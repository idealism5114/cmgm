"""Four functional experts and sample-invariant, horizon-specific routing.

The live H and Z sequences are produced by the existing temporal branch in a
single pass. No diagnostic (detached) cache participates in prediction.
"""
from contextlib import contextmanager
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F

from cmgm.models.horizon_tensor_fusion import predict_by_horizon

GLOBAL_VARIANT = "switching_latent_balanced_four_task_moe_global"
PER_HORIZON_VARIANT = "switching_latent_balanced_four_task_moe_per_horizon"
FILM_VARIANT = "switching_latent_balanced_four_task_moe_per_horizon_film"
FOUR_TASK_VARIANTS = (GLOBAL_VARIANT, PER_HORIZON_VARIANT, FILM_VARIANT)
EXPERT_NAMES = ("recent_change", "long_memory", "spatial_structure", "low_rank_interaction")
HORIZONS = (1, 5, 10, 20)


class RecentChangeExpert(nn.Module):
    """ReLU(W[Z_last, Z_last - mean(Z_last_5)]), from live Z (B,T,64)."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(128, 64)

    def forward(self, latent):
        if latent.ndim != 3 or latent.shape[1:] != (20, 64):
            raise ValueError("Recent expert requires live (B,20,64) latent states")
        last = latent[:, -1]
        change = last - latent[:, -5:].mean(dim=1)
        return F.relu(self.projection(torch.cat((last, change), dim=-1)))


class LongMemoryExpert(nn.Module):
    """ReLU(W[H_last, mean(H_all_20)]), from live H (B,T,128)."""

    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(256, 64)

    def forward(self, memory):
        if memory.ndim != 3 or memory.shape[1:] != (20, 128):
            raise ValueError("Long expert requires live (B,20,128) long memory")
        return F.relu(self.projection(torch.cat((memory[:, -1], memory.mean(dim=1)), dim=-1)))


class SpatialStructureExpert(nn.Module):
    """s + ReLU(W_s s), where s is the original projected spatial vector."""

    def __init__(self):
        super().__init__()
        self.refine = nn.Linear(64, 64)

    def forward(self, spatial):
        return spatial + F.relu(self.refine(spatial))


class LowRankInteractionExpert(nn.Module):
    """t + W_o[(U_s s) ⊙ (U_t t)], with rank 16 and no extra gate."""

    def __init__(self):
        super().__init__()
        self.spatial_rank = nn.Linear(64, 16, bias=False)
        self.temporal_rank = nn.Linear(64, 16, bias=False)
        self.output = nn.Linear(16, 64, bias=False)

    def forward(self, spatial, temporal):
        return temporal + self.output(self.spatial_rank(spatial) * self.temporal_rank(temporal))


def predict_with_task_film(head: nn.Sequential, fused: torch.Tensor,
                           scale_delta: torch.Tensor, shift: torch.Tensor,
                           n_commodities: int) -> torch.Tensor:
    """One shared head body, identity-initialized FiLM, horizon-aligned rows."""
    if fused.ndim != 3 or fused.shape[1:] != (4, 64):
        raise ValueError("Expected (B,4,64) fused states")
    if scale_delta.shape != (4, 64) or shift.shape != (4, 64):
        raise ValueError("FiLM parameters must be (4,64)")
    body = head[:3](fused)
    body = body * (1 + scale_delta.unsqueeze(0)) + shift.unsqueeze(0)
    final = head[3]
    if final.in_features != 64 or final.out_features != 4*n_commodities:
        raise ValueError("Shared head does not match horizon-major commodity rows")
    weights = final.weight.view(4, n_commodities, 64)
    biases = final.bias.view(4, n_commodities)
    return torch.einsum("bhd,hnd->bhn", body, weights) + biases.unsqueeze(0)


class FourTaskMoEFusion(nn.Module):
    """Four shared experts; one global or four task-specific static mixtures."""

    def __init__(self, variant: str):
        super().__init__()
        if variant not in FOUR_TASK_VARIANTS:
            raise ValueError(f"Unsupported four-task MoE: {variant}")
        self.variant = variant
        # Identical construction order and RNG use across all three controls.
        self.recent_expert = RecentChangeExpert()
        self.long_expert = LongMemoryExpert()
        self.spatial_expert = SpatialStructureExpert()
        self.interaction_expert = LowRankInteractionExpert()
        rows = 1 if variant == GLOBAL_VARIANT else 4
        self.logits = nn.Parameter(torch.zeros(rows, 4))
        if variant == FILM_VARIANT:
            self.film_scale_delta = nn.Parameter(torch.zeros(4, 64))
            self.film_shift = nn.Parameter(torch.zeros(4, 64))
        self._swap = None
        self.last_routing = None

    @contextmanager
    def swap_tasks(self, first: int, second: int):
        """VAL-only frozen-model intervention; indices follow (1,5,10,20)."""
        if self.training or self.variant == GLOBAL_VARIANT or (first, second) not in combinations(range(4), 2):
            raise ValueError("Task swap requires eval mode and two distinct horizon indices")
        previous = self._swap
        self._swap = (first, second)
        try:
            yield
        finally:
            self._swap = previous

    def routing_weights(self):
        weights = torch.softmax(self.logits, dim=-1)
        if self.variant == GLOBAL_VARIANT:
            weights = weights.expand(4, -1)
        if self._swap is not None:
            first, second = self._swap
            order = list(range(4))
            order[first], order[second] = order[second], order[first]
            weights = weights[order]
        return weights

    def forward(self, spatial, temporal, live_memory, live_latent):
        batch = spatial.shape[0]
        if spatial.shape != (batch, 64) or temporal.shape != (batch, 64):
            raise ValueError("Expected projected branch vectors (B,64)")
        experts = torch.stack((
            self.recent_expert(live_latent),
            self.long_expert(live_memory),
            self.spatial_expert(spatial),
            self.interaction_expert(spatial, temporal),
        ), dim=1)
        pi = self.routing_weights()
        self.last_routing = pi.detach()
        return torch.einsum("hk,bkd->bhd", pi, experts)

    def predict(self, head, fused, n_commodities):
        if self.variant == FILM_VARIANT:
            return predict_with_task_film(head, fused, self.film_scale_delta,
                                          self.film_shift, n_commodities)
        return predict_by_horizon(head, fused, n_commodities)
