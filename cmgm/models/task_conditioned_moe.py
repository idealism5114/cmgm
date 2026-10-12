"""Matched three-expert D0B fusion with four routing controls.

Routing is a property of the prediction task, not a market-state label.
The short/long and per-horizon routers are static across samples.
"""
from contextlib import contextmanager

import torch
from torch import nn

HORIZONS = (1, 5, 10, 20)
BASE_VARIANT = "switching_latent_balanced_readout"
GLOBAL_VARIANT = "switching_latent_balanced_task_moe_global"
CONTENT_VARIANT = "switching_latent_balanced_task_moe_content"
SHORT_LONG_VARIANT = "switching_latent_balanced_task_moe_short_long"
PER_HORIZON_VARIANT = "switching_latent_balanced_task_moe_per_horizon"
MOE_VARIANTS = (GLOBAL_VARIANT, CONTENT_VARIANT, SHORT_LONG_VARIANT, PER_HORIZON_VARIANT)
ALL_VARIANTS = (BASE_VARIANT, *MOE_VARIANTS)
EXPERT_NAMES = ("temporal", "spatial", "interaction")


class ResidualExpert64(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 64))

    def forward(self, x):
        return x + self.mlp(x)


class InteractionExpert64(nn.Module):
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 64))

    def forward(self, s, t):
        return self.mlp(torch.cat((s, t), dim=-1))


class TaskConditionedThreeExpertFusion(nn.Module):
    """Produce four 64D representations from the same three expert outputs."""

    def __init__(self, variant: str):
        super().__init__()
        if variant not in MOE_VARIANTS:
            raise ValueError(f"Unsupported matched MoE variant: {variant}")
        self.variant = variant
        # Identical expert initialization order across all routing controls.
        self.temporal_expert = ResidualExpert64()
        self.spatial_expert = ResidualExpert64()
        self.interaction_expert = InteractionExpert64()
        # Consume the same constructor RNG in every matched MoE mode. Static
        # controls then remove the inactive content router entirely, so the
        # training dropout stream starts from the same seeded RNG state.
        self.content_router = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Linear(32, 3))
        if variant != CONTENT_VARIANT:
            del self.content_router
            rows = {GLOBAL_VARIANT: 1, SHORT_LONG_VARIANT: 2, PER_HORIZON_VARIANT: 4}[variant]
            self.logits = nn.Parameter(torch.zeros(rows, 3))
        self.register_buffer("group_index", torch.tensor((0, 0, 1, 1)), persistent=False)
        self._intervention = None
        self.last_routing = None

    @contextmanager
    def intervention(self, name: str):
        """Frozen-model VAL diagnosis; never used during fit or model selection."""
        if self.training or name not in ("swap_short_long", *EXPERT_NAMES):
            raise ValueError("Intervention requires eval mode and a known diagnosis")
        previous = self._intervention
        self._intervention = name
        try:
            yield
        finally:
            self._intervention = previous

    @contextmanager
    def swap_tasks(self, first: int, second: int):
        """Exchange two horizon routes for a frozen four-task VAL diagnostic."""
        if (self.training or self.variant != PER_HORIZON_VARIANT
                or first not in range(4) or second not in range(4) or first >= second):
            raise ValueError("Task swap requires eval mode and two ordered horizon indices")
        previous = self._intervention
        self._intervention = (first, second)
        try:
            yield
        finally:
            self._intervention = previous

    def routing_weights(self, h_spatial, h_temporal):
        batch = h_spatial.shape[0]
        if self.variant == CONTENT_VARIANT:
            logits = self.content_router(torch.cat((h_spatial, h_temporal), dim=-1))
            pi = torch.softmax(logits, dim=-1)[:, None, :].expand(-1, 4, -1)
        else:
            weights = torch.softmax(self.logits, dim=-1)
            if self.variant == GLOBAL_VARIANT:
                weights = weights.expand(4, -1)
            elif self.variant == SHORT_LONG_VARIANT:
                weights = weights.index_select(0, self.group_index)
            pi = weights.unsqueeze(0).expand(batch, -1, -1)
        if self._intervention == "swap_short_long":
            if self.variant != SHORT_LONG_VARIANT:
                raise ValueError("Short/long swap applies only to the two-group router")
            pi = pi[:, (2, 3, 0, 1), :]
        elif isinstance(self._intervention, tuple):
            first, second = self._intervention
            order = list(range(4))
            order[first], order[second] = order[second], order[first]
            pi = pi[:, order, :]
        elif self._intervention in EXPERT_NAMES:
            chosen = EXPERT_NAMES.index(self._intervention)
            pi = torch.nn.functional.one_hot(
                torch.full((batch, 4), chosen, device=h_spatial.device), 3
            ).to(dtype=h_spatial.dtype)
        return pi

    def forward(self, s, t, h_spatial, h_temporal):
        if any(z.shape != (s.shape[0], 64) for z in (s, t, h_spatial, h_temporal)):
            raise ValueError("Expected four branch vectors with shape (B,64)")
        experts = torch.stack((self.temporal_expert(t), self.spatial_expert(s),
                               self.interaction_expert(s, t)), dim=1)
        pi = self.routing_weights(h_spatial, h_temporal)
        if pi.shape != (s.shape[0], 4, 3):
            raise AssertionError("Routing shape must be (B,4,3)")
        self.last_routing = pi.detach()
        return torch.einsum("bhk,bkd->bhd", pi, experts)
