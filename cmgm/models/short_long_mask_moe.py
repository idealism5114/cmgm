"""Five-expert D0B fusion with fixed short/long task access masks.

The temporal H/Z sequences are live outputs of one existing temporal pass.
Only the final fusion differs from D0B; no detached diagnostic cache is read.
"""
from contextlib import contextmanager

import torch
from torch import nn

from cmgm.models.horizon_tensor_fusion import predict_by_horizon


GLOBAL_VARIANT = "switching_latent_balanced_short_long_moe_global"
FIXED_VARIANT = "switching_latent_balanced_short_long_moe_mask_equal"
LEARNED_VARIANT = "switching_latent_balanced_short_long_moe_mask_learned"
MASK_MOE_VARIANTS = (GLOBAL_VARIANT, FIXED_VARIANT, LEARNED_VARIANT)
EXPERT_NAMES = ("shared", "short_memory", "short_shock", "long_memory", "long_structure")


class SharedInteractionExpert(nn.Module):
    """E0 = LN((s+t)/2 + W[(U_s s) * (U_t t)]), rank 8."""

    def __init__(self):
        super().__init__()
        self.s_rank = nn.Linear(64, 8, bias=False)
        self.t_rank = nn.Linear(64, 8, bias=False)
        self.out = nn.Linear(8, 64, bias=False)
        self.norm = nn.LayerNorm(64)

    def forward(self, s, t):
        return self.norm(.5 * (s + t) + self.out(self.s_rank(s) * self.t_rank(t)))


class ShortMemoryExpert(nn.Module):
    """ES1 reads Z_last, mean(Z_last5), and their difference."""

    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(192, 16), nn.GELU(), nn.Linear(16, 64))
        self.norm = nn.LayerNorm(64)

    def forward(self, latent):
        last, recent = latent[:, -1], latent[:, -5:].mean(dim=1)
        return self.norm(self.project(torch.cat((last, recent, last - recent), dim=-1)))


class ShortShockExpert(nn.Module):
    """ES2 compares the spatial state with the recent latent change."""

    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(192, 16), nn.GELU(), nn.Linear(16, 64))
        self.norm = nn.LayerNorm(64)

    def forward(self, s, latent):
        recent_change = latent[:, -1] - latent[:, -5:].mean(dim=1)
        conflict = torch.abs(s - recent_change)
        return self.norm(self.project(torch.cat((s, recent_change, conflict), dim=-1)))


class LongMemoryExpert(nn.Module):
    """EL1 reads H_last, mean(H_all20), and slow first-to-last change."""

    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.Linear(384, 16), nn.GELU(), nn.Linear(16, 64))
        self.norm = nn.LayerNorm(64)

    def forward(self, memory):
        last, mean, slow = memory[:, -1], memory.mean(dim=1), memory[:, -1] - memory[:, 0]
        return self.norm(self.project(torch.cat((last, mean, slow), dim=-1)))


class LongStructureExpert(nn.Module):
    """EL2 interacts spatial state with the full-window H mean, rank 8."""

    def __init__(self):
        super().__init__()
        self.memory_project = nn.Linear(128, 64)
        self.s_rank = nn.Linear(64, 8, bias=False)
        self.h_rank = nn.Linear(64, 8, bias=False)
        self.out = nn.Linear(8, 64, bias=False)
        self.norm = nn.LayerNorm(64)

    def forward(self, s, memory):
        long_state = self.memory_project(memory.mean(dim=1))
        interaction = self.out(self.s_rank(s) * self.h_rank(long_state))
        return self.norm(.5 * (s + long_state) + interaction)


class ShortLongMaskedMoE(nn.Module):
    """One global four-specialist route, or disjoint short/long access masks."""

    def __init__(self, variant: str):
        super().__init__()
        if variant not in MASK_MOE_VARIANTS:
            raise ValueError(f"Unsupported short/long MoE: {variant}")
        self.variant = variant
        # Identical expert construction order across the three matched controls.
        self.shared_expert = SharedInteractionExpert()
        self.short_memory_expert = ShortMemoryExpert()
        self.short_shock_expert = ShortShockExpert()
        self.long_memory_expert = LongMemoryExpert()
        self.long_structure_expert = LongStructureExpert()
        self.fusion_norm = nn.LayerNorm(64)
        if variant == GLOBAL_VARIANT:
            self.global_logits = nn.Parameter(torch.zeros(4))
        elif variant == LEARNED_VARIANT:
            self.theta_short = nn.Parameter(torch.zeros(2))
            self.theta_long = nn.Parameter(torch.zeros(2))
        self._intervention = None
        self.last_experts = None  # Detached, read-only diagnosis; never used below.

    def routing_weights(self):
        if self.variant == GLOBAL_VARIANT:
            return {"global": torch.softmax(self.global_logits, dim=0)}
        if self.variant == FIXED_VARIANT:
            half = self.fusion_norm.weight.new_full((2,), .5)
            return {"short": half, "long": half}
        return {"short": torch.softmax(self.theta_short, dim=0),
                "long": torch.softmax(self.theta_long, dim=0)}

    @contextmanager
    def intervention(self, name: str):
        """Frozen-model VAL diagnosis; never used for training or selection."""
        allowed = {"swap_short_long", "remove_short_memory", "remove_short_shock",
                   "remove_long_memory", "remove_long_structure"}
        if self.training or name not in allowed:
            raise ValueError("Known intervention requires eval mode")
        previous = self._intervention
        self._intervention = name
        try:
            yield
        finally:
            self._intervention = previous

    def forward(self, s, t, memory, latent):
        batch = s.shape[0]
        if (s.shape != (batch, 64) or t.shape != (batch, 64)
                or memory.shape != (batch, 20, 128)
                or latent.shape != (batch, 20, 64)):
            raise ValueError("Expected s/t (B,64), live H (B,20,128), live Z (B,20,64)")
        e0 = self.shared_expert(s, t)
        es1 = self.short_memory_expert(latent)
        es2 = self.short_shock_expert(s, latent)
        el1 = self.long_memory_expert(memory)
        el2 = self.long_structure_expert(s, memory)
        self.last_experts = {name: value.detach() for name, value in zip(
            EXPERT_NAMES, (e0, es1, es2, el1, el2))}
        if self._intervention == "remove_short_memory":
            es1 = torch.zeros_like(es1)
        elif self._intervention == "remove_short_shock":
            es2 = torch.zeros_like(es2)
        elif self._intervention == "remove_long_memory":
            el1 = torch.zeros_like(el1)
        elif self._intervention == "remove_long_structure":
            el2 = torch.zeros_like(el2)
        weights = self.routing_weights()
        if self.variant == GLOBAL_VARIANT:
            w = weights["global"]
            fused = self.fusion_norm(e0 + w[0]*es1 + w[1]*es2 + w[2]*el1 + w[3]*el2)
            states = fused[:, None, :].expand(-1, 4, -1)
        else:
            short, long = weights["short"], weights["long"]
            if self._intervention == "swap_short_long":
                short, long = long, short
            z_short = self.fusion_norm(e0 + short[0]*es1 + short[1]*es2)
            z_long = self.fusion_norm(e0 + long[0]*el1 + long[1]*el2)
            states = torch.stack((z_short, z_short, z_long, z_long), dim=1)
        return states

    def predict(self, head, fused, n_commodities):
        return predict_by_horizon(head, fused, n_commodities)
