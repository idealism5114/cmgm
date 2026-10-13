"""Matched static/dynamic short-long routing; no auxiliary losses or warm-up.

A retains the existing expert functions, but uses a three-candidate convex sum.
B uses the original D0B gated representation plus zero-initialized residuals.
Names describe input biases, not established economic specialization.
"""
from contextlib import contextmanager

import torch
from torch import nn
from cmgm.models.short_long_mask_moe import (
    SharedInteractionExpert, ShortMemoryExpert, ShortShockExpert,
    LongMemoryExpert, LongStructureExpert,
)
from cmgm.models.horizon_tensor_fusion import predict_by_horizon

PREFIX = "switching_latent_balanced_"
A_STATIC = PREFIX + "existing_experts_task_static"
A_DYNAMIC = PREFIX + "existing_experts_task_dynamic"
B_STATIC = PREFIX + "residual_experts_task_static"
B_DYNAMIC = PREFIX + "residual_experts_task_dynamic"
DYNAMIC_STUDY_VARIANTS = (A_STATIC, A_DYNAMIC, B_STATIC, B_DYNAMIC)
RESIDUAL_VARIANTS = (B_STATIC, B_DYNAMIC)
EXPERT_NAMES = ("shared", "short_state", "short_interaction", "long_state", "long_interaction")


class ZeroResidual(nn.Module):
    def __init__(self, input_dim, dim=64, hidden=16):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, x):
        return self.mlp(self.norm(x))


class AlignedInteractionResidual(nn.Module):
    def __init__(self, other_dim, dim=64, hidden=16):
        super().__init__()
        self.s_project = nn.Linear(dim, dim)
        self.other_project = nn.Linear(other_dim, dim)
        self.s_norm = nn.LayerNorm(dim)
        self.other_norm = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(3 * dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, s, other):
        a, b = self.s_norm(self.s_project(s)), self.other_norm(self.other_project(other))
        return self.mlp(torch.cat((a, b, a * b), dim=-1))


class GroupRouter(nn.Module):
    def __init__(self, dynamic, dim=64):
        super().__init__()
        self.dynamic = dynamic
        # Same RNG consumption in each matched pair, no inactive parameters.
        self.norm_s, self.norm_t = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.short = nn.Sequential(nn.Linear(2 * dim, 16), nn.GELU(), nn.Linear(16, 3))
        self.long = nn.Sequential(nn.Linear(2 * dim, 16), nn.GELU(), nn.Linear(16, 3))
        for router in (self.short, self.long):
            nn.init.normal_(router[-1].weight, mean=0., std=1e-3)
            nn.init.zeros_(router[-1].bias)
        if not dynamic:
            del self.norm_s, self.norm_t, self.short, self.long
            self.logits = nn.Parameter(torch.zeros(2, 3))

    def forward(self, s, t):
        if not self.dynamic:
            return self.logits.unsqueeze(0).expand(s.shape[0], -1, -1)
        u = torch.cat((self.norm_s(s), self.norm_t(t)), dim=-1)
        return torch.stack((self.short(u), self.long(u)), dim=1)


class ShortLongDynamicMoE(nn.Module):
    def __init__(self, variant, dim=64, memory_dim=128, latent_dim=64, seq_len=20):
        super().__init__()
        if variant not in DYNAMIC_STUDY_VARIANTS:
            raise ValueError(variant)
        # Existing A experts have fixed dimensions; fail explicitly on config drift.
        if (dim, memory_dim, latent_dim, seq_len) != (64, 128, 64, 20):
            raise ValueError("This fixed study requires d=64, H=128, Z=64, T=20")
        self.variant = variant
        self.residual = variant in RESIDUAL_VARIANTS
        self.dynamic = variant in (A_DYNAMIC, B_DYNAMIC)
        self.dim, self.memory_dim, self.latent_dim, self.seq_len = dim, memory_dim, latent_dim, seq_len
        if self.residual:
            self.short_state = ZeroResidual(2 * latent_dim, dim)
            self.short_interaction = AlignedInteractionResidual(latent_dim, dim)
            self.long_state = ZeroResidual(2 * memory_dim, dim)
            self.long_interaction = AlignedInteractionResidual(memory_dim, dim)
        else:
            self.shared = SharedInteractionExpert()
            self.short_state = ShortMemoryExpert()
            self.short_interaction = ShortShockExpert()
            self.long_state = LongMemoryExpert()
            self.long_interaction = LongStructureExpert()
            self.fusion_norm = nn.LayerNorm(dim)
        self.router = GroupRouter(self.dynamic, dim)
        self._intervention = None
        self.last = {}  # detached reporting only, never read by prediction

    @contextmanager
    def intervention(self, name, weights=None):
        """Frozen inference: override routes, shared-only, or mask one logit.

        override weights must be TRAIN means or externally permuted whole pairs.
        No RNG or parameters are changed by this context.
        """
        allowed = ("override", "shared_only", "mask_short_1", "mask_short_2",
                   "mask_long_1", "mask_long_2")
        if self.training or name not in allowed:
            raise ValueError("Known interventions require eval mode")
        previous = self._intervention
        self._intervention = (name, weights)
        try:
            yield
        finally:
            self._intervention = previous

    def routing_weights(self, s, t):
        logits = self.router(s, t)
        if self._intervention is not None:
            name, weights = self._intervention
            if name == "override":
                weights = torch.as_tensor(weights, device=s.device, dtype=s.dtype)
                if weights.shape == (2, 3):
                    weights = weights[None].expand(len(s), -1, -1)
                if (weights.shape != logits.shape or not torch.isfinite(weights).all()
                        or (weights < 0).any() or not torch.allclose(
                            weights.sum(-1), torch.ones_like(weights[..., 0]), atol=1e-6, rtol=0)):
                    raise ValueError("Override must be normalized (2,3) or (B,2,3)")
                return weights
            if name == "shared_only":
                weights = torch.zeros_like(logits)
                weights[..., 0] = 1
                return weights
            _, group, candidate = name.split("_")
            logits = logits.clone()
            logits[:, 0 if group == "short" else 1, int(candidate)] = -torch.inf
        return logits.softmax(-1)

    def forward(self, s, t, memory, latent, z_base=None, return_components=False):
        b = len(s)
        if (s.shape != (b, self.dim) or t.shape != s.shape
                or memory.shape != (b, self.seq_len, self.memory_dim)
                or latent.shape != (b, self.seq_len, self.latent_dim)):
            raise ValueError("Invalid projected states or live H/Z shapes")
        pi = self.routing_weights(s, t)
        if self.residual:
            if z_base is None or z_base.shape != s.shape:
                raise ValueError("B requires the original live D0B gated state")
            last, recent = latent[:, -1], latent[:, -5:].mean(dim=1)
            full = memory.mean(dim=1)
            deltas = torch.stack((
                self.short_state(torch.cat((last, recent), dim=-1)),
                self.short_interaction(s, last - recent),
                self.long_state(torch.cat((memory[:, -1], full), dim=-1)),
                self.long_interaction(s, full)), dim=1)
            group_delta = deltas.reshape(b, 2, 2, self.dim)
            weighted_delta = (pi[..., 1:, None] * group_delta).sum(2)
            groups = z_base[:, None] + weighted_delta
            experts = torch.cat((z_base[:, None], z_base[:, None] + deltas), dim=1)
        else:
            experts = torch.stack((self.shared(s, t), self.short_state(latent),
                                   self.short_interaction(s, latent), self.long_state(memory),
                                   self.long_interaction(s, memory)), dim=1)
            candidates = torch.stack((experts[:, (0, 1, 2)], experts[:, (0, 3, 4)]), dim=1)
            groups = self.fusion_norm((pi[..., None] * candidates).sum(2))
            deltas = weighted_delta = None
        states = groups[:, (0, 0, 1, 1)]
        parts = dict(pi=pi, experts=experts, deltas=deltas,
                     weighted_delta=weighted_delta, z_base=z_base, groups=groups)
        self.last = {k: v.detach() for k, v in parts.items() if v is not None}
        return (states, parts) if return_components else states

    def predict(self, head, states, n_commodities):
        return predict_by_horizon(head, states, n_commodities)
