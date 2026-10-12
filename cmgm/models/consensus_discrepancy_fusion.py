"""Consensus plus discrepancy-residual fusion for two 64D D0B branch states."""
from __future__ import annotations

import torch
from torch import nn


MEAN_VARIANT = "switching_latent_balanced_mean_fusion"
RESIDUAL_VARIANT = "switching_latent_balanced_consensus_discrepancy_fusion"
HIDDEN_DIM = 64
BOTTLENECK_DIM = 16


def mean_fusion(s: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Parameter-free arithmetic mean of matching projected branch states."""
    if s.ndim != 2 or s.shape[-1] != HIDDEN_DIM or t.shape != s.shape:
        raise ValueError("Expected matching projected branch states of shape (B,64)")
    return 0.5 * (s + t)


class ConsensusDiscrepancyFusion(nn.Module):
    """Apply a learned discrepancy residual around the branch mean.

    ``c`` and ``d`` denote the mean and difference representations. These names
    describe the arithmetic only; they do not imply statistical disentanglement.
    """

    def __init__(self, hidden_dim: int = HIDDEN_DIM,
                 bottleneck_dim: int = BOTTLENECK_DIM):
        super().__init__()
        if hidden_dim != HIDDEN_DIM or bottleneck_dim != BOTTLENECK_DIM:
            raise ValueError("ConsensusDiscrepancyFusion is fixed to 64D / 16D")
        self.A_c = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.A_d = nn.Linear(hidden_dim, bottleneck_dim, bias=False)
        self.b = nn.Parameter(torch.zeros(bottleneck_dim))
        self.B = nn.Linear(bottleneck_dim, hidden_dim, bias=False)
        nn.init.zeros_(self.B.weight)

    def forward(self, s: torch.Tensor, t: torch.Tensor, *,
                disable_correction: bool = False,
                return_components: bool = False):
        if s.ndim != 2 or s.shape[-1] != HIDDEN_DIM or t.shape != s.shape:
            raise ValueError("Expected matching projected branch states of shape (B,64)")
        c = 0.5 * (s + t)
        d = s - t
        context = self.A_c(c) + self.b
        r = torch.tanh(context + self.A_d(d)) - torch.tanh(context)
        delta = self.B(r)
        applied_delta = torch.zeros_like(delta) if disable_correction else delta
        fused = c + applied_delta
        if return_components:
            return fused, {
                "c": c, "d": d, "context": context, "r": r,
                "delta": delta, "applied_delta": applied_delta,
            }
        return fused

    @torch.no_grad()
    def diagnostics(self, s: torch.Tensor, t: torch.Tensor, eps: float = 1e-8) -> dict:
        """Summarize per-sample L2 norms; ratio is averaged across samples."""
        _, parts = self(s, t, return_components=True)
        c_norm = parts["c"].norm(dim=-1)
        d_norm = parts["d"].norm(dim=-1)
        delta_norm = parts["delta"].norm(dim=-1)
        ratio = delta_norm / (c_norm + eps)
        return {
            "consensus_l2_mean_per_sample": float(c_norm.mean()),
            "discrepancy_l2_mean_per_sample": float(d_norm.mean()),
            "delta_l2_mean_per_sample": float(delta_norm.mean()),
            "delta_to_consensus_ratio_mean_per_sample": float(ratio.mean()),
            "delta_to_consensus_ratio_median_per_sample": float(ratio.median()),
            "delta_to_consensus_ratio_max_per_sample": float(ratio.max()),
            "samples": int(s.shape[0]),
            "ratio_reduction": "compute each sample's ||delta||2/(||c||2+eps), then report mean/median/max",
        }
