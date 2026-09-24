"""Temporal base plus gated spatial-temporal interaction correction."""
import torch
from torch import nn
from cmgm.models.candidate_moe_fusion import CandidateAwareTwoExpertFusion

VARIANT = 'switching_latent_balanced_candidate_gated_interaction_residual'


class GatedInteractionCorrection(nn.Module):
    """Zero projected spatial OR temporal input gives an exactly zero correction."""
    def __init__(self):
        super().__init__()
        self.spatial = nn.Linear(64, 64, bias=False)
        self.temporal = nn.Linear(64, 64, bias=False)
        self.dropout = nn.Dropout(.1)
        self.output = nn.Linear(64, 64, bias=False)

    def forward(self, s, t):
        return self.output(self.dropout(torch.relu(self.spatial(s)) * torch.relu(self.temporal(t))))


class CandidateGatedInteractionResidual(CandidateAwareTwoExpertFusion):
    """Preserve original construction RNG, then replace only the old ST expert.

    The inherited TemporalResidualExpert and CandidateAwareRouter are reused
    unchanged. No unused ST expert parameters, independent head or auxiliary loss.
    """
    def __init__(self):
        super().__init__()
        del self.interaction_expert
        self.interaction_correction = GatedInteractionCorrection()

    def forward(self, s, t):
        if s.ndim != 2 or s.shape != t.shape or s.shape[-1] != 64:
            raise ValueError('Projected branches must have shape (B,64)')
        base = self.temporal_expert(t)  # Exactly ONE call, including its dropout.
        delta = self.interaction_correction(s, t)
        enhanced = base + delta
        pi = self.router(base, enhanced)
        experts = torch.stack([base, enhanced], dim=1)
        fused = (pi.unsqueeze(-1) * experts).sum(dim=1)
        # Only diagnostic copies are detached. All live paths above stay connected.
        self.last = {**self.router.last, **{k: v.detach() for k, v in dict(
            s=s, t=t, e_base=base, delta_ST=delta, e_T=base, e_ST=enhanced,
            pi=pi, experts=experts, effective_correction=pi[:, 1:2]*delta, h_moe=fused,
        ).items()}}
        return fused
