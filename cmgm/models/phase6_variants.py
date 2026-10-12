"""Phase 6 pre-registered nonlinear cross-branch fusion architectures.

Both variants replace only the final spatial-temporal fusion module. They share
the fusion input convention of the audited GlobalConcatFusion (see
interpretability/phase6/fusion_input_semantics.md):

    x_s = gcn_proj(h_spatial)      x_t = lstm_proj(h_temporal)      c = [x_s; x_t]

M1 GlobalConcatFusion (reused, not retrained here):
    f = GELU(W_c c + b_c)                                          in R^64

M2 ResidualFusion:
    n = GELU(W_1 c + b_1)
    r = W_r c + b_r
    f = LayerNorm(n + r)                                           in R^64

M3 BranchPreservingFusion:
    s' = W_s x_s + b_s
    t' = W_t x_t + b_t
    c' = GELU(W_c c + b_c)
    f  = LayerNorm(W_o [s'; t'; c'] + b_o)                         in R^64

No gate, no dropout, no second hidden layer and no attention is added. Each model
builds the complete native D0B model first and only then replaces the gate, so
every shared parameter keeps the same seeded initialization as the baseline.
"""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from cmgm.config import LSTM_HIDDEN_DIM
from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM

M2_RESIDUAL = "residual_cross_branch_fusion"
M3_BRANCH_PRESERVING = "branch_preserving_fusion"
PHASE6_VARIANTS = (M2_RESIDUAL, M3_BRANCH_PRESERVING)

DESCRIPTION = {
    M2_RESIDUAL: "f = LayerNorm(GELU(W1 c + b1) + (Wr c + br)), c = [s; t]",
    M3_BRANCH_PRESERVING: "f = LayerNorm(Wo [Ws x_s; Wt x_t; GELU(Wc c + bc)])",
}


class ResidualFusion(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.main = nn.Linear(dim * 2, dim)
        self.residual = nn.Linear(dim * 2, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x_s, x_t], dim=-1)
        nonlinear = F.gelu(self.main(x))
        residual = self.residual(x)
        return self.norm(nonlinear + residual)


class BranchPreservingFusion(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__()
        self.spatial_proj = nn.Linear(dim, dim)
        self.temporal_proj = nn.Linear(dim, dim)
        self.interaction = nn.Linear(dim * 2, dim)
        self.out = nn.Linear(dim * 3, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x_s: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        s = self.spatial_proj(x_s)
        t = self.temporal_proj(x_t)
        c = F.gelu(self.interaction(torch.cat([x_s, x_t], dim=-1)))
        return self.norm(self.out(torch.cat([s, t, c], dim=-1)))


class Phase6Fusion(HeteroMixHopCMGM):
    """A pre-registered Phase 6 fusion architecture."""

    def __init__(self, variant: str, **kwargs):
        if variant not in PHASE6_VARIANTS:
            raise ValueError(f"Unknown Phase 6 variant {variant!r}")
        super().__init__(variant="switching_latent_balanced_readout", **kwargs)
        self.phase6_variant = variant
        del self.gate_fc
        if variant == M2_RESIDUAL:
            self.fusion = ResidualFusion(LSTM_HIDDEN_DIM)
        else:
            self.fusion = BranchPreservingFusion(LSTM_HIDDEN_DIM)

    def forward(self, x, edge_index=None, edge_weight=None, debug=False,
                market_descriptor=None):
        if self.variant != "switching_latent_balanced_readout":
            raise RuntimeError("Phase 6 variants only implement the D0B path")
        h_spatial = self._temp_weighted_spatial(x)
        h_temporal = self.switching_latent_transformer(x)
        x_s = self.gcn_proj(h_spatial)
        x_t = self.lstm_proj(h_temporal)
        fused = self.fusion(x_s, x_t)
        pred = self.head(fused)
        return pred.view(x.shape[0], self.n_horizons, self.n_commodities)

    def fusion_metadata(self) -> dict:
        base = HeteroMixHopCMGM(num_nodes=self.num_nodes,
                                n_commodities=self.n_commodities,
                                n_stock=self.n_stock, n_bond=self.n_bond,
                                variant="switching_latent_balanced_readout")
        return dict(phase6_variant=self.phase6_variant,
                    description=DESCRIPTION[self.phase6_variant],
                    fusion_parameters=int(sum(p.numel() for p in self.fusion.parameters())),
                    total_parameters=int(sum(p.numel() for p in self.parameters())),
                    replaced="gate_fc (removed); the prediction head is unchanged")
