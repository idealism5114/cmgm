"""Phase 5B pre-registered fusion controls.

These controls separate the *width* of the fusion representation from the
*operator* that combines the two branches.

Important premise correction
----------------------------
The Phase 5 brief states that GlobalConcatFusion preserves 128 dimensions into
the prediction head and that Original compresses to 64. That is not what the
frozen code does. GlobalConcatFusion computes

    fused = F.gelu(global_concat_fusion(cat([s, t])))   with
    global_concat_fusion = nn.Linear(128, 64)

so it outputs 64 dimensions and feeds exactly the same 64-input head as the
Original model, at an identical parameter count (520549). The two models
therefore differ only in the fusion operator:

    Original:  g = sigmoid(gate_fc(cat([h_s, h_t])));  f = g*t + (1-g)*s
    GCF:       f = gelu(W_cat @ cat([s, t]) + b_cat)

C1 below is consequently GCF without the GELU activation, and it isolates the
activation rather than the width. C2 and C3 keep the pre-registered 128-output
form and therefore widen the head's first layer from 64 to 128 inputs, which
adds parameters; they are structural controls, not capacity-matched ones.

    C1 ConcatLinear64  cat([s,t]) 128 -> Linear 64 -> head (no activation)
    C2 GatedConcat128  [g*t ; (1-g)*s] 128 -> widened head
    C3 MeanConcat128   [0.5t ; 0.5s]   128 -> widened head
"""
from __future__ import annotations

import torch
from torch import nn

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM
from cmgm.config import LSTM_HIDDEN_DIM, FC_HIDDEN_DIM

C1_CONCAT_LINEAR_64 = "ConcatLinear64"
C2_GATED_CONCAT_128 = "GatedConcat128"
C3_MEAN_CONCAT_128 = "MeanConcat128"
PHASE5_VARIANTS = (C1_CONCAT_LINEAR_64, C2_GATED_CONCAT_128, C3_MEAN_CONCAT_128)

DESCRIPTION = {
    C1_CONCAT_LINEAR_64: "f = W_cat @ [s;t] + b_cat (128->64, no activation), then the "
                         "Original head; differs from GCF only by the GELU",
    C2_GATED_CONCAT_128: "f = [g*t ; (1-g)*s] in 128d with the learned gate, then a head "
                         "whose first layer is widened to 128 inputs",
    C3_MEAN_CONCAT_128: "f = [0.5t ; 0.5s] in 128d, no gate, then a widened head",
}


class Phase5FusionControl(HeteroMixHopCMGM):
    """A pre-registered Phase 5B fusion control."""

    def __init__(self, variant: str, **kwargs):
        if variant not in PHASE5_VARIANTS:
            raise ValueError(f"Unknown Phase 5 variant {variant!r}")
        super().__init__(variant="switching_latent_balanced_readout", **kwargs)
        self.phase5_variant = variant
        self.added_modules = []
        # C1: replace the gate with a bias-included 128->64 projection, no activation
        if variant == C1_CONCAT_LINEAR_64:
            del self.gate_fc
            self.concat_linear = nn.Linear(2 * LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM)
            self.added_modules.append("concat_linear")
        # C2: keep the learned gate but stop forcing the branches into one 64d space
        elif variant == C2_GATED_CONCAT_128:
            self.head[0] = nn.Linear(2 * LSTM_HIDDEN_DIM, FC_HIDDEN_DIM)
            self.added_modules.append("head[0] widened to 128 inputs")
        # C3: no gate at all, branches kept side by side in 128d
        elif variant == C3_MEAN_CONCAT_128:
            del self.gate_fc
            self.head[0] = nn.Linear(2 * LSTM_HIDDEN_DIM, FC_HIDDEN_DIM)
            self.added_modules.append("head[0] widened to 128 inputs")

    def forward(self, x, edge_index=None, edge_weight=None, debug=False,
                market_descriptor=None):
        if self.variant != "switching_latent_balanced_readout":
            raise RuntimeError("Phase 5 variants only implement the D0B path")
        h_spatial = self._temp_weighted_spatial(x)
        h_temporal = self.switching_latent_transformer(x)
        s = self.gcn_proj(h_spatial)
        t = self.lstm_proj(h_temporal)
        if self.phase5_variant == C1_CONCAT_LINEAR_64:
            fused = self.concat_linear(torch.cat([s, t], dim=-1))
        elif self.phase5_variant == C2_GATED_CONCAT_128:
            gate = torch.sigmoid(self.gate_fc(torch.cat([h_spatial, h_temporal], dim=-1)))
            fused = torch.cat([gate * t, (1.0 - gate) * s], dim=-1)
        else:
            fused = torch.cat([0.5 * t, 0.5 * s], dim=-1)
        pred = self.head(fused)
        return pred.view(x.shape[0], self.n_horizons, self.n_commodities)

    def control_metadata(self) -> dict:
        return dict(phase5_variant=self.phase5_variant,
                    description=DESCRIPTION[self.phase5_variant],
                    added_modules=list(self.added_modules),
                    fusion_width=int(self.head[0].in_features),
                    has_dynamic_gate=self.phase5_variant == C2_GATED_CONCAT_128,
                    additive_mixing=self.phase5_variant != C1_CONCAT_LINEAR_64,
                    total_parameters=int(sum(p.numel() for p in self.parameters())))
