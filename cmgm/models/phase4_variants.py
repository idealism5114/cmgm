"""Phase 4 pre-registered simplification variants.

Four structural simplifications of the frozen D0B architecture, each removing
exactly one family of mechanisms that Phase 3 diagnosed as functionally
redundant on the trained checkpoint:

    S1 switching_latent_fixed_mean_fusion  remove the learned fusion gate
    S2 switching_latent_no_graph_prior     remove the adaptive structural prior
    S3 switching_latent_uniform_router     remove the learned regime routing
    S4 switching_latent_simplified_all     remove all three together

Implementation contract
-----------------------
Each variant constructs the *complete* native D0B model first, exactly as the
audited baseline does, and only then removes the modules that no longer
participate. That ordering matters twice over: every shared parameter keeps the
same seeded initialization as the baseline, and the removed parameters never
consume RNG. Nothing new is allocated, so no variant adds parameters.

Removal is more than a forward-time bypass: the deleted modules are gone from
the module tree, so they are absent from `state_dict` and from
`sum(p.numel() for p in model.parameters())`, and the parameter deltas in
`parameter_summary.csv` are real.
"""
from __future__ import annotations

import torch
from torch import nn

from cmgm.models.hetero_mixhop_model import HeteroMixHopCMGM

S1_FIXED_MEAN_FUSION = "switching_latent_fixed_mean_fusion"
S2_NO_GRAPH_PRIOR = "switching_latent_no_graph_prior"
S3_UNIFORM_ROUTER = "switching_latent_uniform_router"
S4_SIMPLIFIED_ALL = "switching_latent_simplified_all"
PHASE4_VARIANTS = (S1_FIXED_MEAN_FUSION, S2_NO_GRAPH_PRIOR,
                   S3_UNIFORM_ROUTER, S4_SIMPLIFIED_ALL)

DESCRIPTION = {
    S1_FIXED_MEAN_FUSION: "fused = 0.5*t + 0.5*s; the learned gate_fc is removed",
    S2_NO_GRAPH_PRIOR: "e_ij = q.k/sqrt(d); the adaptive adjacency prior and its "
                       "learner are removed",
    S3_UNIFORM_ROUTER: "p_t = (1/3,1/3,1/3) at every step; the evidence and "
                       "transition parameters are removed, the K generators and "
                       "the Z recursion are retained",
    S4_SIMPLIFIED_ALL: "S1 + S2 + S3 applied together",
}


class Phase4Simplification(HeteroMixHopCMGM):
    """A pre-registered Phase 4 simplification of the D0B model."""

    def __init__(self, variant: str, **kwargs):
        if variant not in PHASE4_VARIANTS:
            raise ValueError(f"Unknown Phase 4 variant {variant!r}")
        if any(k in kwargs for k in ("variant",)):
            raise ValueError("variant is positional for this class")
        super().__init__(variant="switching_latent_balanced_readout", **kwargs)
        self.phase4_variant = variant
        self.removed_modules = []
        # No module construction, no RNG draw, no parameter allocation below:
        # only removals and attribute changes.
        self.mean_fusion = variant in (S1_FIXED_MEAN_FUSION, S4_SIMPLIFIED_ALL)
        self.no_graph_prior = variant in (S2_NO_GRAPH_PRIOR, S4_SIMPLIFIED_ALL)
        self.uniform_router = variant in (S3_UNIFORM_ROUTER, S4_SIMPLIFIED_ALL)

        if self.mean_fusion:
            del self.gate_fc
            self.removed_modules.append("gate_fc")

        if self.no_graph_prior:
            del self.graph_learner
            self.removed_modules.append("graph_learner")
            self.use_learn_graph = False
            # a zero structure buffer keeps _temp_weighted_spatial callable; with
            # prior_scale = 0 its value cannot reach the attention logits at all
            self.register_buffer("static_A", torch.zeros(self.num_nodes,
                                                         self.num_nodes))
            for name in ("attn_mixhop1", "attn_mixhop2"):
                module = getattr(self, name)
                module.prior_scale = 0.0
                if module.graph_prior_heads is not None:
                    raise ValueError("prior removal assumes graph_prior_heads=None")

        if self.uniform_router:
            branch = self.switching_latent_transformer
            # the uniform branch in latent_forward never reads the filter's
            # evidence network or transition matrix
            branch.formal_uniform_switching = True
            filt = branch.regime_filter
            for attr in ("regime_evidence", "transition_logits"):
                delattr(filt, attr)
                self.removed_modules.append(f"regime_filter.{attr}")
            # defensive: switch_loss() must not touch the removed parameter even
            # if it is called before the first forward pass
            filt._last_switch_loss = torch.zeros(())

    # ---------------------------------------------------------------- forward
    def forward(self, x, edge_index=None, edge_weight=None, debug=False,
                market_descriptor=None):
        if self.variant != "switching_latent_balanced_readout":
            raise RuntimeError("Phase 4 variants only implement the D0B path")
        h_spatial = self._temp_weighted_spatial(x)
        h_temporal = self.switching_latent_transformer(x)
        s = self.gcn_proj(h_spatial)
        t = self.lstm_proj(h_temporal)
        if self.mean_fusion:
            fused = 0.5 * t + 0.5 * s
        else:
            gate = torch.sigmoid(self.gate_fc(torch.cat([h_spatial, h_temporal],
                                                        dim=-1)))
            fused = gate * t + (1.0 - gate) * s
        pred = self.head(fused)
        return pred.view(x.shape[0], self.n_horizons, self.n_commodities)

    # ------------------------------------------------------------- reporting
    def simplification_metadata(self) -> dict:
        return dict(
            phase4_variant=self.phase4_variant,
            description=DESCRIPTION[self.phase4_variant],
            removed_modules=list(self.removed_modules),
            mean_fusion=bool(self.mean_fusion),
            no_graph_prior=bool(self.no_graph_prior),
            uniform_router=bool(self.uniform_router),
            total_parameters=int(sum(p.numel() for p in self.parameters())),
            trainable_parameters=int(sum(p.numel() for p in self.parameters()
                                         if p.requires_grad)),
        )


def is_phase4_variant(name: str) -> bool:
    return name in PHASE4_VARIANTS
