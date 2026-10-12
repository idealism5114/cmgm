"""Commodity exposure × horizon-conditioned market shock prediction correction.

The five lag windows use only the already encoded 20-step history.  All
diagnostic interventions are inference-only and do not change parameters.
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LagFactorFusion(nn.Module):
    WINDOWS = (1, 3, 5, 10, 20)
    RANK = 4

    def __init__(self, n_commodities: int = 24):
        super().__init__()
        self.n_commodities = n_commodities
        self.global_fusion = nn.Linear(128, 64)
        self.exposure = nn.Linear(64, self.RANK)
        self.lag_logits = nn.Linear(3, 4 * len(self.WINDOWS))
        self.horizon_embedding = nn.Embedding(4, 16)
        self.shock = nn.Linear(128 + 64 + 16, self.RANK)
        # Nonzero, moderate initial shock: every factor receives gradients at
        # the first step, without an extra gate or a trainable scale.
        nn.init.normal_(self.shock.weight, mean=0.0, std=0.005)
        nn.init.zeros_(self.shock.bias)

    def components(self, spatial: torch.Tensor, temporal: torch.Tensor,
                   commodity_nodes: torch.Tensor, history: torch.Tensor,
                   regime: torch.Tensor, *, uniform_lags: bool = False,
                   exposure_permutation: torch.Tensor | None = None):
        batch = spatial.shape[0]
        if spatial.shape != (batch, 64) or temporal.shape != (batch, 64):
            raise ValueError('Expected spatial and temporal vectors [B,64]')
        if commodity_nodes.shape != (batch, self.n_commodities, 64):
            raise ValueError('Commodity nodes/order do not match [B,Nc,64]')
        if history.shape != (batch, 20, 128) or regime.shape != (batch, 3):
            raise ValueError('Expected 20 live history states and live 3-state probabilities')
        base_state = F.gelu(self.global_fusion(torch.cat((spatial, temporal), dim=-1)))
        exposure = self.exposure(commodity_nodes)
        exposure = exposure - exposure.mean(dim=1, keepdim=True)
        if exposure_permutation is not None:
            if exposure_permutation.shape != (self.n_commodities,):
                raise ValueError('Commodity permutation shape mismatch')
            exposure = exposure.index_select(1, exposure_permutation)
        windows = torch.stack([history[:, -width:].mean(dim=1)
                               for width in self.WINDOWS], dim=1)
        weights = self.lag_logits(regime).view(batch, 4, len(self.WINDOWS)).softmax(dim=-1)
        if uniform_lags:
            weights = torch.full_like(weights, 1.0 / len(self.WINDOWS))
        weighted_history = torch.einsum('bhw,bwd->bhd', weights, windows)
        h_embed = self.horizon_embedding.weight.unsqueeze(0).expand(batch, -1, -1)
        temporal_expanded = temporal.unsqueeze(1).expand(-1, 4, -1)
        shock = self.shock(torch.cat((weighted_history, temporal_expanded, h_embed), dim=-1))
        correction = torch.einsum('bcr,bhr->bhc', exposure, shock) / math.sqrt(self.RANK)
        return base_state, correction, exposure, shock, weights

    def forward(self, spatial: torch.Tensor, temporal: torch.Tensor,
                commodity_nodes: torch.Tensor, history: torch.Tensor,
                regime: torch.Tensor, original_head: nn.Module, *,
                remove_correction: bool = False, uniform_lags: bool = False,
                exposure_permutation: torch.Tensor | None = None,
                return_components: bool = False):
        state, correction, exposure, shock, weights = self.components(
            spatial, temporal, commodity_nodes, history, regime,
            uniform_lags=uniform_lags, exposure_permutation=exposure_permutation,
        )
        global_prediction = original_head(state).view(spatial.shape[0], 4, self.n_commodities)
        prediction = global_prediction if remove_correction else global_prediction + correction
        if not return_components:
            return prediction
        return prediction, dict(global_prediction=global_prediction, correction=correction,
                                exposure=exposure, shock=shock, lag_weights=weights)
