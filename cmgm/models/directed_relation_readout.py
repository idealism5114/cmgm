"""Frozen D0B plus a small, sparse commodity-to-history relation readout."""
from __future__ import annotations

import math

import torch
from torch import nn


LAGS = (0, 1, 2, 3, 5)
TOP_K = 5


class DirectedRelationReadout(nn.Module):
    """Shared projections and head for cross-asset and own-history controls."""

    def __init__(self):
        super().__init__()
        self.query = nn.Linear(128, 32)
        self.key = nn.Linear(64, 32)
        self.value = nn.Linear(64, 32)
        self.predictor = nn.Sequential(nn.Linear(160, 64), nn.ReLU(), nn.Linear(64, 4))
        nn.init.zeros_(self.predictor[-1].weight)
        nn.init.zeros_(self.predictor[-1].bias)

    @staticmethod
    def differences(sequence: torch.Tensor, node_indices: torch.Tensor):
        """Asset-major, lag-minor changes strictly within the observed window."""
        if sequence.ndim != 4 or sequence.size(1) < max(LAGS) + 2:
            raise ValueError('Insufficient observed history for the fixed lag set')
        return torch.stack([
            (sequence[:, -1-lag] - sequence[:, -2-lag]).index_select(1, node_indices)
            for lag in LAGS
        ], dim=2)

    def forward(self, commodity_nodes: torch.Tensor, temporal: torch.Tensor,
                sequence: torch.Tensor, source_indices: torch.Tensor,
                commodity_indices: torch.Tensor, *, cross_asset: bool,
                return_selection: bool = False):
        batch, commodities, hidden = commodity_nodes.shape
        if hidden != 64 or temporal.shape != (batch, 64):
            raise ValueError('Expected commodity nodes [B,C,64] and temporal [B,64]')
        if commodity_indices.numel() != commodities:
            raise ValueError('Commodity output order does not match target order')
        query = self.query(torch.cat((
            commodity_nodes, temporal[:, None].expand(-1, commodities, -1)
        ), dim=-1))
        if cross_asset:
            changes = self.differences(sequence, source_indices)
            candidates = changes.reshape(batch, -1, 64)  # [B, N_source*5, 64]
            keys = self.key(candidates)
            values = self.value(candidates)
            scores = torch.einsum('bcd,bmd->bcm', query, keys) / math.sqrt(32)
            source_asset = source_indices.repeat_interleave(len(LAGS))
        else:
            changes = self.differences(sequence, commodity_indices)
            keys = self.key(changes)  # [B,C,5,32]
            values = self.value(changes)
            scores = torch.einsum('bcd,bcmd->bcm', query, keys) / math.sqrt(32)
            source_asset = None
        selected_score, selected_index = scores.topk(TOP_K, dim=-1)
        attention = selected_score.softmax(dim=-1)
        if cross_asset:
            selected_value = values[:, None].expand(-1, commodities, -1, -1).gather(
                2, selected_index[..., None].expand(-1, -1, -1, values.size(-1))
            )
        else:
            selected_value = values.gather(
                2, selected_index[..., None].expand(-1, -1, -1, values.size(-1))
            )
        message = (attention[..., None] * selected_value).sum(dim=2)
        temporal_broadcast = temporal[:, None].expand(-1, commodities, -1)
        residual = self.predictor(torch.cat((commodity_nodes, temporal_broadcast, message), dim=-1))
        residual = residual.permute(0, 2, 1).contiguous()  # [B,4,C]
        if not return_selection:
            return residual
        if cross_asset:
            asset = source_asset.index_select(0, selected_index.reshape(-1)).reshape_as(selected_index)
            lag_lookup = torch.as_tensor(LAGS,device=selected_index.device).repeat(source_indices.numel())
            selected_lag = lag_lookup.index_select(0, selected_index.reshape(-1)).reshape_as(selected_index)
        else:
            asset = commodity_indices[None, :, None].expand_as(selected_index)
            selected_lag = torch.as_tensor(LAGS,device=selected_index.device)[selected_index]
        return residual, dict(asset_index=asset,lag=selected_lag,
                              candidate_index=selected_index,attention=attention,
                              message=message)


class FrozenD0BDirectedReadout(nn.Module):
    """Run the saved D0B once; optimize only the added relation readout."""

    def __init__(self, backbone: nn.Module, *, commodity_indices: tuple[int, int],
                 stock_count: int, bond_count: int, cross_asset: bool):
        super().__init__()
        start,end=commodity_indices
        if start != stock_count + bond_count or end-start != backbone.n_commodities:
            raise ValueError('Source and target node mapping differs from the dataset')
        if not (0 < stock_count and bond_count > 0 and end <= backbone.num_nodes):
            raise ValueError('Invalid source/commodity node intervals')
        self.backbone=backbone
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.readout=DirectedRelationReadout()
        self.cross_asset=bool(cross_asset)
        self.variant=('frozen_d0b_directed_relation' if cross_asset else
                      'frozen_d0b_own_history')
        self.register_buffer('source_indices',torch.arange(start),persistent=False)
        self.register_buffer('commodity_indices',torch.arange(start,end),persistent=False)

    def train(self, mode: bool=True):
        super().train(mode)
        self.backbone.eval()  # No frozen dropout or running-state updates.
        return self

    def forward(self, x: torch.Tensor, *unused_graph, debug: bool=False,
                return_selection: bool=False):
        with torch.no_grad():
            spatial,nodes,sequence=self.backbone._temp_weighted_spatial(
                x,return_nodes=True,return_sequence=True
            )
            temporal=self.backbone.switching_latent_transformer(x)
            base=self.backbone._market_token_predict(spatial,temporal)
            commodity_nodes=nodes.index_select(1,self.commodity_indices)
        extra=self.readout(commodity_nodes,temporal,sequence,self.source_indices,
                           self.commodity_indices,cross_asset=self.cross_asset,
                           return_selection=return_selection)
        if return_selection:
            correction,selection=extra
            return base+correction,dict(base=base,correction=correction,**selection)
        return base+extra
