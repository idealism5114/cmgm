"""
CMGM v2 model — accepts pre-batched dynamic graphs.

The only difference from model.py's SpatialGCN is that the edge_index
is ALREADY batched by the DataLoader collate function, so we skip
the _batch_edge_index() offset step.

Architecture (same parallel GCN || LSTM as v1):
  GCN branch:     per-timestep Mean+Concat(×3) → mean pool → Linear(2840→64)
  LSTM branch:    raw prices → LSTM(284→64)
  Fusion:         Concat(64+64) → FC(128→64→N_commodities)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from cmgm.config import (
    GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_NUM_LAYERS,
    GCN_DROPOUT,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    FC_HIDDEN_DIM,
)
from cmgm.models.model import MeanConcatGCNLayer, TemporalLSTM, OutputLayer


class DynamicSpatialGCN(nn.Module):
    """
    Same as SpatialGCN but edge_index is ALREADY batched per-sample
    by the collate function — no _batch_edge_index call needed.
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            MeanConcatGCNLayer(GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_DROPOUT),
            *[MeanConcatGCNLayer(GCN_OUTPUT_DIM, GCN_OUTPUT_DIM, GCN_DROPOUT)
              for _ in range(GCN_NUM_LAYERS - 1)],
        ])

    def forward(
        self,
        x_t: torch.Tensor,
        edge_index: torch.Tensor,
        batch_size: int,
        num_nodes: int,
    ) -> torch.Tensor:
        """
        Args:
            x_t: (B, N, 1) — features at one time step
            edge_index: (2, total_E) — pre-batched for all samples in batch
            batch_size: B
            num_nodes: N

        Returns:
            (B, N * GCN_OUTPUT_DIM)
        """
        x_flat = x_t.reshape(batch_size * num_nodes, -1)

        # edge_index is already batched by collate — each sample's nodes
        # occupy [k*N, (k+1)*N-1] range
        for layer in self.layers:
            x_flat = layer(x_flat, edge_index)

        return x_flat.reshape(batch_size, num_nodes * GCN_OUTPUT_DIM)


class CMGM_Dynamic(nn.Module):
    """
    CMGM with per-sample dynamic graphs.

    Same parallel architecture as v1, but uses DynamicSpatialGCN
    that accepts pre-batched edge_index.
    """

    def __init__(self, num_nodes: int, n_commodities: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities

        # GCN branch
        self.spatial = DynamicSpatialGCN()
        gcn_flat_dim = num_nodes * GCN_OUTPUT_DIM
        self.gcn_reduce = nn.Linear(gcn_flat_dim, LSTM_HIDDEN_DIM)

        # LSTM branch
        self.temporal = TemporalLSTM(num_nodes)

        # Fusion: concat GCN(64) + LSTM(64) → FC(128→64→24)
        fusion_dim = LSTM_HIDDEN_DIM * 2
        self.output = OutputLayer(fusion_dim, n_commodities)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, 1) — normalized closing prices
            edge_index: (2, total_E) — pre-batched per-sample graphs
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # ========== GCN Branch ==========
        temporal_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]
            out_t = self.spatial(x_t, edge_index, batch_size, num_nodes)
            temporal_outputs.append(out_t)

        gcn_stack = torch.stack(temporal_outputs, dim=1)
        gcn_pooled = gcn_stack.mean(dim=1)
        gcn_out = self.gcn_reduce(gcn_pooled)

        # ========== LSTM Branch ==========
        x_seq = x.squeeze(-1)
        lstm_out = self.temporal(x_seq)

        # ========== Fusion ==========
        combined = torch.cat([gcn_out, lstm_out], dim=-1)
        pred = self.output(combined)

        if debug:
            print(f"  GCN→({list(gcn_out.shape)}) || LSTM→({list(lstm_out.shape)})"
                  f" → Concat({list(combined.shape)}) → FC → ({list(pred.shape)})")

        return pred
