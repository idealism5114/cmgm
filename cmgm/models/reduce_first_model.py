"""
CMGM_ReduceFirst — GCN per-timestep → reduce to 64 → mean pool over time.
Same as original CMGM, but reduces dimension before (not after) mean pooling.

Original:
    GCN per step (B, T, 2840) → mean pool over T → (B, 2840) → Linear(2840→64)

ReduceFirst:
    GCN per step → reduce per step (B, T, 2840) → Linear(2840→64) per step
    → (B, T, 64) → mean pool over T → (B, 64)

The change means the mean pool operates in a lower-dim space (64 vs 2840),
which reduces noise from irrelevant node dimensions.
"""

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from cmgm.config import (
    GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_NUM_LAYERS, GCN_DROPOUT,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    FC_HIDDEN_DIM,
)


class _MeanConcatGCNLayer(nn.Module):
    """GCN: Mean aggregation + Concat combination."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = GCN_DROPOUT):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        neigh = scatter(x[src_idx], dst_idx, dim=0, dim_size=x.size(0), reduce='mean')
        out = self.lin(torch.cat([x, neigh], dim=-1))
        return self.dropout(torch.relu(out))


class _SpatialGCN(nn.Module):
    """3× MeanConcatGCNLayer(1→10, 10→10, 10→10)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            _MeanConcatGCNLayer(GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_DROPOUT),
            *[_MeanConcatGCNLayer(GCN_OUTPUT_DIM, GCN_OUTPUT_DIM, GCN_DROPOUT)
              for _ in range(GCN_NUM_LAYERS - 1)],
        ])

    def forward(self, x_t, edge_index, batch_size, num_nodes):
        x_flat = x_t.reshape(batch_size * num_nodes, -1)
        batched_ei = self._batch_edge_index(edge_index, num_nodes, batch_size, x_flat.device)
        for layer in self.layers:
            x_flat = layer(x_flat, batched_ei)
        return x_flat.reshape(batch_size, num_nodes * GCN_OUTPUT_DIM)

    @staticmethod
    def _batch_edge_index(edge_index, num_nodes, batch_size, device):
        offsets = torch.arange(batch_size, device=device) * num_nodes
        return edge_index.repeat(1, batch_size) + offsets.repeat_interleave(
            edge_index.shape[1]).unsqueeze(0)


class _TemporalLSTM(nn.Module):
    """LSTM for raw price sequence.  Input: (B, T, N), Output: (B, 64)."""

    def __init__(self, input_size: int):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=LSTM_HIDDEN_DIM,
            num_layers=LSTM_NUM_LAYERS,
            dropout=LSTM_DROPOUT if LSTM_NUM_LAYERS > 1 else 0.0,
            batch_first=True,
        )

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


class CMGM_ReduceFirst(nn.Module):
    """
    Same as original CMGM, except:
      Original:  GCN per step → mean pool → Linear(2840→64)
      This:      GCN per step → Linear(2840→64) per step → mean pool

    Forward: (B, T, N, 1), edge_index, edge_weight → (B, N_commodities)
    """

    def __init__(self, num_nodes: int, n_commodities: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities

        self.spatial = _SpatialGCN()
        gcn_flat_dim = num_nodes * GCN_OUTPUT_DIM  # 2840

        # Per-timestep reduction (before mean pool)
        self.gcn_reduce_per_step = nn.Linear(gcn_flat_dim, LSTM_HIDDEN_DIM)

        self.temporal = _TemporalLSTM(num_nodes)
        fusion_dim = LSTM_HIDDEN_DIM * 2
        self.fc1 = nn.Linear(fusion_dim, FC_HIDDEN_DIM)
        self.dropout = nn.Dropout(GCN_DROPOUT)
        self.fc2 = nn.Linear(FC_HIDDEN_DIM, n_commodities)

    def forward(self, x, edge_index, edge_weight=None, debug=False):
        B, T, N, _ = x.shape

        # GCN per timestep
        temporal_outputs = []
        for t in range(T):
            out_t = self.spatial(x[:, t, :, :], edge_index, B, N)  # (B, 2840)
            temporal_outputs.append(out_t)

        gcn_stack = torch.stack(temporal_outputs, dim=1)  # (B, T, 2840)

        # Reduce per step → mean pool (the key change)
        gcn_reduced = self.gcn_reduce_per_step(
            gcn_stack.reshape(-1, gcn_stack.size(-1))
        ).reshape(B, T, -1)                     # (B, T, 64)
        gcn_out = gcn_reduced.mean(dim=1)       # (B, 64)

        # LSTM
        lstm_out = self.temporal(x.squeeze(-1))  # (B, 64)

        # Fusion
        combined = torch.cat([gcn_out, lstm_out], dim=-1)  # (B, 128)
        pred = self.fc2(self.dropout(torch.relu(self.fc1(combined))))

        if debug:
            print(f"  GCN({T} steps) → reduce→({list(gcn_reduced.shape)}) "
                  f"→ mean→({list(gcn_out.shape)}) || "
                  f"LSTM→({list(lstm_out.shape)}) → "
                  f"FC → ({list(pred.shape)})")

        return pred
