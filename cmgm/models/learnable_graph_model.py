"""
CMGM_LearnableGraph — Edge weights are learnable parameters.

Key idea: instead of using fixed correlation values as edge weights,
make them nn.Parameter instances that are updated via gradient descent
during training. The graph structure (which edges exist) stays fixed,
but the importance of each edge is learned.

Architecture: same as CMGM_ReduceFirst, except:
  - Original:  edge_weight is a fixed external tensor (ignored by GCN)
  - This:      edge_weight is a learnable nn.Parameter, used as
               attention weights in the GCN's mean aggregation

The learnable weights are initialized from |correlation| so the model
starts from a sensible graph and adjusts it during training.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from cmgm.config import (
    GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_NUM_LAYERS, GCN_DROPOUT,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    FC_HIDDEN_DIM,
)


# =============================================================================
# Weighted Mean+Concat GCN Layer
# =============================================================================

class _WeightedMeanConcatGCNLayer(nn.Module):
    """
    GCN layer with weighted mean aggregation.

    For each node v:
      1. h_neighbors = Σ(w_u * h_u) / Σ(|w_u|) for u in N(v)   # Weighted mean
      2. h_combined = [h_v || h_neighbors]                       # Concat
      3. h_v' = ReLU(W * h_combined)                             # Linear
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = GCN_DROPOUT):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor,
                edge_weight: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (N, in_dim) node features
            edge_index: (2, E) batched edges
            edge_weight: (E,) learnable edge weights

        Returns:
            (N, out_dim) updated node features
        """
        src_idx, dst_idx = edge_index[0], edge_index[1]
        src_x = x[src_idx]                         # (E, in_dim)

        # Weighted mean: sum(w * x) / sum(|w|)
        w = edge_weight.unsqueeze(-1)              # (E, 1)
        numer = scatter(src_x * w, dst_idx,
                        dim=0, dim_size=x.size(0), reduce='sum')
        denom = scatter(w.abs().expand_as(src_x), dst_idx,
                        dim=0, dim_size=x.size(0), reduce='sum')
        neigh = numer / (denom + 1e-8)             # (N, in_dim)

        out = self.lin(torch.cat([x, neigh], dim=-1))
        return self.dropout(F.relu(out))


# =============================================================================
# Spatial GCN with learnable edge weights
# =============================================================================

class _LearnableSpatialGCN(nn.Module):
    """3× WeightedMeanConcatGCNLayer using learnable edge weights."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            _WeightedMeanConcatGCNLayer(GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_DROPOUT),
            *[_WeightedMeanConcatGCNLayer(GCN_OUTPUT_DIM, GCN_OUTPUT_DIM, GCN_DROPOUT)
              for _ in range(GCN_NUM_LAYERS - 1)],
        ])

    def forward(self, x_t, edge_index, edge_weight, batch_size, num_nodes):
        """
        Args:
            x_t: (B, N, 1) features at time t
            edge_index: (2, E) original (non-batched) graph
            edge_weight: (E,) learnable weights
        """
        B, N = batch_size, num_nodes
        x_flat = x_t.reshape(B * N, -1)

        # Batch edge_index and weights
        device = x_flat.device
        offsets = torch.arange(B, device=device) * N
        batched_ei = edge_index.repeat(1, B) + offsets.repeat_interleave(
            edge_index.shape[1]).unsqueeze(0)
        batched_ew = edge_weight.repeat(B)    # (E * B,)

        for layer in self.layers:
            x_flat = layer(x_flat, batched_ei, batched_ew)

        return x_flat.reshape(B, N * GCN_OUTPUT_DIM)


# =============================================================================
# LSTM (same as original)
# =============================================================================

class _TemporalLSTM(nn.Module):
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


# =============================================================================
# CMGM_LearnableGraph
# =============================================================================

class CMGM_LearnableGraph(nn.Module):
    """
    CMGM with learnable edge weights.

    Initializes edge weights from |correlation| and updates them during
    training via gradient descent. This lets the model learn which market
    connections are most relevant for prediction.

    Forward: (B, T, N, 1), edge_index, edge_weight → (B, N_commodities)
    """

    def __init__(self, num_nodes: int, n_commodities: int,
                 edge_index: torch.Tensor, init_edge_weight: torch.Tensor):
        """
        Args:
            num_nodes: total number of nodes (284)
            n_commodities: number of prediction targets (24)
            edge_index: (2, E) fixed graph structure
            init_edge_weight: (E,) initial values for learnable weights
                              (typically |correlation| or raw correlation)
        """
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities

        # Register edge_index as a buffer (non-learnable, moves with device)
        self.register_buffer('fixed_edge_index', edge_index)

        # Learnable edge weights, initialized from correlation values
        self.learnable_edge_weight = nn.Parameter(init_edge_weight.clone())

        self.spatial = _LearnableSpatialGCN()
        gcn_flat_dim = num_nodes * GCN_OUTPUT_DIM
        self.gcn_reduce_per_step = nn.Linear(gcn_flat_dim, LSTM_HIDDEN_DIM)

        self.temporal = _TemporalLSTM(num_nodes)
        fusion_dim = LSTM_HIDDEN_DIM * 2
        self.fc1 = nn.Linear(fusion_dim, FC_HIDDEN_DIM)
        self.dropout = nn.Dropout(GCN_DROPOUT)
        self.fc2 = nn.Linear(FC_HIDDEN_DIM, n_commodities)

    def forward(self, x, edge_index=None, edge_weight=None, debug=False):
        """
        Ignores passed edge_index/edge_weight — uses internal learnable graph.

        This ensures compatibility with train.py which passes graph tensors
        as arguments. The model simply ignores them and uses its own.
        """
        B, T, N, _ = x.shape
        ei = self.fixed_edge_index
        ew = self.learnable_edge_weight
        device = x.device

        # Move fixed graph to same device as input (if needed)
        if ei.device != device:
            ei = ei.to(device)

        # GCN per timestep with learnable edge weights
        temporal_outputs = []
        for t in range(T):
            out_t = self.spatial(x[:, t, :, :], ei, ew, B, N)
            temporal_outputs.append(out_t)

        gcn_stack = torch.stack(temporal_outputs, dim=1)

        gcn_reduced = self.gcn_reduce_per_step(
            gcn_stack.reshape(-1, gcn_stack.size(-1))
        ).reshape(B, T, -1)
        gcn_out = gcn_reduced.mean(dim=1)

        lstm_out = self.temporal(x.squeeze(-1))
        combined = torch.cat([gcn_out, lstm_out], dim=-1)
        pred = self.fc2(self.dropout(F.relu(self.fc1(combined))))

        if debug:
            print(f"  GCN({T} steps, learnable_weights={list(ew.shape)})"
                  f" → reduce→({list(gcn_reduced.shape)})"
                  f" → mean→({list(gcn_out.shape)}) || "
                  f"LSTM→({list(lstm_out.shape)}) → "
                  f"FC → ({list(pred.shape)})")

        return pred
