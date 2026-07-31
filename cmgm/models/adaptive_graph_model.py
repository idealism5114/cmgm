"""
AdaptiveCMGM — Learnable graph structure + Dense GCN || LSTM.

Architecture:
    Adaptive Graph:   AdaptiveGraphLearner → A (N, N)
    GCN Branch:       DenseGCN per-timestep(F→10×3) → reduce(2840→64) → mean pool → (B, 64)
    LSTM Branch:      LSTM(N→64) → (B, 64)
    Fusion:           Concat(64+64) → FC(128→64→N_commodities)

The key difference from original CMGM:
  - Graph is learned end-to-end (not pre-computed from correlations)
  - GCN uses dense adjacency matrix A instead of sparse edge_index
  - The graph structure evolves during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cmgm.config import (
    GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_NUM_LAYERS, GCN_DROPOUT,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    FC_HIDDEN_DIM,
)
from cmgm.graph.adaptive_graph import AdaptiveGraphLearner


# =============================================================================
# Dense GCN Layer (operates on dense adjacency matrix A)
# =============================================================================

class _DenseMeanConcatLayer(nn.Module):
    """
    GCN layer with dense adjacency: Mean aggregation + Concat combination.

    For each node v:
      1. h_neighbors = Σ(A[v,u] · h_u) / Σ(A[v,u])    # Weighted mean via dense A
      2. h_combined = [h_v || h_neighbors]              # Concat
      3. h_v' = ReLU(W · h_combined)                    # Linear
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = GCN_DROPOUT):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x:       (B, N, in_dim)  node features
            A_norm:  (N, N)          row-stochastic adjacency (sum=1 per row)

        Returns:
            (B, N, out_dim)  updated node features
        """
        # Weighted neighbor aggregation via dense matrix multiply
        # aggr = A_norm @ x  →  (N, N) @ (B, N, in_dim)  →  (B, N, in_dim)
        B = x.size(0)
        aggr = torch.bmm(A_norm.unsqueeze(0).expand(B, -1, -1), x)  # (B, N, in_dim)

        combined = torch.cat([x, aggr], dim=-1)           # (B, N, 2*in_dim)
        out = self.lin(combined)                          # (B, N, out_dim)
        return self.dropout(F.relu(out))


class _DenseSpatialGCN(nn.Module):
    """
    Dense GCN — 3 layers applied per time step.

    Input per time step:  (B, N, 1)
    Output:               (B, N * 10)  — flattened node dims
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            _DenseMeanConcatLayer(GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_DROPOUT),
            *[_DenseMeanConcatLayer(GCN_OUTPUT_DIM, GCN_OUTPUT_DIM, GCN_DROPOUT)
              for _ in range(GCN_NUM_LAYERS - 1)],
        ])

    def forward(self, x_t: torch.Tensor, A_norm: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t:     (B, N, in_dim)  features at time t
            A_norm:  (N, N)          row-stochastic adjacency

        Returns:
            (B, N * out_dim)  — flattened per node
        """
        for layer in self.layers:
            x_t = layer(x_t, A_norm)
        B, N, D = x_t.shape
        return x_t.reshape(B, N * D)


# =============================================================================
# Temporal LSTM (same as original)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


# =============================================================================
# AdaptiveCMGM
# =============================================================================

class AdaptiveCMGM(nn.Module):
    """
    CMGM with learnable graph structure.

    Forward call convention (compatible with train.py):
        model(x, edge_index, edge_weight, debug)
          → ignores passed edge_index/edge_weight
          → uses internal graph_learner to produce A

    Args:
        num_nodes:      total nodes (284)
        n_commodities:  output dim (24)
        embed_dim:      graph learner embedding dim
        alpha:          graph learner saturation rate
        top_k:          neighbors per node in learned graph
    """

    def __init__(self, num_nodes: int, n_commodities: int,
                 embed_dim: int = 10, alpha: float = 3.0, top_k: int = 5):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities

        # ── Learnable graph ──
        self.graph_learner = AdaptiveGraphLearner(
            num_nodes, embed_dim=embed_dim, alpha=alpha, top_k=top_k,
        )

        # ── Dense GCN branch ──
        self.spatial = _DenseSpatialGCN()
        gcn_flat_dim = num_nodes * GCN_OUTPUT_DIM  # 2840
        self.gcn_reduce_per_step = nn.Linear(gcn_flat_dim, LSTM_HIDDEN_DIM)

        # ── LSTM branch ──
        self.temporal = _TemporalLSTM(num_nodes)

        # ── Fusion ──
        fusion_dim = LSTM_HIDDEN_DIM * 2
        self.fc1 = nn.Linear(fusion_dim, FC_HIDDEN_DIM)
        self.dropout = nn.Dropout(GCN_DROPOUT)
        self.fc2 = nn.Linear(FC_HIDDEN_DIM, n_commodities)

    def forward(self, x: torch.Tensor,
                edge_index=None, edge_weight=None,
                debug: bool = False) -> torch.Tensor:
        """
        Args:
            x:  (B, T, N, 1)  normalized prices
            edge_index: ignored — graph is learned internally
            edge_weight: ignored
        """
        B, T, N, _ = x.shape
        device = x.device

        # ── 1. Learn adjacency matrix ──
        A = self.graph_learner()                              # (N, N)
        # Row-normalize for mean aggregation
        A_norm = A / (A.sum(dim=-1, keepdim=True) + 1e-8)     # (N, N)

        # ── 2. GCN Branch (dense, per timestep) ──
        temporal_outputs = []
        for t in range(T):
            x_t = x[:, t, :, :]                               # (B, N, 1)
            out_t = self.spatial(x_t, A_norm)                 # (B, N*10)
            temporal_outputs.append(out_t)

        gcn_stack = torch.stack(temporal_outputs, dim=1)       # (B, T, 2840)
        gcn_reduced = self.gcn_reduce_per_step(
            gcn_stack.reshape(-1, gcn_stack.size(-1))
        ).reshape(B, T, -1)                                   # (B, T, 64)
        gcn_out = gcn_reduced.mean(dim=1)                     # (B, 64)

        # ── 3. LSTM Branch ──
        lstm_out = self.temporal(x.squeeze(-1))               # (B, 64)

        # ── 4. Fusion ──
        combined = torch.cat([gcn_out, lstm_out], dim=-1)     # (B, 128)
        pred = self.fc2(self.dropout(F.relu(self.fc1(combined))))

        if debug:
            nz = (A > 0).sum().item()
            print(f"  GraphLearner: A=(N={N}×N={N}), non-zero={nz}/{N*N}, "
                  f"top-k={self.graph_learner.top_k}")
            print(f"  GCN→({list(gcn_out.shape)}) || "
                  f"LSTM→({list(lstm_out.shape)}) → "
                  f"Concat({list(combined.shape)}) → "
                  f"FC → ({list(pred.shape)})")

        return pred
