"""
CMGM_Feature — Parallel GCN(21-dim) || LSTM(21-dim) → Concat → FC.

Architecture:
    GCN Branch:
        per-timestep Mean+Concat GCN(×3, F→10) → mean pool → Linear(2840→64)
    LSTM Branch:
        flattened features → LSTM(N*F→64) → (B, 64)
    Fusion:
        Concat(64+64) → FC(128→64→24)

Key change from v1:
  - LSTM now receives ALL features (flattened: N*F per timestep)
    instead of just the price channel.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from cmgm.config import (
    GCN_OUTPUT_DIM, GCN_NUM_LAYERS, GCN_DROPOUT,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    FC_HIDDEN_DIM,
)
from cmgm.data.feature_builder import NUM_FEATURES


# =============================================================================
# Spatial: Mean+Concat GCN (same as v1)
# =============================================================================

class _MeanConcatGCNLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, dropout: float = GCN_DROPOUT):
        super().__init__()
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src_idx, dst_idx = edge_index[0], edge_index[1]
        neigh = scatter(x[src_idx], dst_idx, dim=0, dim_size=x.size(0), reduce='mean')
        out = self.lin(torch.cat([x, neigh], dim=-1))
        out = self.norm(out)
        return self.dropout(F.relu(out))


class _SpatialGCN(nn.Module):
    """3× Mean+Concat GCN layers (F→10, 10→10, 10→10)."""

    def __init__(self, in_dim: int = NUM_FEATURES):
        super().__init__()
        self.layers = nn.ModuleList([
            _MeanConcatGCNLayer(in_dim, GCN_OUTPUT_DIM, GCN_DROPOUT),
            *[_MeanConcatGCNLayer(GCN_OUTPUT_DIM, GCN_OUTPUT_DIM, GCN_DROPOUT)
              for _ in range(GCN_NUM_LAYERS - 1)],
        ])

    def forward(self, x_t: torch.Tensor, edge_index: torch.Tensor,
                batch_size: int, num_nodes: int) -> torch.Tensor:
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


# =============================================================================
# Temporal: LSTM (now uses ALL features)
# =============================================================================

class _TemporalLSTM(nn.Module):
    """
    LSTM over flattened features.
    Input:  (B, T, N * F) — all features for all nodes at each timestep
    Output: (B, 64)
    """

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
# Output: Fusion FC (same as v1)
# =============================================================================

class _OutputLayer(nn.Module):
    """FC: Linear(128→64) → ReLU → Dropout → Linear(64→N_commodities)."""

    def __init__(self, in_dim: int, n_commodities: int):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, FC_HIDDEN_DIM)
        self.dropout = nn.Dropout(GCN_DROPOUT)
        self.fc2 = nn.Linear(FC_HIDDEN_DIM, n_commodities)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


# =============================================================================
# CMGM_Feature: Parallel GCN(all F) || LSTM(all F)
# =============================================================================

class CMGM_Feature(nn.Module):
    """
    Feature-augmented CMGM — both GCN and LSTM use ALL features.

    GCN:   per-timestep, (B, N, F) → GCN → (B, N*10)
    LSTM:  flattened features, (B, T, N*F) → LSTM → (B, 64)
    Fusion: Concat(64+64) → FC → (B, N_commodities)

    Forward:  (B, T, N, F), edge_index, edge_weight → (B, N_commodities)
    """

    def __init__(self, num_nodes: int, n_commodities: int,
                 feat_dim: int = NUM_FEATURES):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities
        self.feat_dim = feat_dim

        # ── GCN branch (all F features) ──
        self.spatial = _SpatialGCN(in_dim=feat_dim)
        gcn_flat_dim = num_nodes * GCN_OUTPUT_DIM          # 2840
        self.gcn_reduce = nn.Linear(gcn_flat_dim, LSTM_HIDDEN_DIM)

        # ── LSTM branch (ALL features flattened) ──
        lstm_input_dim = num_nodes * feat_dim               # 284 * 21 = 5964
        self.temporal = _TemporalLSTM(lstm_input_dim)

        # ── Fusion ──
        fusion_dim = LSTM_HIDDEN_DIM * 2                   # 128
        self.output = _OutputLayer(fusion_dim, n_commodities)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None,
                debug: bool = False) -> torch.Tensor:
        """
        Args:
            x:   (B, T, N, F)  — standardized features
            edge_index: (2, E)
        """
        batch_size, seq_len, num_nodes, n_feat = x.shape

        # ========== GCN Branch (all features, per timestep) ==========
        temporal_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]                                # (B, N, F)
            out_t = self.spatial(x_t, edge_index, batch_size, num_nodes)  # (B, N*10)
            temporal_outputs.append(out_t)

        gcn_stack = torch.stack(temporal_outputs, dim=1)       # (B, T, N*10)
        gcn_pooled = gcn_stack.mean(dim=1)                     # (B, N*10)
        gcn_out = self.gcn_reduce(gcn_pooled)                  # (B, 64)

        # ========== LSTM Branch (all features, flattened) ==========
        x_flat = x.reshape(batch_size, seq_len, -1)            # (B, T, N*F)
        lstm_out = self.temporal(x_flat)                       # (B, 64)

        # ========== Fusion ==========
        combined = torch.cat([gcn_out, lstm_out], dim=-1)      # (B, 128)
        pred = self.output(combined)                           # (B, N_commodities)

        if debug:
            print(f"  GCN(F={n_feat})→({list(gcn_out.shape)}) || "
                  f"LSTM(N*F={num_nodes*n_feat})→({list(lstm_out.shape)}) "
                  f"→ Concat({list(combined.shape)}) → FC → ({list(pred.shape)})")

        return pred
