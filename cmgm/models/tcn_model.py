"""
CMGM_TCN — Parallel GCN || LSTM → Concat → FC.
Replaces mean pooling over GCN's temporal outputs with a Temporal
Convolution Network (TCN).

Architecture:
    GCN branch:
        per-timestep Mean+Concat GCN(×3, 1→10) → Linear(2840→64) per step
        → TCN (4× dilated causal conv1d) → mean pool → (B, 64)

    LSTM branch:
        raw prices → LSTM(284→64) → (B, 64)

    Fusion:
        Concat(64+64) → FC(128→64→24)

Key difference from the original CMGM:
  - Original: mean pool over T → Linear(2840→64)
  - TCN:      Linear(2840→64) per step → TCN over T → (B, 64)

The TCN captures temporal patterns in the spatial (GCN) embeddings,
replacing a simple mean with learned temporal convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import scatter

from cmgm.config import (
    GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_NUM_LAYERS, GCN_DROPOUT,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    FC_HIDDEN_DIM, SEQ_LEN,
)


# =============================================================================
# Spatial: Mean+Concat GCN (identical to model.py)
# =============================================================================

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
        return self.dropout(F.relu(out))


class _SpatialGCN(nn.Module):
    """3× MeanConcatGCNLayer(1→10, 10→10, 10→10)."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            _MeanConcatGCNLayer(GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_DROPOUT),
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
# Temporal Convolution Network (replaces mean pooling)
# =============================================================================

class _TemporalConvNet(nn.Module):
    """
    Causal dilated 1D convolution over the temporal dimension.

    Input:  (B, T, C_in)
    Output: (B, C_out) — after mean pooling over time

    Architecture: 4× DilatedConv1d(C_in→C_out, ..., C_out→C_out)
    with dilations [1, 2, 4, 8] and kernel_size=3.
    """

    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int = 3, dropout: float = GCN_DROPOUT):
        super().__init__()
        self.out_channels = out_channels
        dilations = [1, 2, 4, 8]

        layers = []
        for i, d in enumerate(dilations):
            c_in = in_channels if i == 0 else out_channels
            # Causal padding: output[i] sees only inputs [i - d*(k-1), ..., i]
            padding = d * (kernel_size - 1)
            layers.append(nn.Sequential(
                nn.Conv1d(c_in, out_channels, kernel_size,
                          padding=padding, dilation=d),
                nn.ReLU(),
                nn.Dropout(dropout),
            ))
        self.convs = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, T, C_in)

        Returns:
            (B, C_out) — mean-pooled over the temporal dimension
        """
        x = x.transpose(1, 2)  # (B, C_in, T)
        for conv in self.convs:
            # After causal padding, trim extra frames so output length = T
            out = conv(x)      # (B, C_out, T + padding)
            x = out[:, :, :x.size(-1)]  # trim to original length T
        x = x.transpose(1, 2)  # (B, T, C_out)
        return x.mean(dim=1)   # (B, C_out)


# =============================================================================
# Temporal: LSTM (identical to model.py)
# =============================================================================

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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, (h_n, _) = self.lstm(x)
        return h_n[-1]


# =============================================================================
# Output: Fusion FC (identical to model.py)
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
# CMGM_TCN: Parallel GCN(TCN) || LSTM
# =============================================================================

class CMGM_TCN(nn.Module):
    """
    CMGM with TCN replacing mean pooling over GCN temporal outputs.

    Forward call convention (compatible with train.py / evaluate.py):
        model(X, edge_index, edge_weight, debug)
          X: (B, T, N, 1)  normalized prices
          edge_index: (2, E)
          edge_weight: ignored (unweighted mean)
    """

    def __init__(self, num_nodes: int, n_commodities: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities

        # ── GCN branch ──
        self.spatial = _SpatialGCN()
        gcn_flat_dim = num_nodes * GCN_OUTPUT_DIM  # 2840

        # Per-timestep reduction (replaces post-mean-pool reduction)
        self.gcn_reduce_per_step = nn.Linear(gcn_flat_dim, LSTM_HIDDEN_DIM)  # 2840→64

        # TCN processes the temporal sequence of 64-dim embeddings
        self.tcn = _TemporalConvNet(LSTM_HIDDEN_DIM, LSTM_HIDDEN_DIM)

        # ── LSTM branch ──
        self.temporal = _TemporalLSTM(num_nodes)

        # ── Fusion ──
        fusion_dim = LSTM_HIDDEN_DIM * 2  # 128
        self.output = _OutputLayer(fusion_dim, n_commodities)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None,
                debug: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, 1) — normalized closing prices
            edge_index: (2, E)
            edge_weight: ignored
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # ========== GCN Branch (per timestep → TCN) ==========
        temporal_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]                                 # (B, N, 1)
            out_t = self.spatial(x_t, edge_index, batch_size, num_nodes)  # (B, N*10)
            temporal_outputs.append(out_t)

        gcn_stack = torch.stack(temporal_outputs, dim=1)         # (B, T, 2840)

        # Reduce per timestep: 2840 → 64 at each t
        B, T, D = gcn_stack.shape
        gcn_reduced = self.gcn_reduce_per_step(
            gcn_stack.reshape(-1, D)        # (B*T, 2840)
        ).reshape(B, T, -1)                 # (B, T, 64)

        # TCN processes the temporal dimension
        gcn_out = self.tcn(gcn_reduced)     # (B, 64)

        # ========== LSTM Branch ==========
        x_seq = x.squeeze(-1)               # (B, T, N)
        lstm_out = self.temporal(x_seq)     # (B, 64)

        # ========== Fusion ==========
        combined = torch.cat([gcn_out, lstm_out], dim=-1)  # (B, 128)
        pred = self.output(combined)                       # (B, N_commodities)

        if debug:
            print(f"  GCN(steps={T}) → reduce→({list(gcn_reduced.shape)}) "
                  f"→ TCN→({list(gcn_out.shape)}) || "
                  f"LSTM→({list(lstm_out.shape)}) → "
                  f"Concat({list(combined.shape)}) → "
                  f"FC → ({list(pred.shape)})")

        return pred
