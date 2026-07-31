"""
CMGM_CrossAttn — Cross-Attention Fusion between GCN and LSTM branches.

Replaces simple Concat fusion with cross-attention:
  LSTM final hidden state as QUERY
  GCN per-time-step outputs as KEY / VALUE

Architecture:
  GCN branch:
    per-timestep Mean+Concat GCN(×3, 1→10) → stack (B, T, N*10)
    → gcn_reduce applied per step → (B, T, 64)   [← KEYS / VALUES]
    → mean pool → gcn_reduce → (B, 64)           [← original path, kept]

  LSTM branch:
    raw prices → LSTM(284→64) → (B, 64)           [← QUERY]

  Cross-Attention Fusion (replaces Concat):
    Q = lstm_out.unsqueeze(1)          (B, 1, 64)
    K = V = gcn_steps                  (B, T, 64)
    attn = softmax(Q·K^T / √d)         (B, 1, T)
    attended = attn · V                (B, 64)
    combined = [attended || lstm_out]  (B, 128)
    → FC(128→64→24)

This reuses the existing `gcn_reduce` Linear(2840→64) for per-step
projection — zero additional parameters beyond the original CMGM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cmgm.models.model import (
    MeanConcatGCNLayer, SpatialGCN, TemporalLSTM, OutputLayer,
)
from cmgm.config import (
    GCN_OUTPUT_DIM, GCN_NUM_LAYERS,
    LSTM_HIDDEN_DIM,
    FC_HIDDEN_DIM,
)
from cmgm.data.feature_builder import NUM_FEATURES


class CMGM_CrossAttn(nn.Module):
    """
    CMGM with Cross-Attention Fusion.

    Difference from model.py CMGM:
      GCN per-step → gcn_reduce → (B, T, 64)
          ↓ cross-attention with LSTM query      ← NEW (replaces Concat)
      LSTM → (B, 64)                              ← unchanged
          ↓
      [attended_gcn || lstm] → FC(128→64→24)
    """

    def __init__(self, num_nodes: int, n_commodities: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities

        # ── GCN branch ──
        self.spatial = SpatialGCN()
        gcn_flat_dim = num_nodes * GCN_OUTPUT_DIM  # 2840
        self.gcn_reduce = nn.Linear(gcn_flat_dim, LSTM_HIDDEN_DIM)  # 2840→64

        # ── LSTM branch ──
        self.temporal = TemporalLSTM(num_nodes)

        # ── Cross-Attention Fusion ──
        # Q = LSTM(64), K = V = GCN steps(64)
        # attention dim = 64 (same as hidden)
        # No separate projection needed — reuse gcn_reduce for per-step mapping
        fusion_dim = LSTM_HIDDEN_DIM * 2  # 128 = 64 + 64
        self.output = OutputLayer(fusion_dim, n_commodities)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_weight: torch.Tensor = None,
                debug: bool = False) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, 1) — normalized closing prices
            edge_index: (2, E)
            edge_weight: ignored

        Returns:
            pred: (B, N_commodities)
        """
        batch_size, seq_len, num_nodes, _ = x.shape
        device = x.device

        # ========== GCN Branch ==========
        temporal_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]                                # (B, N, 1)
            out_t = self.spatial(x_t, edge_index,
                                  batch_size, num_nodes)       # (B, N*10)
            temporal_outputs.append(out_t)

        gcn_stack = torch.stack(temporal_outputs, dim=1)       # (B, T, N*10)

        # ── Per-step projection (reuses gcn_reduce) ──
        B, T, D = gcn_stack.shape
        gcn_steps = self.gcn_reduce(
            gcn_stack.reshape(-1, D)                           # (B*T, 2840)
        ).reshape(B, T, -1)                                     # (B, T, 64)

        # ── Also compute original pooled GCN output (kept for reference) ──
        gcn_pooled = gcn_stack.mean(dim=1)                     # (B, 2840)
        gcn_out = self.gcn_reduce(gcn_pooled)                  # (B, 64)

        # ========== LSTM Branch ==========
        x_seq = x.squeeze(-1)                                  # (B, T, N)
        lstm_out = self.temporal(x_seq)                        # (B, 64)

        # ========== Cross-Attention Fusion ==========
        # Q = LSTM output (1 token per sample)
        lstm_q = lstm_out.unsqueeze(1)                         # (B, 1, 64)

        # K = V = GCN per-time-step features
        # Scaled dot-product attention
        d_k = gcn_steps.size(-1)                               # 64
        attn_scores = torch.matmul(lstm_q, gcn_steps.transpose(1, 2))
        #   (B, 1, 64) × (B, 64, T) → (B, 1, T)
        attn_scores = attn_scores / (d_k ** 0.5)
        attn_weights = F.softmax(attn_scores, dim=-1)          # (B, 1, T)

        attended_gcn = torch.matmul(attn_weights, gcn_steps)   # (B, 1, 64)
        attended_gcn = attended_gcn.squeeze(1)                 # (B, 64)

        # Fusion: attended GCN features || LSTM output
        combined = torch.cat([attended_gcn, lstm_out], dim=-1) # (B, 128)

        # ========== Output ==========
        pred = self.output(combined)                           # (B, N_commodities)

        if debug:
            print(f"  GCN→({list(gcn_out.shape)}) | "
                  f"GCN_steps→({list(gcn_steps.shape)}) | "
                  f"LSTM→({list(lstm_out.shape)})")
            print(f"  CrossAttn(LSTM_q × GCN_steps) → "
                  f"attended({list(attended_gcn.shape)}) || "
                  f"LSTM({list(lstm_out.shape)}) → "
                  f"Concat({list(combined.shape)}) → "
                  f"FC → ({list(pred.shape)})")

        return pred
