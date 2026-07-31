"""
CMGM v3 — Learned Graph via Self-Attention.

每个时间步，模型通过多头注意力自己学出资产间的关系（N×N 稠密图），
再用这个学出来的"软图"做 Mean+Concat GCN 聚合。

不需要 graph_builder.py，不需要预定义的相关系数。

Architecture (Parallel):
  Attention-GCN branch: Self-Attention(4头) → Mean+Concat(×3) → mean pool → Linear(2840→64)
  LSTM branch:          raw prices → LSTM(284→64)
  Fusion:               Concat(64+64) → FC(128→64→N_commodities)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from cmgm.config import (
    GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_NUM_LAYERS,
    GCN_DROPOUT,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    FC_HIDDEN_DIM,
)


class AttentionGCNLayer(nn.Module):
    """
    GCN layer with learned attention weights.

    1.  Query/Key projection → multi-head attention (N×N weights)
    2.  Attention-weighted mean over all nodes
    3.  Concat: [node_features || weighted_mean_neighbors]
    4.  Linear + ReLU + Dropout

    No pre-defined graph needed — the N×N attention matrix IS the graph.
    """

    def __init__(self, in_dim: int, out_dim: int,
                 n_heads: int = 4, dropout: float = GCN_DROPOUT):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = max(4, in_dim * 2 // n_heads)

        self.qk_proj = nn.Linear(in_dim, n_heads * self.head_dim * 2)
        self.lin = nn.Linear(in_dim * 2, out_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, N, in_dim)

        Returns:
            out: (B, N, out_dim)
        """
        B, N, D = x.shape

        # ---- Step 1: Multi-head attention ----
        qk = self.qk_proj(x)                        # (B, N, H*d*2)
        qk = qk.view(B, N, self.n_heads, self.head_dim * 2)
        q, k = qk.chunk(2, dim=-1)                  # each (B, N, H, d)
        q = q.transpose(1, 2)                       # (B, H, N, d)
        k = k.transpose(1, 2)                       # (B, H, N, d)

        attn = q @ k.transpose(-2, -1)              # (B, H, N, N)
        attn = attn / (self.head_dim ** 0.5)
        attn = F.softmax(attn, dim=-1)              # row-stochastic

        # ---- Step 2: Weighted mean over all nodes ----
        # For each node i: weighted_mean = Σ_j attn[h,i,j] * x[j]
        # x.unsqueeze(1) → (B, 1, N, D) broadcasts over heads → (B, H, N, D)
        h_neighbors = torch.matmul(attn, x.unsqueeze(1))  # (B, H, N, D)
        h_neighbors = h_neighbors.mean(dim=1)              # (B, N, D)

        # ---- Step 3: Concat + Linear ----
        combined = torch.cat([x, h_neighbors], dim=-1)  # (B, N, D*2)
        out = self.lin(combined)                        # (B, N, out_dim)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class LearnedGraphGCN(nn.Module):
    """
    3× AttentionGCNLayer — no pre-defined graph needed.

    Input per time step:  (B, N, 1)
    Output per time step: (B, N * GCN_OUTPUT_DIM)
    """

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([
            AttentionGCNLayer(GCN_INPUT_DIM, GCN_OUTPUT_DIM,
                              n_heads=4, dropout=GCN_DROPOUT),
            *[AttentionGCNLayer(GCN_OUTPUT_DIM, GCN_OUTPUT_DIM,
                                n_heads=4, dropout=GCN_DROPOUT)
              for _ in range(GCN_NUM_LAYERS - 1)],
        ])

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_t: (B, N, 1)

        Returns:
            (B, N * GCN_OUTPUT_DIM)
        """
        B, N, _ = x_t.shape
        for layer in self.layers:
            x_t = layer(x_t)                        # (B, N, GCN_OUTPUT_DIM)
        return x_t.reshape(B, N * GCN_OUTPUT_DIM)


class TemporalLSTM(nn.Module):
    """LSTM for raw price sequence."""

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


class OutputLayer(nn.Module):
    """Linear → ReLU → Dropout → Linear"""

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


class CMGM_Attention(nn.Module):
    """
    CMGM with learned graph via Self-Attention.

    GCN branch:     Attention → Mean+Concat(×3) → mean pool → Linear(2840→64)
    LSTM branch:    raw prices → LSTM(284→64)
    Fusion:         Concat(64+64) → FC(128→64→24)

    No pre-computed graph needed — edge_index and edge_weight are dummy.
    """

    def __init__(self, num_nodes: int, n_commodities: int):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities

        self.spatial = LearnedGraphGCN()
        gcn_flat_dim = num_nodes * GCN_OUTPUT_DIM
        self.gcn_reduce = nn.Linear(gcn_flat_dim, LSTM_HIDDEN_DIM)

        self.temporal = TemporalLSTM(num_nodes)

        fusion_dim = LSTM_HIDDEN_DIM * 2
        self.output = OutputLayer(fusion_dim, n_commodities)

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor = None,
        edge_weight: torch.Tensor = None,
        debug: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, T, N, 1) — normalized closing prices
            edge_index: ignored (graph is learned)
            edge_weight: ignored
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # ========== GCN Branch (learned graph) ==========
        temporal_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]                         # (B, N, 1)
            out_t = self.spatial(x_t)                    # (B, N*10)
            temporal_outputs.append(out_t)

        gcn_stack = torch.stack(temporal_outputs, dim=1) # (B, T, N*10)
        gcn_pooled = gcn_stack.mean(dim=1)               # (B, N*10)
        gcn_out = self.gcn_reduce(gcn_pooled)            # (B, 64)

        # ========== LSTM Branch ==========
        x_seq = x.squeeze(-1)                            # (B, T, N)
        lstm_out = self.temporal(x_seq)                  # (B, 64)

        # ========== Fusion ==========
        combined = torch.cat([gcn_out, lstm_out], dim=-1)
        pred = self.output(combined)

        if debug:
            print(f"  Attn→({list(gcn_out.shape)}) || LSTM→({list(lstm_out.shape)})"
                  f" → Concat({list(combined.shape)}) → FC → ({list(pred.shape)})")

        return pred
