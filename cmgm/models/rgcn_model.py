"""
RGCN-CMGM Model — Parallel RGCN || LSTM → Concat → FC.

Replaces the Mean+Concat GCN layers with RGCNConv (Relational Graph
Convolution) using 9 pre-defined relation types (Stock/Stock, Stock/Future,
Bond/Bond, etc.).  Relation types are assigned from edge_type produced by
rgcn_graph_builder.build_rgcn_graph().

Architecture (parallel, unchanged from model.py):
    RGCN branch:
        per-timestep RGCN(×3, 9 relations, 1→10) → stack → mean pool
        → Linear(2840→64)
    LSTM branch:
        raw prices → LSTM(284→64)
    Fusion:
        Concat(RGCN_64, LSTM_64) → FC(128→64→N_commodities)

Key differences from model.py:
  - SpatialRGCN uses RGCNConv instead of MeanConcatGCNLayer.
  - RGCNConv in this PyG version takes (x, edge_index, edge_type) only
    — no edge_weight, no built-in self-loop handling (self-loops are
    added explicitly in rgcn_graph_builder).
  - Forward accepts edge_type as the third positional argument
    (train.py passes it through in the `edge_weight` parameter position).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import RGCNConv

from cmgm.config import (
    GCN_INPUT_DIM, GCN_OUTPUT_DIM, GCN_NUM_LAYERS,
    GCN_DROPOUT,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    FC_HIDDEN_DIM,
)
from cmgm.graph.rgcn_graph_builder import NUM_RELATIONS


# =============================================================================
# Spatial: RGCN (per time step)
# =============================================================================

class RGCNLayer(nn.Module):
    """
    Single RGCN layer wrapping torch_geometric.nn.RGCNConv.

    RGCNConv uses relation-specific weight matrices W_r so that each of the
    9 relation types has its own transformation.

    Self-loops are pre-added in the graph builder (rgcn_graph_builder) —
    no built-in self-loop handling from RGCNConv (not supported in this
    PyG version).

    Shape:
        Input:  x (B*N, in_dim), ei (2, E_batched), et (E_batched)
        Output: x (B*N, out_dim)
    """

    def __init__(self, in_dim: int, out_dim: int,
                 num_relations: int = NUM_RELATIONS,
                 dropout: float = GCN_DROPOUT):
        super().__init__()
        self.conv = RGCNConv(
            in_channels=in_dim,
            out_channels=out_dim,
            num_relations=num_relations,
            num_bases=None,          # full rank — one W_r per relation
            aggr='mean',             # mean over neighbor messages
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B*N, in_dim) — flattened node features
            edge_index: (2, E_batched) — batched edges
            edge_type: (E_batched,) — batched relation types (0–8)

        Returns:
            out: (B*N, out_dim)
        """
        out = self.conv(x, edge_index, edge_type)
        out = F.relu(out)
        out = self.dropout(out)
        return out


class SpatialRGCN(nn.Module):
    """
    3× RGCN layers (1→10, 10→10, 10→10) with graph batching.

    Handles per-sample offsetting of edge_index so a single static graph
    is replicated across the batch dimension.

    Input per time step:  (B, N, 1)
    Output per time step: (B, N * GCN_OUTPUT_DIM)  = (B, 2840)
    """

    def __init__(self, num_relations: int = NUM_RELATIONS):
        super().__init__()
        self.num_relations = num_relations
        self.layers = nn.ModuleList([
            RGCNLayer(GCN_INPUT_DIM, GCN_OUTPUT_DIM, num_relations, GCN_DROPOUT),
            *[RGCNLayer(GCN_OUTPUT_DIM, GCN_OUTPUT_DIM, num_relations, GCN_DROPOUT)
              for _ in range(GCN_NUM_LAYERS - 1)],
        ])

    def forward(self, x_t: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor,
                batch_size: int, num_nodes: int) -> torch.Tensor:
        """
        Process one time step with batched RGCN.

        Args:
            x_t:       (B, N, 1) features at time t
            edge_index: (2, E) original (unbatched) edge_index
            edge_type:  (E,) original (unbatched) edge_type
            batch_size: B
            num_nodes:  N

        Returns:
            out: (B, N * GCN_OUTPUT_DIM)
        """
        # Flatten batch: (B * N, in_dim)
        x_flat = x_t.reshape(batch_size * num_nodes, -1)  # (B*N, 1)
        device = x_flat.device

        # Batch graph: replicate across batch with node-index offsets
        batched_ei, batched_et = self._batch_graph(
            edge_index, edge_type, num_nodes, batch_size, device,
        )

        # Sequential RGCN layers
        for layer in self.layers:
            x_flat = layer(x_flat, batched_ei, batched_et)

        # Reshape back: (B, N * out_dim)
        return x_flat.reshape(batch_size, num_nodes * GCN_OUTPUT_DIM)

    @staticmethod
    def _batch_graph(edge_index: torch.Tensor,
                     edge_type: torch.Tensor,
                     num_nodes: int,
                     batch_size: int,
                     device: torch.device):
        """
        Replicate a single graph across the batch dimension.

        For each sample s in [0, B):
            node_idx' = node_idx + s * num_nodes

        edge_index: (2, E) → (2, E*B)
        edge_type:  (E,)   → (E*B,)
        """
        num_edges = edge_index.shape[1]

        # Per-sample offset = sample_idx * num_nodes
        offsets = torch.arange(batch_size, device=device) * num_nodes      # (B,)
        offset_repeat = offsets.repeat_interleave(num_edges)               # (E*B,)

        # Batch edge_index
        batched_ei = edge_index.repeat(1, batch_size)                     # (2, E*B)
        batched_ei = batched_ei + offset_repeat.unsqueeze(0)              # (2, E*B)

        # Batch edge_type — no offset needed, just repeat
        batched_et = edge_type.repeat(batch_size)                          # (E*B,)

        return batched_ei, batched_et


# =============================================================================
# Temporal: LSTM (unchanged from model.py)
# =============================================================================

class TemporalLSTM(nn.Module):
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
# Output: Fusion FC (unchanged from model.py)
# =============================================================================

class OutputLayer(nn.Module):
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
# RGCN-CMGM: Parallel RGCN || LSTM
# =============================================================================

class CMGM_RGCN(nn.Module):
    """
    RGCN-CMGM: Parallel RGCN || LSTM → Concat → FC.

    Forward argument convention:
      The third positional argument is **edge_type** (NOT edge_weight).
      This matches how train.py passes the `edge_weight` argument through.
      Edge weights (correlation magnitude) are stored in a model buffer
      `_rgcn_edge_weight` and accessed inside forward().

    Args passed to train(model, ..., edge_index, edge_type, device, ...):
      - edge_index → second arg of model.forward
      - edge_type  → third arg of model.forward (called edge_type)
    """

    def __init__(self, num_nodes: int, n_commodities: int,
                 num_relations: int = NUM_RELATIONS):
        super().__init__()
        self.num_nodes = num_nodes
        self.n_commodities = n_commodities

        # ── RGCN branch ──
        self.spatial = SpatialRGCN(num_relations=num_relations)
        gcn_flat_dim = num_nodes * GCN_OUTPUT_DIM       # 284 * 10 = 2840
        self.gcn_reduce = nn.Linear(gcn_flat_dim, LSTM_HIDDEN_DIM)  # 2840→64

        # ── LSTM branch ──
        self.temporal = TemporalLSTM(num_nodes)

        # ── Fusion ──
        fusion_dim = LSTM_HIDDEN_DIM * 2               # 128 = 64 + 64
        self.output = OutputLayer(fusion_dim, n_commodities)

    def forward(self, x: torch.Tensor,
                edge_index: torch.Tensor,
                edge_type: torch.Tensor = None,
                debug: bool = False) -> torch.Tensor:
        """
        Args:
            x:          (B, T, N, 1)  — normalized closing prices
            edge_index: (2, E)        — sparse graph edges (with self-loops)
            edge_type:  (E,)          — relation type per edge (0–8)
            debug:      bool          — print shape info

        Returns:
            pred: (B, N_commodities) — next-day commodity price predictions
        """
        batch_size, seq_len, num_nodes, _ = x.shape

        # ── RGCN Branch ──
        temporal_outputs = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]           # (B, N, 1)
            out_t = self.spatial(
                x_t, edge_index, edge_type,
                batch_size, num_nodes,
            )                              # (B, N*10)
            temporal_outputs.append(out_t)

        gcn_stack = torch.stack(temporal_outputs, dim=1)  # (B, T, N*10)
        gcn_pooled = gcn_stack.mean(dim=1)                 # (B, N*10) = (B, 2840)
        gcn_out = self.gcn_reduce(gcn_pooled)              # (B, 64)

        # ── LSTM Branch ──
        x_seq = x.squeeze(-1)                              # (B, T, N)
        lstm_out = self.temporal(x_seq)                    # (B, 64)

        # ── Fusion ──
        combined = torch.cat([gcn_out, lstm_out], dim=-1)  # (B, 128)
        pred = self.output(combined)                       # (B, N_commodities)

        if debug:
            print(f"  RGCN→({list(gcn_out.shape)}) || "
                  f"LSTM→({list(lstm_out.shape)}) → "
                  f"Concat({list(combined.shape)}) → "
                  f"FC → ({list(pred.shape)})")

        return pred
