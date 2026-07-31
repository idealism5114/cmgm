"""
HGT-CMGM Model — Parallel HGT || LSTM → Concat → FC.

Replaces homogeneous GCN/RGCN with HGTConv (Heterogeneous Graph
Transformer) operating on a HeteroData graph with 3 node types
(stock, bond, future) and 9 edge types.

Architecture:
    HGT branch (per time step):
        Input Linear(in_dim→hidden_dim) per node type
        → L× HGTConv(hidden_dim→hidden_dim, heads) → ReLU → Dropout
        → per-type mean pool → concat(3*hidden_dim) → Linear(3*hidden_dim→hidden_dim)
        → stack over T → mean pool → (B, hidden_dim)

    LSTM branch:
        price channel (feature=0) → LSTM(N→hidden_dim) → (B, hidden_dim)

    Fusion:
        Concat(hidden_dim+hidden_dim) → FC(2*hidden_dim→hidden_dim→N_commodities)

Forward argument convention (compatible with train.py):
    model(X_batch, hetero_data, dummy, debug)
    ── X_batch:     (B, T, N, in_dim) — feature matrix
    ── hetero_data:  HeteroData       — passed as the `edge_index` argument
                     (HeteroData.to(device) is called by train.py)
    ── dummy:        ignored          — occupies the `edge_weight` slot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HGTConv

from cmgm.config import (
    GCN_DROPOUT,
    LSTM_HIDDEN_DIM, LSTM_NUM_LAYERS, LSTM_DROPOUT,
    FC_HIDDEN_DIM,
)
from cmgm.data.feature_builder import NUM_FEATURES

# ── All 9 edge types in the same order used by hetero_graph_builder ──
_EDGE_TYPES = [
    ('stock',  'corr',  'stock'),
    ('future', 'corr',  'future'),
    ('bond',   'corr',  'bond'),
    ('stock',  'cross', 'future'),
    ('stock',  'cross', 'bond'),
    ('future', 'cross', 'bond'),
    ('future', 'cross', 'stock'),
    ('bond',   'cross', 'stock'),
    ('bond',   'cross', 'future'),
]
_NODE_TYPES = ['stock', 'future', 'bond']

_METADATA = (_NODE_TYPES, _EDGE_TYPES)


# =============================================================================
# Input Projection (per node type)
# =============================================================================

class _InputProjection(nn.Module):
    """Project in_dim features to hidden_dim per node type."""

    def __init__(self, in_dim: int, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.stock_proj  = nn.Linear(in_dim, hidden_dim)
        self.bond_proj   = nn.Linear(in_dim, hidden_dim)
        self.future_proj = nn.Linear(in_dim, hidden_dim)

    def forward(self, x_t: torch.Tensor,
                n_stock: int, n_bond: int) -> dict:
        """
        Args:
            x_t:    (B, N_total, in_dim)
            n_stock: number of stock nodes
            n_bond:  number of bond nodes

        Returns:
            x_dict: {str: (B * N_type, hidden_dim)}
        """
        H = self.hidden_dim
        x_dict = {}

        # stock:   (B, n_stock, in_dim) → Linear → (B, n_stock, H) → (B*n_stock, H)
        s = x_t[:, :n_stock, :]
        x_dict['stock'] = self.stock_proj(s).reshape(-1, H)

        # bond:    (B, n_bond, in_dim) → Linear → (B, n_bond, H) → (B*n_bond, H)
        b = x_t[:, n_stock:n_stock + n_bond, :]
        x_dict['bond'] = self.bond_proj(b).reshape(-1, H)

        # future:  (B, n_future, in_dim) → Linear → (B, n_future, H) → (B*n_future, H)
        f = x_t[:, n_stock + n_bond:, :]
        x_dict['future'] = self.future_proj(f).reshape(-1, H)

        return x_dict


# =============================================================================
# Spatial: HGT per time step
# =============================================================================

class SpatialHGT(nn.Module):
    """
    2× HGTConv(64→64, 4 heads) with graph batching.

    Input per time step:  (B, N_total, 1) + HeteroData
    Output per time step: (B, 64)  — aggregated heterogeneous embedding
    """

    def __init__(self, in_dim: int = 1, hidden_dim: int = 64,
                 num_heads: int = 4, num_layers: int = 2,
                 dropout: float = GCN_DROPOUT):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.dropout_p  = dropout

        # Input projection in_dim → hidden_dim
        self.input_proj = _InputProjection(in_dim, hidden_dim)

        # HGTConv layers
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(HGTConv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                metadata=_METADATA,
                heads=num_heads,
            ))

        self.dropout = nn.Dropout(dropout)

        # Per-type mean → concat → project 3*hidden_dim → hidden_dim
        self.type_agg = nn.Linear(3 * hidden_dim, hidden_dim)

    def forward(self, x_t: torch.Tensor,
                hetero_data,
                batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Args:
            x_t:         (B, N_total, in_dim) features at time t
            hetero_data: HeteroData with .edge_index per type
            batch_size:  B
            device:      torch device

        Returns:
            h_t: (B, hidden_dim) — aggregated node embedding for this time step
        """
        # ── 1. Input projection ──
        n_stock = hetero_data['stock'].num_nodes
        n_bond  = hetero_data['bond'].num_nodes
        x_dict  = self.input_proj(x_t, n_stock, n_bond)
        #   x_dict['stock']:  (B*n_stock,  hidden_dim)
        #   x_dict['bond']:   (B*n_bond,   hidden_dim)
        #   x_dict['future']: (B*n_future, hidden_dim)

        # ── 2. Batch edge_index_dict ──
        # For each edge type, offset node indices per sample
        batched_ei_dict = {}
        for etype in hetero_data.edge_types:
            src_type, _, dst_type = etype
            ei = hetero_data[etype].edge_index              # (2, E_local)
            num_edges = ei.shape[1]

            n_src = hetero_data[src_type].num_nodes
            n_dst = hetero_data[dst_type].num_nodes

            if num_edges == 0:
                batched_ei_dict[etype] = torch.empty(2, 0, dtype=torch.long, device=device)
                continue

            # Per-sample offsets
            src_off = torch.arange(batch_size, device=device) * n_src
            dst_off = torch.arange(batch_size, device=device) * n_dst

            # Repeat edge_index for batch_size copies
            batched_ei = ei.to(device).repeat(1, batch_size)
            # Offset each copy
            batched_ei[0] += src_off.repeat_interleave(num_edges)
            batched_ei[1] += dst_off.repeat_interleave(num_edges)
            # batched_ei: (2, E_local * B)

            batched_ei_dict[etype] = batched_ei

        # ── 3. HGTConv layers ──
        for conv in self.convs:
            x_dict = conv(x_dict, batched_ei_dict)
            #   x_dict['stock']:  (B*n_stock,  hidden_dim)
            #   x_dict['bond']:   (B*n_bond,   hidden_dim)
            #   x_dict['future']: (B*n_future, hidden_dim)
            x_dict = {k: F.relu(v) for k, v in x_dict.items()}
            x_dict = {k: self.dropout(v) for k, v in x_dict.items()}

        # ── 4. Per-type mean pooling → per-sample ──
        n_future = hetero_data['future'].num_nodes
        stock_pooled = x_dict['stock'].reshape(batch_size, n_stock, -1).mean(dim=1)
        #   (B, hidden_dim)
        bond_pooled = x_dict['bond'].reshape(batch_size, n_bond, -1).mean(dim=1)
        #   (B, hidden_dim)
        future_pooled = x_dict['future'].reshape(batch_size, n_future, -1).mean(dim=1)
        #   (B, hidden_dim)

        type_concat = torch.cat([stock_pooled, bond_pooled, future_pooled], dim=-1)
        #   (B, 3 * hidden_dim) = (B, 192)
        h_t = self.type_agg(type_concat)
        #   (B, hidden_dim) = (B, 64)

        return h_t


# =============================================================================
# Temporal: LSTM (identical to model.py)
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
# Output: Fusion FC (identical to model.py)
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
# HGT-CMGM: Parallel HGT || LSTM
# =============================================================================

class CMGM_HGT(nn.Module):
    """
    HGT-CMGM: Parallel HGT || LSTM → Concat → FC.

    Heterogeneous Graph Transformer captures cross-market relational patterns
    using 9 edge types (3 intra-market + 6 cross-market).

    - HGT branch receives ALL in_dim features per node.
    - LSTM branch receives only the PRICE channel (feature index 0) to keep
      its input size identical to the original CMGM.

    Forward call convention (compatible with train.py / evaluate.py):
        model(x, hetero_data, dummy, debug)
          x:    (B, T, N, in_dim)  feature matrix
          hetero_data: HeteroData, passed in the `edge_index` slot
          dummy: ignored tensor (fills the `edge_weight` slot)
    """

    def __init__(self, num_nodes: int, n_commodities: int,
                 hidden_dim: int = 64, num_heads: int = 4,
                 num_layers: int = 2, in_dim: int = NUM_FEATURES):
        super().__init__()
        self.num_nodes      = num_nodes
        self.n_commodities  = n_commodities
        self.hidden_dim     = hidden_dim
        self.in_dim         = in_dim

        # ── HGT branch (all features) ──
        self.spatial = SpatialHGT(in_dim=in_dim, hidden_dim=hidden_dim,
                                  num_heads=num_heads, num_layers=num_layers)

        # ── LSTM branch (price only: feature index 0) ──
        self.temporal = TemporalLSTM(num_nodes)

        # ── Fusion ──
        fusion_dim = LSTM_HIDDEN_DIM * 2        # 128 = 64 + 64
        self.output = OutputLayer(fusion_dim, n_commodities)

    def forward(self, x: torch.Tensor,
                hetero_data=None,
                dummy=None,
                debug: bool = False) -> torch.Tensor:
        """
        Args:
            x:          (B, T, N, in_dim)  feature matrix
            hetero_data: HeteroData   graph structure (edge_index_dict + num_nodes)
            dummy:      ignored       (placeholder for train.py's edge_weight)
            debug:      bool          print shape info

        Returns:
            pred: (B, N_commodities) next-day commodity price predictions
        """
        batch_size, seq_len, num_nodes, n_feat = x.shape
        device = x.device

        # ── HGT Branch (all features) ──
        temporal_embeds = []
        for t in range(seq_len):
            x_t = x[:, t, :, :]                       # (B, N, in_dim)
            h_t = self.spatial(x_t, hetero_data,
                                batch_size, device)    # (B, hidden_dim)
            temporal_embeds.append(h_t)

        # Stack over time and mean pool
        hgt_stack = torch.stack(temporal_embeds, dim=1)   # (B, T, hidden_dim)
        hgt_out   = hgt_stack.mean(dim=1)                 # (B, hidden_dim) = (B, 64)

        # ── LSTM Branch (price channel only) ──
        x_seq = x[:, :, :, 0]                             # (B, T, N)
        lstm_out = self.temporal(x_seq)                   # (B, 64)

        # ── Fusion ──
        combined = torch.cat([hgt_out, lstm_out], dim=-1)  # (B, 128)
        pred = self.output(combined)                       # (B, N_commodities)

        if debug:
            print(f"  HGT(F={n_feat})→({list(hgt_out.shape)}) || "
                  f"LSTM(price)→({list(lstm_out.shape)}) → "
                  f"Concat({list(combined.shape)}) → "
                  f"FC → ({list(pred.shape)})")

        return pred
