"""
Relational graph builder for RGCN-CMGM.

Extends the standard graph_builder.py with relation type information.
Defines 9 edge types based on source/target node market categories.

Node type mapping:
    Stock = 0   (columns 0 ~ N_stock-1)
    Future = 1  (columns N_stock ~ N_stock+N_future-1)  — commodity futures
    Bond = 2    (columns N_stock+N_future ~ end)

Relation type mapping (9 types):
    0: Stock→Stock    1: Future→Future    2: Bond→Bond
    3: Stock→Future   4: Stock→Bond       5: Future→Bond
    6: Future→Stock   7: Bond→Stock       8: Bond→Future

Notes:
  - Self-loops are added to the graph with self-relation types (0, 1, 2).
  - Edge weights are raw signed correlation values (no symmetric normalization).
  - RGCNConv in this PyG version does NOT support edge_weight or built-in
    self-loop handling, so both are done explicitly in the graph builder.
"""

import numpy as np
import torch
from typing import Dict, Tuple

from cmgm.config import TOP_K_EDGES
from cmgm.graph.graph_builder import (
    CORRELATION_REGISTRY,
    top_k_thresholding,
    symmetrize,
)

# ── Node type constants ──
NODE_STOCK = 0
NODE_FUTURE = 1   # commodity futures (the prediction target)
NODE_BOND = 2

# ── Relation mapping: [src_type, dst_type] → relation_id ──
# Rows = source type, Columns = destination type
_RELATION_MAP = np.array([
    [0, 3, 4],   # src=Stock  → {Stock=0, Future=3, Bond=4}
    [6, 1, 5],   # src=Future → {Stock=6, Future=1, Bond=5}
    [7, 8, 2],   # src=Bond   → {Stock=7, Future=8, Bond=2}
], dtype=np.int64)

RELATION_LABELS = [
    'Stock→Stock',
    'Future→Future',
    'Bond→Bond',
    'Stock→Future',
    'Stock→Bond',
    'Future→Bond',
    'Future→Stock',
    'Bond→Stock',
    'Bond→Future',
]
NUM_RELATIONS = 9


def get_node_types(n_nodes: int, market_indices: Dict) -> np.ndarray:
    """
    Build per-node-type array from market column ranges.

    Returns:
        node_type: (N,) int64, values in {0=Stock, 1=Future, 2=Bond}
    """
    node_type = np.zeros(n_nodes, dtype=np.int64)  # default Stock

    for market_name, node_val in [('bond', NODE_BOND),
                                   ('commodity', NODE_FUTURE)]:
        start, end = market_indices[market_name]
        node_type[start:end] = node_val

    return node_type


def build_rgcn_graph(
    returns: np.ndarray,
    market_indices: Dict,
    method: str = 'volatility_adjusted',
    top_k: int = TOP_K_EDGES,
) -> Dict:
    """
    Build relational graph for RGCN-CMGM.

    Pipeline (modified Section 3.2):
        returns → correlation → top-k (signed) → symmetrize
        → edge_index + edge_weight + edge_type

    Differences from standard build_graph:
      - No self-loops (RGCNConv handles them internally)
      - No symmetric normalization (raw signed correlations as edge_weight)
      - Adds edge_type: (E,) with values in [0, 8]

    Args:
        returns: (T, N) training-asset returns
        market_indices: dict mapping market names to (start, end) col indices
        method: correlation strategy (default: volatility_adjusted)
        top_k: edges to retain per node

    Returns:
        dict with keys:
          edge_index: (2, E)   — sparse adjacency
          edge_type:  (E,)     — relation type per edge
          edge_weight: (E,)    — raw signed correlation value
          corr_raw:   (N, N)   — full correlation matrix
          node_type:  (N,)     — node-type array
          method: str
    """
    print(f"\n{'=' * 60}")
    print(f"RGCN Graph Construction ({method})")
    print(f"{'=' * 60}")

    N = returns.shape[1]

    # ── Step 1: Correlation matrix ──
    print(f"\n[RGCN Step 1] Computing {method} correlation...")
    corr_fn = CORRELATION_REGISTRY[method]
    corr = corr_fn(returns)                      # (N, N)

    # ── Step 2: Top-k thresholding (signed) ──
    print(f"\n[RGCN Step 2] Top-{top_k} thresholding (signed)...")
    adj_values, selected_mask = top_k_thresholding(corr, k=top_k)  # each (N, N)

    # ── Step 3: Symmetrize ──
    print(f"\n[RGCN Step 3] Symmetrizing adjacency...")
    adj_sym = symmetrize(adj_values, selected_mask)  # (N, N)

    # ── Step 4: Add self-loops ──
    # RGCNConv in this PyG version doesn't have built-in self-loop support,
    # so we add self-loops explicitly.
    print(f"\n[RGCN Step 4] Adding self-loops...")
    for i in range(N):
        adj_sym[i, i] = 1.0

    # ── Step 5: edge_index + edge_weight (no symmetric normalization) ──
    print(f"\n[RGCN Step 5] Converting to edge_index / edge_weight...")
    rows, cols = np.where(np.abs(adj_sym) > 1e-8)
    weights = adj_sym[rows, cols]                # raw signed correlation (1.0 for self)

    edge_index = torch.LongTensor(np.stack([rows, cols], axis=0))  # (2, E)
    edge_weight = torch.FloatTensor(weights)                        # (E,)
    print(f"     edge_index: (2, {edge_index.shape[1]}), "
          f"edge_weight: [{edge_weight.min():.4f}, {edge_weight.max():.4f}]")

    # ── Step 6: edge_type per edge ──
    print(f"\n[RGCN Step 6] Computing edge types ({NUM_RELATIONS} relations)...")
    node_type = get_node_types(N, market_indices)
    src_types = node_type[rows]                  # (E,)
    dst_types = node_type[cols]                  # (E,)
    edge_type_np = _RELATION_MAP[src_types, dst_types]  # (E,)
    edge_type = torch.LongTensor(edge_type_np)           # (E,)

    # ── Print relation statistics ──
    for r in range(NUM_RELATIONS):
        count = (edge_type_np == r).sum()
        if count > 0:
            print(f"     [{r}] {RELATION_LABELS[r]}: {count:>6d} edges")

    stock_start, stock_end = market_indices['stock']
    bond_start, bond_end = market_indices['bond']
    commodity_start, commodity_end = market_indices['commodity']
    print(f"\n[RGCN] Node composition: "
          f"Stock={stock_end - stock_start}, "
          f"Bond={bond_end - bond_start}, "
          f"Future={commodity_end - commodity_start}  (total={N})")
    print(f"[RGCN] Total edges: {edge_index.shape[1]} "
          f"(including {N} self-loops), "
          f"weight range: [{edge_weight.min():.4f}, {edge_weight.max():.4f}]")

    return {
        'edge_index': edge_index,
        'edge_type': edge_type,
        'edge_weight': edge_weight,
        'corr_raw': corr,
        'node_type': node_type,
        'method': method,
    }
