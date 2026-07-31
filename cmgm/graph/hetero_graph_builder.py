"""
Heterogeneous graph builder for HGT-CMGM.

Builds a torch_geometric.data.HeteroData object from the volatility_adjusted
correlation matrix, preserving signed cross-market relationships.

Node types:
    'stock'   — CSI 300 constituent stocks  (indices 0…N_stock-1)
    'bond'    — treasury bond futures        (indices N_stock…N_stock+N_bond-1)
    'future'  — commodity futures            (indices N_stock+N_bond…end)

Edge types (9 directed relations):
    (stock,  corr,  stock)    intra-market: Stock → Stock
    (future, corr,  future)   intra-market: Future → Future
    (bond,   corr,  bond)     intra-market: Bond → Bond
    (stock,  cross, future)   cross-market: Stock → Future
    (stock,  cross, bond)     cross-market: Stock → Bond
    (future, cross, bond)     cross-market: Future → Bond
    (future, cross, stock)    cross-market: Future → Stock
    (bond,   cross, stock)    cross-market: Bond → Stock
    (bond,   cross, future)   cross-market: Bond → Future

All 9 edge types are declared in metadata even if some have zero edges.
Self-loops are added with appropriate intra-type relation (corr).

Pipeline:
    returns → volatility_adjusted correlation → top-k (signed) →
    symmetrize → self-loops → HeteroData
"""

import numpy as np
import torch
from typing import Dict

from torch_geometric.data import HeteroData

from cmgm.config import TOP_K_EDGES
from cmgm.graph.graph_builder import (
    CORRELATION_REGISTRY,
    top_k_thresholding,
    symmetrize,
)

# ── All edge type triples (ordered as they appear in metadata) ──
EDGE_TYPE_LIST = [
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
NODE_TYPE_LIST = ['stock', 'future', 'bond']


def _build_type_mappings(market_indices: Dict, n_total: int):
    """
    Create global→local lookup arrays.

    Returns:
        type_names:  (N_total,) str  — 'stock', 'bond', or 'future'
        local_ids:   (N_total,) int  — index within the node's own type
    """
    stock_end     = market_indices['stock'][1]
    bond_start, bond_end = market_indices['bond']
    future_start, future_end = market_indices['commodity']

    n_stock  = stock_end
    n_bond   = bond_end - bond_start
    n_future = future_end - future_start

    type_names = (['stock'] * n_stock
                  + ['bond'] * n_bond
                  + ['future'] * n_future)
    local_ids  = list(range(n_stock)) + list(range(n_bond)) + list(range(n_future))

    type_counts = {'stock': n_stock, 'bond': n_bond, 'future': n_future}
    return type_names, local_ids, type_counts


def build_heterodata(
    returns: np.ndarray,
    market_indices: Dict,
    method: str = 'volatility_adjusted',
    top_k: int = TOP_K_EDGES,
) -> HeteroData:
    """
    Build a HeteroData object from cross-market returns.

    Pipeline (unchanged from graph_builder):
        returns → correlation → top-k (signed) → symmetrize → self-loops

    Then converts the global sparse adjacency into per-(src,rel,dst) edge_index
    with local (per-type) node indices.

    Args:
        returns:        (T, N) training-asset returns
        market_indices: {market: (start, end)} column ranges
        method:         correlation strategy (default volatility_adjusted)
        top_k:          edges to retain per node

    Returns:
        hetero_data: HeteroData with .num_nodes and .edge_index per type
    """
    print(f"\n{'=' * 60}")
    print(f"Heterogeneous Graph Construction ({method})")
    print(f"{'=' * 60}")

    N = returns.shape[1]

    # ── Step 1: Correlation ──
    print(f"\n[HGT Step 1] Computing {method} correlation...")
    corr_fn = CORRELATION_REGISTRY[method]
    corr = corr_fn(returns)                             # (N, N) signed

    # ── Step 2: Top-k thresholding (signed) ──
    print(f"\n[HGT Step 2] Top-{top_k} thresholding (signed)...")
    adj_values, selected_mask = top_k_thresholding(corr, k=top_k)

    # ── Step 3: Symmetrize ──
    print(f"\n[HGT Step 3] Symmetrizing adjacency...")
    adj_sym = symmetrize(adj_values, selected_mask)     # (N, N)

    # ── Step 4: Self-loops ──
    print(f"\n[HGT Step 4] Adding self-loops...")
    np.fill_diagonal(adj_sym, 1.0)

    # ── Step 5: Global → per-type local indices ──
    print(f"\n[HGT Step 5] Converting to HeteroData...")
    type_names, local_ids, type_counts = _build_type_mappings(market_indices, N)

    # Extract non-zero edges
    rows, cols = np.where(np.abs(adj_sym) > 1e-8)       # global node indices

    # Group edges by (src_type, rel, dst_type)
    edge_dict: Dict[tuple, list] = {}
    for u, v in zip(rows, cols):
        utype = type_names[u]
        vtype = type_names[v]
        rel   = 'corr' if utype == vtype else 'cross'
        key   = (utype, rel, vtype)

        if key not in edge_dict:
            edge_dict[key] = [[], []]
        edge_dict[key][0].append(int(local_ids[u]))
        edge_dict[key][1].append(int(local_ids[v]))

    # ── Step 6: Build HeteroData ──
    data = HeteroData()
    for ntype in NODE_TYPE_LIST:
        data[ntype].num_nodes = type_counts[ntype]

    for key in EDGE_TYPE_LIST:
        if key in edge_dict:
            src_list, dst_list = edge_dict[key]
            data[key].edge_index = torch.LongTensor([src_list, dst_list])
        else:
            # Edge type with zero edges — store empty (2, 0) so HGTConv
            # creates parameters for it (it will simply not contribute).
            n_src = type_counts[key[0]]
            data[key].edge_index = torch.empty(2, 0, dtype=torch.long)

    # ── Print summary ──
    n_total = sum(type_counts.values())
    print(f"\n[HGT] Node composition:")
    for ntype in NODE_TYPE_LIST:
        print(f"       {ntype}: {type_counts[ntype]:>4d}")
    print(f"       total: {n_total}")

    print(f"[HGT] Edge types:")
    for key in EDGE_TYPE_LIST:
        e = data[key].edge_index
        print(f"       {key[0]:>6s} → {key[2]:>6s} ({key[1]:>5s}): "
              f"{e.shape[1]:>5d} edges")

    return data
