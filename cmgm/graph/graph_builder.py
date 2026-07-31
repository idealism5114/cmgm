"""
Static graph construction for CMGM (Section 3.2).

Builds a unified adjacency matrix from intra-market and cross-market correlations.

Four correlation strategies (Section 3.2.1–3.2.4):
  1. Linear Pearson Correlation
  2. Volatility-Adjusted Correlation
  3. Skewness/Kurtosis-Adjusted Correlation
  4. Dynamic Time-Series Correlation

Graph construction pipeline:
  Returns → Correlation Matrix → Top-k Thresholding →
  Symmetrization → Self-loops → Normalization (D^{-1/2} A D^{-1/2})
  → edge_index, edge_weight

IMPORTANT: Graph is constructed from TRAINING data only to prevent data leakage.
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, Callable
from scipy.stats import pearsonr, skew, kurtosis

from cmgm.config import (
    CORRELATION_METHOD, TOP_K_EDGES,
    ROLLING_WINDOW, RANDOM_SEED
)
from cmgm.graph.garch_correlation import dcc_garch_correlation


def pearson_correlation(returns: np.ndarray) -> np.ndarray:
    """
    Section 3.2.1: Linear Pearson Correlation.

    Computes the Pearson correlation matrix from asset returns.

    ρ_ij = cov(r_i, r_j) / (σ_i * σ_j)

    Args:
        returns: shape (T, N) — asset return time series

    Returns:
        corr: shape (N, N) — Pearson correlation matrix
    """
    # Center the returns
    returns_centered = returns - returns.mean(axis=0, keepdims=True)  # (T, N)

    # Compute covariance matrix
    cov = returns_centered.T @ returns_centered / (returns.shape[0] - 1)  # (N, N)

    # Standard deviations
    std = np.sqrt(np.diag(cov))  # (N,)

    # Handle zero-variance assets: replace zero std with epsilon
    std = np.maximum(std, 1e-8)

    # Correlation matrix
    corr = cov / np.outer(std, std)  # (N, N)

    # Handle any remaining NaN (shouldn't happen after std fix, but safety)
    corr = np.nan_to_num(corr, nan=0.0)

    # Clip to [-1, 1] for numerical stability
    corr = np.clip(corr, -1.0, 1.0)

    print(f"     [Pearson] Correlation matrix shape: {corr.shape}, range: [{corr.min():.4f}, {corr.max():.4f}]")
    return corr


def volatility_adjusted_correlation(returns: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    """
    Section 3.2.2: Volatility-Adjusted Correlation.

    Steps:
      1. Compute rolling volatility (rolling std of returns)
      2. Normalize returns by dividing by rolling volatility
      3. Compute Pearson correlation on normalized returns

    This reduces the impact of high-volatility periods on correlation estimates.

    Args:
        returns: shape (T, N) — asset return time series
        window: rolling window size in trading days

    Returns:
        corr: shape (N, N) — volatility-adjusted correlation matrix
    """
    T, N = returns.shape

    # Step 1: Compute rolling volatility (rolling standard deviation)
    # Pad to handle edges
    rolling_vol = np.zeros_like(returns)
    for i in range(N):
        # Compute expanding std for early observations, rolling std after window
        for t in range(T):
            if t < window:
                rolling_vol[t, i] = returns[:t+1, i].std() if t > 0 else 1e-8
            else:
                rolling_vol[t, i] = returns[t-window:t, i].std()

    # Ensure no zero volatility
    rolling_vol = np.maximum(rolling_vol, 1e-8)

    # Step 2: Normalize returns by volatility
    normalized_returns = returns / rolling_vol  # (T, N)

    # Step 3: Compute Pearson correlation on normalized returns
    corr = pearson_correlation(normalized_returns)

    print(f"     [VolAdj] Volatility-adjusted correlation computed with window={window}")
    return corr


def skewness_kurtosis_adjusted_correlation(returns: np.ndarray) -> np.ndarray:
    """
    Section 3.2.3: Skewness/Kurtosis-Adjusted Correlation.

    Adjusts the Pearson correlation to account for non-normality in return
    distributions by incorporating skewness and kurtosis adjustments.

    The adjustment factor accounts for the deviation from normality:
      ρ_adj = ρ * (1 + δ_skew * skew_i * skew_j + δ_kurt * (kurt_i - 3) * (kurt_j - 3))

    where δ_skew and δ_kurt are scaling parameters.

    Args:
        returns: shape (T, N) — asset return time series

    Returns:
        corr: shape (N, N) — skewness/kurtosis-adjusted correlation matrix
    """
    T, N = returns.shape

    # Step 1: Base Pearson correlation
    base_corr = pearson_correlation(returns)  # (N, N)

    # Step 2: Compute skewness and excess kurtosis for each asset
    skewness_vals = np.array([skew(returns[:, i]) for i in range(N)])       # (N,)
    kurtosis_vals = np.array([kurtosis(returns[:, i], fisher=True) for i in range(N)])  # (N,) excess kurtosis

    # Step 3: Adjustment factors
    # These scaling parameters control the influence of higher moments
    delta_skew = 0.1
    delta_kurt = 0.05

    # Skewness adjustment matrix
    skew_adj = delta_skew * np.outer(skewness_vals, skewness_vals)  # (N, N)

    # Kurtosis adjustment matrix (excess kurtosis)
    kurt_adj = delta_kurt * np.outer(kurtosis_vals, kurtosis_vals)  # (N, N)

    # Combined adjustment
    adj_matrix = 1.0 + skew_adj + kurt_adj  # (N, N)

    # Step 4: Apply adjustment
    corr = base_corr * adj_matrix  # (N, N)

    # Clip to valid range
    corr = np.clip(corr, -1.0, 1.0)

    print(f"     [SkewKurt] Adjusted correlation computed (skew range: [{skewness_vals.min():.4f}, {skewness_vals.max():.4f}])")
    return corr


def dynamic_correlation(returns: np.ndarray, window: int = ROLLING_WINDOW) -> np.ndarray:
    """
    Section 3.2.4: Dynamic Time-Series Correlation.

    Computes time-varying correlations using a rolling window approach.
    The final correlation matrix is the time-averaged dynamic correlation.

    Steps:
      1. For each time window, compute the Pearson correlation
      2. Average across all windows

    Args:
        returns: shape (T, N) — asset return time series
        window: rolling window size

    Returns:
        corr: shape (N, N) — averaged dynamic correlation matrix
    """
    T, N = returns.shape

    if T <= window:
        print(f"     [Dynamic] Warning: T={T} <= window={window}, using full sample")
        return pearson_correlation(returns)

    # Compute rolling correlations
    n_windows = T - window + 1
    rolling_corrs = np.zeros((n_windows, N, N))

    for t in range(n_windows):
        window_returns = returns[t:t + window]  # (window, N)
        rolling_corrs[t] = pearson_correlation(window_returns)

    # Average across windows
    corr = rolling_corrs.mean(axis=0)  # (N, N)

    print(f"     [Dynamic] Dynamic correlation averaged over {n_windows} windows")
    return corr


# =============================================================================
# Strategy registry — maps config strings to functions
# =============================================================================
CORRELATION_REGISTRY: Dict[str, Callable] = {
    'pearson': pearson_correlation,
    'volatility_adjusted': volatility_adjusted_correlation,
    'skewness_kurtosis_adjusted': skewness_kurtosis_adjusted_correlation,
    'dynamic': dynamic_correlation,
    'dcc_garch': dcc_garch_correlation,
}


def top_k_thresholding(corr: np.ndarray, k: int = TOP_K_EDGES) -> Tuple[np.ndarray, np.ndarray]:
    """
    Retain the strongest top-k edges (by absolute correlation) for each node.

    Section 3.2: "We retain the top-k strongest correlations per asset to
    construct a sparse graph."

    Unlike a simple threshold (which discards negative correlations), we use
    |corr| to select edges — preserving both positive and negative relationships.
    The signed correlation value is kept as the edge weight, which is critical
    in financial markets where strong negative correlations carry information
    about hedging and opposite market movements.

    Args:
        corr: shape (N, N) — correlation matrix
        k: number of strongest edges to retain per node

    Returns:
        adj_values: shape (N, N) — signed correlation values for selected edges, 0 elsewhere
        selected_mask: shape (N, N, bool) — which edges were selected by either node
    """
    N = corr.shape[0]
    adj_values = np.zeros((N, N), dtype=np.float32)
    selected_mask = np.zeros((N, N), dtype=bool)

    for i in range(N):
        abs_corr = np.abs(corr[i])  # (N,)
        abs_corr[i] = -1.0  # exclude self
        top_k_indices = np.argsort(abs_corr)[-k:]  # strongest k by |corr|

        selected_mask[i, top_k_indices] = True
        adj_values[i, top_k_indices] = corr[i, top_k_indices]  # keep signed value

    selected_vals = corr[selected_mask]
    n_pos = (selected_vals > 0).sum()
    n_neg = (selected_vals < 0).sum()
    print(f"     [Threshold] Top-{k} edges per node, "
          f"total={selected_mask.sum()}, "
          f"pos={n_pos}, neg={n_neg}")
    return adj_values, selected_mask


def symmetrize(adj_values: np.ndarray, selected_mask: np.ndarray) -> np.ndarray:
    """
    Symmetrize signed adjacency using selection mask.

    Edge (i,j) exists if EITHER i selected j OR j selected i.
    The signed correlation value is preserved from the original correlation.

    Args:
        adj_values: shape (N, N) — signed correlation values from top-k selection
        selected_mask: shape (N, N, bool) — which edges were selected

    Returns:
        adj_sym: shape (N, N) — symmetric signed adjacency
    """
    # Edge exists if either direction selected it
    mask = selected_mask | selected_mask.T
    adj_sym = np.where(mask, adj_values, 0.0)
    return adj_sym


def add_self_loops(adj: np.ndarray) -> np.ndarray:
    """
    Add self-loops to adjacency matrix: A = A + I.
    """
    N = adj.shape[0]
    np.fill_diagonal(adj, 1.0)
    return adj


def normalize_adjacency(adj: np.ndarray) -> np.ndarray:
    """
    Symmetric normalization: D^{-1/2} A D^{-1/2}

    Section 3.2: "We normalize the adjacency matrix using the symmetric
    normalization proposed by Kipf & Welling (2017)."

    For signed adjacency, degree = sum of absolute values. This ensures the
    normalization factor reflects total connection strength, not net sign.

    Args:
        adj: shape (N, N) — adjacency matrix with self-loops (signed)

    Returns:
        adj_norm: shape (N, N) — normalized adjacency matrix
    """
    # Degree = sum of absolute values (total connection strength)
    d = np.abs(adj).sum(axis=1)  # (N,)

    # Avoid division by zero
    d = np.maximum(d, 1e-8)

    # D^{-1/2}
    d_inv_sqrt = np.diag(1.0 / np.sqrt(d))  # (N, N)

    # Normalize: D^{-1/2} A D^{-1/2}
    adj_norm = d_inv_sqrt @ adj @ d_inv_sqrt  # (N, N)

    print(f"     [Normalize] Adjacency normalized, range: [{adj_norm.min():.4f}, {adj_norm.max():.4f}]")
    return adj_norm


def adjacency_to_edge_index_weight(adj: np.ndarray) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Convert dense adjacency matrix to PyG edge_index and edge_weight format.

    Args:
        adj: shape (N, N) — normalized adjacency matrix

    Returns:
        edge_index: shape (2, E) — edge indices
        edge_weight: shape (E,) — edge weights
    """
    N = adj.shape[0]
    rows, cols = np.where(np.abs(adj) > 1e-8)  # non-zero edges
    weights = adj[rows, cols]

    edge_index = torch.LongTensor(np.stack([rows, cols], axis=0))  # (2, E)
    edge_weight = torch.FloatTensor(weights)                       # (E,)

    print(f"     [EdgeIndex] edge_index shape: {edge_index.shape}, edges: {edge_index.shape[1]}")
    return edge_index, edge_weight


def build_graph(
    returns: np.ndarray,
    market_indices: Dict[str, Tuple[int, int]],
    method: str = CORRELATION_METHOD,
    top_k: int = TOP_K_EDGES,
) -> Dict:
    """
    End-to-end graph construction pipeline (Section 3.2).

    Pipeline:
      Returns → Correlation Matrix → Top-k Thresholding →
      Symmetrization → Self-loops → Normalization → edge_index/edge_weight

    Args:
        returns: shape (T, N) — training returns
        market_indices: Dict mapping market name to (start, end) column indices
        method: correlation strategy name

    Returns:
        dict with keys:
            - 'edge_index': (2, E) tensor
            - 'edge_weight': (E,) tensor
            - 'adj_norm': (N, N) normalized adjacency matrix
            - 'corr_raw': (N, N) raw correlation matrix
            - 'method': method used
    """
    print(f"\n{'=' * 60}")
    print(f"Graph Construction ({method})")
    print(f"{'=' * 60}")

    # Step 1: Compute correlation matrix using selected strategy
    print(f"\n[Graph Step 1] Computing {method} correlation...")
    if method not in CORRELATION_REGISTRY:
        raise ValueError(f"Unknown correlation method: {method}. "
                         f"Choose from {list(CORRELATION_REGISTRY.keys())}")

    corr_fn = CORRELATION_REGISTRY[method]
    corr = corr_fn(returns)  # (N, N)

    # Step 2: Top-k thresholding (signed, preserves negative correlations)
    print(f"\n[Graph Step 2] Top-{top_k} thresholding (signed)...")
    adj_values, selected_mask = top_k_thresholding(corr, k=top_k)

    # Step 3: Symmetrization (preserving signed values)
    print(f"\n[Graph Step 3] Symmetrizing adjacency...")
    adj_sym = symmetrize(adj_values, selected_mask)

    # Step 4: Add self-loops
    print(f"\n[Graph Step 4] Adding self-loops...")
    adj_with_loops = add_self_loops(adj_sym)  # (N, N)

    # Step 5: Normalization: D^{-1/2} A D^{-1/2}
    print(f"\n[Graph Step 5] Symmetric normalization...")
    adj_norm = normalize_adjacency(adj_with_loops)  # (N, N)

    # Step 6: Convert to edge_index and edge_weight
    print(f"\n[Graph Step 6] Converting to edge_index/edge_weight...")
    edge_index, edge_weight = adjacency_to_edge_index_weight(adj_norm)

    # Print market subgraph stats
    print(f"\n[Graph] Market connectivity:")
    for market, (start, end) in market_indices.items():
        sub_adj = adj_norm[start:end, start:end]
        n_edges = (np.abs(sub_adj) > 1e-8).sum()
        n_nodes = end - start
        print(f"       {market}: {n_nodes} nodes, {n_edges} intra-edges")

    return {
        'edge_index': edge_index,
        'edge_weight': edge_weight,
        'adj_norm': adj_norm,
        'corr_raw': corr,
        'method': method,
    }
