"""
Dynamic graph data loading for CMGM v2.

For each sample at day t, the graph is computed from the most recent N days
of returns (default: 60). This ensures:
  - No data leakage: each graph only uses data available before the prediction date
  - Graph evolves over time, capturing changing market regimes
  - Each sample has a unique graph reflecting its market conditions

Returns vs prices alignment:
  returns[t] = (prices[t+1] - prices[t]) / prices[t]
  Sample idx: X = prices[idx:idx+seq_len], y = prices[idx+seq_len]
  Available returns for graph: returns[:idx+seq_len-1]
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, List, Optional, Callable

from cmgm.config import SEQ_LEN, TOP_K_EDGES
from cmgm.graph.graph_builder import (
    top_k_thresholding, symmetrize, add_self_loops,
    normalize_adjacency, adjacency_to_edge_index_weight,
    CORRELATION_REGISTRY,
)
from cmgm.data.data_loader import compute_returns


class DynamicGraphDataset(Dataset):
    """
    PyTorch Dataset with per-sample dynamic graphs.

    Each sample returns (X, y, edge_index, edge_weight) where edge_index
    and edge_weight are computed from a rolling window of returns ending
    before the prediction date.
    """

    def __init__(
        self,
        prices: np.ndarray,
        returns: np.ndarray,
        market_indices: Dict[str, Tuple[int, int]],
        seq_len: int = SEQ_LEN,
        graph_window: int = 60,
        method: str = 'pearson',
        top_k: int = TOP_K_EDGES,
    ):
        """
        Args:
            prices: Normalized prices, shape (T, N)
            returns: Returns from raw prices, shape (T_total_full, N)
                     Must be aligned so returns[t] uses prices[t] and prices[t+1].
                     Can be longer than prices (will be sliced per sample).
            market_indices: Dict mapping market name to (start, end) col indices
            seq_len: Lookback window length
            graph_window: Rolling window size for correlation (trading days)
            method: Correlation strategy name
            top_k: Edges to retain per node
        """
        self.prices = prices
        self.returns = returns
        self.seq_len = seq_len
        self.graph_window = graph_window
        self.method = method
        self.top_k = top_k

        self.commodity_start, self.commodity_end = market_indices['commodity']
        self.n_commodities = self.commodity_end - self.commodity_start
        self.n_samples = len(prices) - seq_len

        print(f"  [DynamicGraph] Pre-computing {self.n_samples} graphs "
              f"(window={graph_window}, method={method}, top_k={top_k})...")
        self.edge_indices, self.edge_weights = self._precompute_graphs()

    def _precompute_graphs(self) -> Tuple[List, List]:
        """Build one graph per sample from a rolling window of returns."""
        corr_fn = CORRELATION_REGISTRY[self.method]

        edge_indices = [None] * self.n_samples
        edge_weights = [None] * self.n_samples

        for idx in range(self.n_samples):
            # Latest safe return: returns[idx+seq_len-2] uses prices[idx+seq_len-2]
            # and prices[idx+seq_len-1] — both known at prediction time.
            # So available returns = returns[:idx+seq_len-1]
            last_ret = idx + self.seq_len - 1   # exclusive end
            first_ret = max(0, last_ret - self.graph_window)
            ret_window = self.returns[first_ret:last_ret]

            if len(ret_window) < 2:
                continue  # Not enough data for correlation

            # Build graph
            corr = corr_fn(ret_window)
            adj_values, selected_mask = top_k_thresholding(corr, k=self.top_k)
            adj_sym = symmetrize(adj_values, selected_mask)
            adj_loop = add_self_loops(adj_sym)
            adj_norm = normalize_adjacency(adj_loop)
            ei, ew = adjacency_to_edge_index_weight(adj_norm)

            edge_indices[idx] = ei
            edge_weights[idx] = ew

        n_valid = sum(ei is not None for ei in edge_indices)
        n_skip = self.n_samples - n_valid
        if n_skip > 0:
            print(f"  [DynamicGraph] {n_valid} graphs computed, "
                  f"{n_skip} skipped (insufficient returns)")

        return edge_indices, edge_weights

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        X = self.prices[idx: idx + self.seq_len]          # (seq_len, N)
        y = self.prices[idx + self.seq_len,
                        self.commodity_start:self.commodity_end]  # (N_commodities,)

        ei = self.edge_indices[idx]
        ew = self.edge_weights[idx]

        # Handle missing graph (early samples)
        if ei is None:
            ei = torch.zeros(2, 0, dtype=torch.long)
            ew = torch.zeros(0)

        return (torch.FloatTensor(X).unsqueeze(-1),
                torch.FloatTensor(y),
                ei, ew)


def dynamic_graph_collate(batch) -> Tuple[torch.Tensor, torch.Tensor,
                                          torch.Tensor, torch.Tensor]:
    """
    Custom collate: each sample has its own graph.

    Each sample's edge_index gets node-index offset so that within a batch:
      - Sample 0 nodes: [0, N-1]
      - Sample 1 nodes: [N, 2N-1]
      - ...
    Then all edge_indices are concatenated into one batched graph.
    """
    X_list, y_list = [], []
    ei_list, ew_list = [], []

    for item in batch:
        X_list.append(item[0])
        y_list.append(item[1])
        ei_list.append(item[2])
        ew_list.append(item[3])

    X_batch = torch.stack(X_list)              # (B, T, N, 1)
    y_batch = torch.stack(y_list)              # (B, N_commodities)
    num_nodes = X_batch.shape[2]

    offset = 0
    batched_eis, batched_ews = [], []
    for ei, ew in zip(ei_list, ew_list):
        if ei.shape[1] > 0:                    # has edges
            batched_eis.append(ei + offset)
            batched_ews.append(ew)
        offset += num_nodes

    if batched_eis:
        edge_index = torch.cat(batched_eis, dim=1)   # (2, total_E)
        edge_weight = torch.cat(batched_ews)         # (total_E,)
    else:
        edge_index = torch.zeros(2, 0, dtype=torch.long)
        edge_weight = torch.zeros(0)

    return X_batch, y_batch, edge_index, edge_weight


def create_dynamic_data_loaders(
    prices_dict: Dict[str, np.ndarray],
    returns_full: np.ndarray,
    market_indices: Dict[str, Tuple[int, int]],
    batch_size: int = 64,
    seq_len: int = SEQ_LEN,
    graph_window: int = 60,
    method: str = 'pearson',
    top_k: int = TOP_K_EDGES,
    **kwargs,
) -> Dict:
    """
    Create DataLoaders with per-sample dynamic graphs.

    Args:
        prices_dict: {'train': (T_train, N), 'val': (T_val, N), 'test': (T_test, N)}
                     — normalized prices
        returns_full: Full returns array from ALL raw prices, shape (T_total-1, N)
        market_indices: Dict of market column ranges
        batch_size: Batch size
        seq_len: Lookback window
        graph_window: Rolling window for correlation
        method: Correlation method
        top_k: Edges per node

    Returns:
        dict with train_loader, val_loader, test_loader, n_nodes, n_commodities
    """
    train_prices = prices_dict['train']
    val_prices = prices_dict['val']
    test_prices = prices_dict['test']
    n_nodes = train_prices.shape[1]
    n_commodities = market_indices['commodity'][1] - market_indices['commodity'][0]

    # Pre-compute offset for returns alignment
    # Full prices = concat([raw_train, raw_val, raw_test])
    # For training samples: position p in full array → returns up to p+seq_len-1
    # Train starts at offset 0 in the full array
    train_offset = 0
    val_offset = len(train_prices)
    test_offset = val_offset + len(val_prices)

    def make_loader(prices, ret_offset, shuffle):
        dataset = DynamicGraphDataset(
            prices,
            returns_full,    # full returns — sliced internally by offset+idx
            market_indices,
            seq_len=seq_len,
            graph_window=graph_window,
            method=method,
            top_k=top_k,
        )
        # We need to adjust returns indexing by offset
        # The simplest approach: slice returns inside the dataset
        # but the dataset doesn't know about the offset
        #
        # Solution: slice returns_full to the relevant portion
        # before passing to DynamicGraphDataset
        pass

    # Actually, let me rethink. The returns_full contains returns for ALL data
    # (train + val + test combined). A training sample at position idx in the
    # training set corresponds to position idx in the full returns array.
    #
    # Let me just slice the returns appropriately for each split.

    # Train returns: from the start, covering training period
    # For training sample idx, we need returns[:train_offset + idx + seq_len - 1]
    # The returns array passed to the dataset must cover this range.
    # The simplest: pass the FULL returns, and let the dataset handle indexing.
    # The dataset uses: returns[:idx+seq_len-1] where idx is the sample index
    # in the dataset's own price array.
    #
    # For the training set:
    # - prices = train_prices, first price at absolute position 0
    # - returns[:train_offset + idx + seq_len - 1]
    # - But we pass the full returns, so the dataset sees returns[0:...]
    # - The offset is 0 for training — correct!
    #
    # For validation set:
    # - prices = val_prices, first price at absolute position train_offset
    # - returns[:train_offset + idx + seq_len - 1]
    # - But the dataset only sees its own prices and doesn't know the offset
    # - It would use returns[:idx+seq_len-1] which gives returns starting from 0
    # - This is WRONG — it should start from train_offset
    #
    # Solution: pass a SLICE of returns_full for each split.

    # End of returns used by a sample at position idx in its own split:
    # For train: split_offset + idx + seq_len - 1
    # We need returns up to that point, so returns[:split_offset + max_idx + seq_len - 1]
    # But we also need the early returns for the graph_window.
    # The first sample needs returns[0:seq_len-1] (which may be very few).
    # So for all splits, we pass returns from the start up to their max index.

    # Actually, the cleanest: pass a single contiguous slice of returns that covers
    # all data from the beginning to the end of each split.

    # For training:
    #   Max sample idx = len(train_prices) - seq_len - 1
    #   Need returns up to max_idx + seq_len - 1 = len(train_prices) - 2
    #   So returns[:len(train_prices)-1] = returns[len(train_prices)-2] max
    #   = train_returns (computed from raw_train)
    #   Perfect!

    # For validation:
    #   Max sample idx = len(val_prices) - seq_len - 1
    #   Returns needed from the beginning up to train_offset + max_idx + seq_len - 1
    #   = train_offset + len(val_prices) - 2
    #   We need returns[:train_offset + len(val_prices) - 1]
    #
    #   But the dataset's internal indexing uses idx starting from 0 within val_prices,
    #   and it does returns[:idx+seq_len-1]. This assumes the returns ALIGN with prices.
    #   If we pass returns_full[:train_offset + len(val_prices) - 1], then
    #   returns[:idx+seq_len-1] gives the right portion since:
    #   - returns[t] = (full_prices[t+1] - full_prices[t]) / full_prices[t]
    #   - The dataset's price at position idx is at absolute position train_offset + idx
    #   - returns[t] aligns with price[t] and price[t+1]
    #   - So the dataset should have returns starting from the same offset as prices
    #
    #   Hmm, but if returns is a prefix of the full returns array aligned with the
    #   prices array we pass, then returns[:idx+seq_len-1] is correct.
    #
    #   For validation: we pass val_prices and returns_full[:val_offset+len(val_prices)-1].
    #   returns includes training + validation returns. For val sample idx:
    #   - returns[:idx+seq_len-1] includes training returns + some validation returns
    #   - That's exactly what we want (past data including training period)

    train_ret_end = train_offset + len(train_prices) - 1
    val_ret_end = val_offset + len(val_prices) - 1
    test_ret_end = test_offset + len(test_prices) - 1

    datasets = {}
    datasets['train'] = DynamicGraphDataset(
        train_prices, returns_full[:train_ret_end],
        market_indices, seq_len, graph_window, method, top_k,
    )
    datasets['val'] = DynamicGraphDataset(
        val_prices, returns_full[:val_ret_end],
        market_indices, seq_len, graph_window, method, top_k,
    )
    datasets['test'] = DynamicGraphDataset(
        test_prices, returns_full[:test_ret_end],
        market_indices, seq_len, graph_window, method, top_k,
    )

    loaders = {}
    for split in ['train', 'val', 'test']:
        loaders[f'{split}_loader'] = DataLoader(
            datasets[split],
            batch_size=batch_size,
            shuffle=(split == 'train'),
            collate_fn=dynamic_graph_collate,
            drop_last=(split != 'test'),
        )

    print(f"\n[DynamicData] Train samples: {len(datasets['train'])}, "
          f"Val: {len(datasets['val'])}, Test: {len(datasets['test'])}")

    return {
        **loaders,
        'n_nodes': n_nodes,
        'n_commodities': n_commodities,
        'market_indices': market_indices,
    }
