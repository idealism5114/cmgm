"""
TCN Dataset — wraps the original data pipeline.
Exists only to keep main_tcn.py self-contained.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict

from cmgm.config import SEQ_LEN, BATCH_SIZE


class TCNMarketSequenceDataset(Dataset):
    """
    Sliding-window dataset — identical target to original MarketSequenceDataset.

    X: (SEQ_LEN, N_total, 1) — normalized closing prices
    y: (N_commodities,)      — next-day normalized commodity price
    """

    def __init__(
        self,
        norm_prices: np.ndarray,   # (T, N)
        market_indices: Dict,
        seq_len: int = SEQ_LEN,
    ):
        self.norm_prices = norm_prices
        self.market_indices = market_indices
        self.seq_len = seq_len

        cs, ce = market_indices['commodity']
        self.n_commodities = ce - cs
        self.n_samples = len(norm_prices) - seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        X = self.norm_prices[idx: idx + self.seq_len]          # (SEQ_LEN, N)
        X = X[..., np.newaxis]                                   # (SEQ_LEN, N, 1)

        cs, ce = self.market_indices['commodity']
        y = self.norm_prices[idx + self.seq_len, cs:ce]          # (N_commodities,)

        return torch.FloatTensor(X), torch.FloatTensor(y)


def create_tcn_loaders(
    norm_prices: np.ndarray,
    market_indices: Dict,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
) -> Dict:
    """Create DataLoaders — same temporal split as the original pipeline."""
    T = norm_prices.shape[0]
    train_end = int(T * 0.7)
    val_end = train_end + int(T * 0.15)

    norm_splits = (
        norm_prices[:train_end],
        norm_prices[train_end:val_end],
        norm_prices[val_end:],
    )

    datasets = {
        k: TCNMarketSequenceDataset(n, market_indices, seq_len)
        for k, n in zip(['train', 'val', 'test'], norm_splits)
    }

    loaders = {}
    for k in ['train', 'val', 'test']:
        loaders[f'tcn_{k}_loader'] = DataLoader(
            datasets[k],
            batch_size=batch_size,
            shuffle=False,
            drop_last=(k == 'train'),
        )

    print(f"[TCNLoader] Train={len(datasets['train'])}, "
          f"Val={len(datasets['val'])}, "
          f"Test={len(datasets['test'])}")
    print(f"[TCNLoader] Target: next-day normalized price (same as original CMGM)")

    return loaders
