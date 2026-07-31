"""
Feature-augmented Dataset — features computed from RAW prices.

X: (SEQ_LEN, N, F) — all 21 features standardized
y: (N_commodities,) — next-day normalized commodity price (for evaluation)

Key changes from v1:
  - Features computed from RAW prices (returns, RSI, BB etc. are meaningful)
  - ALL features (including price) are standardized for cross-asset comparability
  - Both GCN and LSTM branches use the full F-dim feature matrix
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from typing import Dict

from cmgm.config import SEQ_LEN, BATCH_SIZE
from cmgm.data.feature_builder import build_feature_matrix, NUM_FEATURES


class FeatureMarketSequenceDataset(Dataset):
    """
    Sliding-window dataset with F-dim features from raw prices.

    X: (SEQ_LEN, N, F) — standardized features (all assets)
    y: (N_commodities,) — next-day normalized commodity price
    """

    def __init__(
        self,
        feature_matrix: np.ndarray,     # (T, N, F) — standardized
        y_prices: np.ndarray,           # (T, N_comm) — normalized price target
        market_indices: Dict,
        seq_len: int = SEQ_LEN,
    ):
        self.features = feature_matrix
        self.y_prices = y_prices
        self.seq_len = seq_len
        n_comm = y_prices.shape[1]
        self.n_commodities = n_comm
        self.n_samples = len(feature_matrix) - seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        X = self.features[idx: idx + self.seq_len]        # (SEQ_LEN, N, F)
        y = self.y_prices[idx + self.seq_len]              # (N_commodities,)
        return torch.FloatTensor(X), torch.FloatTensor(y)


def create_feature_loaders(
    raw_prices: np.ndarray,         # (T, N) — raw (unnormalized) prices
    norm_prices: np.ndarray,        # (T, N) — normalized [0,1] prices (for y target)
    market_indices: Dict,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
) -> Dict:
    """
    Build feature-augmented DataLoaders from RAW price data.

    Steps:
      1. Compute feature matrix (T, N, F) from RAW prices
      2. Temporal train/val/test split
      3. Standardize ALL features (fit on training portion)
      4. y = next-day normalized commodity price (from norm_prices)
      5. Wrap in FeatureMarketSequenceDataset

    Args:
        raw_prices:  (T, N) raw prices — for feature computation
        norm_prices: (T, N) normalized [0,1] prices — for y target

    Returns:
        dict with:
          feature_train_loader, feature_val_loader, feature_test_loader
          feature_dim: F
          feature_names: list of F names
    """
    T = raw_prices.shape[0]

    # Step 1: Compute features from RAW prices
    features, fnames = build_feature_matrix(raw_prices)    # (T, N, F)

    # Step 2: Temporal split (70/15/15)
    train_end = int(T * 0.7)
    val_end = train_end + int(T * 0.15)

    train_feat = features[:train_end]
    val_feat   = features[train_end:val_end]
    test_feat  = features[val_end:]

    N, F = features.shape[1], features.shape[2]

    # Step 3: Standardize ALL features (including price index 0)
    train_2d = train_feat.reshape(-1, F)                   # (T_tr*N, F)
    scaler = StandardScaler()
    scaler.fit(train_2d)

    def _standardize(subset):
        feat_std = scaler.transform(subset.reshape(-1, F))
        return feat_std.reshape(-1, N, F)

    train_feat = _standardize(train_feat)
    val_feat   = _standardize(val_feat)
    test_feat  = _standardize(test_feat)

    print(f"[Feature] Computed {F} features from RAW prices:")
    print(f"[Feature]   {', '.join(fnames)}")
    print(f"[Feature]   Shape: ({T}, {N}, {F})")
    print(f"[Standardize] ALL {F} features standardized")
    print(f"[Standardize]   Mean ~0, Std ~1 for each feature")

    # Step 4: Extract y targets from normalized prices
    cs, ce = market_indices['commodity']
    train_y = norm_prices[:train_end, cs:ce]
    val_y   = norm_prices[train_end:val_end, cs:ce]
    test_y  = norm_prices[val_end:, cs:ce]

    datasets = {
        'train': FeatureMarketSequenceDataset(train_feat, train_y, market_indices, seq_len),
        'val':   FeatureMarketSequenceDataset(val_feat, val_y, market_indices, seq_len),
        'test':  FeatureMarketSequenceDataset(test_feat, test_y, market_indices, seq_len),
    }

    loaders = {
        f'feature_{k}_loader': DataLoader(
            datasets[k], batch_size=batch_size, shuffle=False,
            drop_last=(k == 'train'),
        )
        for k in ['train', 'val', 'test']
    }

    # Also return the full feature matrix for reference
    train_full = train_feat
    val_full = val_feat
    test_full = test_feat
    feat_concat = np.concatenate([train_full, val_full, test_full], axis=0)

    print(f"\n[FeatureLoader] Train={len(datasets['train'])}, "
          f"Val={len(datasets['val'])}, "
          f"Test={len(datasets['test'])}  |  "
          f"Feature dim F={F}")
    print(f"[FeatureLoader] Range: [{feat_concat.min():.2f}, {feat_concat.max():.2f}]")

    return {
        'feature_train_loader': loaders['feature_train_loader'],
        'feature_val_loader':   loaders['feature_val_loader'],
        'feature_test_loader':  loaders['feature_test_loader'],
        'feature_dim':          F,
        'feature_names':        fnames,
        'feature_full_matrix':  feat_concat,
    }
