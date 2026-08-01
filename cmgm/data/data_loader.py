"""
Data loading and preprocessing for CMGM.

Pipeline (Section 4.1):
  1. Load raw close prices for each market
  2. Align dates across markets (intersection)
  3. Temporal train/val/test split
  4. Per-asset z-score normalization (fit on train only)
  5. Create sliding windows of length SEQ_LEN

Tensor shapes at each step are documented.
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Dict, Tuple, Optional

from cmgm.config import (
    STOCK_FILE, BOND_FILE, COMMODITY_FILE,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    SEQ_LEN, BATCH_SIZE, RANDOM_SEED, TARGET_MARKET,
    TARGET_TYPE, TARGET_HORIZON, ZSCORE_EPS, MULTI_HORIZONS,
)


def set_seed(seed: int = RANDOM_SEED):
    """Set random seeds for reproducibility (Section 4.5)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_stock_prices(filepath: Path) -> pd.DataFrame:
    """Load CSI 300 constituent stock closing prices."""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    df = df.set_index('date')
    df = df.dropna(axis=1, how='all')
    return df


def load_bond_prices(filepath: Path) -> pd.DataFrame:
    """Load treasury futures closing prices."""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    pivoted = df.pivot_table(
        index='date', columns='品种', values='close', aggfunc='first'
    )
    pivoted = pivoted.sort_index()
    pivoted = pivoted.ffill().bfill()
    return pivoted


def load_commodity_prices(filepath: Path) -> pd.DataFrame:
    """Load commodity futures closing prices."""
    df = pd.read_csv(filepath, encoding='utf-8-sig')
    df['date'] = pd.to_datetime(df['date'])
    pivoted = df.pivot_table(
        index='date', columns='品种', values='close', aggfunc='first'
    )
    pivoted = pivoted.sort_index()
    index_futures = ['上证50股指期货', '中证500股指期货', '沪深300股指期货', '中证1000股指期货']
    pivoted = pivoted.drop(columns=[c for c in index_futures if c in pivoted.columns])
    pivoted = pivoted.ffill().bfill()
    pivoted = pivoted.dropna(axis=1, how='any')
    return pivoted


def align_markets(
    stocks: pd.DataFrame,
    bonds: pd.DataFrame,
    commodities: pd.DataFrame
) -> pd.DataFrame:
    """Align all three markets to a common date range (intersection)."""
    common_index = stocks.index.intersection(bonds.index).intersection(commodities.index)
    common_index = common_index.sort_values()

    print(f"[DataLoader] Stock date range: {stocks.index.min()} to {stocks.index.max()} ({len(stocks)} days)")
    print(f"[DataLoader] Bond date range: {bonds.index.min()} to {bonds.index.max()} ({len(bonds)} days)")
    print(f"[DataLoader] Commodity date range: {commodities.index.min()} to {commodities.index.max()} ({len(commodities)} days)")
    print(f"[DataLoader] Common date range: {common_index.min()} to {common_index.max()} ({len(common_index)} days)")

    stocks_aligned = stocks.loc[common_index].copy()
    bonds_aligned = bonds.loc[common_index].copy()
    commodities_aligned = commodities.loc[common_index].copy()

    stocks_aligned = stocks_aligned.dropna(axis=1, how='any')
    bonds_aligned = bonds_aligned.dropna(axis=1, how='any')
    commodities_aligned = commodities_aligned.dropna(axis=1, how='any')

    n_stocks = stocks_aligned.shape[1]
    n_bonds = bonds_aligned.shape[1]
    n_commodities = commodities_aligned.shape[1]

    market_indices = {
        'stock': (0, n_stocks),
        'bond': (n_stocks, n_stocks + n_bonds),
        'commodity': (n_stocks + n_bonds, n_stocks + n_bonds + n_commodities),
    }

    all_prices = pd.concat([stocks_aligned, bonds_aligned, commodities_aligned], axis=1)

    print(f"[DataLoader] Combined price matrix shape: {all_prices.shape}")
    print(f"[DataLoader] Market split: stocks={n_stocks}, bonds={n_bonds}, commodities={n_commodities}")

    return all_prices, market_indices


def temporal_train_val_test_split(
    prices: np.ndarray,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    train_window: int = 0,
):
    """Temporal (non-random) split into train/val/test sets."""
    T = prices.shape[0]
    train_end = int(T * train_ratio)
    val_end = train_end + int(T * val_ratio)

    train_full = prices[:train_end]

    if train_window > 0 and len(train_full) > train_window:
        train_start = train_end - train_window
        train = prices[train_start:train_end]
        print(f"[DataLoader] Rolling window: using last {train_window} of "
              f"{len(train_full)} training days (discarding {len(train_full) - train_window} old days)")
    else:
        train = train_full

    val = prices[train_end:val_end]
    test = prices[val_end:]

    print(f"[DataLoader] Temporal split: train={len(train)}, val={len(val)}, test={len(test)}")
    return train, val, test


def normalize_data(
    train: np.ndarray,
    val: np.ndarray,
    test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    """
    Per-asset Z-score normalization.

    Each asset i is independently normalized:
      z_i = (p_i - μ_i) / (σ_i + ε)

    μ_i = mean(train[:, i]), σ_i = std(train[:, i])
    Fit on training data only, then transform all splits.

    This replaces the old global MinMaxScaler(0,1).  Per-asset z-score
    keeps relative movements comparable across assets with different
    absolute price levels (e.g., ¥5 stocks vs ¥50,000 futures).

    Returns:
        norm_stats: dict with 'mean' (N,) and 'std' (N,) arrays
    """
    mean = train.mean(axis=0).astype(np.float32)           # (N,)
    std  = train.std(axis=0).astype(np.float32)            # (N,)
    std  = np.maximum(std, ZSCORE_EPS)                     # avoid div-by-zero

    train_norm = (train - mean) / std
    val_norm   = (val   - mean) / std
    test_norm  = (test  - mean) / std

    norm_stats = {'mean': mean, 'std': std}

    print(f"[DataLoader] Per-asset z-score normalization fitted on training data only")
    print(f"[DataLoader] mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"[DataLoader] std  range: [{std.min():.6f}, {std.max():.4f}]")
    print(f"[DataLoader] Z-scored train range: [{train_norm.min():.4f}, {train_norm.max():.4f}]")
    return train_norm, val_norm, test_norm, norm_stats


def compute_returns(prices: np.ndarray) -> np.ndarray:
    """Compute simple returns from prices (used ONLY for graph construction)."""
    returns = np.diff(prices, axis=0) / prices[:-1]
    returns = np.nan_to_num(returns, nan=0.0, posinf=0.0, neginf=0.0)
    return returns


# =============================================================================
# Feature computation (7-dim computed from RAW prices)
# =============================================================================

def compute_features(prices: np.ndarray) -> np.ndarray:
    """
    Compute 7-dim features from ***RAW*** prices.

    All rolling features use only past information — no data leakage.
    No scaling is applied — the caller should normalize (MinMaxScaler).

    Features (computed from raw prices, correct returns):
      [0] price        — raw closing price
      [1] return       — daily return (pct_change from raw prices)
      [2] ma5_ratio    — price / 5-day moving average
      [3] ma20_ratio   — price / 20-day moving average
      [4] volatility   — 20-day rolling volatility of daily returns
      [5] rsi_14       — 14-day relative strength index
      [6] macd         — MACD (12, 26)

    Args:
        prices:  (T, N) — RAW prices (original scale, NOT normalized)

    Returns:
        features:  (T, N, 7) — raw features, no scaling applied
    """
    T, N = prices.shape
    df = pd.DataFrame(prices)
    daily_ret = df.pct_change().fillna(0.0)

    layers = []

    # ── [0] Price (raw) ──────────────────────────────────────────────────
    layers.append(df.values[:, :, None])

    # ── [1] Return (from raw prices — correct!) ──────────────────────────
    ret = daily_ret.clip(-0.2, 0.2)
    layers.append(ret.values[:, :, None])

    # ── [2] ma5_ratio ────────────────────────────────────────────────────
    ma5 = df.rolling(5, min_periods=1).mean()
    ratio5 = np.where(ma5.values > 1e-8, df.values / ma5.values, 1.0)
    layers.append(np.clip(ratio5, 0.5, 2.0)[:, :, None])

    # ── [3] ma20_ratio ───────────────────────────────────────────────────
    ma20 = df.rolling(20, min_periods=1).mean()
    ratio20 = np.where(ma20.values > 1e-8, df.values / ma20.values, 1.0)
    layers.append(np.clip(ratio20, 0.5, 2.0)[:, :, None])

    # ── [4] Volatility (20-day, from raw returns) ────────────────────────
    vol = daily_ret.rolling(20, min_periods=1).std().fillna(0.0).clip(0.0, 0.5)
    layers.append(vol.values[:, :, None])

    # ── [5] RSI-14 (from raw prices) ─────────────────────────────────────
    delta = df.diff().fillna(0.0)
    gain = delta.clip(lower=0)
    loss = delta.clip(upper=0).abs()
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean().clip(lower=1e-8)
    rs = avg_gain / avg_loss
    rsi = (100.0 - 100.0 / (1.0 + rs)).fillna(50.0)
    layers.append(rsi.values[:, :, None])

    # ── [6] MACD (12, 26, from raw prices) ────────────────────────────────
    ema12 = df.ewm(span=12, adjust=False).mean()
    ema26 = df.ewm(span=26, adjust=False).mean()
    macd = (ema12 - ema26).fillna(0.0)
    layers.append(macd.values[:, :, None])

    # ── Concatenate & final NaN/Inf guard ────────────────────────────────
    features = np.concatenate(layers, axis=-1).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)

    names = ['price', 'return', 'ma5_ratio', 'ma20_ratio',
             'volatility', 'rsi_14', 'macd']
    print(f"[Features] Computed {features.shape[-1]} features from RAW prices: "
          f"{', '.join(names)}")
    print(f"[Features] Shape: (T={T}, N={N}, F={features.shape[-1]})  "
          f"range: [{features.min():.4f}, {features.max():.4f}]")

    return features


# =============================================================================
# Dataset
# =============================================================================

class MarketSequenceDataset(Dataset):
    """
    PyTorch Dataset for sliding window sequences.

    X: (SEQ_LEN, N_total, F) — normalized prices (+ optional features)
    y: (N_commodities,) — next-day normalized commodity closing prices
    """

    def __init__(
        self,
        prices: np.ndarray,
        market_indices: Dict[str, Tuple[int, int]],
        seq_len: int = SEQ_LEN,
        feature_matrix: np.ndarray = None,
        raw_prices: np.ndarray = None,
        target_type: str = "price",
        horizons: list = None,
    ):
        self.prices = prices
        self.market_indices = market_indices
        self.seq_len = seq_len
        self.feature_matrix = feature_matrix
        self.raw_prices = raw_prices
        self.target_type = target_type
        self.horizons = horizons if horizons is not None else MULTI_HORIZONS

        self.commodity_start, self.commodity_end = market_indices['commodity']
        self.n_commodities = self.commodity_end - self.commodity_start
        # Need enough future prices for the longest horizon
        max_h = max(self.horizons) if self.horizons else 0
        self.n_samples = len(prices) - seq_len - max_h + 1

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int):
        if self.feature_matrix is not None:
            X = self.feature_matrix[idx: idx + self.seq_len]      # (SEQ_LEN, N, F)
        else:
            X = self.prices[idx: idx + self.seq_len]              # (SEQ_LEN, N)
            X = X[..., np.newaxis]                                 # (SEQ_LEN, N, 1)

        cs, ce = self.commodity_start, self.commodity_end

        # ── Target: return(s) or volatility ──────────────────────────────
        if self.target_type in ("return", "volatility") and self.raw_prices is not None:
            # Compute daily returns for the horizon window (needed for vol)
            t_end = idx + self.seq_len - 1  # last known time step
            p_base = self.raw_prices[t_end, cs:ce]  # (N_comm,)

            # Pre-compute daily returns up to max horizon
            max_h = max(self.horizons)
            daily_rets = []
            for d in range(1, max_h + 1):
                p_d = self.raw_prices[t_end + d, cs:ce]
                r_d = (p_d / np.maximum(np.abs(self.raw_prices[t_end + d - 1, cs:ce]), 1e-8)) - 1.0
                r_d = np.clip(r_d, -0.5, 0.5).astype(np.float32)
                daily_rets.append(r_d)  # list of (N_comm,) arrays

            y_list = []
            for h in self.horizons:
                if self.target_type == "return":
                    # ret = p_{t+h} / p_t - 1
                    ret_h = (self.raw_prices[t_end + h, cs:ce] / np.maximum(np.abs(p_base), 1e-8)) - 1.0
                    ret_h = np.clip(ret_h, -1.0, 1.0).astype(np.float32)
                    y_list.append(ret_h)
                else:  # volatility
                    # Realized volatility over h days: sqrt(sum(daily_ret^2))
                    sum_sq = np.zeros(ce - cs, dtype=np.float32)
                    for d in range(h):
                        sum_sq += daily_rets[d] ** 2
                    rv_h = np.sqrt(sum_sq).astype(np.float32)
                    y_list.append(rv_h)
            y = np.stack(y_list, axis=0)                                    # (n_horizons, N_comm)
            if len(self.horizons) == 1:
                y = y[0]                                                    # (N_comm,) — single horizon
        else:
            # Original behaviour: z-scored price at time t+seq_len
            y = self.prices[idx + self.seq_len, cs:ce]

        return torch.FloatTensor(X), torch.FloatTensor(y)


# =============================================================================
# Data loader factory
# =============================================================================

def create_data_loaders(
    stock_file: Path = STOCK_FILE,
    bond_file: Path = BOND_FILE,
    commodity_file: Path = COMMODITY_FILE,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
    train_window: int = 0,
) -> Dict:
    """
    End-to-end data pipeline:
    1. Load raw prices for each market
    2. Align to common dates
    3. Temporal train/val/test split
    4. Normalize with MinMaxScaler (fit on train only)
    5. Compute returns from raw prices for graph construction (train only)
    6. Create DataLoader instances
    """
    print("=" * 60)
    print("CMGM Data Loading Pipeline")
    print("=" * 60)

    # Step 1: Load raw prices
    print("\n[Step 1] Loading raw price data...")
    stocks = load_stock_prices(stock_file)
    bonds = load_bond_prices(bond_file)
    commodities = load_commodity_prices(commodity_file)

    # Step 2: Align markets
    print("\n[Step 2] Aligning markets to common date range...")
    all_prices, market_indices = align_markets(stocks, bonds, commodities)

    # Step 3: Temporal train/val/test split
    print("\n[Step 3] Temporal train/val/test split...")
    raw_prices = all_prices.values
    full_train_end = int(len(raw_prices) * TRAIN_RATIO)
    full_raw_train = raw_prices[:full_train_end]
    raw_train_used, raw_val, raw_test = temporal_train_val_test_split(
        raw_prices, train_window=train_window
    )

    # Step 4: Normalization (per-asset z-score)
    print("\n[Step 4] Per-asset z-score normalization...")
    train_norm_full, val_norm, test_norm, norm_stats = normalize_data(
        full_raw_train, raw_val, raw_test
    )
    if train_window > 0 and len(train_norm_full) > train_window:
        train_norm = train_norm_full[-train_window:]
    else:
        train_norm = train_norm_full

    # Step 6: Compute returns
    print("\n[Step 6] Computing returns for graph construction...")
    train_returns_raw = full_raw_train if train_window == 0 else full_raw_train[-train_window:]
    train_returns = compute_returns(train_returns_raw)
    print(f"     Training returns shape: {train_returns.shape}")

    # Step 7: Create sliding window datasets
    print("\n[Step 7] Creating sliding window datasets...")
    train_dataset = MarketSequenceDataset(
        train_norm, market_indices, seq_len,
        raw_prices=raw_train_used, target_type=TARGET_TYPE,
    )
    val_dataset = MarketSequenceDataset(
        val_norm, market_indices, seq_len,
        raw_prices=raw_val, target_type=TARGET_TYPE,
    )
    test_dataset = MarketSequenceDataset(
        test_norm, market_indices, seq_len,
        raw_prices=raw_test, target_type=TARGET_TYPE,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, drop_last=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, drop_last=False
    )

    print(f"\n[Done] Dataset sizes: train={len(train_dataset)}, val={len(val_dataset)}, test={len(test_dataset)}")
    print(f"[Done] Total nodes: {all_prices.shape[1]}")
    print(f"[Done] Commodity nodes (output dim): {market_indices['commodity'][1] - market_indices['commodity'][0]}")

    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'norm_stats': norm_stats,
        'market_indices': market_indices,
        'n_nodes': all_prices.shape[1],
        'n_commodities': market_indices['commodity'][1] - market_indices['commodity'][0],
        'train_returns': train_returns,
        'raw_prices_train': full_raw_train,
        'raw_prices_val': raw_val,
        'raw_prices_test': raw_test,
        'feature_names': all_prices.columns.tolist(),
    }
