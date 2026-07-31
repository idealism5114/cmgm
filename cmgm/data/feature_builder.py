"""
Feature Builder: compute multi-dimensional features from normalized prices.

All features are computed using ONLY past information (no look-ahead) via
pandas rolling operations.  The output is a (T, N, F) tensor that replaces
the original (T, N, 1) price-only input.

Included features (21 total):
  [0]  price              — normalized closing price (original)
  [1-4]  ret_{1,5,10,20}d  — simple returns (1/5/10/20-day)
  [5-7]  vol_{5,10,20}d    — rolling volatility of daily returns
  [8-10] zscore_{5,10,20}d — rolling Z-score (price dev from moving avg)
  [11-14] price_ma{5,10,20,60} — ratio of price to moving average
  [15]  rsi_14             — relative strength index
  [16]  bb_position        — Bollinger Band position [0,1]
  [17]  skewness_20d       — rolling skewness of daily returns
  [18]  kurtosis_20d       — rolling kurtosis of daily returns
  [19]  roc_10d            — rate of change (10-day)
  [20]  percentile_20d     — percentile rank within 20-day window
"""

import numpy as np
import pandas as pd
from scipy.stats import skew as sp_skew, kurtosis as sp_kurtosis
from typing import Tuple

# =============================================================================
# Feature computation
# =============================================================================

_FEATURE_SPEC = [
    # (name, func(prices_df) -> pd.DataFrame)
]


def build_feature_matrix(prices: np.ndarray) -> Tuple[np.ndarray, list]:
    """
    Compute (T, N, F) feature tensor from normalized price matrix.

    Args:
        prices: (T, N) — normalized prices, stable with MinMaxScaler [0,1]
                (returns and ratios are computed on the normalized prices)

    Returns:
        features:      (T, N, F) float32
        feature_names: list of F strings
    """
    T, N = prices.shape
    df = pd.DataFrame(prices)                       # (T, N)
    daily_ret = df.pct_change().fillna(0.0)          # (T, N)

    layers = []
    names  = []

    # ── 0: price ────────────────────────────────────────────────────────
    layers.append(df.values[:, :, None])             # (T, N, 1)
    names.append('price')

    # ── 1-4: returns (clipped to [-5, 5] to avoid Inf from zero-base) ──
    for period in [1, 5, 10, 20]:
        ret = df.pct_change(periods=period).clip(-5.0, 5.0).fillna(0.0)
        layers.append(ret.values[:, :, None])        # (T, N, 1)
        names.append(f'ret_{period}d')

    # ── 5-7: rolling volatility (clipped) ───────────────────────────────
    for window in [5, 10, 20]:
        vol = daily_ret.rolling(window, min_periods=1).std().fillna(0.0)
        vol = vol.clip(0.0, 5.0)
        layers.append(vol.values[:, :, None])
        names.append(f'vol_{window}d')

    # ── 8-10: Z-score ───────────────────────────────────────────────────
    for window in [5, 10, 20]:
        ma = df.rolling(window, min_periods=1).mean()
        std = df.rolling(window, min_periods=1).std().fillna(1e-8)
        std = std.clip(lower=1e-8)
        z = ((df - ma) / std).clip(-10.0, 10.0).fillna(0.0)
        layers.append(z.values[:, :, None])
        names.append(f'zscore_{window}d')

    # ── 11-14: Price / Moving Average ratio ─────────────────────────────
    for window in [5, 10, 20, 60]:
        ma = df.rolling(window, min_periods=1).mean().clip(lower=1e-8)
        ratio = (df / ma).clip(0.0, 5.0).fillna(1.0)
        layers.append(ratio.values[:, :, None])
        names.append(f'price_ma{window}')

    # ── 15: RSI-14 ──────────────────────────────────────────────────────
    delta = df.diff().fillna(0.0)
    gain = delta.clip(lower=0)
    loss = delta.clip(upper=0).abs()
    avg_gain = gain.rolling(14, min_periods=1).mean()
    avg_loss = loss.rolling(14, min_periods=1).mean().clip(lower=1e-8)
    rs = avg_gain / avg_loss
    rsi = (100.0 - 100.0 / (1.0 + rs)).clip(0.0, 100.0).fillna(50.0)
    layers.append(rsi.values[:, :, None])
    names.append('rsi_14')

    # ── 16: Bollinger Band position ────────────────────────────────────
    ma20 = df.rolling(20, min_periods=1).mean()
    std20 = df.rolling(20, min_periods=1).std().clip(lower=1e-8)
    upper = ma20 + 2.0 * std20
    lower = ma20 - 2.0 * std20
    bb_pos = ((df - lower) / (upper - lower)).clip(0.0, 1.0).fillna(0.5)
    layers.append(bb_pos.values[:, :, None])
    names.append('bb_position')

    # ── 17: Skewness (20d) ─────────────────────────────────────────────
    skew_20 = daily_ret.rolling(20, min_periods=5).apply(
        lambda x: sp_skew(x, bias=False), raw=True).clip(-10.0, 10.0).fillna(0.0)
    layers.append(skew_20.values[:, :, None])
    names.append('skewness_20d')

    # ── 18: Kurtosis (20d) ─────────────────────────────────────────────
    kurt_20 = daily_ret.rolling(20, min_periods=5).apply(
        lambda x: sp_kurtosis(x, bias=False), raw=True).clip(0.0, 50.0).fillna(0.0)
    layers.append(kurt_20.values[:, :, None])
    names.append('kurtosis_20d')

    # ── 19: Rate of Change (10d) — normalised by average price ─────────
    ma10 = df.rolling(10, min_periods=1).mean().clip(lower=1e-8)
    roc = (df - df.shift(10)).fillna(0.0)
    roc_norm = roc / ma10
    roc_norm = roc_norm.clip(-1.0, 1.0).fillna(0.0)
    layers.append(roc_norm.values[:, :, None])
    names.append('roc_10d')

    # ── 20: Percentile rank within 20-day window ───────────────────────
    percentile = np.zeros_like(df)
    for t in range(T):
        start = max(0, t - 19)
        window = df.iloc[start:t + 1].values
        ranks = np.argsort(np.argsort(window, axis=0), axis=0)
        current_rank = ranks[-1, :]
        percentile[t, :] = current_rank / max(window.shape[0] - 1, 1)

    layers.append(percentile[:, :, None])
    names.append('percentile_20d')

    # ── Concatenate & final NaN/Inf guard ──────────────────────────────
    features = np.concatenate(layers, axis=-1).astype(np.float32)
    features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)

    print(f"[Features] Computed {features.shape[-1]} features: "
          f"{', '.join(names)}")
    print(f"[Features] Shape: (T={T}, N={N}, F={features.shape[-1]})  "
          f"range: [{features.min():.4f}, {features.max():.4f}]")

    return features, names


# =============================================================================
# Convenience: feature dimension constant (for GCN_INPUT_DIM override)
# =============================================================================
NUM_FEATURES = 21
