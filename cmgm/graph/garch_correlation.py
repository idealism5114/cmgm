"""
DCC-GARCH correlation estimation for CMGM.

Two-step approach (Engle, 2002):
  1. Fit univariate GARCH(1,1) for each of the N assets
  2. Standardize residuals by conditional volatility → compute correlation matrix

This gives a more accurate static correlation matrix by accounting for
heteroskedasticity (volatility clustering) before computing correlations.

Returns (T, N) → correlation matrix (N, N)
"""

import numpy as np
from typing import Optional
import warnings

from arch import arch_model


def fit_garch_11(returns: np.ndarray, n_jobs: int = 8) -> np.ndarray:
    """
    Fit GARCH(1,1) for each asset and return standardized residuals.

    Args:
        returns: shape (T, N) — asset return time series
        n_jobs: parallel workers

    Returns:
        std_residuals: shape (T, N) — standardized residuals (z_t = ε_t / σ_t)
    """
    T, N = returns.shape
    std_residuals = np.zeros_like(returns)

    # Simple sequential loop (parallel not needed for N=284, ~5min)
    for i in range(N):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            try:
                am = arch_model(returns[:, i], vol='Garch', p=1, q=1,
                                dist='normal', rescale=True)
                res = am.fit(disp='off', show_warning=False, update_freq=0)

                # Conditional volatility
                cond_vol = res.conditional_volatility  # (T,)

                # Standardized residuals
                std_residuals[:, i] = returns[:, i] / np.maximum(cond_vol, 1e-8)

            except Exception:
                # Fallback: use sample std if GARCH fails
                std_residuals[:, i] = returns[:, i] / np.maximum(
                    returns[:, i].std(), 1e-8
                )

        if (i + 1) % 50 == 0:
            print(f"     [GARCH] {i+1}/{N} assets completed")

    return std_residuals


def dcc_garch_correlation(
    returns: np.ndarray,
    n_jobs: int = 8,
) -> np.ndarray:
    """
    Compute DCC-GARCH based unconditional correlation matrix.

    Process:
      1. Fit GARCH(1,1) per asset → standardized residuals
      2. Correlation of standardized residuals = unconditional DCC correlation

    Args:
        returns: shape (T, N) — asset returns (training period only)
        n_jobs: parallel workers for GARCH fitting

    Returns:
        corr: shape (N, N) — DCC-GARCH correlation matrix
    """
    T, N = returns.shape
    print(f"\n     [DCC-GARCH] Fitting GARCH(1,1) for {N} assets (T={T})...")

    # Step 1: GARCH(1,1) filtering
    std_residuals = fit_garch_11(returns, n_jobs=n_jobs)

    # Step 2: Correlation of standardized residuals
    # This is the unconditional correlation matrix ̄Q in DCC framework
    corr = np.corrcoef(std_residuals.T)  # (N, N)

    # Clip for numerical stability
    corr = np.clip(corr, -1.0, 1.0)
    corr = np.nan_to_num(corr, nan=0.0)

    print(f"     [DCC-GARCH] Correlation range: [{corr.min():.4f}, {corr.max():.4f}]")

    return corr
