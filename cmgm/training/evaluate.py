"""
Evaluation metrics for CMGM (Section 4.4).

Metrics reported in NORMALIZED space (matching the loss function and the paper).
Inverse-transformed (original price space) metrics also provided for context.

The paper reports MSE/RMSE/MAE on normalized [0,1] data, which gives
0.00x-level values. Computing on original-scale prices (e.g., 2000–80000 CNY)
inflates metrics by (price_range)², hiding the true model performance.

Paper reference: Ali et al. (2025), AEJ, Section 4.4.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from scipy.stats import skew as scipy_skew
from typing import Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from cmgm.config import (
    CONFIDENCE_LEVEL, NUM_BOOTSTRAP_SAMPLES,
    TARGET_TYPE, SEQ_LEN,
)


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate predictions for all samples in a DataLoader.

    Args:
        model: CMGM model
        loader: DataLoader
        edge_index: Graph edges, shape (2, E)
        edge_weight: Edge weights, shape (E,)
        device: torch device

    Returns:
        preds: Predictions (normalized), shape (N_samples, N_commodities)
        targets: Targets (normalized), shape (N_samples, N_commodities)
    """
    model.eval()
    all_preds = []
    all_targets = []

    for batch in loader:
        if len(batch) == 4:
            X_batch, y_batch, batch_ei, batch_ew = batch
            cur_ei = batch_ei.to(device)
            cur_ew = batch_ew.to(device)
        else:
            X_batch, y_batch = batch
            cur_ei = edge_index.to(device)
            cur_ew = edge_weight.to(device)

        X_batch = X_batch.to(device)
        if hasattr(model, 'graph_learner'):
            pred = model(X_batch, debug=False)
        else:
            pred = model(X_batch, cur_ei, cur_ew, debug=False)

        all_preds.append(pred.cpu().numpy())
        all_targets.append(y_batch.numpy())

    preds = np.concatenate(all_preds, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return preds, targets


def inverse_transform_predictions(
    preds: np.ndarray,                     # (N_samples, N_commodities)
    targets: np.ndarray,                   # (N_samples, N_commodities)
    norm_stats: Dict[str, np.ndarray],     # {'mean': (N,), 'std': (N,)}
    raw_prices_test: np.ndarray,           # (T_test, N_total) — raw prices
    market_indices: Dict,
    target_type: str = "price",
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert predictions/targets to original price space.

    Two modes:
      - 'price'  target: z-score inverse → orig = z * σ + μ
      - 'return' target: implied price = last_known_price × (1 + return)

    For return mode, alignment with MarketSequenceDataset:
      sample i uses raw_prices[i : i+seq_len] as its window and predicts the
      return at time i+seq_len.  The last known price is at i+seq_len-1.
      So for n_samples starting at i=0, the last-known-price slice is
      raw_prices[seq_len-1 : seq_len-1+n_samples, cs:ce].

    Args:
        preds:      Model predictions in target space (z-scored price or return)
        targets:    Ground-truth targets in target space
        norm_stats: Per-asset normalisation parameters
        raw_prices_test: Raw (un-normalised) price array for test period
        market_indices: Dict with 'commodity' key → (start, end)
        target_type: 'price' or 'return'
        seq_len:     Sequence length (needed for return→price alignment)

    Returns:
        preds_orig:   Predictions in original price scale
        targets_orig: Targets in original price scale
    """
    cs, ce = market_indices['commodity']
    commodity_mean = norm_stats['mean'][cs:ce].reshape(1, -1)   # (1, N_comm)
    commodity_std  = norm_stats['std'][cs:ce].reshape(1, -1)    # (1, N_comm)

    if target_type == "price":
        preds_orig   = preds   * commodity_std + commodity_mean
        targets_orig = targets * commodity_std + commodity_mean

    elif target_type == "return":
        n_samples = preds.shape[0]
        # Last known price for each sample
        last_known = raw_prices_test[seq_len - 1 : seq_len - 1 + n_samples, cs:ce]
        # Implied price:  p_{t+1} = p_t × (1 + r)
        preds_orig   = last_known * (1.0 + preds)
        targets_orig = last_known * (1.0 + targets)

    else:
        raise ValueError(f"Unknown target_type: {target_type}")

    return preds_orig.astype(np.float32), targets_orig.astype(np.float32)


def compute_metrics(
    preds: np.ndarray,
    targets: np.ndarray,
) -> Dict[str, float]:
    """
    Compute all evaluation metrics from Section 4.4.

    Args:
        preds: Predictions, shape (N, N_commodities)
        targets: Targets, shape (N, N_commodities)

    Returns:
        dict: All metrics (averaged across commodities where applicable)
    """
    residuals = targets - preds  # (N, N_commodities)

    # Section 4.4.1: MAE
    mae_per_asset = np.mean(np.abs(residuals), axis=0)
    mae = np.mean(mae_per_asset)

    # Section 4.4.2: MSE
    mse_per_asset = np.mean(residuals ** 2, axis=0)
    mse = np.mean(mse_per_asset)

    # Section 4.4.3: RMSE
    rmse_per_asset = np.sqrt(mse_per_asset)
    rmse = np.mean(rmse_per_asset)

    # Section 4.4.4: Residual Mean (bias)
    residual_mean_per_asset = np.mean(residuals, axis=0)
    residual_mean = np.mean(residual_mean_per_asset)

    # Section 4.4.5: Residual Std
    residual_std_per_asset = np.std(residuals, axis=0, ddof=1)
    residual_std = np.mean(residual_std_per_asset)

    # Section 4.4.6: Skewness
    skewness_per_asset = np.array([
        scipy_skew(residuals[:, i], bias=False) for i in range(residuals.shape[1])
    ])
    skewness = np.mean(skewness_per_asset)

    # Directional Accuracy (Hit Ratio)
    # Fraction of predictions where sign(pred) == sign(target).
    # Zero targets are excluded (can't determine correct direction).
    valid = np.abs(targets) > 1e-8  # (N, N_commodities)
    n_valid = valid.sum()
    if n_valid > 0:
        hit_ratio = np.mean(
            (np.sign(preds[valid]) == np.sign(targets[valid])).astype(np.float64)
        )
    else:
        hit_ratio = float('nan')

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'Residual_Mean': residual_mean,
        'Residual_Std': residual_std,
        'Skewness': skewness,
        'Hit_Ratio': hit_ratio,
    }


def compute_confidence_intervals(
    preds: np.ndarray,
    targets: np.ndarray,
    confidence: float = CONFIDENCE_LEVEL,
    n_bootstrap: int = NUM_BOOTSTRAP_SAMPLES,
) -> Dict[str, Tuple[float, float]]:
    """
    Bootstrap confidence intervals for all metrics (Section 4.4).

    Uses the percentile bootstrap method with n_bootstrap resamples.
    """
    rng = np.random.RandomState(42)
    N = preds.shape[0]
    alpha = 1.0 - confidence
    lower_pct = 100 * alpha / 2
    upper_pct = 100 * (1 - alpha / 2)

    residuals_flat = (targets - preds).ravel()

    bootstrap_mae = []
    bootstrap_mse = []
    bootstrap_rmse = []
    bootstrap_res_mean = []
    bootstrap_res_std = []
    bootstrap_skew = []

    for _ in range(n_bootstrap):
        indices = rng.randint(0, len(residuals_flat), size=len(residuals_flat))
        resample = residuals_flat[indices]

        bootstrap_mae.append(np.mean(np.abs(resample)))
        bootstrap_mse.append(np.mean(resample ** 2))
        bootstrap_rmse.append(np.sqrt(np.mean(resample ** 2)))
        bootstrap_res_mean.append(np.mean(resample))
        bootstrap_res_std.append(np.std(resample, ddof=1))
        bootstrap_skew.append(scipy_skew(resample, bias=False))

    ci = {
        'MAE': (
            float(np.percentile(bootstrap_mae, lower_pct)),
            float(np.percentile(bootstrap_mae, upper_pct)),
        ),
        'MSE': (
            float(np.percentile(bootstrap_mse, lower_pct)),
            float(np.percentile(bootstrap_mse, upper_pct)),
        ),
        'RMSE': (
            float(np.percentile(bootstrap_rmse, lower_pct)),
            float(np.percentile(bootstrap_rmse, upper_pct)),
        ),
        'Residual_Mean': (
            float(np.percentile(bootstrap_res_mean, lower_pct)),
            float(np.percentile(bootstrap_res_mean, upper_pct)),
        ),
        'Residual_Std': (
            float(np.percentile(bootstrap_res_std, lower_pct)),
            float(np.percentile(bootstrap_res_std, upper_pct)),
        ),
        'Skewness': (
            float(np.percentile(bootstrap_skew, lower_pct)),
            float(np.percentile(bootstrap_skew, upper_pct)),
        ),
    }
    return ci


def evaluate(
    model: torch.nn.Module,
    test_loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    norm_stats: Dict,
    raw_prices_test: np.ndarray,
    market_indices: Dict,
    device: torch.device,
    compute_ci: bool = True,
    model_name: str = "CMGM",
    target_type: str = "price",
) -> Dict:
    """
    Full evaluation pipeline.

    Reports metrics in BOTH target space (z-scored price or returns —
    matching the loss function) and original price space (for interpretability).

    Args:
        model: Trained model
        test_loader: Test DataLoader
        edge_index: Graph edges
        edge_weight: Graph edge weights
        norm_stats: Per-asset normalisation params {'mean': (N,), 'std': (N,)}
        raw_prices_test: Raw price array for test period (T_test, N)
        market_indices: Dict with market column ranges
        device: torch device
        compute_ci: Whether to compute bootstrap confidence intervals
        model_name: Name for display
        target_type: 'price' or 'return'

    Returns:
        dict: Evaluation results
    """
    print(f"\n{'=' * 60}")
    print(f"{model_name} Evaluation (Section 4.4)")
    print(f"{'=' * 60}")

    commodity_start, commodity_end = market_indices['commodity']

    # Step 1: Generate predictions (always in normalized space)
    print("\n[Step 1] Generating predictions...")
    preds_norm, targets_norm = predict(model, test_loader, edge_index, edge_weight, device)
    print(f"       Predictions shape: {preds_norm.shape}")

    # Step 2: Primary metrics — TARGET space (z-scored price or returns)
    space_label = "Return" if target_type == "return" else "Z-scored"
    print(f"\n[Step 2] Computing metrics in TARGET space ({space_label})...")
    metrics_norm = compute_metrics(preds_norm, targets_norm)

    print(f"\n{'─' * 50}")
    print(f"  Target Space Metrics ({space_label})")
    print(f"{'─' * 50}")
    for name, value in metrics_norm.items():
        print(f"  {name:<18s} {value:.6f}")
    print(f"{'─' * 50}")

    # Step 3: Secondary metrics — ORIGINAL price space
    print("\n[Step 3] Computing metrics in ORIGINAL price space...")
    preds_orig, targets_orig = inverse_transform_predictions(
        preds_norm, targets_norm, norm_stats, raw_prices_test,
        market_indices, target_type=target_type, seq_len=SEQ_LEN,
    )
    metrics_orig = compute_metrics(preds_orig, targets_orig)

    print(f"\n{'─' * 50}")
    print(f"  Original Price Space Metrics (for interpretability)")
    print(f"{'─' * 50}")
    for name, value in metrics_orig.items():
        print(f"  {name:<18s} {value:.4f}")
    print(f"{'─' * 50}")

    results = {
        'metrics_norm': metrics_norm,
        'metrics_orig': metrics_orig,
        'preds_norm': preds_norm,
        'targets_norm': targets_norm,
        'preds_orig': preds_orig,
        'targets_orig': targets_orig,
    }

    # Step 4: Confidence intervals (on normalized space)
    if compute_ci:
        print("\n[Step 4] Bootstrap confidence intervals (normalized space)...")
        ci_norm = compute_confidence_intervals(preds_norm, targets_norm)
        ci_orig = compute_confidence_intervals(preds_orig, targets_orig)

        print(f"\n{'─' * 55}")
        print(f"  Metric           95% CI (normalized)")
        print(f"{'─' * 55}")
        for name, (lower, upper) in ci_norm.items():
            print(f"  {name:<18s} [{lower:.6f}, {upper:.6f}]")
        print(f"{'─' * 55}")

        results['confidence_intervals_norm'] = ci_norm
        results['confidence_intervals_orig'] = ci_orig

    print(f"\n{'=' * 60}\n")
    return results


def evaluate_per_commodity(
    model: torch.nn.Module,
    test_loader: DataLoader,
    edge_index: torch.Tensor,
    edge_weight: torch.Tensor,
    norm_stats: Dict,
    raw_prices_test: np.ndarray,
    market_indices: Dict,
    device: torch.device,
    feature_names: list,
    normalized: bool = True,
    target_type: str = "price",
) -> Dict:
    """
    Per-commodity evaluation.

    Args:
        normalized: If True, compute in target space;
                    if False, compute in original price space.
        target_type: 'price' or 'return'

    Returns:
        dict: Per-commodity MAE, MSE, RMSE
    """
    commodity_start, commodity_end = market_indices['commodity']
    commodity_names = feature_names[commodity_start:commodity_end]

    preds_norm, targets_norm = predict(model, test_loader, edge_index, edge_weight, device)

    if normalized:
        residuals = targets_norm - preds_norm
        space_label = "Target" if target_type == "return" else "Z-scored"
    else:
        preds_orig, targets_orig = inverse_transform_predictions(
            preds_norm, targets_norm, norm_stats, raw_prices_test,
            market_indices, target_type=target_type, seq_len=SEQ_LEN,
        )
        residuals = targets_orig - preds_orig
        space_label = "Original"

    per_commodity = {}
    for i, name in enumerate(commodity_names):
        res = residuals[:, i]
        per_commodity[name] = {
            'MAE': float(np.mean(np.abs(res))),
            'MSE': float(np.mean(res ** 2)),
            'RMSE': float(np.sqrt(np.mean(res ** 2))),
        }

    print(f"\n{'─' * 55}")
    print(f"  Per-Commodity Metrics ({space_label} Space)")
    print(f"{'─' * 55}")
    print(f"  {'Commodity':<20s} {'MAE':<10s} {'MSE':<10s} {'RMSE':<10s}")
    print(f"{'─' * 55}")
    for name, m in per_commodity.items():
        print(f"  {name:<20s} {m['MAE']:<10.6f} {m['MSE']:<10.6f} {m['RMSE']:<10.6f}")
    print(f"{'─' * 55}")

    return per_commodity


# =============================================================================
# Convenience: generic predictor for baselines that dont use edge_index
# =============================================================================
@torch.no_grad()
def predict_no_graph(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray]:
    """Predict for models that dont need graph (LR, SVR, LSTM, BiLSTM)."""
    model.eval()
    all_preds = []
    all_targets = []
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        pred = model(X_batch)
        all_preds.append(pred.cpu().numpy())
        all_targets.append(y_batch.numpy())
    return np.concatenate(all_preds), np.concatenate(all_targets)
