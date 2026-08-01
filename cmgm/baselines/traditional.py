"""
Traditional ML baselines: Linear Regression (Ridge), Support Vector Regression.

Uses PCA dimensionality reduction + regularized models because the flattened
input dimension (20 * N_nodes ≈ 5680) exceeds training samples (≈1400).

Pipeline: Flatten(X) → PCA → StandardScaler → Ridge/LinearSVR → prediction
"""

import numpy as np
import time
from sklearn.linear_model import Ridge
from sklearn.svm import LinearSVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from torch.utils.data import DataLoader
from typing import Dict, Tuple, Optional


def prepare_sklearn_data(
    loader: DataLoader,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert PyTorch DataLoader to sklearn-compatible flat arrays.

    Each sample: X (20, N, 1) → flattened to (20*N,)
                 y (N_commodities,) → kept as-is
    """
    X_list, y_list = [], []
    for X_batch, y_batch in loader:
        B = X_batch.shape[0]
        X_flat = X_batch.reshape(B, -1).numpy()
        # Squeeze single-horizon dim: (B, 1, Nc) → (B, Nc)
        if y_batch.dim() == 3 and y_batch.size(1) == 1:
            y_batch = y_batch.squeeze(1)
        X_list.append(X_flat)
        y_list.append(y_batch.numpy())
    return np.concatenate(X_list, axis=0), np.concatenate(y_list, axis=0)


def _make_pca_pipeline(n_components: int, model):
    """Build PCA → StandardScaler → model pipeline."""
    return make_pipeline(
        PCA(n_components=n_components, random_state=42, whiten=True),
        StandardScaler(),
        model,
    )


def train_linear_regression(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_commodities: int,
    n_components: int = 100,
    alpha: float = 1.0,
) -> Dict:
    """
    PCA + Ridge Regression baseline.

    Pipeline: Flatten → PCA(100) → StandardScaler → Ridge(alpha=1.0)

    Args:
        n_components: PCA components (default 100)
        alpha: Ridge regularization strength

    Returns:
        dict with 'model' (sklearn pipeline), 'train_time'
    """
    print(f"\n{'=' * 50}")
    print(f"PCA+Ridge Baseline (components={n_components}, alpha={alpha})")
    print(f"{'=' * 50}")

    X_train, y_train = prepare_sklearn_data(train_loader)
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")

    model = Ridge(alpha=alpha, fit_intercept=True, random_state=42)
    pipeline = _make_pca_pipeline(n_components, model)

    print(f"\n[Training] PCA(100) → StandardScaler → Ridge(alpha={alpha})...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.2f}s")

    return {'model': pipeline, 'train_time': train_time}


def train_svr(
    train_loader: DataLoader,
    val_loader: DataLoader,
    n_commodities: int,
    n_components: int = 100,
    C: float = 1.0,
    max_iter: int = 2000,
) -> Dict:
    """
    PCA + Linear SVR baseline.

    Pipeline: Flatten → PCA(100) → StandardScaler → LinearSVR(C=1.0)

    Uses LinearSVR (not kernel SVR) for computational efficiency.

    Args:
        n_components: PCA components (default 100)
        C: Regularization parameter
        max_iter: Solver max iterations

    Returns:
        dict with 'model' (sklearn pipeline), 'train_time'
    """
    print(f"\n{'=' * 50}")
    print(f"PCA+LinearSVR Baseline (components={n_components}, C={C})")
    print(f"{'=' * 50}")

    X_train, y_train = prepare_sklearn_data(train_loader)
    print(f"  X_train: {X_train.shape}, y_train: {y_train.shape}")

    base_svr = LinearSVR(
        C=C, epsilon=0.1, max_iter=max_iter, tol=1e-4,
        loss='squared_epsilon_insensitive', random_state=42,
    )
    svr_multi = MultiOutputRegressor(base_svr, n_jobs=1)
    pipeline = _make_pca_pipeline(n_components, svr_multi)

    print(f"\n[Training] PCA(100) → StandardScaler → LinearSVR(C={C})...")
    t0 = time.time()
    pipeline.fit(X_train, y_train)
    train_time = time.time() - t0
    print(f"  Training time: {train_time:.2f}s")

    return {'model': pipeline, 'train_time': train_time}
