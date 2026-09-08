"""Mandatory population metrics for new experiment reports (2026-09-08 onward)."""
import numpy as np


STANDARD = "MAE / MSE / RMSE / Hit%; pooled sample × commodity; RMSE=sqrt(MSE); unmasked sign Hit"


def population_metrics(prediction, target):
    p, y = np.asarray(prediction, dtype=np.float64), np.asarray(target, dtype=np.float64)
    if p.shape != y.shape or not p.size:
        raise ValueError("Prediction and target must have equal nonempty shapes")
    if not (np.isfinite(p).all() and np.isfinite(y).all()):
        raise ValueError("Nonfinite prediction or target")
    e = p - y
    mse = float(np.mean(e ** 2))
    return {"MAE": float(np.mean(np.abs(e))), "MSE": mse,
            "RMSE": float(np.sqrt(mse)), "Hit": float(np.mean(np.sign(p) == np.sign(y)))}
