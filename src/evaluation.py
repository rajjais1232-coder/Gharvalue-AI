"""
evaluation.py — Metric computation and evaluation report helpers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Compute MAE, RMSE, R², and MAPE for regression evaluation."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    # MAPE — guard against zero true values
    with np.errstate(divide="ignore", invalid="ignore"):
        ape = np.abs((y_true - y_pred) / y_true)
        ape = ape[np.isfinite(ape)]
        mape = float(np.mean(ape)) * 100 if len(ape) > 0 else float("nan")

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "R2": float(r2),
        "MAPE": mape,
    }


def build_comparison_table(results: dict[str, dict]) -> pd.DataFrame:
    """
    Build a sorted comparison DataFrame from training results.
    Sorted by RMSE ascending (lower is better).
    """
    rows = []
    for name, res in results.items():
        m = res["metrics"]
        rows.append({
            "Model": name,
            "MAE (₹)": m["MAE"],
            "RMSE (₹)": m["RMSE"],
            "R² Score": m["R2"],
            "Train Samples": m["train_size"],
            "Test Samples": m["test_size"],
        })
    df = pd.DataFrame(rows).sort_values("RMSE (₹)").reset_index(drop=True)
    df.index += 1
    return df


def residuals(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Compute residuals (actual − predicted)."""
    return np.array(y_true) - np.array(y_pred)


def build_actual_vs_predicted_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sample_size: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Return a DataFrame of (actual, predicted, residual) for plotting.
    Downsamples if needed for performance.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    if len(y_true) > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y_true), size=sample_size, replace=False)
        y_true = y_true[idx]
        y_pred = y_pred[idx]

    return pd.DataFrame({
        "Actual (₹)": y_true,
        "Predicted (₹)": y_pred,
        "Residual (₹)": y_true - y_pred,
    })


def explain_metrics() -> dict[str, str]:
    """Return human-readable explanations of each evaluation metric."""
    return {
        "MAE": (
            "Mean Absolute Error — average of |actual − predicted|. "
            "Expressed in rupees; lower is better."
        ),
        "RMSE": (
            "Root Mean Squared Error — penalises large errors more heavily than MAE. "
            "Expressed in rupees; lower is better."
        ),
        "R²": (
            "R² (Coefficient of Determination) — fraction of variance explained by the model. "
            "1.0 is perfect; 0 means no better than predicting the mean; "
            "negative values indicate the model is worse than a constant prediction."
        ),
        "MAPE": (
            "Mean Absolute Percentage Error — average percentage deviation. "
            "Scale-independent; lower is better."
        ),
    }
