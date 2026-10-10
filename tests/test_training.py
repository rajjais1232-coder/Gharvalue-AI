"""
tests/test_training.py — Tests for the model training pipeline.
"""

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

from src.data_loader import _generate_synthetic_data
from src.model_training import (
    train_all_models, select_best_model, get_feature_importance,
    get_model_registry
)
from src.evaluation import compute_metrics, build_comparison_table


@pytest.fixture(scope="module")
def training_results():
    """Train all models once and reuse results across tests."""
    df = _generate_synthetic_data(n=300, seed=1)
    return train_all_models(df)


def test_all_models_trained(training_results):
    """All four models should be in results."""
    expected = {"Linear Regression", "Random Forest", "Extra Trees", "Gradient Boosting"}
    assert set(training_results.keys()) == expected


def test_result_structure(training_results):
    """Each result dict should have required keys."""
    required_keys = {"metrics", "pipeline", "y_test", "y_pred",
                     "num_features", "cat_features", "feature_names"}
    for name, res in training_results.items():
        assert required_keys.issubset(set(res.keys())), (
            f"Missing keys in results for {name}"
        )


def test_pipeline_is_sklearn_pipeline(training_results):
    """Each stored pipeline must be a Scikit-learn Pipeline."""
    for name, res in training_results.items():
        assert isinstance(res["pipeline"], Pipeline), (
            f"Expected Pipeline for {name}, got {type(res['pipeline'])}"
        )


def test_predictions_have_correct_shape(training_results):
    """y_pred and y_test should have the same length."""
    for name, res in training_results.items():
        assert len(res["y_test"]) == len(res["y_pred"]), (
            f"Shape mismatch for {name}"
        )


def test_metrics_are_finite(training_results):
    """All reported metrics should be finite numbers."""
    for name, res in training_results.items():
        m = res["metrics"]
        for metric_name, val in m.items():
            if isinstance(val, (int, float)):
                assert np.isfinite(val), (
                    f"{metric_name} is not finite for {name}: {val}"
                )


def test_select_best_model_returns_valid_name(training_results):
    """select_best_model should return a name present in results."""
    best = select_best_model(training_results)
    assert best in training_results


def test_comparison_table_sorted_by_rmse(training_results):
    """Comparison table should be sorted ascending by RMSE."""
    comp_df = build_comparison_table(training_results)
    rmse_vals = comp_df["RMSE (₹)"].tolist()
    assert rmse_vals == sorted(rmse_vals)


def test_compute_metrics():
    """compute_metrics should return correct keys and finite values."""
    y_true = np.array([100, 200, 300, 400, 500], dtype=float)
    y_pred = np.array([110, 190, 310, 390, 510], dtype=float)
    metrics = compute_metrics(y_true, y_pred)
    assert "MAE" in metrics
    assert "RMSE" in metrics
    assert "R2" in metrics
    assert "MAPE" in metrics
    assert all(np.isfinite(v) for v in metrics.values())


def test_perfect_predictions_r2_is_one():
    """When predictions equal actuals, R² should be 1."""
    y = np.array([1e6, 2e6, 3e6, 4e6, 5e6])
    metrics = compute_metrics(y, y)
    assert abs(metrics["R2"] - 1.0) < 1e-9


def test_feature_importance_tree_model(training_results):
    """Feature importance should return a DataFrame for tree-based models."""
    for model_name in ["Random Forest", "Extra Trees", "Gradient Boosting"]:
        res = training_results[model_name]
        fi = get_feature_importance(
            res["pipeline"], res["feature_names"], model_name
        )
        assert fi is not None, f"Expected feature importance for {model_name}"
        assert "Feature" in fi.columns
        assert "Importance" in fi.columns
        assert len(fi) > 0


def test_too_few_rows_raises_error():
    """Training should fail with a clear error when data is insufficient."""
    df = _generate_synthetic_data(n=5, seed=0)
    with pytest.raises((ValueError, Exception)):
        train_all_models(df)
