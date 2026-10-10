"""
model_training.py — Train, evaluate, compare, and persist ML models.
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                               RandomForestRegressor)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.preprocessing import build_feature_lists, build_preprocessor, prepare_dataframe

MODELS_DIR = Path(__file__).parent.parent / "models"
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
COMPARISON_PATH = MODELS_DIR / "model_comparison.json"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
def get_model_registry() -> dict[str, Any]:
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.08,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
        ),
    }


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_all_models(
    df: pd.DataFrame,
    progress_callback=None,
) -> dict[str, dict]:
    """
    Train and evaluate all models. Returns a dict of results per model name.

    Parameters
    ----------
    df               : Dataset DataFrame (must have 'price' column)
    progress_callback: Optional callable(model_name, step, total) for UI updates

    Returns
    -------
    dict: {model_name: {metrics, pipeline, y_test, y_pred, feature_names}}
    """
    X, y = prepare_dataframe(df)
    num_feats, cat_feats = build_feature_lists(X)

    if len(X) < 50:
        raise ValueError(
            f"Not enough data to train: {len(X)} rows. Need at least 50."
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    preprocessor = build_preprocessor(num_feats, cat_feats)
    results: dict[str, dict] = {}
    registry = get_model_registry()

    for i, (name, estimator) in enumerate(registry.items()):
        if progress_callback:
            progress_callback(name, i, len(registry))

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator),
        ])

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        results[name] = {
            "metrics": {
                "MAE": round(float(mae), 2),
                "RMSE": round(float(rmse), 2),
                "R2": round(float(r2), 4),
                "train_size": len(X_train),
                "test_size": len(X_test),
            },
            "pipeline": pipeline,
            "y_test": y_test.values,
            "y_pred": y_pred,
            "num_features": num_feats,
            "cat_features": cat_feats,
            "feature_names": num_feats + cat_feats,
        }

    return results


def select_best_model(results: dict[str, dict]) -> str:
    """Select the best model by lowest RMSE on the test set."""
    return min(results, key=lambda name: results[name]["metrics"]["RMSE"])


def save_model(
    pipeline: Pipeline,
    model_name: str,
    metrics: dict,
    feature_names: list[str],
    dataset_label: str,
) -> None:
    """Save the model pipeline and metadata to disk."""
    MODELS_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    metadata = {
        "model_name": model_name,
        "trained_at": datetime.now().isoformat(),
        "dataset_label": dataset_label,
        "feature_names": feature_names,
        "metrics": metrics,
        "sklearn_version": _get_sklearn_version(),
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)


def save_comparison(results: dict[str, dict]) -> None:
    """Save model comparison results (metrics only) to disk."""
    MODELS_DIR.mkdir(exist_ok=True)
    comparison = {
        name: res["metrics"] for name, res in results.items()
    }
    with open(COMPARISON_PATH, "w", encoding="utf-8") as fh:
        json.dump(comparison, fh, indent=2)


def load_model() -> tuple[Pipeline | None, dict | None]:
    """
    Load the saved model pipeline and metadata.

    Returns
    -------
    (pipeline, metadata) or (None, None) if not found
    """
    if not MODEL_PATH.exists():
        return None, None
    if not METADATA_PATH.exists():
        return None, None

    pipeline = joblib.load(MODEL_PATH)
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)

    return pipeline, metadata


def load_comparison() -> dict | None:
    if COMPARISON_PATH.exists():
        with open(COMPARISON_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def get_feature_importance(
    pipeline: Pipeline,
    feature_names: list[str],
    model_name: str,
) -> pd.DataFrame | None:
    """Extract feature importances for tree-based models."""
    model = pipeline.named_steps.get("model")
    if model is None:
        return None

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if len(importances) == len(feature_names):
            return (
                pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": importances,
                })
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )
    elif model_name == "Linear Regression" and hasattr(model, "coef_"):
        coefficients = np.abs(model.coef_)
        if len(coefficients) == len(feature_names):
            return (
                pd.DataFrame({
                    "Feature": feature_names,
                    "Importance": coefficients / coefficients.sum(),
                })
                .sort_values("Importance", ascending=False)
                .reset_index(drop=True)
            )
    return None


def _get_sklearn_version() -> str:
    try:
        import sklearn
        return sklearn.__version__
    except Exception:
        return "unknown"
