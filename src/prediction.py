"""
prediction.py — Generate predictions from a trained pipeline.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.model_training import load_model
from src.validation import validate_prediction_input


def predict_price(inputs: dict[str, Any]) -> dict:
    """
    Validate inputs, run the trained model, and return a prediction dict.

    Returns
    -------
    dict with keys:
        success   (bool)
        price     (float | None)
        errors    (list[str])
        warnings  (list[str])
        model_name (str | None)
        is_synthetic (bool)
    """
    result = {
        "success": False,
        "price": None,
        "errors": [],
        "warnings": [],
        "model_name": None,
        "is_synthetic": False,
    }

    # --- Validation ---
    errors = validate_prediction_input(inputs)
    if errors:
        result["errors"] = errors
        return result

    # --- Load model ---
    pipeline, metadata = load_model()
    if pipeline is None:
        result["errors"].append(
            "No trained model found. Please go to Model Evaluation and train a model first."
        )
        return result

    result["model_name"] = metadata.get("model_name", "Unknown")

    # Check if dataset was synthetic
    if "synthetic" in metadata.get("dataset_label", "").lower():
        result["is_synthetic"] = True
        result["warnings"].append(
            "⚠️ This model was trained on SYNTHETIC demo data. "
            "Predictions do not reflect real Indian market prices."
        )

    # --- Build input DataFrame matching training features ---
    feature_names = metadata.get("feature_names", [])
    if not feature_names:
        result["errors"].append("Model metadata is missing feature names.")
        return result

    # Check location coverage
    city = inputs.get("city", "")
    state = inputs.get("state", "")
    if city and city not in _get_known_cities(metadata):
        result["warnings"].append(
            f"'{city}' was not well-represented in training data. "
            "Prediction accuracy may be lower for this location."
        )

    # Build single-row dataframe with all expected columns
    row = {}
    for feat in feature_names:
        row[feat] = inputs.get(feat, np.nan)

    input_df = pd.DataFrame([row])

    # Coerce types
    numeric_cols = ["area", "bedrooms", "bathrooms", "age", "floor",
                    "total_floors", "parking", "carpet_area"]
    for col in numeric_cols:
        if col in input_df.columns:
            input_df[col] = pd.to_numeric(input_df[col], errors="coerce")

    # --- Predict ---
    try:
        price = float(pipeline.predict(input_df)[0])
        if price <= 0:
            result["errors"].append("Model returned an invalid (non-positive) price.")
            return result
        result["success"] = True
        result["price"] = round(price, -3)  # round to nearest ₹1000
    except Exception as exc:
        result["errors"].append(f"Prediction failed: {exc}")

    return result


def format_inr(amount: float) -> str:
    """Format a rupee amount in Indian number format."""
    amount = int(amount)
    if amount >= 1_00_00_000:  # >= 1 Crore
        crore = amount / 1_00_00_000
        return f"₹{crore:.2f} Cr"
    elif amount >= 1_00_000:  # >= 1 Lakh
        lakh = amount / 1_00_000
        return f"₹{lakh:.2f} L"
    else:
        return f"₹{amount:,}"


def format_inr_full(amount: float) -> str:
    """Return full amount with comma grouping in Indian style."""
    amount = int(amount)
    # Indian comma formatting: 1,23,45,678
    s = str(amount)
    if len(s) <= 3:
        return f"₹{s}"
    result = s[-3:]
    s = s[:-3]
    while len(s) > 2:
        result = s[-2:] + "," + result
        s = s[:-2]
    if s:
        result = s + "," + result
    return f"₹{result}"


def get_price_range(price: float, pct: float = 0.10) -> tuple[float, float]:
    """
    Return a simple ±pct% range around the predicted price.
    This is a heuristic interval, not a statistically calibrated prediction interval.
    """
    low = price * (1 - pct)
    high = price * (1 + pct)
    return round(low, -3), round(high, -3)


def _get_known_cities(metadata: dict) -> set[str]:
    """Extract known cities from model metadata if available."""
    return set(metadata.get("known_cities", []))
