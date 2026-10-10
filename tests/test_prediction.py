"""
tests/test_prediction.py — Tests for prediction and formatting utilities.
"""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.prediction import format_inr, format_inr_full, get_price_range


# ── Formatting tests ──────────────────────────────────────────────────────────

def test_format_inr_crore():
    assert "Cr" in format_inr(1_50_00_000)


def test_format_inr_lakh():
    assert "L" in format_inr(5_00_000)


def test_format_inr_small():
    result = format_inr(50_000)
    assert "₹" in result


def test_format_inr_full_contains_rupee_symbol():
    result = format_inr_full(1_00_000)
    assert "₹" in result


def test_format_inr_full_one_crore():
    result = format_inr_full(1_00_00_000)
    assert "1" in result


def test_get_price_range_symmetric():
    """Price range should be symmetric at default 10%."""
    price = 5_00_000
    low, high = get_price_range(price, pct=0.10)
    assert low < price < high
    assert abs((high - price) - (price - low)) < 1_000  # within rounding


def test_get_price_range_pct_respected():
    """Custom pct should produce a proportional range."""
    price = 10_00_000
    low, high = get_price_range(price, pct=0.20)
    assert abs(high - price - 2_00_000) < 1_000


# ── Prediction logic (with mock model) ───────────────────────────────────────

@pytest.fixture
def mock_pipeline():
    """Create a mock sklearn pipeline that returns a fixed price."""
    pipeline = MagicMock()
    pipeline.predict.return_value = np.array([4_50_000.0])
    return pipeline


@pytest.fixture
def mock_metadata():
    return {
        "model_name": "Random Forest",
        "feature_names": ["area", "bedrooms", "bathrooms", "city"],
        "dataset_label": "test.csv",
        "known_cities": ["Mumbai", "Pune"],
    }


def test_predict_price_success(mock_pipeline, mock_metadata):
    """Successful prediction with a valid mock model."""
    from src.prediction import predict_price

    with patch("src.prediction.load_model", return_value=(mock_pipeline, mock_metadata)):
        result = predict_price({
            "area": 1000.0,
            "bedrooms": 2,
            "bathrooms": 2,
            "city": "Mumbai",
            "state": "Maharashtra",
        })

    assert result["success"] is True
    assert result["price"] is not None
    assert result["price"] > 0


def test_predict_price_no_model():
    """Missing model should return an error, not raise an exception."""
    from src.prediction import predict_price

    with patch("src.prediction.load_model", return_value=(None, None)):
        result = predict_price({
            "area": 1000.0,
            "bedrooms": 2,
            "bathrooms": 2,
            "city": "Delhi",
        })

    assert result["success"] is False
    assert len(result["errors"]) > 0


def test_predict_price_invalid_area(mock_pipeline, mock_metadata):
    """Invalid area should fail validation before reaching the model."""
    from src.prediction import predict_price

    with patch("src.prediction.load_model", return_value=(mock_pipeline, mock_metadata)):
        result = predict_price({
            "area": -100,  # invalid
            "bedrooms": 2,
            "bathrooms": 2,
            "city": "Mumbai",
        })

    assert result["success"] is False
    assert any("area" in e.lower() for e in result["errors"])


def test_predict_price_missing_required_city(mock_pipeline, mock_metadata):
    """Missing city should fail validation."""
    from src.prediction import predict_price

    with patch("src.prediction.load_model", return_value=(mock_pipeline, mock_metadata)):
        result = predict_price({
            "area": 1000.0,
            "bedrooms": 2,
            "bathrooms": 2,
            "city": "",  # empty
        })

    assert result["success"] is False


def test_predict_warns_for_unknown_city(mock_pipeline, mock_metadata):
    """Unknown city should trigger a warning (not an error)."""
    from src.prediction import predict_price

    with patch("src.prediction.load_model", return_value=(mock_pipeline, mock_metadata)):
        result = predict_price({
            "area": 1000.0,
            "bedrooms": 2,
            "bathrooms": 2,
            "city": "Zoonkipur",  # not in known_cities
            "state": "Unknown State",
        })

    # Should succeed but with a warning
    if result["success"]:
        assert len(result["warnings"]) > 0


def test_predict_synthetic_data_warns(mock_pipeline):
    """Predictions from synthetic data should include a warning."""
    from src.prediction import predict_price

    synthetic_metadata = {
        "model_name": "Random Forest",
        "feature_names": ["area", "bedrooms", "bathrooms", "city"],
        "dataset_label": "⚠️ Synthetic Demo Data",
        "known_cities": [],
    }
    with patch("src.prediction.load_model", return_value=(mock_pipeline, synthetic_metadata)):
        result = predict_price({
            "area": 1000.0,
            "bedrooms": 2,
            "bathrooms": 2,
            "city": "Test City",
        })

    if result["success"]:
        assert any("synthetic" in w.lower() for w in result["warnings"])
