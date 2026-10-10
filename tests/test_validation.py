"""
tests/test_validation.py — Tests for input and file validation.
"""

import pytest

from src.validation import (
    validate_prediction_input,
    validate_csv_file,
    check_dataset_sufficiency,
)
import pandas as pd
import numpy as np


# ── Prediction input validation ───────────────────────────────────────────────

def test_valid_inputs_pass():
    """A complete valid input dict should return no errors."""
    errors = validate_prediction_input({
        "area": 1200.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "city": "Bangalore",
        "state": "Karnataka",
        "property_type": "Apartment",
        "furnishing": "Furnished",
    })
    assert errors == []


def test_missing_required_area_fails():
    errors = validate_prediction_input({
        "bedrooms": 2, "bathrooms": 1, "city": "Mumbai",
    })
    assert any("area" in e.lower() for e in errors)


def test_missing_city_fails():
    errors = validate_prediction_input({
        "area": 800, "bedrooms": 2, "bathrooms": 1, "city": "",
    })
    assert any("city" in e.lower() for e in errors)


def test_area_below_minimum_fails():
    errors = validate_prediction_input({
        "area": 50,  # below min of 100
        "bedrooms": 2, "bathrooms": 1, "city": "Delhi",
    })
    assert any("area" in e.lower() for e in errors)


def test_area_above_maximum_fails():
    errors = validate_prediction_input({
        "area": 100_000,  # above max of 50000
        "bedrooms": 2, "bathrooms": 1, "city": "Delhi",
    })
    assert any("area" in e.lower() for e in errors)


def test_bedrooms_below_min_fails():
    errors = validate_prediction_input({
        "area": 800, "bedrooms": 0, "bathrooms": 1, "city": "Pune",
    })
    assert any("bedroom" in e.lower() for e in errors)


def test_bathrooms_above_max_fails():
    errors = validate_prediction_input({
        "area": 800, "bedrooms": 2, "bathrooms": 20, "city": "Pune",
    })
    assert any("bathroom" in e.lower() for e in errors)


def test_invalid_furnishing_value_fails():
    errors = validate_prediction_input({
        "area": 800, "bedrooms": 2, "bathrooms": 1,
        "city": "Mumbai", "furnishing": "Partially",
    })
    assert any("furnishing" in e.lower() for e in errors)


def test_carpet_area_greater_than_area_fails():
    errors = validate_prediction_input({
        "area": 800, "bedrooms": 2, "bathrooms": 1,
        "city": "Hyderabad", "carpet_area": 1000,  # > area
    })
    assert any("carpet" in e.lower() for e in errors)


def test_floor_greater_than_total_floors_fails():
    errors = validate_prediction_input({
        "area": 800, "bedrooms": 2, "bathrooms": 1,
        "city": "Chennai", "floor": 20, "total_floors": 5,
    })
    assert any("floor" in e.lower() for e in errors)


def test_optional_fields_absent_passes():
    """Missing optional fields should not cause errors."""
    errors = validate_prediction_input({
        "area": 900, "bedrooms": 2, "bathrooms": 1, "city": "Jaipur",
    })
    assert errors == []


# ── File validation ───────────────────────────────────────────────────────────

def test_valid_csv_file_passes():
    errors = validate_csv_file("data.csv", 1024 * 100)  # 100 KB
    assert errors == []


def test_invalid_extension_fails():
    errors = validate_csv_file("model.exe", 1024)
    assert len(errors) > 0


def test_xlsx_extension_fails():
    errors = validate_csv_file("data.xlsx", 1024)
    assert len(errors) > 0


def test_oversized_file_fails():
    errors = validate_csv_file("huge.csv", 200 * 1024 * 1024)  # 200 MB
    assert len(errors) > 0


# ── Dataset sufficiency ───────────────────────────────────────────────────────

def test_empty_dataframe_is_insufficient():
    issues = check_dataset_sufficiency(pd.DataFrame())
    assert len(issues) > 0


def test_none_dataframe_is_insufficient():
    issues = check_dataset_sufficiency(None)
    assert len(issues) > 0


def test_too_few_rows_insufficient():
    df = pd.DataFrame({
        "price": [100000] * 10,
        "area": [500] * 10,
        "bedrooms": [2] * 10,
        "bathrooms": [1] * 10,
        "city": ["X"] * 10,
    })
    issues = check_dataset_sufficiency(df, min_rows=50)
    assert len(issues) > 0


def test_sufficient_dataset_no_issues():
    from src.data_loader import _generate_synthetic_data
    df = _generate_synthetic_data(n=200, seed=0)
    issues = check_dataset_sufficiency(df, min_rows=50)
    assert issues == []
