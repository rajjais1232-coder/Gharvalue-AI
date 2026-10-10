"""
tests/test_preprocessing.py — Tests for preprocessing pipeline.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import _generate_synthetic_data, _clean_dataset, _normalise_columns
from src.preprocessing import (
    build_feature_lists, build_preprocessor, prepare_dataframe
)


@pytest.fixture
def sample_df():
    """Return a small clean sample DataFrame."""
    return _generate_synthetic_data(n=100, seed=0)


def test_build_feature_lists_returns_present_columns(sample_df):
    """Feature lists should only contain columns present in the dataframe."""
    num_feats, cat_feats = build_feature_lists(sample_df)
    for col in num_feats:
        assert col in sample_df.columns, f"Numeric feature '{col}' not in dataframe"
    for col in cat_feats:
        assert col in sample_df.columns, f"Categorical feature '{col}' not in dataframe"


def test_build_feature_lists_excludes_target(sample_df):
    """'price' should not appear in feature lists."""
    num_feats, cat_feats = build_feature_lists(sample_df)
    assert "price" not in num_feats
    assert "price" not in cat_feats


def test_build_preprocessor_fit_transform(sample_df):
    """Preprocessor should fit and transform without errors."""
    X, y = prepare_dataframe(sample_df)
    num_feats, cat_feats = build_feature_lists(X)
    preprocessor = build_preprocessor(num_feats, cat_feats)
    X_transformed = preprocessor.fit_transform(X)
    assert X_transformed.shape[0] == len(X)
    assert X_transformed.shape[1] == len(num_feats) + len(cat_feats)


def test_preprocessor_handles_missing_values():
    """Preprocessor should handle NaN in both numeric and categorical columns."""
    df = pd.DataFrame({
        "price": [100000, 200000, 300000, 400000, 500000],
        "area": [500, np.nan, 800, 1000, np.nan],
        "bedrooms": [2, 3, np.nan, 4, 2],
        "bathrooms": [1, 2, 2, np.nan, 1],
        "city": ["Mumbai", np.nan, "Pune", "Delhi", "Mumbai"],
    })
    X, y = prepare_dataframe(df)
    num_feats, cat_feats = build_feature_lists(X)
    preprocessor = build_preprocessor(num_feats, cat_feats)
    X_transformed = preprocessor.fit_transform(X)
    # Should have no NaN after imputation
    assert not np.isnan(X_transformed).any()


def test_prepare_dataframe_removes_invalid_prices():
    """Rows with missing or non-positive price should be removed."""
    df = pd.DataFrame({
        "price": [100000, -5000, np.nan, 300000],
        "area": [500, 600, 700, 800],
        "bedrooms": [2, 3, 2, 4],
        "bathrooms": [1, 2, 1, 2],
        "city": ["A", "B", "C", "D"],
    })
    X, y = prepare_dataframe(df)
    assert len(X) == 2  # only rows 0 and 3 are valid
    assert (y > 0).all()


def test_clean_dataset_removes_duplicates():
    """_clean_dataset should drop duplicate rows."""
    df = pd.DataFrame({
        "price": [100000, 100000, 200000],
        "area": [500, 500, 700],
        "bedrooms": [2, 2, 3],
        "bathrooms": [1, 1, 2],
        "city": ["X", "X", "Y"],
    })
    cleaned = _clean_dataset(df)
    assert len(cleaned) < len(df)


def test_normalise_columns_renames_aliases():
    """Column aliases should be renamed to canonical names."""
    df = pd.DataFrame({
        "Price (INR)": [500000],
        "built_up_area": [1200],
        "BHK": [3],
        "baths": [2],
        "CityName": ["Delhi"],
    })
    normalised = _normalise_columns(df)
    assert "price" in normalised.columns
    assert "area" in normalised.columns
    assert "bedrooms" in normalised.columns
    assert "bathrooms" in normalised.columns
    assert "city" in normalised.columns
