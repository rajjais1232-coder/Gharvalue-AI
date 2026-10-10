"""
preprocessing.py — Scikit-learn compatible preprocessing pipeline builders.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------
NUMERIC_FEATURES = ["area", "bedrooms", "bathrooms", "age",
                    "floor", "total_floors", "carpet_area", "parking"]
CATEGORICAL_FEATURES = ["city", "state", "property_type", "furnishing", "locality"]

# Subset that must exist; extras are added when present in the data
CORE_NUMERIC = ["area", "bedrooms", "bathrooms"]
CORE_CATEGORICAL = ["city"]


def build_feature_lists(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """
    Return (numeric_cols, categorical_cols) actually present in df (excluding target).
    """
    num_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
    cat_cols = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    return num_cols, cat_cols


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build a ColumnTransformer that handles:
    - Numeric: median imputation + standard scaling
    - Categorical: most-frequent imputation + ordinal encoding (handles unknowns)
    """
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            dtype=np.float64,
        )),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


def prepare_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X (features) and y (target) from the loaded dataset.
    Drops rows where the target is missing.
    """
    if "price" not in df.columns:
        raise ValueError("Dataset must contain a 'price' column as the target variable.")

    df = df.copy()

    # Drop internal markers
    drop_cols = ["price", "_synthetic"]
    df_clean = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    y = df["price"].copy()
    valid_mask = y.notna() & (y > 0)
    X = df_clean[valid_mask].reset_index(drop=True)
    y = y[valid_mask].reset_index(drop=True)

    return X, y


def get_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    """Extract feature names from a fitted ColumnTransformer."""
    try:
        return list(preprocessor.get_feature_names_out())
    except Exception:
        # Fallback for older sklearn versions
        names = []
        for name, transformer, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            if isinstance(cols, list):
                names.extend(cols)
        return names
