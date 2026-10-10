"""
validation.py — Input validation for prediction requests and uploaded files.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Prediction input schema
# ---------------------------------------------------------------------------
PREDICTION_SCHEMA = {
    "area": {
        "type": float,
        "min": 100,
        "max": 50000,
        "required": True,
        "label": "Built-up area (sq ft)",
    },
    "bedrooms": {
        "type": int,
        "min": 1,
        "max": 10,
        "required": True,
        "label": "Bedrooms (BHK)",
    },
    "bathrooms": {
        "type": int,
        "min": 1,
        "max": 8,
        "required": True,
        "label": "Bathrooms",
    },
    "city": {
        "type": str,
        "required": True,
        "label": "City",
        "min_length": 2,
        "max_length": 80,
    },
    "state": {
        "type": str,
        "required": False,
        "label": "State",
        "min_length": 2,
        "max_length": 80,
    },
    "locality": {
        "type": str,
        "required": False,
        "label": "Locality",
        "max_length": 100,
    },
    "property_type": {
        "type": str,
        "required": False,
        "label": "Property Type",
        "allowed": ["Apartment", "Villa", "Independent House",
                    "Builder Floor", "Penthouse", "Studio", "Other"],
    },
    "furnishing": {
        "type": str,
        "required": False,
        "label": "Furnishing Status",
        "allowed": ["Furnished", "Semi-Furnished", "Unfurnished"],
    },
    "age": {
        "type": float,
        "min": 0,
        "max": 100,
        "required": False,
        "label": "Property age (years)",
    },
    "floor": {
        "type": int,
        "min": 0,
        "max": 100,
        "required": False,
        "label": "Floor number",
    },
    "total_floors": {
        "type": int,
        "min": 1,
        "max": 150,
        "required": False,
        "label": "Total floors",
    },
    "parking": {
        "type": int,
        "min": 0,
        "max": 1,
        "required": False,
        "label": "Parking",
    },
    "carpet_area": {
        "type": float,
        "min": 50,
        "max": 50000,
        "required": False,
        "label": "Carpet area (sq ft)",
    },
}


def validate_prediction_input(inputs: dict[str, Any]) -> list[str]:
    """
    Validate prediction inputs against the schema.

    Returns
    -------
    list[str]
        List of error messages. Empty list means all inputs are valid.
    """
    errors: list[str] = []

    for field, rules in PREDICTION_SCHEMA.items():
        value = inputs.get(field)
        label = rules["label"]

        # Check required fields
        if rules.get("required") and (value is None or value == "" or value != value):
            errors.append(f"{label} is required.")
            continue

        if value is None or value == "":
            continue  # optional field, skip further checks

        # Type coercion check
        expected_type = rules["type"]
        try:
            if expected_type == float:
                value = float(value)
            elif expected_type == int:
                value = int(float(value))  # handle "3.0" → 3
            elif expected_type == str:
                value = str(value).strip()
        except (ValueError, TypeError):
            errors.append(f"{label} must be a {expected_type.__name__}.")
            continue

        # Range checks
        if "min" in rules and value < rules["min"]:
            errors.append(f"{label} must be at least {rules['min']}.")
        if "max" in rules and value > rules["max"]:
            errors.append(f"{label} must not exceed {rules['max']}.")

        # Length checks for strings
        if expected_type == str:
            if "min_length" in rules and len(value) < rules["min_length"]:
                errors.append(
                    f"{label} must be at least {rules['min_length']} characters."
                )
            if "max_length" in rules and len(value) > rules["max_length"]:
                errors.append(
                    f"{label} must not exceed {rules['max_length']} characters."
                )

        # Allowed-values check
        if "allowed" in rules and value not in rules["allowed"]:
            errors.append(
                f"{label} must be one of: {', '.join(rules['allowed'])}. "
                f"Got: '{value}'."
            )

    # Cross-field validation
    area = inputs.get("area")
    carpet = inputs.get("carpet_area")
    if area and carpet:
        try:
            if float(carpet) > float(area):
                errors.append(
                    "Carpet area cannot exceed built-up area."
                )
        except (ValueError, TypeError):
            pass

    floor = inputs.get("floor")
    total = inputs.get("total_floors")
    if floor is not None and total is not None:
        try:
            if int(floor) > int(total):
                errors.append(
                    "Floor number cannot exceed total floors in the building."
                )
        except (ValueError, TypeError):
            pass

    return errors


def validate_csv_file(filename: str, file_size_bytes: int) -> list[str]:
    """Validate an uploaded file before parsing."""
    errors: list[str] = []
    allowed_extensions = {".csv"}
    max_size_mb = 50

    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed_extensions:
        errors.append(
            f"Unsupported file type '{ext}'. Only CSV files are accepted."
        )
    if file_size_bytes > max_size_mb * 1024 * 1024:
        errors.append(
            f"File size {file_size_bytes / 1024 / 1024:.1f} MB exceeds "
            f"the {max_size_mb} MB limit."
        )
    return errors


def check_dataset_sufficiency(df, min_rows: int = 50) -> list[str]:
    """
    Check whether a dataset has enough valid records to train a model.
    Returns a list of warnings/errors.
    """
    from src.data_loader import REQUIRED_COLUMNS
    import pandas as pd

    issues: list[str] = []

    if df is None or df.empty:
        issues.append("Dataset is empty.")
        return issues

    if len(df) < min_rows:
        issues.append(
            f"Dataset has only {len(df)} rows; at least {min_rows} are required."
        )

    missing_req = REQUIRED_COLUMNS - set(df.columns)
    if missing_req:
        issues.append(f"Missing required columns: {missing_req}")

    if "price" in df.columns:
        valid_price = df["price"].dropna()
        if len(valid_price) < min_rows:
            issues.append(
                f"Only {len(valid_price)} rows have a valid price value."
            )
        if (valid_price <= 0).any():
            n_invalid = (valid_price <= 0).sum()
            issues.append(
                f"{n_invalid} rows have non-positive price values."
            )

    return issues
