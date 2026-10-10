"""
data_loader.py — Dataset loading, column normalisation, and synthetic demo data.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Column-name aliases: maps common alternative names → canonical name
# ---------------------------------------------------------------------------
COLUMN_ALIASES: dict[str, list[str]] = {
    "price": ["Price", "PRICE", "sale_price", "SalePrice", "cost", "total_price",
              "price_inr", "Price (INR)", "price_lakhs"],
    "area": ["Area", "AREA", "built_up_area", "builtup_area", "super_area",
             "carpet_area_sqft", "size", "sqft", "square_feet", "area_sqft"],
    "bedrooms": ["Bedrooms", "BEDROOMS", "bhk", "BHK", "bedroom", "bed_rooms",
                 "no_of_bedrooms", "num_bedrooms"],
    "bathrooms": ["Bathrooms", "BATHROOMS", "bathroom", "bath", "baths",
                  "no_of_bathrooms"],
    "state": ["State", "STATE", "state_name", "StateName"],
    "city": ["City", "CITY", "city_name", "CityName", "district", "District"],
    "locality": ["Locality", "LOCALITY", "location", "Location", "neighborhood",
                 "area_name", "sublocality"],
    "property_type": ["Property_Type", "PropertyType", "property_category",
                      "type", "Type", "flat_type"],
    "furnishing": ["Furnishing", "FURNISHING", "furnishing_status",
                   "furnish_type", "furnished"],
    "parking": ["Parking", "PARKING", "parking_spaces", "car_parking"],
    "floor": ["Floor", "FLOOR", "floor_no", "floor_number"],
    "total_floors": ["Total_Floors", "TotalFloors", "total_floor",
                     "number_of_floors"],
    "age": ["Age", "AGE", "property_age", "age_of_property", "age_years",
            "year_built"],
    "carpet_area": ["Carpet_Area", "CarpetArea", "carpet_sqft"],
}

REQUIRED_COLUMNS = {"price", "area", "bedrooms", "bathrooms", "city"}
OPTIONAL_COLUMNS = {
    "state", "locality", "property_type", "furnishing",
    "parking", "floor", "total_floors", "age", "carpet_area",
}
ALL_COLUMNS = REQUIRED_COLUMNS | OPTIONAL_COLUMNS

DATA_DIR = Path(__file__).parent.parent / "data"
DEFAULT_CSV = DATA_DIR / "housing_data.csv"


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_dataset(filepath: Optional[str | Path] = None) -> tuple[pd.DataFrame, str]:
    """
    Load and normalise a housing dataset.

    Returns
    -------
    (df, source_label)
        df          — normalised DataFrame
        source_label — human-readable label for display
    """
    path = Path(filepath) if filepath else DEFAULT_CSV

    if path.exists():
        df = _load_csv(path)
        df = _normalise_columns(df)
        df = _clean_dataset(df)
        label = f"📂 {path.name}"
        return df, label

    # Fall back to synthetic demo data
    df = _generate_synthetic_data()
    label = "⚠️ Synthetic Demo Data (no real dataset loaded)"
    return df, label


def load_uploaded_csv(file_obj) -> tuple[pd.DataFrame, list[str]]:
    """
    Load and validate a user-uploaded CSV file.

    Returns
    -------
    (df, warnings_list)
    """
    warns: list[str] = []
    try:
        df = pd.read_csv(file_obj)
    except Exception as exc:
        raise ValueError(f"Could not parse CSV: {exc}") from exc

    if df.empty:
        raise ValueError("Uploaded file contains no data rows.")
    if len(df.columns) < 2:
        raise ValueError("CSV must have at least 2 columns.")

    df = _normalise_columns(df)

    missing_req = REQUIRED_COLUMNS - set(df.columns)
    if "price" in missing_req:
        # Try to detect price-like column
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        if numeric_cols:
            warns.append(
                f"No 'price' column found. Candidate numeric columns: {numeric_cols}. "
                "Use column mapping to assign the target."
            )
        else:
            raise ValueError(
                "No numeric columns found. Cannot identify a price/target column."
            )
    elif missing_req:
        warns.append(f"Missing columns (will be filled with NaN): {missing_req}")

    df = _clean_dataset(df)
    return df, warns


def get_column_mapping_suggestions(df: pd.DataFrame) -> dict[str, str | None]:
    """Return suggested canonical→actual column mappings for a dataframe."""
    suggestions: dict[str, str | None] = {}
    raw_cols_lower = {c.lower(): c for c in df.columns}
    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            suggestions[canonical] = canonical
            continue
        found = None
        for alias in aliases:
            if alias.lower() in raw_cols_lower:
                found = raw_cols_lower[alias.lower()]
                break
        suggestions[canonical] = found
    return suggestions


def apply_column_mapping(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    """Rename columns according to user-confirmed mapping."""
    rename_map = {v: k for k, v in mapping.items() if v and v != k}
    return df.rename(columns=rename_map)


def get_indian_states() -> list[str]:
    """Return all Indian states and UTs in alphabetical order."""
    return sorted([
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
        "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
        "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
        "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
        "West Bengal",
        # Union Territories
        "Andaman and Nicobar Islands", "Chandigarh",
        "Dadra and Nagar Haveli and Daman and Diu", "Delhi (NCT)",
        "Jammu and Kashmir", "Ladakh", "Lakshadweep", "Puducherry",
    ])


def get_cities_by_state() -> dict[str, list[str]]:
    """Return a curated city list per state (not exhaustive — manual entry supported)."""
    return {
        "Andhra Pradesh": ["Visakhapatnam", "Vijayawada", "Guntur", "Tirupati",
                           "Kurnool", "Nellore", "Rajahmundry"],
        "Assam": ["Guwahati", "Silchar", "Dibrugarh", "Jorhat"],
        "Bihar": ["Patna", "Gaya", "Bhagalpur", "Muzaffarpur"],
        "Chhattisgarh": ["Raipur", "Bilaspur", "Durg", "Bhilai", "Korba",
                         "Rajnandgaon", "Raigarh"],
        "Delhi (NCT)": ["New Delhi", "Dwarka", "Rohini", "Saket", "Vasant Kunj",
                        "Noida Extension", "Greater Noida"],
        "Goa": ["Panaji", "Margao", "Vasco da Gama", "Mapusa"],
        "Gujarat": ["Ahmedabad", "Surat", "Vadodara", "Rajkot", "Gandhinagar",
                    "Bhavnagar", "Jamnagar"],
        "Haryana": ["Gurgaon", "Faridabad", "Panipat", "Ambala", "Hisar",
                    "Rohtak", "Karnal"],
        "Himachal Pradesh": ["Shimla", "Dharamsala", "Manali", "Solan"],
        "Jharkhand": ["Ranchi", "Jamshedpur", "Dhanbad", "Bokaro"],
        "Karnataka": ["Bangalore", "Mysore", "Hubli", "Mangalore", "Belgaum",
                      "Davangere", "Bellary"],
        "Kerala": ["Thiruvananthapuram", "Kochi", "Kozhikode", "Thrissur",
                   "Kollam", "Kannur"],
        "Madhya Pradesh": ["Bhopal", "Indore", "Jabalpur", "Gwalior",
                           "Ujjain", "Sagar"],
        "Maharashtra": ["Mumbai", "Pune", "Nagpur", "Nashik", "Aurangabad",
                        "Thane", "Navi Mumbai", "Kolhapur"],
        "Odisha": ["Bhubaneswar", "Cuttack", "Rourkela", "Berhampur"],
        "Punjab": ["Ludhiana", "Amritsar", "Jalandhar", "Patiala", "Chandigarh"],
        "Rajasthan": ["Jaipur", "Jodhpur", "Udaipur", "Kota", "Bikaner", "Ajmer"],
        "Tamil Nadu": ["Chennai", "Coimbatore", "Madurai", "Tiruchirappalli",
                       "Salem", "Tirunelveli"],
        "Telangana": ["Hyderabad", "Warangal", "Nizamabad", "Karimnagar",
                      "Khammam"],
        "Uttar Pradesh": ["Lucknow", "Kanpur", "Agra", "Varanasi", "Allahabad",
                          "Meerut", "Ghaziabad", "Noida"],
        "Uttarakhand": ["Dehradun", "Haridwar", "Roorkee", "Nainital"],
        "West Bengal": ["Kolkata", "Howrah", "Asansol", "Siliguri", "Durgapur"],
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _load_csv(path: Path) -> pd.DataFrame:
    encodings = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]
    for enc in encodings:
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Cannot read {path.name} with any supported encoding.")


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename DataFrame columns to canonical names where possible."""
    col_map: dict[str, str] = {}
    raw_lower = {c.strip().lower(): c for c in df.columns}

    for canonical, aliases in COLUMN_ALIASES.items():
        if canonical in df.columns:
            continue  # already canonical
        for alias in [canonical] + aliases:
            if alias.strip().lower() in raw_lower:
                original = raw_lower[alias.strip().lower()]
                if original != canonical:
                    col_map[original] = canonical
                break

    if col_map:
        df = df.rename(columns=col_map)

    # Strip whitespace from string columns
    str_cols = df.select_dtypes(include="object").columns
    for col in str_cols:
        df[col] = df[col].astype(str).str.strip()

    return df


def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicates, coerce types, and filter obviously invalid rows."""
    df = df.drop_duplicates().copy()

    # Ensure numeric types for key columns
    for col in ["price", "area", "bedrooms", "bathrooms", "age",
                "floor", "total_floors", "carpet_area"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove rows where price or area is non-positive
    if "price" in df.columns:
        df = df[df["price"] > 0]
    if "area" in df.columns:
        df = df[df["area"] > 0]

    # Clip extreme outliers (>= 3× IQR fence) for price and area
    for col in ["price", "area"]:
        if col in df.columns:
            q1 = df[col].quantile(0.01)
            q99 = df[col].quantile(0.99)
            df = df[(df[col] >= q1) & (df[col] <= q99)]

    # Standardise categorical values
    for col in ["property_type", "furnishing", "state", "city"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()
            df[col] = df[col].replace({"Nan": np.nan, "None": np.nan, "": np.nan})

    # Clip bedroom/bathroom counts to reasonable values
    if "bedrooms" in df.columns:
        df["bedrooms"] = df["bedrooms"].clip(lower=1, upper=10)
    if "bathrooms" in df.columns:
        df["bathrooms"] = df["bathrooms"].clip(lower=1, upper=8)

    df = df.reset_index(drop=True)
    return df


def _generate_synthetic_data(n: int = 1500, seed: int = 42) -> pd.DataFrame:
    """
    Generate a clearly-labelled synthetic demo dataset.
    IMPORTANT: Results from this data are NOT real market data.
    """
    rng = np.random.default_rng(seed)

    city_state_map = {
        "Mumbai": ("Maharashtra", 15000),
        "Pune": ("Maharashtra", 8000),
        "Nagpur": ("Maharashtra", 4500),
        "Bangalore": ("Karnataka", 9000),
        "Mysore": ("Karnataka", 4000),
        "Hyderabad": ("Telangana", 7000),
        "Chennai": ("Tamil Nadu", 7500),
        "Delhi": ("Delhi (NCT)", 12000),
        "Noida": ("Uttar Pradesh", 6000),
        "Gurgaon": ("Haryana", 8500),
        "Kolkata": ("West Bengal", 5500),
        "Ahmedabad": ("Gujarat", 5000),
        "Jaipur": ("Rajasthan", 4500),
        "Lucknow": ("Uttar Pradesh", 4000),
        "Chandigarh": ("Punjab", 5500),
        "Raipur": ("Chhattisgarh", 3500),
        "Bilaspur": ("Chhattisgarh", 2800),
        "Indore": ("Madhya Pradesh", 5000),
        "Bhopal": ("Madhya Pradesh", 4200),
        "Kochi": ("Kerala", 6000),
    }

    cities = list(city_state_map.keys())
    city_col = rng.choice(cities, size=n)

    property_types = ["Apartment", "Villa", "Independent House",
                      "Builder Floor", "Penthouse"]
    furnishing_status = ["Furnished", "Semi-Furnished", "Unfurnished"]

    rows = []
    for city in city_col:
        state, base_price_sqft = city_state_map[city]
        bedrooms = int(rng.choice([1, 2, 3, 4, 5], p=[0.1, 0.3, 0.35, 0.2, 0.05]))
        bathrooms = min(bedrooms + rng.integers(-1, 2), 8)
        bathrooms = max(1, bathrooms)
        area = float(rng.integers(400 + bedrooms * 100, 800 + bedrooms * 300))
        prop_type = rng.choice(property_types)
        furnishing = rng.choice(furnishing_status)
        age = int(rng.integers(0, 30))
        floor = int(rng.integers(0, 15))
        total_floors = max(floor, int(rng.integers(2, 20)))
        parking = int(rng.choice([0, 1], p=[0.3, 0.7]))

        # Price model: area × local_rate × bedroom_factor × furnishing_factor + noise
        bedroom_factor = 1 + (bedrooms - 2) * 0.08
        furnish_factor = {"Furnished": 1.15, "Semi-Furnished": 1.05, "Unfurnished": 1.0}[furnishing]
        type_factor = {"Villa": 1.4, "Penthouse": 1.5, "Independent House": 1.2,
                       "Apartment": 1.0, "Builder Floor": 0.95}[prop_type]
        age_discount = max(0.7, 1 - age * 0.005)
        noise = rng.normal(1.0, 0.08)

        price = (area * base_price_sqft * bedroom_factor
                 * furnish_factor * type_factor * age_discount * noise)
        price = max(price, 500000)  # minimum 5 lakh

        rows.append({
            "price": round(price, -3),
            "area": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "state": state,
            "city": city,
            "locality": f"Sector {rng.integers(1, 50)}",
            "property_type": prop_type,
            "furnishing": furnishing,
            "parking": parking,
            "floor": floor,
            "total_floors": total_floors,
            "age": age,
        })

    df = pd.DataFrame(rows)
    df["_synthetic"] = True  # flag for display purposes
    return df
