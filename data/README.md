# Dataset Directory

## Expected Dataset Format

Place your CSV dataset here as `housing_data.csv`.

### Required Columns
- `price` — Property price in INR (target variable)
- `area` — Built-up area in square feet
- `bedrooms` — Number of bedrooms (BHK)
- `bathrooms` — Number of bathrooms
- `state` — Indian state or Union Territory
- `city` — City or district

### Optional Columns
- `locality` — Locality or neighborhood
- `property_type` — Type (Apartment, Villa, Independent House, etc.)
- `furnishing` — Furnishing status (Furnished, Semi-Furnished, Unfurnished)
- `parking` — Parking availability (Yes/No or 0/1)
- `floor` — Floor number
- `total_floors` — Total number of floors in the building
- `age` — Property age in years
- `carpet_area` — Carpet area in square feet

## Supported Datasets

The application auto-detects column names from common Indian housing datasets:
- MagicBricks / 99acres export format
- Housing.com export format
- Custom CSV with configurable column mapping

## Synthetic Demo Data

If no dataset is present, the application generates a clearly-labelled **synthetic demo dataset**
to demonstrate the application mechanics. Predictions from synthetic data are NOT representative
of the real Indian housing market.

## Dataset Sources (Suggested)

- [Kaggle — Indian Housing Dataset](https://www.kaggle.com/datasets)
- [data.gov.in](https://data.gov.in) — Open Government Data Platform India

## License

Ensure any dataset you use is appropriately licensed for educational / research use.
