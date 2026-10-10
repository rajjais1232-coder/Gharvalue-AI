"""
pages/5_Dataset_Explorer.py — Upload, preview, and validate datasets.
"""

import io
import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Dataset Explorer — GharValue AI",
    page_icon="📁",
    layout="wide",
)

from app import inject_css, init_session_state, ensure_dataset
from src.data_loader import (
    load_uploaded_csv, get_column_mapping_suggestions,
    apply_column_mapping, load_dataset, REQUIRED_COLUMNS, OPTIONAL_COLUMNS
)
from src.validation import validate_csv_file, check_dataset_sufficiency

inject_css()
init_session_state()
ensure_dataset()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 GharValue AI")
    st.markdown("---")
    st.page_link("app.py", label="🏠 Home Dashboard")
    st.page_link("pages/1_Prediction.py", label="🔮 Predict Price")
    st.page_link("pages/2_Market_Insights.py", label="📊 Market Insights")
    st.page_link("pages/3_Model_Evaluation.py", label="🧪 Model Evaluation")
    st.page_link("pages/4_3D_Analytics.py", label="🌐 3D Analytics")
    st.page_link("pages/5_Dataset_Explorer.py", label="📁 Dataset Explorer")
    st.page_link("pages/6_Prediction_History.py", label="📋 Prediction History")
    st.page_link("pages/7_About_Project.py", label="ℹ️ About Project")

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>📁 Dataset Explorer</h1>
    <p>Upload, inspect, and validate your housing dataset</p>
</div>
""", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_current, tab_upload = st.tabs(["Current Dataset", "Upload New Dataset"])

# ────────────────────────────────────────────────────────────────────────────
# Tab 1: Current Dataset
# ────────────────────────────────────────────────────────────────────────────
with tab_current:
    df: pd.DataFrame = st.session_state.dataset
    is_synthetic = "_synthetic" in df.columns
    label = st.session_state.dataset_label

    st.markdown(f"**Source:** {label}")
    if is_synthetic:
        st.warning(
            "**Synthetic Demo Data** is active. Upload a real CSV to replace it.",
            icon="⚠️",
        )

    # Basic info row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", f"{len(df.columns):,}")
    c3.metric("Duplicates", f"{df.duplicated().sum():,}")
    c4.metric("Missing Values", f"{df.isnull().sum().sum():,}")

    st.markdown("---")

    # Preview
    st.markdown("#### Preview (first 50 rows)")
    st.dataframe(df.head(50), use_container_width=True, height=300)

    # Column info
    st.markdown("#### Column Information")
    col_info = pd.DataFrame({
        "Column": df.columns,
        "Type": df.dtypes.values,
        "Non-Null": df.notnull().sum().values,
        "Null": df.isnull().sum().values,
        "Null%": (df.isnull().sum() / len(df) * 100).round(1).values,
        "Unique": df.nunique().values,
    })
    # Annotate canonical vs unknown columns
    all_canonical = REQUIRED_COLUMNS | OPTIONAL_COLUMNS | {"_synthetic"}
    col_info["Status"] = col_info["Column"].apply(
        lambda c: "✅ Required" if c in REQUIRED_COLUMNS
        else ("🔵 Optional" if c in OPTIONAL_COLUMNS else "⚪ Extra")
    )
    st.dataframe(col_info, use_container_width=True, hide_index=True)

    # Missing value report
    st.markdown("#### Missing Value Report")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if missing.empty:
        st.success("No missing values found.", icon="✅")
    else:
        miss_df = pd.DataFrame({
            "Column": missing.index,
            "Missing Count": missing.values,
            "Missing %": (missing / len(df) * 100).round(2).values,
        })
        st.dataframe(miss_df, use_container_width=True, hide_index=True)

    # Descriptive stats
    st.markdown("#### Descriptive Statistics (Numeric Columns)")
    numeric_df = df.select_dtypes(include="number")
    if not numeric_df.empty:
        st.dataframe(numeric_df.describe().round(2), use_container_width=True)

    # Dataset sufficiency check
    issues = check_dataset_sufficiency(df)
    if issues:
        st.markdown("#### ⚠️ Dataset Issues")
        for issue in issues:
            st.warning(issue)
    else:
        st.success("Dataset meets minimum requirements for model training.", icon="✅")

# ────────────────────────────────────────────────────────────────────────────
# Tab 2: Upload Dataset
# ────────────────────────────────────────────────────────────────────────────
with tab_upload:
    st.markdown("""
    Upload a CSV file to replace the current dataset for this session.

    **Required columns:** price, area, bedrooms, bathrooms, city  
    **Optional columns:** state, locality, property_type, furnishing, parking, floor, total_floors, age, carpet_area

    The application auto-detects common column name variants.
    Maximum file size: **50 MB**.
    """)

    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="CSV files only. Max 50 MB. File will not be saved to disk.",
    )

    if uploaded_file is not None:
        # File validation
        file_errors = validate_csv_file(uploaded_file.name, uploaded_file.size)
        if file_errors:
            for err in file_errors:
                st.error(err)
        else:
            st.info(
                f"File: **{uploaded_file.name}** ({uploaded_file.size/1024:.1f} KB)",
                icon="📄",
            )

            try:
                uploaded_df, load_warnings = load_uploaded_csv(uploaded_file)

                for w in load_warnings:
                    st.warning(w, icon="⚠️")

                st.success(
                    f"Loaded {len(uploaded_df):,} rows × {len(uploaded_df.columns)} columns.",
                    icon="✅",
                )
                st.markdown("#### Preview")
                st.dataframe(uploaded_df.head(20), use_container_width=True)

                # Column mapping
                st.markdown("#### Column Mapping")
                suggestions = get_column_mapping_suggestions(uploaded_df)
                st.markdown(
                    "Review and correct the suggested column mappings. "
                    "Set to blank if not applicable."
                )

                mapping_dict = {}
                map_cols = st.columns(3)
                all_df_cols = [""] + sorted(uploaded_df.columns.tolist())
                canonical_labels = sorted(suggestions.keys())

                for i, canonical in enumerate(canonical_labels):
                    suggested = suggestions[canonical] or ""
                    col_idx = i % 3
                    with map_cols[col_idx]:
                        chosen = st.selectbox(
                            f"{canonical}",
                            options=all_df_cols,
                            index=all_df_cols.index(suggested)
                            if suggested in all_df_cols else 0,
                            key=f"col_map_{canonical}",
                        )
                        if chosen:
                            mapping_dict[canonical] = chosen

                # Apply and use
                if st.button("✅ Apply Mapping & Use This Dataset",
                              type="primary", use_container_width=True):
                    try:
                        final_df = apply_column_mapping(uploaded_df, mapping_dict)

                        issues = check_dataset_sufficiency(final_df)
                        blocking = [i for i in issues if "required" in i.lower()
                                    or "empty" in i.lower()]
                        if blocking:
                            for err in blocking:
                                st.error(err)
                        else:
                            if issues:
                                for w in issues:
                                    st.warning(w)
                            st.session_state.dataset = final_df
                            st.session_state.dataset_label = (
                                f"📂 Uploaded: {uploaded_file.name}"
                            )
                            st.session_state.dataset_loaded = True
                            # Clear model cache so it re-evaluates
                            st.session_state.training_results = None
                            st.cache_data.clear()
                            st.success(
                                "Dataset updated for this session. "
                                "Go to **Model Evaluation** to retrain the model.",
                                icon="✅",
                            )
                    except Exception as e:
                        st.error(f"Failed to apply mapping: {e}")

            except ValueError as e:
                st.error(str(e), icon="❌")
            except Exception as e:
                st.error(f"Unexpected error reading file: {e}", icon="❌")

    st.markdown("---")
    st.markdown("#### Required Column Formats")
    format_df = pd.DataFrame([
        {"Column": "price", "Type": "Numeric", "Example": "5500000", "Notes": "INR, required"},
        {"Column": "area", "Type": "Numeric", "Example": "1200", "Notes": "sq ft, required"},
        {"Column": "bedrooms", "Type": "Integer", "Example": "3", "Notes": "1–10, required"},
        {"Column": "bathrooms", "Type": "Integer", "Example": "2", "Notes": "1–8, required"},
        {"Column": "city", "Type": "Text", "Example": "Pune", "Notes": "required"},
        {"Column": "state", "Type": "Text", "Example": "Maharashtra", "Notes": "optional"},
        {"Column": "property_type", "Type": "Text", "Example": "Apartment", "Notes": "optional"},
        {"Column": "furnishing", "Type": "Text", "Example": "Semi-Furnished", "Notes": "optional"},
        {"Column": "parking", "Type": "0/1 or Yes/No", "Example": "1", "Notes": "optional"},
        {"Column": "age", "Type": "Numeric", "Example": "5", "Notes": "years, optional"},
    ])
    st.dataframe(format_df, use_container_width=True, hide_index=True)
