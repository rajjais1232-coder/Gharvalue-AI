"""
pages/2_Market_Insights.py — Interactive market analytics charts.
"""

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Market Insights — GharValue AI",
    page_icon="📊",
    layout="wide",
)

from app import inject_css, init_session_state, ensure_dataset
from src import visualizations as viz

inject_css()
init_session_state()
ensure_dataset()

df: pd.DataFrame = st.session_state.dataset
is_synthetic = "_synthetic" in df.columns

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

    st.markdown("---")
    st.markdown("### Filters")
    states_available = sorted(df["state"].dropna().unique().tolist()) if "state" in df.columns else []
    selected_states = st.multiselect("States", options=states_available, default=[])

    prop_types = sorted(df["property_type"].dropna().unique().tolist()) if "property_type" in df.columns else []
    selected_types = st.multiselect("Property Types", options=prop_types, default=[])

    if "bedrooms" in df.columns:
        min_bhk = int(df["bedrooms"].min())
        max_bhk = int(df["bedrooms"].max())
        bhk_range = st.slider("BHK Range", min_value=min_bhk, max_value=max_bhk,
                              value=(min_bhk, max_bhk))
    else:
        bhk_range = None

# ── Apply filters ─────────────────────────────────────────────────────────────
filtered = df.copy()
if selected_states:
    filtered = filtered[filtered["state"].isin(selected_states)]
if selected_types and "property_type" in filtered.columns:
    filtered = filtered[filtered["property_type"].isin(selected_types)]
if bhk_range and "bedrooms" in filtered.columns:
    filtered = filtered[
        filtered["bedrooms"].between(bhk_range[0], bhk_range[1])
    ]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>📊 Market Insights</h1>
    <p>Interactive property market analytics based on the loaded dataset</p>
</div>
""", unsafe_allow_html=True)

if is_synthetic:
    st.info(
        "Charts are based on **synthetic demo data** — not real market findings.",
        icon="ℹ️",
    )

st.markdown(f"**Showing {len(filtered):,} records** after filters | "
            f"Dataset: {st.session_state.dataset_label}")

if len(filtered) < 10:
    st.warning("Too few records after filtering. Adjust the filters in the sidebar.")
    st.stop()

# ── Row 1: City prices + Price/sqft ──────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(viz.price_by_city(filtered), use_container_width=True)
with col2:
    st.plotly_chart(viz.price_per_sqft(filtered), use_container_width=True)

st.markdown("---")

# ── Row 2: BHK distribution + Price distribution ─────────────────────────────
col3, col4 = st.columns(2)
with col3:
    st.plotly_chart(viz.bhk_distribution(filtered), use_container_width=True)
with col4:
    st.plotly_chart(viz.price_distribution(filtered), use_container_width=True)

st.markdown("---")

# ── Row 3: Area vs Price scatter ─────────────────────────────────────────────
color_by = None
if "property_type" in filtered.columns:
    color_by = "property_type"
elif "state" in filtered.columns:
    color_by = "state"

st.plotly_chart(viz.area_vs_price(filtered, color_col=color_by), use_container_width=True)

st.markdown("---")

# ── Row 4: Property type + Furnishing ────────────────────────────────────────
col5, col6 = st.columns(2)
with col5:
    if "property_type" in filtered.columns:
        st.plotly_chart(viz.property_type_comparison(filtered), use_container_width=True)
    else:
        st.info("Property type column not available in dataset.")
with col6:
    if "furnishing" in filtered.columns:
        st.plotly_chart(viz.furnishing_price_comparison(filtered), use_container_width=True)
    else:
        st.info("Furnishing column not available in dataset.")

# ── Summary stats ─────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Summary Statistics")
stat_cols = [c for c in ["price", "area", "bedrooms", "bathrooms"] if c in filtered.columns]
if stat_cols:
    st.dataframe(
        filtered[stat_cols].describe().round(2),
        use_container_width=True,
    )

st.caption(
    "All charts use actual loaded dataset records. Sample sizes are shown in tooltips. "
    "Charts require a minimum of 5 records per category."
)
