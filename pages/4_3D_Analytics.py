"""
pages/4_3D_Analytics.py — Interactive 3D property scatter plot.
"""

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="3D Analytics — GharValue AI",
    page_icon="🌐",
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
    st.markdown("### 3D Filters")

    if "state" in df.columns:
        states = sorted(df["state"].dropna().unique().tolist())
        sel_states = st.multiselect("Filter by State", states, default=[])
    else:
        sel_states = []

    if "bedrooms" in df.columns:
        min_b = int(df["bedrooms"].dropna().min())
        max_b = int(df["bedrooms"].dropna().max())
        bhk_sel = st.slider("BHK Range", min_b, max_b, (min_b, max_b))
    else:
        bhk_sel = None

    max_records = st.slider("Max records (performance)", 500, 5000, 2000, 250)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🌐 Interactive 3D Analytics</h1>
    <p>Rotate · Zoom · Hover to inspect individual data points</p>
</div>
""", unsafe_allow_html=True)

if is_synthetic:
    st.info("Visualising **synthetic demo data**.", icon="ℹ️")

# ── Filter ────────────────────────────────────────────────────────────────────
filtered = df.copy()
if sel_states and "state" in filtered.columns:
    filtered = filtered[filtered["state"].isin(sel_states)]
if bhk_sel and "bedrooms" in filtered.columns:
    filtered = filtered[filtered["bedrooms"].between(bhk_sel[0], bhk_sel[1])]
if len(filtered) > max_records:
    filtered = filtered.sample(max_records, random_state=42)

st.markdown(
    f"**{len(filtered):,} records** plotted (use sidebar to filter). "
    f"Colour represents **property type** or **state** when available."
)

# ── 3D Chart ──────────────────────────────────────────────────────────────────
fig = viz.scatter_3d(filtered)
st.plotly_chart(fig, use_container_width=True, height=700)

# ── Controls hint ─────────────────────────────────────────────────────────────
st.markdown("""
**Interaction tips:**
- 🖱️ **Click and drag** to rotate the 3D view
- 🔍 **Scroll** to zoom in/out
- 👆 **Hover** over a point to see its details
- 📦 Use the legend to show/hide categories
- 📸 Use the Plotly toolbar (top-right) to save as PNG
""")

# ── Statistics ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Dataset Slice Statistics")

s_col1, s_col2, s_col3, s_col4 = st.columns(4)
with s_col1:
    st.metric("Records", f"{len(filtered):,}")
with s_col2:
    if "price" in filtered.columns:
        med_price = filtered["price"].median()
        st.metric("Median Price", f"₹{med_price/1e5:.1f}L")
with s_col3:
    if "area" in filtered.columns:
        med_area = filtered["area"].median()
        st.metric("Median Area", f"{med_area:,.0f} sq ft")
with s_col4:
    if "bedrooms" in filtered.columns:
        mode_bhk = filtered["bedrooms"].mode()
        st.metric("Most Common BHK", f"{int(mode_bhk.iloc[0])} BHK")
