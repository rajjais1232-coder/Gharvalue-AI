"""
pages/6_Prediction_History.py — Session prediction history with CSV export.
"""

import io
import datetime

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Prediction History — GharValue AI",
    page_icon="📋",
    layout="wide",
)

from app import inject_css, init_session_state, ensure_dataset
from src.prediction import format_inr

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
    <h1>📋 Prediction History</h1>
    <p>All price estimates generated during this session</p>
</div>
""", unsafe_allow_html=True)

st.info(
    "Session history is stored **in memory only**. "
    "It is lost when you close the browser tab or the server restarts. "
    "Export as CSV to save your predictions.",
    icon="ℹ️",
)

history = st.session_state.prediction_history

if not history:
    st.markdown("""
    <div style="text-align:center;padding:3rem;color:#57606a;">
        <div style="font-size:3rem;">📋</div>
        <div style="font-size:1.1rem;margin-top:1rem;">No predictions yet</div>
        <div style="font-size:0.9rem;margin-top:0.5rem;">
            Go to <strong>Predict Price</strong> to generate your first estimate.
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Summary metrics ───────────────────────────────────────────────────────────
hist_df = pd.DataFrame(history)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Predictions", f"{len(hist_df):,}")

if "predicted_price" in hist_df.columns:
    c2.metric("Avg Estimate", format_inr(hist_df["predicted_price"].mean()))
    c3.metric("Min Estimate", format_inr(hist_df["predicted_price"].min()))
    c4.metric("Max Estimate", format_inr(hist_df["predicted_price"].max()))

st.markdown("---")

# ── Filters ───────────────────────────────────────────────────────────────────
filter_col1, filter_col2 = st.columns(2)
with filter_col1:
    if "city" in hist_df.columns:
        cities_f = ["All"] + sorted(hist_df["city"].dropna().unique().tolist())
        sel_city = st.selectbox("Filter by City", cities_f)
    else:
        sel_city = "All"
with filter_col2:
    if "model" in hist_df.columns:
        models_f = ["All"] + sorted(hist_df["model"].dropna().unique().tolist())
        sel_model = st.selectbox("Filter by Model", models_f)
    else:
        sel_model = "All"

filtered_hist = hist_df.copy()
if sel_city != "All" and "city" in filtered_hist.columns:
    filtered_hist = filtered_hist[filtered_hist["city"] == sel_city]
if sel_model != "All" and "model" in filtered_hist.columns:
    filtered_hist = filtered_hist[filtered_hist["model"] == sel_model]

st.markdown(f"Showing **{len(filtered_hist)}** of {len(hist_df)} predictions")

# ── History table ─────────────────────────────────────────────────────────────
display_df = filtered_hist.copy()
if "predicted_price" in display_df.columns:
    display_df["Estimated Price"] = display_df["predicted_price"].apply(format_inr)
    display_df["Price (₹)"] = display_df["predicted_price"].apply(
        lambda v: f"₹{int(v):,}"
    )

display_cols = [c for c in [
    "datetime", "city", "state", "property_type", "bedrooms", "bathrooms",
    "area_sqft", "furnishing", "Estimated Price", "Price (₹)", "model"
] if c in display_df.columns]

st.dataframe(
    display_df[display_cols],
    use_container_width=True,
    hide_index=True,
    height=400,
)

# ── Export ────────────────────────────────────────────────────────────────────
st.markdown("---")
export_col1, export_col2 = st.columns(2)

with export_col1:
    # Export filtered
    csv_buffer = io.StringIO()
    filtered_hist.to_csv(csv_buffer, index=False)
    export_filename = (
        f"gharvalue_predictions_{datetime.datetime.now().strftime('%Y%m%d_%H%M')}.csv"
    )
    st.download_button(
        label="⬇️ Export Filtered as CSV",
        data=csv_buffer.getvalue(),
        file_name=export_filename,
        mime="text/csv",
        use_container_width=True,
    )

with export_col2:
    # Clear history
    if st.button("🗑️ Clear All Session History", use_container_width=True):
        st.session_state.prediction_history = []
        st.rerun()

st.caption(
    "Exported CSV contains all visible columns including raw price values in INR. "
    "This data is not retained after the session ends."
)
