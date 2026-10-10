"""
app.py — GharValue AI: Home Dashboard
Main entry point for the Streamlit application.
"""

import streamlit as st

# Page config must be the first Streamlit call
st.set_page_config(
    page_title="GharValue AI — House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

import pandas as pd
from pathlib import Path

from src.data_loader import load_dataset
from src.model_training import load_model, load_comparison

# ---------------------------------------------------------------------------
# Shared CSS
# ---------------------------------------------------------------------------
SHARED_CSS = """
<style>
    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #16324f 100%);
    }
    section[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    section[data-testid="stSidebar"] .stSelectbox label,
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 { color: #ffffff !important; }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: #f0f4f8;
        border: 1px solid #dde3ed;
        border-radius: 10px;
        padding: 16px 20px;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.78rem;
        color: #57606a;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
        color: #1e3a5f;
    }

    /* Page header */
    .page-header {
        padding: 1.2rem 1.6rem;
        background: linear-gradient(90deg, #1e3a5f, #2563a8);
        border-radius: 12px;
        margin-bottom: 1.5rem;
        color: white;
    }
    .page-header h1 { color: white !important; margin: 0; font-size: 1.7rem; }
    .page-header p  { color: #cbd5e1 !important; margin: 0.3rem 0 0; font-size: 0.95rem; }

    /* Stat cards on home page */
    .stat-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-left: 4px solid #1e3a5f;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }
    .stat-card-value { font-size: 1.4rem; font-weight: 700; color: #1e3a5f; }
    .stat-card-label { font-size: 0.78rem; color: #57606a; margin-top: 2px; }

    /* Status badges */
    .badge-green {
        display: inline-block;
        background: #d1fae5; color: #065f46;
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.78rem; font-weight: 600;
    }
    .badge-yellow {
        display: inline-block;
        background: #fef3c7; color: #92400e;
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.78rem; font-weight: 600;
    }
    .badge-red {
        display: inline-block;
        background: #fee2e2; color: #991b1b;
        border-radius: 20px; padding: 2px 10px;
        font-size: 0.78rem; font-weight: 600;
    }

    /* Remove default padding */
    .block-container { padding-top: 1.5rem !important; }

    /* Divider */
    hr { border-color: #e2e8f0; }
</style>
"""


def inject_css() -> None:
    st.markdown(SHARED_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
def init_session_state() -> None:
    defaults = {
        "prediction_history": [],
        "dataset": None,
        "dataset_label": None,
        "dataset_loaded": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Dataset loading (cached)
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def get_dataset():
    return load_dataset()


def ensure_dataset() -> None:
    if not st.session_state.dataset_loaded:
        with st.spinner("Loading dataset..."):
            df, label = get_dataset()
        st.session_state.dataset = df
        st.session_state.dataset_label = label
        st.session_state.dataset_loaded = True


# ---------------------------------------------------------------------------
# Main page
# ---------------------------------------------------------------------------
def main() -> None:
    inject_css()
    init_session_state()
    ensure_dataset()

    df: pd.DataFrame = st.session_state.dataset
    dataset_label: str = st.session_state.dataset_label
    is_synthetic = "_synthetic" in df.columns

    # ── Sidebar ─────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("## 🏠 GharValue AI")
        st.markdown("*House Price Prediction*")
        st.markdown("---")
        st.markdown("### Navigate")
        st.page_link("app.py", label="🏠 Home Dashboard", icon=None)
        st.page_link("pages/1_Prediction.py", label="🔮 Predict Price")
        st.page_link("pages/2_Market_Insights.py", label="📊 Market Insights")
        st.page_link("pages/3_Model_Evaluation.py", label="🧪 Model Evaluation")
        st.page_link("pages/4_3D_Analytics.py", label="🌐 3D Analytics")
        st.page_link("pages/5_Dataset_Explorer.py", label="📁 Dataset Explorer")
        st.page_link("pages/6_Prediction_History.py", label="📋 Prediction History")
        st.page_link("pages/7_About_Project.py", label="ℹ️ About Project")
        st.markdown("---")
        st.caption("B.Tech IT Project — CGIT Bilaspur")

    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="page-header">
        <h1>🏠 GharValue AI</h1>
        <p>House Price Prediction Using Machine Learning &nbsp;|&nbsp;
           Powered by Scikit-learn &amp; Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

    if is_synthetic:
        st.warning(
            "**Demo Mode:** Running on synthetic data. "
            "Upload a real dataset in **Dataset Explorer** or place `housing_data.csv` "
            "in the `data/` folder, then refresh.",
            icon="⚠️",
        )

    # ── Status Row ───────────────────────────────────────────────────────────
    pipeline, metadata = load_model()
    comparison = load_comparison()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        n_cities = df["city"].nunique() if "city" in df.columns else 0
        st.metric("Cities Covered", f"{n_cities:,}")
    with col3:
        n_states = df["state"].nunique() if "state" in df.columns else 0
        st.metric("States Covered", f"{n_states:,}")
    with col4:
        if pipeline and metadata:
            best = metadata.get("model_name", "—")
            r2 = metadata.get("metrics", {}).get("R2", None)
            label_val = best
            delta_val = f"R² {r2:.3f}" if r2 is not None else None
            st.metric("Best Model", label_val, delta=delta_val)
        else:
            st.metric("Best Model", "Not Trained")

    st.markdown("---")

    # ── Two-column layout: Status + Quick Predict ────────────────────────────
    left, right = st.columns([1, 1], gap="large")

    with left:
        st.markdown("### System Status")

        # Dataset status
        ds_badge = (
            '<span class="badge-yellow">Synthetic Demo</span>'
            if is_synthetic
            else '<span class="badge-green">Real Dataset</span>'
        )
        st.markdown(f"**Dataset:** {dataset_label} &nbsp; {ds_badge}",
                    unsafe_allow_html=True)

        if "price" in df.columns:
            prices = df["price"].dropna()
            st.markdown(f"**Price Range:** ₹{prices.min()/1e5:.1f}L — ₹{prices.max()/1e7:.2f}Cr")
        if "area" in df.columns:
            areas = df["area"].dropna()
            st.markdown(f"**Area Range:** {areas.min():.0f} — {areas.max():.0f} sq ft")

        st.markdown("---")

        model_badge = (
            '<span class="badge-green">Ready</span>'
            if pipeline
            else '<span class="badge-red">Not Trained</span>'
        )
        st.markdown(f"**Model:** {model_badge}", unsafe_allow_html=True)

        if metadata:
            st.markdown(f"**Algorithm:** {metadata.get('model_name', '—')}")
            st.markdown(f"**Trained:** {metadata.get('trained_at', '—')[:10]}")
            m = metadata.get("metrics", {})
            if m:
                st.markdown(
                    f"**Test MAE:** ₹{m.get('MAE', 0)/1e5:.2f}L &nbsp; "
                    f"**RMSE:** ₹{m.get('RMSE', 0)/1e5:.2f}L &nbsp; "
                    f"**R²:** {m.get('R2', 0):.3f}",
                    unsafe_allow_html=True,
                )
        else:
            st.info(
                "No trained model found. Visit **Model Evaluation** to train one.",
                icon="ℹ️",
            )

        if comparison:
            st.markdown("---")
            st.markdown("**Model Comparison (RMSE ₹)**")
            comp_df = pd.DataFrame([
                {"Model": k, "RMSE (₹)": f"₹{v['RMSE']/1e5:.2f}L",
                 "R²": f"{v['R2']:.3f}"}
                for k, v in comparison.items()
            ])
            st.dataframe(comp_df, hide_index=True, use_container_width=True)

    with right:
        st.markdown("### Quick Price Estimate")

        if not pipeline:
            st.info(
                "Train a model first on the **Model Evaluation** page, "
                "then come back for a quick estimate.",
                icon="🔒",
            )
        else:
            with st.form("quick_predict_form"):
                qcol1, qcol2 = st.columns(2)
                with qcol1:
                    area = st.number_input("Area (sq ft)", min_value=100, max_value=50000,
                                           value=1200, step=50)
                    bedrooms = st.selectbox("Bedrooms", [1, 2, 3, 4, 5], index=1)
                with qcol2:
                    bathrooms = st.selectbox("Bathrooms", [1, 2, 3, 4], index=1)
                    city = st.text_input("City", value="Mumbai",
                                        placeholder="e.g. Pune, Bangalore")
                state = st.text_input("State", value="Maharashtra",
                                      placeholder="e.g. Maharashtra")
                submitted = st.form_submit_button("Get Estimate", type="primary",
                                                   use_container_width=True)

            if submitted:
                from src.prediction import predict_price, format_inr, get_price_range
                result = predict_price({
                    "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                    "city": city, "state": state,
                })
                if result["success"]:
                    price = result["price"]
                    low, high = get_price_range(price)
                    from src.prediction import format_inr_full
                    st.success(f"**Estimated Price: {format_inr(price)}**")
                    st.caption(
                        f"Range: {format_inr(low)} – {format_inr(high)} (±10% heuristic)"
                    )
                    st.caption(
                        "This is an ML estimate, not a certified property valuation."
                    )
                    for w in result["warnings"]:
                        st.warning(w)
                    # Save to history
                    import datetime
                    st.session_state.prediction_history.append({
                        "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M"),
                        "city": city,
                        "state": state,
                        "bedrooms": bedrooms,
                        "bathrooms": bathrooms,
                        "area_sqft": area,
                        "predicted_price": price,
                        "model": result["model_name"],
                    })
                else:
                    for err in result["errors"]:
                        st.error(err)

    st.markdown("---")

    # ── Recent Session History ───────────────────────────────────────────────
    if st.session_state.prediction_history:
        st.markdown("### Recent Predictions (This Session)")
        hist_df = pd.DataFrame(st.session_state.prediction_history[-5:])
        hist_df["predicted_price"] = hist_df["predicted_price"].apply(
            lambda v: f"₹{v/1e5:.2f}L" if v >= 1e5 else f"₹{v:,.0f}"
        )
        st.dataframe(
            hist_df[["datetime", "city", "bedrooms", "area_sqft", "predicted_price", "model"]],
            hide_index=True,
            use_container_width=True,
        )
        st.caption("Session history is not permanently stored. Export from Prediction History page.")

    # ── Footer ───────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "GharValue AI &nbsp;|&nbsp; B.Tech IT Project &nbsp;|&nbsp; "
        "Raj Jaiswal (300703324028) &nbsp;|&nbsp; CGIT Bilaspur &nbsp;|&nbsp; "
        "Guide: Samiksha Shukla, Asst. Prof."
    )


if __name__ == "__main__":
    main()
