"""
pages/3_Model_Evaluation.py — Train, compare, and evaluate ML models.
"""

import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Model Evaluation — GharValue AI",
    page_icon="🧪",
    layout="wide",
)

from app import inject_css, init_session_state, ensure_dataset
from src.model_training import (
    train_all_models, select_best_model, save_model, save_comparison,
    load_model, get_feature_importance
)
from src.evaluation import (
    build_comparison_table, build_actual_vs_predicted_df, explain_metrics
)
from src import visualizations as viz

inject_css()
init_session_state()
ensure_dataset()

df = st.session_state.dataset
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

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="page-header">
    <h1>🧪 Model Evaluation</h1>
    <p>Train, compare, and evaluate regression models on the loaded dataset</p>
</div>
""", unsafe_allow_html=True)

if is_synthetic:
    st.info(
        "**Synthetic Demo Data:** Results here demonstrate the evaluation workflow "
        "but do not represent real market predictive accuracy.",
        icon="ℹ️",
    )

# ── Cached training ───────────────────────────────────────────────────────────
if "training_results" not in st.session_state:
    st.session_state.training_results = None

# ── Train button ──────────────────────────────────────────────────────────────
train_col, info_col = st.columns([1, 2])
with train_col:
    if st.button("▶ Train & Evaluate All Models", type="primary", use_container_width=True):
        progress_bar = st.progress(0, text="Starting training...")
        status_text = st.empty()
        models_registry_size = 4

        def progress_cb(name, step, total):
            pct = int((step / total) * 100)
            progress_bar.progress(pct, text=f"Training {name}...")
            status_text.text(f"Training: {name} ({step+1}/{total})")

        try:
            results = train_all_models(df, progress_callback=progress_cb)
            st.session_state.training_results = results
            progress_bar.progress(100, text="Training complete!")
            status_text.empty()
            st.success(f"Trained {len(results)} models successfully.", icon="✅")
            save_comparison(results)
        except Exception as e:
            progress_bar.empty()
            status_text.empty()
            st.error(f"Training failed: {e}", icon="❌")

with info_col:
    st.markdown(
        "**Models:** Linear Regression · Random Forest · Extra Trees · Gradient Boosting  \n"
        "**Split:** 80% train / 20% test (stratified random)  \n"
        "**Preprocessing:** Fitted on training data only (no leakage)  \n"
        "**Best model** is selected by lowest RMSE on the held-out test set."
    )

results = st.session_state.training_results

# ── Show existing saved model info ────────────────────────────────────────────
saved_pipeline, saved_metadata = load_model()
if saved_pipeline and saved_metadata and results is None:
    st.info(
        f"A previously trained model is available: **{saved_metadata.get('model_name')}** "
        f"(trained {saved_metadata.get('trained_at', '')[:10]}). "
        "Retrain to update metrics.",
        icon="ℹ️",
    )

if results is None:
    st.stop()

# ── Comparison table ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Model Comparison")

comp_df = build_comparison_table(results)
best_name = select_best_model(results)

st.dataframe(
    comp_df.style.highlight_min(subset=["MAE (₹)", "RMSE (₹)"], color="#d1fae5")
               .highlight_max(subset=["R² Score"], color="#d1fae5"),
    use_container_width=True,
)
st.caption(f"**Best model (lowest RMSE):** {best_name}")

# ── Comparison chart ──────────────────────────────────────────────────────────
st.plotly_chart(viz.model_comparison_chart(comp_df), use_container_width=True)

# ── Save best model ───────────────────────────────────────────────────────────
st.markdown("---")
save_col1, save_col2 = st.columns([1, 2])
with save_col1:
    model_to_save = st.selectbox(
        "Select model to save as best",
        options=list(results.keys()),
        index=list(results.keys()).index(best_name),
    )
    if st.button("💾 Save Selected Model", type="primary", use_container_width=True):
        res = results[model_to_save]
        feat_names = res["num_features"] + res["cat_features"]

        # Store known cities in metadata
        import json
        from pathlib import Path
        known_cities = []
        if "city" in df.columns:
            known_cities = df["city"].dropna().unique().tolist()

        try:
            save_model(
                pipeline=res["pipeline"],
                model_name=model_to_save,
                metrics=res["metrics"],
                feature_names=feat_names,
                dataset_label=st.session_state.dataset_label,
            )
            # Patch metadata with known_cities
            meta_path = Path("models/model_metadata.json")
            if meta_path.exists():
                with open(meta_path) as f:
                    meta = json.load(f)
                meta["known_cities"] = known_cities
                with open(meta_path, "w") as f:
                    json.dump(meta, f, indent=2)

            st.success(f"Model '{model_to_save}' saved to models/best_model.joblib", icon="✅")
            st.cache_data.clear()
        except Exception as e:
            st.error(f"Save failed: {e}")

with save_col2:
    m = results[model_to_save]["metrics"]
    st.markdown(f"**{model_to_save}** metrics:")
    mc1, mc2, mc3 = st.columns(3)
    mc1.metric("MAE", f"₹{m['MAE']/1e5:.2f}L")
    mc2.metric("RMSE", f"₹{m['RMSE']/1e5:.2f}L")
    mc3.metric("R²", f"{m['R2']:.4f}")

# ── Per-model detail ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### Detailed Model Analysis")
selected_detail = st.selectbox("Select model to inspect", options=list(results.keys()))
res = results[selected_detail]

detail_col1, detail_col2 = st.columns(2)
with detail_col1:
    avp_df = build_actual_vs_predicted_df(res["y_test"], res["y_pred"])
    st.plotly_chart(
        viz.actual_vs_predicted(res["y_test"], res["y_pred"], selected_detail),
        use_container_width=True,
    )
with detail_col2:
    st.plotly_chart(
        viz.residual_histogram(res["y_test"], res["y_pred"], selected_detail),
        use_container_width=True,
    )

# Feature importance
fi_df = get_feature_importance(
    res["pipeline"],
    res["num_features"] + res["cat_features"],
    selected_detail,
)
if fi_df is not None:
    st.plotly_chart(
        viz.feature_importance_chart(fi_df, selected_detail),
        use_container_width=True,
    )
    st.caption(
        "Feature importance shows which features influenced predictions most. "
        "This is NOT a causal relationship — it reflects the model's learned patterns."
    )

# ── Metric explanations ───────────────────────────────────────────────────────
st.markdown("---")
with st.expander("📖 Metric Definitions"):
    for metric, explanation in explain_metrics().items():
        st.markdown(f"**{metric}:** {explanation}")
