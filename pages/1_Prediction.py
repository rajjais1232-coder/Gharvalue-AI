"""
pages/1_Prediction.py — House Price Prediction form.
"""

import datetime

import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Predict Price — GharValue AI",
    page_icon="🔮",
    layout="wide",
)

from app import inject_css, init_session_state, ensure_dataset
from src.data_loader import get_indian_states, get_cities_by_state
from src.prediction import predict_price, format_inr, format_inr_full, get_price_range
from src.validation import validate_prediction_input


inject_css()
init_session_state()
ensure_dataset()

df = st.session_state.dataset

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏠 GharValue AI")
    st.markdown("*House Price Prediction*")
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
    <h1>🔮 House Price Prediction</h1>
    <p>Enter property details to get an ML-based price estimate</p>
</div>
""", unsafe_allow_html=True)

# ── Check model ───────────────────────────────────────────────────────────────
from src.model_training import load_model
pipeline, metadata = load_model()

if not pipeline:
    st.warning(
        "**No trained model found.** "
        "Please go to [Model Evaluation](Model_Evaluation) and train a model first.",
        icon="⚠️",
    )

is_synthetic = "_synthetic" in df.columns
if is_synthetic:
    st.warning(
        "Running on **synthetic demo data**. Predictions do not reflect real market prices.",
        icon="⚠️",
    )

# ── Location helpers ──────────────────────────────────────────────────────────
states_list = get_indian_states()
cities_by_state = get_cities_by_state()

# Training coverage info
trained_cities: set = set()
if metadata:
    trained_cities = set(metadata.get("known_cities", []))
    if not trained_cities and df is not None and "city" in df.columns:
        trained_cities = set(df["city"].dropna().unique())

# ── Main Form ─────────────────────────────────────────────────────────────────
with st.form("prediction_form"):
    st.markdown("#### Location Details")
    loc_col1, loc_col2, loc_col3 = st.columns(3)

    with loc_col1:
        state = st.selectbox(
            "State / Union Territory *",
            options=[""] + states_list,
            index=0,
            help="Select the Indian state or UT where the property is located.",
        )
    with loc_col2:
        state_cities = cities_by_state.get(state, []) if state else []
        city_options = state_cities + ["Other / Enter manually"]
        city_select = st.selectbox(
            "City / District *",
            options=[""] + city_options,
            index=0,
        )
    with loc_col3:
        city_manual = st.text_input(
            "Enter city manually (if not in list)",
            placeholder="e.g. Durg, Korba",
            help="Use this if your city is not in the dropdown above.",
        )

    locality = st.text_input(
        "Locality / Neighbourhood (optional)",
        placeholder="e.g. Sector 12, Koramangala, Banjara Hills",
    )

    st.markdown("---")
    st.markdown("#### Property Details")

    prop_col1, prop_col2 = st.columns(2)
    with prop_col1:
        property_type = st.selectbox(
            "Property Type",
            ["Apartment", "Villa", "Independent House",
             "Builder Floor", "Penthouse", "Studio", "Other"],
            index=0,
        )
        bedrooms = st.select_slider(
            "Bedrooms (BHK) *",
            options=[1, 2, 3, 4, 5, 6, 7, 8],
            value=2,
        )
        bathrooms = st.select_slider(
            "Bathrooms *",
            options=[1, 2, 3, 4, 5, 6],
            value=2,
        )
    with prop_col2:
        area = st.number_input(
            "Built-up Area (sq ft) *",
            min_value=100, max_value=50000, value=1200, step=50,
            help="Total built-up / super area of the property.",
        )
        carpet_area = st.number_input(
            "Carpet Area (sq ft) — optional",
            min_value=0, max_value=50000, value=0, step=50,
            help="Usable carpet area. Leave 0 if unknown.",
        )
        age = st.number_input(
            "Property Age (years) — optional",
            min_value=0, max_value=100, value=0,
            help="Age of the property in years (0 = new).",
        )

    st.markdown("---")
    st.markdown("#### Additional Details (optional)")

    add_col1, add_col2, add_col3 = st.columns(3)
    with add_col1:
        furnishing = st.selectbox(
            "Furnishing Status",
            ["Unfurnished", "Semi-Furnished", "Furnished"],
            index=0,
        )
        parking = st.radio("Parking Available", ["Yes", "No"],
                           index=0, horizontal=True)
    with add_col2:
        floor = st.number_input("Floor Number", min_value=0, max_value=100, value=0)
        total_floors = st.number_input("Total Floors in Building",
                                       min_value=1, max_value=150, value=5)
    with add_col3:
        st.markdown("&nbsp;")
        st.markdown("**Required fields are marked \\***")
        st.markdown("All amounts in Indian Rupees (₹)")

    submitted = st.form_submit_button(
        "🔮 Predict Price", type="primary", use_container_width=True
    )

# ── Process submission ─────────────────────────────────────────────────────────
if submitted:
    # Resolve city
    city = city_manual.strip() if city_manual.strip() else (
        city_select if city_select not in ("", "Other / Enter manually") else ""
    )

    inputs = {
        "state": state or None,
        "city": city,
        "locality": locality or None,
        "property_type": property_type,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "area": area,
        "carpet_area": carpet_area if carpet_area > 0 else None,
        "age": age if age > 0 else None,
        "furnishing": furnishing,
        "parking": 1 if parking == "Yes" else 0,
        "floor": floor,
        "total_floors": total_floors,
    }

    # Validate
    errors = validate_prediction_input(inputs)
    if errors:
        for e in errors:
            st.error(e)
    elif not pipeline:
        st.error("Model not available. Please train a model first.")
    else:
        with st.spinner("Predicting price..."):
            result = predict_price(inputs)

        if result["success"]:
            price = result["price"]
            low, high = get_price_range(price)

            st.markdown("---")
            res_col1, res_col2 = st.columns([1.2, 1])

            with res_col1:
                st.markdown("### Prediction Result")
                st.markdown(
                    f"<div style='background:#d1fae5;border-radius:12px;padding:20px 24px;"
                    f"border-left:5px solid #059669;'>"
                    f"<div style='font-size:0.85rem;color:#065f46;font-weight:600;"
                    f"text-transform:uppercase;letter-spacing:0.05em;'>Estimated Price</div>"
                    f"<div style='font-size:2.2rem;font-weight:800;color:#065f46;'>"
                    f"{format_inr(price)}</div>"
                    f"<div style='font-size:1rem;color:#047857;margin-top:4px;'>"
                    f"{format_inr_full(price)}</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                st.caption(
                    f"Heuristic range (±10%): **{format_inr(low)}** — **{format_inr(high)}**"
                )
                st.caption(
                    f"Model: {result['model_name']} &nbsp;|&nbsp; "
                    f"Estimate only — not a certified valuation."
                )
                for w in result["warnings"]:
                    st.warning(w, icon="⚠️")

                if city and city in trained_cities:
                    st.success(f"'{city}' is represented in the training data.", icon="✅")
                elif city and trained_cities:
                    st.warning(
                        f"'{city}' has limited or no representation in training data. "
                        "Accuracy may be reduced.",
                        icon="⚠️",
                    )

            with res_col2:
                st.markdown("### Input Summary")
                summary = {
                    "Location": f"{city}, {state}" if state else city,
                    "Property Type": property_type,
                    "BHK": f"{bedrooms} Bed / {bathrooms} Bath",
                    "Area": f"{area:,} sq ft",
                    "Furnishing": furnishing,
                    "Age": f"{age} years" if age else "—",
                    "Floor": f"{floor}/{total_floors}",
                    "Parking": parking,
                }
                for k, v in summary.items():
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"padding:4px 0;border-bottom:1px solid #e2e8f0;'>"
                        f"<span style='color:#57606a;font-size:0.85rem;'>{k}</span>"
                        f"<span style='font-weight:600;font-size:0.85rem;'>{v}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                if area > 0:
                    st.markdown(
                        f"<div style='display:flex;justify-content:space-between;"
                        f"padding:4px 0;'>"
                        f"<span style='color:#57606a;font-size:0.85rem;'>Price/sq ft</span>"
                        f"<span style='font-weight:600;font-size:0.85rem;'>"
                        f"₹{price/area:,.0f}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Save to session history
            st.session_state.prediction_history.append({
                "datetime": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "city": city,
                "state": state or "—",
                "locality": locality or "—",
                "property_type": property_type,
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "area_sqft": area,
                "furnishing": furnishing,
                "age_years": age,
                "floor": floor,
                "total_floors": total_floors,
                "parking": parking,
                "predicted_price": price,
                "model": result["model_name"],
            })
            st.success("Prediction saved to session history.", icon="✅")

        else:
            for err in result["errors"]:
                st.error(err, icon="❌")

# ── Info footer ───────────────────────────────────────────────────────────────
st.markdown("---")
with st.expander("ℹ️ How predictions are made"):
    st.markdown("""
    1. Your inputs are validated against expected ranges and data types.
    2. The trained Scikit-learn pipeline (preprocessor + regressor) is loaded.
    3. Inputs are preprocessed using the **same** transformations applied during training
       (imputation → encoding → scaling).
    4. The model outputs a price estimate.
    5. A ±10% heuristic range is shown as an informal interval — it is **not** a
       statistically calibrated prediction interval.

    **Limitations:** The model's accuracy depends on how well your inputs match the
    training data's distribution. Rare cities, localities, or property configurations
    may produce less reliable estimates.
    """)
