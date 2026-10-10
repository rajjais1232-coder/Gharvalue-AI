"""
pages/7_About_Project.py — Academic project information.
"""

import streamlit as st

st.set_page_config(
    page_title="About — GharValue AI",
    page_icon="ℹ️",
    layout="wide",
)

from app import inject_css, init_session_state

inject_css()
init_session_state()

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
    <h1>ℹ️ About GharValue AI</h1>
    <p>B.Tech IT Project — House Price Prediction Using Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# ── Project identity ──────────────────────────────────────────────────────────
id_col1, id_col2 = st.columns(2)

with id_col1:
    st.markdown("### Project Details")
    details = {
        "Project Title": "House Price Prediction Using Machine Learning",
        "Application Name": "GharValue AI",
        "Domain": "Machine Learning / Real Estate Analytics",
        "Platform": "Python · Streamlit · Scikit-learn",
        "Academic Year": "B.Tech 3rd Year — VT Project",
    }
    for k, v in details.items():
        st.markdown(
            f"<div style='padding:6px 0;border-bottom:1px solid #e2e8f0;'>"
            f"<span style='color:#57606a;font-size:0.85rem;'>{k}</span>"
            f"<br><span style='font-weight:600;'>{v}</span></div>",
            unsafe_allow_html=True,
        )

with id_col2:
    st.markdown("### Student & Guide")
    people = {
        "Student Name": "Raj Jaiswal",
        "Roll Number": "300703324028",
        "Department": "Information Technology",
        "Institution": "CHHATTISGARH INSTITUTE OF TECHNOLOGY (CGIT), Bilaspur",
        "Project Guide": "Samiksha Shukla",
        "Guide Designation": "Assistant Professor, IT Department",
    }
    for k, v in people.items():
        st.markdown(
            f"<div style='padding:6px 0;border-bottom:1px solid #e2e8f0;'>"
            f"<span style='color:#57606a;font-size:0.85rem;'>{k}</span>"
            f"<br><span style='font-weight:600;'>{v}</span></div>",
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Content tabs ──────────────────────────────────────────────────────────────
t1, t2, t3, t4, t5 = st.tabs([
    "Problem & Objectives",
    "System Architecture",
    "ML Algorithms",
    "Limitations & Future Scope",
    "References",
])

with t1:
    st.markdown("""
    ### Problem Statement
    Property pricing in India is complex and highly variable — influenced by location,
    infrastructure, property characteristics, market trends, and local demand. Buyers,
    sellers, and investors often lack access to objective, data-driven price estimates
    and rely instead on unverified market opinions.

    ### Objectives
    1. Build a machine-learning pipeline trained on Indian property data.
    2. Provide an interactive web interface for price estimation.
    3. Compare multiple regression algorithms and select the best performer.
    4. Deliver interpretable results with appropriate uncertainty disclosure.
    5. Create a maintainable, modular codebase suitable for extension.

    ### Proposed Solution
    GharValue AI addresses this by:
    - Loading and normalising Indian housing datasets with automatic column detection.
    - Training and comparing four regression algorithms (Linear Regression, Random Forest,
      Extra Trees, Gradient Boosting).
    - Deploying a Streamlit web application that accepts property inputs and returns
      price estimates along with market analytics.
    - Displaying model evaluation metrics, feature importance, and data coverage
      transparently.

    ### Existing System Limitations
    - Traditional valuation methods are manual, slow, and expert-dependent.
    - Most online portals show listing prices, not predicted fair-market values.
    - Existing tools rarely disclose model confidence or data coverage.
    """)

with t2:
    st.markdown("""
    ### System Architecture

    ```
    User Input (Streamlit UI)
         │
         ▼
    Input Validation (src/validation.py)
         │
         ▼
    Preprocessing Pipeline (sklearn ColumnTransformer)
    ├── Numeric: Median Imputation → StandardScaler
    └── Categorical: Mode Imputation → OrdinalEncoder
         │
         ▼
    Trained Regressor (joblib-persisted Pipeline)
         │
         ▼
    Price Prediction (₹)
         │
         ▼
    Result Display + Session History
    ```

    ### Data Flow
    1. **Data Loader** reads CSV, normalises column names, cleans invalid values.
    2. **Preprocessor** is fitted on the training split (80%) only — preventing leakage.
    3. **Model** is trained on transformed training features.
    4. The **complete Pipeline** (preprocessor + model) is saved with joblib.
    5. At inference, the same Pipeline applies identical preprocessing to user inputs.
    6. **Metadata** records feature schema, training date, and metrics.

    ### Module Structure
    | Module | Responsibility |
    |---|---|
    | `src/data_loader.py` | CSV loading, column normalisation, synthetic data |
    | `src/preprocessing.py` | Feature engineering, ColumnTransformer builder |
    | `src/model_training.py` | Train, evaluate, save, load models |
    | `src/prediction.py` | Inference, INR formatting |
    | `src/evaluation.py` | Metrics, comparison tables |
    | `src/validation.py` | Input & file validation |
    | `src/visualizations.py` | Plotly chart builders |
    """)

with t3:
    st.markdown("""
    ### Regression Algorithms Used

    #### 1. Linear Regression
    - Fits a linear relationship: price = β₀ + β₁·area + β₂·bedrooms + …
    - Interpretable coefficients; fast training.
    - Assumes linear relationships — may underfit complex patterns.

    #### 2. Random Forest Regressor
    - Ensemble of decision trees trained on random feature subsets.
    - Reduces overfitting through averaging (bagging).
    - Provides feature importance via impurity-based measures.

    #### 3. Extra Trees Regressor
    - Similar to Random Forest but uses random thresholds for splits.
    - Generally faster to train; sometimes lower variance than Random Forest.

    #### 4. Gradient Boosting Regressor
    - Builds trees sequentially, each correcting the previous one's errors.
    - Often achieves best accuracy on tabular data.
    - Sensitive to hyperparameters; slower to train.

    ### Evaluation Metrics
    | Metric | Meaning |
    |---|---|
    | MAE | Average absolute prediction error in ₹ |
    | RMSE | Penalises large errors more; in ₹ |
    | R² | Fraction of variance explained (1.0 = perfect) |
    | MAPE | Mean absolute % error (scale-independent) |

    **Best model selection:** Lowest RMSE on the held-out 20% test split.

    ### Train/Test Protocol
    - 80/20 stratified random split (seed = 42 for reproducibility).
    - Preprocessing (imputers, encoder, scaler) fitted **only** on training data.
    - Test set used exclusively for evaluation — never seen during training.
    """)

with t4:
    st.markdown("""
    ### Known Limitations

    1. **Data dependency:** Accuracy depends entirely on dataset quality and coverage.
       Under-represented cities, states, or property types will yield less reliable estimates.
    2. **Static model:** The model does not update in real time. Market conditions change.
    3. **No causal inference:** Feature importance reflects correlation, not causation.
    4. **Uncertainty:** The ±10% range is a heuristic, not a calibrated prediction interval.
    5. **Synthetic data mode:** When no real dataset is present, predictions have no
       market validity.
    6. **Locality-level detail:** Granular locality pricing requires far larger, richer datasets.

    ### Future Enhancements

    1. Integrate live data from publicly available Indian housing APIs or datasets.
    2. Implement XGBoost / LightGBM for potentially higher accuracy.
    3. Add calibrated prediction intervals using conformal prediction or quantile regression.
    4. Include infrastructure features (metro distance, school ratings, hospital proximity).
    5. Build a time-series price trend module.
    6. Automated retraining pipeline when new data is added.
    7. Locality-level GeoJSON choropleth maps.
    8. Multi-language support (Hindi, Marathi, Tamil, etc.).
    """)

with t5:
    st.markdown("""
    ### References & Dataset Attribution

    1. Scikit-learn: Machine Learning in Python — Pedregosa et al. (2011), JMLR 12, 2825–2830.
    2. Streamlit — [streamlit.io](https://streamlit.io) (Apache 2.0 License)
    3. Plotly — Interactive charts library — [plotly.com](https://plotly.com)
    4. Breiman, L. (2001). Random Forests. *Machine Learning*, 45, 5–32.
    5. Friedman, J.H. (2001). Greedy Function Approximation: A Gradient Boosting Machine.
       *The Annals of Statistics*, 29(5), 1189–1232.
    6. Pandas — [pandas.pydata.org](https://pandas.pydata.org)
    7. NumPy — [numpy.org](https://numpy.org)
    8. Indian States & Cities data — Government of India open data (census.gov.in)

    **Dataset:** If using a real dataset, credit the source, license, and collection date
    in the `data/README.md` file.

    ### Academic Disclaimer

    > This project is developed for educational purposes as part of the B.Tech IT curriculum
    > at CGIT Bilaspur. Predicted prices are machine-learning estimates based on training data
    > and do NOT constitute a certified property valuation, legal appraisal, or investment advice.
    > The authors make no representations regarding the accuracy, completeness, or fitness for
    > any commercial purpose of the results produced by this application.
    """)

st.markdown("---")
st.caption(
    "GharValue AI &nbsp;|&nbsp; B.Tech IT Project &nbsp;|&nbsp; "
    "Raj Jaiswal (Roll: 300703324028) &nbsp;|&nbsp; CGIT Bilaspur &nbsp;|&nbsp; "
    "Guide: Samiksha Shukla, Asst. Prof., IT Dept."
)
