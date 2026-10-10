# GharValue AI — House Price Prediction Using Machine Learning

> **B.Tech IT Project** | CHHATTISGARH INSTITUTE OF TECHNOLOGY (CGIT), Bilaspur  
> **Student:** Raj Jaiswal (Roll: 300703324028) | **Guide:** Samiksha Shukla, Asst. Prof.

---

## Overview

GharValue AI is a complete machine-learning application that predicts Indian residential property prices from property characteristics. It provides an interactive Streamlit web interface, a reproducible ML pipeline, comparative model evaluation, and interactive analytics charts.

**Key Features:**
- Price prediction using Linear Regression, Random Forest, Extra Trees, and Gradient Boosting
- Automatic dataset loading with column-name normalisation
- Synthetic demo data mode (clearly labelled) when no real dataset is available
- Interactive market analytics and 3D property scatter plots
- Full model evaluation with MAE, RMSE, R², and feature importance
- Dataset upload and column mapping interface
- Session-based prediction history with CSV export
- 37+ automated tests

---

## Technology Stack

| Component | Library |
|---|---|
| Web Application | Streamlit ≥ 1.32 |
| Data Processing | Pandas ≥ 2.0, NumPy ≥ 1.24 |
| Machine Learning | Scikit-learn ≥ 1.4 |
| Visualisations | Plotly ≥ 5.18 |
| Model Persistence | Joblib ≥ 1.3 |
| Testing | Pytest ≥ 7.4 |

---

## Folder Structure

```
GharValue-AI/
├── app.py                        # Home Dashboard (Streamlit entry point)
├── pages/
│   ├── 1_Prediction.py           # Price prediction form
│   ├── 2_Market_Insights.py      # Interactive market charts
│   ├── 3_Model_Evaluation.py     # Train, compare, evaluate models
│   ├── 4_3D_Analytics.py         # Interactive 3D scatter plot
│   ├── 5_Dataset_Explorer.py     # Upload and inspect datasets
│   ├── 6_Prediction_History.py   # Session history + CSV export
│   └── 7_About_Project.py        # Academic information
├── src/
│   ├── __init__.py
│   ├── data_loader.py            # Dataset loading, normalisation, synthetic data
│   ├── preprocessing.py          # Sklearn preprocessing pipeline builders
│   ├── model_training.py         # Train, save, load ML models
│   ├── prediction.py             # Inference + INR formatting
│   ├── evaluation.py             # Metrics + comparison tables
│   ├── visualizations.py         # Plotly chart builders
│   └── validation.py             # Input and file validation
├── data/
│   └── README.md                 # Dataset format documentation
├── models/
│   └── README.md                 # Model artifact documentation
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py
│   ├── test_training.py
│   ├── test_prediction.py
│   └── test_validation.py
├── requirements.txt
├── README.md
└── .streamlit/
    └── config.toml               # Theme and server settings
```

---

## Installation

### Prerequisites
- Python 3.11 or 3.10 (recommended)
- pip ≥ 23

### Steps

```bash
# 1. Clone or extract the project
cd "GharValue AI"

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

# 3. Install dependencies
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## Dataset Setup

### Option A — Use a Real Dataset (Recommended)
1. Obtain an Indian housing dataset (see `data/README.md` for sources).
2. Save it as `data/housing_data.csv`.
3. Required columns: `price`, `area`, `bedrooms`, `bathrooms`, `city`.
4. Optional columns: `state`, `locality`, `property_type`, `furnishing`, `parking`, `floor`, `total_floors`, `age`.
5. The application auto-detects common column name variants.

### Option B — Upload via the App
1. Launch the app (see below).
2. Navigate to **Dataset Explorer**.
3. Upload your CSV and configure column mapping.

### Option C — Demo Mode
If no `data/housing_data.csv` is found, the app automatically generates clearly-labelled **synthetic demo data** to demonstrate the workflow. Predictions in this mode have no market validity.

---

## Train the Model

1. Launch the application.
2. Navigate to **Model Evaluation**.
3. Click **Train & Evaluate All Models**.
4. Review the comparison table and charts.
5. Select the best model and click **Save Selected Model**.
6. The pipeline is saved to `models/best_model.joblib`.

---

## Launch the Application

```bash
streamlit run app.py
```

The application opens at **http://localhost:8501** in your browser.

---

## Run Tests

```bash
# Run all tests with verbose output
python -m pytest -v

# Run a specific test file
python -m pytest tests/test_validation.py -v

# Run with short output
python -m pytest -q
```

---

## Application Pages

| Page | Description |
|---|---|
| 🏠 Home Dashboard | Status overview, quick price estimate, recent history |
| 🔮 Predict Price | Full prediction form with validation and location support |
| 📊 Market Insights | Charts: city prices, area vs price, BHK distribution |
| 🧪 Model Evaluation | Train and compare 4 algorithms; save best model |
| 🌐 3D Analytics | Interactive 3D scatter: Area × BHK × Price |
| 📁 Dataset Explorer | Upload CSV, inspect columns, configure mapping |
| 📋 Prediction History | View and export session predictions as CSV |
| ℹ️ About Project | Academic details, architecture, algorithms, references |

---

## Deployment

### Streamlit Community Cloud (Free)

1. Push the project to a public GitHub repository.
2. Add `data/housing_data.csv` (if size permits; otherwise use demo mode or upload).
3. Do **not** commit `models/best_model.joblib` (>50 MB limit may apply).
4. Visit [share.streamlit.io](https://share.streamlit.io), connect your repo, and deploy.
5. Set the main file to `app.py`.

### Important Deployment Notes
- Model files are not persistent across Streamlit Cloud restarts. Users must retrain the model after each deployment.
- For persistent models, consider Hugging Face Spaces or a VPS.
- Dataset must be present in the repo or uploaded via the Dataset Explorer at runtime.

### Local Production
```bash
streamlit run app.py --server.port 8501 --server.address 0.0.0.0
```

---

## Limitations

1. Model accuracy depends entirely on the training dataset's quality and geographic coverage.
2. Predictions are estimates, not certified property valuations.
3. The ±10% range is a heuristic, not a statistically calibrated prediction interval.
4. The model does not incorporate real-time market data.
5. Rare cities and property types will have higher prediction uncertainty.
6. Session history is lost when the browser tab is closed or server restarts.

---

## Academic Disclaimer

> This project is developed for educational purposes as part of the B.Tech IT curriculum at CGIT Bilaspur. Predicted prices are machine-learning estimates and do NOT constitute a certified property valuation, legal appraisal, or investment advice. No warranty is expressed or implied regarding the accuracy of predictions.

---

## Student Information

| Field | Details |
|---|---|
| Student Name | Raj Jaiswal |
| Roll Number | 300703324028 |
| Institution | CHHATTISGARH INSTITUTE OF TECHNOLOGY (CGIT), Bilaspur |
| Department | Information Technology |
| Project Title | House Price Prediction Using Machine Learning |
| Academic Level | B.Tech 3rd Year — VT Project |
| Project Guide | Samiksha Shukla |
| Guide Designation | Assistant Professor, IT Department |

---

## Screenshots

> Add screenshots of the running application here for the project report.
> Suggested screenshots: Home Dashboard, Prediction form with result, Model Evaluation comparison table, Market Insights charts, 3D Analytics.

---

## License

This project is developed for academic purposes. All third-party libraries are used under their respective open-source licenses (MIT, Apache 2.0, BSD). See individual package licenses for details.
