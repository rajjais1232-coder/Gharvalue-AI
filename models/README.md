# Models Directory

## Model Artifacts

Trained model artifacts are saved here automatically after running the Model Evaluation page.

### Files Generated
- `best_model.joblib` — Best-performing model pipeline (preprocessing + model)
- `model_metadata.json` — Training date, metrics, feature schema, dataset info
- `model_comparison.json` — All model comparison results

### Loading Models

The application loads models automatically. To retrain:
1. Go to the **Model Evaluation** page
2. Click **Train & Evaluate Models**
3. Select the best model and click **Save Best Model**

### Security Note

Only load model artifacts from trusted sources. This application ships with no pre-trained model.
You must train on your own dataset. Never load `.joblib` files from unknown sources.

### Compatibility

Models are saved with the full sklearn Pipeline. Ensure `scikit-learn` version compatibility
when transferring models between environments.
