# 📉 Customer Churn Prediction Dashboard

A production‑ready Streamlit app that serves a trained scikit‑learn Pipeline (ColumnTransformer + Logistic Regression) to predict telecom customer churn, score datasets in batch, inspect model details, and generate SHAP explanations for transparency.

This project is designed for rapid deployment on Streamlit Community Cloud and reproducible local runs.

---

## Table of Contents

- [Key Features](#key-features)
- [Live Demo](#live-demo)
- [Project Structure](#project-structure)
- [Model Overview](#model-overview)
- [Data Inputs and Schema](#data-inputs-and-schema)
- [Local Setup](#local-setup)
- [How to Use the App](#how-to-use-the-app)
  - [Single Prediction](#single-prediction)
  - [Batch Scoring](#batch-scoring)
  - [Model Info](#model-info)
  - [Explanations (SHAP)](#explanations-shap)
- [Business Controls](#business-controls)
  - [Decision Threshold](#decision-threshold)
  - [Expected ROI](#expected-roi)
- [Troubleshooting](#troubleshooting)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

---

## Key Features

- End‑to‑end scikit‑learn Pipeline inference (preprocessing + model) from a single `.pkl`
- Single-record predictions with interactive inputs
- Batch scoring of CSV files with downloadable results
- Adjustable decision threshold for precision/recall trade‑offs
- Simple Expected ROI calculator for retention targeting decisions
- Model transparency:
  - Transformed feature names (post‑scaling and one‑hot)
  - Top logistic regression coefficients
  - SHAP global and local explanations (bar summary and waterfall plots)
- Robust schema handling: app auto‑adds missing columns expected by the model to avoid inference errors

---

## Project Structure

```
.
├── app.py                  # Streamlit application
├── baseline_model.pkl      # Trained scikit-learn Pipeline (preprocessor + classifier)
├── requirements.txt        # Python dependencies (pinned for reproducibility)
├── runtime.txt             # Pin Python version on Streamlit Cloud (e.g., python-3.11.9)
├── README.md               # This file
└── .gitignore
```
---

## Model Overview

- Algorithm: Logistic Regression (binary classification)
- Pipeline:
  - ColumnTransformer
    - StandardScaler on numeric: `['value', 'value_scaled', 'market_share']`
    - OneHotEncoder(handle_unknown='ignore') on categorical:
      `['State', 'Service_provider', 'type_of_connection']`
  - LogisticRegression(random_state=42, max_iter=1000)
- Training insight:
  - Although the training dataset contained many engineered features (e.g., lags, moving averages), the ColumnTransformer is configured to use only the six columns above; the app ensures the full training‑time schema exists by adding missing columns with zeros to keep the pipeline happy.

---

## Data Inputs and Schema

- The pipeline was fitted with a specific input schema (training‑time columns).  
- At inference:
  - For single predictions, the app collects the six used features and auto‑adds other expected columns with zeros.
  - For batch predictions, uploaded CSVs are aligned to the model’s schema (missing columns are added and ordered correctly).

Minimum columns you should provide in uploads (others will be auto‑filled if missing):
- `State` (categorical; must be one of the categories learned during training)
- `Service_provider` (categorical; must be in learned categories)
- `type_of_connection` (categorical; e.g., wireless/wireline)
- `value` (numeric)
- `value_scaled` (numeric; if unavailable, you may supply 0 and rely on `value`)
- `market_share` (numeric, 0–1)

Tip: For best SHAP/global behaviors, use data in ranges similar to the training data.

---

## Local Setup

1) Create and activate a virtual environment (Python 3.11 recommended)
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

2) Install dependencies (pinned to match training)
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

3) Place `baseline_model.pkl` at the project root (or set `MODEL_PATH` env var).

4) Run the app
```bash
streamlit run app.py
```

The app opens at http://localhost:8501.

---

## How to Use the App

### Single Prediction
- Select `State`, `Service_provider`, `type_of_connection`
- Enter numeric values: `value`, `value_scaled`, `market_share`
- Expand “Optional fields” if you want to override default zeros for extra columns
- Click “Predict” to see:
  - Churn probability (class 1)
  - Predicted label using the current threshold
  - Decision hint (target vs do not target)

### Batch Scoring
- Upload a CSV; the app will add missing columns and align order
- Results show:
  - `churn_proba` (probability of churn)
  - `churn_pred` (0/1 using the current threshold)
- Download the full results as CSV
- ROI metrics for targeted customers (predicted=1) appear using the values in the sidebar

### Model Info
- See transformed feature names (as consumed by the model)
- Top coefficients (by absolute value) for Logistic Regression
- Categorical levels learned by the OneHotEncoder (your inputs must use these)

### Explanations (SHAP)
- Upload a representative background CSV (500–2,000 rows recommended)
- Explain either:
  - Manual inputs, or
  - A specific row from your uploaded background
- View:
  - Global importance (SHAP bar plot on background sample)
  - Local waterfall plot for the instance
- Notes:
  - Explanations operate on transformed features (scaled + one‑hot)
  - If no background is provided, the app duplicates the instance (less reliable but works)

---

## Business Controls

### Decision Threshold
- Set the threshold (0–1) for mapping probability → label:
  - Lower threshold → more positives (higher recall, lower precision)
  - Higher threshold → fewer positives (higher precision, lower recall)

### Expected ROI
- Parameters (sidebar):
  - Revenue per retained churner
  - Retention cost per targeted customer
- Calculation (on predicted positives):
  - Expected retained churners ≈ sum of probabilities of targeted rows
  - Expected Gain = (Expected Retained * Revenue) − (Targets * Cost)
  - ROI = Expected Gain / (Targets * Cost)
- Use this to compare thresholds and understand trade‑offs quickly.

---

## Troubleshooting

- “Model file not found”
  - Ensure `baseline_model.pkl` is present at repo root or set `MODEL_PATH` env var.

- “Could not load model” / version mismatch
  - Use Python 3.11 and `scikit-learn==1.4.2` as pinned in `requirements.txt`.
  - If necessary, re‑save the model using the same environment you will deploy.

- Large model file (>100 MB) on GitHub
  - Option A: Re‑export a smaller model (drop unused transformers; `joblib.dump(..., compress=3)`).
  - Option B: Host the model on a URL (S3/GCS/GitHub Releases) and download at runtime with caching.

- SHAP errors about masker shape or dimensions
  - Upload a background CSV with ≥ 2 rows. The app duplicates background if needed, but a real sample is better.
  - The app converts sparse → dense and ensures arrays are 2D to prevent shape mismatches.

- Categories not found
  - Use values present in “Categorical options learned by the model” on the Model Info page.
  - The OneHotEncoder has `handle_unknown='ignore'`; unseen categories won’t break inference but will be ignored.

---

## Roadmap

- Add evaluation metrics (Accuracy, F1, PR‑AUC) when `churn_flag` exists in uploaded CSV
- Add SHAP LinearExplainer toggle (faster for logistic regression, explains log‑odds)
- Persist UI state via query params or session state
- Role‑based access and simple auth for protected deployments

---

## Contributing

1) Fork the repo and create a feature branch:
```bash
git checkout -b feature/my-improvement
```
2) Make changes and add tests where applicable
3) Ensure the app runs locally (`streamlit run app.py`)
4) Submit a PR describing changes, reasoning, and screenshots

---

## Acknowledgements

- [Streamlit](https://streamlit.io) for rapid app development
- [scikit‑learn](https://scikit-learn.org) for the modeling stack
- [SHAP](https://shap.readthedocs.io) for explainability
- Matplotlib, NumPy, Pandas for data and plots

---
