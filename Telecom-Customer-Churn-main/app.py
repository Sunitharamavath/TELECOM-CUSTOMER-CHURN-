import os
from pathlib import Path
import io
import numpy as np
import pandas as pd
import streamlit as st
import joblib

# SHAP + plotting
import shap
import matplotlib.pyplot as plt

# --------------------------
# Configuration
# --------------------------
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="📉",
    layout="wide",
)

MODEL_PATH = os.getenv("MODEL_PATH", "baseline_model.pkl")
EPS = 1e-9

# --------------------------
# Utilities
# --------------------------
@st.cache_resource(show_spinner=True)
def load_model(path: str = MODEL_PATH):
    if not Path(path).exists():
        raise FileNotFoundError(f"Model file not found at: {path}")
    model = joblib.load(path)
    return model

def get_model_components(pipeline):
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    model_input_cols = list(preprocessor.feature_names_in_)
    ohe = preprocessor.named_transformers_["cat"]
    cat_features = ["State", "Service_provider", "type_of_connection"]
    cat_categories = dict(zip(cat_features, ohe.categories_))
    numeric_features = ["value", "value_scaled", "market_share"]
    return preprocessor, classifier, model_input_cols, cat_categories, numeric_features

def build_single_row_df(model_input_cols, form_values: dict):
    row = {col: 0 for col in model_input_cols}
    for k, v in form_values.items():
        if k in row:
            row[k] = v
    numeric_like = {"value", "value_scaled", "market_share", "year", "month",
                    "subscribers_lag_1", "subscribers_lag_3", "subscribers_lag_6",
                    "subscribers_lag_12", "subscribers_ma_3", "subscribers_ma_6",
                    "subscribers_ma_12", "competitor_count", "is_major_operator", "is_metro"}
    for k in list(row.keys()):
        if k in numeric_like and isinstance(row[k], str):
            try:
                row[k] = float(row[k])
            except:
                row[k] = 0.0
    return pd.DataFrame([row], columns=model_input_cols)

def ensure_all_model_columns(df: pd.DataFrame, model_input_cols: list) -> pd.DataFrame:
    df = df.copy()
    for col in model_input_cols:
        if col not in df.columns:
            df[col] = 0
    df = df[model_input_cols]
    return df

def predict_with_threshold(pipeline, df: pd.DataFrame, threshold: float = 0.5):
    proba = pipeline.predict_proba(df)[:, 1]
    pred = (proba >= threshold).astype(int)
    return pred, proba

def compute_expected_roi(pred, proba, revenue_per_customer=100.0, retention_cost=10.0):
    mask = (pred == 1)
    n_target = mask.sum()
    if n_target == 0:
        return 0.0, 0, 0.0, 0.0
    expected_retained = proba[mask].sum()
    expected_gain = expected_retained * revenue_per_customer - n_target * retention_cost
    roi = expected_gain / (n_target * retention_cost + EPS)
    avg_p = float(proba[mask].mean())
    return float(roi), int(n_target), float(expected_retained), avg_p

# NEW: make sure arrays are 2D and dense for SHAP
def ensure_2d(X):
    if hasattr(X, "toarray"):
        X = X.toarray()
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X

# Reusable manual input form
def single_input_form(cat_categories):
    state_opts = list(cat_categories["State"])
    sp_opts = list(cat_categories["Service_provider"])
    conn_opts = list(cat_categories["type_of_connection"])

    col1, col2, col3 = st.columns(3)
    with col1:
        State = st.selectbox("State", options=state_opts, index=min(0, len(state_opts)-1))
        value = st.number_input("value", value=350.0, step=10.0)
    with col2:
        Service_provider = st.selectbox("Service_provider", options=sp_opts, index=min(0, len(sp_opts)-1))
        value_scaled = st.number_input("value_scaled", value=0.85, step=0.01, format="%.4f")
    with col3:
        type_of_connection = st.selectbox("type_of_connection", options=conn_opts, index=min(0, len(conn_opts)-1))
        market_share = st.number_input("market_share", value=0.12, step=0.01, format="%.4f")

    with st.expander("Optional fields (not used by the model but required to exist)"):
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            year = st.number_input("year", value=2025, step=1)
            month = st.number_input("month", value=1, min_value=1, max_value=12, step=1)
        with c2:
            competitor_count = st.number_input("competitor_count", value=0, step=1)
            is_major_operator = st.number_input("is_major_operator", value=0, step=1)
        with c3:
            is_metro = st.number_input("is_metro", value=0, step=1)
            subscribers_lag_1 = st.number_input("subscribers_lag_1", value=0.0, step=1.0)
        with c4:
            subscribers_lag_3 = st.number_input("subscribers_lag_3", value=0.0, step=1.0)
            subscribers_lag_6 = st.number_input("subscribers_lag_6", value=0.0, step=1.0)

        c5, c6, c7 = st.columns(3)
        with c5:
            subscribers_lag_12 = st.number_input("subscribers_lag_12", value=0.0, step=1.0)
        with c6:
            subscribers_ma_3 = st.number_input("subscribers_ma_3", value=0.0, step=1.0)
        with c7:
            subscribers_ma_6 = st.number_input("subscribers_ma_6", value=0.0, step=1.0)
        c8, _ = st.columns(2)
        with c8:
            subscribers_ma_12 = st.number_input("subscribers_ma_12", value=0.0, step=1.0)

    inputs = {
        "State": State,
        "Service_provider": Service_provider,
        "type_of_connection": type_of_connection,
        "value": value,
        "value_scaled": value_scaled,
        "market_share": market_share,
        "year": year,
        "month": month,
        "competitor_count": competitor_count,
        "is_major_operator": is_major_operator,
        "is_metro": is_metro,
        "subscribers_lag_1": subscribers_lag_1,
        "subscribers_lag_3": subscribers_lag_3,
        "subscribers_lag_6": subscribers_lag_6,
        "subscribers_lag_12": subscribers_lag_12,
        "subscribers_ma_3": subscribers_ma_3,
        "subscribers_ma_6": subscribers_ma_6,
        "subscribers_ma_12": subscribers_ma_12,
    }
    return inputs

# --------------------------
# App
# --------------------------
st.title("📉 Customer Churn Prediction Dashboard")

with st.spinner("Loading model..."):
    pipeline = load_model(MODEL_PATH)
preprocessor, classifier, model_input_cols, cat_categories, numeric_features = get_model_components(pipeline)

# Sidebar navigation
page = st.sidebar.radio(
    "Navigation",
    ["Single Prediction", "Batch Scoring", "Model Info", "Explanations (SHAP)"],
    index=0
)

# Global threshold and ROI
st.sidebar.markdown("### Decision Settings")
threshold = st.sidebar.slider("Decision threshold (probability for 'Churn')", 0.05, 0.95, 0.50, 0.01)
st.sidebar.markdown("### ROI Settings (optional)")
rev = st.sidebar.number_input("Revenue per retained churner", min_value=0.0, value=100.0, step=10.0)
cost = st.sidebar.number_input("Retention cost per targeted customer", min_value=0.0, value=10.0, step=1.0)

# --------------------------
# Single Prediction
# --------------------------
if page == "Single Prediction":
    st.header("Single Prediction")
    st.caption("Provide inputs and click Predict. Only a subset of features are used by the model; other features are auto-filled with zeros.")
    inputs = single_input_form(cat_categories)
    if st.button("Predict"):
        try:
            df_row = build_single_row_df(model_input_cols, inputs)
            pred, proba = predict_with_threshold(pipeline, df_row, threshold)
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("Churn probability", f"{proba[0]:.3f}")
            with col_b:
                st.metric("Predicted label", "Churn (1)" if pred[0] == 1 else "No churn (0)")
            st.success("Decision: " + ("Target for retention" if pred[0] == 1 else "Do not target"))
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# --------------------------
# Batch Scoring
# --------------------------
elif page == "Batch Scoring":
    st.header("Batch Scoring")
    st.caption("Upload a CSV. Missing columns will be added automatically and filled with zeros. Extra columns are preserved.")
    with st.expander("Columns expected by the model"):
        st.code(", ".join(model_input_cols), language="text")

    upl = st.file_uploader("Upload CSV", type=["csv"])
    if upl is not None:
        try:
            df_in = pd.read_csv(upl)
            st.write("Input preview:", df_in.head())

            df_pred = ensure_all_model_columns(df_in, model_input_cols)
            pred, proba = predict_with_threshold(pipeline, df_pred, threshold)

            out = df_in.copy()
            out["churn_proba"] = proba
            out["churn_pred"] = pred

            st.write("Scored preview:", out.head())
            csv = out.to_csv(index=False).encode("utf-8")
            st.download_button("Download results CSV", data=csv, file_name="scored_results.csv", mime="text/csv")

            roi, n_target, expected_retained, avg_p = compute_expected_roi(pred, proba, revenue_per_customer=rev, retention_cost=cost)
            st.markdown("### Expected ROI (on targeted set)")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Targets", n_target)
            m2.metric("Expected retained churners", f"{expected_retained:.1f}")
            m3.metric("Avg churn prob (targets)", f"{avg_p:.3f}")
            m4.metric("ROI", f"{roi:.2f}")
        except Exception as e:
            st.error(f"Batch scoring failed: {e}")

# --------------------------
# Model Info
# --------------------------
elif page == "Model Info":
    st.header("Model Info")
    st.markdown("- Model: scikit-learn Pipeline (ColumnTransformer + LogisticRegression)")
    st.markdown(f"- Model file: `{MODEL_PATH}`")

    with st.expander("Raw transformed feature names (after preprocessing)"):
        try:
            transformed_names = preprocessor.get_feature_names_out()
            st.write(pd.DataFrame({"feature": transformed_names}))
        except Exception as e:
            st.warning(f"Could not extract transformed feature names: {e}")

    with st.expander("Logistic regression coefficients (top 20 by absolute value)"):
        try:
            model = classifier
            transformed_names = preprocessor.get_feature_names_out()
            coefs = model.coef_[0]
            coef_df = (
                pd.DataFrame({"feature": transformed_names, "coefficient": coefs, "abs_coef": np.abs(coefs)})
                .sort_values("abs_coef", ascending=False)
                .head(20)
                .drop(columns=["abs_coef"])
            )
            st.dataframe(coef_df.reset_index(drop=True))
        except Exception as e:
            st.warning(f"Could not compute coefficients: {e}")

    with st.expander("Categorical options learned by the model"):
        for k, cats in cat_categories.items():
            st.write(f"- {k}: {', '.join(map(str, cats))}")

    st.info("Note: The model was trained with more columns than it actually uses. The app auto-supplies missing columns with zeros to match training-time schema.")

# --------------------------
# Explanations (SHAP)
# --------------------------
else:
    st.header("Explanations (SHAP)")
    st.caption("Upload a background dataset and explain either a manual input or a specific row.")

    st.markdown("#### 1) Background data (recommended: 500–2,000 representative rows)")
    bg_upl = st.file_uploader("Upload background CSV (used to build the explainer)", type=["csv"], key="bg_csv")
    bg_df = None
    if bg_upl is not None:
        try:
            raw_bg = pd.read_csv(bg_upl)
            st.write("Background preview:", raw_bg.head())
            bg_df = ensure_all_model_columns(raw_bg, model_input_cols)
        except Exception as e:
            st.error(f"Failed to read background CSV: {e}")

    st.markdown("#### 2) Instance to explain")
    mode = st.radio("Choose instance source", ["Manual inputs", "Pick a row from uploaded background"], index=0)

    instance_df = None
    if mode == "Manual inputs":
        inputs = single_input_form(cat_categories)
        if st.button("Explain (manual instance)"):
            try:
                instance_df = build_single_row_df(model_input_cols, inputs)
            except Exception as e:
                st.error(f"Failed to build instance: {e}")
    else:
        if bg_df is None:
            st.warning("Upload a background CSV first to pick a row.")
        else:
            idx = st.number_input("Row index to explain (0-based)", min_value=0, max_value=max(0, len(bg_df)-1), value=0, step=1)
            st.write("Selected row preview:", bg_df.iloc[[idx]].head())
            if st.button("Explain (selected row)"):
                instance_df = bg_df.iloc[[idx]].copy()

    if instance_df is not None:
        try:
            # Transform background and instance through the preprocessor
            if bg_df is None:
                st.warning("No background CSV provided. Using the instance duplicated as background (less reliable).")
                bg_df = instance_df.copy()

            X_bg = preprocessor.transform(bg_df)
            X_bg = ensure_2d(X_bg)

            # If background has < 2 rows, duplicate to keep SHAP masker 2D
            if X_bg.shape[0] < 2:
                reps = 10  # duplicate to provide a minimal background
                X_bg = np.vstack([X_bg] * reps)

            X_inst = preprocessor.transform(instance_df)
            X_inst = ensure_2d(X_inst)

            feature_names = preprocessor.get_feature_names_out()

            # Build a stable masker on the 2D background
            masker = shap.maskers.Independent(X_bg)

            # Probability of class 1 on transformed space
            f = lambda X: classifier.predict_proba(X)[:, 1]

            with st.spinner("Fitting SHAP explainer on background..."):
                explainer = shap.Explainer(f, masker, feature_names=feature_names)

            with st.spinner("Computing SHAP values..."):
                # Limit background sample for global summary to keep it fast
                bg_sample = X_bg[: min(len(X_bg), 500)]
                exp_bg = explainer(bg_sample)
                exp_inst = explainer(X_inst)

            # Predicted probability for the instance
            pred, proba = predict_with_threshold(pipeline, instance_df, threshold)
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Instance churn probability", f"{proba[0]:.3f}")
            with c2:
                st.metric("Predicted label", "Churn (1)" if pred[0] == 1 else "No churn (0)")

            st.markdown("#### Global feature importance (background sample)")
            fig1 = plt.figure(figsize=(8, 5))
            shap.summary_plot(exp_bg.values, bg_sample, feature_names=feature_names, plot_type="bar", show=False)
            st.pyplot(fig1, clear_figure=True)

            st.markdown("#### Local explanation (waterfall) for the selected instance")
            try:
                fig2 = plt.figure(figsize=(8, 6))
                shap.plots.waterfall(exp_inst[0], max_display=25, show=False)
                st.pyplot(fig2, clear_figure=True)
            except Exception:
                fig2 = plt.figure(figsize=(8, 5))
                shap.bar_plot(exp_inst.values[0], feature_names=feature_names, max_display=25, show=False)
                st.pyplot(fig2, clear_figure=True)

            with st.expander("Top features for this instance"):
                vals = exp_inst.values[0]
                df_top = (
                    pd.DataFrame({"feature": feature_names, "shap_value": vals, "abs_val": np.abs(vals)})
                    .sort_values("abs_val", ascending=False)
                    .head(20)
                    .drop(columns=["abs_val"])
                )
                st.dataframe(df_top.reset_index(drop=True))

            st.info("Notes: For best results, upload a representative background (500–2,000 rows). "
                    "SHAP values here explain the post‑preprocessing features (scaled + one‑hot).")
        except Exception as e:
            st.error(f"SHAP explanation failed: {e}")
