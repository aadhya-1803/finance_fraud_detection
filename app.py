import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve, precision_recall_curve
)

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    layout="wide",
    page_icon="assets/favicon.ico" if False else None
)

# ── Load models and test data ────────────────────────────────
@st.cache_resource
def load_models():
    rf  = joblib.load('rf_model.pkl')
    xgb = joblib.load('xgb_model.pkl')
    iso = joblib.load('iso_model.pkl')
    return rf, xgb, iso

@st.cache_data
def load_test_data():
    X = np.load('X_test.npy')
    y = np.load('y_test.npy')
    return X, y

rf_model, xgb_model, iso_model = load_models()
X_test, y_test = load_test_data()

# ── Sidebar controls ─────────────────────────────────────────
st.sidebar.title("Controls")

model_choice = st.sidebar.selectbox(
    "Select Model",
    ["XGBoost", "Random Forest", "Isolation Forest"]
)

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1, max_value=0.9,
    value=0.8, step=0.05,
    help="Applies to XGBoost and Random Forest only"
)

st.sidebar.markdown("---")
st.sidebar.markdown("Threshold only affects probability-based models. Isolation Forest uses a fixed contamination rate.")

# ── Generate predictions based on selection ──────────────────
def get_predictions(model_name, threshold):
    if model_name == "XGBoost":
        probs = xgb_model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        return preds, probs

    elif model_name == "Random Forest":
        probs = rf_model.predict_proba(X_test)[:, 1]
        preds = (probs >= threshold).astype(int)
        return preds, probs

    elif model_name == "Isolation Forest":
        raw = iso_model.predict(X_test)
        preds = np.array([1 if x == -1 else 0 for x in raw])
        return preds, None

preds, probs = get_predictions(model_choice, threshold)

# ── Metrics ───────────────────────────────────────────────────
report = classification_report(y_test, preds, output_dict=True, target_names=["Normal", "Fraud"])
fraud_precision = round(report["Fraud"]["precision"], 3)
fraud_recall    = round(report["Fraud"]["recall"], 3)
fraud_f1        = round(report["Fraud"]["f1-score"], 3)
auc_score       = round(roc_auc_score(y_test, probs), 4) if probs is not None else "N/A"

cm = confusion_matrix(y_test, preds)

# ── Header ────────────────────────────────────────────────────
st.title("Credit Card Fraud Detection Dashboard")
st.markdown(f"Showing results for **{model_choice}** at decision threshold **{threshold}**")
st.markdown("---")

# ── Row 1: KPI Cards ─────────────────────────────────────────
st.subheader("Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", "284,807")
col2.metric("Fraud Cases", "492")
col3.metric("Fraud Rate", "0.17%")
col4.metric("Best AUC-ROC", "0.976")

st.markdown("---")

# ── Row 2: Live Model Metrics ─────────────────────────────────
st.subheader(f"{model_choice} — Live Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Fraud Precision", fraud_precision)
col2.metric("Fraud Recall",    fraud_recall)
col3.metric("F1 Score",        fraud_f1)
col4.metric("AUC-ROC",         auc_score)

st.markdown("---")

# ── Row 3: Confusion Matrix + ROC Curve ──────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Confusion Matrix")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Normal', 'Fraud'],
                yticklabels=['Normal', 'Fraud'], ax=ax)
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')
    ax.set_title(f'{model_choice} — Threshold {threshold}')
    st.pyplot(fig)

with col2:
    if probs is not None:
        st.subheader("ROC Curve")
        fpr, tpr, _ = roc_curve(y_test, probs)
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(fpr, tpr, color='steelblue', lw=2,
                label=f'AUC = {auc_score}')
        ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title('ROC Curve')
        ax.legend(loc='lower right')
        st.pyplot(fig)
    else:
        st.info("ROC Curve not available for Isolation Forest (no probability output).")

st.markdown("---")

# ── Row 4: Precision-Recall Curve ────────────────────────────
if probs is not None:
    st.subheader("Precision vs Recall at Different Thresholds")
    precisions, recalls, thresholds_pr = precision_recall_curve(y_test, probs)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(thresholds_pr, precisions[:-1], label='Precision', color='steelblue')
    ax.plot(thresholds_pr, recalls[:-1],    label='Recall',    color='orange')
    ax.axvline(x=threshold, color='red', linestyle='--',
               label=f'Current threshold ({threshold})')
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Score')
    ax.set_title('Precision and Recall vs Decision Threshold')
    ax.legend()
    st.pyplot(fig)

st.markdown("---")

# ── Row 5: Model Comparison Table ────────────────────────────
st.subheader("Model Comparison")

comparison_df = pd.DataFrame({
    "Model":           ["Isolation Forest", "Random Forest", "XGBoost (Threshold 0.8)"],
    "Fraud Precision": [0.31, 0.81, 0.62],
    "Fraud Recall":    [0.33, 0.81, 0.85],
    "F1 Score":        [0.32, 0.81, 0.72],
    "AUC-ROC":         ["—",  0.9688, 0.976]
})

st.dataframe(comparison_df.set_index("Model"), use_container_width=True)

st.markdown("---")
