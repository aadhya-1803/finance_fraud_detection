
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc

# Page config
st.set_page_config(page_title="Fraud Detection Dashboard",
                   layout="wide",
                   page_icon="🔍")

# Title
st.title("🔍 Credit Card Fraud Detection Dashboard")
st.markdown("**An end-to-end ML project using Random Forest, XGBoost & Isolation Forest**")
st.markdown("---")

# Sidebar
st.sidebar.title("⚙️ Controls")
model_choice = st.sidebar.selectbox("Select Model",
                ["XGBoost (Best)", "Random Forest", "Isolation Forest"])
threshold = st.sidebar.slider("Decision Threshold", 0.1, 0.9, 0.8, 0.05)

# ── Row 1: KPI Cards ──
st.subheader("📊 Dataset Overview")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Transactions", "284,807")
col2.metric("Fraud Cases", "492")
col3.metric("Fraud Rate", "0.17%")
col4.metric("Best AUC-ROC", "0.976")

st.markdown("---")

# ── Row 2: Class Distribution ──
col1, col2 = st.columns(2)

with col1:
    st.subheader("Class Imbalance")
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(['Normal', 'Fraud'], [284315, 492],
           color=['steelblue', 'red'], edgecolor='black')
    ax.set_ylabel("Count")
    ax.set_title("Normal vs Fraud Transactions")
    st.pyplot(fig)

with col2:
    st.subheader("After SMOTE Balancing")
    fig, ax = plt.subplots(figsize=(5,3))
    ax.bar(['Normal', 'Fraud'], [227451, 227451],
           color=['steelblue', 'green'], edgecolor='black')
    ax.set_ylabel("Count")
    ax.set_title("Balanced Training Data")
    st.pyplot(fig)

st.markdown("---")

# ── Row 3: Model Comparison ──
st.subheader("🤖 Model Performance Comparison")

results = pd.DataFrame({
    'Model': ['Isolation Forest', 'Random Forest', 'XGBoost Tuned'],
    'Precision': [0.31, 0.81, 0.62],
    'Recall': [0.33, 0.81, 0.85],
    'F1 Score': [0.32, 0.81, 0.72],
    'AUC-ROC': [0.65, 0.9688, 0.976]
})

col1, col2 = st.columns(2)

with col1:
    st.dataframe(results.set_index('Model'), use_container_width=True)

with col2:
    fig, ax = plt.subplots(figsize=(5,3))
    x = np.arange(len(results['Model']))
    width = 0.2
    ax.bar(x - width, results['Precision'], width, label='Precision', color='steelblue')
    ax.bar(x, results['Recall'], width, label='Recall', color='orange')
    ax.bar(x + width, results['F1 Score'], width, label='F1', color='green')
    ax.set_xticks(x)
    ax.set_xticklabels(results['Model'], rotation=10, fontsize=8)
    ax.set_ylim(0, 1.1)
    ax.legend(fontsize=8)
    ax.set_title("Metrics Comparison")
    st.pyplot(fig)

st.markdown("---")

# ── Row 4: Confusion Matrices ──
st.subheader("🔢 Confusion Matrices")

col1, col2, col3 = st.columns(3)

cms = {
    'Random Forest':    [[56846, 18], [19, 79]],
    'XGBoost Tuned':    [[56814, 50], [15, 83]],
    'Isolation Forest': [[56800, 64], [65, 33]]
}

for col, (name, cm) in zip([col1, col2, col3], cms.items()):
    with col:
        fig, ax = plt.subplots(figsize=(4,3))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Normal','Fraud'],
                    yticklabels=['Normal','Fraud'], ax=ax)
        ax.set_title(name, fontsize=10)
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')
        st.pyplot(fig)

st.markdown("---")
st.markdown("Built with ❤️ using Python, Scikit-learn, XGBoost & Streamlit")
