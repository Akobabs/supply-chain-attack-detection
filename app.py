import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
from PIL import Image

# Streamlit app configuration
st.set_page_config(page_title="Supply-Chain Attack Detection", layout="wide")

# Title and description
st.title("Service Supply-Chain Attack Detection")
st.markdown("""
This dashboard visualizes the testing results of a hybrid machine learning model for detecting supply-chain attacks using the UNSW-NB15 dataset. 
The model includes Random Forest (supervised), Isolation Forest (unsupervised), SHAP explanations, and adversarial robustness testing.
""")

# Load metrics
try:
    with open('metrics.txt', 'r') as f:
        metrics = f.readlines()
    metrics_dict = {line.split(': ')[0]: float(line.split(': ')[1]) for line in metrics}
except FileNotFoundError:
    st.error("Metrics file not found. Run train_model.py first.")
    metrics_dict = {}

# Display metrics
st.header("Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics_dict.get('Accuracy', 0):.2f}")
col2.metric("F1-Score", f"{metrics_dict.get('F1-Score', 0):.2f}")
col3.metric("False Positive Rate", f"{metrics_dict.get('False Positive Rate', 0):.2f}")
col4.metric("Adversarial Accuracy", f"{metrics_dict.get('Adversarial Accuracy', 0):.2f}")

# Display visualizations
st.header("Visualizations")

# Confusion Matrix
st.subheader("Confusion Matrix (Random Forest)")
try:
    cm_img = Image.open('confusion_matrix.png')
    st.image(cm_img, caption="Confusion Matrix showing true vs. predicted labels")
except FileNotFoundError:
    st.warning("Confusion matrix image not found. Run train_model.py to generate.")

# Anomaly Scatterplot
st.subheader("Anomaly Detection (Isolation Forest)")
try:
    anomaly_img = Image.open('anomaly_plot.png')
    st.image(anomaly_img, caption="Scatterplot of anomalies detected in duration vs. source bytes")
except FileNotFoundError:
    st.warning("Anomaly plot image not found. Run train_model.py to generate.")

# SHAP Summary Plot
st.subheader("SHAP Explanations (Feature Importance)")
try:
    shap_img = Image.open('shap_summary.png')
    st.image(shap_img, caption="SHAP summary plot showing feature contributions to predictions")
except FileNotFoundError:
    st.warning("SHAP summary image not found. Run train_model.py to generate.")

# Dataset Info
st.header("Dataset Information")
st.markdown("""
- **Dataset**: UNSW-NB15
- **Features Used**: Protocol, Service, Duration, Source Bytes, Destination Bytes, Source Load, Destination Load, Source Mean Packet Size, Destination Mean Packet Size, Source Jitter, Destination Jitter, Source Inter-Packet Time, Destination Inter-Packet Time
- **Attacks**: Backdoors, Reconnaissance (simulating supply-chain attack vectors)
- **Records**: ~2.54M (filtered for relevant attacks)
""")

# Footer
st.markdown("Developed by Akobabs for an undergraduate project.")