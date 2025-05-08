import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import joblib
from sklearn.model_selection import train_test_split

# Streamlit app configuration
st.set_page_config(page_title="Supply-Chain Attack Detection Dashboard", layout="wide")

# Title and description
st.title("Service Supply-Chain Attack Detection Dashboard")
st.markdown("""
This dashboard visualizes the testing results of a hybrid machine learning model for detecting service supply-chain attacks using the UNSW-NB15 dataset. 
The model uses XGBoost (supervised), Isolation Forest (unsupervised), SHAP for explainability, and tests adversarial robustness with noise. 
You can also make predictions using the saved models.
""")

# Load preprocessed data for predictions
@st.cache_data
def load_data():
    df = pd.read_csv('data/UNSW_NB15_Outputs/combined_raw.csv')
    X = df.drop('binary_label', axis=1)
    y = df['binary_label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_test, y_test

# Load saved models
@st.cache_resource
def load_models():
    xgb_model = joblib.load('data/UNSW_NB15_Outputsmodels/xgb_supervised_model.pkl')
    iso_forest = joblib.load('data/UNSW_NB15_Outputsmodels/isolation_forest_model.pkl')
    return xgb_model, iso_forest

# Load metrics
try:
    with open('data/UNSW_NB15_Outputs/metrics/metrics.txt', 'r') as f:
        metrics = f.readlines()
    metrics_dict = {line.split(': ')[0]: float(line.split(': ')[1]) for line in metrics}
except FileNotFoundError:
    st.error("Metrics file not found at 'metrics/metrics.txt'. Ensure train_model.py has been run.")
    metrics_dict = {}

# Display metrics
st.header("Model Performance Metrics")
col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{metrics_dict.get('Accuracy', 0):.2f}")
col2.metric("F1-Score", f"{metrics_dict.get('F1-Score', 0):.2f}")
col3.metric("False Positive Rate", f"{metrics_dict.get('False Positive Rate', 0):.2f}")
col4.metric("Adversarial Accuracy", f"{metrics_dict.get('Adversarial Accuracy', 0):.2f}")

col5, col6, col7 = st.columns(3)
col5.metric("Cross-Validation F1", f"{metrics_dict.get('Cross-Validation F1', 'N/A')}")
col6.metric("Isolation Forest Accuracy", f"{metrics_dict.get('Isolation Forest Accuracy', 'N/A')}")
col7.metric("Isolation Forest F1-Score", f"{metrics_dict.get('Isolation Forest F1-Score', 'N/A')}")

# Display visualizations
st.header("Visualizations")

# Confusion Matrix
st.subheader("Confusion Matrix (XGBoost)")
try:
    cm_img = Image.open('data/UNSW_NB15_Outputs/plots/confusion_matrix.png')
    st.image(cm_img, caption="Confusion Matrix showing true vs. predicted labels for XGBoost model")
except FileNotFoundError:
    st.warning("Confusion matrix image not found at 'plots/confusion_matrix.png'.")

# Anomaly Scatterplot
st.subheader("Anomaly Detection (Isolation Forest)")
try:
    anomaly_img = Image.open('data/UNSW_NB15_Outputs/plots/anomaly_plot.png')
    st.image(anomaly_img, caption="Scatterplot of anomalies detected in duration vs. source bytes")
except FileNotFoundError:
    st.warning("Anomaly plot image not found at 'plots/anomaly_plot.png'.")

# SHAP Summary Plot
st.subheader("SHAP Explanations (Feature Importance)")
try:
    shap_img = Image.open('data/UNSW_NB15_Outputs/plots/shap_summary.png')
    st.image(shap_img, caption="SHAP summary plot showing feature contributions to XGBoost predictions")
except FileNotFoundError:
    st.warning("SHAP summary image not found at 'plots/shap_summary.png'.")

# Interactive Prediction Section
st.header("Make Predictions")
st.markdown("Select a sample from the test set or input custom values to predict whether it's a supply-chain attack.")

# Load data and models
X_test, y_test = load_data()
xgb_model, iso_forest = load_models()

# Select sample or input custom values
prediction_option = st.selectbox("Choose prediction method:", ["Select a sample from test set", "Input custom values"])

if prediction_option == "Select a sample from test set":
    sample_idx = st.slider("Select a sample index", 0, len(X_test) - 1, 0)
    sample = X_test.iloc[sample_idx:sample_idx+1]
    true_label = y_test.iloc[sample_idx]
    st.write("Selected Sample Features:")
    st.write(sample)

else:
    st.subheader("Input Custom Values")
    custom_input = {}
    features = X_test.columns
    for feature in features:
        custom_input[feature] = st.number_input(f"{feature}", value=float(X_test[feature].mean()), step=0.1)
    sample = pd.DataFrame([custom_input], columns=features)
    true_label = None

# Make predictions
if st.button("Predict"):
    # XGBoost prediction
    xgb_pred = xgb_model.predict(sample)[0]
    xgb_label = "Malicious (Supply-Chain Attack)" if xgb_pred == 1 else "Benign"

    # Isolation Forest prediction
    iso_pred = iso_forest.predict(sample)[0]
    iso_label = "Malicious (Supply-Chain Attack)" if iso_pred == -1 else "Benign"

    # Display results
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    col1.write(f"**XGBoost Prediction**: {xgb_label}")
    col2.write(f"**Isolation Forest Prediction**: {iso_label}")
    if true_label is not None:
        true_label_text = "Malicious (Supply-Chain Attack)" if true_label == 1 else "Benign"
        st.write(f"**True Label**: {true_label_text}")

# Dataset Info
st.header("Dataset Information")
st.markdown("""
- **Dataset**: UNSW-NB15
- **Features Used**: Protocol, Service, Duration, Source Bytes, Destination Bytes, Source Load, Destination Load, Source Mean Packet Size, Destination Mean Packet Size, Source Jitter, Destination Jitter, Source Inter-Packet Time, Destination Inter-Packet Time
- **Attacks**: Backdoors, Reconnaissance (simulating supply-chain attack vectors like API abuse)
- **Records**: Downsampled to 100,000 for training efficiency
""")

# Footer
st.markdown("Developed for service supply-chain attack detection.")