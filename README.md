# Service Supply-Chain Attack Detection

## Overview

This project implements a hybrid machine learning pipeline to detect service supply-chain attacks through network traffic analysis, using the UNSW-NB15 dataset. The system combines:

- **Supervised Learning** (XGBoost)
- **Unsupervised Learning** (Isolation Forest)

It targets Backdoors and Reconnaissance attacks that simulate supply-chain threats, such as API abuse. A Streamlit dashboard is provided for interactive visualization of performance metrics, confusion matrices, anomaly plots, and SHAP explanations.

---

## Dataset

- **Source**: [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset) from UNSW Canberra’s Cyber Range Lab  
- **Size**: 2,540,044 records, 49 features  
- **Subset Used**: 100,000 samples (downsampled for efficiency)  
- **Features**:
  - Time-based: `dur`, `Sjit`, `Djit`, `Sintpkt`, `Dintpkt`  
  - Content-based: `sbytes`, `dbytes`, `Sload`, `Dload`, `smeansz`, `dmeansz`  
  - Categorical: `proto`, `service`  
- **Focus**: Backdoors and Reconnaissance attacks  
- **Balancing**: SMOTE applied to handle class imbalance

---

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Akobabs/supply-chain-attack-detection.git
cd supply-chain-attack-detection
```

### 2. Install Dependencies

```bash
pip install pandas scikit-learn imbalanced-learn xgboost shap seaborn matplotlib streamlit torch joblib
```

### 3. Dataset Preparation

Place the UNSW-NB15 CSV files (`UNSW-NB15_1.csv` to `UNSW-NB15_4.csv`) in the correct folder, e.g.:

```
/content/UNSW_DATASET/CSV Files/
```

---

## Usage

### Step 1: Preprocess the Data

```bash
python preprocess.py
```

- Outputs:
  - `unsw_nb15_preprocessed.csv`
  - Intermediate files (e.g., `missing_handled.csv`)

### Step 2: Train the Models

```bash
python train_model.py
```

- Outputs:
  - `models/xgb_supervised_model.pkl`
  - `models/isolation_forest_model.pkl`
  - `metrics/metrics.txt`
  - Visual plots in `plots/`

### Step 3: Validate the Models

```bash
python validate_models.py
```

- Updates `metrics/metrics.txt` with validation results

### Step 4: Launch the Dashboard

```bash
streamlit run app.py
```

- Access at: http://localhost:8501

For Google Colab users:

```bash
npm install localtunnel
streamlit run app.py &>/dev/null &
npx localtunnel --port 8501
```

---

## File Descriptions

- `preprocess.py` – Cleans and encodes the dataset, applies SMOTE, selects relevant features
- `train_model.py` – Trains XGBoost and Isolation Forest, generates plots and saves models
- `validate_models.py` – Performs cross-validation and model evaluation
- `app.py` – Launches Streamlit dashboard for visualization
- `plots/` – Contains confusion matrix, SHAP plot, and anomaly scatterplot
- `models/` – Saved model files in `.pkl` format
- `metrics/` – Includes `metrics.txt` with evaluation results

---

## Results

### XGBoost

- Accuracy: **1.00**
- F1-Score: **1.00**
- False Positive Rate: **0.00**
- Adversarial Accuracy: **0.72**
- Cross-Validation F1: **(Run `validate_models.py`)**

### Isolation Forest

- Accuracy: **(Pending)**
- F1-Score: **(Pending)**

> **Note**: Perfect scores may indicate overfitting due to SMOTE and downsampling. Cross-validation is strongly recommended.

---

## Visualizations

- **Confusion Matrix** – `confusion_matrix.png`
- **Anomaly Scatterplot** – `anomaly_plot.png`
- **SHAP Summary Plot** – `shap_summary.png`

---

## Limitations

- Overfitting likely due to data balancing and reduced sample size
- Focuses on traffic-level attacks; does not detect compromised dependencies
- Vulnerable to adversarial noise (28% accuracy drop observed)

---

## Future Work

- Apply regularization (e.g., reduce `max_depth` in XGBoost)
- Introduce adversarial training for robustness
- Extend to dependency-level datasets (e.g., [MalOSS](https://github.com/maloss))
- Enable real-time inference in the dashboard

---

## Author

**Akobabs**  
[GitHub Profile](https://github.com/Akobabs)

---

## Next Steps

1. Run model validation (if not already done):

```bash
python validate_models.py
```

2. Update metrics in the `README.md` from `metrics/metrics.txt`

3. Push to GitHub:

```bash
git add .
git commit -m "Add project files and README"
git push origin main
```

---
