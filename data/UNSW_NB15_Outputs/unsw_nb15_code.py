
# Preprocessing Code
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import logging
import os

# Configure logging
logging.basicConfig(
    filename='preprocessing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Define column names for UNSW-NB15
COLUMNS = [
    'srcip', 'sport', 'dstip', 'dsport', 'proto', 'state', 'dur', 'sbytes', 'dbytes', 'sttl',
    'dttl', 'sloss', 'dloss', 'service', 'Sload', 'Dload', 'Spkts', 'Dpkts', 'swin', 'dwin',
    'stcpb', 'dtcpb', 'smeansz', 'dmeansz', 'trans_depth', 'res_bdy_len', 'Sjit', 'Djit',
    'Stime', 'Ltime', 'Sintpkt', 'Dintpkt', 'tcprtt', 'synack', 'ackdat', 'is_sm_ips_ports',
    'ct_state_ttl', 'ct_flw_http_mthd', 'is_ftp_login', 'ct_ftp_cmd', 'ct_srv_src',
    'ct_srv_dst', 'ct_dst_ltm', 'ct_src_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
    'ct_dst_src_ltm', 'attack_cat', 'Label'
]

def load_and_combine_data(file_paths):
    """Load and combine UNSW-NB15 CSV files."""
    try:
        dfs = [pd.read_csv(file, names=COLUMNS, low_memory=False) for file in file_paths]
        df = pd.concat(dfs, ignore_index=True)
        logging.info(f"Combined {len(dfs)} files. Total records: {df.shape[0]}")
        df.to_csv('combined_raw.csv', index=False)
        return df
    except Exception as e:
        logging.error(f"Error combining files: {e}")
        raise

def handle_missing_values(df):
    """Handle missing values in the dataset."""
    try:
        logging.info(f"Missing values before: {df.isnull().sum().sum()}")
        df['attack_cat'] = df['attack_cat'].fillna('Normal')
        df = df.dropna(subset=['Label'])
        df.fillna(df.mean(numeric_only=True), inplace=True)
        logging.info(f"Missing values after: {df.isnull().sum().sum()}")
        df.to_csv('missing_handled.csv', index=False)
        return df
    except Exception as e:
        logging.error(f"Error handling missing values: {e}")
        raise

def filter_relevant_attacks(df):
    """Filter for Backdoors and Reconnaissance attacks."""
    try:
        relevant_attacks = ['Normal', 'Backdoors', 'Reconnaissance']
        df = df[df['attack_cat'].isin(relevant_attacks)]
        df['binary_label'] = df['attack_cat'].apply(lambda x: 0 if x == 'Normal' else 1)
        logging.info(f"Filtered attacks. Records: {df.shape[0]}, Classes: {df['binary_label'].value_counts().to_dict()}")
        df.to_csv('filtered_attacks.csv', index=False)
        return df
    except Exception as e:
        logging.error(f"Error filtering attacks: {e}")
        raise

def select_features(df):
    """Select time and content features for supply-chain attack detection."""
    try:
        features = [
            'proto', 'service', 'dur', 'sbytes', 'dbytes', 'Sload', 'Dload',
            'smeansz', 'dmeansz', 'Sjit', 'Djit', 'Sintpkt', 'Dintpkt', 'binary_label'
        ]
        df = df[features]
        logging.info(f"Selected features: {features}")
        df.to_csv('features_selected.csv', index=False)
        return df
    except Exception as e:
        logging.error(f"Error selecting features: {e}")
        raise

def encode_categorical(df):
    """Encode categorical features."""
    try:
        categorical_cols = ['proto', 'service']
        for col in categorical_cols:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            logging.info(f"Encoded column: {col}")
        df.to_csv('encoded_categorical.csv', index=False)
        return df
    except Exception as e:
        logging.error(f"Error encoding categorical features: {e}")
        raise

def balance_classes(df):
    """Balance classes using SMOTE."""
    try:
        X = df.drop('binary_label', axis=1)
        y = df['binary_label']
        smote = SMOTE(random_state=42)
        X_resampled, y_resampled = smote.fit_resample(X, y)
        df_resampled = pd.concat([pd.DataFrame(X_resampled, columns=X.columns), pd.Series(y_resampled, name='binary_label')], axis=1)
        logging.info(f"Balanced classes. New records: {df_resampled.shape[0]}, Classes: {df_resampled['binary_label'].value_counts().to_dict()}")
        df_resampled.to_csv('balanced_classes.csv', index=False)
        return df_resampled
    except Exception as e:
        logging.error(f"Error balancing classes: {e}")
        raise

def normalize_features(df):
    """Normalize numerical features."""
    try:
        X = df.drop('binary_label', axis=1)
        scaler = StandardScaler()
        X_normalized = scaler.fit_transform(X)
        df_normalized = pd.concat([pd.DataFrame(X_normalized, columns=X.columns), df['binary_label'].reset_index(drop=True)], axis=1)
        logging.info("Normalized features")
        df_normalized.to_csv('normalized_features.csv', index=False)
        return df_normalized
    except Exception as e:
        logging.error(f"Error normalizing features: {e}")
        raise

def main_preprocessing():
    """Main preprocessing pipeline."""
    try:
        file_paths = [
            '/content/UNSW_DATASET/CSV Files/UNSW-NB15_1.csv',
            '/content/UNSW_DATASET/CSV Files/UNSW-NB15_2.csv',
            '/content/UNSW_DATASET/CSV Files/UNSW-NB15_3.csv',
            '/content/UNSW_DATASET/CSV Files/UNSW-NB15_4.csv'
        ]
        
        df = load_and_combine_data(file_paths)
        df = handle_missing_values(df)
        df = filter_relevant_attacks(df)
        df = select_features(df)
        df = encode_categorical(df)
        df = balance_classes(df)
        df = normalize_features(df)
        
        df.to_csv('unsw_nb15_preprocessed.csv', index=False)
        logging.info("Preprocessing complete. Final dataset saved as 'unsw_nb15_preprocessed.csv'")
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise

# Training Code
print("Training model with GPU acceleration...")

import os
import pandas as pd
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.ensemble import IsolationForest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from xgboost import XGBClassifier
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import logging
import joblib

print("GPU available:", torch.cuda.is_available())

os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)
os.makedirs('plots', exist_ok=True)

logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_preprocessed_data(file_path):
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded preprocessed data: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def train_supervised_model(X_train, y_train, X_test, y_test):
    try:
        rf = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor',
                           use_label_encoder=False, eval_metric='logloss', random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr = cm[0, 1] / (cm[0, 1] + cm[0, 0]) if (cm[0, 1] + cm[0, 0]) > 0 else 0

        logging.info(f"Supervised Model - Accuracy: {accuracy:.2f}, F1-Score: {f1:.2f}, FPR: {fpr:.2f}")

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('plots/confusion_matrix.png')
        plt.close()

        joblib.dump(rf, 'models/xgb_supervised_model.pkl')

        return rf, {'accuracy': accuracy, 'f1_score': f1, 'fpr': fpr}
    except Exception as e:
        logging.error(f"Error training supervised model: {e}")
        raise

def train_unsupervised_model(X):
    try:
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X)

        df_plot = pd.DataFrame(X, columns=X.columns)
        df_plot['Anomaly'] = anomalies
        sns.scatterplot(data=df_plot, x='dur', y='sbytes', hue='Anomaly', palette={-1: 'red', 1: 'blue'})
        plt.title('Anomaly Detection with Isolation Forest')
        plt.savefig('plots/anomaly_plot.png')
        plt.close()

        joblib.dump(iso_forest, 'models/isolation_forest_model.pkl')

        logging.info("Unsupervised model trained and anomaly plot saved")
        return iso_forest
    except Exception as e:
        logging.error(f"Error training unsupervised model: {e}")
        raise

def explain_model(rf, X_test):
    try:
        explainer = shap.Explainer(rf)
        shap_values = explainer(X_test)
        shap.summary_plot(shap_values, X_test, show=False)
        plt.savefig('plots/shap_summary.png')
        plt.close()
        logging.info("SHAP summary plot saved")
    except Exception as e:
        logging.error(f"Error generating SHAP explanations: {e}")
        raise

def test_adversarial_robustness(rf, X_test, y_test):
    try:
        X_adv = X_test + np.random.normal(0, 0.1, X_test.shape)
        y_pred_adv = rf.predict(X_adv)
        adv_accuracy = accuracy_score(y_test, y_pred_adv)
        logging.info(f"Adversarial Accuracy: {adv_accuracy:.2f}")
        return adv_accuracy
    except Exception as e:
        logging.error(f"Error testing adversarial robustness: {e}")
        raise

def main_training():
    try:
        df = load_preprocessed_data('/content/unsw_nb15_preprocessed.csv')
        
        logging.info(f"Original dataset size: {df.shape}")
        if df.shape[0] > 100000:
            df = df.sample(100000, random_state=42)
            logging.info(f"Downsampled dataset to: {df.shape}")

        X = df.drop('binary_label', axis=1)
        y = df['binary_label']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")

        rf, metrics = train_supervised_model(X_train, y_train, X_test, y_test)
        iso_forest = train_unsupervised_model(X)
        explain_model(rf, X_test)
        adv_accuracy = test_adversarial_robustness(rf, X_test, y_test)

        with open('metrics/metrics.txt', 'w') as f:
            f.write(f"Accuracy: {metrics['accuracy']:.2f}\n")
            f.write(f"F1-Score: {metrics['f1_score']:.2f}\n")
            f.write(f"False Positive Rate: {metrics['fpr']:.2f}\n")
            f.write(f"Adversarial Accuracy: {adv_accuracy:.2f}\n")

        logging.info("Training complete. All metrics, plots, and models saved.")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main_preprocessing()
    main_training()
