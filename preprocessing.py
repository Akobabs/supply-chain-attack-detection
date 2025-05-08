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
        df.to_csv('combined_raw.csv', index=False)  # Save intermediate file
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

def main():
    """Main preprocessing pipeline."""
    try:
        # Define file paths
        file_paths = ['UNSW-NB15_1.csv', 'UNSW-NB15_2.csv', 'UNSW-NB15_3.csv', 'UNSW-NB15_4.csv']
        
        # Run preprocessing steps
        df = load_and_combine_data(file_paths)
        df = handle_missing_values(df)
        df = filter_relevant_attacks(df)
        df = select_features(df)
        df = encode_categorical(df)
        df = balance_classes(df)
        df = normalize_features(df)
        
        # Save final preprocessed dataset
        df.to_csv('unsw_nb15_preprocessed.csv', index=False)
        logging.info("Preprocessing complete. Final dataset saved as 'unsw_nb15_preprocessed.csv'")
    except Exception as e:
        logging.error(f"Preprocessing failed: {e}")
        raise

if __name__ == "__main__":
    main()