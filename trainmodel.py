print("Training model...")
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import shap
import seaborn as sns
import matplotlib.pyplot as plt
import logging

# Configure logging
logging.basicConfig(
    filename='training.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def load_preprocessed_data(file_path):
    """Load preprocessed dataset."""
    try:
        df = pd.read_csv(file_path)
        logging.info(f"Loaded preprocessed data: {df.shape}")
        return df
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        raise

def train_supervised_model(X_train, y_train, X_test, y_test):
    """Train and evaluate Random Forest classifier."""
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        fpr = cm[0,1] / (cm[0,1] + cm[0,0]) if (cm[0,1] + cm[0,0]) > 0 else 0
        
        logging.info(f"Supervised Model - Accuracy: {accuracy:.2f}, F1-Score: {f1:.2f}, FPR: {fpr:.2f}")
        
        # Save confusion matrix plot
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig('confusion_matrix.png')
        plt.close()
        
        return rf, {'accuracy': accuracy, 'f1_score': f1, 'fpr': fpr}
    except Exception as e:
        logging.error(f"Error training supervised model: {e}")
        raise

def train_unsupervised_model(X):
    """Train Isolation Forest for anomaly detection."""
    try:
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        anomalies = iso_forest.fit_predict(X)
        
        # Save anomaly scatterplot
        df_plot = pd.DataFrame(X, columns=X.columns)
        df_plot['Anomaly'] = anomalies
        sns.scatterplot(data=df_plot, x='dur', y='sbytes', hue='Anomaly', palette={-1: 'red', 1: 'blue'})
        plt.title('Anomaly Detection with Isolation Forest')
        plt.savefig('anomaly_plot.png')
        plt.close()
        
        logging.info("Unsupervised model trained and anomaly plot saved")
        return iso_forest
    except Exception as e:
        logging.error(f"Error training unsupervised model: {e}")
        raise

def explain_model(rf, X_test):
    """Generate SHAP explanations."""
    try:
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=X_test.columns, show=False)
        plt.savefig('shap_summary.png')
        plt.close()
        logging.info("SHAP summary plot saved")
    except Exception as e:
        logging.error(f"Error generating SHAP explanations: {e}")
        raise

def test_adversarial_robustness(rf, X_test, y_test):
    """Test model robustness with noise."""
    try:
        X_adv = X_test + np.random.normal(0, 0.1, X_test.shape)
        y_pred_adv = rf.predict(X_adv)
        adv_accuracy = accuracy_score(y_test, y_pred_adv)
        logging.info(f"Adversarial Accuracy: {adv_accuracy:.2f}")
        return adv_accuracy
    except Exception as e:
        logging.error(f"Error testing adversarial robustness: {e}")
        raise

def main():
    """Main training pipeline."""
    try:
        # Load preprocessed data
        df = load_preprocessed_data('data/unsw_nb15_preprocessed.csv')
        X = df.drop('binary_label', axis=1)
        y = df['binary_label']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logging.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        
        # Train supervised model
        rf, metrics = train_supervised_model(X_train, y_train, X_test, y_test)
        
        # Train unsupervised model
        iso_forest = train_unsupervised_model(X)
        
        # Explain model
        explain_model(rf, X_test)
        
        # Test adversarial robustness
        adv_accuracy = test_adversarial_robustness(rf, X_test, y_test)
        
        # Save metrics
        with open('metrics.txt', 'w') as f:
            f.write(f"Accuracy: {metrics['accuracy']:.2f}\n")
            f.write(f"F1-Score: {metrics['f1_score']:.2f}\n")
            f.write(f"False Positive Rate: {metrics['fpr']:.2f}\n")
            f.write(f"Adversarial Accuracy: {adv_accuracy:.2f}\n")
        
        logging.info("Training complete. Metrics and plots saved")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()