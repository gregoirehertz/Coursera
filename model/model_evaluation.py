import pandas as pd
import joblib
import logging
import sys
import os
import mlflow
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_test_data(test_data_path):
    """
    Load test data from a given path.
    """
    try:
        X_test, y_test = joblib.load(test_data_path)
        logging.info(f"Test data loaded successfully from {test_data_path}")
        return X_test, y_test
    except Exception as e:
        logging.error(f"Error loading test data: {e}")
        sys.exit(1)


def evaluate_model(model_path, X_test, y_test):
    """
    Evaluate the model and return metrics.
    """
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba),
            'classification_report': classification_report(y_test, y_pred)
        }

        logging.info("Model evaluation completed.")
        return metrics
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        return None


if __name__ == "__main__":
    mlflow.set_experiment("fraud_detection_evaluation")

    test_data_path = os.getenv('TEST_DATA_PATH', 'data/test_data.pkl')
    model_path = os.getenv('MODEL_PATH', 'model/saved_models/model.pkl')

    with mlflow.start_run():
        X_test, y_test = load_test_data(test_data_path)
        metrics = evaluate_model(model_path, X_test, y_test)

        if metrics:
            for key, value in metrics.items():
                if key != 'classification_report':
                    mlflow.log_metric(key, value)

            logging.info(metrics['classification_report'])
