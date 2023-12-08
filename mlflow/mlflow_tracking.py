import joblib
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report
import os
import sys
import logging
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score


def log_model_details(model_path, test_data_path, experiment_name, model_name):
    """
    Log model details using MLflow.
    """
    try:
        X_test, y_test = joblib.load(test_data_path)
        model = joblib.load(model_path)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Dynamically log model parameters
            params = model.get_params()
            mlflow.log_params(params)

            # Evaluate model and log metrics
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
            mlflow.log_metric('roc_auc', roc_auc_score(y_test, y_pred_proba))

            # Log classification report as an artifact
            report = classification_report(y_test, y_pred)
            report_file = "classification_report.txt"
            with open(report_file, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_file)

            # Register model
            run_id = mlflow.active_run().info.run_id
            mlflow.register_model(model_uri=f"runs:/{run_id}/model", name=model_name)

    except Exception as e:
        logging.error(f"Error in MLflow tracking: {e}")
        sys.exit(1)


if __name__ == "__main__":
    model_path = os.getenv('MODEL_PATH', 'model/saved_models/model.pkl')
    test_data_path = os.getenv('TEST_DATA_PATH', 'data/test_data.pkl')
    experiment_name = "Fraud Detection Model Performance"
    model_name = "FraudDetectionModel"

    log_model_details(model_path, test_data_path, experiment_name, model_name)
