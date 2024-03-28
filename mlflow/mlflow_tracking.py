import joblib
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
import os
import sys
import logging
import seaborn as sns
import matplotlib.pyplot as plt

def log_model_details(model_path, test_data_path, experiment_name, model_name, additional_params=None):
    """
    Log model details using MLflow.

    Args:
        model_path: Path to the trained model file.
        test_data_path: Path to the test dataset.
        experiment_name: Name of the MLflow experiment.
        model_name: Name to register the model with.
        additional_params: Additional parameters or information to log.

    """
    try:
        X_test, y_test = joblib.load(test_data_path)
        model = joblib.load(model_path)
        mlflow.set_experiment(experiment_name)

        with mlflow.start_run():
            # Log model parameters
            params = model.get_params()
            if additional_params:
                params.update(additional_params)
            mlflow.log_params(params)

            # Evaluate model and log metrics
            y_pred = model.predict(X_test)
            y_pred_proba = model.predict_proba(X_test)[:, 1]
            mlflow.log_metric('accuracy', accuracy_score(y_test, y_pred))
            mlflow.log_metric('roc_auc', roc_auc_score(y_test, y_pred_proba))

            # Log classification report and confusion matrix
            report = classification_report(y_test, y_pred)
            report_file = "assets/classification_report.txt"
            with open(report_file, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_file)

            cm = confusion_matrix(y_test, y_pred)
            plt.figure()
            sns.heatmap(cm, annot=True, fmt='g')
            plt.title('Confusion Matrix')
            plt.ylabel('Actual label')
            plt.xlabel('Predicted label')
            confusion_matrix_path = "assets/confusion_matrix.png"
            plt.savefig(confusion_matrix_path)
            mlflow.log_artifact(confusion_matrix_path)

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

    # Example: additional_params could include environment details or other configurations
    additional_params = {'python_version': sys.version, 'mlflow_version': mlflow.__version__}
    
    log_model_details(model_path, test_data_path, experiment_name, model_name, additional_params)
