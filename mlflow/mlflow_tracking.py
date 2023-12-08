import mlflow
from mlflow.tracking import MlflowClient


def log_model_performance(model_path, metrics):
    """
    Log the model's performance metrics to MLflow.
    Args:
    - model_path: Path to the saved model.
    - metrics: Dictionary of performance metrics.
    """
    mlflow.set_experiment("Fraud Detection Model Performance")

    with mlflow.start_run():
        mlflow.log_params({"model_path": model_path})
        for key, value in metrics.items():
            mlflow.log_metric(key, value)


# Example usage
if __name__ == "__main__":
    model_path = 'model/saved_models/latest_model.pkl'
    metrics = {'accuracy': 0.95, 'precision': 0.90, 'recall': 0.85}
    log_model_performance(model_path, metrics)
