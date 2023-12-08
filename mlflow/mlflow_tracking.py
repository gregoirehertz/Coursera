import joblib
import mlflow
from mlflow.tracking import MlflowClient
from model.model_evaluation import evaluate_model


def log_model_details(model_path, X_test, y_test, params, tags, artifact_paths):
    metrics = evaluate_model(model_path, X_test, y_test)

    if metrics:
        mlflow.set_experiment("Fraud Detection Model Performance")

        with mlflow.start_run():
            mlflow.log_params(params)
            for key, value in metrics.items():
                if key != 'classification_report':
                    mlflow.log_metric(key, value)

            mlflow.set_tags(tags)
            for artifact_path in artifact_paths:
                mlflow.log_artifact(artifact_path)

            run_id = mlflow.active_run().info.run_id
            mlflow.register_model(model_uri=f"runs:/{run_id}/model", name="FraudDetectionModel")

            report_file = "classification_report.txt"
            with open(report_file, "w") as f:
                f.write(metrics['classification_report'])
            mlflow.log_artifact(report_file)


if __name__ == "__main__":
    model_path = 'model/saved_models/model.pkl'
    params = {'num_trees': 100, 'max_depth': 10}
    tags = {'model_type': 'classification', 'framework': 'sklearn'}
    artifact_paths = ['path/to/roc_curve.png', 'path/to/confusion_matrix.png']
    X_test, y_test = joblib.load('model/saved_models/Xy_test.pkl')
    log_model_details(model_path, X_test, y_test, params, tags, artifact_paths)
