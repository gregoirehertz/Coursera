import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import joblib
import logging
import sys
import os
import mlflow
from mlflow.sklearn import log_model

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(data_path):
    """
    Load data from a given path.
    """
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully from {data_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)


def preprocess_data(data):
    """
    Preprocess the data, returning features and labels.
    """
    # Example preprocessing step
    # data.fillna(0, inplace=True)
    X = data.drop('class', axis=1)
    y = data['class']
    return X, y


def train_model(X, y, param_grid, test_size=0.3):
    """
    Train the model with GridSearchCV.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    model = RandomForestClassifier(random_state=42)

    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logging.info(f"Best model parameters: {grid_search.best_params_}")

    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy}, ROC-AUC: {roc_auc}")

    return best_model, X_test, y_test


def save_model(model, model_save_path):
    """
    Save the model to a given path.
    """
    try:
        joblib.dump(model, model_save_path)
        logging.info(f"Model saved to {model_save_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")


if __name__ == '__main__':
    mlflow.set_experiment("fraud_detection")

    data_path = os.getenv('DATA_PATH', 'data/creditcard.csv')
    model_save_path = os.getenv('MODEL_SAVE_PATH', 'model/saved_models/model.pkl')
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}

    with mlflow.start_run():
        data = load_data(data_path)
        X, y = preprocess_data(data)
        model, X_test, y_test = train_model(X, y, param_grid)

        # Log model and parameters
        mlflow.log_params(param_grid)
        mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))
        mlflow.log_metric("roc_auc", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
        log_model(model, "model")

        save_model(model, model_save_path)
