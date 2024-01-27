import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from imblearn.over_sampling import SMOTE
import joblib
import logging
import sys
import os
import mlflow
from mlflow.sklearn import log_model
from sklearn.preprocessing import StandardScaler

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
    Preprocess the data, returning scaled features and labels.
    """
    X = data.drop('class', axis=1)
    y = data['class']

    # Feature scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y


def balance_data(X, y):
    """
    Balance the dataset using SMOTE.
    """
    smote = SMOTE()
    X_balanced, y_balanced = smote.fit_resample(X, y)
    return X_balanced, y_balanced


def train_model(X, y, test_size=0.3, balance=False):
    """
    Train the model and save the test set.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    if balance:
        X_train, y_train = balance_data(X_train, y_train)

    model = RandomForestClassifier(n_estimators=100, max_depth=20, random_state=42)
    model.fit(X_train, y_train)

    return model, X_test, y_test


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

    retrain_flag = os.getenv('RETRAIN_MODEL', 'False')
    if retrain_flag.lower() == 'true':
        logging.info("Retraining model...")
        data_path = os.getenv('DATA_PATH', 'data/creditcard.csv')
        model_save_path = os.getenv('MODEL_SAVE_PATH', 'model/saved_models/model.pkl')

        with mlflow.start_run():
            data = load_data(data_path)
            X, y = preprocess_data(data)
            model, X_test, y_test = train_model(X, y, balance=True)

            # Log model parameters and metrics
            mlflow.log_params(model.get_params())
            mlflow.log_metric("accuracy", accuracy_score(y_test, model.predict(X_test)))
            mlflow.log_metric("roc_auc", roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))
            log_model(model, "model")

            save_model(model, model_save_path)
