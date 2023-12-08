import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
import joblib
import logging
import sys

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def load_data(data_path):
    try:
        data = pd.read_csv(data_path)
        logging.info(f"Data loaded successfully from {data_path}")
        return data
    except Exception as e:
        logging.error(f"Error loading data: {e}")
        sys.exit(1)


def preprocess_data(data):
    X = data.drop('class', axis=1)
    y = data['class']
    return X, y


def evaluate_model(model_path, X_test, y_test):
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        logging.info(f"Model Evaluation Report:\n{report}")
        logging.info(f"Model Accuracy: {accuracy}, ROC-AUC: {roc_auc}")
    except Exception as e:
        logging.error(f"Error in model evaluation: {e}")
        sys.exit(1)


if __name__ == '__main__':
    data_path = 'data/creditcard.csv'
    model_path = 'model/saved_models/model.pkl'

    data = load_data(data_path)
    X, y = preprocess_data(data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    evaluate_model(model_path, X_test, y_test)
