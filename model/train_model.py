import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
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


def save_data(X, y, path):
    joblib.dump((X, y), path)


def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = RandomForestClassifier(random_state=42)

    # Hyperparameter tuning
    param_grid = {'n_estimators': [100, 200], 'max_depth': [10, 20]}
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_
    logging.info(f"Best model parameters: {grid_search.best_params_}")

    # Evaluation
    y_pred = best_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    logging.info(f"Model Accuracy: {accuracy}, ROC-AUC: {roc_auc}")

    save_data(X_test, y_test, 'model/saved_models/Xy_test.pkl')

    return best_model


def save_model(model, model_save_path):
    try:
        joblib.dump(model, model_save_path)
        logging.info(f"Model saved to {model_save_path}")
    except Exception as e:
        logging.error(f"Error saving model: {e}")


# Inside the train_model function, add after the train/test split:


if __name__ == '__main__':
    data_path = 'data/creditcard.csv'
    model_save_path = 'model/saved_models/model.pkl'

    data = load_data(data_path)
    X, y = preprocess_data(data)
    model = train_model(X, y)
    save_model(model, model_save_path)
