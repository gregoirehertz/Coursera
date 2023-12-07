import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import joblib
import os

def train_model(data_path, model_save_path):
    """
    Train a fraud detection model.
    Args:
    - data_path: Path to the training data.
    - model_save_path: Path to save the trained model.
    """
    # Load the dataset
    data = pd.read_csv(data_path)
    X = data.drop('class', axis=1)
    y = data['class']

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Initialize and train the Random Forest Classifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Evaluate the model
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))

    # Save the trained model
    joblib.dump(model, model_save_path)

# Example usage
if __name__ == "__main__":
    data_dir = 'data/simulated_data'
    model_dir = 'model/saved_models'
    
    # Train and save a model for each month's simulated data
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            month = file.split('_')[-1].split('.')[0]
            print(f"Training model for month {month}...")
            train_model(f"{data_dir}/{file}", f"{model_dir}/model_{month}.pkl")
