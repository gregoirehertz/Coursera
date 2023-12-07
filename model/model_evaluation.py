import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def evaluate_model(data_path, model_path):
    data = pd.read_csv(data_path)
    X = data.drop('class', axis=1)
    y = data['class']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = joblib.load(model_path)
    predictions = model.predict(X_test)
    
    return classification_report(y_test, predictions)

if __name__ == "__main__":
    report = evaluate_model('data/creditcard.csv', 'model/fraud_detection_model.pkl')
    print(report)
