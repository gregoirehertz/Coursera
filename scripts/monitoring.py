import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def monitor_model_performance(logs_path):
    logs = pd.read_csv(logs_path)
    accuracy = accuracy_score(logs['actual'], logs['predicted'])
    precision = precision_score(logs['actual'], logs['predicted'])
    recall = recall_score(logs['actual'], logs['predicted'])
    f1 = f1_score(logs['actual'], logs['predicted'])

    return accuracy, precision, recall, f1

if __name__ == "__main__":
    performance_metrics = monitor_model_performance('logs/prediction_logs.csv')
    print(performance_metrics)
