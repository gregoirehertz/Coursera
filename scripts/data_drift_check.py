import pandas as pd
import json
from evidently.dashboard import Dashboard
from evidently.dashboard.tabs import DataDriftTab


def detect_data_drift(reference_data_path, current_data_path, report_path):
    """
    Detects data drift by comparing reference data with current data.

    Args:
        reference_data_path (str): Path to the reference dataset (original training data).
        current_data_path (str): Path to the current dataset to be compared.
        report_path (str): Path to save the data drift report.

    Returns:
        bool: True if data drift is detected, False otherwise.
    """
    # Load datasets
    reference_data = pd.read_csv(reference_data_path)
    current_data = pd.read_csv(current_data_path)

    # Create a data drift report
    data_drift_dashboard = Dashboard(tabs=[DataDriftTab()])
    data_drift_dashboard.calculate(reference_data, current_data)

    # Save the report
    report = data_drift_dashboard.save(output_file=report_path)

    # Analyzing the report to detect drift
    with open(report_path) as f:
        report_json = json.load(f)
        is_drift_found = report_json['data_drift']['data']['metrics']['dataset_drift']

    return is_drift_found


if __name__ == "__main__":
    drift_detected = detect_data_drift(
        'data/creditcard.csv',
        'data/simulated_data/modified_data_month_1.csv',
        'data_drift_report.html'
    )
    if drift_detected:
        print("Data drift detected.")
    else:
        print("No significant data drift found.")
