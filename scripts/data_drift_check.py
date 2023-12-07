import pandas as pd
from sklearn.metrics import mean_absolute_error

def check_data_drift(original_data_path, new_data_path):
    """
    Check for data drift between the original dataset and new data.
    Args:
    - original_data_path: Path to the original dataset.
    - new_data_path: Path to the new dataset.
    """
    original_data = pd.read_csv(original_data_path)
    new_data = pd.read_csv(new_data_path)

    # Simple drift check based on mean absolute error
    mae = mean_absolute_error(original_data, new_data)
    print(f"Mean Absolute Error (Drift Indicator): {mae}")
    return mae > 0.01  # Threshold for drift

# Example usage
if __name__ == "__main__":
    drift_detected = check_data_drift('data/creditcard.csv', 'data/simulated_data/simulated_data_1.csv')
    if drift_detected:
        print("Data drift detected.")
    else:
        print("No significant drift detected.")
