import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def simulate_data_drift(original_data, month_offset):
    """
    Simulate data drift by modifying the original dataset.
    Args:
    - original_data: DataFrame with the original data.
    - month_offset: Integer, number of months to simulate the drift for.
    """
    # Copy the original data
    new_data = original_data.copy()

    # Simulating drift by randomly tweaking numerical columns
    for col in new_data.select_dtypes(include=np.number).columns:
        if col != 'class':
            drift_factor = 1 + (0.01 * month_offset)  # 1% drift per month
            new_data[col] = new_data[col].apply(lambda x: x * np.random.uniform(1, drift_factor))

    # Update 'Time' column to reflect new month
    new_data['Time'] = new_data['Time'].apply(lambda x: x + timedelta(days=30 * month_offset))

    return new_data

# Usage
if __name__ == "__main__":
    original_data = pd.read_csv('data/creditcard.csv')
    month = 1  # Simulate data for the 1st month
    modified_data = simulate_data_drift(original_data, month)
    modified_data.to_csv(f'data/simulated_data/simulated_data_{month}.csv', index=False)
