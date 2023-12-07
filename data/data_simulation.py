import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def simulate_data_drift(original_data, month_offset):
    # Copy the original data
    new_data = original_data.copy()

    # Simulating drift by randomly tweaking numerical columns
    for col in new_data.select_dtypes(include='float').columns:
        if col != 'class':
            drift_factor = 1 + (0.01 * month_offset)  # 1% drift per month
            new_data[col] = new_data[col].apply(lambda x: x * np.random.uniform(1, drift_factor))

    # Modify 'Time' column to reflect new month
    # Assuming each unit in 'Time' is equivalent to a certain time period (e.g., seconds, minutes)
    time_drift_factor = 30 * 24 * 60 * month_offset  # Example: 30 days' worth of minutes
    new_data['Time'] = new_data['Time'] + time_drift_factor

    return new_data


# Usage
if __name__ == "__main__":
    original_data = pd.read_csv('data/creditcard.csv')
    month = 1  # Simulate data for the 1st month
    modified_data = simulate_data_drift(original_data, month)
    modified_data.to_csv(f'data/simulated_data/simulated_data_{month}.csv', index=False)
