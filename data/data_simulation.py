import pandas as pd
import numpy as np
import os

def simulate_data_drift(data, drift_factor, output_path):
    drifted_data = data.copy()

    # Introduce data drift by modifying some features
    drifted_data.iloc[:, 1:10] = data.iloc[:, 1:10] * (1 + np.random.normal(0, drift_factor, size=(len(data), 9)))

    # Ensure the labels remain consistent
    drifted_data.iloc[:, -1] = data.iloc[:, -1]

    # Save the drifted data
    drifted_data.to_csv(output_path, index=False)

if __name__ == "__main__":
    # Load the original dataset
    original_data = pd.read_csv("creditcard.csv")

    # Simulate data drift for each month
    for month in range(1, 13):
        drift_factor = month / 100  # Increase the drift factor with each month
        output_path = f"simulated/creditcard_month_{month}.csv"

        simulate_data_drift(original_data, drift_factor, output_path)