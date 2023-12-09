import pandas as pd
import numpy as np


def modify_data(df, month):
    """
    Modifies the dataset to simulate data drift.

    Args:
        df (DataFrame): The original dataset.
        month (int): The month number used to simulate drift.

    Returns:
        DataFrame: The modified dataset.
    """
    # Example modification: Add a slight trend based on the month
    trend_factor = month * 0.01
    for col in df.select_dtypes(include=np.number).columns:
        if col != 'Class':  # Assuming 'Class' is the label column
            df[col] = df[col] * (1 + np.random.normal(0, trend_factor, len(df)))
    return df


def simulate_monthly_data(input_file, output_dir, num_months=12):
    """
    Simulates monthly data drift. Take a sample dataset and modify it to simulate data drift.

    Args:
        input_file (str): Path to the original dataset.
        output_dir (str): Directory to save the modified datasets.
        num_months (int): Number of months to simulate.
    """
    original_data = pd.read_csv(input_file)
    # take a sample of the original data
    original_data = original_data.sample(frac=0.2, random_state=42)

    for month in range(1, num_months + 1):
        modified_data = modify_data(original_data.copy(), month)
        modified_data.to_csv(f'{output_dir}/modified_data_month_{month}.csv', index=False)


if __name__ == "__main__":
    simulate_monthly_data('data/creditcard.csv', 'data/simulated_data', num_months=12)
