import os
import pandas as pd

def feature_engineering(input_path='data/processed_data.csv', output_path='data/featured_data.csv'):
    """
    Creates new features for better model performance.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"🚨 Processed data file not found at: {input_path}")

    df = pd.read_csv(input_path)

    # Feature: Sum of all feature values (as an example)
    df["feature_sum"] = df.drop(columns=['target']).sum(axis=1)

    # Save enhanced dataset
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Feature Engineering Completed! Saved at {output_path}")

if __name__ == "__main__":
    feature_engineering()
