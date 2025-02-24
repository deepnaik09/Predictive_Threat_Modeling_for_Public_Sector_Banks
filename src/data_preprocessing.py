import os
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

def preprocess_data(input_path='data/generated_data.csv', output_path='data/processed_data.csv'):
    """
    Cleans and normalizes dataset.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"🚨 Data file not found at: {input_path}")

    df = pd.read_csv(input_path)

    # Handle missing values
    imputer = SimpleImputer(strategy='median')
    df[df.columns] = imputer.fit_transform(df)

    # Feature Scaling
    scaler = StandardScaler()
    df[df.columns] = scaler.fit_transform(df)

    # Save processed data
    os.makedirs("data", exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"✅ Processed data saved to {output_path}")

if __name__ == "__main__":
    preprocess_data()
