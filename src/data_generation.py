import os
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification

def generate_data(n_samples=5000, n_features=20, imbalance_ratio=0.3, random_state=42):
    """
    Generates synthetic data for classification tasks.
    """
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[imbalance_ratio, 1 - imbalance_ratio],
        random_state=random_state
    )
    
    columns = [f'feature_{i}' for i in range(n_features)]
    df = pd.DataFrame(X, columns=columns)
    df['target'] = y

    # Ensure data directory exists
    os.makedirs("data", exist_ok=True)

    # Save dataset
    file_path = os.path.join("data", "generated_data.csv")
    df.to_csv(file_path, index=False)
    print(f"✅ Synthetic Data Generated Successfully! 📂 Saved at: {file_path}")

if __name__ == "__main__":
    generate_data()
