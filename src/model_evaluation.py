import os
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score

def evaluate_model(data_path='data/featured_data.csv', model_path='models/trained_model.pkl'):
    """
    Evaluates the trained model.
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"🚨 Model file not found at: {model_path}")

    model = joblib.load(model_path)
    print(f"✅ Loaded model from {model_path}")

    if not os.path.exists(data_path):
        raise FileNotFoundError(f"🚨 Evaluation data file not found at: {data_path}")

    df = pd.read_csv(data_path)
    X = df.drop(columns=['target'])
    y = df['target'].astype(int)

    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)
    print(f"📊 Evaluation Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    evaluate_model()
