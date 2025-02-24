import os
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score

def train_model(input_path='data/featured_data.csv', model_output='models/trained_model.pkl'):
    """
    Trains a RandomForest model and saves it.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"🚨 Feature data file not found at: {input_path}")

    df = pd.read_csv(input_path)

    # Ensure models directory exists
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target'].astype(int)  # Ensure target is categorical

    # Apply SMOTE to balance dataset
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    # Train optimized model
    model = RandomForestClassifier(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X_train, y_train)

    # Save trained model
    model_output_path = os.path.join(model_dir, "trained_model.pkl")
    joblib.dump(model, model_output_path)
    print(f"✅ Model trained and saved at: {model_output_path}")

    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"📊 Model Accuracy: {accuracy:.2f}")

if __name__ == "__main__":
    train_model()
