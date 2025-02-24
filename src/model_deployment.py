import os
import joblib
import pandas as pd
from flask import Flask, request, jsonify

# Define paths
MODEL_PATH = "models/trained_model.pkl"

# Ensure the trained model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"🚨 Model file not found at: {MODEL_PATH}")

# Load trained model
model = joblib.load(MODEL_PATH)
print(f"✅ Loaded model from {MODEL_PATH}")

# Initialize Flask App
app = Flask(__name__)

@app.route("/", methods=["GET"])
def home():
    return "🏦 Welcome to the Loan Default Prediction API! Use /predict to make predictions."

@app.route("/predict", methods=["POST"])
def predict():
    """
    API endpoint to predict loan default probability.
    
    Example Request JSON:
    {
        "Credit_Score": 650,
        "Loan_Amount": 50000,
        "Account_Balance": 200000,
        "Interest_Rate": 5.5,
        "Loan_to_Balance_Ratio": 0.25,
        "High_Credit_Risk": 0
    }
    """
    try:
        # Get JSON input
        data = request.get_json()

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Ensure feature order matches training
        expected_features = ['Credit_Score', 'Loan_Amount', 'Account_Balance', 
                             'Interest_Rate', 'Loan_to_Balance_Ratio', 'High_Credit_Risk']
        df = df[expected_features]

        # Make prediction
        prediction = model.predict(df)

        # Return response
        return jsonify({"Default_Prediction": int(prediction[0])})
    
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=True)
