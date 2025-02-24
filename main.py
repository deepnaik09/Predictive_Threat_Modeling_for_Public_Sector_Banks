import os
import argparse
from src.data_generation import generate_data
from src.data_preprocessing import preprocess_data
from src.feature_engineering import feature_engineering
from src.model_training import train_model
from src.model_evaluation import evaluate_model
from src.model_deployment import app  # Import Flask app

def run_pipeline():
    """Runs the complete AI pipeline."""
    generate_data()
    preprocess_data()
    feature_engineering()
    train_model()
    evaluate_model()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run AI pipeline.")
    parser.add_argument("--step", type=str, choices=["generate", "preprocess", "feature", "train", "evaluate", "deploy", "all"], default="all")
    args = parser.parse_args()

    if args.step == "generate":
        generate_data()
    elif args.step == "preprocess":
        preprocess_data()
    elif args.step == "feature":
        feature_engineering()
    elif args.step == "train":
        train_model()
    elif args.step == "evaluate":
        evaluate_model()
    elif args.step == "deploy":
        print("🚀 Starting API server at http://127.0.0.1:5000/")
        app.run(debug=True)  # Start the Flask API
    else:
        run_pipeline()
