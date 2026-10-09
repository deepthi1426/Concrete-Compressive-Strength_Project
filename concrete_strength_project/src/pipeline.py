"""
Full end-to-end pipeline: acquisition -> preprocessing -> feature engineering
-> training -> evaluation -> optimization -> deployment-ready persistence.

Run this single file to reproduce the entire project:
    python src/pipeline.py
"""

import os

import joblib
from sklearn.model_selection import train_test_split

from data_acquisition import acquire_data
from evaluate import evaluate_all
from feature_engineering import engineer_features
from optimize import optimize_xgboost
from preprocessing import preprocess
from train_models import build_models, train_all

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")


def run_pipeline():
    print("\n=== STAGE 1: Data Acquisition ===")
    raw_df = acquire_data()

    print("\n=== STAGE 2: Preprocessing ===")
    clean_df = preprocess(raw_df)

    print("\n=== STAGE 3: Feature Engineering ===")
    feat_df = engineer_features(clean_df)

    os.makedirs(PROCESSED_DIR, exist_ok=True)
    processed_path = os.path.join(PROCESSED_DIR, "concrete_data_processed.csv")
    feat_df.to_csv(processed_path, index=False)
    print(f"Saved processed dataset to {processed_path}")

    X = feat_df.drop(columns=["strength"])
    y = feat_df["strength"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("\n=== STAGE 4: Model Training (baselines) ===")
    models = train_all(build_models(), X_train, y_train)

    print("\n=== STAGE 5: Evaluation (baselines) ===")
    baseline_results = evaluate_all(models, X_test, y_test, X, y)
    print(baseline_results.to_string(index=False))

    print("\n=== STAGE 6: Optimization (tuning XGBoost) ===")
    best_pipe, study = optimize_xgboost(X_train, y_train, n_trials=50)

    print("\n=== STAGE 5b: Evaluation (optimized model) ===")
    final_results = evaluate_all({"xgboost_tuned": best_pipe}, X_test, y_test, X, y)
    print(final_results.to_string(index=False))

    print("\n=== STAGE 7: Deployment-ready persistence ===")
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "concrete_strength_pipeline.joblib")
    joblib.dump({
        "pipeline": best_pipe,
        "feature_columns": list(X.columns),
        "metrics": final_results.to_dict(orient="records")[0],
    }, model_path)
    print(f"Saved deployment-ready pipeline to {model_path}")

    return best_pipe, final_results


if __name__ == "__main__":
    run_pipeline()