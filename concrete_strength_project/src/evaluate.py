"""
Stage 5: Evaluation
----------------------
Reports R2, RMSE and MAE for each trained model on a held-out test set,
plus 5-fold cross-validated R2 (important on a dataset this small — a single
train/test split can be misleadingly lucky or unlucky).
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, KFold


def evaluate_model(name, pipe, X_test, y_test, X_full=None, y_full=None) -> dict:
    preds = pipe.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)

    result = {"model": name, "RMSE": round(rmse, 3), "MAE": round(mae, 3), "R2": round(r2, 3)}

    if X_full is not None and y_full is not None:
        cv_scores = cross_val_score(pipe, X_full, y_full, cv=KFold(n_splits=5, shuffle=True, random_state=42), scoring="r2")
        result["CV_R2_mean"] = round(cv_scores.mean(), 3)
        result["CV_R2_std"] = round(cv_scores.std(), 3)

    return result


def evaluate_all(models: dict, X_test, y_test, X_full=None, y_full=None) -> pd.DataFrame:
    rows = [evaluate_model(name, pipe, X_test, y_test, X_full, y_full) for name, pipe in models.items()]
    return pd.DataFrame(rows).sort_values("RMSE")


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from data_acquisition import acquire_data
    from preprocessing import preprocess
    from feature_engineering import engineer_features
    from train_models import build_models, train_all

    df = engineer_features(preprocess(acquire_data()))
    X = df.drop(columns=["strength"])
    y = df["strength"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = train_all(build_models(), X_train, y_train)
    results = evaluate_all(models, X_test, y_test, X, y)
    print(results.to_string(index=False))
