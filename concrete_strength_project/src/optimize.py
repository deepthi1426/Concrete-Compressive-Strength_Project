"""
Stage 6: Optimization
------------------------
Uses Optuna to tune the XGBoost model (the best baseline performer) via
5-fold cross-validated RMSE minimization. Returns the best-fit pipeline,
ready to be persisted for deployment.
"""

import numpy as np
import optuna
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial, X, y):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 600),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10.0, log=True),
    }

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(random_state=42, **params)),
    ])

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipe, X, y, cv=kf, scoring="neg_root_mean_squared_error")
    return -scores.mean()


def optimize_xgboost(X, y, n_trials: int = 50):
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, X, y), n_trials=n_trials)

    print("Best RMSE:", study.best_value)
    print("Best params:", study.best_params)

    best_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", XGBRegressor(random_state=42, **study.best_params)),
    ])
    best_pipe.fit(X, y)
    return best_pipe, study


if __name__ == "__main__":
    from data_acquisition import acquire_data
    from preprocessing import preprocess
    from feature_engineering import engineer_features

    df = engineer_features(preprocess(acquire_data()))
    X = df.drop(columns=["strength"])
    y = df["strength"]

    best_pipe, study = optimize_xgboost(X, y, n_trials=50)
