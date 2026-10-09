"""
Stage 4: Model Training
-------------------------
Trains three models of increasing complexity so the report can show a clear
progression and justify the final choice:
  1. Linear Regression   (interpretable baseline)
  2. Random Forest       (captures non-linearity, robust to scale)
  3. XGBoost             (usually best performer on this dataset)

All models are wrapped in an sklearn Pipeline together with a StandardScaler,
so preprocessing travels with the model when it's persisted later
(this avoids the classic bug of forgetting to save/reload the scaler).
"""

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


def build_models() -> dict:
    models = {
        "linear_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression()),
        ]),
        "random_forest": Pipeline([
            ("scaler", StandardScaler()),
            ("model", RandomForestRegressor(n_estimators=300, random_state=42)),
        ]),
        "xgboost": Pipeline([
            ("scaler", StandardScaler()),
            ("model", XGBRegressor(
                n_estimators=400,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
            )),
        ]),
    }
    return models


def train_all(models: dict, X_train, y_train) -> dict:
    for name, pipe in models.items():
        print(f"Training {name}...")
        pipe.fit(X_train, y_train)
    return models


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split

    from data_acquisition import acquire_data
    from preprocessing import preprocess
    from feature_engineering import engineer_features

    df = engineer_features(preprocess(acquire_data()))
    X = df.drop(columns=["strength"])
    y = df["strength"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = build_models()
    models = train_all(models, X_train, y_train)
    print("All models trained.")
