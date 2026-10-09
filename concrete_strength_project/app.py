"""
Deployment: FastAPI service that loads the persisted pipeline and serves
concrete strength predictions from raw mix ingredient inputs.

Run:
    uvicorn app:app --reload
Then open http://127.0.0.1:8000/docs to test it interactively.
"""

import os

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "concrete_strength_pipeline.joblib")

app = FastAPI(title="Concrete Compressive Strength Predictor")

_artifact = None


def get_artifact():
    global _artifact
    if _artifact is None:
        if not os.path.exists(MODEL_PATH):
            raise HTTPException(
                status_code=500,
                detail="Model not found. Run `python src/pipeline.py` first to train and save it.",
            )
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


class MixInput(BaseModel):
    cement: float = Field(..., description="kg per m3 mixture")
    slag: float = Field(..., description="Blast furnace slag, kg per m3")
    fly_ash: float = Field(..., description="kg per m3")
    water: float = Field(..., description="kg per m3")
    superplasticizer: float = Field(..., description="kg per m3")
    coarse_agg: float = Field(..., description="Coarse aggregate, kg per m3")
    fine_agg: float = Field(..., description="Fine aggregate, kg per m3")
    age: int = Field(..., description="Curing age in days (1-365)")


def build_feature_row(payload: MixInput, feature_columns: list) -> pd.DataFrame:
    """Re-creates the exact same feature engineering used in training."""
    row = payload.dict()
    row["age_log"] = np.log1p(row["age"])
    row["water_cement_ratio"] = row["water"] / (row["cement"] or 1e-6)
    row["total_binder"] = row["cement"] + row["slag"] + row["fly_ash"]
    row["agg_ratio"] = row["coarse_agg"] / (row["fine_agg"] or 1e-6)

    # Recreate the one-hot age_bucket columns, defaulting missing ones to 0
    bucket = pd.cut([row["age"]], bins=[0, 7, 28, 90, 365], labels=["early", "standard", "extended", "long_term"])[0]
    for col in feature_columns:
        if col.startswith("age_bucket_"):
            row[col] = 1 if col == f"age_bucket_{bucket}" else 0

    df = pd.DataFrame([row])
    # Ensure column order matches training exactly; fill any missing engineered cols with 0
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


@app.get("/")
def root():
    return {"message": "Concrete Compressive Strength Predictor. See /docs for usage."}


@app.post("/predict")
def predict(payload: MixInput):
    artifact = get_artifact()
    pipe = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]

    X = build_feature_row(payload, feature_columns)
    prediction = pipe.predict(X)[0]

    return {
        "predicted_compressive_strength_mpa": round(float(prediction), 2),
        "model_metrics_on_test_set": artifact.get("metrics", {}),
    }
