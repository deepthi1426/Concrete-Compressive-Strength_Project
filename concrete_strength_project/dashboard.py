"""
Concrete Compressive Strength Dashboard
------------------------------------------
Combines a live prediction tool with dataset/model insight charts.

Run with:
    streamlit run dashboard.py

Requires the pipeline to have already been run once (python src/pipeline.py),
since this loads the saved model from models/concrete_strength_pipeline.joblib
and the standardized dataset from data/raw/concrete_data.csv.
"""

import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "models", "concrete_strength_pipeline.joblib")
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "concrete_data.csv")

st.set_page_config(page_title="Concrete Strength Dashboard", layout="wide")

# ---------- Load model + data ----------

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_data():
    if not os.path.exists(DATA_PATH):
        return None
    return pd.read_csv(DATA_PATH)


artifact = load_model()
raw_df = load_data()

st.title("🧱 Concrete Compressive Strength Dashboard")
st.caption("End-to-end ML pipeline — predict strength from mix design, and explore the data behind it.")

if artifact is None:
    st.error(
        "No trained model found yet. Run `python src/pipeline.py` first "
        "(from the project root) so this dashboard has a model to load."
    )
    st.stop()

if raw_df is None:
    st.warning("Raw dataset not found at data/raw/concrete_data.csv — insight charts will be limited.")

pipe = artifact["pipeline"]
feature_columns = artifact["feature_columns"]
metrics = artifact.get("metrics", {})

# ---------- Feature engineering helper (mirrors src/feature_engineering.py) ----------

def build_feature_row(inputs: dict) -> pd.DataFrame:
    row = dict(inputs)
    row["age_log"] = np.log1p(row["age"])
    row["water_cement_ratio"] = row["water"] / (row["cement"] or 1e-6)
    row["total_binder"] = row["cement"] + row["slag"] + row["fly_ash"]
    row["agg_ratio"] = row["coarse_agg"] / (row["fine_agg"] or 1e-6)

    bucket = pd.cut(
        [row["age"]], bins=[0, 7, 28, 90, 365],
        labels=["early", "standard", "extended", "long_term"]
    )[0]
    for col in feature_columns:
        if col.startswith("age_bucket_"):
            row[col] = 1 if col == f"age_bucket_{bucket}" else 0

    df = pd.DataFrame([row])
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    return df[feature_columns]


# ---------- Layout: two tabs ----------

tab_predict, tab_insights = st.tabs(["🔮 Predict", "📊 Insights"])

# ===================== PREDICT TAB =====================
with tab_predict:
    st.subheader("Enter a mix design")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        cement = st.number_input("Cement (kg/m³)", 100.0, 600.0, 350.0, step=5.0)
        slag = st.number_input("Blast Furnace Slag (kg/m³)", 0.0, 400.0, 0.0, step=5.0)
    with col2:
        fly_ash = st.number_input("Fly Ash (kg/m³)", 0.0, 250.0, 0.0, step=5.0)
        water = st.number_input("Water (kg/m³)", 100.0, 260.0, 180.0, step=1.0)
    with col3:
        superplasticizer = st.number_input("Superplasticizer (kg/m³)", 0.0, 35.0, 5.0, step=0.5)
        coarse_agg = st.number_input("Coarse Aggregate (kg/m³)", 700.0, 1200.0, 970.0, step=5.0)
    with col4:
        fine_agg = st.number_input("Fine Aggregate (kg/m³)", 500.0, 1000.0, 770.0, step=5.0)
        age = st.number_input("Curing Age (days)", 1, 365, 28, step=1)

    if st.button("Predict Strength", type="primary"):
        inputs = {
            "cement": cement, "slag": slag, "fly_ash": fly_ash, "water": water,
            "superplasticizer": superplasticizer, "coarse_agg": coarse_agg,
            "fine_agg": fine_agg, "age": age,
        }
        X_row = build_feature_row(inputs)
        prediction = pipe.predict(X_row)[0]

        wc_ratio = water / cement if cement else 0

        st.success(f"### Predicted Compressive Strength: {prediction:.2f} MPa")
        m1, m2, m3 = st.columns(3)
        m1.metric("Water-Cement Ratio", f"{wc_ratio:.3f}")
        m2.metric("Total Binder", f"{cement + slag + fly_ash:.0f} kg/m³")
        if metrics:
            m3.metric("Model Test RMSE", f"{metrics.get('RMSE', '—')} MPa")

        st.caption(
            "Lower water-cement ratio generally means higher strength "
            "(Abrams' Law) — try lowering water or raising cement to see the effect."
        )

# ===================== INSIGHTS TAB =====================
with tab_insights:
    if raw_df is None:
        st.info("Add data/raw/concrete_data.csv to unlock dataset insight charts.")
    else:
        st.subheader("Dataset overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Samples", len(raw_df))
        c2.metric("Avg. Strength", f"{raw_df['strength'].mean():.1f} MPa")
        c3.metric("Strength Range", f"{raw_df['strength'].min():.0f}–{raw_df['strength'].max():.0f} MPa")

        left, right = st.columns(2)

        with left:
            st.markdown("**Strength distribution**")
            fig, ax = plt.subplots(figsize=(5, 3.5))
            sns.histplot(raw_df["strength"], kde=True, ax=ax, color="#4C72B0")
            ax.set_xlabel("Compressive Strength (MPa)")
            st.pyplot(fig)

        with right:
            st.markdown("**Correlation with strength**")
            corr = raw_df.corr(numeric_only=True)["strength"].drop("strength").sort_values()
            fig, ax = plt.subplots(figsize=(5, 3.5))
            corr.plot(kind="barh", ax=ax, color="#55A868")
            ax.set_xlabel("Correlation with strength")
            st.pyplot(fig)

        st.markdown("**Water-Cement ratio vs Strength**")
        fig, ax = plt.subplots(figsize=(8, 3.5))
        wc = raw_df["water"] / raw_df["cement"]
        sns.scatterplot(x=wc, y=raw_df["strength"], hue=raw_df["age"], palette="viridis", ax=ax, legend=False)
        ax.set_xlabel("Water-Cement Ratio")
        ax.set_ylabel("Strength (MPa)")
        st.pyplot(fig)

        st.subheader("Model performance (on held-out test set)")
        if metrics:
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric("RMSE", f"{metrics.get('RMSE', '—')} MPa")
            mcol2.metric("MAE", f"{metrics.get('MAE', '—')} MPa")
            mcol3.metric("R²", metrics.get("R2", "—"))
        else:
            st.info("No stored metrics found in the model file.")

        # Feature importance, if the underlying model supports it
        try:
            model_step = pipe.named_steps.get("model")
            if hasattr(model_step, "feature_importances_"):
                st.markdown("**Feature importance**")
                importances = pd.Series(model_step.feature_importances_, index=feature_columns).sort_values()
                fig, ax = plt.subplots(figsize=(8, 4))
                importances.plot(kind="barh", ax=ax, color="#C44E52")
                st.pyplot(fig)
        except Exception:
            pass