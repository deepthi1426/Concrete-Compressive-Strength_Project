# Concrete Compressive Strength Prediction — End-to-End ML Project

Predicts the compressive strength (MPa) of concrete from its mix ingredients and
curing age, using the Concrete Compressive Strength dataset (1,030 samples,
100% numeric features — originally UCI ML Repository, Yeh 1998).

## Step 0: Get the dataset (do this first)

1. Go to **https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set**
2. Click **Download**
3. Put the CSV file inside `data/raw/` in this project folder
4. Open `src/data_acquisition.py` and check that `KAGGLE_CSV_PATH` matches
   the exact filename you downloaded (it's usually `Concrete_Data_Yeh.csv`)

That's it — everything else runs automatically from there.

This project covers the full ML lifecycle:

1. Data acquisition
2. Preprocessing
3. Feature engineering
4. Model training
5. Evaluation
6. Hyperparameter optimization
7. Deployment-ready model persistence (+ a FastAPI serving app)

## Project structure

```
concrete_strength_project/
├── README.md
├── requirements.txt
├── data/
│   ├── raw/                # untouched data straight from the source
│   └── processed/          # cleaned, feature-engineered data
├── src/
│   ├── data_acquisition.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── train_models.py
│   ├── evaluate.py
│   ├── optimize.py
│   └── pipeline.py         # runs every stage in order
├── models/                 # saved model + pipeline artifacts (joblib)
├── app.py                  # FastAPI deployment endpoint
└── notebooks/              # optional: for EDA / report screenshots
```

## How to run

```bash
pip install -r requirements.txt

# Make sure you've completed Step 0 above (CSV placed in data/raw/) first

# Run the whole pipeline end-to-end
python src/pipeline.py

# Then serve the trained model
uvicorn app:app --reload
```

Open http://127.0.0.1:8000/docs to test predictions interactively.

## Running in Google Colab (no local install needed)

1. Go to colab.research.google.com → New notebook
2. First cell: `!pip install xgboost optuna -q`
3. Upload the Kaggle CSV using the folder icon on the left sidebar
4. Copy each file in `src/` into its own cell, in this order, and run each:
   `data_acquisition.py` → `preprocessing.py` → `feature_engineering.py` →
   `train_models.py` → `evaluate.py` → `optimize.py`
5. The final cell will print the evaluation table and save the model file,
   which you can download from the folder sidebar.

## Why this dataset satisfies the brief

- **Real-world**: genuine lab-tested concrete mix data (I-Cheng Yeh, 1998),
  used in real civil engineering research.
- **Fully numeric**: 8 numeric inputs (cement, slag, fly ash, water,
  superplasticizer, coarse aggregate, fine aggregate, age) → 1 numeric
  target (compressive strength in MPa). No categorical encoding debates.
- **Genuinely needs preprocessing**: skewed age distribution, outliers in
  ingredient quantities, features on very different numeric scales.
- **License**: CC BY 4.0 — free to use with attribution.
  Source: https://archive.ics.uci.edu/dataset/165/concrete+compressive+strength
