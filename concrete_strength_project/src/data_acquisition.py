"""
Stage 1: Data Acquisition
--------------------------
Loads the Concrete Compressive Strength dataset from a Kaggle CSV download
and saves a standardized raw copy to data/raw/.

Dataset (Kaggle mirror): https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set
Original source: UCI ML Repository, Yeh, I. (1998). CC BY 4.0.

HOW TO GET THE CSV:
  1. Go to the Kaggle link above and click Download.
  2. You'll get a file usually named "Concrete_Data_Yeh.csv" or "concrete_data.csv".
  3. Place it in this project's data/raw/ folder, OR in Colab, upload it via
     the folder icon in the left sidebar.
  4. Update KAGGLE_CSV_PATH below to point to wherever you put it.
"""

import os
import pandas as pd

# ---- EDIT THIS: path to the CSV you downloaded from Kaggle ----
KAGGLE_CSV_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "Concrete_Data_Yeh.csv")

RAW_DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "raw", "concrete_data.csv")

# Kaggle's column names vary slightly by mirror/version -- this map covers the
# common variants and standardizes everything to short, code-friendly names.
COLUMN_MAP = {
    "cement": "cement",
    "Cement": "cement",
    "Cement (component 1)(kg in a m^3 mixture)": "cement",

    "slag": "slag",
    "blast_furnace_slag": "slag",
    "Blast Furnace Slag": "slag",
    "Blast Furnace Slag (component 2)(kg in a m^3 mixture)": "slag",

    "flyash": "fly_ash",
    "fly_ash": "fly_ash",
    "Fly Ash": "fly_ash",
    "Fly Ash (component 3)(kg in a m^3 mixture)": "fly_ash",

    "water": "water",
    "Water": "water",
    "Water  (component 4)(kg in a m^3 mixture)": "water",

    "superplasticizer": "superplasticizer",
    "Superplasticizer": "superplasticizer",
    "Superplasticizer (component 5)(kg in a m^3 mixture)": "superplasticizer",

    "coarseaggregate": "coarse_agg",
    "coarse_aggregate": "coarse_agg",
    "coarse_agg": "coarse_agg",
    "Coarse Aggregate": "coarse_agg",
    "Coarse Aggregate  (component 6)(kg in a m^3 mixture)": "coarse_agg",

    "fineaggregate": "fine_agg",
    "fine_aggregate": "fine_agg",
    "fine_agg": "fine_agg",
    "Fine Aggregate": "fine_agg",
    "Fine Aggregate (component 7)(kg in a m^3 mixture)": "fine_agg",

    "age": "age",
    "Age": "age",
    "Age (day)": "age",

    "csMPa": "strength",
    "strength": "strength",
    "concrete_compressive_strength": "strength",
    "Concrete compressive strength": "strength",
    "Concrete compressive strength(MPa, megapascals) ": "strength",
}


def acquire_data() -> pd.DataFrame:
    """
    Loads the dataset from the Kaggle CSV. Caches a standardized copy to
    data/raw/concrete_data.csv so later stages always read the clean version.
    """
    if os.path.exists(RAW_DATA_PATH):
        print(f"Standardized raw data already present at {RAW_DATA_PATH}, loading from disk.")
        return pd.read_csv(RAW_DATA_PATH)

    if not os.path.exists(KAGGLE_CSV_PATH):
        raise FileNotFoundError(
            f"Could not find '{KAGGLE_CSV_PATH}'.\n"
            "Download the dataset from "
            "https://www.kaggle.com/datasets/elikplim/concrete-compressive-strength-data-set, "
            "place the CSV in data/raw/, and update KAGGLE_CSV_PATH in this file "
            "if the filename differs."
        )

    print(f"Loading dataset from {KAGGLE_CSV_PATH} ...")
    df = pd.read_csv(KAGGLE_CSV_PATH)

    # Standardize column names (strip whitespace first, since some Kaggle
    # mirrors have trailing spaces in headers)
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={c: COLUMN_MAP.get(c, c) for c in df.columns})

    missing_expected = set(["cement", "slag", "fly_ash", "water", "superplasticizer",
                             "coarse_agg", "fine_agg", "age", "strength"]) - set(df.columns)
    if missing_expected:
        raise ValueError(
            f"After renaming, these expected columns are still missing: {missing_expected}. "
            f"Actual columns found: {list(df.columns)}. "
            "Add the correct mapping to COLUMN_MAP in this file."
        )

    os.makedirs(os.path.dirname(RAW_DATA_PATH), exist_ok=True)
    df.to_csv(RAW_DATA_PATH, index=False)
    print(f"Saved standardized dataset ({df.shape[0]} rows, {df.shape[1]} cols) to {RAW_DATA_PATH}")
    return df


if __name__ == "__main__":
    df = acquire_data()
    print(df.head())
    print(df.describe())
