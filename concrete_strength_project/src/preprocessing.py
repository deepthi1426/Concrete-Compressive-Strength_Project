"""
Stage 2: Preprocessing
------------------------
Cleans the raw concrete dataset:
- checks/handles missing values (this dataset has none, but we check anyway
  since a real pipeline should never assume clean data)
- caps extreme outliers using the IQR method
- log-transforms the heavily right-skewed 'age' feature
- separates features/target ready for downstream feature engineering
"""

import numpy as np
import pandas as pd


def check_missing(df: pd.DataFrame) -> pd.DataFrame:
    missing = df.isnull().sum()
    print("Missing values per column:\n", missing[missing > 0] if missing.sum() else "None found.")
    # If any missing values existed, impute numeric columns with the median.
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())
    return df


def cap_outliers_iqr(df: pd.DataFrame, columns, factor: float = 1.5) -> pd.DataFrame:
    """Cap (winsorize) outliers outside [Q1 - factor*IQR, Q3 + factor*IQR]."""
    df = df.copy()
    for col in columns:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - factor * iqr, q3 + factor * iqr
        n_out = ((df[col] < lower) | (df[col] > upper)).sum()
        if n_out > 0:
            print(f"  Capping {n_out} outliers in '{col}' to [{lower:.2f}, {upper:.2f}]")
        df[col] = df[col].clip(lower, upper)
    return df


def log_transform_age(df: pd.DataFrame) -> pd.DataFrame:
    """Age (curing days) is heavily right-skewed (mostly 1/7/28 days, some up to 365)."""
    df = df.copy()
    df["age_log"] = np.log1p(df["age"])
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    print("Checking for missing values...")
    df = check_missing(df)

    numeric_cols = [c for c in df.columns if c != "strength"]
    print("Capping outliers (IQR method)...")
    df = cap_outliers_iqr(df, numeric_cols)

    print("Log-transforming skewed 'age' feature...")
    df = log_transform_age(df)

    return df


if __name__ == "__main__":
    from data_acquisition import acquire_data

    raw_df = acquire_data()
    clean_df = preprocess(raw_df)
    print(clean_df.describe())
