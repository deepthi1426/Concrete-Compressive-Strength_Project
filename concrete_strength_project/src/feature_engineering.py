"""
Stage 3: Feature Engineering
------------------------------
Adds domain-driven features on top of the raw ingredient quantities:

- water_cement_ratio : classic civil-engineering heuristic — lower ratio
  generally means stronger concrete. This is a well-known real formula,
  not an arbitrary derived feature, and is worth explicitly calling out
  in your report as domain knowledge applied to feature engineering.
- total_binder       : cement + slag + fly ash (all three act as the
  "glue" in the mix)
- agg_ratio          : coarse aggregate to fine aggregate ratio
- age_bucket         : early / mid / late curing stage, since strength
  development is non-linear over time
"""

import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Avoid divide-by-zero on the rare mixes with 0 water (shouldn't happen, but be safe)
    df["water_cement_ratio"] = df["water"] / df["cement"].replace(0, 1e-6)

    df["total_binder"] = df["cement"] + df["slag"] + df["fly_ash"]

    df["agg_ratio"] = df["coarse_agg"] / df["fine_agg"].replace(0, 1e-6)

    df["age_bucket"] = pd.cut(
        df["age"],
        bins=[0, 7, 28, 90, 365],
        labels=["early", "standard", "extended", "long_term"],
    )
    # One-hot encode the single categorical we created (keeps everything numeric downstream)
    df = pd.get_dummies(df, columns=["age_bucket"], drop_first=True)

    return df


if __name__ == "__main__":
    from data_acquisition import acquire_data
    from preprocessing import preprocess

    raw_df = acquire_data()
    clean_df = preprocess(raw_df)
    feat_df = engineer_features(clean_df)
    print(feat_df.head())
    print("\nColumns after feature engineering:", list(feat_df.columns))
