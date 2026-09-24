"""
Customer Churn Dataset Preprocessing

Loads the Telco Customer Churn dataset, applies the canonical
data-cleaning logic, converts the churn target to binary format,
and prepares an encoded dataset for machine-learning models.
"""

from pathlib import Path

import pandas as pd

from src.data_cleaning import clean_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_data() -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH)


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the Churn target from Yes/No to 1/0."""
    df = df.copy()

    if "Churn" not in df.columns:
        raise ValueError("Expected 'Churn' column was not found.")

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    if df["Churn"].isna().any():
        raise ValueError("Unexpected values found in Churn column.")

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features for machine-learning models."""
    df = df.copy()

    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    categorical_columns = df.select_dtypes(
        include=["object", "str"]
    ).columns

    df = pd.get_dummies(
        df,
        columns=categorical_columns,
        drop_first=True,
        dtype=int,
    )

    return df


def preprocess_data() -> pd.DataFrame:
    """Run the complete canonical preprocessing pipeline."""
    df = load_data()
    df = clean_data(df)
    df = prepare_target(df)
    df = prepare_features(df)

    return df


if __name__ == "__main__":
    processed_df = preprocess_data()

    print("Preprocessing completed successfully.")
    print(f"Rows: {processed_df.shape[0]}")
    print(f"Columns: {processed_df.shape[1]}")

    print("\nTarget distribution:")
    print(processed_df["Churn"].value_counts())

    print("\nFirst 5 rows:")
    print(processed_df.head())