"""
Customer Churn Dataset Preprocessing

Loads the Telco Customer Churn dataset, cleans the data,
encodes categorical variables, and prepares features for
machine-learning models.
"""

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"


def load_data() -> pd.DataFrame:
    """Load the raw Telco Customer Churn dataset."""
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean missing values and prepare the target column."""
    df = df.copy()

    # TotalCharges contains blank strings in some records.
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    # Remove rows where the target or important numeric field is missing.
    df = df.dropna(subset=["Churn", "TotalCharges"])

    # Convert target to binary format.
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode categorical features for machine-learning models."""
    df = df.copy()

    # CustomerID is an identifier, not a predictive feature.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Convert categorical columns using one-hot encoding.
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
    """Run the complete preprocessing pipeline."""
    df = load_data()
    df = clean_data(df)
    df = prepare_features(df)

    return df


if __name__ == "__main__":
    processed_df = preprocess_data()

    print("Preprocessing completed successfully.")
    print(f"Rows: {processed_df.shape[0]}")
    print(f"Columns: {processed_df.shape[1]}")
    print("\nFirst 5 rows:")
    print(processed_df.head())