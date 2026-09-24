import pandas as pd


def clean_data(df):
    """
    Clean the customer churn dataset.

    Steps:
    1. Create a copy of the dataset.
    2. Convert TotalCharges to numeric.
    3. Handle missing TotalCharges values.
    4. Remove duplicate records.
    5. Reset the index.
    """

    # Create a copy so the original dataset is not modified
    df = df.copy()

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Replace missing TotalCharges values with 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Remove duplicate records
    df = df.drop_duplicates()

    # Reset index after cleaning
    df = df.reset_index(drop=True)

    return df