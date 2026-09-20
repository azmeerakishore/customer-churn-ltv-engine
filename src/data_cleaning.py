import pandas as pd


def clean_data(df):
    """
    Clean the customer churn dataset.
    """

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    # Fill missing TotalCharges with 0
    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Reset index
    df = df.reset_index(drop=True)

    return df