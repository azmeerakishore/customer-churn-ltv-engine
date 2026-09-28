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


def validate_data(df):
    """
    Validate the customer churn dataset before processing.
    """

    # Required columns
    required_columns = [
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn"
    ]

    # Check for missing required columns
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        return False

    # Check whether dataset is empty
    if df.empty:
        return False

    # Check tenure values
    if (df["tenure"] < 0).any():
        return False

    # Check MonthlyCharges values
    if (df["MonthlyCharges"] < 0).any():
        return False

    # Check SeniorCitizen values
    if not df["SeniorCitizen"].isin([0, 1]).all():
        return False

    return True
def validate_input(data):
    """
    Validate incoming customer data.
    """

    required_fields = [
        "customerID",
        "tenure",
        "MonthlyCharges",
        "SeniorCitizen"
    ]

    # Check required fields
    for field in required_fields:
        if field not in data:
            return False

    # Check missing values
    for field in required_fields:
        if pd.isna(data[field]):
            return False

    # Check valid values
    if data["tenure"] < 0:
        return False

    if data["MonthlyCharges"] < 0:
        return False

    if data["SeniorCitizen"] not in [0, 1]:
        return False

    return True