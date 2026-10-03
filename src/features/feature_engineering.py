import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def create_features(df):
    """Create features for churn prediction."""

    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df["TotalCharges"] = df["TotalCharges"].fillna(0)

    df["AverageMonthlyRevenue"] = (
        df["TotalCharges"] /
        df["tenure"].replace(0, 1)
    )

    df["IsLongTermContract"] = (
        df["Contract"] != "Month-to-month"
    ).astype(int)

    df["IsNewCustomer"] = (
        df["tenure"] <= 12
    ).astype(int)

    df["HighMonthlyCharges"] = (
        df["MonthlyCharges"] > 70
    ).astype(int)

    return df


def create_train_test_data(df):
    """Prepare encoded train and test data."""

    df = create_features(df)

    X = df.drop(columns=["customerID", "Churn"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    categorical = X_train.select_dtypes(
        include=["object"]
    ).columns

    encoder = OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    )

    train_encoded = encoder.fit_transform(
        X_train[categorical]
    )

    test_encoded = encoder.transform(
        X_test[categorical]
    )

    train_numeric = X_train.drop(
        columns=categorical
    ).reset_index(drop=True)

    test_numeric = X_test.drop(
        columns=categorical
    ).reset_index(drop=True)

    X_train = pd.concat(
        [
            train_numeric,
            pd.DataFrame(train_encoded)
        ],
        axis=1
    )

    X_test = pd.concat(
        [
            test_numeric,
            pd.DataFrame(test_encoded)
        ],
        axis=1
    )

    return X_train, X_test, y_train, y_test