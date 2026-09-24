from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.preprocessing.preprocess import clean_data, load_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]

RANDOM_STATE = 42
TEST_SIZE = 0.20


def prepare_dataset():
    """Load and clean the Telco churn dataset."""
    df = load_data()
    df = clean_data(df)

    # Remove customer ID because it is not a predictive feature.
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])

    # Separate target from input features.
    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    return X, y


def create_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Create preprocessing transformations for numerical and categorical features."""

    numerical_columns = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        include=["object", "str"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numerical",
                StandardScaler(),
                numerical_columns,
            ),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
                categorical_columns,
            ),
        ]
    )

    return preprocessor


def create_train_test_data():
    """Split the dataset and fit preprocessing only on training data."""

    X, y = prepare_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    preprocessor = create_preprocessor(X_train)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    X_train_processed = pd.DataFrame(
        X_train_processed,
        columns=feature_names,
        index=X_train.index,
    )

    X_test_processed = pd.DataFrame(
        X_test_processed,
        columns=feature_names,
        index=X_test.index,
    )

    return (
        X_train_processed,
        X_test_processed,
        y_train,
        y_test,
        preprocessor,
    )


if __name__ == "__main__":
    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = create_train_test_data()

    print("Feature engineering completed successfully.")
    print(f"Training rows: {X_train.shape[0]}")
    print(f"Testing rows: {X_test.shape[0]}")
    print(f"Training features: {X_train.shape[1]}")
    print(f"Testing features: {X_test.shape[1]}")

    print("\nTarget distribution:")
    print(y_train.value_counts())

    print("\nFirst 5 processed training rows:")
    print(X_train.head())
