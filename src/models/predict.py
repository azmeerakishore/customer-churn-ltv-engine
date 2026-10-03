from pathlib import Path

import joblib
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
DEFAULT_MODEL_NAME = "logistic_regression"
AVAILABLE_MODELS = {
    "logistic_regression",
    "random_forest",
    "xgboost",
    "logistic_regression_tuned",
    "random_forest_tuned",
}
DEFAULT_HORIZON_MONTHS = 12


def load_artifacts(model_name: str = DEFAULT_MODEL_NAME):
    """Load a saved classifier and the matching fitted preprocessor."""
    if model_name not in AVAILABLE_MODELS:
        raise ValueError(
            f"Unknown model '{model_name}'. Choose one of: "
            f"{', '.join(sorted(AVAILABLE_MODELS))}."
        )

    model_path = MODEL_DIR / f"{model_name}.joblib"
    preprocessor_path = MODEL_DIR / "preprocessor.joblib"
    missing_paths = [
        path for path in (model_path, preprocessor_path) if not path.is_file()
    ]
    if missing_paths:
        raise FileNotFoundError(
            "Required model artifacts are missing: "
            + ", ".join(str(path) for path in missing_paths)
        )

    return joblib.load(model_path), joblib.load(preprocessor_path)


def score_customers(
    customers: pd.DataFrame,
    horizon_months: int = DEFAULT_HORIZON_MONTHS,
    model_name: str = DEFAULT_MODEL_NAME,
    model=None,
    preprocessor=None,
) -> pd.DataFrame:
    """Score customers and estimate horizon-based revenue at risk."""
    if customers.empty:
        raise ValueError("At least one customer record is required.")
    if not isinstance(horizon_months, int) or not 1 <= horizon_months <= 120:
        raise ValueError("horizon_months must be an integer from 1 to 120.")

    if model is None or preprocessor is None:
        model, preprocessor = load_artifacts(model_name)

    expected_columns = list(preprocessor.feature_names_in_)
    missing_columns = sorted(set(expected_columns) - set(customers.columns))
    if missing_columns:
        raise ValueError(
            "Customer data is missing required columns: "
            + ", ".join(missing_columns)
        )

    features = customers.loc[:, expected_columns].copy()
    numerical_columns = preprocessor.transformers_[0][2]
    for column in numerical_columns:
        original_values = features[column]
        converted_values = pd.to_numeric(original_values, errors="coerce")
        invalid_values = (
            original_values.notna()
            & original_values.astype(str).str.strip().ne("")
            & converted_values.isna()
        )
        if invalid_values.any():
            raise ValueError(f"{column} must contain valid numbers.")
        features[column] = converted_values
    features["TotalCharges"] = features["TotalCharges"].fillna(0)

    if features.isna().any().any():
        raise ValueError("Customer fields cannot be empty.")
    if features[numerical_columns].isna().any().any():
        raise ValueError("Numeric customer fields must contain valid numbers.")
    if not np.isfinite(features[numerical_columns].to_numpy()).all():
        raise ValueError("Numeric customer fields must be finite numbers.")
    if not features["SeniorCitizen"].isin([0, 1]).all():
        raise ValueError("SeniorCitizen must be 0 or 1.")
    for column in ("tenure", "MonthlyCharges", "TotalCharges"):
        if (features[column] < 0).any():
            raise ValueError(f"{column} cannot be negative.")
    categorical_columns = preprocessor.transformers_[1][2]
    for column in categorical_columns:
        if features[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"{column} cannot be empty.")

    processed = pd.DataFrame(
        preprocessor.transform(features),
        columns=preprocessor.get_feature_names_out(),
        index=features.index,
    )
    probabilities = model.predict_proba(processed)[:, 1]

    result = pd.DataFrame(index=customers.index)
    if "customerID" in customers.columns:
        result["customerID"] = customers["customerID"]
    result["churn_probability"] = probabilities
    result["risk_category"] = pd.cut(
        probabilities,
        bins=[-0.001, 1 / 3, 2 / 3, 1.001],
        labels=["Low", "Medium", "High"],
    ).astype(str)
    result["monthly_charges"] = features["MonthlyCharges"]
    result["revenue_at_risk"] = (
        probabilities * features["MonthlyCharges"] * horizon_months
    )
    result["horizon_months"] = horizon_months
    return result.reset_index(drop=True)