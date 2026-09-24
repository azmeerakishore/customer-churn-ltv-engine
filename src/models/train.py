from pathlib import Path

import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.features.feature_engineering import create_train_test_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"

RANDOM_STATE = 42


def build_models():
    """Create the machine learning models."""

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            min_samples_split=5,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    # XGBoost is optional so the project can still run
    # if the package is not installed.
    try:
        from xgboost import XGBClassifier

        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            colsample_bytree=0.8,
            objective="binary:logistic",
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )

    except ImportError:
        print("XGBoost is not installed. Skipping XGBoost.")

    return models


def train_models(save_models=True):
    """Train all models using the feature-engineering pipeline."""

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = create_train_test_data()

    models = build_models()

    if save_models:
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        joblib.dump(
            preprocessor,
            MODEL_DIR / "preprocessor.joblib",
        )

    trained_models = {}

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(X_train, y_train)

        trained_models[name] = model

        if save_models:
            joblib.dump(
                model,
                MODEL_DIR / f"{name}.joblib",
            )

        print(f"{name} training completed.")

    return (
        trained_models,
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    )


if __name__ == "__main__":

    trained_models, X_train, X_test, y_train, y_test, preprocessor = (
        train_models()
    )

    print("\n" + "=" * 60)
    print("MODEL TRAINING COMPLETED")
    print("=" * 60)

    print(f"Training rows : {X_train.shape[0]}")
    print(f"Testing rows  : {X_test.shape[0]}")
    print(f"Features      : {X_train.shape[1]}")

    print("\nModels trained:")

    for model_name in trained_models:
        print(f"  ✓ {model_name}")

    print(f"\nModels saved to: {MODEL_DIR}")
    