from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.features.feature_engineering import create_train_test_data


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

RANDOM_STATE = 42


def tune_logistic_regression(X_train, y_train):
    """Tune Logistic Regression hyperparameters."""

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    parameter_grid = {
        "C": [0.01, 0.1, 1, 10],
        "solver": ["liblinear", "lbfgs"],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    return search


def tune_random_forest(X_train, y_train):
    """Tune Random Forest hyperparameters."""

    model = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    parameter_grid = {
        "n_estimators": [200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="f1",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    search.fit(X_train, y_train)

    return search


def evaluate_tuned_model(model, X_test, y_test):
    """Evaluate a tuned model."""

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            y_pred,
            zero_division=0,
        ),
        "roc_auc": roc_auc_score(
            y_test,
            y_probability,
        ),
    }


def main():

    print("=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = create_train_test_data()

    print(f"\nTraining rows: {X_train.shape[0]}")
    print(f"Testing rows : {X_test.shape[0]}")
    print(f"Features     : {X_train.shape[1]}")

    # ---------------------------------------------------------
    # Logistic Regression
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TUNING LOGISTIC REGRESSION")
    print("=" * 70)

    logistic_search = tune_logistic_regression(
        X_train,
        y_train,
    )

    print("\nBest Logistic Regression parameters:")
    print(logistic_search.best_params_)

    print(
        f"Best CV F1: "
        f"{logistic_search.best_score_:.4f}"
    )

    logistic_metrics = evaluate_tuned_model(
        logistic_search.best_estimator_,
        X_test,
        y_test,
    )

    print("\nTest Set Results:")

    for metric, value in logistic_metrics.items():
        print(f"{metric.capitalize():12}: {value:.4f}")

    # ---------------------------------------------------------
    # Random Forest
    # ---------------------------------------------------------

    print("\n" + "=" * 70)
    print("TUNING RANDOM FOREST")
    print("=" * 70)

    rf_search = tune_random_forest(
        X_train,
        y_train,
    )

    print("\nBest Random Forest parameters:")
    print(rf_search.best_params_)

    print(
        f"Best CV F1: "
        f"{rf_search.best_score_:.4f}"
    )

    rf_metrics = evaluate_tuned_model(
        rf_search.best_estimator_,
        X_test,
        y_test,
    )

    print("\nTest Set Results:")

    for metric, value in rf_metrics.items():
        print(f"{metric.capitalize():12}: {value:.4f}")

    # ---------------------------------------------------------
    # Save tuned models
    # ---------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        logistic_search.best_estimator_,
        MODEL_DIR / "logistic_regression_tuned.joblib",
    )

    joblib.dump(
        rf_search.best_estimator_,
        MODEL_DIR / "random_forest_tuned.joblib",
    )

    # ---------------------------------------------------------
    # Save comparison
    # ---------------------------------------------------------

    results = pd.DataFrame(
        [
            {
                "model": "logistic_regression_tuned",
                **logistic_metrics,
                "cv_f1": logistic_search.best_score_,
            },
            {
                "model": "random_forest_tuned",
                **rf_metrics,
                "cv_f1": rf_search.best_score_,
            },
        ]
    )

    results.to_csv(
        REPORT_DIR / "tuned_model_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 70)
    print("TUNING COMPLETED")
    print("=" * 70)

    print(
        results.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nSaved:")
    print(
        f"✓ {MODEL_DIR / 'logistic_regression_tuned.joblib'}"
    )
    print(
        f"✓ {MODEL_DIR / 'random_forest_tuned.joblib'}"
    )
    print(
        f"✓ {REPORT_DIR / 'tuned_model_comparison.csv'}"
    )


if __name__ == "__main__":
    main()