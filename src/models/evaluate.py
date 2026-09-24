import json
from pathlib import Path

import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.models.train import train_models


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = PROJECT_ROOT / "reports"


def evaluate_model(model, X_test, y_test):
    """Evaluate one trained classification model."""

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    metrics = {
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
        "average_precision": average_precision_score(
            y_test,
            y_probability,
        ),
        "confusion_matrix": confusion_matrix(
            y_test,
            y_pred,
        ).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            zero_division=0,
        ),
    }

    return metrics


def evaluate_all_models():

    (
        trained_models,
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
    ) = train_models()

    results = {}

    print("\n" + "=" * 70)
    print("MODEL EVALUATION")
    print("=" * 70)

    for name, model in trained_models.items():

        metrics = evaluate_model(
            model,
            X_test,
            y_test,
        )

        results[name] = metrics

        print(f"\n{name.upper()}")
        print("-" * 50)

        print(f"Accuracy          : {metrics['accuracy']:.4f}")
        print(f"Precision         : {metrics['precision']:.4f}")
        print(f"Recall            : {metrics['recall']:.4f}")
        print(f"F1 Score          : {metrics['f1']:.4f}")
        print(f"ROC-AUC           : {metrics['roc_auc']:.4f}")
        print(
            f"Average Precision : "
            f"{metrics['average_precision']:.4f}"
        )

        print("\nConfusion Matrix:")
        print(metrics["confusion_matrix"])

    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Create a clean comparison table.
    comparison_rows = []

    for model_name, metrics in results.items():

        comparison_rows.append(
            {
                "model": model_name,
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "roc_auc": metrics["roc_auc"],
                "average_precision": metrics["average_precision"],
            }
        )

    comparison_df = pd.DataFrame(comparison_rows)

    comparison_df = comparison_df.sort_values(
        by="f1",
        ascending=False,
    )

    comparison_path = REPORT_DIR / "model_comparison.csv"

    comparison_df.to_csv(
        comparison_path,
        index=False,
    )

    # Save complete evaluation results.
    json_path = REPORT_DIR / "model_evaluation.json"

    with open(json_path, "w", encoding="utf-8") as file:
        json.dump(
            results,
            file,
            indent=4,
        )

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        comparison_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.4f}",
        )
    )

    print("\nReports saved:")
    print(f"  ✓ {comparison_path}")
    print(f"  ✓ {json_path}")

    return comparison_df, results


if __name__ == "__main__":
    evaluate_all_models()