from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.features.feature_engineering import create_train_test_data


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_DIR = PROJECT_ROOT / "models"
REPORT_DIR = PROJECT_ROOT / "reports"

MODEL_PATH = MODEL_DIR / "random_forest_tuned.joblib"

FEATURE_REPORT_PATH = REPORT_DIR / "shap_feature_importance.csv"
PLOT_PATH = REPORT_DIR / "shap_summary.png"

SAMPLE_SIZE = 1000
RANDOM_STATE = 42


# ============================================================
# EXTRACT SHAP VALUES
# ============================================================

def extract_shap_values(shap_result):
    """
    Extract SHAP values for the churn class.

    Supports SHAP output formats used by different versions.
    """

    # SHAP Explanation object
    if hasattr(shap_result, "values"):

        values = shap_result.values

        # Shape:
        # samples x features x classes
        if values.ndim == 3:
            return values[:, :, 1]

        # Shape:
        # samples x features
        return values

    # List output
    if isinstance(shap_result, list):

        if len(shap_result) >= 2:
            return shap_result[1]

        return shap_result[0]

    # NumPy array
    if hasattr(shap_result, "ndim"):

        if shap_result.ndim == 3:
            return shap_result[:, :, 1]

        return shap_result

    raise ValueError(
        "Unsupported SHAP output format."
    )


# ============================================================
# MAIN
# ============================================================

def run_shap_analysis():

    print("=" * 70)
    print("SHAP EXPLAINABILITY ANALYSIS")
    print("=" * 70)

    # --------------------------------------------------------
    # Create reports directory
    # --------------------------------------------------------

    REPORT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Check model
    # --------------------------------------------------------

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nTuned model not found:\n{MODEL_PATH}\n\n"
            "Run:\n"
            "python3 -m src.models.tune"
        )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor
    ) = create_train_test_data()

    print(f"\nTest rows: {X_test.shape[0]}")
    print(f"Features : {X_test.shape[1]}")

    # --------------------------------------------------------
    # Load tuned Random Forest
    # --------------------------------------------------------

    print("\nLoading tuned Random Forest...")

    model = joblib.load(MODEL_PATH)

    print("✓ random_forest_tuned.joblib loaded")

    # --------------------------------------------------------
    # Select SHAP sample
    # --------------------------------------------------------

    sample_size = min(
        SAMPLE_SIZE,
        len(X_test)
    )

    X_sample = X_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE
    )

    print(
        f"SHAP sample size: {len(X_sample)}"
    )

    # --------------------------------------------------------
    # Create SHAP explainer
    # --------------------------------------------------------

    print("\nCreating SHAP TreeExplainer...")

    explainer = shap.TreeExplainer(model)

    print("✓ SHAP TreeExplainer created")

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    print("\nCalculating SHAP values...")

    shap_result = explainer(X_sample)

    print("✓ SHAP values calculated")

    # --------------------------------------------------------
    # Extract churn class values
    # --------------------------------------------------------

    shap_values = extract_shap_values(
        shap_result
    )

    print(
        f"SHAP array shape: {shap_values.shape}"
    )

    # --------------------------------------------------------
    # Validate shape
    # --------------------------------------------------------

    if shap_values.shape[0] != X_sample.shape[0]:

        raise ValueError(
            "SHAP rows do not match X_sample rows."
        )

    if shap_values.shape[1] != X_sample.shape[1]:

        raise ValueError(
            "SHAP features do not match X_sample features."
        )

    # --------------------------------------------------------
    # Calculate feature importance
    # --------------------------------------------------------

    mean_abs_shap = (
        abs(shap_values)
        .mean(axis=0)
    )

    feature_names = X_sample.columns.tolist()

    feature_importance = pd.DataFrame(
        {
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap
        }
    )

    feature_importance = (
        feature_importance
        .sort_values(
            "mean_abs_shap",
            ascending=False
        )
        .reset_index(drop=True)
    )

    # --------------------------------------------------------
    # Save CSV
    # --------------------------------------------------------

    feature_importance.to_csv(
        FEATURE_REPORT_PATH,
        index=False
    )

    print(
        "\n✓ Feature importance saved:"
        f"\n  {FEATURE_REPORT_PATH}"
    )

    # --------------------------------------------------------
    # Print top 15
    # --------------------------------------------------------

    print("\nTop 15 churn-related features:")
    print("-" * 70)

    print(
        feature_importance
        .head(15)
        .to_string(index=False)
    )

    # ========================================================
    # SHAP SUMMARY PLOT
    # ========================================================

    print("\nGenerating SHAP summary plot...")

    # --------------------------------------------------------
    # IMPORTANT:
    # Use raw SHAP values + X_sample.
    #
    # This avoids the base_values=None problem from manually
    # constructing shap.Explanation objects.
    # --------------------------------------------------------

    plt.figure(
        figsize=(10, 8)
    )

    shap.summary_plot(
        shap_values,
        X_sample,
        feature_names=X_sample.columns.tolist(),
        show=False
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_PATH,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()

    print(
        "\n✓ SHAP summary plot saved:"
        f"\n  {PLOT_PATH}"
    )

    # ========================================================
    # FINAL RESULT
    # ========================================================

    print("\n" + "=" * 70)
    print("SHAP ANALYSIS COMPLETED")
    print("=" * 70)

    print(
        f"\nFeature importance CSV:"
        f"\n{FEATURE_REPORT_PATH}"
    )

    print(
        f"\nSHAP summary plot:"
        f"\n{PLOT_PATH}"
    )

    print("\nTop 5 features:")

    for i, row in feature_importance.head(5).iterrows():

        print(
            f"{i + 1}. "
            f"{row['feature']} "
            f"({row['mean_abs_shap']:.6f})"
        )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_shap_analysis()