"""
Customer Churn Prediction - Model Evaluation Script
Loads the serialized model pipeline, evaluates on the held-out test dataset,
analyzes metric trade-offs across classification thresholds, and generates
visual evaluation artifacts (Confusion Matrix, ROC Curve, Precision-Recall Curve).
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay
)

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
REPORTS_DIR = BASE_DIR / "reports"


def get_test_data():
    """Load cleaned data and generate the exact test split used in training."""
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.dropna(subset=["TotalCharges"]).copy()

    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    # Exact same random_state & stratify as in train.py
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    return X_test, y_test


def evaluate_thresholds(y_true, y_prob):
    """Evaluate performance metrics across custom decision thresholds."""
    thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
    records = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        records.append({
            "Threshold": t,
            "Accuracy": accuracy_score(y_true, y_pred),
            "Precision": precision_score(y_true, y_pred, zero_division=0),
            "Recall": recall_score(y_true, y_pred, zero_division=0),
            "F1-Score": f1_score(y_true, y_pred, zero_division=0)
        })
    return pd.DataFrame(records)


def run_evaluation():
    print("=" * 60)
    print("EVALUATING CUSTOMER CHURN MODEL PIPELINE")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Run src/train.py first.")

    pipeline = joblib.load(MODEL_PATH)
    X_test, y_test = get_test_data()

    y_test_pred = pipeline.predict(X_test)
    y_test_prob = pipeline.predict_proba(X_test)[:, 1]

    print("\n[1] Overall Metrics (Default 0.5 Threshold):")
    print(f"  Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
    print(f"  Precision: {precision_score(y_test, y_test_pred):.4f}")
    print(f"  Recall:    {recall_score(y_test, y_test_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_test_pred):.4f}")
    print(f"  ROC-AUC:   {roc_auc_score(y_test, y_test_prob):.4f}")

    print("\n[2] Detailed Classification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["Retained (0)", "Churned (1)"]))

    print("\n[3] Threshold Sensitivity Analysis:")
    thresh_df = evaluate_thresholds(y_test, y_test_prob)
    print(thresh_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Generate and save evaluation plots
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")

    # Plot 1: Confusion Matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_test_pred,
        display_labels=["No Churn", "Churn"],
        cmap="Blues",
        ax=ax
    )
    ax.set_title("Test Confusion Matrix (Threshold = 0.50)", fontsize=13, pad=12)
    plt.tight_layout()
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=200)
    plt.close()
    print(f"\nSaved Confusion Matrix plot: {cm_path}")

    # Plot 2: ROC Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    RocCurveDisplay.from_predictions(y_test, y_test_prob, ax=ax, name="Logistic Regression Pipeline")
    ax.plot([0, 1], [0, 1], "k--", label="Random Chance (AUC = 0.50)")
    ax.set_title(f"ROC Curve (AUC = {roc_auc_score(y_test, y_test_prob):.3f})", fontsize=13, pad=12)
    ax.legend(loc="lower right")
    plt.tight_layout()
    roc_path = REPORTS_DIR / "roc_curve.png"
    plt.savefig(roc_path, dpi=200)
    plt.close()
    print(f"Saved ROC Curve plot: {roc_path}")

    # Plot 3: Precision-Recall Curve
    fig, ax = plt.subplots(figsize=(6, 5))
    PrecisionRecallDisplay.from_predictions(y_test, y_test_prob, ax=ax, name="Model PR Curve")
    ax.set_title("Precision-Recall Curve", fontsize=13, pad=12)
    plt.tight_layout()
    pr_path = REPORTS_DIR / "precision_recall_curve.png"
    plt.savefig(pr_path, dpi=200)
    plt.close()
    print(f"Saved Precision-Recall plot: {pr_path}")

    print("=" * 60)
    print("EVALUATION FINISHED SUCCESSFULLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation()
