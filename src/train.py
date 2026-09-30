"""
Customer Churn Prediction - Model Training Pipeline
Trains multiple classification models (Logistic Regression, Random Forest, Gradient Boosting),
compares their performance on a validation set, refits the best model on Train + Val,
evaluates on a held-out Test set, and saves the complete pipeline as a joblib artifact.
"""

from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
    confusion_matrix
)

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"


def load_and_clean_data(data_path: Path) -> pd.DataFrame:
    """Load raw dataset and clean invalid data types and missing values."""
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")

    df = pd.read_csv(data_path)
    print(f"Loaded raw dataset with shape: {df.shape}")

    # TotalCharges has empty spaces (' ') for new customers with tenure = 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_count = df["TotalCharges"].isnull().sum()
    print(f"TotalCharges missing values: {missing_count} (tenure=0 customers)")

    # Drop missing values (0.15% of dataset)
    df = df.dropna(subset=["TotalCharges"]).copy()
    print(f"Dataset shape after removing missing values: {df.shape}")

    return df


def get_feature_lists(X: pd.DataFrame):
    """Categorize numerical and categorical features."""
    numerical_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = [col for col in X.columns if col not in numerical_features]
    return numerical_features, categorical_features


def build_preprocessor(numerical_features: list, categorical_features: list) -> ColumnTransformer:
    """Build Scikit-Learn ColumnTransformer pipeline with imputers and encoders."""
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numerical_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )
    return preprocessor


def train_models():
    """Main training routine comparing multiple models and saving the top performer."""
    print("=" * 60)
    print("STARTING CUSTOMER CHURN MODEL TRAINING PIPELINE")
    print("=" * 60)

    # 1. Load and clean
    df = load_and_clean_data(DATA_PATH)

    # 2. Features and Target
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"No": 0, "Yes": 1})

    numerical_features, categorical_features = get_feature_lists(X)
    print(f"Numerical features ({len(numerical_features)}): {numerical_features}")
    print(f"Categorical features ({len(categorical_features)}): {categorical_features}")

    # 3. Train-Validation-Test Split (60% Train, 20% Val, 20% Test)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
    )

    print(f"\nData Splits:")
    print(f"  Training set:   {X_train.shape[0]} samples ({X_train.shape[0]/len(X):.1%})")
    print(f"  Validation set: {X_val.shape[0]} samples ({X_val.shape[0]/len(X):.1%})")
    print(f"  Test set:       {X_test.shape[0]} samples ({X_test.shape[0]/len(X):.1%})")

    # 4. Define candidate models
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        )
    }

    # 5. Fit & Evaluate candidates on Validation Set
    results = []
    trained_pipelines = {}

    print("\nTraining and validating models...")
    for name, clf in models.items():
        preprocessor = build_preprocessor(numerical_features, categorical_features)
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])

        pipe.fit(X_train, y_train)

        y_val_pred = pipe.predict(X_val)
        y_val_prob = pipe.predict_proba(X_val)[:, 1]

        acc = accuracy_score(y_val, y_val_pred)
        prec = precision_score(y_val, y_val_pred, zero_division=0)
        rec = recall_score(y_val, y_val_pred, zero_division=0)
        f1 = f1_score(y_val, y_val_pred, zero_division=0)
        roc = roc_auc_score(y_val, y_val_prob)

        results.append({
            "Model": name,
            "Accuracy": acc,
            "Precision": prec,
            "Recall": rec,
            "F1 Score": f1,
            "ROC-AUC": roc
        })
        trained_pipelines[name] = pipe

    results_df = pd.DataFrame(results).sort_values(by="ROC-AUC", ascending=False)
    print("\n" + "=" * 60)
    print("VALIDATION SET MODEL COMPARISON")
    print("=" * 60)
    print(results_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # 6. Select Best Model by ROC-AUC
    best_model_name = results_df.iloc[0]["Model"]
    print(f"\n>>> Selected Best Model: '{best_model_name}' (Highest Validation ROC-AUC)")

    # 7. Refit best model on Train + Val
    print(f"\nRefitting '{best_model_name}' on combined Training + Validation data ({len(X_train_val)} samples)...")
    best_pipeline = trained_pipelines[best_model_name]
    best_pipeline.fit(X_train_val, y_train_val)

    # 8. Final evaluation on untouched Test Set
    print("\n" + "=" * 60)
    print("FINAL TEST SET EVALUATION (Unseen Data)")
    print("=" * 60)
    y_test_pred = best_pipeline.predict(X_test)
    y_test_prob = best_pipeline.predict_proba(X_test)[:, 1]

    test_acc = accuracy_score(y_test, y_test_pred)
    test_prec = precision_score(y_test, y_test_pred, zero_division=0)
    test_rec = recall_score(y_test, y_test_pred, zero_division=0)
    test_f1 = f1_score(y_test, y_test_pred, zero_division=0)
    test_roc = roc_auc_score(y_test, y_test_prob)

    print(f"Accuracy:  {test_acc:.4f}")
    print(f"Precision: {test_prec:.4f}")
    print(f"Recall:    {test_rec:.4f}")
    print(f"F1 Score:  {test_f1:.4f}")
    print(f"ROC-AUC:   {test_roc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=["No Churn (0)", "Churn (1)"]))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    # 9. Save Best Pipeline
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)
    print(f"\nSuccessfully serialized and saved model pipeline to:\n  {MODEL_PATH}")

    # 10. Smoke test verification
    print("\nTesting saved pipeline loading...")
    loaded_model = joblib.load(MODEL_PATH)
    sample_input = X_test.iloc[[0]]
    sample_pred = loaded_model.predict(sample_input)[0]
    sample_prob = loaded_model.predict_proba(sample_input)[0][1]
    print(f"Sample test prediction: {sample_pred}, probability: {sample_prob:.4f}")
    print("=" * 60)
    print("TRAINING COMPLETED SUCCESSFULLY!")
    print("=" * 60)

    return best_pipeline, results_df


if __name__ == "__main__":
    train_models()
