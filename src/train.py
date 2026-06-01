"""
Model Training and Tracking
Credit Risk Prediction with MLflow
"""

import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import os
import joblib
import warnings

warnings.filterwarnings("ignore")


# ================================================================
#  DATA LOADING & SPLITTING
# ================================================================


def load_data(data_path, use_woe=False):
    """
    Load processed data and separate features/target.

    Args:
        data_path: Path to processed CSV
        use_woe: If True, use only _woe columns (for Logistic Regression)
                 If False, use all non-woe columns (for tree models)
    """
    df = pd.read_csv(data_path)

    # Always exclude these (leakage or ID columns)
    base_exclude = [
        "CustomerId",  # ID column
        "is_high_risk",  # Target
        "recency",
        "frequency",
        "monetary",  # Used to create target
        "refund_rate",
        "refund_amount",  # Highly correlated with target
    ]

    if use_woe:
        # For Logistic Regression: use ONLY WoE columns
        feature_cols = [
            col
            for col in df.columns
            if col.endswith("_woe") and col not in base_exclude
        ]
        X = df[feature_cols].copy()
    else:
        # For tree models: use all non-WoE columns (already one-hot encoded)
        feature_cols = [
            col
            for col in df.columns
            if col not in base_exclude and not col.endswith("_woe")
        ]
        X = df[feature_cols].copy()

    y = df["is_high_risk"]

    return X, y


def split_data(X, y, test_size=0.3, random_state=42):
    """Split data into train/test sets with stratification."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


# ================================================================
# MODEL EVALUATION
# ================================================================


def evaluate_model(model, X_test, y_test):
    """Calculate all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
    }
    return metrics


def print_metrics(metrics, model_name):
    """Pretty print metrics."""
    print(f"\n  {model_name} Results:")
    print(f"    Recall:  {metrics['recall']:.4f}  ← Most important for credit risk")
    print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"    F1:      {metrics['f1_score']:.4f}")
    print(f"    Prec:    {metrics['precision']:.4f}")
    print(f"    Acc:     {metrics['accuracy']:.4f}")


# ================================================================
# MODEL TRAINING
# ================================================================


def train_logistic_regression(X_train, X_test, y_train, y_test):
    """
    Train Logistic Regression using WoE features.
    WoE columns are already transformed for linear models.
    """
    with mlflow.start_run(run_name="LogisticRegression_WoE"):
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(
            {
                "model_type": "LogisticRegression",
                "features_used": X_train.shape[1],
                "feature_type": "WoE_transformed",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "logistic_regression_model")

        print_metrics(metrics, "Logistic Regression (WoE)")

        return model, metrics


def train_random_forest_baseline(X_train, X_test, y_train, y_test):
    """
    Train Random Forest baseline using clean features (no WoE).
    """
    with mlflow.start_run(run_name="RandomForest_Baseline"):
        model = RandomForestClassifier(random_state=42, n_jobs=-1)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(
            {
                "model_type": "RandomForest",
                "n_estimators": 100,
                "max_depth": "None",
                "features_used": X_train.shape[1],
                "feature_type": "clean_onehot",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "random_forest_baseline")

        print_metrics(metrics, "Random Forest (Baseline)")

        return model, metrics


def train_random_forest_tuned(X_train, X_test, y_train, y_test):
    """
    Train Random Forest with hyperparameter tuning.
    """
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [5, 10, None],
        "min_samples_split": [2, 5, 10],
    }

    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)

    # MOVED INSIDE: Start tracking before fitting to capture the tuning lifecycle
    with mlflow.start_run(run_name="RandomForest_Tuned"):
        print("\n  Running GridSearch (Random Forest)...")
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        metrics = evaluate_model(best_model, X_test, y_test)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "random_forest_tuned")

        print(f"\n  Best params: {grid_search.best_params_}")
        print_metrics(metrics, "Random Forest (Tuned)")

    return best_model, metrics


# ================================================================
#  GRADIENT BOOSTING (XGBoost style with sklearn)
# ================================================================


def train_gradient_boosting_baseline(X_train, X_test, y_train, y_test):
    """
    Train Gradient Boosting baseline.
    Gradient Boosting builds trees sequentially, each correcting previous errors.
    """
    with mlflow.start_run(run_name="GradientBoosting_Baseline"):
        model = GradientBoostingClassifier(random_state=42)
        model.fit(X_train, y_train)

        metrics = evaluate_model(model, X_test, y_test)

        mlflow.log_params(
            {
                "model_type": "GradientBoosting",
                "n_estimators": 100,
                "learning_rate": 0.1,
                "max_depth": 3,
                "features_used": X_train.shape[1],
                "feature_type": "clean_onehot",
            }
        )
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "gradient_boosting_baseline")

        print_metrics(metrics, "Gradient Boosting (Baseline)")

        return model, metrics


def train_gradient_boosting_tuned(X_train, X_test, y_train, y_test):
    """
    Train Gradient Boosting with hyperparameter tuning.
    """
    param_grid = {
        "n_estimators": [50, 100, 150],
        "learning_rate": [0.05, 0.1, 0.2],
        "max_depth": [3, 5, 7],
        "subsample": [0.8, 1.0],
    }

    base_model = GradientBoostingClassifier(random_state=42)

    # MOVED INSIDE: Start tracking before fitting to capture the tuning lifecycle
    with mlflow.start_run(run_name="GradientBoosting_Tuned"):
        print("\n  Running GridSearch (Gradient Boosting - may take 2-3 minutes)...")
        grid_search = GridSearchCV(
            base_model, param_grid, cv=5, scoring="roc_auc", n_jobs=-1, verbose=0
        )
        grid_search.fit(X_train, y_train)

        best_model = grid_search.best_estimator_
        metrics = evaluate_model(best_model, X_test, y_test)

        mlflow.log_params(grid_search.best_params_)
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(best_model, "gradient_boosting_tuned")

        print(f"\n  Best params: {grid_search.best_params_}")
        print_metrics(metrics, "Gradient Boosting (Tuned)")

    return best_model, metrics


# ================================================================
#  MODEL REGISTRATION
# ================================================================


def register_best_model():
    """Find best model by Recall and register to MLflow Registry."""
    experiment = mlflow.get_experiment_by_name("Credit Risk Modeling")

    if experiment is None:
        print("No experiments found.")
        return None

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])

    if len(runs) == 0:
        print("No runs found.")
        return None

    # Sort by RECALL (most important for credit risk)
    best_run = runs.loc[runs["metrics.recall"].idxmax()]
    best_recall = best_run["metrics.recall"]
    best_roc_auc = best_run["metrics.roc_auc"]
    best_model_name = best_run["tags.mlflow.runName"]
    best_run_id = best_run["run_id"]

    print(f"\nBest Model (by Recall): {best_model_name}")
    print(f"   Recall:  {best_recall:.4f}")
    print(f"   ROC-AUC: {best_roc_auc:.4f}")
    print(f"   Run ID: {best_run_id}")

    # Explicitly register using client
    client = MlflowClient()

    # Try the most likely artifact name based on your best model
    if "GradientBoosting_Baseline" in best_model_name:
        artifact_name = "gradient_boosting_baseline"
    elif "GradientBoosting_Tuned" in best_model_name:
        artifact_name = "gradient_boosting_tuned"
    elif "RandomForest_Tuned" in best_model_name:
        artifact_name = "random_forest_tuned"
    elif "RandomForest_Baseline" in best_model_name:
        artifact_name = "random_forest_baseline"
    else:
        artifact_name = "logistic_regression_model"

    model_uri = f"runs:/{best_run_id}/{artifact_name}"
    print(f"   Model URI: {model_uri}")

    try:
        # Register the model
        registered_model = mlflow.register_model(model_uri, "CreditRiskModel")
        print(f"\n Registered: CreditRiskModel (version {registered_model.version})")

        # Transition to Production stage
        client.transition_model_version_stage(
            name="CreditRiskModel", version=registered_model.version, stage="Production"
        )
        print("Set to Production stage")

    except Exception as e:
        print(f"\n Registration failed: {e}")
        print("   Trying alternative artifact names...")

        # Try alternative artifact names
        for alt_name in [
            "gradient_boosting_baseline",
            "random_forest_tuned",
            "logistic_regression_model",
        ]:
            try:
                model_uri = f"runs:/{best_run_id}/{alt_name}"
                registered_model = mlflow.register_model(model_uri, "CreditRiskModel")
                print(f"Registered with {alt_name}: version {registered_model.version}")
                break
            except Exception:
                continue

    return best_run_id


# ================================================================
# MAIN PIPELINE
# ================================================================


def main():
    """Main training pipeline with Model Registry support via SQLite."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(
        script_dir, "..", "data", "processed", "processed_data.csv"
    )
    model_dir = os.path.join(script_dir, "..", "models")
    os.makedirs(model_dir, exist_ok=True)

    print("=" * 60)
    print("TASK 5: CREDIT RISK MODEL TRAINING")
    print("=" * 60)

    # ============================================================
    # FIX: Set MLflow tracking to a SQL backend for Model Registry
    # ============================================================
    db_path = os.path.normpath(os.path.join(model_dir, "mlflow.db"))
    mlflow.set_tracking_uri(f"sqlite:///{db_path}")
    mlflow.set_experiment("Credit Risk Modeling")

    # ============================================================
    # MODEL 1: LOGISTIC REGRESSION (uses WoE features)
    # ============================================================
    print("\n" + "-" * 50)
    print("MODEL 1: LOGISTIC REGRESSION (WoE features)")
    print("-" * 50)

    X, y = load_data(data_path, use_woe=True)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Features: {X_train.shape[1]} (all WoE columns)")

    lr_model, lr_metrics = train_logistic_regression(X_train, X_test, y_train, y_test)

    # ============================================================
    # MODEL 2: RANDOM FOREST BASELINE (clean features, no WoE)
    # ============================================================
    print("\n" + "-" * 50)
    print("MODEL 2: RANDOM FOREST BASELINE (clean features)")
    print("-" * 50)

    X, y = load_data(data_path, use_woe=False)
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Features: {X_train.shape[1]} (one-hot encoded, no WoE)")

    rf_baseline, rf_baseline_metrics = train_random_forest_baseline(
        X_train, X_test, y_train, y_test
    )

    # ============================================================
    # MODEL 3: RANDOM FOREST TUNED
    # ============================================================
    print("\n" + "-" * 50)
    print("MODEL 3: RANDOM FOREST TUNED")
    print("-" * 50)

    rf_tuned, rf_tuned_metrics = train_random_forest_tuned(
        X_train, X_test, y_train, y_test
    )

    # ============================================================
    # MODEL 4: GRADIENT BOOSTING BASELINE
    # ============================================================
    print("\n" + "-" * 50)
    print("MODEL 4: GRADIENT BOOSTING BASELINE")
    print("-" * 50)

    gb_baseline, gb_baseline_metrics = train_gradient_boosting_baseline(
        X_train, X_test, y_train, y_test
    )

    # ============================================================
    # MODEL 5: GRADIENT BOOSTING TUNED
    # ============================================================
    print("\n" + "-" * 50)
    print("MODEL 5: GRADIENT BOOSTING TUNED")
    print("-" * 50)

    gb_tuned, gb_tuned_metrics = train_gradient_boosting_tuned(
        X_train, X_test, y_train, y_test
    )

    # ============================================================
    # COMPARE ALL RESULTS
    # ============================================================
    print("\n" + "=" * 70)
    print("MODEL COMPARISON (Best by RECALL - Most Important for Credit Risk)")
    print("=" * 70)

    results = pd.DataFrame(
        {
            "Model": [
                "Logistic Regression (WoE)",
                "Random Forest (Baseline)",
                "Random Forest (Tuned)",
                "Gradient Boosting (Baseline)",
                "Gradient Boosting (Tuned)",
            ],
            "Recall (↑)": [
                lr_metrics["recall"],
                rf_baseline_metrics["recall"],
                rf_tuned_metrics["recall"],
                gb_baseline_metrics["recall"],
                gb_tuned_metrics["recall"],
            ],
            "ROC-AUC (↑)": [
                lr_metrics["roc_auc"],
                rf_baseline_metrics["roc_auc"],
                rf_tuned_metrics["roc_auc"],
                gb_baseline_metrics["roc_auc"],
                gb_tuned_metrics["roc_auc"],
            ],
            "F1 Score (↑)": [
                lr_metrics["f1_score"],
                rf_baseline_metrics["f1_score"],
                rf_tuned_metrics["f1_score"],
                gb_baseline_metrics["f1_score"],
                gb_tuned_metrics["f1_score"],
            ],
            "Precision (↑)": [
                lr_metrics["precision"],
                rf_baseline_metrics["precision"],
                rf_tuned_metrics["precision"],
                gb_baseline_metrics["precision"],
                gb_tuned_metrics["precision"],
            ],
            "Accuracy (↑)": [
                lr_metrics["accuracy"],
                rf_baseline_metrics["accuracy"],
                rf_tuned_metrics["accuracy"],
                gb_baseline_metrics["accuracy"],
                gb_tuned_metrics["accuracy"],
            ],
        }
    )

    print(results.round(4).to_string(index=False))

    # ============================================================
    # REGISTER BEST MODEL
    # ============================================================
    print("\n" + "-" * 50)
    print("REGISTERING BEST MODEL")
    print("-" * 50)

    register_best_model()

    # Save best model locally (by Recall)
    best_model_path = os.path.join(model_dir, "best_model.pkl")

    # Find best recall among all models
    best_recall = max(
        lr_metrics["recall"],
        rf_baseline_metrics["recall"],
        rf_tuned_metrics["recall"],
        gb_baseline_metrics["recall"],
        gb_tuned_metrics["recall"],
    )

    if gb_tuned_metrics["recall"] == best_recall:
        joblib.dump(gb_tuned, best_model_path)
        print(
            f" Saved: Gradient Boosting Tuned (Recall: {gb_tuned_metrics['recall']:.4f})"
        )
    elif gb_baseline_metrics["recall"] == best_recall:
        joblib.dump(gb_baseline, best_model_path)
        print(
            f" Saved: Gradient Boosting Baseline (Recall: {gb_baseline_metrics['recall']:.4f})"
        )
    elif rf_tuned_metrics["recall"] == best_recall:
        joblib.dump(rf_tuned, best_model_path)
        print(f" Saved: Random Forest Tuned (Recall: {rf_tuned_metrics['recall']:.4f})")
    elif lr_metrics["recall"] == best_recall:
        joblib.dump(lr_model, best_model_path)
        print(f"Saved: Logistic Regression (Recall: {lr_metrics['recall']:.4f})")
    else:
        joblib.dump(rf_baseline, best_model_path)
        print(
            f" Saved: Random Forest Baseline (Recall: {rf_baseline_metrics['recall']:.4f})"
        )

    print("\nView MLflow UI: mlflow ui --backend-store-uri sqlite:///models/mlflow.db")
    print("   Then open: http://localhost:5000")


if __name__ == "__main__":
    main()
