from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

from src.utils.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR


TARGET_COL = "target"

NUMERIC_FEATURES = [
    "team_a_elo_before",
    "team_b_elo_before",
    "elo_diff",
    "abs_elo_diff",
    "team_a_rolling_goals_scored",
    "team_b_rolling_goals_scored",
    "team_a_rolling_goals_conceded",
    "team_b_rolling_goals_conceded",
    "team_a_rolling_goal_diff",
    "team_b_rolling_goal_diff",
    "team_a_rolling_win_rate",
    "team_b_rolling_win_rate",
    "team_a_rolling_points",
    "team_b_rolling_points",
    "rolling_goal_diff_diff",
    "rolling_points_diff",
    "rolling_win_rate_diff",
    "rolling_goals_scored_diff",
    "rolling_goals_conceded_diff",
    "team_a_expected_result",
    "team_b_expected_result",
]

CATEGORICAL_FEATURES = [
    "tournament",
    "neutral_venue",
]


def load_dataset() -> pd.DataFrame:
    """Load match-level modeling dataset."""
    file_path = PROCESSED_DATA_DIR / "match_model_dataset.parquet"
    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values("date").reset_index(drop=True)


def select_modeling_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only relevant columns for modeling."""
    required_cols = ["date", TARGET_COL] + NUMERIC_FEATURES + CATEGORICAL_FEATURES

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required modeling columns: {missing_cols}")

    return df[required_cols].copy()


def temporal_train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split chronologically to avoid leakage.
    Oldest matches go to train, most recent to test.
    """
    split_idx = int(len(df) * (1 - test_size))

    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()

    return train_df, test_df


def build_preprocessor() -> ColumnTransformer:
    """Create preprocessing pipeline for numeric and categorical features."""
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    return preprocessor

# ------------------------------------------------------------------
# Model Definitions
# ------------------------------------------------------------------
# Logistic Regression is the production model used by the simulation engine.
# The calibrated version is included only for benchmarking purposes.
# Benchmark results are documented in:
# experiments/05_match_model_benchmark.ipynb
# docs/modeling.md
# ------------------------------------------------------------------

def build_models(preprocessor: ColumnTransformer) -> Dict[str, Any]:
    """Build candidate ML pipelines."""
    logistic_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                LogisticRegression(
                    max_iter=2000,
                    solver="lbfgs",
                    class_weight=None,
                    random_state=42,
                ),
            ),
        ]
    )

    logistic_calibrated = CalibratedClassifierCV(
        estimator=Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=2000,
                        solver="lbfgs",
                        class_weight=None,
                        random_state=42,
                    ),
                ),
            ]
        ),
        method="isotonic",
        cv=3,
    )

    random_forest_pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    min_samples_split=10,
                    min_samples_leaf=5,
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    return {
        "logistic_regression": logistic_pipeline,
        "logistic_regression_calibrated": logistic_calibrated,
        "random_forest": random_forest_pipeline,
    }


def multiclass_brier_score(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_classes: int,
) -> float:
    """
    Compute multiclass Brier score.
    Lower is better.
    """
    y_true_one_hot = np.eye(n_classes)[y_true]
    return float(np.mean(np.sum((y_prob - y_true_one_hot) ** 2, axis=1)))


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    class_names: List[str],
) -> Tuple[dict, np.ndarray, np.ndarray]:
    """Evaluate model on test set."""
    y_pred = np.asarray(model.predict(X_test))
    y_prob = np.asarray(model.predict_proba(X_test))

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "log_loss": float(log_loss(y_test, y_prob)),
        "brier_score_multiclass": multiclass_brier_score(
            y_true=y_test,
            y_prob=y_prob,
            n_classes=len(class_names),
        ),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=class_names,
            output_dict=True,
            zero_division=0,
        ),
    }

    return metrics, y_pred, y_prob


def save_json(data: dict, output_path: Path) -> None:
    """Save dictionary as JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def save_model_metadata(
    model_name: str,
    label_encoder: LabelEncoder,
) -> None:
    """Save model metadata required for inference."""
    metadata = {
        "model_name": model_name,
        "target_column": TARGET_COL,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "feature_columns": NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        "class_labels": label_encoder.classes_.tolist(),
    }

    output_path = ARTIFACTS_DIR / "models" / f"{model_name}_metadata.json"
    save_json(metadata, output_path)


def save_predictions(
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    label_encoder: LabelEncoder,
    model_name: str,
) -> None:
    """Save test predictions for later analysis."""
    pred_df = X_test.copy()
    pred_df["y_true"] = label_encoder.inverse_transform(np.asarray(y_test))
    pred_df["y_pred"] = label_encoder.inverse_transform(np.asarray(y_pred))

    for idx, class_name in enumerate(label_encoder.classes_):
        pred_df[f"proba_{class_name}"] = np.asarray(y_prob)[:, idx]

    output_path = ARTIFACTS_DIR / "metrics" / f"{model_name}_test_predictions.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_df.to_parquet(output_path, index=False)


def train_and_evaluate() -> None:
    """Main training workflow."""
    print("Loading match modeling dataset...")
    df = load_dataset()
    print(f"Rows loaded: {len(df):,}")

    print("Selecting modeling columns...")
    df = select_modeling_columns(df)

    print("Creating temporal train/test split...")
    train_df, test_df = temporal_train_test_split(df, test_size=0.2)

    print(f"Train rows: {len(train_df):,}")
    print(f"Test rows:  {len(test_df):,}")
    print(f"Train date range: {train_df['date'].min().date()} -> {train_df['date'].max().date()}")
    print(f"Test date range:  {test_df['date'].min().date()} -> {test_df['date'].max().date()}")

    X_train = train_df.drop(columns=[TARGET_COL])
    X_test = test_df.drop(columns=[TARGET_COL])

    y_train_raw = train_df[TARGET_COL].copy()
    y_test_raw = test_df[TARGET_COL].copy()

    label_encoder = LabelEncoder()
    y_train = np.asarray(label_encoder.fit_transform(y_train_raw))
    y_test = np.asarray(label_encoder.transform(y_test_raw))

    print(f"Target classes: {list(label_encoder.classes_)}")
    print(f"Numeric features ({len(NUMERIC_FEATURES)}): {NUMERIC_FEATURES}")
    print(f"Categorical features ({len(CATEGORICAL_FEATURES)}): {CATEGORICAL_FEATURES}")

    preprocessor = build_preprocessor()
    models = build_models(preprocessor)

    all_metrics = {}

    for model_name, model in models.items():
        print(f"\nTraining model: {model_name}")
        model.fit(X_train, y_train)

        metrics, y_pred, y_prob = evaluate_model(
            model=model,
            X_test=X_test,
            y_test=y_test,
            class_names=list(label_encoder.classes_),
        )

        all_metrics[model_name] = metrics

        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Log Loss: {metrics['log_loss']:.4f}")
        print(f"Brier Score (multiclass): {metrics['brier_score_multiclass']:.4f}")

        model_output_path = ARTIFACTS_DIR / "models" / f"{model_name}.joblib"
        model_output_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_output_path)

        save_model_metadata(
            model_name=model_name,
            label_encoder=label_encoder,
        )

        save_predictions(
            X_test=X_test,
            y_test=y_test,
            y_pred=y_pred,
            y_prob=y_prob,
            label_encoder=label_encoder,
            model_name=model_name,
        )

    label_encoder_path = ARTIFACTS_DIR / "models" / "label_encoder.joblib"
    joblib.dump(label_encoder, label_encoder_path)

    metrics_output_path = ARTIFACTS_DIR / "metrics" / "match_outcome_metrics.json"
    save_json(all_metrics, metrics_output_path)

    print("\nTraining completed.")
    print(f"Saved models to: {ARTIFACTS_DIR / 'models'}")
    print(f"Saved metrics to: {ARTIFACTS_DIR / 'metrics'}")


if __name__ == "__main__":
    train_and_evaluate()