from __future__ import annotations

import json
import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DEFAULT_MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "processed"


@dataclass
class MatchPredictionConfig:
    model_name: str = "logistic_regression"
    model_dir: Path = DEFAULT_MODEL_DIR
    latest_features_path: Path = DEFAULT_DATA_DIR / "latest_team_features.parquet"


class MatchPredictor:
    """
    Production-oriented inference wrapper for match outcome prediction.

    Predicts probabilities for team_a:
        - win
        - draw
        - loss
    """

    def __init__(self, config: MatchPredictionConfig | None = None) -> None:
        self.config = config or MatchPredictionConfig()
        self.model = None
        self.feature_columns: list[str] = []
        self.class_labels: list[str] = []
        self.latest_team_features = pd.DataFrame()

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        model_path = self.config.model_dir / f"{self.config.model_name}.joblib"
        metadata_path = self.config.model_dir / f"{self.config.model_name}_metadata.json"
        latest_features_path = self.config.latest_features_path

        if not model_path.exists():
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        if not latest_features_path.exists():
            raise FileNotFoundError(
                f"Latest team features file not found: {latest_features_path}"
            )

        self.model = joblib.load(model_path)

        with open(metadata_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        self.feature_columns = metadata["feature_columns"]
        self.class_labels = metadata["class_labels"]

        self.latest_team_features = pd.read_parquet(latest_features_path)

        required_columns = {
            "team",
            "elo_before",
            "rolling_points",
            "rolling_goal_diff",
            "rolling_win_rate",
        }
        missing = required_columns - set(self.latest_team_features.columns)
        if missing:
            raise ValueError(
                f"latest_team_features.parquet is missing columns: {sorted(missing)}"
            )

        self.latest_team_features = (
            self.latest_team_features
            .sort_values(["team"])
            .drop_duplicates(subset=["team"], keep="last")
            .reset_index(drop=True)
        )

    def _get_team_state(self, team: str) -> pd.Series:
        team_df = self.latest_team_features.loc[
            self.latest_team_features["team"] == team
        ]

        if team_df.empty:
            available = sorted(self.latest_team_features["team"].unique().tolist())
            raise ValueError(
                f"Team '{team}' not found in latest team features. "
                f"Example available teams: {available[:15]}"
            )

        return team_df.iloc[-1]

    def _build_feature_row(
        self,
        team_a: str,
        team_b: str,
        neutral_venue: bool = True,
    ) -> pd.DataFrame:
        a = self._get_team_state(team_a)
        b = self._get_team_state(team_b)

        row = {
            "team_a_elo_before": float(a["elo_before"]),
            "team_a_rolling_points": float(a["rolling_points"]),
            "team_a_rolling_goal_diff": float(a["rolling_goal_diff"]),
            "team_a_rolling_win_rate": float(a["rolling_win_rate"]),
            "team_b_elo_before": float(b["elo_before"]),
            "team_b_rolling_points": float(b["rolling_points"]),
            "team_b_rolling_goal_diff": float(b["rolling_goal_diff"]),
            "team_b_rolling_win_rate": float(b["rolling_win_rate"]),
            "elo_diff": float(a["elo_before"] - b["elo_before"]),
            "rolling_points_diff": float(a["rolling_points"] - b["rolling_points"]),
            "rolling_goal_diff_diff": float(
                a["rolling_goal_diff"] - b["rolling_goal_diff"]
            ),
            "rolling_win_rate_diff": float(
                a["rolling_win_rate"] - b["rolling_win_rate"]
            ),
            "abs_elo_diff": float(abs(a["elo_before"] - b["elo_before"])),
            "neutral_venue": int(neutral_venue),
        }

        feature_row = pd.DataFrame([row])

        missing_features = [c for c in self.feature_columns if c not in feature_row.columns]
        if missing_features:
            raise ValueError(
                f"Feature row is missing required columns: {missing_features}"
            )

        return feature_row[self.feature_columns]

    def predict_proba(
        self,
        team_a: str,
        team_b: str,
        neutral_venue: bool = True,
    ) -> dict[str, Any]:
        X = self._build_feature_row(
            team_a=team_a,
            team_b=team_b,
            neutral_venue=neutral_venue,
        )

        probabilities = self.model.predict_proba(X)[0]
        predicted_idx = int(probabilities.argmax())
        predicted_label = self.class_labels[predicted_idx]

        proba_map = {
            label: float(prob)
            for label, prob in zip(self.class_labels, probabilities)
        }

        return {
            "team_a": team_a,
            "team_b": team_b,
            "neutral_venue": neutral_venue,
            "predicted_label": predicted_label,
            "probabilities": proba_map,
            "features_used": X.iloc[0].to_dict(),
        }


def predict_match(
    team_a: str,
    team_b: str,
    neutral_venue: bool = True,
    model_name: str = "logistic_regression",
) -> dict[str, Any]:
    predictor = MatchPredictor(
        MatchPredictionConfig(model_name=model_name)
    )
    return predictor.predict_proba(
        team_a=team_a,
        team_b=team_b,
        neutral_venue=neutral_venue,
    )


if __name__ == "__main__":
    predictor = MatchPredictor()

    result = predictor.predict_proba(
        team_a="Spain",
        team_b="Brazil",
        neutral_venue=True,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))