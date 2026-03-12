from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from src.utils.config import ARTIFACTS_DIR, PROCESSED_DATA_DIR


@dataclass(slots=True)
class MatchPredictionConfig:
    model_name: str = "logistic_regression"
    latest_features_path: Path = PROCESSED_DATA_DIR / "latest_team_features.parquet"
    models_dir: Path = ARTIFACTS_DIR / "models"


class MatchPredictor:
    """
    Reusable inference wrapper for match outcome prediction.

    Predicts probabilities for team_a:
        - win
        - draw
        - loss
    """

    def __init__(self, config: MatchPredictionConfig | None = None) -> None:
        self.config = config or MatchPredictionConfig()
        self.model = None
        self.metadata: dict[str, Any] = {}
        self.latest_team_features = pd.DataFrame()
        self.feature_columns: list[str] = []
        self.numeric_features: list[str] = []
        self.categorical_features: list[str] = []
        self.class_labels: list[str] = []

        self._load_artifacts()

    def _load_artifacts(self) -> None:
        model_path = self.config.models_dir / f"{self.config.model_name}.joblib"
        metadata_path = (
            self.config.models_dir / f"{self.config.model_name}_metadata.json"
        )
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

        with open(metadata_path, "r", encoding="utf-8") as file:
            self.metadata = json.load(file)

        self.feature_columns = self.metadata["feature_columns"]
        self.numeric_features = self.metadata["numeric_features"]
        self.categorical_features = self.metadata["categorical_features"]
        self.class_labels = self.metadata["class_labels"]

        self.latest_team_features = pd.read_parquet(latest_features_path)
        self.latest_team_features["date"] = pd.to_datetime(
            self.latest_team_features["date"]
        )

        required_latest_cols = {
            "team",
            "date",
            "elo_before",
            "rolling_goals_scored",
            "rolling_goals_conceded",
            "rolling_goal_diff",
            "rolling_win_rate",
            "rolling_points",
        }

        missing_cols = required_latest_cols - set(self.latest_team_features.columns)
        if missing_cols:
            raise ValueError(
                "latest_team_features.parquet is missing required columns: "
                f"{sorted(missing_cols)}"
            )

        self.latest_team_features = (
            self.latest_team_features
            .sort_values(["team", "date"])
            .drop_duplicates(subset=["team"], keep="last")
            .reset_index(drop=True)
        )

    def list_available_teams(self) -> list[str]:
        """Return sorted list of available national teams."""
        return sorted(self.latest_team_features["team"].unique().tolist())

    def _get_team_state(self, team: str) -> pd.Series:
        team_df = self.latest_team_features.loc[
            self.latest_team_features["team"] == team
        ]

        if team_df.empty:
            available = self.list_available_teams()
            raise ValueError(
                f"Team '{team}' not found in latest_team_features.parquet. "
                f"Example available teams: {available[:20]}"
            )

        return team_df.iloc[-1]

    def get_team_strength(self, team: str) -> float:
        """
        Return current pre-tournament strength proxy for a team.

        In v1 this is Elo-based and comes from the latest team snapshot.
        """
        state = self._get_team_state(team)
        return float(state["elo_before"])

    def _build_feature_row(
        self,
        team_a: str,
        team_b: str,
        tournament: str = "FIFA World Cup",
        neutral_venue: int | bool = 1,
    ) -> pd.DataFrame:
        a = self._get_team_state(team_a)
        b = self._get_team_state(team_b)

        neutral_venue = int(neutral_venue)

        row = {
            # Numeric features
            "team_a_elo_before": float(a["elo_before"]),
            "team_b_elo_before": float(b["elo_before"]),
            "elo_diff": float(a["elo_before"] - b["elo_before"]),
            "abs_elo_diff": float(abs(a["elo_before"] - b["elo_before"])),
            "team_a_rolling_goals_scored": float(a["rolling_goals_scored"]),
            "team_b_rolling_goals_scored": float(b["rolling_goals_scored"]),
            "team_a_rolling_goals_conceded": float(a["rolling_goals_conceded"]),
            "team_b_rolling_goals_conceded": float(b["rolling_goals_conceded"]),
            "team_a_rolling_goal_diff": float(a["rolling_goal_diff"]),
            "team_b_rolling_goal_diff": float(b["rolling_goal_diff"]),
            "team_a_rolling_win_rate": float(a["rolling_win_rate"]),
            "team_b_rolling_win_rate": float(b["rolling_win_rate"]),
            "team_a_rolling_points": float(a["rolling_points"]),
            "team_b_rolling_points": float(b["rolling_points"]),
            "rolling_goal_diff_diff": float(
                a["rolling_goal_diff"] - b["rolling_goal_diff"]
            ),
            "rolling_points_diff": float(a["rolling_points"] - b["rolling_points"]),
            "rolling_win_rate_diff": float(
                a["rolling_win_rate"] - b["rolling_win_rate"]
            ),
            "rolling_goals_scored_diff": float(
                a["rolling_goals_scored"] - b["rolling_goals_scored"]
            ),
            "rolling_goals_conceded_diff": float(
                a["rolling_goals_conceded"] - b["rolling_goals_conceded"]
            ),
            "team_a_expected_result": float(
                1 / (1 + 10 ** ((b["elo_before"] - a["elo_before"]) / 400))
            ),
            "team_b_expected_result": float(
                1 / (1 + 10 ** ((a["elo_before"] - b["elo_before"]) / 400))
            ),
            "match_k_factor": float(60.0),
            "match_goal_diff_multiplier": float(1.0),

            # Categorical features
            "tournament": tournament,
            "neutral_venue": neutral_venue,
        }

        feature_row = pd.DataFrame([row])

        missing_feature_columns = [
            col for col in self.feature_columns if col not in feature_row.columns
        ]
        if missing_feature_columns:
            raise ValueError(
                "Constructed feature row is missing required columns: "
                f"{missing_feature_columns}"
            )

        return feature_row[self.feature_columns]

    def predict_proba(
        self,
        team_a: str,
        team_b: str,
        tournament: str = "FIFA World Cup",
        neutral_venue: int | bool = 1,
    ) -> dict[str, Any]:
        X = self._build_feature_row(
            team_a=team_a,
            team_b=team_b,
            tournament=tournament,
            neutral_venue=neutral_venue,
        )

        probabilities = self.model.predict_proba(X)[0]
        predicted_idx = int(probabilities.argmax())
        predicted_label = self.class_labels[predicted_idx]

        probability_map = {
            label: float(prob)
            for label, prob in zip(self.class_labels, probabilities)
        }

        return {
            "team_a": team_a,
            "team_b": team_b,
            "tournament": tournament,
            "neutral_venue": int(neutral_venue),
            "predicted_label": predicted_label,
            "probabilities": probability_map,
            "team_a_snapshot_date": str(self._get_team_state(team_a)["date"].date()),
            "team_b_snapshot_date": str(self._get_team_state(team_b)["date"].date()),
            "features_used": X.iloc[0].to_dict(),
        }

    def predict_match(
        self,
        team_a: str,
        team_b: str,
        tournament: str = "FIFA World Cup",
        neutral_venue: int | bool = 1,
    ) -> dict[str, Any]:
        """
        Alias kept for compatibility with simulation adapters or future services.
        """
        return self.predict_proba(
            team_a=team_a,
            team_b=team_b,
            tournament=tournament,
            neutral_venue=neutral_venue,
        )


def predict_match(
    team_a: str,
    team_b: str,
    tournament: str = "FIFA World Cup",
    neutral_venue: int | bool = 1,
    model_name: str = "logistic_regression",
) -> dict[str, Any]:
    predictor = MatchPredictor(MatchPredictionConfig(model_name=model_name))
    return predictor.predict_proba(
        team_a=team_a,
        team_b=team_b,
        tournament=tournament,
        neutral_venue=neutral_venue,
    )


if __name__ == "__main__":
    predictor = MatchPredictor(
        MatchPredictionConfig(model_name="logistic_regression")
    )

    result = predictor.predict_proba(
        team_a="Spain",
        team_b="Brazil",
        tournament="FIFA World Cup",
        neutral_venue=1,
    )

    print(json.dumps(result, indent=2, ensure_ascii=False))