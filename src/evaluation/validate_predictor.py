from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.models.match_outcome.predict import MatchPredictionConfig, MatchPredictor
from src.utils.config import ARTIFACTS_DIR


DEFAULT_MODEL_NAME = "logistic_regression"
DEFAULT_TOURNAMENT = "FIFA World Cup"
DEFAULT_NEUTRAL_VENUE = 1


TEST_MATCHUPS: list[tuple[str, str]] = [
    ("Spain", "Brazil"),
    ("Brazil", "Spain"),
    ("Argentina", "France"),
    ("France", "Argentina"),
    ("Brazil", "Bolivia"),
    ("Bolivia", "Brazil"),
    ("France", "San Marino"),
    ("San Marino", "France"),
    ("Japan", "Morocco"),
    ("Morocco", "Japan"),
    ("England", "Germany"),
    ("Germany", "England"),
]


def run_validation_suite(
    model_name: str = DEFAULT_MODEL_NAME,
    tournament: str = DEFAULT_TOURNAMENT,
    neutral_venue: int = DEFAULT_NEUTRAL_VENUE,
) -> pd.DataFrame:
    """
    Run a fixed validation suite of matchups and return predictions as a dataframe.
    """
    predictor = MatchPredictor(
        MatchPredictionConfig(model_name=model_name)
    )

    rows: list[dict] = []

    for team_a, team_b in TEST_MATCHUPS:
        try:
            result = predictor.predict_proba(
                team_a=team_a,
                team_b=team_b,
                tournament=tournament,
                neutral_venue=neutral_venue,
            )

            probabilities = result["probabilities"]
            features_used = result["features_used"]

            row = {
                "team_a": result["team_a"],
                "team_b": result["team_b"],
                "predicted_label": result["predicted_label"],
                "proba_draw": probabilities.get("draw"),
                "proba_loss": probabilities.get("loss"),
                "proba_win": probabilities.get("win"),
                "team_a_snapshot_date": result["team_a_snapshot_date"],
                "team_b_snapshot_date": result["team_b_snapshot_date"],
                "team_a_elo_before": features_used.get("team_a_elo_before"),
                "team_b_elo_before": features_used.get("team_b_elo_before"),
                "elo_diff": features_used.get("elo_diff"),
                "abs_elo_diff": features_used.get("abs_elo_diff"),
                "team_a_expected_result": features_used.get("team_a_expected_result"),
                "team_b_expected_result": features_used.get("team_b_expected_result"),
                "tournament": result["tournament"],
                "neutral_venue": result["neutral_venue"],
                "status": "ok",
                "error_message": None,
            }
        except Exception as exc:
            row = {
                "team_a": team_a,
                "team_b": team_b,
                "predicted_label": None,
                "proba_draw": None,
                "proba_loss": None,
                "proba_win": None,
                "team_a_snapshot_date": None,
                "team_b_snapshot_date": None,
                "team_a_elo_before": None,
                "team_b_elo_before": None,
                "elo_diff": None,
                "abs_elo_diff": None,
                "team_a_expected_result": None,
                "team_b_expected_result": None,
                "tournament": tournament,
                "neutral_venue": neutral_venue,
                "status": "error",
                "error_message": str(exc),
            }

        rows.append(row)

    results_df = pd.DataFrame(rows)
    return results_df


def add_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple validation diagnostics to the prediction table.
    """
    df = df.copy()

    if "proba_draw" in df.columns and "proba_loss" in df.columns and "proba_win" in df.columns:
        df["probability_sum"] = (
            df["proba_draw"].fillna(0.0)
            + df["proba_loss"].fillna(0.0)
            + df["proba_win"].fillna(0.0)
        )

    if "proba_draw" in df.columns:
        df["high_draw_flag"] = df["proba_draw"].apply(
            lambda x: 1 if pd.notna(x) and x >= 0.40 else 0
        )

    if "elo_diff" in df.columns and "predicted_label" in df.columns:
        def flag_upset_like(row: pd.Series) -> int:
            if pd.isna(row["elo_diff"]) or pd.isna(row["predicted_label"]):
                return 0

            if row["elo_diff"] >= 100 and row["predicted_label"] == "loss":
                return 1
            if row["elo_diff"] <= -100 and row["predicted_label"] == "win":
                return 1
            return 0

        df["possible_upset_flag"] = df.apply(flag_upset_like, axis=1)

    return df


def print_summary(df: pd.DataFrame) -> None:
    """
    Print a compact validation summary.
    """
    print("\nPredictor validation summary")
    print("-" * 80)
    print(f"Rows evaluated: {len(df):,}")
    print(f"Successful predictions: {(df['status'] == 'ok').sum():,}")
    print(f"Errors: {(df['status'] == 'error').sum():,}")

    if "high_draw_flag" in df.columns:
        print(f"High draw cases (draw >= 0.40): {int(df['high_draw_flag'].sum()):,}")

    if "possible_upset_flag" in df.columns:
        print(f"Possible upset-like predictions: {int(df['possible_upset_flag'].sum()):,}")

    ok_df = df[df["status"] == "ok"].copy()

    if not ok_df.empty:
        print("\nAverage probabilities across successful predictions:")
        print(
            ok_df[["proba_draw", "proba_loss", "proba_win"]]
            .mean()
            .round(4)
            .to_string()
        )

        print("\nTop 10 highest draw probabilities:")
        top_draw = ok_df.sort_values("proba_draw", ascending=False).head(10)
        print(
            top_draw[
                [
                    "team_a",
                    "team_b",
                    "predicted_label",
                    "proba_draw",
                    "proba_loss",
                    "proba_win",
                    "elo_diff",
                ]
            ].to_string(index=False)
        )

        print("\nTop 10 largest Elo gaps:")
        top_elo_gap = ok_df.sort_values("abs_elo_diff", ascending=False).head(10)
        print(
            top_elo_gap[
                [
                    "team_a",
                    "team_b",
                    "predicted_label",
                    "proba_draw",
                    "proba_loss",
                    "proba_win",
                    "elo_diff",
                    "abs_elo_diff",
                ]
            ].to_string(index=False)
        )

    error_df = df[df["status"] == "error"].copy()
    if not error_df.empty:
        print("\nErrors detected:")
        print(
            error_df[
                ["team_a", "team_b", "error_message"]
            ].to_string(index=False)
        )


def save_results(df: pd.DataFrame, model_name: str) -> Path:
    """
    Save validation results to artifacts/metrics.
    """
    output_path = ARTIFACTS_DIR / "metrics" / f"{model_name}_predictor_validation.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    return output_path


def main() -> None:
    print(f"Running predictor validation using model: {DEFAULT_MODEL_NAME}")

    results_df = run_validation_suite(
        model_name=DEFAULT_MODEL_NAME,
        tournament=DEFAULT_TOURNAMENT,
        neutral_venue=DEFAULT_NEUTRAL_VENUE,
    )

    results_df = add_diagnostics(results_df)

    print("\nValidation results table:")
    display_cols = [
        "team_a",
        "team_b",
        "predicted_label",
        "proba_draw",
        "proba_loss",
        "proba_win",
        "elo_diff",
        "abs_elo_diff",
        "status",
    ]
    print(results_df[display_cols].to_string(index=False))

    print_summary(results_df)

    output_path = save_results(results_df, model_name=DEFAULT_MODEL_NAME)
    print(f"\nSaved validation results to: {output_path}")


if __name__ == "__main__":
    main()