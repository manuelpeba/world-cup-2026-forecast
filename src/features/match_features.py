from __future__ import annotations

import pandas as pd

from src.utils.config import PROCESSED_DATA_DIR


def load_filtered_matches() -> pd.DataFrame:
    """Load filtered team-level match dataset with rolling features."""
    file_path = PROCESSED_DATA_DIR / "matches_filtered.parquet"
    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_elo_ratings() -> pd.DataFrame:
    """Load team-level Elo dataset."""
    file_path = PROCESSED_DATA_DIR / "team_elo_ratings.parquet"
    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_match_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a robust team-match key using only columns shared by both
    matches_filtered and team_elo_ratings.
    """
    df = df.copy()

    required_cols = [
        "date",
        "team",
        "opponent",
        "tournament",
        "goals_for",
        "goals_against",
        "neutral_venue",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Cannot build match_key. Missing columns: {missing_cols}"
        )

    df["match_key"] = (
        df["date"].astype(str)
        + "|"
        + df["team"].astype(str)
        + "|"
        + df["opponent"].astype(str)
        + "|"
        + df["tournament"].astype(str)
        + "|"
        + df["goals_for"].astype(str)
        + "|"
        + df["goals_against"].astype(str)
        + "|"
        + df["neutral_venue"].astype(str)
    )

    return df


def print_duplicate_diagnostics(df: pd.DataFrame, name: str) -> None:
    """Print duplicate diagnostics for match_key."""
    dup_count = df["match_key"].duplicated().sum()
    print(f"{name} duplicated match_key rows: {dup_count:,}")

    if dup_count > 0:
        dup_sample = df.loc[df["match_key"].duplicated(keep=False)].sort_values("match_key")
        print(f"\nSample duplicated rows in {name}:")
        print(
            dup_sample[
                [
                    "date",
                    "team",
                    "opponent",
                    "tournament",
                    "team_a",
                    "team_b",
                    "goals_for",
                    "goals_against",
                    "neutral_venue",
                    "match_key",
                ]
            ].head(10)
        )


def deduplicate_match_keys(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop exact duplicate team-match rows if they exist.
    Keeps first occurrence to preserve pipeline continuity.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["match_key"]).copy()
    after = len(df)

    if before != after:
        print(f"Dropped {before - after:,} duplicated rows based on match_key.")

    return df


def merge_team_features_with_elo(
    matches_df: pd.DataFrame, elo_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Merge rolling team features with Elo ratings at the team-match level
    using a robust match_key.
    """
    matches_df = add_match_key(matches_df)
    elo_df = add_match_key(elo_df)

    print_duplicate_diagnostics(matches_df, "matches_df")
    print_duplicate_diagnostics(elo_df, "elo_df")

    matches_df = deduplicate_match_keys(matches_df)
    elo_df = deduplicate_match_keys(elo_df)

    elo_cols = [
        "match_key",
        "elo_before",
        "elo_after",
        "opponent_elo_before",
        "opponent_elo_after",
        "elo_diff_before",
        "expected_result",
        "k_factor",
        "goal_diff_multiplier",
    ]

    merged = matches_df.merge(
        elo_df[elo_cols],
        on="match_key",
        how="inner",
        validate="one_to_one",
    )

    return merged


def build_match_level_dataset(team_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team-level dataset (two rows per match) into match-level dataset.
    One row per real match, with Team A and Team B features side by side.
    """
    team_a_df = team_df[team_df["team"] == team_df["team_a"]].copy()
    team_b_df = team_df[team_df["team"] == team_df["team_b"]].copy()

    key_cols = [
        "date",
        "team_a",
        "team_b",
        "goals_a",
        "goals_b",
        "tournament",
        "city",
        "country",
        "neutral_venue",
        "year",
    ]

    team_a_cols = key_cols + [
        "rolling_goals_scored",
        "rolling_goals_conceded",
        "rolling_goal_diff",
        "rolling_win_rate",
        "rolling_points",
        "elo_before",
        "elo_after",
        "opponent_elo_before",
        "elo_diff_before",
        "expected_result",
        "k_factor",
        "goal_diff_multiplier",
    ]

    team_b_cols = key_cols + [
        "rolling_goals_scored",
        "rolling_goals_conceded",
        "rolling_goal_diff",
        "rolling_win_rate",
        "rolling_points",
        "elo_before",
        "elo_after",
        "opponent_elo_before",
        "elo_diff_before",
        "expected_result",
        "k_factor",
        "goal_diff_multiplier",
    ]

    team_a_df = team_a_df[team_a_cols].rename(
        columns={
            "rolling_goals_scored": "team_a_rolling_goals_scored",
            "rolling_goals_conceded": "team_a_rolling_goals_conceded",
            "rolling_goal_diff": "team_a_rolling_goal_diff",
            "rolling_win_rate": "team_a_rolling_win_rate",
            "rolling_points": "team_a_rolling_points",
            "elo_before": "team_a_elo_before",
            "elo_after": "team_a_elo_after",
            "opponent_elo_before": "team_a_opponent_elo_before",
            "elo_diff_before": "team_a_elo_diff_before",
            "expected_result": "team_a_expected_result",
            "k_factor": "match_k_factor",
            "goal_diff_multiplier": "match_goal_diff_multiplier",
        }
    )

    team_b_df = team_b_df[team_b_cols].rename(
        columns={
            "rolling_goals_scored": "team_b_rolling_goals_scored",
            "rolling_goals_conceded": "team_b_rolling_goals_conceded",
            "rolling_goal_diff": "team_b_rolling_goal_diff",
            "rolling_win_rate": "team_b_rolling_win_rate",
            "rolling_points": "team_b_rolling_points",
            "elo_before": "team_b_elo_before",
            "elo_after": "team_b_elo_after",
            "opponent_elo_before": "team_b_opponent_elo_before",
            "elo_diff_before": "team_b_elo_diff_before",
            "expected_result": "team_b_expected_result",
            "k_factor": "match_k_factor_b",
            "goal_diff_multiplier": "match_goal_diff_multiplier_b",
        }
    )

    match_df = team_a_df.merge(
        team_b_df,
        on=key_cols,
        how="inner",
        validate="one_to_one",
    )

    return match_df


def create_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create differential features and classification target."""
    df = df.copy()

    df["neutral_venue"] = df["neutral_venue"].astype(int)

    df["elo_diff"] = df["team_a_elo_before"] - df["team_b_elo_before"]
    df["abs_elo_diff"] = df["elo_diff"].abs()

    df["rolling_goal_diff_diff"] = (
        df["team_a_rolling_goal_diff"] - df["team_b_rolling_goal_diff"]
    )
    df["rolling_points_diff"] = (
        df["team_a_rolling_points"] - df["team_b_rolling_points"]
    )
    df["rolling_win_rate_diff"] = (
        df["team_a_rolling_win_rate"] - df["team_b_rolling_win_rate"]
    )
    df["rolling_goals_scored_diff"] = (
        df["team_a_rolling_goals_scored"] - df["team_b_rolling_goals_scored"]
    )
    df["rolling_goals_conceded_diff"] = (
        df["team_a_rolling_goals_conceded"] - df["team_b_rolling_goals_conceded"]
    )

    def map_result(goals_a: int, goals_b: int) -> str:
        if goals_a > goals_b:
            return "win"
        if goals_a == goals_b:
            return "draw"
        return "loss"

    df["target"] = df.apply(
        lambda row: map_result(int(row["goals_a"]), int(row["goals_b"])),
        axis=1,
    )

    return df


def run_quality_checks(df: pd.DataFrame) -> None:
    """Basic checks before saving."""
    print("\nQuality checks")
    print("-" * 50)
    print(f"Rows: {len(df):,}")
    print(f"Unique teams (team_a): {df['team_a'].nunique():,}")
    print(f"Date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    print("\nTarget distribution:")
    print(df["target"].value_counts(normalize=True).round(4))

    print("\nneutral_venue distribution:")
    print(df["neutral_venue"].value_counts(dropna=False))

    print("\nDerived feature nulls:")
    print(
        df[
            [
                "elo_diff",
                "abs_elo_diff",
                "rolling_goal_diff_diff",
                "rolling_points_diff",
                "rolling_win_rate_diff",
                "rolling_goals_scored_diff",
                "rolling_goals_conceded_diff",
            ]
        ].isna().sum()
    )

    print("\nDerived feature sample:")
    print(
        df[
            [
                "team_a",
                "team_b",
                "team_a_elo_before",
                "team_b_elo_before",
                "elo_diff",
                "abs_elo_diff",
                "neutral_venue",
                "target",
            ]
        ].head()
    )

    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        print("\nMissing values detected:")
        print(missing.sort_values(ascending=False))
    else:
        print("\nNo missing values detected.")


def save_match_model_dataset(df: pd.DataFrame) -> None:
    """Save final match-level modeling dataset."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "match_model_dataset.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved match model dataset to: {output_path}")


def main() -> None:
    print("Loading filtered matches...")
    matches_df = load_filtered_matches()
    print(f"Filtered team-level rows: {len(matches_df):,}")

    print("Loading Elo ratings...")
    elo_df = load_elo_ratings()
    print(f"Elo team-level rows: {len(elo_df):,}")

    print("Merging rolling team features with Elo...")
    team_df = merge_team_features_with_elo(matches_df, elo_df)
    print(f"Merged team-level rows: {len(team_df):,}")

    print("Building match-level dataset...")
    match_df = build_match_level_dataset(team_df)
    print(f"Match-level rows: {len(match_df):,}")

    print("Creating model features and target...")
    match_df = create_model_features(match_df)

    run_quality_checks(match_df)
    save_match_model_dataset(match_df)


if __name__ == "__main__":
    main()