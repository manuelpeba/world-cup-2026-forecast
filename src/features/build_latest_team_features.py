from __future__ import annotations

import pandas as pd

from src.utils.config import PROCESSED_DATA_DIR


def load_filtered_matches() -> pd.DataFrame:
    """
    Load filtered team-level match dataset with rolling features.
    """
    file_path = PROCESSED_DATA_DIR / "matches_filtered.parquet"
    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_elo_ratings() -> pd.DataFrame:
    """
    Load team-level Elo ratings dataset.
    """
    file_path = PROCESSED_DATA_DIR / "team_elo_ratings.parquet"
    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def add_match_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a robust team-match key using the same logic as match_features.py.
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


def deduplicate_match_keys(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Drop exact duplicate rows based on match_key.
    """
    before = len(df)
    dup_count = df["match_key"].duplicated().sum()
    print(f"{name} duplicated match_key rows: {dup_count:,}")

    df = df.drop_duplicates(subset=["match_key"]).copy()
    after = len(df)

    if before != after:
        print(f"Dropped {before - after:,} duplicated rows from {name}.")

    return df


def merge_features_with_elo(
    matches_df: pd.DataFrame,
    elo_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge filtered team rolling features with Elo ratings using robust match_key.
    """
    matches_df = add_match_key(matches_df)
    elo_df = add_match_key(elo_df)

    matches_df = deduplicate_match_keys(matches_df, "matches_df")
    elo_df = deduplicate_match_keys(elo_df, "elo_df")

    feature_cols = [
        "match_key",
        "date",
        "team",
        "rolling_goals_scored",
        "rolling_goals_conceded",
        "rolling_goal_diff",
        "rolling_win_rate",
        "rolling_points",
    ]

    elo_cols = [
        "match_key",
        "elo_before",
    ]

    missing_feature_cols = [col for col in feature_cols if col not in matches_df.columns]
    if missing_feature_cols:
        raise KeyError(
            f"Missing feature columns in matches_filtered.parquet: {missing_feature_cols}"
        )

    missing_elo_cols = [col for col in elo_cols if col not in elo_df.columns]
    if missing_elo_cols:
        raise KeyError(
            f"Missing Elo columns in team_elo_ratings.parquet: {missing_elo_cols}"
        )

    merged = matches_df[feature_cols].merge(
        elo_df[elo_cols],
        on="match_key",
        how="inner",
        validate="one_to_one",
    )

    return merged


def extract_latest_team_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the most recent available row for each national team.
    """
    df = df.sort_values(["team", "date"]).copy()

    latest_df = (
        df.groupby("team", as_index=False)
        .tail(1)
        .sort_values(["date", "team"])
        .reset_index(drop=True)
    )

    selected_cols = [
        "team",
        "date",
        "elo_before",
        "rolling_goals_scored",
        "rolling_goals_conceded",
        "rolling_goal_diff",
        "rolling_win_rate",
        "rolling_points",
    ]

    latest_df = latest_df[selected_cols].copy()

    return latest_df


def run_quality_checks(df: pd.DataFrame) -> None:
    """
    Print basic checks for latest team feature snapshot.
    """
    print("\nQuality checks")
    print("-" * 50)
    print(f"Teams: {len(df):,}")
    print(f"Unique teams: {df['team'].nunique():,}")
    print(f"Snapshot date range: {df['date'].min().date()} -> {df['date'].max().date()}")

    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        print("\nMissing values detected:")
        print(missing.sort_values(ascending=False))
    else:
        print("\nNo missing values detected.")

    print("\nSample rows:")
    print(df.head(10))


def save_latest_features(df: pd.DataFrame) -> None:
    """
    Save latest team feature snapshot.
    """
    output_path = PROCESSED_DATA_DIR / "latest_team_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)

    print(f"\nSaved latest team features to: {output_path}")


def main() -> None:
    print("Loading filtered matches...")
    matches_df = load_filtered_matches()
    print(f"Filtered team-level rows: {len(matches_df):,}")

    print("Loading Elo ratings...")
    elo_df = load_elo_ratings()
    print(f"Elo team-level rows: {len(elo_df):,}")

    print("Merging rolling features with Elo...")
    merged_df = merge_features_with_elo(matches_df, elo_df)
    print(f"Merged rows: {len(merged_df):,}")

    print("Extracting latest team state...")
    latest_df = extract_latest_team_state(merged_df)

    run_quality_checks(latest_df)
    save_latest_features(latest_df)


if __name__ == "__main__":
    main()