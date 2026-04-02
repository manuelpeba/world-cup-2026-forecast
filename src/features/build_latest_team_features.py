from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.utils.helpers import normalize_team_name
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

    Priority:
    1. data/processed/elo_features.parquet
    2. data/processed/team_elo_ratings.parquet

    Expected minimum columns:
    - team
    - date
    - either:
        * elo_before
      or
        * elo
    """
    preferred_path = PROCESSED_DATA_DIR / "elo_features.parquet"
    fallback_path = PROCESSED_DATA_DIR / "team_elo_ratings.parquet"

    if preferred_path.exists():
        file_path = preferred_path
    elif fallback_path.exists():
        file_path = fallback_path
    else:
        raise FileNotFoundError(
            "No Elo dataset found. Expected one of:\n"
            f"- {preferred_path}\n"
            f"- {fallback_path}"
        )

    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"])

    if "team" not in df.columns:
        raise KeyError(f"'team' column not found in Elo dataset: {file_path}")

    if "date" not in df.columns:
        raise KeyError(f"'date' column not found in Elo dataset: {file_path}")

    # Normalize Elo column name
    if "elo_before" in df.columns:
        elo_col = "elo_before"
    elif "elo" in df.columns:
        elo_col = "elo"
    else:
        raise KeyError(
            f"Elo dataset {file_path} must contain either 'elo_before' or 'elo'."
        )

    df = df.loc[:, ["team", "date", elo_col]].copy()
    df = df.rename(columns={elo_col: "elo_before"})

    return df


def deduplicate_team_date_rows(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Drop exact duplicate rows based on (team, date, elo_before) or (team, date)
    and keep the last available value per team-date.

    This is useful before merge_asof because duplicate keys can create ambiguity.
    """
    before = len(df)

    if "elo_before" in df.columns:
        df = df.sort_values(["team", "date", "elo_before"]).copy()
    else:
        df = df.sort_values(["team", "date"]).copy()

    dup_count = df.duplicated(subset=["team", "date"]).sum()
    print(f"{name} duplicated (team, date) rows: {dup_count:,}")

    df = df.drop_duplicates(subset=["team", "date"], keep="last").copy()
    after = len(df)

    if before != after:
        print(f"Dropped {before - after:,} duplicated rows from {name}.")

    return df


def merge_features_with_elo(
    matches_df: pd.DataFrame,
    elo_df: pd.DataFrame,
    fallback_elo: float = 1500.0,
) -> pd.DataFrame:
    """
    Merge filtered team rolling features with Elo ratings using a temporal join.

    Strategy:
    - For each team and match date, attach the most recent available Elo rating
      on or before that date.
    - Uses merge_asof per team to avoid brittle exact match_key joins.

    Requirements:
    matches_df must contain:
    - team
    - date
    - rolling_goals_scored
    - rolling_goals_conceded
    - rolling_goal_diff
    - rolling_win_rate
    - rolling_points

    elo_df must contain:
    - team
    - date
    - elo_before
    """
    required_match_cols = [
        "team",
        "date",
        "rolling_goals_scored",
        "rolling_goals_conceded",
        "rolling_goal_diff",
        "rolling_win_rate",
        "rolling_points",
    ]
    missing_match_cols = [col for col in required_match_cols if col not in matches_df.columns]
    if missing_match_cols:
        raise KeyError(
            f"Missing required columns in matches_filtered.parquet: {missing_match_cols}"
        )

    required_elo_cols = ["team", "date", "elo_before"]
    missing_elo_cols = [col for col in required_elo_cols if col not in elo_df.columns]
    if missing_elo_cols:
        raise KeyError(
            f"Missing required columns in Elo dataset: {missing_elo_cols}"
        )

    matches_df = matches_df.copy()
    matches_df["team"] = matches_df["team"].apply(normalize_team_name)
    elo_df = elo_df.copy()
    elo_df["team"] = elo_df["team"].apply(normalize_team_name)

    matches_df["date"] = pd.to_datetime(matches_df["date"])
    elo_df["date"] = pd.to_datetime(elo_df["date"])

    matches_df = matches_df.sort_values(["team", "date"]).reset_index(drop=True)
    elo_df = elo_df.sort_values(["team", "date"]).reset_index(drop=True)

    elo_df = deduplicate_team_date_rows(elo_df, "elo_df")

    merged_parts: list[pd.DataFrame] = []

    match_teams = sorted(matches_df["team"].dropna().unique())

    for team in match_teams:
        team_matches = matches_df.loc[matches_df["team"] == team].copy()
        team_elo = elo_df.loc[elo_df["team"] == team, ["date", "elo_before"]].copy()

        team_matches = team_matches.sort_values("date")
        team_elo = team_elo.sort_values(by="date") # type: ignore

        if team_elo.empty:
            team_matches["elo_before"] = pd.NA
            merged_parts.append(team_matches)
            continue

        merged_team = pd.merge_asof(
            team_matches,
            team_elo,
            on="date",
            direction="backward",
        )

        merged_parts.append(merged_team)

    merged = pd.concat(merged_parts, axis=0, ignore_index=True)
    merged = merged.sort_values(["team", "date"]).reset_index(drop=True)

    missing_elo_teams = (
        merged.loc[merged["elo_before"].isna(), "team"]
        .dropna()
        .unique()
        .tolist()
    )

    if missing_elo_teams:
        print(
            "Teams without Elo found after merge. "
            f"Falling back to {fallback_elo:.1f} for: {sorted(missing_elo_teams)}"
        )

    merged["elo_before"] = merged["elo_before"].fillna(fallback_elo)

    return merged


def extract_latest_team_state(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep the most recent available row for each national team.
    """
    required_cols = [
        "team",
        "date",
        "elo_before",
        "rolling_goals_scored",
        "rolling_goals_conceded",
        "rolling_goal_diff",
        "rolling_win_rate",
        "rolling_points",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Missing required columns before extracting latest team state: {missing_cols}"
        )

    df = df.sort_values(["team", "date"]).copy()

    latest_df = (
        df.groupby("team", as_index=False)
        .tail(1)
        .sort_values(["date", "team"])
        .reset_index(drop=True)
    )

    latest_df = latest_df[required_cols].copy()

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