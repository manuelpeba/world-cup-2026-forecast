from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from src.utils.config import PROCESSED_DATA_DIR


MIN_YEAR = 1950

# Tournaments considered relevant for estimating international team strength
ALLOWED_TOURNAMENTS = {
    "Friendly",
    "FIFA World Cup",
    "FIFA World Cup qualification",
    "UEFA Euro",
    "UEFA Euro qualification",
    "UEFA Nations League",
    "AFC Asian Cup",
    "AFC Asian Cup qualification",
    "African Cup of Nations",
    "African Cup of Nations qualification",
    "Copa América",
    "CONCACAF Championship",
    "Gold Cup",
    "CONCACAF Nations League",
    "Confederations Cup",
    "Nehru Cup",
    "Kirin Cup",
    "China Cup",
    "COSAFA Cup",
    "SAFF Cup",
    "WAFF Championship",
    "EAFF Championship",
    "CFU Caribbean Cup",
    "CFU Caribbean Cup qualification",
    "UNCAF Cup",
    "AFC Challenge Cup",
    "AFC Solidarity Cup",
    "Arab Cup",
}


def load_team_match_features() -> pd.DataFrame:
    """Load the team-level feature dataset generated in the previous step."""
    file_path = PROCESSED_DATA_DIR / "team_match_features.parquet"
    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_allowed_teams() -> set[str]:
    """
    Load the allowlist of valid national teams from YAML config.
    Expected format:

    teams:
      - Argentina
      - Brazil
      - France
    """
    config_path = Path("configs/allowed_teams.yaml")

    if not config_path.exists():
        raise FileNotFoundError(
            f"Allowed teams config not found: {config_path}. "
            "Create configs/allowed_teams.yaml before running this pipeline."
        )

    with open(config_path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file)

    if not data or "teams" not in data:
        raise ValueError(
            "Invalid allowed_teams.yaml format. Expected a top-level key called 'teams'."
        )

    teams = data["teams"]

    if not isinstance(teams, list):
        raise ValueError(
            "Invalid allowed_teams.yaml format. 'teams' must be a list of team names."
        )

    return {team.strip() for team in teams if isinstance(team, str) and team.strip()}


def filter_by_year(df: pd.DataFrame, min_year: int = MIN_YEAR) -> pd.DataFrame:
    """Keep only matches from a minimum year onward."""
    return df.loc[df["date"].dt.year >= min_year].copy()


def filter_by_tournament(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only tournaments relevant for international team strength estimation."""
    return df.loc[df["tournament"].isin(ALLOWED_TOURNAMENTS)].copy()


def filter_allowed_teams(df: pd.DataFrame) -> pd.DataFrame:
    """
    Keep only rows where both the team and the opponent belong
    to the curated allowlist of valid national teams.
    """
    allowed_teams = load_allowed_teams()

    mask = (
        df["team"].isin(allowed_teams)
        & df["opponent"].isin(allowed_teams)
    )

    return df.loc[mask].copy()


def drop_missing_rolling_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rolling features require previous matches. The first rows per team will
    contain NaNs after shift(). Those rows are not usable for predictive models.
    """
    rolling_cols = [
        "rolling_goals_scored",
        "rolling_goals_conceded",
        "rolling_goal_diff",
        "rolling_win_rate",
        "rolling_points",
    ]
    return df.dropna(subset=rolling_cols).copy()


def standardize_and_sort(df: pd.DataFrame) -> pd.DataFrame:
    """Final consistency cleanup before saving."""
    return df.sort_values(["date", "team", "opponent"]).reset_index(drop=True)


def print_filtering_summary(before_df: pd.DataFrame, after_df: pd.DataFrame) -> None:
    """Print a summary of the filtering impact."""
    print("\nFiltering summary")
    print("-" * 50)
    print(f"Rows before filtering: {len(before_df):,}")
    print(f"Rows after filtering:  {len(after_df):,}")
    print(f"Rows removed:          {len(before_df) - len(after_df):,}")
    print(f"Teams remaining:       {after_df['team'].nunique():,}")
    print(f"Tournaments remaining: {after_df['tournament'].nunique():,}")

    if not after_df.empty:
        print(
            f"Date range:            "
            f"{after_df['date'].min().date()} -> {after_df['date'].max().date()}"
        )
    else:
        print("Date range:            <empty dataset>")


def save_filtered_matches(df: pd.DataFrame) -> None:
    """Save the cleaned and filtered dataset."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "matches_filtered.parquet"
    df.to_parquet(output_path, index=False)
    print(f"\nSaved filtered dataset to: {output_path}")


def main() -> None:
    print("Loading team match features...")
    df = load_team_match_features()
    before_df = df.copy()

    print(f"Initial rows: {len(df):,}")

    print("Filtering by year...")
    df = filter_by_year(df, min_year=MIN_YEAR)
    print(f"Rows after year filter: {len(df):,}")

    print("Filtering by tournament...")
    df = filter_by_tournament(df)
    print(f"Rows after tournament filter: {len(df):,}")

    print("Filtering allowed national teams...")
    df = filter_allowed_teams(df)
    print(f"Rows after allowed teams filter: {len(df):,}")

    print("Dropping rows without rolling history...")
    df = drop_missing_rolling_rows(df)
    print(f"Rows after dropping missing rolling features: {len(df):,}")

    print("Sorting final dataset...")
    df = standardize_and_sort(df)

    print_filtering_summary(before_df, df)
    save_filtered_matches(df)


if __name__ == "__main__":
    main()