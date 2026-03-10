from __future__ import annotations

from typing import Dict, List

import pandas as pd

from src.utils.config import PROCESSED_DATA_DIR


BASE_ELO = 1500.0
HOME_ADVANTAGE_ELO = 80.0


def load_matches() -> pd.DataFrame:
    """Load filtered team-level match dataset."""
    file_path = PROCESSED_DATA_DIR / "matches_filtered.parquet"
    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"])
    return df.sort_values(["date", "team", "opponent"]).reset_index(drop=True)


def get_tournament_k_factor(tournament: str) -> float:
    """Return K-factor based on tournament importance."""
    if tournament == "Friendly":
        return 20.0

    if "qualification" in tournament.lower():
        return 30.0

    if tournament in {
        "UEFA Nations League",
        "CONCACAF Nations League",
        "CFU Caribbean Cup",
        "CFU Caribbean Cup qualification",
        "COSAFA Cup",
        "SAFF Cup",
        "UNCAF Cup",
        "Arab Cup",
        "EAFF Championship",
        "WAFF Championship",
    }:
        return 35.0

    if tournament in {
        "UEFA Euro",
        "Copa América",
        "African Cup of Nations",
        "AFC Asian Cup",
        "Gold Cup",
        "CONCACAF Championship",
        "Confederations Cup",
    }:
        return 40.0

    if tournament == "FIFA World Cup":
        return 50.0

    return 25.0


def get_goal_difference_multiplier(goals_for: int, goals_against: int) -> float:
    """Return a multiplier based on goal difference."""
    goal_diff = abs(goals_for - goals_against)

    if goal_diff <= 1:
        return 1.0
    if goal_diff == 2:
        return 1.5
    return 1.75


def get_actual_result(goals_for: int, goals_against: int) -> float:
    """Convert match result to Elo score."""
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def expected_result(team_elo: float, opp_elo: float) -> float:
    """Compute expected result using Elo formula."""
    return 1.0 / (1.0 + 10 ** ((opp_elo - team_elo) / 400.0))


def build_match_level_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert team-level dataset into unique match-level rows.
    Since matches_filtered is team-level (two rows per match),
    we reconstruct one row per real match.
    """
    match_cols = [
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

    matches = (
        df[match_cols]
        .drop_duplicates()
        .sort_values(["date", "team_a", "team_b"])
        .reset_index(drop=True)
    )

    return matches


def compute_elo_ratings(matches: pd.DataFrame) -> pd.DataFrame:
    """Compute dynamic Elo ratings match by match."""
    ratings: Dict[str, float] = {}
    records: List[dict] = []

    for row in matches.itertuples(index=False):
        team_a = row.team_a
        team_b = row.team_b
        goals_a = int(row.goals_a)
        goals_b = int(row.goals_b)
        tournament = row.tournament
        neutral_venue = bool(row.neutral_venue)

        elo_a_before = ratings.get(team_a, BASE_ELO)
        elo_b_before = ratings.get(team_b, BASE_ELO)

        elo_a_adjusted = elo_a_before
        elo_b_adjusted = elo_b_before

        if not neutral_venue:
            elo_a_adjusted += HOME_ADVANTAGE_ELO

        expected_a = expected_result(elo_a_adjusted, elo_b_adjusted)
        expected_b = 1.0 - expected_a

        actual_a = get_actual_result(goals_a, goals_b)
        actual_b = 1.0 - actual_a if actual_a != 0.5 else 0.5

        k = get_tournament_k_factor(tournament)
        g = get_goal_difference_multiplier(goals_a, goals_b)

        elo_a_after = elo_a_before + k * g * (actual_a - expected_a)
        elo_b_after = elo_b_before + k * g * (actual_b - expected_b)

        ratings[team_a] = elo_a_after
        ratings[team_b] = elo_b_after

        records.append(
            {
                "date": row.date,
                "team": team_a,
                "opponent": team_b,
                "tournament": tournament,
                "neutral_venue": neutral_venue,
                "goals_for": goals_a,
                "goals_against": goals_b,
                "actual_result": actual_a,
                "expected_result": expected_a,
                "k_factor": k,
                "goal_diff_multiplier": g,
                "elo_before": elo_a_before,
                "elo_after": elo_a_after,
                "opponent_elo_before": elo_b_before,
                "opponent_elo_after": elo_b_after,
            }
        )

        records.append(
            {
                "date": row.date,
                "team": team_b,
                "opponent": team_a,
                "tournament": tournament,
                "neutral_venue": neutral_venue,
                "goals_for": goals_b,
                "goals_against": goals_a,
                "actual_result": actual_b,
                "expected_result": expected_b,
                "k_factor": k,
                "goal_diff_multiplier": g,
                "elo_before": elo_b_before,
                "elo_after": elo_b_after,
                "opponent_elo_before": elo_a_before,
                "opponent_elo_after": elo_a_after,
            }
        )

    elo_df = pd.DataFrame(records).sort_values(["date", "team"]).reset_index(drop=True)
    elo_df["elo_change"] = elo_df["elo_after"] - elo_df["elo_before"]
    elo_df["elo_diff_before"] = elo_df["elo_before"] - elo_df["opponent_elo_before"]

    return elo_df


def save_elo_dataset(df: pd.DataFrame) -> None:
    """Save Elo ratings dataset."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PROCESSED_DATA_DIR / "team_elo_ratings.parquet"
    df.to_parquet(output_path, index=False)
    print(f"Saved Elo ratings dataset to: {output_path}")


def main() -> None:
    print("Loading filtered team-level matches...")
    df = load_matches()
    print(f"Team-level rows loaded: {len(df):,}")

    print("Building unique match-level dataset...")
    matches = build_match_level_dataset(df)
    print(f"Match-level rows: {len(matches):,}")

    print("Computing dynamic Elo ratings...")
    elo_df = compute_elo_ratings(matches)
    print(f"Elo rows generated: {len(elo_df):,}")

    print("Saving Elo dataset...")
    save_elo_dataset(elo_df)

    print("\nElo computation completed.")
    print(f"Teams rated: {elo_df['team'].nunique():,}")
    print(
        f"Date range: {elo_df['date'].min().date()} -> "
        f"{elo_df['date'].max().date()}"
    )


if __name__ == "__main__":
    main()