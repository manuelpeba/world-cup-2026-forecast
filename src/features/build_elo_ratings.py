from __future__ import annotations

from src.utils.helpers import normalize_team_name
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.utils.config import PROCESSED_DATA_DIR


BASE_ELO = 1500.0
DEFAULT_K = 20.0


@dataclass(slots=True)
class EloConfig:
    """
    Configuration for Elo rating generation.
    """

    base_elo: float = BASE_ELO
    default_k: float = DEFAULT_K


def load_filtered_matches() -> pd.DataFrame:
    """
    Load the filtered team-level dataset used for downstream feature generation.

    Expected minimum columns:
    - date
    - team
    - opponent
    - tournament
    - goals_for
    - goals_against
    """
    file_path = PROCESSED_DATA_DIR / "matches_filtered.parquet"

    if not file_path.exists():
        raise FileNotFoundError(
            f"Filtered matches file not found: {file_path}. "
            "Run src.features.team_filtering first."
        )

    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"])

    required_cols = [
        "date",
        "team",
        "opponent",
        "tournament",
        "goals_for",
        "goals_against",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"matches_filtered.parquet is missing required columns: {missing_cols}"
        )
    
    df = pd.read_parquet(file_path)
    df["date"] = pd.to_datetime(df["date"])

    # ✅ NORMALIZE TEAM NAMES EARLY: this ensures consistent matching with Elo dataset and prevents issues downstream.
    df["team"] = df["team"].apply(normalize_team_name)
    df["opponent"] = df["opponent"].apply(normalize_team_name)

    return df


def build_match_key(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a robust key to identify a team-level row belonging to a specific match.

    This is not required for the final latest feature merge anymore, but it is
    useful here to deduplicate source rows and keep Elo generation stable.
    """
    df = df.copy()

    required_cols = [
        "date",
        "team",
        "opponent",
        "tournament",
        "goals_for",
        "goals_against",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Cannot build match_key. Missing columns: {missing_cols}")

    neutral_col = "neutral_venue" if "neutral_venue" in df.columns else None

    if neutral_col is not None:
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
            + df[neutral_col].astype(str)
        )
    else:
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
        )

    return df


def deduplicate_source_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove exact duplicate team-level source rows before Elo generation.
    """
    df = build_match_key(df)
    before = len(df)
    dup_count = df["match_key"].duplicated().sum()
    print(f"Duplicated source match_key rows: {dup_count:,}")

    df = df.drop_duplicates(subset=["match_key"]).copy()
    after = len(df)

    if before != after:
        print(f"Dropped {before - after:,} duplicated source rows.")

    return df.sort_values(["date", "team", "opponent"]).reset_index(drop=True)


def expected_score(elo_a: float, elo_b: float) -> float:
    """
    Standard Elo expected score.
    """
    return 1.0 / (1.0 + 10.0 ** ((elo_b - elo_a) / 400.0))


def actual_score(goals_for: int | float, goals_against: int | float) -> float:
    """
    Convert match result into Elo score:
    - win  -> 1.0
    - draw -> 0.5
    - loss -> 0.0
    """
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def k_factor_for_tournament(tournament: str, default_k: float = DEFAULT_K) -> float:
    """
    Tournament-aware K-factor.

    This is intentionally simple but more realistic than a flat K.
    """
    if tournament == "FIFA World Cup":
        return 60.0
    if tournament == "FIFA World Cup qualification":
        return 40.0
    if tournament in {
        "UEFA Euro",
        "Copa América",
        "African Cup of Nations",
        "AFC Asian Cup",
        "Gold Cup",
        "Confederations Cup",
    }:
        return 35.0
    if "qualification" in str(tournament).lower():
        return 30.0
    if tournament == "Friendly":
        return 20.0

    return default_k


def goal_diff_multiplier(goals_for: int | float, goals_against: int | float) -> float:
    """
    Mild goal-difference multiplier for Elo updates.

    This keeps the system stable while rewarding clearer wins.
    """
    diff = abs(float(goals_for) - float(goals_against))

    if diff <= 1:
        return 1.0
    if diff == 2:
        return 1.5
    if diff == 3:
        return 1.75
    return 1.75 + (diff - 3.0) / 8.0


def extract_canonical_matches(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = [
        "date",
        "team",
        "opponent",
        "tournament",
        "goals_for",
        "goals_against",
    ]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise KeyError(
            f"Cannot extract canonical matches. Missing columns: {missing_cols}"
        )

    # ⚠️ FIX CLAVE: eliminar columnas legacy si existen
    cols_to_drop = ["team_a", "team_b", "goals_a", "goals_b"]
    existing = [c for c in cols_to_drop if c in df.columns]
    if existing:
        df = df.drop(columns=existing)

    canonical = df.loc[df["team"] < df["opponent"]].copy()

    canonical = canonical.rename(
        columns={
            "team": "team_a",
            "opponent": "team_b",
            "goals_for": "goals_a",
            "goals_against": "goals_b",
        }
    )

    keep_cols = ["date", "team_a", "team_b", "tournament", "goals_a", "goals_b"]
    if "neutral_venue" in canonical.columns:
        keep_cols.append("neutral_venue")

    canonical = canonical.loc[:, keep_cols].copy()

    canonical = canonical.sort_values(["date", "team_a", "team_b"]).reset_index(drop=True)

    print("Unique teams after normalization:", df["team"].nunique())

    return canonical


def build_elo_ratings(df: pd.DataFrame, config: EloConfig) -> pd.DataFrame:
    """
    Build team-level Elo states from canonical match history.

    Output is team-level:
    one row per team per match, containing Elo before and after the match.
    """
    canonical_matches = extract_canonical_matches(df)

    current_elo: dict[str, float] = {}
    records: list[dict[str, float | int | str | pd.Timestamp | bool]] = []

    for _, row in canonical_matches.iterrows():
        team_a = str(row["team_a"])
        team_b = str(row["team_b"])
        date = pd.to_datetime(row["date"])
        tournament = str(row["tournament"])
        goals_a = int(row["goals_a"])
        goals_b = int(row["goals_b"])

        neutral_venue = bool(row["neutral_venue"]) if "neutral_venue" in row else True

        elo_a_before = current_elo.get(team_a, config.base_elo)
        elo_b_before = current_elo.get(team_b, config.base_elo)

        exp_a = expected_score(elo_a_before, elo_b_before)
        exp_b = expected_score(elo_b_before, elo_a_before)

        score_a = actual_score(goals_a, goals_b)
        score_b = 1.0 - score_a if score_a != 0.5 else 0.5

        k = k_factor_for_tournament(tournament=tournament, default_k=config.default_k)
        g = goal_diff_multiplier(goals_a, goals_b)

        elo_change_a = k * g * (score_a - exp_a)
        elo_change_b = k * g * (score_b - exp_b)

        elo_a_after = elo_a_before + elo_change_a
        elo_b_after = elo_b_before + elo_change_b

        records.append(
            {
                "date": date,
                "team": team_a,
                "opponent": team_b,
                "tournament": tournament,
                "neutral_venue": neutral_venue,
                "goals_for": goals_a,
                "goals_against": goals_b,
                "actual_result": score_a,
                "expected_result": exp_a,
                "k_factor": k,
                "goal_diff_multiplier": g,
                "elo_before": elo_a_before,
                "elo_after": elo_a_after,
                "opponent_elo_before": elo_b_before,
                "opponent_elo_after": elo_b_after,
                "elo_change": elo_change_a,
                "elo_diff_before": elo_a_before - elo_b_before,
            }
        )

        records.append(
            {
                "date": date,
                "team": team_b,
                "opponent": team_a,
                "tournament": tournament,
                "neutral_venue": neutral_venue,
                "goals_for": goals_b,
                "goals_against": goals_a,
                "actual_result": score_b,
                "expected_result": exp_b,
                "k_factor": k,
                "goal_diff_multiplier": g,
                "elo_before": elo_b_before,
                "elo_after": elo_b_after,
                "opponent_elo_before": elo_a_before,
                "opponent_elo_after": elo_a_after,
                "elo_change": elo_change_b,
                "elo_diff_before": elo_b_before - elo_a_before,
            }
        )

        current_elo[team_a] = elo_a_after
        current_elo[team_b] = elo_b_after

    elo_df = pd.DataFrame(records)
    elo_df = elo_df.sort_values(["date", "team", "opponent"]).reset_index(drop=True)

    return elo_df


def run_quality_checks(elo_df: pd.DataFrame) -> None:
    """
    Print compact quality checks for generated Elo features.
    """
    print("\nQuality checks")
    print("-" * 50)
    print(f"Rows: {len(elo_df):,}")
    print(f"Teams: {elo_df['team'].nunique():,}")
    print(
        f"Date range: {elo_df['date'].min().date()} -> "
        f"{elo_df['date'].max().date()}"
    )

    missing = elo_df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) > 0:
        print("\nMissing values detected:")
        print(missing.sort_values(ascending=False))
    else:
        print("\nNo missing values detected.")

    print("\nLatest top 20 teams by Elo:")
    latest_elo = (
        elo_df.sort_values(["team", "date"])
        .groupby("team", as_index=False)
        .tail(1)
        .sort_values("elo_after", ascending=False)
        .reset_index(drop=True)
    )
    print(latest_elo.loc[:, ["team", "elo_after"]].head(20).to_string(index=False))


def save_elo_features(elo_df: pd.DataFrame) -> Path:
    """
    Save team-level Elo dataset for downstream feature joining.
    """
    output_path = PROCESSED_DATA_DIR / "elo_features.parquet"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    elo_df.to_parquet(output_path, index=False)
    print(f"\nSaved Elo features to: {output_path}")
    return output_path


def main() -> None:
    print("Loading filtered matches...")
    df = load_filtered_matches()
    print(f"Team-level rows loaded: {len(df):,}")

    print("Deduplicating source rows...")
    df = deduplicate_source_rows(df)
    print(f"Rows after deduplication: {len(df):,}")

    print("Building Elo ratings...")
    config = EloConfig()
    elo_df = build_elo_ratings(df=df, config=config)
    print(f"Generated Elo rows: {len(elo_df):,}")

    run_quality_checks(elo_df)
    save_elo_features(elo_df)


if __name__ == "__main__":
    main()