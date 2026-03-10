import pandas as pd
from pathlib import Path

from src.utils.config import INTERIM_DATA_DIR, PROCESSED_DATA_DIR


ROLLING_WINDOW = 10


def load_matches():

    file_path = INTERIM_DATA_DIR / "matches_clean.parquet"

    df = pd.read_parquet(file_path)

    df = df.sort_values("date")

    return df


def create_team_match_rows(df):

    """
    Transform matches into team-level rows.
    Each match becomes two rows:
    one for each team.
    """

    home = df.copy()
    home["team"] = home["team_a"]
    home["opponent"] = home["team_b"]
    home["goals_for"] = home["goals_a"]
    home["goals_against"] = home["goals_b"]

    away = df.copy()
    away["team"] = away["team_b"]
    away["opponent"] = away["team_a"]
    away["goals_for"] = away["goals_b"]
    away["goals_against"] = away["goals_a"]

    teams = pd.concat([home, away])

    teams["goal_diff"] = teams["goals_for"] - teams["goals_against"]

    teams["win"] = (teams["goals_for"] > teams["goals_against"]).astype(int)
    teams["draw"] = (teams["goals_for"] == teams["goals_against"]).astype(int)
    teams["loss"] = (teams["goals_for"] < teams["goals_against"]).astype(int)

    teams["points"] = teams["win"] * 3 + teams["draw"]

    return teams


def compute_rolling_features(df):

    df = df.sort_values(["team", "date"])

    df["rolling_goals_scored"] = (
        df.groupby("team")["goals_for"]
        .transform(lambda x: x.shift().rolling(ROLLING_WINDOW).mean())
    )

    df["rolling_goals_conceded"] = (
        df.groupby("team")["goals_against"]
        .transform(lambda x: x.shift().rolling(ROLLING_WINDOW).mean())
    )

    df["rolling_goal_diff"] = (
        df.groupby("team")["goal_diff"]
        .transform(lambda x: x.shift().rolling(ROLLING_WINDOW).mean())
    )

    df["rolling_win_rate"] = (
        df.groupby("team")["win"]
        .transform(lambda x: x.shift().rolling(ROLLING_WINDOW).mean())
    )

    df["rolling_points"] = (
        df.groupby("team")["points"]
        .transform(lambda x: x.shift().rolling(ROLLING_WINDOW).mean())
    )

    return df


def save_features(df):

    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DATA_DIR / "team_match_features.parquet"

    df.to_parquet(output_path, index=False)

    print(f"Saved team features to {output_path}")


def main():

    print("Loading matches dataset...")
    matches = load_matches()

    print("Creating team match rows...")
    teams = create_team_match_rows(matches)

    print("Computing rolling performance features...")
    teams = compute_rolling_features(teams)

    print("Saving dataset...")
    save_features(teams)


if __name__ == "__main__":
    main()