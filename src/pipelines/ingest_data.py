import pandas as pd
import os
from pathlib import Path

DATA_URL = "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"


def download_matches():

    print("Downloading international matches dataset...")

    df = pd.read_csv(DATA_URL)

    return df


def clean_matches(df):

    print("Cleaning dataset...")

    df["date"] = pd.to_datetime(df["date"])

    df = df.rename(
        columns={
            "home_team": "team_a",
            "away_team": "team_b",
            "home_score": "goals_a",
            "away_score": "goals_b",
            "neutral": "neutral_venue",
        }
    )

    df["year"] = df["date"].dt.year

    df["goal_diff"] = df["goals_a"] - df["goals_b"]

    return df


def save_dataset(df):

    output_path = Path("data/interim")

    output_path.mkdir(parents=True, exist_ok=True)

    file_path = output_path / "matches_clean.parquet"

    df.to_parquet(file_path, index=False)

    print(f"Dataset saved to {file_path}")


def main():

    df = download_matches()

    df = clean_matches(df)

    save_dataset(df)


if __name__ == "__main__":
    main()