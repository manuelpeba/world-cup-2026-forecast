from __future__ import annotations

from typing import Any

import pandas as pd

from src.simulation.structures import TournamentRunResult
from src.simulation.tournament import extract_stage_presence_flags


DEFAULT_STAGE_ORDER = [
    "group_stage_exit",
    "round_of_32",
    "round_of_16",
    "quarterfinal",
    "semifinal",
    "final",
    "champion",
]


def validate_simulation_results(
    simulation_results: list[TournamentRunResult],
) -> None:
    """
    Validate simulation outputs before aggregation.
    """
    if not simulation_results:
        raise ValueError("simulation_results cannot be empty.")

    simulation_ids = [result.simulation_id for result in simulation_results]
    if len(simulation_ids) != len(set(simulation_ids)):
        raise ValueError("simulation_results contains duplicated simulation_id values.")


def extract_all_teams(
    simulation_results: list[TournamentRunResult],
) -> list[str]:
    """
    Extract the full set of teams participating in the simulations.

    Teams are inferred from group tables stored in each tournament run.
    """
    validate_simulation_results(simulation_results)

    teams: set[str] = set()

    for run_result in simulation_results:
        for group_rows in run_result.group_tables.values():
            for row in group_rows:
                teams.add(row.team)

    if not teams:
        raise ValueError("No teams could be extracted from simulation_results.")

    return sorted(teams)


def infer_available_stage_columns(
    stage_presence_df: pd.DataFrame,
) -> list[str]:
    """
    Infer which stage columns are available in a stage presence dataframe.

    Supports both:
        - v1: group_stage_exit, round_of_16, quarterfinal, semifinal, final, champion
        - v2: group_stage_exit, round_of_32, round_of_16, quarterfinal, semifinal, final, champion
    """
    return [col for col in DEFAULT_STAGE_ORDER if col in stage_presence_df.columns]


def build_stage_presence_dataframe(
    simulation_results: list[TournamentRunResult],
) -> pd.DataFrame:
    """
    Build a long-format dataframe with one row per team per simulation.

    Output columns always include:
        - simulation_id
        - team

    Stage columns depend on tournament format:
        v1:
            - group_stage_exit
            - round_of_16
            - quarterfinal
            - semifinal
            - final
            - champion

        v2:
            - group_stage_exit
            - round_of_32
            - round_of_16
            - quarterfinal
            - semifinal
            - final
            - champion
    """
    validate_simulation_results(simulation_results)
    all_teams = extract_all_teams(simulation_results)

    records: list[dict[str, Any]] = []

    for run_result in simulation_results:
        records.extend(
            extract_stage_presence_flags(
                run_result=run_result,
                all_teams=all_teams,
            )
        )

    if not records:
        raise ValueError("No stage-presence records were generated.")

    df = pd.DataFrame(records)

    base_required_columns = {"simulation_id", "team", "group_stage_exit", "champion"}
    missing_base_columns = base_required_columns - set(df.columns)
    if missing_base_columns:
        raise ValueError(
            "Stage presence dataframe is missing required base columns: "
            f"{sorted(missing_base_columns)}"
        )

    available_stage_columns = infer_available_stage_columns(df)

    if "round_of_16" not in available_stage_columns:
        raise ValueError(
            "Stage presence dataframe must contain at least 'round_of_16'."
        )

    ordered_columns = ["simulation_id", "team"] + available_stage_columns
    df = df.loc[:, ordered_columns]

    return df.sort_values(["team", "simulation_id"]).reset_index(drop=True)


def aggregate_stage_probabilities(
    stage_presence_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate team-level stage presence flags into probabilities.

    Since stage flags are binary, their mean across simulations equals the
    estimated probability of reaching that stage.
    """
    required_columns = {"team", "group_stage_exit", "champion"}
    missing_columns = required_columns - set(stage_presence_df.columns)
    if missing_columns:
        raise ValueError(
            f"stage_presence_df is missing required columns: {sorted(missing_columns)}"
        )

    stage_columns = infer_available_stage_columns(stage_presence_df)
    aggregation_columns = ["team"] + stage_columns

    probability_table = (
        stage_presence_df.loc[:, aggregation_columns]
        .groupby("team", as_index=False)[stage_columns]
        .mean()
    )

    return probability_table


def build_team_probability_table(
    simulation_results: list[TournamentRunResult],
) -> pd.DataFrame:
    """
    Build the main team probability table for forecasting outputs.

    Output columns depend on tournament format.

    v1 example:
        - team
        - group_stage_exit_prob
        - round_of_16_prob
        - quarterfinal_prob
        - semifinal_prob
        - final_prob
        - champion_prob

    v2 example:
        - team
        - group_stage_exit_prob
        - round_of_32_prob
        - round_of_16_prob
        - quarterfinal_prob
        - semifinal_prob
        - final_prob
        - champion_prob
    """
    stage_presence_df = build_stage_presence_dataframe(simulation_results)
    probability_table = aggregate_stage_probabilities(stage_presence_df)

    rename_map = {
        col: f"{col}_prob"
        for col in infer_available_stage_columns(stage_presence_df)
    }

    probability_table = probability_table.rename(columns=rename_map)

    sort_priority = [
        col for col in [
            "champion_prob",
            "final_prob",
            "semifinal_prob",
            "quarterfinal_prob",
            "round_of_16_prob",
            "round_of_32_prob",
        ]
        if col in probability_table.columns
    ]

    probability_table = probability_table.sort_values(
        by=sort_priority,
        ascending=False,
    ).reset_index(drop=True)

    return probability_table


def add_advancement_complements(
    probability_table: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add complementary advancement metrics useful for dashboards.
    """
    required_columns = {
        "group_stage_exit_prob",
        "champion_prob",
    }
    missing_columns = required_columns - set(probability_table.columns)
    if missing_columns:
        raise ValueError(
            f"probability_table is missing required columns: {sorted(missing_columns)}"
        )

    df = probability_table.copy()
    df["advance_from_group_prob"] = 1.0 - df["group_stage_exit_prob"]
    df["non_champion_prob"] = 1.0 - df["champion_prob"]

    return df


def build_champion_distribution(
    simulation_results: list[TournamentRunResult],
) -> pd.DataFrame:
    """
    Build a champion frequency / probability table.
    """
    validate_simulation_results(simulation_results)

    champions = [result.champion for result in simulation_results if result.champion]
    if not champions:
        raise ValueError("No champions found in simulation_results.")

    champion_df = (
        pd.Series(champions, name="team")
        .value_counts(dropna=False)
        .reset_index()
    )
    champion_df.columns = ["team", "titles"]
    champion_df["champion_prob"] = champion_df["titles"] / len(simulation_results)

    return champion_df.sort_values(
        by=["champion_prob", "titles"],
        ascending=False,
    ).reset_index(drop=True)


def build_stage_counts_table(
    simulation_results: list[TournamentRunResult],
) -> pd.DataFrame:
    """
    Build a count-based table instead of probabilities.

    Useful for QA, debugging, or sanity checks before normalizing.
    """
    stage_presence_df = build_stage_presence_dataframe(simulation_results)
    stage_columns = infer_available_stage_columns(stage_presence_df)

    counts_table = (
        stage_presence_df
        .groupby("team", as_index=False)[stage_columns]
        .sum()
    )

    rename_map = {
        col: f"{col}_count"
        for col in stage_columns
    }
    counts_table = counts_table.rename(columns=rename_map)
    counts_table["num_simulations"] = len(simulation_results)

    sort_priority = [
        col for col in [
            "champion_count",
            "final_count",
            "semifinal_count",
            "quarterfinal_count",
            "round_of_16_count",
            "round_of_32_count",
        ]
        if col in counts_table.columns
    ]

    return counts_table.sort_values(
        by=sort_priority,
        ascending=False,
    ).reset_index(drop=True)


def aggregate_simulation_results(
    simulation_results: list[TournamentRunResult],
    include_complements: bool = True,
) -> dict[str, pd.DataFrame]:
    """
    Main aggregation entry point.

    Returns:
        - stage_presence
        - stage_counts
        - team_probabilities
        - champion_distribution
    """
    stage_presence_df = build_stage_presence_dataframe(simulation_results)
    stage_counts_df = build_stage_counts_table(simulation_results)
    team_probabilities_df = build_team_probability_table(simulation_results)
    champion_distribution_df = build_champion_distribution(simulation_results)

    if include_complements:
        team_probabilities_df = add_advancement_complements(team_probabilities_df)

    return {
        "stage_presence": stage_presence_df,
        "stage_counts": stage_counts_df,
        "team_probabilities": team_probabilities_df,
        "champion_distribution": champion_distribution_df,
    }


def round_probability_columns(
    probability_table: pd.DataFrame,
    decimals: int = 4,
) -> pd.DataFrame:
    """
    Round probability columns for cleaner presentation.
    """
    df = probability_table.copy()

    probability_columns = [
        col for col in df.columns
        if col.endswith("_prob")
    ]

    for col in probability_columns:
        df[col] = df[col].round(decimals)

    return df


def build_summary_metadata(
    simulation_results: list[TournamentRunResult],
) -> pd.DataFrame:
    """
    Build a compact metadata summary from the first simulation run.

    Assumes all runs were generated under the same simulation configuration.
    """
    validate_simulation_results(simulation_results)

    first_result = simulation_results[0]
    metadata = dict(first_result.metadata or {})

    metadata["num_simulations"] = len(simulation_results)
    metadata["num_teams"] = len(extract_all_teams(simulation_results))

    return pd.DataFrame([metadata])


def build_match_log_dataframe(
    simulation_results: list[TournamentRunResult],
) -> pd.DataFrame:
    """
    Flatten all match logs across all simulations into one dataframe.
    """
    validate_simulation_results(simulation_results)

    records: list[dict[str, Any]] = []

    for run_result in simulation_results:
        for match in (run_result.match_results or []):
            records.append(
                {
                    "simulation_id": run_result.simulation_id,
                    "stage": match.stage,
                    "team_a": match.team_a,
                    "team_b": match.team_b,
                    "outcome": match.outcome,
                    "winner": match.winner,
                    "decided_by": match.decided_by,
                    "team_a_win_prob": match.probabilities.team_a_win,
                    "draw_prob": match.probabilities.draw,
                    "team_a_loss_prob": match.probabilities.team_a_loss,
                    "team_a_goals": match.team_a_goals,
                    "team_b_goals": match.team_b_goals,
                }
            )

    return pd.DataFrame(records)
