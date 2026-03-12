from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.simulation.aggregation import (
    aggregate_simulation_results,
    build_match_log_dataframe,
    build_summary_metadata,
    round_probability_columns,
)
from src.simulation.config import SimulationConfig
from src.simulation.structures import TournamentRunResult


def ensure_output_dir(output_dir: Path | str) -> Path:
    """
    Create the output directory if it does not exist and return it as Path.
    """
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_dataframe_parquet(
    df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """
    Save dataframe as parquet.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def save_dataframe_csv(
    df: pd.DataFrame,
    output_path: Path | str,
) -> Path:
    """
    Save dataframe as CSV.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def save_json(
    payload: dict[str, Any] | list[dict[str, Any]],
    output_path: Path | str,
    indent: int = 2,
) -> Path:
    """
    Save serializable payload as JSON.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8") as file:
        json.dump(payload, file, indent=indent, ensure_ascii=False)

    return path


def save_summary_metadata(
    simulation_results: list[TournamentRunResult],
    output_dir: Path | str,
    file_name: str = "summary_metadata.json",
) -> Path:
    """
    Build and save summary metadata as JSON.
    """
    output_dir = ensure_output_dir(output_dir)

    metadata_df = build_summary_metadata(simulation_results)
    metadata_record = metadata_df.iloc[0].to_dict()

    return save_json(
        payload=metadata_record,
        output_path=output_dir / file_name,
    )


def save_team_advancement_probs(
    team_probabilities_df: pd.DataFrame,
    output_dir: Path | str,
    parquet_name: str = "team_probabilities.parquet",
    csv_name: str = "team_probabilities.csv",
    decimals: int = 4,
) -> dict[str, Path]:
    """
    Save team probability table in parquet and CSV formats.
    """
    output_dir = ensure_output_dir(output_dir)
    formatted_df = round_probability_columns(team_probabilities_df, decimals=decimals)

    parquet_path = save_dataframe_parquet(formatted_df, output_dir / parquet_name)
    csv_path = save_dataframe_csv(formatted_df, output_dir / csv_name)

    return {
        "parquet": parquet_path,
        "csv": csv_path,
    }


def save_champion_distribution(
    champion_distribution_df: pd.DataFrame,
    output_dir: Path | str,
    parquet_name: str = "champion_distribution.parquet",
    csv_name: str = "champion_distribution.csv",
    decimals: int = 4,
) -> dict[str, Path]:
    """
    Save champion distribution table in parquet and CSV formats.
    """
    output_dir = ensure_output_dir(output_dir)
    df = champion_distribution_df.copy()

    if "champion_prob" in df.columns:
        df["champion_prob"] = df["champion_prob"].round(decimals)

    parquet_path = save_dataframe_parquet(df, output_dir / parquet_name)
    csv_path = save_dataframe_csv(df, output_dir / csv_name)

    return {
        "parquet": parquet_path,
        "csv": csv_path,
    }


def save_match_level_logs(
    match_logs_df: pd.DataFrame,
    output_dir: Path | str,
    parquet_name: str = "match_logs.parquet",
) -> Path:
    """
    Save match-level logs as parquet.
    """
    output_dir = ensure_output_dir(output_dir)
    return save_dataframe_parquet(match_logs_df, output_dir / parquet_name)


def save_stage_presence_table(
    stage_presence_df: pd.DataFrame,
    output_dir: Path | str,
    parquet_name: str = "stage_presence.parquet",
    csv_name: str = "stage_presence.csv",
) -> dict[str, Path]:
    """
    Optional export for the long-format stage presence table.
    Useful for QA and debugging.
    """
    output_dir = ensure_output_dir(output_dir)

    parquet_path = save_dataframe_parquet(stage_presence_df, output_dir / parquet_name)
    csv_path = save_dataframe_csv(stage_presence_df, output_dir / csv_name)

    return {
        "parquet": parquet_path,
        "csv": csv_path,
    }


def save_stage_counts_table(
    stage_counts_df: pd.DataFrame,
    output_dir: Path | str,
    parquet_name: str = "stage_counts.parquet",
    csv_name: str = "stage_counts.csv",
) -> dict[str, Path]:
    """
    Optional export for count-based progression table.
    Useful for QA and sanity checks.
    """
    output_dir = ensure_output_dir(output_dir)

    parquet_path = save_dataframe_parquet(stage_counts_df, output_dir / parquet_name)
    csv_path = save_dataframe_csv(stage_counts_df, output_dir / csv_name)

    return {
        "parquet": parquet_path,
        "csv": csv_path,
    }


def export_simulation_outputs(
    simulation_results: list[TournamentRunResult],
    output_dir: Path | str,
    decimals: int = 4,
    export_stage_presence: bool = False,
    export_stage_counts: bool = False,
) -> dict[str, Path]:
    """
    Main reporting entry point.

    This function aggregates all simulation outputs and exports the main
    forecasting artifacts:
        - team_probabilities.parquet/csv
        - champion_distribution.parquet/csv
        - match_logs.parquet
        - summary_metadata.json

    Optional QA/debugging exports:
        - stage_presence.parquet/csv
        - stage_counts.parquet/csv
    """
    output_dir = ensure_output_dir(output_dir)

    aggregated = aggregate_simulation_results(
        simulation_results=simulation_results,
        include_complements=True,
    )

    team_probabilities_df = aggregated["team_probabilities"]
    champion_distribution_df = aggregated["champion_distribution"]
    stage_presence_df = aggregated["stage_presence"]
    stage_counts_df = aggregated["stage_counts"]

    match_logs_df = build_match_log_dataframe(simulation_results)
    summary_metadata_df = build_summary_metadata(simulation_results)

    exported_paths: dict[str, Path] = {}

    team_prob_paths = save_team_advancement_probs(
        team_probabilities_df=team_probabilities_df,
        output_dir=output_dir,
        parquet_name="team_probabilities.parquet",
        csv_name="team_probabilities.csv",
        decimals=decimals,
    )
    exported_paths["team_probabilities_parquet"] = team_prob_paths["parquet"]
    exported_paths["team_probabilities_csv"] = team_prob_paths["csv"]

    champion_paths = save_champion_distribution(
        champion_distribution_df=champion_distribution_df,
        output_dir=output_dir,
        parquet_name="champion_distribution.parquet",
        csv_name="champion_distribution.csv",
        decimals=decimals,
    )
    exported_paths["champion_distribution_parquet"] = champion_paths["parquet"]
    exported_paths["champion_distribution_csv"] = champion_paths["csv"]

    match_logs_path = save_match_level_logs(
        match_logs_df=match_logs_df,
        output_dir=output_dir,
        parquet_name="match_logs.parquet",
    )
    exported_paths["match_logs_parquet"] = match_logs_path

    metadata_record = summary_metadata_df.iloc[0].to_dict()
    summary_metadata_path = save_json(
        payload=metadata_record,
        output_path=Path(output_dir) / "summary_metadata.json",
    )
    exported_paths["summary_metadata_json"] = summary_metadata_path

    if export_stage_presence:
        stage_presence_paths = save_stage_presence_table(
            stage_presence_df=stage_presence_df,
            output_dir=output_dir,
            parquet_name="stage_presence.parquet",
            csv_name="stage_presence.csv",
        )
        exported_paths["stage_presence_parquet"] = stage_presence_paths["parquet"]
        exported_paths["stage_presence_csv"] = stage_presence_paths["csv"]

    if export_stage_counts:
        stage_counts_paths = save_stage_counts_table(
            stage_counts_df=stage_counts_df,
            output_dir=output_dir,
            parquet_name="stage_counts.parquet",
            csv_name="stage_counts.csv",
        )
        exported_paths["stage_counts_parquet"] = stage_counts_paths["parquet"]
        exported_paths["stage_counts_csv"] = stage_counts_paths["csv"]

    return exported_paths


def export_simulation_outputs_from_config(
    simulation_results: list[TournamentRunResult],
    simulation_config: SimulationConfig,
    decimals: int = 4,
    export_stage_presence: bool = False,
    export_stage_counts: bool = False,
) -> dict[str, Path]:
    """
    Convenience wrapper that uses SimulationConfig.output_dir.
    """
    return export_simulation_outputs(
        simulation_results=simulation_results,
        output_dir=simulation_config.output_dir,
        decimals=decimals,
        export_stage_presence=export_stage_presence,
        export_stage_counts=export_stage_counts,
    )