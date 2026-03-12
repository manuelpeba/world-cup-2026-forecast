from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.simulation.aggregation import (
    aggregate_simulation_results,
    round_probability_columns,
)
from src.simulation.config import SimulationConfig, TournamentConfig
from src.simulation.reporting import export_simulation_outputs
from src.simulation.tournament import simulate_many_tournaments


def load_groups_from_json(input_path: Path) -> dict[str, list[str]]:
    """
    Load tournament groups from a JSON file.

    Expected format:
    {
        "A": ["Spain", "Brazil", "Japan", "Mexico"],
        "B": ["France", "Argentina", "USA", "Morocco"]
    }
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Groups file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError("Groups JSON must be a dictionary of group_name -> list[teams].")

    groups: dict[str, list[str]] = {}

    for group_name, teams in payload.items():
        if not isinstance(group_name, str):
            raise ValueError("All group names must be strings.")
        if not isinstance(teams, list) or not all(isinstance(team, str) for team in teams):
            raise ValueError(
                f"Group '{group_name}' must map to a list of team names."
            )
        groups[group_name] = teams

    return groups


def load_round_of_16_mapping_from_json(
    input_path: Path | None,
) -> list[tuple[str, str, str]] | None:
    """
    Load optional round-of-16 bracket mapping from a JSON file.

    Expected format:
    [
        ["R16_1", "A1", "B2"],
        ["R16_2", "C1", "D2"]
    ]
    """
    if input_path is None:
        return None

    if not input_path.exists():
        raise FileNotFoundError(f"Round-of-16 mapping file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, list):
        raise ValueError("Round-of-16 mapping JSON must be a list.")

    mapping: list[tuple[str, str, str]] = []

    for item in payload:
        if (
            not isinstance(item, list)
            or len(item) != 3
            or not all(isinstance(x, str) for x in item)
        ):
            raise ValueError(
                "Each round-of-16 mapping row must be a list of 3 strings: "
                "[slot_id, side_a_ref, side_b_ref]."
            )
        mapping.append((item[0], item[1], item[2]))

    return mapping


def load_group_match_schedule_from_json(
    input_path: Path | None,
) -> dict[str, list[tuple[str, str]]] | None:
    """
    Load optional explicit group match schedule from JSON.

    Expected format:
    {
        "A": [["Spain", "Brazil"], ["Japan", "Mexico"], ...],
        "B": [["France", "USA"], ["Argentina", "Morocco"], ...]
    }
    """
    if input_path is None:
        return None

    if not input_path.exists():
        raise FileNotFoundError(f"Group match schedule file not found: {input_path}")

    with open(input_path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    if not isinstance(payload, dict):
        raise ValueError(
            "Group match schedule JSON must be a dictionary of "
            "group_name -> list[[team_a, team_b]]."
        )

    schedule: dict[str, list[tuple[str, str]]] = {}

    for group_name, matches in payload.items():
        if not isinstance(group_name, str):
            raise ValueError("All group names in match schedule must be strings.")
        if not isinstance(matches, list):
            raise ValueError(
                f"Group '{group_name}' schedule must be a list of matches."
            )

        parsed_matches: list[tuple[str, str]] = []
        for match in matches:
            if (
                not isinstance(match, list)
                or len(match) != 2
                or not all(isinstance(x, str) for x in match)
            ):
                raise ValueError(
                    f"Invalid match entry in group '{group_name}'. "
                    "Expected [team_a, team_b]."
                )
            parsed_matches.append((match[0], match[1]))

        schedule[group_name] = parsed_matches

    return schedule


def build_simulation_config(args: argparse.Namespace) -> SimulationConfig:
    """
    Build SimulationConfig from CLI args.
    """
    return SimulationConfig(
        num_simulations=args.num_simulations,
        random_seed=args.random_seed,
        model_name=args.model_name,
        neutral_venue=args.neutral_venue,
        cache_predictions=not args.disable_prediction_cache,
        max_cached_matchups=args.max_cached_matchups,
        allow_draws_in_group_stage=True,
        allow_draws_in_knockout=False,
        knockout_draw_resolution=args.knockout_draw_resolution,
        save_match_logs=True,
        output_dir=Path(args.output_dir),
    )


def build_tournament_config(args: argparse.Namespace) -> TournamentConfig:
    """
    Build TournamentConfig from CLI args.
    """
    return TournamentConfig(
        tournament_name=args.tournament_name,
        tournament_id=args.tournament_id,
        points_win=args.points_win,
        points_draw=args.points_draw,
        points_loss=args.points_loss,
        group_size=args.group_size,
        teams_advancing_per_group=args.teams_advancing_per_group,
        features_path=Path(args.features_path),
        model_artifacts_dir=Path(args.model_artifacts_dir),
    )


def print_run_summary(
    simulation_config: SimulationConfig,
    tournament_config: TournamentConfig,
    groups: dict[str, list[str]],
) -> None:
    """
    Print simulation setup summary to console.
    """
    total_teams = sum(len(teams) for teams in groups.values())

    print("=" * 80)
    print("WORLD CUP SIMULATION RUN")
    print("=" * 80)
    print(f"Tournament:              {tournament_config.tournament_name}")
    print(f"Tournament ID:           {tournament_config.tournament_id}")
    print(f"Model:                   {simulation_config.model_name}")
    print(f"Simulations:             {simulation_config.num_simulations:,}")
    print(f"Random seed:             {simulation_config.random_seed}")
    print(f"Neutral venue:           {simulation_config.neutral_venue}")
    print(f"Knockout draw handling:  {simulation_config.knockout_draw_resolution}")
    print(f"Groups:                  {len(groups)}")
    print(f"Teams:                   {total_teams}")
    print(f"Output dir:              {simulation_config.output_dir}")
    print("=" * 80)


def print_top_probability_table(
    aggregated_outputs: dict[str, pd.DataFrame],
    top_n: int = 10,
) -> None:
    """
    Print a compact top-N team probability table to console.
    """
    probability_table = aggregated_outputs["team_probabilities"].copy()
    probability_table = round_probability_columns(probability_table, decimals=4)

    display_columns = [
        col for col in [
            "team",
            "advance_from_group_prob",
            "quarterfinal_prob",
            "semifinal_prob",
            "final_prob",
            "champion_prob",
        ]
        if col in probability_table.columns
    ]

    print("\nTOP TEAMS BY CHAMPION PROBABILITY")
    print("-" * 80)
    print(probability_table.loc[:, display_columns].head(top_n).to_string(index=False))


def run_simulation_pipeline(
    groups: dict[str, list[str]],
    simulation_config: SimulationConfig,
    tournament_config: TournamentConfig,
    group_match_schedule: dict[str, list[tuple[str, str]]] | None = None,
    round_of_16_mapping: list[tuple[str, str, str]] | None = None,
    export_stage_presence: bool = False,
    export_stage_counts: bool = False,
    decimals: int = 4,
) -> dict[str, Any]:
    """
    End-to-end simulation pipeline:
        1. simulate many tournaments
        2. aggregate results
        3. export artifacts
        4. return useful objects for downstream use
    """
    simulation_results = simulate_many_tournaments(
        groups=groups,
        simulation_config=simulation_config,
        tournament_config=tournament_config,
        group_match_schedule=group_match_schedule,
        round_of_16_mapping=round_of_16_mapping,
    )

    aggregated_outputs = aggregate_simulation_results(
        simulation_results=simulation_results,
        include_complements=True,
    )

    exported_paths = export_simulation_outputs(
        simulation_results=simulation_results,
        output_dir=simulation_config.output_dir,
        decimals=decimals,
        export_stage_presence=export_stage_presence,
        export_stage_counts=export_stage_counts,
    )

    return {
        "simulation_results": simulation_results,
        "aggregated_outputs": aggregated_outputs,
        "exported_paths": exported_paths,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    """
    Build CLI parser for the simulation runner.
    """
    parser = argparse.ArgumentParser(
        description="Run World Cup tournament simulations end-to-end."
    )

    parser.add_argument(
        "--groups-path",
        type=str,
        required=True,
        help="Path to JSON file with tournament groups.",
    )
    parser.add_argument(
        "--group-match-schedule-path",
        type=str,
        default=None,
        help="Optional path to JSON file with explicit group-stage schedule.",
    )
    parser.add_argument(
        "--round-of-16-mapping-path",
        type=str,
        default=None,
        help="Optional path to JSON file with custom round-of-16 mapping.",
    )

    parser.add_argument(
        "--num-simulations",
        type=int,
        default=10_000,
        help="Number of Monte Carlo tournament simulations.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="logistic_regression",
        help="Model artifact name to load.",
    )
    parser.add_argument(
        "--neutral-venue",
        type=int,
        default=1,
        help="Whether matches are simulated as neutral venue (0/1).",
    )
    parser.add_argument(
        "--knockout-draw-resolution",
        type=str,
        default="elo_weighted",
        choices=["coin_flip", "elo_weighted"],
        help="Method to resolve knockout draws.",
    )
    parser.add_argument(
        "--disable-prediction-cache",
        action="store_true",
        help="Disable matchup probability cache.",
    )
    parser.add_argument(
        "--max-cached-matchups",
        type=int,
        default=50_000,
        help="Maximum number of matchup probabilities to cache.",
    )

    parser.add_argument(
        "--tournament-name",
        type=str,
        default="FIFA World Cup 2026 - Simplified",
        help="Tournament name used in model inference and reporting.",
    )
    parser.add_argument(
        "--tournament-id",
        type=str,
        default="world_cup_2026_simplified",
        help="Stable tournament identifier.",
    )
    parser.add_argument(
        "--group-size",
        type=int,
        default=4,
        help="Number of teams per group.",
    )
    parser.add_argument(
        "--teams-advancing-per-group",
        type=int,
        default=2,
        help="Number of teams advancing from each group.",
    )
    parser.add_argument(
        "--points-win",
        type=int,
        default=3,
        help="Points awarded for a win in group stage.",
    )
    parser.add_argument(
        "--points-draw",
        type=int,
        default=1,
        help="Points awarded for a draw in group stage.",
    )
    parser.add_argument(
        "--points-loss",
        type=int,
        default=0,
        help="Points awarded for a loss in group stage.",
    )

    parser.add_argument(
        "--features-path",
        type=str,
        default="data/processed/latest_team_features.parquet",
        help="Path to latest team features parquet.",
    )
    parser.add_argument(
        "--model-artifacts-dir",
        type=str,
        default="artifacts/models",
        help="Directory containing trained model artifacts.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/outputs/simulation",
        help="Directory where simulation artifacts will be exported.",
    )

    parser.add_argument(
        "--export-stage-presence",
        action="store_true",
        help="Export long-format stage presence table.",
    )
    parser.add_argument(
        "--export-stage-counts",
        action="store_true",
        help="Export count-based stage progression table.",
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=4,
        help="Number of decimals for exported probability tables.",
    )
    parser.add_argument(
        "--top-n-preview",
        type=int,
        default=10,
        help="Number of teams to print in console preview.",
    )

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    groups = load_groups_from_json(Path(args.groups_path))
    group_match_schedule = load_group_match_schedule_from_json(
        Path(args.group_match_schedule_path)
        if args.group_match_schedule_path is not None
        else None
    )
    round_of_16_mapping = load_round_of_16_mapping_from_json(
        Path(args.round_of_16_mapping_path)
        if args.round_of_16_mapping_path is not None
        else None
    )

    simulation_config = build_simulation_config(args)
    tournament_config = build_tournament_config(args)

    print_run_summary(
        simulation_config=simulation_config,
        tournament_config=tournament_config,
        groups=groups,
    )

    outputs = run_simulation_pipeline(
        groups=groups,
        simulation_config=simulation_config,
        tournament_config=tournament_config,
        group_match_schedule=group_match_schedule,
        round_of_16_mapping=round_of_16_mapping,
        export_stage_presence=args.export_stage_presence,
        export_stage_counts=args.export_stage_counts,
        decimals=args.round_decimals,
    )

    aggregated_outputs = outputs["aggregated_outputs"]
    exported_paths = outputs["exported_paths"]

    print_top_probability_table(
        aggregated_outputs=aggregated_outputs,
        top_n=args.top_n_preview,
    )

    print("\nEXPORTED ARTIFACTS")
    print("-" * 80)
    for artifact_name, artifact_path in exported_paths.items():
        print(f"{artifact_name}: {artifact_path}")

    print("\nSimulation pipeline completed successfully.")


if __name__ == "__main__":
    main()