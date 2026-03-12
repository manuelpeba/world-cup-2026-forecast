from __future__ import annotations

from typing import Any

import numpy as np

from src.simulation.config import SimulationConfig, TournamentConfig
from src.simulation.group_stage import simulate_group_stage
from src.simulation.knockout_stage import simulate_knockout_stage
from src.simulation.predictor_adapter import SimulationPredictor
from src.simulation.sampling import build_rng
from src.simulation.structures import MatchSimulationResult, TournamentRunResult


def validate_tournament_groups(
    groups: dict[str, list[str]],
    tournament_config: TournamentConfig,
) -> None:
    """
    Validate tournament group composition before simulation.
    """
    if not groups:
        raise ValueError("groups cannot be empty.")

    all_teams: list[str] = []

    for group_name, group_teams in groups.items():
        if len(group_teams) != tournament_config.group_size:
            raise ValueError(
                f"Group '{group_name}' has {len(group_teams)} teams, but "
                f"group_size={tournament_config.group_size}."
            )

        if len(set(group_teams)) != len(group_teams):
            raise ValueError(
                f"Group '{group_name}' contains duplicate teams: {group_teams}"
            )

        all_teams.extend(group_teams)

    if len(set(all_teams)) != len(all_teams):
        raise ValueError(
            "A team appears more than once across tournament groups."
        )

    expected_knockout_teams = (
        len(groups) * tournament_config.teams_advancing_per_group
    )
    if expected_knockout_teams % 2 != 0:
        raise ValueError(
            "The number of teams advancing from groups must be even "
            "to build a knockout stage."
        )


def simulate_one_tournament(
    simulation_id: int,
    groups: dict[str, list[str]],
    predictor: SimulationPredictor,
    simulation_config: SimulationConfig,
    tournament_config: TournamentConfig,
    rng: np.random.Generator,
    group_match_schedule: dict[str, list[tuple[str, str]]] | None = None,
    round_of_16_mapping: list[tuple[str, str, str]] | None = None,
) -> TournamentRunResult:
    """
    Simulate a single full tournament run.

    Flow:
        1. simulate group stage
        2. simulate knockout stage
        3. return structured run result
    """
    validate_tournament_groups(
        groups=groups,
        tournament_config=tournament_config,
    )

    (
        group_stage_tables,
        group_qualified_map,
        qualified_teams_flat,
        group_stage_match_results,
    ) = simulate_group_stage(
        groups=groups,
        predictor=predictor,
        simulation_config=simulation_config,
        tournament_config=tournament_config,
        rng=rng,
        group_match_schedule=group_match_schedule,
    )

    knockout_output = simulate_knockout_stage(
        group_qualified_map=group_qualified_map,
        predictor=predictor,
        simulation_config=simulation_config,
        rng=rng,
        round_of_16_mapping=round_of_16_mapping,
    )

    knockout_match_results = list(
        knockout_output["all_knockout_results"]  # type: ignore[arg-type]
    )
    all_match_results = group_stage_match_results + knockout_match_results

    tournament_result = TournamentRunResult(
        simulation_id=simulation_id,
        group_stage_tables=group_stage_tables,
        group_stage_match_results=group_stage_match_results,
        knockout_match_results=knockout_match_results,
        all_match_results=all_match_results,
        qualified_teams=qualified_teams_flat,
        round_of_16_teams=list(
            knockout_output["round_of_16_teams"]  # type: ignore[arg-type]
        ),
        quarterfinalists=list(
            knockout_output["quarterfinalists"]  # type: ignore[arg-type]
        ),
        semifinalists=list(
            knockout_output["semifinalists"]  # type: ignore[arg-type]
        ),
        finalists=list(
            knockout_output["finalists"]  # type: ignore[arg-type]
        ),
        champion=str(knockout_output["champion"]),
        metadata={
            "tournament_name": tournament_config.tournament_name,
            "tournament_id": tournament_config.tournament_id,
            "model_name": simulation_config.model_name,
            "neutral_venue": simulation_config.neutral_venue,
            "knockout_draw_resolution": simulation_config.knockout_draw_resolution,
            "num_groups": len(groups),
            "group_size": tournament_config.group_size,
            "teams_advancing_per_group": tournament_config.teams_advancing_per_group,
        },
    )

    return tournament_result


def simulate_many_tournaments(
    groups: dict[str, list[str]],
    simulation_config: SimulationConfig,
    tournament_config: TournamentConfig,
    group_match_schedule: dict[str, list[tuple[str, str]]] | None = None,
    round_of_16_mapping: list[tuple[str, str, str]] | None = None,
) -> list[TournamentRunResult]:
    """
    Run N Monte Carlo tournament simulations.

    The predictor is instantiated only once and reused across all runs.
    The random number generator is also reused to guarantee reproducibility
    under a fixed random seed.
    """
    validate_tournament_groups(
        groups=groups,
        tournament_config=tournament_config,
    )

    predictor = SimulationPredictor(
        simulation_config=simulation_config,
        tournament_config=tournament_config,
    )
    rng = build_rng(simulation_config.random_seed)

    results: list[TournamentRunResult] = []

    for simulation_id in range(simulation_config.num_simulations):
        run_result = simulate_one_tournament(
            simulation_id=simulation_id,
            groups=groups,
            predictor=predictor,
            simulation_config=simulation_config,
            tournament_config=tournament_config,
            rng=rng,
            group_match_schedule=group_match_schedule,
            round_of_16_mapping=round_of_16_mapping,
        )
        results.append(run_result)

    return results


def extract_stage_presence_flags(
    run_result: TournamentRunResult,
    all_teams: list[str],
) -> list[dict[str, Any]]:
    """
    Convert one tournament run into team-level binary stage flags.

    Useful downstream for aggregation into advancement probabilities.
    """
    qualified_set = set(run_result.qualified_teams)
    r16_set = set(run_result.round_of_16_teams)
    qf_set = set(run_result.quarterfinalists)
    sf_set = set(run_result.semifinalists)
    final_set = set(run_result.finalists)
    champion = run_result.champion

    records: list[dict[str, Any]] = []

    for team in all_teams:
        records.append(
            {
                "simulation_id": run_result.simulation_id,
                "team": team,
                "group_stage_exit": int(team not in qualified_set),
                "round_of_16": int(team in r16_set),
                "quarterfinal": int(team in qf_set),
                "semifinal": int(team in sf_set),
                "final": int(team in final_set),
                "champion": int(team == champion),
            }
        )

    return records


def flatten_tournament_run(
    run_result: TournamentRunResult,
) -> dict[str, Any]:
    """
    Produce a flat summary of one tournament run.
    """
    return {
        "simulation_id": run_result.simulation_id,
        "qualified_teams": run_result.qualified_teams,
        "round_of_16_teams": run_result.round_of_16_teams,
        "quarterfinalists": run_result.quarterfinalists,
        "semifinalists": run_result.semifinalists,
        "finalists": run_result.finalists,
        "champion": run_result.champion,
        "group_stage_match_count": len(run_result.group_stage_match_results),
        "knockout_match_count": len(run_result.knockout_match_results),
        "all_match_count": len(run_result.all_match_results),
        "metadata": run_result.metadata,
    }


def collect_all_match_logs(
    run_result: TournamentRunResult,
) -> list[MatchSimulationResult]:
    """
    Return all match logs stored in the tournament result.
    """
    return list(run_result.all_match_results)