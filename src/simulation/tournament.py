from __future__ import annotations

from typing import Any

import numpy as np

from src.simulation.bracket_builder import build_round_of_32_bracket
from src.simulation.config import SimulationConfig, TournamentConfig
from src.simulation.group_stage import simulate_group_stage
from src.simulation.knockout_stage import (
    simulate_knockout_from_initial_matches,
    simulate_knockout_stage,
)
from src.simulation.predictor_adapter import SimulationPredictor
from src.simulation.qualification import (
    build_knockout_qualifiers,
    collect_auto_qualifiers,
    collect_third_place_teams,
    rank_third_place_teams,
    select_best_thirds,
)
from src.simulation.sampling import build_rng
from src.simulation.structures import MatchSimulationResult, TournamentRunResult


def validate_tournament_groups(
    groups: dict[str, list[str]],
    tournament_config: TournamentConfig,
) -> None:
    """
    Validate tournament group structure before simulation.
    """
    if not groups:
        raise ValueError("groups cannot be empty.")

    all_teams: list[str] = []

    for group_name, group_teams in groups.items():
        if len(group_teams) != tournament_config.group_size:
            raise ValueError(
                f"Group '{group_name}' has {len(group_teams)} teams, "
                f"expected {tournament_config.group_size}."
            )

        if len(set(group_teams)) != len(group_teams):
            raise ValueError(f"Group '{group_name}' contains duplicate teams.")

        all_teams.extend(group_teams)

    if len(set(all_teams)) != len(all_teams):
        raise ValueError("A team appears more than once across tournament groups.")

    expected_knockout_teams = (
        len(groups) * tournament_config.teams_advancing_per_group
    )

    if expected_knockout_teams % 2 != 0:
        raise ValueError(
            "The number of teams advancing from groups must be even."
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
    Simulate one tournament using the v1 32-team format.

    Semantics:
        - qualified_teams: group-stage qualifiers = round_of_16 teams
        - quarterfinalists: winners of round_of_16
        - semifinalists: winners of quarterfinals
        - finalists: winners of semifinals
        - champion: winner of final
    """
    validate_tournament_groups(groups, tournament_config)

    (
        group_tables,
        group_qualified_map,
        qualified_teams_flat,
        group_stage_results,
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

    knockout_results = list(knockout_output["all_knockout_results"])
    match_results = group_stage_results + knockout_results

    tournament_result = TournamentRunResult(
        simulation_id=simulation_id,
        group_tables=group_tables,
        group_stage_results=group_stage_results,
        qualified_teams=qualified_teams_flat,
        round_of_16_teams=qualified_teams_flat,
        quarterfinalists=list(knockout_output["quarterfinalists"]),
        semifinalists=list(knockout_output["semifinalists"]),
        finalists=list(knockout_output["finalists"]),
        champion=str(knockout_output["champion"]),
        match_results=match_results,
        metadata={
            "tournament_name": tournament_config.tournament_name,
            "tournament_id": tournament_config.tournament_id,
            "model_name": simulation_config.model_name,
            "neutral_venue": simulation_config.neutral_venue,
            "knockout_draw_resolution": simulation_config.knockout_draw_resolution,
            "num_groups": len(groups),
            "group_size": tournament_config.group_size,
            "teams_advancing_per_group": tournament_config.teams_advancing_per_group,
            "initial_knockout_round": "round_of_16",
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
    Run many v1 tournament simulations.
    """
    validate_tournament_groups(groups, tournament_config)

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
    Build per-team stage flags for one simulation run.

    Backward-compatible semantics:
        - v1:
            qualified_teams = round_of_16 teams
        - v2:
            qualified_teams = round_of_32 teams
            round_of_16_teams = winners of round_of_32
    """
    qualified_set = set(run_result.qualified_teams)
    r16_set = set(run_result.round_of_16_teams or [])
    qf_set = set(run_result.quarterfinalists or [])
    sf_set = set(run_result.semifinalists or [])
    final_set = set(run_result.finalists or [])
    champion = run_result.champion

    initial_knockout_round = (
        run_result.metadata.get("initial_knockout_round")
        if run_result.metadata is not None
        else None
    )

    records: list[dict[str, Any]] = []

    for team in all_teams:
        record: dict[str, Any] = {
            "simulation_id": run_result.simulation_id,
            "team": team,
            "group_stage_exit": int(team not in qualified_set),
            "quarterfinal": int(team in qf_set),
            "semifinal": int(team in sf_set),
            "final": int(team in final_set),
            "champion": int(team == champion),
        }

        if initial_knockout_round == "round_of_32":
            record["round_of_32"] = int(team in qualified_set)
            record["round_of_16"] = int(team in r16_set)
        else:
            # v1 compatibility
            record["round_of_16"] = int(team in qualified_set)

        records.append(record)

    return records


def flatten_tournament_run(
    run_result: TournamentRunResult,
) -> dict[str, Any]:
    """
    Convert one tournament run into a compact serializable dictionary.
    """
    return {
        "simulation_id": run_result.simulation_id,
        "qualified_teams": run_result.qualified_teams,
        "round_of_16_teams": run_result.round_of_16_teams,
        "quarterfinalists": run_result.quarterfinalists,
        "semifinalists": run_result.semifinalists,
        "finalists": run_result.finalists,
        "champion": run_result.champion,
        "group_stage_match_count": len(run_result.group_stage_results),
        "all_match_count": len(run_result.match_results or []),
        "metadata": run_result.metadata,
    }


def collect_all_match_logs(
    run_result: TournamentRunResult,
) -> list[MatchSimulationResult]:
    """
    Return all match logs for one tournament run.
    """
    return list(run_result.match_results or [])


def simulate_one_tournament_v2(
    simulation_id: int,
    groups: dict[str, list[str]],
    predictor: SimulationPredictor,
    simulation_config: SimulationConfig,
    tournament_config: TournamentConfig,
    rng: np.random.Generator,
    bracket_config: dict[str, Any],
    group_match_schedule: dict[str, list[tuple[str, str]]] | None = None,
) -> TournamentRunResult:
    """
    Simulate one tournament using the v2 48-team format.

    Assumed structure:
        - 12 groups of 4
        - top 2 qualify automatically
        - best 8 third-placed teams qualify
        - knockout starts at round_of_32

    Semantics:
        - qualified_teams: round_of_32 teams
        - round_of_16_teams: winners of round_of_32
        - quarterfinalists: winners of round_of_16
        - semifinalists: winners of quarterfinals
        - finalists: winners of semifinals
        - champion: winner of final
    """
    validate_tournament_groups(groups, tournament_config)

    (
        group_tables,
        _group_qualified_map_v1,
        _qualified_teams_flat_v1,
        group_stage_results,
    ) = simulate_group_stage(
        groups=groups,
        predictor=predictor,
        simulation_config=simulation_config,
        tournament_config=tournament_config,
        rng=rng,
        group_match_schedule=group_match_schedule,
    )

    auto_qualifiers_map = collect_auto_qualifiers(
        group_tables=group_tables,
        auto_qualifiers_per_group=2,
    )

    third_place_rows = collect_third_place_teams(group_tables)
    ranked_thirds = rank_third_place_teams(third_place_rows)
    best_thirds = select_best_thirds(ranked_thirds, num_best_thirds=8)

    qualified_teams_flat = build_knockout_qualifiers(
        auto_qualifiers_map=auto_qualifiers_map,
        best_thirds=best_thirds,
    )

    round_of_32_matches = build_round_of_32_bracket(
        group_tables=group_tables,
        best_thirds=best_thirds,
        bracket_config=bracket_config,
    )

    knockout_output = simulate_knockout_from_initial_matches(
        initial_matches=round_of_32_matches,
        predictor=predictor,
        simulation_config=simulation_config,
        rng=rng,
        rounds=[
            "round_of_32",
            "round_of_16",
            "quarterfinals",
            "semifinals",
            "final",
        ],
    )

    teams_by_round = knockout_output["teams_by_round"]
    knockout_results = list(knockout_output["all_knockout_results"])
    match_results = group_stage_results + knockout_results

    tournament_result = TournamentRunResult(
        simulation_id=simulation_id,
        group_tables=group_tables,
        group_stage_results=group_stage_results,
        qualified_teams=qualified_teams_flat,
        round_of_16_teams=list(teams_by_round.get("round_of_16", [])),
        quarterfinalists=list(teams_by_round.get("quarterfinals", [])),
        semifinalists=list(teams_by_round.get("semifinals", [])),
        finalists=list(teams_by_round.get("final", [])),
        champion=str(knockout_output["champion"]),
        match_results=match_results,
        metadata={
            "tournament_name": tournament_config.tournament_name,
            "tournament_id": tournament_config.tournament_id,
            "model_name": simulation_config.model_name,
            "neutral_venue": simulation_config.neutral_venue,
            "knockout_draw_resolution": simulation_config.knockout_draw_resolution,
            "num_groups": len(groups),
            "group_size": tournament_config.group_size,
            "teams_advancing_per_group": 2,
            "best_third_qualifiers": 8,
            "initial_knockout_round": "round_of_32",
        },
    )

    return tournament_result


def simulate_many_tournaments_v2(
    groups: dict[str, list[str]],
    simulation_config: SimulationConfig,
    tournament_config: TournamentConfig,
    bracket_config: dict[str, Any],
    group_match_schedule: dict[str, list[tuple[str, str]]] | None = None,
) -> list[TournamentRunResult]:
    """
    Run many v2 tournament simulations.
    """
    validate_tournament_groups(groups, tournament_config)

    predictor = SimulationPredictor(
        simulation_config=simulation_config,
        tournament_config=tournament_config,
    )

    rng = build_rng(simulation_config.random_seed)

    results: list[TournamentRunResult] = []

    for simulation_id in range(simulation_config.num_simulations):
        run_result = simulate_one_tournament_v2(
            simulation_id=simulation_id,
            groups=groups,
            predictor=predictor,
            simulation_config=simulation_config,
            tournament_config=tournament_config,
            rng=rng,
            bracket_config=bracket_config,
            group_match_schedule=group_match_schedule,
        )
        results.append(run_result)

    return results


