from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.simulation.config import SimulationConfig, TournamentConfig
from src.simulation.predictor_adapter import SimulationPredictor
from src.simulation.sampling import outcome_to_points, sample_match_outcome
from src.simulation.structures import (
    GroupTableRow,
    MatchSimulationResult,
    MatchProbabilities,
)


def build_round_robin_matches(group_teams: list[str]) -> list[tuple[str, str]]:
    """
    Build a full round-robin schedule for a group.

    For a standard 4-team group this returns 6 matches:
        A vs B
        A vs C
        A vs D
        B vs C
        B vs D
        C vs D
    """
    matches: list[tuple[str, str]] = []
    n_teams = len(group_teams)

    for i in range(n_teams):
        for j in range(i + 1, n_teams):
            matches.append((group_teams[i], group_teams[j]))

    return matches


def initialize_group_table(
    teams: list[str],
    predictor: SimulationPredictor,
    rng: np.random.Generator,
) -> dict[str, GroupTableRow]:
    """
    Initialize empty group standings rows for all teams.

    Since v1 does not model scorelines, we keep goal metrics at zero and
    rely on points + pre-tournament Elo + tiny seeded noise as tie-breakers.
    """
    table: dict[str, GroupTableRow] = {}

    for team in teams:
        table[team] = GroupTableRow(
            team=team,
            played=0,
            wins=0,
            draws=0,
            losses=0,
            goals_for=0,
            goals_against=0,
            points=0,
            pre_tournament_elo=predictor.get_team_strength(team),
            tie_break_noise=float(rng.uniform(0.0, 1e-6)),
        )

    return table


def simulate_group_match(
    team_a: str,
    team_b: str,
    predictor: SimulationPredictor,
    rng: np.random.Generator,
    stage: str = "group_stage",
) -> MatchSimulationResult:
    """
    Simulate one group-stage match from model probabilities.

    Outcome is always sampled from team A perspective:
        - win  => team A wins
        - draw => draw
        - loss => team A loses
    """
    probabilities: MatchProbabilities = predictor.predict_match_proba(
        team_a=team_a,
        team_b=team_b,
    )

    outcome = sample_match_outcome(probabilities=probabilities, rng=rng)

    winner: str | None = None
    if outcome == "win":
        winner = team_a
    elif outcome == "loss":
        winner = team_b

    return MatchSimulationResult(
        stage=stage,
        team_a=team_a,
        team_b=team_b,
        outcome=outcome,
        probabilities=probabilities,
        winner=winner,
        decided_by="regular_time",
        team_a_goals=None,
        team_b_goals=None,
    )


def update_group_table(
    table: dict[str, GroupTableRow],
    match_result: MatchSimulationResult,
    tournament_config: TournamentConfig,
) -> None:
    """
    Update standings table in place after one simulated group-stage match.

    v1 updates:
        - played
        - wins/draws/losses
        - points

    Goal-based columns remain zero because the current model does not
    simulate scorelines yet.
    """
    team_a = match_result.team_a
    team_b = match_result.team_b
    outcome = match_result.outcome

    row_a = table[team_a]
    row_b = table[team_b]

    row_a.played += 1
    row_b.played += 1

    points_a, points_b = outcome_to_points(
        outcome=outcome,
        points_win=tournament_config.points_win,
        points_draw=tournament_config.points_draw,
        points_loss=tournament_config.points_loss,
    )

    row_a.points += points_a
    row_b.points += points_b

    if outcome == "win":
        row_a.wins += 1
        row_b.losses += 1
    elif outcome == "draw":
        row_a.draws += 1
        row_b.draws += 1
    elif outcome == "loss":
        row_a.losses += 1
        row_b.wins += 1
    else:
        raise ValueError(f"Unsupported match outcome: {outcome}")


def rank_group_table(table: dict[str, GroupTableRow]) -> list[GroupTableRow]:
    """
    Rank a group table using v1 approximate tie-break rules.

    Order:
        1. points
        2. pre_tournament_elo
        3. tie_break_noise

    This is a deliberate approximation because the current system does not
    produce scorelines for official FIFA tie-break criteria.
    """
    rows = list(table.values())

    ranked = sorted(
        rows,
        key=lambda row: (
            row.points,
            row.pre_tournament_elo,
            row.tie_break_noise,
        ),
        reverse=True,
    )

    return ranked


def simulate_group(
    group_name: str,
    group_teams: list[str],
    predictor: SimulationPredictor,
    simulation_config: SimulationConfig,
    tournament_config: TournamentConfig,
    rng: np.random.Generator,
    group_matches: list[tuple[str, str]] | None = None,
) -> tuple[list[GroupTableRow], list[str], list[MatchSimulationResult]]:
    """
    Simulate one full group.

    Args:
        group_name:
            Group identifier, e.g. "A".
        group_teams:
            Teams in the group.
        predictor:
            Reusable simulation predictor adapter.
        simulation_config:
            Runtime simulation settings.
        tournament_config:
            Tournament rules.
        rng:
            Reproducible random generator.
        group_matches:
            Optional explicit schedule. If omitted, a full round-robin schedule
            is generated automatically.

    Returns:
        ranked_table:
            Final ranked standings rows.
        qualified_teams:
            Teams advancing from the group.
        match_results:
            All simulated match logs for the group.
    """
    if len(group_teams) != tournament_config.group_size:
        raise ValueError(
            f"Group '{group_name}' has {len(group_teams)} teams, but "
            f"tournament_config.group_size={tournament_config.group_size}."
        )

    if not simulation_config.allow_draws_in_group_stage:
        raise ValueError(
            "Group stage requires allow_draws_in_group_stage=True in v1."
        )

    schedule = group_matches or build_round_robin_matches(group_teams)
    table = initialize_group_table(
        teams=group_teams,
        predictor=predictor,
        rng=rng,
    )

    match_results: list[MatchSimulationResult] = []

    for team_a, team_b in schedule:
        result = simulate_group_match(
            team_a=team_a,
            team_b=team_b,
            predictor=predictor,
            rng=rng,
            stage=f"group_{group_name}",
        )
        update_group_table(
            table=table,
            match_result=result,
            tournament_config=tournament_config,
        )
        match_results.append(result)

    ranked_table = rank_group_table(table)
    qualified_teams = [
        row.team for row in ranked_table[: tournament_config.teams_advancing_per_group]
    ]

    return ranked_table, qualified_teams, match_results


def simulate_group_stage(
    groups: dict[str, list[str]],
    predictor: SimulationPredictor,
    simulation_config: SimulationConfig,
    tournament_config: TournamentConfig,
    rng: np.random.Generator,
    group_match_schedule: dict[str, list[tuple[str, str]]] | None = None,
) -> tuple[
    dict[str, list[GroupTableRow]],
    dict[str, list[str]],
    list[str],
    list[MatchSimulationResult],
]:
    """
    Simulate the full group stage across all groups.

    Args:
        groups:
            Mapping like:
                {
                    "A": ["Spain", "Brazil", "Japan", "Mexico"],
                    "B": [...],
                }
        predictor:
            Reusable prediction adapter.
        simulation_config:
            Runtime simulation settings.
        tournament_config:
            Tournament rules.
        rng:
            Reproducible random generator.
        group_match_schedule:
            Optional explicit schedule per group.

    Returns:
        group_tables:
            Final ranked standings per group.
        group_qualified_map:
            Qualified teams per group.
        qualified_teams_flat:
            Flat ordered list of all qualified teams.
        all_match_results:
            All group-stage match logs.
    """
    group_tables: dict[str, list[GroupTableRow]] = {}
    group_qualified_map: dict[str, list[str]] = {}
    qualified_teams_flat: list[str] = []
    all_match_results: list[MatchSimulationResult] = []

    for group_name, group_teams in groups.items():
        explicit_schedule = None
        if group_match_schedule is not None:
            explicit_schedule = group_match_schedule.get(group_name)

        ranked_table, qualified_teams, match_results = simulate_group(
            group_name=group_name,
            group_teams=group_teams,
            predictor=predictor,
            simulation_config=simulation_config,
            tournament_config=tournament_config,
            rng=rng,
            group_matches=explicit_schedule,
        )

        group_tables[group_name] = ranked_table
        group_qualified_map[group_name] = qualified_teams
        qualified_teams_flat.extend(qualified_teams)
        all_match_results.extend(match_results)

    return (
        group_tables,
        group_qualified_map,
        qualified_teams_flat,
        all_match_results,
    )


def group_table_to_records(
    ranked_table: list[GroupTableRow],
    group_name: str,
) -> list[dict[str, float | int | str]]:
    """
    Convert ranked group standings to serializable records for reporting.
    """
    records: list[dict[str, float | int | str]] = []

    for position, row in enumerate(ranked_table, start=1):
        records.append(
            {
                "group": group_name,
                "position": position,
                "team": row.team,
                "played": row.played,
                "wins": row.wins,
                "draws": row.draws,
                "losses": row.losses,
                "goals_for": row.goals_for,
                "goals_against": row.goals_against,
                "goal_difference": row.goal_difference,
                "points": row.points,
                "pre_tournament_elo": row.pre_tournament_elo,
            }
        )

    return records


def clone_group_table_rows(rows: list[GroupTableRow]) -> list[GroupTableRow]:
    """
    Return safe copies of group table rows.
    Useful if downstream code mutates objects.
    """
    return [replace(row) for row in rows]