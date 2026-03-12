from __future__ import annotations

from typing import Mapping

import numpy as np

from src.simulation.config import SimulationConfig
from src.simulation.predictor_adapter import SimulationPredictor
from src.simulation.sampling import sample_match_outcome, sample_penalty_winner
from src.simulation.structures import KnockoutMatch, MatchSimulationResult, MatchProbabilities


DEFAULT_ROUND_OF_16_MAPPING: list[tuple[str, str, str]] = [
    ("R16_1", "A1", "B2"),
    ("R16_2", "C1", "D2"),
    ("R16_3", "E1", "F2"),
    ("R16_4", "G1", "H2"),
    ("R16_5", "B1", "A2"),
    ("R16_6", "D1", "C2"),
    ("R16_7", "F1", "E2"),
    ("R16_8", "H1", "G2"),
]


def build_group_position_index(
    group_qualified_map: Mapping[str, list[str]],
) -> dict[str, str]:
    position_index: dict[str, str] = {}

    for group_name, qualified_teams in group_qualified_map.items():
        if len(qualified_teams) < 2:
            raise ValueError(
                f"Group '{group_name}' must provide at least 2 qualified teams "
                "to build a standard knockout bracket."
            )

        position_index[f"{group_name}1"] = qualified_teams[0]
        position_index[f"{group_name}2"] = qualified_teams[1]

    return position_index


def build_round_of_16_bracket(
    group_qualified_map: Mapping[str, list[str]],
    knockout_mapping: list[tuple[str, str, str]] | None = None,
) -> list[KnockoutMatch]:
    mapping = knockout_mapping or DEFAULT_ROUND_OF_16_MAPPING
    position_index = build_group_position_index(group_qualified_map)

    matches: list[KnockoutMatch] = []

    for slot_id, side_a_ref, side_b_ref in mapping:
        if side_a_ref not in position_index:
            raise KeyError(
                f"Knockout mapping reference '{side_a_ref}' not found in group qualifiers."
            )
        if side_b_ref not in position_index:
            raise KeyError(
                f"Knockout mapping reference '{side_b_ref}' not found in group qualifiers."
            )

        matches.append(
            KnockoutMatch(
                stage="round_of_16",
                slot_id=slot_id,
                team_a=position_index[side_a_ref],
                team_b=position_index[side_b_ref],
            )
        )

    return matches


def simulate_knockout_match(
    match: KnockoutMatch,
    predictor: SimulationPredictor,
    simulation_config: SimulationConfig,
    rng: np.random.Generator,
) -> MatchSimulationResult:
    probabilities: MatchProbabilities = predictor.predict_match_proba(
        team_a=match.team_a,
        team_b=match.team_b,
    )

    outcome = sample_match_outcome(probabilities=probabilities, rng=rng)

    if outcome == "win":
        winner = match.team_a
        decided_by = "regular_time"
    elif outcome == "loss":
        winner = match.team_b
        decided_by = "regular_time"
    elif outcome == "draw":
        if simulation_config.allow_draws_in_knockout:
            winner = None
            decided_by = "regular_time"
        else:
            winner = sample_penalty_winner(
                team_a=match.team_a,
                team_b=match.team_b,
                rng=rng,
                method=simulation_config.knockout_draw_resolution,
                team_a_elo=predictor.get_team_strength(match.team_a),
                team_b_elo=predictor.get_team_strength(match.team_b),
            )
            decided_by = simulation_config.knockout_draw_resolution
    else:
        raise ValueError(f"Unsupported knockout outcome: {outcome}")

    return MatchSimulationResult(
        stage=match.stage,
        team_a=match.team_a,
        team_b=match.team_b,
        outcome=outcome,
        probabilities=probabilities,
        winner=winner,
        decided_by=decided_by,
        team_a_goals=None,
        team_b_goals=None,
    )


def build_next_round_matches(
    previous_round_results: list[MatchSimulationResult],
    next_stage: str,
) -> list[KnockoutMatch]:
    winners = [result.winner for result in previous_round_results]

    if any(winner is None for winner in winners):
        raise ValueError(
            f"Cannot build next knockout round '{next_stage}' because at least one "
            "previous-round match has no winner."
        )

    winner_list = [winner for winner in winners if winner is not None]

    if len(winner_list) % 2 != 0:
        raise ValueError(
            f"Cannot build next knockout round '{next_stage}' with an odd number of winners."
        )

    next_matches: list[KnockoutMatch] = []

    for i in range(0, len(winner_list), 2):
        slot_number = i // 2 + 1
        next_matches.append(
            KnockoutMatch(
                stage=next_stage,
                slot_id=f"{next_stage}_{slot_number}",
                team_a=winner_list[i],
                team_b=winner_list[i + 1],
            )
        )

    return next_matches


def simulate_knockout_round(
    matches: list[KnockoutMatch],
    predictor: SimulationPredictor,
    simulation_config: SimulationConfig,
    rng: np.random.Generator,
) -> tuple[list[MatchSimulationResult], list[str]]:
    results: list[MatchSimulationResult] = []

    for match in matches:
        result = simulate_knockout_match(
            match=match,
            predictor=predictor,
            simulation_config=simulation_config,
            rng=rng,
        )
        results.append(result)

    winners = [result.winner for result in results]
    if any(winner is None for winner in winners):
        raise ValueError(
            f"Knockout round '{matches[0].stage if matches else 'unknown'}' "
            "contains matches without a resolved winner."
        )

    winner_list = [winner for winner in winners if winner is not None]
    return results, winner_list


def simulate_knockout_stage(
    group_qualified_map: Mapping[str, list[str]],
    predictor: SimulationPredictor,
    simulation_config: SimulationConfig,
    rng: np.random.Generator,
    round_of_16_mapping: list[tuple[str, str, str]] | None = None,
) -> dict[str, list[str] | list[MatchSimulationResult] | str]:
    round_of_16_matches = build_round_of_16_bracket(
        group_qualified_map=group_qualified_map,
        knockout_mapping=round_of_16_mapping,
    )

    round_of_16_results, round_of_16_winners = simulate_knockout_round(
        matches=round_of_16_matches,
        predictor=predictor,
        simulation_config=simulation_config,
        rng=rng,
    )

    quarterfinal_matches = build_next_round_matches(
        previous_round_results=round_of_16_results,
        next_stage="quarterfinals",
    )
    quarterfinal_results, quarterfinal_winners = simulate_knockout_round(
        matches=quarterfinal_matches,
        predictor=predictor,
        simulation_config=simulation_config,
        rng=rng,
    )

    semifinal_matches = build_next_round_matches(
        previous_round_results=quarterfinal_results,
        next_stage="semifinals",
    )
    semifinal_results, semifinal_winners = simulate_knockout_round(
        matches=semifinal_matches,
        predictor=predictor,
        simulation_config=simulation_config,
        rng=rng,
    )

    final_matches = build_next_round_matches(
        previous_round_results=semifinal_results,
        next_stage="final",
    )
    final_results, final_winners = simulate_knockout_round(
        matches=final_matches,
        predictor=predictor,
        simulation_config=simulation_config,
        rng=rng,
    )

    if len(final_winners) != 1:
        raise ValueError(
            f"Final round must produce exactly one champion, got {len(final_winners)}."
        )

    champion = final_winners[0]

    all_match_results: list[MatchSimulationResult] = (
        round_of_16_results
        + quarterfinal_results
        + semifinal_results
        + final_results
    )

    return {
        "round_of_16_teams": round_of_16_winners,
        "quarterfinalists": quarterfinal_winners,
        "semifinalists": semifinal_winners,
        "finalists": semifinal_winners,
        "champion": champion,
        "round_of_16_results": round_of_16_results,
        "quarterfinal_results": quarterfinal_results,
        "semifinal_results": semifinal_results,
        "final_results": final_results,
        "all_knockout_results": all_match_results,
    }


def flatten_knockout_results(
    knockout_output: dict[str, list[str] | list[MatchSimulationResult] | str],
) -> list[MatchSimulationResult]:
    results = knockout_output.get("all_knockout_results", [])
    if not isinstance(results, list):
        raise TypeError("'all_knockout_results' must be a list.")
    return results


def knockout_results_to_records(
    match_results: list[MatchSimulationResult],
) -> list[dict[str, str | float | None]]:
    records: list[dict[str, str | float | None]] = []

    for result in match_results:
        records.append(
            {
                "stage": result.stage,
                "team_a": result.team_a,
                "team_b": result.team_b,
                "outcome": result.outcome,
                "winner": result.winner,
                "decided_by": result.decided_by,
                "team_a_win_prob": result.probabilities.team_a_win,
                "draw_prob": result.probabilities.draw,
                "team_a_loss_prob": result.probabilities.team_a_loss,
            }
        )

    return records