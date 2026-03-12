from __future__ import annotations

from typing import Literal

import numpy as np

from src.simulation.structures import MatchProbabilities


MatchOutcome = Literal["win", "draw", "loss"]


def build_rng(random_seed: int | None = None) -> np.random.Generator:
    """
    Build a reusable random generator for reproducible simulations.
    """
    return np.random.default_rng(random_seed)


def sample_match_outcome(
    probabilities: MatchProbabilities,
    rng: np.random.Generator,
) -> MatchOutcome:
    """
    Sample a single match outcome from team A perspective.

    Returns:
        - "win": team A wins
        - "draw": draw
        - "loss": team A loses
    """
    probabilities.validate()

    outcomes = np.array(["win", "draw", "loss"], dtype=object)
    probs = np.array(
        [
            probabilities.team_a_win,
            probabilities.draw,
            probabilities.team_a_loss,
        ],
        dtype=float,
    )

    sampled = rng.choice(outcomes, p=probs)
    return str(sampled)  # type: ignore[return-value]


def elo_win_probability(
    team_a_elo: float,
    team_b_elo: float,
) -> float:
    """
    Standard Elo expected-score transformation.

    Interpreted here as a pragmatic proxy for winning a knockout tiebreak
    after a drawn match.
    """
    return 1.0 / (1.0 + 10 ** ((team_b_elo - team_a_elo) / 400.0))


def sample_penalty_winner(
    team_a: str,
    team_b: str,
    rng: np.random.Generator,
    method: str = "elo_weighted",
    team_a_elo: float | None = None,
    team_b_elo: float | None = None,
) -> str:
    """
    Resolve a knockout draw.

    Methods:
        - "coin_flip": 50/50 winner
        - "elo_weighted": win probability derived from Elo strengths
    """
    if method == "coin_flip":
        return str(rng.choice([team_a, team_b]))

    if method == "elo_weighted":
        if team_a_elo is None or team_b_elo is None:
            raise ValueError(
                "team_a_elo and team_b_elo are required for method='elo_weighted'."
            )

        p_team_a = elo_win_probability(team_a_elo=team_a_elo, team_b_elo=team_b_elo)
        winner = rng.choice([team_a, team_b], p=[p_team_a, 1.0 - p_team_a])
        return str(winner)

    raise ValueError("method must be one of {'coin_flip', 'elo_weighted'}")


def outcome_to_points(
    outcome: MatchOutcome,
    points_win: int = 3,
    points_draw: int = 1,
    points_loss: int = 0,
) -> tuple[int, int]:
    """
    Convert sampled match outcome into points from team A vs team B perspective.
    """
    if outcome == "win":
        return points_win, points_loss
    if outcome == "draw":
        return points_draw, points_draw
    if outcome == "loss":
        return points_loss, points_win

    raise ValueError(f"Unsupported outcome: {outcome}")