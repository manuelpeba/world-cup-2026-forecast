from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class MatchProbabilities:
    """
    Probabilistic output of the match prediction model from team A perspective.
    """

    team_a_win: float
    draw: float
    team_a_loss: float

    def as_dict(self) -> dict[str, float]:
        return {
            "win": self.team_a_win,
            "draw": self.draw,
            "loss": self.team_a_loss,
        }

    def validate(self, tolerance: float = 1e-6) -> None:
        values = [self.team_a_win, self.draw, self.team_a_loss]
        if any(v < 0.0 or v > 1.0 for v in values):
            raise ValueError("All probabilities must be between 0 and 1.")

        if abs(sum(values) - 1.0) > tolerance:
            raise ValueError(
                f"Probabilities must sum to 1.0. Got {sum(values):.8f}."
            )


@dataclass(slots=True)
class MatchSimulationResult:
    """
    Result of one simulated match.
    """

    stage: str
    team_a: str
    team_b: str
    outcome: str
    probabilities: MatchProbabilities
    winner: str | None = None
    decided_by: str | None = None
    team_a_goals: int | None = None
    team_b_goals: int | None = None

    def is_draw(self) -> bool:
        return self.outcome == "draw"


@dataclass(slots=True)
class GroupTableRow:
    """
    Group standings row for one team.
    """

    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0

    # Proxy fields for approximate tiebreaks when no score model is available
    pre_tournament_elo: float = 0.0
    tie_break_noise: float = 0.0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass(slots=True)
class KnockoutMatch:
    """
    Structural representation of a knockout fixture.
    """

    stage: str
    slot_id: str
    team_a: str
    team_b: str


@dataclass(slots=True)
class TournamentRunResult:
    """
    Full output of a single simulated tournament run.
    """

    simulation_id: int
    group_stage_tables: dict[str, list[GroupTableRow]] = field(default_factory=dict)
    group_stage_match_results: list[MatchSimulationResult] = field(default_factory=list)
    knockout_match_results: list[MatchSimulationResult] = field(default_factory=list)
    all_match_results: list[MatchSimulationResult] = field(default_factory=list)
    qualified_teams: list[str] = field(default_factory=list)
    round_of_16_teams: list[str] = field(default_factory=list)
    quarterfinalists: list[str] = field(default_factory=list)
    semifinalists: list[str] = field(default_factory=list)
    finalists: list[str] = field(default_factory=list)
    champion: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)