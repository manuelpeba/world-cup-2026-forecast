from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class MatchProbabilities:
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
    team: str
    played: int = 0
    wins: int = 0
    draws: int = 0
    losses: int = 0
    goals_for: int = 0
    goals_against: int = 0
    points: int = 0
    pre_tournament_elo: float = 0.0
    tie_break_noise: float = 0.0

    @property
    def goal_difference(self) -> int:
        return self.goals_for - self.goals_against


@dataclass(slots=True)
class KnockoutMatch:
    stage: str
    slot_id: str
    team_a: str
    team_b: str


@dataclass(slots=True)
class TournamentRunResult:
    simulation_id: int
    group_tables: dict[str, list[GroupTableRow]]
    group_stage_results: list[MatchSimulationResult]
    qualified_teams: list[str]
    round_of_16_teams: list[str] | None = None
    quarterfinalists: list[str] | None = None
    semifinalists: list[str] | None = None
    finalists: list[str] | None = None
    champion: str | None = None
    match_results: list[MatchSimulationResult] | None = None
    metadata: dict[str, Any] | None = None
