from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass(slots=True)
class SimulationConfig:
    """
    Runtime configuration for Monte Carlo tournament simulations.
    """

    num_simulations: int = 10_000
    random_seed: Optional[int] = 42
    model_name: str = "xgboost_match_outcome"
    neutral_venue: int = 1
    cache_predictions: bool = True
    max_cached_matchups: int = 50_000

    # Knockout resolution
    allow_draws_in_group_stage: bool = True
    allow_draws_in_knockout: bool = False
    knockout_draw_resolution: str = "elo_weighted"  # {"coin_flip", "elo_weighted"}

    # Reporting / persistence
    save_match_logs: bool = False
    output_dir: Path = field(default_factory=lambda: Path("data/outputs/simulation"))

    def __post_init__(self) -> None:
        if self.num_simulations <= 0:
            raise ValueError("num_simulations must be > 0")

        if self.knockout_draw_resolution not in {"coin_flip", "elo_weighted"}:
            raise ValueError(
                "knockout_draw_resolution must be one of "
                "{'coin_flip', 'elo_weighted'}"
            )


@dataclass(slots=True)
class TournamentConfig:
    """
    Static tournament configuration.

    This object should describe the tournament format as data, keeping the
    simulation engine generic and reusable across competitions.
    """

    tournament_name: str = "FIFA World Cup 2026 - Simplified"
    tournament_id: str = "world_cup_2026_simplified"
    points_win: int = 3
    points_draw: int = 1
    points_loss: int = 0

    # Format metadata
    group_size: int = 4
    teams_advancing_per_group: int = 2

    # Paths expected by the predictor adapter
    features_path: Path = field(
        default_factory=lambda: Path("data/processed/latest_team_features.parquet")
    )
    model_artifacts_dir: Path = field(
        default_factory=lambda: Path("artifacts/models/match_outcome")
    )

    def __post_init__(self) -> None:
        if self.group_size < 2:
            raise ValueError("group_size must be >= 2")

        if self.teams_advancing_per_group < 1:
            raise ValueError("teams_advancing_per_group must be >= 1")

        if self.teams_advancing_per_group >= self.group_size:
            raise ValueError(
                "teams_advancing_per_group must be smaller than group_size"
            )