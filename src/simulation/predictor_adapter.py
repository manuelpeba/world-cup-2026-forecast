from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

from src.models.match_outcome.predict import MatchPredictionConfig, MatchPredictor
from src.simulation.config import SimulationConfig, TournamentConfig
from src.simulation.structures import MatchProbabilities


@dataclass(frozen=True, slots=True)
class MatchupKey:
    """
    Cache key for repeated probability lookups.
    """

    team_a: str
    team_b: str
    tournament_name: str
    neutral_venue: int
    model_name: str


class LRUCache:
    """
    Minimal LRU cache implementation for matchup probability reuse.
    """

    def __init__(self, max_size: int = 50_000) -> None:
        self.max_size = max_size
        self._store: OrderedDict[MatchupKey, MatchProbabilities] = OrderedDict()

    def get(self, key: MatchupKey) -> MatchProbabilities | None:
        if key not in self._store:
            return None
        self._store.move_to_end(key)
        return self._store[key]

    def set(self, key: MatchupKey, value: MatchProbabilities) -> None:
        self._store[key] = value
        self._store.move_to_end(key)

        if len(self._store) > self.max_size:
            self._store.popitem(last=False)

    def __len__(self) -> int:
        return len(self._store)


class SimulationPredictor:
    """
    Lightweight wrapper between the trained match model and the simulation engine.

    Responsibilities:
    - load predictor once
    - expose a stable interface to simulation modules
    - cache repeated matchup probabilities
    """

    def __init__(
        self,
        simulation_config: SimulationConfig,
        tournament_config: TournamentConfig,
    ) -> None:
        self.simulation_config = simulation_config
        self.tournament_config = tournament_config
        self.predictor = self._load_predictor()
        self._cache = (
            LRUCache(max_size=simulation_config.max_cached_matchups)
            if simulation_config.cache_predictions
            else None
        )

    def _load_predictor(self) -> MatchPredictor:
        config = MatchPredictionConfig(
            model_name=self.simulation_config.model_name,
            latest_features_path=self.tournament_config.features_path,
            models_dir=self.tournament_config.model_artifacts_dir,
        )
        return MatchPredictor(config=config)

    def get_team_strength(self, team: str, default: float = 1500.0) -> float:
        try:
            return float(self.predictor.get_team_strength(team))
        except Exception:
            return default

    def predict_match_proba(
        self,
        team_a: str,
        team_b: str,
    ) -> MatchProbabilities:
        """
        Return probabilities from team A perspective:
        - win: team A wins
        - draw: draw
        - loss: team A loses
        """
        cache_key = MatchupKey(
            team_a=team_a,
            team_b=team_b,
            tournament_name=self.tournament_config.tournament_name,
            neutral_venue=self.simulation_config.neutral_venue,
            model_name=self.simulation_config.model_name,
        )

        if self._cache is not None:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        raw_pred = self.predictor.predict_proba(
            team_a=team_a,
            team_b=team_b,
            tournament=self.tournament_config.tournament_name,
            neutral_venue=self.simulation_config.neutral_venue,
        )

        probabilities = self._normalize_prediction_output(raw_pred)
        probabilities.validate()

        if self._cache is not None:
            self._cache.set(cache_key, probabilities)

        return probabilities

    @staticmethod
    def _normalize_prediction_output(raw_pred: dict) -> MatchProbabilities:
        """
        Normalize predictor output into MatchProbabilities.

        Supported formats:

        1) Nested current MatchPredictor output:
        {
            "team_a": ...,
            "team_b": ...,
            "predicted_label": ...,
            "probabilities": {
                "win": ...,
                "draw": ...,
                "loss": ...
            },
            ...
        }

        2) Flat output:
        {
            "win": ...,
            "draw": ...,
            "loss": ...
        }

        3) Alternative flat output:
        {
            "team_a_win": ...,
            "draw": ...,
            "team_a_loss": ...
        }
        """
        if not isinstance(raw_pred, dict):
            raise TypeError(
                "Predictor output must be a dictionary-like object with probabilities."
            )

        # Current project format: nested probabilities dict
        if "probabilities" in raw_pred and isinstance(raw_pred["probabilities"], dict):
            proba = raw_pred["probabilities"]

            if {"win", "draw", "loss"}.issubset(proba.keys()):
                return MatchProbabilities(
                    team_a_win=float(proba["win"]),
                    draw=float(proba["draw"]),
                    team_a_loss=float(proba["loss"]),
                )

        # Flat canonical format
        if {"win", "draw", "loss"}.issubset(raw_pred.keys()):
            return MatchProbabilities(
                team_a_win=float(raw_pred["win"]),
                draw=float(raw_pred["draw"]),
                team_a_loss=float(raw_pred["loss"]),
            )

        # Flat alternative format
        if {"team_a_win", "draw", "team_a_loss"}.issubset(raw_pred.keys()):
            return MatchProbabilities(
                team_a_win=float(raw_pred["team_a_win"]),
                draw=float(raw_pred["draw"]),
                team_a_loss=float(raw_pred["team_a_loss"]),
            )

        raise ValueError(
            "Unsupported predictor output format. Expected either: "
            "1) {'probabilities': {'win','draw','loss'}}, "
            "2) flat {'win','draw','loss'}, or "
            "3) flat {'team_a_win','draw','team_a_loss'}."
        )
    