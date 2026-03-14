from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from math import ceil

from src.simulation.config import SimulationConfig, TournamentConfig
from src.simulation.structures import TournamentRunResult
from src.simulation.tournament import simulate_many_tournaments


def _run_simulation_batch(
    *,
    groups: dict[str, list[str]],
    simulation_config: SimulationConfig,
    tournament_config: TournamentConfig,
    group_match_schedule: dict[str, list[tuple[str, str]]] | None,
    round_of_16_mapping: list[tuple[str, str, str]] | None,
    simulation_id_offset: int,
) -> list[TournamentRunResult]:
    """
    Worker entrypoint for one simulation batch.

    Each worker runs a smaller Monte Carlo block with its own random seed.
    After execution, simulation IDs are re-based using simulation_id_offset.
    """
    batch_results = simulate_many_tournaments(
        groups=groups,
        simulation_config=simulation_config,
        tournament_config=tournament_config,
        group_match_schedule=group_match_schedule,
        round_of_16_mapping=round_of_16_mapping,
    )

    rebased_results: list[TournamentRunResult] = []

    for i, result in enumerate(batch_results):
        rebased_results.append(
            replace(result, simulation_id=simulation_id_offset + i)
        )

    return rebased_results


def _split_simulation_counts(
    total_simulations: int,
    num_workers: int,
) -> list[int]:
    """
    Split total number of simulations into near-equal worker chunks.
    """
    if total_simulations <= 0:
        raise ValueError("total_simulations must be > 0")

    if num_workers <= 0:
        raise ValueError("num_workers must be > 0")

    base = total_simulations // num_workers
    remainder = total_simulations % num_workers

    counts = [base] * num_workers
    for i in range(remainder):
        counts[i] += 1

    return [count for count in counts if count > 0]


def simulate_many_tournaments_parallel(
    groups: dict[str, list[str]],
    simulation_config: SimulationConfig,
    tournament_config: TournamentConfig,
    group_match_schedule: dict[str, list[tuple[str, str]]] | None = None,
    round_of_16_mapping: list[tuple[str, str, str]] | None = None,
    num_workers: int = 4,
) -> list[TournamentRunResult]:
    """
    Parallel Monte Carlo tournament simulation.

    Strategy:
    - split total simulations into worker batches
    - assign each worker a distinct random seed
    - run batches in separate processes
    - concatenate and sort final results
    """
    if num_workers <= 1:
        return simulate_many_tournaments(
            groups=groups,
            simulation_config=simulation_config,
            tournament_config=tournament_config,
            group_match_schedule=group_match_schedule,
            round_of_16_mapping=round_of_16_mapping,
        )

    batch_sizes = _split_simulation_counts(
        total_simulations=simulation_config.num_simulations,
        num_workers=num_workers,
    )

    base_seed = simulation_config.random_seed or 42

    futures = []
    results: list[TournamentRunResult] = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        simulation_id_offset = 0

        for worker_idx, batch_size in enumerate(batch_sizes):
            worker_config = replace(
                simulation_config,
                num_simulations=batch_size,
                random_seed=base_seed + worker_idx + 1,
            )

            future = executor.submit(
                _run_simulation_batch,
                groups=groups,
                simulation_config=worker_config,
                tournament_config=tournament_config,
                group_match_schedule=group_match_schedule,
                round_of_16_mapping=round_of_16_mapping,
                simulation_id_offset=simulation_id_offset,
            )
            futures.append(future)
            simulation_id_offset += batch_size

        for future in as_completed(futures):
            batch_results = future.result()
            results.extend(batch_results)

    results.sort(key=lambda x: x.simulation_id)
    return results