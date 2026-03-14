from __future__ import annotations

from typing import Any

from src.simulation.structures import GroupTableRow


def extract_group_rankings(
    group_tables: dict[str, list[GroupTableRow]],
) -> dict[str, dict[str, GroupTableRow]]:
    """
    Convert ranked group tables into positional access by group.

    Returns:
        {
            "A": {
                "A1": GroupTableRow(...),
                "A2": GroupTableRow(...),
                "A3": GroupTableRow(...),
                "A4": GroupTableRow(...),
            },
            ...
        }
    """
    rankings: dict[str, dict[str, GroupTableRow]] = {}

    for group_name, rows in group_tables.items():
        if not rows:
            raise ValueError(f"Group '{group_name}' has no ranked rows.")

        rankings[group_name] = {}
        for position, row in enumerate(rows, start=1):
            rankings[group_name][f"{group_name}{position}"] = row

    return rankings


def collect_auto_qualifiers(
    group_tables: dict[str, list[GroupTableRow]],
    auto_qualifiers_per_group: int = 2,
) -> dict[str, list[str]]:
    """
    Collect top-N teams per group that qualify automatically.
    """
    qualified: dict[str, list[str]] = {}

    for group_name, rows in group_tables.items():
        if len(rows) < auto_qualifiers_per_group:
            raise ValueError(
                f"Group '{group_name}' has only {len(rows)} ranked rows; "
                f"cannot extract top {auto_qualifiers_per_group}."
            )

        qualified[group_name] = [row.team for row in rows[:auto_qualifiers_per_group]]

    return qualified


def collect_third_place_teams(
    group_tables: dict[str, list[GroupTableRow]],
) -> list[dict[str, Any]]:
    """
    Extract 3rd-placed teams from each group into a flat ranking-ready structure.
    """
    third_place_rows: list[dict[str, Any]] = []

    for group_name, rows in group_tables.items():
        if len(rows) < 3:
            raise ValueError(
                f"Group '{group_name}' must contain at least 3 ranked teams "
                "to extract a third-placed team."
            )

        row = rows[2]
        third_place_rows.append(
            {
                "team": row.team,
                "group": group_name,
                "group_position": 3,
                "points": row.points,
                "goal_difference": row.goal_difference,
                "goals_for": row.goals_for,
                "goals_against": row.goals_against,
                "wins": row.wins,
                "draws": row.draws,
                "losses": row.losses,
                "pre_tournament_elo": row.pre_tournament_elo,
                "tie_break_noise": row.tie_break_noise,
            }
        )

    return third_place_rows


def rank_third_place_teams(
    third_place_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """
    Rank third-placed teams globally.

    Current ranking logic is aligned with the project's current group-stage
    approximation:
        1. points
        2. goal_difference
        3. goals_for
        4. wins
        5. pre_tournament_elo
        6. tie_break_noise

    Note:
    Because v1 does not simulate scorelines yet, goal-based values may remain
    zero unless that capability is added later. The fallback on Elo and tiny
    seeded noise keeps the ranking deterministic and simulation-safe.
    """
    ranked = sorted(
        third_place_rows,
        key=lambda row: (
            row["points"],
            row["goal_difference"],
            row["goals_for"],
            row["wins"],
            row["pre_tournament_elo"],
            row["tie_break_noise"],
        ),
        reverse=True,
    )

    output: list[dict[str, Any]] = []
    for i, row in enumerate(ranked, start=1):
        enriched = dict(row)
        enriched["third_place_rank"] = i
        output.append(enriched)

    return output


def select_best_thirds(
    ranked_third_place_rows: list[dict[str, Any]],
    num_best_thirds: int = 8,
) -> list[dict[str, Any]]:
    """
    Select the best-ranked third-placed teams.
    """
    if len(ranked_third_place_rows) < num_best_thirds:
        raise ValueError(
            f"Cannot select {num_best_thirds} best third-placed teams from only "
            f"{len(ranked_third_place_rows)} rows."
        )

    selected: list[dict[str, Any]] = []

    for i, row in enumerate(ranked_third_place_rows):
        enriched = dict(row)
        enriched["qualified_as_best_third"] = int(i < num_best_thirds)
        if i < num_best_thirds:
            selected.append(enriched)

    return selected


def build_knockout_qualifiers(
    auto_qualifiers_map: dict[str, list[str]],
    best_thirds: list[dict[str, Any]],
) -> list[str]:
    """
    Return a flat list of all knockout-qualified teams.
    """
    qualified: list[str] = []

    for group_name in sorted(auto_qualifiers_map.keys()):
        qualified.extend(auto_qualifiers_map[group_name])

    qualified.extend(row["team"] for row in best_thirds)

    if len(set(qualified)) != len(qualified):
        raise ValueError("Duplicate team detected in knockout qualifiers.")

    return qualified
