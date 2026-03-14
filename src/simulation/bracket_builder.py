from __future__ import annotations

from typing import Any

from src.simulation.structures import GroupTableRow, KnockoutMatch


def build_group_position_index(
    group_tables: dict[str, list[GroupTableRow]],
) -> dict[str, str]:
    """
    Build a team lookup by group-position reference.

    Example:
        {
            "A1": "Spain",
            "A2": "Japan",
            "A3": "Canada",
            "A4": "Ghana",
            ...
        }
    """
    position_index: dict[str, str] = {}

    for group_name, rows in group_tables.items():
        for position, row in enumerate(rows, start=1):
            position_index[f"{group_name}{position}"] = row.team

    return position_index


def build_best_third_index(
    best_thirds: list[dict[str, Any]],
) -> dict[str, str]:
    """
    Build a lookup like:
        {
            "3A": "Canada",
            "3C": "Mexico",
            ...
        }
    """
    index: dict[str, str] = {}

    for row in best_thirds:
        group_name = row["group"]
        team = row["team"]
        ref = f"3{group_name}"

        if ref in index:
            raise ValueError(f"Duplicate best-third reference detected: '{ref}'.")

        index[ref] = team

    return index


def resolve_third_place_assignment_key(
    best_thirds: list[dict[str, Any]],
) -> str:
    """
    Convert the set of qualified third-place groups into a stable lookup key.

    Example:
        A-C-D-F-H-I-J-L
    """
    groups = sorted(row["group"] for row in best_thirds)
    return "-".join(groups)


def build_fallback_third_assignment(
    best_third_index: dict[str, str],
) -> dict[str, str]:
    """
    Development fallback for combinations not explicitly present in the bracket config.

    This does NOT represent the official FIFA 2026 bracket logic.
    It simply maps the selected best-third teams to abstract slots in
    alphabetical order of their group references:

        ["3B", "3C", "3D", "3E", "3G", "3J", "3K", "3L"]

    becomes:

        {
            "3X1": "3B",
            "3X2": "3C",
            ...
            "3X8": "3L"
        }

    This is useful for:
        - smoke tests
        - development
        - validating the simulation engine end-to-end

    Once the official complete mapping is available, this fallback can be
    removed or disabled.
    """
    sorted_best_third_refs = sorted(best_third_index.keys())

    if len(sorted_best_third_refs) != 8:
        raise ValueError(
            "Fallback third-place assignment requires exactly 8 qualified "
            f"best-third references, got {len(sorted_best_third_refs)}."
        )

    return {
        f"3X{i + 1}": ref
        for i, ref in enumerate(sorted_best_third_refs)
    }


def resolve_team_reference(
    ref: str,
    position_index: dict[str, str],
    best_third_index: dict[str, str],
    third_assignment: dict[str, str],
) -> str:
    """
    Resolve a bracket reference into a concrete team name.

    Supported references:
        - fixed group position refs: A1, B2, C3, ...
        - abstract best-third slots: 3X1, 3X2, ...
        - concrete best-third refs: 3A, 3B, ...
    """
    if ref in position_index:
        return position_index[ref]

    if ref in best_third_index:
        return best_third_index[ref]

    if ref in third_assignment:
        concrete_ref = third_assignment[ref]
        if concrete_ref not in best_third_index:
            raise KeyError(
                f"Resolved best-third reference '{concrete_ref}' is not available "
                "among selected best third-placed teams."
            )
        return best_third_index[concrete_ref]

    raise KeyError(f"Unknown bracket reference: '{ref}'.")


def validate_bracket_matches(
    matches: list[KnockoutMatch],
    expected_num_matches: int = 16,
) -> None:
    """
    Validate the initial knockout bracket.
    """
    if len(matches) != expected_num_matches:
        raise ValueError(
            f"Expected {expected_num_matches} matches in initial bracket, "
            f"got {len(matches)}."
        )

    teams: list[str] = []
    for match in matches:
        if not match.team_a or not match.team_b:
            raise ValueError(
                f"Match '{match.slot_id}' contains an empty team reference."
            )
        teams.extend([match.team_a, match.team_b])

    if len(set(teams)) != len(teams):
        raise ValueError("Initial knockout bracket contains duplicate teams.")


def build_round_of_32_bracket(
    group_tables: dict[str, list[GroupTableRow]],
    best_thirds: list[dict[str, Any]],
    bracket_config: dict[str, Any],
) -> list[KnockoutMatch]:
    """
    Build the initial round-of-32 bracket from group results and selected best thirds.

    Expected bracket_config structure:
        {
            "bracket": {
                "fixed_slots": [
                    ["R32_1", "A1", "3X1"],
                    ...
                ],
                "third_place_assignments": {
                    "A-C-D-F-H-I-J-L": {
                        "3X1": "3A",
                        ...
                    }
                }
            }
        }

    Resolution strategy:
        1. Try exact assignment from bracket_config["bracket"]["third_place_assignments"]
        2. If not found, use fallback assignment by alphabetical order of qualified
           best-third group references (development mode)
    """
    position_index = build_group_position_index(group_tables)
    best_third_index = build_best_third_index(best_thirds)
    combination_key = resolve_third_place_assignment_key(best_thirds)

    bracket_rules = bracket_config["bracket"]
    fixed_slots = bracket_rules["fixed_slots"]
    third_place_assignments = bracket_rules.get("third_place_assignments", {})

    if combination_key in third_place_assignments:
        third_assignment = third_place_assignments[combination_key]
    else:
        third_assignment = build_fallback_third_assignment(best_third_index)

    matches: list[KnockoutMatch] = []

    for slot_id, side_a_ref, side_b_ref in fixed_slots:
        team_a = resolve_team_reference(
            ref=side_a_ref,
            position_index=position_index,
            best_third_index=best_third_index,
            third_assignment=third_assignment,
        )
        team_b = resolve_team_reference(
            ref=side_b_ref,
            position_index=position_index,
            best_third_index=best_third_index,
            third_assignment=third_assignment,
        )

        matches.append(
            KnockoutMatch(
                stage="round_of_32",
                slot_id=slot_id,
                team_a=team_a,
                team_b=team_b,
            )
        )

    validate_bracket_matches(matches)
    return matches
