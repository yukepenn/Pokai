"""Team building utilities using set catalog."""

from __future__ import annotations

from typing import Iterable, List, Optional

from .config import DEFAULT_FORMAT
from .eval import evaluate_team_vs_random, evaluate_team_vs_team
from .set_catalog import SetCatalog
from .teams import import_text_to_packed

# Import the summary types for return annotations
from .eval import MatchupEvalSummary, TeamEvalSummary


def build_team_import_from_set_ids(
    set_ids: Iterable[str],
    *,
    catalog: Optional[SetCatalog] = None,
) -> str:
    """
    Given a sequence of set IDs, build a Showdown team import string
    by concatenating the import_text of each SetEntry, separated by blank lines.

    Args:
        set_ids: Sequence of set IDs from the catalog.
        catalog: SetCatalog instance. If None, loads from default catalog path.

    Returns:
        Complete Showdown team import string (6 Pokemon).

    Raises:
        KeyError: If any set_id is not found in the catalog.
        ValueError: If set_ids contains duplicate IDs.
    """
    if catalog is None:
        catalog = SetCatalog.from_yaml()

    set_ids_list = list(set_ids)
    
    # Validate: should be exactly 6 sets
    if len(set_ids_list) != 6:
        raise ValueError(f"Expected exactly 6 set IDs, got {len(set_ids_list)}")

    # Validate: no duplicates
    if len(set_ids_list) != len(set(set_ids_list)):
        raise ValueError("Duplicate set IDs found in team")

    entries = [catalog.get(sid) for sid in set_ids_list]

    # Optional: validate that all entries have compatible format_id
    format_ids = {entry.format_id for entry in entries}
    if len(format_ids) > 1:
        raise ValueError(
            f"Mixed format IDs found: {format_ids}. "
            "All sets in a team should have the same format."
        )

    lines: List[str] = []
    for entry in entries:
        # Strip trailing whitespace for cleanliness, but keep internal structure
        text = entry.import_text.strip()
        lines.append(text)

    return "\n\n".join(lines) + "\n"


def evaluate_catalog_team_vs_random(
    set_ids: Iterable[str],
    *,
    format_id: str = DEFAULT_FORMAT,
    n: int = 100,
    team_as: str = "both",
    save_logs: bool = False,
    catalog: Optional[SetCatalog] = None,
) -> TeamEvalSummary:
    """
    Convenience wrapper to evaluate a catalog-defined team vs random opponents.

    This function:
    - Loads the set catalog (if not provided).
    - Builds a team import from set_ids.
    - Converts it to a packed team string.
    - Calls evaluate_team_vs_random(team_packed=...).

    Args:
        set_ids: Sequence of 6 set IDs from the catalog.
        format_id: Format ID for evaluation.
        n: Number of battles to run.
        team_as: Which side to place the team on: "p1", "p2", or "both".
        save_logs: If True, save each battle log to data/battles_raw/.
        catalog: SetCatalog instance. If None, loads from default catalog path.

    Returns:
        TeamEvalSummary from evaluate_team_vs_random.
    """
    if catalog is None:
        catalog = SetCatalog.from_yaml()

    team_import = build_team_import_from_set_ids(set_ids, catalog=catalog)
    team_packed = import_text_to_packed(team_import, format_id=format_id)

    summary = evaluate_team_vs_random(
        team_packed=team_packed,
        format_id=format_id,
        n=n,
        team_as=team_as,
        save_logs=save_logs,
    )
    return summary


def evaluate_catalog_matchup(
    team_a_set_ids: Iterable[str],
    team_b_set_ids: Iterable[str],
    *,
    format_id: str = DEFAULT_FORMAT,
    n: int = 100,
    catalog: Optional[SetCatalog] = None,
) -> MatchupEvalSummary:
    """
    Convenience wrapper to evaluate a matchup between two catalog-defined teams.

    This function:
    - Builds import text for Team A and Team B from set IDs.
    - Calls evaluate_team_vs_team(team_a_import, team_b_import, ...).

    Args:
        team_a_set_ids: Sequence of 6 set IDs for Team A.
        team_b_set_ids: Sequence of 6 set IDs for Team B.
        format_id: Format ID for evaluation.
        n: Number of battles to run.
        catalog: SetCatalog instance. If None, loads from default catalog path.

    Returns:
        MatchupEvalSummary from evaluate_team_vs_team.
    """
    if catalog is None:
        catalog = SetCatalog.from_yaml()

    team_a_import = build_team_import_from_set_ids(team_a_set_ids, catalog=catalog)
    team_b_import = build_team_import_from_set_ids(team_b_set_ids, catalog=catalog)

    summary = evaluate_team_vs_team(
        team_a_import=team_a_import,
        team_b_import=team_b_import,
        format_id=format_id,
        n=n,
    )
    return summary

