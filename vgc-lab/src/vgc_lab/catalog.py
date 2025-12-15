"""Catalog and team building: sets, teams, pools, and team construction.

This module consolidates:
- set_catalog.py: Set catalog for team building
- team_pool.py: Team pool abstraction for opponent sampling
- teams.py: Team format conversion utilities
- team_builder.py: Team building utilities using set catalog
"""

from __future__ import annotations

import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from .config import DEFAULT_FORMAT, PROJECT_ROOT

# ============================================================================
# Set Catalog
# ============================================================================

CATALOG_PATH = PROJECT_ROOT / "data" / "catalog" / "sets_regf.yaml"


@dataclass
class SetEntry:
    """A single Pokemon set entry in the catalog."""

    id: str
    format_id: str
    import_text: str
    description: Optional[str] = None


class SetCatalog:
    """Catalog of Pokemon sets for team building."""

    def __init__(self, entries: Dict[str, SetEntry]):
        """Initialize catalog with a dictionary of set_id -> SetEntry."""
        self._entries = entries

    @classmethod
    def from_yaml(cls, path: Path = CATALOG_PATH) -> "SetCatalog":
        """
        Load the catalog from the given YAML path.

        Args:
            path: Path to the YAML catalog file.

        Returns:
            SetCatalog instance.

        Raises:
            FileNotFoundError: If the catalog file doesn't exist.
            ValueError: If the YAML structure is invalid.
        """
        if not path.exists():
            raise FileNotFoundError(f"Catalog file not found: {path}")

        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)

        if not isinstance(data, list):
            raise ValueError(f"Expected YAML root to be a list, got {type(data)}")

        entries: Dict[str, SetEntry] = {}
        for entry_dict in data:
            if not isinstance(entry_dict, dict):
                raise ValueError(f"Expected entry to be a dict, got {type(entry_dict)}")

            set_id = entry_dict.get("id")
            if not set_id:
                raise ValueError("Entry missing required 'id' field")

            format_id = entry_dict.get("format", "gen9vgc2026regf")
            import_text = entry_dict.get("import_text", "")
            description = entry_dict.get("description")

            entry = SetEntry(
                id=set_id,
                format_id=format_id,
                import_text=import_text,
                description=description,
            )
            entries[set_id] = entry

        return cls(entries)

    def get(self, set_id: str) -> SetEntry:
        """
        Return a SetEntry by id.

        Args:
            set_id: The set identifier.

        Returns:
            SetEntry instance.

        Raises:
            KeyError: If the set_id is not found in the catalog.
        """
        if set_id not in self._entries:
            raise KeyError(f"Set ID not found in catalog: {set_id}")
        return self._entries[set_id]

    def ids(self) -> List[str]:
        """
        Return all set IDs in the catalog.

        Returns:
            List of set ID strings.
        """
        return list(self._entries.keys())

    def __len__(self) -> int:
        """Return the number of sets in the catalog."""
        return len(self._entries)

    def __contains__(self, set_id: str) -> bool:
        """Check if a set_id exists in the catalog."""
        return set_id in self._entries


# ============================================================================
# Team Pool
# ============================================================================

TEAMS_CATALOG_PATH = PROJECT_ROOT / "data" / "catalog" / "teams_regf.yaml"


@dataclass
class TeamDef:
    """Definition of a team by its set IDs."""

    id: str
    format_id: str
    set_ids: List[str]
    description: Optional[str] = None


class TeamPool:
    """Pool of team definitions for opponent sampling."""

    def __init__(self, teams: Dict[str, TeamDef]):
        """Initialize pool with a dictionary of team_id -> TeamDef."""
        self._teams = teams

    @classmethod
    def from_yaml(cls, path: Path = TEAMS_CATALOG_PATH) -> "TeamPool":
        """
        Load team definitions from a YAML file.

        Args:
            path: Path to the YAML teams catalog file.

        Returns:
            TeamPool instance.

        Raises:
            FileNotFoundError: If the catalog file doesn't exist.
            ValueError: If the YAML structure is invalid.
        """
        if not path.exists():
            raise FileNotFoundError(f"Teams catalog file not found: {path}")

        text = path.read_text(encoding="utf-8")
        data = yaml.safe_load(text)

        if not isinstance(data, list):
            raise ValueError(f"Expected YAML root to be a list, got {type(data)}")

        teams: Dict[str, TeamDef] = {}
        for entry_dict in data:
            if not isinstance(entry_dict, dict):
                raise ValueError(f"Expected entry to be a dict, got {type(entry_dict)}")

            team_id = entry_dict.get("id")
            if not team_id:
                raise ValueError("Entry missing required 'id' field")

            format_id = entry_dict.get("format", "gen9vgc2026regf")
            set_ids = entry_dict.get("set_ids", [])
            description = entry_dict.get("description")

            if not isinstance(set_ids, list):
                raise ValueError(f"Expected 'set_ids' to be a list, got {type(set_ids)}")

            if not all(isinstance(sid, str) for sid in set_ids):
                raise ValueError("All 'set_ids' must be strings")

            if len(set_ids) != 6:
                raise ValueError(f"Expected exactly 6 set IDs per team, got {len(set_ids)}")

            team_def = TeamDef(
                id=team_id,
                format_id=format_id,
                set_ids=set_ids,
                description=description,
            )
            teams[team_id] = team_def

        return cls(teams)

    def ids(self) -> List[str]:
        """
        Return a list of team IDs in the pool.

        Returns:
            List of team ID strings.
        """
        return list(self._teams.keys())

    def get(self, team_id: str) -> TeamDef:
        """
        Return a TeamDef by ID.

        Args:
            team_id: The team identifier.

        Returns:
            TeamDef instance.

        Raises:
            KeyError: If the team_id is not found in the pool.
        """
        if team_id not in self._teams:
            raise KeyError(f"Team ID not found in pool: {team_id}")
        return self._teams[team_id]

    def all(self) -> List[TeamDef]:
        """
        Return all TeamDefs in the pool.

        Returns:
            List of TeamDef instances.
        """
        return list(self._teams.values())

    def sample(
        self,
        n: int = 1,
        *,
        exclude_ids: Optional[Iterable[str]] = None,
        rng=None,
    ) -> List[TeamDef]:
        """
        Sample n teams from the pool (with replacement).

        Args:
            n: Number of teams to sample.
            exclude_ids: Optional set of team IDs to exclude from sampling.
            rng: Optional random number generator. If None, uses random module.

        Returns:
            List of TeamDef instances (may contain duplicates if n > pool size).
        """
        if rng is None:
            rng = random

        available_ids = list(self._teams.keys())
        if exclude_ids:
            exclude_set = set(exclude_ids)
            available_ids = [tid for tid in available_ids if tid not in exclude_set]

        if not available_ids:
            raise ValueError(
                "No teams available for sampling (possibly all excluded)"
            )

        sampled_ids = rng.choices(available_ids, k=n)
        return [self._teams[tid] for tid in sampled_ids]

    def __len__(self) -> int:
        """Return the number of teams in the pool."""
        return len(self._teams)

    def __contains__(self, team_id: str) -> bool:
        """Check if a team_id exists in the pool."""
        return team_id in self._teams


# ============================================================================
# Team Format Conversion
# ============================================================================


def import_text_to_packed(
    import_text: str,
    *,
    format_id: str = DEFAULT_FORMAT,
    timeout_seconds: int = 30,
) -> str:
    """
    Convert Showdown import/export team text to a packed team string
    using the Node script js/import_to_packed.js.

    Args:
        import_text: Multi-line Showdown team text (6 Pokémon).
        format_id:   Format ID for validation context (default: gen9vgc2026regf).
        timeout_seconds: Subprocess timeout.

    Returns:
        Packed team string (suitable for Teams.unpack and run_random_selfplay_json).

    Raises:
        RuntimeError on conversion failure.
    """
    cmd = [
        "node",
        str(PROJECT_ROOT / "js" / "import_to_packed.js"),
        "--format",
        format_id,
    ]
    
    proc = subprocess.run(
        cmd,
        input=import_text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        cwd=str(PROJECT_ROOT),
    )
    
    if proc.returncode != 0:
        raise RuntimeError(
            f"import_text_to_packed failed with code {proc.returncode}: "
            f"{proc.stderr.decode('utf-8', errors='ignore')}"
        )

    packed = proc.stdout.decode("utf-8", errors="strict").strip()
    if not packed:
        raise RuntimeError("import_text_to_packed: empty packed result")
    
    return packed


# ============================================================================
# Team Building
# ============================================================================

# Note: These functions import from .analysis.


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
):
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
    from .analysis import evaluate_team_vs_random, TeamEvalSummary

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
):
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
    from .analysis import evaluate_team_vs_team, MatchupEvalSummary

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
