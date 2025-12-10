"""Team pool abstraction for opponent sampling."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml

from .config import PROJECT_ROOT

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

