"""Set catalog for team building."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from .config import PROJECT_ROOT

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

