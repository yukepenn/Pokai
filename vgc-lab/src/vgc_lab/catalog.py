"""
Catalog module for loading and managing Pokemon sets from YAML files.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import random
import yaml

from .core import PROJECT_ROOT, ShowdownClient

CATALOG_ROOT = PROJECT_ROOT / "data" / "catalog"


class PokemonSetDef:
    """Represents a single Pokemon set definition from sets_regf.yaml."""

    def __init__(
        self,
        id: str,
        format: str,
        description: str,
        import_text: str,
        species: str,
        item: Optional[str],
    ) -> None:
        self.id = id
        self.format = format
        self.description = description
        self.import_text = import_text
        self.species = species
        self.item = item


def _normalize_species(raw: str) -> str:
    """
    Normalize species names to something stable across nickname/gender variations.
    """
    raw = raw.strip()
    # Strip gender suffixes like "(M)" or "(F)"
    if raw.endswith("(M)") or raw.endswith("(F)"):
        return raw[:-3].strip()
    # Handle nickname patterns like "Nickname (Species)"
    m = re.match(r"^(.*)\(([^()]+)\)\s*$", raw)
    if m:
        nickname_part, species_part = m.groups()
        return species_part.strip()
    return raw


def _parse_species_item(import_text: str) -> tuple[str, Optional[str]]:
    """
    Parse species and item from the first line of a Showdown export.

    Examples:
        "Tornadus @ Focus Sash" -> ("Tornadus", "Focus Sash")
        "Amoonguss" -> ("Amoonguss", None)
        "Dejoy (Tornadus) @ Focus Sash" -> ("Tornadus", "Focus Sash")
        "Tornadus (M) @ Focus Sash" -> ("Tornadus", "Focus Sash")
    """
    first_line = import_text.strip().splitlines()[0]
    if "@" in first_line:
        left, right = first_line.split("@", 1)
        species = _normalize_species(left)
        item = right.strip() or None
        return species, item
    species = _normalize_species(first_line)
    return species, None


def load_sets(path: Optional[Path] = None) -> Dict[str, PokemonSetDef]:
    """
    Load all set definitions from sets_regf.yaml.

    Returns:
        A dict mapping set_id -> PokemonSetDef.

    Raises:
        ValueError: If duplicate set ids are found in the YAML file.
    """
    if path is None:
        path = CATALOG_ROOT / "sets_regf.yaml"
    rows = yaml.safe_load(path.read_text(encoding="utf-8")) or []
    sets: Dict[str, PokemonSetDef] = {}
    for row in rows:
        set_id = row["id"]
        if set_id in sets:
            raise ValueError(f"Duplicate set id in sets_regf.yaml: {set_id}")
        import_text = row["import_text"]
        species, item = _parse_species_item(import_text)
        s = PokemonSetDef(
            id=set_id,
            format=row["format"],
            description=row.get("description", ""),
            import_text=import_text,
            species=species,
            item=item,
        )
        sets[set_id] = s
    return sets


def sample_team_sets_random(
    sets: Dict[str, PokemonSetDef],
    format_id: str = "gen9vgc2026regf",
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    Randomly sample 6 set_ids that form a legal team (no duplicate species or items).

    Args:
        sets: Dict mapping set_id -> PokemonSetDef
        format_id: Format to filter sets by
        rng: Optional random number generator

    Returns:
        List of 6 set_ids

    Raises:
        RuntimeError: If not enough distinct species/items to form a team of 6
    """
    rng = rng or random.Random()
    pool = [s for s in sets.values() if s.format == format_id]
    rng.shuffle(pool)

    chosen: List[PokemonSetDef] = []
    used_species: set[str] = set()
    used_items: set[str] = set()

    for s in pool:
        if s.species in used_species:
            continue
        if s.item is not None and s.item in used_items:
            continue
        chosen.append(s)
        used_species.add(s.species)
        if s.item is not None:
            used_items.add(s.item)
        if len(chosen) == 6:
            break

    if len(chosen) < 6:
        raise RuntimeError("Not enough distinct species/items to form a team of 6")

    return [s.id for s in chosen]


def build_export_from_set_ids(
    set_ids: List[str],
    sets: Dict[str, PokemonSetDef],
) -> str:
    """
    Concatenate the `import_text` of each set_id into a single Showdown export
    (blank line between mons).
    """
    texts = [sets[sid].import_text.strip() for sid in set_ids]
    return "\n\n".join(texts)


def build_packed_team_from_set_ids(
    set_ids: List[str],
    sets: Dict[str, PokemonSetDef],
    client: ShowdownClient,
) -> str:
    """
    Construct a packed team string from a list of set_ids using ShowdownClient.pack_team().
    """
    export_text = build_export_from_set_ids(set_ids, sets)
    return client.pack_team(export_text)

