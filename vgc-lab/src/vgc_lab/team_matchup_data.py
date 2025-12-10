"""Dataset I/O utilities for pairwise team matchup records."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional

from .config import DATA_ROOT

MATCHUP_DATASET_DIR = DATA_ROOT / "datasets" / "team_matchups"
MATCHUP_JSONL_PATH = MATCHUP_DATASET_DIR / "team_matchups.jsonl"


@dataclass
class TeamMatchupRecord:
    """Single record of team A vs team B matchup statistics."""

    # Team identities (if known – may be None for random teams)
    team_a_id: Optional[str]
    team_b_id: Optional[str]

    # Team compositions (as set_ids, consistent with SetCatalog)
    team_a_set_ids: list[str]
    team_b_set_ids: list[str]

    format_id: str

    # Evaluation stats
    n_battles: int
    n_a_wins: int
    n_b_wins: int
    n_ties: int
    avg_turns: Optional[float]

    # Metadata
    created_at: str  # ISO timestamp
    meta: Dict[str, object] = field(default_factory=dict)  # free-form, e.g. {"source": "catalog_vs_auto_top", "note": "..."}


def ensure_matchup_dir() -> None:
    """Create MATCHUP_DATASET_DIR if it doesn't exist."""
    MATCHUP_DATASET_DIR.mkdir(parents=True, exist_ok=True)


def to_json_dict(record: TeamMatchupRecord) -> Dict[str, object]:
    """
    Convert TeamMatchupRecord to a JSON-serializable dictionary.

    Args:
        record: TeamMatchupRecord instance.

    Returns:
        Dictionary ready for json.dumps.
    """
    return {
        "team_a_id": record.team_a_id,
        "team_b_id": record.team_b_id,
        "team_a_set_ids": record.team_a_set_ids,
        "team_b_set_ids": record.team_b_set_ids,
        "format_id": record.format_id,
        "n_battles": record.n_battles,
        "n_a_wins": record.n_a_wins,
        "n_b_wins": record.n_b_wins,
        "n_ties": record.n_ties,
        "avg_turns": record.avg_turns,
        "created_at": record.created_at,
        **record.meta,  # Merge meta fields at top level for convenience
    }


def from_json_dict(d: Dict[str, object]) -> TeamMatchupRecord:
    """
    Create TeamMatchupRecord from a JSON dictionary.

    Args:
        d: Dictionary loaded from JSON.

    Returns:
        TeamMatchupRecord instance.
    """
    # Extract main fields
    team_a_id = d.get("team_a_id")
    team_b_id = d.get("team_b_id")
    team_a_set_ids = d.get("team_a_set_ids", [])
    team_b_set_ids = d.get("team_b_set_ids", [])
    format_id = d.get("format_id", "gen9vgc2026regf")
    n_battles = int(d.get("n_battles", 0))
    n_a_wins = int(d.get("n_a_wins", 0))
    n_b_wins = int(d.get("n_b_wins", 0))
    n_ties = int(d.get("n_ties", 0))
    avg_turns = d.get("avg_turns")
    if avg_turns is not None:
        avg_turns = float(avg_turns)
    created_at = d.get("created_at", datetime.now(timezone.utc).isoformat())

    # Extract meta (everything else)
    main_fields = {
        "team_a_id",
        "team_b_id",
        "team_a_set_ids",
        "team_b_set_ids",
        "format_id",
        "n_battles",
        "n_a_wins",
        "n_b_wins",
        "n_ties",
        "avg_turns",
        "created_at",
    }
    meta = {k: v for k, v in d.items() if k not in main_fields}

    return TeamMatchupRecord(
        team_a_id=team_a_id,
        team_b_id=team_b_id,
        team_a_set_ids=list(team_a_set_ids),
        team_b_set_ids=list(team_b_set_ids),
        format_id=format_id,
        n_battles=n_battles,
        n_a_wins=n_a_wins,
        n_b_wins=n_b_wins,
        n_ties=n_ties,
        avg_turns=avg_turns,
        created_at=str(created_at),
        meta=meta,
    )


def append_matchup_record(
    record: TeamMatchupRecord, path: Optional[Path] = None
) -> None:
    """
    Append a TeamMatchupRecord as a JSON line to the dataset.

    Args:
        record: TeamMatchupRecord to serialize and append.
        path: Optional explicit path to JSONL file (defaults to MATCHUP_JSONL_PATH).
    """
    ensure_matchup_dir()
    output_path = path or MATCHUP_JSONL_PATH

    data = to_json_dict(record)

    with output_path.open("a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


def iter_team_matchup_records(
    path: Optional[Path] = None,
) -> Iterator[TeamMatchupRecord]:
    """
    Iterate over team matchup records in a JSONL file.

    Args:
        path: Path to JSONL file (defaults to MATCHUP_JSONL_PATH).

    Yields:
        TeamMatchupRecord instances.
    """
    input_path = path or MATCHUP_JSONL_PATH

    if not input_path.exists():
        return

    with input_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
                yield from_json_dict(obj)
            except (json.JSONDecodeError, ValueError, KeyError) as e:
                # Skip malformed lines with a warning?
                continue

