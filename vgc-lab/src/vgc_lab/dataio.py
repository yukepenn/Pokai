"""Data I/O: battle logs, matchup records, and teams_vs_pool datasets.

This module consolidates:
- battle_logger.py: Battle result models and persistence utilities
- team_matchup_data.py: Dataset I/O utilities for pairwise team matchup records
- teams_vs_pool_data.py: Dataset I/O utilities for teams_vs_pool JSONL
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Iterator, Optional

if TYPE_CHECKING:
    from .team_search import CandidateTeamResult

from pydantic import BaseModel, Field

from .config import BATTLES_RAW_DIR, BATTLES_JSON_DIR, DATA_ROOT, PROJECT_ROOT

# ============================================================================
# Battle Logger
# ============================================================================


class BattleResult(BaseModel):
    """Structured battle result data."""

    format_id: str = Field(..., description="Format ID (e.g., gen9vgc2026regf)")
    p1_name: str = Field(..., description="Player 1 name")
    p2_name: str = Field(..., description="Player 2 name")
    winner: str = Field(..., description="Winner: 'p1', 'p2', 'tie', or 'unknown'")
    raw_log_path: Path = Field(..., description="Path to raw battle log file")
    turns: Optional[int] = Field(None, description="Number of turns (if detectable)")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat(),
        }


def save_raw_log(format_id: str, log_text: str) -> Path:
    """
    Save raw battle log to data/battles_raw/<timestamp>_<format_id>.log.

    Args:
        format_id: Format ID (e.g., "gen9vgc2026regf")
        log_text: Full battle log text

    Returns:
        Path to the saved log file
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]  # milliseconds
    filename = f"{timestamp}_{format_id}.log"
    filepath = BATTLES_RAW_DIR / filename

    filepath.write_text(log_text, encoding="utf-8")
    return filepath


def infer_winner_from_log(log_text: str, p1_name: str, p2_name: str) -> str:
    """
    Infer winner from battle log using simple heuristics.

    Looks for lines like:
    - `|win|P1_NAME` or `|win|P2_NAME`
    - `|tie|` or tie-related patterns

    Args:
        log_text: Full battle log text
        p1_name: Player 1 name
        p2_name: Player 2 name

    Returns:
        "p1", "p2", "tie", or "unknown"
    """
    lines = log_text.split("\n")

    for line in lines:
        # Check for win messages
        if line.startswith("|win|"):
            winner_name = line[5:].strip()
            if winner_name == p1_name:
                return "p1"
            elif winner_name == p2_name:
                return "p2"

        # Check for tie patterns
        if "|tie|" in line or "|draw|" in line.lower():
            return "tie"

    return "unknown"


def count_turns(log_text: str) -> Optional[int]:
    """
    Attempt to count turns from battle log.

    Looks for `|turn|N` messages.

    Args:
        log_text: Full battle log text

    Returns:
        Number of turns if detectable, None otherwise
    """
    lines = log_text.split("\n")
    max_turn = 0

    for line in lines:
        if line.startswith("|turn|"):
            try:
                turn_num = int(line[6:].strip())
                max_turn = max(max_turn, turn_num)
            except ValueError:
                continue

    return max_turn if max_turn > 0 else None


def save_battle_result(result: BattleResult) -> Path:
    """
    Save BattleResult as JSON to data/battles_json/<timestamp>_<format_id>.json.

    Args:
        result: BattleResult instance

    Returns:
        Path to the saved JSON file
    """
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
    filename = f"{timestamp}_{result.format_id}.json"
    filepath = BATTLES_JSON_DIR / filename

    # Convert to dict and handle Path/datetime serialization
    data = result.model_dump()
    data["raw_log_path"] = str(result.raw_log_path)
    data["created_at"] = result.created_at.isoformat()

    filepath.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return filepath


class FullBattleRecord(BaseModel):
    """Full battle record for completed battles."""

    format_id: str = Field(..., description="Format ID (e.g., gen9vgc2026regf)")
    p1_name: str = Field(..., description="Player 1 name")
    p2_name: str = Field(..., description="Player 2 name")
    winner_side: str = Field(
        ..., description="Winner side: 'p1', 'p2', 'tie', or 'unknown'"
    )
    winner_name: Optional[str] = Field(
        None, description="Winner player name (null if tie)"
    )
    turns: Optional[int] = Field(None, description="Number of turns")
    raw_log_path: Path = Field(..., description="Path to raw battle log file")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    meta: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat(),
        }


# Full battle dataset location
FULL_BATTLE_DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "full_battles"
FULL_BATTLE_DATASET_PATH = FULL_BATTLE_DATASET_DIR / "full_battles.jsonl"


def ensure_full_battle_dataset_dir() -> None:
    """Ensure the full battle dataset directory exists."""
    FULL_BATTLE_DATASET_DIR.mkdir(parents=True, exist_ok=True)


def append_full_battle_record(record: FullBattleRecord) -> None:
    """
    Append one full battle record as a JSON line to FULL_BATTLE_DATASET_PATH.

    Args:
        record: FullBattleRecord instance to append
    """
    ensure_full_battle_dataset_dir()

    # Convert to dict, handling Path and datetime serialization
    data = record.model_dump(mode="json")
    if record.raw_log_path is not None:
        data["raw_log_path"] = str(record.raw_log_path)
    data["created_at"] = record.created_at.isoformat()

    # Write as a single JSON line
    json_line = json.dumps(data, ensure_ascii=False)
    with FULL_BATTLE_DATASET_PATH.open("a", encoding="utf-8") as f:
        f.write(json_line + "\n")


# ============================================================================
# Team Matchup Data
# ============================================================================

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


# ============================================================================
# Teams vs Pool Data
# ============================================================================

TEAMS_VS_POOL_DIR = DATA_ROOT / "datasets" / "teams_vs_pool"
TEAMS_VS_POOL_JSONL_PATH = TEAMS_VS_POOL_DIR / "teams_vs_pool.jsonl"


def ensure_teams_vs_pool_dir() -> None:
    """Ensure the teams_vs_pool dataset directory exists."""
    TEAMS_VS_POOL_DIR.mkdir(parents=True, exist_ok=True)


def append_result_to_dataset(
    result: "CandidateTeamResult",
    *,
    source: str = "random",
    team_id: Optional[str] = None,
    jsonl_path: Optional[Path] = None,
) -> None:
    """
    Append a CandidateTeamResult to the teams_vs_pool JSONL dataset.

    This preserves the existing JSONL schema used by scripts/teams_vs_pool_dataset.py.

    Args:
        result: CandidateTeamResult to serialize and append.
        source: Source tag for the team (e.g., "random", "catalog", "model_guided_v1").
        team_id: Optional team ID (for catalog teams or named teams).
        jsonl_path: Optional explicit path to JSONL file (defaults to TEAMS_VS_POOL_JSONL_PATH).
    """
    ensure_teams_vs_pool_dir()
    path = jsonl_path or TEAMS_VS_POOL_JSONL_PATH

    summary = result.pool_summary
    win_rate = result.win_rate

    record = {
        "team_set_ids": result.team_set_ids,
        "format_id": summary.format_id,
        "n_opponents": summary.n_opponents,
        "n_battles_per_opponent": summary.n_battles_per_opponent,
        "n_battles_total": summary.n_battles_total,
        "n_wins": summary.n_wins,
        "n_losses": summary.n_losses,
        "n_ties": summary.n_ties,
        "win_rate": win_rate,
        "avg_turns": summary.avg_turns,
        "opponent_counts": summary.opponent_counts,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if team_id is not None:
        record["team_id"] = team_id

    with path.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")
