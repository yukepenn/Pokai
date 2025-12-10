"""Battle result models and persistence utilities."""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from .config import BATTLES_RAW_DIR, BATTLES_JSON_DIR, PROJECT_ROOT


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

