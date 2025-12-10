"""Dataset I/O utilities for teams_vs_pool JSONL."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import DATA_ROOT, DEFAULT_FORMAT
from .team_search import CandidateTeamResult

TEAMS_VS_POOL_DIR = DATA_ROOT / "datasets" / "teams_vs_pool"
TEAMS_VS_POOL_JSONL_PATH = TEAMS_VS_POOL_DIR / "teams_vs_pool.jsonl"


def ensure_teams_vs_pool_dir() -> None:
    """Ensure the teams_vs_pool dataset directory exists."""
    TEAMS_VS_POOL_DIR.mkdir(parents=True, exist_ok=True)


def append_result_to_dataset(
    result: CandidateTeamResult,
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

