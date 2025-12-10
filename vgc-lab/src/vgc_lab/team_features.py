"""Feature extraction utilities for teams vs pool dataset."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List

import json

from .config import PROJECT_ROOT
from .team_encoding import SetIdVocab, build_vocab_from_default_catalog

DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"


@dataclass
class TeamsVsPoolRecord:
    """A single record from the teams_vs_pool dataset."""

    team_set_ids: List[str]
    win_rate: float
    n_battles_total: int
    source: str
    meta: Dict[str, object]


def iter_teams_vs_pool_records(
    path: Path,
) -> Iterator[TeamsVsPoolRecord]:
    """
    Iterate over records in a teams_vs_pool JSONL file.

    Expected fields in each JSON object:
      - "team_set_ids": list[str]
      - "win_rate": float
      - "n_battles_total": int
      - "source": str  (e.g., "random" or "catalog")
      - plus any additional fields (stored in meta)

    Args:
        path: Path to the JSONL file.

    Yields:
        TeamsVsPoolRecord instances.
    """
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            obj = json.loads(line)
            team_set_ids = obj.get("team_set_ids", [])
            win_rate = float(obj.get("win_rate", 0.0))
            n_battles_total = int(obj.get("n_battles_total", 0))
            source = obj.get("source", "unknown")

            meta = dict(obj)
            # Remove main fields from meta to avoid duplication
            for key in ("team_set_ids", "win_rate", "n_battles_total", "source"):
                meta.pop(key, None)

            yield TeamsVsPoolRecord(
                team_set_ids=team_set_ids,
                win_rate=win_rate,
                n_battles_total=n_battles_total,
                source=source,
                meta=meta,
            )


def team_bag_of_sets_feature(
    team_set_ids: List[str],
    vocab: SetIdVocab,
) -> List[int]:
    """
    Build a bag-of-sets feature vector for a team.

    - Dimension = len(vocab.idx_to_id)
    - Each position i counts how many times that set appears in the team
      (for now, it's usually 0 or 1).

    Args:
        team_set_ids: List of set ID strings.
        vocab: SetIdVocab instance for encoding.

    Returns:
        List of integers (length = vocab size) representing bag-of-sets counts.
    """
    vec = [0] * len(vocab.idx_to_id)
    indices = vocab.encode_ids(team_set_ids)
    for idx in indices:
        if 0 <= idx < len(vec):
            vec[idx] += 1
    return vec


def build_default_vocab() -> SetIdVocab:
    """
    Convenience: just call build_vocab_from_default_catalog().

    Returns:
        SetIdVocab instance built from the default catalog.
    """
    return build_vocab_from_default_catalog()

