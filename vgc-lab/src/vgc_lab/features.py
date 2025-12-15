"""Feature extraction and encoding: vocab, bag-of-sets, and dataset iteration.

This module consolidates:
- team_encoding.py: Team encoding utilities for RL/ML integration
- team_features.py: Feature extraction utilities for teams vs pool dataset
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Sequence

from .catalog import SetCatalog

# ============================================================================
# Team Encoding
# ============================================================================


@dataclass
class SetIdVocab:
    """Mapping between set IDs and integer indices (0..N-1)."""

    id_to_idx: Dict[str, int]
    idx_to_id: List[str]

    @classmethod
    def from_catalog(cls, catalog: SetCatalog) -> "SetIdVocab":
        """
        Build a vocabulary from all set IDs in the catalog.

        The order is deterministic (sorted).

        Args:
            catalog: SetCatalog instance.

        Returns:
            SetIdVocab instance.
        """
        ids = sorted(catalog.ids())
        id_to_idx = {set_id: idx for idx, set_id in enumerate(ids)}
        idx_to_id = ids.copy()

        return cls(id_to_idx=id_to_idx, idx_to_id=idx_to_id)

    @classmethod
    def from_idx_to_id(cls, idx_to_id: List[str]) -> "SetIdVocab":
        """
        Build a vocabulary from a list of set IDs (by index order).

        Args:
            idx_to_id: List of set ID strings, where index i corresponds to the ID at position i.

        Returns:
            SetIdVocab instance.
        """
        id_to_idx = {set_id: idx for idx, set_id in enumerate(idx_to_id)}
        return cls(id_to_idx=id_to_idx, idx_to_id=idx_to_id.copy())

    def encode_ids(self, set_ids: Sequence[str]) -> List[int]:
        """
        Encode a sequence of set IDs to integer indices.

        Args:
            set_ids: Sequence of set ID strings.

        Returns:
            List of integer indices.

        Raises:
            KeyError: If any set ID is not in the vocabulary.
        """
        return [self.id_to_idx[set_id] for set_id in set_ids]

    def decode_indices(self, indices: Sequence[int]) -> List[str]:
        """
        Decode a sequence of indices back to set IDs.

        Args:
            indices: Sequence of integer indices.

        Returns:
            List of set ID strings.

        Raises:
            IndexError: If any index is out of range.
        """
        return [self.idx_to_id[idx] for idx in indices]

    def __len__(self) -> int:
        """Return the vocabulary size."""
        return len(self.idx_to_id)


def one_hot_team(
    indices: Sequence[int],
    vocab_size: int,
) -> List[List[int]]:
    """
    Convert a sequence of indices (team) into a list of one-hot vectors.

    Each inner list has length vocab_size and contains 0/1 integers.

    Args:
        indices: Sequence of integer indices (e.g., a team of 6 set indices).
        vocab_size: Size of the vocabulary (total number of possible sets).

    Returns:
        List of one-hot vectors, one per index in the input.

    Raises:
        ValueError: If any index is out of range [0, vocab_size).
    """
    result = []
    for idx in indices:
        if idx < 0 or idx >= vocab_size:
            raise ValueError(
                f"Index {idx} out of range [0, {vocab_size})"
            )
        one_hot = [0] * vocab_size
        one_hot[idx] = 1
        result.append(one_hot)
    return result


def build_vocab_from_default_catalog() -> SetIdVocab:
    """
    Convenience helper: load SetCatalog.from_yaml() and build a SetIdVocab.

    Returns:
        SetIdVocab instance built from the default catalog.
    """
    catalog = SetCatalog.from_yaml()
    return SetIdVocab.from_catalog(catalog)


# ============================================================================
# Team Features
# ============================================================================

from .config import PROJECT_ROOT

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
