"""Team encoding utilities for RL/ML integration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from .set_catalog import SetCatalog


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

