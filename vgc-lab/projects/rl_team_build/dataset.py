"""PyTorch Dataset for team-building episodes.

This module defines the PyTorch Dataset for TeamBuildEpisode rows,
mapping set_ids to integer indices suitable for training.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Collection, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from vgc_lab import (
    TeamBuildEpisode,
    PokemonSetDef,
    load_sets,
    iter_team_build_episodes,
)


@dataclass
class EncodedTeamEpisode:
    """Internal representation of a team episode with integer indices."""

    set_indices: torch.LongTensor  # shape: (6,)
    reward: torch.FloatTensor  # shape: ()


def build_set_id_index(sets: Dict[str, PokemonSetDef]) -> Tuple[Dict[str, int], List[str]]:
    """
    Build a mapping from set_id -> integer index and a reverse list.

    Args:
        sets: Dict mapping set_id -> PokemonSetDef

    Returns:
        Tuple of (id2idx dict, idx2id list) where idx2id[i] is the set_id at index i.
    """
    set_ids = sorted(sets.keys())  # Deterministic ordering
    id2idx = {sid: idx for idx, sid in enumerate(set_ids)}
    idx2id = set_ids
    return id2idx, idx2id


def encode_episode_to_indices(
    ep: TeamBuildEpisode,
    set_id_to_index: Dict[str, int],
) -> EncodedTeamEpisode:
    """
    Convert a TeamBuildEpisode into integer indices + reward tensor.

    Args:
        ep: TeamBuildEpisode with chosen_set_ids of length 6
        set_id_to_index: Mapping from set_id -> integer index

    Returns:
        EncodedTeamEpisode with set_indices tensor (6,) and reward scalar

    Assumptions:
      - ep.chosen_set_ids has length 6
      - All set_ids exist in set_id_to_index
    """
    indices = [set_id_to_index[sid] for sid in ep.chosen_set_ids]
    set_indices = torch.LongTensor(indices)
    reward = torch.FloatTensor([ep.reward])
    return EncodedTeamEpisode(set_indices=set_indices, reward=reward)


class TeamBuildDataset(Dataset[Tuple[torch.LongTensor, torch.FloatTensor]]):
    """
    PyTorch Dataset wrapping TeamBuildEpisode rows into (team_indices, reward) pairs.

    This dataset loads episodes from the JSONL file and converts them into
    integer-indexed representations suitable for neural network training.
    """

    def __init__(
        self,
        format_id: str = "gen9vgc2026regf",
        allowed_policy_ids: Optional[Collection[str]] = None,
        min_created_at: Optional[datetime] = None,
    ) -> None:
        """Initialize the dataset.

        Args:
            format_id: Format ID to filter episodes (e.g., "gen9vgc2026regf")
            allowed_policy_ids: Optional collection of policy_ids to include (None = all)
            min_created_at: Optional minimum creation date to include episodes
        """
        # Load sets and build index mapping
        sets = load_sets()
        self._id2idx, self._idx2id = build_set_id_index(sets)

        # Load and encode episodes with filtering
        self._encoded_episodes: List[EncodedTeamEpisode] = []
        self._policy_id_counts: Dict[str, int] = {}

        for ep in iter_team_build_episodes():
            # Filter by format_id
            if ep.format_id != format_id:
                continue

            # Filter by policy_id if specified
            if allowed_policy_ids is not None and ep.policy_id not in allowed_policy_ids:
                continue

            # Filter by creation date if specified
            if min_created_at is not None and ep.created_at < min_created_at:
                continue

            # Encode and store
            encoded = encode_episode_to_indices(ep, self._id2idx)
            self._encoded_episodes.append(encoded)

            # Count policy_ids
            self._policy_id_counts[ep.policy_id] = self._policy_id_counts.get(
                ep.policy_id, 0
            ) + 1

    def __len__(self) -> int:
        """Return the number of episodes in the dataset."""
        return len(self._encoded_episodes)

    def __getitem__(self, idx: int) -> Tuple[torch.LongTensor, torch.FloatTensor]:
        """
        Get a single episode.

        Args:
            idx: Index into the dataset

        Returns:
            Tuple of (set_indices, reward) where:
              - set_indices: torch.LongTensor of shape (6,)
              - reward: torch.FloatTensor scalar
        """
        encoded = self._encoded_episodes[idx]
        return encoded.set_indices, encoded.reward

    @property
    def num_sets(self) -> int:
        """Return the total number of unique sets in the catalog."""
        return len(self._idx2id)

    @property
    def idx_to_set_id(self) -> List[str]:
        """Return the list mapping index -> set_id."""
        return self._idx2id

    @property
    def policy_id_counts(self) -> Dict[str, int]:
        """Return a dict mapping policy_id -> count of episodes."""
        return self._policy_id_counts.copy()

