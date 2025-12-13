"""Team-building policy using a trained TeamValueModel.

This module provides a policy interface to score teams and sample strong teams
by evaluating multiple random candidates using a trained value model.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import random

import torch

from vgc_lab import PokemonSetDef, load_sets, sample_team_sets_random
from vgc_lab.core import PROJECT_ROOT
from .dataset import build_set_id_index
from .train_value import TeamValueModel


def load_team_value_checkpoint(
    ckpt_path: Optional[torch.types.PathLike] = None,
    map_location: Optional[str] = None,
) -> Dict:
    """
    Load the team value model checkpoint saved by train_value_model().

    Args:
        ckpt_path: Path to checkpoint file. Defaults to PROJECT_ROOT / "checkpoints" / "team_value.pt"
        map_location: Device to load checkpoint on (e.g., "cpu", "cuda"). Defaults to "cpu"

    Returns:
        A dict with at least:
          - "model_state": state_dict for TeamValueModel
          - "num_sets": number of distinct set_ids used during training

    Raises:
        FileNotFoundError: If the checkpoint file does not exist
    """
    if ckpt_path is None:
        ckpt_path = PROJECT_ROOT / "checkpoints" / "team_value.pt"

    path_obj = Path(ckpt_path) if not isinstance(ckpt_path, Path) else ckpt_path

    if not path_obj.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {path_obj}. "
            "Train a model first using: python -m projects.rl_team_build.train_value"
        )

    return torch.load(path_obj, map_location=map_location or "cpu")


@dataclass
class SetIdIndexMapping:
    """
    Mapping between set_ids and integer indices, shared with the training code.
    """

    id_to_index: Dict[str, int]
    index_to_id: List[str]


def build_mapping_from_sets(
    sets: Dict[str, PokemonSetDef],
) -> SetIdIndexMapping:
    """
    Build a SetIdIndexMapping using the same sorted ordering as the training dataset.

    Args:
        sets: Dict mapping set_id -> PokemonSetDef

    Returns:
        SetIdIndexMapping with id_to_index and index_to_id
    """
    id_to_index, index_to_id = build_set_id_index(sets)
    return SetIdIndexMapping(id_to_index=id_to_index, index_to_id=index_to_id)


class TeamValuePolicy:
    """
    A simple team-building policy based on the trained TeamValueModel.

    Responsibilities:
      - Load the trained value model and set_id index mapping.
      - Score arbitrary 6-mon teams.
      - Sample strong teams by evaluating multiple random candidates.
    """

    def __init__(
        self,
        ckpt_path: Optional[torch.types.PathLike] = None,
        device: Optional[str] = None,
        format_id: str = "gen9vgc2026regf",
    ) -> None:
        """Initialize the policy.

        Loads the trained value model and set_id index mapping, validating
        that the checkpoint is compatible with the current sets catalog.

        Args:
            ckpt_path: Path to checkpoint file. Defaults to PROJECT_ROOT / "checkpoints" / "team_value.pt"
            device: Device to run model on ("cuda" or "cpu"). Defaults to "cuda" if available, else "cpu"
            format_id: Format ID to filter sets (e.g., "gen9vgc2026regf")

        Raises:
            FileNotFoundError: If checkpoint not found
            ValueError: If checkpoint num_sets doesn't match current sets catalog
        """
        # Device selection
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        # Load sets and filter by format
        all_sets = load_sets()
        # Filter sets by format_id - we need all sets for the mapping, but filter for sampling
        self._sets = {sid: s for sid, s in all_sets.items() if s.format == format_id}
        self._format_id = format_id

        # Build id↔index mapping
        self._mapping = build_mapping_from_sets(all_sets)

        # Load checkpoint
        ckpt = load_team_value_checkpoint(ckpt_path, map_location=device)

        # Validate checkpoint consistency with strict checks
        ckpt_set_ids = ckpt.get("set_ids")
        ckpt_num_sets = ckpt.get("num_sets")
        ckpt_format_id = ckpt.get("format_id")

        # Check if checkpoint has new metadata format
        if ckpt_set_ids is None:
            raise ValueError(
                "Checkpoint is from an older version and is incompatible. "
                "The checkpoint format has been updated. Please re-train the value model "
                "using: python -m projects.rl_team_build.train_value"
            )

        # Validate format_id matches
        if ckpt_format_id != format_id:
            raise ValueError(
                f"Checkpoint format_id mismatch: checkpoint is for '{ckpt_format_id}', "
                f"but requested format is '{format_id}'. "
                "Please use a checkpoint trained for the requested format."
            )

        # Strict compatibility check: both count and exact list must match
        if ckpt_num_sets != len(self._mapping.index_to_id):
            raise ValueError(
                f"TeamValue checkpoint is incompatible with current sets_regf.yaml. "
                f"Checkpoint expects {ckpt_num_sets} total sets, "
                f"but current catalog has {len(self._mapping.index_to_id)} total sets. "
                "Please re-train the value model."
            )

        if ckpt_set_ids != self._mapping.index_to_id:
            raise ValueError(
                f"TeamValue checkpoint is incompatible with current sets_regf.yaml. "
                f"The set_ids list in the checkpoint does not match the current catalog. "
                "The catalog may have been updated. Please re-train the value model."
            )

        # Instantiate and load model
        self._model = TeamValueModel(num_sets=len(self._mapping.index_to_id))
        self._model.load_state_dict(ckpt["model_state"])
        self._model.to(device)
        self._model.eval()

    def score_team(self, set_ids: List[str]) -> float:
        """Compute the predicted value for a given 6-mon team.

        Args:
            set_ids: List of 6 set_ids

        Returns:
            Scalar value (float) predicted by the model

        Raises:
            ValueError: If set_ids doesn't have length 6
            KeyError: If any set_id is not in the mapping
        """
        if len(set_ids) != 6:
            raise ValueError(f"Expected 6 set_ids, got {len(set_ids)}")

        # Map set_ids to indices
        try:
            indices = [self._mapping.id_to_index[sid] for sid in set_ids]
        except KeyError as e:
            missing_id = str(e).strip("'")
            raise KeyError(
                f"Unknown set_id: {missing_id}. "
                f"Available set_ids are from the catalog loaded during training."
            ) from e

        # Build tensor
        team_indices = torch.LongTensor([indices])  # shape: (1, 6)
        team_indices = team_indices.to(self.device)

        # Forward pass
        with torch.no_grad():
            pred = self._model(team_indices)  # shape: (1,)

        return float(pred.item())

    def sample_team(
        self,
        n_candidates: int = 128,
        rng: Optional[random.Random] = None,
    ) -> List[str]:
        """Sample a strong 6-mon team by evaluating multiple random candidates.

        Args:
            n_candidates: Number of random candidate teams to consider (default: 128)
            rng: Optional random.Random instance for reproducibility

        Returns:
            List of 6 set_ids forming the selected team.

        Raises:
            RuntimeError: If sample_team_sets_random fails (e.g., not enough distinct species/items)
        """
        if n_candidates <= 0:
            n_candidates = 1

        rng = rng or random.Random()

        best_score = float("-inf")
        best_team: Optional[List[str]] = None

        for _ in range(n_candidates):
            try:
                candidate_set_ids = sample_team_sets_random(
                    self._sets, format_id=self._format_id, rng=rng
                )
                score = self.score_team(candidate_set_ids)

                if score > best_score:
                    best_score = score
                    best_team = candidate_set_ids
            except RuntimeError as e:
                # Propagate with helpful message if sampling fails
                raise RuntimeError(
                    f"Failed to sample candidate team: {e}. "
                    "This may indicate insufficient distinct species/items in the catalog."
                ) from e

        if best_team is None:
            raise RuntimeError("No valid team candidates found")

        return best_team


if __name__ == "__main__":
    policy = TeamValuePolicy()
    team = policy.sample_team(n_candidates=32)
    print("Sampled team (set_ids):", team)
    score = policy.score_team(team)
    print("Predicted value:", score)

