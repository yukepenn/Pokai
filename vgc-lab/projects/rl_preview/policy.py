"""Team-preview policy using a trained PreviewModel.

This module provides a policy interface to score bring-4 actions and choose
which 4 mons to bring given 6v6 teams.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import random

import torch
import torch.nn as nn

from vgc_lab.core import PROJECT_ROOT
from vgc_lab.catalog import load_sets
from .dataset import BRING4_COMBOS
from .train_preview import PreviewModel


@dataclass
class SetIdMapping:
    """Mapping between set_ids and indices for the preview model."""

    set_ids: List[str]
    id_to_index: dict[str, int]


def _build_mapping_from_ckpt(
    ckpt: dict, format_id: str
) -> SetIdMapping:
    """Build SetIdMapping from checkpoint, validating compatibility.

    Args:
        ckpt: Checkpoint dict with "set_ids", "num_sets", "format_id"
        format_id: Expected format ID

    Returns:
        SetIdMapping with set_ids and id_to_index

    Raises:
        ValueError: If checkpoint is incompatible with current catalog/format
    """
    ckpt_set_ids = ckpt.get("set_ids")
    ckpt_num_sets = ckpt.get("num_sets")
    ckpt_format_id = ckpt.get("format_id")

    if ckpt_set_ids is None:
        raise ValueError(
            "Preview checkpoint missing 'set_ids'. "
            "Please re-train using: python -m projects.rl_preview.train_preview"
        )

    if ckpt_num_sets is None:
        raise ValueError(
            "Preview checkpoint missing 'num_sets'. "
            "Please re-train using: python -m projects.rl_preview.train_preview"
        )

    if ckpt_format_id != format_id:
        raise ValueError(
            f"Preview checkpoint format_id mismatch: checkpoint is for '{ckpt_format_id}', "
            f"but requested format is '{format_id}'. "
            "Please use a checkpoint trained for the requested format."
        )

    if len(ckpt_set_ids) != ckpt_num_sets:
        raise ValueError(
            f"Preview checkpoint inconsistency: len(set_ids)={len(ckpt_set_ids)} "
            f"but num_sets={ckpt_num_sets}. "
            "Please re-train using: python -m projects.rl_preview.train_preview"
        )

    # Load current catalog sets and filter by format
    all_sets = load_sets()
    format_sets = {sid: s for sid, s in all_sets.items() if s.format == format_id}
    current_format_set_ids = sorted(format_sets.keys())

    # Verify exact match
    if current_format_set_ids != ckpt_set_ids:
        raise ValueError(
            "Preview checkpoint is incompatible with current catalog / format. "
            "The set_ids list in the checkpoint does not match the current catalog. "
            "Please re-train using: python -m projects.rl_preview.train_preview"
        )

    # Build id_to_index mapping
    id_to_index = {sid: i for i, sid in enumerate(ckpt_set_ids)}

    return SetIdMapping(set_ids=ckpt_set_ids, id_to_index=id_to_index)


class PreviewPolicy:
    """A team-preview policy based on the trained PreviewModel.

    Responsibilities:
    - Load the trained preview model and set_id mapping.
    - Score all 15 bring-4 actions given (self_team, opp_team).
    - Choose an action (argmax).
    - Convert action index to Showdown "team XXXX" command string.
    """

    def __init__(
        self,
        format_id: str = "gen9vgc2026regf",
        ckpt_path: Optional[Path] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> None:
        """Initialize the policy.

        Loads the trained preview model and validates checkpoint compatibility.

        Args:
            format_id: Format ID (e.g., "gen9vgc2026regf")
            ckpt_path: Path to checkpoint file. Defaults to PROJECT_ROOT / "checkpoints/preview_bring4.pt"
            device: Device to run model on ("cuda" or "cpu")

        Raises:
            FileNotFoundError: If checkpoint not found
            ValueError: If checkpoint is incompatible with current catalog/format
        """
        if ckpt_path is None:
            ckpt_path = PROJECT_ROOT / "checkpoints" / "preview_bring4.pt"

        path_obj = Path(ckpt_path) if not isinstance(ckpt_path, Path) else ckpt_path

        if not path_obj.exists():
            raise FileNotFoundError(
                f"Checkpoint not found at {path_obj}. "
                "Train a model first using: python -m projects.rl_preview.train_preview"
            )

        # Load checkpoint
        ckpt = torch.load(path_obj, map_location=device)

        # Build mapping and validate compatibility
        self.mapping = _build_mapping_from_ckpt(ckpt, format_id)
        self.format_id = format_id
        self.device = device

        # Get hyperparams from checkpoint (with fallback to defaults)
        hyperparams = ckpt.get("hyperparams", {})
        embed_dim = hyperparams.get("embed_dim", 64)
        hidden_dim = hyperparams.get("hidden_dim", 128)

        # Create and load model
        self.model = PreviewModel(
            num_sets=len(self.mapping.set_ids),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
        )
        self.model.load_state_dict(ckpt["model_state"])
        self.model.to(device)
        self.model.eval()

    def _encode_team(self, set_ids: List[str]) -> torch.Tensor:
        """Encode a list of set_ids into a tensor of indices.

        Args:
            set_ids: List of 6 set_ids

        Returns:
            Tensor of shape (1, 6) on self.device

        Raises:
            KeyError: If any set_id is not in the mapping
        """
        if len(set_ids) != 6:
            raise ValueError(f"Expected 6 set_ids, got {len(set_ids)}")

        try:
            indices = [self.mapping.id_to_index[sid] for sid in set_ids]
        except KeyError as e:
            missing_id = str(e).strip("'")
            raise KeyError(
                f"Unknown set_id: {missing_id}. "
                f"Available set_ids are from the catalog loaded during training."
            ) from e

        return torch.tensor([indices], dtype=torch.long, device=self.device)

    def score_actions(
        self,
        self_set_ids: List[str],
        opp_set_ids: List[str],
    ) -> torch.Tensor:
        """Score all 15 bring-4 actions given self/opp teams.

        Args:
            self_set_ids: List of 6 set_ids for self team
            opp_set_ids: List of 6 set_ids for opponent team

        Returns:
            Logits over 15 actions as a 1D tensor of shape (15,) on CPU
        """
        self_team = self._encode_team(self_set_ids)
        opp_team = self._encode_team(opp_set_ids)

        with torch.no_grad():
            logits = self.model(self_team, opp_team)  # shape (1, 15)

        return logits.squeeze(0).detach().cpu()

    def choose_action_argmax(
        self,
        self_set_ids: List[str],
        opp_set_ids: List[str],
    ) -> Tuple[int, Tuple[int, int, int, int]]:
        """Choose the best bring-4 action using argmax.

        Args:
            self_set_ids: List of 6 set_ids for self team
            opp_set_ids: List of 6 set_ids for opponent team

        Returns:
            Tuple of (action_index, bring4_slots) where bring4_slots is a tuple of 0-based indices
        """
        logits = self.score_actions(self_set_ids, opp_set_ids)
        action_index = int(torch.argmax(logits).item())
        combo = BRING4_COMBOS[action_index]  # tuple of 4 integers (0-based)
        return action_index, combo

    def action_to_showdown_team(
        self,
        action_index: int,
    ) -> str:
        """Convert an action_index into a Showdown 'team XXXX' command string.

        Args:
            action_index: Integer in [0, 14]

        Returns:
            String like "team 1234" where digits are 1-based slot numbers
        """
        if action_index < 0 or action_index >= len(BRING4_COMBOS):
            raise ValueError(f"Invalid action_index: {action_index}. Must be in [0, {len(BRING4_COMBOS)-1}]")

        combo = BRING4_COMBOS[action_index]
        # Convert 0-based indices to 1-based slot numbers
        slot_nums = sorted([i + 1 for i in combo])
        # Build string: "team " + "".join(str(n) for n in slot_nums)
        return "team " + "".join(str(n) for n in slot_nums)

    def choose_showdown_command(
        self,
        self_set_ids: List[str],
        opp_set_ids: List[str],
    ) -> str:
        """Choose a bring-4 action and return as Showdown command string.

        Args:
            self_set_ids: List of 6 set_ids for self team
            opp_set_ids: List of 6 set_ids for opponent team

        Returns:
            String like "team 1234" ready to send to Showdown
        """
        action_index, _ = self.choose_action_argmax(self_set_ids, opp_set_ids)
        return self.action_to_showdown_team(action_index)


if __name__ == "__main__":
    # Simple smoke test
    all_sets = load_sets()
    format_id = "gen9vgc2026regf"
    format_set_ids = [sid for sid, s in all_sets.items() if s.format == format_id]

    if len(format_set_ids) < 12:
        print("Not enough sets for a smoke test.")
    else:
        rng = random.Random(123)
        self_team = rng.sample(format_set_ids, 6)
        opp_team = rng.sample(format_set_ids, 6)

        policy = PreviewPolicy(format_id=format_id)
        action_index, combo = policy.choose_action_argmax(self_team, opp_team)
        cmd = policy.action_to_showdown_team(action_index)

        print("Self team:", self_team)
        print("Opp team:", opp_team)
        print("Chosen action_index:", action_index, "combo:", combo, "showdown cmd:", cmd)

