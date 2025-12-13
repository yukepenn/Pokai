"""PyTorch Dataset for team-preview / bring-4 decision.

This module builds a dataset that joins TeamBuildEpisode and BattleTrajectory
to produce samples for predicting which 4 mons to bring given 6v6 teams.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Collection, Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset

from vgc_lab.core import Paths, BattleTrajectory, get_paths
from vgc_lab.datasets import TeamBuildEpisode, iter_team_build_episodes, iter_trajectories
from vgc_lab.catalog import load_sets

# Canonical list of all 6-choose-4 combinations (15 total)
BRING4_COMBOS: List[Tuple[int, int, int, int]] = sorted(
    itertools.combinations(range(6), 4)
)

# Map from combo tuple to index
BRING4_COMBO_TO_INDEX: Dict[Tuple[int, int, int, int], int] = {
    combo: idx for idx, combo in enumerate(BRING4_COMBOS)
}


@dataclass
class PreviewExample:
    """A single team-preview / bring-4 example."""

    battle_id: str
    side: str  # "p1" or "p2"
    self_set_ids: List[str]  # len == 6
    opp_set_ids: List[str]  # len == 6
    bring4_slots: Tuple[int, int, int, int]  # 0-based indices into self_set_ids
    action_index: int  # 0..14, index into BRING4_COMBOS
    reward: float


def parse_team_preview_choice(choice: str) -> Optional[Tuple[Tuple[int, int, int, int], int]]:
    """Parse a team-preview choice string into bring4_slots tuple and action_index.

    Supported formats:
    - "default": Brings first 4 Pokemon (slots 0,1,2,3)
    - "team 1234": Explicit team selection where digits represent slots 1..6

    Args:
        choice: Choice string from BattleStep.choice

    Returns:
        Tuple of (bring4_slots, action_index) or None if parsing fails
    """
    choice = choice.strip()

    # Handle "default" - means bring first 4 Pokemon (slots 0,1,2,3)
    if choice == "default":
        slots_tuple = (0, 1, 2, 3)
        action_index = BRING4_COMBO_TO_INDEX.get(slots_tuple)
        if action_index is not None:
            return slots_tuple, action_index
        return None

    # Handle "team XXXX" format
    if not choice.startswith("team"):
        return None

    # Extract substring after "team "
    parts = choice.split(" ", 1)
    if len(parts) < 2:
        return None

    digit_str = parts[1].strip()

    # Extract only digits '1'..'6' and convert to 0-based indices
    slots = []
    for char in digit_str:
        if char in "123456":
            slot = int(char) - 1  # Convert to 0-based
            if slot not in slots:  # Avoid duplicates
                slots.append(slot)

    if len(slots) != 4:
        return None

    # Sort and convert to tuple
    slots_tuple = tuple(sorted(slots))

    # Map to action_index
    action_index = BRING4_COMBO_TO_INDEX.get(slots_tuple)
    if action_index is None:
        return None

    return slots_tuple, action_index


class PreviewDataset(Dataset[Dict[str, torch.Tensor]]):
    """PyTorch Dataset for team-preview / bring-4 prediction.

    Joins TeamBuildEpisode and BattleTrajectory to produce samples where:
    - self_team: 6 set_ids for the current side
    - opp_team: 6 set_ids for the opponent side
    - action: which 4 slots were chosen (15-way classification)
    - reward: final episodic reward from that side's perspective
    """

    def __init__(
        self,
        format_id: str = "gen9vgc2026regf",
        allowed_policy_ids: Optional[Collection[str]] = None,
        min_created_at: Optional[datetime] = None,
        include_default_choices: bool = False,
        paths: Optional[Paths] = None,
    ) -> None:
        """Initialize the preview dataset.

        Args:
            format_id: Format ID to filter episodes (e.g., "gen9vgc2026regf")
            allowed_policy_ids: Optional collection of policy_ids to include (None = all)
            min_created_at: Optional minimum creation date to include episodes
            include_default_choices: If False, skip "default" team-preview choices (default: False)
            paths: Optional Paths instance (defaults to get_paths())
        """
        paths = paths or get_paths()

        # Load sets and build index
        all_sets = load_sets()
        format_sets = {sid: s for sid, s in all_sets.items() if s.format == format_id}
        if not format_sets:
            raise ValueError(f"No sets found for format_id={format_id!r}")

        self.set_ids_sorted = sorted(format_sets.keys())
        self.set_id_to_index = {sid: i for i, sid in enumerate(self.set_ids_sorted)}
        self.num_sets = len(self.set_ids_sorted)

        # Load TeamBuildEpisode entries and build index
        ep_index: Dict[Tuple[str, str], TeamBuildEpisode] = {}
        for ep in iter_team_build_episodes(paths):
            if ep.format_id != format_id:
                continue
            if allowed_policy_ids is not None and ep.policy_id not in allowed_policy_ids:
                continue
            if min_created_at is not None and ep.created_at < min_created_at:
                continue

            if not ep.battle_ids:
                continue
            battle_id = ep.battle_ids[0]
            ep_index[(battle_id, ep.side)] = ep

        # Iterate over BattleTrajectory objects and build examples
        self._examples: List[PreviewExample] = []
        action_counts: Dict[int, int] = {}
        self._num_default_skipped = 0
        self._num_parse_failures = 0

        for traj in iter_trajectories(paths):
            if traj.format_id != format_id:
                continue

            for side in ("p1", "p2"):
                # Look up episodes for both sides
                ep_self = ep_index.get((traj.battle_id, side))
                opp_side = "p1" if side == "p2" else "p2"
                ep_opp = ep_index.get((traj.battle_id, opp_side))

                if ep_self is None or ep_opp is None:
                    continue

                # Get reward
                reward = traj.reward_p1 if side == "p1" else traj.reward_p2

                # Find team-preview step
                steps = traj.steps_p1 if side == "p1" else traj.steps_p2
                preview_step = None
                for step in steps:
                    if step.request_type == "team-preview":
                        preview_step = step
                        break

                if preview_step is None:
                    continue

                # Filter out "default" choices if not included
                if not include_default_choices and preview_step.choice.strip() == "default":
                    self._num_default_skipped += 1
                    continue

                # Parse bring-4 choice
                parsed = parse_team_preview_choice(preview_step.choice)
                if parsed is None:
                    self._num_parse_failures += 1
                    continue

                bring4_slots, action_index = parsed

                # Validate set_ids length
                if len(ep_self.chosen_set_ids) != 6 or len(ep_opp.chosen_set_ids) != 6:
                    continue

                # Build example
                example = PreviewExample(
                    battle_id=traj.battle_id,
                    side=side,
                    self_set_ids=ep_self.chosen_set_ids,
                    opp_set_ids=ep_opp.chosen_set_ids,
                    bring4_slots=bring4_slots,
                    action_index=action_index,
                    reward=reward,
                )
                self._examples.append(example)

                # Count action
                action_counts[action_index] = action_counts.get(action_index, 0) + 1

        self._action_counts = action_counts

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self._examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single example.

        Args:
            idx: Index into the dataset

        Returns:
            Dict with keys: self_team, opp_team, action, reward
            All values are torch.Tensor
        """
        ex = self._examples[idx]

        # Convert set_ids to indices
        self_idx = torch.tensor(
            [self.set_id_to_index[sid] for sid in ex.self_set_ids], dtype=torch.long
        )
        opp_idx = torch.tensor(
            [self.set_id_to_index[sid] for sid in ex.opp_set_ids], dtype=torch.long
        )

        return {
            "self_team": self_idx,  # shape (6,)
            "opp_team": opp_idx,  # shape (6,)
            "action": torch.tensor(ex.action_index, dtype=torch.long),
            "reward": torch.tensor(ex.reward, dtype=torch.float32),
        }

    @property
    def num_examples(self) -> int:
        """Return the number of examples."""
        return len(self._examples)

    @property
    def action_counts(self) -> Dict[int, int]:
        """Return a copy of the action label histogram."""
        return self._action_counts.copy()

    @property
    def num_default_skipped(self) -> int:
        """Number of 'default' team-preview choices skipped (when include_default_choices=False)."""
        return self._num_default_skipped

    @property
    def num_parse_failures(self) -> int:
        """Number of team-preview choices that failed to parse into a valid bring-4 action."""
        return self._num_parse_failures


if __name__ == "__main__":
    # Quick test
    ds = PreviewDataset()
    print(f"Preview examples: {ds.num_examples}")
    print(f"Num sets: {ds.num_sets}")
    print(f"Action counts: {ds.action_counts}")
    print(f"Default choices skipped: {ds.num_default_skipped}")
    print(f"Parse failures: {ds.num_parse_failures}")

