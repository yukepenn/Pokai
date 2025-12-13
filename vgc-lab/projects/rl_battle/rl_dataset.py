"""Offline RL transition dataset for battle trajectories.

This module provides utilities to convert BattleTrajectory objects into
RL-style transitions (state, action, reward, next_state, done) suitable
for offline RL algorithms like DQN, PPO, etc.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
from torch.utils.data import Dataset

from vgc_lab.core import BattleTrajectory, DEFAULT_FORMAT, Paths, get_paths
from vgc_lab.datasets import iter_trajectories
from vgc_lab.features import encode_trajectory_both_sides

from .dataset import encode_battle_state_to_vec, parse_move_choice


@dataclass
class BattleTransition:
    """A single RL-style transition from a battle trajectory."""

    state: np.ndarray
    action_index: int
    reward: float
    next_state: np.ndarray
    done: bool
    battle_id: str
    step_index: int
    side: str  # "p1" or "p2"


@dataclass
class RlBattleDatasetConfig:
    """Configuration for BattleTransitionDataset."""

    format_id: str = DEFAULT_FORMAT
    vec_dim: int = 256
    max_trajectories: Optional[int] = None
    min_turns: Optional[int] = None
    allowed_policy_ids: Optional[List[str]] = None
    forbidden_policy_ids: Optional[List[str]] = None


def trajectory_to_transitions(
    traj: BattleTrajectory,
    *,
    vec_dim: int = 256,
) -> List[BattleTransition]:
    """
    Convert a single BattleTrajectory into a list of RL-style transitions
    (state, action, reward, next_state, done) for both sides.

    Args:
        traj: BattleTrajectory to convert
        vec_dim: Dimension of state vectors

    Returns:
        List of BattleTransition objects
    """
    # Get encoded trajectory from both sides
    encoded = encode_trajectory_both_sides(traj)

    if not encoded:
        return []

    # Determine side boundaries: first len(steps_p1) are p1, rest are p2
    p1_count = len(traj.steps_p1)
    p2_count = len(traj.steps_p2)

    transitions: List[BattleTransition] = []

    # Pre-encode all states for efficient next_state lookup
    encoded_states: List[np.ndarray] = []
    for state_dict, _, _, _ in encoded:
        state_vec = encode_battle_state_to_vec(state_dict, vec_dim=vec_dim)
        encoded_states.append(state_vec)

    for i, (state_dict, choice_str, reward, done) in enumerate(encoded):
        # Encode current state
        state = encoded_states[i]

        # Parse action index
        action_index = parse_move_choice(choice_str)
        if action_index is None:
            # Skip transitions with unparseable actions
            continue

        # Determine next_state
        if done:
            # Terminal state: use state itself as next_state
            next_state = state.copy()
        elif i + 1 < len(encoded_states):
            # Non-terminal: use next step's state
            next_state = encoded_states[i + 1]
        else:
            # Edge case: should not happen if done flag is correct
            next_state = state.copy()

        # Determine side based on index
        if i < p1_count:
            side = "p1"
            step_index = i
        else:
            side = "p2"
            step_index = i - p1_count

        transition = BattleTransition(
            state=state,
            action_index=action_index,
            reward=reward,
            next_state=next_state,
            done=done,
            battle_id=traj.battle_id,
            step_index=step_index,
            side=side,
        )
        transitions.append(transition)

    return transitions


class BattleTransitionDataset(Dataset[BattleTransition]):
    """PyTorch Dataset for offline RL transitions from battle trajectories."""

    def __init__(
        self,
        cfg: RlBattleDatasetConfig,
        paths: Optional[Paths] = None,
    ) -> None:
        """Initialize dataset from trajectories.

        Args:
            cfg: Configuration for dataset loading
            paths: Paths instance (defaults to get_paths())
        """
        if paths is None:
            paths = get_paths()

        self.cfg = cfg
        self._transitions: List[BattleTransition] = []

        num_trajectories_processed = 0

        for traj in iter_trajectories(paths):
            # Filter by format_id
            if traj.format_id != cfg.format_id:
                continue

            # Filter by min_turns
            if cfg.min_turns is not None:
                if traj.turns is None or traj.turns < cfg.min_turns:
                    continue

            # Filter by policy IDs if specified
            if cfg.allowed_policy_ids is not None or cfg.forbidden_policy_ids is not None:
                # Try to extract policy IDs from meta
                meta = traj.meta or {}
                policy_ids = []

                # Check for side-specific policy IDs
                if "battle_policy_id_p1" in meta:
                    policy_ids.append(meta["battle_policy_id_p1"])
                if "battle_policy_id_p2" in meta:
                    policy_ids.append(meta["battle_policy_id_p2"])
                if "battle_policy_id" in meta:
                    policy_ids.append(meta["battle_policy_id"])

                # Apply filters
                if cfg.allowed_policy_ids is not None:
                    if not any(pid in cfg.allowed_policy_ids for pid in policy_ids):
                        continue

                if cfg.forbidden_policy_ids is not None:
                    if any(pid in cfg.forbidden_policy_ids for pid in policy_ids):
                        continue

            # Convert trajectory to transitions
            traj_transitions = trajectory_to_transitions(traj, vec_dim=cfg.vec_dim)
            self._transitions.extend(traj_transitions)

            num_trajectories_processed += 1

            # Check max_trajectories limit
            if cfg.max_trajectories is not None and num_trajectories_processed >= cfg.max_trajectories:
                break

    def __len__(self) -> int:
        """Return the number of transitions in the dataset."""
        return len(self._transitions)

    def __getitem__(self, idx: int) -> BattleTransition:
        """Get a single transition by index."""
        return self._transitions[idx]
