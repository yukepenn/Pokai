"""Tests for RL battle transition dataset."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vgc_lab.core import (
    BattleStep,
    BattleTrajectory,
    DEFAULT_FORMAT,
    Paths,
    get_paths,
)
from projects.rl_battle.dataset import encode_battle_state_to_vec, parse_move_choice
from projects.rl_battle.rl_dataset import (
    BattleTransition,
    BattleTransitionDataset,
    RlBattleDatasetConfig,
    trajectory_to_transitions,
)


def test_trajectory_to_transitions_unit():
    """Unit test for trajectory_to_transitions function."""
    # Create a minimal BattleTrajectory
    traj = BattleTrajectory(
        battle_id="test_battle_1",
        format_id=DEFAULT_FORMAT,
        p1_name="Bot1",
        p2_name="Bot2",
        winner_side="p1",
        winner_name="Bot1",
        turns=3,
        raw_log_path=Path("dummy.log"),
        reward_p1=1.0,
        reward_p2=-1.0,
        steps_p1=[
            BattleStep(
                side="p1",
                step_index=0,
                request_type="move",
                request={"dummy": 0},
                choice="move 1",
            ),
            BattleStep(
                side="p1",
                step_index=1,
                request_type="move",
                request={"dummy": 1},
                choice="move 2",
            ),
        ],
        steps_p2=[
            BattleStep(
                side="p2",
                step_index=0,
                request_type="move",
                request={"dummy": 2},
                choice="move 1",
            ),
        ],
    )

    # Mock encode_trajectory_both_sides to return expected format
    mock_encoded = [
        ({"dummy": 0}, "move 1", 1.0, False),  # p1 step 0
        ({"dummy": 1}, "move 2", 1.0, False),  # p1 step 1
        ({"dummy": 2}, "move 1", -1.0, True),  # p2 step 0 (terminal)
    ]

    with patch("projects.rl_battle.rl_dataset.encode_trajectory_both_sides") as mock_fn:
        mock_fn.return_value = mock_encoded

        transitions = trajectory_to_transitions(traj, vec_dim=16)

    # Assertions
    assert len(transitions) == 3, f"Expected 3 transitions, got {len(transitions)}"

    # Check first transition (p1, non-terminal)
    t0 = transitions[0]
    assert t0.side == "p1"
    assert t0.step_index == 0
    assert t0.done is False
    assert t0.reward == 1.0
    assert t0.battle_id == "test_battle_1"
    assert isinstance(t0.state, np.ndarray)
    assert t0.state.shape == (16,)
    assert isinstance(t0.next_state, np.ndarray)
    assert t0.next_state.shape == (16,)
    # next_state should be different from state (not terminal)
    assert not np.array_equal(t0.state, t0.next_state)

    # Check action_index
    expected_action_0 = parse_move_choice("move 1")
    assert t0.action_index == expected_action_0, f"Expected {expected_action_0}, got {t0.action_index}"

    # Check second transition (p1, non-terminal)
    t1 = transitions[1]
    assert t1.side == "p1"
    assert t1.step_index == 1
    assert t1.done is False
    assert t1.reward == 1.0
    expected_action_1 = parse_move_choice("move 2")
    assert t1.action_index == expected_action_1

    # Check third transition (p2, terminal)
    t2 = transitions[2]
    assert t2.side == "p2"
    assert t2.step_index == 0  # p2 step_index resets to 0
    assert t2.done is True
    assert t2.reward == -1.0
    expected_action_2 = parse_move_choice("move 1")
    assert t2.action_index == expected_action_2
    # Terminal: next_state should be a copy of state
    assert np.array_equal(t2.state, t2.next_state)


def test_battle_transition_dataset_integration(tmp_path: Path):
    """Integration test for BattleTransitionDataset."""
    # Create temporary Paths
    data_root = tmp_path / "data"
    paths = Paths(
        data_root=data_root,
        battles_raw=data_root / "battles_raw",
        battles_json=data_root / "battles_json",
        datasets_root=data_root / "datasets",
        full_battles=data_root / "datasets" / "full_battles",
        team_preview=data_root / "datasets" / "team_preview",
        trajectories=data_root / "datasets" / "trajectories",
        team_build=data_root / "datasets" / "team_build",
    )

    # Create trajectories directory
    paths.trajectories.mkdir(parents=True, exist_ok=True)

    # Create a minimal BattleTrajectory
    traj = BattleTrajectory(
        battle_id="test_integration_1",
        format_id=DEFAULT_FORMAT,
        p1_name="Bot1",
        p2_name="Bot2",
        winner_side="p1",
        winner_name="Bot1",
        turns=2,
        raw_log_path=Path("dummy.log"),
        reward_p1=1.0,
        reward_p2=-1.0,
        steps_p1=[
            BattleStep(
                side="p1",
                step_index=0,
                request_type="move",
                request={"test": "request"},
                choice="move 1",
            ),
        ],
        steps_p2=[
            BattleStep(
                side="p2",
                step_index=0,
                request_type="move",
                request={"test": "request"},
                choice="move 2",
            ),
        ],
    )

    # Write trajectory to JSONL file
    traj_json = traj.model_dump(mode="json")
    with paths.trajectories_jsonl.open("w", encoding="utf-8") as f:
        f.write(json.dumps(traj_json, ensure_ascii=False) + "\n")

    # Create dataset
    cfg = RlBattleDatasetConfig(format_id=DEFAULT_FORMAT, vec_dim=64)
    dataset = BattleTransitionDataset(cfg, paths=paths)

    # Assertions
    assert len(dataset) > 0, "Dataset should contain at least one transition"

    # Get first sample
    sample = dataset[0]
    assert isinstance(sample, BattleTransition)
    assert isinstance(sample.state, np.ndarray)
    assert sample.state.shape[0] == cfg.vec_dim
    assert 0 <= sample.action_index < 256, f"action_index should be reasonable, got {sample.action_index}"
    assert isinstance(sample.reward, float)
    assert isinstance(sample.done, bool)
    assert sample.battle_id == "test_integration_1"
    assert sample.side in ("p1", "p2")


def test_battle_transition_dataset_filters(tmp_path: Path):
    """Test that BattleTransitionDataset applies filters correctly."""
    data_root = tmp_path / "data"
    paths = Paths(
        data_root=data_root,
        battles_raw=data_root / "battles_raw",
        battles_json=data_root / "battles_json",
        datasets_root=data_root / "datasets",
        full_battles=data_root / "datasets" / "full_battles",
        team_preview=data_root / "datasets" / "team_preview",
        trajectories=data_root / "datasets" / "trajectories",
        team_build=data_root / "datasets" / "team_build",
    )
    paths.trajectories.mkdir(parents=True, exist_ok=True)

    # Create two trajectories with different turns
    traj1 = BattleTrajectory(
        battle_id="short_battle",
        format_id=DEFAULT_FORMAT,
        p1_name="Bot1",
        p2_name="Bot2",
        winner_side="p1",
        winner_name="Bot1",
        turns=1,  # Short battle
        raw_log_path=Path("dummy1.log"),
        reward_p1=1.0,
        reward_p2=-1.0,
        steps_p1=[
            BattleStep(
                side="p1",
                step_index=0,
                request_type="move",
                request={},
                choice="move 1",
            ),
        ],
        steps_p2=[],
    )

    traj2 = BattleTrajectory(
        battle_id="long_battle",
        format_id=DEFAULT_FORMAT,
        p1_name="Bot1",
        p2_name="Bot2",
        winner_side="p1",
        winner_name="Bot1",
        turns=10,  # Long battle
        raw_log_path=Path("dummy2.log"),
        reward_p1=1.0,
        reward_p2=-1.0,
        steps_p1=[
            BattleStep(
                side="p1",
                step_index=0,
                request_type="move",
                request={},
                choice="move 1",
            ),
        ],
        steps_p2=[],
    )

    # Write both trajectories
    with paths.trajectories_jsonl.open("w", encoding="utf-8") as f:
        f.write(json.dumps(traj1.model_dump(mode="json"), ensure_ascii=False) + "\n")
        f.write(json.dumps(traj2.model_dump(mode="json"), ensure_ascii=False) + "\n")

    # Test min_turns filter
    cfg = RlBattleDatasetConfig(format_id=DEFAULT_FORMAT, min_turns=5, vec_dim=64)
    dataset = BattleTransitionDataset(cfg, paths=paths)

    # Should only contain transitions from long_battle
    battle_ids = {t.battle_id for t in dataset._transitions}
    assert "short_battle" not in battle_ids
    assert "long_battle" in battle_ids
