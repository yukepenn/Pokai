"""Tests for battle DQN policy."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from projects.rl_battle.policy import BattleDqnPolicy, BattleDqnPolicyConfig, load_battle_dqn_checkpoint
from projects.rl_battle.train_dqn import BattleQNetwork


def test_battle_dqn_policy_forward_shape(tmp_path: Path):
    """Test that BattleDqnPolicy produces correct output shapes."""
    vec_dim = 8
    num_actions = 3

    # Create a tiny synthetic checkpoint
    # Use default hidden_dim=256 to match what load_battle_dqn_checkpoint will create
    model = BattleQNetwork(input_dim=vec_dim, num_actions=num_actions)
    ckpt_path = tmp_path / "test_dqn_policy.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": vec_dim,
            "num_actions": num_actions,
            "format_id": "gen9vgc2026regf",
            "hyperparams": {},
        },
        ckpt_path,
    )

    # Build policy config
    cfg = BattleDqnPolicyConfig(
        format_id="gen9vgc2026regf",
        vec_dim=vec_dim,
        num_actions=num_actions,
        ckpt_path=ckpt_path,
        device="cpu",
    )

    # Instantiate policy
    policy = BattleDqnPolicy(cfg)

    # Create a minimal fake request dict
    # encode_state_from_request accepts any dict and wraps it
    request = {"example": "dummy", "active": [{"moves": []}]}

    # Test score_actions
    q_values = policy.score_actions(request)
    assert q_values.shape == (num_actions,), f"Expected shape ({num_actions},), got {q_values.shape}"
    assert isinstance(q_values, torch.Tensor)

    # Test choose_showdown_command
    choice = policy.choose_showdown_command(request)
    assert isinstance(choice, str)
    assert choice.startswith("move "), f"Expected choice to start with 'move ', got {choice!r}"

    # Test choose_action_argmax
    action_idx, choice_str = policy.choose_action_argmax(request)
    assert isinstance(action_idx, int)
    assert 0 <= action_idx < num_actions
    assert isinstance(choice_str, str)
    assert choice_str.startswith("move ")
    assert choice_str == choice


def test_battle_dqn_policy_checkpoint_mismatch_input_dim(tmp_path: Path):
    """Test that mismatched input_dim raises ValueError."""
    vec_dim_checkpoint = 16
    vec_dim_config = 8
    num_actions = 4

    # Create checkpoint with input_dim=16
    model = BattleQNetwork(input_dim=vec_dim_checkpoint, num_actions=num_actions)
    ckpt_path = tmp_path / "test_dqn_policy_mismatch.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": vec_dim_checkpoint,
            "num_actions": num_actions,
            "format_id": "gen9vgc2026regf",
            "hyperparams": {},
        },
        ckpt_path,
    )

    # Build config with vec_dim=8 (mismatch)
    cfg = BattleDqnPolicyConfig(
        format_id="gen9vgc2026regf",
        vec_dim=vec_dim_config,  # Mismatch!
        num_actions=num_actions,
        ckpt_path=ckpt_path,
        device="cpu",
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="input_dim.*does not match"):
        load_battle_dqn_checkpoint(cfg)


def test_battle_dqn_policy_checkpoint_mismatch_num_actions(tmp_path: Path):
    """Test that mismatched num_actions raises ValueError."""
    vec_dim = 8
    num_actions_checkpoint = 4
    num_actions_config = 3

    # Create checkpoint with num_actions=4
    model = BattleQNetwork(input_dim=vec_dim, num_actions=num_actions_checkpoint)
    ckpt_path = tmp_path / "test_dqn_policy_mismatch_actions.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": vec_dim,
            "num_actions": num_actions_checkpoint,
            "format_id": "gen9vgc2026regf",
            "hyperparams": {},
        },
        ckpt_path,
    )

    # Build config with num_actions=3 (mismatch)
    cfg = BattleDqnPolicyConfig(
        format_id="gen9vgc2026regf",
        vec_dim=vec_dim,
        num_actions=num_actions_config,  # Mismatch!
        ckpt_path=ckpt_path,
        device="cpu",
    )

    # Should raise ValueError
    with pytest.raises(ValueError, match="num_actions.*does not match"):
        load_battle_dqn_checkpoint(cfg)


def test_battle_dqn_policy_missing_checkpoint():
    """Test that missing checkpoint raises FileNotFoundError."""
    cfg = BattleDqnPolicyConfig(
        format_id="gen9vgc2026regf",
        vec_dim=8,
        num_actions=3,
        ckpt_path=Path("/nonexistent/path/battle_dqn.pt"),
        device="cpu",
    )

    with pytest.raises(FileNotFoundError, match="not found"):
        load_battle_dqn_checkpoint(cfg)
