"""Tests for battle DQN training."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from projects.rl_battle.rl_dataset import BattleTransition
from projects.rl_battle.train_dqn import BattleDqnConfig, BattleQNetwork, train_battle_dqn


def test_battle_q_network_forward_shape():
    """Test that BattleQNetwork produces correct output shapes."""
    input_dim = 16
    num_actions = 3
    batch_size = 5

    network = BattleQNetwork(input_dim=input_dim, num_actions=num_actions, hidden_dim=32)

    # Create random input tensor
    x = torch.randn(batch_size, input_dim)

    # Forward pass
    output = network(x)

    # Assertions
    assert output.shape == (batch_size, num_actions), f"Expected shape ({batch_size}, {num_actions}), got {output.shape}"
    assert isinstance(output, torch.Tensor)


def test_train_battle_dqn_completes_and_saves_checkpoint(tmp_path: Path):
    """Test that a tiny training run completes and saves a checkpoint."""
    # Create a fake dataset with minimal transitions
    vec_dim = 16
    num_actions = 3
    num_transitions = 32

    # Generate fake transitions
    fake_transitions = []
    for i in range(num_transitions):
        transition = BattleTransition(
            state=np.random.randn(vec_dim).astype(np.float32),
            action_index=i % num_actions,
            reward=1.0 if i % 3 == 0 else 0.0,
            next_state=np.random.randn(vec_dim).astype(np.float32),
            done=(i >= num_transitions - 3),  # Last 3 are terminal
            battle_id=f"test_battle_{i // 10}",
            step_index=i % 10,
            side="p1" if i % 2 == 0 else "p2",
        )
        fake_transitions.append(transition)

    # Create a mock dataset class
    class FakeDataset:
        def __init__(self, transitions):
            self._transitions = transitions

        def __len__(self):
            return len(self._transitions)

        def __getitem__(self, idx):
            return self._transitions[idx]

    fake_dataset = FakeDataset(fake_transitions)

    # Create config with minimal training settings
    ckpt_path = tmp_path / "battle_dqn_test.pt"
    cfg = BattleDqnConfig(
        format_id="gen9vgc2026regf",
        vec_dim=vec_dim,
        num_actions=num_actions,
        epochs=1,
        steps_per_epoch=5,
        batch_size=8,
        device="cpu",
        ckpt_path=ckpt_path,
    )

    # Patch BattleTransitionDataset to return our fake dataset
    with patch("projects.rl_battle.train_dqn.BattleTransitionDataset") as mock_dataset_class:
        mock_dataset_class.return_value = fake_dataset

        # Run training
        returned_path = train_battle_dqn(cfg)

    # Assertions
    assert returned_path == ckpt_path
    assert ckpt_path.exists(), f"Checkpoint file should exist at {ckpt_path}"

    # Load checkpoint and verify contents
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    assert "model_state" in checkpoint
    assert "num_actions" in checkpoint
    assert checkpoint["num_actions"] == num_actions
    assert checkpoint["input_dim"] == vec_dim
    assert "hyperparams" in checkpoint

    # Verify model can be loaded
    model = BattleQNetwork(input_dim=vec_dim, num_actions=num_actions)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # Test forward pass with loaded model
    test_input = torch.randn(1, vec_dim)
    with torch.no_grad():
        output = model(test_input)
    assert output.shape == (1, num_actions)


def test_train_battle_dqn_empty_dataset_raises_error():
    """Test that training raises ValueError when dataset is empty."""
    cfg = BattleDqnConfig(
        format_id="gen9vgc2026regf",
        vec_dim=16,
        num_actions=3,
        epochs=1,
        steps_per_epoch=1,
        batch_size=8,
    )

    # Create a mock empty dataset
    class EmptyDataset:
        def __len__(self):
            return 0

        def __getitem__(self, idx):
            raise IndexError

    with patch("projects.rl_battle.train_dqn.BattleTransitionDataset") as mock_dataset_class:
        mock_dataset_class.return_value = EmptyDataset()

        with pytest.raises(ValueError, match="empty"):
            train_battle_dqn(cfg)
