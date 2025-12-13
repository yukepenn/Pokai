"""Training script for offline DQN on battle trajectories."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from vgc_lab.core import DEFAULT_FORMAT, PROJECT_ROOT
from .rl_dataset import BattleTransition, BattleTransitionDataset, RlBattleDatasetConfig


def _collate_transitions(batch: list[BattleTransition]) -> list[BattleTransition]:
    """Collate function for DataLoader to handle BattleTransition objects."""
    # For now, just return the list as-is; we'll unpack manually in training loop
    return batch


class BattleQNetwork(nn.Module):
    """
    Simple MLP Q-network for battle RL.

    Input: state vector of dimension vec_dim.
    Output: Q-values for each discrete action.
    """

    def __init__(self, input_dim: int, num_actions: int, hidden_dim: int = 256) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class BattleDqnConfig:
    """Configuration for battle DQN training."""

    format_id: str = DEFAULT_FORMAT
    vec_dim: int = 256
    num_actions: int = 4

    gamma: float = 0.99
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 3
    steps_per_epoch: int = 100
    target_update_interval: int = 100

    max_trajectories: Optional[int] = None
    device: str = "cpu"
    seed: int = 42

    ckpt_path: Path = PROJECT_ROOT / "checkpoints" / "battle_dqn.pt"


def train_battle_dqn(cfg: BattleDqnConfig) -> Path:
    """
    Train a simple offline DQN on BattleTransitionDataset.

    This is a minimal, baseline implementation intended for experimentation,
    not a highly optimized RL library.

    Args:
        cfg: BattleDqnConfig with training hyperparameters.

    Returns:
        Path to the saved checkpoint.
    """
    # Set seeds for reproducibility
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Build dataset
    dataset_cfg = RlBattleDatasetConfig(
        format_id=cfg.format_id,
        vec_dim=cfg.vec_dim,
        max_trajectories=cfg.max_trajectories,
    )
    dataset = BattleTransitionDataset(dataset_cfg)

    if len(dataset) == 0:
        raise ValueError(
            "BattleTransitionDataset is empty; cannot train DQN model. "
            "Generate trajectories first using online-selfplay or gen-full commands."
        )

    # Build DataLoader with custom collate function
    loader = DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=_collate_transitions,
    )

    # Instantiate Q-network and target network
    device = torch.device(cfg.device)
    q_net = BattleQNetwork(cfg.vec_dim, cfg.num_actions).to(device)
    target_net = BattleQNetwork(cfg.vec_dim, cfg.num_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())
    target_net.eval()

    # Optimizer & loss
    optimizer = torch.optim.Adam(q_net.parameters(), lr=cfg.lr)
    mse_loss = nn.MSELoss()

    # Training loop
    total_steps = 0

    for epoch in range(cfg.epochs):
        q_net.train()
        epoch_loss = 0.0
        steps_in_epoch = 0

        loader_iter = iter(loader)

        for step_idx in range(cfg.steps_per_epoch):
            try:
                batch = next(loader_iter)
            except StopIteration:
                # Recreate iterator if exhausted
                loader_iter = iter(loader)
                batch = next(loader_iter)

            # Stack batch into tensors
            states = torch.stack([torch.from_numpy(t.state) for t in batch]).float().to(device)
            actions = torch.tensor([t.action_index for t in batch], dtype=torch.long, device=device)
            rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device)
            next_states = torch.stack([torch.from_numpy(t.next_state) for t in batch]).float().to(device)
            dones = torch.tensor([t.done for t in batch], dtype=torch.float32, device=device)

            # Q(s,a)
            q_values = q_net(states)
            q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # Q target using target_net
            with torch.no_grad():
                q_next = target_net(next_states)
                max_q_next, _ = q_next.max(dim=1)
                target = rewards + cfg.gamma * (1.0 - dones) * max_q_next

            # Compute loss and optimize
            loss = mse_loss(q_sa, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            steps_in_epoch += 1
            total_steps += 1

            # Update target network periodically
            if total_steps % cfg.target_update_interval == 0:
                target_net.load_state_dict(q_net.state_dict())

        avg_loss = epoch_loss / max(steps_in_epoch, 1)
        print(f"Epoch {epoch + 1}/{cfg.epochs}: avg_loss={avg_loss:.6f}, steps={steps_in_epoch}")

    # Save checkpoint
    ckpt_path = cfg.ckpt_path
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, Any] = {
        "model_state": q_net.state_dict(),
        "input_dim": cfg.vec_dim,
        "num_actions": cfg.num_actions,
        "format_id": cfg.format_id,
        "hyperparams": {
            "gamma": cfg.gamma,
            "batch_size": cfg.batch_size,
            "lr": cfg.lr,
            "epochs": cfg.epochs,
            "steps_per_epoch": cfg.steps_per_epoch,
            "target_update_interval": cfg.target_update_interval,
        },
    }
    torch.save(checkpoint, ckpt_path)
    print(f"DQN checkpoint saved to: {ckpt_path}")
    return ckpt_path
