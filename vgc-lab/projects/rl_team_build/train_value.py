"""Training script for TeamValueModel.

This module trains a simple MLP to predict team values from team-building episodes,
saving checkpoints for use by the policy module.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Collection, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader

from vgc_lab.core import PROJECT_ROOT
from .dataset import TeamBuildDataset


class TeamValueModel(nn.Module):
    """
    Simple MLP that maps a 6-mon team (as set indices) to a scalar value.

    This is intentionally small and generic:
      - Embedding layer for set indices
      - Mean pooling over the 6 mons
      - 2-layer MLP to a single scalar
    """

    def __init__(
        self,
        num_sets: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        """
        Initialize the model.

        Args:
            num_sets: Number of unique sets in the catalog (embedding vocabulary size)
            embed_dim: Embedding dimension for each set
            hidden_dim: Hidden dimension for the MLP
        """
        super().__init__()
        self.embedding = nn.Embedding(num_sets, embed_dim)
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.relu = nn.ReLU()

    def forward(self, team_indices: torch.LongTensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            team_indices: LongTensor of shape (batch_size, 6)

        Returns:
            Tensor of shape (batch_size,) with predicted values
        """
        # Embed: (batch, 6) -> (batch, 6, embed_dim)
        emb = self.embedding(team_indices)

        # Pool: (batch, 6, embed_dim) -> (batch, embed_dim)
        team_repr = emb.mean(dim=1)

        # MLP: (batch, embed_dim) -> (batch, 1)
        h = self.relu(self.fc1(team_repr))
        out = self.fc2(h)

        # Squeeze to (batch,)
        return out.squeeze(-1)


def train_value_model(
    batch_size: int = 64,
    lr: float = 1e-3,
    epochs: int = 5,
    format_id: str = "gen9vgc2026regf",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    allowed_policy_ids: Optional[Collection[str]] = None,
    min_created_at: Optional[datetime] = None,
) -> Path:
    """Train a TeamValueModel on TeamBuildEpisode data and save a checkpoint.

    Args:
        batch_size: Batch size for training
        lr: Learning rate
        epochs: Number of training epochs
        format_id: Format ID to filter episodes
        device: Device to run training on ("cuda" or "cpu")
        allowed_policy_ids: Optional list of policy_ids to include (None = all)
        min_created_at: Optional minimum creation date to include episodes

    Returns:
        Path to the saved checkpoint file
    """
    print(f"Initializing dataset (format={format_id})...")
    dataset = TeamBuildDataset(
        format_id=format_id,
        allowed_policy_ids=allowed_policy_ids,
        min_created_at=min_created_at,
    )

    if len(dataset) == 0:
        print("WARNING: No episodes found in dataset. Skipping training.")
        return PROJECT_ROOT / "checkpoints" / "team_value.pt"

    print(f"Loaded {len(dataset)} episodes, {dataset.num_sets} unique sets")

    # Build DataLoader
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Create model (use default hyperparams, save them in checkpoint)
    embed_dim = 64
    hidden_dim = 128
    model = TeamValueModel(
        num_sets=dataset.num_sets, embed_dim=embed_dim, hidden_dim=hidden_dim
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    print(f"Training on device: {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Training loop
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        for batch_indices, batch_rewards in dataloader:
            # Move to device
            batch_indices = batch_indices.to(device)
            batch_rewards = batch_rewards.to(device).squeeze(-1)

            # Forward pass
            preds = model(batch_indices)

            # Compute loss
            loss = criterion(preds, batch_rewards)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            num_batches += 1

        avg_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        print(f"Epoch {epoch + 1}/{epochs}: loss = {avg_loss:.6f}")

    # Save checkpoint with rich metadata
    ckpt_dir = PROJECT_ROOT / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / "team_value.pt"

    torch.save(
        {
            "model_state": model.state_dict(),
            "num_sets": dataset.num_sets,
            "set_ids": dataset.idx_to_set_id,  # Sorted list matching build_set_id_index ordering
            "format_id": format_id,
            "hyperparams": {
                "embed_dim": embed_dim,
                "hidden_dim": hidden_dim,
                "lr": lr,
                "epochs": epochs,
                "batch_size": batch_size,
            },
        },
        ckpt_path,
    )

    print(f"Checkpoint saved to: {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    train_value_model()

