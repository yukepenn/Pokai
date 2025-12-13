"""Training script for PreviewModel (bring-4 prediction).

This module trains a simple BC model that predicts which 4 mons to bring
given encoded self/opp teams (15-way classification).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Collection, Dict, List, Optional

import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split, Subset

from vgc_lab.core import PROJECT_ROOT
from .dataset import PreviewDataset, BRING4_COMBOS


class PreviewModel(nn.Module):
    """Simple BC model for predicting bring-4 action given self/opp teams.

    Architecture:
    - Embedding layer for set indices
    - Mean pooling over self_team and opp_team separately
    - Concatenate pooled embeddings
    - 2-layer MLP to 15-way classification (logits)
    """

    def __init__(self, num_sets: int, embed_dim: int = 64, hidden_dim: int = 128) -> None:
        """Initialize the model.

        Args:
            num_sets: Number of unique sets (embedding vocabulary size)
            embed_dim: Embedding dimension for each set
            hidden_dim: Hidden dimension for the MLP
        """
        super().__init__()
        self.embedding = nn.Embedding(num_sets, embed_dim)
        self.fc1 = nn.Linear(embed_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, len(BRING4_COMBOS))  # 15 = C(6,4)

    def forward(self, self_team: torch.Tensor, opp_team: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            self_team: (B, 6) int64 indices
            opp_team: (B, 6) int64 indices

        Returns:
            logits: (B, len(BRING4_COMBOS))
        """
        emb_self = self.embedding(self_team).mean(dim=1)  # (B, embed_dim)
        emb_opp = self.embedding(opp_team).mean(dim=1)  # (B, embed_dim)
        x = torch.cat([emb_self, emb_opp], dim=-1)  # (B, embed_dim * 2)
        x = torch.relu(self.fc1(x))  # (B, hidden_dim)
        logits = self.fc2(x)  # (B, len(BRING4_COMBOS))
        return logits


@dataclass
class TrainPreviewConfig:
    """Configuration for preview model training."""

    format_id: str = "gen9vgc2026regf"
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 5
    train_frac: float = 0.8
    seed: int = 42
    allowed_policy_ids: Optional[Collection[str]] = None
    min_created_at: Optional[datetime] = None
    include_default_choices: bool = False
    # Reward-aware training options
    use_reward_weights: bool = False
    min_reward_for_training: Optional[float] = None
    max_reward_for_training: Optional[float] = None
    reward_weight_alpha: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt_path: Path = PROJECT_ROOT / "checkpoints" / "preview_bring4.pt"


def train_preview_model(config: TrainPreviewConfig) -> Path:
    """Train a PreviewModel on preview/bring-4 data and save a checkpoint.

    Args:
        config: TrainPreviewConfig with training hyperparameters and dataset filters

    Returns:
        Path to the saved checkpoint file

    Raises:
        ValueError: If dataset is empty
    """
    # Set seeds for reproducibility
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Build dataset
    ds = PreviewDataset(
        format_id=config.format_id,
        allowed_policy_ids=config.allowed_policy_ids,
        min_created_at=config.min_created_at,
        include_default_choices=config.include_default_choices,
    )

    if ds.num_examples == 0:
        raise ValueError(
            "PreviewDataset is empty (num_examples=0). "
            "This usually means you only have old 'default' team-preview data. "
            "Generate new battles with explicit 'team XXXX' choices using "
            "projects.rl_team_build.selfplay before training the preview model."
        )

    num_raw_examples = ds.num_examples
    print(f"Loaded {num_raw_examples} preview examples, {ds.num_sets} unique sets")
    print(f"Action distribution: {dict(sorted(ds.action_counts.items()))}")

    # Apply reward filtering if specified
    keep_indices: List[int] = []
    sample_weights: Optional[List[float]] = None

    if config.min_reward_for_training is not None or config.max_reward_for_training is not None:
        # Filter by reward range
        for i in range(len(ds)):
            example = ds._examples[i]
            reward = example.reward

            # Check min threshold
            if config.min_reward_for_training is not None:
                if reward < config.min_reward_for_training:
                    continue

            # Check max threshold
            if config.max_reward_for_training is not None:
                if reward > config.max_reward_for_training:
                    continue

            keep_indices.append(i)

        if len(keep_indices) == 0:
            raise ValueError(
                f"Reward filtering removed all examples. "
                f"min_reward={config.min_reward_for_training}, "
                f"max_reward={config.max_reward_for_training}. "
                "Check your thresholds or dataset reward distribution."
            )

        # Apply filter by wrapping in Subset
        ds = Subset(ds, keep_indices)
        print(f"After reward filtering: {len(keep_indices)} examples kept")

    # Compute reward weights if enabled
    if config.use_reward_weights:
        # Use original dataset if we didn't filter, otherwise use filtered indices
        source_ds = ds.dataset if isinstance(ds, Subset) else ds
        source_indices = keep_indices if isinstance(ds, Subset) else list(range(len(source_ds)))

        sample_weights = []
        for idx in source_indices:
            example = source_ds._examples[idx]
            reward = example.reward
            # Compute weight: w_i = 1.0 + alpha * reward_i, clamped to [0.1, 5.0]
            weight = 1.0 + config.reward_weight_alpha * reward
            weight = max(0.1, min(5.0, weight))
            sample_weights.append(weight)

        # Attach weights to Subset if we have one, otherwise we'll handle it differently
        if isinstance(ds, Subset):
            ds.sample_weights = sample_weights  # type: ignore
        else:
            # Create a Subset wrapper just to attach weights
            ds_wrapped = Subset(ds, list(range(len(ds))))
            ds_wrapped.sample_weights = sample_weights  # type: ignore
            ds = ds_wrapped

        weight_stats = {
            "min": min(sample_weights),
            "max": max(sample_weights),
            "mean": sum(sample_weights) / len(sample_weights),
        }
        print(
            f"Reward weights enabled (alpha={config.reward_weight_alpha}): "
            f"min={weight_stats['min']:.3f}, max={weight_stats['max']:.3f}, "
            f"mean={weight_stats['mean']:.3f}"
        )

    # Compute train/val split
    num_examples_after_filter = len(ds)
    print(f"Training examples: {num_raw_examples} raw -> {num_examples_after_filter} after filtering")
    
    train_size = int(config.train_frac * num_examples_after_filter)
    val_size = num_examples_after_filter - train_size
    if val_size < 1 and num_examples_after_filter > 0:
        val_size = 1
        train_size = num_examples_after_filter - 1

    # Split with seeded generator
    generator = torch.Generator().manual_seed(config.seed)
    train_ds, val_ds = random_split(ds, [train_size, val_size], generator=generator)

    # Store weight mapping for train set if using reward weights
    train_sample_weights: Optional[List[float]] = None
    use_shuffle = True
    if config.use_reward_weights and sample_weights is not None:
        # Map train_ds indices back to sample_weights
        train_sample_weights = []
        if isinstance(train_ds, Subset):
            # train_ds.indices maps train_ds position -> ds position
            # sample_weights is indexed by ds position
            for idx in train_ds.indices:
                train_sample_weights.append(sample_weights[idx])
        else:
            train_sample_weights = sample_weights[:train_size]
        
        # Disable shuffle when using weights to maintain index alignment
        use_shuffle = False

    # Create DataLoaders
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, shuffle=use_shuffle)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size, shuffle=False)

    # Create model
    embed_dim = 64
    hidden_dim = 128
    model = PreviewModel(
        num_sets=ds.num_sets,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
    ).to(config.device)

    # Optimizer and loss
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    loss_fn = nn.CrossEntropyLoss()

    print(f"Training on device: {config.device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")

    # Training loop
    for epoch in range(config.epochs):
        # Train phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        # Use enumerate to track batch position for weight lookup
        for batch_idx, batch in enumerate(train_loader):
            self_team = batch["self_team"].to(config.device)
            opp_team = batch["opp_team"].to(config.device)
            action = batch["action"].to(config.device)

            # Forward pass
            logits = model(self_team, opp_team)

            # Compute loss (with optional reward weighting)
            if config.use_reward_weights and train_sample_weights is not None:
                # Get per-sample loss
                per_sample_loss = loss_fn(logits, action, reduction="none")  # shape: (B,)

                # Get weights for this batch
                # Note: DataLoader with shuffle=True randomizes order, but we can still map via indices
                # Since we can't easily get exact indices from shuffled batches, we use the order
                # in train_sample_weights which matches train_ds.indices order
                batch_start_idx = batch_idx * config.batch_size
                batch_weights = []
                for i in range(len(action)):
                    sample_idx = batch_start_idx + i
                    if sample_idx < len(train_sample_weights):
                        batch_weights.append(train_sample_weights[sample_idx])
                    else:
                        batch_weights.append(1.0)  # fallback for last incomplete batch

                weights_tensor = torch.tensor(batch_weights, dtype=torch.float32, device=config.device)

                # Weighted loss
                loss = (per_sample_loss * weights_tensor).mean()
            else:
                # Standard loss (unchanged behavior)
                loss = loss_fn(logits, action)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            train_correct += (preds == action).sum().item()
            train_total += action.size(0)

        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        train_acc = train_correct / train_total if train_total > 0 else 0.0

        # Validation phase
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch in val_loader:
                self_team = batch["self_team"].to(config.device)
                opp_team = batch["opp_team"].to(config.device)
                action = batch["action"].to(config.device)

                logits = model(self_team, opp_team)
                preds = logits.argmax(dim=1)
                val_correct += (preds == action).sum().item()
                val_total += action.size(0)

        val_acc = val_correct / val_total if val_total > 0 else 0.0

        print(
            f"Epoch {epoch + 1}/{config.epochs}: train_loss={avg_train_loss:.6f}, "
            f"train_acc={train_acc:.4f}, val_acc={val_acc:.4f}"
        )

    # Build checkpoint dict
    ckpt = {
        "model_state": model.state_dict(),
        "num_sets": ds.num_sets,
        "set_ids": ds.set_ids_sorted,  # IMPORTANT: align with policy expectations
        "format_id": config.format_id,
        "hyperparams": {
            "embed_dim": embed_dim,
            "hidden_dim": hidden_dim,
            "lr": config.lr,
            "epochs": config.epochs,
            "batch_size": config.batch_size,
            "weight_decay": config.weight_decay,
        },
    }

    # Ensure output directory exists
    config.ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # Save checkpoint
    torch.save(ckpt, config.ckpt_path)
    print(f"Preview model checkpoint saved to: {config.ckpt_path}")
    return config.ckpt_path


if __name__ == "__main__":
    cfg = TrainPreviewConfig()
    ckpt_path = train_preview_model(cfg)
    print(f"Checkpoint saved to: {ckpt_path}")
