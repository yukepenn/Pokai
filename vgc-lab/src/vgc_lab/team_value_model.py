"""Neural team value model for offline learning."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from .team_encoding import SetIdVocab
from .team_features import (
    TeamsVsPoolRecord,
    build_default_vocab,
    iter_teams_vs_pool_records,
)


@dataclass
class TeamValueSample:
    """A single training sample for the team value model."""

    set_indices: List[int]  # length = team_size (6)
    win_rate: float  # in [0, 1]
    n_battles: int  # for weighting
    source: str  # "random" / "catalog" / etc.


def load_team_value_samples(
    jsonl_path: Path,
    *,
    min_battles: int = 3,
    include_sources: Optional[List[str]] = None,
    rng: Optional[random.Random] = None,
) -> Tuple[List[TeamValueSample], SetIdVocab]:
    """
    Load teams_vs_pool.jsonl and build a list of TeamValueSample plus a SetIdVocab.

    Args:
        jsonl_path: Path to the teams_vs_pool.jsonl file.
        min_battles: Filter out extremely noisy samples with very few battles.
        include_sources: If not None, only include records whose 'source' is in this list.
        rng: Optional random.Random for shuffling or splitting (if needed).

    Returns:
        Tuple of (samples list, vocab instance).

    Raises:
        ValueError: If a team does not have exactly 6 sets.
    """
    if rng is None:
        rng = random.Random()

    vocab = build_default_vocab()
    samples: List[TeamValueSample] = []

    for record in iter_teams_vs_pool_records(jsonl_path):
        # Filter by min_battles
        if record.n_battles_total < min_battles:
            continue

        # Filter by source
        if include_sources is not None and record.source not in include_sources:
            continue

        # Validate team size
        if len(record.team_set_ids) != 6:
            raise ValueError(
                f"Expected exactly 6 set IDs per team, got {len(record.team_set_ids)}"
            )

        # Encode team_set_ids to indices
        try:
            indices = vocab.encode_ids(record.team_set_ids)
        except KeyError as e:
            # Skip if any set ID is not in vocab (shouldn't happen with default vocab)
            continue

        sample = TeamValueSample(
            set_indices=indices,
            win_rate=record.win_rate,
            n_battles=record.n_battles_total,
            source=record.source,
        )
        samples.append(sample)

    # Shuffle samples
    rng.shuffle(samples)

    return samples, vocab


class TeamValueTorchDataset(Dataset):
    """PyTorch Dataset for team value samples."""

    def __init__(self, samples: List[TeamValueSample], vocab_size: int, team_size: int = 6):
        """
        Initialize dataset.

        Args:
            samples: List of TeamValueSample instances.
            vocab_size: Size of the vocabulary (for validation).
            team_size: Expected team size (default: 6).
        """
        self.samples = samples
        self.vocab_size = vocab_size
        self.team_size = team_size

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single sample.

        Returns:
            Tuple of (indices, win_rate, weight) tensors.
        """
        sample = self.samples[idx]
        indices = torch.tensor(sample.set_indices, dtype=torch.long)  # shape [team_size]
        win_rate = torch.tensor(sample.win_rate, dtype=torch.float32)  # scalar
        weight = torch.tensor(
            min(sample.n_battles, 20) / 20.0,
            dtype=torch.float32,
        )  # cap at 20 for stability
        return indices, win_rate, weight


class TeamValueModel(nn.Module):
    """Neural network model to predict team win rate."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 64,
        hidden_dim: int = 256,
        team_size: int = 6,
    ):
        """
        Initialize the model.

        Args:
            vocab_size: Number of unique set IDs in vocabulary.
            embedding_dim: Dimension of set embeddings.
            hidden_dim: Dimension of hidden layers in MLP.
            team_size: Number of sets per team (default: 6).
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.team_size = team_size

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, set_indices: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            set_indices: LongTensor of shape [batch_size, team_size]

        Returns:
            FloatTensor of shape [batch_size], predicted win_rate in [0,1]
        """
        emb = self.embedding(set_indices)  # [B, T, D]
        team_emb = emb.mean(dim=1)  # [B, D]
        logits = self.mlp(team_emb).squeeze(-1)  # [B]
        win_pred = self.sigmoid(logits)  # [B]
        return win_pred


@dataclass
class TrainingConfig:
    """Configuration for training the team value model."""

    jsonl_path: Path
    out_path: Optional[Path] = None  # Optional path to save checkpoint (if None, don't save)
    batch_size: int = 64
    epochs: int = 50
    lr: float = 3e-4
    min_battles: int = 3
    include_sources: Optional[List[str]] = None
    embedding_dim: int = 64
    hidden_dim: int = 256
    team_size: int = 6
    val_ratio: float = 0.2
    seed: int = 42
    device: str = "cpu"  # or "cuda" if available


def train_team_value_model(
    config: TrainingConfig,
    device: Optional[str] = None,
) -> Tuple[TeamValueModel, SetIdVocab, Dict[str, float]]:
    """
    Train a TeamValueModel on teams_vs_pool.jsonl using weighted MSE on win_rate.

    Args:
        config: Training configuration.
        device: Optional device override (if None, uses config.device).

    Returns:
        Tuple of (trained model, vocab instance, training metadata dict).
        Metadata dict contains: train_loss, val_loss, val_corr, n_train, n_val (all floats).
    """
    # Use device override if provided
    train_device = device if device is not None else config.device
    # Set random seeds for reproducibility
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load samples and vocab
    samples, vocab = load_team_value_samples(
        config.jsonl_path,
        min_battles=config.min_battles,
        include_sources=config.include_sources,
        rng=random.Random(config.seed),
    )

    if not samples:
        raise ValueError(f"No samples found in {config.jsonl_path} after filtering")

    # Build dataset
    dataset = TeamValueTorchDataset(samples, vocab_size=len(vocab), team_size=config.team_size)

    # Split train/val
    val_size = int(len(dataset) * config.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed)
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False
    )

    # Instantiate model
    model = TeamValueModel(
        vocab_size=len(vocab),
        embedding_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        team_size=config.team_size,
    )
    model.to(train_device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Training loop
    for epoch in range(config.epochs):
        # Train phase
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for indices, win_rate, weight in train_loader:
            indices = indices.to(train_device)
            win_rate = win_rate.to(train_device)
            weight = weight.to(train_device)

            optimizer.zero_grad()
            pred = model(indices)
            loss = (weight * (pred - win_rate) ** 2).mean()
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(indices)
            train_count += len(indices)

        avg_train_loss = train_loss_sum / train_count if train_count > 0 else 0.0

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for indices, win_rate, weight in val_loader:
                indices = indices.to(train_device)
                win_rate = win_rate.to(train_device)
                weight = weight.to(train_device)

                pred = model(indices)
                loss = (weight * (pred - win_rate) ** 2).mean()

                val_loss_sum += loss.item() * len(indices)
                val_count += len(indices)

                preds_list.append(pred.cpu())
                targets_list.append(win_rate.cpu())

        avg_val_loss = val_loss_sum / val_count if val_count > 0 else 0.0

        # Compute correlation
        if preds_list:
            all_preds = torch.cat(preds_list)
            all_targets = torch.cat(targets_list)
            correlation = torch.corrcoef(torch.stack([all_preds, all_targets]))[0, 1]
            if torch.isnan(correlation):
                correlation = 0.0
        else:
            correlation = 0.0

        # Print summary
        if (epoch + 1) % max(1, config.epochs // 10) == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{config.epochs}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={avg_val_loss:.4f}, "
                f"val_corr={correlation:.4f}"
            )

    model.eval()

    # Prepare training metadata (use final epoch metrics)
    train_meta = {
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_corr": float(correlation),
        "n_train": train_count,
        "n_val": val_count,
    }

    # Save checkpoint if out_path is specified
    if config.out_path is not None:
        save_team_value_checkpoint(model, vocab, config, config.out_path)

    return model, vocab, train_meta


def save_team_value_checkpoint(
    model: TeamValueModel,
    vocab: SetIdVocab,
    config: TrainingConfig,
    path: Path,
) -> None:
    """
    Save model checkpoint to disk.

    Args:
        model: Trained TeamValueModel instance.
        vocab: SetIdVocab instance used for training.
        config: Training configuration.
        path: Path to save the checkpoint.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab_idx_to_id": vocab.idx_to_id,
            "model_config": {
                "vocab_size": model.vocab_size,
                "embedding_dim": model.embedding_dim,
                "hidden_dim": config.hidden_dim,
                "team_size": model.team_size,
            },
        },
        path,
    )


def load_team_value_checkpoint(
    path: Path, device: str = "cpu"
) -> Tuple[TeamValueModel, SetIdVocab, dict]:
    """
    Load model checkpoint from disk.

    Args:
        path: Path to the checkpoint file.
        device: Device to load the model on ("cpu" or "cuda").

    Returns:
        Tuple of (model, vocab, model_config dict).
    """
    ckpt = torch.load(path, map_location=device)
    idx_to_id = ckpt["vocab_idx_to_id"]
    model_cfg = ckpt["model_config"]

    vocab = SetIdVocab.from_idx_to_id(idx_to_id)

    model = TeamValueModel(
        vocab_size=model_cfg["vocab_size"],
        embedding_dim=model_cfg["embedding_dim"],
        hidden_dim=model_cfg["hidden_dim"],
        team_size=model_cfg["team_size"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, vocab, model_cfg

