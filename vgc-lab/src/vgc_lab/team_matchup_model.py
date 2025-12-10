"""Neural team matchup model for pairwise win probability prediction."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# Sequence already imported above

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, random_split

from .team_encoding import SetIdVocab
from .team_matchup_data import TeamMatchupRecord, iter_team_matchup_records
from .set_catalog import SetCatalog


@dataclass
class MatchupTrainingConfig:
    """Configuration for training the team matchup model."""

    jsonl_path: Path
    out_path: Optional[Path] = None
    batch_size: int = 64
    lr: float = 1e-3
    epochs: int = 40
    device: str = "cuda"
    min_battles: int = 10
    max_records: Optional[int] = None
    val_ratio: float = 0.2
    seed: int = 42
    embedding_dim: int = 64
    hidden_dim: int = 128


def _collate_matchup_batch(
    batch: List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]
) -> Tuple[List[torch.Tensor], List[torch.Tensor], torch.Tensor]:
    """
    Custom collate function for matchup batches.

    Args:
        batch: List of (a_indices, b_indices, win_rate) tuples.

    Returns:
        Tuple of (a_indices_list, b_indices_list, win_rates_tensor).
    """
    a_indices_list = [item[0] for item in batch]
    b_indices_list = [item[1] for item in batch]
    win_rates = torch.stack([item[2] for item in batch], dim=0)

    return a_indices_list, b_indices_list, win_rates


class TeamMatchupTorchDataset(Dataset):
    """PyTorch Dataset for team matchup records."""

    def __init__(self, records: Sequence[TeamMatchupRecord], vocab: SetIdVocab):
        """
        Initialize dataset.

        Args:
            records: Sequence of TeamMatchupRecord instances.
            vocab: SetIdVocab for encoding team set_ids to indices.
        """
        self.records = list(records)
        self.vocab = vocab

    def __len__(self) -> int:
        return len(self.records)

    def _encode_team(self, set_ids: Sequence[str]) -> torch.Tensor:
        """
        Encode a team's set_ids to indices.

        Args:
            set_ids: Sequence of set ID strings.

        Returns:
            LongTensor of indices.
        """
        indices = self.vocab.encode_ids(set_ids)
        return torch.tensor(indices, dtype=torch.long)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a single matchup record.

        Returns:
            Tuple of (a_indices, b_indices, win_rate_a).
        """
        rec = self.records[idx]

        a_idx = self._encode_team(rec.team_a_set_ids)
        b_idx = self._encode_team(rec.team_b_set_ids)

        if rec.n_battles > 0:
            win_rate_a = rec.n_a_wins / rec.n_battles
        else:
            win_rate_a = 0.5

        win_rate_a_tensor = torch.tensor(win_rate_a, dtype=torch.float32)

        return a_idx, b_idx, win_rate_a_tensor


class TeamMatchupModel(nn.Module):
    """Neural network model to predict P(A wins vs B)."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 128,
    ):
        """
        Initialize the model.

        Args:
            vocab_size: Number of unique set IDs in vocabulary.
            embed_dim: Dimension of set embeddings.
            hidden_dim: Dimension of hidden layers in MLP.
        """
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.embedding = nn.Embedding(vocab_size, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),  # output ~ probability win_rate_a in [0,1]
        )

    def encode_team_batch(self, batch_indices: List[torch.Tensor]) -> torch.Tensor:
        """
        Encode a batch of teams (lists of set indices) to team embeddings.

        Args:
            batch_indices: List of 1D LongTensors, each containing set indices for a team.

        Returns:
            FloatTensor of shape (batch, embed_dim).
        """
        embs = []
        for idx in batch_indices:
            emb = self.embedding(idx)  # (num_sets, embed_dim)
            team_emb = emb.mean(dim=0)  # (embed_dim,)
            embs.append(team_emb)
        return torch.stack(embs, dim=0)  # (batch, embed_dim)

    def forward(
        self, a_indices_list: List[torch.Tensor], b_indices_list: List[torch.Tensor]
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            a_indices_list: List of 1D LongTensors for team A set indices.
            b_indices_list: List of 1D LongTensors for team B set indices.

        Returns:
            FloatTensor of shape (batch,), predicted P(A wins vs B) in [0,1].
        """
        a_embed = self.encode_team_batch(a_indices_list)  # (batch, embed_dim)
        b_embed = self.encode_team_batch(b_indices_list)  # (batch, embed_dim)

        # Concatenate team embeddings
        x = torch.cat([a_embed, b_embed], dim=-1)  # (batch, embed_dim * 2)

        # Predict win rate
        out = self.mlp(x).squeeze(-1)  # (batch,)

        return out


def train_team_matchup_model(
    config: MatchupTrainingConfig,
    device_override: Optional[str] = None,
) -> Tuple[TeamMatchupModel, SetIdVocab, Dict[str, float]]:
    """
    Train a TeamMatchupModel on matchup JSONL data.

    Args:
        config: Training configuration.
        device_override: Optional device override (if None, uses config.device).

    Returns:
        Tuple of (trained model, vocab, training metadata dict).
    """
    # Use device override if provided
    train_device = device_override if device_override is not None else config.device

    # Set random seeds
    random.seed(config.seed)
    torch.manual_seed(config.seed)

    # Load records
    records = list(iter_team_matchup_records(config.jsonl_path))

    if not records:
        raise ValueError(f"No records found in {config.jsonl_path}")

    print(f"Loaded {len(records)} matchup records")

    # Filter by min_battles
    filtered = [r for r in records if r.n_battles >= config.min_battles]
    print(
        f"After min_battles={config.min_battles} filter: {len(filtered)} records"
    )

    if not filtered:
        raise ValueError(
            f"No records match min_battles={config.min_battles} threshold"
        )

    # Optionally subsample
    if config.max_records is not None and len(filtered) > config.max_records:
        random.shuffle(filtered)
        filtered = filtered[: config.max_records]
        print(f"Subsampled to {len(filtered)} records")

    # Build vocab from all unique set_ids
    all_set_ids = set()
    for rec in filtered:
        all_set_ids.update(rec.team_a_set_ids)
        all_set_ids.update(rec.team_b_set_ids)

    # Use SetCatalog to get deterministic ordering
    catalog = SetCatalog.from_yaml()
    vocab = SetIdVocab.from_catalog(catalog)

    # Filter records to only include teams with all set_ids in vocab
    valid_records = []
    for rec in filtered:
        try:
            vocab.encode_ids(rec.team_a_set_ids)
            vocab.encode_ids(rec.team_b_set_ids)
            valid_records.append(rec)
        except KeyError:
            continue

    print(f"Valid records (all set_ids in vocab): {len(valid_records)}")

    if not valid_records:
        raise ValueError("No valid records after vocab filtering")

    # Build dataset
    dataset = TeamMatchupTorchDataset(valid_records, vocab)

    # Split train/val
    val_size = int(len(dataset) * config.val_ratio)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size], generator=torch.Generator().manual_seed(config.seed)
    )

    # Instantiate model
    model = TeamMatchupModel(
        vocab_size=len(vocab),
        embed_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
    )
    model.to(train_device)

    # Create data loaders with custom collate
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=_collate_matchup_batch
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=_collate_matchup_batch
    )

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Training loop
    best_val_loss = float("inf")
    for epoch in range(config.epochs):
        # Train phase
        model.train()
        train_loss_sum = 0.0
        train_count = 0

        for a_indices_list, b_indices_list, win_rates in train_loader:
            optimizer.zero_grad()

            pred = model(a_indices_list, b_indices_list)

            loss = nn.functional.mse_loss(pred, win_rates.to(train_device))

            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * len(win_rates)
            train_count += len(win_rates)

        avg_train_loss = train_loss_sum / train_count if train_count > 0 else 0.0

        # Validation phase
        model.eval()
        val_loss_sum = 0.0
        val_count = 0
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for a_indices_list, b_indices_list, win_rates in val_loader:
                pred = model(a_indices_list, b_indices_list)

                loss = nn.functional.mse_loss(pred, win_rates.to(train_device))

                val_loss_sum += loss.item() * len(win_rates)
                val_count += len(win_rates)

                preds_list.append(pred.cpu())
                targets_list.append(win_rates.cpu())

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

        best_val_loss = min(best_val_loss, avg_val_loss)

        # Print summary
        if (epoch + 1) % max(1, config.epochs // 10) == 0 or epoch == 0:
            print(
                f"Epoch {epoch + 1}/{config.epochs}: "
                f"train_loss={avg_train_loss:.4f}, "
                f"val_loss={avg_val_loss:.4f}, "
                f"val_corr={correlation:.4f}"
            )

    model.eval()

    # Prepare training metadata
    train_meta = {
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "val_corr": float(correlation),
        "best_val_loss": best_val_loss,
        "n_train": train_count,
        "n_val": val_count,
        "n_records": len(valid_records),
    }

    # Save checkpoint if out_path is specified
    if config.out_path is not None:
        save_team_matchup_checkpoint(model, vocab, config, config.out_path)

    return model, vocab, train_meta


def save_team_matchup_checkpoint(
    model: TeamMatchupModel,
    vocab: SetIdVocab,
    config: MatchupTrainingConfig,
    path: Path,
) -> None:
    """
    Save model checkpoint to disk.

    Args:
        model: Trained TeamMatchupModel instance.
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
                "embed_dim": model.embed_dim,
                "hidden_dim": config.hidden_dim,
            },
        },
        path,
    )


def load_team_matchup_checkpoint(
    path: Path, device: str = "cpu"
) -> Tuple[TeamMatchupModel, SetIdVocab, dict]:
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

    from .team_encoding import SetIdVocab

    vocab = SetIdVocab.from_idx_to_id(idx_to_id)

    model = TeamMatchupModel(
        vocab_size=model_cfg["vocab_size"],
        embed_dim=model_cfg["embed_dim"],
        hidden_dim=model_cfg["hidden_dim"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()

    return model, vocab, model_cfg


def predict_matchup_win_prob(
    model: TeamMatchupModel,
    vocab: SetIdVocab,
    team_a_set_ids: Sequence[str],
    team_b_set_ids: Sequence[str],
    *,
    device: str = "cpu",
) -> float:
    """
    Convenience helper: given set_ids for teams A and B, use the matchup model to
    predict P(A wins vs B).

    - Moves model to eval mode.
    - Encodes set_ids to indices via vocab.
    - Runs a forward pass on a single pair.
    - Returns a Python float in [0, 1].
    - Does not modify model training state (no gradients).

    Args:
        model: Trained TeamMatchupModel instance.
        vocab: SetIdVocab for encoding set_ids to indices.
        team_a_set_ids: Sequence of set ID strings for team A.
        team_b_set_ids: Sequence of set ID strings for team B.
        device: Device to run inference on ("cpu" or "cuda").

    Returns:
        Predicted probability that team A wins vs team B (float in [0, 1]).

    Raises:
        KeyError: If any set_id is not in the vocabulary.
    """
    model.eval()

    # Encode set_ids to indices
    a_indices = vocab.encode_ids(team_a_set_ids)
    b_indices = vocab.encode_ids(team_b_set_ids)

    # Convert to tensors
    a_tensor = torch.tensor([a_indices], dtype=torch.long, device=device)
    b_tensor = torch.tensor([b_indices], dtype=torch.long, device=device)

    # Predict
    with torch.no_grad():
        pred = model([a_tensor[0]], [b_tensor[0]]).item()

    return float(pred)

