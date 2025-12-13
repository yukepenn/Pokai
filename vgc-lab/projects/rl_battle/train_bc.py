"""Training script for in-battle behavior cloning."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split

from vgc_lab.core import PROJECT_ROOT, SanitizeReason

from .dataset import BattleStepDataset, BattleStepDatasetConfig


@dataclass
class BattleBCConfig:
    """Configuration for battle BC training."""

    format_id: str = "gen9vgc2026regf"
    batch_size: int = 256
    lr: float = 1e-3
    epochs: int = 5
    val_frac: float = 0.2
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    # Outcome-aware weighting options
    use_outcome_weights: bool = False
    winner_weight: float = 2.0
    loser_weight: float = 0.5
    draw_weight: float = 1.0
    # Sanitize reason filtering: if None, include all steps regardless of sanitize_reason.
    # If provided, only steps with sanitize_reason in this list are included.
    # Default: ["ok", "fixed_pass"] - only include clean steps and steps where we fixed a "pass"
    train_allowed_sanitize_reasons: Optional[List[SanitizeReason]] = field(
        default_factory=lambda: ["ok", "fixed_pass"]
    )


class BattlePolicyBC(nn.Module):
    """Simple MLP for battle move choice prediction."""

    def __init__(self, input_dim: int, hidden_dim: int = 256, num_actions: int = 4) -> None:
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


def train_battle_bc(cfg: BattleBCConfig) -> Path:
    """
    Train a simple behavior cloning policy on battle steps with full joint action choices.

    Args:
        cfg: BattleBCConfig with training hyperparameters.

    Returns:
        Path to the saved checkpoint.
    """
    torch.manual_seed(cfg.seed)

    # Build dataset
    ds_cfg = BattleStepDatasetConfig(
        format_id=cfg.format_id,
        train_allowed_sanitize_reasons=cfg.train_allowed_sanitize_reasons,
    )
    dataset = BattleStepDataset(ds_cfg)

    if dataset.num_examples == 0:
        raise ValueError("BattleStepDataset is empty; cannot train BC model.")

    num_actions = dataset.num_actions
    if num_actions <= 0:
        raise ValueError("BattleStepDataset has no actions; cannot train BC model.")

    has_outcomes = getattr(dataset, "has_outcomes", False)
    print(f"BattleStepDataset has outcomes: {has_outcomes}")

    if cfg.use_outcome_weights and not has_outcomes:
        print("WARNING: use_outcome_weights=True but dataset has no outcomes; "
              "falling back to uniform weights.")

    # Train/val split
    val_size = int(len(dataset) * cfg.val_frac)
    train_size = len(dataset) - val_size
    train_ds, val_ds = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(cfg.seed),
    )

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)

    input_dim = dataset.num_features

    model = BattlePolicyBC(input_dim=input_dim, num_actions=num_actions).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    criterion = nn.CrossEntropyLoss(reduction="none")

    def run_epoch(loader: DataLoader, train: bool) -> Tuple[float, float]:
        if train:
            model.train()
        else:
            model.eval()

        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for batch in loader:
            state = batch["state"].to(cfg.device)
            action = batch["action"].to(cfg.device)

            logits = model(state)
            per_sample_loss = criterion(logits, action)  # shape: (batch_size,)

            # Default weights = 1.0
            weights = torch.ones_like(per_sample_loss, device=cfg.device)

            if cfg.use_outcome_weights and "outcome" in batch:
                outcome = batch["outcome"].to(cfg.device)  # -1, 0, +1
                # Map outcome -> weight
                weights = torch.where(outcome > 0, cfg.winner_weight, weights)
                weights = torch.where(outcome < 0, cfg.loser_weight, weights)
                # outcome == 0 stays at draw_weight
                weights = torch.where(outcome == 0, cfg.draw_weight, weights)

            # Weighted mean loss
            loss = (per_sample_loss * weights).mean()

            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * state.size(0)
            preds = logits.argmax(dim=1)
            total_correct += (preds == action).sum().item()
            total_examples += state.size(0)

        avg_loss = total_loss / max(total_examples, 1)
        acc = total_correct / max(total_examples, 1)
        return avg_loss, acc

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = run_epoch(train_loader, train=True)
        val_loss, val_acc = run_epoch(val_loader, train=False)
        print(
            f"Epoch {epoch}/{cfg.epochs}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

    # Save checkpoint
    ckpt_path = PROJECT_ROOT / "checkpoints" / "battle_bc.pt"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state": model.state_dict(),
            "input_dim": input_dim,
            "num_actions": num_actions,
            "format_id": cfg.format_id,
            "id_to_choice": dataset.id_to_choice,
            "action_mapping": {
                "description": (
                    "Joint action over full Showdown choice strings; "
                    "class id -> choice_str via id_to_choice"
                ),
                "num_actions": num_actions,
            },
            "hyperparams": {
                "lr": cfg.lr,
                "epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "use_outcome_weights": cfg.use_outcome_weights,
                "winner_weight": cfg.winner_weight,
                "loser_weight": cfg.loser_weight,
                "draw_weight": cfg.draw_weight,
            },
        },
        ckpt_path,
    )
    print(f"Battle BC checkpoint saved to: {ckpt_path}")
    return ckpt_path


if __name__ == "__main__":
    cfg = BattleBCConfig()
    train_battle_bc(cfg)

