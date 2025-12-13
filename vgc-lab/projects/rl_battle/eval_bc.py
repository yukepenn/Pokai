"""Offline evaluation for battle BC model."""

from __future__ import annotations

from typing import Tuple

import torch
from torch.utils.data import DataLoader

from .dataset import BattleStepDataset, BattleStepDatasetConfig
from .policy import BattleBCPolicy, BattleBCPolicyConfig


def evaluate_battle_bc(batch_size: int = 512, topk: int = 1) -> Tuple[float, float]:
    """
    Evaluate the battle BC policy on the dataset.

    Args:
        batch_size: Batch size for evaluation.
        topk: If > 1, compute top-k accuracy (true action in top-k predictions).
              If <= 1, compute standard argmax accuracy.

    Returns:
        Tuple of (accuracy: float, total_examples: float).
    """
    ds = BattleStepDataset(BattleStepDatasetConfig())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    cfg = BattleBCPolicyConfig()
    policy = BattleBCPolicy(cfg)

    total = 0
    correct = 0

    with torch.no_grad():
        for batch in loader:
            states = batch["state"].to(cfg.device)
            actions = batch["action"].to(cfg.device)
            logits = policy.model(states)

            if topk <= 1:
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == actions).sum().item()
            else:
                topk_vals, topk_idx = torch.topk(logits, k=topk, dim=-1)
                matches = (topk_idx == actions.unsqueeze(-1)).any(dim=-1)
                correct += matches.sum().item()

            total += actions.size(0)

    acc = correct / max(total, 1)
    print(f"Offline BC top-{topk} accuracy on dataset: {acc:.4f} ({correct}/{total})")
    print(f"Num actions (vocab size): {policy.num_actions}")
    return acc, float(total)


if __name__ == "__main__":
    evaluate_battle_bc()

