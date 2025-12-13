"""Offline evaluation script for PreviewModel.

This module computes offline top-1 accuracy on bring-4 actions
using the current PreviewDataset and PreviewPolicy.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Collection, Dict, Optional

import torch
from torch.utils.data import DataLoader

from .dataset import PreviewDataset
from .policy import PreviewPolicy


@dataclass
class EvalPreviewConfig:
    """Configuration for preview model offline evaluation."""

    format_id: str = "gen9vgc2026regf"
    batch_size: int = 64
    allowed_policy_ids: Optional[Collection[str]] = None
    min_created_at: Optional[datetime] = None
    include_default_choices: bool = False
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def run_eval_preview(config: EvalPreviewConfig) -> Dict[str, float]:
    """Run offline evaluation of PreviewPolicy on PreviewDataset.

    Args:
        config: EvalPreviewConfig with dataset filters and device

    Returns:
        Dict with "num_examples" and "accuracy" keys

    Raises:
        ValueError: If dataset is empty
    """
    # Create dataset
    ds = PreviewDataset(
        format_id=config.format_id,
        allowed_policy_ids=config.allowed_policy_ids,
        min_created_at=config.min_created_at,
        include_default_choices=config.include_default_choices,
    )

    if ds.num_examples == 0:
        raise ValueError(
            "PreviewDataset is empty; generate battles first using "
            "python -m scripts.cli gen-full or projects.rl_team_build.selfplay. "
            "If you only have old 'default' team-preview choices, you may need to generate new "
            "battles with the updated js/random_selfplay.js that logs explicit 'team XXXX' choices."
        )

    # Create DataLoader
    loader = DataLoader(ds, batch_size=config.batch_size, shuffle=False)

    # Create policy
    policy = PreviewPolicy(format_id=config.format_id, device=config.device)

    # Get set_ids mapping from dataset
    set_ids_sorted = ds.set_ids_sorted

    # Evaluate
    correct = 0
    total = 0

    for batch in loader:
        batch_size = batch["self_team"].size(0)

        for i in range(batch_size):
            # Convert indices back to set_ids
            self_idx = batch["self_team"][i].tolist()
            opp_idx = batch["opp_team"][i].tolist()

            self_set_ids = [set_ids_sorted[j] for j in self_idx]
            opp_set_ids = [set_ids_sorted[j] for j in opp_idx]

            # Get predicted action
            predicted_action_index, _ = policy.choose_action_argmax(
                self_set_ids, opp_set_ids
            )

            # Get true action
            true_action = int(batch["action"][i].item())

            if predicted_action_index == true_action:
                correct += 1
            total += 1

    accuracy = correct / total if total > 0 else 0.0

    return {
        "num_examples": total,
        "accuracy": accuracy,
    }


if __name__ == "__main__":
    cfg = EvalPreviewConfig()
    summary = run_eval_preview(cfg)
    print(
        f"Preview offline eval: num_examples={summary['num_examples']}, "
        f"accuracy={summary['accuracy']:.4f}"
    )

