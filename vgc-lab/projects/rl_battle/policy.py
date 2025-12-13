"""In-battle BC policy for inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from vgc_lab import encode_state_from_request, get_paths
from vgc_lab.core import PROJECT_ROOT

from .dataset import encode_battle_state_to_vec
from .train_bc import BattlePolicyBC


@dataclass
class BattleBCPolicyConfig:
    """Configuration for BattleBCPolicy."""

    format_id: str = "gen9vgc2026regf"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def load_battle_bc_checkpoint(
    cfg: BattleBCPolicyConfig,
) -> Tuple[BattlePolicyBC, Dict]:
    """
    Load the battle BC checkpoint and return (model, metadata).

    Raises a clear error if the checkpoint is missing or incompatible.
    """
    ckpt_path = PROJECT_ROOT / "checkpoints" / "battle_bc.pt"

    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Battle BC checkpoint not found: {ckpt_path}. "
            "Train a model first using: python -m projects.rl_battle.train_bc"
        )

    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    # Validate format_id
    ckpt_format_id = ckpt.get("format_id")
    if ckpt_format_id is not None and ckpt_format_id != cfg.format_id:
        raise ValueError(
            f"Battle BC checkpoint format_id={ckpt_format_id!r} "
            f"does not match config format_id={cfg.format_id!r}."
        )
    
    # Check for id_to_choice (new format)
    if "id_to_choice" not in ckpt:
        raise ValueError(
            "Battle BC checkpoint missing 'id_to_choice'. "
            "It was probably created with an older move-only "
            "version. Please retrain the BC model."
        )
    
    input_dim = ckpt["input_dim"]
    num_actions = ckpt["num_actions"]
    id_to_choice = ckpt["id_to_choice"]

    model = BattlePolicyBC(input_dim=input_dim, num_actions=num_actions)
    model.load_state_dict(ckpt["model_state"])
    model.to(cfg.device)
    model.eval()

    return model, ckpt


class BattleBCPolicy:
    """Policy for in-battle move choices using a trained BC model."""

    def __init__(self, cfg: BattleBCPolicyConfig | None = None) -> None:
        """Initialize policy by loading checkpoint.

        Args:
            cfg: Configuration. If None, uses default BattleBCPolicyConfig().
        """
        if cfg is None:
            cfg = BattleBCPolicyConfig()
        self.cfg = cfg
        self.model, self.ckpt = load_battle_bc_checkpoint(cfg)
        self.id_to_choice = self.ckpt["id_to_choice"]
        self.num_actions = self.ckpt["num_actions"]

    def encode_request(self, request: dict) -> torch.Tensor:
        """
        Convert a Showdown request dict to a 1D float tensor of shape (input_dim,).

        Args:
            request: Showdown request dictionary.

        Returns:
            Tensor of shape (1, input_dim) on the configured device.
        """
        state_dict = encode_state_from_request(request)
        state_vec = encode_battle_state_to_vec(
            state_dict, vec_dim=self.ckpt["input_dim"]
        )
        state = torch.from_numpy(state_vec).unsqueeze(0)  # (1, D)
        return state.to(self.cfg.device)

    def score_actions(self, request: dict) -> torch.Tensor:
        """
        Return logits over actions for the given request: shape (num_actions,).

        Args:
            request: Showdown request dictionary.

        Returns:
            Logits tensor of shape (num_actions,) on the configured device.
        """
        state = self.encode_request(request)
        with torch.no_grad():
            logits = self.model(state)[0]  # (num_actions,)
        return logits

    def choose_action(
        self,
        request: dict,
        *,
        temperature: float = 0.0,
    ) -> Tuple[int, str]:
        """
        Choose an action index and return the corresponding Showdown choice string.

        If temperature <= 0, use greedy argmax. Otherwise, softmax-sample.

        Args:
            request: Showdown request dictionary.
            temperature: Sampling temperature. <= 0 for greedy, > 0 for sampling.

        Returns:
            Tuple of (action_index: int, choice_str: str) where choice_str is the full
            Showdown choice string from the vocabulary.
        """
        logits = self.score_actions(request)
        if temperature <= 0.0:
            action_index = int(torch.argmax(logits).item())
        else:
            probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
            action_index = int(torch.multinomial(probs, num_samples=1).item())

        if not (0 <= action_index < len(self.id_to_choice)):
            raise IndexError(
                f"Predicted action_index {action_index} out of range "
                f"for id_to_choice (size={len(self.id_to_choice)})"
            )

        choice_str = self.id_to_choice[action_index]
        return action_index, choice_str


if __name__ == "__main__":
    # Simple smoke test: load policy and run on a dummy request.
    from .dataset import BattleStepDataset, BattleStepDatasetConfig

    ds = BattleStepDataset(BattleStepDatasetConfig())
    print(f"Loaded BattleStepDataset with {ds.num_examples} examples.")

    cfg = BattleBCPolicyConfig()
    policy = BattleBCPolicy(cfg)
    print("Loaded BattleBCPolicy with checkpoint format_id:", policy.ckpt["format_id"])
    print("Num actions (vocab size):", policy.num_actions)

    # Verify that encode_request works on a dummy dict
    dummy_request = {"example": "dummy"}  # minimal smoke test
    logits = policy.score_actions(dummy_request)
    print(f"Dummy logits shape: {logits.shape}")
    action_idx, choice_str = policy.choose_action(dummy_request)
    print(f"Dummy action: index={action_idx}, choice={choice_str!r}")

