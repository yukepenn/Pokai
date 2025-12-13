"""In-battle BC policy for inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import torch
from torch import nn

from vgc_lab import encode_state_from_request, get_paths
from vgc_lab.core import DEFAULT_FORMAT, PROJECT_ROOT

from .dataset import encode_battle_state_to_vec
from .train_bc import BattlePolicyBC
from .train_dqn import BattleQNetwork


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


@dataclass
class BattleDqnPolicyConfig:
    """Configuration for BattleDqnPolicy."""

    format_id: str = DEFAULT_FORMAT
    ckpt_path: Path = PROJECT_ROOT / "checkpoints" / "battle_dqn.pt"
    vec_dim: int = 256
    num_actions: int = 4
    device: str = "cpu"


def load_battle_dqn_checkpoint(
    cfg: BattleDqnPolicyConfig,
) -> Tuple[BattleQNetwork, Dict]:
    """
    Load a BattleQNetwork and metadata from a DQN checkpoint.

    Validates that input_dim / num_actions in the checkpoint are compatible
    with the config.

    Raises a clear error if the checkpoint is missing or incompatible.
    """
    if not cfg.ckpt_path.exists():
        raise FileNotFoundError(
            f"Battle DQN checkpoint not found: {cfg.ckpt_path}. "
            "Train a model first using: python -m scripts.cli train-battle-dqn"
        )

    ckpt = torch.load(cfg.ckpt_path, map_location=cfg.device)

    # Validate format_id
    ckpt_format_id = ckpt.get("format_id")
    if ckpt_format_id is not None and ckpt_format_id != cfg.format_id:
        raise ValueError(
            f"Battle DQN checkpoint format_id={ckpt_format_id!r} "
            f"does not match config format_id={cfg.format_id!r}."
        )

    input_dim = int(ckpt.get("input_dim", cfg.vec_dim))
    num_actions = int(ckpt.get("num_actions", cfg.num_actions))

    # Basic compatibility checks
    if input_dim != cfg.vec_dim:
        raise ValueError(
            f"Checkpoint input_dim={input_dim} does not match cfg.vec_dim={cfg.vec_dim}"
        )
    if num_actions != cfg.num_actions:
        raise ValueError(
            f"Checkpoint num_actions={num_actions} does not match cfg.num_actions={cfg.num_actions}"
        )

    model = BattleQNetwork(input_dim, num_actions)
    model.load_state_dict(ckpt["model_state"])
    model.to(cfg.device)
    model.eval()

    return model, ckpt


class BattleDqnPolicy:
    """
    Policy wrapper around a BattleQNetwork.

    This policy:
    - Encodes Showdown requests into fixed-size state vectors
    - Produces Q-values for a small discrete action set
    - Chooses a greedy action (argmax over Q-values)
    - Maps action indices to simple 'move N' commands
    """

    def __init__(self, cfg: BattleDqnPolicyConfig) -> None:
        """Initialize policy by loading checkpoint.

        Args:
            cfg: Configuration.
        """
        self.cfg = cfg
        self.model, self.ckpt = load_battle_dqn_checkpoint(cfg)
        self.device = cfg.device
        self.vec_dim = cfg.vec_dim
        self.num_actions = cfg.num_actions

    def _encode_request(self, request: Dict[str, Any]) -> torch.Tensor:
        """
        Convert a Showdown request dict to a 1D float tensor of shape (vec_dim,).

        Args:
            request: Showdown request dictionary.

        Returns:
            Tensor of shape (vec_dim,) on the configured device.
        """
        state_dict = encode_state_from_request(request)
        vec = encode_battle_state_to_vec(state_dict, vec_dim=self.vec_dim)
        x = torch.from_numpy(vec).float().to(self.device)
        return x

    def score_actions(self, request: Dict[str, Any]) -> torch.Tensor:
        """
        Return Q-values for all actions for a single request.

        Args:
            request: Showdown request dictionary.

        Returns:
            Q-values tensor of shape [num_actions] on the configured device.
        """
        x = self._encode_request(request).unsqueeze(0)  # [1, vec_dim]
        with torch.no_grad():
            q_values = self.model(x)  # [1, num_actions]
        return q_values.squeeze(0)  # [num_actions]

    def choose_action_argmax(self, request: Dict[str, Any]) -> Tuple[int, str]:
        """
        Choose the greedy action (argmax Q) and return (action_index, choice_str).

        choice_str uses 'move N' with 1-based indexing, consistent with parse_move_choice().

        Args:
            request: Showdown request dictionary.

        Returns:
            Tuple of (action_index: int, choice_str: str) where choice_str is like "move 1".
        """
        q_values = self.score_actions(request)
        action_index = int(torch.argmax(q_values).item())
        if not (0 <= action_index < self.num_actions):
            raise ValueError(
                f"Invalid action_index {action_index} for num_actions={self.num_actions}"
            )

        # For now, we map 0..(num_actions-1) -> 'move 1'..'move num_actions'
        # This is consistent with parse_move_choice() which maps 'move N' (1-indexed) to index N-1
        choice_str = f"move {action_index + 1}"
        return action_index, choice_str

    def choose_showdown_command(self, request: Dict[str, Any]) -> str:
        """
        Convenience method: just return the Showdown command string.

        Args:
            request: Showdown request dictionary.

        Returns:
            Showdown choice string like "move 1".
        """
        _, choice = self.choose_action_argmax(request)
        return choice


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

