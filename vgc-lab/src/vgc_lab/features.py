"""
Feature encoding utilities for RL/BC training.

This module provides simple encoder skeletons for converting raw data structures
into feature representations suitable for machine learning models.
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .core import BattleStep, BattleTrajectory
from .datasets import TeamBuildEpisode
from .catalog import PokemonSetDef


# --- In-battle encoders ---


def encode_state_from_request(request: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a Showdown request dict into a feature dict.

    For now, we keep this as a thin wrapper around the raw request;
    future work can add structured features here.
    """
    return {"raw_request": request}


def encode_step(step: BattleStep) -> Tuple[Dict[str, Any], str]:
    """
    Encode a single BattleStep into (state_features, action_string).
    """
    state = encode_state_from_request(step.request)
    action = step.choice
    return state, action


def encode_trajectory_side(
    traj: BattleTrajectory,
    side: str = "p1",
) -> List[Tuple[Dict[str, Any], str, float, bool]]:
    """
    Encode a trajectory from a specific side's perspective ("p1" or "p2").

    Returns a list of (state, action, reward, done) tuples.
    Currently uses only the final outcome as reward and marks all steps as done=True.
    """
    if side == "p1":
        steps = traj.steps_p1
        reward = traj.reward_p1
    elif side == "p2":
        steps = traj.steps_p2
        reward = traj.reward_p2
    else:
        raise ValueError(f"Invalid side: {side!r}, expected 'p1' or 'p2'.")

    transitions: List[Tuple[Dict[str, Any], str, float, bool]] = []
    for step in steps:
        s, a = encode_step(step)
        transitions.append((s, a, reward, True))
    return transitions


def encode_trajectory(traj: BattleTrajectory) -> List[Tuple[Dict[str, Any], str, float, bool]]:
    """
    Simple BC-style encoding of a trajectory from p1's perspective.

    Returns a list of (state, action, reward, done) tuples.
    """
    return encode_trajectory_side(traj, side="p1")


def encode_trajectory_both_sides(traj: BattleTrajectory) -> List[Tuple[Dict[str, Any], str, float, bool]]:
    """
    Encode a trajectory from both sides' perspectives by concatenating p1 and p2 views.
    """
    return encode_trajectory_side(traj, "p1") + encode_trajectory_side(traj, "p2")


# --- Team-building encoders ---


def encode_team_from_set_ids(
    set_ids: List[str],
    set_dict: Dict[str, PokemonSetDef],
) -> Dict[str, Any]:
    """
    Encode a 6-mon team specified by set_ids into a simple feature dict.

    This is intentionally simple:
      - species list
      - items list
      - raw set_ids for embedding in training code
    """
    species = [set_dict[sid].species for sid in set_ids]
    items = [set_dict[sid].item for sid in set_ids]
    return {
        "set_ids": set_ids,
        "species": species,
        "items": items,
    }


def encode_team_build_episode(
    ep: TeamBuildEpisode,
    set_dict: Dict[str, PokemonSetDef],
) -> Dict[str, Any]:
    """
    Encode a TeamBuildEpisode into features + reward for downstream training.
    """
    features = encode_team_from_set_ids(ep.chosen_set_ids, set_dict)
    return {
        "features": features,
        "reward": ep.reward,
    }

