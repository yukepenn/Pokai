"""
Dataset utilities for loading and managing battle and team-building datasets.
"""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Union

from pydantic import BaseModel, Field

from .core import Paths, get_paths, BattleStep, BattleTrajectory


@dataclass
class TurnView:
    """A turn-based view of a trajectory with separate step lists for p1 and p2."""

    turn: int
    p1_steps: List[BattleStep]
    p2_steps: List[BattleStep]


class TeamBuildStep(BaseModel):
    """
    A single step in the team-building process.

    Right now we keep it simple: just the chosen set_id at each index.
    Later we can add available_set_ids and action masks if needed.
    """

    step_index: int = Field(..., description="0..5, index of the Pokemon being chosen")
    available_set_ids: Optional[List[str]] = Field(
        None,
        description="Optional list of available set_ids at this step (for RL masking).",
    )
    chosen_set_id: str = Field(..., description="The chosen set_id at this step.")


class TeamBuildEpisode(BaseModel):
    """
    A high-level episode for team-building, separate from in-battle trajectories.

    One episode represents building a single 6-mon team for a given side (p1 or p2),
    along with the final reward from that side's perspective.
    """

    episode_id: str = Field(..., description="Globally unique identifier for this episode.")
    side: str = Field(..., description="'p1' or 'p2'.")
    format_id: str = Field(..., description="Battle format id, e.g. 'gen9vgc2026regf'.")
    policy_id: str = Field(
        ...,
        description="Identifier for the team-building policy, e.g. 'random_sets_v1'.",
    )
    chosen_set_ids: List[str] = Field(
        ...,
        description="List of 6 set_ids used to build the team (order is preserved).",
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the episode was created.",
    )
    reward: float = Field(
        ...,
        description="Final reward from this side's perspective (e.g. +1 / 0 / -1).",
    )
    battle_ids: List[str] = Field(
        default_factory=list,
        description="List of battle_ids that this team participated in.",
    )
    steps: List[TeamBuildStep] = Field(
        default_factory=list,
        description="Detailed team-building sequence (currently simple).",
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata for analysis.",
    )

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat(),
        }


def append_team_build_episode(
    episode: TeamBuildEpisode,
    paths: Optional[Paths] = None,
) -> None:
    """
    Append a TeamBuildEpisode as one line of JSON to data/datasets/team_build/episodes.jsonl.
    """
    if paths is None:
        paths = get_paths()
    from .core import ensure_paths

    ensure_paths(paths)
    json_path = paths.team_build_jsonl
    json_line = episode.model_dump_json()
    with json_path.open("a", encoding="utf-8") as f:
        f.write(json_line + "\n")


def iter_team_build_episodes(
    paths: Optional[Paths] = None,
) -> Iterator[TeamBuildEpisode]:
    """
    Iterate over all TeamBuildEpisode entries from the JSONL file.
    """
    if paths is None:
        paths = get_paths()
    json_path = paths.team_build_jsonl
    if not json_path.exists():
        return
    with json_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield TeamBuildEpisode.model_validate_json(line)


def iter_trajectories(paths: Optional[Paths] = None) -> Iterator[BattleTrajectory]:
    """
    Iterate over all BattleTrajectory rows from trajectories.jsonl.
    """
    if paths is None:
        paths = get_paths()
    json_path = paths.trajectories_jsonl
    if not json_path.exists():
        return
    with json_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield BattleTrajectory.model_validate_json(line)


def group_steps_by_turn(
    traj: BattleTrajectory,
    side: Optional[str] = None,
) -> Dict[int, List[BattleStep]]:
    """
    Utility to group BattleStep objects by turn number (for turn-based views).

    Args:
        traj: BattleTrajectory to process
        side: If "p1", only use traj.steps_p1. If "p2", only use traj.steps_p2.
              If None, use both steps_p1 and steps_p2 concatenated (backward compatible).

    Returns:
        Dict mapping turn number -> list of BattleStep objects for that turn.
    """
    if side == "p1":
        steps = list(traj.steps_p1)
    elif side == "p2":
        steps = list(traj.steps_p2)
    else:
        steps = list(traj.steps_p1) + list(traj.steps_p2)

    result: Dict[int, List[BattleStep]] = {}
    for step in steps:
        if step.turn is None:
            continue
        result.setdefault(step.turn, []).append(step)
    return result


def summarize_sanitize_reasons(dataset_root: Union[str, Path]) -> Dict[str, int]:
    """
    Summarize sanitize_reason distribution across all battle trajectories.

    Args:
        dataset_root: Path to trajectories.jsonl file or directory containing it.
            If a directory, looks for trajectories/trajectories.jsonl inside it.

    Returns:
        Dictionary mapping reason -> count, including "none" for steps with None sanitize_reason.
    """
    dataset_path = Path(dataset_root)
    if dataset_path.is_dir():
        # If directory, assume standard structure
        paths = Paths(
            data_root=dataset_path,
            battles_raw=dataset_path / "battles_raw",
            battles_json=dataset_path / "battles_json",
            datasets_root=dataset_path / "datasets",
            full_battles=dataset_path / "datasets" / "full_battles",
            team_preview=dataset_path / "datasets" / "team_preview",
            trajectories=dataset_path / "datasets" / "trajectories",
            team_build=dataset_path / "datasets" / "team_build",
        )
        jsonl_path = paths.trajectories_jsonl
    else:
        # Assume it's a direct path to the JSONL file
        jsonl_path = dataset_path

    if not jsonl_path.exists():
        raise FileNotFoundError(f"Trajectories file not found: {jsonl_path}")

    counts: Counter[str] = Counter()

    # Iterate over all trajectories and count sanitize_reason values
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                traj = BattleTrajectory(**data)

                # Count reasons from p1 and p2 steps
                for step in traj.steps_p1:
                    reason = step.sanitize_reason
                    counts["none" if reason is None else reason] += 1

                for step in traj.steps_p2:
                    reason = step.sanitize_reason
                    counts["none" if reason is None else reason] += 1

            except Exception as e:  # noqa: BLE001
                # Skip malformed lines
                continue

    return dict(counts)


def build_turn_views(traj: BattleTrajectory) -> List[TurnView]:
    """
    Build a turn-based view of a trajectory, with separate step lists for p1 and p2.

    Returns a list of TurnView objects, one per turn, sorted by turn number.
    """
    by_turn_p1: Dict[int, List[BattleStep]] = {}
    by_turn_p2: Dict[int, List[BattleStep]] = {}

    for step in traj.steps_p1:
        if step.turn is None:
            continue
        by_turn_p1.setdefault(step.turn, []).append(step)

    for step in traj.steps_p2:
        if step.turn is None:
            continue
        by_turn_p2.setdefault(step.turn, []).append(step)

    turns = sorted(set(by_turn_p1.keys()) | set(by_turn_p2.keys()))
    return [
        TurnView(
            turn=t,
            p1_steps=by_turn_p1.get(t, []),
            p2_steps=by_turn_p2.get(t, []),
        )
        for t in turns
    ]

