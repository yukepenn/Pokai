"""Evaluation script for team-building policies.

This module provides a dedicated evaluation harness that runs self-play
between policies WITHOUT writing new TeamBuildEpisode rows to the dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from .selfplay import SelfPlayConfig, run_selfplay


@dataclass
class EvalConfig:
    """Configuration for team-building policy evaluation."""

    num_episodes: int = 100
    format_id: str = "gen9vgc2026regf"
    p1_policy: str = "team_value"  # "team_value" or "random"
    p2_policy: str = "random"  # "team_value" or "random"
    seed: Optional[int] = None


def run_eval(config: EvalConfig) -> Dict[str, Any]:
    """Run evaluation without writing new episodes.

    Executes self-play battles between policies and returns statistics,
    but does NOT append new TeamBuildEpisode rows to the dataset.

    Args:
        config: EvalConfig with num_episodes, policies, format_id, etc.

    Returns:
        Summary dict from run_selfplay with keys: num_episodes, p1_wins,
        p1_losses, p1_ties, avg_reward_p1
    """
    # Build SelfPlayConfig with write_team_build_episodes=False
    selfplay_cfg = SelfPlayConfig(
        num_episodes=config.num_episodes,
        format_id=config.format_id,
        p1_policy=config.p1_policy,
        p2_policy=config.p2_policy,
        seed=config.seed if config.seed is not None else 42,
        write_team_build_episodes=False,  # IMPORTANT: eval does not write episodes
        policy_id_team_value="team_value_policy_v1",
        policy_id_random="random_sets_v1",
    )

    summary = run_selfplay(selfplay_cfg)

    # Print concise summary
    win_rate = summary["p1_wins"] / summary["num_episodes"] if summary["num_episodes"] > 0 else 0.0
    print(
        f"Eval: p1={config.p1_policy} vs p2={config.p2_policy}, "
        f"episodes={summary['num_episodes']}, "
        f"winrate={win_rate:.2%}, "
        f"avg_reward={summary['avg_reward_p1']:.2f}"
    )

    return summary


if __name__ == "__main__":
    cfg = EvalConfig()
    summary = run_eval(cfg)
    print("Eval summary:", summary)

