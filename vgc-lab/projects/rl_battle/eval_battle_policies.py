"""Evaluation harness for battle policies via online self-play."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List

from .online_selfplay import (
    OnlineSelfPlayConfig,
    OnlineSelfPlaySummary,
    PythonPolicyKind,
    run_online_selfplay,
)


@dataclass
class BattleEvalConfig:
    """Configuration for battle policy evaluation."""

    format_id: str = "gen9vgc2026regf"
    num_runs: int = 3
    episodes_per_run: int = 20
    p1_policy: str = "python_external_v1"
    p2_policy: str = "node_random_v1"
    p1_python_policy: PythonPolicyKind = "dqn"
    p2_python_policy: PythonPolicyKind = "random"
    seed: int = 0
    strict_invalid_choice: bool = True
    debug: bool = False


def run_battle_eval(cfg: BattleEvalConfig) -> Dict[str, Any]:
    """
    Run multiple online self-play evaluations and aggregate results.

    Args:
        cfg: BattleEvalConfig with evaluation parameters.

    Returns:
        Dict with aggregated results:
          - total_episodes: Total number of episodes across all runs
          - total_errors: Total number of errors across all runs
          - total_p1_wins: Total number of P1 wins across all runs
          - total_p2_wins: Total number of P2 wins across all runs
          - total_draws: Total number of draws across all runs
          - p1_win_rate: P1 win rate (p1_wins / (p1_wins + p2_wins + draws))
          - runs: List of OnlineSelfPlaySummary for each run
    """
    # Initialize accumulators
    total_episodes = 0
    total_errors = 0
    total_p1_wins = 0
    total_p2_wins = 0
    total_draws = 0
    runs: List[OnlineSelfPlaySummary] = []

    # Run multiple evaluations
    for run_idx in range(cfg.num_runs):
        # Compute run-specific seed
        run_seed = cfg.seed + run_idx

        # Construct config for this run
        cfg_run = OnlineSelfPlayConfig(
            num_episodes=cfg.episodes_per_run,
            format_id=cfg.format_id,
            p1_policy=cfg.p1_policy,
            p2_policy=cfg.p2_policy,
            p1_python_policy=cfg.p1_python_policy,
            p2_python_policy=cfg.p2_python_policy,
            seed=run_seed,
            strict_invalid_choice=cfg.strict_invalid_choice,
            debug=cfg.debug,
            write_trajectories=False,  # Disable trajectory writing during eval
        )

        # Run online self-play for this run
        summary = run_online_selfplay(cfg_run)

        # Append summary to runs list
        runs.append(summary)

        # Update accumulators
        total_episodes += summary["episodes"]
        total_errors += summary["errors"]
        total_p1_wins += summary["p1_wins"]
        total_p2_wins += summary["p2_wins"]
        total_draws += summary["draws"]

    # Compute win rate
    total_games = total_p1_wins + total_p2_wins + total_draws
    if total_games > 0:
        p1_win_rate = total_p1_wins / total_games
    else:
        p1_win_rate = 0.0

    # Return aggregated results
    return {
        "total_episodes": total_episodes,
        "total_errors": total_errors,
        "total_p1_wins": total_p1_wins,
        "total_p2_wins": total_p2_wins,
        "total_draws": total_draws,
        "p1_win_rate": p1_win_rate,
        "runs": runs,
    }
