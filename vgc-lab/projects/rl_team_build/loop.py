"""Value iteration loop for team-building.

This module orchestrates multiple rounds of self-play and re-training,
implementing a basic value iteration loop that improves the team value policy
over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    import torch
except ImportError:
    torch = None  # type: ignore

from .selfplay import SelfPlayConfig, run_selfplay
from .train_value import train_value_model


@dataclass
class ValueIterationConfig:
    """Configuration for value iteration loop."""

    num_iters: int = 3
    episodes_per_iter: int = 50
    format_id: str = "gen9vgc2026regf"
    seed: int = 42
    # Preview model training options (all default to False/None to preserve existing behavior)
    train_preview_per_iter: bool = False
    preview_epochs: int = 5
    preview_use_reward_weights: bool = False
    preview_min_reward: Optional[float] = None
    preview_max_reward: Optional[float] = None
    preview_reward_alpha: float = 1.0
    preview_device: str = "cuda" if torch and torch.cuda.is_available() else "cpu"
    # Battle BC training options (all default to False to preserve existing behavior)
    train_battle_bc_per_iter: bool = False
    battle_bc_epochs: int = 5
    battle_bc_batch_size: int = 256
    battle_bc_val_frac: float = 0.2
    battle_bc_use_outcome_weights: bool = False
    battle_bc_winner_weight: float = 2.0
    battle_bc_loser_weight: float = 0.5
    battle_bc_draw_weight: float = 1.0
    # Online self-play options (Python-driven policies)
    use_python_policies_for_battle: bool = False


def run_value_iteration(config: ValueIterationConfig) -> None:
    """Run value iteration loop: self-play → retrain → (optional) preview/battle BC training → repeat.

    For each iteration:
      1. Runs self-play to generate new episodes using current value policy
      2. Retrains the value model on all available team_build episodes
      3. Optionally trains the preview model if config.train_preview_per_iter is True
      4. Optionally trains the battle BC model if config.train_battle_bc_per_iter is True
      5. Updates checkpoints for next iteration

    Args:
        config: ValueIterationConfig with num_iters, episodes_per_iter, etc.
    """
    for iter_idx in range(config.num_iters):
        print(f"\n{'='*60}")
        print(f"Iteration {iter_idx + 1}/{config.num_iters}")
        print(f"{'='*60}")

        # Run self-play to generate new episodes
        if config.use_python_policies_for_battle:
            # Use Python-driven online self-play (preview + battle policies)
            print(f"[Iter {iter_idx + 1}] Using Python-driven online self-play...")
            try:
                from projects.rl_battle.online_selfplay import OnlineSelfPlayConfig, run_online_selfplay

                online_cfg = OnlineSelfPlayConfig(
                    num_episodes=config.episodes_per_iter,
                    format_id=config.format_id,
                    p1_policy="python_external_v1",  # Use Python policies
                    p2_policy="node_random_v1",  # Keep random for now
                    seed=config.seed + iter_idx,
                    write_trajectories=True,
                )
                online_summary = run_online_selfplay(online_cfg)

                # Convert to same format as run_selfplay output
                summary = {
                    "num_episodes": online_summary["num_episodes"],
                    "p1_wins": online_summary["p1_wins"],
                    "p1_losses": online_summary["p2_wins"],
                    "p1_ties": online_summary["num_episodes"] - online_summary["p1_wins"] - online_summary["p2_wins"],
                    "avg_reward_p1": (online_summary["p1_wins"] - online_summary["p2_wins"]) / max(online_summary["num_episodes"], 1),
                }
            except ImportError as e:
                print(f"[Iter {iter_idx + 1}] Online self-play not available: {e}. Falling back to standard self-play.")
                config.use_python_policies_for_battle = False
                # Fall through to standard self-play
            except Exception as e:
                print(f"[Iter {iter_idx + 1}] Online self-play failed: {e}. Falling back to standard self-play.")
                config.use_python_policies_for_battle = False
                # Fall through to standard self-play

        if not config.use_python_policies_for_battle:
            # Use standard self-play (team-value only, Node random for preview/battle)
            selfplay_cfg = SelfPlayConfig(
                num_episodes=config.episodes_per_iter,
                format_id=config.format_id,
                p1_policy="team_value",
                p2_policy="random",
                seed=config.seed + iter_idx,
                write_team_build_episodes=True,
                policy_id_team_value=f"team_value_iter_{iter_idx+1}",
                policy_id_random="random_sets_v1",
            )
            summary = run_selfplay(selfplay_cfg)

        # Print self-play summary
        win_rate = summary["p1_wins"] / summary["num_episodes"] if summary["num_episodes"] > 0 else 0.0
        print(f"Self-play: {summary['p1_wins']}/{summary['num_episodes']} wins "
              f"(win rate: {win_rate:.2%}), avg_reward={summary['avg_reward_p1']:.2f}")

        # Retrain value model on updated dataset
        print("Retraining value model...")
        train_value_model()

        # Optionally train preview model if enabled
        if config.train_preview_per_iter:
            print(f"\n[Iter {iter_idx + 1}/{config.num_iters}] Training PreviewModel...")
            try:
                from projects.rl_preview.train_preview import TrainPreviewConfig, train_preview_model
                from projects.rl_preview.eval import EvalPreviewConfig, run_eval_preview

                # Build preview training config
                prev_train_cfg = TrainPreviewConfig(
                    format_id=config.format_id,
                    epochs=config.preview_epochs,
                    device=config.preview_device,
                    include_default_choices=False,
                    use_reward_weights=config.preview_use_reward_weights,
                    min_reward_for_training=config.preview_min_reward,
                    max_reward_for_training=config.preview_max_reward,
                    reward_weight_alpha=config.preview_reward_alpha,
                )

                # Train the preview model
                ckpt_path = train_preview_model(prev_train_cfg)

                # Run offline eval
                print(f"[Iter {iter_idx + 1}/{config.num_iters}] Preview offline eval...")
                prev_eval_cfg = EvalPreviewConfig(
                    format_id=prev_train_cfg.format_id,
                    device=prev_train_cfg.device,
                    include_default_choices=False,
                )
                eval_summary = run_eval_preview(prev_eval_cfg)
                print(
                    f"[Iter {iter_idx + 1}] Preview eval: "
                    f"num_examples={eval_summary['num_examples']}, "
                    f"accuracy={eval_summary['accuracy']:.4f} "
                    f"(random baseline ≈ 1/15 ≈ 0.0667)"
                )
            except ValueError as e:
                # Keep error message clear but don't crash the whole loop
                print(f"[Iter {iter_idx + 1}] Preview training skipped: {e}")
            except ImportError as e:
                print(f"[Iter {iter_idx + 1}] Preview training skipped (import error): {e}")

        # Optionally train battle BC model if enabled
        if config.train_battle_bc_per_iter:
            print(f"\n[ValueIter] Training battle BC for iteration {iter_idx + 1}...")
            try:
                from projects.rl_battle.train_bc import BattleBCConfig, train_battle_bc
                from projects.rl_battle.eval_bc import evaluate_battle_bc

                bc_cfg = BattleBCConfig(
                    format_id=config.format_id,
                    batch_size=config.battle_bc_batch_size,
                    lr=1e-3,
                    epochs=config.battle_bc_epochs,
                    val_frac=config.battle_bc_val_frac,
                    seed=config.seed,
                    use_outcome_weights=config.battle_bc_use_outcome_weights,
                    winner_weight=config.battle_bc_winner_weight,
                    loser_weight=config.battle_bc_loser_weight,
                    draw_weight=config.battle_bc_draw_weight,
                )

                ckpt_path = train_battle_bc(bc_cfg)
                print(f"[ValueIter] Battle BC checkpoint updated: {ckpt_path}")

                # Run offline evaluation
                acc, total = evaluate_battle_bc(batch_size=config.battle_bc_batch_size)
                print(
                    f"[ValueIter] Battle BC offline accuracy: {acc:.4f} "
                    f"on {int(total)} examples"
                )
            except ValueError as e:
                print(f"[Iter {iter_idx + 1}] Battle BC training skipped: {e}")
            except ImportError as e:
                print(f"[Iter {iter_idx + 1}] Battle BC training skipped (import error): {e}")

    print(f"\n{'='*60}")
    print("Value iteration complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    config = ValueIterationConfig()
    run_value_iteration(config)

