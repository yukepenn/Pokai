#!/usr/bin/env python3
"""
Unified CLI for vgc-lab.

Replaces all individual scripts with a single Typer-based CLI.
"""

import shutil
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import torch

import typer

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab import (
    ShowdownClient,
    BattleStore,
    get_paths,
    load_sets,
    sample_team_sets_random,
    build_packed_team_from_set_ids,
    TeamBuildEpisode,
    TeamBuildStep,
    append_team_build_episode,
    parse_team_preview_snapshot,
)

# Import preview dataset for stats command
try:
    from projects.rl_preview.dataset import PreviewDataset
except ImportError:
    PreviewDataset = None  # type: ignore

try:
    from projects.rl_battle.dataset import BattleStepDataset, BattleStepDatasetConfig
except ImportError:
    BattleStepDataset = None  # type: ignore
    BattleStepDatasetConfig = None  # type: ignore

try:
    from projects.rl_battle.rl_dataset import BattleTransitionDataset, RlBattleDatasetConfig
except ImportError:
    BattleTransitionDataset = None  # type: ignore
    RlBattleDatasetConfig = None  # type: ignore

try:
    from projects.rl_battle.train_dqn import BattleDqnConfig, train_battle_dqn
except ImportError:
    BattleDqnConfig = None  # type: ignore
    train_battle_dqn = None  # type: ignore

app = typer.Typer(help="Unified CLI for vgc-lab")


@app.command("demo-battle")
def demo_battle(
    format_id: str = typer.Option("gen9vgc2026regf", "--format-id", help="Format ID"),
):
    """
    Run a single random selfplay battle and log the result.
    Replaces: run_demo_battle.py
    """
    paths = get_paths()
    client = ShowdownClient(paths, format_id=format_id)
    store = BattleStore(paths)

    try:
        battle = client.run_random_selfplay_json(format_id=format_id)
        record = store.append_full_battle_from_json(battle)

        # If the JS output includes a request-level trajectory, save it as well.
        if "trajectory" in battle:
            store.append_trajectory_from_battle(record, battle)

        typer.echo("=" * 60)
        typer.echo("BATTLE SUMMARY")
        typer.echo("=" * 60)
        typer.echo(f"Format: {record.format_id}")
        typer.echo(f"Winner: {record.winner_name} ({record.winner_side})")
        typer.echo(f"Turns: {record.turns if record.turns is not None else 'unknown'}")
        result_json_path = paths.battles_json / f"{record.battle_id}.json"
        typer.echo(f"Raw log: {record.raw_log_path}")
        typer.echo(f"Result JSON: {result_json_path}")
        typer.echo("=" * 60)
    except RuntimeError as e:
        error_msg = str(e)
        if "Node.js not found" in error_msg or "node" in error_msg.lower():
            typer.echo(
                "ERROR: Node.js is not installed or not in PATH.\n"
                "Please install Node.js 16+ from https://nodejs.org/\n"
                "Then ensure 'node' and 'npm' are available in your PATH.",
                err=True,
            )
        elif "Showdown directory not found" in error_msg:
            typer.echo(
                f"ERROR: {error_msg}\n"
                "Ensure pokemon-showdown/ is a sibling directory of vgc-lab/.",
                err=True,
            )
        else:
            typer.echo(f"ERROR: {error_msg}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        import traceback

        traceback.print_exc()
        raise typer.Exit(1)


@app.command("gen-full")
def gen_full(
    n: int = typer.Argument(100, help="Number of battles to generate"),
    format_id: str = typer.Option("gen9vgc2026regf", "--format-id", help="Format ID"),
):
    """
    Generate a full_battles JSONL dataset using random selfplay.
    Replaces: generate_full_battle_dataset.py
    """
    paths = get_paths()
    client = ShowdownClient(paths, format_id=format_id)
    store = BattleStore(paths)

    successful = 0
    failed = 0

    typer.echo(f"Generating {n} full battle records...")
    typer.echo(f"Format: {format_id}")
    typer.echo()

    for i in range(1, n + 1):
        try:
            battle = client.run_random_selfplay_json(format_id=format_id)
            record = store.append_full_battle_from_json(battle)

            # If the JS output includes a request-level trajectory, save it as well.
            if "trajectory" in battle:
                store.append_trajectory_from_battle(record, battle)

            successful += 1

            typer.echo(f"[{i}/{n}] winner={record.winner_side}, turns={record.turns}")

            # Small delay to avoid OneDrive race conditions
            if i < n:
                time.sleep(0.1)

        except RuntimeError as e:
            failed += 1
            typer.echo(f"  Error [{i}]: {e}", err=True)
        except Exception as e:
            failed += 1
            typer.echo(f"  Unexpected error [{i}]: {e}", err=True)

    # Print summary
    typer.echo()
    typer.echo("=" * 60)
    typer.echo("FULL BATTLE DATASET GENERATION SUMMARY")
    typer.echo("=" * 60)
    typer.echo(f"Total attempts: {n}")
    typer.echo(f"Successful: {successful}")
    typer.echo(f"Failed: {failed}")
    typer.echo()
    typer.echo(f"Dataset file: {store.paths.full_battles_jsonl}")
    if store.paths.full_battles_jsonl.exists():
        line_count = sum(1 for _ in store.paths.full_battles_jsonl.open("r", encoding="utf-8"))
        typer.echo(f"Total lines in dataset: {line_count}")
    typer.echo("=" * 60)


@app.command("gen-preview")
def gen_preview(
    n: int = typer.Argument(100, help="Number of snapshots to generate"),
    format_id: str = typer.Option("gen9vgc2026regf", "--format-id", help="Format ID"),
):
    """
    Generate team preview snapshots from random battles.
    Replaces: generate_team_preview_dataset.py
    """
    from datetime import datetime, timezone

    paths = get_paths()
    client = ShowdownClient(paths, format_id=format_id)
    store = BattleStore(paths)

    successful = 0
    failed = 0
    skipped = 0

    typer.echo(f"Generating {n} team preview snapshots...")
    typer.echo(f"Format: {format_id}")
    typer.echo()

    for i in range(1, n + 1):
        try:
            # Run battle via JSON
            battle = client.run_random_selfplay_json(format_id=format_id)
            log_text = battle["log"]

            # Save raw log first and derive battle_id
            battle_id, raw_path = store.save_raw_log(
                log_text=log_text,
                format_id=battle["format_id"],
            )

            # Parse team preview snapshot from JSON
            snapshot = parse_team_preview_snapshot(
                battle_json=battle,
                battle_id=battle_id,
                raw_log_path=raw_path,
            )

            store.append_preview_snapshot(snapshot)
            successful += 1

            if i % 10 == 0 or i == n:
                typer.echo(
                    f"  Progress: {i}/{n} (successful: {successful}, failed: {failed})"
                )

        except RuntimeError as e:
            failed += 1
            typer.echo(f"  Error [{i}]: {e}", err=True)
        except Exception as e:
            failed += 1
            typer.echo(f"  Unexpected error [{i}]: {e}", err=True)

    # Print summary
    typer.echo()
    typer.echo("=" * 60)
    typer.echo("DATASET GENERATION SUMMARY")
    typer.echo("=" * 60)
    typer.echo(f"Total attempts: {n}")
    typer.echo(f"Successful snapshots: {successful}")
    typer.echo(f"Failed: {failed}")
    typer.echo()
    typer.echo(f"Dataset file: {store.paths.team_preview_jsonl}")
    if store.paths.team_preview_jsonl.exists():
        line_count = sum(
            1 for _ in store.paths.team_preview_jsonl.open("r", encoding="utf-8")
        )
        typer.echo(f"Total lines in dataset: {line_count}")
    typer.echo("=" * 60)


@app.command("pack-team")
def pack_team(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Input file (or read from stdin)"),
):
    """
    Read an export-format team file and print the packed format.
    Replaces: pack_team_cli.py
    """
    paths = get_paths()
    client = ShowdownClient(paths)

    try:
        if file:
            if not file.exists():
                typer.echo(f"Error: File not found: {file}", err=True)
                raise typer.Exit(1)
            input_text = file.read_text(encoding="utf-8")
        else:
            input_text = sys.stdin.read()

        if not input_text.strip():
            typer.echo("Error: No input provided", err=True)
            raise typer.Exit(1)

        packed = client.pack_team(input_text)
        typer.echo(packed)

    except RuntimeError as e:
        error_msg = str(e)
        if "Node.js not found" in error_msg:
            typer.echo(
                "Error: Node.js not found. Please install Node.js 16+ from https://nodejs.org/",
                err=True,
            )
        else:
            typer.echo(f"Error: {error_msg}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


@app.command("validate-team")
def validate_team(
    file: Optional[Path] = typer.Option(None, "--file", "-f", help="Input file (or read from stdin)"),
    format_id: str = typer.Option(
        "gen9vgc2026regf", "--format-id", "-F", help="Format ID to validate against"
    ),
):
    """
    Validate a team (export or packed) against a format using Showdown.
    Replaces: validate_team_cli.py
    """
    paths = get_paths()
    client = ShowdownClient(paths)

    try:
        if file:
            if not file.exists():
                typer.echo(f"Error: File not found: {file}", err=True)
                raise typer.Exit(1)
            input_text = file.read_text(encoding="utf-8")
        else:
            input_text = sys.stdin.read()

        if not input_text.strip():
            typer.echo("Error: No input provided", err=True)
            raise typer.Exit(1)

        # Detect if input is export format
        looks_like_export = "@" in input_text or "EVs:" in input_text or "Ability:" in input_text

        if looks_like_export:
            typer.echo("Detected export format, converting to packed...", err=True)
            packed_team = client.pack_team(input_text)
        else:
            packed_team = input_text.strip()

        # Validate
        validation_output = client.validate_team(packed_team, format_id=format_id)

        if validation_output:
            typer.echo("Validation FAILED:", err=True)
            typer.echo(validation_output, err=True)
            raise typer.Exit(1)
        else:
            typer.echo(f"Team is valid for format: {format_id}")
            raise typer.Exit(0)

    except RuntimeError as e:
        error_msg = str(e)
        if "Node.js not found" in error_msg:
            typer.echo(
                "Error: Node.js not found. Please install Node.js 16+ from https://nodejs.org/",
                err=True,
            )
        else:
            typer.echo(f"Error: {error_msg}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


@app.command("clean-data")
def clean_data(
    confirm: bool = typer.Option(
        False,
        "--confirm",
        help="Actually delete all logs and dataset files under data/",
    ),
):
    """
    Delete all battle logs and dataset JSONL files to reset the data directory.

    This removes files under:
    - data/battles_raw/
    - data/battles_json/
    - data/datasets/full_battles/
    - data/datasets/team_preview/
    - data/datasets/trajectories/
    - data/datasets/team_build/

    Directories themselves are kept.
    """
    if not confirm:
        typer.echo("This will delete ALL logs and dataset files under data/.")
        typer.echo("Re-run with --confirm to perform deletion.")
        raise typer.Exit(0)

    paths = get_paths()

    targets = [
        paths.battles_raw,
        paths.battles_json,
        paths.full_battles,
        paths.team_preview,
        paths.trajectories,
        paths.team_build,
    ]

    for directory in targets:
        if not directory.exists():
            continue
        for p in directory.iterdir():
            if p.is_file():
                p.unlink()
            elif p.is_dir():
                shutil.rmtree(p)

    typer.echo("All logs and dataset files deleted. Directories remain.")


@app.command("preview-dataset-stats")
def preview_dataset_stats(
    format_id: str = typer.Option("gen9vgc2026regf", "--format-id", help="Format ID"),
    include_default: bool = typer.Option(
        False, "--include-default", help="Include 'default' team-preview choices"
    ),
):
    """
    Print statistics about the PreviewDataset.

    Shows number of examples, action distribution, and filtering statistics.
    Useful for debugging whether the dataset contains diverse bring-4 actions.
    """
    if PreviewDataset is None:
        typer.echo("Error: PreviewDataset not available. Is projects.rl_preview installed?")
        raise typer.Exit(1)

    ds = PreviewDataset(
        format_id=format_id,
        include_default_choices=include_default,
    )

    typer.echo(f"Preview Dataset Statistics (format_id={format_id})")
    typer.echo("=" * 60)
    typer.echo(f"Number of examples: {ds.num_examples}")
    typer.echo(f"Number of sets: {ds.num_sets}")
    typer.echo(f"Default choices skipped: {ds.num_default_skipped}")
    typer.echo(f"Parse failures: {ds.num_parse_failures}")
    typer.echo("\nAction distribution (action_index -> count):")
    for action_idx in sorted(ds.action_counts.keys()):
        count = ds.action_counts[action_idx]
        typer.echo(f"  Action {action_idx:2d}: {count:4d} examples")


@app.command("train-battle-bc")
def train_battle_bc_cmd(
    format_id: str = typer.Option("gen9vgc2026regf", "--format-id", help="Format ID"),
    batch_size: int = typer.Option(256, "--batch-size", help="Batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    epochs: int = typer.Option(5, "--epochs", help="Number of epochs"),
    val_frac: float = typer.Option(0.2, "--val-frac", help="Validation fraction"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    use_outcome_weights: bool = typer.Option(
        False, "--use-outcome-weights", help="Use outcome-aware sample weights"
    ),
    winner_weight: float = typer.Option(2.0, "--winner-weight", help="Weight for winner steps"),
    loser_weight: float = typer.Option(0.5, "--loser-weight", help="Weight for loser steps"),
    draw_weight: float = typer.Option(1.0, "--draw-weight", help="Weight for draw steps"),
    allowed_sanitize_reasons: Optional[str] = typer.Option(
        None,
        "--allowed-sanitize-reasons",
        help="Comma-separated list of allowed sanitize_reason values (e.g., 'ok,fixed_pass'). "
        "If not provided, defaults to ['ok', 'fixed_pass'].",
    ),
):
    """
    Train a battle BC policy using behavior cloning on battle trajectories.

    Steps with sanitize_reason not in the allowed list are filtered out during training.
    """
    from projects.rl_battle.train_bc import BattleBCConfig, train_battle_bc
    from vgc_lab.core import SanitizeReason

    # Parse allowed_sanitize_reasons
    train_allowed_sanitize_reasons: Optional[list[SanitizeReason]] = None
    if allowed_sanitize_reasons:
        reasons_list = [r.strip() for r in allowed_sanitize_reasons.split(",")]
        # Validate that all are valid SanitizeReason values
        valid_reasons = {"ok", "fixed_pass", "fixed_disabled_move", "fixed_switch_to_move", "fallback_switch", "fallback_pass"}
        invalid = [r for r in reasons_list if r not in valid_reasons]
        if invalid:
            typer.echo(f"Error: Invalid sanitize_reason values: {invalid}", err=True)
            typer.echo(f"Valid values: {sorted(valid_reasons)}", err=True)
            raise typer.Exit(1)
        train_allowed_sanitize_reasons = reasons_list  # type: ignore

    cfg = BattleBCConfig(
        format_id=format_id,
        batch_size=batch_size,
        lr=lr,
        epochs=epochs,
        val_frac=val_frac,
        seed=seed,
        device="cuda" if torch.cuda.is_available() else "cpu",
        use_outcome_weights=use_outcome_weights,
        winner_weight=winner_weight,
        loser_weight=loser_weight,
        draw_weight=draw_weight,
        train_allowed_sanitize_reasons=train_allowed_sanitize_reasons,
    )

    typer.echo(f"Training battle BC policy...")
    typer.echo(f"  format_id: {format_id}")
    typer.echo(f"  allowed_sanitize_reasons: {cfg.train_allowed_sanitize_reasons}")
    ckpt_path = train_battle_bc(cfg)
    typer.echo(f"Checkpoint saved to: {ckpt_path}")


@app.command("battle-bc-stats")
def battle_bc_stats(
    format_id: str = typer.Option("gen9vgc2026regf", "--format-id", help="Format ID"),
    top_k: int = typer.Option(20, "--top-k", help="Number of most frequent actions to show"),
    allowed_sanitize_reasons: Optional[str] = typer.Option(
        None,
        "--allowed-sanitize-reasons",
        help="Comma-separated list of allowed sanitize_reason values for filtering stats",
    ),
):
    """
    Print statistics about the BattleStepDataset used for Battle BC.

    Shows number of examples, vocab size, invalid-choice filtering stats,
    and the top-K most frequent joint action strings.
    """
    if BattleStepDataset is None or BattleStepDatasetConfig is None:
        typer.echo("Error: BattleStepDataset not available. Is projects.rl_battle installed?")
        raise typer.Exit(1)

    from vgc_lab.core import SanitizeReason

    # Parse allowed_sanitize_reasons
    train_allowed_sanitize_reasons: Optional[list[SanitizeReason]] = None
    if allowed_sanitize_reasons:
        reasons_list = [r.strip() for r in allowed_sanitize_reasons.split(",")]
        train_allowed_sanitize_reasons = reasons_list  # type: ignore

    ds_cfg = BattleStepDatasetConfig(
        format_id=format_id, train_allowed_sanitize_reasons=train_allowed_sanitize_reasons
    )
    ds = BattleStepDataset(ds_cfg)

    typer.echo(f"Battle BC Dataset Statistics (format_id={format_id})")
    typer.echo("=" * 60)
    typer.echo(f"Number of examples: {ds.num_examples}")
    typer.echo(f"Number of features: {ds.num_features}")
    typer.echo(f"Vocab size (num_actions): {ds.num_actions}")
    if hasattr(ds, "num_total_steps"):
        typer.echo(f"Total steps seen: {ds.num_total_steps}")
    if hasattr(ds, "num_invalid_choices"):
        typer.echo(f"Invalid choices skipped: {ds.num_invalid_choices}")
    if hasattr(ds, "policy_counts"):
        typer.echo("\nExamples per battle_policy_id:")
        for pid, count in sorted(ds.policy_counts.items(), key=lambda kv: kv[0]):
            typer.echo(f"  {pid!r}: {count}")
    typer.echo("\nAction distribution (id -> count), top-K:")

    from collections import Counter

    counter = Counter(ds.action_counts)
    for idx, (action_id, count) in enumerate(
        sorted(counter.items(), key=lambda kv: kv[1], reverse=True)[:top_k],
        start=1,
    ):
        choice_str = ds.id_to_choice[action_id] if 0 <= action_id < len(ds.id_to_choice) else "<unknown>"
        typer.echo(f"  #{idx:2d} id={action_id:3d} count={count:5d} choice={choice_str!r}")


@app.command("battle-rl-dataset-stats")
def battle_rl_dataset_stats(
    format_id: str = typer.Option("gen9vgc2026regf", "--format-id", help="Battle format id"),
    max_trajectories: Optional[int] = typer.Option(None, "--max-trajectories", help="Max trajectories to load"),
) -> None:
    """
    Show basic statistics for the offline RL battle transition dataset.
    """
    if BattleTransitionDataset is None or RlBattleDatasetConfig is None:
        typer.echo("Error: BattleTransitionDataset not available. Is projects.rl_battle installed?")
        raise typer.Exit(1)

    cfg = RlBattleDatasetConfig(format_id=format_id, max_trajectories=max_trajectories)
    dataset = BattleTransitionDataset(cfg)

    num_transitions = len(dataset)
    num_episodes = len({t.battle_id for t in dataset._transitions})
    avg_reward = sum(t.reward for t in dataset._transitions) / max(1, num_transitions)
    done_fraction = sum(t.done for t in dataset._transitions) / max(1, num_transitions)

    typer.echo("Battle RL dataset stats:")
    typer.echo(f"  Format:          {format_id}")
    typer.echo(f"  Episodes:        {num_episodes}")
    typer.echo(f"  Transitions:     {num_transitions}")
    typer.echo(f"  Avg reward:      {avg_reward:.4f}")
    typer.echo(f"  Done fraction:   {done_fraction:.4f}")


@app.command("train-battle-dqn")
def train_battle_dqn_cmd(
    format_id: str = typer.Option("gen9vgc2026regf", "--format-id", help="Battle format id"),
    vec_dim: int = typer.Option(256, "--vec-dim", help="State vector dimension"),
    num_actions: int = typer.Option(4, "--num-actions", help="Number of discrete actions"),
    max_trajectories: Optional[int] = typer.Option(
        None,
        "--max-trajectories",
        help="Max trajectories to load for RL training",
    ),
    epochs: int = typer.Option(3, "--epochs", help="Number of training epochs"),
    steps_per_epoch: int = typer.Option(100, "--steps-per-epoch", help="Optimization steps per epoch"),
    batch_size: int = typer.Option(64, "--batch-size", help="Mini-batch size"),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate"),
    device: str = typer.Option("cpu", "--device", help="Torch device"),
) -> None:
    """
    Train a simple offline DQN on battle trajectories.
    """
    if BattleDqnConfig is None or train_battle_dqn is None:
        typer.echo("Error: BattleDqnConfig not available. Is projects.rl_battle installed?")
        raise typer.Exit(1)

    cfg = BattleDqnConfig(
        format_id=format_id,
        vec_dim=vec_dim,
        num_actions=num_actions,
        max_trajectories=max_trajectories,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        batch_size=batch_size,
        lr=lr,
        device=device,
    )
    ckpt_path = train_battle_dqn(cfg)
    typer.echo(f"Saved DQN checkpoint to: {ckpt_path}")


@app.command("gen-battles-from-sets")
def gen_battles_from_sets(
    n: int = typer.Argument(100, help="Number of battles to generate"),
    format_id: str = typer.Option("gen9vgc2026regf", "--format-id"),
    policy_id: str = typer.Option("random_sets_v1", "--policy-id"),
):
    """
    Generate battles where each side's 6-mon team is built from sets_regf.yaml.

    For each battle:
      - p1 and p2 each build a team of 6 sets (no duplicate species or item).
      - Battle is played via random selfplay in Node (LoggingRandomPlayerAI).
      - We store:
          * FullBattleRecord + BattleTrajectory
          * Two TeamBuildEpisode rows (one for each side)
    """
    paths = get_paths()
    client = ShowdownClient(paths, format_id=format_id)
    store = BattleStore(paths)

    # Load all sets once
    sets = load_sets()

    successful = 0
    failed = 0

    typer.echo(f"Generating {n} battles from sets_regf.yaml (format={format_id})...")

    for i in range(1, n + 1):
        try:
            # 1) Sample teams
            p1_set_ids = sample_team_sets_random(sets, format_id=format_id)
            p2_set_ids = sample_team_sets_random(sets, format_id=format_id)

            p1_steps = [
                TeamBuildStep(step_index=j, chosen_set_id=sid)
                for j, sid in enumerate(p1_set_ids)
            ]
            p2_steps = [
                TeamBuildStep(step_index=j, chosen_set_id=sid)
                for j, sid in enumerate(p2_set_ids)
            ]

            # 2) Build packed teams
            p1_packed = build_packed_team_from_set_ids(p1_set_ids, sets, client)
            p2_packed = build_packed_team_from_set_ids(p2_set_ids, sets, client)

            # 3) Run battle
            battle_json = client.run_random_selfplay_json(
                format_id=format_id,
                p1_name="TeamBuilderP1",
                p2_name="TeamBuilderP2",
                p1_packed_team=p1_packed,
                p2_packed_team=p2_packed,
            )

            # Optionally store the set_ids inside the battle_json meta
            battle_json["p1_set_ids"] = p1_set_ids
            battle_json["p2_set_ids"] = p2_set_ids

            record = store.append_full_battle_from_json(battle_json)
            store.append_trajectory_from_battle(record, battle_json)

            # 4) Compute rewards for team-building episodes
            win_side = record.winner_side
            if win_side == "p1":
                reward_p1, reward_p2 = 1.0, -1.0
            elif win_side == "p2":
                reward_p1, reward_p2 = -1.0, 1.0
            else:
                reward_p1 = reward_p2 = 0.0

            # 5) Create and append TeamBuildEpisode for each side
            ep_id_p1 = f"team_ep_{uuid.uuid4().hex}"
            ep_id_p2 = f"team_ep_{uuid.uuid4().hex}"

            ep_p1 = TeamBuildEpisode(
                episode_id=ep_id_p1,
                side="p1",
                format_id=format_id,
                policy_id=policy_id,
                chosen_set_ids=p1_set_ids,
                reward=reward_p1,
                battle_ids=[record.battle_id],
                steps=p1_steps,
                meta={},
            )
            ep_p2 = TeamBuildEpisode(
                episode_id=ep_id_p2,
                side="p2",
                format_id=format_id,
                policy_id=policy_id,
                chosen_set_ids=p2_set_ids,
                reward=reward_p2,
                battle_ids=[record.battle_id],
                steps=p2_steps,
                meta={},
            )

            append_team_build_episode(ep_p1, paths)
            append_team_build_episode(ep_p2, paths)

            successful += 1
            typer.echo(f"[{i}/{n}] battle={record.battle_id} winner={record.winner_side}")

        except Exception as e:
            failed += 1
            typer.echo(f"  Error [{i}]: {e}", err=True)

    typer.echo()
    typer.echo("Generation summary:")
    typer.echo(f"  Successful: {successful}")
    typer.echo(f"  Failed: {failed}")
    typer.echo(f"  Team-build dataset: {paths.team_build_jsonl}")


@app.command("value-iter")
def value_iter(
    num_iters: int = typer.Option(3, "--num-iters", help="Number of value iteration iterations"),
    episodes_per_iter: int = typer.Option(
        50, "--episodes-per-iter", help="Number of episodes to generate per iteration"
    ),
    format_id: str = typer.Option("gen9vgc2026regf", "--format-id", help="Format ID"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
    train_preview_per_iter: bool = typer.Option(
        False, "--train-preview-per-iter", help="Train PreviewModel each iteration"
    ),
    preview_epochs: int = typer.Option(
        5, "--preview-epochs", help="PreviewModel epochs per iteration"
    ),
    preview_use_reward_weights: bool = typer.Option(
        False, "--preview-use-reward-weights", help="Enable reward-weighted BC for preview"
    ),
    preview_min_reward: Optional[float] = typer.Option(
        None, "--preview-min-reward", help="Optional min reward filter for preview data"
    ),
    preview_reward_alpha: float = typer.Option(
        1.0, "--preview-reward-alpha", help="Reward weight alpha for preview"
    ),
    train_battle_bc_per_iter: bool = typer.Option(
        False, "--train-battle-bc-per-iter", help="If set, train battle BC once per value-iteration step"
    ),
    battle_bc_epochs: int = typer.Option(
        5, "--battle-bc-epochs", help="Number of epochs for battle BC training per iteration"
    ),
    battle_bc_batch_size: int = typer.Option(
        256, "--battle-bc-batch-size", help="Batch size for battle BC training/eval"
    ),
    battle_bc_use_outcome_weights: bool = typer.Option(
        False,
        "--battle-bc-use-outcome-weights",
        help="If set, use outcome-aware sample weights for battle BC (winner/loser/draw).",
    ),
    battle_bc_winner_weight: float = typer.Option(
        2.0,
        "--battle-bc-winner-weight",
        help="Sample weight for winner-side steps when outcome-weighted BC is enabled.",
    ),
    battle_bc_loser_weight: float = typer.Option(
        0.5,
        "--battle-bc-loser-weight",
        help="Sample weight for loser-side steps when outcome-weighted BC is enabled.",
    ),
    battle_bc_draw_weight: float = typer.Option(
        1.0,
        "--battle-bc-draw-weight",
        help="Sample weight for draw/unknown steps when outcome-weighted BC is enabled.",
    ),
    use_python_policies_for_battle: bool = typer.Option(
        False,
        "--use-python-policies-for-battle",
        help="If set, use Python-driven online self-play (PreviewPolicy + BattleBCPolicy) "
             "instead of Node random policies for preview/battle actions.",
    ),
):
    """
    Run value iteration loop: self-play → retrain value model → (optional) train preview/battle BC → repeat.

    This orchestrates iterative improvement of the team value policy, optionally co-training
    the preview/bring-4 model and battle BC model alongside the value model.
    
    Battle BC can be outcome-weighted (winner/loser/draw sample weights) when enabled.
    """
    from projects.rl_team_build.loop import ValueIterationConfig, run_value_iteration

    cfg = ValueIterationConfig(
        num_iters=num_iters,
        episodes_per_iter=episodes_per_iter,
        format_id=format_id,
        seed=seed,
        train_preview_per_iter=train_preview_per_iter,
        preview_epochs=preview_epochs,
        preview_use_reward_weights=preview_use_reward_weights,
        preview_min_reward=preview_min_reward,
        preview_reward_alpha=preview_reward_alpha,
        train_battle_bc_per_iter=train_battle_bc_per_iter,
        battle_bc_epochs=battle_bc_epochs,
        battle_bc_batch_size=battle_bc_batch_size,
        battle_bc_use_outcome_weights=battle_bc_use_outcome_weights,
        battle_bc_winner_weight=battle_bc_winner_weight,
        battle_bc_loser_weight=battle_bc_loser_weight,
        battle_bc_draw_weight=battle_bc_draw_weight,
        use_python_policies_for_battle=use_python_policies_for_battle,
    )
    run_value_iteration(cfg)


@app.command("online-selfplay")
def online_selfplay_cmd(
    num_episodes: int = typer.Option(3, "--num-episodes", help="Number of battles to run"),
    format_id: str = typer.Option("gen9vgc2026regf", "--format-id", help="Format ID"),
    p1_policy: str = typer.Option(
        "python_external_v1",
        "--p1-policy",
        help="Policy for p1: 'node_random_v1' or 'python_external_v1'",
    ),
    p2_policy: str = typer.Option(
        "node_random_v1",
        "--p2-policy",
        help="Policy for p2: 'node_random_v1' or 'python_external_v1'",
    ),
    strict_invalid_choice: bool = typer.Option(
        True,
        "--strict/--no-strict",
        help="Strict mode (default): invalid choices cause hard failures. --no-strict is EXPERIMENTAL and may cause hangs.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Enable verbose debug logging",
    ),
):
    """
    Run a few battles using the Python-driven online self-play bridge.

    This is a smoke test for the Node <-> Python protocol, PreviewPolicy, and BattleBCPolicy.
    Battles are logged to the usual datasets (full_battles, trajectories).
    """
    from projects.rl_battle.online_selfplay import OnlineSelfPlayConfig, run_online_selfplay

    cfg = OnlineSelfPlayConfig(
        num_episodes=num_episodes,
        format_id=format_id,
        p1_policy=p1_policy,
        p2_policy=p2_policy,
        seed=42,
        write_trajectories=True,
        strict_invalid_choice=strict_invalid_choice,
        debug=debug,
    )

    typer.echo(f"Running {num_episodes} online self-play battles...")
    typer.echo(f"  p1_policy: {cfg.p1_policy}")
    typer.echo(f"  p2_policy: {cfg.p2_policy}")

    try:
        summary = run_online_selfplay(cfg)
        typer.echo("\nOnline self-play summary:")
        typer.echo(f"  Episodes: {summary['episodes']}")
        typer.echo(f"  Errors:   {summary['errors']}")
        typer.echo(f"  P1 wins:  {summary['p1_wins']}")
        typer.echo(f"  P2 wins:  {summary['p2_wins']}")
        typer.echo(f"  Draws:    {summary['draws']}")
        
        # Exit with non-zero code if there were errors
        if summary['errors'] > 0:
            raise typer.Exit(code=1)
    except typer.Exit:
        raise
    except Exception as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command("inspect-selfplay")
def inspect_selfplay(
    max_battles: Optional[int] = typer.Option(
        None, "--max-battles", help="Maximum number of battles to load for summary"
    ),
) -> None:
    """
    Inspect and summarize saved online self-play battles.
    
    This is a read-only diagnostic tool that loads battles from the dataset
    created by online-selfplay and prints aggregate statistics.
    """
    from projects.rl_battle.inspect_selfplay import summarize_selfplay
    
    stats = summarize_selfplay(max_battles=max_battles)
    
    typer.echo("Self-play dataset summary")
    typer.echo("-" * 25)
    typer.echo(f"Total battles: {stats['total_battles']}")
    typer.echo("Winner counts:")
    typer.echo(f"  p1:      {stats['winner_counts']['p1']:4d}")
    typer.echo(f"  p2:      {stats['winner_counts']['p2']:4d}")
    typer.echo(f"  tie:     {stats['winner_counts']['tie']:4d}")
    typer.echo(f"  unknown: {stats['winner_counts']['unknown']:4d}")
    
    if stats['avg_turns'] is not None:
        typer.echo(f"Avg turns:   {stats['avg_turns']:.1f}")
    else:
        typer.echo("Avg turns:   N/A")
    
    if stats['median_turns'] is not None:
        typer.echo(f"Median turns: {stats['median_turns']:.1f}")
    else:
        typer.echo("Median turns: N/A")


@app.command("analyze-sanitizer")
def analyze_sanitizer(
    dataset_root: Path = typer.Argument(..., help="Root directory for self-play trajectories or path to trajectories.jsonl"),
) -> None:
    """
    Analyze sanitize_reason distribution across battle trajectories.

    Shows how often each sanitization reason occurs in the dataset.
    """
    from vgc_lab.datasets import summarize_sanitize_reasons

    try:
        counts = summarize_sanitize_reasons(dataset_root)
        total = sum(counts.values()) or 1

        typer.echo(f"Total steps with sanitize_reason: {total}")
        typer.echo()
        typer.echo(f"{'Reason':<30s} {'Count':>10s} {'Percentage':>10s}")
        typer.echo("=" * 50)

        # Sort by count descending, then by reason name
        for reason in sorted(counts.keys(), key=lambda r: (-counts[r], r)):
            c = counts[reason]
            pct = 100.0 * c / total
            typer.echo(f"{reason:<30s} {c:>10d} {pct:>9.2f}%")

    except FileNotFoundError as e:
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

