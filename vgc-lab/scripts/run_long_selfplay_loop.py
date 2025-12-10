#!/usr/bin/env python3
"""CLI: Long-running self-play orchestrator for continuous improvement."""

from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Optional

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import (
    DATA_ROOT,
    MODELS_DIR,
    PROJECT_ROOT,
    ensure_paths,
)

app = typer.Typer()


@app.command()
def main(
    time_budget_hours: float = typer.Option(
        8.0,
        "--time-budget-hours",
        help="Maximum time to run in hours.",
    ),
    device: str = typer.Option(
        "cuda",
        "--device",
        help="Device for model training ('cpu' or 'cuda').",
    ),
    max_iters: int = typer.Option(
        1000,
        "--max-iters",
        help="Maximum number of iterations (safeguard).",
    ),
    # Hyperparameters
    team_iter_epochs: int = typer.Option(
        20,
        "--team-iter-epochs",
        help="Epochs for team value model training per iteration.",
    ),
    team_iter_n_samples: int = typer.Option(
        20,
        "--team-iter-n-samples",
        help="Number of new team samples per iteration.",
    ),
    matchup_max_pairs: int = typer.Option(
        50,
        "--matchup-max-pairs",
        help="Maximum matchup pairs to evaluate per iteration.",
    ),
    matchup_epochs: int = typer.Option(
        20,
        "--matchup-epochs",
        help="Epochs for matchup model training per iteration.",
    ),
    n_battles_per_matchup: int = typer.Option(
        6,
        "--n-battles-per-matchup",
        help="Number of battles per matchup pair.",
    ),
):
    """
    Run a long-running self-play loop that continuously:
    - Expands teams_vs_pool dataset
    - Retrains team value model
    - Analyzes and exports top teams
    - Generates matchup dataset
    - Trains matchup model
    - Optionally generates battle trajectories
    """
    ensure_paths()

    start_time = time.time()
    time_budget_seconds = time_budget_hours * 3600.0
    iter_idx = 1

    # Paths
    teams_auto_yaml = DATA_ROOT / "catalog" / "teams_regf_auto.yaml"
    team_value_ckpt = MODELS_DIR / "team_value_mlp.pt"
    matchup_ckpt = MODELS_DIR / "team_matchup_mlp.pt"

    typer.echo(f"=== Long self-play loop starting ===")
    typer.echo(f"Time budget: {time_budget_hours} hours")
    typer.echo(f"Device: {device}")
    typer.echo(f"Max iterations: {max_iters}")
    typer.echo()

    while iter_idx <= max_iters and (time.time() - start_time) < time_budget_seconds:
        elapsed_hours = (time.time() - start_time) / 3600.0
        remaining_hours = time_budget_hours - elapsed_hours

        typer.echo(f"\n{'='*60}")
        typer.echo(f"=== Iteration {iter_idx} ===")
        typer.echo(f"Elapsed: {elapsed_hours:.2f} hours | Remaining: {remaining_hours:.2f} hours")
        typer.echo(f"{'='*60}\n")

        # 1) Expand teams_vs_pool dataset
        typer.echo("[1/5] Expanding teams_vs_pool dataset...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "teams_vs_pool_dataset.py"),
                    "--n-samples",
                    str(team_iter_n_samples),
                    "--include-catalog-teams",
                ],
                check=True,
                capture_output=False,
            )
            typer.echo("  ✓ Dataset expanded")
        except subprocess.CalledProcessError as e:
            typer.echo(f"  ✗ Error: {e}", err=True)
            iter_idx += 1
            continue

        # 2) Retrain team value model
        typer.echo("[2/5] Training team value model...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "train_team_value_model.py"),
                    "--epochs",
                    str(team_iter_epochs),
                    "--device",
                    device,
                    "--out",
                    str(team_value_ckpt),
                ],
                check=True,
                capture_output=False,
            )
            typer.echo("  ✓ Team value model trained")
        except subprocess.CalledProcessError as e:
            typer.echo(f"  ✗ Error: {e}", err=True)
            iter_idx += 1
            continue

        # 3) Analyze teams and export top teams
        typer.echo("[3/5] Analyzing teams and exporting top teams...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "analyze_teams_vs_pool.py"),
                    "--min-battles",
                    "3",
                    "--min-win-rate",
                    "0.4",
                    "--top-k",
                    "20",
                ],
                check=True,
                capture_output=False,
            )
            typer.echo("  ✓ Top teams exported")
        except subprocess.CalledProcessError as e:
            typer.echo(f"  ✗ Error: {e}", err=True)
            # Continue anyway - this is less critical

        # 4) Generate/refresh matchup dataset and train matchup model
        typer.echo("[4/5] Generating matchup dataset...")
        try:
            # Use auto teams if available, else base catalog
            pool_yaml = str(teams_auto_yaml) if teams_auto_yaml.exists() else str(
                DATA_ROOT / "catalog" / "teams_regf.yaml"
            )

            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "generate_team_matchup_dataset.py"),
                    "--pool-yaml",
                    pool_yaml,
                    "--max-pairs",
                    str(matchup_max_pairs),
                    "--n-battles-per-pair",
                    str(n_battles_per_matchup),
                ],
                check=True,
                capture_output=False,
            )
            typer.echo("  ✓ Matchup dataset generated")
        except subprocess.CalledProcessError as e:
            typer.echo(f"  ✗ Error: {e}", err=True)
            iter_idx += 1
            continue

        typer.echo("[4.5/5] Training matchup model...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "train_team_matchup_model.py"),
                    "--epochs",
                    str(matchup_epochs),
                    "--min-battles",
                    "4",
                    "--device",
                    device,
                    "--out",
                    str(matchup_ckpt),
                ],
                check=True,
                capture_output=False,
            )
            typer.echo("  ✓ Matchup model trained")
        except subprocess.CalledProcessError as e:
            typer.echo(f"  ✗ Error: {e}", err=True)
            # Continue anyway - matchup model is less critical

        # 5) Optionally: generate some battle trajectories
        typer.echo("[5/5] Updating battle trajectories...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).parent / "dump_battle_trajectories.py"),
                ],
                check=True,
                capture_output=False,
            )
            typer.echo("  ✓ Battle trajectories updated")
        except subprocess.CalledProcessError as e:
            typer.echo(f"  ✗ Warning: {e}", err=True)
            # This is optional, so we continue

        typer.echo(f"\n✓ Iteration {iter_idx} complete\n")

        iter_idx += 1

        # Small pause between iterations
        time.sleep(1)

    elapsed_hours = (time.time() - start_time) / 3600.0
    typer.echo(f"\n{'='*60}")
    typer.echo(f"Long self-play loop finished.")
    typer.echo(f"Total iterations: {iter_idx - 1}")
    typer.echo(f"Total time: {elapsed_hours:.2f} hours")
    typer.echo(f"{'='*60}\n")


if __name__ == "__main__":
    app()

