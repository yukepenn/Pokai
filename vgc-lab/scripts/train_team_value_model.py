#!/usr/bin/env python3
"""CLI: Train a team value model on teams_vs_pool.jsonl."""

from __future__ import annotations

from pathlib import Path

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import PROJECT_ROOT
from vgc_lab.team_value_model import (
    TrainingConfig,
    train_team_value_model,
)

DEFAULT_JSONL = PROJECT_ROOT / "data" / "datasets" / "teams_vs_pool" / "teams_vs_pool.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "data" / "models" / "team_value_mlp.pt"

app = typer.Typer()


@app.command()
def main(
    jsonl_path: Path = typer.Option(
        str(DEFAULT_JSONL),
        "--jsonl",
        help="Path to teams_vs_pool.jsonl dataset.",
    ),
    out_path: Path = typer.Option(
        str(DEFAULT_OUT),
        "--out",
        help="Path to save the trained model checkpoint.",
    ),
    epochs: int = typer.Option(50, "--epochs", "-e"),
    batch_size: int = typer.Option(64, "--batch-size", "-b"),
    lr: float = typer.Option(3e-4, "--lr"),
    min_battles: int = typer.Option(
        3, "--min-battles", help="Minimum battles per sample to include."
    ),
    include_random: bool = typer.Option(
        True, "--include-random/--no-include-random"
    ),
    include_catalog: bool = typer.Option(
        True, "--include-catalog/--no-include-catalog"
    ),
    device: str = typer.Option(
        "cpu", "--device", help='"cpu" or "cuda" if available.'
    ),
):
    """Train a neural team value model on teams_vs_pool.jsonl."""
    try:
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            typer.echo(f"ERROR: Dataset file not found: {jsonl_path}", err=True)
            raise typer.Exit(1)

        sources = []
        if include_random:
            sources.append("random")
        if include_catalog:
            sources.append("catalog")
        if not sources:
            typer.echo(
                "ERROR: At least one of --include-random or --include-catalog must be enabled",
                err=True,
            )
            raise typer.Exit(1)

        cfg = TrainingConfig(
            jsonl_path=jsonl_path,
            out_path=Path(out_path),
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            min_battles=min_battles,
            include_sources=sources,
            device=device,
        )

        typer.echo(f"Training team value model on {jsonl_path}")
        typer.echo(f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}, device={device}")
        typer.echo(f"Including sources: {sources}")

        model, vocab, train_meta = train_team_value_model(cfg, device=device)

        typer.echo(f"\nSaved team value model to {out_path}")
        typer.echo(f"Vocab size: {len(vocab)}")
        typer.echo(
            f"Final metrics: train_loss={train_meta['train_loss']:.4f}, "
            f"val_loss={train_meta['val_loss']:.4f}, "
            f"val_corr={train_meta['val_corr']:.4f}"
        )
    except ValueError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

