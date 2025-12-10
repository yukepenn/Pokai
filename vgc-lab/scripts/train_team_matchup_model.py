#!/usr/bin/env python3
"""CLI: Train a team matchup model on team_matchups.jsonl."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import MODELS_DIR, PROJECT_ROOT
from vgc_lab.team_matchup_data import MATCHUP_JSONL_PATH
from vgc_lab.team_matchup_model import (
    MatchupTrainingConfig,
    train_team_matchup_model,
)

DEFAULT_OUT = MODELS_DIR / "team_matchup_mlp.pt"

app = typer.Typer()


@app.command()
def main(
    jsonl_path: Path = typer.Option(
        str(MATCHUP_JSONL_PATH),
        "--jsonl",
        help="Path to team_matchups.jsonl dataset.",
    ),
    out_path: Path = typer.Option(
        str(DEFAULT_OUT),
        "--out",
        help="Path to save the trained model checkpoint.",
    ),
    epochs: int = typer.Option(40, "--epochs", "-e"),
    batch_size: int = typer.Option(64, "--batch-size", "-b"),
    lr: float = typer.Option(1e-3, "--lr"),
    min_battles: int = typer.Option(
        10, "--min-battles", help="Minimum battles per matchup to include."
    ),
    max_records: Optional[int] = typer.Option(
        None, "--max-records", help="Optional cap on number of records for quick tests."
    ),
    device: str = typer.Option(
        "cuda", "--device", help='"cpu" or "cuda" if available.'
    ),
):
    """Train a neural team matchup model on team_matchups.jsonl."""
    try:
        jsonl_path = Path(jsonl_path)
        if not jsonl_path.exists():
            typer.echo(f"ERROR: Dataset file not found: {jsonl_path}", err=True)
            raise typer.Exit(1)

        cfg = MatchupTrainingConfig(
            jsonl_path=jsonl_path,
            out_path=Path(out_path),
            batch_size=batch_size,
            epochs=epochs,
            lr=lr,
            min_battles=min_battles,
            max_records=max_records,
            device=device,
        )

        typer.echo(f"Training team matchup model on {jsonl_path}")
        typer.echo(f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}, device={device}")

        model, vocab, train_meta = train_team_matchup_model(cfg, device_override=device)

        typer.echo(f"\nSaved team matchup model to {out_path}")
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

