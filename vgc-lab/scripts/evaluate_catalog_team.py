#!/usr/bin/env python3
"""CLI: Evaluate a catalog-defined team vs random opponents."""

from __future__ import annotations

from typing import List

import typer

# Add src to PYTHONPATH
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import DEFAULT_FORMAT, ensure_paths
from vgc_lab.team_builder import evaluate_catalog_team_vs_random

app = typer.Typer()


@app.command()
def main(
    set_ids: List[str] = typer.Argument(
        ...,
        help="List of set IDs from the catalog to form a 6-Pokemon team.",
    ),
    n: int = typer.Option(
        50,
        "--n",
        "-n",
        help="Number of self-play battles to run.",
    ),
    format_id: str = typer.Option(
        DEFAULT_FORMAT,
        "--format",
        "-f",
        help="Showdown format ID (default: gen9vgc2026regf).",
    ),
    team_as: str = typer.Option(
        "both",
        "--team-as",
        help='Which side to place the team on: "p1", "p2", or "both".',
    ),
    save_logs: bool = typer.Option(
        False,
        "--save-logs",
        help="If set, save logs to data/battles_raw/.",
    ),
):
    """
    Evaluate a team defined by set IDs from the catalog.
    """
    try:
        ensure_paths()

        # Validate: must have exactly 6 set IDs
        if len(set_ids) != 6:
            typer.echo(
                f"ERROR: Expected exactly 6 set IDs, got {len(set_ids)}",
                err=True,
            )
            raise typer.Exit(1)

        summary = evaluate_catalog_team_vs_random(
            set_ids=set_ids,
            format_id=format_id,
            n=n,
            team_as=team_as,
            save_logs=save_logs,
        )

        winrate = summary.n_wins / summary.n_battles if summary.n_battles > 0 else 0.0
        typer.echo(f"Format: {summary.format_id}")
        typer.echo(f"Team role: {summary.team_role}")
        typer.echo(f"Battles: {summary.n_battles}")
        typer.echo(f"Wins: {summary.n_wins}, Losses: {summary.n_losses}, Ties: {summary.n_ties}")
        typer.echo(f"Win rate: {winrate:.3f}")
        if summary.avg_turns is not None:
            typer.echo(f"Average turns: {summary.avg_turns:.2f}")
        typer.echo(f"Evaluated at: {summary.created_at.isoformat()}")
    except KeyError as e:
        typer.echo(f"ERROR: Set ID not found in catalog: {e}", err=True)
        raise typer.Exit(1)
    except ValueError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)
    except RuntimeError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

