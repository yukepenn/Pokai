#!/usr/bin/env python3
"""CLI: Evaluate matchup between two teams using RandomPlayerAI."""

from __future__ import annotations

from pathlib import Path

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import DEFAULT_FORMAT
from vgc_lab.eval import evaluate_team_vs_team

app = typer.Typer()


@app.command()
def main(
    team_import_path: Path = typer.Argument(
        ...,
        help="Path to Showdown import text for Team A.",
    ),
    opp_import_path: Path = typer.Argument(
        ...,
        help="Path to Showdown import text for Team B.",
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
):
    """Evaluate a matchup between two teams."""
    try:
        if not team_import_path.exists():
            typer.echo(f"ERROR: Team A import file not found: {team_import_path}", err=True)
            raise typer.Exit(1)
        if not opp_import_path.exists():
            typer.echo(f"ERROR: Team B import file not found: {opp_import_path}", err=True)
            raise typer.Exit(1)

        team_a_import = team_import_path.read_text(encoding="utf-8")
        team_b_import = opp_import_path.read_text(encoding="utf-8")

        summary = evaluate_team_vs_team(
            team_a_import=team_a_import,
            team_b_import=team_b_import,
            format_id=format_id,
            n=n,
        )

        total = summary.n_battles
        winrate_a = summary.n_a_wins / total if total > 0 else 0.0
        winrate_b = summary.n_b_wins / total if total > 0 else 0.0

        typer.echo(f"Format: {summary.format_id}")
        typer.echo(f"Battles: {summary.n_battles}")
        typer.echo(f"Team A wins: {summary.n_a_wins}")
        typer.echo(f"Team B wins: {summary.n_b_wins}")
        typer.echo(f"Ties: {summary.n_ties}")
        typer.echo(f"Team A win rate: {winrate_a:.3f}")
        typer.echo(f"Team B win rate: {winrate_b:.3f}")
        if summary.avg_turns is not None:
            typer.echo(f"Average turns: {summary.avg_turns:.2f}")
        typer.echo(f"Evaluated at: {summary.created_at.isoformat()}")
    except RuntimeError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

