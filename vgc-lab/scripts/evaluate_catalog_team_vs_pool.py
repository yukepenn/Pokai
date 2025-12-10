#!/usr/bin/env python3
"""CLI: Evaluate a catalog-defined team vs a catalog-defined pool."""

from __future__ import annotations

from pathlib import Path
from typing import List

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import DEFAULT_FORMAT, ensure_paths
from vgc_lab.team_pool import TeamPool, TEAMS_CATALOG_PATH
from vgc_lab.team_search import evaluate_team_against_pool

app = typer.Typer()


@app.command()
def main(
    set_ids: List[str] = typer.Argument(
        ...,
        help="List of set IDs from the catalog forming a 6-Pokemon team.",
    ),
    n_opponents: int = typer.Option(
        5,
        "--n-opponents",
        help="Number of opponent teams sampled from the pool.",
    ),
    n_battles_per_opponent: int = typer.Option(
        5,
        "--n-per",
        help="Number of battles per opponent matchup.",
    ),
    format_id: str = typer.Option(
        DEFAULT_FORMAT,
        "--format",
        "-f",
        help="Showdown format ID (default: gen9vgc2026regf).",
    ),
    teams_yaml: str = typer.Option(
        str(TEAMS_CATALOG_PATH),
        "--teams-yaml",
        help="Path to teams YAML catalog.",
    ),
):
    """
    Evaluate a catalog-defined team vs a catalog-defined pool of teams.
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

        pool = TeamPool.from_yaml(Path(teams_yaml))

        summary = evaluate_team_against_pool(
            team_set_ids=set_ids,
            pool=pool,
            format_id=format_id,
            n_opponents=n_opponents,
            n_battles_per_opponent=n_battles_per_opponent,
        )

        winrate = (
            summary.n_wins / summary.n_battles_total
            if summary.n_battles_total > 0
            else 0.0
        )

        typer.echo(f"Format: {summary.format_id}")
        typer.echo(f"Team set IDs: {', '.join(summary.team_set_ids)}")
        typer.echo(
            f"Battles: {summary.n_battles_total} "
            f"(opponents={summary.n_opponents}, per_opponent={summary.n_battles_per_opponent})"
        )
        typer.echo(
            f"Wins: {summary.n_wins}, Losses: {summary.n_losses}, Ties: {summary.n_ties}"
        )
        typer.echo(f"Win rate vs pool: {winrate:.3f}")
        if summary.avg_turns is not None:
            typer.echo(f"Average turns: {summary.avg_turns:.2f}")
        if summary.opponent_counts:
            typer.echo("Opponent usage counts:")
            for opp_id, count in summary.opponent_counts.items():
                typer.echo(f"  - {opp_id}: {count}")
    except FileNotFoundError as e:
        typer.echo(f"ERROR: File not found: {e}", err=True)
        raise typer.Exit(1)
    except KeyError as e:
        typer.echo(f"ERROR: {e}", err=True)
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

