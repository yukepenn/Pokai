#!/usr/bin/env python3
"""CLI: Random search baseline over catalog-defined teams vs pool."""

from __future__ import annotations

from pathlib import Path
from typing import List

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import DEFAULT_FORMAT, ensure_paths
from vgc_lab.set_catalog import SetCatalog
from vgc_lab.team_pool import TeamPool, TEAMS_CATALOG_PATH
from vgc_lab.team_search import random_search_over_pool

app = typer.Typer()


@app.command()
def main(
    n_candidates: int = typer.Option(
        20,
        "--n-candidates",
        help="Number of random candidate teams to evaluate.",
    ),
    team_size: int = typer.Option(
        6,
        "--team-size",
        help="Number of Pokémon (set IDs) per team.",
    ),
    n_opponents: int = typer.Option(
        5,
        "--n-opponents",
        help="Number of opponent teams sampled from the pool for each candidate.",
    ),
    n_battles_per_opponent: int = typer.Option(
        5,
        "--n-per",
        help="Number of battles per opponent matchup.",
    ),
    top_k: int = typer.Option(
        5,
        "--top-k",
        help="How many top teams to print.",
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
    Run a random search over catalog-defined sets, evaluating teams vs the team pool.
    """
    try:
        ensure_paths()

        catalog = SetCatalog.from_yaml()
        pool = TeamPool.from_yaml(Path(teams_yaml))

        if len(catalog) < team_size:
            typer.echo(
                f"ERROR: Catalog has only {len(catalog)} sets, cannot build teams of size {team_size}",
                err=True,
            )
            raise typer.Exit(1)

        results = random_search_over_pool(
            catalog=catalog,
            pool=pool,
            n_candidates=n_candidates,
            team_size=team_size,
            n_opponents=n_opponents,
            n_battles_per_opponent=n_battles_per_opponent,
        )

        if top_k <= 0:
            top_k = len(results)
        top_k = min(top_k, len(results))

        typer.echo(f"Evaluated {len(results)} candidate teams.")
        typer.echo(f"Showing top {top_k} by win rate vs pool:\n")

        for i, r in enumerate(results[:top_k], start=1):
            winrate = r.win_rate
            summary = r.pool_summary
            typer.echo(f"[#{i}] win_rate={winrate:.3f}, battles={summary.n_battles_total}")
            typer.echo(f"     team_set_ids: {', '.join(r.team_set_ids)}")
    except ValueError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

