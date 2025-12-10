#!/usr/bin/env python3
"""CLI: Generate dataset of random teams vs pool evaluations."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import DEFAULT_FORMAT, ensure_paths, PROJECT_ROOT
from vgc_lab.set_catalog import SetCatalog
from vgc_lab.team_pool import TeamPool, TEAMS_CATALOG_PATH
from vgc_lab.team_search import (
    CandidateTeamResult,
    evaluate_team_against_pool,
    random_search_over_pool,
)

# Dataset location
DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "teams_vs_pool"
DATASET_PATH = DATASET_DIR / "teams_vs_pool.jsonl"


def ensure_dataset_dir() -> None:
    """Ensure the dataset directory exists."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def append_result_to_dataset(
    result: CandidateTeamResult,
    *,
    source: str = "random",
    team_id: Optional[str] = None,
) -> None:
    """
    Append one CandidateTeamResult as a JSON line to the dataset.

    Args:
        result: CandidateTeamResult to serialize and append.
        source: Source of the team ("random" or "catalog").
        team_id: Optional team ID (for catalog teams).
    """
    ensure_dataset_dir()

    summary = result.pool_summary
    win_rate = result.win_rate

    record = {
        "team_set_ids": result.team_set_ids,
        "format_id": summary.format_id,
        "n_opponents": summary.n_opponents,
        "n_battles_per_opponent": summary.n_battles_per_opponent,
        "n_battles_total": summary.n_battles_total,
        "n_wins": summary.n_wins,
        "n_losses": summary.n_losses,
        "n_ties": summary.n_ties,
        "win_rate": win_rate,
        "avg_turns": summary.avg_turns,
        "opponent_counts": summary.opponent_counts,
        "source": source,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    if team_id is not None:
        record["team_id"] = team_id

    with DATASET_PATH.open("a", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")


app = typer.Typer()


@app.command()
def main(
    n_samples: int = typer.Option(
        50,
        "--n-samples",
        help="Number of random teams to evaluate and write to dataset.",
    ),
    team_size: int = typer.Option(
        6,
        "--team-size",
        help="Number of Pokémon (set IDs) per team.",
    ),
    n_opponents: int = typer.Option(
        5,
        "--n-opponents",
        help="Number of opponent teams sampled from the pool for each team.",
    ),
    n_battles_per_opponent: int = typer.Option(
        5,
        "--n-per",
        help="Number of battles per opponent matchup.",
    ),
    include_catalog_teams: bool = typer.Option(
        False,
        "--include-catalog-teams/--no-include-catalog-teams",
        help="Also evaluate all catalog teams vs the pool.",
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
    Generate a dataset of random teams evaluated against the team pool.
    """
    try:
        ensure_paths()
        ensure_dataset_dir()

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
            n_candidates=n_samples,
            team_size=team_size,
            n_opponents=n_opponents,
            n_battles_per_opponent=n_battles_per_opponent,
        )

        n_random_written = 0
        for i, result in enumerate(results, start=1):
            append_result_to_dataset(result, source="random")
            n_random_written += 1
            winrate = result.win_rate
            typer.echo(
                f"[Random {i}/{n_samples}] win_rate={winrate:.3f}, "
                f"team={', '.join(result.team_set_ids[:2])}..."
            )

        n_catalog_written = 0
        if include_catalog_teams:
            catalog_teams = pool.all()
            typer.echo(f"\nEvaluating {len(catalog_teams)} catalog teams...")
            for i, team_def in enumerate(catalog_teams, start=1):
                try:
                    summary = evaluate_team_against_pool(
                        team_set_ids=team_def.set_ids,
                        pool=pool,
                        n_opponents=n_opponents,
                        n_battles_per_opponent=n_battles_per_opponent,
                        catalog=catalog,
                    )
                    result = CandidateTeamResult(
                        team_set_ids=team_def.set_ids,
                        pool_summary=summary,
                    )
                    append_result_to_dataset(
                        result,
                        source="catalog",
                        team_id=team_def.id,
                    )
                    n_catalog_written += 1
                    winrate = result.win_rate
                    typer.echo(
                        f"[Catalog {i}/{len(catalog_teams)}] "
                        f"team_id={team_def.id}, win_rate={winrate:.3f}"
                    )
                except Exception as e:
                    typer.echo(
                        f"[Catalog {i}/{len(catalog_teams)}] ERROR evaluating "
                        f"team_id={team_def.id}: {e}",
                        err=True,
                    )

        # Count total records in file
        total_records = 0
        if DATASET_PATH.exists():
            with DATASET_PATH.open("r", encoding="utf-8") as f:
                total_records = sum(1 for line in f if line.strip())

        typer.echo(f"\nDataset written to: {DATASET_PATH}")
        typer.echo(f"Random samples written: {n_random_written}")
        if include_catalog_teams:
            typer.echo(f"Catalog teams evaluated: {n_catalog_written}")
        typer.echo(f"Total records in file: {total_records}")
    except ValueError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

