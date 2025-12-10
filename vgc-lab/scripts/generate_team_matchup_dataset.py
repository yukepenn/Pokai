#!/usr/bin/env python3
"""CLI: Generate pairwise team matchup dataset."""

from __future__ import annotations

from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Optional

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import DEFAULT_FORMAT, ensure_paths
from vgc_lab.eval import evaluate_team_vs_team
from vgc_lab.team_builder import build_team_import_from_set_ids
from vgc_lab.team_matchup_data import (
    MATCHUP_JSONL_PATH,
    TeamMatchupRecord,
    append_matchup_record,
    ensure_matchup_dir,
)
from vgc_lab.team_pool import TeamPool, TEAMS_CATALOG_PATH

# Default pool YAML paths
AUTO_POOL_PATH = Path("data/catalog/teams_regf_auto.yaml")
DEFAULT_POOL_PATH = TEAMS_CATALOG_PATH

app = typer.Typer()


@app.command()
def main(
    pool_yaml: Optional[Path] = typer.Option(
        None,
        "--pool-yaml",
        help="Which team catalog to load. Defaults to teams_regf_auto.yaml if exists, else teams_regf.yaml.",
    ),
    max_pairs: int = typer.Option(
        200,
        "--max-pairs",
        help="Maximum number of team pairs to evaluate.",
    ),
    n_battles_per_pair: int = typer.Option(
        12,
        "--n-battles-per-pair",
        help="Number of battles for each A vs B evaluation.",
    ),
    dataset_path: Optional[Path] = typer.Option(
        None,
        "--dataset-path",
        help="Optional override for the JSONL output path.",
    ),
    format_id: str = typer.Option(
        DEFAULT_FORMAT,
        "--format",
        help="Showdown format ID.",
    ),
):
    """
    Evaluate team-vs-team matchups for teams from a catalog and save results.
    """
    try:
        ensure_paths()
        ensure_matchup_dir()

        # Determine pool YAML path
        if pool_yaml is None:
            if AUTO_POOL_PATH.exists():
                pool_yaml = AUTO_POOL_PATH
            else:
                pool_yaml = DEFAULT_POOL_PATH

        pool_yaml = Path(pool_yaml)
        if not pool_yaml.exists():
            typer.echo(f"ERROR: Team pool file not found: {pool_yaml}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Loading team pool from: {pool_yaml}")
        pool = TeamPool.from_yaml(pool_yaml)

        team_list = pool.all()
        typer.echo(f"Loaded {len(team_list)} teams from pool.")

        if len(team_list) < 2:
            typer.echo("ERROR: Need at least 2 teams to evaluate matchups.", err=True)
            raise typer.Exit(1)

        # Generate pairs
        all_pairs = list(combinations(team_list, 2))
        if len(all_pairs) > max_pairs:
            typer.echo(
                f"Total possible pairs: {len(all_pairs)}, limiting to {max_pairs}"
            )
            pairs_to_evaluate = all_pairs[:max_pairs]
        else:
            pairs_to_evaluate = all_pairs

        typer.echo(f"Evaluating {len(pairs_to_evaluate)} team pairs...")
        typer.echo(f"Battles per pair: {n_battles_per_pair}")
        typer.echo(f"Format: {format_id}\n")

        output_path = dataset_path or MATCHUP_JSONL_PATH

        # Evaluate each pair
        for idx, (team_a, team_b) in enumerate(pairs_to_evaluate, start=1):
            try:
                # Build import text for both teams
                team_a_import = build_team_import_from_set_ids(
                    team_a.set_ids, catalog=None  # Will use default catalog
                )
                team_b_import = build_team_import_from_set_ids(
                    team_b.set_ids, catalog=None
                )

                # Evaluate matchup
                matchup = evaluate_team_vs_team(
                    team_a_import=team_a_import,
                    team_b_import=team_b_import,
                    format_id=format_id,
                    n=n_battles_per_pair,
                )

                # Create record
                win_rate_a = matchup.n_a_wins / matchup.n_battles if matchup.n_battles > 0 else 0.0

                record = TeamMatchupRecord(
                    team_a_id=team_a.id,
                    team_b_id=team_b.id,
                    team_a_set_ids=team_a.set_ids,
                    team_b_set_ids=team_b.set_ids,
                    format_id=format_id,
                    n_battles=matchup.n_battles,
                    n_a_wins=matchup.n_a_wins,
                    n_b_wins=matchup.n_b_wins,
                    n_ties=matchup.n_ties,
                    avg_turns=matchup.avg_turns,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    meta={
                        "source": "catalog_pairwise_v1",
                        "n_battles_per_pair": n_battles_per_pair,
                    },
                )

                append_matchup_record(record, path=output_path)

                typer.echo(
                    f"[{idx}/{len(pairs_to_evaluate)}] "
                    f"A={team_a.id}, B={team_b.id}, "
                    f"win_rate_a={win_rate_a:.3f} "
                    f"(wins={matchup.n_a_wins} / {matchup.n_battles})"
                )
            except Exception as e:
                typer.echo(
                    f"[{idx}/{len(pairs_to_evaluate)}] ERROR evaluating "
                    f"A={team_a.id}, B={team_b.id}: {e}",
                    err=True,
                )
                continue

        typer.echo(f"\nDataset written/appended to: {output_path}")
        typer.echo(f"Total pairs evaluated: {len(pairs_to_evaluate)}")

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

