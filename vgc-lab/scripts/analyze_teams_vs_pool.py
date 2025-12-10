#!/usr/bin/env python3
"""CLI: Analyze teams_vs_pool dataset and list top teams."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Optional

import typer
import yaml

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import DEFAULT_FORMAT, ensure_paths
from vgc_lab.set_catalog import SetCatalog
from vgc_lab.team_analysis import (
    TeamAggregateStats,
    load_and_aggregate_teams_vs_pool,
)
from vgc_lab.teams_vs_pool_data import TEAMS_VS_POOL_JSONL_PATH

app = typer.Typer()


@app.command()
def main(
    top_k: int = typer.Option(
        20,
        "--top-k",
        help="Number of top teams to display.",
    ),
    min_battles: int = typer.Option(
        20,
        "--min-battles",
        help="Minimum total battles required for a team to be considered.",
    ),
    min_records: int = typer.Option(
        1,
        "--min-records",
        help="Minimum number of JSONL records for a team to be considered.",
    ),
    source_prefix: Optional[str] = typer.Option(
        None,
        "--source-prefix",
        help="If set, only count/consider teams that have at least one record with source starting with this prefix "
        "(e.g. 'auto_iter_' or 'auto_iter_2_'). If None, use all sources.",
    ),
    jsonl_path: Optional[Path] = typer.Option(
        None,
        "--jsonl-path",
        help="Path to teams_vs_pool.jsonl. Defaults to the configured TEAMS_VS_POOL_JSONL_PATH.",
    ),
    output_csv: Optional[Path] = typer.Option(
        None,
        "--output-csv",
        help="Optional path to write a CSV of aggregated team stats.",
    ),
    output_teams_yaml: Optional[Path] = typer.Option(
        None,
        "--output-teams-yaml",
        help="Optional path to write a YAML catalog of the top teams.",
    ),
):
    """
    Analyze the teams_vs_pool dataset:

    - Aggregate evaluations by unique team_set_ids combination.
    - Filter by minimum battles / records.
    - Optionally restrict to teams that have at least one record from a given source prefix.
    - Print the top-K teams ranked by aggregated win_rate.
    - Optionally export:
        - a CSV of aggregates,
        - a YAML catalog file of the top teams for later use (e.g. as TeamPool, Showdown testing, RL).
    """
    try:
        ensure_paths()

        if jsonl_path is None:
            jsonl_path = TEAMS_VS_POOL_JSONL_PATH

        typer.echo(f"Using dataset: {jsonl_path}")

        if not jsonl_path.exists():
            typer.echo(f"ERROR: Dataset file not found: {jsonl_path}", err=True)
            raise typer.Exit(1)

        aggregates = load_and_aggregate_teams_vs_pool(jsonl_path)
        typer.echo(f"Loaded {len(aggregates)} aggregated teams before filtering.")

        # Apply source prefix filter
        if source_prefix is not None:
            filtered_by_source = [
                agg
                for agg in aggregates
                if any(src.startswith(source_prefix) for src in agg.source_counts.keys())
            ]
            aggregates = filtered_by_source
            typer.echo(
                f"After source_prefix='{source_prefix}' filter: {len(aggregates)} teams"
            )

        # Filter by min_battles and min_records
        filtered = [
            agg
            for agg in aggregates
            if agg.n_battles_total >= min_battles and agg.n_records >= min_records
        ]

        typer.echo(
            f"After min_battles={min_battles}, min_records={min_records} filter: {len(filtered)} teams"
        )

        if not filtered:
            typer.echo("No teams match the filtering criteria.", err=True)
            raise typer.Exit(1)

        # Sort by win_rate descending, then by n_battles_total descending as tiebreaker
        filtered.sort(key=lambda a: (a.win_rate, a.n_battles_total), reverse=True)
        top = filtered[:top_k]

        # Pretty-print the top teams
        typer.echo(f"\n{'='*80}")
        typer.echo(f"Top {len(top)} teams (ranked by win_rate):")
        typer.echo(f"{'='*80}\n")

        for rank, agg in enumerate(top, start=1):
            typer.echo(
                f"[#{rank}] win_rate={agg.win_rate:.3f} "
                f"(wins={agg.n_wins} / battles={agg.n_battles_total}, records={agg.n_records})"
            )
            typer.echo(f"     set_ids: {', '.join(agg.team_set_ids)}")
            source_str = ", ".join(
                f"{src}={count}" for src, count in sorted(agg.source_counts.items())
            )
            typer.echo(f"     sources: {source_str}")
            typer.echo()

        # Optional CSV output
        if output_csv is not None:
            output_csv = Path(output_csv)
            output_csv.parent.mkdir(parents=True, exist_ok=True)

            with output_csv.open("w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)

                # Header
                writer.writerow(
                    [
                        "rank",
                        "win_rate",
                        "n_battles_total",
                        "n_wins",
                        "n_losses",
                        "n_ties",
                        "n_records",
                        "team_set_ids",
                        "source_counts_json",
                    ]
                )

                # Rows
                for rank, agg in enumerate(top, start=1):
                    team_set_ids_str = " ".join(agg.team_set_ids)
                    source_counts_json = json.dumps(agg.source_counts, ensure_ascii=False)

                    writer.writerow(
                        [
                            rank,
                            f"{agg.win_rate:.6f}",
                            agg.n_battles_total,
                            agg.n_wins,
                            agg.n_losses,
                            agg.n_ties,
                            agg.n_records,
                            team_set_ids_str,
                            source_counts_json,
                        ]
                    )

            typer.echo(f"Wrote CSV to: {output_csv}")

        # Optional YAML export of top teams
        if output_teams_yaml is not None:
            output_teams_yaml = Path(output_teams_yaml)
            output_teams_yaml.parent.mkdir(parents=True, exist_ok=True)

            # Validate set_ids against catalog
            catalog = SetCatalog.from_yaml()
            yaml_teams = []

            for rank, agg in enumerate(top, start=1):
                # Validate all set_ids exist in catalog
                missing_set_ids = [
                    sid for sid in agg.team_set_ids if sid not in catalog._entries
                ]

                if missing_set_ids:
                    typer.echo(
                        f"WARNING: Team rank #{rank} has missing set_ids in catalog: {missing_set_ids}",
                        err=True,
                    )
                    continue

                team_id = f"auto_top_{rank:04d}"
                description = (
                    f"Auto-discovered top team rank #{rank} "
                    f"(win_rate={agg.win_rate:.3f}, battles={agg.n_battles_total}, "
                    f"records={agg.n_records}, min_battles={min_battles}, "
                    f"min_records={min_records})"
                )

                yaml_teams.append(
                    {
                        "id": team_id,
                        "format": DEFAULT_FORMAT,
                        "description": description,
                        "set_ids": agg.team_set_ids,
                    }
                )

            if yaml_teams:
                with output_teams_yaml.open("w", encoding="utf-8") as f:
                    yaml.dump(
                        yaml_teams,
                        f,
                        default_flow_style=False,
                        sort_keys=False,
                        allow_unicode=True,
                    )
                typer.echo(f"Wrote YAML catalog to: {output_teams_yaml}")
                typer.echo(f"Exported {len(yaml_teams)} teams.")
            else:
                typer.echo(
                    "WARNING: No teams could be exported to YAML (all had missing set_ids)",
                    err=True,
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

