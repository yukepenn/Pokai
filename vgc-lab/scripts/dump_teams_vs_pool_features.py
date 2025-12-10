#!/usr/bin/env python3
"""CLI: Dump teams_vs_pool features to CSV for ML experiments."""

from __future__ import annotations

import csv
from pathlib import Path

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import PROJECT_ROOT
from vgc_lab.team_features import (
    iter_teams_vs_pool_records,
    team_bag_of_sets_feature,
    build_default_vocab,
)

DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"
DEFAULT_INPUT = DATASETS_ROOT / "teams_vs_pool" / "teams_vs_pool.jsonl"
DEFAULT_OUTPUT = DATASETS_ROOT / "teams_vs_pool" / "features.csv"

app = typer.Typer()


@app.command()
def main(
    input_path: str = typer.Option(
        None,
        "--input",
        help="Path to teams_vs_pool.jsonl (default: datasets root).",
    ),
    output_csv: str = typer.Option(
        None,
        "--output",
        help="Output CSV path (default: data/datasets/teams_vs_pool/features.csv).",
    ),
):
    """
    Convert teams_vs_pool JSONL records into a CSV of features for ML.
    """
    try:
        # Resolve paths
        if input_path is None:
            input_path = DEFAULT_INPUT
        else:
            input_path = Path(input_path)

        if output_csv is None:
            output_csv = DEFAULT_OUTPUT
        else:
            output_csv = Path(output_csv)

        if not input_path.exists():
            typer.echo(f"ERROR: Input file not found: {input_path}", err=True)
            raise typer.Exit(1)

        vocab = build_default_vocab()
        V = len(vocab.idx_to_id)

        records = list(iter_teams_vs_pool_records(input_path))

        if not records:
            typer.echo("WARNING: No records found in input file", err=True)
            raise typer.Exit(1)

        # Ensure output directory exists
        output_csv.parent.mkdir(parents=True, exist_ok=True)

        with output_csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)

            # Header
            header = ["win_rate", "n_battles", "source", "set_id_indices"]
            header += [f"feat_{i}" for i in range(V)]
            writer.writerow(header)

            for rec in records:
                indices = vocab.encode_ids(rec.team_set_ids)
                feat_vec = team_bag_of_sets_feature(rec.team_set_ids, vocab)
                indices_str = " ".join(str(i) for i in indices)
                row = [rec.win_rate, rec.n_battles_total, rec.source, indices_str]
                row += feat_vec
                writer.writerow(row)

        typer.echo(f"Wrote {len(records)} records to {output_csv}")
        typer.echo(f"Feature vector dimension: {V}")
    except KeyError as e:
        typer.echo(f"ERROR: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

