#!/usr/bin/env python3
"""CLI: Dump battle trajectories from battle JSON files to JSONL."""

from __future__ import annotations

from pathlib import Path

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.battle_trajectory_data import (
    BATTLES_JSON_DIR,
    TRAJECTORY_JSONL_PATH,
    dump_trajectories_to_jsonl,
)

app = typer.Typer()


@app.command()
def main(
    battles_json_dir: Path = typer.Option(
        str(BATTLES_JSON_DIR),
        "--battles-json-dir",
        help="Directory containing battle JSON files.",
    ),
    out_path: Path = typer.Option(
        str(TRAJECTORY_JSONL_PATH),
        "--out-path",
        help="Output JSONL file path for trajectories.",
    ),
):
    """
    Parse battle JSON files and dump turn-by-turn trajectories to JSONL.
    """
    try:
        battles_json_dir = Path(battles_json_dir)
        out_path = Path(out_path)

        if not battles_json_dir.exists():
            typer.echo(f"ERROR: Battles JSON directory not found: {battles_json_dir}", err=True)
            raise typer.Exit(1)

        typer.echo(f"Processing battles from: {battles_json_dir}")
        typer.echo(f"Output trajectory file: {out_path}")

        n_battles, n_steps = dump_trajectories_to_jsonl(battles_json_dir, out_path)

        typer.echo(f"\nProcessed {n_battles} battles")
        typer.echo(f"Wrote {n_steps} battle steps to {out_path}")
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

