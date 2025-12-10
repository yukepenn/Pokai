#!/usr/bin/env python3
"""
Pack a team from export format to packed format.

Usage:
    python scripts/pack_team_cli.py --file my_team.txt
    python scripts/pack_team_cli.py < export.txt > packed.txt
"""

import sys
from pathlib import Path

import typer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.showdown_cli import pack_team

app = typer.Typer(help="Convert Pokemon Showdown team from export to packed format")


@app.command()
def main(file: Path = typer.Option(None, "--file", "-f", help="Input file (or read from stdin)")):
    """
    Convert a team from export format to packed format.

    If --file is provided, read from that file. Otherwise, read from stdin.
    """
    try:
        if file:
            if not file.exists():
                typer.echo(f"Error: File not found: {file}", err=True)
                raise typer.Exit(1)
            input_text = file.read_text(encoding="utf-8")
        else:
            input_text = sys.stdin.read()

        if not input_text.strip():
            typer.echo("Error: No input provided", err=True)
            raise typer.Exit(1)

        packed = pack_team(input_text)
        print(packed)

    except RuntimeError as e:
        error_msg = str(e)
        if "Node.js not found" in error_msg:
            typer.echo("Error: Node.js not found. Please install Node.js 16+ from https://nodejs.org/", err=True)
        else:
            typer.echo(f"Error: {error_msg}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

