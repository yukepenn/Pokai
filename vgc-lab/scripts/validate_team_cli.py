#!/usr/bin/env python3
"""
Validate a Pokemon Showdown team against a format.

Usage:
    python scripts/validate_team_cli.py --file my_team.txt
    python scripts/validate_team_cli.py < team.txt
"""

import sys
from pathlib import Path

import typer

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import DEFAULT_FORMAT
from vgc_lab.showdown_cli import pack_team, validate_team

app = typer.Typer(help="Validate a Pokemon Showdown team")


@app.command()
def main(
    file: Path = typer.Option(None, "--file", "-f", help="Input file (or read from stdin)"),
    format_id: str = typer.Option(
        DEFAULT_FORMAT, "--format", "-F", help="Format ID to validate against"
    ),
):
    """
    Validate a team against a format.

    If the input looks like export format (contains '@' or 'EVs:'), it will be
    converted to packed format first.
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

        # Detect if input is export format
        looks_like_export = "@" in input_text or "EVs:" in input_text or "Ability:" in input_text

        if looks_like_export:
            typer.echo("Detected export format, converting to packed...", err=True)
            packed_team = pack_team(input_text)
        else:
            packed_team = input_text.strip()

        # Validate
        validation_output = validate_team(packed_team, format_id)

        if validation_output:
            typer.echo("Validation FAILED:", err=True)
            typer.echo(validation_output, err=True)
            raise typer.Exit(1)
        else:
            typer.echo(f"Team is valid for format: {format_id}")
            raise typer.Exit(0)

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

