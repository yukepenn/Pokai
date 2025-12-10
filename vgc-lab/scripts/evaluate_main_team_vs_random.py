#!/usr/bin/env python3
"""CLI: Evaluate my main team against random opponents using RandomPlayerAI."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import DEFAULT_FORMAT, ensure_paths
from vgc_lab.eval import evaluate_team_vs_random
from vgc_lab.teams import import_text_to_packed

app = typer.Typer()

MY_MAIN_TEAM_IMPORT = """\
Amoonguss @ Sitrus Berry
Ability: Regenerator
Level: 50
Tera Type: Water
EVs: 220 HP / 164 Def / 4 SpA / 116 SpD / 4 Spe
Bold Nature
- Protect
- Sludge Bomb
- Spore
- Rage Powder

Flutter Mane @ Booster Energy
Ability: Protosynthesis
Level: 50
Tera Type: Fairy
EVs: 116 HP / 76 Def / 116 SpA / 4 SpD / 196 Spe
Modest Nature
IVs: 0 Atk
- Protect
- Moonblast
- Shadow Ball
- Dazzling Gleam

Tornadus @ Focus Sash
Ability: Prankster
Level: 50
Tera Type: Ghost
EVs: 36 HP / 12 Def / 204 SpA / 4 SpD / 252 Spe
Timid Nature
IVs: 0 Atk
- Protect
- Bleakwind Storm
- Tailwind
- Rain Dance

Incineroar @ Safety Goggles
Ability: Intimidate
Level: 50
Tera Type: Grass
EVs: 252 HP / 4 Atk / 84 Def / 92 SpD / 76 Spe
Careful Nature
- Fake Out
- Parting Shot
- Flare Blitz
- Knock Off

Landorus @ Life Orb
Ability: Sheer Force
Level: 50
Tera Type: Steel
EVs: 116 HP / 12 Def / 116 SpA / 12 SpD / 252 Spe
Modest Nature
IVs: 0 Atk
- Protect
- Earth Power
- Substitute
- Sludge Bomb

Urshifu-Rapid-Strike @ Choice Scarf
Ability: Unseen Fist
Level: 50
Tera Type: Water
EVs: 60 HP / 156 Atk / 4 Def / 124 SpD / 164 Spe
Adamant Nature
- Close Combat
- Surging Strikes
- Aqua Jet
- U-turn
"""


@app.command()
def main(
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
    team_as: str = typer.Option(
        "p1",
        "--team-as",
        help='Which side to place the fixed team on: "p1", "p2", or "both".',
    ),
    save_logs: bool = typer.Option(
        False,
        "--save-logs",
        help="Save individual battle logs to data/battles_raw/.",
    ),
):
    """Evaluate the main team against random opponents."""
    try:
        if save_logs:
            ensure_paths()
        packed = import_text_to_packed(MY_MAIN_TEAM_IMPORT, format_id=format_id)
        summary = evaluate_team_vs_random(
            team_packed=packed,
            format_id=format_id,
            n=n,
            team_as=team_as,
            save_logs=save_logs,
        )

        winrate = summary.n_wins / summary.n_battles if summary.n_battles > 0 else 0.0
        typer.echo(f"Format: {summary.format_id}")
        typer.echo(f"Team role: {summary.team_role}")
        typer.echo(f"Battles: {summary.n_battles}")
        typer.echo(f"Wins: {summary.n_wins}, Losses: {summary.n_losses}, Ties: {summary.n_ties}")
        typer.echo(f"Win rate: {winrate:.3f}")
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

