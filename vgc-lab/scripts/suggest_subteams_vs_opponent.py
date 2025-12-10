#!/usr/bin/env python3
"""CLI: Suggest 4-of-6 subteams vs an opponent using the matchup model."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import MODELS_DIR, PROJECT_ROOT
from vgc_lab.subteam_selection import rank_subteams_vs_opponent
from vgc_lab.team_matchup_model import load_team_matchup_checkpoint
from vgc_lab.team_pool import TeamPool, TEAMS_CATALOG_PATH

# Default paths
DEFAULT_CKPT = MODELS_DIR / "team_matchup_mlp.pt"
AUTO_POOL_PATH = PROJECT_ROOT / "data" / "catalog" / "teams_regf_auto.yaml"

app = typer.Typer()


@app.command()
def main(
    our_team_id: str = typer.Argument(..., help="Our team ID from the catalog."),
    opp_team_id: str = typer.Argument(..., help="Opponent team ID from the catalog."),
    teams_yaml: Optional[Path] = typer.Option(
        None,
        "--teams-yaml",
        help="Which catalog to load. Defaults to teams_regf_auto.yaml if exists, else teams_regf.yaml.",
    ),
    ckpt: Path = typer.Option(
        str(DEFAULT_CKPT),
        "--ckpt",
        help="Path to the trained matchup model checkpoint.",
    ),
    max_subteams: int = typer.Option(
        15,
        "--max-subteams",
        help="Maximum number of subteams to evaluate (default 15 for 6 choose 4).",
    ),
    device: str = typer.Option("cpu", "--device"),
):
    """
    Suggest best 4-of-6 subteams from our team vs an opponent team.
    """
    try:
        ckpt = Path(ckpt)
        if not ckpt.exists():
            typer.echo(f"ERROR: Checkpoint file not found: {ckpt}", err=True)
            raise typer.Exit(1)

        # Determine teams YAML path
        if teams_yaml is None:
            if AUTO_POOL_PATH.exists():
                teams_yaml = AUTO_POOL_PATH
            else:
                teams_yaml = TEAMS_CATALOG_PATH

        teams_yaml = Path(teams_yaml)
        if not teams_yaml.exists():
            typer.echo(f"ERROR: Team pool file not found: {teams_yaml}", err=True)
            raise typer.Exit(1)

        # Load checkpoint
        model, vocab, model_cfg = load_team_matchup_checkpoint(ckpt, device=device)

        # Load team pool
        pool = TeamPool.from_yaml(teams_yaml)

        typer.echo(f"Our team: {our_team_id}")
        typer.echo(f"Opponent team: {opp_team_id}")
        typer.echo()

        # Rank subteams
        ranked = rank_subteams_vs_opponent(
            our_team_id=our_team_id,
            opp_team_id=opp_team_id,
            team_pool=pool,
            model=model,
            vocab=vocab,
            max_subteams=max_subteams,
            device=device,
        )

        if not ranked:
            typer.echo("ERROR: No valid subteams could be evaluated", err=True)
            raise typer.Exit(1)

        typer.echo(f"Best 4-of-6 subteams (approximate P(win)):")
        typer.echo()

        for rank, (subteam_tuple, p_win) in enumerate(ranked, start=1):
            subteam_list = list(subteam_tuple)
            typer.echo(f"  {rank}. {subteam_list}  P(win) = {p_win:.3f}")

    except KeyError as e:
        typer.echo(f"ERROR: Team ID not found: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

