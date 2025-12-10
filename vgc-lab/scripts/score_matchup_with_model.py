#!/usr/bin/env python3
"""CLI: Score a matchup between two catalog teams with the trained matchup model."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import MODELS_DIR, PROJECT_ROOT
from vgc_lab.team_pool import TeamPool, TEAMS_CATALOG_PATH
from vgc_lab.team_matchup_model import (
    load_team_matchup_checkpoint,
    predict_matchup_win_prob,
)

# Default paths
DEFAULT_CKPT = MODELS_DIR / "team_matchup_mlp.pt"
AUTO_POOL_PATH = PROJECT_ROOT / "data" / "catalog" / "teams_regf_auto.yaml"

app = typer.Typer()


@app.command()
def main(
    team_a_id: str = typer.Argument(..., help="Team ID of team A from the catalog."),
    team_b_id: str = typer.Argument(..., help="Team ID of team B from the catalog."),
    ckpt_path: Path = typer.Option(
        str(DEFAULT_CKPT),
        "--ckpt",
        help="Path to the trained matchup model checkpoint.",
    ),
    teams_yaml: Optional[Path] = typer.Option(
        None,
        "--teams-yaml",
        help="Which catalog to load. Defaults to teams_regf_auto.yaml if exists, else teams_regf.yaml.",
    ),
    device: str = typer.Option("cpu", "--device"),
):
    """Score a matchup between two catalog teams using the trained matchup model."""
    try:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            typer.echo(f"ERROR: Checkpoint file not found: {ckpt_path}", err=True)
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
        model, vocab, model_cfg = load_team_matchup_checkpoint(ckpt_path, device=device)

        # Load team pool
        pool = TeamPool.from_yaml(teams_yaml)

        # Look up teams
        try:
            team_a = pool.get(team_a_id)
            team_b = pool.get(team_b_id)
        except KeyError as e:
            typer.echo(f"ERROR: Team ID not found in pool: {e}", err=True)
            raise typer.Exit(1)

        # Predict matchup
        try:
            pred = predict_matchup_win_prob(
                model=model,
                vocab=vocab,
                team_a_set_ids=team_a.set_ids,
                team_b_set_ids=team_b.set_ids,
                device=device,
            )
        except KeyError as e:
            typer.echo(
                f"ERROR: Set ID {e} not found in model vocabulary. "
                f"The model was trained on a different set catalog.",
                err=True,
            )
            raise typer.Exit(1)

        typer.echo(f"Team A ID: {team_a_id}")
        typer.echo(f"Team A set_ids: {', '.join(team_a.set_ids)}")
        typer.echo()
        typer.echo(f"Team B ID: {team_b_id}")
        typer.echo(f"Team B set_ids: {', '.join(team_b.set_ids)}")
        typer.echo()
        typer.echo(f"Predicted P(A wins vs B): {pred:.3f}")
        typer.echo(
            f"(Model vocab_size={model_cfg['vocab_size']}, "
            f"embed_dim={model_cfg['embed_dim']})"
        )
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

