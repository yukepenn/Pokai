#!/usr/bin/env python3
"""CLI: Score a catalog-defined team with the trained value model."""

from __future__ import annotations

from pathlib import Path

import torch
import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import PROJECT_ROOT
from vgc_lab.team_pool import TeamPool
from vgc_lab.team_value_model import load_team_value_checkpoint

DEFAULT_CKPT = PROJECT_ROOT / "data" / "models" / "team_value_mlp.pt"

app = typer.Typer()


@app.command()
def main(
    team_id: str = typer.Argument(..., help="Team ID from teams_regf.yaml."),
    ckpt_path: Path = typer.Option(
        str(DEFAULT_CKPT),
        "--ckpt",
        help="Path to the trained model checkpoint.",
    ),
    device: str = typer.Option("cpu", "--device"),
):
    """Score a catalog team using the trained value model."""
    try:
        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            typer.echo(f"ERROR: Checkpoint file not found: {ckpt_path}", err=True)
            raise typer.Exit(1)

        # Load checkpoint
        model, vocab, model_cfg = load_team_value_checkpoint(ckpt_path, device=device)

        # Load team pool and resolve team_id -> set_ids
        pool = TeamPool.from_yaml()  # uses default teams_regf.yaml
        team_def = pool.get(team_id)
        set_ids = team_def.set_ids

        # Encode team using the vocab from the checkpoint
        try:
            indices = vocab.encode_ids(set_ids)  # should be 6 indices
        except KeyError as e:
            typer.echo(
                f"ERROR: Set ID {e} from team '{team_id}' not found in model vocabulary. "
                f"The model was trained on a different set catalog.",
                err=True,
            )
            raise typer.Exit(1)

        tensor = torch.tensor([indices], dtype=torch.long, device=device)  # [1, team_size]

        with torch.no_grad():
            pred = model(tensor).item()

        typer.echo(f"Team ID: {team_id}")
        typer.echo(f"Set IDs: {', '.join(set_ids)}")
        typer.echo(f"Predicted win_rate vs pool: {pred:.3f}")
        typer.echo(
            f"(Model vocab_size={model_cfg['vocab_size']}, "
            f"embed_dim={model_cfg['embedding_dim']})"
        )
    except KeyError as e:
        typer.echo(f"ERROR: Team ID not found in pool: {e}", err=True)
        raise typer.Exit(1)
    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

