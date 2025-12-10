#!/usr/bin/env python3
"""CLI: Suggest counter teams vs a meta using the matchup model."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import MODELS_DIR, PROJECT_ROOT
from vgc_lab.matchup_meta_analysis import (
    MetaDistribution,
    MetaTeam,
    build_uniform_meta,
    normalize_meta,
    rank_teams_vs_meta,
)
from vgc_lab.team_matchup_model import load_team_matchup_checkpoint
from vgc_lab.team_pool import TeamPool, TEAMS_CATALOG_PATH

# Default paths
DEFAULT_CKPT = MODELS_DIR / "team_matchup_mlp.pt"
AUTO_POOL_PATH = PROJECT_ROOT / "data" / "catalog" / "teams_regf_auto.yaml"

app = typer.Typer()


@app.command()
def main(
    teams_yaml: Optional[Path] = typer.Option(
        None,
        "--teams-yaml",
        help="Catalog defining both meta teams and candidate teams. "
        "Defaults to teams_regf_auto.yaml if exists, else teams_regf.yaml.",
    ),
    ckpt_path: Path = typer.Option(
        str(DEFAULT_CKPT),
        "--ckpt",
        help="Path to the trained matchup model checkpoint.",
    ),
    meta_json: Optional[Path] = typer.Option(
        None,
        "--meta-json",
        help="Optional JSON file describing the meta distribution. "
        "If not provided, uses uniform meta over all teams in teams-yaml.",
    ),
    candidate_prefix: Optional[str] = typer.Option(
        None,
        "--candidate-prefix",
        help="If provided, restrict candidates to team IDs starting with this prefix "
        "(e.g., 'auto_top_' or 'my_').",
    ),
    top_k: int = typer.Option(
        10,
        "--top-k",
        help="Number of best counter candidates to show.",
    ),
    device: str = typer.Option("cpu", "--device"),
):
    """
    Suggest counter teams vs a meta using the matchup model.

    Finds candidate teams that maximize expected win_rate vs the given meta distribution.
    """
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

        # Load team pool
        pool = TeamPool.from_yaml(teams_yaml)

        # Load matchup model
        model, vocab, model_cfg = load_team_matchup_checkpoint(ckpt_path, device=device)

        # Build meta distribution
        if meta_json is not None:
            meta_json = Path(meta_json)
            if not meta_json.exists():
                typer.echo(f"ERROR: Meta JSON file not found: {meta_json}", err=True)
                raise typer.Exit(1)

            with meta_json.open("r", encoding="utf-8") as f:
                meta_data = json.load(f)

            # Parse meta entries
            entries = []
            for entry_data in meta_data.get("entries", []):
                team_id = entry_data.get("team_id")
                weight = float(entry_data.get("weight", 1.0))

                # Validate team exists
                try:
                    pool.get(team_id)
                    entries.append(MetaTeam(team_id=team_id, weight=weight))
                except KeyError:
                    typer.echo(
                        f"WARNING: Team '{team_id}' from meta JSON not found in pool, skipping",
                        err=True,
                    )
                    continue

            if not entries:
                typer.echo(
                    "ERROR: No valid teams found in meta JSON after filtering", err=True
                )
                raise typer.Exit(1)

            meta = MetaDistribution(entries=entries)
            meta = normalize_meta(meta)
            typer.echo(f"Meta: {len(meta.entries)} teams (from {meta_json})")
        else:
            # Use uniform meta over all teams
            all_team_ids = pool.ids()
            meta = build_uniform_meta(pool, all_team_ids)
            meta = normalize_meta(meta)
            typer.echo(f"Meta: {len(meta.entries)} teams (uniform)")

        # Build candidate list
        all_candidate_ids = pool.ids()

        if candidate_prefix is not None:
            candidate_ids = [
                tid for tid in all_candidate_ids if tid.startswith(candidate_prefix)
            ]
            typer.echo(f"Candidates filtered by prefix '{candidate_prefix}': {len(candidate_ids)} teams")
        else:
            candidate_ids = all_candidate_ids

        if not candidate_ids:
            typer.echo("ERROR: No candidate teams found", err=True)
            raise typer.Exit(1)

        typer.echo(f"Evaluating {len(candidate_ids)} candidate teams...")
        typer.echo()

        # Rank candidates vs meta
        ranked = rank_teams_vs_meta(
            candidate_team_ids=candidate_ids,
            meta=meta,
            team_pool=pool,
            model=model,
            vocab=vocab,
            device=device,
        )

        if not ranked:
            typer.echo("ERROR: No valid rankings computed", err=True)
            raise typer.Exit(1)

        # Display top-K
        typer.echo(f"Top {min(top_k, len(ranked))} counter candidates:")
        typer.echo()

        for rank, (team_id, expected_win) in enumerate(ranked[:top_k], start=1):
            typer.echo(
                f"  {rank}. {team_id:<30} E[win_rate vs meta] = {expected_win:.3f}"
            )

        typer.echo()
        typer.echo(
            "Note: These are model-based estimates (not guaranteed, but useful for guiding "
            "further real evaluations)."
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

