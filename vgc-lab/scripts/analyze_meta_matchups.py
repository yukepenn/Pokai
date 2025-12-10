#!/usr/bin/env python3
"""CLI: Analyze a team's matchup vs a meta using the matchup model."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import MODELS_DIR, PROJECT_ROOT
from vgc_lab.matchup_meta_analysis import (
    build_uniform_meta,
    normalize_meta,
    predict_matchup_win_prob,
)
from vgc_lab.team_matchup_model import load_team_matchup_checkpoint
from vgc_lab.team_pool import TeamPool, TEAMS_CATALOG_PATH

# Default paths
DEFAULT_CKPT = MODELS_DIR / "team_matchup_mlp.pt"
AUTO_POOL_PATH = PROJECT_ROOT / "data" / "catalog" / "teams_regf_auto.yaml"

app = typer.Typer()


@app.command()
def main(
    team_id: str = typer.Argument(..., help="Team ID to analyze (e.g., 'my_main_team' or 'auto_top_0001')."),
    teams_yaml: Optional[Path] = typer.Option(
        None,
        "--teams-yaml",
        help="Which catalog to load. Defaults to teams_regf_auto.yaml if exists, else teams_regf.yaml.",
    ),
    ckpt_path: Path = typer.Option(
        str(DEFAULT_CKPT),
        "--ckpt",
        help="Path to the trained matchup model checkpoint.",
    ),
    top_k_best: int = typer.Option(
        10,
        "--top-k-best",
        help="Number of best matchups to show.",
    ),
    top_k_worst: int = typer.Option(
        10,
        "--top-k-worst",
        help="Number of worst matchups to show.",
    ),
    device: str = typer.Option("cpu", "--device"),
):
    """
    Analyze a team's matchups vs a meta using the matchup model.

    Computes:
    - Overall expected win rate vs the meta
    - Best individual matchups (highest P(win))
    - Worst individual matchups (lowest P(win))
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

        # Validate focus team exists
        try:
            focus_team = pool.get(team_id)
        except KeyError:
            typer.echo(f"ERROR: Team ID '{team_id}' not found in pool", err=True)
            raise typer.Exit(1)

        # Load matchup model
        model, vocab, model_cfg = load_team_matchup_checkpoint(ckpt_path, device=device)

        # Build uniform meta over all teams in the pool
        all_team_ids = pool.ids()
        meta = build_uniform_meta(pool, all_team_ids)
        meta = normalize_meta(meta)

        typer.echo(f"Focus team: {team_id}")
        typer.echo(f"Meta size: {len(meta.entries)} teams")
        typer.echo()

        # Compute matchups vs each opponent
        matchup_results: List[Tuple[str, float]] = []

        for meta_entry in meta.entries:
            opp_id = meta_entry.team_id
            if opp_id == team_id:
                continue  # Skip self-matchup

            try:
                opp_team = pool.get(opp_id)
                p_win = predict_matchup_win_prob(
                    model=model,
                    vocab=vocab,
                    team_a_set_ids=focus_team.set_ids,
                    team_b_set_ids=opp_team.set_ids,
                    device=device,
                )
                matchup_results.append((opp_id, p_win))
            except KeyError:
                continue

        if not matchup_results:
            typer.echo("ERROR: No valid matchups found", err=True)
            raise typer.Exit(1)

        # Compute overall expected win rate (average of all matchups)
        overall_win_rate = sum(p for _, p in matchup_results) / len(matchup_results)

        typer.echo(f"Expected win_rate vs meta (model): {overall_win_rate:.3f}")
        typer.echo()

        # Sort by win probability
        matchup_results.sort(key=lambda x: x[1], reverse=True)

        # Show best matchups
        typer.echo(f"Best matchups (top {top_k_best}):")
        for rank, (opp_id, p_win) in enumerate(matchup_results[:top_k_best], start=1):
            typer.echo(f"  {rank}. {opp_id:<30} P(win) = {p_win:.3f}")
        typer.echo()

        # Show worst matchups
        typer.echo(f"Worst matchups (bottom {top_k_worst}):")
        for rank, (opp_id, p_win) in enumerate(
            reversed(matchup_results[-top_k_worst:]), start=1
        ):
            typer.echo(f"  {rank}. {opp_id:<30} P(win) = {p_win:.3f}")

    except Exception as e:
        typer.echo(f"Unexpected error: {e}", err=True)
        import traceback
        traceback.print_exc()
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

