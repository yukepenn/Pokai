#!/usr/bin/env python3
"""CLI: Model-guided team search vs pool and dataset append."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import random

import torch
import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import PROJECT_ROOT, ensure_paths
from vgc_lab.set_catalog import SetCatalog
from vgc_lab.team_pool import TeamPool
from vgc_lab.team_value_model import load_team_value_checkpoint
from vgc_lab.team_search import evaluate_top_model_candidates_against_pool
from vgc_lab.teams_vs_pool_data import (
    append_result_to_dataset,
    TEAMS_VS_POOL_JSONL_PATH,
)

DEFAULT_CKPT = PROJECT_ROOT / "data" / "models" / "team_value_mlp.pt"

app = typer.Typer()


@app.command()
def main(
    ckpt_path: Path = typer.Option(
        str(DEFAULT_CKPT),
        "--ckpt",
        help="Path to the trained team value model checkpoint.",
    ),
    n_proposals: int = typer.Option(
        500,
        "--n-proposals",
        help="Number of distinct candidate teams to propose with the model.",
    ),
    top_k: int = typer.Option(
        20,
        "--top-k",
        help="Number of top candidates (by predicted win_rate) to evaluate vs the pool.",
    ),
    n_opponents: int = typer.Option(
        5,
        "--n-opponents",
        help="Number of opponent teams sampled from the pool for each candidate.",
    ),
    n_per_opponent: int = typer.Option(
        5,
        "--n-per",
        help="Number of battles vs each opponent.",
    ),
    seed: int = typer.Option(123, "--seed", help="Random seed for reproducibility."),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device",
        help='"cpu" or "cuda". Defaults to CUDA if available.',
    ),
    source_tag: str = typer.Option(
        "model_guided_v1",
        "--source",
        help="Source tag for dataset records (e.g., 'model_guided_v1').",
    ),
    jsonl_path: Optional[Path] = typer.Option(
        None,
        "--jsonl",
        help="Optional explicit path to teams_vs_pool.jsonl (default uses TEAMS_VS_POOL_JSONL_PATH).",
    ),
):
    """
    Run a single model-guided search iteration:

    - Load the trained team value model.
    - Use it to propose n_proposals candidate teams from the SetCatalog.
    - Take top_k candidates by predicted win_rate.
    - Evaluate each candidate vs TeamPool in the real environment.
    - Append results to teams_vs_pool.jsonl with source=source_tag.
    """
    try:
        ensure_paths()

        ckpt_path = Path(ckpt_path)
        if not ckpt_path.exists():
            typer.echo(f"ERROR: Checkpoint file not found: {ckpt_path}", err=True)
            raise typer.Exit(1)

        rng = random.Random(seed)

        # Load model and vocab
        model, vocab, model_cfg = load_team_value_checkpoint(ckpt_path, device=device)
        typer.echo(
            f"Loaded model from {ckpt_path} "
            f"(vocab_size={model_cfg['vocab_size']}, "
            f"embed_dim={model_cfg['embedding_dim']})"
        )

        # Load catalog and pool
        catalog = SetCatalog.from_yaml()
        pool = TeamPool.from_yaml()

        typer.echo(
            f"Proposing {n_proposals} candidates, evaluating top {top_k} vs pool "
            f"({n_opponents} opponents, {n_per_opponent} battles each)..."
        )

        # Run model-guided search
        pairs = evaluate_top_model_candidates_against_pool(
            model=model,
            vocab=vocab,
            catalog=catalog,
            pool=pool,
            n_proposals=n_proposals,
            top_k=top_k,
            n_opponents=n_opponents,
            n_per_opponent=n_per_opponent,
            rng=rng,
            device=device,
        )

        path = jsonl_path or TEAMS_VS_POOL_JSONL_PATH
        typer.echo(f"Appending results to dataset: {path}")

        for i, (proposal, result) in enumerate(pairs, start=1):
            append_result_to_dataset(result, source=source_tag, jsonl_path=path)
            typer.echo(
                f"[{i}/{len(pairs)}] pred={proposal.pred_win_rate:.3f}, "
                f"actual={result.win_rate:.3f}, "
                f"sets={', '.join(proposal.set_ids[:2])}..."
            )

        typer.echo("Model-guided search iteration complete.")
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

