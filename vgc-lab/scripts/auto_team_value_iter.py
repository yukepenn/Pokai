#!/usr/bin/env python3
"""CLI: Run multi-iteration team value training + model-guided search."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import random

import torch
import typer

# Add src to PYTHONPATH
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import ensure_paths, MODELS_DIR
from vgc_lab.set_catalog import SetCatalog
from vgc_lab.team_pool import TeamPool
from vgc_lab.team_value_model import TrainingConfig, train_team_value_model
from vgc_lab.team_search import (
    CandidateTeamResult,
    evaluate_team_against_pool,
    evaluate_top_model_candidates_against_pool,
    mutate_team_set_ids,
    random_search_over_pool,
)
from vgc_lab.teams_vs_pool_data import (
    TEAMS_VS_POOL_JSONL_PATH,
    append_result_to_dataset,
)

app = typer.Typer()


@app.command()
def main(
    n_iters: int = typer.Option(
        5,
        "--n-iters",
        help="Number of outer iterations of train + search.",
    ),
    # Data / search workload
    init_random_samples: int = typer.Option(
        0,
        "--init-random-samples",
        help="Number of random candidates to evaluate once before iterations (if dataset is very small).",
    ),
    per_iter_random_samples: int = typer.Option(
        0,
        "--per-iter-random-samples",
        help="Number of random candidates to evaluate per iteration (exploration).",
    ),
    per_iter_model_proposals: int = typer.Option(
        500,
        "--per-iter-model-proposals",
        help="Number of candidates proposed by the model per iteration.",
    ),
    per_iter_top_k: int = typer.Option(
        40,
        "--per-iter-top-k",
        help="Number of top model-proposed candidates to evaluate per iteration.",
    ),
    # Mutation settings
    n_mutation_bases: int = typer.Option(
        0,
        "--n-mutation-bases",
        help="Number of top model-guided candidates to use as bases for mutation (0 = disable).",
    ),
    n_mutations_per_base: int = typer.Option(
        1,
        "--n-mutations-per-base",
        help="Number of mutated variants to create per base team when mutation is enabled.",
    ),
    # Pool evaluation settings
    n_opponents: int = typer.Option(
        5,
        "--n-opponents",
        help="Number of opponent teams sampled from the pool for each evaluation.",
    ),
    n_per_opponent: int = typer.Option(
        5,
        "--n-per",
        help="Number of battles per opponent.",
    ),
    # Training hyperparameters
    epochs_per_iter: int = typer.Option(
        20,
        "--epochs-per-iter",
        help="Number of training epochs per iteration.",
    ),
    batch_size: int = typer.Option(
        64,
        "--batch-size",
        help="Batch size for training.",
    ),
    lr: float = typer.Option(
        3e-4,
        "--lr",
        help="Learning rate for training.",
    ),
    # Device and randomness
    seed: int = typer.Option(
        123,
        "--seed",
        help="Random seed for reproducibility.",
    ),
    device: str = typer.Option(
        "cuda" if torch.cuda.is_available() else "cpu",
        "--device",
        help='"cpu" or "cuda". Defaults to CUDA if available.',
    ),
):
    """
    Run multi-iteration team value training + model-guided search:

    For each iteration:

    1) Train (or re-train) the team value model on the current teams_vs_pool.jsonl.
    2) Use the model to propose many candidate teams and evaluate the top-K vs the TeamPool.
    3) Optionally: add some random-search candidates.
    4) Optionally: take some of the best proposals and evaluate mutated variants.

    All evaluation results are appended to teams_vs_pool.jsonl so that subsequent
    iterations train on a richer dataset.
    """
    try:
        ensure_paths()
        rng = random.Random(seed)

        typer.echo(f"Using device: {device}")
        typer.echo(f"Dataset path: {TEAMS_VS_POOL_JSONL_PATH}")

        catalog = SetCatalog.from_yaml()
        pool = TeamPool.from_yaml()

        # Optional initial random exploration
        if init_random_samples > 0:
            typer.echo(f"\n=== Initial Random Search ===")
            typer.echo(f"Evaluating {init_random_samples} random candidates")

            random_results = random_search_over_pool(
                catalog=catalog,
                pool=pool,
                n_candidates=init_random_samples,
                n_opponents=n_opponents,
                n_battles_per_opponent=n_per_opponent,
                rng=rng,
            )

            for i, res in enumerate(random_results, start=1):
                append_result_to_dataset(res, source="auto_init_random")
                typer.echo(
                    f"[Init Random {i}/{len(random_results)}] "
                    f"win_rate={res.win_rate:.3f}"
                )

        # Outer loop over iterations
        for iter_idx in range(1, n_iters + 1):
            typer.echo(f"\n{'='*50}")
            typer.echo(f"Iteration {iter_idx}/{n_iters}")
            typer.echo(f"{'='*50}")

            # Training step
            ckpt_path = MODELS_DIR / f"team_value_mlp_iter{iter_idx:03d}.pt"

            train_cfg = TrainingConfig(
                jsonl_path=TEAMS_VS_POOL_JSONL_PATH,
                out_path=ckpt_path,
                epochs=epochs_per_iter,
                batch_size=batch_size,
                lr=lr,
                min_battles=3,
                include_sources=None,  # Include all sources by default
                device=device,
                seed=seed + iter_idx,  # Different seed per iteration
            )

            typer.echo(f"[Iter {iter_idx}] Training model -> {ckpt_path}")
            model, vocab, train_meta = train_team_value_model(train_cfg, device=device)

            typer.echo(
                f"[Iter {iter_idx}] train_loss={train_meta.get('train_loss', 0.0):.4f}, "
                f"val_loss={train_meta.get('val_loss', 0.0):.4f}, "
                f"val_corr={train_meta.get('val_corr', 0.0):.4f}"
            )

            # Model-guided search step
            typer.echo(
                f"[Iter {iter_idx}] Model-guided proposals: {per_iter_model_proposals} candidates, "
                f"top_k={per_iter_top_k}, n_opponents={n_opponents}, n_per={n_per_opponent}"
            )

            proposal_result_pairs = evaluate_top_model_candidates_against_pool(
                model=model,
                vocab=vocab,
                catalog=catalog,
                pool=pool,
                n_proposals=per_iter_model_proposals,
                top_k=per_iter_top_k,
                n_opponents=n_opponents,
                n_per_opponent=n_per_opponent,
                rng=rng,
                device=device,
            )

            for i, (proposal, result) in enumerate(proposal_result_pairs, start=1):
                append_result_to_dataset(
                    result, source=f"auto_iter_{iter_idx}_model"
                )
                typer.echo(
                    f"[Iter {iter_idx}] [Model {i}/{len(proposal_result_pairs)}] "
                    f"pred={proposal.pred_win_rate:.3f}, "
                    f"actual={result.win_rate:.3f}"
                )

            # Optional mutation step
            if n_mutation_bases > 0 and n_mutations_per_base > 0:
                typer.echo(
                    f"[Iter {iter_idx}] Mutation: bases={n_mutation_bases}, "
                    f"mutations_per_base={n_mutations_per_base}"
                )

                # Use the already-selected top model-guided proposals as bases
                base_proposals = [p for (p, _) in proposal_result_pairs][
                    :n_mutation_bases
                ]

                seen_keys = set()  # to avoid duplicated mutated teams
                mutation_count = 0

                for base_idx, base in enumerate(base_proposals, start=1):
                    for m in range(n_mutations_per_base):
                        mutated_set_ids = mutate_team_set_ids(
                            base.set_ids,
                            catalog=catalog,
                            rng=rng,
                        )
                        key = tuple(sorted(mutated_set_ids))
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        # Evaluate mutated team vs pool
                        mut_summary = evaluate_team_against_pool(
                            mutated_set_ids,
                            pool=pool,
                            n_opponents=n_opponents,
                            n_battles_per_opponent=n_per_opponent,
                            catalog=catalog,
                        )

                        mut_result = CandidateTeamResult(
                            team_set_ids=mutated_set_ids, pool_summary=mut_summary
                        )

                        append_result_to_dataset(
                            mut_result,
                            source=f"auto_iter_{iter_idx}_mut",
                        )
                        mutation_count += 1
                        typer.echo(
                            f"[Iter {iter_idx}] [Mut base={base_idx}, m={m+1}] "
                            f"actual={mut_result.win_rate:.3f}"
                        )

                typer.echo(
                    f"[Iter {iter_idx}] Mutation complete: {mutation_count} unique variants evaluated"
                )

            # Optional per-iteration random exploration
            if per_iter_random_samples > 0:
                typer.echo(
                    f"[Iter {iter_idx}] Random exploration: {per_iter_random_samples} candidates"
                )

                random_results = random_search_over_pool(
                    catalog=catalog,
                    pool=pool,
                    n_candidates=per_iter_random_samples,
                    n_opponents=n_opponents,
                    n_battles_per_opponent=n_per_opponent,
                    rng=rng,
                )

                for res in random_results:
                    append_result_to_dataset(
                        res, source=f"auto_iter_{iter_idx}_random"
                    )

                typer.echo(
                    f"[Iter {iter_idx}] Random exploration: {len(random_results)} candidates evaluated"
                )

        typer.echo(f"\n{'='*50}")
        typer.echo("Auto-iteration complete!")
        typer.echo(f"Final dataset: {TEAMS_VS_POOL_JSONL_PATH}")
        typer.echo(f"{'='*50}")

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

