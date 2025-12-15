"""High-level pipeline functions for vgc-lab workflows."""

from __future__ import annotations

import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional

# Import torch early to check availability
try:
    import torch
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    CUDA_AVAILABLE = False


def _resolve_device(device: str | None) -> str:
    """
    Resolve the effective device.

    - If device is None or empty, use 'cuda' when available, else 'cpu'.
    - Otherwise, return device as-is.
    """
    if not device:
        return "cuda" if CUDA_AVAILABLE else "cpu"
    return device

from .config import DEFAULT_FORMAT, MODELS_DIR, PROJECT_ROOT, ensure_paths
from .dataio import (
    FULL_BATTLE_DATASET_PATH,
    FullBattleRecord,
    append_full_battle_record,
    ensure_full_battle_dataset_dir,
    save_raw_log,
)
from .showdown import (
    DATASET_PATH,
    PreviewOutcomeRecord,
    append_preview_outcome_record,
    append_snapshot_to_dataset,
    ensure_dataset_dir,
    ensure_preview_outcome_dataset_dir,
    extract_lineups_from_log,
    generate_random_team,
    parse_team_preview_snapshot,
    run_random_selfplay,
    run_random_selfplay_json,
    simulate_battle,
)
from .dataio import (
    MATCHUP_JSONL_PATH,
    TeamMatchupRecord,
    append_matchup_record,
    ensure_matchup_dir,
)
from .dataio import (
    TEAMS_VS_POOL_JSONL_PATH,
    append_result_to_dataset,
)
from .analysis import (
    CandidateTeamResult,
    evaluate_team_against_pool,
    evaluate_top_model_candidates_against_pool,
    mutate_team_set_ids,
    random_search_over_pool,
)
from .catalog import (
    SetCatalog,
    TEAMS_CATALOG_PATH,
    TeamPool,
    build_team_import_from_set_ids,
    evaluate_catalog_team_vs_random,
)
from .analysis import (
    evaluate_team_vs_team,
    load_and_aggregate_teams_vs_pool,
)
from .models import (
    MatchupTrainingConfig,
    TrainingConfig,
    train_team_matchup_model as _train_team_matchup_model_impl,
    train_team_value_model as _train_team_value_model_impl,
)
from .analysis import (
    MetaDistribution,
    MetaTeam,
    build_uniform_meta,
    normalize_meta,
    rank_teams_vs_meta,
)
from .models import (
    load_team_matchup_checkpoint,
    load_team_value_checkpoint,
    predict_matchup_win_prob,
)
from .catalog import import_text_to_packed
from .showdown import pack_team as _pack_team, validate_team as _validate_team
from .dataio import (
    BattleResult,
    count_turns,
    infer_winner_from_log,
    save_battle_result,
)

# Default paths
AUTO_POOL_PATH = PROJECT_ROOT / "data" / "catalog" / "teams_regf_auto.yaml"
DEFAULT_VALUE_CKPT = MODELS_DIR / "team_value_mlp.pt"
DEFAULT_MATCHUP_CKPT = MODELS_DIR / "team_matchup_mlp.pt"
DEFAULT_MATCHUP_JSONL = MATCHUP_JSONL_PATH
DEFAULT_VALUE_JSONL = TEAMS_VS_POOL_JSONL_PATH

# Hardcoded main team import text (from evaluate_main_team_vs_random.py)
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


# ============================================================================
# Dataset Generation Functions
# ============================================================================


def generate_full_battle_dataset(
    n: int = 10, format_id: str | None = None
) -> None:
    """
    Generate `n` complete random self-play battles using Showdown's internal RandomPlayer AI.

    Logic moved from scripts/generate_full_battle_dataset.py.
    """
    if format_id is None:
        format_id = DEFAULT_FORMAT

    ensure_paths()
    ensure_full_battle_dataset_dir()

    successful = 0
    failed = 0

    print(f"Generating {n} full battle records...")
    print(f"Format: {format_id}")
    print()

    for i in range(1, n + 1):
        try:
            log_text, winner_side, turns, p1_name, p2_name = run_random_selfplay(
                format_id=format_id
            )

            raw_log_path = save_raw_log(format_id, log_text)

            winner_name = None
            if winner_side == "p1":
                winner_name = p1_name
            elif winner_side == "p2":
                winner_name = p2_name

            record = FullBattleRecord(
                format_id=format_id,
                p1_name=p1_name,
                p2_name=p2_name,
                winner_side=winner_side,
                winner_name=winner_name,
                turns=turns,
                raw_log_path=raw_log_path,
                created_at=datetime.now(timezone.utc),
            )

            append_full_battle_record(record)
            successful += 1
            print(f"[{i}/{n}] winner={winner_side}, turns={turns}")

            if i < n:
                time.sleep(0.1)

        except RuntimeError as e:
            failed += 1
            print(f"  Error [{i}]: {e}")
        except Exception as e:
            failed += 1
            print(f"  Unexpected error [{i}]: {e}")

    print()
    print("=" * 60)
    print("FULL BATTLE DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total attempts: {n}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()
    print(f"Dataset file: {FULL_BATTLE_DATASET_PATH}")
    if FULL_BATTLE_DATASET_PATH.exists():
        line_count = sum(1 for _ in FULL_BATTLE_DATASET_PATH.open("r", encoding="utf-8"))
        print(f"Total lines in dataset: {line_count}")
    print("=" * 60)


def generate_preview_outcome_dataset(
    n: int = 10, sleep_ms: int = 50, format_id: str | None = None
) -> None:
    """
    Generate a preview+outcome dataset from random self-play battles.

    Logic moved from scripts/generate_preview_outcome_dataset.py.
    """
    if format_id is None:
        format_id = DEFAULT_FORMAT

    ensure_paths()
    ensure_preview_outcome_dataset_dir()

    for i in range(1, n + 1):
        try:
            data = run_random_selfplay_json(format_id=format_id)
        except RuntimeError as e:
            print(f"[{i}/{n}] WARNING: run_random_selfplay_json failed: {e}")
            continue

        log_text = data.get("log", "")
        format_id_actual = data.get("format_id", format_id)
        p1_name = data.get("p1_name", "Bot1")
        p2_name = data.get("p2_name", "Bot2")
        winner_side = data.get("winner_side", "unknown")
        winner_name = data.get("winner_name")

        p1_team_public = data.get("p1_team_public", [])
        p2_team_public = data.get("p2_team_public", [])

        p1_chosen_indices: List[int] = []
        p2_chosen_indices: List[int] = []
        p1_lead_indices: List[int] = []
        p2_lead_indices: List[int] = []

        if log_text and p1_team_public and p2_team_public:
            try:
                (
                    p1_chosen_indices,
                    p2_chosen_indices,
                    p1_lead_indices,
                    p2_lead_indices,
                ) = extract_lineups_from_log(log_text, p1_team_public, p2_team_public)
            except Exception as e:
                print(f"[WARN] Failed to extract lineups: {e}")

        raw_log_path = save_raw_log(format_id_actual, log_text)

        record = PreviewOutcomeRecord(
            format_id=format_id_actual,
            p1_name=p1_name,
            p2_name=p2_name,
            p1_team_public=p1_team_public,
            p2_team_public=p2_team_public,
            p1_chosen_indices=p1_chosen_indices,
            p2_chosen_indices=p2_chosen_indices,
            p1_lead_indices=p1_lead_indices,
            p2_lead_indices=p2_lead_indices,
            winner_side=winner_side,
            winner_name=winner_name,
            raw_log_path=raw_log_path,
            created_at=datetime.now(timezone.utc),
            meta={},
        )
        append_preview_outcome_record(record)
        print(f"[{i}/{n}] winner_side={winner_side}, raw_log={raw_log_path.name}")

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)


def generate_team_preview_dataset(
    n: int = 100, format_id: str | None = None
) -> None:
    """
    Generate team preview dataset from simulated battles.

    Logic moved from scripts/generate_team_preview_dataset.py.
    """
    if format_id is None:
        format_id = DEFAULT_FORMAT

    ensure_paths()
    ensure_dataset_dir()

    successful = 0
    failed = 0
    skipped = 0

    print(f"Generating {n} team preview snapshots...")
    print(f"Format: {format_id}")
    print()

    for i in range(1, n + 1):
        try:
            p1_team = generate_random_team(format_id)
            p2_team = generate_random_team(format_id)

            log_text = simulate_battle(
                format_id=format_id,
                p1_name="Bot1",
                p1_packed_team=p1_team,
                p2_name="Bot2",
                p2_packed_team=p2_team,
            )

            raw_log_path = save_raw_log(format_id, log_text)

            try:
                snapshot = parse_team_preview_snapshot(
                    log_text,
                    format_id=format_id,
                    raw_log_path=raw_log_path,
                )
                append_snapshot_to_dataset(snapshot)
                successful += 1

                if i % 10 == 0:
                    print(
                        f"  Progress: {i}/{n} (successful: {successful}, failed: {failed}, skipped: {skipped})"
                    )
            except ValueError as e:
                skipped += 1
                print(f"  Warning [{i}]: Skipping due to parsing error: {e}")
                print(f"    Log: {raw_log_path}")

        except RuntimeError as e:
            failed += 1
            print(f"  Error [{i}]: simulate_battle failed: {e}")
        except Exception as e:
            failed += 1
            print(f"  Error [{i}]: Unexpected error: {e}")

    print()
    print("=" * 60)
    print("DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total attempts: {n}")
    print(f"Successful snapshots: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (parse errors): {skipped}")
    print()
    print(f"Dataset file: {DATASET_PATH}")
    if DATASET_PATH.exists():
        line_count = sum(1 for _ in DATASET_PATH.open("r", encoding="utf-8"))
        print(f"Total lines in dataset: {line_count}")
    print("=" * 60)


def generate_team_matchup_dataset(
    max_pairs: int = 200,
    n_battles_per_pair: int = 12,
    dataset_path: str | None = None,
    format_id: str | None = None,
    pool_yaml: str | None = None,
) -> None:
    """
    Evaluate team-vs-team matchups for teams from a catalog and save results.

    Logic moved from scripts/generate_team_matchup_dataset.py.
    """
    if format_id is None:
        format_id = DEFAULT_FORMAT

    ensure_paths()
    ensure_matchup_dir()

    if pool_yaml is None:
        if AUTO_POOL_PATH.exists():
            pool_yaml = str(AUTO_POOL_PATH)
        else:
            pool_yaml = str(TEAMS_CATALOG_PATH)

    pool_yaml_path = Path(pool_yaml)
    if not pool_yaml_path.exists():
        raise FileNotFoundError(f"Team pool file not found: {pool_yaml_path}")

    pool = TeamPool.from_yaml(pool_yaml_path)
    team_list = pool.all()
    print(f"Loaded {len(team_list)} teams from pool.")

    if len(team_list) < 2:
        raise ValueError("Need at least 2 teams to evaluate matchups.")

    from itertools import combinations

    all_pairs = list(combinations(team_list, 2))
    if len(all_pairs) > max_pairs:
        print(f"Total possible pairs: {len(all_pairs)}, limiting to {max_pairs}")
        pairs_to_evaluate = all_pairs[:max_pairs]
    else:
        pairs_to_evaluate = all_pairs

    print(f"Evaluating {len(pairs_to_evaluate)} team pairs...")
    print(f"Battles per pair: {n_battles_per_pair}")
    print(f"Format: {format_id}\n")

    output_path = Path(dataset_path) if dataset_path else MATCHUP_JSONL_PATH

    for idx, (team_a, team_b) in enumerate(pairs_to_evaluate, start=1):
        try:
            team_a_import = build_team_import_from_set_ids(
                team_a.set_ids, catalog=None
            )
            team_b_import = build_team_import_from_set_ids(team_b.set_ids, catalog=None)

            matchup = evaluate_team_vs_team(
                team_a_import=team_a_import,
                team_b_import=team_b_import,
                format_id=format_id,
                n=n_battles_per_pair,
            )

            win_rate_a = (
                matchup.n_a_wins / matchup.n_battles if matchup.n_battles > 0 else 0.0
            )

            record = TeamMatchupRecord(
                team_a_id=team_a.id,
                team_b_id=team_b.id,
                team_a_set_ids=team_a.set_ids,
                team_b_set_ids=team_b.set_ids,
                format_id=format_id,
                n_battles=matchup.n_battles,
                n_a_wins=matchup.n_a_wins,
                n_b_wins=matchup.n_b_wins,
                n_ties=matchup.n_ties,
                avg_turns=matchup.avg_turns,
                created_at=datetime.now(timezone.utc).isoformat(),
                meta={
                    "source": "catalog_pairwise_v1",
                    "n_battles_per_pair": n_battles_per_pair,
                },
            )

            append_matchup_record(record, path=output_path)

            print(
                f"[{idx}/{len(pairs_to_evaluate)}] "
                f"A={team_a.id}, B={team_b.id}, "
                f"win_rate_a={win_rate_a:.3f} "
                f"(wins={matchup.n_a_wins} / {matchup.n_battles})"
            )
        except Exception as e:
            print(
                f"[{idx}/{len(pairs_to_evaluate)}] ERROR evaluating "
                f"A={team_a.id}, B={team_b.id}: {e}"
            )
            continue

    print(f"\nDataset written/appended to: {output_path}")
    print(f"Total pairs evaluated: {len(pairs_to_evaluate)}")


def generate_teams_vs_pool_dataset(
    n_samples: int = 50,
    team_size: int = 6,
    n_opponents: int = 5,
    n_battles_per_opponent: int = 5,
    include_catalog_teams: bool = False,
    format_id: str | None = None,
    teams_yaml: str | None = None,
) -> None:
    """
    Generate dataset of random teams evaluated against a team pool.

    Logic moved from scripts/teams_vs_pool_dataset.py.
    """
    if format_id is None:
        format_id = DEFAULT_FORMAT

    ensure_paths()

    catalog = SetCatalog.from_yaml()
    pool_yaml_path = Path(teams_yaml) if teams_yaml else TEAMS_CATALOG_PATH
    pool = TeamPool.from_yaml(pool_yaml_path)

    if len(catalog) < team_size:
        raise ValueError(
            f"Catalog has only {len(catalog)} sets, cannot build teams of size {team_size}"
        )

    results = random_search_over_pool(
        catalog=catalog,
        pool=pool,
        n_candidates=n_samples,
        team_size=team_size,
        n_opponents=n_opponents,
        n_battles_per_opponent=n_battles_per_opponent,
    )

    n_random_written = 0
    for i, result in enumerate(results, start=1):
        append_result_to_dataset(result, source="random")
        n_random_written += 1
        winrate = result.win_rate
        print(
            f"[Random {i}/{n_samples}] win_rate={winrate:.3f}, "
            f"team={', '.join(result.team_set_ids[:2])}..."
        )

    n_catalog_written = 0
    if include_catalog_teams:
        catalog_teams = pool.all()
        print(f"\nEvaluating {len(catalog_teams)} catalog teams...")
        for i, team_def in enumerate(catalog_teams, start=1):
            try:
                summary = evaluate_team_against_pool(
                    team_set_ids=team_def.set_ids,
                    pool=pool,
                    n_opponents=n_opponents,
                    n_battles_per_opponent=n_battles_per_opponent,
                    catalog=catalog,
                )
                result = CandidateTeamResult(
                    team_set_ids=team_def.set_ids, pool_summary=summary
                )
                append_result_to_dataset(
                    result, source="catalog", team_id=team_def.id
                )
                n_catalog_written += 1
                winrate = result.win_rate
                print(
                    f"[Catalog {i}/{len(catalog_teams)}] "
                    f"team_id={team_def.id}, win_rate={winrate:.3f}"
                )
            except Exception as e:
                print(
                    f"[Catalog {i}/{len(catalog_teams)}] ERROR evaluating "
                    f"team_id={team_def.id}: {e}"
                )

    total_records = 0
    if TEAMS_VS_POOL_JSONL_PATH.exists():
        with TEAMS_VS_POOL_JSONL_PATH.open("r", encoding="utf-8") as f:
            total_records = sum(1 for line in f if line.strip())

    print(f"\nDataset written to: {TEAMS_VS_POOL_JSONL_PATH}")
    print(f"Random samples written: {n_random_written}")
    if include_catalog_teams:
        print(f"Catalog teams evaluated: {n_catalog_written}")
    print(f"Total records in file: {total_records}")


# ============================================================================
# Model Training Functions
# ============================================================================


def train_team_value_model(
    jsonl_path: Path | str | None = None,
    out_path: Path | str | None = None,
    epochs: int = 50,
    batch_size: int = 64,
    lr: float = 3e-4,
    min_battles: int = 3,
    include_random: bool = True,
    include_catalog: bool = True,
    device: str | None = None,
) -> None:
    """
    Train a neural team value model on teams_vs_pool.jsonl.

    Logic moved from scripts/train_team_value_model.py.
    """
    device = _resolve_device(device)
    jsonl_path_actual = Path(jsonl_path) if jsonl_path else DEFAULT_VALUE_JSONL
    out_path_actual = Path(out_path) if out_path else DEFAULT_VALUE_CKPT

    if not jsonl_path_actual.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path_actual}")

    sources = []
    if include_random:
        sources.append("random")
    if include_catalog:
        sources.append("catalog")
    if not sources:
        raise ValueError(
            "At least one of include_random or include_catalog must be enabled"
        )

    cfg = TrainingConfig(
        jsonl_path=jsonl_path_actual,
        out_path=out_path_actual,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        min_battles=min_battles,
        include_sources=sources,
        device=device,
    )

    print(f"Training team value model on {jsonl_path_actual}")
    print(
        f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}, device={device}"
    )
    print(f"Including sources: {sources}")

    model, vocab, train_meta = _train_team_value_model_impl(cfg, device_override=device)

    print(f"\nSaved team value model to {out_path_actual}")
    print(f"Vocab size: {len(vocab.idx_to_id) if hasattr(vocab, 'idx_to_id') else len(vocab)}")
    print(
        f"Final metrics: train_loss={train_meta['train_loss']:.4f}, "
        f"val_loss={train_meta['val_loss']:.4f}, "
        f"val_corr={train_meta['val_corr']:.4f}"
    )


def train_team_matchup_model(
    jsonl_path: Path | str | None = None,
    out_path: Path | str | None = None,
    epochs: int = 40,
    batch_size: int = 64,
    lr: float = 1e-3,
    min_battles: int = 10,
    max_records: int | None = None,
    device: str | None = None,
) -> None:
    """
    Train a neural team matchup model on team_matchups.jsonl.

    Logic moved from scripts/train_team_matchup_model.py.
    """
    device = _resolve_device(device)
    jsonl_path_actual = Path(jsonl_path) if jsonl_path else DEFAULT_MATCHUP_JSONL
    out_path_actual = Path(out_path) if out_path else DEFAULT_MATCHUP_CKPT

    if not jsonl_path_actual.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path_actual}")

    cfg = MatchupTrainingConfig(
        jsonl_path=jsonl_path_actual,
        out_path=out_path_actual,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        min_battles=min_battles,
        max_records=max_records,
        device=device,
    )

    print(f"Training team matchup model on {jsonl_path_actual}")
    print(
        f"Config: epochs={epochs}, batch_size={batch_size}, lr={lr}, device={device}"
    )

    model, vocab, train_meta = _train_team_matchup_model_impl(cfg, device_override=device)

    print(f"\nSaved team matchup model to {out_path_actual}")
    print(f"Vocab size: {len(vocab.idx_to_id) if hasattr(vocab, 'idx_to_id') else len(vocab)}")
    print(
        f"Final metrics: train_loss={train_meta['train_loss']:.4f}, "
        f"val_loss={train_meta['val_loss']:.4f}, "
        f"val_corr={train_meta['val_corr']:.4f}"
    )


# ============================================================================
# Team Search Functions
# ============================================================================


def random_catalog_team_search(
    n_candidates: int = 20,
    team_size: int = 6,
    n_opponents: int = 5,
    n_battles_per_opponent: int = 5,
    top_k: int = 5,
    format_id: str | None = None,
    teams_yaml: str | None = None,
) -> None:
    """
    Run a random search over catalog-defined sets, evaluating teams vs the team pool.

    Logic moved from scripts/random_catalog_team_search.py.
    """
    if format_id is None:
        format_id = DEFAULT_FORMAT

    ensure_paths()

    catalog = SetCatalog.from_yaml()
    pool_yaml_path = Path(teams_yaml) if teams_yaml else TEAMS_CATALOG_PATH
    pool = TeamPool.from_yaml(pool_yaml_path)

    if len(catalog) < team_size:
        raise ValueError(
            f"Catalog has only {len(catalog)} sets, cannot build teams of size {team_size}"
        )

    results = random_search_over_pool(
        catalog=catalog,
        pool=pool,
        n_candidates=n_candidates,
        team_size=team_size,
        n_opponents=n_opponents,
        n_battles_per_opponent=n_battles_per_opponent,
    )

    if top_k <= 0:
        top_k = len(results)
    top_k = min(top_k, len(results))

    print(f"Evaluated {len(results)} candidate teams.")
    print(f"Showing top {top_k} by win rate vs pool:\n")

    for i, r in enumerate(results[:top_k], start=1):
        winrate = r.win_rate
        summary = r.pool_summary
        print(f"[#{i}] win_rate={winrate:.3f}, battles={summary.n_battles_total}")
        print(f"     team_set_ids: {', '.join(r.team_set_ids)}")


def model_guided_team_search(
    ckpt_path: str | None = None,
    n_proposals: int = 500,
    top_k: int = 20,
    n_opponents: int = 5,
    n_per_opponent: int = 5,
    seed: int = 123,
    device: str | None = None,
    source_tag: str = "model_guided_v1",
    jsonl_path: str | None = None,
) -> None:
    """
    Run a single model-guided search iteration.

    Logic moved from scripts/model_guided_team_search.py.
    """
    import random

    ensure_paths()

    ckpt_path_actual = Path(ckpt_path) if ckpt_path else DEFAULT_VALUE_CKPT
    if not ckpt_path_actual.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path_actual}")

    device = _resolve_device(device)

    rng = random.Random(seed)

    model, vocab, model_cfg = load_team_value_checkpoint(ckpt_path_actual, device=device)
    print(
        f"Loaded model from {ckpt_path_actual} "
        f"(vocab_size={model_cfg['vocab_size']}, "
        f"embed_dim={model_cfg['embedding_dim']})"
    )

    catalog = SetCatalog.from_yaml()
    pool = TeamPool.from_yaml()

    print(
        f"Proposing {n_proposals} candidates, evaluating top {top_k} vs pool "
        f"({n_opponents} opponents, {n_per_opponent} battles each)..."
    )

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

    path = Path(jsonl_path) if jsonl_path else TEAMS_VS_POOL_JSONL_PATH
    print(f"Appending results to dataset: {path}")

    for i, (proposal, result) in enumerate(pairs, start=1):
        append_result_to_dataset(result, source=source_tag, jsonl_path=path)
        print(
            f"[{i}/{len(pairs)}] pred={proposal.pred_win_rate:.3f}, "
            f"actual={result.win_rate:.3f}, "
            f"sets={', '.join(proposal.set_ids[:2])}..."
        )

    print("Model-guided search iteration complete.")


def auto_team_value_iter(
    n_iters: int = 5,
    init_random_samples: int = 0,
    per_iter_random_samples: int = 0,
    per_iter_model_proposals: int = 500,
    per_iter_top_k: int = 40,
    n_mutation_bases: int = 0,
    n_mutations_per_base: int = 1,
    n_opponents: int = 5,
    n_per_opponent: int = 5,
    epochs_per_iter: int = 20,
    batch_size: int = 64,
    lr: float = 3e-4,
    seed: int = 123,
    device: str | None = None,
) -> None:
    """
    Run multi-iteration team value training + model-guided search.

    Logic moved from scripts/auto_team_value_iter.py.
    """
    import random

    ensure_paths()

    device = _resolve_device(device)

    rng = random.Random(seed)

    print(f"Using device: {device}")
    print(f"Dataset path: {TEAMS_VS_POOL_JSONL_PATH}")

    catalog = SetCatalog.from_yaml()
    pool = TeamPool.from_yaml()

    if init_random_samples > 0:
        print(f"\n=== Initial Random Search ===")
        print(f"Evaluating {init_random_samples} random candidates")

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
            print(f"[Init Random {i}/{len(random_results)}] win_rate={res.win_rate:.3f}")

    for iter_idx in range(1, n_iters + 1):
        print(f"\n{'='*50}")
        print(f"Iteration {iter_idx}/{n_iters}")
        print(f"{'='*50}")

        ckpt_path = MODELS_DIR / f"team_value_mlp_iter{iter_idx:03d}.pt"

        train_cfg = TrainingConfig(
            jsonl_path=TEAMS_VS_POOL_JSONL_PATH,
            out_path=ckpt_path,
            epochs=epochs_per_iter,
            batch_size=batch_size,
            lr=lr,
            min_battles=3,
            include_sources=None,
            device=device,
            seed=seed + iter_idx,
        )

        print(f"[Iter {iter_idx}] Training model -> {ckpt_path}")
        model, vocab, train_meta = _train_team_value_model_impl(train_cfg, device_override=device)

        print(
            f"[Iter {iter_idx}] train_loss={train_meta.get('train_loss', 0.0):.4f}, "
            f"val_loss={train_meta.get('val_loss', 0.0):.4f}, "
            f"val_corr={train_meta.get('val_corr', 0.0):.4f}"
        )

        print(
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
            append_result_to_dataset(result, source=f"auto_iter_{iter_idx}_model")
            print(
                f"[Iter {iter_idx}] [Model {i}/{len(proposal_result_pairs)}] "
                f"pred={proposal.pred_win_rate:.3f}, "
                f"actual={result.win_rate:.3f}"
            )

        if n_mutation_bases > 0 and n_mutations_per_base > 0:
            print(
                f"[Iter {iter_idx}] Mutation: bases={n_mutation_bases}, "
                f"mutations_per_base={n_mutations_per_base}"
            )

            base_proposals = [p for (p, _) in proposal_result_pairs][:n_mutation_bases]

            seen_keys = set()
            mutation_count = 0

            for base_idx, base in enumerate(base_proposals, start=1):
                for m in range(n_mutations_per_base):
                    mutated_set_ids = mutate_team_set_ids(
                        base.set_ids, catalog=catalog, rng=rng
                    )
                    key = tuple(sorted(mutated_set_ids))
                    if key in seen_keys:
                        continue
                    seen_keys.add(key)

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

                    append_result_to_dataset(mut_result, source=f"auto_iter_{iter_idx}_mut")
                    mutation_count += 1
                    print(
                        f"[Iter {iter_idx}] [Mut base={base_idx}, m={m+1}] "
                        f"actual={mut_result.win_rate:.3f}"
                    )

            print(
                f"[Iter {iter_idx}] Mutation complete: {mutation_count} unique variants evaluated"
            )

        if per_iter_random_samples > 0:
            print(
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
                append_result_to_dataset(res, source=f"auto_iter_{iter_idx}_random")

            print(
                f"[Iter {iter_idx}] Random exploration: {len(random_results)} candidates evaluated"
            )

    print(f"\n{'='*50}")
    print("Auto-iteration complete!")
    print(f"Final dataset: {TEAMS_VS_POOL_JSONL_PATH}")
    print(f"{'='*50}")


# ============================================================================
# Evaluation & Analysis Functions
# ============================================================================


def evaluate_catalog_team(
    set_ids: list[str],
    n: int = 50,
    format_id: str | None = None,
    team_as: str = "both",
    save_logs: bool = False,
) -> None:
    """
    Evaluate a catalog-defined team vs random opponents.

    Logic moved from scripts/evaluate_catalog_team.py.
    """
    if format_id is None:
        format_id = DEFAULT_FORMAT

    if save_logs:
        ensure_paths()

    if len(set_ids) != 6:
        raise ValueError(f"Expected exactly 6 set IDs, got {len(set_ids)}")

    summary = evaluate_catalog_team_vs_random(
        set_ids=set_ids,
        format_id=format_id,
        n=n,
        team_as=team_as,
        save_logs=save_logs,
    )

    winrate = summary.n_wins / summary.n_battles if summary.n_battles > 0 else 0.0
    print(f"Format: {summary.format_id}")
    print(f"Team role: {summary.team_role}")
    print(f"Battles: {summary.n_battles}")
    print(
        f"Wins: {summary.n_wins}, Losses: {summary.n_losses}, Ties: {summary.n_ties}"
    )
    print(f"Win rate: {winrate:.3f}")
    if summary.avg_turns is not None:
        print(f"Average turns: {summary.avg_turns:.2f}")
    print(f"Evaluated at: {summary.created_at.isoformat()}")


def evaluate_catalog_team_vs_pool(
    set_ids: list[str],
    n_opponents: int = 5,
    n_battles_per_opponent: int = 5,
    format_id: str | None = None,
    teams_yaml: str | None = None,
) -> None:
    """
    Evaluate a catalog-defined team vs a catalog-defined pool.

    Logic moved from scripts/evaluate_catalog_team_vs_pool.py.
    """
    if format_id is None:
        format_id = DEFAULT_FORMAT

    ensure_paths()

    if len(set_ids) != 6:
        raise ValueError(f"Expected exactly 6 set IDs, got {len(set_ids)}")

    pool_yaml_path = Path(teams_yaml) if teams_yaml else TEAMS_CATALOG_PATH
    pool = TeamPool.from_yaml(pool_yaml_path)

    summary = evaluate_team_against_pool(
        team_set_ids=set_ids,
        pool=pool,
        format_id=format_id,
        n_opponents=n_opponents,
        n_battles_per_opponent=n_battles_per_opponent,
    )

    winrate = (
        summary.n_wins / summary.n_battles_total
        if summary.n_battles_total > 0
        else 0.0
    )

    print(f"Format: {summary.format_id}")
    print(f"Team set IDs: {', '.join(summary.team_set_ids)}")
    print(
        f"Battles: {summary.n_battles_total} "
        f"(opponents={summary.n_opponents}, per_opponent={summary.n_battles_per_opponent})"
    )
    print(
        f"Wins: {summary.n_wins}, Losses: {summary.n_losses}, Ties: {summary.n_ties}"
    )
    print(f"Win rate vs pool: {winrate:.3f}")
    if summary.avg_turns is not None:
        print(f"Average turns: {summary.avg_turns:.2f}")
    if summary.opponent_counts:
        print("Opponent usage counts:")
        for opp_id, count in summary.opponent_counts.items():
            print(f"  - {opp_id}: {count}")


def evaluate_main_team_vs_random(
    n: int = 50,
    format_id: str | None = None,
    team_as: str = "p1",
    save_logs: bool = False,
) -> None:
    """
    Evaluate the main team against random opponents.

    Logic moved from scripts/evaluate_main_team_vs_random.py.
    """
    if format_id is None:
        format_id = DEFAULT_FORMAT

    if save_logs:
        ensure_paths()

    from .analysis import evaluate_team_vs_random

    packed = import_text_to_packed(MY_MAIN_TEAM_IMPORT, format_id=format_id)
    summary = evaluate_team_vs_random(
        team_packed=packed,
        format_id=format_id,
        n=n,
        team_as=team_as,
        save_logs=save_logs,
    )

    winrate = summary.n_wins / summary.n_battles if summary.n_battles > 0 else 0.0
    print(f"Format: {summary.format_id}")
    print(f"Team role: {summary.team_role}")
    print(f"Battles: {summary.n_battles}")
    print(
        f"Wins: {summary.n_wins}, Losses: {summary.n_losses}, Ties: {summary.n_ties}"
    )
    print(f"Win rate: {winrate:.3f}")
    if summary.avg_turns is not None:
        print(f"Average turns: {summary.avg_turns:.2f}")
    print(f"Evaluated at: {summary.created_at.isoformat()}")


def evaluate_matchup(
    team_a_path: Path | str,
    team_b_path: Path | str,
    n: int = 50,
    format_id: str | None = None,
) -> None:
    """
    Evaluate matchup between two teams using RandomPlayerAI.

    Logic moved from scripts/evaluate_matchup.py.
    """
    ensure_paths()
    
    if format_id is None:
        format_id = DEFAULT_FORMAT

    team_path = Path(team_a_path)
    opp_path = Path(team_b_path)

    if not team_path.exists():
        raise FileNotFoundError(f"Team A import file not found: {team_path}")
    if not opp_path.exists():
        raise FileNotFoundError(f"Team B import file not found: {opp_path}")

    team_a_import = team_path.read_text(encoding="utf-8")
    team_b_import = opp_path.read_text(encoding="utf-8")

    matchup = evaluate_team_vs_team(
        team_a_import=team_a_import,
        team_b_import=team_b_import,
        format_id=format_id,
        n=n,
    )

    total = matchup.n_battles
    winrate_a = matchup.n_a_wins / total if total > 0 else 0.0
    winrate_b = matchup.n_b_wins / total if total > 0 else 0.0

    print(f"Format: {matchup.format_id}")
    print(f"Battles: {matchup.n_battles}")
    print(f"Team A wins: {matchup.n_a_wins}")
    print(f"Team B wins: {matchup.n_b_wins}")
    print(f"Ties: {matchup.n_ties}")
    print(f"Team A win rate: {winrate_a:.3f}")
    print(f"Team B win rate: {winrate_b:.3f}")
    if matchup.avg_turns is not None:
        print(f"Average turns: {matchup.avg_turns:.2f}")
    print(f"Evaluated at: {matchup.created_at.isoformat()}")


def analyze_teams_vs_pool(
    top_k: int = 20,
    min_battles: int = 20,
    min_records: int = 1,
    source_prefix: str | None = None,
    jsonl_path: str | None = None,
    output_csv: str | None = None,
    output_teams_yaml: str | None = None,
) -> None:
    """
    Analyze the teams_vs_pool dataset and list/discover top teams.

    Logic moved from scripts/analyze_teams_vs_pool.py.
    """
    import csv
    import json
    from typing import Tuple

    jsonl_path_actual = (
        Path(jsonl_path) if jsonl_path else TEAMS_VS_POOL_JSONL_PATH
    )

    if not jsonl_path_actual.exists():
        raise FileNotFoundError(f"Dataset file not found: {jsonl_path_actual}")

    aggregates = load_and_aggregate_teams_vs_pool(jsonl_path_actual)
    print(f"Loaded {len(aggregates)} aggregated teams before filtering.")

    if source_prefix is not None:
        filtered_by_source = [
            agg
            for agg in aggregates
            if any(src.startswith(source_prefix) for src in agg.source_counts.keys())
        ]
        aggregates = filtered_by_source
        print(f"After source_prefix='{source_prefix}' filter: {len(aggregates)} teams")

    filtered = [
        agg
        for agg in aggregates
        if agg.n_battles_total >= min_battles and agg.n_records >= min_records
    ]

    print(
        f"After min_battles={min_battles}, min_records={min_records} filter: {len(filtered)} teams"
    )

    if not filtered:
        raise ValueError("No teams match the filtering criteria.")

    filtered.sort(key=lambda a: (a.win_rate, a.n_battles_total), reverse=True)
    top = filtered[:top_k]

    print(f"\n{'='*80}")
    print(f"Top {len(top)} teams (ranked by win_rate):")
    print(f"{'='*80}\n")

    for rank, agg in enumerate(top, start=1):
        print(
            f"[#{rank}] win_rate={agg.win_rate:.3f} "
            f"(wins={agg.n_wins} / battles={agg.n_battles_total}, records={agg.n_records})"
        )
        print(f"     set_ids: {', '.join(agg.team_set_ids)}")
        source_str = ", ".join(
            f"{src}={count}" for src, count in sorted(agg.source_counts.items())
        )
        print(f"     sources: {source_str}")
        print()

    if output_csv is not None:
        output_csv_path = Path(output_csv)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        with output_csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "rank",
                    "win_rate",
                    "n_battles_total",
                    "n_wins",
                    "n_losses",
                    "n_ties",
                    "n_records",
                    "team_set_ids",
                    "source_counts_json",
                ]
            )

            for rank, agg in enumerate(top, start=1):
                team_set_ids_str = " ".join(agg.team_set_ids)
                source_counts_json = json.dumps(agg.source_counts, ensure_ascii=False)
                writer.writerow(
                    [
                        rank,
                        f"{agg.win_rate:.6f}",
                        agg.n_battles_total,
                        agg.n_wins,
                        agg.n_losses,
                        agg.n_ties,
                        agg.n_records,
                        team_set_ids_str,
                        source_counts_json,
                    ]
                )

        print(f"Wrote CSV to: {output_csv_path}")

    if output_teams_yaml is not None:
        import yaml

        output_teams_yaml_path = Path(output_teams_yaml)
        output_teams_yaml_path.parent.mkdir(parents=True, exist_ok=True)

        catalog = SetCatalog.from_yaml()
        yaml_teams = []

        for rank, agg in enumerate(top, start=1):
            missing_set_ids = [
                sid for sid in agg.team_set_ids if sid not in catalog._entries
            ]

            if missing_set_ids:
                print(
                    f"WARNING: Team rank #{rank} has missing set_ids in catalog: {missing_set_ids}"
                )
                continue

            team_id = f"auto_top_{rank:04d}"
            description = (
                f"Auto-discovered top team rank #{rank} "
                f"(win_rate={agg.win_rate:.3f}, battles={agg.n_battles_total}, "
                f"records={agg.n_records}, min_battles={min_battles}, "
                f"min_records={min_records})"
            )

            yaml_teams.append(
                {
                    "id": team_id,
                    "format": DEFAULT_FORMAT,
                    "description": description,
                    "set_ids": agg.team_set_ids,
                }
            )

        if yaml_teams:
            with output_teams_yaml_path.open("w", encoding="utf-8") as f:
                yaml.dump(
                    yaml_teams,
                    f,
                    default_flow_style=False,
                    sort_keys=False,
                    allow_unicode=True,
                )
            print(f"Wrote YAML catalog to: {output_teams_yaml_path}")
            print(f"Exported {len(yaml_teams)} teams.")
        else:
            print(
                "WARNING: No teams could be exported to YAML (all had missing set_ids)"
            )


def analyze_meta_matchups(
    team_id: str,
    teams_yaml: Path | str | None = None,
    ckpt_path: Path | str | None = None,
    top_k_best: int = 10,
    top_k_worst: int = 10,
    device: str | None = None,
) -> None:
    """
    Analyze a team's matchup vs a meta using the matchup model.

    Logic moved from scripts/analyze_meta_matchups.py.
    """
    from typing import Tuple

    device = _resolve_device(device)
    ckpt_path_actual = (
        Path(ckpt_path) if ckpt_path else DEFAULT_MATCHUP_CKPT
    )
    if not ckpt_path_actual.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path_actual}")

    if teams_yaml is None:
        if AUTO_POOL_PATH.exists():
            teams_yaml_actual = AUTO_POOL_PATH
        else:
            teams_yaml_actual = TEAMS_CATALOG_PATH
    else:
        teams_yaml_actual = Path(teams_yaml)

    if not teams_yaml_actual.exists():
        raise FileNotFoundError(f"Team pool file not found: {teams_yaml_actual}")

    pool = TeamPool.from_yaml(teams_yaml_actual)

    try:
        focus_team = pool.get(team_id)
    except KeyError:
        raise KeyError(f"Team ID '{team_id}' not found in pool")

    model, vocab, model_cfg = load_team_matchup_checkpoint(
        ckpt_path_actual, device=device
    )

    all_team_ids = pool.ids()
    meta = build_uniform_meta(pool, all_team_ids)
    meta = normalize_meta(meta)

    print(f"Focus team: {team_id}")
    print(f"Meta size: {len(meta.entries)} teams")
    print()

    matchup_results: List[Tuple[str, float]] = []

    for meta_entry in meta.entries:
        opp_id = meta_entry.team_id
        if opp_id == team_id:
            continue

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
        raise ValueError("No valid matchups found")

    overall_win_rate = sum(p for _, p in matchup_results) / len(matchup_results)

    print(f"Expected win_rate vs meta (model): {overall_win_rate:.3f}")
    print()

    matchup_results.sort(key=lambda x: x[1], reverse=True)

    print(f"Best matchups (top {top_k_best}):")
    for rank, (opp_id, p_win) in enumerate(matchup_results[:top_k_best], start=1):
        print(f"  {rank}. {opp_id:<30} P(win) = {p_win:.3f}")
    print()

    print(f"Worst matchups (bottom {top_k_worst}):")
    for rank, (opp_id, p_win) in enumerate(
        reversed(matchup_results[-top_k_worst:]), start=1
    ):
        print(f"  {rank}. {opp_id:<30} P(win) = {p_win:.3f}")


def suggest_counter_teams(
    teams_yaml: Path | str | None = None,
    ckpt_path: Path | str | None = None,
    meta_json: Path | str | None = None,
    candidate_prefix: str | None = None,
    top_k: int = 10,
    device: str | None = None,
) -> None:
    """
    Suggest counter teams vs a meta using the matchup model.

    Logic moved from scripts/suggest_counter_teams.py.
    """
    import json

    device = _resolve_device(device)
    ckpt_path_actual = (
        Path(ckpt_path) if ckpt_path else DEFAULT_MATCHUP_CKPT
    )
    if not ckpt_path_actual.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path_actual}")

    if teams_yaml is None:
        if AUTO_POOL_PATH.exists():
            teams_yaml_actual = AUTO_POOL_PATH
        else:
            teams_yaml_actual = TEAMS_CATALOG_PATH
    else:
        teams_yaml_actual = Path(teams_yaml)

    if not teams_yaml_actual.exists():
        raise FileNotFoundError(f"Team pool file not found: {teams_yaml_actual}")

    pool = TeamPool.from_yaml(teams_yaml_actual)
    model, vocab, model_cfg = load_team_matchup_checkpoint(
        ckpt_path_actual, device=device
    )

    if meta_json is not None:
        meta_json_path = Path(meta_json)
        if not meta_json_path.exists():
            raise FileNotFoundError(f"Meta JSON file not found: {meta_json_path}")

        with meta_json_path.open("r", encoding="utf-8") as f:
            meta_data = json.load(f)

        entries = []
        for entry_data in meta_data.get("entries", []):
            team_id = entry_data.get("team_id")
            weight = float(entry_data.get("weight", 1.0))

            try:
                pool.get(team_id)
                entries.append(MetaTeam(team_id=team_id, weight=weight))
            except KeyError:
                print(f"WARNING: Team '{team_id}' from meta JSON not found in pool, skipping")
                continue

        if not entries:
            raise ValueError("No valid teams found in meta JSON after filtering")

        meta = MetaDistribution(entries=entries)
        meta = normalize_meta(meta)
        print(f"Meta: {len(meta.entries)} teams (from {meta_json_path})")
    else:
        all_team_ids = pool.ids()
        meta = build_uniform_meta(pool, all_team_ids)
        meta = normalize_meta(meta)
        print(f"Meta: {len(meta.entries)} teams (uniform)")

    all_candidate_ids = pool.ids()

    if candidate_prefix is not None:
        candidate_ids = [
            tid for tid in all_candidate_ids if tid.startswith(candidate_prefix)
        ]
        print(f"Candidates filtered by prefix '{candidate_prefix}': {len(candidate_ids)} teams")
    else:
        candidate_ids = all_candidate_ids

    if not candidate_ids:
        raise ValueError("No candidate teams found")

    print(f"Evaluating {len(candidate_ids)} candidate teams...")
    print()

    ranked = rank_teams_vs_meta(
        candidate_team_ids=candidate_ids,
        meta=meta,
        team_pool=pool,
        model=model,
        vocab=vocab,
        device=device,
    )

    if not ranked:
        raise ValueError("No valid rankings computed")

    print(f"Top {min(top_k, len(ranked))} counter candidates:")
    print()

    for rank, (team_id, expected_win) in enumerate(ranked[:top_k], start=1):
        print(
            f"  {rank}. {team_id:<30} E[win_rate vs meta] = {expected_win:.3f}"
        )

    print()
    print(
        "Note: These are model-based estimates (not guaranteed, but useful for guiding "
        "further real evaluations)."
    )


# ============================================================================
# Utility Functions
# ============================================================================


def dump_teams_vs_pool_features(
    input_path: str | None = None,
    output_csv: str | None = None,
) -> None:
    """
    Convert teams_vs_pool JSONL records into a CSV of features for ML.

    Logic moved from scripts/dump_teams_vs_pool_features.py.
    """
    import csv
    from .features import (
        build_default_vocab,
        iter_teams_vs_pool_records,
        team_bag_of_sets_feature,
    )

    DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"
    DEFAULT_INPUT = DATASETS_ROOT / "teams_vs_pool" / "teams_vs_pool.jsonl"
    DEFAULT_OUTPUT = DATASETS_ROOT / "teams_vs_pool" / "features.csv"

    input_path_actual = Path(input_path) if input_path else DEFAULT_INPUT
    output_csv_actual = Path(output_csv) if output_csv else DEFAULT_OUTPUT

    if not input_path_actual.exists():
        raise FileNotFoundError(f"Input file not found: {input_path_actual}")

    vocab = build_default_vocab()
    V = len(vocab.idx_to_id)

    records = list(iter_teams_vs_pool_records(input_path_actual))

    if not records:
        raise ValueError("No records found in input file")

    output_csv_actual.parent.mkdir(parents=True, exist_ok=True)

    with output_csv_actual.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        header = ["win_rate", "n_battles", "source", "set_id_indices"]
        header += [f"feat_{i}" for i in range(V)]
        writer.writerow(header)

        for rec in records:
            indices = vocab.encode_ids(rec.team_set_ids)
            feat_vec = team_bag_of_sets_feature(rec.team_set_ids, vocab)
            indices_str = " ".join(str(i) for i in indices)
            row = [rec.win_rate, rec.n_battles_total, rec.source, indices_str]
            row += feat_vec
            writer.writerow(row)

    print(f"Wrote {len(records)} records to {output_csv_actual}")
    print(f"Feature vector dimension: {V}")


def pack_team(file: str | None = None) -> None:
    """
    Convert team from export format to packed format.

    Logic moved from scripts/pack_team_cli.py.
    """
    if file:
        file_path = Path(file)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        input_text = file_path.read_text(encoding="utf-8")
    else:
        input_text = sys.stdin.read()

    if not input_text.strip():
        raise ValueError("No input provided")

    packed_result = _pack_team(input_text)
    print(packed_result)


def validate_team(file: str | None = None, format_id: str | None = None) -> None:
    """
    Validate a Pokemon Showdown team against a format.

    Logic moved from scripts/validate_team_cli.py.
    """
    if format_id is None:
        format_id = DEFAULT_FORMAT

    if file:
        file_path = Path(file)
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        input_text = file_path.read_text(encoding="utf-8")
    else:
        input_text = sys.stdin.read()

    if not input_text.strip():
        raise ValueError("No input provided")

    looks_like_export = (
        "@" in input_text or "EVs:" in input_text or "Ability:" in input_text
    )

    if looks_like_export:
        print("Detected export format, converting to packed...", file=sys.stderr)
        packed_team = _pack_team(input_text)
    else:
        packed_team = input_text.strip()

    validation_output = _validate_team(packed_team, format_id)

    if validation_output:
        print("Validation FAILED:", file=sys.stderr)
        print(validation_output, file=sys.stderr)
        sys.exit(1)
    else:
        print(f"Team is valid for format: {format_id}")
        sys.exit(0)


def score_catalog_team_with_model(
    team_id: str,
    ckpt_path: str | None = None,
    device: str | None = None,
) -> None:
    """
    Score a catalog team using the trained value model.

    Logic moved from scripts/score_catalog_team_with_model.py.
    """
    import torch
    from .catalog import TeamPool
    from .models import load_team_value_checkpoint

    device = _resolve_device(device)
    ckpt_path_actual = Path(ckpt_path) if ckpt_path else DEFAULT_VALUE_CKPT
    if not ckpt_path_actual.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path_actual}")

    model, vocab, model_cfg = load_team_value_checkpoint(ckpt_path_actual, device=device)

    pool = TeamPool.from_yaml()
    try:
        team_def = pool.get(team_id)
    except KeyError:
        raise KeyError(f"Team ID not found in pool: {team_id}")

    set_ids = team_def.set_ids

    try:
        indices = vocab.encode_ids(set_ids)
    except KeyError as e:
        raise KeyError(
            f"Set ID {e} from team '{team_id}' not found in model vocabulary. "
            f"The model was trained on a different set catalog."
        )

    tensor = torch.tensor([indices], dtype=torch.long, device=device)

    with torch.no_grad():
        pred = model(tensor).item()

    print(f"Team ID: {team_id}")
    print(f"Set IDs: {', '.join(set_ids)}")
    print(f"Predicted win_rate vs pool: {pred:.3f}")
    print(
        f"(Model vocab_size={model_cfg['vocab_size']}, "
        f"embed_dim={model_cfg['embedding_dim']})"
    )


def score_matchup_with_model(
    team_a_id: str,
    team_b_id: str,
    ckpt_path: Path | str | None = None,
    teams_yaml: Path | str | None = None,
    device: str | None = None,
) -> None:
    """
    Score a matchup between two catalog teams using the trained matchup model.

    Logic moved from scripts/score_matchup_with_model.py.
    """
    # predict_matchup_win_prob is already imported at top level

    device = _resolve_device(device)
    ckpt_path_actual = Path(ckpt_path) if ckpt_path else DEFAULT_MATCHUP_CKPT
    if not ckpt_path_actual.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {ckpt_path_actual}")

    if teams_yaml is None:
        if AUTO_POOL_PATH.exists():
            teams_yaml_actual = AUTO_POOL_PATH
        else:
            from .catalog import TEAMS_CATALOG_PATH
            teams_yaml_actual = TEAMS_CATALOG_PATH
    else:
        teams_yaml_actual = Path(teams_yaml)

    if not teams_yaml_actual.exists():
        raise FileNotFoundError(f"Team pool file not found: {teams_yaml_actual}")

    model, vocab, model_cfg = load_team_matchup_checkpoint(ckpt_path_actual, device=device)

    pool = TeamPool.from_yaml(teams_yaml_actual)

    try:
        team_a = pool.get(team_a_id)
        team_b = pool.get(team_b_id)
    except KeyError as e:
        raise KeyError(f"Team ID not found in pool: {e}")

    try:
        pred = predict_matchup_win_prob(
            model=model,
            vocab=vocab,
            team_a_set_ids=team_a.set_ids,
            team_b_set_ids=team_b.set_ids,
            device=device,
        )
    except KeyError as e:
        raise KeyError(
            f"Set ID {e} not found in model vocabulary. "
            f"The model was trained on a different set catalog."
        )

    print(f"Team A ID: {team_a_id}")
    print(f"Team A set_ids: {', '.join(team_a.set_ids)}")
    print()
    print(f"Team B ID: {team_b_id}")
    print(f"Team B set_ids: {', '.join(team_b.set_ids)}")
    print()
    print(f"Predicted P(A wins vs B): {pred:.3f}")
    print(f"(Model vocab_size={model_cfg['vocab_size']}, embed_dim={model_cfg['embed_dim']})")


def run_demo_battle() -> None:
    """
    Run a demo battle between two randomly generated teams.

    Logic moved from scripts/run_demo_battle.py.
    """
    ensure_paths()

    print(f"Generating random teams for format: {DEFAULT_FORMAT}")
    p1_team = generate_random_team(DEFAULT_FORMAT)
    p2_team = generate_random_team(DEFAULT_FORMAT)

    print(f"P1 team (first 100 chars): {p1_team[:100]}...")
    print(f"P2 team (first 100 chars): {p2_team[:100]}...")

    print("\nRunning battle simulation (one-shot CLI, no streaming bots)...")
    battle_log = simulate_battle(
        format_id=DEFAULT_FORMAT,
        p1_name="Bot1",
        p1_packed_team=p1_team,
        p2_name="Bot2",
        p2_packed_team=p2_team,
    )

    print(f"Battle log length: {len(battle_log)} characters")

    raw_log_path = save_raw_log(DEFAULT_FORMAT, battle_log)
    print(f"\nSaved raw log to: {raw_log_path}")

    winner_side = infer_winner_from_log(battle_log, "Bot1", "Bot2")
    turns = count_turns(battle_log)

    result = BattleResult(
        format_id=DEFAULT_FORMAT,
        p1_name="Bot1",
        p2_name="Bot2",
        winner=winner_side,
        raw_log_path=raw_log_path,
        turns=turns,
    )

    result_json_path = save_battle_result(result)
    print(f"Saved result JSON to: {result_json_path}")

    print("\n" + "=" * 60)
    print("BATTLE SUMMARY")
    print("=" * 60)
    print(f"Format: {result.format_id}")
    print(f"Winner side: {result.winner}")
    print(f"Turns: {result.turns if result.turns is not None else 'unknown'}")
    print(f"Raw log: {result.raw_log_path}")
    print(f"Result JSON: {result_json_path}")
    print("=" * 60)
