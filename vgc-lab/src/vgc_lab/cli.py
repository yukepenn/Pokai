"""Unified CLI entrypoint for vgc-lab."""

import argparse
import sys
from pathlib import Path

from . import core


def _str_to_bool(v: str) -> bool:
    """Convert string to boolean for argparse."""
    if v.lower() in ("yes", "true", "t", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def _parse_dataset_full_battles(args):
    """Parse arguments for dataset full-battles command."""
    core.generate_full_battle_dataset(n=args.n, format_id=args.format)


def _parse_dataset_preview_outcome(args):
    """Parse arguments for dataset preview-outcome command."""
    core.generate_preview_outcome_dataset(
        n=args.n, sleep_ms=args.sleep_ms, format_id=args.format
    )


def _parse_dataset_team_preview(args):
    """Parse arguments for dataset team-preview command."""
    core.generate_team_preview_dataset(n=args.n, format_id=args.format)


def _parse_dataset_team_matchups(args):
    """Parse arguments for dataset team-matchups command."""
    core.generate_team_matchup_dataset(
        max_pairs=args.max_pairs,
        n_battles_per_pair=args.n_battles_per_pair,
        dataset_path=args.dataset_path,
        format_id=args.format,
        pool_yaml=args.pool_yaml,
    )


def _parse_dataset_teams_vs_pool(args):
    """Parse arguments for dataset teams-vs-pool command."""
    core.generate_teams_vs_pool_dataset(
        n_samples=args.n_samples,
        team_size=args.team_size,
        n_opponents=args.n_opponents,
        n_battles_per_opponent=args.n_per,
        include_catalog_teams=args.include_catalog_teams,
        format_id=args.format,
        teams_yaml=args.teams_yaml,
    )


def _parse_train_value_model(args):
    """Parse arguments for train value-model command."""
    core.train_team_value_model(
        jsonl_path=args.jsonl,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_battles=args.min_battles,
        include_random=args.include_random,
        include_catalog=args.include_catalog,
        device=args.device,
    )


def _parse_train_matchup_model(args):
    """Parse arguments for train matchup-model command."""
    core.train_team_matchup_model(
        jsonl_path=args.jsonl,
        out_path=args.out,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        min_battles=args.min_battles,
        max_records=args.max_records,
        device=args.device,
    )


def _parse_search_random_over_pool(args):
    """Parse arguments for search random-over-pool command."""
    core.random_catalog_team_search(
        n_candidates=args.n_candidates,
        team_size=args.team_size,
        n_opponents=args.n_opponents,
        n_battles_per_opponent=args.n_per,
        top_k=args.top_k,
        format_id=args.format,
        teams_yaml=args.teams_yaml,
    )


def _parse_search_model_guided(args):
    """Parse arguments for search model-guided command."""
    core.model_guided_team_search(
        ckpt_path=args.ckpt,
        n_proposals=args.n_proposals,
        top_k=args.top_k,
        n_opponents=args.n_opponents,
        n_per_opponent=args.n_per,
        seed=args.seed,
        device=args.device,
        source_tag=args.source,
        jsonl_path=args.jsonl,
    )


def _parse_search_auto_value_iter(args):
    """Parse arguments for search auto-value-iter command."""
    core.auto_team_value_iter(
        n_iters=args.n_iters,
        init_random_samples=args.init_random_samples,
        per_iter_random_samples=args.per_iter_random_samples,
        per_iter_model_proposals=args.per_iter_model_proposals,
        per_iter_top_k=args.per_iter_top_k,
        n_mutation_bases=args.n_mutation_bases,
        n_mutations_per_base=args.n_mutations_per_base,
        n_opponents=args.n_opponents,
        n_per_opponent=args.n_per,
        epochs_per_iter=args.epochs_per_iter,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
    )


def _parse_eval_catalog_team(args):
    """Parse arguments for eval catalog-team command."""
    core.evaluate_catalog_team(
        set_ids=args.set_ids,
        n=args.n,
        format_id=args.format,
        team_as=args.team_as,
        save_logs=args.save_logs,
    )


def _parse_eval_catalog_vs_pool(args):
    """Parse arguments for eval catalog-vs-pool command."""
    core.evaluate_catalog_team_vs_pool(
        set_ids=args.set_ids,
        n_opponents=args.n_opponents,
        n_battles_per_opponent=args.n_per,
        format_id=args.format,
        teams_yaml=args.teams_yaml,
    )


def _parse_eval_main_team_vs_random(args):
    """Parse arguments for eval main-team-vs-random command."""
    core.evaluate_main_team_vs_random(
        n=args.n, format_id=args.format, team_as=args.team_as, save_logs=args.save_logs
    )


def _parse_eval_matchup(args):
    """Parse arguments for eval matchup command."""
    core.evaluate_matchup(
        team_a_path=Path(args.team_a),
        team_b_path=Path(args.team_b),
        n=args.n,
        format_id=args.format,
    )


def _parse_eval_score_catalog_team(args):
    """Parse arguments for eval score-catalog-team command."""
    core.score_catalog_team_with_model(
        team_id=args.team_id,
        ckpt_path=args.ckpt,
        device=args.device,
    )


def _parse_eval_score_matchup(args):
    """Parse arguments for eval score-matchup command."""
    core.score_matchup_with_model(
        team_a_id=args.team_a_id,
        team_b_id=args.team_b_id,
        ckpt_path=args.ckpt,
        teams_yaml=args.teams_yaml,
        device=args.device,
    )


def _parse_analyze_teams_vs_pool(args):
    """Parse arguments for analyze teams-vs-pool command."""
    core.analyze_teams_vs_pool(
        top_k=args.top_k,
        min_battles=args.min_battles,
        min_records=args.min_records,
        source_prefix=args.source_prefix,
        jsonl_path=args.jsonl,
        output_csv=args.output_csv,
        output_teams_yaml=args.output_teams_yaml,
    )


def _parse_analyze_meta_matchups(args):
    """Parse arguments for analyze meta-matchups command."""
    core.analyze_meta_matchups(
        team_id=args.team_id,
        teams_yaml=args.teams_yaml,
        ckpt_path=args.ckpt,
        top_k_best=args.top_k_best,
        top_k_worst=args.top_k_worst,
        device=args.device,
    )


def _parse_analyze_suggest_counters(args):
    """Parse arguments for analyze suggest-counters command."""
    core.suggest_counter_teams(
        teams_yaml=args.teams_yaml,
        ckpt_path=args.ckpt,
        meta_json=args.meta_json,
        candidate_prefix=args.candidate_prefix,
        top_k=args.top_k,
        device=args.device,
    )


def _parse_tools_pack_team(args):
    """Parse arguments for tools pack-team command."""
    core.pack_team(file=args.file)


def _parse_tools_validate_team(args):
    """Parse arguments for tools validate-team command."""
    core.validate_team(file=args.file, format_id=args.format)


def _parse_tools_dump_features(args):
    """Parse arguments for tools dump-features command."""
    core.dump_teams_vs_pool_features(input_path=args.input, output_csv=args.output)


def _parse_demo_battle(args):
    """Parse arguments for demo battle command."""
    core.run_demo_battle()


def main(argv: list[str] | None = None) -> None:
    """Main CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="vgc-lab",
        description="Pokemon Showdown CLI wrapper for Gen 9 VGC Reg F battle simulation and team analysis",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    subparsers.required = True

    # ========================================================================
    # Dataset generation commands
    # ========================================================================
    dataset_parser = subparsers.add_parser("dataset", help="Dataset generation commands")
    dataset_subparsers = dataset_parser.add_subparsers(dest="dataset_command")
    dataset_subparsers.required = True

    # dataset full-battles
    parser_full_battles = dataset_subparsers.add_parser(
        "full-battles", help="Generate full battle dataset"
    )
    parser_full_battles.add_argument(
        "--n", type=int, default=10, help="Number of full battles to generate"
    )
    parser_full_battles.add_argument(
        "--format", type=str, help="Showdown format ID (default: gen9vgc2026regf)"
    )
    parser_full_battles.set_defaults(func=_parse_dataset_full_battles)

    # dataset preview-outcome
    parser_preview_outcome = dataset_subparsers.add_parser(
        "preview-outcome", help="Generate preview+outcome dataset"
    )
    parser_preview_outcome.add_argument(
        "--n",
        type=int,
        default=10,
        help="Number of preview+outcome samples to generate",
    )
    parser_preview_outcome.add_argument(
        "--sleep-ms",
        type=int,
        default=50,
        help="Sleep between battles in milliseconds",
    )
    parser_preview_outcome.add_argument(
        "--format", type=str, help="Showdown format ID"
    )
    parser_preview_outcome.set_defaults(func=_parse_dataset_preview_outcome)

    # dataset team-preview
    parser_team_preview = dataset_subparsers.add_parser(
        "team-preview", help="Generate team preview dataset"
    )
    parser_team_preview.add_argument(
        "--n", type=int, default=100, help="Number of snapshots to generate"
    )
    parser_team_preview.add_argument("--format", type=str, help="Showdown format ID")
    parser_team_preview.set_defaults(func=_parse_dataset_team_preview)

    # dataset team-matchups
    parser_team_matchups = dataset_subparsers.add_parser(
        "team-matchups", help="Generate pairwise team matchup dataset"
    )
    parser_team_matchups.add_argument(
        "--max-pairs",
        type=int,
        default=200,
        help="Maximum number of team pairs to evaluate",
    )
    parser_team_matchups.add_argument(
        "--n-battles-per-pair",
        type=int,
        default=12,
        help="Number of battles for each A vs B evaluation",
    )
    parser_team_matchups.add_argument(
        "--dataset-path", type=str, help="Optional override for JSONL output path"
    )
    parser_team_matchups.add_argument(
        "--format", type=str, help="Showdown format ID"
    )
    parser_team_matchups.add_argument(
        "--pool-yaml",
        type=str,
        help="Team catalog path (defaults to teams_regf_auto.yaml if exists)",
    )
    parser_team_matchups.set_defaults(func=_parse_dataset_team_matchups)

    # dataset teams-vs-pool
    parser_teams_vs_pool = dataset_subparsers.add_parser(
        "teams-vs-pool", help="Generate teams vs pool dataset"
    )
    parser_teams_vs_pool.add_argument(
        "--n-samples",
        type=int,
        default=50,
        help="Number of random teams to evaluate",
    )
    parser_teams_vs_pool.add_argument(
        "--team-size", type=int, default=6, help="Number of Pokémon per team"
    )
    parser_teams_vs_pool.add_argument(
        "--n-opponents",
        type=int,
        default=5,
        help="Number of opponent teams sampled per candidate",
    )
    parser_teams_vs_pool.add_argument(
        "--n-per",
        type=int,
        default=5,
        help="Number of battles per opponent",
    )
    parser_teams_vs_pool.add_argument(
        "--include-catalog-teams",
        action="store_true",
        help="Also evaluate all catalog teams",
    )
    parser_teams_vs_pool.add_argument("--format", type=str, help="Showdown format ID")
    parser_teams_vs_pool.add_argument(
        "--teams-yaml", type=str, help="Path to teams YAML catalog"
    )
    parser_teams_vs_pool.set_defaults(func=_parse_dataset_teams_vs_pool)

    # ========================================================================
    # Model training commands
    # ========================================================================
    train_parser = subparsers.add_parser("train", help="Model training commands")
    train_subparsers = train_parser.add_subparsers(dest="train_command")
    train_subparsers.required = True

    # train value-model
    parser_value_model = train_subparsers.add_parser(
        "value-model", help="Train team value model"
    )
    parser_value_model.add_argument(
        "--jsonl", type=str, help="Path to teams_vs_pool.jsonl dataset"
    )
    parser_value_model.add_argument(
        "--out", type=str, help="Path to save model checkpoint"
    )
    parser_value_model.add_argument(
        "--epochs", "-e", type=int, default=50, help="Training epochs"
    )
    parser_value_model.add_argument(
        "--batch-size", "-b", type=int, default=64, help="Batch size"
    )
    parser_value_model.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate"
    )
    parser_value_model.add_argument(
        "--min-battles",
        type=int,
        default=3,
        help="Minimum battles per sample to include",
    )
    # Default is True for both; use --no-* flags to exclude
    parser_value_model.add_argument(
        "--no-include-random",
        dest="include_random",
        action="store_false",
        default=True,
        help="Exclude 'random' source (default: include)",
    )
    parser_value_model.add_argument(
        "--no-include-catalog",
        dest="include_catalog",
        action="store_false",
        default=True,
        help="Exclude 'catalog' source (default: include)",
    )
    parser_value_model.add_argument(
        "--device",
        type=str,
        help="Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'",
    )
    parser_value_model.set_defaults(func=_parse_train_value_model)

    # train matchup-model
    parser_matchup_model = train_subparsers.add_parser(
        "matchup-model", help="Train team matchup model"
    )
    parser_matchup_model.add_argument(
        "--jsonl", type=str, help="Path to team_matchups.jsonl dataset"
    )
    parser_matchup_model.add_argument(
        "--out", type=str, help="Path to save model checkpoint"
    )
    parser_matchup_model.add_argument(
        "--epochs", "-e", type=int, default=40, help="Training epochs"
    )
    parser_matchup_model.add_argument(
        "--batch-size", "-b", type=int, default=64, help="Batch size"
    )
    parser_matchup_model.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate"
    )
    parser_matchup_model.add_argument(
        "--min-battles",
        type=int,
        default=10,
        help="Minimum battles per matchup to include",
    )
    parser_matchup_model.add_argument(
        "--max-records", type=int, help="Optional cap on number of records"
    )
    parser_matchup_model.add_argument(
        "--device",
        type=str,
        help="Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'",
    )
    parser_matchup_model.set_defaults(func=_parse_train_matchup_model)

    # ========================================================================
    # Team search commands
    # ========================================================================
    search_parser = subparsers.add_parser("search", help="Team search commands")
    search_subparsers = search_parser.add_subparsers(dest="search_command")
    search_subparsers.required = True

    # search random-over-pool
    parser_random_search = search_subparsers.add_parser(
        "random-over-pool", help="Random search over catalog teams vs pool"
    )
    parser_random_search.add_argument(
        "--n-candidates",
        type=int,
        default=20,
        help="Number of random candidate teams to evaluate",
    )
    parser_random_search.add_argument(
        "--team-size", type=int, default=6, help="Number of Pokémon per team"
    )
    parser_random_search.add_argument(
        "--n-opponents",
        type=int,
        default=5,
        help="Number of opponent teams sampled",
    )
    parser_random_search.add_argument(
        "--n-per",
        type=int,
        default=5,
        help="Number of battles per opponent",
    )
    parser_random_search.add_argument(
        "--top-k", type=int, default=5, help="How many top teams to print"
    )
    parser_random_search.add_argument("--format", type=str, help="Showdown format ID")
    parser_random_search.add_argument(
        "--teams-yaml", type=str, help="Path to teams YAML catalog"
    )
    parser_random_search.set_defaults(func=_parse_search_random_over_pool)

    # search model-guided
    parser_model_guided = search_subparsers.add_parser(
        "model-guided", help="Model-guided team search"
    )
    parser_model_guided.add_argument(
        "--ckpt", type=str, help="Path to trained team value model checkpoint"
    )
    parser_model_guided.add_argument(
        "--n-proposals",
        type=int,
        default=500,
        help="Number of candidate teams to propose",
    )
    parser_model_guided.add_argument(
        "--top-k",
        type=int,
        default=20,
        help="Number of top candidates to evaluate",
    )
    parser_model_guided.add_argument(
        "--n-opponents",
        type=int,
        default=5,
        help="Number of opponent teams sampled",
    )
    parser_model_guided.add_argument(
        "--n-per", type=int, default=5, help="Number of battles vs each opponent"
    )
    parser_model_guided.add_argument(
        "--seed", type=int, default=123, help="Random seed"
    )
    parser_model_guided.add_argument(
        "--device",
        type=str,
        help="Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'",
    )
    parser_model_guided.add_argument(
        "--source",
        type=str,
        default="model_guided_v1",
        help="Source tag for dataset records",
    )
    parser_model_guided.add_argument(
        "--jsonl", type=str, help="Optional explicit path to teams_vs_pool.jsonl"
    )
    parser_model_guided.set_defaults(func=_parse_search_model_guided)

    # search auto-value-iter
    parser_auto_iter = search_subparsers.add_parser(
        "auto-value-iter", help="Multi-iteration training + model-guided search"
    )
    parser_auto_iter.add_argument(
        "--n-iters", type=int, default=5, help="Number of outer iterations"
    )
    parser_auto_iter.add_argument(
        "--init-random-samples",
        type=int,
        default=0,
        help="Initial random candidates to evaluate",
    )
    parser_auto_iter.add_argument(
        "--per-iter-random-samples",
        type=int,
        default=0,
        help="Random candidates per iteration",
    )
    parser_auto_iter.add_argument(
        "--per-iter-model-proposals",
        type=int,
        default=500,
        help="Model proposals per iteration",
    )
    parser_auto_iter.add_argument(
        "--per-iter-top-k",
        type=int,
        default=40,
        help="Top candidates to evaluate per iteration",
    )
    parser_auto_iter.add_argument(
        "--n-mutation-bases",
        type=int,
        default=0,
        help="Number of top candidates to use as bases for mutation (0 = disable)",
    )
    parser_auto_iter.add_argument(
        "--n-mutations-per-base",
        type=int,
        default=1,
        help="Number of mutated variants per base team",
    )
    parser_auto_iter.add_argument(
        "--n-opponents",
        type=int,
        default=5,
        help="Number of opponent teams sampled",
    )
    parser_auto_iter.add_argument(
        "--n-per", type=int, default=5, help="Number of battles per opponent"
    )
    parser_auto_iter.add_argument(
        "--epochs-per-iter",
        type=int,
        default=20,
        help="Training epochs per iteration",
    )
    parser_auto_iter.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for training"
    )
    parser_auto_iter.add_argument(
        "--lr", type=float, default=3e-4, help="Learning rate"
    )
    parser_auto_iter.add_argument(
        "--seed", type=int, default=123, help="Random seed"
    )
    parser_auto_iter.add_argument(
        "--device",
        type=str,
        help="Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'",
    )
    parser_auto_iter.set_defaults(func=_parse_search_auto_value_iter)

    # ========================================================================
    # Evaluation commands
    # ========================================================================
    eval_parser = subparsers.add_parser("eval", help="Evaluation commands")
    eval_subparsers = eval_parser.add_subparsers(dest="eval_command")
    eval_subparsers.required = True

    # eval catalog-team
    parser_eval_catalog = eval_subparsers.add_parser(
        "catalog-team", help="Evaluate catalog team vs random opponents"
    )
    parser_eval_catalog.add_argument(
        "set_ids", nargs="+", help="List of 6 set IDs forming the team"
    )
    parser_eval_catalog.add_argument(
        "--n", type=int, default=50, help="Number of self-play battles"
    )
    parser_eval_catalog.add_argument("--format", type=str, help="Showdown format ID")
    parser_eval_catalog.add_argument(
        "--team-as",
        type=str,
        default="both",
        choices=["p1", "p2", "both"],
        help='Which side to place the team on',
    )
    parser_eval_catalog.add_argument(
        "--save-logs", action="store_true", help="Save logs to data/battles_raw/"
    )
    parser_eval_catalog.set_defaults(func=_parse_eval_catalog_team)

    # eval catalog-vs-pool
    parser_eval_catalog_pool = eval_subparsers.add_parser(
        "catalog-vs-pool", help="Evaluate catalog team vs pool"
    )
    parser_eval_catalog_pool.add_argument(
        "set_ids", nargs="+", help="List of 6 set IDs forming the team"
    )
    parser_eval_catalog_pool.add_argument(
        "--n-opponents",
        type=int,
        default=5,
        help="Number of opponent teams sampled",
    )
    parser_eval_catalog_pool.add_argument(
        "--n-per",
        type=int,
        default=5,
        help="Number of battles per opponent",
    )
    parser_eval_catalog_pool.add_argument("--format", type=str, help="Showdown format ID")
    parser_eval_catalog_pool.add_argument(
        "--teams-yaml", type=str, help="Path to teams YAML catalog"
    )
    parser_eval_catalog_pool.set_defaults(func=_parse_eval_catalog_vs_pool)

    # eval main-team-vs-random
    parser_eval_main = eval_subparsers.add_parser(
        "main-team-vs-random", help="Evaluate main team vs random opponents"
    )
    parser_eval_main.add_argument(
        "--n", type=int, default=50, help="Number of self-play battles"
    )
    parser_eval_main.add_argument("--format", type=str, help="Showdown format ID")
    parser_eval_main.add_argument(
        "--team-as",
        type=str,
        default="p1",
        choices=["p1", "p2", "both"],
        help='Which side to place the team on',
    )
    parser_eval_main.add_argument(
        "--save-logs", action="store_true", help="Save logs to data/battles_raw/"
    )
    parser_eval_main.set_defaults(func=_parse_eval_main_team_vs_random)

    # eval matchup
    parser_eval_matchup = eval_subparsers.add_parser(
        "matchup", help="Evaluate matchup between two teams"
    )
    parser_eval_matchup.add_argument(
        "team_a", type=str, help="Path to Team A import text"
    )
    parser_eval_matchup.add_argument(
        "team_b", type=str, help="Path to Team B import text"
    )
    parser_eval_matchup.add_argument(
        "--n", type=int, default=50, help="Number of battles"
    )
    parser_eval_matchup.add_argument("--format", type=str, help="Showdown format ID")
    parser_eval_matchup.set_defaults(func=_parse_eval_matchup)

    # eval score-catalog-team
    parser_score_catalog = eval_subparsers.add_parser(
        "score-catalog-team", help="Score a catalog team using the value model"
    )
    parser_score_catalog.add_argument(
        "team_id", type=str, help="Team ID from teams_regf.yaml"
    )
    parser_score_catalog.add_argument(
        "--ckpt", type=str, help="Path to trained value model checkpoint"
    )
    parser_score_catalog.add_argument(
        "--device",
        type=str,
        help="Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'",
    )
    parser_score_catalog.set_defaults(func=_parse_eval_score_catalog_team)

    # eval score-matchup
    parser_score_matchup = eval_subparsers.add_parser(
        "score-matchup", help="Score a matchup using the matchup model"
    )
    parser_score_matchup.add_argument(
        "team_a_id", type=str, help="Team ID of team A"
    )
    parser_score_matchup.add_argument(
        "team_b_id", type=str, help="Team ID of team B"
    )
    parser_score_matchup.add_argument(
        "--ckpt", type=str, help="Path to trained matchup model checkpoint"
    )
    parser_score_matchup.add_argument(
        "--teams-yaml", type=str, help="Team catalog path"
    )
    parser_score_matchup.add_argument(
        "--device",
        type=str,
        help="Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'",
    )
    parser_score_matchup.set_defaults(func=_parse_eval_score_matchup)

    # ========================================================================
    # Analysis commands
    # ========================================================================
    analyze_parser = subparsers.add_parser("analyze", help="Analysis commands")
    analyze_subparsers = analyze_parser.add_subparsers(dest="analyze_command")
    analyze_subparsers.required = True

    # analyze teams-vs-pool
    parser_analyze_teams = analyze_subparsers.add_parser(
        "teams-vs-pool", help="Analyze teams_vs_pool dataset and discover top teams"
    )
    parser_analyze_teams.add_argument(
        "--top-k", type=int, default=20, help="Number of top teams to display"
    )
    parser_analyze_teams.add_argument(
        "--min-battles",
        type=int,
        default=20,
        help="Minimum total battles required",
    )
    parser_analyze_teams.add_argument(
        "--min-records",
        type=int,
        default=1,
        help="Minimum number of JSONL records required",
    )
    parser_analyze_teams.add_argument(
        "--source-prefix",
        type=str,
        help="Filter by source prefix (e.g., 'auto_iter_')",
    )
    parser_analyze_teams.add_argument(
        "--jsonl", type=str, help="Path to teams_vs_pool.jsonl"
    )
    parser_analyze_teams.add_argument(
        "--output-csv", type=str, help="Optional path to write CSV"
    )
    parser_analyze_teams.add_argument(
        "--output-teams-yaml",
        type=str,
        help="Optional path to write YAML catalog",
    )
    parser_analyze_teams.set_defaults(func=_parse_analyze_teams_vs_pool)

    # analyze meta-matchups
    parser_analyze_meta = analyze_subparsers.add_parser(
        "meta-matchups", help="Analyze team's matchup vs meta using matchup model"
    )
    parser_analyze_meta.add_argument(
        "team_id", type=str, help="Team ID to analyze"
    )
    parser_analyze_meta.add_argument(
        "--teams-yaml", type=str, help="Team catalog path"
    )
    parser_analyze_meta.add_argument(
        "--ckpt", type=str, help="Path to trained matchup model checkpoint"
    )
    parser_analyze_meta.add_argument(
        "--top-k-best", type=int, default=10, help="Number of best matchups to show"
    )
    parser_analyze_meta.add_argument(
        "--top-k-worst",
        type=int,
        default=10,
        help="Number of worst matchups to show",
    )
    parser_analyze_meta.add_argument(
        "--device",
        type=str,
        help="Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'",
    )
    parser_analyze_meta.set_defaults(func=_parse_analyze_meta_matchups)

    # analyze suggest-counters
    parser_suggest = analyze_subparsers.add_parser(
        "suggest-counters", help="Suggest counter teams vs a meta"
    )
    parser_suggest.add_argument(
        "--teams-yaml", type=str, help="Team catalog path"
    )
    parser_suggest.add_argument(
        "--ckpt", type=str, help="Path to trained matchup model checkpoint"
    )
    parser_suggest.add_argument(
        "--meta-json", type=str, help="Optional JSON file describing meta distribution"
    )
    parser_suggest.add_argument(
        "--candidate-prefix",
        type=str,
        help="Restrict candidates to team IDs starting with this prefix",
    )
    parser_suggest.add_argument(
        "--top-k", type=int, default=10, help="Number of best counter candidates to show"
    )
    parser_suggest.add_argument(
        "--device",
        type=str,
        help="Device ('cpu' or 'cuda'); default is 'cuda' if available, else 'cpu'",
    )
    parser_suggest.set_defaults(func=_parse_analyze_suggest_counters)

    # ========================================================================
    # Tools commands
    # ========================================================================
    tools_parser = subparsers.add_parser("tools", help="Utility tools")
    tools_subparsers = tools_parser.add_subparsers(dest="tools_command")
    tools_subparsers.required = True

    # tools pack-team
    parser_pack = tools_subparsers.add_parser(
        "pack-team", help="Convert team from export to packed format"
    )
    parser_pack.add_argument(
        "--file", "-f", type=str, help="Input file (or read from stdin)"
    )
    parser_pack.set_defaults(func=_parse_tools_pack_team)

    # tools validate-team
    parser_validate = tools_subparsers.add_parser(
        "validate-team", help="Validate a Pokemon Showdown team"
    )
    parser_validate.add_argument(
        "--file", "-f", type=str, help="Input file (or read from stdin)"
    )
    parser_validate.add_argument(
        "--format", "-F", type=str, help="Format ID to validate against"
    )
    parser_validate.set_defaults(func=_parse_tools_validate_team)

    # tools dump-features
    parser_dump = tools_subparsers.add_parser(
        "dump-features", help="Dump teams_vs_pool features to CSV"
    )
    parser_dump.add_argument(
        "--input", type=str, help="Path to teams_vs_pool.jsonl"
    )
    parser_dump.add_argument(
        "--output", type=str, help="Output CSV path"
    )
    parser_dump.set_defaults(func=_parse_tools_dump_features)

    # ========================================================================
    # Demo commands
    # ========================================================================
    demo_parser = subparsers.add_parser("demo", help="Demo commands")
    demo_subparsers = demo_parser.add_subparsers(dest="demo_command")
    demo_subparsers.required = True

    # demo battle
    parser_demo_battle = demo_subparsers.add_parser(
        "battle", help="Run a demo battle"
    )
    parser_demo_battle.set_defaults(func=_parse_demo_battle)

    # Parse arguments and execute
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
