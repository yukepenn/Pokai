"""vgc-lab: Python wrapper for Pokemon Showdown CLI."""

from .core import (
    analyze_meta_matchups,
    analyze_teams_vs_pool,
    auto_team_value_iter,
    dump_teams_vs_pool_features,
    evaluate_catalog_team,
    evaluate_catalog_team_vs_pool,
    evaluate_main_team_vs_random,
    evaluate_matchup,
    generate_full_battle_dataset,
    generate_preview_outcome_dataset,
    generate_team_matchup_dataset,
    generate_team_preview_dataset,
    generate_teams_vs_pool_dataset,
    model_guided_team_search,
    pack_team,
    random_catalog_team_search,
    run_demo_battle,
    score_catalog_team_with_model,
    score_matchup_with_model,
    suggest_counter_teams,
    train_team_matchup_model,
    train_team_value_model,
    validate_team,
)

# Re-export key types for convenience
from .analysis import (
    CandidateTeamResult,
    MatchupEvalSummary,
    MetaDistribution,
    TeamEvalSummary,
)
from .catalog import SetCatalog, TeamPool
from .models import TeamMatchupModel, TeamValueModel

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # Dataset generation
    "generate_full_battle_dataset",
    "generate_preview_outcome_dataset",
    "generate_team_preview_dataset",
    "generate_team_matchup_dataset",
    "generate_teams_vs_pool_dataset",
    # Training
    "train_team_value_model",
    "train_team_matchup_model",
    "auto_team_value_iter",
    # Search
    "random_catalog_team_search",
    "model_guided_team_search",
    # Evaluation / analysis
    "evaluate_catalog_team",
    "evaluate_catalog_team_vs_pool",
    "evaluate_main_team_vs_random",
    "evaluate_matchup",
    "analyze_teams_vs_pool",
    "analyze_meta_matchups",
    "suggest_counter_teams",
    # Tools
    "pack_team",
    "validate_team",
    "dump_teams_vs_pool_features",
    "run_demo_battle",
    "score_catalog_team_with_model",
    "score_matchup_with_model",
    # Key types
    "SetCatalog",
    "TeamPool",
    "TeamValueModel",
    "TeamMatchupModel",
    "TeamEvalSummary",
    "MatchupEvalSummary",
    "MetaDistribution",
    "CandidateTeamResult",
]
