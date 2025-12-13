"""vgc-lab: Python wrapper for Pokemon Showdown CLI."""

from .catalog import (
    PokemonSetDef,
    build_packed_team_from_set_ids,
    load_sets,
    sample_team_sets_random,
)
from .core import (
    BattleStep,
    BattleStore,
    BattleTrajectory,
    FullBattleRecord,
    Paths,
    RandomAgent,
    ShowdownClient,
    TeamPreviewSnapshot,
    get_paths,
    parse_team_preview_snapshot,
)
from .datasets import (
    TeamBuildEpisode,
    TeamBuildStep,
    TurnView,
    append_team_build_episode,
    build_turn_views,
    iter_team_build_episodes,
    iter_trajectories,
    group_steps_by_turn,
)
from .features import (
    encode_state_from_request,
    encode_step,
    encode_team_build_episode,
    encode_team_from_set_ids,
    encode_trajectory,
    encode_trajectory_both_sides,
    encode_trajectory_side,
)

__version__ = "0.1.0"

__all__ = [
    "Paths",
    "get_paths",
    "ShowdownClient",
    "BattleStore",
    "RandomAgent",
    "FullBattleRecord",
    "TeamPreviewSnapshot",
    "parse_team_preview_snapshot",
    "BattleStep",
    "BattleTrajectory",
    "PokemonSetDef",
    "load_sets",
    "sample_team_sets_random",
    "build_packed_team_from_set_ids",
    "TeamBuildEpisode",
    "TeamBuildStep",
    "TurnView",
    "append_team_build_episode",
    "iter_team_build_episodes",
    "iter_trajectories",
    "group_steps_by_turn",
    "build_turn_views",
    "encode_state_from_request",
    "encode_step",
    "encode_trajectory",
    "encode_trajectory_side",
    "encode_trajectory_both_sides",
    "encode_team_from_set_ids",
    "encode_team_build_episode",
]

