"""Self-play harness for team-building evaluation.

This module runs battles between TeamValuePolicy-driven teams and baseline (random) teams,
recording statistics and optionally appending TeamBuildEpisode rows for future training.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from statistics import mean
from typing import Dict, List, Optional

import random

from vgc_lab import (
    ShowdownClient,
    BattleStore,
    PokemonSetDef,
    get_paths,
    load_sets,
    sample_team_sets_random,
    build_packed_team_from_set_ids,
    TeamBuildEpisode,
    TeamBuildStep,
    append_team_build_episode,
)
from .policy import TeamValuePolicy


@dataclass
class SelfPlayConfig:
    """
    Configuration for team-building self-play evaluation.
    """

    num_episodes: int = 50
    format_id: str = "gen9vgc2026regf"
    p1_policy: str = "team_value"  # "team_value" or "random"
    p2_policy: str = "random"  # "team_value" or "random"
    seed: int = 42
    write_team_build_episodes: bool = True
    policy_id_team_value: str = "team_value_policy_v1"
    policy_id_random: str = "random_sets_v1"


def _pick_team(
    policy_name: str,
    value_policy: Optional[TeamValuePolicy],
    sets: Dict[str, PokemonSetDef],
    format_id: str,
    rng: random.Random,
) -> List[str]:
    """Select a 6-mon team according to the given policy name.

    Args:
        policy_name: "team_value" or "random"
        value_policy: TeamValuePolicy instance (required if policy_name == "team_value")
        sets: Dict mapping set_id -> PokemonSetDef (filtered by format_id)
        format_id: Format ID for filtering
        rng: Random number generator for reproducibility

    Returns:
        List of 6 set_ids forming the selected team.

    Raises:
        ValueError: If policy_name is invalid
        AssertionError: If policy_name == "team_value" but value_policy is None
    """
    if policy_name == "team_value":
        assert value_policy is not None, "value_policy must be provided for 'team_value' policy"
        return value_policy.sample_team(n_candidates=128, rng=rng)
    elif policy_name == "random":
        return sample_team_sets_random(sets, format_id=format_id, rng=rng)
    else:
        raise ValueError(f"Unknown policy name: {policy_name!r}. Expected 'team_value' or 'random'.")


def _append_team_build_episode_for_side(
    side: str,
    policy_name: str,
    policy_id_team_value: str,
    policy_id_random: str,
    format_id: str,
    set_ids: List[str],
    reward: float,
    battle_id: str,
) -> None:
    """Append a TeamBuildEpisode for a given side.

    Creates and appends a TeamBuildEpisode row using the same structure
    as scripts/cli.py gen-battles-from-sets.

    Args:
        side: "p1" or "p2"
        policy_name: "team_value" or "random"
        policy_id_team_value: Policy ID to use for team_value policy
        policy_id_random: Policy ID to use for random policy
        format_id: Format ID
        set_ids: List of 6 set_ids
        reward: Reward from this side's perspective (float)
        battle_id: Battle ID to associate with this episode
    """
    policy_id = policy_id_team_value if policy_name == "team_value" else policy_id_random

    steps = [
        TeamBuildStep(step_index=i, chosen_set_id=sid) for i, sid in enumerate(set_ids)
    ]

    episode = TeamBuildEpisode(
        episode_id=f"team_ep_{uuid.uuid4().hex}",
        side=side,
        format_id=format_id,
        policy_id=policy_id,
        chosen_set_ids=set_ids,
        reward=reward,
        battle_ids=[battle_id],
        steps=steps,
        meta={"source": "selfplay_rl_team_build"},
    )

    append_team_build_episode(episode)


def run_selfplay(config: SelfPlayConfig) -> Dict[str, float]:
    """Run self-play episodes between TeamValuePolicy and baseline teams.

    Executes battles between policy-driven teams and random baseline teams,
    logging results via BattleStore and optionally appending TeamBuildEpisodes
    for future training.

    Args:
        config: SelfPlayConfig containing num_episodes, policies, format_id, etc.

    Returns:
        Summary dict with keys: num_episodes, p1_wins, p1_losses, p1_ties, avg_reward_p1.
        All values are numeric (int or float).
    """
    # Set up RNG and clients
    rng = random.Random(config.seed)
    paths = get_paths()
    client = ShowdownClient(paths, format_id=config.format_id)
    store = BattleStore(paths)

    # Load sets
    all_sets = load_sets()
    sets = {sid: s for sid, s in all_sets.items() if s.format == config.format_id}

    if not sets:
        raise ValueError(f"No sets found for format_id: {config.format_id}")

    # Initialize TeamValuePolicy if needed
    value_policy = None
    if config.p1_policy == "team_value" or config.p2_policy == "team_value":
        value_policy = TeamValuePolicy(format_id=config.format_id)

    # Main loop
    results_p1: List[float] = []

    for ep in range(config.num_episodes):
        # Pick teams
        p1_set_ids = _pick_team(config.p1_policy, value_policy, sets, config.format_id, rng)
        p2_set_ids = _pick_team(config.p2_policy, value_policy, sets, config.format_id, rng)

        # Build packed teams
        p1_packed = build_packed_team_from_set_ids(p1_set_ids, sets, client)
        p2_packed = build_packed_team_from_set_ids(p2_set_ids, sets, client)

        # Run battle
        battle_json = client.run_random_selfplay_json(
            format_id=config.format_id,
            p1_name="PolicyP1",
            p2_name="BaselineP2",
            p1_packed_team=p1_packed,
            p2_packed_team=p2_packed,
        )

        # Store battle record and trajectory
        record = store.append_full_battle_from_json(battle_json)
        store.append_trajectory_from_battle(record, battle_json)

        # Get reward from p1's perspective
        # Use the same logic as in scripts/cli.py gen-battles-from-sets
        win_side = record.winner_side
        if win_side == "p1":
            reward_p1 = 1.0
        elif win_side == "p2":
            reward_p1 = -1.0
        else:
            reward_p1 = 0.0

        results_p1.append(reward_p1)

        # Append TeamBuildEpisode rows if requested
        if config.write_team_build_episodes:
            _append_team_build_episode_for_side(
                side="p1",
                policy_name=config.p1_policy,
                policy_id_team_value=config.policy_id_team_value,
                policy_id_random=config.policy_id_random,
                format_id=config.format_id,
                set_ids=p1_set_ids,
                reward=reward_p1,
                battle_id=record.battle_id,
            )
            # Also log p2's episode
            reward_p2 = -reward_p1
            _append_team_build_episode_for_side(
                side="p2",
                policy_name=config.p2_policy,
                policy_id_team_value=config.policy_id_team_value,
                policy_id_random=config.policy_id_random,
                format_id=config.format_id,
                set_ids=p2_set_ids,
                reward=reward_p2,
                battle_id=record.battle_id,
            )

    # Compute summary
    total = len(results_p1)
    p1_wins = sum(1 for r in results_p1 if r > 0)
    p1_losses = sum(1 for r in results_p1 if r < 0)
    p1_ties = total - p1_wins - p1_losses
    avg_reward = mean(results_p1) if results_p1 else 0.0

    summary = {
        "num_episodes": total,
        "p1_wins": p1_wins,
        "p1_losses": p1_losses,
        "p1_ties": p1_ties,
        "avg_reward_p1": float(avg_reward),
    }

    print(
        f"Self-play summary: episodes={total}, "
        f"p1_wins={p1_wins}, losses={p1_losses}, ties={p1_ties}, "
        f"avg_reward_p1={avg_reward:.2f}"
    )

    return summary


if __name__ == "__main__":
    config = SelfPlayConfig(
        num_episodes=20,
        p1_policy="team_value",
        p2_policy="random",
        seed=123,
        write_team_build_episodes=True,
    )
    summary = run_selfplay(config)
    print("Final summary:", summary)

