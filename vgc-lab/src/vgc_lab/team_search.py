"""Team pool evaluation utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .config import DEFAULT_FORMAT
from .eval import evaluate_team_vs_team
from .set_catalog import SetCatalog
from .team_builder import build_team_import_from_set_ids
from .team_pool import TeamPool, TeamDef


@dataclass
class PoolEvalSummary:
    """Summary of team vs pool evaluation results."""

    format_id: str
    team_set_ids: List[str]
    n_opponents: int
    n_battles_per_opponent: int
    n_battles_total: int
    n_wins: int
    n_losses: int
    n_ties: int
    avg_turns: Optional[float]
    opponent_counts: Dict[str, int]


def evaluate_team_against_pool(
    team_set_ids: Iterable[str],
    pool: TeamPool,
    *,
    format_id: str = DEFAULT_FORMAT,
    n_opponents: int = 5,
    n_battles_per_opponent: int = 5,
    catalog: Optional[SetCatalog] = None,
) -> PoolEvalSummary:
    """
    Evaluate a catalog-defined team against a pool of catalog teams.

    For each of `n_opponents`:
      - sample an opponent TeamDef from the pool (with replacement),
      - run a matchup using evaluate_team_vs_team(..., n=n_battles_per_opponent),
      - accumulate wins/losses/ties from the perspective of our team.

    Args:
        team_set_ids: Sequence of 6 set IDs defining our team.
        pool: TeamPool instance to sample opponents from.
        format_id: Format ID for evaluation.
        n_opponents: Number of opponent teams to sample.
        n_battles_per_opponent: Number of battles per opponent matchup.
        catalog: SetCatalog instance. If None, loads from default catalog path.

    Returns:
        PoolEvalSummary with aggregated statistics.
    """
    if catalog is None:
        catalog = SetCatalog.from_yaml()

    # Build import for our team once
    team_set_ids_list = list(team_set_ids)
    team_import = build_team_import_from_set_ids(team_set_ids_list, catalog=catalog)

    # Counters
    n_wins = 0
    n_losses = 0
    n_ties = 0
    total_turns = 0
    n_turns_samples = 0
    opponent_counts: Dict[str, int] = {}

    # Sample opponent teams
    opponents = pool.sample(n=n_opponents)

    for opp in opponents:
        opponent_counts[opp.id] = opponent_counts.get(opp.id, 0) + 1

        opp_import = build_team_import_from_set_ids(opp.set_ids, catalog=catalog)

        matchup = evaluate_team_vs_team(
            team_a_import=team_import,
            team_b_import=opp_import,
            format_id=format_id,
            n=n_battles_per_opponent,
        )

        # From Team A's perspective (our team is Team A)
        n_wins += matchup.n_a_wins
        n_losses += matchup.n_b_wins
        n_ties += matchup.n_ties

        if matchup.avg_turns is not None:
            total_turns += matchup.avg_turns * matchup.n_battles
            n_turns_samples += matchup.n_battles

    n_battles_total = n_opponents * n_battles_per_opponent
    avg_turns = (
        total_turns / n_turns_samples if n_turns_samples > 0 else None
    )

    return PoolEvalSummary(
        format_id=format_id,
        team_set_ids=team_set_ids_list,
        n_opponents=n_opponents,
        n_battles_per_opponent=n_battles_per_opponent,
        n_battles_total=n_battles_total,
        n_wins=n_wins,
        n_losses=n_losses,
        n_ties=n_ties,
        avg_turns=avg_turns,
        opponent_counts=opponent_counts,
    )


def sample_random_team_set_ids(
    catalog: SetCatalog,
    *,
    team_size: int = 6,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    Sample a random team (a list of set IDs) from the catalog.

    For now:
      - Sample without replacement from all available set IDs.
      - Assume team_size <= number of sets in the catalog.
      - The caller is responsible for ensuring team_size is valid.

    Args:
        catalog: SetCatalog instance to sample from.
        team_size: Number of sets to include in the team (default: 6).
        rng: Optional random number generator. If None, uses random module.

    Returns:
        List of set ID strings (length = team_size).

    Raises:
        ValueError: If team_size is larger than the number of available sets.
    """
    if rng is None:
        rng = random

    all_ids = catalog.ids()

    if len(all_ids) < team_size:
        raise ValueError(
            f"Catalog has {len(all_ids)} sets, cannot sample team of size {team_size}"
        )

    return rng.sample(all_ids, k=team_size)


@dataclass
class CandidateTeamResult:
    """Result of evaluating a candidate team against the pool."""

    team_set_ids: List[str]
    pool_summary: PoolEvalSummary

    @property
    def win_rate(self) -> float:
        """Calculate win rate from the pool summary."""
        if self.pool_summary.n_battles_total == 0:
            return 0.0
        return self.pool_summary.n_wins / self.pool_summary.n_battles_total


def random_search_over_pool(
    *,
    catalog: SetCatalog,
    pool: TeamPool,
    n_candidates: int = 20,
    team_size: int = 6,
    n_opponents: int = 5,
    n_battles_per_opponent: int = 5,
    rng: Optional[random.Random] = None,
) -> List[CandidateTeamResult]:
    """
    Run a simple random search over teams sampled from the catalog.

    For each candidate:
      - sample a random team (set ID list),
      - evaluate it against the pool with evaluate_team_against_pool,
      - collect the results.

    Args:
        catalog: SetCatalog instance to sample teams from.
        pool: TeamPool instance to evaluate against.
        n_candidates: Number of candidate teams to evaluate.
        team_size: Number of sets per team (default: 6).
        n_opponents: Number of opponent teams to sample for each candidate.
        n_battles_per_opponent: Number of battles per opponent matchup.
        rng: Optional random number generator. If None, creates a new one.

    Returns:
        A list of CandidateTeamResult, sorted by win_rate descending.
    """
    if rng is None:
        rng = random.Random()

    results: List[CandidateTeamResult] = []

    for _ in range(n_candidates):
        team_set_ids = sample_random_team_set_ids(
            catalog, team_size=team_size, rng=rng
        )

        summary = evaluate_team_against_pool(
            team_set_ids=team_set_ids,
            pool=pool,
            n_opponents=n_opponents,
            n_battles_per_opponent=n_battles_per_opponent,
            catalog=catalog,
        )

        results.append(
            CandidateTeamResult(team_set_ids=team_set_ids, pool_summary=summary)
        )

    # Sort by win_rate descending
    results.sort(key=lambda r: r.win_rate, reverse=True)
    return results

