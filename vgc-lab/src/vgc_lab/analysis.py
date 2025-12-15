"""Analysis: evaluation, team aggregation, meta analysis, and team search.

This module consolidates:
- eval.py: Team evaluation utilities
- team_analysis.py: Team aggregation and analysis utilities
- matchup_meta_analysis.py: Meta and matchup analysis utilities using the team matchup model
- team_search.py: Team pool evaluation utilities
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from .catalog import SetCatalog, TeamPool, TeamDef, build_team_import_from_set_ids, import_text_to_packed
from .config import DEFAULT_FORMAT
from .dataio import save_raw_log
from .features import SetIdVocab, TeamsVsPoolRecord, iter_teams_vs_pool_records
from .models import TeamMatchupModel, predict_matchup_win_prob
from .showdown import run_random_selfplay_json

# Forward reference for type hints (avoid circular import)
if TYPE_CHECKING:
    from .models import TeamValueModel

# ============================================================================
# Team Evaluation
# ============================================================================


@dataclass
class TeamEvalSummary:
    """Summary of team evaluation results."""

    format_id: str
    team_role: str  # "p1" or "p2"
    team_packed: str
    n_battles: int
    n_wins: int
    n_losses: int
    n_ties: int
    avg_turns: Optional[float]
    created_at: datetime


def evaluate_team_vs_random(
    team_packed: str,
    *,
    format_id: str = DEFAULT_FORMAT,
    n: int = 100,
    team_as: str = "p1",
    save_logs: bool = False,
) -> TeamEvalSummary:
    """
    Evaluate a fixed packed team against random opponents, using RandomPlayerAI.

    Args:
        team_packed: Showdown packed team string for a 6-Pokémon team.
        format_id:   Format ID, default is DEFAULT_FORMAT (gen9vgc2026regf).
        n:           Number of battles to run.
        team_as:     Which side to place the fixed team on: "p1", "p2", or "both".
                     If "both", alternates between P1 and P2 (50/50 split).
        save_logs:   If True, save each battle log to data/battles_raw/.

    Returns:
        TeamEvalSummary with basic statistics.
    """
    assert team_as in ("p1", "p2", "both"), f"team_as must be 'p1', 'p2', or 'both', got {team_as!r}"

    n_wins = 0
    n_losses = 0
    n_ties = 0
    turns_sum = 0
    turns_count = 0

    for i in range(n):
        # Determine which side the fixed team should be on for this battle
        if team_as == "both":
            # Alternate: even indices → P1, odd indices → P2
            current_side = "p1" if i % 2 == 0 else "p2"
        else:
            current_side = team_as

        # Run battle with fixed team on appropriate side
        if current_side == "p1":
            data = run_random_selfplay_json(
                format_id=format_id,
                p1_packed_team=team_packed,
                p2_packed_team=None,
            )
        else:
            data = run_random_selfplay_json(
                format_id=format_id,
                p1_packed_team=None,
                p2_packed_team=team_packed,
            )

        winner_side = data.get("winner_side", "unknown")
        turns = data.get("turns", None)
        log_text = data.get("log", "")

        # Optionally save battle log
        if save_logs and log_text:
            save_raw_log(format_id, log_text)

        # Count wins/losses/ties from fixed team's perspective
        if winner_side == "tie" or winner_side == "unknown":
            n_ties += 1
        else:
            if current_side == "p1":
                if winner_side == "p1":
                    n_wins += 1
                elif winner_side == "p2":
                    n_losses += 1
                else:
                    n_ties += 1
            else:  # current_side == "p2"
                if winner_side == "p2":
                    n_wins += 1
                elif winner_side == "p1":
                    n_losses += 1
                else:
                    n_ties += 1

        if isinstance(turns, int):
            turns_sum += turns
            turns_count += 1

    avg_turns: Optional[float]
    if turns_count > 0:
        avg_turns = turns_sum / turns_count
    else:
        avg_turns = None

    # Set team_role for summary
    team_role = "both" if team_as == "both" else team_as

    return TeamEvalSummary(
        format_id=format_id,
        team_role=team_role,
        team_packed=team_packed,
        n_battles=n,
        n_wins=n_wins,
        n_losses=n_losses,
        n_ties=n_ties,
        avg_turns=avg_turns,
        created_at=datetime.now(timezone.utc),
    )


@dataclass
class MatchupEvalSummary:
    """Summary of team vs team matchup evaluation results."""

    format_id: str
    team_a_import: str
    team_b_import: str
    team_a_packed: str
    team_b_packed: str
    n_battles: int
    n_a_wins: int
    n_b_wins: int
    n_ties: int
    avg_turns: Optional[float]
    created_at: datetime


def evaluate_team_vs_team(
    team_a_import: str,
    team_b_import: str,
    *,
    format_id: str = DEFAULT_FORMAT,
    n: int = 100,
) -> MatchupEvalSummary:
    """
    Evaluate Team A vs Team B using RandomPlayerAI on both sides.

    Team A and Team B are provided in Showdown import/export text format.
    We import+pack both, then run many self-play battles.

    Args:
        team_a_import: Showdown team text for Team A.
        team_b_import: Showdown team text for Team B.
        format_id:     Format ID, default is gen9vgc2026regf.
        n:             Number of battles to run.

    Returns:
        MatchupEvalSummary with win/loss/tie stats and average turns.
    """
    # Convert both import texts to packed strings
    team_a_packed = import_text_to_packed(team_a_import, format_id=format_id)
    team_b_packed = import_text_to_packed(team_b_import, format_id=format_id)

    n_a_wins = 0
    n_b_wins = 0
    n_ties = 0
    turns_sum = 0
    turns_count = 0

    for i in range(n):
        # Alternate sides to avoid seat bias
        # Even i: Team A as P1, Team B as P2
        # Odd i:  Team B as P1, Team A as P2
        if i % 2 == 0:
            # Team A as P1, Team B as P2
            data = run_random_selfplay_json(
                format_id=format_id,
                p1_packed_team=team_a_packed,
                p2_packed_team=team_b_packed,
            )
            winner_side = data.get("winner_side", "unknown")
            # Count from Team A's perspective
            if winner_side == "p1":
                n_a_wins += 1
            elif winner_side == "p2":
                n_b_wins += 1
            else:  # tie or unknown
                n_ties += 1
        else:
            # Team B as P1, Team A as P2
            data = run_random_selfplay_json(
                format_id=format_id,
                p1_packed_team=team_b_packed,
                p2_packed_team=team_a_packed,
            )
            winner_side = data.get("winner_side", "unknown")
            # Count from Team A's perspective (A is P2 here)
            if winner_side == "p2":
                n_a_wins += 1
            elif winner_side == "p1":
                n_b_wins += 1
            else:  # tie or unknown
                n_ties += 1

        turns = data.get("turns", None)
        if isinstance(turns, int):
            turns_sum += turns
            turns_count += 1

    avg_turns: Optional[float]
    if turns_count > 0:
        avg_turns = turns_sum / turns_count
    else:
        avg_turns = None

    return MatchupEvalSummary(
        format_id=format_id,
        team_a_import=team_a_import,
        team_b_import=team_b_import,
        team_a_packed=team_a_packed,
        team_b_packed=team_b_packed,
        n_battles=n,
        n_a_wins=n_a_wins,
        n_b_wins=n_b_wins,
        n_ties=n_ties,
        avg_turns=avg_turns,
        created_at=datetime.now(timezone.utc),
    )


# ============================================================================
# Team Analysis
# ============================================================================


@dataclass
class TeamAggregateStats:
    """Aggregate statistics for a unique team (identified by its set_ids)."""

    # Canonical representation of this team (we'll store sorted set_ids for stability)
    team_set_ids: List[str]

    # Evaluation counts
    n_records: int
    n_battles_total: int
    n_wins: int
    n_losses: int
    n_ties: int

    # Derived metrics
    win_rate: float

    # Source breakdown (e.g. random / catalog / auto_iter_X_model / auto_iter_X_mut / auto_iter_X_random)
    source_counts: Dict[str, int]


def aggregate_teams(
    records: Iterable[TeamsVsPoolRecord],
) -> List[TeamAggregateStats]:
    """
    Aggregate teams_vs_pool records by unique team_set_ids combination.

    - Teams are identified by the *multiset* of set_ids, ignoring order.
      We can use a sorted tuple of set_ids as the key.
    - For each key, aggregate:
        - n_records: number of JSONL records for this team.
        - n_battles_total: sum over records.
        - n_wins / n_losses / n_ties: sums over records.
        - win_rate: n_wins / n_battles_total if n_battles_total > 0, else 0.0.
        - source_counts: Counter over "source" fields.

    - Return a list of TeamAggregateStats, without sorting; sorting will be done
      by the CLI script.

    Args:
        records: Iterable of TeamsVsPoolRecord instances.

    Returns:
        List of TeamAggregateStats, one per unique team.
    """
    # Aggregate by team key (sorted tuple of set_ids)
    team_dict: Dict[tuple, Dict] = {}

    for record in records:
        # Use sorted tuple as key for canonical representation
        key = tuple(sorted(record.team_set_ids))

        if key not in team_dict:
            team_dict[key] = {
                "team_set_ids": list(record.team_set_ids),  # Original order (or we can sort later)
                "n_records": 0,
                "n_battles_total": 0,
                "n_wins": 0,
                "n_losses": 0,
                "n_ties": 0,
                "source_counter": Counter(),
            }

        agg = team_dict[key]
        agg["n_records"] += 1
        agg["n_battles_total"] += record.n_battles_total
        # Extract wins/losses/ties from meta dict
        agg["n_wins"] += record.meta.get("n_wins", 0) or 0
        agg["n_losses"] += record.meta.get("n_losses", 0) or 0
        agg["n_ties"] += record.meta.get("n_ties", 0) or 0

        # Track source
        source = record.source or "unknown"
        agg["source_counter"][source] += 1

    # Convert to TeamAggregateStats
    results: List[TeamAggregateStats] = []

    for key, agg_data in team_dict.items():
        team_set_ids_sorted = sorted(key)  # Canonical sorted list

        n_battles = agg_data["n_battles_total"]
        n_wins = agg_data["n_wins"]
        n_losses = agg_data["n_losses"]
        n_ties = agg_data["n_ties"]

        # Calculate win_rate
        if n_battles > 0:
            win_rate = n_wins / n_battles
        else:
            win_rate = 0.0

        stats = TeamAggregateStats(
            team_set_ids=team_set_ids_sorted,
            n_records=agg_data["n_records"],
            n_battles_total=n_battles,
            n_wins=n_wins,
            n_losses=n_losses,
            n_ties=n_ties,
            win_rate=win_rate,
            source_counts=dict(agg_data["source_counter"]),
        )
        results.append(stats)

    return results


def load_and_aggregate_teams_vs_pool(jsonl_path: Path) -> List[TeamAggregateStats]:
    """
    Convenience wrapper: read JSONL and aggregate teams.

    Args:
        jsonl_path: Path to teams_vs_pool.jsonl file.

    Returns:
        List of TeamAggregateStats, one per unique team.
    """
    records = list(iter_teams_vs_pool_records(jsonl_path))
    return aggregate_teams(records)


# ============================================================================
# Meta Analysis
# ============================================================================


@dataclass
class MetaTeam:
    """A single team entry in a meta distribution."""

    team_id: str
    weight: float  # relative weight (not necessarily normalized yet)


@dataclass
class MetaDistribution:
    """Discrete meta distribution over team IDs."""

    entries: List[MetaTeam]

    def normalized(self) -> "MetaDistribution":
        """
        Return a new MetaDistribution with weights normalized to sum to 1.0.

        Returns:
            New MetaDistribution instance with normalized weights.
        """
        total_weight = sum(entry.weight for entry in self.entries)
        if total_weight == 0.0:
            # If all weights are zero, return uniform distribution
            if not self.entries:
                return self
            uniform_weight = 1.0 / len(self.entries)
            normalized_entries = [
                MetaTeam(team_id=entry.team_id, weight=uniform_weight)
                for entry in self.entries
            ]
            return MetaDistribution(entries=normalized_entries)

        normalized_entries = [
            MetaTeam(team_id=entry.team_id, weight=entry.weight / total_weight)
            for entry in self.entries
        ]
        return MetaDistribution(entries=normalized_entries)


def build_uniform_meta(team_pool: TeamPool, team_ids: Iterable[str]) -> MetaDistribution:
    """
    Create a uniform distribution over the given team IDs.

    Args:
        team_pool: TeamPool instance (for validation).
        team_ids: Iterable of team IDs to include in the meta.

    Returns:
        MetaDistribution with uniform weights (not normalized yet).

    Raises:
        KeyError: If any team_id is not found in the pool.
    """
    team_ids_list = list(team_ids)
    weight = 1.0  # Uniform weight

    entries = []
    for team_id in team_ids_list:
        # Validate team exists
        team_pool.get(team_id)
        entries.append(MetaTeam(team_id=team_id, weight=weight))

    return MetaDistribution(entries=entries)


def normalize_meta(meta: MetaDistribution) -> MetaDistribution:
    """
    Normalize a meta distribution so weights sum to 1.0.

    Args:
        meta: MetaDistribution to normalize.

    Returns:
        New MetaDistribution with normalized weights.
    """
    return meta.normalized()


def compute_expected_win_vs_meta(
    team_id: str,
    meta: MetaDistribution,
    team_pool: TeamPool,
    model: TeamMatchupModel,
    vocab: SetIdVocab,
    *,
    device: str = "cpu",
) -> float:
    """
    Compute expected win rate of a team vs a meta distribution.

    For each entry (opponent_id, weight) in meta:
      - Look up both teams in TeamPool (their set_ids).
      - Use predict_matchup_win_prob to get P(my_team wins vs opp).
      - Accumulate weight * p_win.

    If some opponent teams have set_ids not in vocab, skip them and renormalize
    remaining weights.

    Args:
        team_id: ID of the team to evaluate.
        meta: MetaDistribution of opponent teams.
        team_pool: TeamPool instance to look up teams.
        model: Trained TeamMatchupModel instance.
        vocab: SetIdVocab for encoding.
        device: Device for model inference.

    Returns:
        Expected win rate (float in [0, 1]).

    Raises:
        KeyError: If team_id is not found in the pool.
    """
    # Get our team
    my_team = team_pool.get(team_id)
    my_set_ids = my_team.set_ids

    # Accumulate weighted wins
    total_weight = 0.0
    weighted_win_sum = 0.0

    valid_entries = []

    for meta_entry in meta.entries:
        opp_id = meta_entry.team_id
        weight = meta_entry.weight

        try:
            # Look up opponent team
            opp_team = team_pool.get(opp_id)
            opp_set_ids = opp_team.set_ids

            # Predict win probability
            p_win = predict_matchup_win_prob(
                model=model,
                vocab=vocab,
                team_a_set_ids=my_set_ids,
                team_b_set_ids=opp_set_ids,
                device=device,
            )

            weighted_win_sum += weight * p_win
            total_weight += weight
            valid_entries.append((opp_id, p_win, weight))

        except KeyError:
            # Skip if team or set_ids not found
            continue

    # Handle case where no valid entries
    if total_weight == 0.0:
        return 0.5  # Default to 50% if no valid matchups

    # Expected win rate (weighted average)
    expected_win = weighted_win_sum / total_weight

    return float(expected_win)


def rank_teams_vs_meta(
    candidate_team_ids: Iterable[str],
    meta: MetaDistribution,
    team_pool: TeamPool,
    model: TeamMatchupModel,
    vocab: SetIdVocab,
    *,
    device: str = "cpu",
) -> List[Tuple[str, float]]:
    """
    Rank candidate teams by their expected win rate vs a meta.

    Args:
        candidate_team_ids: Iterable of team IDs to evaluate.
        meta: MetaDistribution of opponent teams.
        team_pool: TeamPool instance to look up teams.
        model: Trained TeamMatchupModel instance.
        vocab: SetIdVocab for encoding.
        device: Device for model inference.

    Returns:
        List of (team_id, expected_win_rate_vs_meta) tuples, sorted by
        expected_win_rate descending.
    """
    results: List[Tuple[str, float]] = []

    for team_id in candidate_team_ids:
        try:
            expected_win = compute_expected_win_vs_meta(
                team_id=team_id,
                meta=meta,
                team_pool=team_pool,
                model=model,
                vocab=vocab,
                device=device,
            )
            results.append((team_id, expected_win))
        except KeyError:
            # Skip if team not found
            continue
        except Exception:
            # Skip on any other error (e.g., set_id not in vocab)
            continue

    # Sort by expected win rate descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results


# ============================================================================
# Team Search
# ============================================================================


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


@dataclass
class ModelGuidedProposal:
    """A candidate team proposal with predicted win rate from the value model."""

    set_ids: List[str]
    pred_win_rate: float


def propose_candidates_with_model(
    model: "TeamValueModel",  # Forward reference to avoid circular import
    vocab: SetIdVocab,
    catalog: SetCatalog,
    *,
    n_proposals: int,
    team_size: int = 6,
    rng: Optional[random.Random] = None,
    device: str = "cpu",
    max_attempts: int = 10000,
    batch_size: int = 256,
) -> List[ModelGuidedProposal]:
    """
    Use the value model to propose promising candidate teams.

    - Sample random teams from the catalog via sample_random_team_set_ids().
    - Deduplicate by sorted tuple of set IDs.
    - Score them in batches with the neural value model.
    - Return a list of ModelGuidedProposal sorted by pred_win_rate DESC.

    Args:
        model: Trained TeamValueModel instance.
        vocab: SetIdVocab for encoding set IDs to indices.
        catalog: SetCatalog to sample teams from.
        n_proposals: Number of distinct candidate teams to propose.
        team_size: Number of sets per team (default: 6).
        rng: Optional random number generator. If None, uses random module.
        device: Device to run model inference on ("cpu" or "cuda").
        max_attempts: Maximum number of sampling attempts before giving up.
        batch_size: Batch size for model inference.

    Returns:
        List of ModelGuidedProposal sorted by pred_win_rate descending.
    """
    if rng is None:
        rng = random.Random()

    # Collect distinct teams
    seen_teams: set[Tuple[str, ...]] = set()
    team_list: List[List[str]] = []

    attempts = 0
    while len(team_list) < n_proposals and attempts < max_attempts:
        attempts += 1
        try:
            team_set_ids = sample_random_team_set_ids(
                catalog, team_size=team_size, rng=rng
            )
            team_key = tuple(sorted(team_set_ids))
            if team_key not in seen_teams:
                seen_teams.add(team_key)
                team_list.append(team_set_ids)
        except ValueError:
            # Skip if sampling fails
            continue

    if len(team_list) < n_proposals:
        import warnings
        warnings.warn(
            f"Only found {len(team_list)} distinct teams after {attempts} attempts "
            f"(requested {n_proposals}). Returning what we have."
        )

    if not team_list:
        return []

    # Score teams in batches
    model.eval()
    proposals: List[ModelGuidedProposal] = []

    with torch.no_grad():
        for batch_start in range(0, len(team_list), batch_size):
            batch_end = min(batch_start + batch_size, len(team_list))
            batch_teams = team_list[batch_start:batch_end]

            # Encode batch
            batch_indices = []
            for team_set_ids in batch_teams:
                try:
                    indices = vocab.encode_ids(team_set_ids)
                    batch_indices.append(indices)
                except KeyError:
                    # Skip teams with unknown set IDs
                    continue

            if not batch_indices:
                continue

            # Convert to tensor
            indices_tensor = torch.tensor(batch_indices, dtype=torch.long, device=device)

            # Forward pass
            pred_win_rates = model(indices_tensor).cpu().tolist()

            # Create proposals
            for team_set_ids, pred_win_rate in zip(batch_teams, pred_win_rates):
                proposals.append(
                    ModelGuidedProposal(set_ids=team_set_ids, pred_win_rate=pred_win_rate)
                )

    # Sort by predicted win rate descending
    proposals.sort(key=lambda p: p.pred_win_rate, reverse=True)
    return proposals


def evaluate_top_model_candidates_against_pool(
    model: "TeamValueModel",  # Forward reference
    vocab: SetIdVocab,
    catalog: SetCatalog,
    pool: TeamPool,
    *,
    n_proposals: int,
    top_k: int,
    n_opponents: int,
    n_per_opponent: int,
    rng: Optional[random.Random] = None,
    device: str = "cpu",
) -> List[Tuple[ModelGuidedProposal, CandidateTeamResult]]:
    """
    Use model to propose candidates, take top-k, and evaluate vs pool.

    - Use propose_candidates_with_model(...) to get n_proposals candidates.
    - Take the top_k proposals by predicted win_rate.
    - For each, evaluate vs the TeamPool using evaluate_team_against_pool(...).
    - Return a list of (proposal, result) pairs.

    This does NOT write to disk; a higher-level CLI or function will decide
    how/where to persist these results (e.g., JSONL via teams_vs_pool_data).

    Args:
        model: Trained TeamValueModel instance.
        vocab: SetIdVocab for encoding.
        catalog: SetCatalog to sample from.
        pool: TeamPool to evaluate against.
        n_proposals: Number of candidates to propose.
        top_k: Number of top candidates to evaluate.
        n_opponents: Number of opponent teams to sample per candidate.
        n_per_opponent: Number of battles per opponent.
        rng: Optional random number generator.
        device: Device for model inference.

    Returns:
        List of (ModelGuidedProposal, CandidateTeamResult) pairs.
    """
    # Propose candidates
    proposals = propose_candidates_with_model(
        model=model,
        vocab=vocab,
        catalog=catalog,
        n_proposals=n_proposals,
        rng=rng,
        device=device,
    )

    # Take top-k
    selected = proposals[:top_k]

    # Evaluate each selected candidate
    pairs: List[Tuple[ModelGuidedProposal, CandidateTeamResult]] = []

    for proposal in selected:
        summary = evaluate_team_against_pool(
            team_set_ids=proposal.set_ids,
            pool=pool,
            n_opponents=n_opponents,
            n_battles_per_opponent=n_per_opponent,
            catalog=catalog,
        )
        result = CandidateTeamResult(
            team_set_ids=proposal.set_ids, pool_summary=summary
        )
        pairs.append((proposal, result))

    return pairs


def mutate_team_set_ids(
    base_set_ids: List[str],
    catalog: SetCatalog,
    *,
    n_mutations: int = 1,
    rng: Optional[random.Random] = None,
) -> List[str]:
    """
    Return a mutated copy of base_set_ids.

    Mutation rule (simple, but safe):

    - Work on a copy of base_set_ids.
    - For each mutation step:
        - Pick a random slot index i in [0, len(team)-1].
        - Sample a new set_id from the catalog (catalog.ids()) that:
            - Is different from the current set_id at i.
            - Is not already present in the team, if possible (avoid duplicates).
        - Replace slot i with the new set_id.
    - If the catalog is too small to find a distinct replacement, leave that slot unchanged.

    This is a generic local search primitive; we'll use it in auto-iteration to explore
    neighborhoods around top candidates.

    Args:
        base_set_ids: List of 6 set IDs to mutate.
        catalog: SetCatalog to sample new set IDs from.
        n_mutations: Number of mutation steps to perform (default: 1).
        rng: Optional random number generator. If None, uses random module.

    Returns:
        A new list of set IDs (mutated copy of base_set_ids).
    """
    if rng is None:
        rng = random

    mutated = list(base_set_ids)  # Work on a copy
    all_set_ids = catalog.ids()

    for _ in range(n_mutations):
        slot_idx = rng.randint(0, len(mutated) - 1)
        current_id = mutated[slot_idx]

        # Find candidates: different from current and ideally not in team
        candidates = [sid for sid in all_set_ids if sid != current_id]

        if not candidates:
            # Cannot mutate if catalog is too small
            continue

        # Prefer candidates not already in the team
        not_in_team = [sid for sid in candidates if sid not in mutated]
        if not_in_team:
            candidates = not_in_team

        # Sample a replacement
        new_id = rng.choice(candidates)
        mutated[slot_idx] = new_id

    return mutated
