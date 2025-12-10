"""Team pool evaluation utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Tuple

import torch

from .config import DEFAULT_FORMAT
from .eval import evaluate_team_vs_team
from .set_catalog import SetCatalog
from .team_builder import build_team_import_from_set_ids
from .team_pool import TeamPool, TeamDef
from .team_encoding import SetIdVocab

# Forward reference for type hints (avoid circular import)
if TYPE_CHECKING:
    from .team_value_model import TeamValueModel


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

