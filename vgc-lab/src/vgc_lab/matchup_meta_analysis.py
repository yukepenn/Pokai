"""Meta and matchup analysis utilities using the team matchup model."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from .team_encoding import SetIdVocab
from .team_matchup_model import TeamMatchupModel, predict_matchup_win_prob
from .team_pool import TeamPool


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

