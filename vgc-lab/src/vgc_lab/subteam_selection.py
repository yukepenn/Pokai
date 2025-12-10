"""Subteam selection utilities for BO3-style 4-of-6 selection."""

from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Sequence, Tuple

from .team_encoding import SetIdVocab
from .team_matchup_model import TeamMatchupModel, predict_matchup_win_prob
from .team_pool import TeamPool


def iter_subteams_4_of_6(set_ids: Sequence[str]) -> Iterable[Tuple[str, ...]]:
    """
    Yield all 4-of-6 combinations from a 6-set team.

    Args:
        set_ids: Sequence of set ID strings (should be length 6 for VGC).

    Yields:
        Tuples of 4 set IDs (all combinations).

    Note:
        If len(set_ids) < 6, yields empty or the full team depending on length.
        If len(set_ids) > 6, yields all combinations (may be many).
    """
    if len(set_ids) < 4:
        # Can't form a 4-of-6 subteam
        return
    if len(set_ids) == 4:
        yield tuple(set_ids)
        return
    if len(set_ids) == 6:
        # Standard VGC case: 6 choose 4 = 15 combinations
        for combo in combinations(set_ids, 4):
            yield tuple(combo)
    else:
        # Handle other sizes (for future flexibility)
        for combo in combinations(set_ids, 4):
            yield tuple(combo)


def estimate_subteam_win_prob_vs_team(
    subteam_set_ids: Sequence[str],
    opponent_set_ids: Sequence[str],
    model: TeamMatchupModel,
    vocab: SetIdVocab,
    *,
    device: str = "cpu",
) -> float:
    """
    Use the existing matchup model to approximate P(subteam wins vs opponent full team).

    This is an approximation: we treat the 4-set subteam as if it were a full 6-set team
    and compare it to the opponent's full 6-set team.

    Args:
        subteam_set_ids: Sequence of 4 set IDs for the subteam.
        opponent_set_ids: Sequence of 6 set IDs for the opponent's full team.
        model: Trained TeamMatchupModel instance.
        vocab: SetIdVocab for encoding.
        device: Device for model inference.

    Returns:
        Predicted probability that the subteam wins (float in [0, 1]).
    """
    return predict_matchup_win_prob(
        model=model,
        vocab=vocab,
        team_a_set_ids=list(subteam_set_ids),
        team_b_set_ids=list(opponent_set_ids),
        device=device,
    )


def rank_subteams_vs_opponent(
    our_team_id: str,
    opp_team_id: str,
    team_pool: TeamPool,
    model: TeamMatchupModel,
    vocab: SetIdVocab,
    *,
    max_subteams: int = 15,
    device: str = "cpu",
) -> List[Tuple[Tuple[str, ...], float]]:
    """
    Rank subteams (4-of-6) from our team by estimated win_prob vs opponent.

    Args:
        our_team_id: ID of our team in the pool.
        opp_team_id: ID of opponent team in the pool.
        team_pool: TeamPool instance to look up teams.
        model: Trained TeamMatchupModel instance.
        vocab: SetIdVocab for encoding.
        max_subteams: Maximum number of subteams to evaluate (default 15 for 6 choose 4).
        device: Device for model inference.

    Returns:
        List of (subteam_set_ids_tuple, estimated_p_win) tuples, sorted by p_win descending.
    """
    # Look up teams
    our_team = team_pool.get(our_team_id)
    opp_team = team_pool.get(opp_team_id)

    our_set_ids = our_team.set_ids
    opp_set_ids = opp_team.set_ids

    # Generate subteams
    all_subteams = list(iter_subteams_4_of_6(our_set_ids))

    # Limit if needed
    if len(all_subteams) > max_subteams:
        all_subteams = all_subteams[:max_subteams]

    # Evaluate each subteam
    results: List[Tuple[Tuple[str, ...], float]] = []

    for subteam_tuple in all_subteams:
        try:
            p_win = estimate_subteam_win_prob_vs_team(
                subteam_set_ids=subteam_tuple,
                opponent_set_ids=opp_set_ids,
                model=model,
                vocab=vocab,
                device=device,
            )
            results.append((subteam_tuple, p_win))
        except Exception:
            # Skip if evaluation fails (e.g., set_id not in vocab)
            continue

    # Sort by win probability descending
    results.sort(key=lambda x: x[1], reverse=True)

    return results

