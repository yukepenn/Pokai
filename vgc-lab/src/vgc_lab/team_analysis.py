"""Team aggregation and analysis utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

from .team_features import TeamsVsPoolRecord, iter_teams_vs_pool_records


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

        # Try to get win_rate from meta if it exists, otherwise calculate
        # But we'll recalculate from aggregates for consistency
        # If the records have explicit win_rate in meta, we might need to weight-average
        # For now, we calculate from aggregated wins/battles which is more robust

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

