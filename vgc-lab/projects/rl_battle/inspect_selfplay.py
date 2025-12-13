"""
Analysis tool for inspecting saved online self-play battles.

This module provides utilities to load and summarize battle data persisted
by BattleStore.append_full_battle_from_json().
"""

import json
import statistics
from pathlib import Path
from typing import Dict, List, Optional

from datetime import datetime, timezone

from vgc_lab.core import FullBattleRecord, get_paths


def iter_full_battles(max_battles: Optional[int] = None, paths=None):
    """
    Iterate over FullBattleRecord entries from the full_battles.jsonl file.
    
    Args:
        max_battles: Maximum number of battles to yield (None = all)
        paths: Paths instance (defaults to get_paths())
    
    Yields:
        FullBattleRecord instances
    """
    if paths is None:
        paths = get_paths()
    
    json_path = paths.full_battles_jsonl
    if not json_path.exists():
        return
    
    count = 0
    with json_path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_battles is not None and count >= max_battles:
                break
            
            line = line.strip()
            if not line:
                continue
            
            try:
                data = json.loads(line)
                # Convert raw_log_path string back to Path
                if "raw_log_path" in data:
                    data["raw_log_path"] = Path(data["raw_log_path"])
                # Convert created_at ISO string back to datetime
                if "created_at" in data:
                    data["created_at"] = datetime.fromisoformat(data["created_at"])
                
                record = FullBattleRecord(**data)
                yield record
                count += 1
            except (json.JSONDecodeError, KeyError, ValueError) as e:
                # Skip malformed lines but continue
                continue


def summarize_selfplay(max_battles: Optional[int] = None, paths=None) -> Dict[str, any]:
    """
    Load up to `max_battles` battles from the existing BattleStore
    and return simple aggregate stats.
    
    Args:
        max_battles: Maximum number of battles to load (None = all available)
        paths: Paths instance (defaults to get_paths())
    
    Returns:
        Dictionary with:
        - total_battles: int
        - winner_counts: {"p1": int, "p2": int, "tie": int, "unknown": int}
        - avg_turns: float | None
        - median_turns: float | None
    """
    winner_counts = {"p1": 0, "p2": 0, "tie": 0, "unknown": 0}
    turns_list: List[int] = []
    
    for record in iter_full_battles(max_battles=max_battles, paths=paths):
        # Count winners
        winner_side = record.winner_side
        if winner_side in winner_counts:
            winner_counts[winner_side] += 1
        else:
            # Normalize unknown winner_side values
            winner_counts["unknown"] += 1
        
        # Collect turns
        if record.turns is not None:
            turns_list.append(record.turns)
    
    total_battles = sum(winner_counts.values())
    
    # Calculate turn statistics
    avg_turns = statistics.mean(turns_list) if turns_list else None
    median_turns = statistics.median(turns_list) if turns_list else None
    
    return {
        "total_battles": total_battles,
        "winner_counts": winner_counts,
        "avg_turns": avg_turns,
        "median_turns": median_turns,
    }


def main() -> None:
    """CLI entry point for inspect_selfplay."""
    import sys
    
    max_battles = None
    if len(sys.argv) > 1:
        try:
            max_battles = int(sys.argv[1])
        except ValueError:
            print(f"Invalid max_battles argument: {sys.argv[1]}", file=sys.stderr)
            sys.exit(1)
    
    stats = summarize_selfplay(max_battles=max_battles)
    
    print("Self-play dataset summary")
    print("-" * 25)
    print(f"Total battles: {stats['total_battles']}")
    print("Winner counts:")
    print(f"  p1:      {stats['winner_counts']['p1']:4d}")
    print(f"  p2:      {stats['winner_counts']['p2']:4d}")
    print(f"  tie:     {stats['winner_counts']['tie']:4d}")
    print(f"  unknown: {stats['winner_counts']['unknown']:4d}")
    
    if stats['avg_turns'] is not None:
        print(f"Avg turns:   {stats['avg_turns']:.1f}")
    else:
        print("Avg turns:   N/A")
    
    if stats['median_turns'] is not None:
        print(f"Median turns: {stats['median_turns']:.1f}")
    else:
        print("Median turns: N/A")


if __name__ == "__main__":
    main()

