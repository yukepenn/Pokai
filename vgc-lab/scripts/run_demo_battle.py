#!/usr/bin/env python3
"""Demo script: Generate two random teams and simulate a battle (non-interactive)."""

import sys
from pathlib import Path

# Add src to path so we can import vgc_lab when running as a script
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.battle_logger import (
    BattleResult,
    save_battle_result,
    save_raw_log,
    infer_winner_from_log,
    count_turns,
)
from vgc_lab.config import DEFAULT_FORMAT, ensure_paths
from vgc_lab.showdown_cli import generate_random_team, simulate_battle


def main() -> None:
    """Run a demo battle between two randomly generated teams using simulate_battle only."""
    try:
        # Ensure data directories exist
        ensure_paths()

        print(f"Generating random teams for format: {DEFAULT_FORMAT}")
        p1_team = generate_random_team(DEFAULT_FORMAT)
        p2_team = generate_random_team(DEFAULT_FORMAT)

        print(f"P1 team (first 100 chars): {p1_team[:100]}...")
        print(f"P2 team (first 100 chars): {p2_team[:100]}...")

        print("\nRunning battle simulation (one-shot CLI, no streaming bots)...")
        battle_log = simulate_battle(
            format_id=DEFAULT_FORMAT,
            p1_name="Bot1",
            p1_packed_team=p1_team,
            p2_name="Bot2",
            p2_packed_team=p2_team,
        )

        print(f"Battle log length: {len(battle_log)} characters")

        # Save raw log
        raw_log_path = save_raw_log(DEFAULT_FORMAT, battle_log)
        print(f"\nSaved raw log to: {raw_log_path}")

        # Infer winner and turn count (best-effort)
        winner_side = infer_winner_from_log(battle_log, "Bot1", "Bot2")
        turns = count_turns(battle_log)

        result = BattleResult(
            format_id=DEFAULT_FORMAT,
            p1_name="Bot1",
            p2_name="Bot2",
            winner=winner_side,  # "p1", "p2", "tie", or "unknown"
            raw_log_path=raw_log_path,
            turns=turns,
        )

        result_json_path = save_battle_result(result)
        print(f"Saved result JSON to: {result_json_path}")

        # Print summary
        print("\n" + "=" * 60)
        print("BATTLE SUMMARY")
        print("=" * 60)
        print(f"Format: {result.format_id}")
        print(f"Winner side: {result.winner}")
        print(f"Turns: {result.turns if result.turns is not None else 'unknown'}")
        print(f"Raw log: {result.raw_log_path}")
        print(f"Result JSON: {result_json_path}")
        print("=" * 60)

    except RuntimeError as e:
        error_msg = str(e)
        if "Node.js not found" in error_msg or "node" in error_msg.lower():
            print(
                "ERROR: Node.js is not installed or not in PATH.\n"
                "Please install Node.js 16+ from https://nodejs.org/\n"
                "Then ensure 'node' and 'npm' are available in your PATH."
            )
        elif "Showdown directory not found" in error_msg:
            print(
                f"ERROR: {error_msg}\n"
                "Ensure pokemon-showdown/ is a sibling directory of vgc-lab/."
            )
        else:
            print(f"ERROR: {error_msg}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
