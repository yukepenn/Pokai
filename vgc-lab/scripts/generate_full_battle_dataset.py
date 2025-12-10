#!/usr/bin/env python3
"""Generate full battle dataset from random self-play battles."""

import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import typer

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.battle_logger import (
    append_full_battle_record,
    ensure_full_battle_dataset_dir,
    FullBattleRecord,
    save_raw_log,
)
from vgc_lab.config import DEFAULT_FORMAT, ensure_paths
from vgc_lab.showdown_cli import run_random_selfplay


def main(
    n: int = typer.Option(10, "--n", "-n", help="Number of full battles to generate"),
):
    """
    Generate `n` complete random self-play battles using Showdown's internal RandomPlayer AI.

    For each battle:
      - Call run_random_selfplay(...)
      - Save raw log
      - Build a FullBattleRecord
      - Append it to full_battles.jsonl
    """
    ensure_paths()
    ensure_full_battle_dataset_dir()

    successful = 0
    failed = 0

    print(f"Generating {n} full battle records...")
    print(f"Format: {DEFAULT_FORMAT}")
    print()

    for i in range(1, n + 1):
        try:
            # Run random selfplay battle
            log_text, winner_side, turns, p1_name, p2_name = run_random_selfplay(
                format_id=DEFAULT_FORMAT
            )

            # Save raw log
            raw_log_path = save_raw_log(DEFAULT_FORMAT, log_text)

            # Determine winner_name from winner_side
            winner_name = None
            if winner_side == "p1":
                winner_name = p1_name
            elif winner_side == "p2":
                winner_name = p2_name
            # else: tie or unknown, winner_name stays None

            # Create FullBattleRecord
            record = FullBattleRecord(
                format_id=DEFAULT_FORMAT,
                p1_name=p1_name,
                p2_name=p2_name,
                winner_side=winner_side,
                winner_name=winner_name,
                turns=turns,
                raw_log_path=raw_log_path,
                created_at=datetime.now(timezone.utc),
            )

            # Append to dataset
            append_full_battle_record(record)
            successful += 1

            print(f"[{i}/{n}] winner={winner_side}, turns={turns}")

            # Small delay to avoid OneDrive race conditions
            if i < n:
                time.sleep(0.1)

        except RuntimeError as e:
            failed += 1
            print(f"  Error [{i}]: {e}")

        except Exception as e:
            failed += 1
            print(f"  Unexpected error [{i}]: {e}")

    # Print summary
    print()
    print("=" * 60)
    print("FULL BATTLE DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total attempts: {n}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print()

    from vgc_lab.battle_logger import FULL_BATTLE_DATASET_PATH

    print(f"Dataset file: {FULL_BATTLE_DATASET_PATH}")
    if FULL_BATTLE_DATASET_PATH.exists():
        line_count = sum(1 for _ in FULL_BATTLE_DATASET_PATH.open("r", encoding="utf-8"))
        print(f"Total lines in dataset: {line_count}")
    print("=" * 60)


if __name__ == "__main__":
    typer.run(main)

