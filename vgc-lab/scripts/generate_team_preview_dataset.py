#!/usr/bin/env python3
"""Generate team preview dataset from simulated battles."""

import sys
from pathlib import Path

import typer

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.battle_logger import save_raw_log
from vgc_lab.config import DEFAULT_FORMAT, ensure_paths
from vgc_lab.showdown_cli import generate_random_team, simulate_battle
from vgc_lab.team_preview import (
    append_snapshot_to_dataset,
    ensure_dataset_dir,
    parse_team_preview_snapshot,
)

def main(
    n: int = typer.Option(100, "--n", "-n", help="Number of snapshots to generate"),
):
    """
    Generate `n` team preview snapshots using random Reg F teams.

    Each snapshot:
      - generates two random teams
      - runs simulate_battle once (non-interactive)
      - saves raw log
      - parses a TeamPreviewSnapshot
      - appends it to the JSONL dataset
    """
    ensure_paths()
    ensure_dataset_dir()

    successful = 0
    failed = 0
    skipped = 0

    print(f"Generating {n} team preview snapshots...")
    print(f"Format: {DEFAULT_FORMAT}")
    print()

    for i in range(1, n + 1):
        try:
            # Generate random teams
            p1_team = generate_random_team(DEFAULT_FORMAT)
            p2_team = generate_random_team(DEFAULT_FORMAT)

            # Run battle simulation
            log_text = simulate_battle(
                format_id=DEFAULT_FORMAT,
                p1_name="Bot1",
                p1_packed_team=p1_team,
                p2_name="Bot2",
                p2_packed_team=p2_team,
            )

            # Save raw log
            raw_log_path = save_raw_log(DEFAULT_FORMAT, log_text)

            # Parse snapshot
            try:
                snapshot = parse_team_preview_snapshot(
                    log_text,
                    format_id=DEFAULT_FORMAT,
                    raw_log_path=raw_log_path,
                )

                # Append to dataset
                append_snapshot_to_dataset(snapshot)
                successful += 1

                if i % 10 == 0:
                    print(f"  Progress: {i}/{n} (successful: {successful}, failed: {failed}, skipped: {skipped})")

            except ValueError as e:
                # Missing teamPreview info - skip this one
                skipped += 1
                print(f"  Warning [{i}]: Skipping due to parsing error: {e}")
                print(f"    Log: {raw_log_path}")

        except RuntimeError as e:
            # simulate_battle raised an error
            failed += 1
            print(f"  Error [{i}]: simulate_battle failed: {e}")

        except Exception as e:
            # Unexpected error
            failed += 1
            print(f"  Error [{i}]: Unexpected error: {e}")

    # Print summary
    print()
    print("=" * 60)
    print("DATASET GENERATION SUMMARY")
    print("=" * 60)
    print(f"Total attempts: {n}")
    print(f"Successful snapshots: {successful}")
    print(f"Failed: {failed}")
    print(f"Skipped (parse errors): {skipped}")
    print()
    from vgc_lab.team_preview import DATASET_PATH

    print(f"Dataset file: {DATASET_PATH}")
    if DATASET_PATH.exists():
        line_count = sum(1 for _ in DATASET_PATH.open("r", encoding="utf-8"))
        print(f"Total lines in dataset: {line_count}")
    print("=" * 60)


if __name__ == "__main__":
    typer.run(main)

