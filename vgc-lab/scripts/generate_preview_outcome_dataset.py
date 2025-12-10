#!/usr/bin/env python3
"""Generate a preview+outcome dataset from random self-play battles."""

import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List
import time

import typer

# Add src to PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from vgc_lab.config import DEFAULT_FORMAT, ensure_paths
from vgc_lab.battle_logger import save_raw_log
from vgc_lab.showdown_cli import run_random_selfplay_json
from vgc_lab.lineup import extract_lineups_from_log
from vgc_lab.team_preview import (
    PreviewOutcomeRecord,
    append_preview_outcome_record,
    ensure_preview_outcome_dataset_dir,
)


def main(
    n: int = typer.Option(10, "--n", "-n", help="Number of preview+outcome samples to generate"),
    sleep_ms: int = typer.Option(50, "--sleep-ms", help="Sleep between battles (to ease OneDrive)"),
):
    """
    For each sample:
      - Run a random self-play battle via the Node script (random teams, random AIs).
      - Save the raw battle log into data/battles_raw/.
      - Create a PreviewOutcomeRecord with:
          * format_id, p1_name, p2_name
          * p1_team_public / p2_team_public
          * winner_side / winner_name
          * raw_log_path
          * created_at
      - Append it as one JSON line to data/datasets/preview_outcome/preview_outcome.jsonl
    """
    ensure_paths()
    ensure_preview_outcome_dataset_dir()

    for i in range(1, n + 1):
        try:
            data = run_random_selfplay_json(format_id=DEFAULT_FORMAT)
        except RuntimeError as e:
            print(f"[{i}/{n}] WARNING: run_random_selfplay_json failed: {e}")
            continue

        log_text = data.get("log", "")
        format_id = data.get("format_id", DEFAULT_FORMAT)
        p1_name = data.get("p1_name", "Bot1")
        p2_name = data.get("p2_name", "Bot2")
        winner_side = data.get("winner_side", "unknown")
        winner_name = data.get("winner_name")

        p1_team_public = data.get("p1_team_public", [])
        p2_team_public = data.get("p2_team_public", [])

        # Extract lineup information from log
        p1_chosen_indices: List[int] = []
        p2_chosen_indices: List[int] = []
        p1_lead_indices: List[int] = []
        p2_lead_indices: List[int] = []

        if log_text and p1_team_public and p2_team_public:
            try:
                (
                    p1_chosen_indices,
                    p2_chosen_indices,
                    p1_lead_indices,
                    p2_lead_indices,
                ) = extract_lineups_from_log(log_text, p1_team_public, p2_team_public)
            except Exception as e:
                print(f"[WARN] Failed to extract lineups: {e}")

        # Save raw log
        raw_log_path = save_raw_log(format_id, log_text)

        record = PreviewOutcomeRecord(
            format_id=format_id,
            p1_name=p1_name,
            p2_name=p2_name,
            p1_team_public=p1_team_public,
            p2_team_public=p2_team_public,
            p1_chosen_indices=p1_chosen_indices,
            p2_chosen_indices=p2_chosen_indices,
            p1_lead_indices=p1_lead_indices,
            p2_lead_indices=p2_lead_indices,
            winner_side=winner_side,
            winner_name=winner_name,
            raw_log_path=raw_log_path,
            created_at=datetime.now(timezone.utc),
            meta={},
        )
        append_preview_outcome_record(record)
        print(f"[{i}/{n}] winner_side={winner_side}, raw_log={raw_log_path.name}")

        if sleep_ms > 0:
            time.sleep(sleep_ms / 1000.0)


if __name__ == "__main__":
    typer.run(main)

