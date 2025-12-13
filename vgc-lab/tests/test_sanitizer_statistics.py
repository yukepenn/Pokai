"""Smoke test for sanitizer statistics CLI."""

import json
import sys
import tempfile
from pathlib import Path

# Add the repository root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from vgc_lab.core import BattleStep, BattleTrajectory
from vgc_lab.datasets import summarize_sanitize_reasons


def test_summarize_sanitize_reasons():
    """Test that summarize_sanitize_reasons correctly counts sanitize_reason values."""
    from datetime import datetime, timezone

    # Create a temporary directory structure
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        trajectories_dir = tmp_path / "datasets" / "trajectories"
        trajectories_dir.mkdir(parents=True)

        jsonl_path = trajectories_dir / "trajectories.jsonl"

        # Create a minimal trajectory with various sanitize_reason values
        traj = BattleTrajectory(
            battle_id="test-battle-1",
            format_id="gen9vgc2026regf",
            p1_name="p1",
            p2_name="p2",
            winner_side="p1",
            winner_name="p1",
            turns=5,
            raw_log_path=Path("test.log"),
            created_at=datetime.now(timezone.utc),
            reward_p1=1.0,
            reward_p2=-1.0,
            steps_p1=[
                BattleStep(
                    side="p1",
                    step_index=0,
                    request_type="move",
                    request={},
                    choice="move 1",
                    sanitize_reason="ok",
                ),
                BattleStep(
                    side="p1",
                    step_index=1,
                    request_type="move",
                    request={},
                    choice="move 2",
                    sanitize_reason="fixed_pass",
                ),
                BattleStep(
                    side="p1",
                    step_index=2,
                    request_type="move",
                    request={},
                    choice="move 3",
                    sanitize_reason="fixed_disabled_move",
                ),
            ],
            steps_p2=[
                BattleStep(
                    side="p2",
                    step_index=0,
                    request_type="move",
                    request={},
                    choice="move 1",
                    sanitize_reason="ok",
                ),
                BattleStep(
                    side="p2",
                    step_index=1,
                    request_type="move",
                    request={},
                    choice="move 1",
                    sanitize_reason=None,  # None should count as "none"
                ),
            ],
        )

        # Write trajectory to JSONL
        with jsonl_path.open("w", encoding="utf-8") as f:
            json.dump(traj.model_dump(mode="json"), f, ensure_ascii=False)
            f.write("\n")

        # Test with directory path
        counts = summarize_sanitize_reasons(tmp_path)
        assert counts["ok"] == 2
        assert counts["fixed_pass"] == 1
        assert counts["fixed_disabled_move"] == 1
        assert counts["none"] == 1

        # Test with direct file path
        counts2 = summarize_sanitize_reasons(jsonl_path)
        assert counts2 == counts


if __name__ == "__main__":
    test_summarize_sanitize_reasons()
    print("✓ test_summarize_sanitize_reasons passed")

