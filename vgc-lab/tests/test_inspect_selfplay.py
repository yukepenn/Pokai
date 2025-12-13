"""
Unit tests for inspect_selfplay analysis tool.
"""

import json
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from vgc_lab.core import Paths


def test_summarize_selfplay_with_sample_data():
    """Test summarize_selfplay with a small in-memory dataset."""
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from projects.rl_battle.inspect_selfplay import summarize_selfplay
    
    # Create temporary JSONL file with sample battle records
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        full_battles_path = tmp_path / "full_battles.jsonl"
        
        # Write 3 sample battle records
        battles = [
            {
                "battle_id": "test_001",
                "format_id": "gen9vgc2026regf",
                "p1_name": "Bot1",
                "p2_name": "Bot2",
                "winner_side": "p1",
                "winner_name": "Bot1",
                "turns": 10,
                "raw_log_path": str(tmp_path / "test_001.log"),
                "created_at": "2024-01-01T00:00:00+00:00",
                "meta": {},
            },
            {
                "battle_id": "test_002",
                "format_id": "gen9vgc2026regf",
                "p1_name": "Bot1",
                "p2_name": "Bot2",
                "winner_side": "p2",
                "winner_name": "Bot2",
                "turns": 8,
                "raw_log_path": str(tmp_path / "test_002.log"),
                "created_at": "2024-01-01T00:01:00+00:00",
                "meta": {},
            },
            {
                "battle_id": "test_003",
                "format_id": "gen9vgc2026regf",
                "p1_name": "Bot1",
                "p2_name": "Bot2",
                "winner_side": "tie",
                "winner_name": None,
                "turns": 12,
                "raw_log_path": str(tmp_path / "test_003.log"),
                "created_at": "2024-01-01T00:02:00+00:00",
                "meta": {},
            },
        ]
        
        with full_battles_path.open("w", encoding="utf-8") as f:
            for battle in battles:
                f.write(json.dumps(battle, ensure_ascii=False) + "\n")
        
        # Create a mock Paths object that points to our temporary file
        # We'll patch iter_full_battles to use the temp file directly
        from unittest.mock import MagicMock
        
        mock_paths = MagicMock(spec=Paths)
        mock_paths.full_battles_jsonl = full_battles_path
        
        with patch("projects.rl_battle.inspect_selfplay.get_paths", return_value=mock_paths):
            stats = summarize_selfplay(max_battles=None, paths=mock_paths)
        
        # Verify stats
        assert stats["total_battles"] == 3
        assert stats["winner_counts"]["p1"] == 1
        assert stats["winner_counts"]["p2"] == 1
        assert stats["winner_counts"]["tie"] == 1
        assert stats["winner_counts"]["unknown"] == 0
        
        # Verify turn statistics (mean of 10, 8, 12 = 10.0, median = 10.0)
        assert stats["avg_turns"] == pytest.approx(10.0)
        assert stats["median_turns"] == pytest.approx(10.0)


def test_summarize_selfplay_empty_dataset():
    """Test summarize_selfplay with no battles."""
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from projects.rl_battle.inspect_selfplay import summarize_selfplay
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        
        # Create a mock Paths object with non-existent JSONL path
        from unittest.mock import MagicMock
        
        mock_paths = MagicMock(spec=Paths)
        mock_paths.full_battles_jsonl = tmp_path / "nonexistent.jsonl"
        
        with patch("projects.rl_battle.inspect_selfplay.get_paths", return_value=mock_paths):
            stats = summarize_selfplay(max_battles=None, paths=mock_paths)
        
        # Should return zeros and None
        assert stats["total_battles"] == 0
        assert stats["winner_counts"] == {"p1": 0, "p2": 0, "tie": 0, "unknown": 0}
        assert stats["avg_turns"] is None
        assert stats["median_turns"] is None


def test_summarize_selfplay_max_battles_limit():
    """Test that max_battles limits the number of battles loaded."""
    import sys
    from pathlib import Path
    
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    
    from projects.rl_battle.inspect_selfplay import summarize_selfplay
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        full_battles_path = tmp_path / "full_battles.jsonl"
        
        # Write 5 battles
        with full_battles_path.open("w", encoding="utf-8") as f:
            for i in range(5):
                battle = {
                    "battle_id": f"test_{i:03d}",
                    "format_id": "gen9vgc2026regf",
                    "p1_name": "Bot1",
                    "p2_name": "Bot2",
                    "winner_side": "p1" if i % 2 == 0 else "p2",
                    "winner_name": "Bot1" if i % 2 == 0 else "Bot2",
                    "turns": 10 + i,
                    "raw_log_path": str(tmp_path / f"test_{i:03d}.log"),
                    "created_at": "2024-01-01T00:00:00+00:00",
                    "meta": {},
                }
                f.write(json.dumps(battle, ensure_ascii=False) + "\n")
        
        # Create a mock Paths object that points to our temporary file
        from unittest.mock import MagicMock
        
        mock_paths = MagicMock(spec=Paths)
        mock_paths.full_battles_jsonl = full_battles_path
        
        with patch("projects.rl_battle.inspect_selfplay.get_paths", return_value=mock_paths):
            stats = summarize_selfplay(max_battles=3, paths=mock_paths)
        
        # Should only load 3 battles
        assert stats["total_battles"] == 3

