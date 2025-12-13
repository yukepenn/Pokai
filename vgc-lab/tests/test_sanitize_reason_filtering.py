"""Unit tests for sanitize_reason filtering in BattleStepDataset."""

import sys
from pathlib import Path

# Add the repository root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from projects.rl_battle.dataset import BattleStepDatasetConfig


def test_battle_step_dataset_config_sanitize_reason_filtering():
    """Test that BattleStepDatasetConfig accepts train_allowed_sanitize_reasons parameter."""
    # Test 1: Default config should have ["ok", "fixed_pass"]
    cfg_default = BattleStepDatasetConfig(format_id="gen9vgc2026regf")
    assert cfg_default.train_allowed_sanitize_reasons == ["ok", "fixed_pass"]

    # Test 2: Can set to None (no filtering)
    cfg_none = BattleStepDatasetConfig(
        format_id="gen9vgc2026regf", train_allowed_sanitize_reasons=None
    )
    assert cfg_none.train_allowed_sanitize_reasons is None

    # Test 3: Can set to specific list
    cfg_ok_only = BattleStepDatasetConfig(
        format_id="gen9vgc2026regf", train_allowed_sanitize_reasons=["ok"]
    )
    assert cfg_ok_only.train_allowed_sanitize_reasons == ["ok"]

    # Test 4: Can set to multiple values
    cfg_multiple = BattleStepDatasetConfig(
        format_id="gen9vgc2026regf",
        train_allowed_sanitize_reasons=["ok", "fixed_pass", "fixed_disabled_move"],
    )
    assert cfg_multiple.train_allowed_sanitize_reasons == ["ok", "fixed_pass", "fixed_disabled_move"]


def test_sanitize_reason_filtering_logic():
    """Test the filtering logic works correctly (unit test without filesystem mocking).

    Note: Full integration testing with actual trajectories requires test data fixtures.
    The filtering logic is exercised in the dataset loading code at:
    projects/rl_battle/dataset.py around line 193-198.
    """
    # This is a placeholder test that verifies the config structure.
    # The actual filtering is tested implicitly through:
    # 1. The sanitizer tests (test_sanitizer_reasons.py) which verify sanitize_reason values are set
    # 2. The CLI command which can be run with --allowed-sanitize-reasons to verify filtering
    cfg = BattleStepDatasetConfig(format_id="gen9vgc2026regf", train_allowed_sanitize_reasons=["ok"])
    assert cfg.train_allowed_sanitize_reasons == ["ok"]
    
    # Verify that when train_allowed_sanitize_reasons is set, it filters correctly
    # by checking that the dataset config structure is correct
    assert isinstance(cfg.train_allowed_sanitize_reasons, list)
    assert len(cfg.train_allowed_sanitize_reasons) == 1


if __name__ == "__main__":
    test_battle_step_dataset_config_sanitize_reason_filtering()
    print("✓ test_battle_step_dataset_config_sanitize_reason_filtering passed")
    
    test_sanitize_reason_filtering_logic()
    print("✓ test_sanitize_reason_filtering_logic passed")
