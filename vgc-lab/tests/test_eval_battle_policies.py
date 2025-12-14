"""Tests for battle policy evaluation harness."""

from __future__ import annotations

from unittest.mock import patch

import pytest

from projects.rl_battle.eval_battle_policies import BattleEvalConfig, run_battle_eval


def test_run_battle_eval_aggregates_runs(monkeypatch):
    """Test that run_battle_eval correctly aggregates results across multiple runs."""
    # Create a fake summary to return from mocked run_online_selfplay
    fake_summary = {
        "episodes": 10,
        "errors": 1,
        "p1_wins": 6,
        "p2_wins": 3,
        "draws": 1,
    }

    # Mock run_online_selfplay to return our fake summary
    def _mock_run_online_selfplay(cfg):
        return fake_summary

    monkeypatch.setattr(
        "projects.rl_battle.eval_battle_policies.run_online_selfplay",
        _mock_run_online_selfplay,
    )

    # Build config
    cfg = BattleEvalConfig(
        num_runs=3,
        episodes_per_run=10,
        format_id="gen9vgc2026regf",
    )

    # Run evaluation
    result = run_battle_eval(cfg)

    # Assertions
    assert result["total_episodes"] == 30  # 3 runs * 10 episodes
    assert result["total_errors"] == 3  # 3 runs * 1 error
    assert result["total_p1_wins"] == 18  # 3 runs * 6 wins
    assert result["total_p2_wins"] == 9  # 3 runs * 3 wins
    assert result["total_draws"] == 3  # 3 runs * 1 draw

    # Check win rate: 18 / (18 + 9 + 3) = 18 / 30 = 0.6
    expected_win_rate = 18 / (18 + 9 + 3)
    assert abs(result["p1_win_rate"] - expected_win_rate) < 1e-6

    # Check runs list
    assert len(result["runs"]) == 3
    for run_summary in result["runs"]:
        assert run_summary == fake_summary


def test_run_battle_eval_handles_zero_games(monkeypatch):
    """Test that run_battle_eval handles zero games gracefully."""
    # Create a fake summary with all zero counts
    fake_summary = {
        "episodes": 10,
        "errors": 10,  # All episodes errored
        "p1_wins": 0,
        "p2_wins": 0,
        "draws": 0,
    }

    def _mock_run_online_selfplay(cfg):
        return fake_summary

    monkeypatch.setattr(
        "projects.rl_battle.eval_battle_policies.run_online_selfplay",
        _mock_run_online_selfplay,
    )

    cfg = BattleEvalConfig(
        num_runs=2,
        episodes_per_run=10,
    )

    result = run_battle_eval(cfg)

    # Assertions
    assert result["total_episodes"] == 20
    assert result["total_errors"] == 20
    assert result["total_p1_wins"] == 0
    assert result["total_p2_wins"] == 0
    assert result["total_draws"] == 0
    assert result["p1_win_rate"] == 0.0  # Should be 0.0 when no games completed
