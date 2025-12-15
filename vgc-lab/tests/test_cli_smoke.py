"""Minimal smoke tests for vgc-lab CLI and imports."""

import subprocess
import sys

import vgc_lab


def test_import_vgc_lab():
    """Test that vgc_lab can be imported and key functions exist."""
    assert hasattr(vgc_lab, "generate_full_battle_dataset")
    assert hasattr(vgc_lab, "analyze_teams_vs_pool")
    assert hasattr(vgc_lab, "train_team_value_model")
    assert hasattr(vgc_lab, "train_team_matchup_model")
    assert hasattr(vgc_lab, "model_guided_team_search")
    assert hasattr(vgc_lab, "evaluate_catalog_team")
    assert hasattr(vgc_lab, "score_catalog_team_with_model")
    assert hasattr(vgc_lab, "score_matchup_with_model")


def test_cli_help_runs():
    """Test that the CLI help command runs without errors."""
    result = subprocess.run(
        [sys.executable, "-m", "vgc_lab.cli", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )
    assert "dataset" in result.stdout
    assert "train" in result.stdout
    assert "search" in result.stdout
    assert "eval" in result.stdout
    assert "analyze" in result.stdout
    assert "tools" in result.stdout
    assert "demo" in result.stdout


def test_cli_dataset_subcommand_exists():
    """Test that dataset subcommands are available."""
    result = subprocess.run(
        [sys.executable, "-m", "vgc_lab.cli", "dataset", "--help"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
        text=True,
    )
    assert "full-battles" in result.stdout
    assert "preview-outcome" in result.stdout
    assert "team-preview" in result.stdout
    assert "team-matchups" in result.stdout
    assert "teams-vs-pool" in result.stdout
