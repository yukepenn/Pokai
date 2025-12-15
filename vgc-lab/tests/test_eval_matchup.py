"""Test for eval matchup CLI command."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vgc_lab import cli


def test_eval_matchup_cli_wiring():
    """Test that eval matchup CLI correctly wires to core.evaluate_matchup."""
    # Create temporary team import files
    team_a_content = "Pikachu @ Light Ball\nAbility: Static\nIVs: 0 Atk\n- Thunderbolt\n- Thunder\n"
    team_b_content = "Charizard @ Charcoal\nAbility: Blaze\nIVs: 0 Atk\n- Flamethrower\n- Fire Blast\n"

    with tempfile.TemporaryDirectory() as tmpdir:
        team_a_path = Path(tmpdir) / "team_a.txt"
        team_b_path = Path(tmpdir) / "team_b.txt"
        
        team_a_path.write_text(team_a_content, encoding="utf-8")
        team_b_path.write_text(team_b_content, encoding="utf-8")

        # Mock core.evaluate_matchup to capture arguments
        with patch("vgc_lab.cli.core.evaluate_matchup") as mock_eval:
            mock_eval.return_value = None  # Function returns None
            
            # Invoke CLI
            cli.main([
                "eval", "matchup",
                str(team_a_path),
                str(team_b_path),
                "--n", "3",
            ])
            
            # Verify it was called once with correct arguments
            assert mock_eval.call_count == 1
            call_kwargs = mock_eval.call_args.kwargs
            assert call_kwargs["team_a_path"] == team_a_path
            assert call_kwargs["team_b_path"] == team_b_path
            assert call_kwargs["n"] == 3
            assert call_kwargs["format_id"] is None  # Default
