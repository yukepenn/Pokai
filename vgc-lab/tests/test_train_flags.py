"""Test for train value-model CLI flags."""

from unittest.mock import MagicMock, patch

import pytest

from vgc_lab import cli


def test_train_value_model_flags():
    """Test that --no-include-random and --no-include-catalog flags work correctly."""
    # Mock core.train_team_value_model to capture arguments
    with patch("vgc_lab.cli.core.train_team_value_model") as mock_train:
        mock_train.return_value = None
        
        # Test with --no-include-random
        with patch("vgc_lab.cli.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            cli.main([
                "train", "value-model",
                "--jsonl", "dummy.jsonl",
                "--out", "dummy.pt",
                "--epochs", "1",
                "--no-include-random",
            ])
            
            call_kwargs = mock_train.call_args.kwargs
            assert call_kwargs["include_random"] is False
            assert call_kwargs["include_catalog"] is True  # Default
        
        # Reset mock
        mock_train.reset_mock()
        
        # Test with --no-include-catalog
        with patch("vgc_lab.cli.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            cli.main([
                "train", "value-model",
                "--jsonl", "dummy.jsonl",
                "--out", "dummy.pt",
                "--epochs", "1",
                "--no-include-catalog",
            ])
            
            call_kwargs = mock_train.call_args.kwargs
            assert call_kwargs["include_random"] is True  # Default
            assert call_kwargs["include_catalog"] is False
        
        # Reset mock
        mock_train.reset_mock()
        
        # Test with both flags (both should be False)
        with patch("vgc_lab.cli.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            cli.main([
                "train", "value-model",
                "--jsonl", "dummy.jsonl",
                "--out", "dummy.pt",
                "--epochs", "1",
                "--no-include-random",
                "--no-include-catalog",
            ])
            
            call_kwargs = mock_train.call_args.kwargs
            assert call_kwargs["include_random"] is False
            assert call_kwargs["include_catalog"] is False
