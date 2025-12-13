"""Regression tests for specific [Invalid choice] errors encountered in strict-mode self-play."""

import sys
from pathlib import Path
from typing import Any, Dict

import pytest

# Add the repository root to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

from projects.rl_battle.online_selfplay import (
    _emergency_valid_choice,
    _sanitize_choice_for_doubles_with_reasons,
)
from src.vgc_lab.core import SanitizeReason


def _create_mock_request(
    active_moves: list[list[Dict[str, Any]]],
    force_switch: list[bool] | None = None,
    team_conditions: list[str] | None = None,
) -> Dict[str, Any]:
    """Helper to create a minimal request object for testing."""
    active_data = []
    for moves_list in active_moves:
        active_data.append({"moves": moves_list})

    team_data = []
    if team_conditions:
        for i, cond in enumerate(team_conditions):
            is_active = i < len(active_moves)
            # Convert condition to string if it's a dict
            if isinstance(cond, dict):
                cond_str = cond.get("condition", "")
            else:
                cond_str = str(cond)
            team_data.append({"condition": cond_str, "active": is_active})
    else:
        # Default to some healthy bench if not specified
        team_data = [
            {"condition": "100/100", "active": False},
            {"condition": "100/100", "active": False},
            {"condition": "100/100", "active": False},
            {"condition": "100/100", "active": False},
        ]
        # Add active mons
        for i in range(len(active_moves)):
            if i < len(team_data):
                team_data[i]["active"] = True
            else:
                team_data.append({"condition": "100/100", "active": True})

    return {
        "active": active_data,
        "forceSwitch": force_switch if force_switch is not None else [False] * len(active_moves),
        "side": {"pokemon": team_data},
    }


def test_mixed_force_switch_no_duplicate_switches():
    """
    Regression: forceSwitch=[false,true] should not result in duplicate switch targets.
    
    Error: "Can't switch: You sent more switches than Pokémon that need to switch"
    """
    request = _create_mock_request(
        active_moves=[
            [{"id": "tackle", "disabled": False, "target": "normal"}],  # Slot 0: active, not forced
            [],  # Slot 1: no moves (fainted?), forced
        ],
        force_switch=[False, True],  # Slot 0 NOT forced, Slot 1 IS forced
        team_conditions=[
            {"condition": "100/100", "active": True},   # Active mon for slot 0
            {"condition": "0 fnt", "active": True},     # Fainted active for slot 1
            {"condition": "100/100", "active": False},  # Bench 3
            {"condition": "100/100", "active": False},  # Bench 4
        ],
    )
    
    # The sanitizer should handle this correctly
    choice = "pass, pass"  # Policy outputs pass for both
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons(choice, request)
    
    # Slot 0 (not forced) should become a move
    # Slot 1 (forced) should become a switch
    assert "switch" in choice_str, f"Expected a switch in choice, got {choice_str!r}"
    assert choice_str.count("switch") == 1, f"Expected exactly one switch (only forced slot), got {choice_str!r}"
    assert not choice_str.startswith("switch"), f"Slot 0 should be a move, not switch. Got {choice_str!r}"
    
    # Emergency function should also handle this correctly
    emergency_choice = _emergency_valid_choice(request, "force-switch")
    assert "switch" in emergency_choice, f"Expected a switch in emergency choice, got {emergency_choice!r}"
    assert emergency_choice.count("switch") == 1, f"Expected exactly one switch in emergency, got {emergency_choice!r}"
    
    # Parse and verify no duplicate targets
    parts = [p.strip() for p in emergency_choice.split(",")]
    switch_targets = []
    for part in parts:
        if part.startswith("switch"):
            target = part.split()[1] if len(part.split()) > 1 else None
            if target:
                assert target not in switch_targets, f"Duplicate switch target {target} in {emergency_choice!r}"
                switch_targets.append(target)


def test_ally_targeting_move_needs_negative_target():
    """
    Regression: Ally-targeting moves like Coaching need negative target (-1), not positive.
    
    Error: "Invalid target for Coaching"
    """
    request = _create_mock_request(
        active_moves=[
            [{"id": "coaching", "disabled": False, "target": "ally"}],  # Coaching targets ally
        ],
    )
    
    choice = "move 1"  # Missing target
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons(choice, request)
    
    # Should have -1 target for ally moves
    assert "-1" in choice_str, f"Ally-targeting move should get -1 target, got {choice_str!r}"
    assert "move 1 -1" in choice_str or choice_str == "move 1 -1", f"Expected 'move 1 -1', got {choice_str!r}"


def test_move_request_with_enabled_moves_never_returns_pass():
    """
    Regression: Move request with active Pokémon that have enabled moves should NEVER return "pass".
    
    Error: "Can't pass: Your [Pokemon] must make a move (or switch)"
    """
    request = _create_mock_request(
        active_moves=[
            [
                {"id": "heavyslam", "disabled": False, "target": "normal"},
                {"id": "bodypress", "disabled": False, "target": "normal"},
                {"id": "helpinghand", "disabled": False, "target": "adjacentAlly"},
                {"id": "shedtail", "disabled": False, "target": "self"},
            ],
            [
                {"id": "shadowclaw", "disabled": False, "target": "normal"},
                {"id": "shadowsneak", "disabled": False, "target": "normal"},
                {"id": "swordsdance", "disabled": False, "target": "self"},
                {"id": "playrough", "disabled": False, "target": "normal"},
            ],
        ],
        force_switch=[False, False],
        team_conditions=[
            {"condition": "100/156", "active": True},
            {"condition": "16/141", "active": True},
            {"condition": "201/201", "active": False},
            {"condition": "131/131", "active": False},
        ],
    )
    
    # Even if policy outputs "pass", sanitizer should convert to valid moves
    choice = "pass"
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons(choice, request)
    
    # Should NEVER be "pass" since there are enabled moves
    assert choice_str != "pass", f"Sanitizer returned 'pass' when moves are available: {choice_str!r}"
    assert "pass" not in choice_str.lower(), f"Sanitizer returned choice containing 'pass': {choice_str!r}"
    assert "move" in choice_str.lower(), f"Should have moves in choice: {choice_str!r}"
    
    # All reasons should be valid
    assert all(r in ("ok", "fixed_pass", "fixed_disabled_move", "fixed_switch_to_move", "fallback_switch", "fallback_pass") for r in reasons)


def test_self_targeting_move_no_target():
    """
    Regression: Self-targeting moves like Jungle Healing should NOT have a numeric target.
    
    Error: "You can't choose a target for Jungle Healing"
    """
    request = _create_mock_request(
        active_moves=[
            [{"id": "junglehealing", "disabled": False, "target": "self"}],  # Self-targeting
        ],
    )
    
    choice = "move 1 1"  # Has target but shouldn't
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons(choice, request)
    
    # Should NOT have numeric target
    assert "move 1 1" not in choice_str, f"Self-targeting move should not have numeric target, got {choice_str!r}"
    assert choice_str == "move 1" or choice_str.startswith("move 1 "), f"Expected 'move 1' (possibly with flags), got {choice_str!r}"
    # Should not contain a standalone "1" after "move 1" (unless it's part of a flag or something else)
    parts = choice_str.split()
    # move 1 should be followed by flags or nothing, not a numeric target
    if len(parts) > 2:
        assert parts[2] not in ("1", "-1"), f"Self-targeting move should not have numeric target, got {choice_str!r}"


def test_force_switch_empty_active_no_moves_for_non_forced():
    """
    Regression: force-switch request with empty active array should not generate moves.
    
    Error: [Invalid choice] when active_len=0 and forceSwitch=[False, True]
    The sanitizer tries to generate "move 1" for slot 0, but there are no active mons.
    """
    request = {
        "forceSwitch": [False, True],  # Slot 0 NOT forced, Slot 1 IS forced
        "active": [],  # Empty - all active mons have fainted
        "side": {
            "pokemon": [
                {"condition": "0 fnt", "active": True},  # Fainted active for slot 0
                {"condition": "0 fnt", "active": True},  # Fainted active for slot 1
                {"condition": "100/100", "active": False},  # Bench 3
                {"condition": "100/100", "active": False},  # Bench 4
            ]
        },
    }
    
    # Policy tries to switch slot 0 and pass slot 1
    choice = "switch 4, pass"
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons(choice, request)
    
    # When active is empty in force-switch context, Showdown requires ALL slots to switch
    # (not just the explicitly forced ones), because there are no active mons to use moves
    assert "switch" in choice_str, f"Expected switches in choice, got {choice_str!r}"
    # Both slots should be switches since active is empty
    assert choice_str.count("switch") == 2, f"Expected two switches (all slots when active is empty), got {choice_str!r}"
    # Should not contain "move" since there are no active mons
    assert "move" not in choice_str.lower(), f"Should not generate moves when active is empty, got {choice_str!r}"
    # Should not contain "pass" - all slots should switch
    assert "pass" not in choice_str.lower(), f"Should not contain 'pass' when active is empty, got {choice_str!r}"

