"""Test sanitizer with mixed force-switch scenarios."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from projects.rl_battle.online_selfplay import _sanitize_choice_for_doubles_with_reasons


def test_mixed_force_switch_converts_non_forced_to_move():
    """Test that in a mixed force-switch scenario, non-forced slots use moves, not switches."""
    request = {
        "forceSwitch": [True, False],  # Slot 0 forced, slot 1 not forced
        "active": [
            {
                "moves": [
                    {"target": "normal", "disabled": False},  # move 1 - enabled
                    {"target": "self", "disabled": False},    # move 2 - enabled
                ]
            },
            {
                "moves": [
                    {"target": "normal", "disabled": False},  # move 1 - enabled
                    {"target": "self", "disabled": False},    # move 2 - enabled
                ]
            },
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "0 fnt"},   # slot 0 fainted (forced switch)
                {"active": True, "condition": "100/100"},  # slot 1 active (not forced)
                {"active": False, "condition": "100/100"},  # bench mon 3
                {"active": False, "condition": "100/100"},  # bench mon 4
            ]
        },
    }
    # Policy wants to switch for both slots
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons("switch 3, switch 4", request)
    
    assert isinstance(choice_str, str)
    assert len(reasons) == 2
    
    # Slot 0 (forced) should get a switch
    # Slot 1 (non-forced) should get a move (converted from switch)
    parts = choice_str.split(", ")
    assert len(parts) == 2
    assert parts[0].startswith("switch"), f"Slot 0 (forced) should be switch, got {parts[0]}"
    assert parts[1].startswith("move"), f"Slot 1 (non-forced) should be move, got {parts[1]}"
    
    assert reasons[1] == "fixed_switch_to_move", f"Slot 1 should have reason 'fixed_switch_to_move', got {reasons[1]}"


def test_pure_pass_choice_converted_to_moves():
    """Test that a pure 'pass' choice is converted to moves when moves are available."""
    request = {
        "forceSwitch": [False, False],  # Both slots non-forced
        "active": [
            {
                "moves": [
                    {"target": "normal", "disabled": False},  # move 1 - enabled
                    {"target": "self", "disabled": False},    # move 2 - enabled
                ]
            },
            {
                "moves": [
                    {"target": "normal", "disabled": False},  # move 1 - enabled
                    {"target": "self", "disabled": False},    # move 2 - enabled
                ]
            },
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": True, "condition": "100/100"},
            ]
        },
    }
    # Policy sends just "pass"
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons("pass", request)
    
    assert isinstance(choice_str, str)
    assert len(reasons) == 2
    
    # Both slots should get moves (not pass)
    parts = choice_str.split(", ")
    assert len(parts) == 2
    assert parts[0].startswith("move"), f"Slot 0 should be move, got {parts[0]}"
    assert parts[1].startswith("move"), f"Slot 1 should be move, got {parts[1]}"
    
    assert reasons[0] == "fixed_pass", f"Slot 0 should have reason 'fixed_pass', got {reasons[0]}"
    assert reasons[1] == "fixed_pass", f"Slot 1 should have reason 'fixed_pass', got {reasons[1]}"


if __name__ == "__main__":
    test_mixed_force_switch_converts_non_forced_to_move()
    print("✓ test_mixed_force_switch_converts_non_forced_to_move passed")
    
    test_pure_pass_choice_converted_to_moves()
    print("✓ test_pure_pass_choice_converted_to_moves passed")
    
    print("\nAll mixed force-switch tests passed!")

