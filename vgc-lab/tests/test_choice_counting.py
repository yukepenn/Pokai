"""Tests for choice counting and encoding logic."""

import sys
from pathlib import Path

# Add the repository root to sys.path so we can import from projects.rl_battle.
sys.path.insert(0, str(Path(__file__).parent.parent))

from projects.rl_battle.online_selfplay import _compute_required_slots, _sanitize_choice_for_doubles


def test_singles_requires_one_choice():
    """Test that singles battles require exactly 1 choice."""
    request = {
        "active": [
            {
                "moves": [
                    {"id": "tackle", "disabled": False, "target": "normal"},
                    {"id": "growl", "disabled": False, "target": "allAdjacentFoes"},
                ]
            }
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": False, "condition": "100/100"},
            ]
        },
    }
    required = _compute_required_slots(request)
    assert required == 1, f"Expected 1 choice for singles, got {required}"


def test_doubles_two_active_requires_two_choices():
    """Test that doubles with 2 unfainted active Pokémon requires 2 choices."""
    request = {
        "active": [
            {
                "moves": [
                    {"id": "tackle", "disabled": False, "target": "normal"},
                ]
            },
            {
                "moves": [
                    {"id": "scratch", "disabled": False, "target": "normal"},
                ]
            },
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": True, "condition": "100/100"},
                {"active": False, "condition": "100/100"},
            ]
        },
    }
    required = _compute_required_slots(request)
    assert required == 2, f"Expected 2 choices for doubles with 2 active, got {required}"


def test_doubles_one_fainted_requires_one_choice():
    """Test that doubles with 1 unfainted and 1 fainted requires only 1 choice."""
    request = {
        "active": [
            {
                "moves": [
                    {"id": "tackle", "disabled": False, "target": "normal"},
                ]
            },
            {
                # No moves = fainted Pokémon
                "moves": [],
            },
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": True, "condition": "0 fnt"},
                {"active": False, "condition": "100/100"},
            ]
        },
    }
    required = _compute_required_slots(request)
    assert required == 1, f"Expected 1 choice when one Pokémon is fainted, got {required}"


def test_force_switch_uses_force_switch_length():
    """Test that force-switch requests use the forceSwitch array length."""
    request = {
        "active": [
            {
                "moves": [
                    {"id": "tackle", "disabled": False, "target": "normal"},
                ]
            },
            {
                "moves": [
                    {"id": "scratch", "disabled": False, "target": "normal"},
                ]
            },
        ],
        "forceSwitch": [False, True],  # Slot 1 forced to switch
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": True, "condition": "100/100"},
            ]
        },
    }
    required = _compute_required_slots(request)
    assert required == 2, f"Expected 2 choices for 2 force-switch flags, got {required}"


def test_sanitizer_trims_excess_choices():
    """Test that sanitizer trims choices when more are provided than required."""
    # Singles battle but policy returns 2 choices
    request = {
        "active": [
            {
                "moves": [
                    {"id": "tackle", "disabled": False, "target": "normal"},
                ]
            },
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
            ]
        },
    }
    # Policy incorrectly returns 2 choices for a singles battle
    raw_choice = "move 1, move 1"
    sanitized = _sanitize_choice_for_doubles(raw_choice, request)
    # Should be trimmed to 1 choice
    parts = [p.strip() for p in sanitized.split(",")]
    assert len(parts) == 1, f"Expected 1 choice part, got {len(parts)}: {sanitized}"
    assert parts[0].startswith("move"), f"Expected move action, got {parts[0]}"


def test_sanitizer_prevents_doubles_excess_in_singles():
    """Test that we never produce 'move 1, move 1 1' style errors in singles."""
    request = {
        "active": [
            {
                "moves": [
                    {"id": "tackle", "disabled": False, "target": "normal"},
                ]
            },
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
            ]
        },
    }
    # Simulate the bug pattern: policy returns doubles-style choice for singles
    raw_choice = "move 1, move 1 1"
    sanitized = _sanitize_choice_for_doubles(raw_choice, request)
    # Should be exactly 1 choice, no comma
    assert "," not in sanitized, f"Singles choice should not contain comma: {sanitized}"
    assert sanitized.startswith("move"), f"Should start with 'move': {sanitized}"


if __name__ == "__main__":
    test_singles_requires_one_choice()
    print("✓ test_singles_requires_one_choice passed")

    test_doubles_two_active_requires_two_choices()
    print("✓ test_doubles_two_active_requires_two_choices passed")

    test_doubles_one_fainted_requires_one_choice()
    print("✓ test_doubles_one_fainted_requires_one_choice passed")

    test_force_switch_uses_force_switch_length()
    print("✓ test_force_switch_uses_force_switch_length passed")

    test_sanitizer_trims_excess_choices()
    print("✓ test_sanitizer_trims_excess_choices passed")

    test_sanitizer_prevents_doubles_excess_in_singles()
    print("✓ test_sanitizer_prevents_doubles_excess_in_singles passed")

    print("\nAll tests passed!")

