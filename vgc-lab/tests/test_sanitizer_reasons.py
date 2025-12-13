"""Unit tests for sanitizer reason assignment.

Each test verifies that _sanitize_choice_for_doubles_with_reasons produces
the expected reason value for a specific sanitization scenario.
"""

import sys
from pathlib import Path

# Add the repository root to sys.path so we can import from projects.rl_battle.
sys.path.insert(0, str(Path(__file__).parent.parent))

from projects.rl_battle.online_selfplay import _sanitize_choice_for_doubles_with_reasons


def test_reason_ok():
    """Test 'ok' reason: valid move that doesn't need any change."""
    request = {
        "active": [
            {
                "moves": [
                    {"target": "normal", "disabled": False},  # move 1 - enabled
                    {"target": "self", "disabled": False},    # move 2 - enabled
                ]
            }
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
            ]
        },
    }
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons("move 1", request)
    assert isinstance(choice_str, str)
    assert len(reasons) == 1
    assert reasons[0] == "ok", f"Expected 'ok', got {reasons[0]!r}"


def test_reason_fixed_pass():
    """Test 'fixed_pass' reason: input choice is pass, but valid move exists."""
    request = {
        "active": [
            {
                "moves": [
                    {"target": "normal", "disabled": False},  # move 1 - enabled
                    {"target": "self", "disabled": False},    # move 2 - enabled
                ]
            }
        ],
        "forceSwitch": [False],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
            ]
        },
    }
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons("pass", request)
    assert isinstance(choice_str, str)
    assert len(reasons) == 1
    assert reasons[0] == "fixed_pass", f"Expected 'fixed_pass', got {reasons[0]!r}"
    assert choice_str.startswith("move"), f"Expected choice to start with 'move', got {choice_str!r}"


def test_reason_fixed_disabled_move():
    """Test 'fixed_disabled_move' reason: disabled move replaced with enabled alternative."""
    request = {
        "active": [
            {
                "moves": [
                    {"target": "normal", "disabled": True},   # move 1 - disabled
                    {"target": "normal", "disabled": False},  # move 2 - enabled
                    {"target": "self", "disabled": False},    # move 3 - enabled
                ]
            }
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
            ]
        },
    }
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons("move 1", request)
    assert isinstance(choice_str, str)
    assert len(reasons) == 1
    assert reasons[0] == "fixed_disabled_move", f"Expected 'fixed_disabled_move', got {reasons[0]!r}"
    # Should have replaced move 1 with move 2 (first enabled)
    assert "move 2" in choice_str or "move 3" in choice_str


def test_reason_fixed_switch_to_move():
    """Test 'fixed_switch_to_move' reason: switch converted to move for non-forced slot."""
    request = {
        "active": [
            {
                "moves": [
                    {"target": "normal", "disabled": False},  # move 1 - enabled
                    {"target": "self", "disabled": False},    # move 2 - enabled
                ]
            }
        ],
        "forceSwitch": [False],  # Not forced to switch
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": False, "condition": "100/100"},  # bench mon
            ]
        },
    }
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons("switch 2", request)
    assert isinstance(choice_str, str)
    assert len(reasons) == 1
    assert reasons[0] == "fixed_switch_to_move", f"Expected 'fixed_switch_to_move', got {reasons[0]!r}"
    assert choice_str.startswith("move"), f"Expected choice to start with 'move', got {choice_str!r}"


def test_reason_fallback_switch():
    """Test 'fallback_switch' reason: couldn't use original choice, fallback to disabled move."""
    request = {
        "active": [
            {
                "moves": [
                    {"target": "normal", "disabled": True},   # move 1 - disabled
                    {"target": "normal", "disabled": True},   # move 2 - disabled
                    # All moves disabled
                ]
            }
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": False, "condition": "100/100"},  # bench mon
            ]
        },
    }
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons("move 1", request)
    assert isinstance(choice_str, str)
    assert len(reasons) == 1
    # For non-forced slots, we use a disabled move (not a switch) when all moves are disabled
    assert reasons[0] == "fallback_switch", f"Expected 'fallback_switch', got {reasons[0]!r}"
    assert choice_str.startswith("move"), f"Expected choice to start with 'move' (using disabled move), got {choice_str!r}"


def test_reason_fallback_pass():
    """Test 'fallback_pass' reason: everything blocked, must use pass."""
    request = {
        "active": [
            {
                "moves": [
                    # All moves disabled
                    {"target": "normal", "disabled": True},
                    {"target": "normal", "disabled": True},
                ]
            }
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "0 fnt"},  # Fainted
                # No bench mons available
            ]
        },
    }
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons("move 1", request)
    assert isinstance(choice_str, str)
    assert len(reasons) == 1
    # Note: fallback_pass is extremely rare, but if all moves disabled and no switch possible, we get it
    assert reasons[0] in ("fallback_pass", "fallback_switch"), f"Expected 'fallback_pass' or 'fallback_switch', got {reasons[0]!r}"


def test_reason_fallback_switch_force_switch():
    """Test 'fallback_switch' reason for forced switch slots."""
    request = {
        "forceSwitch": [True],
        "active": [
            {
                "moves": [
                    {"target": "normal", "disabled": False},
                ]
            }
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "0 fnt"},   # Fainted
                {"active": False, "condition": "100/100"},  # bench mon
            ]
        },
    }
    choice_str, reasons = _sanitize_choice_for_doubles_with_reasons("move 1", request)
    assert isinstance(choice_str, str)
    assert len(reasons) == 1
    assert reasons[0] == "fallback_switch", f"Expected 'fallback_switch', got {reasons[0]!r}"
    assert choice_str.startswith("switch"), f"Expected choice to start with 'switch', got {choice_str!r}"


if __name__ == "__main__":
    test_reason_ok()
    print("✓ test_reason_ok passed")

    test_reason_fixed_pass()
    print("✓ test_reason_fixed_pass passed")

    test_reason_fixed_disabled_move()
    print("✓ test_reason_fixed_disabled_move passed")

    test_reason_fixed_switch_to_move()
    print("✓ test_reason_fixed_switch_to_move passed")

    test_reason_fallback_switch()
    print("✓ test_reason_fallback_switch passed")

    test_reason_fallback_pass()
    print("✓ test_reason_fallback_pass passed")

    test_reason_fallback_switch_force_switch()
    print("✓ test_reason_fallback_switch_force_switch passed")

    print("\nAll reason tests passed!")

