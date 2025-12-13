"""Tests for _sanitize_choice_for_doubles function."""

import sys
from pathlib import Path

# Add the repository root to sys.path so we can import from projects.rl_battle.
sys.path.insert(0, str(Path(__file__).parent.parent))

from projects.rl_battle.online_selfplay import _sanitize_choice_for_doubles


def test_double_force_switch_deduplicates_bench_targets():
    """Test that double force-switch uses distinct bench targets."""
    request = {
        "forceSwitch": [True, True],
        "side": {
            "pokemon": [
                {"condition": "0 fnt", "active": True},   # slot 1 fainted
                {"condition": "0 fnt", "active": True},   # slot 2 fainted
                {"condition": "100/100", "active": False},  # bench 3
                {"condition": "100/100", "active": False},  # bench 4
            ]
        },
    }
    out = _sanitize_choice_for_doubles("switch 3, pass", request)
    # Should assign distinct bench targets
    assert out in {"switch 3, switch 4", "switch 4, switch 3"}


def test_mixed_force_switch_allows_pass_on_free_slot():
    """Test that mixed force-switch only forces the required slot."""
    request = {
        "forceSwitch": [True, False],
        "active": [
            {"moves": [{"target": "normal"}]},  # slot 0 - forced
            {"moves": [{"target": "normal"}]},  # slot 1 - free
        ],
        "side": {
            "pokemon": [
                {"condition": "0 fnt", "active": True},
                {"condition": "100/100", "active": True},
                {"condition": "100/100", "active": False},
            ]
        },
    }
    out = _sanitize_choice_for_doubles("switch 3, pass", request)
    # First slot must be a switch; second slot is allowed to stay 'pass'.
    assert out.startswith("switch ")
    # But second part could be pass or a move
    parts = out.split(", ")
    assert len(parts) == 2
    assert parts[0].startswith("switch ")


def test_adds_default_target_for_targeted_moves():
    """Test that targeted moves get default target '1' if missing."""
    request = {
        "active": [
            {
                "moves": [
                    {"target": "normal"},  # move 1
                    {"target": "self"},    # move 2
                ]
            }
        ]
    }
    out = _sanitize_choice_for_doubles("move 1", request)
    assert out == "move 1 1"

    out2 = _sanitize_choice_for_doubles("move 2", request)
    assert out2 == "move 2"


def test_fixes_disabled_move_by_selecting_enabled_alternative():
    """Test that disabled moves are replaced with first enabled move."""
    request = {
        "active": [
            {
                "moves": [
                    {"disabled": False},  # move 1 - enabled
                    {"disabled": False},  # move 2 - enabled
                    {"disabled": False},  # move 3 - enabled
                    {"disabled": False},  # move 4 - enabled
                ]
            },
            {
                "moves": [
                    {"disabled": True},   # move 1 - disabled
                    {"disabled": False},  # move 2 - enabled
                    {"disabled": False},  # move 3 - enabled
                    {"disabled": False},  # move 4 - enabled
                ]
            }
        ]
    }
    # Slot 0 chooses move 1 (enabled), slot 1 chooses move 1 (disabled) -> should fix slot 1
    out = _sanitize_choice_for_doubles("move 1, move 1", request)
    # Slot 0 should stay as move 1, slot 1 should be fixed to move 2 (first enabled)
    assert out.startswith("move 1,")
    assert "move 2" in out


def test_replaces_pass_with_move_when_legal_moves_available():
    """Test that 'pass' is replaced with a move when legal moves exist."""
    request = {
        "active": [
            {
                "moves": [
                    {"disabled": False},  # move 1 - enabled
                    {"disabled": False},  # move 2 - enabled
                ]
            }
        ],
        "forceSwitch": [False]  # Not a forced switch
    }
    out = _sanitize_choice_for_doubles("pass", request)
    # Should replace 'pass' with the first enabled move
    assert out == "move 1"


def test_fixes_target_mismatch_when_replacing_disabled_move():
    """
    Test that when replacing a disabled move, we rebuild the choice correctly
    based on the new move's target type, not reusing the old target.
    """
    request = {
        "active": [
            {
                "moves": [
                    {"target": "normal", "disabled": True},   # move 1 - disabled, needs target
                    {"target": "self", "disabled": False},    # move 2 - enabled, NO target
                ]
            }
        ]
    }
    # Original choice is "move 1 1" (disabled move with explicit target)
    # Should become "move 2" (no target, since move 2 is "self" target)
    out = _sanitize_choice_for_doubles("move 1 1", request)
    # Should NOT have "move 2 1" - the "1" target should be dropped
    assert out == "move 2"
    
    # Test the reverse: move 1 needs target, move 2 doesn't
    request2 = {
        "active": [
            {
                "moves": [
                    {"target": "self", "disabled": True},     # move 1 - disabled, NO target
                    {"target": "normal", "disabled": False},  # move 2 - enabled, needs target
                ]
            }
        ]
    }
    # Original choice is "move 1" (disabled move, no target)
    # Should become "move 2 1" (with target, since move 2 is "normal" target)
    out2 = _sanitize_choice_for_doubles("move 1", request2)
    assert out2 == "move 2 1"


def test_adds_target_for_single_target_moves_after_trimming():
    """
    Test that after trimming choices to the required count, we still add
    numeric targets for moves that need them (e.g. Heat Crash needs a target).

    Scenario: Doubles where one Pokemon is fainted, so we only need 1 choice,
    but the move needs a target.
    """
    request = {
        "active": [
            {
                "moves": [
                    {
                        "id": "heatcrash",
                        "disabled": False,
                        "target": "normal",  # Needs a numeric target
                    }
                ]
            },
            {
                # Second slot exists but Pokemon is fainted
                "moves": []
            },
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},  # Unfainted
                {"active": True, "condition": "0 fnt"},    # Fainted
            ]
        },
    }

    # Test case 1: Raw choice "move 1" should become "move 1 1"
    out1 = _sanitize_choice_for_doubles("move 1", request)
    assert out1 == "move 1 1", f"Expected 'move 1 1', got {out1!r}"

    # Test case 2: Raw choice "move 1, move 1 1" (doubles format) should be trimmed to "move 1 1"
    out2 = _sanitize_choice_for_doubles("move 1, move 1 1", request)
    assert out2 == "move 1 1", f"Expected 'move 1 1', got {out2!r}"

    # Test case 3: Raw choice "move 1, pass" should be trimmed to "move 1 1"
    out3 = _sanitize_choice_for_doubles("move 1, pass", request)
    assert out3 == "move 1 1", f"Expected 'move 1 1', got {out3!r}"

    # Test case 4: Move with "adjacentFoe" target (like many VGC moves)
    request_adjacent = {
        "active": [
            {
                "moves": [
                    {
                        "id": "flamethrower",
                        "disabled": False,
                        "target": "adjacentFoe",  # Needs target
                    }
                ]
            },
            {"moves": []},
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": True, "condition": "0 fnt"},
            ]
        },
    }
    out4 = _sanitize_choice_for_doubles("move 1", request_adjacent)
    assert out4 == "move 1 1", f"Expected 'move 1 1', got {out4!r}"

    # Test case 5: Move with "self" target should NOT get a numeric target
    request_self = {
        "active": [
            {
                "moves": [
                    {
                        "id": "protect",
                        "disabled": False,
                        "target": "self",  # Does NOT need target
                    }
                ]
            },
            {"moves": []},
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
                {"active": True, "condition": "0 fnt"},
            ]
        },
    }
    out5 = _sanitize_choice_for_doubles("move 1", request_self)
    assert out5 == "move 1", f"Expected 'move 1', got {out5!r} (self-target moves don't need numeric target)"


def test_no_pass_if_enabled_moves_exist():
    """
    Test that 'pass' is replaced with a move when enabled moves exist.

    A request with one active mon whose moves list contains at least one
    non-disabled move. choice = "pass" → sanitized choice must start with
    "move", not "pass".
    """
    request = {
        "active": [
            {
                "moves": [
                    {"target": "normal", "disabled": False},  # move 1 - enabled
                    {"target": "normal", "disabled": False},  # move 2 - enabled
                ]
            }
        ],
        "forceSwitch": [False],  # Not a forced switch
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
            ]
        },
    }
    out = _sanitize_choice_for_doubles("pass", request)
    # Should replace 'pass' with a move, not keep it
    assert out.startswith("move"), f"Expected choice to start with 'move', got {out!r}"
    assert "pass" not in out.lower(), f"Choice should not contain 'pass', got {out!r}"


def test_force_switch_slots_always_switch():
    """
    Test that force-switch slots always output switches, even if policy sends moves.

    forceSwitch = [True, False], active length 2, side.pokemon with at least one bench.
    Raw choice "move 1, move 1".
    Sanitized choice: slot 0 must be "switch N"; slot 1 must be "move ..." or "switch ...".
    """
    request = {
        "forceSwitch": [True, False],
        "active": [
            {"moves": [{"target": "normal", "disabled": False}]},  # slot 0 - forced
            {"moves": [{"target": "normal", "disabled": False}]},  # slot 1 - free
        ],
        "side": {
            "pokemon": [
                {"condition": "0 fnt", "active": True},   # slot 0 fainted
                {"condition": "100/100", "active": True},  # slot 1 alive
                {"condition": "100/100", "active": False},  # bench 3
            ]
        },
    }
    out = _sanitize_choice_for_doubles("move 1, move 1", request)
    parts = out.split(", ")
    assert len(parts) == 2
    # Slot 0 must be a switch (forced)
    assert parts[0].startswith("switch"), f"Slot 0 should be a switch, got {parts[0]!r}"
    # Slot 1 can be a move (it's not forced)
    assert parts[1].startswith("move") or parts[1].startswith("switch"), \
        f"Slot 1 should be move or switch, got {parts[1]!r}"


def test_self_and_allAdjacent_moves_have_no_numeric_target():
    """
    Test that self-target and allAdjacent moves never have numeric targets.

    Construct fake requests with active[0].moves[0].target = "self" or "allAdjacentFoes".
    Raw choice "move 1 1".
    Sanitized result must be "move 1" (no numeric target).
    """
    # Test "self" target
    request_self = {
        "active": [
            {
                "moves": [
                    {"target": "self", "disabled": False},  # Protect-like
                ]
            }
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
            ]
        },
    }
    out_self = _sanitize_choice_for_doubles("move 1 1", request_self)
    assert out_self == "move 1", f"Expected 'move 1' (no target), got {out_self!r}"

    # Test "allAdjacentFoes" target (Blizzard-like)
    request_blizzard = {
        "active": [
            {
                "moves": [
                    {"target": "allAdjacentFoes", "disabled": False},
                ]
            }
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
            ]
        },
    }
    out_blizzard = _sanitize_choice_for_doubles("move 1 1", request_blizzard)
    assert out_blizzard == "move 1", f"Expected 'move 1' (no target), got {out_blizzard!r}"

    # Test "allAdjacent" target
    request_all_adjacent = {
        "active": [
            {
                "moves": [
                    {"target": "allAdjacent", "disabled": False},
                ]
            }
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},
            ]
        },
    }
    out_all_adjacent = _sanitize_choice_for_doubles("move 1 1", request_all_adjacent)
    assert out_all_adjacent == "move 1", f"Expected 'move 1' (no target), got {out_all_adjacent!r}"


def test_single_target_moves_get_numeric_target_after_trimming():
    """
    Test that single-target moves get numeric targets even after trimming.

    Doubles-like request where:
    - active has 2 entries
    - side.pokemon has only one unfainted active mon
    - The move's target is "normal"
    - Raw choice "move 1, move 1" → after sanitization and trimming to one slot,
      result must be "move 1 1" (or "move 1 2").
    """
    request = {
        "active": [
            {
                "moves": [
                    {"target": "normal", "disabled": False},  # Needs target
                ]
            },
            {
                # Second slot exists but Pokemon is fainted
                "moves": []
            },
        ],
        "side": {
            "pokemon": [
                {"active": True, "condition": "100/100"},  # Unfainted
                {"active": True, "condition": "0 fnt"},    # Fainted
            ]
        },
    }
    out = _sanitize_choice_for_doubles("move 1, move 1", request)
    # Should be trimmed to one slot and have target
    assert out == "move 1 1", f"Expected 'move 1 1', got {out!r}"


def test_pure_force_switch_request():
    """
    Test pure force-switch request with no active slots.

    active = [], forceSwitch = [True, True], side.pokemon has two healthy bench mons.
    Raw choice arbitrary.
    Sanitized choice must be "switch X, switch Y" with X≠Y, both valid team indices.
    """
    request = {
        "active": [],  # No active slots
        "forceSwitch": [True, True],
        "side": {
            "pokemon": [
                {"condition": "0 fnt", "active": True},   # fainted
                {"condition": "0 fnt", "active": True},   # fainted
                {"condition": "100/100", "active": False},  # bench 3
                {"condition": "100/100", "active": False},  # bench 4
            ]
        },
    }
    # Test with arbitrary raw choice
    out = _sanitize_choice_for_doubles("move 1, move 1", request)
    parts = out.split(", ")
    assert len(parts) == 2
    # Both must be switches
    assert parts[0].startswith("switch"), f"Part 0 should be switch, got {parts[0]!r}"
    assert parts[1].startswith("switch"), f"Part 1 should be switch, got {parts[1]!r}"
    # They must be distinct
    idx0 = int(parts[0].split()[1])
    idx1 = int(parts[1].split()[1])
    assert idx0 != idx1, f"Switch targets should be distinct, got {idx0} and {idx1}"
    # Both should be valid bench indices (3 or 4)
    assert idx0 in [3, 4], f"Switch target should be 3 or 4, got {idx0}"
    assert idx1 in [3, 4], f"Switch target should be 3 or 4, got {idx1}"


if __name__ == "__main__":
    test_double_force_switch_deduplicates_bench_targets()
    print("✓ test_double_force_switch_deduplicates_bench_targets passed")

    test_mixed_force_switch_allows_pass_on_free_slot()
    print("✓ test_mixed_force_switch_allows_pass_on_free_slot passed")

    test_adds_default_target_for_targeted_moves()
    print("✓ test_adds_default_target_for_targeted_moves passed")

    test_fixes_disabled_move_by_selecting_enabled_alternative()
    print("✓ test_fixes_disabled_move_by_selecting_enabled_alternative passed")

    test_replaces_pass_with_move_when_legal_moves_available()
    print("✓ test_replaces_pass_with_move_when_legal_moves_available passed")

    test_fixes_target_mismatch_when_replacing_disabled_move()
    print("✓ test_fixes_target_mismatch_when_replacing_disabled_move passed")

    test_adds_target_for_single_target_moves_after_trimming()
    print("✓ test_adds_target_for_single_target_moves_after_trimming passed")

    print("\nAll tests passed!")

