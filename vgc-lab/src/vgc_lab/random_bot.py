"""Random bot utilities for making battle choices."""

from __future__ import annotations

import random
from typing import Any, Optional


def _get_rng(rng: Optional[random.Random]) -> random.Random:
    """Return a usable RNG (either the provided one or the module-level RNG)."""
    return rng if rng is not None else random


def choose_team_preview(request: dict[str, Any], rng: Optional[random.Random] = None) -> str:
    """Choose a random subset of Pokémon for team preview.

    The Showdown protocol for team preview expects commands like:
        "team 1234"
    where the digits are 1-based indexes into the side's `pokemon` list.

    Args:
        request: A JSON object from a `|request|` line with `"teamPreview": true`.
        rng: Optional random number generator; if None, use module-level random.

    Returns:
        A command string such as "team 1234".
    """
    rng = _get_rng(rng)

    side = request.get("side", {})
    pokemon = side.get("pokemon", [])
    total = len(pokemon)
    if total == 0:
        # Nothing to choose – just send "team" so the simulator can continue.
        return "team"

    max_chosen = request.get("maxChosenTeamSize") or total
    max_chosen = max(1, min(int(max_chosen), total))

    available_slots = list(range(1, total + 1))  # 1-based indices
    chosen = rng.sample(available_slots, k=max_chosen)
    chosen.sort()

    choice_str = "team " + "".join(str(slot) for slot in chosen)
    return choice_str


def choose_turn_action(
    request: dict[str, Any],
    rng: Optional[random.Random] = None,
) -> str:
    """Choose a random set of actions for a single turn.

    This supports doubles by choosing one action per active Pokémon, and then
    joining them with ", " to form a single command string.

    Args:
        request: A JSON request object (not a teamPreview request).
        rng: Optional random number generator.

    Returns:
        A command string like "move 1, move 2" or "move 1, switch 3".
    """
    rng = _get_rng(rng)

    side = request.get("side", {})
    active = side.get("active", [])
    all_pokemon = side.get("pokemon", [])

    # Build a quick bench index list: 1-based indices of non-active, non-fainted mons.
    bench_candidates: list[int] = []
    for idx, mon in enumerate(all_pokemon, start=1):
        is_active = bool(mon.get("active"))
        condition = mon.get("condition", "")
        fainted = condition.startswith("0 fnt")
        if not is_active and not fainted:
            bench_candidates.append(idx)

    choices: list[str] = []

    for mon_state in active:
        moves = mon_state.get("moves", []) or []
        can_switch = bool(mon_state.get("canSwitch", False))

        # Collect available (non-disabled) moves as (slot_index, move_info)
        available_moves: list[tuple[int, Any]] = []
        for idx, move in enumerate(moves, start=1):
            if move is None:
                continue
            if move.get("disabled"):
                continue
            available_moves.append((idx, move))

        # Very dumb logic:
        # - If we have at least one available move, 80% of the time use a move;
        # - Otherwise, if we can switch and have bench candidates, switch;
        # - Otherwise, pass.
        if available_moves and (not can_switch or rng.random() < 0.8):
            move_idx, _ = rng.choice(available_moves)
            choices.append(f"move {move_idx}")
        elif can_switch and bench_candidates:
            slot = rng.choice(bench_candidates)
            choices.append(f"switch {slot}")
        else:
            choices.append("pass")

    if not choices:
        return "pass"

    # Doubles: one command per active mon, joined by ", "
    return ", ".join(choices)


def choose_action(
    request: dict[str, Any],
    rng: Optional[random.Random] = None,
) -> str:
    """High-level dispatcher: team preview vs normal turn.

    Args:
        request: JSON request dict decoded from a `|request|` line.
        rng: Optional RNG, for deterministic testing.

    Returns:
        A Showdown command string for this side (e.g. "team 1234", "move 1, move 2").
    """
    if request.get("teamPreview"):
        return choose_team_preview(request, rng=rng)
    return choose_turn_action(request, rng=rng)
