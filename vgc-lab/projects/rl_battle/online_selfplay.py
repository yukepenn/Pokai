"""Python-driven online self-play for Pokémon Showdown battles.

This module implements the Python side of the Node↔Python bridge for running
battles where Python policies (e.g., BC policies) make decisions in real-time.

Main Public API:
    run_online_selfplay(cfg: OnlineSelfPlayConfig) -> OnlineSelfPlaySummary
        Run multiple episodes of online self-play and return a summary with
        episode counts, errors, and win statistics.

    Python↔Node Protocol:
    Communication uses JSON messages over stdin/stdout:
    
    Node → Python:
        {"type": "request", "side": "p1"|"p2", "request_type": "preview"|"move"|"force-switch"|"wait", "request": {...}}
    
    Python → Node:
        {"type": "action", "choice": "..."}
    
    Node → Python (battle end):
        {"type": "result", "format_id": "...", "p1_name": "...", "p2_name": "...",
         "winner_side": "p1"|"p2"|"tie"|"unknown", "winner_name": "...", "turns": N,
         "log": "...", "p1_team_packed": "...", "p2_team_packed": "...",
         "tier_name": "...", "trajectory": {"p1": [...], "p2": [...]}, "meta": {...}}
    
    The "result" message is then passed to BattleStore.append_full_battle_from_json()
    to persist the battle and trajectory data.
    
    Step Schema:
    The battle_json["trajectory"]["p1"] and ["p2"] arrays contain step dictionaries.
    Each step is a JSON serialization of BattleStep (see vgc_lab.core.BattleStep), with fields:
    
      - side: str
            Side ID: "p1" or "p2"
      - step_index: int
            Monotone index per side, starting from 0
      - request_type: str
            High-level type: "preview", "move", "force-switch", "wait", etc.
      - rqid: int | None
            Showdown request id (rqid) if present
      - turn: int | None
            Battle turn number if available on the request
      - request: Dict[str, Any]
            Raw Showdown request JSON snapshot (already JSON-serializable)
      - choice: str
            Choice string sent to Showdown, e.g. "team 1234" or "move 1 2, switch 3"
      - sanitize_reason: str | None
            Reason for sanitization applied: "ok", "fixed_pass", "fixed_disabled_move",
            "fixed_switch_to_move", "fallback_switch", "fallback_pass", or None
    
    The trajectory structure is:
        {
            "p1": [step1_dict, step2_dict, ...],
            "p2": [step1_dict, step2_dict, ...],
        }
    
    where each step_dict is produced via BattleStep.model_dump(mode="json").
    This matches exactly what BattleStore.append_full_battle_from_json() expects.
"""

import json
import os
import random
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, TypedDict

from vgc_lab.core import BattleStore, PROJECT_ROOT, SanitizeReason

from .policy import (
    BattleBCPolicy,
    BattleBCPolicyConfig,
    BattleDqnPolicy,
    BattleDqnPolicyConfig,
)


def _compute_required_slots(request: Dict[str, Any]) -> int:
    """
    Compute the number of choice slots required for this request.

    Invariant: This determines how many actions we must send to Showdown.

    Args:
        request: Showdown request dict with "forceSwitch", "active", and "side" fields.

    Returns:
        Number of required slots (1-2 for doubles).
    """
    force_switch = request.get("forceSwitch") or []
    active = request.get("active") or []
    side = request.get("side") or {}
    team = side.get("pokemon") or []

    if not isinstance(force_switch, list):
        force_switch = []
    if not isinstance(active, list):
        active = []
    if not isinstance(team, list):
        team = []

    # Rule 1: If we have force-switch flags, we need one choice per forceSwitch flag
    if isinstance(force_switch, list) and any(bool(x) for x in force_switch):
        return len(force_switch)

    # Rule 2: Normal move context - count unfainted active mons
    unfainted_active = 0
    for mon in team:
        if not mon:
            continue
        if not mon.get("active"):
            continue
        cond = str(mon.get("condition") or "").lower()
        if "fnt" in cond:
            continue
        unfainted_active += 1

    if unfainted_active > 0:
        return unfainted_active

    # Rule 3: Fallback - use length of active if we couldn't infer from side.pokemon
    if isinstance(active, list) and active:
        return len(active)

    # Rule 4: Absolute fallback
    return 1


def _sanitize_choice_for_doubles_with_reasons(
    choice: str, request: Dict[str, Any]
) -> Tuple[str, List[SanitizeReason]]:
    """
    Sanitize a choice string and return both the sanitized choice and reasons per slot.

    This is the SINGLE SOURCE OF TRUTH for sanitizing Python policy outputs.
    It enforces all Showdown constraints for "move" and "force-switch" requests.

    Invariants enforced:
    1. Correct number of slots (based on unfainted active or force-switch flags)
    2. Force-switch slots always output "switch N" if possible
    3. Non-forced slots never output "pass" if any legal move/switch exists
    4. Moves that need targets have numeric targets; those that don't, don't
    5. No duplicate switch targets in force-switch scenarios

    Args:
        choice: Raw choice string from Python policy (e.g., "move 1", "move 1, move 2")
        request: Showdown request dict with "active", "forceSwitch", "side" fields

    Returns:
        Tuple of (sanitized_choice_string, reasons_list) where reasons_list[i] describes
        the sanitization applied to slot i. Possible reasons:
        - "ok": No change needed
        - "fixed_pass": Changed 'pass' to a move
        - "fixed_disabled_move": Changed disabled/invalid move to enabled move
        - "fixed_switch_to_move": Changed switch to move for non-forced slot
        - "fallback_switch": Used fallback switch target
        - "fallback_pass": Used 'pass' as last resort (extremely rare)
    """
    if not isinstance(choice, str) or not choice:
        choice = "pass"

    # ========================================================================
    # STEP 1: Compute required slots and prepare data structures
    # ========================================================================
    required_slots = _compute_required_slots(request)
    active = request.get("active") or []
    force_switch = request.get("forceSwitch") or []
    side = request.get("side") or {}
    team = side.get("pokemon") or []

    if not isinstance(active, list):
        active = []
    if not isinstance(force_switch, list):
        force_switch = []
    if not isinstance(team, list):
        team = []

    # Split choice into per-slot parts
    raw_parts = [p.strip() for p in choice.split(",")]
    parts: List[str] = [p for p in raw_parts if p.strip()]

    # Pad/trim to required_slots
    while len(parts) < required_slots:
        parts.append("pass")
    if len(parts) > required_slots:
        parts = parts[:required_slots]

    # Track reasons for each slot (parallel to parts)
    reasons: List[str] = ["ok"] * len(parts)

    # ========================================================================
    # STEP 2: Precompute bench indices and helper functions
    # ========================================================================
    bench_indices: List[int] = []
    for i, mon in enumerate(team):
        if not mon:
            continue
        cond = str(mon.get("condition", ""))
        is_fainted = "fnt" in cond.lower()
        is_active = bool(mon.get("active"))
        if not is_fainted and not is_active:
            bench_indices.append(i + 1)  # 1-based indexing

    used_switch_targets: List[int] = []

    def _take_switch_target() -> int | None:
        """Return a unique switch target index, preferring bench mons."""
        # 1) Prefer true bench: non-fainted and not active
        for idx in bench_indices:
            if idx not in used_switch_targets:
                used_switch_targets.append(idx)
                return idx

        # 2) Fallback: ANY non-fainted mon (even if active), to avoid 'pass'
        for i, mon in enumerate(team):
            if not mon:
                continue
            cond = str(mon.get("condition", ""))
            is_fainted = "fnt" in cond.lower()
            if is_fainted:
                continue
            idx = i + 1
            if idx not in used_switch_targets:
                used_switch_targets.append(idx)
                return idx

        # 3) Truly nothing left: all mons fainted
        return None

    def _needs_explicit_target(target: str) -> bool:
        """
        Return True if this move target type requires a numeric target argument.

        Uses a denylist approach: only these targets don't need numeric targets.
        """
        t = (target or "").lower()
        no_target_needed = {
            "self",
            "allyside",
            "foeside",
            "all",
            "allyteam",
            "alladjacent",
            "alladjacentfoes",
            "randomnormal",
        }
        return t not in no_target_needed and t != ""

    def _build_move_choice(
        slot: Dict[str, Any], move_index: int, extra_tokens: List[str] | None = None
    ) -> str:
        """
        Build a move choice string with correct target handling.

        Args:
            slot: One element of request["active"]
            move_index: 1-based move index
            extra_tokens: Optional list of flags like "tera", "zmove", etc.

        Returns:
            Move choice string like "move 1 1" or "move 2" (no target for self-target moves)
        """
        moves = slot.get("moves") or []
        if extra_tokens is None:
            extra_tokens = []

        toks: List[str] = ["move", str(move_index)]

        # Check if this move needs a numeric target
        if 1 <= move_index <= len(moves):
            mv = moves[move_index - 1] or {}
            target = mv.get("target") or ""
            if _needs_explicit_target(target):
                # For ally-targeting moves, use -1; for foe-targeting, use 1
                default_target = "-1" if "ally" in target.lower() else "1"
                toks.append(default_target)

        # Append allowed flag tokens
        for tok in extra_tokens:
            if tok.lower() in ("tera", "zmove", "z", "mega", "dynamax"):
                toks.append(tok)

        return " ".join(toks)

    def _first_enabled_move_idx(slot: Dict[str, Any]) -> int | None:
        """Return 1-based index of first non-disabled move, or None."""
        moves = slot.get("moves") or []
        for j, move in enumerate(moves):
            if not move.get("disabled"):
                return j + 1
        return None

    # ========================================================================
    # STEP 3: First pass - process each slot (forced switch vs non-forced)
    # ========================================================================
    for slot_idx in range(required_slots):
        slot_forced = bool(force_switch[slot_idx]) if slot_idx < len(force_switch) else False
        raw_part = parts[slot_idx]
        tokens = (raw_part or "").split()
        if not tokens:
            tokens = ["pass"]

        cmd = tokens[0].lower()

        # --------------------------------------------------------------------
        # FORCED SWITCH SLOT: must output "switch N" if possible
        # --------------------------------------------------------------------
        if slot_forced:
            if cmd != "switch":
                # Not a switch - override with a switch target
                target_idx = _take_switch_target()
                if target_idx is not None:
                    parts[slot_idx] = f"switch {target_idx}"
                    reasons[slot_idx] = "fallback_switch"
                else:
                    # All mons fainted - last resort
                    parts[slot_idx] = "pass"
                    reasons[slot_idx] = "fallback_pass"
            else:
                # Already "switch" - validate and deduplicate
                target_idx: int | None = None
                if len(tokens) >= 2:
                    try:
                        target_idx = int(tokens[1])
                    except ValueError:
                        target_idx = None

                if target_idx is not None:
                    if target_idx in used_switch_targets:
                        # Duplicate - try to get a new one
                        new_idx = _take_switch_target()
                        if new_idx is not None:
                            parts[slot_idx] = f"switch {new_idx}"
                            reasons[slot_idx] = "fallback_switch"
                        else:
                            # Can't get new one, keep existing (better than nothing)
                            parts[slot_idx] = f"switch {target_idx}"
                            # Keep reason as "ok" since we kept the original switch
                    else:
                        # Valid and unused
                        used_switch_targets.append(target_idx)
                        parts[slot_idx] = f"switch {target_idx}"
                        reasons[slot_idx] = "ok"  # Valid forced switch
                else:
                    # "switch" but missing index
                    new_idx = _take_switch_target()
                    if new_idx is not None:
                        parts[slot_idx] = f"switch {new_idx}"
                        reasons[slot_idx] = "fallback_switch"
                    else:
                        parts[slot_idx] = "pass"
                        reasons[slot_idx] = "fallback_pass"
            continue

        # --------------------------------------------------------------------
        # NON-FORCED SLOT: prefer moves, allow switches, never pass if avoidable
        # --------------------------------------------------------------------
        # Special case: if active is empty (all active mons fainted)
        # Showdown quirk: When active is completely empty in a force-switch request,
        # ALL slots must switch (not just the explicitly forced ones)
        # This is because there are no active mons to use moves, so switches are required
        if not active or len(active) == 0:
            # When active is empty, treat all slots as requiring switches
            target_idx = _take_switch_target()
            if target_idx is not None:
                parts[slot_idx] = f"switch {target_idx}"
                reasons[slot_idx] = "fallback_switch"
            else:
                # No valid switch target - last resort pass (should be extremely rare)
                parts[slot_idx] = "pass"
                reasons[slot_idx] = "fallback_pass"
            continue
        
        # Handle case where active has fewer entries than required_slots
        # (shouldn't happen in normal doubles, but handle gracefully)
        if slot_idx >= len(active):
            # No active entry for this slot - use last active entry as fallback
            # or use first active entry if available
            slot_req = active[-1] if active else {}
        else:
            slot_req = active[slot_idx]
        moves = slot_req.get("moves") or [] if slot_req else []
        
        # Safety: if we don't have moves for this slot, try to use moves from any active entry
        if not moves and active:
            for act in active:
                act_moves = act.get("moves") or []
                if act_moves:
                    slot_req = act
                    moves = act_moves
                    break

        if cmd == "move":
            # Parse move index and collect extra tokens
            move_idx_str = tokens[1] if len(tokens) >= 2 else "1"
            try:
                move_idx = int(move_idx_str)
            except ValueError:
                move_idx = 1

            # Collect extra flag tokens (tera, mega, etc.)
            extra_tokens_list: List[str] = []
            for tok in tokens[2:]:
                if tok.lower() in ("tera", "zmove", "z", "mega", "dynamax"):
                    extra_tokens_list.append(tok)

            # Validate move index and disabled status
            if (
                move_idx >= 1
                and move_idx <= len(moves)
                and not moves[move_idx - 1].get("disabled")
            ):
                # Valid move - use it
                parts[slot_idx] = _build_move_choice(slot_req, move_idx, extra_tokens_list)
                reasons[slot_idx] = "ok"  # Valid move, no change needed
            else:
                # Invalid or disabled - try to find enabled alternative
                alt_idx = _first_enabled_move_idx(slot_req)
                if alt_idx is not None:
                    parts[slot_idx] = _build_move_choice(slot_req, alt_idx, extra_tokens_list)
                    reasons[slot_idx] = "fixed_disabled_move"
                else:
                    # No enabled moves - for non-forced slots, cannot use switch
                    # Use first move (even if disabled) or pass as last resort
                    if moves and len(moves) > 0:
                        # Use first move even if disabled (Showdown allows disabled moves in choice)
                        parts[slot_idx] = _build_move_choice(slot_req, 1, extra_tokens_list)
                        reasons[slot_idx] = "fallback_switch"  # Treated as fallback since enabled moves unavailable
                    else:
                        # Try one more time to find moves from any active entry
                        if active:
                            for act in active:
                                act_moves = act.get("moves") or []
                                if act_moves:
                                    parts[slot_idx] = _build_move_choice(act, 1, extra_tokens_list)
                                    reasons[slot_idx] = "fallback_switch"
                                    break
                            else:
                                parts[slot_idx] = "pass"  # Last resort
                                reasons[slot_idx] = "fallback_pass"
                        else:
                            parts[slot_idx] = "pass"  # Last resort
                            reasons[slot_idx] = "fallback_pass"

        elif cmd == "switch":
            # Policy wants to switch, but for non-forced slots in doubles,
            # Showdown does NOT allow switches - only moves are allowed.
            # We MUST convert switches to moves (or pass as last resort).
            alt_move_idx = _first_enabled_move_idx(slot_req)
            if alt_move_idx is not None:
                # Convert switch to move for non-forced slots
                parts[slot_idx] = _build_move_choice(slot_req, alt_move_idx, [])
                reasons[slot_idx] = "fixed_switch_to_move"
            else:
                # No enabled moves available - cannot switch in non-forced slot
                # We must use a move (even if disabled) - Showdown will handle disabled moves
                # Use the first move available (index 1)
                if moves and len(moves) > 0:
                    # Use first move even if disabled (Showdown allows disabled moves in choice)
                    parts[slot_idx] = _build_move_choice(slot_req, 1, [])
                    reasons[slot_idx] = "fallback_switch"  # Treated as fallback since moves are disabled
                else:
                    # Try one more time to find moves from any active entry
                    if active:
                        for act in active:
                            act_moves = act.get("moves") or []
                            if act_moves:
                                parts[slot_idx] = _build_move_choice(act, 1, [])
                                reasons[slot_idx] = "fallback_switch"
                                break
                        else:
                            # No moves at all - extremely rare, use pass as absolute last resort
                            parts[slot_idx] = "pass"
                            reasons[slot_idx] = "fallback_pass"
                    else:
                        # No moves at all - extremely rare, use pass as absolute last resort
                        parts[slot_idx] = "pass"
                        reasons[slot_idx] = "fallback_pass"

        else:
            # "pass" or anything else - replace with move/switch if possible
            # ALWAYS prefer a move if any moves exist (even disabled ones)
            alt_move_idx = _first_enabled_move_idx(slot_req)
            if alt_move_idx is not None:
                parts[slot_idx] = _build_move_choice(slot_req, alt_move_idx, [])
                reasons[slot_idx] = "fixed_pass"
            elif moves and len(moves) > 0:
                # No enabled moves, but moves exist - use first move even if disabled
                # Showdown allows disabled moves in choice
                parts[slot_idx] = _build_move_choice(slot_req, 1, [])
                reasons[slot_idx] = "fixed_pass"  # We're fixing "pass" to a move
            elif active:
                # Try to find moves from any active entry
                for act in active:
                    act_moves = act.get("moves") or []
                    if act_moves:
                        slot_req = act
                        moves = act_moves
                        parts[slot_idx] = _build_move_choice(slot_req, 1, [])
                        reasons[slot_idx] = "fixed_pass"
                        break
                else:
                    # Truly no moves available - use pass as last resort
                    # Note: Debug logging removed - cfg.debug not available in sanitizer
                    parts[slot_idx] = "pass"
                    reasons[slot_idx] = "fallback_pass"
            else:
                # No active entries at all - this should have been caught earlier
                # but handle it defensively
                # Note: Debug logging removed - cfg.debug not available in sanitizer
                parts[slot_idx] = "pass"
                reasons[slot_idx] = "fallback_pass"

    # ========================================================================
    # STEP 4: Second pass - target correction only (after slot processing)
    # ========================================================================
    # This pass ensures moves have correct targets: add if needed, remove if not allowed
    for slot_idx in range(len(parts)):
        if slot_idx >= len(active):
            continue
        part = parts[slot_idx]
        tokens = part.split()
        if not tokens or tokens[0].lower() != "move":
            continue

        # Parse move index
        move_idx: int | None = None
        if len(tokens) >= 2:
            try:
                move_idx = int(tokens[1])
            except ValueError:
                continue

        if move_idx is None or move_idx < 1:
            continue

        slot = active[slot_idx] or {}
        moves = slot.get("moves") or []
        if move_idx > len(moves):
            continue

        move_data = moves[move_idx - 1] or {}
        target = str(move_data.get("target", "")).lower()

        # Find numeric target token if present
        numeric_target_idx: int | None = None
        for i in range(2, len(tokens)):
            try:
                int(tokens[i])  # If this parses as int, it's a numeric target
                numeric_target_idx = i
                break
            except ValueError:
                continue

        needs_target = _needs_explicit_target(target)

        if needs_target:
            # Move needs a target - add one if missing
            if numeric_target_idx is None:
                # For ally-targeting moves, use -1 (ally); for foe-targeting, use 1 (foe slot 1)
                # Default to 1 (foe) unless target type suggests ally
                default_target = "-1" if "ally" in target.lower() else "1"
                tokens.insert(2, default_target)
                parts[slot_idx] = " ".join(tokens)
        else:
            # Move does NOT need a target - remove it if present
            if numeric_target_idx is not None:
                tokens.pop(numeric_target_idx)
                parts[slot_idx] = " ".join(tokens)

    # ========================================================================
    # STEP 5: Final join and debug logging
    # ========================================================================
    choice_string = ", ".join(parts)

    # Assert that all reasons are valid (type check at runtime for safety)
    assert all(
        r in ("ok", "fixed_pass", "fixed_disabled_move", "fixed_switch_to_move", "fallback_switch", "fallback_pass")
        for r in reasons
    ), f"Unexpected sanitize_reason in {reasons}"

    # Debug logging removed - cfg.debug not available in this function
    # Note: This function is called from sanitizer which doesn't have access to cfg
    
    return choice_string, reasons


def _sanitize_choice_for_doubles(choice: str, request: Dict[str, Any]) -> str:
    """
    Sanitize a choice string to ensure it's valid for Showdown doubles battles.

    Backward-compatible wrapper that returns only the choice string.
    Use _sanitize_choice_for_doubles_with_reasons if you need sanitization reasons.
    """
    choice_str, _ = _sanitize_choice_for_doubles_with_reasons(choice, request)
    return choice_str


def _random_preview_choice(request: Dict[str, Any]) -> str:
    """Generate a random team preview choice (bring-4) from a preview request."""
    side = request.get("side") or {}
    team = side.get("pokemon") or []
    if not isinstance(team, list):
        team = []

    total = len(team)
    if total == 0:
        return "team 1"

    max_chosen = request.get("maxChosenTeamSize", 4)
    chosen = list(range(1, min(max_chosen, total) + 1))
    return "team " + "".join(str(x) for x in chosen)


def _emergency_valid_choice(request: Dict[str, Any], req_type: str) -> str:
    """
    Generate a minimal valid choice when the sanitizer fails.
    
    This is a last-resort fallback that tries to construct a choice
    that won't be rejected by Showdown, even if it's not optimal.
    
    Note: This function is called when:
    1. The sanitizer throws an exception (shouldn't happen, but defensive)
    2. The sanitizer returns a choice containing "pass" and the safety check catches it
    
    It uses manual logic to construct a valid choice without calling the sanitizer.
    """
    if req_type == "preview":
        return _random_preview_choice(request)
    
    # Manual fallback - construct choice without using sanitizer
    active = request.get("active") or []
    force_switch = request.get("forceSwitch") or []
    side = request.get("side") or {}
    team = side.get("pokemon") or []
    
    # Count required slots using same logic as sanitizer
    required_slots = _compute_required_slots(request)
    if required_slots == 0:
        required_slots = 1  # Fallback to at least one slot
    
    # Track used switch targets to avoid duplicates
    used_switch_targets: List[int] = []
    
    parts = []
    for slot_idx in range(required_slots):
        # Only treat as forced if explicitly marked in forceSwitch array
        # Don't infer from fainted mons - let the sanitizer handle that
        is_forced = force_switch[slot_idx] if slot_idx < len(force_switch) and force_switch[slot_idx] else False
        
        # Special case: if active is empty, ALL slots must switch (same logic as sanitizer)
        if not active or len(active) == 0:
            # Find a switch target (avoiding duplicates)
            switch_target = None
            for i, mon in enumerate(team):
                if isinstance(mon, dict):
                    condition = mon.get("condition", "")
                    # Handle condition as string or dict
                    if isinstance(condition, dict):
                        condition = str(condition.get("condition", ""))
                    condition_str = str(condition).lower()
                    is_active_mon = mon.get("active", False)
                    target_idx = i + 1  # 1-indexed
                    # Look for bench mon (not active, not fainted, not already used)
                    if not is_active_mon and "fnt" not in condition_str and target_idx not in used_switch_targets:
                        switch_target = target_idx
                        used_switch_targets.append(target_idx)
                        break
            
            if switch_target:
                parts.append(f"switch {switch_target}")
            else:
                # No valid unused switch target - try any non-active mon
                for i, mon in enumerate(team):
                    if isinstance(mon, dict):
                        target_idx = i + 1
                        if not mon.get("active", False) and target_idx not in used_switch_targets:
                            used_switch_targets.append(target_idx)
                            parts.append(f"switch {target_idx}")
                            break
                else:
                    # Truly desperate: use first team member (even if duplicate)
                    parts.append("switch 1")
            continue
        
        if is_forced:
            # Try to find a switch target (avoiding duplicates)
            switch_target = None
            for i, mon in enumerate(team):
                if isinstance(mon, dict):
                    condition = mon.get("condition", "")
                    # Handle condition as string or dict
                    if isinstance(condition, dict):
                        condition = str(condition.get("condition", ""))
                    condition_str = str(condition).lower()
                    is_active_mon = mon.get("active", False)
                    target_idx = i + 1  # 1-indexed
                    # Look for bench mon (not active, not fainted, not already used)
                    if not is_active_mon and "fnt" not in condition_str and target_idx not in used_switch_targets:
                        switch_target = target_idx
                        used_switch_targets.append(target_idx)
                        break
            
            if switch_target:
                parts.append(f"switch {switch_target}")
            else:
                # No valid unused switch target - try any non-active mon
                for i, mon in enumerate(team):
                    if isinstance(mon, dict):
                        target_idx = i + 1
                        if not mon.get("active", False) and target_idx not in used_switch_targets:
                            used_switch_targets.append(target_idx)
                            parts.append(f"switch {target_idx}")
                            break
                else:
                    # Truly desperate: use first team member (even if duplicate)
                    parts.append("switch 1")
        else:
            # Non-forced: try to use a move
            slot_req = active[slot_idx] if slot_idx < len(active) else {}
            moves = slot_req.get("moves") or []
            if moves and len(moves) > 0:
                move_id = 1
                move = moves[0]
                target = move.get("target", "")
                # Check if move needs explicit target using same logic as sanitizer
                t = (target or "").lower()
                no_target_needed = {
                    "self", "allyside", "foeside", "all", "allyteam",
                    "alladjacent", "alladjacentfoes", "randomnormal",
                }
                if t not in no_target_needed:
                    # For ally-targeting moves, use -1; for foe-targeting, use 1
                    default_target = "-1" if "ally" in t else "1"
                    parts.append(f"move {move_id} {default_target}")
                else:
                    parts.append(f"move {move_id}")
            else:
                # No moves available - this is very unusual, but try move 1 anyway
                parts.append("move 1")
    
    # parts should never be empty since required_slots >= 1, but be defensive
    if not parts:
        parts.append("move 1")  # Absolute last resort
    
    return ", ".join(parts)


# Re-export RandomAgent from core for convenience
from vgc_lab.core import RandomAgent  # noqa: E402


class OnlineSelfPlaySummary(TypedDict):
    """
    Canonical summary structure returned by online self-play.
    
    This type defines the stable API contract for run_online_selfplay() results.
    All counts are non-negative integers, and the invariant holds:
        episodes == errors + p1_wins + p2_wins + draws
    """
    episodes: int  # Total attempted episodes (should always equal cfg.num_episodes)
    errors: int    # Episodes that failed or didn't yield a valid result
    p1_wins: int   # Number of episodes won by player 1
    p2_wins: int   # Number of episodes won by player 2
    draws: int     # Number of episodes ending in tie or "unknown" winner


PythonPolicyKind = Literal["random", "bc", "dqn"]


@dataclass
class OnlineSelfPlayConfig:
    """
    Configuration for Python-driven online self-play.

    Used by scripts/cli.py online-selfplay with:
      - num_episodes
      - format_id
      - p1_policy / p2_policy: Node-side policy types ("node_random_v1", "python_external_v1")
      - p1_python_policy / p2_python_policy: Python-side policy kinds when using "python_external_v1"
        (one of: "random", "bc", "dqn")
      - seed
      - write_trajectories
      - strict_invalid_choice
      - debug

    Policy selection semantics:
      - If p1_policy == "node_random_v1", p1_python_policy is ignored (Node handles it).
      - If p1_policy == "python_external_v1", p1_python_policy determines which Python policy to use:
        - "random": RandomAgent
        - "bc": BattleBCPolicy
        - "dqn": BattleDqnPolicy
      - Same logic applies to p2_policy / p2_python_policy.
    """

    num_episodes: int = 3
    format_id: str = "gen9vgc2026regf"
    p1_policy: str = "python_external_v1"
    p2_policy: str = "node_random_v1"
    p1_python_policy: PythonPolicyKind = "random"
    p2_python_policy: PythonPolicyKind = "random"
    seed: int = 42
    write_trajectories: bool = True
    strict_invalid_choice: bool = True
    debug: bool = False


class PythonPolicyRouter:
    """
    Route Showdown requests to Python-side policies (RandomAgent / BattleBCPolicy / BattleDqnPolicy).

    - For team preview: use RandomAgent.choose_team_preview() to pick bring-4.
    - For move / force-switch: route based on per-side python policy kind configured in cfg.
    """

    def __init__(
        self,
        cfg: OnlineSelfPlayConfig,
        rng: Optional[random.Random] = None,
    ) -> None:
        """
        Initialize router with configuration.

        Args:
            cfg: OnlineSelfPlayConfig with format_id and python policy kinds per side.
            rng: Optional random number generator (defaults to new Random instance).
        """
        self.format_id = cfg.format_id
        self.rng = rng or random.Random()
        self.p1_kind = cfg.p1_python_policy
        self.p2_kind = cfg.p2_python_policy

        # Always create RandomAgent
        self.random_agent = RandomAgent(self.rng)

        # Optionally create BC policy if needed
        self.bc_policy: Optional[BattleBCPolicy] = None
        if self.p1_kind == "bc" or self.p2_kind == "bc":
            try:
                self.bc_policy = BattleBCPolicy(BattleBCPolicyConfig(format_id=cfg.format_id))
                print(
                    f"[online-selfplay] Loaded BattleBCPolicy "
                    f"(format_id={cfg.format_id}, num_actions={self.bc_policy.num_actions})",
                    file=sys.stderr,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    "[online-selfplay] Failed to load BattleBCPolicy: "
                    f"{e}. Will use RandomAgent fallback if BC is requested.",
                    file=sys.stderr,
                )

        # Optionally create DQN policy if needed
        self.dqn_policy: Optional[BattleDqnPolicy] = None
        if self.p1_kind == "dqn" or self.p2_kind == "dqn":
            try:
                self.dqn_policy = BattleDqnPolicy(
                    BattleDqnPolicyConfig(format_id=cfg.format_id, device="cpu")
                )
                print(
                    f"[online-selfplay] Loaded BattleDqnPolicy "
                    f"(format_id={cfg.format_id}, num_actions={self.dqn_policy.num_actions})",
                    file=sys.stderr,
                )
            except Exception as e:  # noqa: BLE001
                print(
                    "[online-selfplay] Failed to load BattleDqnPolicy: "
                    f"{e}. Will use RandomAgent fallback if DQN is requested.",
                    file=sys.stderr,
                )

    # ------------------------------------------------------------------
    # Internal helpers for different request types
    # ------------------------------------------------------------------

    def _choose_team_preview(self, request: Dict[str, Any]) -> str:
        """Use RandomAgent to choose bring-4 from the preview request."""
        return self.random_agent.choose_team_preview(request)

    def _choose_move_like(self, request: Dict[str, Any], side: str) -> str:
        """
        Handle normal move / force-switch style requests.

        Routes to the appropriate policy based on side and configured python policy kind.
        Note: sanitization happens at the protocol level in _run_single_episode.

        Args:
            request: Showdown request dict.
            side: "p1" or "p2" to determine which policy kind to use.
        """
        # Determine policy kind for this side
        if side == "p1":
            kind = self.p1_kind
        elif side == "p2":
            kind = self.p2_kind
        else:
            print(
                f"[online-selfplay] Unknown side {side!r}, falling back to RandomAgent",
                file=sys.stderr,
            )
            return self.random_agent.choose_turn_action(request)

        # Route based on policy kind
        if kind == "random":
            return self.random_agent.choose_turn_action(request)

        elif kind == "bc":
            if self.bc_policy is None:
                print(
                    "[online-selfplay] BC policy requested but not available, "
                    "falling back to RandomAgent",
                    file=sys.stderr,
                )
                return self.random_agent.choose_turn_action(request)
            try:
                _, raw_choice = self.bc_policy.choose_action(request, temperature=0.0)
                return raw_choice
            except Exception as e:  # noqa: BLE001
                print(
                    "[online-selfplay] BattleBCPolicy error, "
                    f"falling back to RandomAgent. Error: {e}",
                    file=sys.stderr,
                )
                return self.random_agent.choose_turn_action(request)

        elif kind == "dqn":
            if self.dqn_policy is None:
                print(
                    "[online-selfplay] DQN policy requested but not available, "
                    "falling back to RandomAgent",
                    file=sys.stderr,
                )
                return self.random_agent.choose_turn_action(request)
            try:
                raw_choice = self.dqn_policy.choose_showdown_command(request)
                return raw_choice
            except Exception as e:  # noqa: BLE001
                print(
                    "[online-selfplay] BattleDqnPolicy error, "
                    f"falling back to RandomAgent. Error: {e}",
                    file=sys.stderr,
                )
                return self.random_agent.choose_turn_action(request)

        else:
            raise ValueError(
                f"Unknown python policy kind: {kind!r} for side {side}. "
                "Expected one of: 'random', 'bc', 'dqn'"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def choose_for_request(self, msg: Dict[str, Any]) -> str:
        """
        Decide a Showdown choice string for a given protocol message.

        The message comes from Node (py_policy_selfplay.js) and has shape:
          {
            "type": "request",
            "side": "p1" | "p2",
            "request_type": "preview" | "move" | "force-switch" | "wait",
            "request": { ... Showdown request ... }
          }
        
        This method routes to appropriate policy methods based on side and configured
        python policy kind, but does NOT sanitize. Sanitization happens in
        _run_single_episode before sending to Node.
        """
        req_type = msg.get("request_type", "unknown")
        request = msg.get("request", {}) or {}
        side = msg.get("side", "p1")  # Default to p1 if missing

        if req_type == "preview":
            return self._choose_team_preview(request)
        elif req_type in ("move", "force-switch"):
            return self._choose_move_like(request, side)
        elif req_type == "wait":
            return "pass"
        else:
            # Unknown request type - return pass as safe fallback
            return "pass"


def _spawn_stderr_drain_thread(proc: subprocess.Popen, episode_idx: int) -> None:
    """Continuously mirror Node's stderr lines into Python's stderr.

    This runs in a daemon thread so it won't block process shutdown.
    """
    if proc.stderr is None:
        return

    def _drain() -> None:
        for line in proc.stderr:
            print(f"[online-selfplay] (episode {episode_idx + 1}) [node-stderr] {line.rstrip()}", file=sys.stderr)

    thread = threading.Thread(target=_drain, daemon=True)
    thread.start()


def _run_single_episode(
    cfg: OnlineSelfPlayConfig,
    store: BattleStore,
    router: PythonPolicyRouter,
    episode_idx: int,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    Run a single episode of online self-play via Node.js bridge.
    
    Args:
        cfg: Online self-play configuration
        store: BattleStore instance for persisting battles (used internally)
        router: Policy router for handling Python-side policy decisions
        episode_idx: Zero-based episode index for logging
    
    Returns:
        Tuple of (battle_json, error):
        - battle_json: Dict[str, Any] containing battle result with trajectory data,
          or None if the episode failed. The dict matches the schema expected by
          BattleStore.append_full_battle_from_json().
        - error: Optional[str] error message if episode failed, None otherwise.
        
    Note:
        This function does not persist battles itself. The caller (run_online_selfplay)
        is responsible for calling store.append_full_battle_from_json(battle_json)
        when battle_json is not None and persistence is desired.
    """
    from vgc_lab.core import BattleStep

    # Launch Node.js process
    node_script = PROJECT_ROOT / "js" / "py_policy_selfplay.js"
    cmd = [
        "node",
        str(node_script),
        "--format-id",
        cfg.format_id,
        "--p1-name",
        "Bot1",
        "--p2-name",
        "Bot2",
        "--p1-policy",
        cfg.p1_policy,
        "--p2-policy",
        cfg.p2_policy,
    ]

    print(
        f"[online-selfplay] Episode {episode_idx + 1}: launching Node: {' '.join(cmd)}",
        file=sys.stderr,
    )

    # Prepare environment with strict_invalid_choice and debug flags
    env = os.environ.copy()
    env["PY_POLICY_STRICT_INVALID_CHOICE"] = "1" if cfg.strict_invalid_choice else "0"
    if cfg.debug:
        env["PY_POLICY_DEBUG"] = "1"

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line buffered
        env=env,
    )

    battle_json: Optional[Dict[str, Any]] = None
    error_msg: Optional[str] = None
    steps_p1: List[BattleStep] = []
    steps_p2: List[BattleStep] = []

    # Start background stderr mirror for Node process
    _spawn_stderr_drain_thread(proc, episode_idx)

    try:
        assert proc.stdout is not None
        assert proc.stdin is not None

        # Main protocol loop: read lines from Node's stdout
        max_messages = 500  # Hard cap to prevent infinite loops
        messages_seen = 0
        
        while True:
            raw_line = proc.stdout.readline()
            if not raw_line:
                # Node has exited; if we never saw a "result", treat as error.
                break

            messages_seen += 1
            if messages_seen > max_messages:
                error_msg = f"Exceeded max_messages={max_messages} without receiving a final result"
                print(f"[online-selfplay] {error_msg}", file=sys.stderr)
                battle_json = None
                break

            line = raw_line.strip()
            if not line:
                continue

            # Each line from Node should be a JSON object.
            try:
                msg = json.loads(line)
            except json.JSONDecodeError:
                # Non-JSON lines (if any) are logged and ignored.
                if cfg.debug:
                    print(
                        f"[online-selfplay] Non-JSON line from Node stdout: {line}",
                        file=sys.stderr,
                    )
                continue

            msg_type = msg.get("type")

            if msg_type == "request":
                side = msg.get("side", "unknown")
                req_type = msg.get("request_type", "unknown")
                request = msg.get("request", {}) or {}

                # 'wait' requests are handled entirely on the Node side.
                if req_type == "wait":
                    continue

                sanitize_reason: Optional[str] = None
                raw_choice: Optional[str] = None  # Track for debugging
                try:
                    if req_type == "preview":
                        # Team preview: do NOT go through _sanitize_choice_for_doubles.
                        raw_choice = router._choose_team_preview(request)
                        choice_str = raw_choice.strip()

                        # If the policy returns something non-standard like 'pass', fall back
                        # to a safe random preview choice.
                        if not choice_str.startswith("team"):
                            choice_str = _random_preview_choice(request)
                        sanitize_reason = None  # Preview choices don't get sanitized
                    else:
                        # 'move' or 'force-switch'
                        raw_choice = router._choose_move_like(request)
                        choice_str, reasons = _sanitize_choice_for_doubles_with_reasons(
                            raw_choice, request
                        )
                        # For multi-slot choices, use first non-"ok" reason if any, otherwise "ok"
                        sanitize_reason = "ok"
                        for r in reasons:
                            if r != "ok":
                                sanitize_reason = r
                                break

                except Exception as e:  # noqa: BLE001
                    print(
                        "[online-selfplay] Error in Python policy for request: "
                        f"{e}. Using fallback choice.",
                        file=sys.stderr,
                    )
                    # For move/force-switch requests, use sanitizer with "pass" to generate a valid choice
                    if req_type not in ("preview", "wait"):
                        try:
                            choice_str, reasons = _sanitize_choice_for_doubles_with_reasons(
                                "pass", request
                            )
                            sanitize_reason = reasons[0] if reasons else None
                        except Exception as e2:  # noqa: BLE001
                            print(
                                f"[online-selfplay] CRITICAL: Sanitizer failed: {e2}. "
                                "Using emergency fallback choice generator.",
                                file=sys.stderr,
                            )
                            # Last resort: try to generate a safe choice manually
                            choice_str = _emergency_valid_choice(request, req_type)
                            sanitize_reason = None  # Emergency fallback - reason unknown
                    else:
                        # Preview or wait: use appropriate fallback
                        if req_type == "preview":
                            choice_str = _random_preview_choice(request)
                        else:
                            choice_str = "pass"  # Wait requests should pass
                        sanitize_reason = None

                # Final safety check: ensure we never send "pass" for move/force-switch requests
                # Note: The sanitizer now handles empty active by making all slots switch,
                # so "pass" should not appear in the sanitized choice
                if req_type in ("move", "force-switch"):
                    # Check if choice contains "pass" (either as entire choice or as a slot)
                    choice_parts = [p.strip() for p in choice_str.split(",")]
                    if "pass" in choice_parts:
                        # The sanitizer should not produce "pass" for move/force-switch requests
                        # except in truly impossible scenarios (all mons fainted, no bench available)
                        # For now, treat any "pass" as an error and use emergency fallback
                        if cfg.debug:
                            sys.stderr.write(
                                f"[online-selfplay] CRITICAL: About to send choice containing 'pass' "
                                f"for {req_type} request: {choice_str!r}. Using emergency fallback.\n"
                            )
                        choice_str = _emergency_valid_choice(request, req_type)
                        sanitize_reason = "fallback_pass"
                        
                        # Double-check emergency function didn't return pass
                        # If it did, we're in a truly impossible scenario (all mons fainted, no bench)
                        # In that case, we have no choice but to send "pass" and let Showdown handle it
                        if "pass" in choice_str:
                            if cfg.debug:
                                sys.stderr.write(
                                    f"[online-selfplay] WARN: Emergency function returned 'pass': "
                                    f"{choice_str!r}. This indicates an impossible scenario.\n"
                                )

                # Absolute final check - should never send "pass" for move/force-switch
                if req_type in ("move", "force-switch") and choice_str.strip() == "pass":
                    if cfg.debug:
                        sys.stderr.write(
                            f"[online-selfplay] CRITICAL FINAL: About to send pure 'pass' for {req_type}. "
                            f"Force correcting to 'move 1'.\n"
                        )
                    choice_str = "move 1"

                # Trace logging for strict mode debugging (only for non-preview requests)
                if cfg.debug and req_type != "preview":
                    sys.stderr.write(
                        "[online-selfplay] TRACE choice "
                        f"req_type={req_type!r}, "
                        f"raw_choice={raw_choice!r}, "
                        f"sanitized_choice={choice_str!r}, "
                        f"sanitize_reason={sanitize_reason!r}, "
                        f"forceSwitch={request.get('forceSwitch')!r}, "
                        f"active_len={len(request.get('active') or [])}\n"
                    )

                if cfg.debug:
                    sys.stderr.write(
                        f"[online-selfplay] Sending choice to Node: req_type={req_type!r}, "
                        f"choice={choice_str!r}\n"
                    )

                out = {"type": "action", "choice": choice_str}
                proc.stdin.write(json.dumps(out) + "\n")
                proc.stdin.flush()

                # Log the step
                step = BattleStep(
                    side=side,
                    step_index=len(steps_p1) if side == "p1" else len(steps_p2),
                    request_type=req_type,
                    rqid=request.get("rqid"),
                    turn=request.get("turn"),
                    request=request,
                    choice=choice_str,
                    sanitize_reason=sanitize_reason,
                )
                if side == "p1":
                    steps_p1.append(step)
                else:
                    steps_p2.append(step)

            elif msg_type == "result":
                # Store the result message with collected steps
                battle_json = msg.copy()
                # Add trajectory steps to the result message
                battle_json["trajectory"] = {
                    "p1": [s.model_dump(mode="json") for s in steps_p1],
                    "p2": [s.model_dump(mode="json") for s in steps_p2],
                }
                break

            else:
                # Unknown message type – log and ignore.
                if cfg.debug:
                    print(
                        f"[online-selfplay] unknown msg type: {msg_type!r}",
                        file=sys.stderr,
                    )

        proc.wait()

        if battle_json is None:
            error_msg = (
                "Node process exited without emitting final battle JSON "
                f"(returncode={proc.returncode})"
            )
            print(f"[online-selfplay] {error_msg}", file=sys.stderr)
            return None, error_msg

        # Return the battle_json dict for use with append_full_battle_from_json
        return battle_json, None

    finally:
        proc.stdout.close()
        proc.stdin.close()
        proc.wait()


def _validate_summary(summary: OnlineSelfPlaySummary) -> None:
    """
    Validate that the summary satisfies the required invariants.
    
    Raises:
        ValueError: If any invariant is violated.
    """
    # Check all counts are non-negative integers
    for key, value in summary.items():
        if not isinstance(value, int):
            raise ValueError(f"Summary field '{key}' must be int, got {type(value).__name__}")
        if value < 0:
            raise ValueError(f"Summary field '{key}' must be non-negative, got {value}")
    
    # Check invariant: episodes == errors + p1_wins + p2_wins + draws
    total = summary["errors"] + summary["p1_wins"] + summary["p2_wins"] + summary["draws"]
    if summary["episodes"] != total:
        raise ValueError(
            f"Summary invariant violated: episodes ({summary['episodes']}) != "
            f"errors + p1_wins + p2_wins + draws ({total})"
        )


def run_online_selfplay(cfg: OnlineSelfPlayConfig) -> OnlineSelfPlaySummary:
    """
    Run multiple episodes of online self-play through the Node.js bridge.
    
    This function orchestrates multiple battle episodes where Python policies
    interact with the Pokemon Showdown simulator via the Node.js bridge.
    
    Args:
        cfg: Configuration for online self-play including number of episodes,
             policies, format, and whether to write trajectories.
    
    Returns:
        A summary dictionary with the following fields:
        - episodes (int): Total number of attempted episodes (regardless of success)
        - errors (int): Number of episodes that failed with an error
        - p1_wins (int): Number of episodes won by player 1
        - p2_wins (int): Number of episodes won by player 2
        - draws (int): Number of episodes that ended in a draw/tie
        
        Note: episodes == errors + p1_wins + p2_wins + draws
    
    Protocol:
        Python↔Node communication uses JSON messages over stdin/stdout:
        - Node sends {"type": "request", "side": "p1"|"p2", "request_type": ..., "request": {...}}
        - Python responds with {"type": "action", "choice": "..."}
        - Node sends {"type": "result", ...} when battle completes
        
        The "result" message contains: format_id, p1_name, p2_name, winner_side,
        winner_name, turns, log, p1_team_packed, p2_team_packed, tier_name,
        trajectory (with p1/p2 steps), and meta.
        
    Persistence:
        If cfg.write_trajectories is True, each successful battle_json is passed
        to BattleStore.append_full_battle_from_json(), which saves:
        - Raw battle log to data/battles_raw/
        - Full battle record to data/datasets/full_battles/full_battles.jsonl
        - Battle trajectory to data/datasets/trajectories/trajectories.jsonl
    """
    store = BattleStore()
    router = PythonPolicyRouter(cfg=cfg, rng=random.Random(cfg.seed))

    episodes = 0  # Total attempted episodes
    errors = 0    # Failed episodes
    p1_wins = 0
    p2_wins = 0
    draws = 0

    for i in range(cfg.num_episodes):
        episodes += 1  # Count each attempt
        battle_json, error = _run_single_episode(cfg, store, router, i)
        if error:
            errors += 1
            print(f"[online-selfplay] Episode {i+1} failed: {error}", file=sys.stderr)
        elif battle_json:
            # Successful episode - persist if requested
            if cfg.write_trajectories:
                # This saves both the full battle record and the trajectory
                record = store.append_full_battle_from_json(battle_json)
            
            # Track outcome
            winner_side = battle_json.get("winner_side", "unknown")
            if winner_side == "p1":
                p1_wins += 1
            elif winner_side == "p2":
                p2_wins += 1
            else:  # "tie" or "unknown"
                draws += 1

    print(f"\nEpisodes: {episodes}")
    print(f"Errors: {errors}")
    
    # Build summary dict with stable schema
    summary: OnlineSelfPlaySummary = {
        "episodes": episodes,
        "errors": errors,
        "p1_wins": p1_wins,
        "p2_wins": p2_wins,
        "draws": draws,
    }
    
    # Validate invariants before returning
    _validate_summary(summary)
    
    return summary
