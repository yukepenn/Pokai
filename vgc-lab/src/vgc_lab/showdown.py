"""Showdown bridge: CLI commands, random bot, lineup extraction, and team preview.

This module consolidates:
- showdown_cli.py: Wrapper around pokemon-showdown CLI commands
- random_bot.py: Random bot utilities for making battle choices
- lineup.py: Lineup extraction from battle logs
- team_preview.py: Team preview and open team sheet models and parsing
"""

from __future__ import annotations

import json
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

from .config import SHOWDOWN_ROOT, DEFAULT_FORMAT, PROJECT_ROOT

# ============================================================================
# Random Bot Functions
# ============================================================================


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


# ============================================================================
# Lineup Extraction
# ============================================================================


def extract_lineups_from_log(
    log_text: str,
    p1_team_public: List[Dict],
    p2_team_public: List[Dict],
) -> Tuple[List[int], List[int], List[int], List[int]]:
    """
    Parse a Showdown battle log and determine, for each side:

    - Which 4 Pokémon from the 6 in team_public were actually brought.
    - Which 2 were the initial leads.

    Args:
        log_text: Full battle log text
        p1_team_public: List of dicts with public team info for p1
        p2_team_public: List of dicts with public team info for p2

    Returns:
        Tuple of (p1_chosen_indices, p2_chosen_indices, p1_lead_indices, p2_lead_indices)
        All indices are 0-based into the corresponding team_public list.
    """
    # Build species -> index mappings
    p1_species_to_idx: Dict[str, int] = {}
    p2_species_to_idx: Dict[str, int] = {}

    for idx, pokemon in enumerate(p1_team_public):
        species = pokemon.get("species", "")
        if species:
            p1_species_to_idx[species] = idx

    for idx, pokemon in enumerate(p2_team_public):
        species = pokemon.get("species", "")
        if species:
            p2_species_to_idx[species] = idx

    # Track chosen and leads
    p1_chosen_set: set[int] = set()
    p2_chosen_set: set[int] = set()
    p1_lead_indices: List[int] = []
    p2_lead_indices: List[int] = []

    battle_started = False
    lines = log_text.splitlines()

    for line in lines:
        # Mark when battle starts
        if line.strip() == "|start":
            battle_started = True
            continue

        if not battle_started:
            continue

        # Parse switch/drag lines
        if line.startswith("|switch|") or line.startswith("|drag|"):
            parts = line.split("|")
            if len(parts) < 3:
                continue

            # Extract side (p1a, p1b, p2a, p2b, etc.)
            switch_part = parts[2]
            if not switch_part.startswith(("p1", "p2")):
                continue

            is_p1 = switch_part.startswith("p1")
            side_idx = 1 if is_p1 else 2

            # Extract species from the details part (after the next |)
            if len(parts) < 4:
                continue

            details = parts[3]
            # Format: "Species, L50, M" or "Species, L50" or "Necrozma-Dawn-Wings, L50"
            # Extract species name (everything before the first comma)
            species_name = details.split(",")[0].strip()

            # Map to index
            species_map = p1_species_to_idx if is_p1 else p2_species_to_idx
            chosen_set = p1_chosen_set if is_p1 else p2_chosen_set
            lead_list = p1_lead_indices if is_p1 else p2_lead_indices

            if species_name in species_map:
                idx = species_map[species_name]
                chosen_set.add(idx)

                # Track leads (first 2 switches per side after |start)
                # Only count |switch| lines, not |drag| lines for leads
                if line.startswith("|switch|") and len(lead_list) < 2:
                    lead_list.append(idx)
            else:
                # Species not found - log a warning (but don't crash)
                print(
                    f"[WARN] Could not map species '{species_name}' to team_public for {'p1' if is_p1 else 'p2'}"
                )

    # Convert sets to sorted lists for consistency
    p1_chosen_indices = sorted(list(p1_chosen_set))
    p2_chosen_indices = sorted(list(p2_chosen_set))

    # Limit to 4 (VGC format)
    if len(p1_chosen_indices) > 4:
        print(f"[WARN] p1 has more than 4 chosen Pokémon: {p1_chosen_indices}")
        p1_chosen_indices = p1_chosen_indices[:4]
    if len(p2_chosen_indices) > 4:
        print(f"[WARN] p2 has more than 4 chosen Pokémon: {p2_chosen_indices}")
        p2_chosen_indices = p2_chosen_indices[:4]

    return (
        p1_chosen_indices,
        p2_chosen_indices,
        p1_lead_indices,
        p2_lead_indices,
    )


# ============================================================================
# Team Preview Models and Functions
# ============================================================================

# Dataset location
DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "team_preview"
DATASET_PATH = DATASET_DIR / "team_preview.jsonl"

PREVIEW_OUTCOME_DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "preview_outcome"
PREVIEW_OUTCOME_DATASET_PATH = PREVIEW_OUTCOME_DATASET_DIR / "preview_outcome.jsonl"


class PokemonPublicInfo(BaseModel):
    """Public information about a single Pokémon at team preview."""

    ident: str = Field(..., description="Identifier, e.g., 'p1: Cinderace'")
    details: str = Field(..., description="Details string, e.g., 'Cinderace, L50, M'")
    name: str = Field(..., description="Species name, e.g., 'Cinderace'")
    item: Optional[str] = Field(None, description="Item name or None")
    ability: Optional[str] = Field(None, description="Ability name or None")
    moves: list[str] = Field(default_factory=list, description="List of move IDs/names")
    tera_type: Optional[str] = Field(None, description="Tera type, e.g., 'Fighting'")
    raw_stats: Optional[dict[str, int]] = Field(None, description="Raw stats dict if available")


class SideTeamPreview(BaseModel):
    """Team preview information for one side."""

    side_id: str = Field(..., description="Side ID: 'p1' or 'p2'")
    player_name: str = Field(..., description="Player name")
    max_chosen_team_size: int = Field(..., description="Maximum chosen team size (usually 4 for VGC)")
    pokemon: list[PokemonPublicInfo] = Field(..., description="List of 6 Pokémon")


class TeamPreviewSnapshot(BaseModel):
    """Complete team preview snapshot for a battle."""

    format_id: str = Field(..., description="Format ID, e.g., 'gen9vgc2026regf'")
    tier_name: str = Field(..., description="Tier name, e.g., '[Gen 9] VGC 2026 Reg F'")
    p1_preview: SideTeamPreview
    p2_preview: SideTeamPreview
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_log_path: Optional[Path] = Field(None, description="Path to source log file")
    meta: dict[str, Any] = Field(default_factory=dict, description="Optional metadata")

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat(),
        }


class PreviewOutcomeRecord(BaseModel):
    """
    One open-team-sheet snapshot plus battle outcome.

    Teams are stored as raw dicts coming from the Node script, so they can evolve
    without breaking older data (they live under p1_team_public / p2_team_public).
    """

    format_id: str = Field(..., description="Format ID (e.g., gen9vgc2026regf)")
    p1_name: str
    p2_name: str

    # Public team info from Node (list of dicts as emitted by js/random_selfplay.js)
    p1_team_public: List[Dict[str, Any]]
    p2_team_public: List[Dict[str, Any]]

    # Indices into p1_team_public / p2_team_public
    p1_chosen_indices: List[int] = Field(default_factory=list)
    p2_chosen_indices: List[int] = Field(default_factory=list)
    p1_lead_indices: List[int] = Field(default_factory=list)
    p2_lead_indices: List[int] = Field(default_factory=list)

    winner_side: str = Field(..., description='"p1" / "p2" / "tie" / "unknown"')
    winner_name: Optional[str]

    raw_log_path: Path
    created_at: datetime

    meta: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat(),
        }


def ensure_dataset_dir() -> None:
    """Ensure the dataset directory exists."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


def ensure_preview_outcome_dataset_dir() -> None:
    """Ensure the preview+outcome dataset directory exists."""
    PREVIEW_OUTCOME_DATASET_DIR.mkdir(parents=True, exist_ok=True)


def extract_team_preview_requests(log_text: str) -> dict[str, dict]:
    """
    Parse the raw battle log and return a mapping:
        {"p1": request_json_for_p1, "p2": request_json_for_p2}
    Only include requests where teamPreview is true.

    Args:
        log_text: Full battle log text

    Returns:
        Dictionary mapping side_id to request JSON dict

    Raises:
        ValueError: If expected teamPreview requests are missing
    """
    lines = log_text.splitlines()
    requests: dict[str, dict] = {}
    current_side: Optional[str] = None
    i = 0

    while i < len(lines):
        line = lines[i].strip()

        if line == "sideupdate":
            # Next non-empty line should be side ID
            i += 1
            while i < len(lines) and not lines[i].strip():
                i += 1
            if i >= len(lines):
                break
            current_side = lines[i].strip()
            i += 1
            continue

        if line.startswith("|request|") and current_side:
            try:
                req_json_str = line[len("|request|"):]
                request_dict = json.loads(req_json_str)

                # Only capture teamPreview requests
                if request_dict.get("teamPreview") is True:
                    requests[current_side] = request_dict

                current_side = None  # Reset after processing
            except (json.JSONDecodeError, KeyError):
                # Skip malformed requests
                current_side = None

        i += 1

    return requests


def _parse_pokemon_name(details: str) -> str:
    """
    Extract Pokémon name from details string.
    Examples:
        "Cinderace, L50, M" -> "Cinderace"
        "Ogerpon-Cornerstone, L50, F" -> "Ogerpon-Cornerstone"
    """
    # Split on comma and take first part
    parts = details.split(",")
    return parts[0].strip()


def build_side_team_preview(side_id: str, request_json: dict) -> SideTeamPreview:
    """
    Convert one side's request JSON (with teamPreview true) into a SideTeamPreview.

    Args:
        side_id: Side ID ("p1" or "p2")
        request_json: Request JSON dict with teamPreview=true

    Returns:
        SideTeamPreview instance
    """
    side_data = request_json.get("side", {})
    player_name = side_data.get("name", "")
    max_chosen_team_size = request_json.get("maxChosenTeamSize", 4)

    pokemon_list: list[PokemonPublicInfo] = []
    raw_pokemon = side_data.get("pokemon", [])

    for pokemon_data in raw_pokemon:
        details = pokemon_data.get("details", "")
        name = _parse_pokemon_name(details)

        # Extract moves
        moves = pokemon_data.get("moves", [])
        if isinstance(moves, list):
            moves = [str(m) if not isinstance(m, dict) else str(m.get("id", m)) for m in moves]

        # Extract stats (optional)
        raw_stats = pokemon_data.get("stats")
        if isinstance(raw_stats, dict):
            # Convert to int if possible
            stats_int: dict[str, int] = {}
            for key, value in raw_stats.items():
                try:
                    stats_int[key] = int(value)
                except (ValueError, TypeError):
                    pass
            raw_stats = stats_int if stats_int else None

        pokemon_info = PokemonPublicInfo(
            ident=pokemon_data.get("ident", ""),
            details=details,
            name=name,
            item=pokemon_data.get("item"),
            ability=pokemon_data.get("ability"),
            moves=moves,
            tera_type=pokemon_data.get("teraType"),
            raw_stats=raw_stats,
        )
        pokemon_list.append(pokemon_info)

    return SideTeamPreview(
        side_id=side_id,
        player_name=player_name,
        max_chosen_team_size=max_chosen_team_size,
        pokemon=pokemon_list,
    )


def parse_team_preview_snapshot(
    log_text: str,
    *,
    format_id: str | None = None,
    raw_log_path: Path | None = None,
) -> TeamPreviewSnapshot:
    """
    Parse a battle log and extract team preview snapshot.

    Args:
        log_text: Full battle log text
        format_id: Format ID (defaults to DEFAULT_FORMAT if not provided)
        raw_log_path: Optional path to source log file

    Returns:
        TeamPreviewSnapshot instance

    Raises:
        ValueError: If required teamPreview requests are missing
    """
    if format_id is None:
        format_id = DEFAULT_FORMAT

    # Extract tier name
    tier_name = format_id  # Default fallback
    for line in log_text.splitlines():
        if line.startswith("|tier|"):
            tier_name = line[len("|tier|"):].strip()
            break

    # Extract team preview requests
    requests = extract_team_preview_requests(log_text)

    if "p1" not in requests or "p2" not in requests:
        missing = [sid for sid in ["p1", "p2"] if sid not in requests]
        raise ValueError(f"Missing teamPreview requests for sides: {missing}")

    # Build side previews
    p1_preview = build_side_team_preview("p1", requests["p1"])
    p2_preview = build_side_team_preview("p2", requests["p2"])

    return TeamPreviewSnapshot(
        format_id=format_id,
        tier_name=tier_name,
        p1_preview=p1_preview,
        p2_preview=p2_preview,
        created_at=datetime.now(timezone.utc),
        raw_log_path=raw_log_path,
    )


def append_snapshot_to_dataset(snapshot: TeamPreviewSnapshot) -> None:
    """
    Append one snapshot as a JSON line to DATASET_PATH.

    Args:
        snapshot: TeamPreviewSnapshot to append
    """
    ensure_dataset_dir()

    # Convert to dict, handling Path and datetime serialization
    data = snapshot.model_dump(mode="json")
    if snapshot.raw_log_path is not None:
        data["raw_log_path"] = str(snapshot.raw_log_path)
    data["created_at"] = snapshot.created_at.isoformat()

    # Write as a single JSON line
    json_line = json.dumps(data, ensure_ascii=False)
    with DATASET_PATH.open("a", encoding="utf-8") as f:
        f.write(json_line + "\n")


def append_preview_outcome_record(record: PreviewOutcomeRecord) -> None:
    """
    Append one record as a JSON line to PREVIEW_OUTCOME_DATASET_PATH.

    Args:
        record: PreviewOutcomeRecord instance to append
    """
    ensure_preview_outcome_dataset_dir()

    data = record.model_dump()
    # Convert non-JSON types
    data["raw_log_path"] = str(data["raw_log_path"])
    data["created_at"] = data["created_at"].isoformat()

    with PREVIEW_OUTCOME_DATASET_PATH.open("a", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


# ============================================================================
# Showdown CLI Functions
# ============================================================================


def run_showdown_command(args: list[str], input_text: Optional[str] = None) -> str:
    """
    Run `node pokemon-showdown <args>` in SHOWDOWN_ROOT, optionally feeding `input_text` to stdin.

    Args:
        args: List of arguments to pass after "pokemon-showdown" (e.g., ["generate-team", "gen9ou"])
        input_text: Optional text to send to stdin

    Returns:
        stdout as a string (stripped of trailing whitespace)

    Raises:
        RuntimeError: On non-zero exit code or if `node` is not found
        FileNotFoundError: If SHOWDOWN_ROOT doesn't exist
    """
    if not SHOWDOWN_ROOT.exists():
        raise FileNotFoundError(
            f"Showdown directory not found at {SHOWDOWN_ROOT}. "
            "Ensure pokemon-showdown/ is a sibling of vgc-lab/"
        )

    cmd = ["node", "pokemon-showdown"] + args

    try:
        result = subprocess.run(
            cmd,
            cwd=str(SHOWDOWN_ROOT),
            input=input_text,
            text=True,
            capture_output=True,
            check=False,  # We'll check exit code manually
        )
    except FileNotFoundError as e:
        if "node" in str(e).lower():
            raise RuntimeError(
                "Node.js not found. Please install Node.js 16+ from https://nodejs.org/"
            ) from e
        raise

    if result.returncode != 0:
        error_msg = (
            f"Command failed: {' '.join(cmd)}\n"
            f"Exit code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )
        raise RuntimeError(error_msg)

    return result.stdout.rstrip("\n\r")


def pack_team(export_text: str) -> str:
    """
    Convert a team from export format to packed format.

    Uses: echo "EXPORT" | node pokemon-showdown pack-team

    Args:
        export_text: Team in export/teambuilder format

    Returns:
        Packed team string (trailing newline removed)
    """
    return run_showdown_command(["pack-team"], input_text=export_text)


def validate_team(packed_team: str, format_id: str = DEFAULT_FORMAT) -> str:
    """
    Validate a team against a format.

    Uses: echo "TEAM" | node pokemon-showdown validate-team <format_id>

    Args:
        packed_team: Team in packed format
        format_id: Format ID to validate against

    Returns:
        Validation output (empty string if valid, error messages if invalid)

    Note:
        Exit code 0 = valid, exit code 1 = invalid (but we return stdout in both cases)
    """
    try:
        return run_showdown_command(["validate-team", format_id], input_text=packed_team)
    except RuntimeError as e:
        # On validation failure, the command exits with code 1, but stderr contains
        # the error messages. We want to return those messages, not raise.
        # Check if it's a validation error (exit code 1) vs. a real error
        if "Exit code: 1" in str(e):
            # Extract stderr from the error message
            parts = str(e).split("STDERR:\n", 1)
            if len(parts) > 1:
                return parts[1].rstrip()
            return str(e)
        raise


def generate_random_team(format_id: str = DEFAULT_FORMAT) -> str:
    """
    Generate a random team for a format.

    Uses: node pokemon-showdown generate-team <format_id>

    Args:
        format_id: Format ID (defaults to gen9vgc2026regf)

    Returns:
        Packed team string
    """
    return run_showdown_command(["generate-team", format_id])


def simulate_battle(
    format_id: str,
    p1_name: str,
    p1_packed_team: str,
    p2_name: str,
    p2_packed_team: str,
) -> str:
    """
    Run a simulated battle via CLI.

    IMPORTANT NOTE: The `simulate-battle` command requires manual choice input
    (`>p1 move 1`, `>p2 switch 3`, etc.). This function currently only sends the
    initial setup and will wait for choices. For fully automated battles, you
    would need to:
    1. Parse the battle output to detect choice requests
    2. Generate choices (random, heuristic, or RL-based)
    3. Send choices back to the simulator

    This is a basic implementation that sets up the battle. Future versions
    should include choice-making logic for automated play.

    Protocol sent:
    ```
    >start {"formatid":"FORMAT_ID"}
    >player p1 {"name":"P1_NAME","team":"P1_TEAM"}
    >player p2 {"name":"P2_NAME","team":"P2_TEAM"}
    ```

    Args:
        format_id: Format ID (e.g., "gen9vgc2026regf")
        p1_name: Player 1 name
        p1_packed_team: Player 1 team in packed format
        p2_name: Player 2 name
        p2_packed_team: Player 2 team in packed format

    Returns:
        Full battle log output (may be incomplete if choices are required)

    TODO:
        Implement automatic choice-making based on battle stream parsing.
        For now, this will hang or fail if choices are required.
    """
    # Build the battle input protocol
    start_cmd = f'>start {json.dumps({"formatid": format_id})}'
    p1_cmd = f'>player p1 {json.dumps({"name": p1_name, "team": p1_packed_team})}'
    p2_cmd = f'>player p2 {json.dumps({"name": p2_name, "team": p2_packed_team})}'

    input_text = f"{start_cmd}\n{p1_cmd}\n{p2_cmd}\n"

    return run_showdown_command(["simulate-battle"], input_text=input_text)


def play_battle_with_random_bots(
    format_id: str,
    p1_name: str,
    p1_packed_team: str,
    p2_name: str,
    p2_packed_team: str,
    *,
    max_requests: int = 512,
    max_turns: int = 200,
    rng: Optional[random.Random] = None,
) -> Tuple[str, str, Optional[int]]:
    """Run a full battle using Showdown's `simulate-battle` and two random bots.

    This function:
      * Spawns `node pokemon-showdown simulate-battle` in SHOWDOWN_ROOT.
      * Sends `>start`, `>player p1`, `>player p2`.
      * For every `sideupdate` + `|request|` pair, calls `choose_action` to
        generate a command and sends it back to the simulator.
      * Tracks `|turn|N` lines to estimate the number of turns.
      * Detects `|win|` / `|tie|` to determine a winner.
      * Enforces `max_requests` and `max_turns` to avoid hangs.

    Args:
        format_id: Format ID (e.g., "gen9vgc2026regf")
        p1_name: Player 1 name
        p1_packed_team: Player 1 team in packed format
        p2_name: Player 2 name
        p2_packed_team: Player 2 team in packed format
        max_requests: Maximum number of requests to process (safety cap)
        max_turns: Maximum number of turns (safety cap)
        rng: Optional random number generator (for testing). If None, creates a new one.

    Returns:
        log_text: Full stdout from simulate-battle.
        winner_side: "p1", "p2", "tie", or "unknown".
        turns: The last observed turn number, or None if not found.
    """
    if not SHOWDOWN_ROOT.exists():
        raise RuntimeError(f"Showdown directory not found: {SHOWDOWN_ROOT}")

    cmd = ["node", "pokemon-showdown", "simulate-battle"]
    proc = subprocess.Popen(
        cmd,
        cwd=SHOWDOWN_ROOT,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    if proc.stdin is None or proc.stdout is None:
        raise RuntimeError("Failed to start simulate-battle with proper pipes")

    if rng is None:
        rng = random.Random()

    log_lines: list[str] = []

    def send(command: str) -> None:
        """Send a single command line to the simulator and flush immediately."""
        if proc.stdin is None:
            return
        proc.stdin.write(command + "\n")
        proc.stdin.flush()

    # Initial setup: start + players
    start_payload = json.dumps({"formatid": format_id})
    p1_payload = json.dumps({"name": p1_name, "team": p1_packed_team})
    p2_payload = json.dumps({"name": p2_name, "team": p2_packed_team})

    send(f">start {start_payload}")
    send(f">player p1 {p1_payload}")
    send(f">player p2 {p2_payload}")

    winner_side: str = "unknown"
    turns: Optional[int] = None
    requests_seen = 0

    # Main read loop: read stdout line by line, respond to `sideupdate` blocks.
    try:
        while True:
            raw_line = proc.stdout.readline()
            # If we got EOF and the process has exited, stop.
            if raw_line == "" and proc.poll() is not None:
                break
            if raw_line == "":
                # Process is still alive but not producing output.
                # To avoid a hard hang, stop after this condition:
                break

            line = raw_line.rstrip("\n")
            log_lines.append(line)

            # Turn counting
            if line.startswith("|turn|"):
                parts = line.split("|")
                if len(parts) >= 3:
                    try:
                        turns = int(parts[2])
                    except ValueError:
                        pass

            # Winner detection
            if line.startswith("|win|"):
                # Example: "|win|Bot1"
                winner_name = line.split("|", 2)[2] if "|" in line else ""
                if winner_name == p1_name:
                    winner_side = "p1"
                elif winner_name == p2_name:
                    winner_side = "p2"
                else:
                    winner_side = "unknown"
                break
            if line == "|tie|" or line.startswith("|tie|"):
                winner_side = "tie"
                break

            # sideupdate block: next two lines are <sideid> and "|request|..."
            if line == "sideupdate":
                side_id_line = proc.stdout.readline()
                if side_id_line == "" and proc.poll() is not None:
                    break
                side_id = side_id_line.rstrip("\n")
                log_lines.append(side_id)

                req_line = proc.stdout.readline()
                if req_line == "" and proc.poll() is not None:
                    break
                req_line = req_line.rstrip("\n")
                log_lines.append(req_line)

                if not req_line.startswith("|request|"):
                    # No request payload here, nothing to do.
                    continue

                try:
                    req_json = req_line.split("|request|", 1)[1]
                    request = json.loads(req_json)
                except Exception:
                    # If we can't parse the request, skip making a choice.
                    continue

                requests_seen += 1
                if requests_seen > max_requests:
                    # Safety cap: stop feeding choices; we'll bail out below.
                    break

                choice = choose_action(request, rng=rng)
                send(f">{side_id} {choice}")

                # Also enforce a turn-based safety cap.
                if turns is not None and turns >= max_turns:
                    break

        # After breaking out of the loop, drain remaining stdout/stderr briefly.
        try:
            stdout_rest, stderr_rest = proc.communicate(timeout=0.5)
        except Exception:
            stdout_rest, stderr_rest = "", ""

        if stdout_rest:
            for extra_line in stdout_rest.splitlines():
                log_lines.append(extra_line)
        if stderr_rest:
            # Include stderr in the log to help debugging.
            for extra_line in stderr_rest.splitlines():
                log_lines.append(f"[stderr] {extra_line}")

    finally:
        # Ensure the process is not left running.
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

    log_text = "\n".join(log_lines)
    return log_text, winner_side, turns


def run_random_selfplay_json(
    format_id: str = DEFAULT_FORMAT,
    p1_name: str = "Bot1",
    p2_name: str = "Bot2",
    p1_packed_team: Optional[str] = None,
    p2_packed_team: Optional[str] = None,
    timeout_seconds: int = 60,
) -> Dict[str, Any]:
    """
    Call `node js/random_selfplay.js` and return the parsed JSON object.

    The JSON object contains:
      - format_id, p1_name, p2_name
      - winner_side, winner_name
      - turns
      - log
      - p1_team_packed, p2_team_packed
      - p1_team_public, p2_team_public

    Args:
        format_id: Format ID (defaults to DEFAULT_FORMAT)
        p1_name: Player 1 name (default: "Bot1")
        p2_name: Player 2 name (default: "Bot2")
        p1_packed_team: Optional packed team string for p1 (if None, generates random)
        p2_packed_team: Optional packed team string for p2 (if None, generates random)
        timeout_seconds: Timeout for subprocess execution (default: 60)

    Returns:
        Dictionary containing all battle data from the Node script

    Raises:
        RuntimeError: If the Node script fails or times out
    """
    cmd = [
        "node",
        "js/random_selfplay.js",
        "--format",
        format_id,
        "--p1-name",
        p1_name,
        "--p2-name",
        p2_name,
    ]

    if p1_packed_team is not None:
        cmd.extend(["--p1-team", p1_packed_team])
    if p2_packed_team is not None:
        cmd.extend(["--p2-team", p2_packed_team])

    try:
        result = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,  # We'll check return code manually
        )
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"random_selfplay timed out after {timeout_seconds} seconds")
    except FileNotFoundError as e:
        if "node" in str(e).lower():
            raise RuntimeError(
                "Node.js not found. Please install Node.js 16+ from https://nodejs.org/"
            ) from e
        raise

    if result.returncode != 0:
        error_msg = result.stderr.strip() if result.stderr else "Unknown error"
        raise RuntimeError(f"random_selfplay failed: {error_msg}")

    # Parse JSON output
    try:
        data = json.loads(result.stdout.strip())
    except json.JSONDecodeError as e:
        raise RuntimeError(f"Failed to parse JSON output from random_selfplay: {e}")

    return data


def run_random_selfplay(
    format_id: str = DEFAULT_FORMAT,
    p1_name: str = "Bot1",
    p2_name: str = "Bot2",
    p1_packed_team: Optional[str] = None,
    p2_packed_team: Optional[str] = None,
    timeout_seconds: int = 60,
) -> Tuple[str, str, Optional[int], str, str]:
    """
    Call the Node random selfplay script and return battle results.

    Uses Showdown's internal BattleStream + RandomPlayerAI to run a complete battle.

    This is a thin wrapper around run_random_selfplay_json() that extracts
    the legacy tuple return format for backwards compatibility.

    Args:
        format_id: Format ID (defaults to DEFAULT_FORMAT)
        p1_name: Player 1 name (default: "Bot1")
        p2_name: Player 2 name (default: "Bot2")
        p1_packed_team: Optional packed team string for p1 (if None, generates random)
        p2_packed_team: Optional packed team string for p2 (if None, generates random)
        timeout_seconds: Timeout for subprocess execution (default: 60)

    Returns:
        Tuple of (log_text, winner_side, turns, p1_name, p2_name):
        - log_text: Full battle log as string
        - winner_side: "p1", "p2", "tie", or "unknown"
        - turns: Number of turns (int or None)
        - p1_name: Player 1 name
        - p2_name: Player 2 name

    Raises:
        RuntimeError: If the Node script fails or times out
    """
    data = run_random_selfplay_json(
        format_id=format_id,
        p1_name=p1_name,
        p2_name=p2_name,
        p1_packed_team=p1_packed_team,
        p2_packed_team=p2_packed_team,
        timeout_seconds=timeout_seconds,
    )

    log_text = data.get("log", "")
    winner_side = data.get("winner_side", "unknown")
    turns = data.get("turns")
    p1_name_resp = data.get("p1_name", p1_name)
    p2_name_resp = data.get("p2_name", p2_name)

    return log_text, winner_side, turns, p1_name_resp, p2_name_resp
