"""Battle trajectory dataset utilities for turn-level RL preparation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .config import BATTLES_JSON_DIR, DATA_ROOT


TRAJECTORY_DATASET_DIR = DATA_ROOT / "datasets" / "battle_trajectories"
TRAJECTORY_JSONL_PATH = TRAJECTORY_DATASET_DIR / "battle_trajectories.jsonl"


@dataclass
class BattleStep:
    """A single step in a battle trajectory."""

    battle_id: str
    turn_index: int

    # High-level features
    p1_active_species: List[str]  # Currently active Pokémon species names
    p2_active_species: List[str]
    p1_team_species: List[str]  # All Pokémon in team (from team preview)
    p2_team_species: List[str]

    weather: Optional[str] = None
    terrain: Optional[str] = None

    # Actions taken this turn (string-encoded for now)
    p1_actions: List[str] = field(default_factory=list)
    p2_actions: List[str] = field(default_factory=list)

    # Final outcome info (only filled at last step, or repeated)
    winner: Optional[str] = None  # "p1", "p2", or None (tie)


def iter_battle_json_files(battles_json_dir: Path) -> Iterable[Path]:
    """
    Yield all JSON battle files (one per battle) from the given directory.

    Args:
        battles_json_dir: Directory containing battle JSON files.

    Yields:
        Path objects to JSON files.
    """
    if not battles_json_dir.exists():
        return

    for json_file in battles_json_dir.glob("*.json"):
        if json_file.is_file():
            yield json_file


def parse_battle_json(path: Path) -> List[BattleStep]:
    """
    Parse a single battle JSON (metadata) and its associated raw log into BattleStep list.

    The JSON file contains metadata including raw_log_path.
    We read the raw log to extract turn-by-turn information.

    Args:
        path: Path to battle JSON file.

    Returns:
        List of BattleStep instances for this battle.

    Raises:
        FileNotFoundError: If raw log file is missing.
        ValueError: If battle structure cannot be parsed.
    """
    # Load battle metadata
    with path.open("r", encoding="utf-8") as f:
        battle_data = json.load(f)

    battle_id = path.stem  # e.g., "20251209_233407_296_gen9vgc2026regf"
    format_id = battle_data.get("format_id", "unknown")
    winner = battle_data.get("winner", "unknown")
    if winner == "unknown":
        winner = battle_data.get("winner_side", "unknown")

    # Read raw log file
    raw_log_path_str = battle_data.get("raw_log_path", "")
    if not raw_log_path_str:
        return []  # Skip if no log path

    # Handle path - could be absolute or relative
    raw_log_path = Path(raw_log_path_str.strip())  # Strip any whitespace
    
    if not raw_log_path.exists():
        # Try extracting filename and looking in battles_raw directory
        log_filename = raw_log_path.name
        # Try multiple possible locations
        possible_paths = [
            DATA_ROOT / "battles_raw" / log_filename,
            Path(raw_log_path_str.strip()),
        ]
        
        found = False
        for possible_path in possible_paths:
            try:
                if possible_path.exists():
                    raw_log_path = possible_path
                    found = True
                    break
            except (OSError, ValueError):
                continue
        
        if not found:
            return []  # Skip if log not found

    log_text = raw_log_path.read_text(encoding="utf-8")

    # Extract team information from team preview
    p1_team_species: List[str] = []
    p2_team_species: List[str] = []

    # Parse |poke| lines for initial team
    for line in log_text.splitlines():
        if line.startswith("|poke|p1|"):
            # Format: |poke|p1|Species, L50| or |poke|p1|Species, L50, M|
            parts = line.split("|")
            if len(parts) >= 4:
                species_part = parts[3].split(",")[0].strip()
                if species_part and species_part not in p1_team_species:
                    p1_team_species.append(species_part)
        elif line.startswith("|poke|p2|"):
            parts = line.split("|")
            if len(parts) >= 4:
                species_part = parts[3].split(",")[0].strip()
                if species_part and species_part not in p2_team_species:
                    p2_team_species.append(species_part)

    # Parse turns
    steps: List[BattleStep] = []
    lines = log_text.splitlines()

    current_turn = 0
    current_weather: Optional[str] = None
    current_terrain: Optional[str] = None
    p1_active: List[str] = []
    p2_active: List[str] = []
    p1_actions_turn: List[str] = []
    p2_actions_turn: List[str] = []

    battle_started = False

    for i, line in enumerate(lines):
        line = line.strip()

        # Detect battle start
        if line == "|start" and not battle_started:
            battle_started = True
            # Extract initial active Pokémon from switches right after |start|
            continue

        # Track turns
        if line.startswith("|turn|"):
            # Save previous turn if any
            if current_turn > 0 and battle_started:
                step = BattleStep(
                    battle_id=battle_id,
                    turn_index=current_turn,
                    p1_active_species=list(p1_active),
                    p2_active_species=list(p2_active),
                    p1_team_species=p1_team_species.copy() if p1_team_species else [],
                    p2_team_species=p2_team_species.copy() if p2_team_species else [],
                    weather=current_weather,
                    terrain=current_terrain,
                    p1_actions=list(p1_actions_turn),
                    p2_actions=list(p2_actions_turn),
                    winner=None,  # Will fill at end
                )
                steps.append(step)

            # Start new turn
            try:
                current_turn = int(line[6:].strip())
                p1_actions_turn = []
                p2_actions_turn = []
            except ValueError:
                continue
            continue

        # Track active Pokémon (switches) - update active list
        if line.startswith("|switch|") or line.startswith("|drag|"):
            # Format: |switch|p1a: Species|Species, L50|...
            parts = line.split("|")
            if len(parts) >= 3:
                slot_full = parts[2]  # e.g., "p1a: Umbreon"
                species_part = parts[3].split(",")[0].strip() if len(parts) >= 4 else ""
                
                # Extract slot identifier (p1a or p1b)
                slot_base = slot_full.split(":")[0].strip() if ":" in slot_full else slot_full.strip()
                
                if slot_base.startswith("p1"):
                    slot_idx = 0 if slot_base.endswith("a") else (1 if slot_base.endswith("b") else 0)
                    while len(p1_active) <= slot_idx:
                        p1_active.append("")
                    p1_active[slot_idx] = species_part
                    # Keep only active slots (max 2 for doubles)
                    if len(p1_active) > 2:
                        p1_active = p1_active[:2]
                elif slot_base.startswith("p2"):
                    slot_idx = 0 if slot_base.endswith("a") else (1 if slot_base.endswith("b") else 0)
                    while len(p2_active) <= slot_idx:
                        p2_active.append("")
                    p2_active[slot_idx] = species_part
                    # Keep only active slots (max 2 for doubles)
                    if len(p2_active) > 2:
                        p2_active = p2_active[:2]

        # Track moves
        if line.startswith("|move|"):
            # Format: |move|p1a: Species|Move Name|...
            parts = line.split("|")
            if len(parts) >= 4:
                slot = parts[2].split(":")[0]
                move = parts[3].strip()
                if slot.startswith("p1"):
                    p1_actions_turn.append(f"move:{move}")
                elif slot.startswith("p2"):
                    p2_actions_turn.append(f"move:{move}")

        # Track switches (as actions)
        if line.startswith("|switch|") or line.startswith("|drag|"):
            parts = line.split("|")
            if len(parts) >= 3:
                slot = parts[2].split(":")[0]
                species = parts[3].split(",")[0].strip() if len(parts) >= 4 else ""
                if slot.startswith("p1"):
                    p1_actions_turn.append(f"switch:{species}")
                elif slot.startswith("p2"):
                    p2_actions_turn.append(f"switch:{species}")

        # Track weather
        if line.startswith("|weather|"):
            parts = line.split("|")
            if len(parts) >= 3:
                current_weather = parts[2].strip()

        # Track terrain
        if line.startswith("|fieldstart|") and "terrain" in line.lower():
            parts = line.split("|")
            if len(parts) >= 3:
                terrain_part = parts[2].strip()
                # Extract terrain type (e.g., "mistyterrain" -> "Misty")
                current_terrain = terrain_part.replace("terrain", "").capitalize()

    # Add final step with winner
    if current_turn > 0 and battle_started:
        final_winner = None
        if winner == "p1":
            final_winner = "p1"
        elif winner == "p2":
            final_winner = "p2"
        elif winner == "tie":
            final_winner = None

        step = BattleStep(
            battle_id=battle_id,
            turn_index=current_turn,
            p1_active_species=list(p1_active),
            p2_active_species=list(p2_active),
            p1_team_species=p1_team_species.copy() if p1_team_species else [],
            p2_team_species=p2_team_species.copy() if p2_team_species else [],
            weather=current_weather,
            terrain=current_terrain,
            p1_actions=list(p1_actions_turn),
            p2_actions=list(p2_actions_turn),
            winner=final_winner,
        )
        steps.append(step)

    # Also add winner to all steps (or at least the last one)
    if steps:
        steps[-1].winner = final_winner

    return steps


def battle_step_to_dict(step: BattleStep) -> Dict[str, object]:
    """
    Convert BattleStep to a JSON-serializable dictionary.

    Args:
        step: BattleStep instance.

    Returns:
        Dictionary ready for json.dumps.
    """
    return {
        "battle_id": step.battle_id,
        "turn_index": step.turn_index,
        "p1_active_species": step.p1_active_species,
        "p2_active_species": step.p2_active_species,
        "p1_team_species": step.p1_team_species,
        "p2_team_species": step.p2_team_species,
        "weather": step.weather,
        "terrain": step.terrain,
        "p1_actions": step.p1_actions,
        "p2_actions": step.p2_actions,
        "winner": step.winner,
    }


def dump_trajectories_to_jsonl(
    battles_json_dir: Path,
    out_path: Path,
) -> tuple[int, int]:
    """
    Iterate over all JSON battle files, parse BattleStep lists, and write to JSONL.

    Args:
        battles_json_dir: Directory containing battle JSON files.
        out_path: Path to output JSONL file.

    Returns:
        Tuple of (n_battles_processed, n_steps_written).
    """
    # Ensure output directory exists
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_battles_processed = 0
    n_steps_written = 0
    n_battles_skipped = 0

    with out_path.open("w", encoding="utf-8") as out_file:
        for json_file in iter_battle_json_files(battles_json_dir):
            try:
                steps = parse_battle_json(json_file)
                if not steps:
                    n_battles_skipped += 1
                    continue

                for step in steps:
                    data = battle_step_to_dict(step)
                    json_line = json.dumps(data, ensure_ascii=False)
                    out_file.write(json_line + "\n")
                    n_steps_written += 1

                n_battles_processed += 1
            except Exception as e:
                # Skip on error but don't print (to avoid noise)
                n_battles_skipped += 1
                continue

    return n_battles_processed, n_steps_written

