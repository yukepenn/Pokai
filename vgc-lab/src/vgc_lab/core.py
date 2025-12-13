"""
Single core module for vgc-lab: config, Showdown client, battle logging, datasets, and team preview.

This module consolidates all functionality from config.py, showdown_cli.py, random_bot.py,
battle_logger.py, and team_preview.py into a single, coherent module.
"""

from __future__ import annotations

import json
import random
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

# Type-safe sanitization reason values
SanitizeReason = Literal[
    "ok",
    "fixed_pass",
    "fixed_disabled_move",
    "fixed_switch_to_move",
    "fallback_switch",
    "fallback_pass",
]

from pydantic import BaseModel, Field

# ============================================================================
# 1. Paths / Config
# ============================================================================

PROJECT_ROOT: Path = Path(__file__).parent.parent.parent.resolve()
SHOWDOWN_ROOT: Path = PROJECT_ROOT.parent / "pokemon-showdown"
DEFAULT_FORMAT: str = "gen9vgc2026regf"
DEFAULT_BO3_FORMAT: str = "gen9vgc2026regfbo3"


@dataclass
class Paths:
    """Centralized path configuration."""

    data_root: Path
    battles_raw: Path
    battles_json: Path
    datasets_root: Path
    full_battles: Path
    team_preview: Path
    trajectories: Path
    team_build: Path

    @property
    def full_battles_jsonl(self) -> Path:
        """Path to full_battles.jsonl file."""
        return self.full_battles / "full_battles.jsonl"

    @property
    def team_preview_jsonl(self) -> Path:
        """Path to team_preview.jsonl file."""
        return self.team_preview / "team_preview.jsonl"

    @property
    def trajectories_jsonl(self) -> Path:
        """Path to trajectories.jsonl file."""
        return self.trajectories / "trajectories.jsonl"

    @property
    def team_build_jsonl(self) -> Path:
        """Path to episodes.jsonl file."""
        return self.team_build / "episodes.jsonl"


def get_paths() -> Paths:
    """Get the default Paths configuration."""
    data_root = PROJECT_ROOT / "data"
    return Paths(
        data_root=data_root,
        battles_raw=data_root / "battles_raw",
        battles_json=data_root / "battles_json",
        datasets_root=data_root / "datasets",
        full_battles=data_root / "datasets" / "full_battles",
        team_preview=data_root / "datasets" / "team_preview",
        trajectories=data_root / "datasets" / "trajectories",
        team_build=data_root / "datasets" / "team_build",
    )


def ensure_paths(paths: Optional[Paths] = None) -> None:
    """Create all necessary directories if they don't exist."""
    if paths is None:
        paths = get_paths()

    paths.battles_raw.mkdir(parents=True, exist_ok=True)
    paths.battles_json.mkdir(parents=True, exist_ok=True)
    paths.full_battles.mkdir(parents=True, exist_ok=True)
    paths.team_preview.mkdir(parents=True, exist_ok=True)
    paths.trajectories.mkdir(parents=True, exist_ok=True)
    paths.team_build.mkdir(parents=True, exist_ok=True)


# ============================================================================
# 2. Data Models
# ============================================================================


@dataclass
class FullBattleRecord:
    """Full battle record for completed battles (matches existing JSONL schema)."""

    battle_id: str
    format_id: str
    p1_name: str
    p2_name: str
    winner_side: str  # "p1", "p2", "tie", or "unknown"
    winner_name: Optional[str]
    turns: Optional[int]
    raw_log_path: Path
    created_at: datetime
    meta: Dict[str, Any] = field(default_factory=dict)


class PokemonPublicInfo(BaseModel):
    """Public information about a single Pokémon at team preview."""

    ident: str = Field(..., description="Identifier, e.g., 'p1: Cinderace'")
    details: str = Field(..., description="Details string, e.g., 'Cinderace, L50, M'")
    name: str = Field(..., description="Species name, e.g., 'Cinderace'")
    item: Optional[str] = Field(None, description="Item name or None")
    ability: Optional[str] = Field(None, description="Ability name or None")
    moves: List[str] = Field(default_factory=list, description="List of move IDs/names")
    tera_type: Optional[str] = Field(None, description="Tera type, e.g., 'Fighting'")
    raw_stats: Optional[Dict[str, int]] = Field(None, description="Raw stats dict if available")


class SideTeamPreview(BaseModel):
    """Team preview information for one side."""

    side_id: str = Field(..., description="Side ID: 'p1' or 'p2'")
    player_name: str = Field(..., description="Player name")
    max_chosen_team_size: int = Field(..., description="Maximum chosen team size (usually 4 for VGC)")
    pokemon: List[PokemonPublicInfo] = Field(..., description="List of 6 Pokémon")


class TeamPreviewSnapshot(BaseModel):
    """Complete team preview snapshot for a battle."""

    battle_id: str = Field(..., description="Unique battle id that matches logs and datasets")
    format_id: str = Field(..., description="Format ID, e.g., 'gen9vgc2026regf'")
    tier_name: str = Field(..., description="Tier name, e.g., '[Gen 9] VGC 2026 Reg F'")
    p1_preview: SideTeamPreview
    p2_preview: SideTeamPreview
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    raw_log_path: Optional[Path] = Field(None, description="Path to source log file")
    meta: Dict[str, Any] = Field(default_factory=dict, description="Optional metadata")

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat(),
        }


class BattleStep(BaseModel):
    """Single request/response step for one side (p1 or p2)."""

    side: str = Field(..., description="Side ID: 'p1' or 'p2'")
    step_index: int = Field(..., description="Monotone index per side, starting from 0")
    request_type: str = Field(
        ...,
        description="High-level type: 'preview', 'move', 'force-switch', 'wait', etc.",
    )
    rqid: Optional[int] = Field(
        None,
        description="Showdown request id (rqid) if present",
    )
    turn: Optional[int] = Field(
        None,
        description="Battle turn number if available on the request",
    )
    request: Dict[str, Any] = Field(
        ...,
        description="Raw Showdown request JSON snapshot (already JSON-serializable)",
    )
    choice: str = Field(
        ...,
        description="Choice string sent to Showdown, e.g. 'team 1234' or 'move 1 2, switch 3'",
    )
    sanitize_reason: Optional[SanitizeReason] = Field(
        None,
        description="Reason for sanitization applied to this slot: 'ok', 'fixed_pass', "
        "'fixed_disabled_move', 'fixed_switch_to_move', 'fallback_switch', 'fallback_pass'",
    )


class BattleTrajectory(BaseModel):
    """
    Full request/response trajectory for a battle, suitable for RL / BC.

    This is a higher-level dataset than FullBattleRecord:
    - FullBattleRecord: summary (who won, turns, where log is)
    - BattleTrajectory: per-request steps for both sides (p1/p2),
      plus a stable battle_id and episodic rewards per side.
    """

    battle_id: str = Field(..., description="Unique battle id tying together logs and datasets")
    format_id: str = Field(..., description="Format ID, e.g. 'gen9vgc2026regf'")
    p1_name: str = Field(..., description="Player 1 name")
    p2_name: str = Field(..., description="Player 2 name")
    winner_side: str = Field(
        ...,
        description="Winner side: 'p1', 'p2', 'tie', or 'unknown'",
    )
    winner_name: Optional[str] = Field(
        None,
        description="Winner name if known, else None",
    )
    turns: Optional[int] = Field(
        None,
        description="Number of turns if available",
    )
    raw_log_path: Path = Field(..., description="Path to the raw battle log file")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        description="Timestamp when the trajectory was recorded",
    )
    reward_p1: float = Field(
        ...,
        description="Final episodic reward from p1 perspective (1, 0, or -1)",
    )
    reward_p2: float = Field(
        ...,
        description="Final episodic reward from p2 perspective (-reward_p1)",
    )
    steps_p1: List[BattleStep] = Field(
        default_factory=list,
        description="Trajectory steps for side p1",
    )
    steps_p2: List[BattleStep] = Field(
        default_factory=list,
        description="Trajectory steps for side p2",
    )
    meta: Dict[str, Any] = Field(
        default_factory=dict,
        description="Optional metadata (e.g., seeds, simulation parameters, agent types, env version)",
    )

    class Config:
        json_encoders = {
            Path: str,
            datetime: lambda v: v.isoformat(),
        }


# ============================================================================
# 3. Showdown Client
# ============================================================================


class ShowdownClient:
    """Encapsulates all Showdown / Node calls."""

    def __init__(self, paths: Optional[Paths] = None, format_id: str = DEFAULT_FORMAT) -> None:
        self.paths = paths or get_paths()
        self.format_id = format_id

    def run_showdown_command(self, args: List[str], input_text: Optional[str] = None) -> str:
        """
        Run `node pokemon-showdown <args>` in SHOWDOWN_ROOT.

        Args:
            args: List of arguments to pass after "pokemon-showdown"
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
                check=False,
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

    def pack_team(self, export_text: str) -> str:
        """Convert a team from export format to packed format."""
        return self.run_showdown_command(["pack-team"], input_text=export_text)

    def validate_team(self, team_text: str, format_id: Optional[str] = None) -> str:
        """
        Validate a team against a format.

        Returns:
            Empty string if valid, error messages if invalid
        """
        if format_id is None:
            format_id = self.format_id

        try:
            return self.run_showdown_command(["validate-team", format_id], input_text=team_text)
        except RuntimeError as e:
            # On validation failure, extract error messages
            if "Exit code: 1" in str(e):
                parts = str(e).split("STDERR:\n", 1)
                if len(parts) > 1:
                    return parts[1].rstrip()
                return str(e)
            raise

    def generate_random_team(self, format_id: Optional[str] = None) -> str:
        """Generate a random team for a format."""
        if format_id is None:
            format_id = self.format_id
        return self.run_showdown_command(["generate-team", format_id])

    def run_random_selfplay_json(
        self,
        format_id: Optional[str] = None,
        p1_name: str = "Bot1",
        p2_name: str = "Bot2",
        p1_packed_team: Optional[str] = None,
        p2_packed_team: Optional[str] = None,
        timeout_seconds: int = 60,
    ) -> Dict[str, Any]:
        """
        Run random selfplay via Node.js script and return JSON.

        Returns:
            Dict with keys: format_id, p1_name, p2_name, winner_side, winner_name,
            turns, log (raw log text)
        """
        if format_id is None:
            format_id = self.format_id

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
                check=False,
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

        try:
            return json.loads(result.stdout.strip())
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse JSON output from random_selfplay: {e}")


# ============================================================================
# 4. Random Agent (optional, kept for compatibility)
# ============================================================================


class RandomAgent:
    """Random decision-making agent for battles."""

    def __init__(self, rng: Optional[random.Random] = None) -> None:
        self.rng = rng if rng is not None else random.Random()

    def choose_team_preview(self, request: Dict[str, Any]) -> str:
        """Choose a random subset of Pokémon for team preview."""
        side = request.get("side", {})
        pokemon = side.get("pokemon", [])
        total = len(pokemon)
        if total == 0:
            return "team"

        max_chosen = request.get("maxChosenTeamSize") or total
        max_chosen = max(1, min(int(max_chosen), total))

        available_slots = list(range(1, total + 1))
        chosen = self.rng.sample(available_slots, k=max_chosen)
        chosen.sort()

        return "team " + "".join(str(slot) for slot in chosen)

    def choose_turn_action(self, request: Dict[str, Any]) -> str:
        """Choose a random set of actions for a single turn."""
        side = request.get("side", {})
        active = side.get("active", [])
        all_pokemon = side.get("pokemon", [])

        bench_candidates: List[int] = []
        for idx, mon in enumerate(all_pokemon, start=1):
            is_active = bool(mon.get("active"))
            condition = mon.get("condition", "")
            fainted = condition.startswith("0 fnt")
            if not is_active and not fainted:
                bench_candidates.append(idx)

        choices: List[str] = []

        for mon_state in active:
            moves = mon_state.get("moves", []) or []
            can_switch = bool(mon_state.get("canSwitch", False))

            available_moves: List[tuple[int, Any]] = []
            for idx, move in enumerate(moves, start=1):
                if move is None:
                    continue
                if move.get("disabled"):
                    continue
                available_moves.append((idx, move))

            if available_moves and (not can_switch or self.rng.random() < 0.8):
                move_idx, _ = self.rng.choice(available_moves)
                choices.append(f"move {move_idx}")
            elif can_switch and bench_candidates:
                slot = self.rng.choice(bench_candidates)
                choices.append(f"switch {slot}")
            else:
                choices.append("pass")

        if not choices:
            return "pass"

        return ", ".join(choices)

    def choose_action(self, request: Dict[str, Any]) -> str:
        """High-level dispatcher: team preview vs normal turn."""
        if request.get("teamPreview"):
            return self.choose_team_preview(request)
        return self.choose_turn_action(request)


# ============================================================================
# 5. Battle Store
# ============================================================================


class BattleStore:
    """Handles saving logs, JSON, and JSONL datasets in a consistent way."""

    def __init__(self, paths: Optional[Paths] = None) -> None:
        self.paths = paths or get_paths()
        ensure_paths(self.paths)

    def save_raw_log(self, log_text: str, format_id: str) -> tuple[str, Path]:
        """
        Save raw battle log and return (battle_id, filepath).

        battle_id is a stable identifier used to tie together:
        - raw log
        - raw battle JSON
        - full_battles / team_preview / trajectories entries
        """
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")[:-3]
        battle_id = f"{timestamp}_{format_id}"
        filename = f"{battle_id}.log"
        filepath = self.paths.battles_raw / filename
        filepath.write_text(log_text, encoding="utf-8")
        return battle_id, filepath

    def infer_winner_from_log(self, log_text: str, p1_name: str, p2_name: str) -> str:
        """Infer winner from battle log using simple heuristics."""
        lines = log_text.split("\n")

        for line in lines:
            if line.startswith("|win|"):
                winner_name = line[5:].strip()
                if winner_name == p1_name:
                    return "p1"
                elif winner_name == p2_name:
                    return "p2"

            if "|tie|" in line or "|draw|" in line.lower():
                return "tie"

        return "unknown"

    def count_turns(self, log_text: str) -> Optional[int]:
        """Attempt to count turns from battle log."""
        lines = log_text.split("\n")
        max_turn = 0

        for line in lines:
            if line.startswith("|turn|"):
                try:
                    turn_num = int(line[6:].strip())
                    max_turn = max(max_turn, turn_num)
                except ValueError:
                    continue

        return max_turn if max_turn > 0 else None

    def append_full_battle_from_json(self, battle_json: Dict[str, Any]) -> FullBattleRecord:
        """
        Helper that saves log, builds FullBattleRecord, and appends to JSONL.

        Args:
            battle_json: Dict from run_random_selfplay_json() with keys:
                format_id, p1_name, p2_name, winner_side, winner_name, turns, log

        Returns:
            FullBattleRecord instance
        """
        log_text = battle_json.get("log", "")
        format_id = battle_json.get("format_id", self.paths.data_root.name)  # fallback
        p1_name = battle_json.get("p1_name", "Bot1")
        p2_name = battle_json.get("p2_name", "Bot2")
        winner_side = battle_json.get("winner_side", "unknown")
        winner_name = battle_json.get("winner_name")
        turns = battle_json.get("turns")

        # Save raw log and derive battle_id
        battle_id, raw_log_path = self.save_raw_log(log_text, format_id)

        # Save full battle JSON (for debugging / future extensions)
        json_path = self.paths.battles_json / f"{battle_id}.json"
        json_path.write_text(json.dumps(battle_json, ensure_ascii=False), encoding="utf-8")

        # Create record
        record = FullBattleRecord(
            battle_id=battle_id,
            format_id=format_id,
            p1_name=p1_name,
            p2_name=p2_name,
            winner_side=winner_side,
            winner_name=winner_name,
            turns=turns,
            raw_log_path=raw_log_path,
            created_at=datetime.now(timezone.utc),
            meta={},
        )

        # Append to JSONL
        data = {
            "battle_id": record.battle_id,
            "format_id": record.format_id,
            "p1_name": record.p1_name,
            "p2_name": record.p2_name,
            "winner_side": record.winner_side,
            "winner_name": record.winner_name,
            "turns": record.turns,
            "raw_log_path": str(record.raw_log_path),
            "created_at": record.created_at.isoformat(),
            "meta": record.meta,
        }

        json_line = json.dumps(data, ensure_ascii=False)
        with self.paths.full_battles_jsonl.open("a", encoding="utf-8") as f:
            f.write(json_line + "\n")

        return record

    def append_preview_snapshot(self, snapshot: TeamPreviewSnapshot) -> None:
        """Append TeamPreviewSnapshot to team_preview.jsonl."""
        data = snapshot.model_dump(mode="json")
        if snapshot.raw_log_path is not None:
            data["raw_log_path"] = str(snapshot.raw_log_path)
        data["created_at"] = snapshot.created_at.isoformat()

        json_line = json.dumps(data, ensure_ascii=False)
        with self.paths.team_preview_jsonl.open("a", encoding="utf-8") as f:
            f.write(json_line + "\n")

    def append_trajectory_from_battle(
        self,
        record: FullBattleRecord,
        battle_json: Dict[str, Any],
    ) -> None:
        """
        Append a BattleTrajectory row to trajectories.jsonl, if trajectory data is present
        in the battle_json.

        Args:
            record: FullBattleRecord returned by append_full_battle_from_json.
            battle_json: Dict returned by ShowdownClient.run_random_selfplay_json(),
                expected to contain a 'trajectory' key with:
                    {
                        "p1": [BattleStep-like dicts],
                        "p2": [BattleStep-like dicts],
                    }
        """
        traj_data = battle_json.get("trajectory")
        if not traj_data:
            # Nothing to write
            return

        steps_p1_raw = traj_data.get("p1") or []
        steps_p2_raw = traj_data.get("p2") or []

        # Validate and normalize steps through the Pydantic models
        steps_p1 = [BattleStep.model_validate(s) for s in steps_p1_raw]
        steps_p2 = [BattleStep.model_validate(s) for s in steps_p2_raw]

        # Simple episodic rewards from p1 perspective
        reward_map = {"p1": 1.0, "p2": -1.0, "tie": 0.0, "unknown": 0.0}
        reward_p1 = reward_map.get(record.winner_side, 0.0)
        reward_p2 = -reward_p1

        # Merge meta information coming from Node (py_policy_selfplay.js).
        # battle_json["meta"] is expected to contain:
        #   - battle_policy_id_p1 / battle_policy_id_p2 / battle_policy_id
        #   - agent_p1 / agent_p2
        #   - env_version
        raw_meta = battle_json.get("meta") or {}
        meta: Dict[str, Any] = {
            "agent_p1": raw_meta.get("agent_p1", battle_json.get("agent_p1", "RandomPlayerAI")),
            "agent_p2": raw_meta.get("agent_p2", battle_json.get("agent_p2", "RandomPlayerAI")),
            "env_version": raw_meta.get(
                "env_version",
                battle_json.get("env_version", "pokemon-showdown@local"),
            ),
        }

        # Preserve explicit battle_policy_id_* fields if present
        for key in ("battle_policy_id_p1", "battle_policy_id_p2", "battle_policy_id"):
            if key in raw_meta:
                meta[key] = raw_meta[key]

        trajectory = BattleTrajectory(
            battle_id=record.battle_id,
            format_id=record.format_id,
            p1_name=record.p1_name,
            p2_name=record.p2_name,
            winner_side=record.winner_side,
            winner_name=record.winner_name,
            turns=record.turns,
            raw_log_path=record.raw_log_path,
            created_at=record.created_at,
            reward_p1=reward_p1,
            reward_p2=reward_p2,
            steps_p1=steps_p1,
            steps_p2=steps_p2,
            meta=meta,
        )

        data = trajectory.model_dump(mode="json")
        json_line = json.dumps(data, ensure_ascii=False)
        with self.paths.trajectories_jsonl.open("a", encoding="utf-8") as f:
            f.write(json_line + "\n")


# ============================================================================
# 6. Team Preview Parsing
# ============================================================================


def _parse_packed_team_to_public_info(
    packed: str,
    side_id: str,
    player_name: str,
    max_chosen_team_size: int = 4,
) -> SideTeamPreview:
    """
    Parse a packed team string into a SideTeamPreview.

    Packed format: "mon1]mon2]mon3]..." where each mon is pipe-separated fields.
    """
    pokemon_list: List[PokemonPublicInfo] = []
    mons_raw = [m for m in packed.split("]") if m]

    for mon_raw in mons_raw:
        parts = mon_raw.split("|")
        
        # Extract fields (be resilient to length mismatches)
        # Packed format: species|nickname|item|ability|moves|...
        species = parts[0].strip() if len(parts) > 0 and parts[0] else "Unknown"
        nickname = parts[1].strip() if len(parts) > 1 and parts[1] else None
        item = parts[2].strip() if len(parts) > 2 and parts[2] else None
        ability = parts[3].strip() if len(parts) > 3 and parts[3] else None
        
        # Moves are comma-separated in parts[4]
        moves: List[str] = []
        if len(parts) > 4 and parts[4]:
            moves = [m.strip() for m in parts[4].split(",") if m.strip()]
        
        # Tera type is in parts[14] (if present)
        tera_type: Optional[str] = None
        if len(parts) > 14 and parts[14]:
            tera_type = parts[14].strip()

        pokemon_info = PokemonPublicInfo(
            ident=f"{side_id}: {species}",
            details=species,  # Simplified for packed format
            name=species,
            item=item if item else None,
            ability=ability if ability else None,
            moves=moves,
            tera_type=tera_type if tera_type else None,
            raw_stats=None,  # Not available in packed format
        )
        pokemon_list.append(pokemon_info)

    return SideTeamPreview(
        side_id=side_id,
        player_name=player_name,
        max_chosen_team_size=max_chosen_team_size,
        pokemon=pokemon_list,
    )


def parse_team_preview_snapshot(
    battle_json: Dict[str, Any],
    battle_id: str,
    raw_log_path: Path,
    created_at: Optional[datetime] = None,
) -> TeamPreviewSnapshot:
    """
    Parse a battle JSON from run_random_selfplay_json() into a TeamPreviewSnapshot.

    Args:
        battle_json: Dict with keys: format_id, p1_name, p2_name, p1_team_packed,
                     p2_team_packed, tier_name (optional), etc.
        raw_log_path: Path to the raw battle log file
        created_at: Optional datetime (defaults to now if not provided)

    Returns:
        TeamPreviewSnapshot instance
    """
    format_id = battle_json["format_id"]
    p1_name = battle_json.get("p1_name", "Player 1")
    p2_name = battle_json.get("p2_name", "Player 2")
    p1_team_packed = battle_json["p1_team_packed"]
    p2_team_packed = battle_json["p2_team_packed"]
    
    tier_name = battle_json.get("tier_name") or "[Gen 9] VGC 2026 Reg F"
    
    if created_at is None:
        created_at = datetime.now(timezone.utc)

    p1_preview = _parse_packed_team_to_public_info(
        packed=p1_team_packed,
        side_id="p1",
        player_name=p1_name,
        max_chosen_team_size=4,
    )
    
    p2_preview = _parse_packed_team_to_public_info(
        packed=p2_team_packed,
        side_id="p2",
        player_name=p2_name,
        max_chosen_team_size=4,
    )

    return TeamPreviewSnapshot(
        battle_id=battle_id,
        format_id=format_id,
        tier_name=tier_name,
        p1_preview=p1_preview,
        p2_preview=p2_preview,
        created_at=created_at,
        raw_log_path=raw_log_path,
    )

