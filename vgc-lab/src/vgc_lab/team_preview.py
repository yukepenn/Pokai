"""Team preview and open team sheet models and parsing."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from .config import DEFAULT_FORMAT, PROJECT_ROOT

# Dataset location
DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "team_preview"
DATASET_PATH = DATASET_DIR / "team_preview.jsonl"


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


def ensure_dataset_dir() -> None:
    """Ensure the dataset directory exists."""
    DATASET_DIR.mkdir(parents=True, exist_ok=True)


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


# Preview+Outcome dataset location
PREVIEW_OUTCOME_DATASET_DIR = PROJECT_ROOT / "data" / "datasets" / "preview_outcome"
PREVIEW_OUTCOME_DATASET_PATH = PREVIEW_OUTCOME_DATASET_DIR / "preview_outcome.jsonl"


def ensure_preview_outcome_dataset_dir() -> None:
    """Ensure the preview+outcome dataset directory exists."""
    PREVIEW_OUTCOME_DATASET_DIR.mkdir(parents=True, exist_ok=True)


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

