"""Configuration and path management for vgc-lab."""

from pathlib import Path
from typing import Final

# Project root: the directory containing pyproject.toml
PROJECT_ROOT: Final[Path] = Path(__file__).parent.parent.parent.resolve()

# Showdown root: sibling directory containing pokemon-showdown
SHOWDOWN_ROOT: Final[Path] = PROJECT_ROOT.parent / "pokemon-showdown"

# Format IDs (confirmed from pokemon-showdown/config/formats.ts)
DEFAULT_FORMAT: Final[str] = "gen9vgc2026regf"
DEFAULT_FORMAT_BO3: Final[str] = "gen9vgc2026regfbo3"

# Data directories
DATA_ROOT: Final[Path] = PROJECT_ROOT / "data"
TEAMS_DIR: Final[Path] = DATA_ROOT / "teams"
BATTLES_RAW_DIR: Final[Path] = DATA_ROOT / "battles_raw"
BATTLES_JSON_DIR: Final[Path] = DATA_ROOT / "battles_json"
MODELS_DIR: Final[Path] = DATA_ROOT / "models"


def ensure_paths() -> None:
    """
    Create the data directories (teams, battles_raw, battles_json, models) if they don't exist.

    This should be called at the start of scripts that need to write data.
    """
    TEAMS_DIR.mkdir(parents=True, exist_ok=True)
    BATTLES_RAW_DIR.mkdir(parents=True, exist_ok=True)
    BATTLES_JSON_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

