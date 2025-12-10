"""Team format conversion utilities."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from .config import DEFAULT_FORMAT, PROJECT_ROOT


def import_text_to_packed(
    import_text: str,
    *,
    format_id: str = DEFAULT_FORMAT,
    timeout_seconds: int = 30,
) -> str:
    """
    Convert Showdown import/export team text to a packed team string
    using the Node script js/import_to_packed.js.

    Args:
        import_text: Multi-line Showdown team text (6 Pokémon).
        format_id:   Format ID for validation context (default: gen9vgc2026regf).
        timeout_seconds: Subprocess timeout.

    Returns:
        Packed team string (suitable for Teams.unpack and run_random_selfplay_json).

    Raises:
        RuntimeError on conversion failure.
    """
    cmd = [
        "node",
        str(PROJECT_ROOT / "js" / "import_to_packed.js"),
        "--format",
        format_id,
    ]
    
    proc = subprocess.run(
        cmd,
        input=import_text.encode("utf-8"),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=timeout_seconds,
        cwd=str(PROJECT_ROOT),
    )
    
    if proc.returncode != 0:
        raise RuntimeError(
            f"import_text_to_packed failed with code {proc.returncode}: "
            f"{proc.stderr.decode('utf-8', errors='ignore')}"
        )

    packed = proc.stdout.decode("utf-8", errors="strict").strip()
    if not packed:
        raise RuntimeError("import_text_to_packed: empty packed result")
    
    return packed

