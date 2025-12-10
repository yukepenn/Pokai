"""Wrapper around pokemon-showdown CLI commands using subprocess."""

import json
import random
import subprocess
from pathlib import Path
from typing import Any, Optional, Tuple

from .config import SHOWDOWN_ROOT, DEFAULT_FORMAT
from .random_bot import choose_action


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
    import json

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

