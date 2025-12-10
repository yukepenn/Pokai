"""Team evaluation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .battle_logger import save_raw_log
from .config import DEFAULT_FORMAT
from .showdown_cli import run_random_selfplay_json
from .teams import import_text_to_packed


@dataclass
class TeamEvalSummary:
    """Summary of team evaluation results."""

    format_id: str
    team_role: str  # "p1" or "p2"
    team_packed: str
    n_battles: int
    n_wins: int
    n_losses: int
    n_ties: int
    avg_turns: Optional[float]
    created_at: datetime


def evaluate_team_vs_random(
    team_packed: str,
    *,
    format_id: str = DEFAULT_FORMAT,
    n: int = 100,
    team_as: str = "p1",
    save_logs: bool = False,
) -> TeamEvalSummary:
    """
    Evaluate a fixed packed team against random opponents, using RandomPlayerAI.

    Args:
        team_packed: Showdown packed team string for a 6-Pokémon team.
        format_id:   Format ID, default is DEFAULT_FORMAT (gen9vgc2026regf).
        n:           Number of battles to run.
        team_as:     Which side to place the fixed team on: "p1", "p2", or "both".
                     If "both", alternates between P1 and P2 (50/50 split).
        save_logs:   If True, save each battle log to data/battles_raw/.

    Returns:
        TeamEvalSummary with basic statistics.
    """
    assert team_as in ("p1", "p2", "both"), f"team_as must be 'p1', 'p2', or 'both', got {team_as!r}"

    n_wins = 0
    n_losses = 0
    n_ties = 0
    turns_sum = 0
    turns_count = 0

    for i in range(n):
        # Determine which side the fixed team should be on for this battle
        if team_as == "both":
            # Alternate: even indices → P1, odd indices → P2
            current_side = "p1" if i % 2 == 0 else "p2"
        else:
            current_side = team_as

        # Run battle with fixed team on appropriate side
        if current_side == "p1":
            data = run_random_selfplay_json(
                format_id=format_id,
                p1_packed_team=team_packed,
                p2_packed_team=None,
            )
        else:
            data = run_random_selfplay_json(
                format_id=format_id,
                p1_packed_team=None,
                p2_packed_team=team_packed,
            )

        winner_side = data.get("winner_side", "unknown")
        turns = data.get("turns", None)
        log_text = data.get("log", "")

        # Optionally save battle log
        if save_logs and log_text:
            save_raw_log(format_id, log_text)

        # Count wins/losses/ties from fixed team's perspective
        if winner_side == "tie" or winner_side == "unknown":
            n_ties += 1
        else:
            if current_side == "p1":
                if winner_side == "p1":
                    n_wins += 1
                elif winner_side == "p2":
                    n_losses += 1
                else:
                    n_ties += 1
            else:  # current_side == "p2"
                if winner_side == "p2":
                    n_wins += 1
                elif winner_side == "p1":
                    n_losses += 1
                else:
                    n_ties += 1

        if isinstance(turns, int):
            turns_sum += turns
            turns_count += 1

    avg_turns: Optional[float]
    if turns_count > 0:
        avg_turns = turns_sum / turns_count
    else:
        avg_turns = None

    # Set team_role for summary
    team_role = "both" if team_as == "both" else team_as

    return TeamEvalSummary(
        format_id=format_id,
        team_role=team_role,
        team_packed=team_packed,
        n_battles=n,
        n_wins=n_wins,
        n_losses=n_losses,
        n_ties=n_ties,
        avg_turns=avg_turns,
        created_at=datetime.now(timezone.utc),
    )


@dataclass
class MatchupEvalSummary:
    """Summary of team vs team matchup evaluation results."""

    format_id: str
    team_a_import: str
    team_b_import: str
    team_a_packed: str
    team_b_packed: str
    n_battles: int
    n_a_wins: int
    n_b_wins: int
    n_ties: int
    avg_turns: Optional[float]
    created_at: datetime


def evaluate_team_vs_team(
    team_a_import: str,
    team_b_import: str,
    *,
    format_id: str = DEFAULT_FORMAT,
    n: int = 100,
) -> MatchupEvalSummary:
    """
    Evaluate Team A vs Team B using RandomPlayerAI on both sides.

    Team A and Team B are provided in Showdown import/export text format.
    We import+pack both, then run many self-play battles.

    Args:
        team_a_import: Showdown team text for Team A.
        team_b_import: Showdown team text for Team B.
        format_id:     Format ID, default is gen9vgc2026regf.
        n:             Number of battles to run.

    Returns:
        MatchupEvalSummary with win/loss/tie stats and average turns.
    """
    # Convert both import texts to packed strings
    team_a_packed = import_text_to_packed(team_a_import, format_id=format_id)
    team_b_packed = import_text_to_packed(team_b_import, format_id=format_id)

    n_a_wins = 0
    n_b_wins = 0
    n_ties = 0
    turns_sum = 0
    turns_count = 0

    for i in range(n):
        # Alternate sides to avoid seat bias
        # Even i: Team A as P1, Team B as P2
        # Odd i:  Team B as P1, Team A as P2
        if i % 2 == 0:
            # Team A as P1, Team B as P2
            data = run_random_selfplay_json(
                format_id=format_id,
                p1_packed_team=team_a_packed,
                p2_packed_team=team_b_packed,
            )
            winner_side = data.get("winner_side", "unknown")
            # Count from Team A's perspective
            if winner_side == "p1":
                n_a_wins += 1
            elif winner_side == "p2":
                n_b_wins += 1
            else:  # tie or unknown
                n_ties += 1
        else:
            # Team B as P1, Team A as P2
            data = run_random_selfplay_json(
                format_id=format_id,
                p1_packed_team=team_b_packed,
                p2_packed_team=team_a_packed,
            )
            winner_side = data.get("winner_side", "unknown")
            # Count from Team A's perspective (A is P2 here)
            if winner_side == "p2":
                n_a_wins += 1
            elif winner_side == "p1":
                n_b_wins += 1
            else:  # tie or unknown
                n_ties += 1

        turns = data.get("turns", None)
        if isinstance(turns, int):
            turns_sum += turns
            turns_count += 1

    avg_turns: Optional[float]
    if turns_count > 0:
        avg_turns = turns_sum / turns_count
    else:
        avg_turns = None

    return MatchupEvalSummary(
        format_id=format_id,
        team_a_import=team_a_import,
        team_b_import=team_b_import,
        team_a_packed=team_a_packed,
        team_b_packed=team_b_packed,
        n_battles=n,
        n_a_wins=n_a_wins,
        n_b_wins=n_b_wins,
        n_ties=n_ties,
        avg_turns=avg_turns,
        created_at=datetime.now(timezone.utc),
    )

