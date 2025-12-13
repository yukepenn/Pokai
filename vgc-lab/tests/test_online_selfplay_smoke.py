import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
NODE_SCRIPT = PROJECT_ROOT / "js" / "py_policy_selfplay.js"


def test_online_selfplay_node_vs_node_completes():
    """Smoke test: Node-only selfplay (random vs random) should emit a result JSON."""
    assert NODE_SCRIPT.is_file(), f"Node script not found: {NODE_SCRIPT}"

    cmd = [
        "node",
        str(NODE_SCRIPT),
        "--format-id",
        "gen9vgc2026regf",
        "--p1-policy",
        "node_random_v1",
        "--p2-policy",
        "node_random_v1",
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    stdout, stderr = proc.communicate(timeout=25)

    # 确认进程正常退出
    assert proc.returncode == 0, f"Node exited with code {proc.returncode}, stderr:\n{stderr}"

    lines = [ln.strip() for ln in stdout.splitlines() if ln.strip()]
    assert lines, f"No stdout from Node. stderr:\n{stderr}"

    last_json = None
    for line in reversed(lines):
        try:
            msg = json.loads(line)
        except json.JSONDecodeError:
            continue
        last_json = msg
        break

    assert last_json is not None, f"No JSON lines in stdout. Tail:\n" + "\n".join(lines[-5:])
    assert last_json.get("type") == "result", f"Expected type='result', got {last_json!r}"


def test_run_online_selfplay_returns_valid_summary():
    """Test that run_online_selfplay returns a valid summary dictionary."""
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from projects.rl_battle.online_selfplay import OnlineSelfPlayConfig, run_online_selfplay

    cfg = OnlineSelfPlayConfig(
        num_episodes=2,
        format_id="gen9vgc2026regf",
        p1_policy="node_random_v1",
        p2_policy="node_random_v1",
        seed=42,
        write_trajectories=True,
        strict_invalid_choice=True,
        debug=False,
    )

    summary = run_online_selfplay(cfg)

    # Basic shape - all five required keys must be present
    assert isinstance(summary, dict)
    required_keys = ("episodes", "errors", "p1_wins", "p2_wins", "draws")
    for key in required_keys:
        assert key in summary, f"Missing key in summary: {key}"

    # Check types - all values must be non-negative integers
    assert isinstance(summary["episodes"], int), f"episodes must be int, got {type(summary['episodes'])}"
    assert isinstance(summary["errors"], int), f"errors must be int, got {type(summary['errors'])}"
    assert isinstance(summary["p1_wins"], int), f"p1_wins must be int, got {type(summary['p1_wins'])}"
    assert isinstance(summary["p2_wins"], int), f"p2_wins must be int, got {type(summary['p2_wins'])}"
    assert isinstance(summary["draws"], int), f"draws must be int, got {type(summary['draws'])}"
    
    # Check non-negativity
    assert summary["episodes"] >= 0
    assert summary["errors"] >= 0
    assert summary["p1_wins"] >= 0
    assert summary["p2_wins"] >= 0
    assert summary["draws"] >= 0

    # Check values
    assert summary["episodes"] == 2
    assert summary["errors"] == 0
    
    # Critical invariant: episodes == errors + p1_wins + p2_wins + draws
    total = summary["errors"] + summary["p1_wins"] + summary["p2_wins"] + summary["draws"]
    assert total == summary["episodes"], (
        f"Invariant violated: episodes ({summary['episodes']}) != "
        f"errors + p1_wins + p2_wins + draws ({total})"
    )


def test_online_selfplay_cli_completes():
    """Smoke test: Python CLI online-selfplay should complete 1 episode with node-vs-node."""
    env = os.environ.copy()
    # 确保 src/vgc_lab 可以被 import
    env["PYTHONPATH"] = str(PROJECT_ROOT)

    cmd = [
        sys.executable,
        "-m",
        "scripts.cli",
        "online-selfplay",
        "--num-episodes",
        "1",
        "--p1-policy",
        "node_random_v1",
        "--p2-policy",
        "node_random_v1",
        "--strict",  # 默认严格模式
    ]

    proc = subprocess.Popen(
        cmd,
        cwd=str(PROJECT_ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )

    stdout, stderr = proc.communicate(timeout=35)

    # CLI should print summary with all fields
    assert "Episodes:" in stdout, f"No 'Episodes:' in stdout.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    assert "Errors:" in stdout, f"No 'Errors:' in stdout.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    assert "P1 wins:" in stdout, f"No 'P1 wins:' in stdout.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    assert "P2 wins:" in stdout, f"No 'P2 wins:' in stdout.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    assert "Draws:" in stdout, f"No 'Draws:' in stdout.\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"
    assert "Episodes: 1" in stdout, f"Expected 'Episodes: 1' in stdout.\nSTDOUT:\n{stdout}"
    assert "Errors:   0" in stdout, f"Expected 'Errors:   0' in stdout.\nSTDOUT:\n{stdout}"

    # CLI should exit with code 0 when there are no errors
    assert proc.returncode == 0, f"CLI exited with code {proc.returncode}, stderr:\n{stderr}"
