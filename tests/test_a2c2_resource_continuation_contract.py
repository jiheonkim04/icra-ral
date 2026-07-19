from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER = REPO_ROOT / "scripts" / "run_a2c2_resource_smoke.py"
FROZEN_RUNNER = REPO_ROOT / "scripts" / "run_a2c2_problem_verification.py"
MONITOR = REPO_ROOT / "scripts" / "monitor_a2c2_resource_smoke.ps1"
WSL_LAUNCHER = REPO_ROOT / "scripts" / "run_a2c2_resource_smoke_wsl.sh"


def test_resource_smoke_mode_is_syntax_valid_and_outcome_suppressing() -> None:
    text = RUNNER.read_text(encoding="utf-8")
    ast.parse(text)

    assert '"RESOURCE_ONLY_ACTUAL_PATH_SMOKE"' in text
    assert '"task_success_persisted": False' in text
    assert '"task_success_counted": False' in text
    assert '"reward_persisted": False' in text
    assert '"scientific_episode_row_persisted": False' in text
    assert '"ours_designed_or_executed": False' in text
    assert '"prior_retrained": False' in text
    assert "import run_a2c2_problem_verification as frozen" in text


def test_frozen_scientific_runner_has_no_resource_smoke_mode() -> None:
    text = FROZEN_RUNNER.read_text(encoding="utf-8")

    assert '"resource-smoke"' not in text


def test_host_monitor_enforces_frozen_launch_and_runtime_ceilings() -> None:
    text = MONITOR.read_text(encoding="utf-8")

    assert "A2C2_HOST_MEMORY_BASELINE_UNSAFE" in text
    assert "-gt 0.70" in text
    assert "-gt 0.82" in text
    assert 'memory=${CapGiB}GB' in text
    assert 'swap=0' in text
    assert "pagefile_current_growth_mib" in text
    assert "Start-Process" in text
    assert "-WindowStyle Hidden" in text
    assert "internal_report_fallback" in text
    assert "Stale internal smoke report must be archived" in text
    assert "--terminate Ubuntu-22.04" in text
    assert 'exitCodeSource = "host_ceiling_guard"' in text


def test_wsl_launcher_uses_the_accepted_a2c2_environment() -> None:
    text = WSL_LAUNCHER.read_text(encoding="utf-8")

    assert "/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python" in text
    assert "/home/jiheon/.venvs/tca_map_sim/bin/python" not in text
