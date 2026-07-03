import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "26_plan_tiny_head_only_pilot.ps1"


def _run_planner(tmp_path, extra_env=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the tiny head-only pilot planner")

    report_path = tmp_path / "tiny_head_only_pilot_plan_report.json"
    env = os.environ.copy()
    env.update(extra_env or {})
    return subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-ReportPath",
            str(report_path),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def _json_from_stdout(stdout):
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_tiny_head_only_pilot_planner_is_planning_only(tmp_path):
    result = _run_planner(tmp_path)
    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)

    assert report["configs_pass_policy"] is True
    assert report["safe_to_run_training_now"] is False
    assert report["ready_to_request_tiny_training_approval"] is True
    assert report["policy"]["planning_only"] is True
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False


def test_tiny_head_only_pilot_planner_refuses_training_gate(tmp_path):
    result = _run_planner(tmp_path, {"ALLOW_TINY_TRAINING": "1"})
    assert result.returncode == 20
    assert "dangerous gates" in (result.stdout + result.stderr)
