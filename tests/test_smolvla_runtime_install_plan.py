import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "18_plan_smolvla_runtime_install.ps1"


def _run_plan(tmp_path, extra_env=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the runtime install planner")

    report_path = tmp_path / "runtime_install_plan_report.json"
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


def test_runtime_install_planner_is_planning_only(tmp_path):
    result = _run_plan(tmp_path)
    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)

    assert report["policy"]["planning_only"] is True
    assert report["policy"]["risk_assessment_required_before_install"] is True
    assert report["policy"]["installs_performed"] is False
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["heavy_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    planned_distributions = {item["distribution"] for item in report["packages"]}
    assert "torch" in planned_distributions
    assert "num2words" in planned_distributions


def test_runtime_install_planner_refuses_dangerous_gates(tmp_path):
    result = _run_plan(tmp_path, {"ALLOW_HEAVY_IMPORT": "1"})
    assert result.returncode == 20
    assert "dangerous gates" in (result.stderr + result.stdout)
