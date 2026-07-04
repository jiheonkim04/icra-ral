import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "62_plan_tiny_diagnostic_rollout.ps1"


def _run_planner(tmp_path, extra_env=None, reset_report=None, extra_args=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the tiny diagnostic rollout planner")

    reset_report_path = tmp_path / "bounded_simulator_reset_step_smoke_report.json"
    if reset_report is not None:
        reset_report_path.write_text(json.dumps(reset_report), encoding="utf-8")
    json_report = tmp_path / "tiny_diagnostic_rollout_plan_report.json"
    md_report = tmp_path / "tiny_diagnostic_rollout_plan_report.md"

    env = os.environ.copy()
    for key in (
        "ALLOW_TINY_ROLLOUT",
        "ALLOW_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_TINY_TRAINING",
        "ALLOW_SIMULATOR_RESET_STEP",
        "ALLOW_SIMULATOR_RENDER_SMOKE",
    ):
        env.pop(key, None)
    env.update(extra_env or {})

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-ResetStepReportPath",
            str(reset_report_path),
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(md_report),
            *(extra_args or []),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    start = result.stdout.find("{")
    assert start >= 0, result.stdout + result.stderr
    return result, json.loads(result.stdout[start:]), json_report, md_report


def test_tiny_diagnostic_rollout_plan_stops_without_reset_report(tmp_path):
    result, report, json_report, md_report = _run_planner(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["ready_for_tiny_diagnostic_rollout_execution"] is False
    assert report["policy"]["planning_only"] is True
    assert report["policy"]["rollouts_performed"] is False
    assert any("reset/step smoke" in reason for reason in report["stop_reasons"])
    assert json_report.exists()
    assert md_report.exists()


def test_tiny_diagnostic_rollout_plan_proceeds_for_planning_only_after_reset_pass(tmp_path):
    result, report, _, _ = _run_planner(
        tmp_path,
        reset_report={"bounded_simulator_reset_step_smoke_passed": True},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["risk_envelope_inside_budget"] is True
    assert report["ready_for_tiny_diagnostic_rollout_plan"] is True
    assert report["ready_for_tiny_diagnostic_rollout_execution"] is True
    assert report["ready_for_rollout"] is False
    assert report["execution_authorized_by_this_planner"] is True
    assert report["policy"]["rollouts_performed"] is False
    assert report["risk_assessment"]["rollout_would_run_now"] is False


def test_tiny_diagnostic_rollout_plan_allows_five_task_boundary(tmp_path):
    result, report, _, _ = _run_planner(
        tmp_path,
        reset_report={"bounded_simulator_reset_step_smoke_passed": True},
        extra_args=["-TaskCount", "5", "-ExpectedRuntimeMinutes", "30", "-ExpectedVramGb", "14"],
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["risk_envelope_inside_budget"] is True
    assert report["ready_for_tiny_diagnostic_rollout_execution"] is True
    assert report["risk_assessment"]["task_count"] == 5
    assert report["risk_assessment"]["expected_runtime_minutes"] == 30
    assert report["risk_assessment"]["expected_vram_gb"] == 14


def test_tiny_diagnostic_rollout_plan_refuses_execution_gates(tmp_path):
    result, report, _, _ = _run_planner(
        tmp_path,
        extra_env={"ALLOW_TINY_ROLLOUT": "1"},
        reset_report={"bounded_simulator_reset_step_smoke_passed": True},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_TINY_ROLLOUT" in report["dangerous_execution_gates_set"]
    assert report["policy"]["rollouts_performed"] is False
