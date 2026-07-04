import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_SCRIPT = REPO_ROOT / "scripts" / "64_plan_libero_robosuite_diagnostic_rollout.ps1"
RUN_SCRIPT = REPO_ROOT / "scripts" / "65_bounded_libero_robosuite_diagnostic_rollout.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for diagnostic rollout tests")
    return exe


def _run_plan(tmp_path, extra_env=None, extra_args=None, prereq=True):
    if prereq:
        (tmp_path / "import.json").write_text(json.dumps({"bounded_simulator_import_smoke_passed": True}), encoding="utf-8")
        (tmp_path / "render.json").write_text(json.dumps({"bounded_simulator_render_smoke_passed": True}), encoding="utf-8")
        (tmp_path / "reset.json").write_text(json.dumps({"bounded_simulator_reset_step_smoke_passed": True}), encoding="utf-8")
        (tmp_path / "tiny.json").write_text(json.dumps({"bounded_tiny_diagnostic_rollout_passed": True}), encoding="utf-8")

    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    env = os.environ.copy()
    for key in (
        "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
        "ALLOW_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_TINY_TRAINING",
        "ALLOW_SIMULATOR_RESET_STEP",
        "ALLOW_SIMULATOR_RENDER_SMOKE",
        "ALLOW_TINY_ROLLOUT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PLAN_SCRIPT),
            "-ImportSmokeReportPath",
            str(tmp_path / "import.json"),
            "-RenderSmokeReportPath",
            str(tmp_path / "render.json"),
            "-ResetStepReportPath",
            str(tmp_path / "reset.json"),
            "-TinyDiagnosticReportPath",
            str(tmp_path / "tiny.json"),
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


def _run_runner(tmp_path, extra_env=None, extra_args=None):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"ready_for_libero_robosuite_diagnostic_rollout_execution": True}), encoding="utf-8")
    json_report = tmp_path / "run.json"
    md_report = tmp_path / "run.md"
    env = os.environ.copy()
    for key in (
        "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
        "ALLOW_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_TINY_TRAINING",
        "ALLOW_SIMULATOR_RESET_STEP",
        "ALLOW_SIMULATOR_RENDER_SMOKE",
        "ALLOW_TINY_ROLLOUT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(RUN_SCRIPT),
            "-PlanReportPath",
            str(plan),
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


def test_libero_robosuite_diagnostic_plan_stops_without_prereqs(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path, prereq=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["ready_for_libero_robosuite_diagnostic_rollout_execution"] is False
    assert report["policy"]["planning_only"] is True
    assert report["policy"]["diagnostic_rollouts_performed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_libero_robosuite_diagnostic_plan_allows_bounded_green_path(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, extra_args=["-TaskCount", "5", "-MaxStepsPerTask", "5"])

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["risk_envelope_inside_budget"] is True
    assert report["ready_for_libero_robosuite_diagnostic_rollout_execution"] is True
    assert report["ready_for_benchmark_rollout"] is False
    assert report["ready_for_paper_claim"] is False
    assert report["risk_assessment"]["task_count"] == 5
    assert report["risk_assessment"]["max_steps_per_task"] == 5


def test_libero_robosuite_diagnostic_plan_refuses_execution_gates(tmp_path):
    result, report, _, _ = _run_plan(
        tmp_path,
        extra_env={"ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT" in report["dangerous_execution_gates_set"]
    assert report["policy"]["diagnostic_rollouts_performed"] is False


def test_bounded_libero_robosuite_diagnostic_runner_requires_task_local_gate(tmp_path):
    result, report, json_report, md_report = _run_runner(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["bounded_libero_robosuite_diagnostic_rollout_passed"] is False
    assert report["policy"]["simulator_environment_attempted"] is False
    assert report["policy"]["diagnostic_rollouts_performed"] is False
    assert report["policy"]["benchmark_rollouts_performed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_bounded_libero_robosuite_diagnostic_runner_refuses_broad_rollout_gate(tmp_path):
    result, report, _, _ = _run_runner(
        tmp_path,
        extra_env={
            "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT": "1",
            "ALLOW_ROLLOUT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUT" in report["reason"]
    assert report["policy"]["simulator_environment_attempted"] is False


def test_bounded_libero_robosuite_diagnostic_runner_rejects_too_many_tasks(tmp_path):
    result, report, _, _ = _run_runner(
        tmp_path,
        extra_env={"ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT": "1"},
        extra_args=["-TaskCount", "6"],
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "TaskCount" in report["reason"]
    assert report["policy"]["simulator_environment_attempted"] is False
