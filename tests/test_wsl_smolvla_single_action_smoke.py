import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_SCRIPT = REPO_ROOT / "scripts" / "69_plan_wsl_smolvla_single_action_smoke.ps1"
RUN_SCRIPT = REPO_ROOT / "scripts" / "70_bounded_wsl_smolvla_single_action_smoke.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for WSL SmolVLA single-action smoke tests")
    return exe


def _run_plan(tmp_path, policy_ready=True, runtime_ready=True, extra_env=None):
    policy_report = tmp_path / "policy.json"
    runtime_report = tmp_path / "runtime.json"
    policy_report.write_text(
        json.dumps({"ready_for_tiny_learned_policy_rollout_execution": policy_ready}),
        encoding="utf-8",
    )
    runtime_report.write_text(
        json.dumps({"ready_for_wsl_smolvla_runtime": runtime_ready}),
        encoding="utf-8",
    )
    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    env = os.environ.copy()
    for key in (
        "ALLOW_WSL_SMOLVLA_SINGLE_ACTION",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_TINY_TRAINING",
        "ALLOW_GPU_TRAINING",
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
            "-PolicyReadinessReportPath",
            str(policy_report),
            "-WslRuntimePlanPath",
            str(runtime_report),
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(md_report),
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


def _run_runner(tmp_path, extra_env=None):
    json_report = tmp_path / "run.json"
    md_report = tmp_path / "run.md"
    env = os.environ.copy()
    for key in (
        "ALLOW_WSL_SMOLVLA_SINGLE_ACTION",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_TINY_TRAINING",
        "ALLOW_GPU_TRAINING",
        "ALLOW_SIMULATOR_RESET_STEP",
        "ALLOW_SIMULATOR_RENDER_SMOKE",
        "ALLOW_TINY_ROLLOUT",
        "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
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
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(md_report),
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


def test_wsl_smolvla_single_action_plan_allows_green_prereqs(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["ready_for_wsl_smolvla_single_action_smoke"] is True
    assert report["policy"]["planning_only"] is True
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_wsl_smolvla_single_action_plan_stops_without_runtime_ready(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, runtime_ready=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["ready_for_wsl_smolvla_single_action_smoke"] is False
    assert any("WSL SmolVLA runtime" in reason for reason in report["stop_reasons"])


def test_wsl_smolvla_single_action_runner_requires_gate(tmp_path):
    result, report, json_report, md_report = _run_runner(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert "ALLOW_WSL_SMOLVLA_SINGLE_ACTION=1" in report["recommended_next_step"]
    assert json_report.exists()
    assert md_report.exists()


def test_wsl_smolvla_single_action_runner_refuses_policy_rollout_gate(tmp_path):
    result, report, _, _ = _run_runner(
        tmp_path,
        extra_env={
            "ALLOW_WSL_SMOLVLA_SINGLE_ACTION": "1",
            "ALLOW_POLICY_ROLLOUT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_POLICY_ROLLOUT" in report["reason"]
    assert report["policy"]["model_load_performed"] is False
