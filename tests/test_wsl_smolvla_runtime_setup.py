import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_SCRIPT = REPO_ROOT / "scripts" / "67_plan_wsl_smolvla_runtime_setup.ps1"
SETUP_SCRIPT = REPO_ROOT / "scripts" / "68_setup_wsl_smolvla_runtime_deps.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for WSL SmolVLA runtime setup tests")
    return exe


def _run_plan(tmp_path, extra_args=None, readiness=True):
    readiness_report = tmp_path / "policy_readiness.json"
    readiness_report.write_text(
        json.dumps({"ready_for_tiny_learned_policy_rollout_plan": readiness}),
        encoding="utf-8",
    )
    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(PLAN_SCRIPT),
            "-PolicyReadinessReportPath",
            str(readiness_report),
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(md_report),
            "-SkipLiveWslProbe",
            *(extra_args or []),
        ],
        cwd=REPO_ROOT,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    start = result.stdout.find("{")
    assert start >= 0, result.stdout + result.stderr
    return result, json.loads(result.stdout[start:]), json_report, md_report


def _run_setup(tmp_path, extra_env=None):
    plan_report = tmp_path / "plan.json"
    plan_report.write_text(
        json.dumps(
            {
                "decision": "proceed",
                "setup_required": True,
                "reason": "test plan",
            }
        ),
        encoding="utf-8",
    )
    json_report = tmp_path / "setup.json"
    md_report = tmp_path / "setup.md"
    env = os.environ.copy()
    for key in (
        "ALLOW_WSL_SMOLVLA_RUNTIME_SETUP",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_HEAVY_IMPORT",
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
            str(SETUP_SCRIPT),
            "-PlanReportPath",
            str(plan_report),
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


def test_wsl_smolvla_runtime_plan_is_safe_and_recommends_setup(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["setup_required"] is True
    assert report["policy"]["planning_only"] is True
    assert report["policy"]["installs_performed"] is False
    assert report["policy"]["package_downloads_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert "ALLOW_WSL_SMOLVLA_RUNTIME_SETUP=1" in report["recommended_next_step"]
    assert json_report.exists()
    assert md_report.exists()


def test_wsl_smolvla_runtime_plan_stops_without_policy_prereq(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, readiness=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["setup_required"] is True
    assert report["ready_for_wsl_smolvla_runtime"] is False
    assert any("scripts\\66_plan_libero_policy_rollout_readiness.ps1" in reason for reason in report["stop_reasons"])


def test_wsl_smolvla_runtime_setup_requires_task_local_gate(tmp_path):
    result, report, json_report, md_report = _run_setup(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["execution"]["gate_present"] is False
    assert report["execution"]["executed"] is False
    assert report["policy"]["package_installs_performed"] is False
    assert report["policy"]["package_downloads_performed"] is False
    assert "ALLOW_WSL_SMOLVLA_RUNTIME_SETUP=1" in report["recommended_next_step"]
    assert json_report.exists()
    assert md_report.exists()


def test_wsl_smolvla_runtime_setup_refuses_unrelated_rollout_gate(tmp_path):
    result, report, _, _ = _run_setup(
        tmp_path,
        extra_env={
            "ALLOW_WSL_SMOLVLA_RUNTIME_SETUP": "1",
            "ALLOW_POLICY_ROLLOUT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_POLICY_ROLLOUT" in report["reason"]
    assert report["execution"]["executed"] is False
    assert report["policy"]["package_installs_performed"] is False
