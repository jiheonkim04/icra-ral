import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAN_SCRIPT = REPO_ROOT / "scripts" / "71_plan_tiny_learned_policy_rollout.ps1"
RUN_SCRIPT = REPO_ROOT / "scripts" / "72_bounded_tiny_learned_policy_rollout.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for tiny learned-policy rollout tests")
    return exe


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_TINY_TRAINING",
        "ALLOW_GPU_TRAINING",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _run_plan(tmp_path, policy_ready=True, single_action_passed=True, extra_env=None):
    policy_report = tmp_path / "policy.json"
    single_action_report = tmp_path / "single_action.json"
    policy_report.write_text(
        json.dumps({"ready_for_tiny_learned_policy_rollout_execution": policy_ready}),
        encoding="utf-8",
    )
    single_action_report.write_text(
        json.dumps({"wsl_smolvla_single_action_smoke_passed": single_action_passed}),
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
            str(policy_report),
            "-SingleActionReportPath",
            str(single_action_report),
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(md_report),
        ],
        cwd=REPO_ROOT,
        env=_clean_env(extra_env),
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
        env=_clean_env(extra_env),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    start = result.stdout.find("{")
    assert start >= 0, result.stdout + result.stderr
    return result, json.loads(result.stdout[start:]), json_report, md_report


def test_tiny_learned_policy_rollout_plan_allows_green_prereqs(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["ready_for_tiny_learned_policy_rollout_execution"] is True
    assert report["policy"]["planning_only"] is True
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_tiny_learned_policy_rollout_plan_stops_without_single_action(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, single_action_passed=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["ready_for_tiny_learned_policy_rollout_execution"] is False
    assert any("single-action smoke" in reason for reason in report["stop_reasons"])


def test_tiny_learned_policy_rollout_runner_requires_gate(tmp_path):
    result, report, json_report, md_report = _run_runner(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["diagnostic_rollouts_performed"] is False
    assert "ALLOW_TINY_LEARNED_POLICY_ROLLOUT=1" in report["recommended_next_step"]
    assert json_report.exists()
    assert md_report.exists()


def test_tiny_learned_policy_rollout_runner_refuses_broad_rollout_gate(tmp_path):
    result, report, _, _ = _run_runner(
        tmp_path,
        extra_env={
            "ALLOW_TINY_LEARNED_POLICY_ROLLOUT": "1",
            "ALLOW_ROLLOUT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUT" in report["reason"]
    assert report["policy"]["model_load_performed"] is False
