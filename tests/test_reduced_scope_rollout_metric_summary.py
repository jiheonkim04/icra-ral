import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "76_generate_reduced_scope_rollout_metric_summary.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for reduced-scope metric summary tests")
    return exe


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_ROLLOUT",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
        "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_reduced_scope_report(path):
    payload = {
        "bounded_reduced_scope_learned_policy_rollout_passed": True,
        "rollout_result": {
            "result": {"passed": True, "tasks_completed": 1, "total_steps_performed": 10},
            "tasks": [
                {
                    "task_name": "task_a",
                    "success_check": False,
                    "reward_sum": 0.0,
                    "steps_performed": 10,
                    "policy_calls": 10,
                    "last_inference_sec": 0.15,
                    "last_policy_action_shape": [1, 6],
                    "action_dim": 7,
                    "last_env_action_preview": [-0.3, 0.1, 0.7, 0.8, 0.4, -0.2, 0.0],
                    "error": None,
                }
            ],
        },
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_summary(tmp_path, input_report=None, extra_env=None):
    input_report = input_report or tmp_path / "rollout.json"
    json_report = tmp_path / "summary.json"
    md_report = tmp_path / "summary.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-InputReportPath",
            str(input_report),
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


def test_reduced_scope_metric_summary_extracts_action_metrics(tmp_path):
    input_report = tmp_path / "rollout.json"
    _write_reduced_scope_report(input_report)

    result, report, json_report, md_report = _run_summary(tmp_path, input_report)

    assert result.returncode == 0, result.stderr
    assert report["reduced_scope_rollout_metric_summary_passed"] is True
    assert report["evidence_label"] == "reduced_scope_learned_policy_diagnostic"
    assert report["claims"]["standard_success_claimed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert report["metric_summary"]["total_steps"] == 10
    assert report["metric_summary"]["policy_calls"] == 10
    assert report["metric_summary"]["diagnostic_success_rate"] == 0.0
    assert report["metric_summary"]["reward_sum_total"] == 0.0
    assert report["metric_summary"]["policy_action_shapes"] == [[1, 6]]
    assert report["metric_summary"]["env_action_dims"] == [7]
    assert report["metric_summary"]["last_env_action_gripper_component"] == 0.0
    assert report["metric_summary"]["last_env_action_max_abs"] == 0.8
    assert json_report.exists()
    assert md_report.exists()


def test_reduced_scope_metric_summary_refuses_execution_gate(tmp_path):
    input_report = tmp_path / "rollout.json"
    _write_reduced_scope_report(input_report)

    result, report, _, _ = _run_summary(
        tmp_path,
        input_report,
        extra_env={"ALLOW_BOUNDED_LEARNED_POLICY_MATRIX": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["reduced_scope_rollout_metric_summary_passed"] is False
    assert "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX" in report["reason"]


def test_reduced_scope_metric_summary_handles_missing_input(tmp_path):
    missing = tmp_path / "missing.json"

    result, report, json_report, md_report = _run_summary(tmp_path, missing)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["reduced_scope_rollout_metric_summary_passed"] is False
    assert "Missing input report" in report["reason"]
    assert json_report.exists()
    assert md_report.exists()
