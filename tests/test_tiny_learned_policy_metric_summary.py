import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "73_generate_tiny_learned_policy_metric_summary.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for tiny learned-policy metric summary tests")
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
        "ALLOW_OPENVLA_OFT",
        "ALLOW_RUNTIME_INSTALL",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_rollout_report(path):
    payload = {
        "tiny_learned_policy_rollout_passed": True,
        "rollout_result": {
            "runtime": {"elapsed_sec": 30.6},
            "result": {"passed": True, "tasks_completed": 1, "total_steps_performed": 3},
            "tasks": [
                {
                    "task_name": "task_a",
                    "success_check": False,
                    "reward_sum": 0.0,
                    "steps_performed": 3,
                    "policy_calls": 3,
                    "last_inference_sec": 0.25,
                    "last_policy_action_shape": [1, 6],
                    "action_dim": 7,
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


def test_tiny_learned_policy_metric_summary_extracts_diagnostic_metrics(tmp_path):
    input_report = tmp_path / "rollout.json"
    _write_rollout_report(input_report)

    result, report, json_report, md_report = _run_summary(tmp_path, input_report)

    assert result.returncode == 0, result.stderr
    assert report["tiny_learned_policy_metric_summary_passed"] is True
    assert report["evidence_label"] == "tiny_learned_policy_diagnostic"
    assert report["claims"]["standard_success_claimed"] is False
    assert report["claims"]["benchmark_success_claimed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["metric_summary"]["source_rollout_passed"] is True
    assert report["metric_summary"]["total_steps"] == 3
    assert report["metric_summary"]["policy_calls"] == 3
    assert report["metric_summary"]["diagnostic_success_count"] == 0
    assert report["metric_summary"]["diagnostic_success_rate"] == 0.0
    assert report["metric_summary"]["policy_action_shapes"] == [[1, 6]]
    assert report["metric_summary"]["env_action_dims"] == [7]
    assert json_report.exists()
    assert md_report.exists()


def test_tiny_learned_policy_metric_summary_refuses_execution_gate(tmp_path):
    input_report = tmp_path / "rollout.json"
    _write_rollout_report(input_report)

    result, report, _, _ = _run_summary(
        tmp_path,
        input_report,
        extra_env={"ALLOW_TINY_LEARNED_POLICY_ROLLOUT": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["tiny_learned_policy_metric_summary_passed"] is False
    assert "ALLOW_TINY_LEARNED_POLICY_ROLLOUT" in report["reason"]


def test_tiny_learned_policy_metric_summary_handles_missing_input(tmp_path):
    missing = tmp_path / "missing.json"

    result, report, json_report, md_report = _run_summary(tmp_path, missing)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["tiny_learned_policy_metric_summary_passed"] is False
    assert "Missing input report" in report["reason"]
    assert json_report.exists()
    assert md_report.exists()
