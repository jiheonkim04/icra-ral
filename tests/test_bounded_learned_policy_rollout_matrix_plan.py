import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "74_plan_bounded_learned_policy_rollout_matrix.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for bounded rollout matrix planner tests")
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


def _write_metric_summary(path, *, success_rate=0.0, passed=True):
    path.write_text(
        json.dumps(
            {
                "tiny_learned_policy_metric_summary_passed": passed,
                "metric_summary": {
                    "source_rollout_passed": passed,
                    "total_steps": 3,
                    "policy_calls": 3,
                    "diagnostic_success_count": 1 if success_rate > 0 else 0,
                    "diagnostic_success_rate": success_rate,
                    "reward_sum_total": 1.0 if success_rate > 0 else 0.0,
                },
            }
        ),
        encoding="utf-8",
    )


def _run_plan(tmp_path, metric_path=None, extra_env=None):
    metric_path = metric_path or tmp_path / "metric.json"
    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-MetricSummaryReportPath",
            str(metric_path),
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


def test_matrix_planner_reduces_scope_when_diagnostic_success_is_zero(tmp_path):
    metric = tmp_path / "metric.json"
    _write_metric_summary(metric, success_rate=0.0)

    result, report, json_report, md_report = _run_plan(tmp_path, metric)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "reduce_scope"
    assert report["ready_for_reduced_scope_learned_policy_runner"] is True
    assert report["ready_for_bounded_small_learned_policy_matrix_runner"] is False
    assert report["recommended_rung"] == "one_task_longer_diagnostic"
    assert report["risk_assessment"]["recommended_task_count"] == 1
    assert report["risk_assessment"]["recommended_steps_per_task"] == 10
    assert report["policy"]["rollouts_performed"] is False
    assert report["evidence_policy"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_matrix_planner_allows_small_matrix_when_success_is_positive(tmp_path):
    metric = tmp_path / "metric.json"
    _write_metric_summary(metric, success_rate=0.5)

    result, report, _, _ = _run_plan(tmp_path, metric)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["ready_for_reduced_scope_learned_policy_runner"] is True
    assert report["ready_for_bounded_small_learned_policy_matrix_runner"] is True
    assert report["recommended_rung"] == "bounded_small_matrix"


def test_matrix_planner_stops_when_metric_summary_missing(tmp_path):
    missing = tmp_path / "missing.json"

    result, report, _, _ = _run_plan(tmp_path, missing)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["ready_for_reduced_scope_learned_policy_runner"] is False
    assert any("Missing metric summary" in reason for reason in report["stop_reasons"])


def test_matrix_planner_refuses_execution_gate(tmp_path):
    metric = tmp_path / "metric.json"
    _write_metric_summary(metric, success_rate=0.0)

    result, report, _, _ = _run_plan(
        tmp_path,
        metric,
        extra_env={"ALLOW_BOUNDED_LEARNED_POLICY_MATRIX": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_BOUNDED_LEARNED_POLICY_MATRIX" in reason for reason in report["stop_reasons"])
