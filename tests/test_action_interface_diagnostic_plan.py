import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "77_plan_action_interface_diagnostics.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for action-interface diagnostic planner tests")
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
        "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_summary(path):
    path.write_text(
        json.dumps(
            {
                "reduced_scope_rollout_metric_summary_passed": True,
                "metric_summary": {
                    "diagnostic_success_rate": 0.0,
                    "reward_sum_total": 0.0,
                    "policy_action_shapes": [[1, 6]],
                    "env_action_dims": [7],
                    "last_env_action_gripper_component": 0.0,
                    "last_env_action_max_abs": 0.793,
                    "last_env_action_l2": 1.222,
                },
            }
        ),
        encoding="utf-8",
    )


def _run_plan(tmp_path, summary_path=None, extra_env=None):
    summary_path = summary_path or tmp_path / "summary.json"
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    (ckpt / "config.json").write_text(json.dumps({"max_action_dim": 6}), encoding="utf-8")
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
            str(summary_path),
            "-SmolVlaCkptPath",
            str(ckpt),
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


def test_action_interface_plan_prioritizes_dim_and_gripper_mismatch(tmp_path):
    summary = tmp_path / "summary.json"
    _write_summary(summary)

    result, report, json_report, md_report = _run_plan(tmp_path, summary)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["ready_for_action_interface_audit"] is True
    assert report["ready_for_zero_action_vs_policy_action_diagnostic"] is True
    assert report["observed_signals"]["action_dim_mismatch"] is True
    assert report["observed_signals"]["gripper_padded_zero"] is True
    assert report["observed_signals"]["nontrivial_action_magnitude"] is True
    priorities = {item["name"]: item["priority"] for item in report["diagnostics"]}
    assert priorities["action_dimension_and_gripper_mapping"] == "high"
    assert priorities["action_normalization_and_scale"] == "high"
    assert report["policy"]["rollouts_performed"] is False
    assert report["evidence_policy"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_action_interface_plan_stops_when_summary_missing(tmp_path):
    missing = tmp_path / "missing.json"

    result, report, _, _ = _run_plan(tmp_path, missing)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["ready_for_action_interface_audit"] is False
    assert any("Missing reduced-scope metric summary" in reason for reason in report["stop_reasons"])


def test_action_interface_plan_refuses_execution_gate(tmp_path):
    summary = tmp_path / "summary.json"
    _write_summary(summary)

    result, report, _, _ = _run_plan(
        tmp_path,
        summary,
        extra_env={"ALLOW_BOUNDED_LEARNED_POLICY_MATRIX": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_BOUNDED_LEARNED_POLICY_MATRIX" in reason for reason in report["stop_reasons"])
