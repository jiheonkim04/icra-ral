import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "79_compare_zero_action_policy_diagnostic.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for zero-action policy diagnostic comparison tests")
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


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_inputs(tmp_path):
    zero = tmp_path / "zero.json"
    learned = tmp_path / "learned.json"
    audit = tmp_path / "audit.json"
    _write_json(
        zero,
        {
            "bounded_libero_robosuite_diagnostic_rollout_passed": True,
            "policy": {
                "simulator_environment_created": True,
                "learned_policy_inference_performed": False,
                "zero_action_policy_only": True,
            },
            "rollout_result": {
                "tasks_completed": 1,
                "task_summaries": [
                    {
                        "task_name": "task_a",
                        "steps_performed": 3,
                        "reward_sum": 0.0,
                        "success_check": False,
                    }
                ],
            },
        },
    )
    _write_json(
        learned,
        {
            "reduced_scope_rollout_metric_summary_passed": True,
            "metric_summary": {
                "source_runner_passed": True,
                "tasks_observed": 1,
                "tasks_completed": 1,
                "total_steps": 10,
                "policy_calls": 10,
                "diagnostic_success_count": 0,
                "diagnostic_success_rate": 0.0,
                "reward_sum_total": 0.0,
                "last_env_action_max_abs": 0.8,
                "last_env_action_l2": 1.2,
                "last_env_action_gripper_component": 0.0,
                "last_env_action_preview": [-0.3, 0.1, 0.7, 0.8, 0.4, -0.2, 0.0],
                "policy_action_shapes": [[1, 6]],
                "env_action_dims": [7],
                "failure_modes": [{"task_name": "task_a", "failure": "diagnostic_success_check_false"}],
            },
        },
    )
    _write_json(
        audit,
        {
            "action_interface_metadata_audit_passed": True,
            "high_priority_findings": [
                "action_dim_mismatch",
                "gripper_constant_zero",
                "state_truncation_risk",
            ],
        },
    )
    return zero, learned, audit


def _run_comparison(tmp_path, zero=None, learned=None, audit=None, extra_env=None):
    zero, learned, audit = (zero, learned, audit) if zero else _write_inputs(tmp_path)
    json_report = tmp_path / "comparison.json"
    md_report = tmp_path / "comparison.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-ZeroActionReportPath",
            str(zero),
            "-LearnedPolicySummaryPath",
            str(learned),
            "-ActionAuditReportPath",
            str(audit),
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


def test_zero_action_policy_comparison_prioritizes_adapter_patch(tmp_path):
    result, report, json_report, md_report = _run_comparison(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["zero_action_policy_diagnostic_comparison_passed"] is True
    assert report["comparison"]["zero_action_env_plumbing_passed"] is True
    assert report["comparison"]["policy_action_nontrivial"] is True
    assert report["comparison"]["learned_policy_outperformed_zero_action"] is False
    assert report["ready_for_action_state_adapter_patch_plan"] is True
    assert report["ready_for_rollout_scaling"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_zero_action_policy_comparison_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_comparison(
        tmp_path,
        extra_env={"ALLOW_BOUNDED_LEARNED_POLICY_MATRIX": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_BOUNDED_LEARNED_POLICY_MATRIX" in reason for reason in report["stop_reasons"])


def test_zero_action_policy_comparison_handles_missing_zero_report(tmp_path):
    zero, learned, audit = _write_inputs(tmp_path)
    zero.unlink()

    result, report, json_report, md_report = _run_comparison(tmp_path, zero, learned, audit)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("Missing input report" in reason for reason in report["stop_reasons"])
    assert json_report.exists()
    assert md_report.exists()
