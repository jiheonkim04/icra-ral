import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "82_plan_adapter_strategy_action_scale_diagnostics.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for adapter strategy diagnostics planner tests")
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


def _make_inputs(tmp_path, *, adapter_metadata_present=True):
    audit = tmp_path / "audit.json"
    metric = tmp_path / "metric.json"
    comparison = tmp_path / "comparison.json"
    source = tmp_path / "rollout.py"
    _write_json(audit, {"ready_for_adapter_strategy_diagnosis": True})
    _write_json(
        metric,
        {
            "metric_summary": {
                "adapter_metadata_present": adapter_metadata_present,
                "action_adapter_strategies": ["policy_6d_delta_pose_plus_gripper_zero_hold"],
                "state_adapters": ["diagnostic_eef_pos_quat_xyz_6d_state_adapter"],
                "image_source_keys": {"observation.images.camera1": "agentview_image"},
                "diagnostic_success_rate": 0.0,
                "reward_sum_total": 0.0,
                "last_env_action_max_abs": 0.8,
                "last_env_action_gripper_component": 0.0,
            }
        },
    )
    _write_json(
        comparison,
        {
            "ready_for_adapter_strategy_diagnosis": True,
            "comparison": {"adapter_wiring_clean": True},
        },
    )
    source.write_text("adapt_policy_action_to_env_action(policy_action, action_dim)\n", encoding="utf-8")
    return audit, metric, comparison, source


def _run_plan(tmp_path, *, adapter_metadata_present=True, extra_env=None):
    audit, metric, comparison, source = _make_inputs(
        tmp_path,
        adapter_metadata_present=adapter_metadata_present,
    )
    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-ActionAuditReportPath",
            str(audit),
            "-MetricSummaryReportPath",
            str(metric),
            "-ZeroComparisonReportPath",
            str(comparison),
            "-RolloutBridgeSourcePath",
            str(source),
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


def test_adapter_strategy_action_scale_plan_goes_green(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["adapter_strategy_action_scale_diagnostics_plan_passed"] is True
    assert report["ready_for_adapter_strategy_diagnostic_runner_implementation"] is True
    assert report["ready_for_rollout_scaling"] is False
    assert report["inputs"]["adapter_metadata_present"] is True
    assert report["diagnostic_plan"]["max_tasks"] == 1
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_adapter_strategy_action_scale_plan_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_plan(
        tmp_path,
        extra_env={"ALLOW_BOUNDED_LEARNED_POLICY_MATRIX": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_BOUNDED_LEARNED_POLICY_MATRIX" in reason for reason in report["stop_reasons"])


def test_adapter_strategy_action_scale_plan_requires_adapter_metadata(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, adapter_metadata_present=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["adapter_strategy_action_scale_diagnostics_plan_passed"] is False
    assert any("Adapter metadata is missing" in reason for reason in report["stop_reasons"])
