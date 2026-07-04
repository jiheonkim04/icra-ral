import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "78_audit_action_interface_metadata.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for action-interface metadata audit tests")
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


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_inputs(tmp_path):
    plan = tmp_path / "plan.json"
    metric = tmp_path / "metric.json"
    ckpt = tmp_path / "ckpt"
    source = tmp_path / "rollout.py"
    ckpt.mkdir()
    _write_json(plan, {"observed_signals": {"policy_action_dim": 6, "env_action_dim": 7, "gripper_component": 0.0, "diagnostic_success_rate": 0.0, "reward_sum_total": 0.0, "action_max_abs": 0.7, "action_l2": 1.1}})
    _write_json(metric, {"metric_summary": {"diagnostic_success_rate": 0.0, "reward_sum_total": 0.0}})
    _write_json(
        ckpt / "config.json",
        {
            "input_features": {"observation.state": {"shape": [6]}, "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]}},
            "output_features": {"action": {"shape": [6]}},
            "normalization_mapping": {"ACTION": "MEAN_STD"},
        },
    )
    _write_json(
        ckpt / "policy_preprocessor.json",
        {"steps": [{"registry_name": "normalizer_processor", "config": {"features": {"observation.state": {"shape": [6]}, "action": {"shape": [6]}}}}]},
    )
    _write_json(
        ckpt / "policy_postprocessor.json",
        {"steps": [{"registry_name": "unnormalizer_processor", "config": {"features": {"action": {"shape": [6]}}}}]},
    )
    source.write_text(
        'keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos", "robot0_joint_vel"]\nvalues = values[:dim]\n',
        encoding="utf-8",
    )
    return plan, metric, ckpt, source


def _run_audit(tmp_path, extra_env=None, missing_plan=False):
    plan, metric, ckpt, source = _make_inputs(tmp_path)
    if missing_plan:
        plan.unlink()
    json_report = tmp_path / "audit.json"
    md_report = tmp_path / "audit.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-PlanReportPath",
            str(plan),
            "-MetricSummaryReportPath",
            str(metric),
            "-SmolVlaCkptPath",
            str(ckpt),
            "-SourceFilePath",
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


def test_action_interface_metadata_audit_finds_high_priority_mismatches(tmp_path):
    result, report, json_report, md_report = _run_audit(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["action_interface_metadata_audit_passed"] is True
    assert "action_dim_mismatch" in report["high_priority_findings"]
    assert "gripper_constant_zero" in report["high_priority_findings"]
    assert "state_truncation_risk" in report["high_priority_findings"]
    assert report["ready_for_zero_action_vs_policy_action_diagnostic"] is True
    assert report["ready_for_action_adapter_patch_plan"] is True
    assert report["policy"]["rollouts_performed"] is False
    assert report["evidence_policy"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_action_interface_metadata_audit_recognizes_wired_adapter_strategy_diagnosis(tmp_path):
    plan, metric, ckpt, source = _make_inputs(tmp_path)
    source.write_text(
        "from tca_map.smolvla.interface_adapters import (\n"
        "    adapt_policy_action_to_env_action,\n"
        "    adapt_observation_state,\n"
        "    select_image_source,\n"
        ")\n"
        "adapt_policy_action_to_env_action(policy_action, action_dim)\n"
        "adapt_observation_state(obs, fields, dim)\n"
        "select_image_source(obs, feature_key)\n",
        encoding="utf-8",
    )
    json_report = tmp_path / "audit.json"
    md_report = tmp_path / "audit.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-PlanReportPath",
            str(plan),
            "-MetricSummaryReportPath",
            str(metric),
            "-SmolVlaCkptPath",
            str(ckpt),
            "-SourceFilePath",
            str(source),
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(md_report),
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    start = result.stdout.find("{")
    assert start >= 0, result.stdout + result.stderr
    report = json.loads(result.stdout[start:])

    assert result.returncode == 0, result.stderr
    assert "action_dim_mismatch_explicit_adapter_in_use" in report["high_priority_findings"]
    assert "gripper_zero_hold_strategy_requires_validation" in report["high_priority_findings"]
    assert report["ready_for_action_adapter_patch_plan"] is False
    assert report["ready_for_adapter_strategy_diagnosis"] is True
    assert "adapter-strategy/action-scale" in report["recommended_next_step"]


def test_action_interface_metadata_audit_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_audit(tmp_path, extra_env={"ALLOW_BOUNDED_LEARNED_POLICY_MATRIX": "1"})

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_BOUNDED_LEARNED_POLICY_MATRIX" in reason for reason in report["stop_reasons"])


def test_action_interface_metadata_audit_stops_when_plan_missing(tmp_path):
    result, report, _, _ = _run_audit(tmp_path, missing_plan=True)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("action-interface plan" in reason for reason in report["stop_reasons"])
