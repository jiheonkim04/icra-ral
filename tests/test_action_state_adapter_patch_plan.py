import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "80_plan_action_state_adapter_patch.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for action/state adapter patch plan tests")
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


def _make_inputs(tmp_path):
    audit = tmp_path / "audit.json"
    comparison = tmp_path / "comparison.json"
    ckpt = tmp_path / "ckpt"
    source = tmp_path / "bridge.py"
    ckpt.mkdir()
    _write_json(
        audit,
        {
            "action_interface_metadata_audit_passed": True,
            "high_priority_findings": [
                "action_dim_mismatch",
                "gripper_constant_zero",
                "state_truncation_risk",
                "camera_feature_name_mismatch",
            ],
        },
    )
    _write_json(
        comparison,
        {
            "zero_action_policy_diagnostic_comparison_passed": True,
            "ready_for_action_state_adapter_patch_plan": True,
            "comparison": {
                "policy_action_nontrivial": True,
                "learned_policy_outperformed_zero_action": False,
                "zero_action": {"reward_sum_total": 0.0},
                "learned_policy": {
                    "reward_sum_total": 0.0,
                    "policy_action_shapes": [[1, 6]],
                    "env_action_dims": [7],
                    "last_env_action_gripper_component": 0.0,
                },
            },
        },
    )
    _write_json(
        ckpt / "config.json",
        {
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        },
    )
    _write_json(
        ckpt / "policy_preprocessor.json",
        {
            "steps": [
                {
                    "registry_name": "normalizer_processor",
                    "config": {
                        "features": {
                            "observation.state": {"type": "STATE", "shape": [6]},
                            "observation.image": {"type": "VISUAL", "shape": [3, 256, 256]},
                            "action": {"type": "ACTION", "shape": [6]},
                        }
                    },
                }
            ]
        },
    )
    _write_json(
        ckpt / "policy_postprocessor.json",
        {
            "steps": [
                {
                    "registry_name": "unnormalizer_processor",
                    "config": {"features": {"action": {"type": "ACTION", "shape": [6]}}},
                }
            ]
        },
    )
    source.write_text(
        "values.extend([0.0] * (action_dim - len(values)))\nreturn values[:action_dim]\nvalues = values[:dim]\ndef _select_image_array():\n    return 'agentview_image'\n",
        encoding="utf-8",
    )
    return audit, comparison, ckpt, source


def _run_plan(tmp_path, audit=None, comparison=None, ckpt=None, source=None, extra_env=None):
    if audit is None:
        audit, comparison, ckpt, source = _make_inputs(tmp_path)
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
            "-ZeroPolicyComparisonPath",
            str(comparison),
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


def test_action_state_adapter_patch_plan_requires_pure_adapter_work(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["action_state_adapter_patch_plan_passed"] is True
    assert report["ready_for_pure_adapter_implementation"] is True
    assert report["ready_for_rollout_scaling"] is False
    assert report["patch_plan"]["action_adapter"]["required"] is True
    assert report["patch_plan"]["state_adapter"]["required"] is True
    assert report["patch_plan"]["camera_adapter"]["required"] is True
    assert "silent zero padding without report metadata" in report["patch_plan"]["action_adapter"]["forbidden_shortcuts"]
    assert report["claims"]["paper_grade_claim_made"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_action_state_adapter_patch_plan_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_plan(
        tmp_path,
        extra_env={"ALLOW_BOUNDED_LEARNED_POLICY_MATRIX": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_BOUNDED_LEARNED_POLICY_MATRIX" in reason for reason in report["stop_reasons"])


def test_action_state_adapter_patch_plan_handles_missing_comparison(tmp_path):
    audit, comparison, ckpt, source = _make_inputs(tmp_path)
    comparison.unlink()

    result, report, json_report, md_report = _run_plan(tmp_path, audit, comparison, ckpt, source)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("Missing input" in reason for reason in report["stop_reasons"])
    assert json_report.exists()
    assert md_report.exists()
