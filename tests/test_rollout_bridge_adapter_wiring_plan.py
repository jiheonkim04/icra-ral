import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "81_plan_rollout_bridge_adapter_wiring.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for rollout bridge wiring planner tests")
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
    patch_plan = tmp_path / "patch_plan.json"
    single = tmp_path / "single.json"
    bridge = tmp_path / "bridge.py"
    adapter = tmp_path / "adapter.py"
    _write_json(
        patch_plan,
        {
            "action_state_adapter_patch_plan_passed": True,
            "ready_for_pure_adapter_implementation": True,
        },
    )
    _write_json(
        single,
        {
            "policy": {"adapter_metadata_recorded": True},
            "interface": {
                "adapter_metadata": {
                    "action_adapter": {"adapter_mode": "policy_6d_delta_pose_plus_gripper_zero_hold"},
                    "state_adapter": {"adapter": "diagnostic_eef_pos_quat_xyz_6d_state_adapter"},
                }
            },
        },
    )
    bridge.write_text(
        "values.extend([0.0] * (action_dim - len(values)))\nreturn values[:action_dim]\nvalues = values[:dim]\ndef _select_image_array():\n    pass\n",
        encoding="utf-8",
    )
    adapter.write_text(
        "def adapt_policy_action_to_env_action(): pass\n"
        "def adapt_observation_state(): pass\n"
        "def select_image_source(): pass\n",
        encoding="utf-8",
    )
    return patch_plan, single, bridge, adapter


def _run_plan(tmp_path, patch_plan=None, single=None, bridge=None, adapter=None, extra_env=None):
    if patch_plan is None:
        patch_plan, single, bridge, adapter = _make_inputs(tmp_path)
    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-PatchPlanReportPath",
            str(patch_plan),
            "-SingleSampleReportPath",
            str(single),
            "-RolloutBridgeSourcePath",
            str(bridge),
            "-AdapterSourcePath",
            str(adapter),
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


def test_rollout_bridge_adapter_wiring_plan_goes_green(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["rollout_bridge_adapter_wiring_plan_passed"] is True
    assert report["ready_for_rollout_bridge_adapter_wiring"] is True
    assert report["ready_for_rollout_execution"] is False
    assert report["source_audit"]["bridge_needs_action_adapter"] is True
    assert report["source_audit"]["bridge_needs_state_adapter"] is True
    assert report["source_audit"]["bridge_needs_image_adapter"] is True
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_rollout_bridge_adapter_wiring_plan_accepts_already_wired_bridge(tmp_path):
    patch_plan, single, bridge, adapter = _make_inputs(tmp_path)
    bridge.write_text(
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

    result, report, _, _ = _run_plan(tmp_path, patch_plan, single, bridge, adapter)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["rollout_bridge_adapter_wiring_plan_passed"] is True
    assert report["ready_for_rollout_bridge_adapter_wiring"] is False
    assert report["rollout_bridge_adapter_wiring_complete"] is True
    assert report["ready_for_rollout_execution"] is False


def test_rollout_bridge_adapter_wiring_plan_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_plan(
        tmp_path,
        extra_env={"ALLOW_BOUNDED_LEARNED_POLICY_MATRIX": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_BOUNDED_LEARNED_POLICY_MATRIX" in reason for reason in report["stop_reasons"])


def test_rollout_bridge_adapter_wiring_plan_requires_single_sample_metadata(tmp_path):
    patch_plan, single, bridge, adapter = _make_inputs(tmp_path)
    _write_json(single, {"policy": {"adapter_metadata_recorded": False}, "interface": {}})

    result, report, _, _ = _run_plan(tmp_path, patch_plan, single, bridge, adapter)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("Single-sample adapter metadata" in reason for reason in report["stop_reasons"])
