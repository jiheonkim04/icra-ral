import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "105_plan_offline_demo_conditioned_action_decoding.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for offline decoding plan tests")
    return exe


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_ROLLOUT",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_RUNTIME_INSTALL",
        "ALLOW_SIMULATOR_IMPORT_SMOKE",
        "ALLOW_SIMULATOR_RENDER_SMOKE",
        "ALLOW_SIMULATOR_RESET_STEP",
        "ALLOW_TINY_ROLLOUT",
        "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_inputs(tmp_path, *, hdf5_exists=True, alignment_ready=True):
    smolvla = tmp_path / "smolvla"
    data = tmp_path / "data" / "libero"
    reports = tmp_path / "reports"
    hdf5_path = data / "libero_10" / "task_demo.hdf5"
    if hdf5_exists:
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        hdf5_path.write_bytes(b"marker")
    for name in (
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "model.safetensors",
    ):
        (smolvla / name).parent.mkdir(parents=True, exist_ok=True)
        (smolvla / name).write_text("{}", encoding="utf-8")
    _write_json(
        reports / "alignment.json",
        {
            "smolvla_libero_checkpoint_task_alignment_audit_passed": alignment_ready,
            "ready_for_offline_demonstration_conditioned_action_decoding_plan": alignment_ready,
            "task_summary": {
                "hdf5_path": str(hdf5_path),
                "selected_task_name": "task",
                "selected_bddl_language": "do the task",
            },
            "checkpoint_summary": {"action_shape": [6]},
            "evidence_summary": {
                "hdf5_action_dim": 7,
                "best_gripper_strategy_for_first_demo_action": "policy_6d_delta_pose_plus_gripper_close",
            },
        },
    )
    _write_json(
        reports / "hdf5.json",
        {"paths": {"hdf5_path": str(hdf5_path)}, "hdf5_summary": {"action_dim": 7}},
    )
    _write_json(
        reports / "offline.json",
        {
            "paths": {"hdf5_path": str(hdf5_path)},
            "reproduction": {
                "best_action_adapter_strategy_for_first_demo_action": "policy_6d_delta_pose_plus_gripper_close"
            },
        },
    )
    return smolvla, data, reports


def _run_plan(tmp_path, *, hdf5_exists=True, alignment_ready=True, extra_env=None):
    smolvla, data, reports = _make_inputs(
        tmp_path,
        hdf5_exists=hdf5_exists,
        alignment_ready=alignment_ready,
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
            "-SmolVLACkpt",
            str(smolvla),
            "-LiberoDataRoot",
            str(data),
            "-AlignmentAuditPath",
            str(reports / "alignment.json"),
            "-Hdf5AuditPath",
            str(reports / "hdf5.json"),
            "-OfflineAdapterReportPath",
            str(reports / "offline.json"),
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


def test_offline_demo_conditioned_action_decoding_plan_proceeds(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["offline_demo_conditioned_action_decoding_plan_passed"] is True
    assert report["decision"] == "proceed"
    assert report["ready_for_bounded_offline_demo_action_decoding_runner"] is True
    assert report["ready_for_rollout_scaling"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["risk_assessment"]["future_runner_model_inference"] is True
    assert report["risk_assessment"]["simulator_will_run"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_offline_demo_conditioned_action_decoding_plan_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_plan(
        tmp_path,
        extra_env={"ALLOW_OFFLINE_DEMO_ACTION_DECODING": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["offline_demo_conditioned_action_decoding_plan_passed"] is False
    assert "ALLOW_OFFLINE_DEMO_ACTION_DECODING" in report["risk_assessment"]["reason"]
    assert report["policy"]["model_load_performed"] is False


def test_offline_demo_conditioned_action_decoding_plan_stops_when_hdf5_missing(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path, hdf5_exists=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["offline_demo_conditioned_action_decoding_plan_passed"] is False
    assert any("HDF5" in reason for reason in report["stop_reasons"])
    assert json_report.exists()
    assert md_report.exists()
