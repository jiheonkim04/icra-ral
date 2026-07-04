import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "109_plan_repeated_offline_demo_action_decoding.ps1"
PYTHON = r"C:\Users\jiheo\miniconda3\envs\tca_map\python.exe"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for repeated offline decoding plan tests")
    return exe


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
        "ALLOW_REPEATED_OFFLINE_DEMO_DECODING",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_ROLLOUT",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_RUNTIME_INSTALL",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_hdf5(path, *, timesteps=5):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((timesteps, 7), dtype=np.float32))
        demo.create_dataset("states", data=np.zeros((timesteps, 16), dtype=np.float32))
        demo.attrs["init_state"] = np.zeros((16,), dtype=np.float32)
        obs = demo.create_group("obs")
        obs.create_dataset("ee_states", data=np.zeros((timesteps, 6), dtype=np.float32))
        obs.create_dataset("agentview_rgb", data=np.zeros((timesteps, 8, 8, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.zeros((timesteps, 8, 8, 3), dtype=np.uint8))


def _make_inputs(tmp_path, *, audit_ready=True, hdf5_exists=True):
    ckpt = tmp_path / "smolvla"
    for name in (
        "config.json",
        "policy_preprocessor.json",
        "policy_postprocessor.json",
        "model.safetensors",
        "policy_preprocessor_step_5_normalizer_processor.safetensors",
        "policy_postprocessor_step_0_unnormalizer_processor.safetensors",
    ):
        (ckpt / name).parent.mkdir(parents=True, exist_ok=True)
        (ckpt / name).write_text("{}", encoding="utf-8")
    hdf5_path = tmp_path / "data" / "libero_10" / "task_demo.hdf5"
    if hdf5_exists:
        _write_hdf5(hdf5_path)
    audit = tmp_path / "audit.json"
    offline = tmp_path / "offline.json"
    _write_json(
        audit,
        {
            "vlm_loading_policy_action_normalization_audit_passed": audit_ready,
            "ready_for_repeated_offline_decoding_plan": audit_ready,
            "checkpoint_summary": {
                "config_load_vlm_weights": True,
                "observed_load_vlm_weights": False,
                "config_normalization_mapping": {"ACTION": "MEAN_STD"},
            },
            "offline_alignment_summary": {
                "sample": {"hdf5_path": str(hdf5_path)},
                "action_adapter_metadata": {"strategy": "policy_6d_delta_pose_plus_gripper_close"},
            },
        },
    )
    _write_json(offline, {"sample": {"hdf5_path": str(hdf5_path)}})
    return ckpt, audit, offline


def _run_plan(tmp_path, *, audit_ready=True, hdf5_exists=True, extra_env=None):
    ckpt, audit, offline = _make_inputs(tmp_path, audit_ready=audit_ready, hdf5_exists=hdf5_exists)
    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            PYTHON,
            "-SmolVLACkpt",
            str(ckpt),
            "-VlmActionAuditPath",
            str(audit),
            "-OfflineDecodingReportPath",
            str(offline),
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


def test_repeated_offline_demo_action_decoding_plan_proceeds(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["repeated_offline_demo_action_decoding_plan_passed"] is True
    assert report["decision"] == "proceed"
    assert report["ready_for_bounded_repeated_offline_demo_action_decoding_runner"] is True
    assert report["ready_for_rollout_scaling"] is False
    assert report["risk_assessment"]["planned_policy_inference_calls"] == 3
    assert report["planned_sample"]["hdf5"]["selected_timesteps"] == [0, 2, 4]
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_repeated_offline_demo_action_decoding_plan_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_plan(
        tmp_path,
        extra_env={"ALLOW_REPEATED_OFFLINE_DEMO_DECODING": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["repeated_offline_demo_action_decoding_plan_passed"] is False
    assert any("ALLOW_REPEATED_OFFLINE_DEMO_DECODING" in reason for reason in report["stop_reasons"])
    assert report["policy"]["model_load_performed"] is False


def test_repeated_offline_demo_action_decoding_plan_stops_without_audit_ready(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path, audit_ready=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["repeated_offline_demo_action_decoding_plan_passed"] is False
    assert any("audit" in reason.lower() for reason in report["stop_reasons"])
    assert json_report.exists()
    assert md_report.exists()
