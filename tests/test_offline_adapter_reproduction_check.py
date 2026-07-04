import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "95_check_offline_adapter_reproduction.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for offline adapter reproduction tests")
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
        "ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC",
        "ALLOW_ACTION_SCALE_DIAGNOSTIC",
        "ALLOW_PROMPT_FORMAT_DIAGNOSTIC",
        "ALLOW_CAMERA_SOURCE_DIAGNOSTIC",
        "ALLOW_STATE_SUFFICIENCY_DIAGNOSTIC",
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_inputs(tmp_path, *, include_hdf5_audit=True):
    h5py = pytest.importorskip("h5py")
    smolvla = tmp_path / "smolvla"
    libero_data = tmp_path / "data" / "libero"
    hdf5_path = libero_data / "libero_10" / "demo_task_demo.hdf5"
    hdf5_audit = tmp_path / "hdf5_audit.json"

    _write_json(
        smolvla / "config.json",
        {
            "input_features": {
                "observation.state": {"type": "STATE", "shape": [6]},
                "observation.images.camera1": {"type": "VISUAL", "shape": [3, 256, 256]},
                "observation.images.camera2": {"type": "VISUAL", "shape": [3, 256, 256]},
                "observation.images.camera3": {"type": "VISUAL", "shape": [3, 256, 256]},
            },
            "output_features": {"action": {"type": "ACTION", "shape": [6]}},
        },
    )
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        actions = np.zeros((4, 7), dtype=np.float64)
        actions[0] = np.array([0.0, 0.05, -0.02, 0.0, 0.0, 0.0, -1.0])
        actions[1:, -1] = -1.0
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("rewards", data=np.zeros((4,), dtype=np.uint8))
        demo.create_dataset("dones", data=np.array([0, 0, 0, 1], dtype=np.uint8))
        obs = demo.create_group("obs")
        ee_pos = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        ee_ori = np.array([0.1, 0.2, 0.3], dtype=np.float64)
        obs.create_dataset("ee_pos", data=np.tile(ee_pos, (4, 1)))
        obs.create_dataset("ee_ori", data=np.tile(ee_ori, (4, 1)))
        obs.create_dataset("ee_states", data=np.tile(np.concatenate([ee_pos, ee_ori]), (4, 1)))
        obs.create_dataset("agentview_rgb", data=np.zeros((4, 128, 128, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.zeros((4, 128, 128, 3), dtype=np.uint8))
    if include_hdf5_audit:
        _write_json(
            hdf5_audit,
            {
                "decision": "no_go_rollout_scaling",
                "high_severity_issue_count": 2,
                "ready_for_rollout_scaling": False,
            },
        )
    return smolvla, libero_data, hdf5_path, hdf5_audit


def _run_check(tmp_path, *, include_hdf5_audit=True, extra_env=None):
    smolvla, libero_data, hdf5_path, hdf5_audit = _make_inputs(
        tmp_path,
        include_hdf5_audit=include_hdf5_audit,
    )
    json_report = tmp_path / "repro.json"
    md_report = tmp_path / "repro.md"
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
            str(libero_data),
            "-Hdf5Path",
            str(hdf5_path),
            "-Hdf5AuditPath",
            str(hdf5_audit),
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


def test_offline_adapter_reproduction_prefers_close_for_negative_gripper(tmp_path):
    result, report, json_report, md_report = _run_check(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["offline_adapter_reproduction_check_passed"] is True
    assert report["decision"] == "no_go_rollout_scaling"
    assert report["ready_for_rollout_scaling"] is False
    assert report["reproduction"]["best_action_adapter_strategy_for_first_demo_action"].endswith("gripper_close")
    assert report["reproduction"]["state_reproduction"]["rebuilt_matches_ee_states"] is True
    assert any(issue["axis"] == "gripper_adapter_strategy" and issue["severity"] == "high" for issue in report["issues"])
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_offline_adapter_reproduction_stops_when_hdf5_audit_missing(tmp_path):
    result, report, _, _ = _run_check(tmp_path, include_hdf5_audit=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["offline_adapter_reproduction_check_passed"] is False
    assert any("Missing JSON file" in reason for reason in report["stop_reasons"])


def test_offline_adapter_reproduction_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_check(
        tmp_path,
        extra_env={"ALLOW_HEAVY_IMPORT": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_HEAVY_IMPORT" in reason for reason in report["stop_reasons"])
    assert report["policy"]["model_inference_performed"] is False
