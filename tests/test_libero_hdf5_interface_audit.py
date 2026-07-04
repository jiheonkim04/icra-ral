import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "94_audit_libero_hdf5_interface.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for HDF5 interface audit tests")
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


def _make_inputs(tmp_path, *, include_compat=True):
    h5py = pytest.importorskip("h5py")
    smolvla = tmp_path / "smolvla"
    libero = tmp_path / "LIBERO"
    libero_data = tmp_path / "data" / "libero"
    hdf5_path = libero_data / "libero_10" / "demo_task_demo.hdf5"
    compat = tmp_path / "compat.json"

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
    bddl = libero / "libero" / "libero" / "bddl_files" / "libero_10" / "demo_task.bddl"
    bddl.parent.mkdir(parents=True, exist_ok=True)
    bddl.write_text("(:language pick up the mug)\n", encoding="utf-8")
    hdf5_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.create_dataset("actions", data=np.zeros((4, 7), dtype=np.float64))
        demo.create_dataset("rewards", data=np.zeros((4,), dtype=np.uint8))
        demo.create_dataset("dones", data=np.array([0, 0, 0, 1], dtype=np.uint8))
        demo.create_dataset("states", data=np.zeros((4, 47), dtype=np.float64))
        demo.create_dataset("robot_states", data=np.zeros((4, 9), dtype=np.float64))
        obs = demo.create_group("obs")
        obs.create_dataset("ee_states", data=np.zeros((4, 6), dtype=np.float64))
        obs.create_dataset("ee_pos", data=np.zeros((4, 3), dtype=np.float64))
        obs.create_dataset("ee_ori", data=np.zeros((4, 3), dtype=np.float64))
        obs.create_dataset("agentview_rgb", data=np.zeros((4, 128, 128, 3), dtype=np.uint8))
        obs.create_dataset("eye_in_hand_rgb", data=np.zeros((4, 128, 128, 3), dtype=np.uint8))
    if include_compat:
        _write_json(
            compat,
            {
                "decision": "no_go_rollout_scaling",
                "high_severity_issue_count": 4,
                "ready_for_rollout_scaling": False,
            },
        )
    return smolvla, libero, libero_data, hdf5_path, compat


def _run_audit(tmp_path, *, include_compat=True, extra_env=None):
    smolvla, libero, libero_data, hdf5_path, compat = _make_inputs(tmp_path, include_compat=include_compat)
    json_report = tmp_path / "audit.json"
    md_report = tmp_path / "audit.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-SmolVLACkpt",
            str(smolvla),
            "-LiberoRoot",
            str(libero),
            "-LiberoDataRoot",
            str(libero_data),
            "-Hdf5Path",
            str(hdf5_path),
            "-CompatibilityAuditPath",
            str(compat),
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


def test_hdf5_interface_audit_detects_action_and_camera_gaps(tmp_path):
    result, report, json_report, md_report = _run_audit(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["libero_hdf5_interface_audit_passed"] is True
    assert report["decision"] == "no_go_rollout_scaling"
    assert report["ready_for_rollout_scaling"] is False
    assert report["hdf5_summary"]["action_dim"] == 7
    assert report["policy_config"]["action_shape"] == [6]
    assert any(issue["axis"] == "action_dimension" and issue["severity"] == "high" for issue in report["issues"])
    assert any(issue["axis"] == "camera_count" for issue in report["issues"])
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_hdf5_interface_audit_stops_when_compat_missing(tmp_path):
    result, report, _, _ = _run_audit(tmp_path, include_compat=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["libero_hdf5_interface_audit_passed"] is False
    assert any("Missing JSON file" in reason for reason in report["stop_reasons"])


def test_hdf5_interface_audit_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_audit(
        tmp_path,
        extra_env={"ALLOW_ROLLOUT": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_ROLLOUT" in reason for reason in report["stop_reasons"])
    assert report["policy"]["simulator_environment_created"] is False
