import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "97_audit_hdf5_rollout_alignment.ps1"
CLOSE = "policy_6d_delta_pose_plus_gripper_close"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for HDF5 rollout alignment audit tests")
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
        "ALLOW_GRIPPER_CLOSE_COMPAT_DIAGNOSTIC",
        "ALLOW_HDF5_REPLAY_DIAGNOSTIC",
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_hdf5(path):
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        demo.attrs["init_state"] = np.arange(47, dtype=np.float64)
        demo.attrs["model_file"] = "<mujoco/>"
        actions = np.zeros((4, 7), dtype=np.float64)
        actions[0] = np.array([0.0, 0.05, -0.02, 0.0, 0.0, 0.0, -1.0])
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("states", data=np.zeros((4, 47), dtype=np.float64))
        demo.create_dataset("rewards", data=np.zeros((4,), dtype=np.uint8))
        demo.create_dataset("dones", data=np.array([0, 0, 0, 1], dtype=np.uint8))
        obs = demo.create_group("obs")
        obs.create_dataset("ee_states", data=np.zeros((4, 6), dtype=np.float64))
        obs.create_dataset("agentview_rgb", data=np.zeros((4, 128, 128, 3), dtype=np.uint8))
    return path


def _make_inputs(tmp_path, *, rollout_task_name="demo_task"):
    hdf5_path = _make_hdf5(tmp_path / "data" / "demo_task_demo.hdf5")
    gripper_plan = tmp_path / "gripper_plan.json"
    repro = tmp_path / "repro.json"
    previous = tmp_path / "previous.json"
    hdf5_audit = tmp_path / "hdf5_audit.json"
    source = tmp_path / "rollout.py"
    _write_json(gripper_plan, {"decision": "reduce_scope"})
    _write_json(
        repro,
        {
            "offline_adapter_reproduction_check_passed": True,
            "paths": {"hdf5_path": str(hdf5_path)},
        },
    )
    _write_json(
        previous,
        {
            "variants": [
                {
                    "strategy": CLOSE,
                    "passed": True,
                    "diagnostic_success_rate": 0.0,
                    "reward_sum": 0.0,
                    "inner_report": {
                        "tasks": [
                            {
                                "task_name": rollout_task_name,
                                "bddl_file": "/tmp/demo_task.bddl",
                                "success_check": False,
                                "reward_sum": 0.0,
                            }
                        ]
                    },
                }
            ]
        },
    )
    _write_json(hdf5_audit, {"decision": "no_go_rollout_scaling"})
    source.write_text("obs = env.reset()\n", encoding="utf-8")
    return gripper_plan, repro, previous, hdf5_audit, source


def _run_audit(tmp_path, *, rollout_task_name="demo_task", extra_env=None):
    gripper_plan, repro, previous, hdf5_audit, source = _make_inputs(
        tmp_path,
        rollout_task_name=rollout_task_name,
    )
    json_report = tmp_path / "alignment.json"
    md_report = tmp_path / "alignment.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-GripperClosePlanReportPath",
            str(gripper_plan),
            "-OfflineReproductionReportPath",
            str(repro),
            "-PreviousAdapterStrategyReportPath",
            str(previous),
            "-Hdf5AuditReportPath",
            str(hdf5_audit),
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


def test_hdf5_rollout_alignment_audit_reduces_scope_for_init_state_gap(tmp_path):
    result, report, json_report, md_report = _run_audit(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "reduce_scope"
    assert report["hdf5_rollout_alignment_audit_passed"] is True
    assert report["alignment"]["task_name_matches"] is True
    assert report["hdf5_demo"]["init_state_present"] is True
    assert report["rollout_bridge"]["source_sets_hdf5_initial_state"] is False
    assert report["previous_close_diagnostic"]["duplicate_zero_signal"] is True
    assert report["ready_for_hdf5_initial_state_replay_plan"] is True
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_hdf5_rollout_alignment_audit_stops_on_task_mismatch(tmp_path):
    result, report, _, _ = _run_audit(tmp_path, rollout_task_name="different_task")

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["hdf5_rollout_alignment_audit_passed"] is False
    assert any("do not match" in reason for reason in report["stop_reasons"])


def test_hdf5_rollout_alignment_audit_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_audit(
        tmp_path,
        extra_env={"ALLOW_HDF5_REPLAY_DIAGNOSTIC": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_HDF5_REPLAY_DIAGNOSTIC" in reason for reason in report["stop_reasons"])
