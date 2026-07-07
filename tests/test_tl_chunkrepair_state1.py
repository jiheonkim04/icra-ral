import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tca_map.tl_chunkrepair import state1_diagnostic as tl


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "181_tl_chunkrepair_state1_diagnostic.ps1"


def _base_actions() -> np.ndarray:
    actions = np.zeros((16, 7), dtype=np.float64)
    actions[:, 6] = -1.0
    actions[4:13, 6] = 1.0
    actions[8:13, 2] = 0.05
    actions[13:, 6] = -1.0
    return actions


def _write_demo(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    actions_np = _base_actions()
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        demo.attrs["num_samples"] = int(actions_np.shape[0])
        demo.attrs["model_file"] = "<xml/>"
        actions = demo.create_dataset("actions", shape=actions_np.shape, dtype="f4")
        rewards = demo.create_dataset("rewards", shape=(actions_np.shape[0],), dtype="f4")
        dones = demo.create_dataset("dones", shape=(actions_np.shape[0],), dtype="i1")
        states = demo.create_dataset("states", shape=(actions_np.shape[0], 47), dtype="f4")
        obs = demo.create_group("obs")
        ee_pos = obs.create_dataset("ee_pos", shape=(actions_np.shape[0], 3), dtype="f4")
        obs.create_dataset("gripper_states", shape=(actions_np.shape[0], 2), dtype="f4")
        actions[:, :] = actions_np
        for step in range(actions_np.shape[0]):
            rewards[step] = 1.0 if step == actions_np.shape[0] - 2 else 0.0
            dones[step] = 1 if step == actions_np.shape[0] - 2 else 0
            states[step, :] = 0.0
            ee_pos[step, :] = [0.1, 0.2, 0.3 + 0.01 * max(0, step - 8)]


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    demo_path = tmp_path / "data" / "libero_10" / "task_demo.hdf5"
    _write_demo(demo_path)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "ready_for_tiny_offline_counterfactual_split": True,
                "counterfactual_pairs": [
                    {
                        "pair_id": "libero_10:task__vs__counter",
                        "suite": "libero_10",
                        "positive_task_id": "task",
                        "counterfactual_task_id": "counter",
                        "positive_instruction": "put the object in the basket",
                        "counterfactual_instruction": "put another object in the drawer",
                        "positive_demo_file": str(demo_path),
                        "counterfactual_demo_file": str(demo_path),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    readiness = tmp_path / "readiness.json"
    readiness.write_text(json.dumps({"risk_gate_status": "green", "rollout_diagnostic_authorized": True}), encoding="utf-8")
    return manifest, readiness


def test_monitor_detects_release_and_open_transport():
    actions = _base_actions()
    actions[9:13, 6] = -1.0

    report = tl.monitor_chunk(actions, {"gripper_close_index": 4, "lift_index": 8, "safe_release_index": 13})

    assert report["violation_count"] >= 2
    assert report["violations"]["keep_grasp_until_placement"] is True
    assert report["violations"]["do_not_move_object_while_gripper_open"] is True


def test_tl_repair_reduces_temporal_violations_and_edits_locally():
    actions = _base_actions()
    actions[9:13, 6] = -1.0

    repaired, info = tl.repair_tl_chunk(actions, {"gripper_close_index": 4, "lift_index": 8, "safe_release_index": 13})

    assert info["input_monitor"]["violation_count"] > info["output_monitor"]["violation_count"]
    assert np.all(repaired[8:13, 6] >= 0.0)
    assert tl._edit_metrics(actions, repaired)["changed_steps"] > 0


def test_build_case_creates_requested_temporal_perturbations(tmp_path):
    manifest, _readiness = _fixture(tmp_path)

    case = tl.build_tl_chunkrepair_case(manifest, max_steps_cap=16, post_signal_margin=1, offset_steps=3)

    assert set(tl.PERTURBATION_NAMES) == set(case["perturbations"])
    assert case["hdf5_eef_source"]["available"] is True
    raw = case["perturbations"]["inserted_unsafe_contact_action"]["actions"]
    assert tl.monitor_chunk(raw, case["event_anchors"])["violations"]["avoid_forbidden_contact_before_safe_phase"] is True


def test_tl_chunkrepair_script_requires_task_local_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for TL-ChunkRepair script tests")
    manifest, readiness = _fixture(tmp_path)
    report_json = tmp_path / "report.json"
    env = os.environ.copy()
    env.pop(tl.TASK_GATE, None)
    for gate in tl.FORBIDDEN_GATES:
        env.pop(gate, None)

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-ManifestPath",
            str(manifest),
            "-ReadinessReportPath",
            str(readiness),
            "-JsonReportPath",
            str(report_json),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
            "-MaxStepsCap",
            "16",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    data = json.loads(report_json.read_text(encoding="utf-8-sig"))
    assert data["policy"]["replay_or_rollout_performed"] is False
    assert f"{tl.TASK_GATE}=1 is required" in data["result"]["blocked_reason"]
