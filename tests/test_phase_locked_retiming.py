import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tca_map.phase_locked import retiming


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "180_phase_locked_retiming_diagnostic.ps1"


def _write_demo(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        demo.attrs["num_samples"] = 12
        demo.attrs["model_file"] = "<xml/>"
        actions = demo.create_dataset("actions", shape=(12, 7), dtype="f4")
        rewards = demo.create_dataset("rewards", shape=(12,), dtype="f4")
        dones = demo.create_dataset("dones", shape=(12,), dtype="i1")
        states = demo.create_dataset("states", shape=(12, 47), dtype="f4")
        obs = demo.create_group("obs")
        ee_pos = obs.create_dataset("ee_pos", shape=(12, 3), dtype="f4")
        obj_pos = obs.create_dataset("moka_pot_pos", shape=(12, 3), dtype="f4")
        obs.create_dataset("gripper_states", shape=(12, 2), dtype="f4")
        for step in range(12):
            actions[step, :] = [0.2, 0.0, 0.05 if step >= 6 else 0.0, 0.01, -0.02, 0.03, -1.0 if step < 4 else 1.0]
            rewards[step] = 1.0 if step == 10 else 0.0
            dones[step] = 1 if step == 10 else 0
            states[step, :] = 0.0
            ee_pos[step, :] = [0.1 + 0.01 * step, 0.2, 0.3 + 0.01 * max(0, step - 6)]
            obj_pos[step, :] = [0.18, 0.2, 0.3 + 0.015 * max(0, step - 6)]


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
                        "positive_instruction": "put the moka pot on the stove",
                        "counterfactual_instruction": "put the black bowl in the drawer",
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


def test_extract_event_anchors_finds_gripper_motion_and_lift():
    actions = np.zeros((12, 7), dtype=np.float64)
    actions[:4, 6] = -1.0
    actions[4:, 6] = 1.0
    eef = np.asarray([[0.1 + 0.01 * step, 0.0, 0.2 + 0.01 * max(0, step - 6)] for step in range(12)])
    obj = np.asarray([[0.18, 0.0, 0.2 + 0.015 * max(0, step - 6)] for step in range(12)])

    anchors = retiming.extract_event_anchors(actions, eef, obj)

    assert anchors["gripper_close_index"] == 4
    assert anchors["object_motion_onset_index"] == 7
    assert anchors["lift_index"] >= 8
    assert anchors["demo_eef_object_distance"]["available"] is True


def test_build_phase_locked_case_creates_required_perturbations(tmp_path):
    manifest, _readiness = _fixture(tmp_path)

    case = retiming.build_phase_locked_case(manifest, max_steps_cap=12, post_signal_margin=1, offset_steps=3)

    assert case["target_horizon"] == 11
    assert set(retiming.PERTURBATION_NAMES) == set(case["perturbations"])
    assert case["hdf5_eef_source"]["available"] is True
    assert case["hdf5_object_source"]["available"] is True
    delayed = case["perturbations"]["gripper_close_delayed"]["actions"]
    assert retiming._first_gripper_nonnegative(case["actions"]) == 4
    assert retiming._first_gripper_nonnegative(delayed) == 7


def test_event_locked_index_uses_event_anchors_without_modifying_action_values(tmp_path):
    manifest, _readiness = _fixture(tmp_path)
    case = retiming.build_phase_locked_case(manifest, max_steps_cap=12, post_signal_margin=1, offset_steps=3)
    obs = {"ee_pos": np.asarray([0.14, 0.2, 0.3]), "moka_pot_pos": np.asarray([0.18, 0.2, 0.3])}

    index, trace = retiming.select_event_locked_index(
        case=case,
        obs=obs,
        target_key="moka_pot_pos",
        demo_object=[0.18, 0.2, 0.3],
        start_object=[0.18, 0.2, 0.3],
        previous_index=None,
    )

    assert 0 <= index < case["actions"].shape[0]
    assert trace["selected_index"] == index
    action = case["actions"][index]
    assert any(np.allclose(action, row) for row in case["actions"])


def test_phase_locked_script_requires_task_local_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for phase-locked script tests")
    manifest, readiness = _fixture(tmp_path)
    report_json = tmp_path / "report.json"
    env = os.environ.copy()
    env.pop(retiming.TASK_GATE, None)
    for gate in retiming.FORBIDDEN_GATES:
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
            "12",
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
    assert f"{retiming.TASK_GATE}=1 is required" in data["result"]["blocked_reason"]
