import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tca_map.resetspec import retarget


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "170_resetspec_retarget_diagnostic.ps1"


def _write_demo(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        demo.attrs["num_samples"] = 8
        demo.attrs["model_file"] = "<xml/>"
        actions = demo.create_dataset("actions", shape=(8, 7), dtype="f4")
        rewards = demo.create_dataset("rewards", shape=(8,), dtype="f4")
        dones = demo.create_dataset("dones", shape=(8,), dtype="i1")
        states = demo.create_dataset("states", shape=(8, 47), dtype="f4")
        obs = demo.create_group("obs")
        ee_pos = obs.create_dataset("ee_pos", shape=(8, 3), dtype="f4")
        obs.create_dataset("gripper_states", shape=(8, 2), dtype="f4")
        for step in range(8):
            actions[step, :] = [0.2, 0.0, 0.0, 0.01, -0.02, 0.03, -1.0 if step < 4 else 1.0]
            rewards[step] = 1.0 if step == 6 else 0.0
            dones[step] = 1 if step == 6 else 0
            states[step, :] = 0.0
            ee_pos[step, :] = [0.1 + 0.01 * step, 0.2, 0.3]


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


def test_build_resetspec_case_reads_hdf5_eef_and_baselines(tmp_path):
    manifest, _readiness = _fixture(tmp_path)

    case = retarget.build_resetspec_case(manifest, max_steps_cap=8, post_signal_margin=2, global_scale=0.5)

    assert case["target_horizon"] == 8
    assert case["hdf5_metadata"]["first_done_index"] == 6
    assert case["hdf5_eef_source"]["available"] is True
    assert case["translation_unit"]["meters_per_action_unit"] > 0
    assert np.allclose(case["static_variant_actions"]["default_reset_diagonal_affine_replay"], case["actions"])
    assert np.allclose(case["static_variant_actions"]["default_reset_global_scale_replay"][:, :6], case["actions"][:, :6] * 0.5)


def test_retarget_translation_action_uses_current_eef_and_desired_eef():
    action, available = retarget.retarget_translation_action(
        raw_action=np.zeros((7,), dtype=np.float64),
        current_eef=[0.0, 0.0, 0.0],
        desired_eef=np.asarray([0.02, -0.01, 0.03], dtype=np.float64),
        meters_per_action_unit=0.01,
    )

    assert available is True
    assert np.allclose(action[:3], [2.0, -1.0, 3.0])
    assert np.allclose(action[3:], [0.0, 0.0, 0.0, 0.0])


def test_resetspec_script_requires_task_local_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for ResetSpec script tests")
    manifest, readiness = _fixture(tmp_path)
    report_json = tmp_path / "report.json"
    env = os.environ.copy()
    env.pop(retarget.TASK_GATE, None)
    for gate in retarget.FORBIDDEN_GATES:
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
            "8",
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
    assert f"{retarget.TASK_GATE}=1 is required" in data["result"]["blocked_reason"]
