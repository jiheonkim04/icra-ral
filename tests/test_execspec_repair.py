import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tca_map.execspec import repair


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "165_execspec_calibrated_repair.ps1"


def _write_demo(path: Path, offset: float, reward_index: int = 19) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        demo.attrs["num_samples"] = 24
        demo.attrs["model_file"] = "<xml/>"
        actions = demo.create_dataset("actions", shape=(24, 7), dtype="f4")
        rewards = demo.create_dataset("rewards", shape=(24,), dtype="f4")
        dones = demo.create_dataset("dones", shape=(24,), dtype="i1")
        states = demo.create_dataset("states", shape=(24, 47), dtype="f4")
        for step in range(24):
            phase = step / 23.0
            actions[step, :] = [
                offset + 0.20 * phase,
                -0.10 + 0.06 * phase,
                0.02 + 0.05 * phase,
                0.08 * phase,
                -0.07 * phase,
                0.05 * phase,
                -1.0 if step < 10 else 1.0,
            ]
            rewards[step] = 1.0 if step == reward_index else 0.0
            dones[step] = 1 if step == reward_index else 0
            states[step, :] = 0.0


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data" / "libero_10"
    eval_demo = data / "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
    cal_a = data / "calibration_a_demo.hdf5"
    cal_b = data / "calibration_b_demo.hdf5"
    _write_demo(eval_demo, 0.0)
    _write_demo(cal_a, 0.2)
    _write_demo(cal_b, -0.1)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "counterfactual_pairs": [
                    {
                        "pair_id": "libero_10:eval__vs__cf",
                        "suite": "libero_10",
                        "positive_task_id": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
                        "counterfactual_task_id": "cf_task",
                        "positive_instruction": "turn on the stove and put the moka pot on it",
                        "counterfactual_instruction": "put the black bowl in the drawer",
                        "positive_demo_file": str(eval_demo),
                        "counterfactual_demo_file": str(cal_a),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return manifest, tmp_path / "data"


def _args(tmp_path: Path, manifest: Path, data_root: Path) -> argparse.Namespace:
    return argparse.Namespace(
        manifest=str(manifest),
        data_root=str(data_root),
        libero_root="/tmp/libero",
        robosuite_root="/tmp/robosuite",
        max_calibration_demos=2,
        max_eval_demos=1,
        max_actions_per_demo=24,
        max_steps_cap=24,
        post_signal_margin=5,
        camera_size=64,
        replay_mismatches="gripper_sign_flip",
        report_json=str(tmp_path / "report.json"),
        report_md=str(tmp_path / "report.md"),
    )


def test_split_excludes_heldout_eval_demo(tmp_path):
    manifest, data_root = _write_fixture(tmp_path)

    split = repair.build_data_split(
        manifest_path=manifest,
        data_root=data_root,
        max_calibration_demos=2,
        max_eval_demos=1,
    )

    calibration = {Path(path).name for path in split["calibration_paths"]}
    evaluation = {Path(path).name for path in split["eval_paths"]}
    assert calibration.isdisjoint(evaluation)
    assert "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5" in evaluation
    assert split["leakage_detected"] is False


def test_full_repair_beats_simple_baselines_without_eval_leakage(tmp_path, monkeypatch):
    manifest, data_root = _write_fixture(tmp_path)
    monkeypatch.delenv(repair.TASK_GATE, raising=False)
    for gate in repair.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report = repair.build_state2_report(_args(tmp_path, manifest, data_root))

    assert report["result"]["passed"] is True
    assert report["policy"]["replay_or_rollout_performed"] is False
    assert report["split"]["leakage_detected"] is False
    assert report["split"]["calibration_demo_count"] == 2
    gripper = report["heldout_action_metrics"]["gripper_sign_flip"]["repair_methods"]["full_execspec_repair"]
    translation = report["heldout_action_metrics"]["translation_scale_mismatch"]["repair_methods"]["full_execspec_repair"]
    assert gripper["beats_identity"] is True
    assert gripper["beats_clipping_only"] is True
    assert gripper["beats_global_affine"] is True
    assert translation["beats_identity"] is True
    assert report["summary"]["full_repair_beats_identity_on_action_drift"] is True


def test_repair_script_runs_action_metrics_without_replay_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for repair script tests")
    manifest, data_root = _write_fixture(tmp_path)
    report_json = tmp_path / "report.json"
    env = os.environ.copy()
    env.pop(repair.TASK_GATE, None)
    for gate in repair.FORBIDDEN_GATES:
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
            "-DataRoot",
            str(data_root),
            "-JsonReportPath",
            str(report_json),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
            "-MaxCalibrationDemos",
            "2",
            "-MaxEvalDemos",
            "1",
            "-MaxActionsPerDemo",
            "24",
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
    assert data["result"]["passed"] is True
    assert data["policy"]["replay_or_rollout_performed"] is False
    assert data["replay_skip_reason"].startswith(repair.TASK_GATE)


def test_gripper_calibration_learns_sign_flip():
    target = np.asarray([[-0.1, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0], [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]])
    source = repair.apply_mismatch(target, "gripper_sign_flip")
    params = repair.fit_repair_parameters(source, target)
    repaired = repair.apply_repair(source, "full_execspec_repair", params)

    assert np.allclose(repaired[:, 6], target[:, 6])
