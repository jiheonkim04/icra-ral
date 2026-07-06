import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tca_map.execspec import replay_validation


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "166_execspec_replay_validation.ps1"


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


def _fixture(tmp_path: Path) -> tuple[Path, Path]:
    data = tmp_path / "data" / "libero_10"
    names = [
        "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5",
        "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo.hdf5",
        "KITCHEN_SCENE6_put_the_yellow_and_white_mug_in_the_microwave_and_close_it_demo.hdf5",
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5",
        "LIVING_ROOM_SCENE1_put_both_the_alphabet_soup_and_the_cream_cheese_box_in_the_basket_demo.hdf5",
        "LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5",
        "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5",
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5",
    ]
    paths = []
    for index, name in enumerate(names):
        path = data / name
        _write_demo(path, offset=0.03 * index)
        paths.append(path)
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
                        "positive_demo_file": str(paths[0]),
                        "counterfactual_demo_file": str(paths[1]),
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
        max_calibration_demos=5,
        max_eval_demos=3,
        max_replay_eval_demos=3,
        max_actions_per_demo=24,
        max_steps_cap=24,
        post_signal_margin=5,
        camera_size=64,
        replay_mismatches="gripper_sign_flip,translation_scale_mismatch",
        replay_methods="wrong_executable_spec_replay,clipping_only,global_affine_calibration,diagonal_affine_calibration,gripper_only_calibration,full_execspec_repair",
        calibration_sensitivity_sizes="1,3,5",
        include_default_reset_sanity=False,
        report_json=str(tmp_path / "report.json"),
        report_md=str(tmp_path / "report.md"),
    )


def test_state3_split_keeps_eval_out_of_calibration(tmp_path):
    manifest, data_root = _fixture(tmp_path)

    split = replay_validation.build_validation_split(
        manifest_path=manifest,
        data_root=data_root,
        max_calibration_demos=5,
        max_eval_demos=3,
    )

    calibration = {Path(path).name for path in split["calibration_paths"]}
    evaluation = {Path(path).name for path in split["eval_paths"]}
    assert len(calibration) == 5
    assert len(evaluation) == 3
    assert calibration.isdisjoint(evaluation)
    assert "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5" in evaluation
    assert split["leakage_detected"] is False


def test_state3_action_aggregation_and_sensitivity_without_replay(tmp_path, monkeypatch):
    manifest, data_root = _fixture(tmp_path)
    monkeypatch.delenv(replay_validation.TASK_GATE, raising=False)
    for gate in replay_validation.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report = replay_validation.build_state3_report(_args(tmp_path, manifest, data_root))

    assert report["result"]["passed"] is True
    assert report["policy"]["replay_or_rollout_performed"] is False
    assert report["split"]["eval_demo_count"] == 3
    assert report["split"]["leakage_detected"] is False
    assert report["heldout_action_metrics"]["aggregate"]["full_repair_beats_identity_on_action_drift"] is True
    assert report["heldout_action_metrics"]["aggregate"]["full_repair_beats_clipping_only_on_action_drift"] is True
    assert {item["calibration_demo_count"] for item in report["calibration_sensitivity"]} == {1, 3, 5}
    assert report["summary"]["continue_or_kill"] == "needs_replay_validation"


def test_replay_aggregation_detects_multi_demo_recovery():
    def result(variant: str, reward: float, success: bool, done: int | None = None) -> dict:
        return {
            "variant": variant,
            "reward_sum": reward,
            "final_success": success,
            "done_seen": done is not None,
            "first_done_index": done,
        }

    cases = []
    for index, mismatch in enumerate(["gripper_sign_flip", "translation_scale_mismatch"]):
        case = {
            "eval_demo_path": f"demo_{index}.hdf5",
            "task_id": f"task_{index}",
            "mismatch_type": mismatch,
            "replay_results": [
                result("correct_7d_expert_action_replay", 1.0, True, 10),
                result("wrong_executable_spec_replay", 0.0, False, None),
                result("clipping_only", 0.0, False, None),
                result("global_affine_calibration", 0.0, False, None),
                result("full_execspec_repair", 1.0, True, 10),
            ],
        }
        case["summary"] = replay_validation._replay_case_summary(case)
        cases.append(case)

    aggregate = replay_validation.aggregate_replay_cases(cases)

    assert aggregate["degraded_case_count"] == 2
    assert aggregate["success_recovered_count"] == 2
    assert aggregate["success_recovery_rate"] == 1.0
    assert aggregate["eval_demos_with_success_recovery"] == 2
    assert aggregate["mismatches_with_success_recovery"] == 2
    assert aggregate["simple_baseline_match_count"] == 0


def test_state3_script_runs_action_only_without_replay_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for replay validation script tests")
    manifest, data_root = _fixture(tmp_path)
    report_json = tmp_path / "report.json"
    env = os.environ.copy()
    env.pop(replay_validation.TASK_GATE, None)
    for gate in replay_validation.FORBIDDEN_GATES:
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
            "5",
            "-MaxEvalDemos",
            "3",
            "-MaxReplayEvalDemos",
            "3",
            "-MaxActionsPerDemo",
            "24",
            "-SkipDefaultResetSanity",
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
    assert data["replay_skip_reason"].startswith(replay_validation.TASK_GATE)
