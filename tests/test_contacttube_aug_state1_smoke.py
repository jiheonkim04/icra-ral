import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from tca_map.contacttube_aug import state1_smoke


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "180_contacttube_aug_state1_smoke.ps1"


def _write_demo(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    steps = 36
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        demo.attrs["num_samples"] = steps
        demo.attrs["model_file"] = "<xml/>"
        actions = demo.create_dataset("actions", shape=(steps, 7), dtype="f4")
        dones = demo.create_dataset("dones", shape=(steps,), dtype="i1")
        rewards = demo.create_dataset("rewards", shape=(steps,), dtype="f4")
        states = demo.create_dataset("states", shape=(steps, 47), dtype="f4")
        obs = demo.create_group("obs")
        ee_pos = obs.create_dataset("ee_pos", shape=(steps, 3), dtype="f4")
        obj_pos = obs.create_dataset("moka_pot_pos", shape=(steps, 3), dtype="f4")
        gripper = obs.create_dataset("gripper_states", shape=(steps, 2), dtype="f4")
        for step in range(steps):
            phase = step / float(steps - 1)
            close = step >= 10
            release = step >= 31
            obj_z = 0.02 + (0.035 if step >= 18 else 0.0) - (0.02 if step >= 29 else 0.0)
            obj_x = 0.35 + (0.025 if step >= 14 else 0.0)
            eef_x = 0.20 + 0.18 * phase
            if step >= 10:
                eef_x = obj_x + 0.02
            actions[step, :] = [
                0.05 * np.cos(phase),
                0.03 * np.sin(phase),
                0.04 if 18 <= step < 25 else 0.0,
                0.01,
                -0.01,
                0.0,
                -1.0 if (not close or release) else 1.0,
            ]
            dones[step] = 1 if step == 32 else 0
            rewards[step] = 1.0 if step == 32 else 0.0
            states[step, :] = 0.0
            ee_pos[step, :] = [eef_x, 0.1, obj_z + 0.02]
            obj_pos[step, :] = [obj_x, 0.1, obj_z]
            gripper[step, :] = [1.0 if close else 0.0, 1.0 if close else 0.0]


def _write_manifest(tmp_path: Path) -> Path:
    positive = tmp_path / "data" / "libero_10" / "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
    _write_demo(positive)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "counterfactual_pairs": [
                    {
                        "pair_id": "libero_10:demo__vs__cf",
                        "suite": "libero_10",
                        "positive_task_id": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
                        "counterfactual_task_id": "cf_task",
                        "positive_instruction": "turn on the stove and put the moka pot on it",
                        "counterfactual_instruction": "put the black bowl in the drawer",
                        "positive_demo_file": str(positive),
                        "counterfactual_demo_file": str(positive),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_green_readiness(path: Path) -> None:
    path.write_text(json.dumps({"risk_gate_status": "green", "rollout_diagnostic_authorized": True}), encoding="utf-8")


def test_extract_contact_tube_finds_events():
    steps = 16
    actions = np.zeros((steps, 7), dtype=np.float64)
    actions[:, 6] = -1.0
    actions[5:13, 6] = 1.0
    eef = np.zeros((steps, 3), dtype=np.float64)
    obj = np.zeros((steps, 3), dtype=np.float64)
    obj[:, :] = [0.3, 0.1, 0.02]
    obj[8:, 0] += 0.01
    obj[10:, 2] += 0.03
    eef[:, :] = obj + np.asarray([0.015, 0.0, 0.02])

    tube = state1_smoke.extract_contact_tube(actions=actions, eef_positions=eef, object_positions=obj, source="unit")

    assert tube["observable"] is True
    assert tube["object_pose_available"] is True
    assert tube["gripper_close_index"] == 5
    assert tube["release_index"] == 13
    assert tube["object_motion_onset_index"] == 8
    assert tube["lift_index"] == 10
    assert tube["contact_window"]["available"] is True
    assert tube["distance_profile_summary"]["available"] is True


def test_build_contacttube_case_uses_hdf5_tube(tmp_path):
    manifest = _write_manifest(tmp_path)

    case = state1_smoke.build_contacttube_case(manifest, max_steps_cap=34, post_signal_margin=2, seed=3)

    assert case["target_horizon"] == 34
    assert case["hdf5_eef_source"]["available"] is True
    assert case["hdf5_object_source"]["available"] is True
    assert case["hdf5_contact_tube"]["object_pose_available"] is True
    assert case["hdf5_contact_tube"]["contact_window"]["available"] is True
    assert set(case["static_variant_actions"]) == {
        "exact_init_noop_upper_bound",
        "raw_demo_replay",
        "random_pose_jitter",
        "random_action_jitter",
    }
    assert "simple_object_relative_translation_retarget" in case["dynamic_variants"]
    assert state1_smoke.METHOD_VARIANT in case["dynamic_variants"]


def test_summary_kills_when_simple_retarget_matches_contacttube():
    def variant(name: str, score: float) -> dict:
        return {
            "variant": name,
            "passed": True,
            "reward_sum": 0.0,
            "final_success": False,
            "done_seen": False,
            "controller_valid_action_rate": 1.0,
            "clip_rate_step": 0.0,
            "contact_tube_metrics": {"contact_tube_preservation_error": score},
        }

    report = {
        "policy": {"replay_or_rollout_performed": True},
        "cases": [
            {
                "hdf5_contact_tube": {"observable": True, "object_pose_available": True},
                "variants": [
                    {
                        "variant": "exact_init_noop_upper_bound",
                        "reward_sum": 1.0,
                        "final_success": True,
                        "done_seen": True,
                        "contact_tube": {"observable": True, "object_pose_available": True},
                    },
                    variant("random_pose_jitter", 0.50),
                    variant("random_action_jitter", 0.40),
                    variant("simple_object_relative_translation_retarget", 0.10),
                    variant(state1_smoke.METHOD_VARIANT, 0.10),
                ],
            }
        ],
    }

    summary = state1_smoke.summarize_report(report)

    assert summary["continue_or_kill"] == "kill"
    assert summary["simple_object_relative_matches_or_beats_contacttube_aug"] is True


def test_contacttube_script_requires_task_local_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for ContactTube-Aug script tests")
    manifest = _write_manifest(tmp_path)
    readiness = tmp_path / "readiness.json"
    _write_green_readiness(readiness)
    report_json = tmp_path / "report.json"
    env = os.environ.copy()
    env.pop(state1_smoke.TASK_GATE, None)
    for gate in state1_smoke.FORBIDDEN_GATES:
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
            "34",
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
    assert "ALLOW_CONTACTTUBE_AUG_STATE1=1 is required" in data["result"]["blocked_reason"]

