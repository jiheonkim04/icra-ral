import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tca_map.datasets.libero_action_source_audit_matched_init_diagnostic import build_action_source_audit_case


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "142_action_source_audit_matched_init_diagnostic.ps1"


def _write_demo(path: Path, offset: float, reward_index: int = 59) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        demo.attrs["num_samples"] = 90
        demo.attrs["model_file"] = "<xml/>"
        actions = demo.create_dataset("actions", shape=(90, 7), dtype="f4")
        dones = demo.create_dataset("dones", shape=(90,), dtype="i1")
        rewards = demo.create_dataset("rewards", shape=(90,), dtype="f4")
        states = demo.create_dataset("states", shape=(90, 47), dtype="f4")
        obs = demo.create_group("obs")
        ee_pos = obs.create_dataset("ee_pos", shape=(90, 3), dtype="f4")
        gripper = obs.create_dataset("gripper_states", shape=(90, 2), dtype="f4")
        for row in range(90):
            actions[row, :] = offset + row * 0.001
            actions[row, 6] = -1.0 if row < 70 else 1.0
            dones[row] = 1 if row == reward_index else 0
            rewards[row] = 1.0 if row == reward_index else 0.0
            states[row, :] = 0.0
            ee_pos[row, :] = [0.1, 0.2, 0.3]
            gripper[row, :] = [0.0, 0.0]


def _write_manifest(tmp_path: Path) -> Path:
    positive = tmp_path / "data" / "libero_10" / "task_positive_demo.hdf5"
    counter = tmp_path / "data" / "libero_10" / "task_counter_demo.hdf5"
    _write_demo(positive, 0.1)
    _write_demo(counter, 0.3)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "ready_for_tiny_offline_counterfactual_split": True,
                "counterfactual_pairs": [
                    {
                        "pair_id": "libero_10:task_positive__vs__task_counter",
                        "suite": "libero_10",
                        "positive_task_id": "task_positive",
                        "counterfactual_task_id": "task_counter",
                        "positive_instruction": "put the moka pot on the stove",
                        "counterfactual_instruction": "put the black bowl in the drawer",
                        "positive_demo_file": str(positive),
                        "counterfactual_demo_file": str(counter),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_green_readiness(path: Path) -> None:
    path.write_text(json.dumps({"risk_gate_status": "green", "rollout_diagnostic_authorized": True}), encoding="utf-8")


def test_build_action_source_audit_case_marks_fixed_prior_as_expert_replay(tmp_path):
    manifest = _write_manifest(tmp_path)

    case = build_action_source_audit_case(manifest, max_steps_cap=80, post_signal_margin=20)

    fixed = case["action_source_audit"]["fixed_prior_tca_candidate_replay"]
    actionmap = case["action_source_audit"]["actionmap_style_target_agnostic_mean"]
    hard = case["action_source_audit"]["hard_learned_target_tca_candidate_replay"]

    assert case["target_horizon"] == 79
    assert fixed["fixed_prior_action_equals"] == "hdf5_expert_action"
    assert fixed["match_to_hdf5_expert"]["near_match_rate"] == 1.0
    assert fixed["uses_future_hdf5_actions_unavailable_at_deployment"] is True
    assert fixed["valid_for_method_rollout_claim"] is False
    assert actionmap["mean_or_aggregate_of_hdf5_actions"] is True
    assert actionmap["uses_future_hdf5_actions_unavailable_at_deployment"] is True
    assert hard["selected_from_offline_candidate_set"] is True


def test_action_source_audit_script_requires_task_local_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for action-source audit script tests")
    manifest = _write_manifest(tmp_path)
    readiness = tmp_path / "readiness.json"
    _write_green_readiness(readiness)
    report = tmp_path / "report.json"
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
            str(report),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
            "-MaxStepsCap",
            "80",
        ],
        cwd=REPO_ROOT,
        env={key: value for key, value in os.environ.items() if key != "ALLOW_ACTION_SOURCE_AUDIT_ROLLOUT"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    data = json.loads(report.read_text(encoding="utf-8-sig"))
    assert data["policy"]["diagnostic_rollouts_performed"] is False
    assert "ALLOW_ACTION_SOURCE_AUDIT_ROLLOUT=1 is required" in data["result"]["reason"]
