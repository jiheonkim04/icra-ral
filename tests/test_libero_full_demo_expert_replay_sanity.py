import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tca_map.datasets.libero_full_demo_expert_replay_sanity import build_full_demo_expert_replay_case


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "141_full_demo_expert_replay_sanity.ps1"


def _write_demo(path: Path) -> None:
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
            actions[row, :] = row * 0.001
            actions[row, 6] = -1.0 if row < 70 else 1.0
            dones[row] = 1 if row == 59 else 0
            rewards[row] = 1.0 if row == 59 else 0.0
            states[row, :] = 0.0
            ee_pos[row, :] = [0.1, 0.2, 0.3]
            gripper[row, :] = [0.0, 0.0]


def _write_manifest(tmp_path: Path) -> Path:
    positive = tmp_path / "data" / "libero_10" / "task_positive_demo.hdf5"
    _write_demo(positive)
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
                        "counterfactual_demo_file": str(positive),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _write_green_readiness(path: Path) -> None:
    path.write_text(json.dumps({"risk_gate_status": "green", "rollout_diagnostic_authorized": True}), encoding="utf-8")


def test_build_full_demo_expert_replay_case_uses_first_signal_margin(tmp_path):
    manifest = _write_manifest(tmp_path)

    case = build_full_demo_expert_replay_case(manifest, max_steps_cap=80, post_signal_margin=20)

    assert case["hdf5_metadata"]["first_positive_reward_index"] == 59
    assert case["hdf5_metadata"]["first_done_index"] == 59
    assert case["target_horizon"] == 79
    assert case["action_diagnostics"]["hdf5_action_distribution_matches_expected_range"] is True
    assert [variant["name"] for variant in case["variants"]] == [
        "zero_action_exact_init",
        "hdf5_expert_replay_exact_init",
        "hdf5_expert_replay_default_reset",
    ]


def test_full_demo_expert_replay_script_requires_task_local_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for full-demo expert replay script tests")
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
        env={key: value for key, value in os.environ.items() if key != "ALLOW_FULL_DEMO_EXPERT_REPLAY"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    data = json.loads(report.read_text(encoding="utf-8-sig"))
    assert data["policy"]["diagnostic_rollouts_performed"] is False
    assert "ALLOW_FULL_DEMO_EXPERT_REPLAY=1 is required" in data["result"]["reason"]
