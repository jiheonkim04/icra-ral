import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tca_map.datasets.libero_zero_reward_rollout_diagnosis import (
    _best_object_key,
    build_zero_reward_diagnosis_cases,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "140_zero_reward_rollout_diagnosis.ps1"


def _write_demo(path: Path, offset: float) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        actions = demo.create_dataset("actions", shape=(60, 7), dtype="f4")
        dones = demo.create_dataset("dones", shape=(60,), dtype="i1")
        rewards = demo.create_dataset("rewards", shape=(60,), dtype="f4")
        for row in range(60):
            actions[row, :] = offset + row * 0.001
            actions[row, 6] = -1.0
            dones[row] = 1 if row == 59 else 0
            rewards[row] = 1.0 if row == 59 else 0.0


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


def test_build_zero_reward_diagnosis_cases_has_horizons_and_expert_replay(tmp_path):
    manifest = _write_manifest(tmp_path)
    cases = build_zero_reward_diagnosis_cases(manifest, horizons=[10, 25, 50], max_tasks=1)

    assert len(cases) == 1
    case = cases[0]
    assert case["horizons"] == [10, 25, 50]
    assert case["positive_demo_metadata"]["full_action_steps"] == 60
    assert case["positive_demo_metadata"]["first_done_index"] == 59
    assert case["action_diagnostics"]["fixed_prior_actions_identical_to_expert_replay"] is True
    assert [variant["name"] for variant in case["variants"]] == [
        "zero_action",
        "actionmap_style_target_agnostic_mean",
        "hdf5_expert_replay",
        "fixed_semantic_target_prior_tca_proxy",
    ]


def test_best_object_key_matches_instruction_tokens():
    obs = {
        "robot0_eef_pos": [0.0, 0.0, 0.0],
        "moka_pot_1_pos": [1.0, 0.0, 0.0],
        "black_bowl_1_pos": [0.0, 1.0, 0.0],
    }

    audit = _best_object_key(obs, "turn on the stove and put the moka pot on it")

    assert audit["best_key"] == "moka_pot_1_pos"
    assert audit["best_score"] >= 2


def test_zero_reward_script_requires_task_local_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for zero-reward diagnosis script tests")
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
            "-Horizons",
            "10,25,50",
        ],
        cwd=REPO_ROOT,
        env={key: value for key, value in os.environ.items() if key != "ALLOW_ZERO_REWARD_ROLLOUT_DIAGNOSIS"},
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    data = json.loads(report.read_text(encoding="utf-8-sig"))
    assert data["policy"]["diagnostic_rollouts_performed"] is False
    assert "ALLOW_ZERO_REWARD_ROLLOUT_DIAGNOSIS=1 is required" in data["result"]["reason"]
