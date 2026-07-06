import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tca_map.execspec.exact_init_mismatch_replay import build_execspec_replay_case


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "164_execspec_exact_init_mismatch_replay.ps1"


def _write_demo(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        demo.attrs["num_samples"] = 90
        demo.attrs["model_file"] = "<xml/>"
        actions = demo.create_dataset("actions", shape=(90, 7), dtype="f4")
        rewards = demo.create_dataset("rewards", shape=(90,), dtype="f4")
        dones = demo.create_dataset("dones", shape=(90,), dtype="i1")
        states = demo.create_dataset("states", shape=(90, 47), dtype="f4")
        for step in range(90):
            phase = step / 89.0
            actions[step, :] = [
                0.10 + 0.20 * phase,
                -0.05 + 0.10 * phase,
                0.02 + 0.08 * phase,
                0.12 * phase,
                -0.10 * phase,
                0.06 * phase,
                -1.0 if step < 45 else 1.0,
            ]
            rewards[step] = 1.0 if step == 59 else 0.0
            dones[step] = 1 if step == 59 else 0
            states[step, :] = 0.0


def _write_manifest(tmp_path: Path) -> Path:
    positive = tmp_path / "data" / "libero_10" / "task_positive_demo.hdf5"
    _write_demo(positive)
    path = tmp_path / "manifest.json"
    path.write_text(
        json.dumps(
            {
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
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_build_execspec_replay_case_contains_mismatch_variants(tmp_path):
    manifest = _write_manifest(tmp_path)

    case = build_execspec_replay_case(
        manifest,
        max_steps_cap=80,
        post_signal_margin=20,
        replay_variants="correct_7d_expert_action_replay,gripper_sign_flip",
    )

    assert case["target_horizon"] == 79
    assert [variant["name"] for variant in case["variants"]] == [
        "correct_7d_expert_action_replay",
        "gripper_sign_flip",
    ]
    assert case["action_diagnostics"]["gripper_sign_flip"]["metrics"]["gripper_mismatch_rate"] == 1.0
    assert case["action_diagnostics"]["correct_7d_expert_action_replay"]["metrics"]["action_l2_mean"] == 0.0


def test_execspec_replay_script_requires_task_local_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for ExecSpec replay script tests")
    manifest = _write_manifest(tmp_path)
    report = tmp_path / "report.json"
    env = os.environ.copy()
    env.pop("ALLOW_EXECSPEC_MISMATCH_REPLAY", None)

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-ManifestPath",
            str(manifest),
            "-JsonReportPath",
            str(report),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
            "-MaxStepsCap",
            "80",
            "-ReplayVariants",
            "correct_7d_expert_action_replay,gripper_sign_flip",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0
    data = json.loads(report.read_text(encoding="utf-8-sig"))
    assert data["policy"]["replay_or_rollout_performed"] is False
    assert "ALLOW_EXECSPEC_MISMATCH_REPLAY=1 is required" in data["result"]["reason"]
