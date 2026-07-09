import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from tca_map.smolvla_lora_baseline import libero_ee_state_features as ee_features
from tca_map.smolvla_lora_baseline import replay_bridge


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "236_smolvla_7d_replay_bridge.ps1"


def _args(tmp_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        hdf5_path=str(tmp_path / "missing.hdf5"),
        smolvla_ckpt=str(tmp_path / "smolvla"),
        adapter_artifact=str(tmp_path / "adapter.pt"),
        output_dir=str(tmp_path / "runs"),
        report_path=str(tmp_path / "report.json"),
        data_root=str(tmp_path / "data"),
        libero_root=str(tmp_path / "LIBERO"),
        robosuite_root=str(tmp_path / "robosuite"),
        adapter_steps=1,
        adapter_hidden_dim=8,
        lora_rank=8,
        lora_learning_rate=1e-3,
        seed=25,
        max_replay_steps=4,
        post_signal_margin=0,
        camera_size=32,
    )


def _write_demo(path: Path) -> None:
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.attrs["init_state"] = [0.0] * 47
        actions = demo.create_dataset("actions", shape=(5, 7), dtype="f4")
        rewards = demo.create_dataset("rewards", shape=(5,), dtype="f4")
        dones = demo.create_dataset("dones", shape=(5,), dtype="i1")
        obs = demo.create_group("obs")
        ee = obs.create_dataset("ee_states", shape=(5, 6), dtype="f4")
        for step in range(5):
            actions[step, :] = [0.1 * step, 0.0, -0.1, 0.01, 0.02, 0.03, -1.0 if step < 3 else 1.0]
            rewards[step] = 1.0 if step == 4 else 0.0
            dones[step] = 1 if step == 4 else 0
            ee[step, :] = [step, step + 1, step + 2, 0.1, 0.2, 0.3]


def test_replay_bridge_decision_set_is_exact():
    assert replay_bridge.FINAL_DECISIONS == {
        "READY_FOR_METHOD_AFTER_REPLAY_BRIDGE",
        "ADAPTER_ACTION_RANGE_ISSUE",
        "OFFLINE_TO_CONTROL_GAP",
        "EXPERT_REPLAY_STILL_BLOCKED",
        "ENV_BLOCKED_INSTALL_FAILED",
        "TOO_HEAVY_LOCAL",
    }


def test_default_mujoco_gl_uses_glfw_on_windows(monkeypatch):
    monkeypatch.setattr(replay_bridge.os, "name", "nt")

    assert replay_bridge._default_mujoco_gl() == "glfw"


def test_ensure_mujoco_gl_default_preserves_existing_value(monkeypatch):
    monkeypatch.setenv("MUJOCO_GL", "wgl")

    assert replay_bridge._ensure_mujoco_gl_default() == "wgl"


def test_replay_bridge_requires_gate(tmp_path, monkeypatch):
    monkeypatch.delenv(replay_bridge.BRIDGE_GATE, raising=False)
    monkeypatch.delenv(replay_bridge.TRAINING_GATE, raising=False)
    monkeypatch.delenv(replay_bridge.REPLAY_GATE, raising=False)

    report, code = replay_bridge.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "ENV_BLOCKED_INSTALL_FAILED"
    assert replay_bridge.BRIDGE_GATE in report["summary"]["exact_next_step"]
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["replay_control_performed"] is False


def test_action_validity_reports_shape_and_clip_rate():
    actions = np.asarray([[0.0, 0.1, -0.2, 0.0, 0.0, 0.0, 1.0], [1.2, 0.0, 0.0, 0.0, 0.0, 0.0, -1.4]], dtype=np.float32)

    validity = replay_bridge._action_validity(actions)

    assert validity["shape_exactly_7d"] is True
    assert validity["clip_rate_step"] == 0.5
    assert validity["controller_valid_rate_proxy"] == 0.5
    assert validity["silent_broadcast_or_truncation_detected"] is False


def test_demo_feature_uses_ee_states_and_timestep_fraction(tmp_path):
    path = tmp_path / "task_demo.hdf5"
    _write_demo(path)

    feature = replay_bridge._feature_for_demo_timestep(path, "demo_0", 2)

    assert feature.shape == (7,)
    assert feature[:6].tolist() == pytest.approx([2.0, 3.0, 4.0, 0.1, 0.2, 0.3])
    assert feature[6] == pytest.approx(0.5)


def test_observation_feature_accepts_env_style_obs():
    obs = {"robot0_eef_pos": [1.0, 2.0, 3.0], "robot0_eef_quat": [0.4, 0.5, 0.6, 0.7]}

    feature, metadata = replay_bridge._observation_feature(obs, 0.25)

    expected_ori = ee_features.quat_xyzw_to_hdf5_axis_angle(obs["robot0_eef_quat"])
    assert feature.tolist() == pytest.approx([1.0, 2.0, 3.0, *expected_ori.tolist(), 0.25])
    assert metadata["orientation_convention"] == ee_features.ORIENTATION_CONVENTION
    assert metadata["uses_quat_first3_fallback"] is False
    assert metadata["uses_eval_action_label"] is False


def test_script_runs_blocked_without_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for replay bridge script tests")
    env = os.environ.copy()
    env.pop(replay_bridge.BRIDGE_GATE, None)
    env.pop(replay_bridge.TRAINING_GATE, None)
    env.pop(replay_bridge.REPLAY_GATE, None)
    report_path = tmp_path / "report.json"

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-Hdf5Path",
            str(tmp_path / "missing.hdf5"),
            "-SmolVlaCkpt",
            str(tmp_path / "missing_ckpt"),
            "-AdapterArtifact",
            str(tmp_path / "adapter.pt"),
            "-ReportPath",
            str(report_path),
            "-AdapterSteps",
            "1",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    data = json.loads(report_path.read_text(encoding="utf-8-sig"))
    assert data["decision"] == "ENV_BLOCKED_INSTALL_FAILED"
    assert replay_bridge.BRIDGE_GATE in data["summary"]["exact_next_step"]
