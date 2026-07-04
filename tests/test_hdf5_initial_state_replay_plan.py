import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "98_plan_hdf5_initial_state_replay.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for HDF5 replay planner tests")
    return exe


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_ROLLOUT",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
        "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
        "ALLOW_HDF5_REPLAY_DIAGNOSTIC",
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_hdf5(path, *, include_init_state=True):
    h5py = pytest.importorskip("h5py")
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data/demo_0")
        if include_init_state:
            demo.attrs["init_state"] = np.arange(47, dtype=np.float64)
        demo.attrs["model_file"] = "<mujoco/>"
        actions = np.zeros((3, 7), dtype=np.float64)
        actions[0, -1] = -1.0
        demo.create_dataset("actions", data=actions)
        demo.create_dataset("states", data=np.zeros((3, 47), dtype=np.float64))
    return path


def _make_inputs(tmp_path, *, include_init_state=True):
    hdf5_path = _make_hdf5(tmp_path / "data" / "demo_task_demo.hdf5", include_init_state=include_init_state)
    align = tmp_path / "align.json"
    repro = tmp_path / "repro.json"
    libero_env = tmp_path / "env_wrapper.py"
    libero_readme = tmp_path / "README.md"
    robosuite_playback = tmp_path / "playback.py"
    _write_json(align, {"ready_for_hdf5_initial_state_replay_plan": True})
    _write_json(repro, {"paths": {"hdf5_path": str(hdf5_path)}})
    libero_env.write_text("def set_init_state(self, init_state):\n    return self.regenerate_obs_from_state(init_state)\n", encoding="utf-8")
    libero_readme.write_text("obs = env.set_init_state(init_states[0])\n", encoding="utf-8")
    robosuite_playback.write_text("env.reset_from_xml_string(xml)\nenv.sim.set_state_from_flattened(states[0])\n", encoding="utf-8")
    return align, repro, libero_env, libero_readme, robosuite_playback


def _run_plan(tmp_path, *, include_init_state=True, extra_env=None):
    align, repro, libero_env, libero_readme, robosuite_playback = _make_inputs(
        tmp_path,
        include_init_state=include_init_state,
    )
    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Hdf5AlignmentAuditReportPath",
            str(align),
            "-OfflineReproductionReportPath",
            str(repro),
            "-LiberoEnvWrapperPath",
            str(libero_env),
            "-LiberoReadmePath",
            str(libero_readme),
            "-RoboSuitePlaybackPath",
            str(robosuite_playback),
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(md_report),
        ],
        cwd=REPO_ROOT,
        env=_clean_env(extra_env),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    start = result.stdout.find("{")
    assert start >= 0, result.stdout + result.stderr
    return result, json.loads(result.stdout[start:]), json_report, md_report


def test_hdf5_initial_state_replay_plan_goes_green(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["hdf5_initial_state_replay_plan_passed"] is True
    assert report["ready_for_bounded_hdf5_replay_runner"] is True
    assert report["hdf5_inputs"]["init_state_present"] is True
    assert report["source_support"]["libero_env_set_init_state"] is True
    assert report["source_support"]["robosuite_playback_sets_state"] is True
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_hdf5_initial_state_replay_plan_requires_init_state(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, include_init_state=False)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["hdf5_initial_state_replay_plan_passed"] is False
    assert any("init_state" in reason for reason in report["stop_reasons"])


def test_hdf5_initial_state_replay_plan_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_plan(
        tmp_path,
        extra_env={"ALLOW_HDF5_REPLAY_DIAGNOSTIC": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_HDF5_REPLAY_DIAGNOSTIC" in reason for reason in report["stop_reasons"])
