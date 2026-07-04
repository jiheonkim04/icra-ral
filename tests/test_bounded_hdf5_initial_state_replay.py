import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "100_bounded_hdf5_initial_state_replay.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for bounded HDF5 replay runner tests")
    return exe


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_HDF5_REPLAY_DIAGNOSTIC",
        "ALLOW_ROLLOUT",
        "ALLOW_ROLLOUTS",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_TINY_TRAINING",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
        "ALLOW_BOUNDED_LEARNED_POLICY_MATRIX",
        "ALLOW_ADAPTER_STRATEGY_DIAGNOSTIC",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_runner(tmp_path, *, extra_env=None, extra_args=None):
    plan = tmp_path / "plan.json"
    hdf5 = tmp_path / "demo.hdf5"
    hdf5.write_bytes(b"placeholder")
    _write_json(
        plan,
        {
            "ready_for_bounded_hdf5_replay_runner": True,
            "hdf5_inputs": {"hdf5_path": str(hdf5)},
            "recommended_next_step": "test",
        },
    )
    json_report = tmp_path / "run.json"
    md_report = tmp_path / "run.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-PlanReportPath",
            str(plan),
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(md_report),
            *(extra_args or []),
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


def test_bounded_hdf5_replay_runner_requires_task_local_gate(tmp_path):
    result, report, json_report, md_report = _run_runner(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["bounded_hdf5_initial_state_replay_passed"] is False
    assert report["policy"]["simulator_environment_attempted"] is False
    assert report["policy"]["hdf5_replay_diagnostic_performed"] is False
    assert report["policy"]["learned_policy_inference_performed"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_bounded_hdf5_replay_runner_refuses_broad_rollout_gate(tmp_path):
    result, report, _, _ = _run_runner(
        tmp_path,
        extra_env={
            "ALLOW_HDF5_REPLAY_DIAGNOSTIC": "1",
            "ALLOW_ROLLOUT": "1",
        },
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUT" in report["reason"]
    assert report["policy"]["simulator_environment_attempted"] is False


def test_bounded_hdf5_replay_runner_rejects_multiple_steps_first_runner(tmp_path):
    result, report, _, _ = _run_runner(
        tmp_path,
        extra_env={"ALLOW_HDF5_REPLAY_DIAGNOSTIC": "1"},
        extra_args=["-MaxReplaySteps", "2"],
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "exactly one replay step" in report["reason"]
    assert report["policy"]["simulator_environment_attempted"] is False
