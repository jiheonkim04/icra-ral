import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
        "ALLOW_REPEATED_OFFLINE_DEMO_DECODING",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_ROLLOUT",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_RUNTIME_INSTALL",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_plan(path, *, ready=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ready_for_bounded_repeated_offline_demo_action_decoding_runner": ready,
                "inputs": {"hdf5_path": str(path.parent / "missing.hdf5")},
                "planned_sample": {"hdf5": {"selected_timesteps": [0, 1, 2], "demo_name": "demo_0"}},
            }
        ),
        encoding="utf-8",
    )


def _run_module(tmp_path, *, extra_env=None, ready=True):
    plan = tmp_path / "plan.json"
    report = tmp_path / "report.json"
    _write_plan(plan, ready=ready)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "tca_map.smolvla.repeated_offline_demo_action_decoding",
            "--plan-report",
            str(plan),
            "--report-path",
            str(report),
            "--smolvla-ckpt",
            str(tmp_path / "smolvla"),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--hf-home",
            str(tmp_path / "hf_home"),
        ],
        cwd=REPO_ROOT,
        env=_clean_env(extra_env),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    payload = json.loads(report.read_text(encoding="utf-8"))
    return result, payload


def test_repeated_offline_demo_action_decoding_requires_task_gate(tmp_path):
    result, report = _run_module(tmp_path)

    assert result.returncode != 0
    assert report["decision"] == "stop"
    assert report["repeated_offline_demo_action_decoding_passed"] is False
    assert "ALLOW_REPEATED_OFFLINE_DEMO_DECODING" in report["recommended_next_step"]
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False


def test_repeated_offline_demo_action_decoding_refuses_forbidden_gate(tmp_path):
    result, report = _run_module(
        tmp_path,
        extra_env={
            "ALLOW_REPEATED_OFFLINE_DEMO_DECODING": "1",
            "ALLOW_DOWNLOADS": "1",
        },
    )

    assert result.returncode != 0
    assert report["decision"] == "stop"
    assert "ALLOW_DOWNLOADS" in report["recommended_next_step"]
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["downloads_performed"] is False


def test_repeated_offline_demo_action_decoding_stops_when_plan_not_ready(tmp_path):
    result, report = _run_module(
        tmp_path,
        ready=False,
        extra_env={"ALLOW_REPEATED_OFFLINE_DEMO_DECODING": "1"},
    )

    assert result.returncode != 0
    assert report["decision"] == "stop"
    assert "plan did not authorize" in report["recommended_next_step"]
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
