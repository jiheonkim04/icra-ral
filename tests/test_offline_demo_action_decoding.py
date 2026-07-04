import json
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _make_plan(tmp_path: Path, *, hdf5_exists=True) -> tuple[Path, Path, Path]:
    hdf5_path = tmp_path / "data" / "task_demo.hdf5"
    if hdf5_exists:
        hdf5_path.parent.mkdir(parents=True, exist_ok=True)
        hdf5_path.write_bytes(b"marker")
    smolvla = tmp_path / "smolvla"
    smolvla.mkdir()
    for name in ("config.json", "policy_preprocessor.json", "policy_postprocessor.json", "model.safetensors"):
        (smolvla / name).write_text("{}", encoding="utf-8")
    plan = tmp_path / "plan.json"
    _write_json(
        plan,
        {
            "ready_for_bounded_offline_demo_action_decoding_runner": True,
            "inputs": {"hdf5_path": str(hdf5_path)},
            "planned_sample": {
                "selected_language": "do the task",
                "expert_adapter_strategy": "policy_6d_delta_pose_plus_gripper_close",
            },
        },
    )
    return plan, smolvla, hdf5_path


def _run_module(tmp_path: Path, *, gate=False, extra_env=None, hdf5_exists=True):
    plan, smolvla, _hdf5_path = _make_plan(tmp_path, hdf5_exists=hdf5_exists)
    report = tmp_path / "report.json"
    env = os.environ.copy()
    env.pop("ALLOW_OFFLINE_DEMO_ACTION_DECODING", None)
    for key in (
        "ALLOW_DOWNLOADS",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_ROLLOUT",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    if gate:
        env["ALLOW_OFFLINE_DEMO_ACTION_DECODING"] = "1"
    env.update(extra_env or {})
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "tca_map.smolvla.offline_demo_action_decoding",
            "--plan-report",
            str(plan),
            "--smolvla-ckpt",
            str(smolvla),
            "--checkpoint-root",
            str(tmp_path / "checkpoints"),
            "--hf-home",
            str(tmp_path / "hf_home"),
            "--report-path",
            str(report),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )


def _report(stdout: str) -> dict:
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_offline_demo_action_decoding_requires_task_gate(tmp_path):
    result = _run_module(tmp_path, gate=False)
    report = _report(result.stdout)

    assert result.returncode == 2
    assert report["offline_demo_action_decoding_passed"] is False
    assert report["policy"]["task_local_gate_set"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False


def test_offline_demo_action_decoding_refuses_forbidden_gate_before_loading(tmp_path):
    result = _run_module(tmp_path, gate=True, extra_env={"ALLOW_ROLLOUTS": "1"})
    report = _report(result.stdout)

    assert result.returncode == 3
    assert "ALLOW_ROLLOUTS" in report["error"]["message"]
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["simulator_environment_created"] is False
    assert report["policy"]["rollouts_performed"] is False


def test_offline_demo_action_decoding_stops_when_hdf5_missing_before_loading(tmp_path):
    result = _run_module(tmp_path, gate=True, hdf5_exists=False)
    report = _report(result.stdout)

    assert result.returncode == 7
    assert "HDF5" in report["error"]["message"]
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
