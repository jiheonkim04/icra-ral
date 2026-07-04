import json
import os
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "115_plan_vlm_enabled_repeated_offline_decoding.ps1"
PYTHON = r"C:\Users\jiheo\miniconda3\envs\tca_map\python.exe"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for VLM-enabled repeated offline plan tests")
    return exe


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_VLM_ENABLED_LOAD_SMOKE",
        "ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING",
        "ALLOW_REPEATED_OFFLINE_DEMO_DECODING",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_hdf5(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "w") as handle:
        demo = handle.create_group("data").create_group("demo_0")
        demo.create_dataset("actions", data=np.zeros((5, 7), dtype=np.float32))


def _write_inputs(tmp_path: Path, *, previous_signal="weak", load_passed=True):
    hdf5_path = tmp_path / "demo.hdf5"
    _write_hdf5(hdf5_path)
    load = tmp_path / "load.json"
    previous = tmp_path / "previous.json"
    plan = tmp_path / "plan.json"
    load.write_text(
        json.dumps(
            {
                "vlm_enabled_load_smoke_passed": load_passed,
                "load": {
                    "load_vlm_weights": True,
                    "device": "cpu",
                    "parameter_count": 450046176,
                    "cuda_max_allocated_mb": 0,
                    "load_elapsed_sec": 8.4,
                },
                "policy": {"model_inference_performed": False},
            }
        ),
        encoding="utf-8",
    )
    previous.write_text(
        json.dumps(
            {
                "repeated_offline_demo_action_decoding_passed": True,
                "metrics": {
                    "load_vlm_weights": False,
                    "offline_alignment_signal": previous_signal,
                    "mean_action_l1_to_expert": 0.412322,
                    "mean_action_mse_to_expert": 0.286972,
                    "clipped_values_total": 3,
                },
            }
        ),
        encoding="utf-8",
    )
    plan.write_text(
        json.dumps(
            {
                "ready_for_bounded_repeated_offline_demo_action_decoding_runner": True,
                "inputs": {"hdf5_path": str(hdf5_path)},
                "planned_sample": {"hdf5": {"selected_timesteps": [0, 2, 4]}},
            }
        ),
        encoding="utf-8",
    )
    return load, previous, plan


def _run_plan(tmp_path: Path, *, extra_env=None, previous_signal="weak", load_passed=True):
    load, previous, plan = _write_inputs(tmp_path, previous_signal=previous_signal, load_passed=load_passed)
    json_report = tmp_path / "report.json"
    md_report = tmp_path / "report.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            PYTHON,
            "-VlmLoadSmokeReportPath",
            str(load),
            "-PreviousRepeatedReportPath",
            str(previous),
            "-PreviousRepeatedPlanPath",
            str(plan),
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


def test_vlm_enabled_repeated_offline_decoding_plan_proceeds(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "proceed"
    assert report["vlm_enabled_repeated_offline_decoding_plan_passed"] is True
    assert report["ready_for_bounded_vlm_enabled_repeated_offline_decoding_runner"] is True
    assert report["baseline_to_compare"]["previous_offline_alignment_signal"] == "weak"
    assert report["vlm_load_summary"]["load_vlm_weights"] is True
    assert report["risk_assessment"]["planned_policy_inference_calls"] == 3
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_vlm_enabled_repeated_offline_decoding_plan_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_plan(
        tmp_path,
        extra_env={"ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING" in reason for reason in report["stop_reasons"])
    assert report["policy"]["model_load_performed"] is False


def test_vlm_enabled_repeated_offline_decoding_plan_stops_when_not_prioritized(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, previous_signal="moderate")

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert any("not weak" in reason for reason in report["stop_reasons"])
