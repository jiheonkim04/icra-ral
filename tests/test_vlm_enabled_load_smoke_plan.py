import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "113_plan_vlm_enabled_load_smoke.ps1"
PYTHON = r"C:\Users\jiheo\miniconda3\envs\tca_map\python.exe"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for VLM-enabled load smoke planner tests")
    return exe


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_VLM_ENABLED_LOAD_SMOKE",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
        "ALLOW_REPEATED_OFFLINE_DEMO_DECODING",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_OPENVLA_OFT",
        "ALLOW_RUNTIME_INSTALL",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _make_layout(tmp_path: Path):
    smolvla = tmp_path / "smolvla"
    ckpt = tmp_path / "checkpoints"
    hf_home = tmp_path / "hf_home"
    dep = hf_home / "HuggingFaceTB" / "SmolVLM2-500M-Video-Instruct"
    smolvla.mkdir()
    ckpt.mkdir()
    dep.mkdir(parents=True)
    (smolvla / "config.json").write_text("{}", encoding="utf-8")
    (smolvla / "model.safetensors").write_text("x", encoding="utf-8")
    (smolvla / "policy_preprocessor.json").write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "registry_name": "tokenizer_processor",
                        "config": {"tokenizer_name": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    for name in ("tokenizer.json", "tokenizer_config.json", "config.json", "processor_config.json"):
        (dep / name).write_text("{}", encoding="utf-8")
    return smolvla, ckpt, hf_home, dep


def _write_reports(tmp_path: Path, dep: Path, *, acquired=True):
    acquisition = tmp_path / "acquisition.json"
    risk = tmp_path / "risk.json"
    acquisition.write_text(
        json.dumps(
            {
                "vlm_required_files_acquisition_passed": acquired,
                "decision": "acquisition_complete" if acquired else "stop",
                "ready_for_bounded_vlm_enabled_load_smoke_plan": acquired,
                "risk_assessment": {
                    "source_repo": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                    "source_url": "https://huggingface.co/HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
                    "target_path": str(dep),
                    "expected_new_disk_gb": 1.895,
                },
            }
        ),
        encoding="utf-8",
    )
    risk.write_text(
        json.dumps(
            {
                "decision": "proceed",
                "ready_for_vlm_weight_acquisition_plan": True,
            }
        ),
        encoding="utf-8",
    )
    return acquisition, risk


def _run_plan(tmp_path: Path, *, extra_env=None, acquired=True):
    smolvla, ckpt, hf_home, dep = _make_layout(tmp_path)
    acquisition, risk = _write_reports(tmp_path, dep, acquired=acquired)
    json_report = tmp_path / "plan.json"
    md_report = tmp_path / "plan.md"
    env = _clean_env(
        {
            "SMOLVLA_CKPT": str(smolvla),
            "CHECKPOINT_ROOT": str(ckpt),
            "HF_HOME": str(hf_home),
            **(extra_env or {}),
        }
    )
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            PYTHON,
            "-PathsFile",
            str(tmp_path / "missing_paths.local.yaml"),
            "-AcquisitionReportPath",
            str(acquisition),
            "-RiskReportPath",
            str(risk),
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(md_report),
            "-MinTotalRamGb",
            "0",
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    start = result.stdout.find("{")
    assert start >= 0, result.stdout + result.stderr
    return result, json.loads(result.stdout[start:]), json_report, md_report


def test_vlm_enabled_load_smoke_plan_is_plan_only_and_green(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["vlm_enabled_load_smoke_plan_passed"] is True
    assert report["decision"] == "proceed"
    assert report["ready_for_bounded_vlm_enabled_load_smoke_runner"] is True
    assert report["policy"]["plan_only"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert "ALLOW_VLM_ENABLED_LOAD_SMOKE=1" in report["risk_assessment"]["required_future_gates"]
    assert json_report.exists()
    assert md_report.exists()


def test_vlm_enabled_load_smoke_plan_refuses_execution_gates(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, extra_env={"ALLOW_HEAVY_IMPORT": "1"})

    assert result.returncode == 2
    assert report["decision"] == "stop"
    assert report["vlm_enabled_load_smoke_plan_passed"] is False
    assert "ALLOW_HEAVY_IMPORT" in report["risk_assessment"]["reason"]
    assert report["policy"]["model_load_performed"] is False


def test_vlm_enabled_load_smoke_plan_stops_without_acquisition(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, acquired=False)

    assert result.returncode == 0
    assert report["decision"] == "stop"
    assert report["ready_for_bounded_vlm_enabled_load_smoke_runner"] is False
    assert any("acquisition" in reason.lower() for reason in report["stop_reasons"])
