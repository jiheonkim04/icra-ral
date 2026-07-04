import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "111_plan_vlm_enabled_loading_risk.ps1"
PYTHON = r"C:\Users\jiheo\miniconda3\envs\tca_map\python.exe"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for VLM loading risk plan tests")
    return exe


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
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_metadata(path, *, gated=False, include_size=True):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": "HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        "private": False,
        "gated": gated,
        "license": "apache-2.0",
        "siblings": [
            {"rfilename": "config.json", "size": 3767 if include_size else None},
            {"rfilename": "tokenizer.json", "size": 4900000 if include_size else None},
            {"rfilename": "tokenizer_config.json", "size": 1200 if include_size else None},
            {"rfilename": "vocab.json", "size": 777000 if include_size else None},
            {"rfilename": "merges.txt", "size": 466391 if include_size else None},
            {"rfilename": "preprocessor_config.json", "size": 900 if include_size else None},
            {"rfilename": "processor_config.json", "size": 900 if include_size else None},
            {"rfilename": "model.safetensors", "size": 2029990624 if include_size else None},
        ],
        "metadata_source": "test_fixture",
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _run_plan(tmp_path, *, extra_env=None, gated=False, include_size=True):
    metadata = tmp_path / "metadata.json"
    json_report = tmp_path / "report.json"
    md_report = tmp_path / "report.md"
    _write_metadata(metadata, gated=gated, include_size=include_size)
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            PYTHON,
            "-MetadataJsonPath",
            str(metadata),
            "-TargetRoot",
            str(tmp_path / "hf_home" / "HuggingFaceTB" / "SmolVLM2-500M-Video-Instruct"),
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


def test_vlm_enabled_loading_risk_plan_green_with_public_metadata(tmp_path):
    result, report, json_report, md_report = _run_plan(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["vlm_enabled_loading_risk_plan_passed"] is True
    assert report["decision"] == "proceed"
    assert report["ready_for_vlm_weight_acquisition_plan"] is True
    assert report["ready_for_bounded_vlm_enabled_load_smoke_plan"] is False
    assert report["source"]["official_source"] is True
    assert report["source"]["token_login_license_payment_required"] is False
    assert report["risk_assessment"]["expected_new_disk_gb"] > 1.0
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_vlm_enabled_loading_risk_plan_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_plan(tmp_path, extra_env={"ALLOW_DOWNLOADS": "1"})

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["vlm_enabled_loading_risk_plan_passed"] is False
    assert any("ALLOW_DOWNLOADS" in reason for reason in report["stop_reasons"])
    assert report["policy"]["downloads_performed"] is False


def test_vlm_enabled_loading_risk_plan_stops_for_gated_or_unknown_size(tmp_path):
    _, gated_report, _, _ = _run_plan(tmp_path / "gated", gated=True)
    _, size_report, _, _ = _run_plan(tmp_path / "nosize", include_size=False)

    assert gated_report["decision"] == "stop"
    assert any("private or gated" in reason for reason in gated_report["stop_reasons"])
    assert size_report["decision"] == "stop"
    assert any("size" in reason.lower() for reason in size_report["stop_reasons"])
