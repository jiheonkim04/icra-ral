import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "107_summarize_offline_demo_action_decoding.ps1"


def _powershell():
    exe = shutil.which("powershell")
    if exe is None:
        pytest.skip("PowerShell is required for offline decoding summary tests")
    return exe


def _clean_env(extra_env=None):
    env = os.environ.copy()
    for key in (
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_ROLLOUT",
        "ALLOW_POLICY_ROLLOUT",
        "ALLOW_BENCHMARK_ROLLOUT",
        "ALLOW_OPENVLA_OFT",
    ):
        env.pop(key, None)
    env.update(extra_env or {})
    return env


def _write_source(path, *, action_l1=0.375016):
    path.write_text(
        json.dumps(
            {
                "offline_demo_action_decoding_passed": True,
                "metrics": {
                    "action_l1_to_expert": action_l1,
                    "action_mse_to_expert": 0.227689,
                    "policy6_l1_to_expert_first6": 0.570848,
                    "action_finite": True,
                    "load_vlm_weights": False,
                },
            }
        ),
        encoding="utf-8",
    )


def _run_summary(tmp_path, *, action_l1=0.375016, extra_env=None, missing=False):
    source = tmp_path / "offline.json"
    if not missing:
        _write_source(source, action_l1=action_l1)
    json_report = tmp_path / "summary.json"
    md_report = tmp_path / "summary.md"
    result = subprocess.run(
        [
            _powershell(),
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-SourceReportPath",
            str(source),
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


def test_offline_decoding_summary_blocks_rollout_scaling_for_weak_alignment(tmp_path):
    result, report, json_report, md_report = _run_summary(tmp_path)

    assert result.returncode == 0, result.stderr
    assert report["offline_demo_action_decoding_summary_passed"] is True
    assert report["decision"] == "no_go_rollout_scaling"
    assert report["metrics"]["offline_alignment_signal"] == "weak"
    assert report["ready_for_rollout_scaling"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["claims"]["paper_grade_claim_made"] is False
    assert json_report.exists()
    assert md_report.exists()


def test_offline_decoding_summary_refuses_execution_gate(tmp_path):
    result, report, _, _ = _run_summary(
        tmp_path,
        extra_env={"ALLOW_OFFLINE_DEMO_ACTION_DECODING": "1"},
    )

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert "ALLOW_OFFLINE_DEMO_ACTION_DECODING" in report["reason"]
    assert report["policy"]["model_load_performed"] is False


def test_offline_decoding_summary_stops_when_source_missing(tmp_path):
    result, report, json_report, md_report = _run_summary(tmp_path, missing=True)

    assert result.returncode == 0, result.stderr
    assert report["decision"] == "stop"
    assert report["offline_demo_action_decoding_summary_passed"] is False
    assert "Missing or unreadable offline decoding report" in report["reason"]
    assert json_report.exists()
    assert md_report.exists()
