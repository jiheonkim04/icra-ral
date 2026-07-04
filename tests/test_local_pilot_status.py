import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "39_generate_local_pilot_status.ps1"


def _clean_env(extra=None):
    env = os.environ.copy()
    for gate in [
        "ALLOW_DOWNLOADS",
        "ALLOW_HEAVY_IMPORT",
        "ALLOW_GPU_TRAINING",
        "ALLOW_TINY_TRAINING",
        "ALLOW_ROLLOUTS",
        "ALLOW_RUNTIME_INSTALL",
        "ALLOW_SINGLE_SAMPLE_INFERENCE",
        "ALLOW_CLOUD_HANDOFF",
    ]:
        env.pop(gate, None)
    env.update(extra or {})
    return env


def _run_script(tmp_path, extra_env=None):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the local pilot status report")

    json_report = tmp_path / "local_pilot_status_report.json"
    markdown_report = tmp_path / "local_pilot_status_report.md"
    return subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-JsonReportPath",
            str(json_report),
            "-MarkdownReportPath",
            str(markdown_report),
        ],
        cwd=REPO_ROOT,
        env=_clean_env(extra_env),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    ), json_report, markdown_report


def _json_from_stdout(stdout):
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_local_pilot_status_is_summary_only(tmp_path):
    result, json_report, markdown_report = _run_script(tmp_path)

    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)

    assert report["policy"]["summary_only"] is True
    assert report["policy"]["risk_assessed_autonomy_policy"] is True
    assert report["policy"]["offline_proxy_only"] is True
    assert report["policy"]["not_standard_success"] is True
    assert report["policy"]["not_paper_grade"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert "simulator readiness/import-render smoke if already installed locally" in report["risk_assessed_next_gates"]
    assert "OpenVLA-OFT execution" in report["hard_stop_boundaries"]
    assert "token or secret access" in report["external_irreversible_stop_gates"]
    assert "local_pilot_status_passed" in report
    assert "bounded_local_pilot_extension" in report["source_reports"]
    assert "libero_metadata_subset" in report["source_reports"]
    assert "libero_offline_interface" in report["source_reports"]
    assert "libero_offline_counterfactual_split" in report["source_reports"]
    assert "libero_offline_head_comparison" in report["source_reports"]
    assert "libero_offline_lora_comparison" in report["source_reports"]
    assert "libero_offline_bounded_pilot" in report["source_reports"]
    assert "simulator_readiness" in report["source_reports"]
    assert "bounded_simulator_import_smoke" in report["source_reports"]
    assert "bounded_local_pilot_extension_passed" in report["status"]
    assert "libero_metadata_subset_ready" in report["status"]
    assert "libero_offline_interface_ready" in report["status"]
    assert "libero_offline_counterfactual_split_ready" in report["status"]
    assert "libero_offline_actionmap_tca_ready" in report["status"]
    assert "libero_offline_head_comparison_passed" in report["status"]
    assert "libero_ready_for_required_tiny_lora_comparison" in report["status"]
    assert "libero_offline_lora_comparison_passed" in report["status"]
    assert "libero_ready_for_bounded_local_pilot_report" in report["status"]
    assert "libero_offline_bounded_pilot_report_passed" in report["status"]
    assert "libero_ready_for_simulator_readiness_risk_assessment" in report["status"]
    assert "simulator_readiness_report_present" in report["status"]
    assert "simulator_readiness_decision" in report["status"]
    assert "simulator_effective_runtime_platform" in report["status"]
    assert "simulator_path_ready" in report["status"]
    assert "simulator_dataset_path_ready" in report["status"]
    assert "simulator_import_smoke_ready" in report["status"]
    assert "simulator_render_smoke_ready" in report["status"]
    assert "simulator_rollout_ready" in report["status"]
    assert "simulator_stop_reasons" in report["status"]
    assert "bounded_simulator_import_smoke_report_present" in report["status"]
    assert "bounded_simulator_import_smoke_passed" in report["status"]
    assert "bounded_simulator_import_smoke_decision" in report["status"]
    assert "bounded_simulator_import_smoke_imports_attempted" in report["status"]
    assert "bounded_simulator_import_smoke_rollouts_performed" in report["status"]
    assert report["status"]["simulator_render_smoke_ready"] is False
    assert report["status"]["simulator_rollout_ready"] is False
    assert report["status"]["bounded_simulator_import_smoke_rollouts_performed"] is False
    assert "libero_rollout_ready" in report["status"]
    assert json_report.exists()
    assert markdown_report.exists()


def test_local_pilot_status_refuses_execution_gate(tmp_path):
    result, _, _ = _run_script(tmp_path, {"ALLOW_TINY_TRAINING": "1"})

    assert result.returncode == 20
    assert "execution gates" in (result.stdout + result.stderr)
