import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "31_generate_go_no_go_report.ps1"


def test_go_no_go_status_generator_is_summary_only(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the go/no-go status generator")

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

    json_report = tmp_path / "go_no_go_status_report.json"
    markdown_report = tmp_path / "go_no_go_status_report.md"
    result = subprocess.run(
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
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    start = result.stdout.find("{")
    assert start >= 0, result.stdout
    report = json.loads(result.stdout[start:])

    assert report["policy"]["summary_only"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["policy"]["risk_assessed_autonomy_policy"] is True
    assert report["decision"].startswith("no_go")
    assert report["ready_for_bounded_local_pilot"] in {True, False}
    assert "bounded_local_pilot_extension" in report["completed_safe_smokes"]
    assert "bounded_local_pilot_extension" in report
    assert report["bounded_local_pilot_extension"].get("not_paper_grade") in {True, False, None}
    assert "libero_data_gates" in report
    assert report["libero_data_gates"]["ready_for_rollout"] is False
    assert report["runtime_reports_available"]["libero_metadata_subset_report"] in {True, False}
    assert report["runtime_reports_available"]["libero_offline_interface_smoke_report"] in {True, False}
    assert report["runtime_reports_available"]["libero_offline_counterfactual_split_report"] in {True, False}
    assert report["runtime_reports_available"]["libero_offline_actionmap_tca_comparison_report"] in {True, False}
    assert report["runtime_reports_available"]["libero_offline_lora_comparison_report"] in {True, False}
    assert report["runtime_reports_available"]["libero_offline_bounded_pilot_report"] in {True, False}
    assert report["runtime_reports_available"]["simulator_readiness_plan_report"] in {True, False}
    assert report["runtime_reports_available"]["bounded_simulator_import_smoke_report"] in {True, False}
    assert report["runtime_reports_available"]["wsl_simulator_dependency_report"] in {True, False}
    assert report["libero_data_gates"]["ready_for_tiny_offline_counterfactual_split"] in {True, False}
    assert report["libero_data_gates"]["ready_for_tiny_offline_actionmap_tca_comparison"] in {True, False}
    assert report["libero_data_gates"]["offline_actionmap_tca_comparison_passed"] in {True, False}
    assert report["libero_data_gates"]["ready_for_required_tiny_lora_comparison"] in {True, False}
    assert report["libero_data_gates"]["offline_lora_comparison_passed"] in {True, False}
    assert report["libero_data_gates"]["ready_for_bounded_local_pilot_report"] in {True, False}
    assert report["libero_data_gates"]["offline_bounded_pilot_report_passed"] in {True, False}
    assert report["libero_data_gates"]["ready_for_simulator_readiness_risk_assessment"] in {True, False}
    assert "simulator_readiness_gates" in report
    assert report["simulator_readiness_gates"]["report_present"] in {True, False}
    assert report["simulator_readiness_gates"]["ready_for_simulator_import_smoke"] in {True, False}
    assert report["simulator_readiness_gates"]["ready_for_simulator_render_smoke"] is False
    assert report["simulator_readiness_gates"]["ready_for_libero_rollout"] is False
    assert report["simulator_readiness_gates"]["simulator_imports_performed"] is False
    assert report["simulator_readiness_gates"]["render_smoke_performed"] is False
    assert report["simulator_readiness_gates"]["rollouts_performed"] is False
    assert report["simulator_readiness_gates"]["bounded_import_smoke_report_present"] in {True, False}
    assert report["simulator_readiness_gates"]["bounded_import_smoke_passed"] in {True, False}
    assert report["simulator_readiness_gates"]["bounded_import_smoke_rollouts_performed"] is False
    assert report["simulator_readiness_gates"]["wsl_dependency_report_present"] in {True, False}
    assert report["simulator_readiness_gates"]["wsl_ready_for_user_level_pip_install"] in {True, False}
    assert report["simulator_readiness_gates"]["wsl_ready_for_simulator_import_retry"] in {True, False}
    assert report["blocked_for_larger_paper_grade_stage"] is True
    assert "lora_qlora_planning" in report
    assert report["lora_qlora_planning"]["qlora_safe_to_run_now"] is False
    assert "paper-grade empirical claims" in report["no_go_for"]
    assert (
        "risk-assessed bounded local SmolVLA pilot tasks inside budget" in report["go_for"]
        or report["all_safe_smokes_passed"] is False
    )
    assert "downloads" in report["risk_assessment_required_for"]
    assert "token/secret/API key access" in report["external_irreversible_stop_gates"]
    assert json_report.exists()
    assert markdown_report.exists()


def test_go_no_go_status_generator_refuses_dangerous_gate(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the go/no-go status generator")

    env = os.environ.copy()
    env["ALLOW_DOWNLOADS"] = "1"
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-JsonReportPath",
            str(tmp_path / "go_no_go_status_report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "go_no_go_status_report.md"),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 20
    assert "dangerous gates" in (result.stdout + result.stderr)
