import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.smolvla.candidate_generation_contract_check import (
    FORBIDDEN_GATES,
    _synthetic_contract,
    _validate_contract,
    run_contract_check,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "131_check_candidate_generation_contract.ps1"


def _write_readiness(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "candidate_generation_readiness_plan_passed": True,
                "ready_for_candidate_generation_contract_checker": True,
                "ready_for_real_candidate_generation_smoke_execution": False,
            }
        ),
        encoding="utf-8",
    )


def _clean_env(extra=None):
    env = os.environ.copy()
    for gate in FORBIDDEN_GATES:
        env.pop(gate, None)
    env.update(extra or {})
    return env


def _json_from_stdout(stdout: str):
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_candidate_generation_contract_check_passes_without_model_execution(tmp_path, monkeypatch):
    for gate in FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    readiness = tmp_path / "readiness.json"
    _write_readiness(readiness)

    report = run_contract_check(
        readiness_report=readiness,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
        candidate_count=4,
        heatmap_grid=8,
    )

    assert report["candidate_generation_contract_check_passed"] is True
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["privileged_inference_used"] is False
    assert report["policy"]["external_verifier_used"] is False
    assert report["ready_for_real_candidate_generation_smoke_execution"] is False
    assert report["selection"]["selected_candidate_index"] is not None


def test_candidate_generation_contract_validator_rejects_privileged_metadata():
    action_heatmap, masked_heatmap, target_heatmap, metadata = _synthetic_contract(candidate_count=4, heatmap_grid=8)
    metadata["object_pose"] = [0.0, 0.0, 0.0]

    errors = _validate_contract(action_heatmap, masked_heatmap, target_heatmap, metadata)

    assert any("forbidden privileged key" in error for error in errors)


def test_candidate_generation_contract_refuses_dangerous_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_HEAVY_IMPORT", "1")
    readiness = tmp_path / "readiness.json"
    _write_readiness(readiness)

    report = run_contract_check(
        readiness_report=readiness,
        report_json=tmp_path / "report.json",
        report_md=tmp_path / "report.md",
    )

    assert report["candidate_generation_contract_check_passed"] is False
    assert "ALLOW_HEAVY_IMPORT" in report["policy"]["forbidden_gates_set"]
    assert report["policy"]["heavy_model_imports_performed"] is False


def test_candidate_generation_contract_script_runs(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for candidate-generation contract script tests")
    readiness = tmp_path / "readiness.json"
    _write_readiness(readiness)

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-ReadinessReportPath",
            str(readiness),
            "-JsonReportPath",
            str(tmp_path / "report.json"),
            "-MarkdownReportPath",
            str(tmp_path / "report.md"),
        ],
        cwd=REPO_ROOT,
        env=_clean_env(),
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    report = _json_from_stdout(result.stdout)
    assert report["candidate_generation_contract_check_passed"] is True
    assert report["policy"]["model_inference_performed"] is False
