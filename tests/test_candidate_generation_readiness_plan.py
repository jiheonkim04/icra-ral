import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tca_map.smolvla import candidate_generation_readiness_plan as plan


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "130_plan_candidate_generation_readiness.ps1"


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _args(tmp_path: Path):
    synthesis = tmp_path / "synthesis.json"
    load = tmp_path / "load.json"
    single = tmp_path / "single.json"
    feature = tmp_path / "feature.json"
    _write_json(
        synthesis,
        {
            "scaleup_attribution_gap_synthesis_passed": True,
            "tca_select_ambiguity_stress_included": True,
            "ready_for_paper_claim": False,
            "input_summary": {
                "evidence_row_count": 10,
                "tca_select_inference_attribution_gap_status": "offline_ambiguity_stress_proxy_present",
            },
            "recommended_next_step": "Plan a report-only learned-policy candidate-generation readiness check",
        },
    )
    _write_json(load, {"result": {"passed": True}})
    _write_json(single, {"result": {"passed": True}})
    _write_json(feature, {"cache_valid": True})
    return argparse.Namespace(
        synthesis_report=str(synthesis),
        load_only_report=str(load),
        single_sample_report=str(single),
        feature_cache_report=str(feature),
        report_path=str(tmp_path / "report.json"),
        markdown_report_path=str(tmp_path / "report.md"),
    )


def _clean_env(extra=None):
    env = os.environ.copy()
    for gate in plan.FORBIDDEN_GATES:
        env.pop(gate, None)
    env.update(extra or {})
    return env


def _json_from_stdout(stdout: str):
    start = stdout.find("{")
    assert start >= 0, stdout
    return json.loads(stdout[start:])


def test_candidate_generation_readiness_plan_passes_without_execution(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = plan.build_report(_args(tmp_path))

    assert code == 0
    assert report["candidate_generation_readiness_plan_passed"] is True
    assert report["decision"] == "candidate_generation_readiness_plan_ready"
    assert report["ready_for_candidate_generation_contract_checker"] is True
    assert report["ready_for_real_candidate_generation_smoke_plan"] is True
    assert report["ready_for_real_candidate_generation_smoke_execution"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert "external verifier model" in report["planned_contract"]["forbidden_future_inputs"]
    assert report["candidate_generation_blockers"] == []


def test_candidate_generation_readiness_plan_refuses_inference_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_SINGLE_SAMPLE_INFERENCE", "1")

    report, code = plan.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_SINGLE_SAMPLE_INFERENCE" in report["recommended_next_step"]
    assert report["policy"]["model_inference_performed"] is False


def test_candidate_generation_readiness_plan_records_missing_prior_smoke(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    Path(args.single_sample_report).unlink()

    report, code = plan.build_report(args)

    assert code == 0
    assert report["candidate_generation_readiness_plan_passed"] is True
    assert report["ready_for_candidate_generation_contract_checker"] is True
    assert report["ready_for_real_candidate_generation_smoke_plan"] is True
    assert any("single-sample interface" in item for item in report["candidate_generation_blockers"])


def test_candidate_generation_readiness_plan_script_runs(tmp_path):
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for candidate-generation readiness script tests")

    args = _args(tmp_path)
    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-Python",
            sys.executable,
            "-SynthesisReportPath",
            args.synthesis_report,
            "-LoadOnlyReportPath",
            args.load_only_report,
            "-SingleSampleReportPath",
            args.single_sample_report,
            "-FeatureCacheReportPath",
            args.feature_cache_report,
            "-ReportPath",
            args.report_path,
            "-MarkdownReportPath",
            args.markdown_report_path,
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
    assert report["candidate_generation_readiness_plan_passed"] is True
    assert report["policy"]["heavy_model_imports_performed"] is False
