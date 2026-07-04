import argparse
import json
from pathlib import Path

from tca_map.smolvla import real_candidate_generation_smoke_plan as plan


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _args(tmp_path: Path):
    contract = tmp_path / "contract.json"
    runtime = tmp_path / "runtime.json"
    load = tmp_path / "load.json"
    single = tmp_path / "single.json"
    _write_json(
        contract,
        {
            "candidate_generation_contract_check_passed": True,
            "ready_for_real_candidate_generation_smoke_plan": True,
        },
    )
    _write_json(
        runtime,
        {
            "runtime_dependencies": {"ready_for_load_only_runtime": True},
            "gpu": {"memory_total_mb": 16303},
        },
    )
    _write_json(load, {"result": {"passed": True}})
    _write_json(single, {"result": {"passed": True}, "interface": {"cuda_max_allocated_mb": 0}})
    return argparse.Namespace(
        contract_report=str(contract),
        runtime_deps_report=str(runtime),
        load_only_report=str(load),
        single_sample_report=str(single),
        report_path=str(tmp_path / "report.json"),
        markdown_report_path=str(tmp_path / "report.md"),
    )


def test_real_candidate_generation_smoke_plan_green_without_execution(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = plan.build_report(_args(tmp_path))

    assert code == 0
    assert report["real_candidate_generation_smoke_plan_passed"] is True
    assert report["decision"] == "proceed_bounded_real_candidate_generation_smoke_implementation"
    assert report["ready_for_real_candidate_generation_smoke_implementation"] is True
    assert report["ready_for_real_candidate_generation_smoke_execution"] is False
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["model_inference_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert report["implementation_constraints"]["candidate_count_max"] == 4
    assert "ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1" in report["required_future_gates"]


def test_real_candidate_generation_smoke_plan_refuses_execution_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_REAL_CANDIDATE_GENERATION_SMOKE", "1")

    report, code = plan.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_REAL_CANDIDATE_GENERATION_SMOKE" in report["recommended_next_step"]
    assert report["policy"]["model_inference_performed"] is False


def test_real_candidate_generation_smoke_plan_records_blockers(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    _write_json(Path(args.runtime_deps_report), {"runtime_dependencies": {"ready_for_load_only_runtime": False}})

    report, code = plan.build_report(args)

    assert code == 0
    assert report["real_candidate_generation_smoke_plan_passed"] is True
    assert report["decision"] == "blocked"
    assert report["ready_for_real_candidate_generation_smoke_implementation"] is False
    assert any("runtime dependency" in item for item in report["blockers"])
