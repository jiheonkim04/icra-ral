import argparse
import json
from pathlib import Path

from tca_map.smolvla import real_candidate_generation_smoke_summary as summary


def _write_smoke(path: Path, *, passed: bool = True, wrong_target: bool = False) -> None:
    path.write_text(
        json.dumps(
            {
                "real_candidate_generation_smoke_passed": passed,
                "policy": {
                    "bounded_single_sample_only": True,
                    "downloads_performed": False,
                    "training_performed": False,
                    "rollouts_performed": False,
                    "simulator_environment_created": False,
                    "openvla_oft_executed": False,
                    "tokens_read_or_written": False,
                    "paper_grade_claims_made": False,
                    "external_verifier_used": False,
                    "privileged_inference_used": False,
                    "model_inference_performed": passed,
                    "candidate_generation_performed": passed,
                },
                "result": {"passed": passed, "elapsed_sec": 38.2},
                "runtime": {
                    "device": "cpu",
                    "single_sample_inference_elapsed_sec": 1.7,
                    "selection_latency_ms": 0.3,
                    "cuda_max_allocated_mb": 0.0,
                },
                "generation": {
                    "candidate_count": 4,
                    "heatmap_grid": 8,
                    "action_dim": 6,
                },
                "selection": {
                    "selected_candidate_index": 0,
                    "selected_target_index": 0 if not wrong_target else 1,
                    "selected_action_l1_to_seed": 0.0,
                    "wrong_target_proxy": wrong_target,
                },
            }
        ),
        encoding="utf-8",
    )


def _args(tmp_path: Path) -> argparse.Namespace:
    smoke = tmp_path / "smoke.json"
    _write_smoke(smoke)
    return argparse.Namespace(
        smoke_report=str(smoke),
        report_path=str(tmp_path / "summary.json"),
        markdown_report_path=str(tmp_path / "summary.md"),
    )


def test_real_candidate_generation_summary_marks_engineering_evidence_ready(tmp_path, monkeypatch):
    for gate in summary.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = summary.build_report(_args(tmp_path))

    assert code == 0
    assert report["decision"] == "real_candidate_generation_smoke_engineering_evidence_ready"
    assert report["metrics"]["smoke_passed"] is True
    assert report["metrics"]["candidate_count"] == 4
    assert report["metrics"]["wrong_target_proxy"] is False
    assert report["ready_for_candidate_generation_comparison_plan"] is True
    assert report["policy"]["model_inference_performed"] is False
    assert report["ready_for_paper_claim"] is False


def test_real_candidate_generation_summary_refuses_execution_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_REAL_CANDIDATE_GENERATION_SMOKE", "1")

    report, code = summary.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_REAL_CANDIDATE_GENERATION_SMOKE" in report["recommended_next_step"]
    assert report["policy"]["model_inference_performed"] is False


def test_real_candidate_generation_summary_records_wrong_target_gap(tmp_path, monkeypatch):
    for gate in summary.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    _write_smoke(Path(args.smoke_report), wrong_target=True)

    report, code = summary.build_report(args)

    assert code == 0
    assert report["decision"] == "reduce_scope"
    assert report["ready_for_candidate_generation_comparison_plan"] is False
    assert any("wrong-target" in gap for gap in report["gaps"])
