import argparse
import json
from pathlib import Path

from tca_map.smolvla import tca_select_ambiguity_stress_plan as plan


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _args(tmp_path: Path):
    synthesis = tmp_path / "synthesis.json"
    _write_json(
        synthesis,
        {
            "scaleup_attribution_gap_synthesis_passed": True,
            "bounded_lora_scaleup_included": True,
            "ready_for_paper_claim": False,
            "input_summary": {
                "bounded_select_action_l1_delta": 0.0,
                "bounded_select_wrong_target_delta": 0.0,
            },
            "recommended_next_step": "Create a report-only TCA-Select stress-test plan",
        },
    )
    return argparse.Namespace(
        synthesis_report=str(synthesis),
        max_pairs=16,
        max_records=64,
        candidate_count=8,
        temperature=0.5,
        max_runtime_seconds=300,
        report_path=str(tmp_path / "report.json"),
        markdown_report_path=str(tmp_path / "report.md"),
    )


def test_tca_select_ambiguity_stress_plan_passes_and_blocks_claims(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = plan.build_report(_args(tmp_path))

    assert code == 0
    assert report["tca_select_ambiguity_stress_plan_passed"] is True
    assert report["decision"] == "proceed_offline_tca_select_ambiguity_stress_runner"
    assert report["ready_for_offline_tca_select_ambiguity_stress_runner"] is True
    assert report["ready_for_paper_claim"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert "selection_wrong_target_proxy_delta_vs_top_heatmap" in report["planned_metrics"]
    assert "external verifier model" in report["stress_test_design"]["forbidden_inputs"]


def test_tca_select_ambiguity_stress_plan_refuses_forbidden_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_ROLLOUTS", "1")

    report, code = plan.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUTS" in report["recommended_next_step"]
    assert report["policy"]["rollouts_performed"] is False


def test_tca_select_ambiguity_stress_plan_requires_synthesis(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    _write_json(Path(args.synthesis_report), {"scaleup_attribution_gap_synthesis_passed": False})

    report, code = plan.build_report(args)

    assert code != 0
    assert "did not pass" in report["recommended_next_step"]
