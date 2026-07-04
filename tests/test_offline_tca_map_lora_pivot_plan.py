import argparse
import json
from pathlib import Path

from tca_map.smolvla import offline_tca_map_lora_pivot_plan as plan


def _write_provenance(path: Path, *, no_go=True):
    path.write_text(
        json.dumps(
            {
                "checkpoint_task_provenance_resolution_passed": True,
                "decision": "no_go_learned_policy_rollout_scaling" if no_go else "review_required",
            }
        ),
        encoding="utf-8",
    )


def _write_head(path: Path, *, passed=True):
    path.write_text(
        json.dumps(
            {
                "libero_offline_head_comparison_passed": passed,
                "arms": {
                    "actionmap_head_only_proxy": {"metrics": {"wrong_target_proxy_rate": 1.0}},
                    "tca_map_head_only_proxy": {"metrics": {"wrong_target_proxy_rate": 0.0}},
                },
                "comparison": {
                    "tca_map_vs_actionmap": {
                        "action_l1_delta": -0.1,
                        "wrong_target_proxy_rate_delta": -1.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_lora(path: Path, *, passed=True):
    path.write_text(
        json.dumps(
            {
                "libero_offline_lora_comparison_passed": passed,
                "arms": [
                    {"arm": "actionmap_lora", "metrics": {"wrong_target_proxy_rate": 1.0}},
                    {"arm": "tca_map_lora", "metrics": {"wrong_target_proxy_rate": 0.5}},
                ],
                "comparison": {
                    "tca_lora_vs_actionmap_lora": {
                        "action_l1_delta": -0.01,
                        "wrong_target_proxy_rate_delta": -0.5,
                    }
                },
            }
        ),
        encoding="utf-8",
    )


def _write_bounded(path: Path, *, passed=True):
    path.write_text(json.dumps({"libero_offline_bounded_pilot_report_passed": passed}), encoding="utf-8")


def _args(tmp_path: Path):
    provenance = tmp_path / "provenance.json"
    head = tmp_path / "head.json"
    lora = tmp_path / "lora.json"
    bounded = tmp_path / "bounded.json"
    _write_provenance(provenance)
    _write_head(head)
    _write_lora(lora)
    _write_bounded(bounded)
    return argparse.Namespace(
        provenance_report=str(provenance),
        head_report=str(head),
        lora_report=str(lora),
        bounded_pilot_report=str(bounded),
        report_path=str(tmp_path / "report.json"),
        markdown_report_path=str(tmp_path / "report.md"),
    )


def test_pivot_plan_selects_offline_evidence_table_after_provenance_no_go(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = plan.build_report(_args(tmp_path))

    assert code == 0
    assert report["offline_tca_map_lora_pivot_plan_passed"] is True
    assert report["decision"] == "pivot_offline_evidence_ladder"
    assert report["selected_next_step"] == "consolidate_offline_tca_lora_evidence_table_and_gap_report"
    assert report["ready_for_offline_evidence_table"] is True
    assert report["ready_for_learned_policy_rollout_scaling"] is False
    assert report["ready_for_paper_claim"] is False


def test_pivot_plan_refuses_forbidden_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_ROLLOUTS", "1")

    report, code = plan.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUTS" in report["recommended_next_step"]
    assert report["policy"]["rollouts_performed"] is False


def test_pivot_plan_requests_regeneration_when_offline_reports_missing(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    _write_lora(Path(args.lora_report), passed=False)

    report, code = plan.build_report(args)

    assert code == 0
    assert report["decision"] == "regenerate_offline_reports"
    assert report["selected_next_step"] == "rerun_missing_offline_head_or_lora_reports"


def test_pivot_plan_reviews_if_provenance_does_not_no_go(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    _write_provenance(Path(args.provenance_report), no_go=False)

    report, code = plan.build_report(args)

    assert code == 0
    assert report["decision"] == "review_required"
    assert report["selected_next_step"] == "review_rollout_provenance_before_pivot"
