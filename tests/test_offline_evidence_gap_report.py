import argparse
import json
from pathlib import Path

from tca_map.smolvla import offline_evidence_gap_report as report_mod


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _args(tmp_path: Path):
    pivot = tmp_path / "pivot.json"
    head = tmp_path / "head.json"
    lora = tmp_path / "lora.json"
    scaleup = tmp_path / "scaleup.json"
    provenance = tmp_path / "provenance.json"
    bounded = tmp_path / "bounded.json"
    _write_json(
        pivot,
        {
            "ready_for_offline_evidence_table": True,
            "ready_for_lora_scaleup_plan": True,
            "ready_for_libero_aligned_checkpoint_source_plan": True,
        },
    )
    _write_json(
        head,
        {
            "arms": {
                "actionmap_head_only_proxy": {
                    "metrics": {
                        "action_l1": 0.1,
                        "wrong_target_proxy_rate": 1.0,
                        "counterfactual_separation_margin": 0.0,
                        "offline_standard_proxy": 0.0,
                        "target_top1_accuracy": 0.0,
                    },
                    "trainable_parameter_count": 0,
                },
                "tca_map_head_only_proxy": {
                    "metrics": {
                        "action_l1": 0.0,
                        "wrong_target_proxy_rate": 0.0,
                        "counterfactual_separation_margin": 0.1,
                        "offline_standard_proxy": 1.0,
                        "target_top1_accuracy": 1.0,
                    },
                    "trainable_parameter_count": 0,
                },
                "tca_map_distributional_select_proxy": {
                    "metrics": {
                        "action_l1": 0.0,
                        "wrong_target_proxy_rate": 0.0,
                        "counterfactual_separation_margin": 0.1,
                        "offline_standard_proxy": 1.0,
                        "target_top1_accuracy": 1.0,
                    },
                    "trainable_parameter_count": 0,
                },
            },
            "comparison": {"tca_map_vs_actionmap": {"action_l1_delta": -0.1}},
        },
    )
    _write_json(
        lora,
        {
            "arms": [
                {
                    "arm": "actionmap_lora",
                    "metrics": {
                        "action_l1": 0.11,
                        "wrong_target_proxy_rate": 1.0,
                        "counterfactual_separation_margin": 0.0,
                        "offline_standard_proxy": 0.0,
                        "target_top1_accuracy": 0.0,
                    },
                    "trainable_lora_parameter_count": 84,
                },
                {
                    "arm": "tca_map_lora",
                    "metrics": {
                        "action_l1": 0.10,
                        "wrong_target_proxy_rate": 0.5,
                        "counterfactual_separation_margin": 0.01,
                        "offline_standard_proxy": 0.4,
                        "target_top1_accuracy": 0.5,
                    },
                    "trainable_lora_parameter_count": 168,
                },
                {
                    "arm": "tca_map_lora_distributional_select",
                    "metrics": {
                        "action_l1": 0.10,
                        "wrong_target_proxy_rate": 0.5,
                        "counterfactual_separation_margin": 0.01,
                        "offline_standard_proxy": 0.4,
                        "target_top1_accuracy": 0.5,
                    },
                    "trainable_lora_parameter_count": 168,
                },
            ],
            "comparison": {"tca_lora_vs_actionmap_lora": {"wrong_target_proxy_rate_delta": -0.5}},
        },
    )
    _write_json(
        scaleup,
        {
            "bounded_lora_offline_scaleup_passed": True,
            "record_count": 16,
            "arms": [
                {
                    "arm": "actionmap_lora",
                    "metrics": {
                        "action_l1": 0.12,
                        "wrong_target_proxy_rate": 1.0,
                        "counterfactual_separation_margin": 0.0,
                        "offline_standard_proxy": 0.0,
                        "target_top1_accuracy": 0.0,
                    },
                    "trainable_lora_parameter_count": 84,
                },
                {
                    "arm": "tca_map_lora",
                    "metrics": {
                        "action_l1": 0.10,
                        "wrong_target_proxy_rate": 0.56,
                        "counterfactual_separation_margin": -0.001,
                        "offline_standard_proxy": 0.38,
                        "target_top1_accuracy": 0.44,
                    },
                    "trainable_lora_parameter_count": 168,
                },
                {
                    "arm": "tca_map_lora_distributional_select",
                    "metrics": {
                        "action_l1": 0.10,
                        "wrong_target_proxy_rate": 0.56,
                        "counterfactual_separation_margin": -0.001,
                        "offline_standard_proxy": 0.38,
                        "target_top1_accuracy": 0.44,
                    },
                    "trainable_lora_parameter_count": 168,
                },
            ],
            "comparison": {
                "tca_lora_vs_actionmap_lora": {"wrong_target_proxy_rate_delta": -0.44},
                "tca_select_lora_vs_tca_lora": {"wrong_target_proxy_rate_delta": 0.0},
            },
        },
    )
    _write_json(provenance, {"recommended_next_step": "do not scale rollout"})
    _write_json(bounded, {"libero_offline_bounded_pilot_report_passed": True})
    return argparse.Namespace(
        pivot_report=str(pivot),
        head_report=str(head),
        lora_report=str(lora),
        bounded_lora_scaleup_report=str(scaleup),
        provenance_report=str(provenance),
        bounded_pilot_report=str(bounded),
        report_path=str(tmp_path / "report.json"),
        markdown_report_path=str(tmp_path / "report.md"),
    )


def test_evidence_gap_report_builds_table_and_blocks_claims(tmp_path, monkeypatch):
    for gate in report_mod.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = report_mod.build_report(_args(tmp_path))

    assert code == 0
    assert report["offline_evidence_gap_report_passed"] is True
    assert report["decision"] == "offline_evidence_table_ready"
    assert len(report["evidence_table"]) == 9
    assert report["bounded_lora_scaleup_included"] is True
    assert report["bounded_lora_scaleup_record_count"] == 16
    assert report["deltas"]["bounded_lora_tca_vs_actionmap_lora"]["wrong_target_proxy_rate_delta"] == -0.44
    assert report["ready_for_lora_scaleup_plan"] is True
    assert report["ready_for_learned_policy_rollout_scaling"] is False
    assert report["ready_for_paper_claim"] is False


def test_evidence_gap_report_refuses_forbidden_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_OPENVLA_OFT", "1")

    report, code = report_mod.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_OPENVLA_OFT" in report["recommended_next_step"]
    assert report["policy"]["openvla_oft_executed"] is False


def test_evidence_gap_report_requires_pivot_authorization(tmp_path, monkeypatch):
    for gate in report_mod.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    _write_json(Path(args.pivot_report), {"ready_for_offline_evidence_table": False})

    report, code = report_mod.build_report(args)

    assert code != 0
    assert report["decision"] == "stop"
    assert "did not authorize" in report["recommended_next_step"]
