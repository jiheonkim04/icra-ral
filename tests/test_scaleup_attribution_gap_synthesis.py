import argparse
import json
from pathlib import Path

from tca_map.smolvla import scaleup_attribution_gap_synthesis as synth


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _args(tmp_path: Path):
    evidence = tmp_path / "evidence.json"
    _write_json(
        evidence,
        {
            "offline_evidence_gap_report_passed": True,
            "bounded_lora_scaleup_included": True,
            "bounded_lora_scaleup_record_count": 16,
            "evidence_table": [{} for _ in range(9)],
            "gap_table": [
                {"id": "standard_success", "status": "blocked"},
                {"id": "learned_policy_rollout", "status": "blocked_for_current_checkpoint"},
                {"id": "required_lora_track", "status": "bounded_proxy_present"},
            ],
            "deltas": {
                "bounded_lora_tca_vs_actionmap_lora": {
                    "action_l1_delta": -0.004,
                    "wrong_target_proxy_rate_delta": -0.44,
                },
                "bounded_lora_tca_select_vs_tca_lora": {
                    "action_l1_delta": 0.0,
                    "wrong_target_proxy_rate_delta": 0.0,
                },
            },
        },
    )
    return argparse.Namespace(
        evidence_report=str(evidence),
        report_path=str(tmp_path / "report.json"),
        markdown_report_path=str(tmp_path / "report.md"),
    )


def test_scaleup_attribution_gap_synthesis_blocks_claims(tmp_path, monkeypatch):
    for gate in synth.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = synth.build_report(_args(tmp_path))

    assert code == 0
    assert report["scaleup_attribution_gap_synthesis_passed"] is True
    assert report["decision"] == "scaleup_attribution_gaps_ready"
    assert report["bounded_lora_scaleup_included"] is True
    assert report["ready_for_learned_policy_rollout_scaling"] is False
    assert report["ready_for_paper_claim"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert any("Distributional TCA-Select adds no extra LoRA proxy gain" in item for item in report["findings"])
    assert report["input_summary"]["bounded_lora_wrong_target_delta"] == -0.44


def test_scaleup_attribution_gap_synthesis_refuses_forbidden_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_TINY_TRAINING", "1")

    report, code = synth.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_TINY_TRAINING" in report["recommended_next_step"]
    assert report["policy"]["training_performed"] is False


def test_scaleup_attribution_gap_synthesis_requires_passed_evidence(tmp_path, monkeypatch):
    for gate in synth.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    _write_json(Path(args.evidence_report), {"offline_evidence_gap_report_passed": False})

    report, code = synth.build_report(args)

    assert code != 0
    assert "did not pass" in report["recommended_next_step"]
