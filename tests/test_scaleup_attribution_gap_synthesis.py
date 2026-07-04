import argparse
import json
from pathlib import Path

from tca_map.smolvla import scaleup_attribution_gap_synthesis as synth


def _write_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_stress_report(path: Path, passed: bool = True):
    _write_json(
        path,
        {
            "tca_select_ambiguity_stress_passed": passed,
            "record_count": 16,
            "policy": {
                "offline_proxy_only": True,
                "model_load_performed": False,
                "training_performed": False,
                "rollouts_performed": False,
                "gpu_jobs_performed": False,
                "openvla_oft_executed": False,
                "privileged_inference_used": False,
                "external_verifier_used": False,
            },
            "metrics": {
                "top_heatmap_wrong_target_proxy_rate": 1.0,
                "selected_wrong_target_proxy_rate": 0.0,
                "selection_wrong_target_proxy_delta_vs_top_heatmap": -1.0,
                "top_heatmap_action_l1": 0.164299,
                "selected_action_l1": 0.0,
                "selection_action_l1_delta_vs_top_heatmap": -0.164299,
                "latency_ms": 0.428231,
            },
        },
    )


def _args(tmp_path: Path, include_stress: bool = True):
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
                {"id": "tca_select_inference_attribution", "status": "offline_ambiguity_stress_proxy_present"},
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
    stress = tmp_path / "stress.json"
    if include_stress:
        _write_stress_report(stress)
    return argparse.Namespace(
        evidence_report=str(evidence),
        tca_select_stress_report=str(stress),
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
    assert report["tca_select_ambiguity_stress_included"] is True
    assert report["ready_for_learned_policy_rollout_scaling"] is False
    assert report["ready_for_paper_claim"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["rollouts_performed"] is False
    assert any("Distributional TCA-Select adds no extra LoRA proxy gain" in item for item in report["findings"])
    assert any("offline ambiguity stress test provides selection-specific proxy evidence" in item for item in report["findings"])
    assert report["input_summary"]["bounded_lora_wrong_target_delta"] == -0.44
    assert report["input_summary"]["tca_select_inference_attribution_gap_status"] == "offline_ambiguity_stress_proxy_present"
    assert report["input_summary"]["stress_selection_wrong_target_proxy_delta"] == -1.0
    assert report["input_summary"]["stress_selection_action_l1_delta"] == -0.164299
    assert "candidate-generation readiness check" in report["recommended_next_step"]


def test_scaleup_attribution_gap_synthesis_handles_missing_stress_report(tmp_path, monkeypatch):
    for gate in synth.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = synth.build_report(_args(tmp_path, include_stress=False))

    assert code == 0
    assert report["scaleup_attribution_gap_synthesis_passed"] is True
    assert report["tca_select_ambiguity_stress_included"] is False
    assert any("No TCA-Select ambiguity stress report was available" in item for item in report["findings"])
    assert report["input_summary"]["stress_selection_wrong_target_proxy_delta"] is None


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
