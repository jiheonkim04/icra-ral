import argparse
import json
from pathlib import Path

from tca_map.smolvla import bounded_lora_offline_scaleup_plan as plan


def _write_evidence(path: Path, *, passed=True, ready=True):
    path.write_text(
        json.dumps(
            {
                "offline_evidence_gap_report_passed": passed,
                "decision": "offline_evidence_table_ready",
                "ready_for_lora_scaleup_plan": ready,
                "ready_for_offline_proxy_extension": True,
                "ready_for_learned_policy_rollout_scaling": False,
                "evidence_table": [
                    {"arm": "ActionMap + LoRA"},
                    {"arm": "TCA-Map + LoRA"},
                    {"arm": "TCA-Map + LoRA + Distributional TCA-Select"},
                ],
            }
        ),
        encoding="utf-8",
    )


def _args(tmp_path: Path):
    evidence = tmp_path / "evidence.json"
    _write_evidence(evidence)
    return argparse.Namespace(
        evidence_gap_report=str(evidence),
        max_pairs=16,
        max_samples=64,
        max_steps=64,
        max_runtime_minutes=20,
        lora_rank=4,
        report_path=str(tmp_path / "plan.json"),
        markdown_report_path=str(tmp_path / "plan.md"),
    )


def test_plan_authorizes_bounded_cpu_only_lora_runner(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = plan.build_report(_args(tmp_path))

    assert code == 0
    assert report["bounded_lora_offline_scaleup_plan_passed"] is True
    assert report["decision"] == "proceed_bounded_offline_lora_scaleup_runner"
    assert report["ready_for_bounded_lora_offline_scaleup_runner"] is True
    assert report["ready_for_learned_policy_rollout_scaling"] is False
    assert report["limits"]["max_steps"] == 64
    assert report["limits"]["device"] == "cpu"
    assert report["limits"]["full_finetuning_allowed"] is False
    assert report["required_future_gates"] == ["ALLOW_TINY_TRAINING=1"]


def test_plan_clamps_requested_limits_to_safe_defaults(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    args.max_pairs = 999
    args.max_samples = 999
    args.max_steps = 999
    args.max_runtime_minutes = 999
    args.lora_rank = 999

    report, code = plan.build_report(args)

    assert code == 0
    assert report["limits"]["max_pairs"] == 16
    assert report["limits"]["max_samples"] == 64
    assert report["limits"]["max_steps"] == 64
    assert report["limits"]["max_runtime_minutes"] == 20
    assert report["limits"]["lora_rank"] == 4


def test_plan_refuses_forbidden_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_TINY_TRAINING", "1")

    report, code = plan.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_TINY_TRAINING" in report["recommended_next_step"]
    assert report["policy"]["training_performed"] is False


def test_plan_requires_evidence_authorization(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    _write_evidence(Path(args.evidence_gap_report), ready=False)

    report, code = plan.build_report(args)

    assert code != 0
    assert report["decision"] == "stop"
    assert "did not authorize" in report["recommended_next_step"]
