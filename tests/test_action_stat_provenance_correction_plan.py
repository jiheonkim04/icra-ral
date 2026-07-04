import argparse
import json
from pathlib import Path

from tca_map.smolvla import action_stat_provenance_correction_plan as plan


def _write_audit(path: Path, *, passed=True, mismatch=True):
    path.write_text(
        json.dumps(
            {
                "action_normalization_provenance_audit_passed": passed,
                "decision": "no_go_rollout_scaling",
                "diagnosis": {
                    "checkpoint_action_stats_appear_non_libero_scale": mismatch,
                    "checkpoint_action_stats_prefix_mismatch_risk": mismatch,
                    "libero_expert_actions_appear_unit_scaled": True,
                    "policy_action_shape": [6],
                    "config_action_normalization": "MEAN_STD",
                },
                "action_stats": {
                    "action_stat_prefixes": ["so100"],
                    "action_mean_range": {"max_abs": 120.0},
                    "action_std_range": {"max": 50.0},
                },
                "sample_action_ranges": {
                    "expert_action_preview_range": {"max_abs": 1.0},
                    "clipped_values_total": 3,
                },
            }
        ),
        encoding="utf-8",
    )


def _args(tmp_path: Path):
    audit = tmp_path / "audit.json"
    _write_audit(audit)
    return argparse.Namespace(
        audit_report=str(audit),
        report_path=str(tmp_path / "plan.json"),
        markdown_report_path=str(tmp_path / "plan.md"),
    )


def test_plan_selects_libero_action_stat_audit_when_mismatch_present(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = plan.build_report(_args(tmp_path))

    assert code == 0
    assert report["decision"] == "reduce_scope"
    assert report["action_stat_provenance_correction_plan_passed"] is True
    assert report["ready_for_libero_action_stat_audit"] is True
    assert report["selected_next_step"] == "libero_action_stat_subset_audit"
    assert report["ready_for_rollout_scaling"] is False


def test_plan_refuses_forbidden_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_ROLLOUTS", "1")

    report, code = plan.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_ROLLOUTS" in report["recommended_next_step"]
    assert report["policy"]["rollouts_performed"] is False


def test_plan_requires_passed_audit(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    _write_audit(Path(args.audit_report), passed=False)

    report, code = plan.build_report(args)

    assert code != 0
    assert report["decision"] == "stop"
    assert "did not pass" in report["recommended_next_step"]
