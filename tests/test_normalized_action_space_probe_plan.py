import argparse
import json
from pathlib import Path

from tca_map.smolvla import normalized_action_space_probe_plan as plan


def _write_audit(path: Path, *, passed=True, prefixes=None, scale=True, dim=True):
    if prefixes is None:
        prefixes = ["so100", "so100-blue"]
    path.write_text(
        json.dumps(
            {
                "libero_action_stat_subset_audit_passed": passed,
                "decision": "no_go_rollout_scaling",
                "libero_action_stats": {
                    "count": 2500,
                    "dim": 7,
                    "max_abs": 1.0,
                },
                "comparison_to_checkpoint": {
                    "checkpoint_action_mean_max_abs": 125.0 if scale else 0.4,
                    "checkpoint_action_std_max": 59.0 if scale else 0.2,
                    "checkpoint_action_stat_prefixes": prefixes,
                    "scale_mismatch_confirmed": scale,
                    "dimension_mismatch_confirmed": dim,
                    "policy_action_shape": [6],
                    "libero_action_dim": 7,
                },
            }
        ),
        encoding="utf-8",
    )


def _write_vlm_summary(path: Path):
    path.write_text(
        json.dumps({"comparison": {"vlm_enabled_alignment_signal": "weak"}}),
        encoding="utf-8",
    )


def _args(tmp_path: Path):
    audit = tmp_path / "libero_action_stats.json"
    vlm = tmp_path / "vlm_summary.json"
    _write_audit(audit)
    _write_vlm_summary(vlm)
    return argparse.Namespace(
        libero_action_stat_report=str(audit),
        vlm_summary_report=str(vlm),
        report_path=str(tmp_path / "plan.json"),
        markdown_report_path=str(tmp_path / "plan.md"),
    )


def test_plan_selects_checkpoint_provenance_resolution_for_so100_mismatch(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)

    report, code = plan.build_report(_args(tmp_path))

    assert code == 0
    assert report["normalized_action_space_probe_plan_passed"] is True
    assert report["decision"] == "reduce_scope"
    assert report["selected_next_step"] == "checkpoint_task_provenance_resolution"
    assert report["ready_for_checkpoint_task_provenance_resolution"] is True
    assert report["ready_for_bounded_normalized_action_space_probe_runner"] is False
    assert report["ready_for_rollout_scaling"] is False


def test_plan_can_route_to_normalized_probe_plan_when_mismatch_is_not_so100(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    _write_audit(Path(args.libero_action_stat_report), prefixes=["unknown"], scale=True, dim=True)

    report, code = plan.build_report(args)

    assert code == 0
    assert report["decision"] == "reduce_scope"
    assert report["selected_next_step"] == "bounded_normalized_action_space_probe_plan"
    assert report["ready_for_checkpoint_task_provenance_resolution"] is True
    assert report["ready_for_bounded_normalized_action_space_probe_runner"] is False


def test_plan_refuses_forbidden_gate(tmp_path, monkeypatch):
    monkeypatch.setenv("ALLOW_NORMALIZED_ACTION_SPACE_PROBE", "1")

    report, code = plan.build_report(_args(tmp_path))

    assert code != 0
    assert report["decision"] == "stop"
    assert "ALLOW_NORMALIZED_ACTION_SPACE_PROBE" in report["recommended_next_step"]
    assert report["policy"]["model_load_performed"] is False
    assert report["policy"]["rollouts_performed"] is False


def test_plan_requires_passed_libero_action_stat_audit(tmp_path, monkeypatch):
    for gate in plan.FORBIDDEN_GATES:
        monkeypatch.delenv(gate, raising=False)
    args = _args(tmp_path)
    _write_audit(Path(args.libero_action_stat_report), passed=False)

    report, code = plan.build_report(args)

    assert code != 0
    assert report["decision"] == "stop"
    assert "did not pass" in report["recommended_next_step"]
