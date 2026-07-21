from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest

from scripts import run_epoch9e_joint_continuation as continuation
from scripts import adjudicate_epoch9e_joint_continuation as final_adjudicator


ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load(name: str) -> dict:
    return json.loads((REPORTS / name).read_text(encoding="utf-8"))


def test_explicit_continuation_authority_and_scope_are_append_only() -> None:
    authority = load("epoch9e_failfast_continuation_authorization.json")
    correction = load("epoch9e_failfast_root_cause_and_scope_correction.json")
    assert authority["authority_sha256"] == "A70E137D2C92E0395F47E13CF99692702F0FDB86E2CC714B5C219D66618EC9E7"
    assert authority["starting_checkpoint"] == "4f57ecb94a3c84e0a5889bc0bd60cbd53ad415e8"
    assert authority["controller_or_scientific_change_authorized"] is False
    assert authority["rerun_20261134_authorized"] is False
    assert correction["historical_terminal_artifacts_edited"] is False
    assert correction["scientific_miss"]["immutable"] is True
    assert correction["scientific_miss"]["frozen_response_window_steps"] == 0
    assert correction["fixed_handling_20261134"]["exact_pair"].startswith("fixed adverse/nonflip")


def test_frozen_scientific_hashes_and_existing_traces_are_exact() -> None:
    correction = load("epoch9e_failfast_root_cause_and_scope_correction.json")
    hashes = correction["frozen_hashes"]
    for key in ("controller", "joint_protocol", "original_runner", "original_adjudicator", "original_host_wrapper", "original_execution_seal", "interrupted_result", "interrupted_adjudication"):
        row = hashes[key]
        assert sha256(ROOT / row["path"]) == row["sha256"]
    traces = hashes["existing_traces"]
    assert len(traces) == 4
    assert sum(not row["response_window_valid"] for row in traces) == 1
    for row in traces:
        assert sha256(ROOT / row["path"]) == row["sha256"]


def test_continuation_schedule_contains_only_untouched_committed_rows() -> None:
    schedule = load("epoch9e_continuation_schedule_audit.json")
    assert schedule["pending_primary_count"] == 22
    assert schedule["pending_sham_count"] == 12
    assert schedule["primary_identity_order"] == [identity for identity in range(20261135, 20261146) for _ in (0, 1)]
    assert schedule["primary_assignment_order"] == [value for _ in range(11) for value in ("A", "B")]
    assert schedule["preflight_actions_executed"] == 0
    assert schedule["preflight_reward_done_success_accessed"] is False
    assert schedule["pending_scientific_outcomes_opened"] is False
    assert all("20261134" in key for key in schedule["never_recompute_keys"])


def test_missing_pair_rule_is_frozen_conservative_and_cannot_help() -> None:
    sensitivity = load("epoch9e_missing_pair_sensitivity_protocol.json")
    observed_a = sensitivity["observed_assignment_A_back_heavy_response_m"]
    assert sensitivity["missing_contrast_physically_admissible_range_m"][0] == -observed_a
    assert sensitivity["worst_case_missing_contrast_m"] == -observed_a
    assert sensitivity["assignment_B_missing_response_physically_admissible_range_m"] == [0.0, 0.05]
    assert "nonpositive" in sensitivity["rules"]["binary_fixed_denominator"]
    assert "Never label" in sensitivity["rules"]["reporting"]
    assert sensitivity["outcomes_after_20261134_accessed_before_freeze"] is False


def test_missing_window_is_finalized_and_following_row_executes(monkeypatch: pytest.MonkeyPatch) -> None:
    manifests = [
        ("primary", {"scene_id": "missing", "probe_order": ["front", "back"]}),
        ("primary", {"scene_id": "following", "probe_order": ["front", "back"]}),
    ]
    executed: list[str] = []
    persisted: list[dict] = []

    def execute(_kind: str, manifest: dict) -> dict:
        executed.append(manifest["scene_id"])
        if manifest["scene_id"] == "missing":
            return {
                "row_key": "primary:missing",
                "completed": False,
                "exception": continuation.MISSING_PREFIX + "/frozen/missing_back.npz",
            }
        return {"row_key": "primary:following", "completed": True, "exception": None}

    monkeypatch.setattr(
        continuation,
        "structure_missing_response_failure",
        lambda row, _manifest: {**row, "failure_class": continuation.MISSING_CLASS},
    )
    continuation.process_schedule(manifests, set(), execute, persisted.append)
    assert executed == ["missing", "following"]
    assert persisted[0]["completed"] is False
    assert persisted[0]["failure_class"] == continuation.MISSING_CLASS
    assert persisted[1]["completed"] is True


def test_non_authorized_failure_is_not_swallowed_and_following_row_does_not_execute() -> None:
    manifests = [
        ("primary", {"scene_id": "infrastructure"}),
        ("primary", {"scene_id": "must_not_run"}),
    ]
    executed: list[str] = []
    persisted: list[dict] = []

    def execute(_kind: str, manifest: dict) -> dict:
        executed.append(manifest["scene_id"])
        return {"row_key": f"primary:{manifest['scene_id']}", "completed": False, "exception": "OSError: device lost"}

    with pytest.raises(RuntimeError, match="stops safely"):
        continuation.process_schedule(manifests, set(), execute, persisted.append)
    assert executed == ["infrastructure"]
    assert persisted[0]["failure_class"] == "UNAUTHORIZED_OR_INFRASTRUCTURE_FAILURE"


def test_genuine_executor_exception_propagates_without_being_classified() -> None:
    persisted: list[dict] = []

    def execute(_kind: str, _manifest: dict) -> dict:
        raise MemoryError("synthetic infrastructure stop")

    with pytest.raises(MemoryError, match="synthetic infrastructure stop"):
        continuation.process_schedule([("primary", {"scene_id": "x"})], set(), execute, persisted.append)
    assert persisted == []


def test_continuation_wrapper_calls_frozen_scientific_functions_only() -> None:
    source = inspect.getsource(continuation.main)
    assert "frozen.run_primary" in source
    assert "frozen.run_sham" in source
    assert "run_nondrag_probe" not in source
    assert "probe_candidate" not in source
    assert "back_heavy_threshold_m" not in source
    assert "20261134" not in inspect.getsource(continuation.process_schedule)


def test_historical_missing_trace_is_finite_contacted_but_never_a_valid_response() -> None:
    original = load("epoch9b_v2_task_preservation_protocol.json")
    path = REPORTS / "epoch9e_joint_certification" / "traces" / "epoch9e_joint_base_20261134_assignment_B_back.npz"
    audit = final_adjudicator.trace_probe_audit(path, "back", original)
    assert audit["finite_bounded_actions"] is True
    assert audit["sampled_target_contact"] is True
    assert audit["intended_contact_or_excitation"] is True
    assert audit["response_window_steps"] == 0
    assert audit["response_window_valid"] is False
    assert audit["nondrag_liftoff_planar_commands_exact_zero"] is True


def test_final_gate_function_keeps_original_denominators_and_sensitivity() -> None:
    counts = {
        "finite_bounded_actions": 48,
        "intended_contact_or_excitation": 46,
        "both_candidates_excited": 22,
        "full_trajectory_lane_reachable": 48,
        "collisions": 0,
        "identity_swaps": 0,
        "falls": 0,
        "workspace_exits": 0,
        "unrecoverable_track_losses": 0,
        "rank_correct": 20,
        "rank_by_heavy_position": {"front": {"correct": 10, "total": 12}, "back": {"correct": 10, "total": 12}},
        "exact_pair_correct_flips": 9,
        "completion_oracle": 20,
        "completion_by_heavy_position": {"front": {"success": 9, "total": 12}, "back": {"success": 11, "total": 12}},
    }
    paired = {
        "fixed_denominator_one_sided_exact_sign_p": 0.009,
        "complete_case_student_t_95_interval_m": [0.001, 0.003],
        "complete_case_adjusted_hc3": {"valid": True, "estimate_m": 0.002, "hc3_95_interval_m": [0.0001, 0.004]},
        "worst_case_augmented_student_t_95_interval_m": [0.0002, 0.003],
        "worst_case_augmented_adjusted_hc3": {"valid": True, "estimate_m": 0.001, "hc3_95_interval_m": [0.00001, 0.003]},
    }
    controls = {"position_order": True, "sham": True}
    integrity = {"complete_fixed_manifest": True, "controller_and_information_boundary": True, "trace_hashes_and_disclosures": True, "execution_and_resource": True}
    gates = final_adjudicator.final_gates(counts, paired, controls, integrity)
    assert all(gates.values())
    paired["worst_case_augmented_student_t_95_interval_m"] = [-0.001, 0.003]
    assert final_adjudicator.final_gates(counts, paired, controls, integrity)["worst_case_sensitivity_student_t_interval_positive"] is False


def test_tipping_point_reports_full_range_survival_only_when_worst_case_passes() -> None:
    positive = final_adjudicator.tipping_point([0.02] * 11, [-0.001, 0.01])
    assert positive["classification"] == "SURVIVES_FULL_ADMISSIBLE_RANGE"
    adverse = final_adjudicator.tipping_point([0.001] * 11, [-0.05, 0.05])
    assert adverse["classification"] in {"INTERIOR_TIPPING_POINT", "NO_ADMISSIBLE_VALUE_RESCUES_INTERVAL"}


def test_host_wrapper_captures_authoritative_python_exit_and_only_resumes_missing_keys() -> None:
    source = (ROOT / "scripts" / "run_epoch9e_joint_continuation_host.ps1").read_text(encoding="utf-8")
    assert "authoritative_runner_exit_code" in source
    assert "runner_exit_code_attempt_" in source
    assert "code=$?" in source
    assert "--resume" in source
    assert "missing_key_only" in source
    assert "run_epoch9e_joint_continuation.py" in source
    assert "run_epoch9e_joint_certification.py" not in source
