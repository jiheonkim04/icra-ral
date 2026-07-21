from __future__ import annotations

import hashlib
import json
from pathlib import Path


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
