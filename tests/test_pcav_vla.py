import numpy as np
import pytest

from tca_map.smolvla.pcav_vla import (
    EXPANDED_PHASE_QUOTAS,
    INITIAL_PHASE_QUOTAS,
    PROPOSAL_HASH,
    TARGET_TASK_IDENTITIES,
    action_validity,
    candidate_diversity,
    classify_stage0a,
    grouped_action_error,
    oracle_headroom,
    partition_identity_audit,
    phase_for_frame,
    row_key,
    select_stage0_rows,
    stable_seed,
    validate_partial_payload,
)


def _population() -> list[dict[str, object]]:
    rows = []
    for task in TARGET_TASK_IDENTITIES:
        for episode in range(30):
            for frame in range(0, 90):
                rows.append(
                    {
                        "task_identity": task,
                        "episode": episode,
                        "frame": frame,
                        "episode_length": 100,
                    }
                )
    return rows


def _completed_row(task: str = TARGET_TASK_IDENTITIES[0], episode: int = 0, frame: int = 0) -> dict:
    return {
        "task_identity": task,
        "episode": episode,
        "frame": frame,
        "phase": "early",
        "candidates": [{"candidate_index": index} for index in range(4)],
    }


def _passing_audit(**overrides: object) -> dict[str, object]:
    audit: dict[str, object] = {
        "exception_count": 0,
        "duplicate_key_count": 0,
        "mapping_passed": True,
        "partition_passed": True,
        "reload_passed": True,
        "source_health_passed": True,
        "manifest_passed": True,
        "base_identity_max_abs_error": 0.0,
        "all_base_candidates_valid": True,
        "fraction_rows_with_valid_alternative": 1.0,
        "confirmatory_observations_decoded": 0,
        "confirmatory_actions_computed": 0,
        "fraction_rows_with_two_unique_chunks": 1.0,
        "median_nonzero_pairwise_rms_l2": 0.01,
        "completed_row_count": 24,
        "headroom": {
            "pass_threshold": True,
            "strictly_better_row_count": 8,
        },
    }
    audit.update(overrides)
    return audit


def test_stable_seed_is_deterministic_and_partitioned() -> None:
    assert stable_seed(PROPOSAL_HASH, "row", 0) == stable_seed(PROPOSAL_HASH, "row", 0)
    assert stable_seed(PROPOSAL_HASH, "row", 0) != stable_seed(PROPOSAL_HASH, "row", 1)


def test_phase_boundaries_require_future_target() -> None:
    assert phase_for_frame(0, 100) == "early"
    assert phase_for_frame(30, 100) == "middle"
    assert phase_for_frame(60, 100) == "late"
    with pytest.raises(ValueError, match="valid future"):
        phase_for_frame(90, 100)


def test_row_selection_obeys_exact_phase_quotas_and_expansion_is_superset() -> None:
    population = _population()
    initial = select_stage0_rows(population, INITIAL_PHASE_QUOTAS)
    expanded = select_stage0_rows(population, EXPANDED_PHASE_QUOTAS)
    assert len(initial) == 24
    assert len(expanded) == 96
    assert {row_key(row) for row in initial} <= {row_key(row) for row in expanded}
    for task in TARGET_TASK_IDENTITIES:
        task_rows = [row for row in initial if row["task_identity"] == task]
        assert sum(row["phase"] == "early" for row in task_rows) == 3
        assert sum(row["phase"] == "middle" for row in task_rows) == 2
        assert sum(row["phase"] == "late" for row in task_rows) == 3


def test_partition_audit_detects_overlap_and_duplicates() -> None:
    row = _completed_row()
    clean = partition_identity_audit({"discovery": [row], "validation": [_completed_row(episode=1)]})
    assert clean["passed"] is True
    dirty = partition_identity_audit({"discovery": [row, row], "validation": [row]})
    assert dirty["passed"] is False
    assert dirty["duplicate_counts"]["discovery"] == 1
    assert dirty["pairwise_overlap_counts"]["discovery__validation"] == 1


def test_candidate_diversity_uses_full_chunk_hashes() -> None:
    base = np.zeros((50, 7), dtype=np.float32)
    changed = base.copy()
    changed[0, 0] = 0.1
    audit = candidate_diversity([base, changed, changed.copy(), base.copy()])
    assert audit["unique_chunk_count"] == 2
    assert audit["median_nonzero_pairwise_rms_l2"] > 0.0


def test_grouped_action_error_separates_units() -> None:
    expert = np.zeros((50, 7), dtype=np.float32)
    candidate = expert.copy()
    candidate[:10, 0] = 1.0
    metrics = grouped_action_error(
        candidate,
        expert,
        {"translation": 1.0, "rotation": 1.0, "gripper": 1.0},
    )
    assert metrics["translation"] == 1.0
    assert metrics["rotation"] == 0.0
    assert metrics["gripper"] == 0.0
    assert metrics["aggregate"] == pytest.approx(1.0 / 3.0)


def test_oracle_headroom_requires_fraction_and_effect() -> None:
    passing = oracle_headroom([[1.0, 0.9, 1.1, 1.2]] * 6 + [[1.0, 1.0, 1.1, 1.2]] * 18)
    assert passing["materially_better_fraction"] == 0.25
    assert passing["pass_threshold"] is True
    absent = oracle_headroom([[1.0, 1.0, 1.1, 1.2]] * 24)
    assert absent["strictly_better_row_count"] == 0
    assert absent["pass_threshold"] is False


def test_action_validity_is_base_relative_and_absolute() -> None:
    base = np.zeros((50, 7), dtype=np.float32)
    valid = base.copy()
    valid[0, 0] = 1.1
    assert action_validity(valid, base)["passed"] is True
    invalid = base.copy()
    invalid[:, 0] = 1.3
    audit = action_validity(invalid, base)
    assert audit["passed"] is False
    assert audit["observed"]["absolute_max"] > audit["limits"]["absolute_max"]


def test_partial_validation_rejects_duplicates_and_bad_candidates() -> None:
    row = _completed_row()
    payload = {
        "proposal_hash": PROPOSAL_HASH,
        "planned_row_count": 24,
        "completed_row_count": 1,
        "completed_row_keys": [row_key(row)],
        "rows": [row],
        "exception_count": 0,
    }
    assert validate_partial_payload(payload)["missing_row_count"] == 23
    duplicate = dict(payload)
    duplicate["completed_row_count"] = 2
    duplicate["completed_row_keys"] = [row_key(row), row_key(row)]
    duplicate["rows"] = [row, row]
    with pytest.raises(ValueError, match="duplicate"):
        validate_partial_payload(duplicate)
    malformed = dict(payload)
    malformed["rows"] = [{**row, "candidates": row["candidates"][:3]}]
    with pytest.raises(ValueError, match="four candidates"):
        validate_partial_payload(malformed)


def test_stage0_decisions_separate_pass_expansion_headroom_and_implementation() -> None:
    assert classify_stage0a(_passing_audit()) == "PCAV_STAGE_0A_PASS_STAGE_0B_ALLOWED"

    unresolved = _passing_audit(
        headroom={"pass_threshold": False, "strictly_better_row_count": 2}
    )
    assert classify_stage0a(unresolved) == "PCAV_STAGE_0A_UNRESOLVED_EXPANSION_REQUIRED"

    expanded = dict(unresolved)
    expanded["completed_row_count"] = 96
    assert classify_stage0a(expanded) == "PCAV_STAGE_0A_NO_USABLE_HEADROOM"

    no_headroom = _passing_audit(
        headroom={"pass_threshold": False, "strictly_better_row_count": 0}
    )
    assert classify_stage0a(no_headroom) == "PCAV_STAGE_0A_NO_USABLE_HEADROOM"

    collapsed = _passing_audit(fraction_rows_with_two_unique_chunks=0.5)
    assert classify_stage0a(collapsed) == "PCAV_STAGE_0A_DESIGN_FAILURE_CANDIDATES_COLLAPSED"

    broken = _passing_audit(exception_count=1)
    assert classify_stage0a(broken) == "PCAV_STAGE_0A_IMPLEMENTATION_OR_DATA_FAILURE"

    bad_manifest = _passing_audit(manifest_passed=False)
    assert classify_stage0a(bad_manifest) == "PCAV_STAGE_0A_IMPLEMENTATION_OR_DATA_FAILURE"

    no_valid_alternative = _passing_audit(fraction_rows_with_valid_alternative=0.5)
    assert classify_stage0a(no_valid_alternative) == "PCAV_STAGE_0A_DESIGN_FAILURE_CANDIDATES_COLLAPSED"
