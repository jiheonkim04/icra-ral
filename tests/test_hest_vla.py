from __future__ import annotations

import numpy as np
import pytest

from tca_map.smolvla.hest_vla import (
    ACTION_DIM,
    HORIZON,
    Stage0ADecisionInputs,
    classify_stage0a,
    cumulative_arm_energy,
    gripper_transition,
    hest_transform,
    moving_average_control,
    no_endpoint_ablation,
    parse_sha256_registry,
    smooth_cumulative_path,
    spline_proxy,
    support_bounds,
    support_valid,
    validate_manifest,
)


def _chunk() -> np.ndarray:
    time = np.linspace(0.0, 1.0, HORIZON)
    chunk = np.zeros((HORIZON, ACTION_DIM), dtype=np.float64)
    for dim in range(6):
        chunk[:, dim] = 0.05 * np.sin((dim + 2) * np.pi * time) + 0.01 * ((-1.0) ** np.arange(HORIZON))
    chunk[:, 6] = 1.0
    chunk[25:, 6] = -1.0
    return chunk


def test_hest_preserves_first_endpoint_and_gripper() -> None:
    base = _chunk()
    output, fallback = hest_transform(base, alpha=1.0)
    assert fallback is None
    np.testing.assert_allclose(output[0, :6], base[0, :6], atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(output[:, :6].sum(axis=0), base[:, :6].sum(axis=0), atol=1e-12, rtol=0.0)
    assert np.array_equal(output[:, 6], base[:, 6])
    assert cumulative_arm_energy(output) < cumulative_arm_energy(base)


def test_variants_are_distinct_and_gripper_ablation_is_live() -> None:
    base = _chunk()
    hest, _ = hest_transform(base)
    prior = spline_proxy(base)
    no_endpoint = no_endpoint_ablation(base)
    moving = moving_average_control(base)
    assert not np.allclose(hest, prior)
    assert not np.allclose(hest, no_endpoint)
    assert not np.allclose(hest, moving)
    assert not np.array_equal(prior[:, 6], base[:, 6])
    assert np.array_equal(no_endpoint[:, 6], base[:, 6])
    assert np.array_equal(moving[:, 6], base[:, 6])


def test_support_violation_falls_back_to_entire_base_chunk() -> None:
    base = _chunk()
    lower = base.min(axis=0)
    upper = base.max(axis=0)
    lower[:6] = base[:, :6].mean(axis=0)
    upper[:6] = lower[:6]
    output, fallback = hest_transform(base, lower=lower, upper=upper, tolerance=0.0)
    assert fallback == "support_violation"
    assert np.array_equal(output, base)


def test_support_bounds_and_validity() -> None:
    base = _chunk()
    lower, upper = support_bounds([base, base * np.array([1, 1, 1, 1, 1, 1, 1])])
    assert support_valid(base, lower, upper)
    invalid = base.copy()
    invalid[0, 0] = upper[0] + 10.0
    assert not support_valid(invalid, lower, upper)


def test_gripper_transition_and_shape_guard() -> None:
    assert gripper_transition(_chunk())
    with pytest.raises(ValueError, match="expected action chunk"):
        hest_transform(np.zeros((49, 7)))
    with pytest.raises(ValueError, match="increments must have shape"):
        smooth_cumulative_path(np.zeros((49, 6)))


def test_sha256_registry_parser_accepts_filename_record() -> None:
    digest = "E56B4717BDF949E1A4371457058DFC662E0D79C70D9E2FBEF35A5415FD0F0527"
    assert parse_sha256_registry(f"SHA256 researcher_proposal.md\n{digest}\n") == digest
    assert parse_sha256_registry(f"SHA256  {digest}  researcher_proposal.md\n") == digest
    assert parse_sha256_registry("SHA256 researcher_proposal.md\nnot-a-digest\n") == ""


def test_manifest_validation_detects_duplicates_and_overlap() -> None:
    manifest = [
        {"window_key": "a", "partition": "discovery"},
        {"window_key": "b", "partition": "validation"},
    ]
    audit = validate_manifest(manifest, [{"window_key": "a"}, {"window_key": "b"}])
    assert audit["key_sets_equal"]
    assert audit["partition_overlap_count"] == 0
    duplicate = validate_manifest(manifest, [{"window_key": "a"}, {"window_key": "a"}])
    assert duplicate["duplicate_partial_key_count"] == 1
    assert duplicate["missing_manifest_key_count"] == 1


def _decision_inputs(**overrides: object) -> Stage0ADecisionInputs:
    values: dict[str, object] = {
        "proposal_hash_ok": True,
        "manifest_audit_ok": True,
        "source_finite_shape_ok": True,
        "arm_support_noncollapsed": True,
        "validation_transition_count": 8,
        "endpoint_max_error": 1e-12,
        "first_action_max_error": 1e-12,
        "gripper_max_error": 0.0,
        "all_variant_support_valid": True,
        "acting_fraction": 1.0,
        "median_energy_reduction": 0.5,
        "comparator_distinct": True,
        "roundtrip_max_error": 0.0,
        "exception_count": 0,
    }
    values.update(overrides)
    return Stage0ADecisionInputs(**values)  # type: ignore[arg-type]


def test_stage0a_decision_taxonomy() -> None:
    assert classify_stage0a(_decision_inputs()) == "HEST_STAGE_0A_PASS_STAGE_0B_ALLOWED"
    assert classify_stage0a(_decision_inputs(validation_transition_count=7)) == "HEST_STAGE_0A_DATA_FAILURE"
    assert classify_stage0a(_decision_inputs(endpoint_max_error=1e-4)) == "HEST_STAGE_0A_IMPLEMENTATION_FAILURE"
    assert classify_stage0a(_decision_inputs(acting_fraction=0.2)) == "HEST_STAGE_0A_NO_HEADROOM"
    assert (
        classify_stage0a(_decision_inputs(comparator_distinct=False))
        == "HEST_STAGE_0A_DESIGN_FAILURE_EQUIVALENT_CONTROL"
    )
