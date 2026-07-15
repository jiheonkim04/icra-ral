from __future__ import annotations

import numpy as np
import pytest

from tca_map.smolvla.haste_vla import (
    Stage0ADecisionInputs,
    classify_stage0a,
    construct_event_label,
    displacement_statistics,
    event_stratum,
    fit_constant_hazard,
    hazard_nll_from_probabilities,
    normalize_displacement,
    offset_quintile,
    validate_manifest,
)


def _actions(length: int = 80) -> np.ndarray:
    actions = np.zeros((length, 7), dtype=np.float64)
    actions[:, :6] = np.arange(length, dtype=np.float64)[:, None] * 0.001
    actions[:, 6] = 1.0
    actions[12:, 6] = -1.0
    actions[47:, 6] = 1.0
    return actions


def test_event_label_uses_first_transition_and_masks_after_event() -> None:
    actions = _actions()
    label = construct_event_label(actions, frame_index=5, event_horizon=20)
    assert label["transition_offset"] == 7
    assert label["likelihood_term_count"] == 7
    assert label["survival_mask"].sum() == 7
    assert label["event_target"].sum() == 1
    assert label["event_target"][6] == 1
    np.testing.assert_allclose(label["relative_displacement"], actions[5:13, :6].sum(axis=0))


def test_boundary_censoring_does_not_claim_unobserved_intervals() -> None:
    actions = _actions(15)
    label = construct_event_label(actions, frame_index=13, event_horizon=50)
    assert label["censored"]
    assert label["valid_interval_count"] == 1
    assert label["survival_mask"].sum() == 1
    assert label["event_target"].sum() == 0


def test_invalid_frame_and_horizon_are_rejected() -> None:
    with pytest.raises(ValueError, match="at least one observable"):
        construct_event_label(_actions(), frame_index=79, event_horizon=20)
    with pytest.raises(ValueError, match="event horizon"):
        construct_event_label(_actions(), frame_index=0, event_horizon=10)


def test_displacement_normalization_uses_floor() -> None:
    values = [np.ones(6), np.ones(6)]
    mean, std = displacement_statistics(values)
    assert np.all(std == 1e-6)
    np.testing.assert_allclose(normalize_displacement(np.ones(6), mean, std), 0.0)


def test_constant_hazard_and_nll_are_finite() -> None:
    actions = _actions()
    rows = []
    for frame in (0, 5, 20, 60):
        label = construct_event_label(actions, frame, 20)
        rows.append({"event_horizon": 20, **label})
    hazard = fit_constant_hazard(rows, 20)
    assert hazard.shape == (20,)
    assert np.all((hazard > 0.0) & (hazard < 1.0))
    assert np.isfinite(hazard_nll_from_probabilities(hazard, rows))


def test_strata_and_quintiles() -> None:
    assert event_stratum({"transition_offset": None}) == "censored"
    assert event_stratum({"transition_offset": 10}) == "event_near"
    assert event_stratum({"transition_offset": 11}) == "event_far"
    assert {offset_quintile(offset, 50) for offset in (1, 11, 21, 31, 41)} == {0, 1, 2, 3, 4}


def test_manifest_validation() -> None:
    manifest = [
        {"event_row_key": "a", "partition": "discovery"},
        {"event_row_key": "b", "partition": "validation"},
    ]
    audit = validate_manifest(manifest, [{"event_row_key": "a"}, {"event_row_key": "b"}])
    assert audit["key_sets_equal"]
    assert audit["partition_overlap_count"] == 0
    duplicate = validate_manifest(manifest, [{"event_row_key": "a"}, {"event_row_key": "a"}])
    assert duplicate["duplicate_partial_key_count"] == 1
    assert duplicate["missing_manifest_key_count"] == 1


def _decision_inputs(**overrides: object) -> Stage0ADecisionInputs:
    values: dict[str, object] = {
        "proposal_hash_ok": True,
        "manifest_integrity_ok": True,
        "finite_source_and_features": True,
        "discovery_uncensored_count": 256,
        "discovery_censored_count": 256,
        "minimum_validation_uncensored_per_task": 32,
        "occupied_offset_quintile_count": 5,
        "displacement_variance_all_positive": True,
        "maximum_uncensored_task_fraction": 0.25,
        "base_event_near_headroom": True,
        "hazard_probe_improvement": 0.10,
        "displacement_probe_improvement": 0.10,
        "identity_max_error": 0.0,
        "base_hash_unchanged": True,
        "checkpoint_reload_ok": True,
        "exception_count": 0,
    }
    values.update(overrides)
    return Stage0ADecisionInputs(**values)  # type: ignore[arg-type]


def test_stage0a_decision_taxonomy() -> None:
    assert classify_stage0a(_decision_inputs()) == "HASTE_STAGE_0A_PASS_STAGE_0B_ALLOWED"
    assert classify_stage0a(_decision_inputs(discovery_uncensored_count=127)) == "HASTE_STAGE_0A_DATA_FAILURE"
    assert classify_stage0a(_decision_inputs(identity_max_error=1e-3)) == "HASTE_STAGE_0A_IMPLEMENTATION_FAILURE"
    assert classify_stage0a(_decision_inputs(base_event_near_headroom=False)) == "HASTE_STAGE_0A_NO_HEADROOM"
    assert classify_stage0a(_decision_inputs(hazard_probe_improvement=0.0)) == "HASTE_STAGE_0A_DESIGN_FAILURE"
