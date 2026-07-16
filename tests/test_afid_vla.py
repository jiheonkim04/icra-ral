import json
from pathlib import Path

import numpy as np

from scripts.run_afid_vla_stage0 import POLICY_PROBE, _serializer_preflight
from tca_map.smolvla.afid_vla import (
    ACTION_DIM,
    HORIZON,
    PROPOSAL_HASH,
    Stage0DecisionInputs,
    action_delta_summary,
    afid_row_key,
    apply_afid_gate,
    apply_finevla_proxy,
    binary_prediction_metrics,
    canonical_json_sha256,
    classify_stage0,
    clean_retention_summary,
    extract_action_factors,
    factor_keys,
    factor_label_health,
    factor_mask,
    fit_linear_factor_predictor,
    fit_residual_scale,
    gradient_smoke,
    group_clip,
    json_default,
    mask_health,
    predict_factor_confidence,
    validate_manifest,
)


def test_afid_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "AFID-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "horizon": np.int64(HORIZON),
        "action_dimension": np.int64(ACTION_DIM),
        "decision_inputs": _healthy_inputs(),
    }
    digest = canonical_json_sha256(fixture)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture, sort_keys=True, default=json_default), encoding="utf-8")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert canonical_json_sha256(parsed) == digest


def test_runner_serializer_preflight_writes_parses_and_reproduces_hash(tmp_path: Path) -> None:
    path = tmp_path / "stage_0_serializer_preflight.json"
    result = _serializer_preflight(path)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert persisted["parsed"] is True
    assert persisted["fixture_hash"] == persisted["reproduced_hash"]
    assert persisted["fixture"]["manifest_row"]["policy_probe"] == POLICY_PROBE
    assert persisted["fixture"]["decision"] == "AFID_STAGE_0_PASS_TO_BOUNDED_VALIDATION"


def _manifest_row(split: str, demo: int, start: int, policy: str = "afid_full") -> dict[str, object]:
    row: dict[str, object] = {
        "split": split,
        "task_suite": "libero_spatial",
        "task_id": "libero_spatial/task_3",
        "demo_id": demo,
        "window_start": start,
        "factor_key": "axis:1|dir:1|grip:0|rot:0|term:0",
        "policy": policy,
        "policy_probe": POLICY_PROBE,
    }
    row["row_key"] = afid_row_key(row)
    return row


def test_manifest_validation_detects_duplicates_extra_and_split_overlap() -> None:
    manifest = [
        _manifest_row("discovery", 0, 10, "smolvla_base"),
        _manifest_row("discovery", 0, 10, "afid_full"),
        _manifest_row("validation", 8, 10, "afid_full"),
    ]
    completed = [{"row_key": row["row_key"]} for row in manifest]
    healthy = validate_manifest(manifest, completed)
    assert healthy["key_sets_equal"] is True
    assert healthy["duplicate_partial_key_count"] == 0
    duplicate = validate_manifest(manifest, completed + [completed[0]])
    assert duplicate["duplicate_partial_key_count"] == 1
    extra = validate_manifest(manifest, completed + [{"row_key": "off-manifest"}])
    assert extra["extra_partial_key_count"] == 1
    overlapped = [_manifest_row("discovery", 0, 10), _manifest_row("validation", 0, 10)]
    overlap_summary = validate_manifest(overlapped, [{"row_key": row["row_key"]} for row in overlapped])
    assert overlap_summary["split_overlap_key_count"] == 1


def test_factor_extraction_mask_gate_identity_and_proxy() -> None:
    rng = np.random.default_rng(20263301)
    base = rng.normal(scale=0.01, size=(8, HORIZON, ACTION_DIM))
    expert = base.copy()
    expert[:4, :, 0] += 0.04
    expert[4:, :, 1] -= 0.04
    expert[:, :, 6] += np.where(np.arange(8)[:, None] % 2 == 0, 0.24, -0.24)
    residual = expert - base
    scale = fit_residual_scale(residual)
    mask = factor_mask(residual, scale)
    labels = extract_action_factors(base, expert)
    health = factor_label_health(labels)
    keys = factor_keys(labels)
    predictor = fit_linear_factor_predictor(base, mask)
    confidence = predict_factor_confidence(predictor, base)
    changed, gate = apply_afid_gate(base, residual, mask, np.ones_like(confidence))
    identity, _ = apply_afid_gate(base, residual, mask, np.zeros_like(confidence))
    inactive, _ = apply_afid_gate(base, residual, np.zeros_like(mask), np.ones_like(confidence))
    finevla = apply_finevla_proxy(base, residual, np.ones_like(confidence))
    assert len(keys) == len(base)
    assert health["usable_factor_count"] >= 1
    assert mask_health(mask)["factor_mask_noncollapsed"] is True
    assert np.max(np.abs(identity - base)) == 0.0
    assert np.max(np.abs(inactive - base)) == 0.0
    assert np.mean(np.abs(changed - base)) > 0.0
    assert finevla.shape == base.shape
    assert group_clip(residual).shape == base.shape
    assert action_delta_summary(base, changed)["action_deltas_bounded"] is True
    assert clean_retention_summary(base, identity, inactive)["clean_retention_ok"] is True
    assert gradient_smoke(base, changed - base, gate, expert)["finite_objectives_and_gradients"] is True


def test_factor_predictor_metrics_are_above_collapsed_baseline() -> None:
    target = np.asarray([0, 0, 1, 1, 1, 1], dtype=bool)
    predicted = np.asarray([0, 1, 1, 1, 1, 1], dtype=bool)
    metrics = binary_prediction_metrics(predicted, target)
    assert metrics["accuracy"] > metrics["majority_accuracy"]
    assert metrics["macro_f1"] > metrics["majority_macro_f1"]


def test_stage0_decision_taxonomy() -> None:
    assert classify_stage0(_healthy_inputs()) == "AFID_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert (
        classify_stage0(_healthy_inputs(factor_labels_noncollapsed=False))
        == "AFID_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(factor_conditioned_oracle_reduction=0.0))
        == "AFID_STAGE_0_NO_USABLE_HEADROOM"
    )
    assert (
        classify_stage0(_healthy_inputs(factor_predictor_beats_majority=False))
        == "AFID_STAGE_0_DESIGN_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(identity_max_abs_error=1e-4))
        == "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(confirmatory_records_read=1))
        == "AFID_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    )


def _healthy_inputs(**overrides: object) -> Stage0DecisionInputs:
    values: dict[str, object] = {
        "proposal_hash_ok": True,
        "serializer_preflight_ok": True,
        "official_prior_asset_check_persisted": True,
        "manifest_integrity_ok": True,
        "source_alignment_ok": True,
        "action_semantics_ok": True,
        "base_chunks_valid": True,
        "factor_labels_noncollapsed": True,
        "usable_factor_count": 2,
        "factor_mask_global_positive_fraction": 0.25,
        "validation_task_mask_fraction_min": 0.25,
        "validation_task_mask_fraction_max": 0.25,
        "factor_predictor_beats_majority": True,
        "factor_predictor_beats_task_phase": True,
        "factor_conditioned_oracle_reduction": 0.05,
        "finevla_proxy_residual_headroom": 0.03,
        "afid_differs_from_base": True,
        "afid_differs_from_finevla_proxy": True,
        "afid_differs_from_no_factor": True,
        "afid_differs_from_standard_lora": True,
        "identity_max_abs_error": 0.0,
        "inactive_gate_max_abs_error": 0.0,
        "finite_objectives_and_gradients": True,
        "expected_parameter_gradient_nonzero": True,
        "frozen_base_gradient_count": 0,
        "weighted_gradient_norm_ratio_max": 1.0,
        "gate_activation_fraction": 0.25,
        "action_deltas_bounded": True,
        "action_validity_ok": True,
        "clean_retention_ok": True,
        "reward_read_count": 0,
        "success_read_count": 0,
        "done_read_count": 0,
        "confirmatory_records_read": 0,
        "closed_loop_experiment_happened": False,
        "simulator_load_count": 0,
        "training_happened": False,
        "validation_search_happened": False,
        "exception_count": 0,
    }
    values.update(overrides)
    return Stage0DecisionInputs(**values)
