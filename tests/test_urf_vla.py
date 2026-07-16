import hashlib
import json
from pathlib import Path

import numpy as np

from scripts.run_urf_vla_stage0 import (
    POLICY_PROBE,
    _load_resume,
    _partial_payload,
    _serializer_preflight,
)
from tca_map.smolvla.urf_vla import (
    ACTION_DIM,
    CHUNK_SIZE,
    DEFAULT_G_MAX,
    PROPOSAL_HASH,
    ROUTE_POSITIVE_MAX,
    ROUTE_POSITIVE_MIN,
    Stage0DecisionInputs,
    action_chunk,
    action_delta_summary,
    apply_urf_residual,
    canonical_json_sha256,
    classify_stage0,
    fit_residual_scale,
    heteroscedastic_huber_nll,
    json_default,
    normalized_residual,
    route_label_health,
    route_labels,
    route_logits,
    route_thresholds,
    uncertainty_monotonicity,
    urf_row_key,
    validate_manifest,
)


def test_urf_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "URF-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "chunk_size": np.int64(CHUNK_SIZE),
        "action_dimension": np.int64(ACTION_DIM),
        "base_chunk": np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32),
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
    assert persisted["fixture"]["action_dimension"] == ACTION_DIM


def _manifest_row(
    partition: str,
    demo: int,
    frame: int,
    model_or_probe: str = "urf_full",
) -> dict[str, object]:
    row: dict[str, object] = {
        "partition": partition,
        "suite": "libero_spatial",
        "task_identity": "libero_spatial/task_3",
        "source_edge_sha256": "ABC",
        "demo_id": demo,
        "frame_index": frame,
        "model_or_probe": model_or_probe,
        "proxy_variant": model_or_probe,
        "policy_probe": POLICY_PROBE,
    }
    if model_or_probe.startswith("urf"):
        row["g_max"] = DEFAULT_G_MAX
        row["lambda_clean"] = 0.2
        row["tau_g_family"] = "lcb_alpha_m1_alpha_u1"
    row["row_key"] = urf_row_key(row)
    return row


def test_manifest_validation_uses_variant_and_detects_source_overlap() -> None:
    manifest = [
        _manifest_row("discovery", 0, 1, "urf_full"),
        _manifest_row("discovery", 0, 1, "urf_no_uncertainty_route_ablation"),
        _manifest_row("validation", 8, 1, "urf_full"),
    ]
    completed = [{"row_key": row["row_key"]} for row in manifest]
    healthy = validate_manifest(manifest, completed)
    assert healthy["key_sets_equal"] is True
    assert healthy["split_overlap_key_count"] == 0
    duplicate = validate_manifest(manifest, completed + [completed[0]])
    assert duplicate["duplicate_partial_key_count"] == 1
    extra = validate_manifest(manifest, completed + [{"row_key": "off-manifest"}])
    assert extra["extra_partial_key_count"] == 1

    overlapped = [_manifest_row("discovery", 0, 1), _manifest_row("validation", 0, 1)]
    overlap_summary = validate_manifest(overlapped, [{"row_key": row["row_key"]} for row in overlapped])
    assert overlap_summary["split_overlap_key_count"] == 1


def test_action_chunk_residual_scale_and_route_labels_are_shape_stable() -> None:
    rng = np.random.default_rng(20263001)
    actions = rng.normal(scale=0.01, size=(80, ACTION_DIM))
    chunk = action_chunk(actions, 5)
    assert chunk.shape == (CHUNK_SIZE, ACTION_DIM)

    base = rng.normal(scale=0.01, size=(12, CHUNK_SIZE, ACTION_DIM))
    expert = base + rng.normal(scale=0.02, size=base.shape)
    scale = fit_residual_scale(base, expert)
    residual = normalized_residual(base, expert, scale["scale"])
    thresholds = route_thresholds(residual)
    labels = route_labels(residual, thresholds)
    health = route_label_health(labels)
    assert residual.shape == base.shape
    assert thresholds.shape == (ACTION_DIM,)
    assert labels.shape == base.shape
    assert ROUTE_POSITIVE_MIN <= health["route_label_positive_fraction"] <= ROUTE_POSITIVE_MAX


def test_urf_zero_eta_is_identity_and_uncertainty_changes_gate() -> None:
    rng = np.random.default_rng(20263002)
    base = rng.normal(scale=0.01, size=(3, CHUNK_SIZE, ACTION_DIM))
    residual = rng.normal(scale=0.2, size=base.shape)
    scale = np.full(ACTION_DIM, 0.1)
    low_uncertainty = np.full(base.shape, -4.0)
    high_uncertainty = np.full(base.shape, 2.0)

    identity = apply_urf_residual(base, residual, low_uncertainty, scale, eta=0.0)
    low = apply_urf_residual(base, residual, low_uncertainty, scale, eta=1.0)
    high = apply_urf_residual(base, residual, high_uncertainty, scale, eta=1.0)
    assert np.max(np.abs(identity - base)) == 0.0
    assert np.mean(np.abs(low - base)) > np.mean(np.abs(high - base))
    assert np.mean(np.abs(route_logits(residual, low_uncertainty) - route_logits(residual, high_uncertainty))) > 0.0
    assert action_delta_summary(base, low)["delta_finite"] is True


def test_heteroscedastic_objective_and_uncertainty_monotonicity() -> None:
    rng = np.random.default_rng(20263003)
    target = rng.normal(size=(10, CHUNK_SIZE, ACTION_DIM))
    good_mean = target + rng.normal(scale=0.01, size=target.shape)
    bad_mean = np.zeros_like(target)
    log_var = np.full_like(target, -2.0)
    assert heteroscedastic_huber_nll(target, good_mean, log_var) < heteroscedastic_huber_nll(target, bad_mean, log_var)

    predicted_std = np.linspace(0.1, 2.0, target.size)
    residual_error = predicted_std + np.linspace(0.0, 0.1, target.size)
    mono = uncertainty_monotonicity(predicted_std, residual_error)
    assert mono["uncertainty_strata_noncollapsed"] is True
    assert mono["uncertainty_monotonicity_passed"] is True


def test_resume_validates_hashes_and_manifest_membership(tmp_path: Path) -> None:
    manifest_row = _manifest_row("discovery", 0, 1)
    feature_cache = tmp_path / "feature.npz"
    base_cache = tmp_path / "base.npz"
    np.savez_compressed(feature_cache, feature=np.zeros(64, dtype=np.float16))
    base_chunk = np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
    np.savez_compressed(base_cache, base_chunk=base_chunk)
    completed = {
        "row_key": manifest_row["row_key"],
        "feature_cache_path": str(feature_cache),
        "feature_cache_sha256": hashlib.sha256(feature_cache.read_bytes()).hexdigest().upper(),
        "base_chunk_cache_path": str(base_cache),
        "base_chunk_cache_sha256": hashlib.sha256(base_cache.read_bytes()).hexdigest().upper(),
        "base_chunk_sha256": hashlib.sha256(base_chunk.astype(np.float32).tobytes()).hexdigest().upper(),
    }
    partial = _partial_payload("MANIFEST", 1, [completed], exception_count=1, last_exception="transient")
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(partial), encoding="utf-8")
    rows, exception_count, last_exception = _load_resume(path, [manifest_row], "MANIFEST")
    assert rows == [completed]
    assert exception_count == 1
    assert last_exception == "transient"


def _healthy_inputs(**overrides: object) -> Stage0DecisionInputs:
    values: dict[str, object] = {
        "proposal_hash_ok": True,
        "serializer_preflight_ok": True,
        "official_prior_asset_check_persisted": True,
        "manifest_integrity_ok": True,
        "source_alignment_ok": True,
        "feature_action_proprio_finite_aligned": True,
        "split_integrity_ok": True,
        "minimum_discovery_windows": 512,
        "minimum_validation_windows": 128,
        "all_tasks_reported": True,
        "maximum_validation_task_fraction": 0.25,
        "residual_scales_noncollapsed": True,
        "residual_targets_noncollapsed": True,
        "route_labels_noncollapsed": True,
        "route_positive_fraction": 0.20,
        "uncertainty_strata_noncollapsed": True,
        "task_phase_action_group_coverage_ok": True,
        "base_residual_headroom_ok": True,
        "hetero_beats_homoscedastic_relative": 0.05,
        "hetero_beats_homoscedastic_absolute_huber": 0.005,
        "hetero_beats_task_phase_relative": 0.05,
        "hetero_beats_task_phase_absolute_huber": 0.005,
        "uncertainty_enters_route_gate": True,
        "uncertainty_monotonicity_spearman": 0.20,
        "uncertainty_binned_monotonic": False,
        "sureflow_proxy_headroom_relative": 0.05,
        "sureflow_proxy_headroom_absolute_huber": 0.005,
        "no_uncertainty_ablation_distinct": True,
        "urf_beats_ablation_relative": 0.05,
        "urf_beats_ablation_absolute_huber": 0.005,
        "route_activation_fraction": 0.20,
        "route_all_zero": False,
        "route_all_one": False,
        "route_globally_active": False,
        "action_validity_ok": True,
        "identity_max_abs_error": 0.0,
        "checkpoint_reload_ok": True,
        "finite_objectives_and_gradients": True,
        "urf_gradient_nonzero": True,
        "frozen_parameter_gradient_count": 0,
        "weighted_gradient_norm_ratio_max": 1.0,
        "action_deltas_bounded": True,
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


def test_stage0_decision_taxonomy() -> None:
    assert classify_stage0(_healthy_inputs()) == "URF_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert (
        classify_stage0(_healthy_inputs(route_labels_noncollapsed=False))
        == "URF_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    )
    assert classify_stage0(_healthy_inputs(base_residual_headroom_ok=False)) == "URF_STAGE_0_NO_USABLE_HEADROOM"
    assert (
        classify_stage0(_healthy_inputs(uncertainty_enters_route_gate=False))
        == "URF_STAGE_0_DESIGN_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(uncertainty_monotonicity_spearman=0.0, uncertainty_binned_monotonic=False))
        == "URF_STAGE_0_DESIGN_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(sureflow_proxy_headroom_relative=0.0, sureflow_proxy_headroom_absolute_huber=0.0))
        == "URF_STAGE_0_NO_USABLE_HEADROOM"
    )
    assert classify_stage0(_healthy_inputs(route_all_one=True)) == "URF_STAGE_0_DESIGN_FAILURE"
    assert (
        classify_stage0(_healthy_inputs(urf_gradient_nonzero=False))
        == "URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(confirmatory_records_read=1))
        == "URF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
