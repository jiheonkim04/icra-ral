import json
from pathlib import Path

import numpy as np

from scripts.run_tsc_vla_stage0 import (
    POLICY_PROBE,
    _load_resume,
    _partial_payload,
    _serializer_preflight,
)
from tca_map.smolvla.tsc_vla import (
    ACTION_DIM,
    CHUNK_SIZE,
    DIAGNOSTIC_ALPHA,
    PROPOSAL_HASH,
    VISUAL_FEATURE_DIM,
    Stage0DecisionInputs,
    action_chunk,
    apply_masked_completion,
    binary_cross_entropy,
    canonical_json_sha256,
    classify_stage0,
    fit_completion_model,
    fit_mask_label_stats,
    fit_structured_mask_probe,
    hard_mask,
    json_default,
    make_error_mask_labels,
    predict_completion_residual,
    predict_structured_mask_scores,
    raw_tsc_feature,
    trivial_mask_probability,
    tsc_row_key,
    unselected_clamp_error,
    validate_manifest,
)


def test_tsc_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "TSC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "diagnostic_alpha": np.float32(DIAGNOSTIC_ALPHA),
        "base_chunk": np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32),
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


def _manifest_row(partition: str, demo: int, frame: int, variant: str = "tsc_full") -> dict[str, object]:
    row: dict[str, object] = {
        "partition": partition,
        "suite": "libero_spatial",
        "task_identity": "libero_spatial/task_3",
        "source_edge_sha256": "ABC",
        "demo_id": demo,
        "frame_index": frame,
        "proxy_variant": variant,
        "policy_probe": POLICY_PROBE,
    }
    row["row_key"] = tsc_row_key(row)
    return row


def test_manifest_validation_uses_tsc_identity() -> None:
    manifest = [_manifest_row("discovery", 0, 1), _manifest_row("validation", 8, 1)]
    completed = [{"row_key": row["row_key"]} for row in manifest]
    healthy = validate_manifest(manifest, completed)
    assert healthy["key_sets_equal"] is True
    duplicate = validate_manifest(manifest, completed + [completed[0]])
    assert duplicate["duplicate_partial_key_count"] == 1
    extra = validate_manifest(manifest, completed + [{"row_key": "off-manifest"}])
    assert extra["extra_partial_key_count"] == 1


def test_action_chunk_feature_and_mask_label_shapes() -> None:
    actions = np.arange(80 * ACTION_DIM, dtype=np.float64).reshape(80, ACTION_DIM) / 100.0
    assert action_chunk(actions, 4).shape == (CHUNK_SIZE, ACTION_DIM)
    feature = raw_tsc_feature(np.zeros(VISUAL_FEATURE_DIM), np.zeros(8), 2, 0.5)
    assert feature.shape == (VISUAL_FEATURE_DIM + 8 + 1 + 4,)
    rng = np.random.default_rng(20262800)
    residual = rng.normal(scale=0.01, size=(4, CHUNK_SIZE, ACTION_DIM))
    residual[:, :, 0] = np.linspace(-1.0, 1.0, CHUNK_SIZE).reshape(1, CHUNK_SIZE)
    residual[:, 10:20, 3] = 2.0
    stats = fit_mask_label_stats(residual)
    labels = make_error_mask_labels(residual, stats)
    assert labels.shape == residual.shape
    assert 0 < int(labels.sum()) < labels.size


def test_structured_mask_probe_beats_trivial_on_synthetic_signal() -> None:
    rng = np.random.default_rng(20262801)
    features = rng.normal(size=(120, 4))
    labels = np.zeros((120, CHUNK_SIZE, ACTION_DIM), dtype=bool)
    labels[:, :, 2] = features[:, 0].reshape(-1, 1) > 0.0
    model = fit_structured_mask_probe(features[:90], labels[:90])
    probability = predict_structured_mask_scores(model, features[90:])
    structured_bce = binary_cross_entropy(probability, labels[90:])
    trivial = np.broadcast_to(trivial_mask_probability(labels[:90]), labels[90:].shape)
    assert structured_bce < binary_cross_entropy(trivial, labels[90:])


def test_masked_completion_clamps_unselected_cells_and_predicts_shape() -> None:
    rng = np.random.default_rng(20262802)
    features = rng.normal(size=(80, 5))
    base = rng.normal(size=(80, CHUNK_SIZE, ACTION_DIM)) * 0.01
    residual = np.zeros_like(base)
    residual[:, :, 1] = features[:, 0].reshape(-1, 1) * 0.05
    expert = base + residual
    model = fit_completion_model(features[:60], base[:60], expert[:60])
    prediction = predict_completion_residual(model, features[60:], base[60:])
    mask = np.zeros_like(prediction, dtype=bool)
    mask[:, :, 1] = True
    completed = apply_masked_completion(base[60:], prediction, mask)
    assert completed.shape == base[60:].shape
    assert unselected_clamp_error(base[60:], completed, mask) == 0.0
    assert np.mean(np.abs(completed[:, :, 1] - base[60:, :, 1])) > 0.0
    assert hard_mask(mask.astype(float)).dtype == bool


def test_resume_validates_feature_and_base_chunk_hashes(tmp_path: Path) -> None:
    manifest_row = _manifest_row("discovery", 0, 1)
    feature_cache = tmp_path / "feature.npz"
    base_cache = tmp_path / "base.npz"
    np.savez_compressed(feature_cache, feature=np.zeros(VISUAL_FEATURE_DIM, dtype=np.float16))
    base_chunk = np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
    np.savez_compressed(base_cache, base_chunk=base_chunk)
    completed = {
        "row_key": manifest_row["row_key"],
        "feature_cache_path": str(feature_cache),
        "feature_cache_sha256": __import__("hashlib").sha256(feature_cache.read_bytes()).hexdigest().upper(),
        "base_chunk_cache_path": str(base_cache),
        "base_chunk_cache_sha256": __import__("hashlib").sha256(base_cache.read_bytes()).hexdigest().upper(),
        "base_chunk_sha256": __import__("hashlib").sha256(base_chunk.astype(np.float32).tobytes()).hexdigest().upper(),
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
        "minimum_discovery_windows": 512,
        "minimum_validation_windows": 128,
        "all_tasks_reported": True,
        "maximum_validation_task_fraction": 0.25,
        "labels_noncollapsed_discovery": True,
        "labels_noncollapsed_validation": True,
        "structured_mask_beats_trivial": True,
        "structured_mask_beats_magnitude": True,
        "tsc_beats_prior_relative": 0.05,
        "tsc_beats_prior_absolute_huber": 0.0,
        "tsc_beats_ablation_relative": 0.05,
        "tsc_beats_ablation_absolute_huber": 0.0,
        "unselected_cell_clamp_max_error": 0.0,
        "changed_cell_fraction": 0.10,
        "deltas_finite_and_bounded": True,
        "tsc_distinct_from_prior_and_ablation": True,
        "finite_objectives_and_gradients": True,
        "tsc_gradient_nonzero": True,
        "gradient_ratio_at_most_100": True,
        "frozen_parameter_gradient_count": 0,
        "identity_max_error": 0.0,
        "base_hash_unchanged": True,
        "checkpoint_reload_ok": True,
        "action_validity_ok": True,
        "reward_read_count": 0,
        "success_read_count": 0,
        "done_read_count": 0,
        "confirmatory_records_read": 0,
        "exception_count": 0,
    }
    values.update(overrides)
    return Stage0DecisionInputs(**values)


def test_stage0_decision_taxonomy() -> None:
    assert classify_stage0(_healthy_inputs()) == "TSC_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert (
        classify_stage0(_healthy_inputs(labels_noncollapsed_validation=False))
        == "TSC_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    )
    assert classify_stage0(_healthy_inputs(structured_mask_beats_magnitude=False)) == "TSC_STAGE_0_NO_USABLE_HEADROOM"
    assert classify_stage0(_healthy_inputs(changed_cell_fraction=0.0)) == "TSC_STAGE_0_DESIGN_FAILURE"
    assert (
        classify_stage0(_healthy_inputs(serializer_preflight_ok=False))
        == "TSC_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(confirmatory_records_read=1))
        == "TSC_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
