import json
from pathlib import Path

import numpy as np

from scripts.run_cfr_vla_stage0 import (
    POLICY_PROBE,
    _array_sha256,
    _load_resume,
    _partial_payload,
    _serializer_preflight,
)
from tca_map.smolvla.cfr_vla import (
    ACTION_DIM,
    CHUNK_SIZE,
    DFM_BINS,
    PROPOSAL_HASH,
    REFINEMENT_STEPS,
    VISUAL_FEATURE_DIM,
    Stage0DecisionInputs,
    action_chunk,
    apply_discovery_zscore,
    canonical_json_sha256,
    cfr_row_key,
    classify_stage0,
    fit_dfm_proxy,
    fit_discovery_zscore,
    fit_iterative_refinement,
    flattened_chunks,
    json_default,
    phase_bin,
    predict_dfm_proxy,
    predict_iterative_refinement,
    raw_refinement_feature,
    validate_manifest,
)


def test_cfr_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "CFR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "refinement_steps": np.int64(REFINEMENT_STEPS),
        "dfm_bins": np.int64(DFM_BINS),
        "base_chunk": np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32),
    }
    digest = canonical_json_sha256(fixture)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture, sort_keys=True, default=json_default), encoding="utf-8")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert canonical_json_sha256(parsed) == digest
    assert parsed["refinement_steps"] == REFINEMENT_STEPS


def test_runner_serializer_preflight_writes_parses_and_reproduces_hash(tmp_path: Path) -> None:
    path = tmp_path / "stage_0_serializer_preflight.json"
    result = _serializer_preflight(path)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert persisted["parsed"] is True
    assert persisted["fixture_hash"] == persisted["reproduced_hash"]
    assert persisted["fixture"]["manifest_row"]["policy_probe"] == POLICY_PROBE


def _manifest_row(
    partition: str,
    demo: int,
    frame: int,
    steps: int = REFINEMENT_STEPS,
    variant: str = "cfr_full",
) -> dict[str, object]:
    row: dict[str, object] = {
        "partition": partition,
        "suite": "libero_spatial",
        "task_identity": "libero_spatial/task_3",
        "source_edge_sha256": "ABC",
        "demo_id": demo,
        "frame_index": frame,
        "refinement_steps": steps,
        "proxy_variant": variant,
        "policy_probe": POLICY_PROBE,
    }
    row["row_key"] = cfr_row_key(row)
    return row


def test_manifest_validation_includes_refinement_identity() -> None:
    manifest = [_manifest_row("discovery", 0, 1), _manifest_row("validation", 8, 1, variant="cfr_no_iter")]
    completed = [{"row_key": row["row_key"]} for row in manifest]
    healthy = validate_manifest(manifest, completed)
    assert healthy["key_sets_equal"] is True
    duplicate = validate_manifest(manifest, completed + [completed[0]])
    assert duplicate["duplicate_partial_key_count"] == 1
    extra = validate_manifest(manifest, completed + [{"row_key": "off-manifest"}])
    assert extra["extra_partial_key_count"] == 1


def test_action_chunk_phase_and_refinement_feature_shapes() -> None:
    actions = np.arange(80 * ACTION_DIM, dtype=np.float64).reshape(80, ACTION_DIM) / 100.0
    assert action_chunk(actions, 4).shape == (CHUNK_SIZE, ACTION_DIM)
    assert phase_bin(0.0) == 0
    assert phase_bin(0.999) == 9
    assert phase_bin(1.0) == 9
    feature = raw_refinement_feature(np.zeros(VISUAL_FEATURE_DIM), np.zeros(8), 2, 0.5)
    assert feature.shape == (VISUAL_FEATURE_DIM + 8 + 1 + 4,)
    stats = fit_discovery_zscore(np.vstack([feature, feature + 1e-3]))
    transformed = apply_discovery_zscore(stats, np.vstack([feature]))
    assert np.allclose(transformed[:, -4:], feature[-4:].reshape(1, -1))


def test_iterative_cfr_refinement_recovers_structured_residuals() -> None:
    rng = np.random.default_rng(20262700)
    features = rng.normal(size=(120, 5))
    basis = rng.normal(size=(5, CHUNK_SIZE * ACTION_DIM)) * 0.01
    base = rng.normal(size=(120, CHUNK_SIZE, ACTION_DIM)) * 0.02
    expert = base + (features @ basis).reshape(120, CHUNK_SIZE, ACTION_DIM)
    model = fit_iterative_refinement(features[:90], base[:90], expert[:90])
    prediction = predict_iterative_refinement(model, features[90:], base[90:])
    assert prediction.shape == expert[90:].shape
    assert np.mean(np.square(prediction - expert[90:])) < 1e-6


def test_dfm_proxy_is_iterative_full_sequence_and_legal_task_phase() -> None:
    base = np.zeros((6, CHUNK_SIZE, ACTION_DIM), dtype=np.float64)
    expert = np.zeros_like(base)
    expert[:3] += 0.2
    expert[3:] -= 0.1
    tasks = ["a", "a", "a", "b", "b", "b"]
    phases = [0.1, 0.12, 0.14, 0.8, 0.82, 0.84]
    model = fit_dfm_proxy(base, expert, tasks, phases)
    prediction = predict_dfm_proxy(model, base[:2], ["a", "b"], [0.11, 0.83])
    assert prediction.shape == (2, CHUNK_SIZE, ACTION_DIM)
    assert np.allclose(flattened_chunks(prediction)[0].mean(), 0.2)
    assert np.allclose(flattened_chunks(prediction)[1].mean(), -0.1)


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
        "base_chunk_sha256": _array_sha256(base_chunk),
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
        "base_to_expert_residual_variance_all_positive": True,
        "residual_probe_relative_improvement": 0.05,
        "residual_probe_absolute_huber_improvement": 0.0,
        "dfm_proxy_headroom_relative_improvement": 0.05,
        "dfm_proxy_headroom_absolute_huber_improvement": 0.0,
        "iterative_cfr_distinct_from_no_iterative": True,
        "finite_objectives_and_gradients": True,
        "cfr_gradient_nonzero": True,
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
    assert classify_stage0(_healthy_inputs()) == "CFR_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert (
        classify_stage0(_healthy_inputs(base_to_expert_residual_variance_all_positive=False))
        == "CFR_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(residual_probe_relative_improvement=0.04, residual_probe_absolute_huber_improvement=0.004))
        == "CFR_STAGE_0_NO_USABLE_HEADROOM"
    )
    assert classify_stage0(_healthy_inputs(cfr_gradient_nonzero=False)) == "CFR_STAGE_0_DESIGN_FAILURE"
    assert (
        classify_stage0(_healthy_inputs(serializer_preflight_ok=False))
        == "CFR_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(confirmatory_records_read=1))
        == "CFR_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
