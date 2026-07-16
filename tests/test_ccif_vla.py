import json
from pathlib import Path

import numpy as np

from scripts.run_ccif_vla_stage0 import (
    POLICY_PROBE,
    _load_resume,
    _partial_payload,
    _serializer_preflight,
)
from tca_map.smolvla.ccif_vla import (
    ACTION_DIM,
    CHUNK_SIZE,
    INTENT_DIM,
    PROPOSAL_HASH,
    VISUAL_FEATURE_DIM,
    WAYPOINT_INDICES,
    Stage0DecisionInputs,
    action_chunk,
    apply_ccif_residual,
    canonical_json_sha256,
    ccif_row_key,
    classify_stage0,
    coarse_intent,
    endpoint_only_intent,
    fit_intent_normalizer,
    fit_intent_probe,
    fit_task_phase_mean_intent,
    intent_template,
    json_default,
    mean_huber,
    normalize_intent,
    predict_intent_probe,
    predict_task_phase_mean_intent,
    raw_ccif_feature,
    validate_manifest,
)


def test_ccif_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "CCIF-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "intent_dimension": np.int64(INTENT_DIM),
        "waypoint_indices": np.asarray(WAYPOINT_INDICES, dtype=np.int64),
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
    assert persisted["fixture"]["intent_dimension"] == INTENT_DIM


def _manifest_row(partition: str, demo: int, frame: int, model_or_probe: str = "ccif_full") -> dict[str, object]:
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
    row["row_key"] = ccif_row_key(row)
    return row


def test_manifest_validation_uses_policy_identity() -> None:
    manifest = [
        _manifest_row("discovery", 0, 1, "ccif_full"),
        _manifest_row("discovery", 0, 1, "endpoint_only_intent"),
        _manifest_row("validation", 8, 1, "ccif_full"),
    ]
    completed = [{"row_key": row["row_key"]} for row in manifest]
    healthy = validate_manifest(manifest, completed)
    assert healthy["key_sets_equal"] is True
    duplicate = validate_manifest(manifest, completed + [completed[0]])
    assert duplicate["duplicate_partial_key_count"] == 1
    extra = validate_manifest(manifest, completed + [{"row_key": "off-manifest"}])
    assert extra["extra_partial_key_count"] == 1


def test_action_chunk_feature_and_coarse_intent_shape() -> None:
    actions = np.zeros((80, ACTION_DIM), dtype=np.float64)
    actions[:, 0] = 1.0
    actions[:, 3] = 0.5
    actions[:, 6] = np.linspace(-1.0, 1.0, 80)
    chunk = action_chunk(actions, 4)
    intent = coarse_intent(chunk)
    assert intent.shape == (1, INTENT_DIM)
    assert INTENT_DIM == 31
    assert intent.shape[1] != 37
    assert np.isclose(intent[0, 0], 1.0)
    assert np.isclose(intent[0, 7 + 3], 20.0)
    feature = raw_ccif_feature(np.zeros(VISUAL_FEATURE_DIM), np.zeros(8), 2, 0.5, chunk)
    assert feature.shape == (VISUAL_FEATURE_DIM + 8 + 1 + INTENT_DIM + 4,)


def test_intent_normalization_endpoint_and_template_are_shape_stable() -> None:
    rng = np.random.default_rng(20262901)
    chunks = rng.normal(scale=0.02, size=(12, CHUNK_SIZE, ACTION_DIM))
    chunks[:, :, 0] += np.linspace(0.0, 0.05, 12).reshape(-1, 1)
    chunks[:, :, 6] += np.linspace(-0.2, 0.2, 12).reshape(-1, 1)
    raw = coarse_intent(chunks)
    stats = fit_intent_normalizer(raw)
    normalized = normalize_intent(stats, raw)
    endpoint = endpoint_only_intent(normalized, stats)
    template = intent_template(normalized, stats)
    assert normalized.shape == (12, INTENT_DIM)
    assert endpoint.shape == normalized.shape
    assert template.shape == (12, CHUNK_SIZE, ACTION_DIM)
    assert np.isfinite(template).all()


def test_deployment_intent_probe_beats_task_phase_mean_on_synthetic_signal() -> None:
    rng = np.random.default_rng(20262902)
    features = rng.normal(size=(160, 6))
    intents = rng.normal(scale=0.05, size=(160, INTENT_DIM))
    intents[:, 0] = features[:, 0] * 0.8
    intents[:, 10] = features[:, 1] * -0.4
    task = np.arange(160) % 4
    phase = np.linspace(0.0, 1.0, 160)
    model = fit_intent_probe(features[:120], intents[:120])
    prediction = predict_intent_probe(model, features[120:])
    mean_model = fit_task_phase_mean_intent(task[:120], phase[:120], intents[:120])
    task_phase = predict_task_phase_mean_intent(mean_model, task[120:], phase[120:])
    assert mean_huber(prediction, intents[120:]) < mean_huber(task_phase, intents[120:])


def test_ccif_zero_gate_is_identity_and_nonzero_gate_changes_actions() -> None:
    rng = np.random.default_rng(20262903)
    base = rng.normal(scale=0.01, size=(3, CHUNK_SIZE, ACTION_DIM))
    residual = rng.normal(scale=0.02, size=base.shape)
    raw = coarse_intent(base + residual)
    stats = fit_intent_normalizer(np.concatenate([raw, raw + 1e-3], axis=0))
    normalized = normalize_intent(stats, raw)
    identity = apply_ccif_residual(base, residual, normalized, stats, gate=0.0, residual_cap=0.1, beta=0.0)
    changed = apply_ccif_residual(base, residual, normalized, stats, gate=1.0, residual_cap=0.1, beta=0.1)
    assert np.max(np.abs(identity - base)) == 0.0
    assert np.mean(np.abs(changed - base)) > 0.0


def test_resume_validates_feature_and_base_chunk_hashes(tmp_path: Path) -> None:
    manifest_row = _manifest_row("discovery", 0, 1)
    manifest_row["feature_key"] = "FEATURE"
    feature_cache = tmp_path / "feature.npz"
    base_cache = tmp_path / "base.npz"
    np.savez_compressed(feature_cache, feature=np.zeros(VISUAL_FEATURE_DIM, dtype=np.float16))
    base_chunk = np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
    np.savez_compressed(base_cache, base_chunk=base_chunk)
    completed = {
        "row_key": manifest_row["row_key"],
        "feature_key": manifest_row["feature_key"],
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
        "split_integrity_ok": True,
        "minimum_discovery_windows": 512,
        "minimum_validation_windows": 128,
        "all_tasks_reported": True,
        "maximum_validation_task_fraction": 0.25,
        "labels_noncollapsed_discovery": True,
        "labels_noncollapsed_validation": True,
        "collapsed_intent_component_count": 0,
        "intent_probe_beats_task_phase_mean": True,
        "intent_probe_relative_improvement": 0.05,
        "intent_probe_absolute_huber": 0.0,
        "endpoint_only_explains_ccif": False,
        "ccif_beats_prior_relative": 0.05,
        "ccif_beats_prior_absolute_huber": 0.0,
        "ccif_beats_ablation_relative": 0.05,
        "ccif_beats_ablation_absolute_huber": 0.0,
        "action_validity_ok": True,
        "identity_max_abs_error": 0.0,
        "checkpoint_reload_ok": True,
        "finite_objectives_and_gradients": True,
        "ccif_gradient_nonzero": True,
        "frozen_parameter_gradient_count": 0,
        "weighted_gradient_norm_ratio_max": 1.0,
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
    assert classify_stage0(_healthy_inputs()) == "CCIF_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert (
        classify_stage0(_healthy_inputs(labels_noncollapsed_validation=False))
        == "CCIF_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(intent_probe_beats_task_phase_mean=False))
        == "CCIF_STAGE_0_DESIGN_FAILURE"
    )
    assert classify_stage0(_healthy_inputs(endpoint_only_explains_ccif=True)) == "CCIF_STAGE_0_DESIGN_FAILURE"
    assert (
        classify_stage0(_healthy_inputs(ccif_beats_prior_relative=0.0, ccif_beats_prior_absolute_huber=0.0))
        == "CCIF_STAGE_0_NO_USABLE_HEADROOM"
    )
    assert (
        classify_stage0(_healthy_inputs(ccif_gradient_nonzero=False))
        == "CCIF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(confirmatory_records_read=1))
        == "CCIF_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
