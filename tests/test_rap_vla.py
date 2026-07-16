import json
from pathlib import Path

import numpy as np

from scripts.run_rap_vla_stage0 import (
    POLICY_PROBE,
    _array_sha256,
    _evenly_spaced,
    _load_resume,
    _partial_payload,
    _serializer_preflight,
)
from tca_map.smolvla.rap_vla import (
    ACTION_DIM,
    CHUNK_SIZE,
    PROPOSAL_HASH,
    TOP_K,
    VISUAL_FEATURE_DIM,
    Stage0DecisionInputs,
    action_chunk,
    apply_discovery_zscore,
    canonical_json_sha256,
    classify_stage0,
    fit_discovery_zscore,
    fit_ridge,
    flattened_chunks,
    json_default,
    phase_bin,
    prediction_metrics,
    predict_ridge,
    rap_row_key,
    raw_retrieval_feature,
    retrieve_topk_same_task,
    task_phase_mean_chunks,
    uniform_anchor,
    validate_manifest,
)


def test_rap_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "RAP-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "feature": np.arange(4, dtype=np.float32),
        "top_k": np.int64(TOP_K),
    }
    digest = canonical_json_sha256(fixture)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture, sort_keys=True, default=json_default), encoding="utf-8")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert canonical_json_sha256(parsed) == digest
    assert parsed["feature"] == [0.0, 1.0, 2.0, 3.0]
    assert parsed["top_k"] == TOP_K


def test_runner_serializer_preflight_writes_parses_and_reproduces_hash(tmp_path: Path) -> None:
    path = tmp_path / "stage_0_serializer_preflight.json"
    result = _serializer_preflight(path)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert persisted["parsed"] is True
    assert persisted["fixture_hash"] == persisted["reproduced_hash"]
    assert persisted["fixture"]["manifest_row"]["policy_probe"] == POLICY_PROBE


def test_evenly_spaced_sampler_is_deterministic_unique_and_covers_endpoints() -> None:
    rows = [{"index": index} for index in range(97)]
    first = _evenly_spaced(rows, 13)
    second = _evenly_spaced(rows, 13)
    indices = [row["index"] for row in first]
    assert first == second
    assert len(indices) == len(set(indices)) == 13
    assert indices[0] == 0
    assert indices[-1] == 96


def _manifest_row(partition: str, demo: int, frame: int, probe: str = POLICY_PROBE) -> dict[str, object]:
    row: dict[str, object] = {
        "partition": partition,
        "suite": "libero_spatial",
        "task_identity": "libero_spatial/task_3",
        "source_edge_sha256": "ABC",
        "demo_id": demo,
        "frame_index": frame,
        "top_k": TOP_K,
        "policy_probe": probe,
    }
    row["row_key"] = rap_row_key(row)
    return row


def test_manifest_validation_detects_duplicates_missing_extra_and_overlap() -> None:
    manifest = [_manifest_row("discovery", 0, 1), _manifest_row("validation", 8, 1)]
    completed = [{"row_key": row["row_key"]} for row in manifest]
    healthy = validate_manifest(manifest, completed)
    assert healthy["key_sets_equal"] is True
    assert healthy["duplicate_partial_key_count"] == 0
    duplicate = validate_manifest(manifest, completed + [completed[0]])
    assert duplicate["duplicate_partial_key_count"] == 1
    missing = validate_manifest(manifest, completed[:1])
    assert missing["missing_manifest_key_count"] == 1
    extra = validate_manifest(manifest, completed + [{"row_key": "off-manifest"}])
    assert extra["extra_partial_key_count"] == 1


def test_action_chunk_and_phase_bin_follow_frozen_stage0_convention() -> None:
    actions = np.arange(80 * ACTION_DIM, dtype=np.float64).reshape(80, ACTION_DIM) / 100.0
    chunk = action_chunk(actions, 4)
    assert chunk.shape == (CHUNK_SIZE, ACTION_DIM)
    assert np.allclose(chunk[0], actions[4])
    assert phase_bin(0.0) == 0
    assert phase_bin(0.999) == 9
    assert phase_bin(1.0) == 9


def test_discovery_zscore_preserves_task_one_hot_without_validation_leakage() -> None:
    visual = np.zeros(VISUAL_FEATURE_DIM, dtype=np.float64)
    rows = []
    for task in range(4):
        for offset in range(3):
            rows.append(raw_retrieval_feature(visual + offset + task, np.ones(8) * offset, task, offset / 2))
    features = np.asarray(rows)
    stats = fit_discovery_zscore(features[:8])
    transformed = apply_discovery_zscore(stats, features)
    assert transformed.shape[1] == VISUAL_FEATURE_DIM + 8 + 1 + 4
    assert np.allclose(transformed[:, -4:], features[:, -4:])


def test_same_task_topk_excludes_self_and_uniform_anchor_uses_legal_memory() -> None:
    memory_features = np.asarray([[0.0], [1.0], [2.0], [100.0], [101.0]], dtype=np.float64)
    query_features = np.asarray([[0.0], [100.0]], dtype=np.float64)
    memory_tasks = ["a", "a", "a", "b", "b"]
    query_tasks = ["a", "b"]
    retrievals = retrieve_topk_same_task(
        query_features,
        memory_features,
        query_tasks,
        memory_tasks,
        k=2,
        query_keys=["m0", "m3"],
        memory_keys=["m0", "m1", "m2", "m3", "m4"],
    )
    assert retrievals[0]["indices"].tolist() == [1, 2]
    assert retrievals[1]["indices"].tolist() == [4]
    chunks = np.zeros((5, CHUNK_SIZE, ACTION_DIM), dtype=np.float64)
    chunks[1] += 2.0
    chunks[2] += 4.0
    chunks[3] += 9.0
    chunks[4] += 11.0
    anchors = uniform_anchor(chunks, retrievals)
    assert np.allclose(anchors[0], 3.0)
    assert np.allclose(anchors[1], 11.0)


def test_retrieval_anchor_beats_task_phase_mean_on_structured_memory() -> None:
    chunks = np.zeros((6, CHUNK_SIZE, ACTION_DIM), dtype=np.float64)
    chunks[:3] += np.asarray([0.0, 2.0, 4.0]).reshape(3, 1, 1)
    chunks[3:] += np.asarray([10.0, 11.0, 12.0]).reshape(3, 1, 1)
    tasks = ["a", "a", "a", "b", "b", "b"]
    phases = [0.51, 0.52, 0.9, 0.1, 0.5, 0.9]
    query_tasks = ["a"]
    query_phases = [0.52]
    retrieval = [{"indices": np.asarray([1]), "distances": np.asarray([0.0]), "available_count": 3}]
    anchor = uniform_anchor(chunks, retrieval)
    baseline = task_phase_mean_chunks(chunks, tasks, phases, query_tasks, query_phases)
    target = chunks[1:2]
    metrics = prediction_metrics(flattened_chunks(anchor), flattened_chunks(baseline), flattened_chunks(target))
    assert metrics["relative_mse_improvement"] > 0.99


def test_ridge_residual_probe_recovers_affine_residuals() -> None:
    rng = np.random.default_rng(20262500)
    features = rng.normal(size=(240, 9))
    coefficient = rng.normal(size=(9, 14))
    residuals = features @ coefficient
    model = fit_ridge(features[:180], residuals[:180])
    predicted = predict_ridge(model, features[180:])
    metrics = prediction_metrics(predicted, np.zeros_like(predicted), residuals[180:])
    assert np.max(np.abs(predicted - residuals[180:])) < 1e-4
    assert metrics["relative_mse_improvement"] > 0.999


def test_resume_preserves_prior_exception_and_validates_feature_hash(tmp_path: Path) -> None:
    manifest_row = _manifest_row("discovery", 0, 1)
    cache = tmp_path / "feature.npz"
    np.savez_compressed(cache, feature=np.zeros(VISUAL_FEATURE_DIM, dtype=np.float16))
    completed = {
        "row_key": manifest_row["row_key"],
        "feature_cache_path": str(cache),
    }
    completed["feature_cache_sha256"] = __import__("hashlib").sha256(cache.read_bytes()).hexdigest().upper()
    partial = _partial_payload(
        "MANIFEST",
        1,
        [completed],
        exception_count=1,
        last_exception="transient feature extraction failure",
    )
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(partial), encoding="utf-8")
    rows, exception_count, last_exception = _load_resume(path, [manifest_row], "MANIFEST")
    assert rows == [completed]
    assert exception_count == 1
    assert last_exception == "transient feature extraction failure"


def test_resume_ignores_zero_row_premanifest_blocker_without_repeating_rows(tmp_path: Path) -> None:
    manifest_row = _manifest_row("discovery", 0, 1)
    partial = _partial_payload(
        None,
        None,
        [],
        exception_count=1,
        last_exception="pre-manifest wrapper quoting failure",
    )
    path = tmp_path / "partial.json"
    path.write_text(json.dumps(partial), encoding="utf-8")
    rows, exception_count, last_exception = _load_resume(path, [manifest_row], "MANIFEST")
    assert rows == []
    assert exception_count == 0
    assert last_exception is None


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
        "median_topk_unique_demos": 3.0,
        "top1_max_source_fraction": 0.25,
        "anchor_relative_improvement": 0.10,
        "anchor_absolute_huber_improvement": 0.0,
        "residual_variance_all_positive": True,
        "residual_probe_relative_improvement": 0.05,
        "residual_probe_absolute_huber_improvement": 0.0,
        "anchor_and_residual_paths_distinct": True,
        "finite_objectives_and_gradients": True,
        "rap_gradient_nonzero": True,
        "gradient_ratio_at_most_100": True,
        "frozen_parameter_gradient_count": 0,
        "identity_max_error": 0.0,
        "base_hash_unchanged": True,
        "checkpoint_reload_ok": True,
        "action_validity_ok": True,
        "exception_count": 0,
    }
    values.update(overrides)
    return Stage0DecisionInputs(**values)


def test_stage0_decision_taxonomy() -> None:
    assert classify_stage0(_healthy_inputs()) == "RAP_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert (
        classify_stage0(_healthy_inputs(median_topk_unique_demos=2.0))
        == "RAP_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(anchor_relative_improvement=0.09, anchor_absolute_huber_improvement=0.009))
        == "RAP_STAGE_0_NO_USABLE_HEADROOM"
    )
    assert classify_stage0(_healthy_inputs(rap_gradient_nonzero=False)) == "RAP_STAGE_0_DESIGN_FAILURE"
    assert (
        classify_stage0(_healthy_inputs(serializer_preflight_ok=False))
        == "RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(exception_count=1))
        == "RAP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
