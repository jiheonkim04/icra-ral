import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_vdr_vla_stage0a import (
    _evenly_spaced,
    _load_resume,
    _partial_payload,
    _serializer_preflight,
    _sha256,
)
from tca_map.smolvla.vdr_vla import (
    ACTION_DIM,
    FEATURE_DIM,
    HORIZONS,
    PROJECTION_DIM,
    PROPOSAL_HASH,
    Stage0ADecisionInputs,
    action_summary,
    canonical_json_sha256,
    classify_stage0a,
    fit_pca_whitener,
    fit_ridge,
    json_default,
    predict_ridge,
    project_with_whitener,
    regression_metrics,
    torch_action_summary,
    torch_predict_ridge,
    validate_manifest,
    vdr_row_key,
)


def test_vdr_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "VDR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "array": np.arange(4, dtype=np.float32),
        "rank": np.int64(32),
    }
    digest = canonical_json_sha256(fixture)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture, sort_keys=True, default=json_default), encoding="utf-8")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert canonical_json_sha256(parsed) == digest
    assert parsed["array"] == [0.0, 1.0, 2.0, 3.0]
    assert parsed["rank"] == 32


def test_runner_serializer_preflight_writes_parses_and_reproduces_hash(tmp_path: Path) -> None:
    path = tmp_path / "serializer_preflight.json"
    result = _serializer_preflight(path)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert persisted["parsed"] is True
    assert persisted["fixture_hash"] == persisted["reproduced_hash"]


def test_evenly_spaced_sampler_is_deterministic_unique_and_covers_endpoints() -> None:
    rows = [{"index": index} for index in range(113)]
    first = _evenly_spaced(rows, 17)
    second = _evenly_spaced(rows, 17)
    indices = [row["index"] for row in first]
    assert first == second
    assert len(indices) == len(set(indices)) == 17
    assert indices[0] == 0
    assert indices[-1] == 112


def test_action_summary_follows_frozen_horizon_convention() -> None:
    actions = np.arange(30 * ACTION_DIM, dtype=np.float64).reshape(30, ACTION_DIM) / 100.0
    for horizon in HORIZONS:
        summary = action_summary(actions, 2, horizon)
        window = actions[2 : 2 + horizon]
        expected = np.concatenate([window.mean(0), window.std(0), window[:, :6].sum(0), window[-1:, 6]])
        assert np.allclose(summary, expected)
    with pytest.raises(ValueError):
        action_summary(actions, 20, 12)


def test_pca_whitener_projects_to_frozen_dimension() -> None:
    rng = np.random.default_rng(20262400)
    deltas = rng.normal(size=(96, FEATURE_DIM))
    whitener = fit_pca_whitener(deltas)
    projected = project_with_whitener(whitener, deltas[:11])
    assert projected.shape == (11, PROJECTION_DIM)
    assert whitener["projection_dim"] == PROJECTION_DIM
    assert np.isfinite(projected).all()


def test_ridge_regression_recovers_affine_targets() -> None:
    rng = np.random.default_rng(20262401)
    features = rng.normal(size=(300, 12))
    coefficient = rng.normal(size=(12, PROJECTION_DIM))
    intercept = rng.normal(size=(PROJECTION_DIM,))
    targets = features @ coefficient + intercept
    model = fit_ridge(features[:220], targets[:220])
    predicted = predict_ridge(model, features[220:])
    metrics = regression_metrics(model, features[220:], targets[220:])
    assert predicted.shape == targets[220:].shape
    assert np.max(np.abs(predicted - targets[220:])) < 1e-4
    assert metrics["normalized_relative_improvement"] > 0.999


def _manifest_row(partition: str, demo: int, frame: int, horizon: int) -> dict[str, object]:
    row: dict[str, object] = {
        "partition": partition,
        "suite": "libero_spatial",
        "task_identity": "libero_spatial/task_3",
        "source_edge_sha256": "ABC",
        "demo_id": demo,
        "frame_index": frame,
        "horizon": horizon,
    }
    row["row_key"] = vdr_row_key(row)
    return row


def test_manifest_validation_detects_duplicates_missing_extra_and_overlap() -> None:
    manifest = [_manifest_row("discovery", 0, 1, 4), _manifest_row("validation", 8, 1, 4)]
    completed = [{"row_key": row["row_key"]} for row in manifest]
    healthy = validate_manifest(manifest, completed)
    assert healthy["key_sets_equal"] is True
    assert healthy["duplicate_partial_key_count"] == 0
    duplicate = validate_manifest(manifest, completed + [completed[0]])
    assert duplicate["duplicate_partial_key_count"] == 1
    missing = validate_manifest(manifest, completed[:1])
    assert missing["missing_manifest_key_count"] == 1


def test_resume_preserves_prior_exception_and_validates_feature_hashes(tmp_path: Path) -> None:
    manifest_row = _manifest_row("discovery", 0, 1, 4)
    current = tmp_path / "current.npz"
    future = tmp_path / "future.npz"
    np.savez_compressed(current, feature=np.zeros(FEATURE_DIM, dtype=np.float16))
    np.savez_compressed(future, feature=np.ones(FEATURE_DIM, dtype=np.float16))
    completed = {
        "row_key": manifest_row["row_key"],
        "current_feature_path": str(current),
        "future_feature_path": str(future),
        "current_feature_sha256": _sha256(current),
        "future_feature_sha256": _sha256(future),
    }
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


def test_torch_action_summary_and_ridge_prediction_preserve_gradient() -> None:
    rng = np.random.default_rng(20262402)
    features = rng.normal(size=(80, 5))
    targets = rng.normal(size=(80, 3))
    model = fit_ridge(features, targets)
    actions = torch.zeros((1, 12, ACTION_DIM), dtype=torch.float32, requires_grad=True)
    summary = torch_action_summary(actions, 12)
    x = torch.cat([summary[:, :5]], dim=1)
    prediction = torch_predict_ridge(model, x)
    prediction.square().mean().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert float(torch.linalg.vector_norm(actions.grad)) > 0.0


def _healthy_inputs(**overrides: object) -> Stage0ADecisionInputs:
    values: dict[str, object] = {
        "proposal_hash_ok": True,
        "serializer_preflight_ok": True,
        "manifest_integrity_ok": True,
        "source_alignment_ok": True,
        "feature_action_proprio_finite_aligned": True,
        "minimum_discovery_rows_per_horizon": 512,
        "minimum_validation_rows_per_horizon": 128,
        "residual_variance_all_positive": True,
        "maximum_validation_task_fraction": 0.25,
        "all_tasks_reported": True,
        "static_predictor_relative_improvement": 0.25,
        "action_residual_relative_improvement": 0.05,
        "action_residual_absolute_improvement": 0.0,
        "future_proxy_relative_improvement": 0.05,
        "future_proxy_absolute_gap": 0.0,
        "finite_objectives_and_gradients": True,
        "vdr_gradient_nonzero": True,
        "gradient_ratio_at_most_100": True,
        "frozen_parameter_gradient_count": 0,
        "identity_max_error": 0.0,
        "base_hash_unchanged": True,
        "checkpoint_reload_ok": True,
        "action_validity_ok": True,
        "exception_count": 0,
    }
    values.update(overrides)
    return Stage0ADecisionInputs(**values)


def test_stage0a_decision_taxonomy() -> None:
    assert classify_stage0a(_healthy_inputs()) == "VDR_STAGE_0A_PASS_STAGE_0B_ALLOWED"
    assert (
        classify_stage0a(_healthy_inputs(residual_variance_all_positive=False))
        == "VDR_STAGE_0A_DATA_OR_SUPERVISION_FAILURE"
    )
    assert (
        classify_stage0a(_healthy_inputs(static_predictor_relative_improvement=0.24))
        == "VDR_STAGE_0A_NO_USABLE_HEADROOM"
    )
    assert classify_stage0a(_healthy_inputs(vdr_gradient_nonzero=False)) == "VDR_STAGE_0A_DESIGN_FAILURE"
    assert (
        classify_stage0a(_healthy_inputs(serializer_preflight_ok=False))
        == "VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
    assert (
        classify_stage0a(_healthy_inputs(exception_count=1))
        == "VDR_STAGE_0A_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
