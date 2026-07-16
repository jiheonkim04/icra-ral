import json
from pathlib import Path

import numpy as np

from scripts.run_amp_vla_stage0 import POLICY_PROBE, _serializer_preflight
from tca_map.smolvla.amp_vla import (
    ACTION_DIM,
    CHUNK_SIZE,
    LATENT_DIMS,
    PROPOSAL_HASH,
    Stage0DecisionInputs,
    amp_row_key,
    canonical_json_sha256,
    classify_stage0,
    decode_manifold,
    encode_manifold,
    fit_action_manifold,
    json_default,
    manifold_consistency,
    mean_huber,
    phase_bin,
    predict_ridge,
    project_to_manifold,
    fit_ridge,
    task_phase_mean_chunks,
    task_phase_mean_coordinates,
    validate_manifest,
)


def test_amp_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "AMP-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "latent_dims": np.asarray(LATENT_DIMS, dtype=np.int64),
    }
    digest = canonical_json_sha256(fixture)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture, sort_keys=True, default=json_default), encoding="utf-8")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert canonical_json_sha256(parsed) == digest
    assert parsed["latent_dims"] == list(LATENT_DIMS)


def test_runner_serializer_preflight_writes_parses_and_reproduces_hash(tmp_path: Path) -> None:
    path = tmp_path / "stage_0_serializer_preflight.json"
    result = _serializer_preflight(path)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert persisted["parsed"] is True
    assert persisted["fixture_hash"] == persisted["reproduced_hash"]
    assert persisted["fixture"]["manifest_row"]["policy_probe"] == POLICY_PROBE


def _manifest_row(partition: str, demo: int, frame: int, latent_dim: int = 8) -> dict[str, object]:
    row: dict[str, object] = {
        "partition": partition,
        "suite": "libero_spatial",
        "task_identity": "libero_spatial/task_3",
        "source_edge_sha256": "ABC",
        "demo_id": demo,
        "frame_index": frame,
        "latent_dim": latent_dim,
        "policy_probe": POLICY_PROBE,
    }
    row["row_key"] = amp_row_key(row)
    return row


def test_manifest_validation_includes_latent_dim_in_identity() -> None:
    manifest = [_manifest_row("discovery", 0, 1, 8), _manifest_row("validation", 8, 1, 16)]
    completed = [{"row_key": row["row_key"]} for row in manifest]
    healthy = validate_manifest(manifest, completed)
    assert healthy["key_sets_equal"] is True
    duplicate = validate_manifest(manifest, completed + [completed[0]])
    assert duplicate["duplicate_partial_key_count"] == 1
    extra = validate_manifest(manifest, completed + [{"row_key": "off-manifest"}])
    assert extra["extra_partial_key_count"] == 1


def test_phase_bin_and_action_shape_are_frozen() -> None:
    assert phase_bin(0.0) == 0
    assert phase_bin(0.999) == 9
    assert phase_bin(1.0) == 9
    chunks = np.zeros((3, CHUNK_SIZE, ACTION_DIM), dtype=np.float64)
    model = fit_action_manifold(chunks + np.arange(3).reshape(3, 1, 1), latent_dim=2)
    assert model["latent_dim"] == 2


def test_action_manifold_reconstructs_low_rank_chunks_and_projects_off_support() -> None:
    basis = np.linspace(-1.0, 1.0, CHUNK_SIZE * ACTION_DIM).reshape(CHUNK_SIZE, ACTION_DIM)
    chunks = np.asarray([(scale * basis) for scale in np.linspace(-0.5, 0.5, 12)], dtype=np.float64)
    model = fit_action_manifold(chunks, latent_dim=1)
    encoded = encode_manifold(model, chunks)
    decoded = decode_manifold(model, encoded)
    assert decoded.shape == chunks.shape
    assert mean_huber(decoded, chunks) < 1e-10
    noisy = chunks + 0.25
    projected = project_to_manifold(model, noisy)
    assert manifold_consistency(model, projected) < manifold_consistency(model, noisy)


def test_task_phase_mean_action_and_coordinate_baselines() -> None:
    chunks = np.zeros((4, CHUNK_SIZE, ACTION_DIM), dtype=np.float64)
    chunks[0] += 1.0
    chunks[1] += 2.0
    chunks[2] += 10.0
    chunks[3] += 12.0
    tasks = ["a", "a", "b", "b"]
    phases = [0.11, 0.12, 0.11, 0.12]
    action_mean = task_phase_mean_chunks(chunks, tasks, phases, ["a"], [0.115])
    assert np.allclose(action_mean, 1.5)
    coords = np.asarray([[1.0, 0.0], [3.0, 0.0], [10.0, 1.0], [12.0, 1.0]])
    coord_mean = task_phase_mean_coordinates(coords, tasks, phases, ["b"], [0.115])
    assert np.allclose(coord_mean, [[11.0, 1.0]])


def test_ridge_coordinate_probe_recovers_affine_coordinates() -> None:
    rng = np.random.default_rng(20262600)
    features = rng.normal(size=(120, 5))
    weights = rng.normal(size=(5, 3))
    coords = features @ weights
    model = fit_ridge(features[:90], coords[:90])
    pred = predict_ridge(model, features[90:])
    assert np.max(np.abs(pred - coords[90:])) < 1e-4


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
        "coordinate_variance_all_positive": True,
        "manifold_reconstruction_relative_improvement": 0.10,
        "manifold_reconstruction_absolute_huber_improvement": 0.0,
        "coordinate_probe_relative_improvement": 0.05,
        "coordinate_probe_absolute_huber_improvement": 0.0,
        "abot_proxy_headroom_relative_improvement": 0.05,
        "abot_proxy_headroom_absolute_huber_improvement": 0.0,
        "clipping_explains_projection": False,
        "projection_path_distinct": True,
        "finite_objectives_and_gradients": True,
        "amp_gradient_nonzero": True,
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
    assert classify_stage0(_healthy_inputs()) == "AMP_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert (
        classify_stage0(_healthy_inputs(coordinate_variance_all_positive=False))
        == "AMP_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(manifold_reconstruction_relative_improvement=0.09))
        == "AMP_STAGE_0_NO_USABLE_HEADROOM"
    )
    assert (
        classify_stage0(_healthy_inputs(coordinate_probe_relative_improvement=0.04))
        == "AMP_STAGE_0_DESIGN_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(serializer_preflight_ok=False))
        == "AMP_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
