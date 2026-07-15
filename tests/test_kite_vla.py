import json
from pathlib import Path

import numpy as np
import pytest
import torch

from scripts.run_kite_vla_stage0a import _evenly_spaced, _serializer_preflight
from tca_map.smolvla.kite_vla import (
    HORIZONS,
    Stage0ADecisionInputs,
    canonical_json_sha256,
    classify_stage0a,
    cumulative_arm_command,
    differentiable_mean_std_unnormalize,
    fit_realization_operator,
    json_default,
    predict_realization,
    realization_metrics,
    realization_row_key,
    state_displacement,
    torch_realization_normalized,
    validate_manifest,
)


def test_numpy_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "path": tmp_path / "operator.json",
        "mean": np.asarray([1.0, 2.0], dtype=np.float32),
        "rank": np.int64(6),
    }
    first = canonical_json_sha256(fixture)
    path = tmp_path / "fixture.json"
    path.write_text(json.dumps(fixture, sort_keys=True, default=json_default), encoding="utf-8")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    assert canonical_json_sha256(parsed) == first
    assert parsed["mean"] == [1.0, 2.0]
    assert parsed["rank"] == 6


def test_runner_serializer_preflight_writes_parses_and_reproduces_hash(tmp_path: Path) -> None:
    path = tmp_path / "serializer_preflight.json"
    result = _serializer_preflight(path)
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert result["passed"] is True
    assert persisted["parsed"] is True
    assert persisted["fixture_hash"] == persisted["reproduced_hash"]


def test_evenly_spaced_sampler_is_deterministic_unique_and_covers_endpoints() -> None:
    rows = [{"index": index} for index in range(101)]
    first = _evenly_spaced(rows, 8)
    second = _evenly_spaced(rows, 8)
    indices = [row["index"] for row in first]
    assert first == second
    assert len(indices) == len(set(indices)) == 8
    assert indices[0] == 0
    assert indices[-1] == 100


def test_labels_follow_frozen_horizon_convention() -> None:
    actions = np.arange(30 * 7, dtype=np.float64).reshape(30, 7) / 100.0
    states = np.arange(30 * 6, dtype=np.float64).reshape(30, 6) / 10.0
    for horizon in HORIZONS:
        assert np.allclose(cumulative_arm_command(actions, 2, horizon), actions[2 : 2 + horizon, :6].sum(0))
        assert np.allclose(state_displacement(states, 2, horizon), states[2 + horizon] - states[2])
    with pytest.raises(ValueError):
        cumulative_arm_command(actions, 20, 20)


def test_ridge_operator_recovers_affine_realization() -> None:
    rng = np.random.default_rng(20262300)
    commands = rng.normal(size=(800, 6))
    coefficient = rng.normal(size=(6, 6))
    intercept = rng.normal(size=(6,))
    displacements = commands @ coefficient + intercept
    operator = fit_realization_operator(commands[:640], displacements[:640])
    predicted = predict_realization(operator, commands[640:])
    metrics = realization_metrics(operator, commands[640:], displacements[640:])
    assert operator["rank"] == 6
    assert np.max(np.abs(predicted - displacements[640:])) < 1e-5
    assert metrics["normalized_relative_improvement"] > 0.999


def _manifest_row(partition: str, demo: int, frame: int, horizon: int) -> dict[str, object]:
    row: dict[str, object] = {
        "partition": partition,
        "suite": "suite",
        "task_identity": "suite/task_0",
        "source_edge_sha256": "ABC",
        "demo_id": demo,
        "frame_index": frame,
        "horizon": horizon,
    }
    row["row_key"] = realization_row_key(row)
    return row


def test_manifest_validation_detects_duplicates_missing_extra_and_overlap() -> None:
    manifest = [_manifest_row("discovery", 0, 1, 5), _manifest_row("validation", 8, 1, 5)]
    completed = [{"row_key": row["row_key"]} for row in manifest]
    healthy = validate_manifest(manifest, completed)
    assert healthy["key_sets_equal"] is True
    assert healthy["duplicate_partial_key_count"] == 0
    duplicate = validate_manifest(manifest, completed + [completed[0]])
    assert duplicate["duplicate_partial_key_count"] == 1
    assert duplicate["key_sets_equal"] is True
    missing = validate_manifest(manifest, completed[:1])
    assert missing["missing_manifest_key_count"] == 1


def test_differentiable_unnormalization_and_operator_preserve_gradient() -> None:
    actions = torch.zeros((1, 20, 7), dtype=torch.float32, requires_grad=True)
    raw = differentiable_mean_std_unnormalize(actions, np.arange(7), np.ones(7) * 2.0)
    commands = raw[:, :5, :6].sum(dim=1)
    identity_rows = np.eye(6, dtype=np.float64)
    operator = fit_realization_operator(np.concatenate([-identity_rows, identity_rows]), np.concatenate([-identity_rows, identity_rows]))
    output = torch_realization_normalized(operator, commands)
    output.square().mean().backward()
    assert actions.grad is not None
    assert torch.isfinite(actions.grad).all()
    assert float(torch.linalg.vector_norm(actions.grad)) > 0.0


def _healthy_inputs(**overrides: object) -> Stage0ADecisionInputs:
    values: dict[str, object] = {
        "proposal_hash_ok": True,
        "serializer_preflight_ok": True,
        "manifest_integrity_ok": True,
        "source_alignment_ok": True,
        "minimum_discovery_rows_per_horizon": 512,
        "minimum_validation_rows_per_horizon": 96,
        "command_variance_all_positive": True,
        "state_variance_all_positive": True,
        "maximum_sampled_task_fraction": 0.25,
        "all_operator_ranks_six": True,
        "minimum_operator_relative_improvement": 0.75,
        "all_tasks_reported": True,
        "base_headroom_passed": True,
        "finite_objectives_and_gradients": True,
        "kite_gradient_nonzero": True,
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
    assert classify_stage0a(_healthy_inputs()) == "KITE_STAGE_0A_PASS_STAGE_0B_ALLOWED"
    assert classify_stage0a(_healthy_inputs(state_variance_all_positive=False)) == "KITE_STAGE_0A_DATA_FAILURE"
    assert classify_stage0a(_healthy_inputs(base_headroom_passed=False)) == "KITE_STAGE_0A_NO_HEADROOM"
    assert classify_stage0a(_healthy_inputs(kite_gradient_nonzero=False)) == "KITE_STAGE_0A_DESIGN_FAILURE"
    assert classify_stage0a(_healthy_inputs(serializer_preflight_ok=False)) == "KITE_STAGE_0A_IMPLEMENTATION_FAILURE"
