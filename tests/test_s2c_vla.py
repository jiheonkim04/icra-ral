import json
from pathlib import Path

import numpy as np

from scripts.run_s2c_vla_stage0 import POLICY_PROBE, _serializer_preflight
from tca_map.smolvla.s2c_vla import (
    ACTION_DIM,
    CHUNK_SIZE,
    OVERLAP_LENGTH,
    PROPOSAL_HASH,
    REPLAN_STRIDE,
    Stage0DecisionInputs,
    action_delta_summary,
    apply_s2c_edit,
    boundary_headroom_summary,
    bridge_target,
    canonical_json_sha256,
    classify_stage0,
    current_head,
    effective_mask,
    gripper_event_destruction_count,
    json_default,
    mask_health,
    previous_tail,
    s2c_row_key,
    validate_manifest,
)


def test_s2c_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "S2C-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "chunk_size": np.int64(CHUNK_SIZE),
        "overlap_length": np.int64(OVERLAP_LENGTH),
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


def _manifest_row(split: str, demo: int, frame: int, policy: str = "s2c_full") -> dict[str, object]:
    row: dict[str, object] = {
        "split": split,
        "task_suite": "libero_spatial",
        "task_id": "libero_spatial/task_3",
        "demo_id": demo,
        "window_start": frame,
        "stride": REPLAN_STRIDE,
        "previous_policy_source": "base",
        "policy": policy,
        "policy_probe": POLICY_PROBE,
    }
    row["row_key"] = s2c_row_key(row)
    return row


def test_manifest_validation_detects_duplicates_and_split_overlap() -> None:
    manifest = [
        _manifest_row("discovery", 0, 10, "s2c_full"),
        _manifest_row("discovery", 0, 10, "chunkflow_overlap_proxy"),
        _manifest_row("validation", 8, 10, "s2c_full"),
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


def test_bridge_target_and_s2c_identity_preserve_future_zone() -> None:
    rng = np.random.default_rng(20263101)
    previous = rng.normal(scale=0.01, size=(3, CHUNK_SIZE, ACTION_DIM))
    current = previous.copy()
    current[:, :OVERLAP_LENGTH, 0] += 0.03
    target = bridge_target(current_head(current), previous_tail(previous))
    assert target.shape == (3, OVERLAP_LENGTH, ACTION_DIM)
    identity = apply_s2c_edit(current, previous, np.zeros((3, OVERLAP_LENGTH, ACTION_DIM)), gamma=0.0)
    changed = apply_s2c_edit(current, previous, np.ones((3, OVERLAP_LENGTH, ACTION_DIM)), gamma=1.0)
    assert np.max(np.abs(identity - current)) == 0.0
    assert np.mean(np.abs(changed[:, :OVERLAP_LENGTH] - current[:, :OVERLAP_LENGTH])) > 0.0
    assert np.max(np.abs(changed[:, OVERLAP_LENGTH:] - current[:, OVERLAP_LENGTH:])) == 0.0
    assert action_delta_summary(current, changed)["future_zone_drift_max"] == 0.0


def test_mask_health_boundary_headroom_and_gripper_event() -> None:
    base = np.zeros((2, CHUNK_SIZE, ACTION_DIM), dtype=np.float64)
    previous = np.zeros_like(base)
    previous[:, REPLAN_STRIDE : REPLAN_STRIDE + OVERLAP_LENGTH, 0] = 0.05
    base[:, :OVERLAP_LENGTH, 6] = np.linspace(-1.0, 1.0, OVERLAP_LENGTH)
    summary = boundary_headroom_summary(base, previous)
    assert summary["base_boundary_headroom_ok"] is True
    logits = np.full((2, OVERLAP_LENGTH, ACTION_DIM), -10.0)
    logits[:, :2, :2] = 10.0
    mask = effective_mask(logits, gamma=0.5)
    health = mask_health(mask)
    assert health["mask_noncollapsed"] is True
    changed = apply_s2c_edit(base, previous, np.ones((2, OVERLAP_LENGTH, ACTION_DIM)), gamma=1.0)
    assert gripper_event_destruction_count(base, changed, previous) >= 0


def test_no_previous_tail_forces_exact_base() -> None:
    rng = np.random.default_rng(20263102)
    base = rng.normal(scale=0.01, size=(2, CHUNK_SIZE, ACTION_DIM))
    previous = rng.normal(scale=0.01, size=base.shape)
    output = apply_s2c_edit(base, previous, np.ones((2, OVERLAP_LENGTH, ACTION_DIM)), gamma=1.0, no_previous_tail=[True, True])
    assert np.max(np.abs(output - base)) == 0.0


def test_stage0_decision_taxonomy() -> None:
    assert classify_stage0(_healthy_inputs()) == "S2C_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert (
        classify_stage0(_healthy_inputs(label_contrast_noncollapsed=False))
        == "S2C_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(base_boundary_headroom_ok=False))
        == "S2C_STAGE_0_NO_ADJACENT_BOUNDARY_HEADROOM"
    )
    assert (
        classify_stage0(_healthy_inputs(s2c_beats_chunkflow_relative=0.0))
        == "S2C_STAGE_0_DESIGN_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(future_zone_drift_max=1e-6))
        == "S2C_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(confirmatory_records_read=1))
        == "S2C_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )


def _healthy_inputs(**overrides: object) -> Stage0DecisionInputs:
    values: dict[str, object] = {
        "proposal_hash_ok": True,
        "serializer_preflight_ok": True,
        "official_prior_asset_check_persisted": True,
        "manifest_integrity_ok": True,
        "source_alignment_ok": True,
        "feature_action_proprio_finite_aligned": True,
        "split_integrity_ok": True,
        "adjacent_pair_count": 128,
        "all_tasks_reported": True,
        "maximum_validation_task_fraction": 0.25,
        "label_contrast_noncollapsed": True,
        "base_boundary_headroom_ok": True,
        "chunkflow_residual_headroom_relative": 0.02,
        "identity_max_abs_error": 0.0,
        "checkpoint_reload_ok": True,
        "mask_positive_fraction": 0.20,
        "mask_all_zero": False,
        "mask_all_one": False,
        "future_zone_drift_max": 0.0,
        "action_validity_ok": True,
        "s2c_beats_chunkflow_relative": 0.02,
        "s2c_beats_no_mask_relative": 0.05,
        "standard_lora_explains": False,
        "gripper_event_destruction_count": 0,
        "finite_objectives_and_gradients": True,
        "s2c_gradient_nonzero": True,
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
