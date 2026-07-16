import json
from pathlib import Path

import numpy as np

from scripts.run_mhs_vla_stage0 import CONFIG_LABEL, POLICY_PROBE, _safe_masked_mean, _serializer_preflight, _write_json
from tca_map.smolvla.mhs_vla import (
    ACTION_DIM,
    HISTORY_LENGTH,
    HORIZON,
    PROPOSAL_HASH,
    Stage0DecisionInputs,
    action_delta_summary,
    apply_mhs_residual,
    build_current_features,
    build_history_features,
    canonical_json_sha256,
    classify_stage0,
    clean_retention_summary,
    construct_history_labels,
    gradient_smoke,
    group_clip,
    history_identity_for,
    history_predictability_diagnostics,
    json_default,
    label_health,
    mean_huber,
    mhs_row_key,
    normalize_z_targets,
    residual_targets,
    standard_lora_proxy,
    validate_manifest,
)


def test_mhs_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "MHS-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "horizon": np.int64(HORIZON),
        "action_dimension": np.int64(ACTION_DIM),
        "history_length": np.int64(HISTORY_LENGTH),
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
    assert persisted["fixture"]["manifest_row"]["probe_label"] == POLICY_PROBE
    assert persisted["fixture"]["config_label"] == CONFIG_LABEL
    assert persisted["fixture"]["decision"] == "MHS_STAGE_0_PASS_TO_BOUNDED_VALIDATION"


def test_json_writer_and_empty_masks_are_resume_safe(tmp_path: Path) -> None:
    assert _safe_masked_mean(np.asarray([1.0]), np.asarray([False])) == 0.0
    diagnostics = history_predictability_diagnostics(
        np.asarray([0, 1]),
        np.asarray([False, False]),
        ["libero_goal/task_5", "libero_goal/task_5"],
        np.asarray([-1, -1]),
        np.asarray([-1, -1]),
    )
    assert diagnostics["bce_defined"] is False
    assert np.isfinite(diagnostics["history_bce"])
    path = tmp_path / "strict.json"
    _write_json(path, {"finite": 1.0, "undefined": float("inf")})
    persisted = json.loads(path.read_text(encoding="utf-8"))
    assert persisted == {"finite": 1.0, "undefined": None}


def _manifest_row(split: str, demo: int, start: int, policy: str = "mhs_full") -> dict[str, object]:
    row: dict[str, object] = {
        "split": split,
        "task_suite": "libero_goal",
        "task_id": "libero_goal/task_5",
        "demo_id": demo,
        "window_start": start,
        "policy": policy,
        "probe_label": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
    }
    row["history_identity"] = history_identity_for(row)
    row["row_key"] = mhs_row_key(row)
    return row


def test_manifest_validation_detects_duplicate_extra_and_split_overlap() -> None:
    manifest = [
        _manifest_row("discovery", 0, 10, "smolvla_base"),
        _manifest_row("discovery", 0, 10, "mhs_full"),
        _manifest_row("validation", 8, 10, "mhs_full"),
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
    overlapped[1]["history_identity"] = overlapped[0]["history_identity"]
    overlapped[1]["row_key"] = mhs_row_key(overlapped[1])
    overlap_summary = validate_manifest(overlapped, [{"row_key": row["row_key"]} for row in overlapped])
    assert overlap_summary["split_overlap_key_count"] == 1


def test_history_labels_predictability_and_targets_are_noncollapsed() -> None:
    base, expert, history, splits, task_ids = _synthetic_history_problem()
    current = build_current_features(base, task_ids)
    hist = build_history_features(history, task_ids)
    labels = construct_history_labels(base, expert, current, hist, splits=splits, task_ids=task_ids)
    health = label_health(labels["m"], labels["valid_mask"], task_ids)
    z_norm = normalize_z_targets(labels["z"], [split == "discovery" for split in splits])
    predictability = history_predictability_diagnostics(
        labels["m"],
        labels["valid_mask"],
        task_ids,
        labels["current_neighbor"],
        labels["history_neighbor"],
    )
    assert health["labels_noncollapsed"] is True
    assert health["positive_count"] > 0
    assert health["negative_count"] > 0
    assert z_norm["z_iqr_valid"] is True
    assert np.isfinite(predictability["history_predictability_margin"])


def test_mhs_identity_residual_and_proxy_diagnostics() -> None:
    base, expert, history, splits, task_ids = _synthetic_history_problem()
    current = build_current_features(base, task_ids)
    hist = build_history_features(history, task_ids)
    labels = construct_history_labels(base, expert, current, hist, splits=splits, task_ids=task_ids)
    residual = residual_targets(base, expert)
    clipped = group_clip(residual)
    gate = labels["m"].astype(float)
    mhs, gate_array = apply_mhs_residual(base, clipped, gate)
    identity, _ = apply_mhs_residual(base, np.zeros_like(clipped), np.zeros(len(base)))
    inactive, _ = apply_mhs_residual(base, clipped, np.zeros(len(base)))
    no_history, _ = apply_mhs_residual(base, 0.25 * clipped, gate)
    standard = standard_lora_proxy(base, clipped)
    delta = action_delta_summary(base, mhs)
    clean = clean_retention_summary(base, identity, inactive)
    gradient = gradient_smoke(base, clipped, np.maximum(gate, 1.0), expert)
    assert mean_huber(mhs, expert) < mean_huber(no_history, expert)
    assert mean_huber(mhs, expert) < mean_huber(standard, expert)
    assert delta["action_deltas_bounded"] is True
    assert clean["clean_retention_ok"] is True
    assert gradient["expected_parameter_gradient_nonzero"] is True
    assert float(np.mean(gate_array > 0.5)) > 0.0


def test_stage0_decision_taxonomy() -> None:
    assert classify_stage0(_healthy_inputs()) == "MHS_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert (
        classify_stage0(_healthy_inputs(labels_noncollapsed=False))
        == "MHS_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(history_predictability_margin=0.0))
        == "MHS_STAGE_0_NO_USABLE_HEADROOM"
    )
    assert (
        classify_stage0(_healthy_inputs(mhs_beats_mtil_proxy=False))
        == "MHS_STAGE_0_DESIGN_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(identity_max_abs_error=1e-4))
        == "MHS_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(confirmatory_records_read=1))
        == "MHS_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )


def _synthetic_history_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str], list[str]]:
    rng = np.random.default_rng(20263501)
    n = 12
    base = np.zeros((n, HORIZON, ACTION_DIM), dtype=np.float64)
    expert = base.copy()
    history = rng.normal(scale=0.001, size=(n, HISTORY_LENGTH, ACTION_DIM))
    task_ids = ["libero_goal/task_5"] * 6 + ["libero_10/task_5"] * 6
    splits = ["discovery"] * 8 + ["validation"] * 4
    positive_indexes = {0, 2, 8, 10}
    for idx in range(n):
        history[idx, :, 0] -= 5.0
    for idx in positive_indexes:
        expert[idx, :, 0] += 6.0
        expert[idx, :, 1] -= 4.0
        expert[idx, :, 6] += 5.0
        history[idx, :, 0] += 5.0
    return base, expert, history, splits, task_ids


def _healthy_inputs(**overrides: object) -> Stage0DecisionInputs:
    values: dict[str, object] = {
        "proposal_hash_ok": True,
        "serializer_preflight_ok": True,
        "official_prior_asset_check_persisted": True,
        "preflight_passed": True,
        "manifest_integrity_ok": True,
        "source_alignment_ok": True,
        "action_semantics_ok": True,
        "base_chunks_valid": True,
        "history_windows_valid": True,
        "labels_noncollapsed": True,
        "enough_discovery_windows": True,
        "enough_validation_windows": True,
        "validation_task_coverage_ok": True,
        "maximum_validation_task_fraction": 0.25,
        "validation_unmasked_label_count": 128,
        "validation_positive_count": 16,
        "validation_negative_count": 112,
        "validation_positive_fraction": 0.125,
        "largest_positive_task_fraction": 0.50,
        "z_iqr_valid": True,
        "history_predictability_margin": 0.03,
        "history_neighbor_margin": 0.02,
        "base_residual_activity": True,
        "mtil_proxy_headroom": 0.05,
        "mhs_beats_mtil_proxy": True,
        "mhs_beats_no_history_ablation": True,
        "mhs_beats_standard_lora": True,
        "mhs_differs_from_base": True,
        "mhs_differs_from_ablation": True,
        "identity_max_abs_error": 0.0,
        "checkpoint_reload_ok": True,
        "finite_objectives_and_gradients": True,
        "expected_parameter_gradient_nonzero": True,
        "frozen_base_gradient_count": 0,
        "weighted_gradient_norm_ratio_max": 1.0,
        "intervention_fraction": 0.25,
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
