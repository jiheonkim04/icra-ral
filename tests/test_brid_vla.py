import json
from pathlib import Path

import numpy as np

from scripts.run_brid_vla_stage0 import CONFIG_LABEL, POLICY_PROBE, _serializer_preflight
from tca_map.smolvla.brid_vla import (
    ACTION_DIM,
    DIFFUSION_STEP_COUNT,
    HORIZON,
    PROPOSAL_HASH,
    Stage0DecisionInputs,
    action_delta_summary,
    apply_brid_residual,
    brid_row_key,
    canonical_json_sha256,
    classify_stage0,
    clean_retention_summary,
    deterministic_noise,
    gradient_smoke,
    group_clip,
    json_default,
    mean_huber,
    noise_identity_for,
    raw_diffusion_proxy_metrics,
    residual_health,
    residual_oracle_metrics,
    residual_targets,
    score_prediction_diagnostics,
    standard_lora_proxy,
    validate_manifest,
)


def test_brid_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "BRID-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "horizon": np.int64(HORIZON),
        "action_dimension": np.int64(ACTION_DIM),
        "diffusion_step_count": np.int64(DIFFUSION_STEP_COUNT),
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
    assert persisted["fixture"]["decision"] == "BRID_STAGE_0_PASS_TO_BOUNDED_VALIDATION"


def _manifest_row(split: str, demo: int, start: int, policy: str = "brid_full") -> dict[str, object]:
    row: dict[str, object] = {
        "split": split,
        "task_suite": "libero_spatial",
        "task_id": "libero_spatial/task_3",
        "demo_id": demo,
        "window_start": start,
        "diffusion_step": 2,
        "policy": policy,
        "probe_label": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
    }
    row["noise_identity"] = noise_identity_for(row)
    row["row_key"] = brid_row_key(row)
    return row


def test_manifest_validation_detects_duplicate_extra_and_split_overlap() -> None:
    manifest = [
        _manifest_row("discovery", 0, 10, "smolvla_base"),
        _manifest_row("discovery", 0, 10, "brid_full"),
        _manifest_row("validation", 8, 10, "brid_full"),
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
    overlapped[1]["noise_identity"] = overlapped[0]["noise_identity"]
    overlapped[1]["row_key"] = brid_row_key(overlapped[1])
    overlap_summary = validate_manifest(overlapped, [{"row_key": row["row_key"]} for row in overlapped])
    assert overlap_summary["split_overlap_key_count"] == 1


def test_noise_identity_is_deterministic_and_shape_checked() -> None:
    row = _manifest_row("validation", 8, 12)
    assert noise_identity_for(row) == noise_identity_for(row)
    noise = deterministic_noise(str(row["noise_identity"]))
    assert noise.shape == (HORIZON, ACTION_DIM)
    assert np.isfinite(noise).all()


def test_residual_score_identity_and_proxy_diagnostics() -> None:
    rng = np.random.default_rng(20263401)
    base = rng.normal(scale=0.01, size=(8, HORIZON, ACTION_DIM))
    expert = base.copy()
    active_scale = np.linspace(0.75, 1.25, 4)[:, None]
    expert[:4, :, 0] += 0.03 * active_scale
    expert[:4, :, 1] -= 0.018 * active_scale
    expert[:4, :, 6] += np.where(np.arange(4)[:, None] % 2 == 0, 0.20, -0.20) * active_scale
    residual = residual_targets(base, expert)
    clipped = group_clip(residual)
    scores = np.asarray([0.05, 0.05, 0.05, 0.05, 0.0, 0.0, 0.0, 0.0])
    brid, gate = apply_brid_residual(base, clipped, scores)
    identity, _ = apply_brid_residual(base, clipped, np.zeros(8), residual_gain=0.0)
    inactive, _ = apply_brid_residual(base, clipped, np.zeros(8))
    standard = standard_lora_proxy(base, clipped)
    raw = base + 0.50 * clipped
    oracle = base + clipped
    task_ids = ["a"] * 4 + ["b"] * 4
    health = residual_health(residual, splits=["discovery"] * 6 + ["validation"] * 2, task_ids=task_ids)
    oracle_metrics = residual_oracle_metrics(base, expert, oracle)
    raw_metrics = raw_diffusion_proxy_metrics(base, expert, raw)
    delta = action_delta_summary(base, brid)
    clean = clean_retention_summary(base, identity, inactive)
    gradient = gradient_smoke(base, clipped, gate, expert)
    assert health["residual_noncollapsed"] is True
    assert oracle_metrics["residual_oracle_headroom_ok"] is True
    assert raw_metrics["raw_diffusion_proxy_headroom"] > 0.0
    assert mean_huber(brid, expert) < mean_huber(base, expert)
    assert mean_huber(brid, expert) < mean_huber(raw, expert)
    assert mean_huber(brid, expert) < mean_huber(standard, expert)
    assert delta["action_deltas_bounded"] is True
    assert clean["clean_retention_ok"] is True
    assert gradient["expected_parameter_gradient_nonzero"] is True


def test_score_prediction_diagnostics_beats_trivial_baselines() -> None:
    rng = np.random.default_rng(20263402)
    noise = rng.normal(scale=2.0, size=(6, HORIZON, ACTION_DIM))
    keys = ["shared"] * 6
    metrics = score_prediction_diagnostics(noise, task_phase_keys=keys, brid_prediction=noise)
    assert metrics["score_predictable"] is True
    assert metrics["brid_score_huber"] == 0.0
    assert metrics["score_prediction_huber_improvement"] > 0.02


def test_stage0_decision_taxonomy() -> None:
    assert classify_stage0(_healthy_inputs()) == "BRID_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert (
        classify_stage0(_healthy_inputs(residual_targets_noncollapsed=False))
        == "BRID_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(residual_oracle_huber_reduction=0.0))
        == "BRID_STAGE_0_NO_RESIDUAL_HEADROOM"
    )
    assert (
        classify_stage0(_healthy_inputs(score_predictable=False))
        == "BRID_STAGE_0_DESIGN_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(identity_max_abs_error=1e-4))
        == "BRID_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(confirmatory_records_read=1))
        == "BRID_STAGE_0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    )


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
        "residual_targets_noncollapsed": True,
        "enough_discovery_windows": True,
        "enough_validation_windows": True,
        "validation_task_coverage_ok": True,
        "maximum_validation_task_fraction": 0.25,
        "noise_identity_valid": True,
        "score_predictable": True,
        "residual_oracle_huber_reduction": 0.05,
        "raw_diffusion_proxy_headroom": 0.05,
        "brid_beats_base": True,
        "brid_beats_raw_diffusion_proxy": True,
        "brid_beats_no_base_residual_ablation": True,
        "brid_beats_standard_lora": True,
        "brid_differs_from_base": True,
        "brid_differs_from_ablation": True,
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
