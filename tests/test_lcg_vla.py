import json
from pathlib import Path

import numpy as np

from scripts.run_lcg_vla_stage0 import POLICY_PROBE, _serializer_preflight
from tca_map.smolvla.lcg_vla import (
    ACTION_DIM,
    CAG_PROXY_BETAS,
    HORIZON,
    NULL_INSTRUCTION,
    PROPOSAL_HASH,
    Stage0DecisionInputs,
    action_delta_summary,
    apply_cag_proxy,
    apply_lcg_gate,
    apply_no_language_ablation,
    canonical_json_sha256,
    classify_stage0,
    clean_retention_summary,
    construct_language_contrast,
    contrast_residual_noncollapse,
    fit_discovery_contrast_scale,
    gradient_smoke,
    group_clip,
    json_default,
    language_mask,
    lcg_row_key,
    mask_health,
    scalar_contrast_residual_probe,
    validate_manifest,
)


def test_lcg_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "LCG-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "horizon": np.int64(HORIZON),
        "action_dimension": np.int64(ACTION_DIM),
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
    assert persisted["fixture"]["null_instruction"] == NULL_INSTRUCTION
    assert persisted["fixture"]["decision"] == "LCG_STAGE_0_PASS_TO_BOUNDED_VALIDATION"


def _manifest_row(partition: str, demo: int, frame: int, policy: str = "lcg_full") -> dict[str, object]:
    row: dict[str, object] = {
        "partition": partition,
        "suite": "libero_spatial",
        "task_identity": "libero_spatial/task_3",
        "source_edge_sha256": "ABC",
        "demo_id": demo,
        "frame_index": frame,
        "instruction_variant": "original_vs_null",
        "model_or_probe": policy,
        "policy_probe": POLICY_PROBE,
    }
    row["row_key"] = lcg_row_key(row)
    return row


def test_manifest_validation_detects_duplicates_extra_and_split_overlap() -> None:
    manifest = [
        _manifest_row("discovery", 0, 10, "smolvla_base"),
        _manifest_row("discovery", 0, 10, "lcg_full"),
        _manifest_row("validation", 8, 10, "lcg_full"),
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


def test_language_contrast_mask_gate_identity_and_cag_proxy() -> None:
    rng = np.random.default_rng(20263201)
    base = rng.normal(scale=0.01, size=(4, HORIZON, ACTION_DIM))
    null = base.copy()
    null[:, :, 0] -= 0.04
    residual = np.zeros_like(base)
    residual[:, :, 0] = 0.03
    contrast = construct_language_contrast(base, null)
    scale = fit_discovery_contrast_scale(contrast)
    mask = language_mask(contrast, scale)
    identity = apply_lcg_gate(base, residual, mask, residual_gain=0.0)
    changed = apply_lcg_gate(base, residual, mask, residual_gain=1.0)
    inactive = apply_lcg_gate(base, residual, np.zeros((4, HORIZON, 1)), residual_gain=1.0)
    assert np.max(np.abs(identity - base)) == 0.0
    assert np.max(np.abs(inactive - base)) == 0.0
    assert np.mean(np.abs(changed - base)) > 0.0
    assert clean_retention_summary(base, identity, inactive)["clean_retention_ok"] is True
    assert action_delta_summary(base, changed)["action_deltas_bounded"] is True
    cag = apply_cag_proxy(base, null, beta=CAG_PROXY_BETAS[1])
    ablation = apply_no_language_ablation(base, residual)
    assert cag.shape == base.shape
    assert ablation.shape == base.shape
    assert group_clip(residual).shape == base.shape
    assert mask_health(mask)["language_mask_all_zero"] is False


def test_contrast_residual_probe_and_gradient_smoke_are_nonzero() -> None:
    rng = np.random.default_rng(20263202)
    contrast = rng.normal(scale=0.03, size=(16, HORIZON, ACTION_DIM))
    residual = 0.5 * contrast
    base = rng.normal(scale=0.01, size=(16, HORIZON, ACTION_DIM))
    target = base + group_clip(residual)
    model = scalar_contrast_residual_probe(contrast, residual)
    predicted = model["slope"].reshape(1, 1, ACTION_DIM) * contrast
    summary = contrast_residual_noncollapse(contrast, residual)
    gradient = gradient_smoke(base, predicted, np.ones((16, HORIZON, 1)), target)
    assert summary["contrast_noncollapsed"] is True
    assert summary["residual_labels_noncollapsed"] is True
    assert summary["contrast_residual_spearman"] > 0.9
    assert gradient["finite_objectives_and_gradients"] is True
    assert gradient["expected_parameter_gradient_nonzero"] is True


def test_stage0_decision_taxonomy() -> None:
    assert classify_stage0(_healthy_inputs()) == "LCG_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert (
        classify_stage0(_healthy_inputs(contrast_noncollapsed=False))
        == "LCG_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(masked_residual_headroom=0.0))
        == "LCG_STAGE_0_NO_USABLE_HEADROOM"
    )
    assert (
        classify_stage0(_healthy_inputs(contrast_residual_spearman=0.0))
        == "LCG_STAGE_0_DESIGN_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(identity_max_abs_error=1e-4))
        == "LCG_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
    )
    assert (
        classify_stage0(_healthy_inputs(confirmatory_records_read=1))
        == "LCG_STAGE_0_IMPLEMENTATION_OR_OBJECTIVE_SCALE_FAILURE"
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
        "minimum_discovery_windows": 512,
        "minimum_validation_windows": 128,
        "all_tasks_reported": True,
        "maximum_validation_task_fraction": 0.25,
        "contrast_noncollapsed": True,
        "residual_labels_noncollapsed": True,
        "contrast_positive_fraction": 0.25,
        "language_mask_all_zero": False,
        "language_mask_all_one": False,
        "gate_activation_fraction": 0.25,
        "contrast_residual_spearman": 0.10,
        "contrast_probe_beats_task_phase_baseline": True,
        "contrast_probe_relative_improvement": 0.02,
        "best_cag_proxy_score": 0.1,
        "cag_proxy_residual_headroom": 0.05,
        "lcg_beats_cag_proxy_relative": 0.01,
        "masked_residual_headroom": 0.05,
        "cag_coefficient_equivalence": False,
        "no_language_ablation_explains": False,
        "lora_explains": False,
        "identity_max_abs_error": 0.0,
        "inactive_gate_max_abs_error": 0.0,
        "action_validity_ok": True,
        "clean_retention_ok": True,
        "finite_objectives_and_gradients": True,
        "expected_parameter_gradient_nonzero": True,
        "frozen_base_gradient_count": 0,
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
