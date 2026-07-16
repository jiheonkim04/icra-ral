import json
from pathlib import Path

import numpy as np

from scripts.run_dccg_vla_stage0 import CONFIG_LABEL, POLICY_PROBE, _serializer_preflight, main as dccg_stage0_main
from tca_map.smolvla.dccg_vla import (
    ACTION_DIM,
    FEATURE_COUNT,
    HORIZON,
    PROPOSAL_HASH,
    Stage0DecisionInputs,
    action_delta_summary,
    action_validity_summary,
    apply_dccg_guidance,
    canonical_json_sha256,
    classify_stage0,
    coherence_energy,
    coherence_features,
    deployment_bin_key,
    dccg_row_key,
    feature_health,
    fit_demo_statistics,
    gradient_smoke,
    gripper_event_summary,
    json_default,
    no_demo_calibration_stats,
    smoothing_simple_killer,
    validate_manifest,
)


def test_dccg_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "DCCG-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "horizon": np.int64(HORIZON),
        "action_dimension": np.int64(ACTION_DIM),
        "feature_count": np.int64(FEATURE_COUNT),
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
    assert persisted["fixture"]["decision"] == "DCCG_STAGE_0_PASS_TO_VALIDATION_SEARCH"


def test_runner_stage0_missing_cache_records_data_failure(tmp_path: Path) -> None:
    (tmp_path / "proposal_hash.txt").write_text(PROPOSAL_HASH, encoding="utf-8")
    exit_code = dccg_stage0_main(
        [
            "--report-root",
            str(tmp_path),
            "--ccif-partial",
            str(tmp_path / "missing_partial.json"),
            "--ccif-manifest",
            str(tmp_path / "missing_manifest.json"),
        ]
    )
    assert exit_code == 0
    result = json.loads((tmp_path / "stage_0_result.json").read_text(encoding="utf-8"))
    manifest = json.loads((tmp_path / "stage_0_manifest.json").read_text(encoding="utf-8"))
    partial = json.loads((tmp_path / "stage_0_partial.json").read_text(encoding="utf-8"))
    status = json.loads((tmp_path / "stage_0_status.json").read_text(encoding="utf-8"))
    assert result["final_decision"] == "DCCG_STAGE_0_DATA_FAILURE"
    assert result["valid_scientific_result"] is False
    assert result["completed_model_row_count"] == 0
    assert result["planned_model_row_count"] == 0
    assert result["exception_count"] == 0
    assert result["manifest_summary"]["key_sets_equal"] is True
    assert result["cache_coverage"]["matching_frozen_dccg_rows"] == 0
    assert manifest["rows"] == []
    assert partial["rows"] == []
    assert status["state"] == "completed"
    assert (tmp_path / "stage_0_exit_code.txt").read_text(encoding="utf-8").strip() == "0"


def _manifest_row(split: str, demo: int, start: int, policy: str = "dccg_full") -> dict[str, object]:
    row: dict[str, object] = {
        "split": split,
        "task_suite": "libero_goal",
        "task_id": "libero_goal/task_5",
        "demo_id": demo,
        "window_start": start,
        "bin_key": "libero_goal|q1|t1|r0|g1|c1",
        "policy": policy,
        "probe_label": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
    }
    row["row_key"] = dccg_row_key(row)
    return row


def test_manifest_validation_detects_duplicate_extra_and_split_overlap() -> None:
    manifest = [
        _manifest_row("discovery", 0, 10, "smolvla_base"),
        _manifest_row("discovery", 0, 10, "dccg_full"),
        _manifest_row("validation", 30, 10, "dccg_full"),
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
    overlapped[1]["row_key"] = dccg_row_key(overlapped[1])
    overlap_summary = validate_manifest(overlapped, [{"row_key": row["row_key"]} for row in overlapped])
    assert overlap_summary["split_overlap_key_count"] == 1


def test_coherence_features_statistics_gradient_and_guidance_are_active() -> None:
    demo, base, jitter = _synthetic_coherence_problem()
    bin_keys = [deployment_bin_key(chunk, task_family="libero_goal", queue_index=12) for chunk in demo]
    features = coherence_features(demo)
    stats = fit_demo_statistics(features, bin_keys)
    global_stats = no_demo_calibration_stats(features)
    health = feature_health(features, bin_keys)
    gradient = gradient_smoke(jitter[:1], [bin_keys[0]], stats)
    guided, gate = apply_dccg_guidance(jitter[:1], gradient["gradient"], [1.0], gamma=0.10)
    identity, _ = apply_dccg_guidance(jitter[:1], gradient["gradient"], [1.0], gamma=0.0)
    smoothing = smoothing_simple_killer(jitter[:1])
    dccg_energy = coherence_energy(guided, [bin_keys[0]], stats)[0]
    jitter_energy = coherence_energy(jitter[:1], [bin_keys[0]], stats)[0]
    global_energy = coherence_energy(jitter[:1], ["global"], global_stats)[0]
    delta = action_delta_summary(jitter[:1], guided)
    grip = gripper_event_summary(jitter[:1], guided)
    validity = action_validity_summary(guided)
    assert features.shape == (len(demo), FEATURE_COUNT)
    assert health["features_noncollapsed"] is True
    assert gradient["finite_nonzero_gradients"] is True
    assert gate[0] == 1.0
    assert np.max(np.abs(identity - jitter[:1])) == 0.0
    assert dccg_energy >= 0.0
    assert jitter_energy >= 0.0
    assert global_energy >= 0.0
    assert np.max(np.abs(guided - jitter[:1])) > 0.0
    assert np.max(np.abs(guided - smoothing)) > 0.0
    assert delta["action_deltas_bounded"] is True
    assert grip["gripper_event_preservation_ok"] is True
    assert validity["action_validity_ok"] is True


def test_stage0_decision_taxonomy() -> None:
    assert classify_stage0(_healthy_inputs()) == "DCCG_STAGE_0_PASS_TO_VALIDATION_SEARCH"
    assert classify_stage0(_healthy_inputs(features_noncollapsed=False)) == "DCCG_STAGE_0_DATA_FAILURE"
    assert classify_stage0(_healthy_inputs(base_acg_headroom=0.0)) == "DCCG_STAGE_0_NO_HEADROOM"
    assert classify_stage0(_healthy_inputs(dccg_differs_from_smoothing=False)) == "DCCG_STAGE_0_DESIGN_FAILURE"
    assert classify_stage0(_healthy_inputs(finite_nonzero_gradients=False)) == "DCCG_STAGE_0_IMPLEMENTATION_FAILURE"
    assert classify_stage0(_healthy_inputs(confirmatory_records_read=1)) == "DCCG_STAGE_0_IMPLEMENTATION_FAILURE"


def _synthetic_coherence_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20263601)
    n = 8
    base = np.zeros((n, HORIZON, ACTION_DIM), dtype=np.float64)
    demo = base.copy()
    for idx in range(n):
        demo[idx, :, 0] = np.linspace(0.0, 0.03 + 0.003 * idx, HORIZON)
        demo[idx, :, 1] = 0.01 * np.sin(np.linspace(0.0, np.pi, HORIZON) + idx / 3.0)
        demo[idx, 18:24, 6] = 0.35
    jitter = demo.copy()
    jitter[:, :, 0] += rng.normal(scale=0.012, size=(n, HORIZON))
    jitter[:, 20:23, 6] = demo[:, 20:23, 6]
    return demo, base, jitter


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
        "features_noncollapsed": True,
        "bins_noncollapsed": True,
        "enough_discovery_windows": True,
        "enough_validation_windows": True,
        "validation_task_coverage_ok": True,
        "maximum_validation_task_fraction": 0.25,
        "gate_activation_fraction": 0.25,
        "base_acg_headroom": 0.05,
        "dccg_differs_from_base": True,
        "dccg_differs_from_acg": True,
        "dccg_differs_from_ablation": True,
        "dccg_differs_from_smoothing": True,
        "finite_nonzero_gradients": True,
        "exact_base_passthrough_ok": True,
        "gripper_event_preservation_ok": True,
        "normalized_action_validity_ok": True,
        "postprocessed_action_validity_ok": True,
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
