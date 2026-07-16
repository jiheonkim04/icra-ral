import json
from pathlib import Path

import numpy as np

from scripts.run_cspr_vla_stage0 import CONFIG_LABEL, POLICY_PROBE, _serializer_preflight, main as cspr_stage0_main
from tca_map.smolvla.cspr_vla import (
    ACTION_DIM,
    HORIZON,
    POLICY_ROWS,
    PROPOSAL_HASH,
    Stage0DecisionInputs,
    action_delta_summary,
    action_validity_summary,
    apply_cspr_refinement,
    base_criticality_proxy,
    canonical_json_sha256,
    classify_stage0,
    clean_retention_summary,
    construct_criticality_labels,
    critical_step_threshold_simple_killer,
    criticality_predictability_diagnostics,
    cspr_row_key,
    gradient_smoke,
    group_clip,
    json_default,
    label_health,
    residual_targets,
    uniform_refinement_ablation,
    validate_manifest,
)


def test_cspr_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "CSPR-VLA",
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
    assert persisted["fixture"]["manifest_row"]["probe_label"] == POLICY_PROBE
    assert persisted["fixture"]["config_label"] == CONFIG_LABEL
    assert persisted["fixture"]["decision"] == "CSPR_STAGE_0_PASS_TO_BOUNDED_VALIDATION"


def test_runner_stage0_missing_cache_records_data_failure(tmp_path: Path) -> None:
    (tmp_path / "proposal_hash.txt").write_text(PROPOSAL_HASH, encoding="utf-8")
    for name in [
        "researcher_proposal.md",
        "reviewer_attack.md",
        "researcher_rebuttal.md",
        "mathematical_mechanism_audit.md",
        "preregistration.md",
        "prototype_protocol.md",
    ]:
        (tmp_path / name).write_text(name, encoding="utf-8")
    exit_code = cspr_stage0_main(
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
    assert result["final_decision"] == "CSPR_STAGE_0_DATA_FAILURE"
    assert result["valid_scientific_result"] is False
    assert result["completed_model_row_count"] == 0
    assert result["planned_model_row_count"] == 0
    assert result["exception_count"] == 0
    assert result["manifest_summary"]["key_sets_equal"] is True
    assert manifest["rows"] == []
    assert partial["rows"] == []
    assert status["state"] == "completed"
    assert (tmp_path / "stage_0_exit_code.txt").read_text(encoding="utf-8").strip() == "0"


def _manifest_row(split: str, demo: int, frame: int, policy: str = "cspr_full") -> dict[str, object]:
    row: dict[str, object] = {
        "split": split,
        "task_suite": "libero_goal",
        "task_identity": "libero_goal/task_5",
        "demo_id": demo,
        "frame_index": frame,
        "source_edge_sha256": "source_a",
        "model_or_probe": policy,
        "config_label": CONFIG_LABEL,
        "probe_label": POLICY_PROBE,
    }
    row["row_key"] = cspr_row_key(row)
    return row


def test_manifest_validation_detects_duplicate_extra_and_split_overlap() -> None:
    manifest = [
        _manifest_row("discovery", 0, 10, "smolvla_base"),
        _manifest_row("discovery", 0, 10, "cspr_full"),
        _manifest_row("validation", 8, 10, "cspr_full"),
    ]
    completed = [{"row_key": row["row_key"]} for row in manifest]
    healthy = validate_manifest(manifest, completed)
    assert healthy["key_sets_equal"] is True
    duplicate = validate_manifest(manifest, completed + [completed[0]])
    assert duplicate["duplicate_partial_key_count"] == 1
    extra = validate_manifest(manifest, completed + [{"row_key": "off-manifest"}])
    assert extra["extra_partial_key_count"] == 1
    overlapped = [_manifest_row("discovery", 0, 10), _manifest_row("validation", 0, 10)]
    overlap_summary = validate_manifest(overlapped, [{"row_key": row["row_key"]} for row in overlapped])
    assert overlap_summary["split_overlap_key_count"] == 1


def test_criticality_labels_refinement_and_gradients_are_active() -> None:
    base, expert, discovery = _synthetic_problem()
    labels = construct_criticality_labels(base, expert, discovery)
    residual = residual_targets(base, expert)
    legal_score = base_criticality_proxy(expert)
    health = label_health(labels["labels"], ["libero_goal/task_5"] * len(base), list(range(len(base))))
    predict = criticality_predictability_diagnostics(
        labels["labels"],
        legal_score,
        ["libero_goal/task_5"] * len(base),
        list(range(len(base))),
    )
    cspr, gate = apply_cspr_refinement(base, residual, labels["score"], tau=labels["q_tau"])
    uniform, _ = uniform_refinement_ablation(base, residual, intervention_fraction=float(np.mean(gate)))
    simple, _ = critical_step_threshold_simple_killer(base, residual)
    identity, _ = apply_cspr_refinement(base, np.zeros_like(residual), labels["score"], tau=labels["q_tau"])
    inactive, _ = apply_cspr_refinement(base, residual, labels["score"], tau=float(np.max(labels["score"]) + 1.0))
    delta = action_delta_summary(base, cspr)
    clean = clean_retention_summary(base, identity, inactive)
    gradient = gradient_smoke(base, group_clip(residual), gate, expert, labels["labels"])
    validity = action_validity_summary(cspr)
    assert labels["criticality_score_variance_ok"] is True
    assert health["labels_noncollapsed"] is True
    assert predict["legal_score_balanced_accuracy"] >= 0.5
    assert np.max(np.abs(cspr - base)) > 0.0
    assert np.max(np.abs(cspr - uniform)) > 0.0
    assert np.isfinite(simple).all()
    assert delta["action_deltas_bounded"] is True
    assert clean["clean_retention_ok"] is True
    assert gradient["finite_nonzero_gradients"] is True
    assert validity["action_validity_ok"] is True


def test_stage0_decision_taxonomy() -> None:
    assert classify_stage0(_healthy_inputs()) == "CSPR_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert classify_stage0(_healthy_inputs(labels_noncollapsed=False)) == "CSPR_STAGE_0_DATA_FAILURE"
    assert classify_stage0(_healthy_inputs(base_residual_headroom=0.0)) == "CSPR_STAGE_0_NO_USABLE_HEADROOM"
    assert classify_stage0(_healthy_inputs(cspr_beats_comparators=False)) == "CSPR_STAGE_0_DESIGN_FAILURE"
    assert classify_stage0(_healthy_inputs(finite_nonzero_gradients=False)) == "CSPR_STAGE_0_IMPLEMENTATION_FAILURE"
    assert (
        classify_stage0(_healthy_inputs(cspr_beats_comparators=False, weighted_gradient_norm_ratio_max=129.0))
        == "CSPR_STAGE_0_IMPLEMENTATION_FAILURE"
    )
    assert classify_stage0(_healthy_inputs(confirmatory_records_read=1)) == "CSPR_STAGE_0_IMPLEMENTATION_FAILURE"


def _synthetic_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20263701)
    n = 8
    base = rng.normal(scale=0.003, size=(n, HORIZON, ACTION_DIM)).astype(np.float64)
    expert = base.copy()
    for index in range(n):
        expert[index, 8:18, 0] += 0.050 + 0.002 * index
        expert[index, 16:24, 3] -= 0.040
        expert[index, 22:25, 6] += 0.35 if index % 2 == 0 else -0.35
    discovery = np.asarray([True] * 6 + [False] * 2, dtype=bool)
    return base, expert, discovery


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
        "feature_caches_valid": True,
        "labels_noncollapsed": True,
        "criticality_score_variance_ok": True,
        "enough_discovery_rows": True,
        "enough_validation_rows": True,
        "validation_task_coverage_ok": True,
        "maximum_validation_task_fraction": 0.25,
        "validation_positive_count": 16,
        "validation_negative_count": 128,
        "validation_positive_fraction": 0.10,
        "largest_positive_task_fraction": 0.50,
        "criticality_predictability_margin": 0.03,
        "base_residual_headroom": 0.01,
        "dysl_residual_headroom": 0.01,
        "simple_killer_residual_headroom": 0.01,
        "cspr_beats_comparators": True,
        "cspr_differs_from_base": True,
        "cspr_differs_from_ablation": True,
        "simple_killer_explains_gain": False,
        "identity_reload_error": 0.0,
        "finite_nonzero_gradients": True,
        "frozen_base_gradient_count": 0,
        "weighted_gradient_norm_ratio_max": 10.0,
        "intervention_fraction": 0.10,
        "action_deltas_bounded": True,
        "action_validity_ok": True,
        "clean_retention_ok": True,
        "reward_read_count": 0,
        "success_read_count": 0,
        "done_read_count": 0,
        "confirmatory_records_read": 0,
        "simulator_load_count": 0,
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "validation_search_happened": False,
        "exception_count": 0,
    }
    values.update(overrides)
    return Stage0DecisionInputs(**values)
