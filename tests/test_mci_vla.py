import json
from pathlib import Path

import numpy as np

from scripts.run_mci_vla_stage0 import CONFIG_LABEL, POLICY_PROBE, _serializer_preflight, main as mci_stage0_main
from tca_map.smolvla.mci_vla import (
    ACTION_DIM,
    HORIZON,
    POLICY_ROWS,
    PROPOSAL_HASH,
    PROPRIO_DIM,
    TRANSFORMATION_FAMILIES,
    VISUAL_FEATURE_DIM,
    Stage0DecisionInputs,
    action_delta_summary,
    action_validity_summary,
    apply_mci_adapter,
    augmentation_only_lora_killer,
    canonical_json_sha256,
    classify_stage0,
    clean_retention_summary,
    consistency_code,
    consistency_observability_diagnostics,
    identity_passthrough,
    json_default,
    mci_no_consistency_code_ablation,
    mci_row_key,
    objective_gradient_smoke,
    representation_health,
    residual_targets,
    rovla_multiconsistency_proxy,
    transformed_inputs,
    validate_manifest,
)


def test_mci_serializer_roundtrip_and_hash_are_stable(tmp_path: Path) -> None:
    fixture = {
        "method": "MCI-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "horizon": np.int64(HORIZON),
        "action_dimension": np.int64(ACTION_DIM),
        "policy_rows": POLICY_ROWS,
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
    assert persisted["fixture"]["decision"] == "MCI_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert persisted["fixture"]["no_deterministic_action_kl"] is True


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
    exit_code = mci_stage0_main(
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
    assert result["final_decision"] == "MCI_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    assert result["valid_scientific_result"] is False
    assert result["completed_model_row_count"] == 0
    assert result["planned_model_row_count"] == 0
    assert result["exception_count"] == 0
    assert result["manifest_summary"]["key_sets_equal"] is True
    assert manifest["rows"] == []
    assert partial["rows"] == []
    assert status["state"] == "completed"
    assert (tmp_path / "stage_0_exit_code.txt").read_text(encoding="utf-8").strip() == "0"


def _manifest_row(split: str, demo: int, frame: int, family: str = "instruction", policy: str = "mci_full") -> dict[str, object]:
    row: dict[str, object] = {
        "split": split,
        "task_suite": "libero_goal",
        "task_identity": "libero_goal/task_5",
        "demo_id": demo,
        "window_start": frame,
        "transform_family": family,
        "policy": policy,
        "probe_label": POLICY_PROBE,
        "config_label": CONFIG_LABEL,
    }
    row["row_key"] = mci_row_key(row)
    return row


def test_manifest_validation_detects_duplicate_extra_and_split_overlap() -> None:
    manifest = [
        _manifest_row("discovery", 0, 10, "instruction", "smolvla_base"),
        _manifest_row("discovery", 0, 10, "observation_proprioception", "mci_full"),
        _manifest_row("validation", 8, 10, "instruction", "mci_full"),
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


def test_consistency_code_adapter_comparators_and_gradients_are_active() -> None:
    base, expert, features, proprio, tasks = _synthetic_problem()
    code = consistency_code(features, proprio, tasks, base)
    tf, tp, tt, tb, metadata = transformed_inputs(features, proprio, tasks, base, family="observation_proprioception")
    transformed_code = consistency_code(tf, tp, tt, tb)
    residual = residual_targets(base, expert)
    mci, gate, _ = apply_mci_adapter(base, residual, code)
    ablation, _ = mci_no_consistency_code_ablation(base, residual, intervention_fraction=float(np.mean(gate > 0.5)))
    rovla = rovla_multiconsistency_proxy(base)
    killer = augmentation_only_lora_killer(base, residual)
    identity, identity_gate = identity_passthrough(base)
    clean = clean_retention_summary(base, identity, base)
    delta = action_delta_summary(base, mci)
    validity = action_validity_summary(mci)
    rep = representation_health(code)
    gradient = objective_gradient_smoke(base, expert, tb, code, transformed_code, mci, gate)
    observability = _observability_smoke(features, proprio, tasks, base)
    assert metadata["uses_future_or_privileged_input"] is False
    assert code.shape == (len(base), 16)
    assert transformed_code.shape == code.shape
    assert np.max(np.abs(mci - base)) > 0.0
    assert np.max(np.abs(mci - rovla)) > 0.0
    assert np.max(np.abs(mci - ablation)) > 0.0
    assert np.max(np.abs(mci - killer)) > 0.0
    assert np.max(np.abs(identity - base)) == 0.0
    assert identity_gate.shape == base.shape
    assert clean["clean_retention_ok"] is True
    assert 0.02 <= float(np.mean(gate > 0.5)) <= 0.80
    assert delta["action_deltas_bounded"] is True
    assert validity["action_validity_ok"] is True
    assert rep["representation_noncollapsed"] is True
    assert gradient["finite_nonzero_gradients"] is True
    assert gradient["frozen_base_gradient_count"] == 0
    assert observability["consistency_signal_predictable"] is True


def test_stage0_decision_taxonomy() -> None:
    assert classify_stage0(_healthy_inputs()) == "MCI_STAGE_0_PASS_TO_BOUNDED_VALIDATION"
    assert classify_stage0(_healthy_inputs(transformations_noncollapsed=False)) == "MCI_STAGE_0_DATA_OR_SUPERVISION_FAILURE"
    assert classify_stage0(_healthy_inputs(rovla_residual_headroom=0.0)) == "MCI_STAGE_0_NO_HEADROOM"
    assert classify_stage0(_healthy_inputs(mci_beats_comparators=False)) == "MCI_STAGE_0_DESIGN_FAILURE"
    assert classify_stage0(_healthy_inputs(finite_nonzero_gradients=False)) == "MCI_STAGE_0_IMPLEMENTATION_FAILURE"
    assert classify_stage0(_healthy_inputs(confirmatory_records_read=1)) == "MCI_STAGE_0_IMPLEMENTATION_FAILURE"


def _synthetic_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str]]:
    rng = np.random.default_rng(20263801)
    n = 16
    base = rng.normal(scale=0.003, size=(n, HORIZON, ACTION_DIM)).astype(np.float64)
    expert = base.copy()
    for index in range(n):
        expert[index, 8:18, 0] += 0.045 + 0.001 * index
        expert[index, 18:26, 4] -= 0.035
        expert[index, 24:28, 6] += 0.25 if index % 2 == 0 else -0.25
    features = rng.normal(size=(n, VISUAL_FEATURE_DIM)).astype(np.float64)
    proprio = rng.normal(scale=0.05, size=(n, PROPRIO_DIM)).astype(np.float64)
    tasks = ["libero_goal/task_5" if index < n // 2 else "libero_10/task_5" for index in range(n)]
    return base, expert, features, proprio, tasks


def _observability_smoke(features: np.ndarray, proprio: np.ndarray, tasks: list[str], base: np.ndarray) -> dict[str, object]:
    code = consistency_code(features, proprio, tasks, base)
    scores: list[float] = []
    targets: list[int] = []
    obs_tasks: list[str] = []
    frames: list[int] = []
    magnitudes: list[float] = []
    families: list[str] = []
    for family in TRANSFORMATION_FAMILIES:
        tf, tp, tt, tb, _ = transformed_inputs(features, proprio, tasks, base, family=family)
        transformed_code = consistency_code(tf, tp, tt, tb)
        distances = np.linalg.norm(code - transformed_code, axis=1)
        negative = np.linalg.norm(transformed_code - np.roll(code, 1, axis=0), axis=1)
        for index in range(len(code)):
            scores.extend([float(-distances[index]), float(-negative[index])])
            targets.extend([1, 0])
            obs_tasks.extend([tasks[index], tasks[index]])
            frames.extend([index, index])
            magnitudes.extend([float(np.mean(np.abs(base[index]))), float(np.mean(np.abs(base[index])))])
            families.extend([family, family])
    return consistency_observability_diagnostics(scores, targets, obs_tasks, frames, magnitudes, families)


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
        "transformations_noncollapsed": True,
        "enough_discovery_rows": True,
        "enough_validation_rows": True,
        "validation_task_coverage_ok": True,
        "maximum_validation_task_fraction": 0.25,
        "minimum_validation_pairs_per_family": 32,
        "positive_contrast_count": 32,
        "negative_contrast_count": 32,
        "representation_dims_fraction_above_floor": 0.85,
        "consistency_predictability_margin": 0.03,
        "base_transformed_pair_headroom": 0.01,
        "rovla_residual_headroom": 0.01,
        "augmentation_residual_headroom": 0.01,
        "mci_beats_comparators": True,
        "mci_differs_from_base": True,
        "mci_differs_from_rovla": True,
        "mci_differs_from_ablation": True,
        "mci_differs_from_augmentation_only_lora": True,
        "exact_base_passthrough_ok": True,
        "identity_reload_error": 0.0,
        "finite_nonzero_gradients": True,
        "frozen_base_gradient_count": 0,
        "weighted_gradient_norm_ratio_max": 10.0,
        "intervention_fraction": 0.30,
        "action_deltas_bounded": True,
        "action_validity_rate": 1.0,
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
