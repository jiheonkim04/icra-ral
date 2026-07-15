import json
from pathlib import Path

import pytest

from tca_map.smolvla.rar_vla import FORBIDDEN_INFERENCE_KEYS, LEGAL_INFERENCE_FEATURES, PROPOSAL_HASH


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "rar_vla"
RAR_PROPOSAL_HASH = "723C16C3885A974E2CA12D90BC36267FA6E86827AC9D2A1E0E0E475E16FB0E56"


def _read_json(name: str) -> dict:
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def test_rar_stage_0_records_fixed_design_failure() -> None:
    report = _read_json("development_audit.json")

    assert report["method"] == "RAR-VLA"
    assert report["proposal_hash"] == RAR_PROPOSAL_HASH == PROPOSAL_HASH
    assert report["final_decision"] == "DESIGN_FAILURE"
    assert report["stage_0_completed"] is True
    assert report["stage_0_passed"] is False
    assert report["stage_0_failure_class"] == "DESIGN_FAILURE"
    assert report["hard_stop_reasons"] == ["residual predictability margin below minimum: -0.038376"]

    assert report["training_happened"] is False
    assert report["validation_search_happened"] is False
    assert report["closed_loop_experiment_happened"] is False
    assert report["confirmatory_test_tuning_happened"] is False

    assert report["scoreable_development_records"] == 2800
    assert report["train_records"] == 1200
    assert report["validation_records"] == 400
    assert report["reserved_records_not_used"] == 1200
    assert report["selected_task_count"] == 40
    assert report["duplicate_sample_keys"] == 0
    assert report["duplicate_frame_keys"] == 0

    observability = report["residual_observability_summary"]
    assert observability["best_trivial_baseline"] == "zero_residual"
    assert observability["residual_predictability_margin"] == pytest.approx(-0.03837609884238533)
    assert observability["best_trivial_rmse"] == pytest.approx(0.16559729909097304)
    assert observability["full_probe_rmse"] == pytest.approx(0.1719540079557317)
    assert observability["best_trivial_rmse"] < observability["full_probe_rmse"]

    assert report["residual_headroom_l2_validation"] == pytest.approx(0.08630366897708504)
    assert report["initial_action_delta_p95"] == 0.0
    assert report["base_action_validity"] == 1.0
    assert report["gradient_audit"]["valid"] is True
    assert report["gradient_audit"]["residual_head_gradient_norm"] > 0.0
    assert report["gradient_audit"]["gate_surrogate_gradient_norm"] > 0.0


def test_rar_source_history_and_split_manifests_preserve_no_leakage_boundary() -> None:
    source_gate = _read_json("source_gate_manifest.json")
    split_manifest = _read_json("split_manifest.json")
    history_manifest = _read_json("history_feature_manifest.json")

    assert set(LEGAL_INFERENCE_FEATURES).issubset(set(source_gate["legal_inference_features"]))
    assert set(FORBIDDEN_INFERENCE_KEYS).issubset(set(source_gate["forbidden_inference_keys"]))
    assert source_gate["source_gate_passed"] is True
    assert source_gate["future_actions_used_at_inference"] is False
    assert source_gate["cala_latents_used_at_inference"] is False
    assert source_gate["previous_actions_are_causal_only"] is True
    assert source_gate["privileged_object_pose_available_as_dataset_feature"] is False
    assert source_gate["used_inference_features_for_stage_0_probe"] == [
        "observation.state",
        "base_action",
        "previous_base_actions",
        "state_delta",
        "language_or_task_instruction_proxy",
    ]

    assert split_manifest["duplicate_sample_keys"] == 0
    assert split_manifest["duplicate_frame_keys"] == 0
    assert split_manifest["split_overlap"] == {
        "train_reserved": 0,
        "train_validation": 0,
        "validation_reserved": 0,
    }

    assert history_manifest["history_horizon"] == 8
    assert history_manifest["history_source"] == "previous_base_actions_and_current_state_only"
    assert history_manifest["future_actions_used_at_inference"] is False
    assert history_manifest["cala_latents_used_at_inference"] is False
    assert history_manifest["reanchor_feature"] == "base_action_diff_from_previous"
    assert history_manifest["train_residual_summary"]["residual_variance_nonzero_dims"] == 7
    assert history_manifest["validation_residual_summary"]["residual_variance_nonzero_dims"] == 7
    assert history_manifest["discontinuity_diagnostics"]["diagnostic_type"] == "frame_local_base_difference_proxy"
