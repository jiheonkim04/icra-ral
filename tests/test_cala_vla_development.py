import json
from pathlib import Path

import pytest

from tca_map.smolvla.cala_vla import FORBIDDEN_INFERENCE_KEYS, LEGAL_INFERENCE_FEATURES, PROPOSAL_HASH


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = REPO_ROOT / "reports" / "cala_vla"
CALA_PROPOSAL_HASH = "5B3933C9C0FD5AE5F07FDB0CEC447B48040238FB6D872D97E545E3D93E257E76"


def _read_json(name: str) -> dict:
    return json.loads((REPORT_DIR / name).read_text(encoding="utf-8"))


def test_cala_stage_0_records_fixed_design_failure() -> None:
    report = _read_json("development_audit.json")

    assert report["method"] == "CALA-VLA"
    assert report["proposal_hash"] == CALA_PROPOSAL_HASH == PROPOSAL_HASH
    assert report["final_decision"] == "DESIGN_FAILURE"
    assert report["stage_0_completed"] is True
    assert report["stage_0_passed"] is False
    assert report["stage_0_failure_class"] == "DESIGN_FAILURE"
    assert report["hard_stop_reasons"] == ["latent predictability margin below minimum: -0.011718"]

    assert report["training_happened"] is False
    assert report["validation_search_happened"] is False
    assert report["closed_loop_experiment_happened"] is False
    assert report["confirmatory_test_tuning_happened"] is False

    assert report["scoreable_development_records"] == 2800
    assert report["train_records"] == 1200
    assert report["validation_records"] == 400
    assert report["reserved_records_not_used"] == 1200
    assert report["duplicate_sample_keys"] == 0
    assert report["duplicate_frame_keys"] == 0

    predictability = report["latent_predictability_summary"]
    assert predictability["best_trivial_baseline"] == "action_history_only"
    assert predictability["accuracy_margin"] == pytest.approx(-0.01171824382857035)
    assert predictability["best_trivial_rmse"] < predictability["full_probe_rmse"]


def test_cala_source_and_split_manifests_preserve_no_leakage_boundary() -> None:
    source_gate = _read_json("source_gate_manifest.json")
    split_manifest = _read_json("split_manifest.json")
    latent_manifest = _read_json("latent_label_manifest.json")

    assert set(LEGAL_INFERENCE_FEATURES).issubset(set(source_gate["legal_inference_features"]))
    assert set(FORBIDDEN_INFERENCE_KEYS).issubset(set(source_gate["forbidden_inference_keys"]))
    assert source_gate["source_gate_passed"] is True
    assert source_gate["future_action_segments_used_at_inference"] is False
    assert source_gate["latent_labels_used_at_inference"] is False
    assert source_gate["future_action_segments_used_for_training_only"] is True
    assert source_gate["privileged_object_pose_available_as_dataset_feature"] is False

    assert split_manifest["duplicate_sample_keys"] == 0
    assert split_manifest["duplicate_frame_keys"] == 0
    assert split_manifest["split_overlap"] == {
        "train_reserved": 0,
        "train_validation": 0,
        "validation_reserved": 0,
    }

    assert latent_manifest["inference_uses_future_action_segment"] is False
    assert latent_manifest["latent_encoder"] == "oat_lite_summary_mean_first_last_diff_std"
    assert latent_manifest["latent_horizon"] == 16
    assert latent_manifest["train_label_summary"]["latent_variance_nonzero_dims"] == 35
    assert latent_manifest["validation_label_summary"]["latent_variance_nonzero_dims"] == 35
