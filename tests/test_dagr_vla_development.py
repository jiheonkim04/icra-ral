import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
DAGR = REPO_ROOT / "reports" / "dagr_vla"


def test_dagr_stage_0_audit_passes_without_confirmatory_use() -> None:
    audit = json.loads((DAGR / "development_audit.json").read_text(encoding="utf-8"))

    assert audit["final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert audit["closed_loop_experiment_happened"] is False
    assert audit["training_happened"] is False
    assert audit["confirmatory_test_tuning_happened"] is False
    assert audit["scoreable_development_records"] == 1600
    assert audit["reserved_records_not_used"] == 1200
    assert audit["duplicate_sample_keys"] == 0
    assert audit["duplicate_frame_keys"] == 0
    assert audit["split_overlap"] == {"train_reserved": 0, "train_validation": 0, "validation_reserved": 0}
    assert audit["hard_stop_reasons"] == []
    assert audit["base_action_validity"] == 1.0
    assert audit["validation_any_route_fraction"] == 0.865
    assert audit["route_probe_summary"]["translation"]["accuracy_margin"] >= 0.02
    assert audit["route_probe_summary"]["rotation"]["accuracy_margin"] >= 0.02
    assert audit["route_probe_summary"]["gripper"]["accuracy_margin"] >= 0.02


def test_dagr_validation_search_freezes_selected_config() -> None:
    validation = json.loads((DAGR / "validation_search.json").read_text(encoding="utf-8"))
    selected = validation["selected_config"]

    assert validation["final_decision"] == "VALIDATION_SEARCH_SELECT_CONFIG_REQUIRES_ADAPTER_TRAINING"
    assert validation["closed_loop_experiment_happened"] is False
    assert validation["confirmatory_test_tuning_happened"] is False
    assert validation["tried_config_count"] == 6
    assert selected["config_id"] == "dagr_a020_route_mlp"
    assert selected["residual_alpha"] == 0.2
    assert selected["route_architecture"] == "mlp"
    assert selected["score_terms"]["total"] == 0.8571740870493018
    assert selected["initial_delta_p95"] == 0.0
    assert selected["checkpoint_reload_max_abs_diff"] == 0.0
    assert selected["validation_metrics"]["action_validity"] == 1.0
    assert selected["hard_stop_reasons"] == []

    for item in validation["tried_configs"]:
        assert (REPO_ROOT / item["checkpoint_path"]).exists()

