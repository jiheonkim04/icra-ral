import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
REPORTS = REPO_ROOT / "reports"


def test_eac_stage_0_audit_passes_without_rollout_or_tuning() -> None:
    report = json.loads((REPORTS / "eac_vla" / "stage_0_audit.json").read_text(encoding="utf-8"))

    assert report["final_decision"] == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    assert report["closed_loop_experiment_happened"] is False
    assert report["training_happened"] is False
    assert report["validation_search_happened"] is False
    assert report["confirmatory_test_tuning_happened"] is False
    assert report["hard_stop_reasons"] == []
    assert report["scoreable_validation_records"] == 2000
    assert report["validation_unique_frames"] == 400
    assert report["reserved_records_not_used_for_tuning"] == 6000

    split = report["split_manifest"]
    assert split["confirmatory_records_used_for_tuning"] is False
    assert split["validation_reserved_frame_overlap"] == 0
    assert split["validation_reserved_sample_overlap"] == 0

    queue = report["queue_surface_manifest"]
    assert queue["queue_helper_present"] is True
    assert queue["chunk_shape_ok"] is True
    assert queue["full_chunk_values_available_in_artifact"] is False
    assert queue["runtime_full_chunk_check_required_before_validation_search"] is True

    dispersion = report["dispersion_manifest"]
    assert dispersion["first_two_dispersion_summary"]["p95"] == 0.0007983036317792467
    assert dispersion["first_two_dispersion_summary"]["nonzero_fraction"] == 1.0
    assert dispersion["commitment_counts"] == {"2": 136, "8": 132, "50": 132}
    assert dispersion["max_commitment_share"] == 0.34

    passthrough = report["action_value_passthrough_summary"]
    assert passthrough["max"] <= 1e-6
