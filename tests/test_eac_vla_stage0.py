import json
from collections import Counter
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


def test_eac_runtime_queue_check_preserves_full_chunk_prefixes() -> None:
    report = json.loads((REPORTS / "eac_vla" / "runtime_queue_check.json").read_text(encoding="utf-8"))

    assert report["final_decision"] == "EAC_RUNTIME_QUEUE_CHECK_PASS_VALIDATION_SEARCH_ALLOWED"
    assert report["closed_loop_experiment_happened"] is False
    assert report["training_happened"] is False
    assert report["validation_search_happened"] is False
    assert report["confirmatory_test_tuning_happened"] is False
    assert report["hard_stop_reasons"] == []

    chunk = report["chunk_check"]
    assert chunk["raw_action_chunk_shape"] == [1, 50, 7]
    assert chunk["postprocessed_chunk_shape"] == [50, 7]
    assert chunk["postprocessed_chunk_finite"] is True
    assert chunk["select_action_matches_predict_chunk_first"] is True
    assert chunk["select_action_vs_chunk0_max_abs_diff"] == 0.0

    queue = report["queue_check"]
    assert queue["queue_owner_present"] is True
    assert queue["queue_len_before_select_action"] == 0
    assert queue["queue_len_after_select_action"] == 49

    prefix = report["prefix_preservation"]
    assert prefix["all_prefixes_value_preserving"] is True
    assert prefix["commitment_lengths"] == [1, 2, 4, 8, 16, 50]
    assert prefix["max_prefix_abs_diff"] == 0.0
    assert prefix["max_queue_pop_abs_diff"] == 0.0
    assert all(not check["action_values_modified"] for check in prefix["checks"])


def test_eac_validation_search_selects_frozen_config_without_test_tuning() -> None:
    report = json.loads((REPORTS / "eac_vla" / "validation_search.json").read_text(encoding="utf-8"))
    selected = json.loads((REPORTS / "eac_vla" / "selected_config.json").read_text(encoding="utf-8"))

    assert report["final_decision"] == "EAC_VALIDATION_SEARCH_SELECT_CONFIG_STAGE_A_MANIFEST_READY"
    assert report["closed_loop_experiment_happened"] is False
    assert report["training_happened"] is False
    assert report["validation_search_happened"] is True
    assert report["confirmatory_test_tuning_happened"] is False
    assert report["confirmatory_records_used_for_tuning"] is False
    assert report["tried_config_count"] == 6
    assert report["hard_stop_reasons"] == []
    assert report["selected_config_id"] == "eac_q33_aggressive_1_4_50"
    assert selected["config_id"] == report["selected_config_id"]
    assert selected["validation_score"] == 0.7530415186081504
    assert selected["commitment_map"] == {"short": 1, "medium": 4, "long": 50}
    assert selected["commitment_counts"] == {"1": 132, "4": 136, "50": 132}
    assert selected["score_components"]["clean_action_value_passthrough"] == 1.0
    assert selected["score_components"]["runtime_action_validity"] == 1.0


def test_eac_stage_a_manifest_freezes_matched_five_policy_plan() -> None:
    manifest = json.loads((REPORTS / "eac_vla" / "stage_a_manifest.json").read_text(encoding="utf-8"))

    assert manifest["final_decision"] == "EAC_STAGE_A_PLAN_FROZEN_PREFLIGHT_PENDING"
    assert manifest["closed_loop_experiment_happened"] is False
    assert manifest["training_happened"] is False
    assert manifest["validation_search_happened"] is False
    assert manifest["confirmatory_test_tuning_happened"] is False
    assert manifest["config_id"] == "eac_q33_aggressive_1_4_50"
    assert manifest["canonical_payload_sha256"] == "63E96D0629F3D34E4801EB1084D094CB287EC4F2F2FCD96373981787EDA9954C"
    assert manifest["planned_episode_count"] == 50
    assert manifest["paired_cases_per_policy"] == 10
    assert manifest["reset_seeds"] == [20261211, 20261212]
    assert manifest["policy_order"] == [
        "frozen_smolvla_fixed_queue",
        "aac_entropy_proxy",
        "eac_full",
        "eac_no_calibration_no_hysteresis_ablation",
        "fixed_short_replan_baseline",
    ]
    assert manifest["errors"] == []
    assert manifest["confirmatory_test_identities_used_for_training_or_validation"] is False
    assert manifest["policy_order_affects_env_initialization"] is False
    assert manifest["fixed_task_balanced_allocation"] is True
    assert manifest["no_post_hoc_task_or_reset_selection"] is True

    episodes = manifest["episodes"]
    assert len({episode["episode_id"] for episode in episodes}) == 50
    assert Counter(episode["policy"] for episode in episodes) == {
        "frozen_smolvla_fixed_queue": 10,
        "aac_entropy_proxy": 10,
        "eac_full": 10,
        "eac_no_calibration_no_hysteresis_ablation": 10,
        "fixed_short_replan_baseline": 10,
    }
    pair_sets = {
        policy: {episode["pair_id"] for episode in episodes if episode["policy"] == policy}
        for policy in manifest["policy_order"]
    }
    assert len({tuple(sorted(pair_ids)) for pair_ids in pair_sets.values()}) == 1

    labels = {identity["policy"]: identity["proxy_or_reproduction_label"] for identity in manifest["policy_identities"]}
    assert labels["aac_entropy_proxy"] == "faithful transparent local proxy, not an official AAC reproduction"


def test_eac_stage_a_preflight_preserves_action_values_before_rollout() -> None:
    report = json.loads((REPORTS / "eac_vla" / "stage_a_preflight.json").read_text(encoding="utf-8"))

    assert report["final_decision"] == "EAC_STAGE_A_PREFLIGHT_PASS_RUNNER_IMPLEMENTATION_PENDING"
    assert report["closed_loop_experiment_happened"] is False
    assert report["training_happened"] is False
    assert report["validation_search_happened"] is False
    assert report["confirmatory_test_tuning_happened"] is False
    assert report["planned_episode_count"] == 50
    assert report["paired_cases_per_policy"] == 10
    assert report["policy_count"] == 5
    assert report["checkpoint_policy_count"] == 0
    assert report["cuda_ok"] is True
    assert report["cuda_device_name"] == "NVIDIA GeForce RTX 5080"
    assert report["policy_output_shape"] == [50, 7]
    assert report["policy_output_shape_ok"] is True
    assert report["policy_output_finite"] is True
    assert report["all_policy_prefixes_value_preserving"] is True
    assert report["no_accidental_checkpoint_reuse"] is True
    assert report["old_custom_libero_7d_route_used"] is False
    assert report["errors"] == []
    assert all(not record["action_values_modified"] for record in report["preflight_records"])
    assert {record["policy"] for record in report["preflight_records"]} == set(report["policy_order"])


def test_eac_stage_a_runner_validation_authorizes_frozen_rollout() -> None:
    report = json.loads((REPORTS / "eac_vla" / "stage_a_runner_validation.json").read_text(encoding="utf-8"))

    assert report["final_decision"] == "EAC_STAGE_A_RUNNER_VALIDATED_READY_FOR_ROLLOUT"
    assert report["closed_loop_experiment_happened"] is False
    assert report["training_happened"] is False
    assert report["validation_search_happened"] is False
    assert report["confirmatory_test_tuning_happened"] is False
    assert report["planned_episode_count"] == 50
    assert report["policy_count"] == 5
    assert report["runtime_samples_for_dynamic_schedulers"] == 2
    assert report["runtime_calibration"]["validation_frame_count"] == 400
    assert report["runtime_calibration"]["eac_quantile_thresholds"]["low"] == 0.1383995528485192
    assert report["runtime_calibration"]["eac_quantile_thresholds"]["high"] == 0.3085939397201893
    assert report["all_policy_prefixes_value_preserving"] is True
    assert report["any_action_values_modified"] is False
    assert report["stage_a_rollout_allowed"] is True
    assert report["errors"] == []
    assert {record["policy"] for record in report["runner_validation_records"]} == set(report["policy_order"])
    assert all(not record["action_values_modified"] for record in report["runner_validation_records"])


def test_eac_stage_a_result_requires_stage_b_without_retuning() -> None:
    report = json.loads((REPORTS / "eac_vla" / "stage_a_result.json").read_text(encoding="utf-8"))
    summary = report["summary"]["policy_summary"]

    assert report["final_decision"] == "EAC_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"
    assert report["closed_loop_experiment_happened"] is True
    assert report["training_happened"] is False
    assert report["validation_search_happened"] is False
    assert report["confirmatory_test_tuning_happened"] is False
    assert report["scaleup"]["completed_episode_count"] == 50
    assert report["scaleup"]["infrastructure_failure_count"] == 0
    assert report["stage_b_required"] is True
    assert report["valid_current_formulation_kill"] is False
    assert summary["frozen_smolvla_fixed_queue"]["successes"] == 7
    assert summary["aac_entropy_proxy"]["successes"] == 9
    assert summary["eac_full"]["successes"] == 8
    assert summary["eac_no_calibration_no_hysteresis_ablation"]["successes"] == 7
    assert summary["fixed_short_replan_baseline"]["successes"] == 7
    assert summary["eac_full"]["action_values_modified"] is False
    assert summary["eac_full"]["action_validity_all_finite"] is True
    assert summary["eac_full"]["commitment_counts"] == {"1": 150, "4": 25, "50": 33}
    assert report["summary"]["paired_vs_eac_full"]["aac_entropy_proxy"]["paired_delta_eac_minus_policy"] == -0.1
    assert report["summary"]["paired_vs_eac_full"]["frozen_smolvla_fixed_queue"]["paired_delta_eac_minus_policy"] == 0.1
