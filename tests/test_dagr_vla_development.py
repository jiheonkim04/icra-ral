import json
from pathlib import Path

from tca_map.smolvla.dagr_vla_stage_a import (
    STAGE_A_POLICY_ORDER,
    STAGE_B_RESET_SEEDS,
    _stage_a_decision,
    _task_index_for_task,
    _task_index_map_from_artifact,
    validate_stage_b_manifest,
    validate_stage_a_manifest,
)


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


def test_dagr_policy_identities_are_disk_reloadable() -> None:
    manifest = json.loads((DAGR / "policy_checkpoint_manifest.json").read_text(encoding="utf-8"))

    assert manifest["final_decision"] == "DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY"
    assert manifest["stage_a_allowed"] is True
    assert manifest["closed_loop_experiment_happened"] is False
    assert manifest["confirmatory_test_identities_used"] is False
    assert manifest["policy_identities"] == [
        "frozen_smolvla",
        "dam_static_component_proxy",
        "dagr_full",
        "dagr_no_dynamic_route_ablation",
        "gripper_transition_heuristic",
    ]
    variants = {row["variant"]: row for row in manifest["variant_results"]}
    assert set(variants) == {"dagr_full", "dam_static_component_proxy", "dagr_no_dynamic_route_ablation"}
    for row in variants.values():
        assert row["final_decision"] == "DAGR_POLICY_CHECKPOINT_VERIFIED"
        assert row["disk_reload"] is True
        assert row["initial_delta_p95"] == 0.0
        assert row["validation"]["action_validity"] == 1.0
        for filename in row["required_files"]:
            assert (REPO_ROOT / row["checkpoint_path"] / filename).exists()

    assert (REPO_ROOT / manifest["heuristic"]["checkpoint_path"] / "heuristic_config.json").exists()


def test_dagr_stage_a_manifest_is_frozen_and_matched() -> None:
    manifest = json.loads((DAGR / "stage_a_manifest.json").read_text(encoding="utf-8"))

    validate_stage_a_manifest(manifest)
    assert manifest["final_decision"] == "DAGR_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert manifest["closed_loop_experiment_happened"] is False
    assert manifest["confirmatory_test_tuning_happened"] is False
    assert manifest["planned_episode_count"] == 50
    assert manifest["stage_a_reset_seeds"] == [20261205, 20261206]
    assert manifest["stage_a_pair_count_per_policy"] == 10
    assert manifest["canonical_payload_sha256"] == "8379E47D3C3C73E21ADDD285491750E7406B8389578C0003278E5E187EA27E7B"
    assert manifest["policy_order"] == [
        "frozen_smolvla",
        "dam_static_component_proxy",
        "dagr_full",
        "dagr_no_dynamic_route_ablation",
        "gripper_transition_heuristic",
    ]
    keys = {(row["policy"], row["suite"], row["task_id"], row["reset_seed"]) for row in manifest["episodes"]}
    assert len(keys) == len(manifest["episodes"])
    pair_sets = {
        policy: {
            (row["suite"], row["task_id"], row["reset_seed"])
            for row in manifest["episodes"]
            if row["policy"] == policy
        }
        for policy in manifest["policy_order"]
    }
    assert len({tuple(sorted(values)) for values in pair_sets.values()}) == 1


def test_dagr_stage_a_task_indices_match_stable_artifact() -> None:
    manifest = json.loads((DAGR / "stage_a_manifest.json").read_text(encoding="utf-8"))
    task_index_map = _task_index_map_from_artifact(REPO_ROOT / "reports" / "official_smolvla_stable_prediction_artifact.json")

    mapped = {
        f"{task['suite']}/task_{task['task_id']}": _task_index_for_task(task, task_index_map)
        for task in manifest["tasks"]
    }

    assert mapped == {
        "libero_spatial/task_0": 34,
        "libero_spatial/task_8": 36,
        "libero_object/task_6": 27,
        "libero_goal/task_4": 18,
        "libero_10/task_2": 3,
    }


def test_dagr_stage_b_manifest_is_frozen_all_task_expansion() -> None:
    manifest = json.loads((DAGR / "stage_b_manifest.json").read_text(encoding="utf-8"))

    validate_stage_b_manifest(manifest)
    assert manifest["final_decision"] == "DAGR_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT"
    assert manifest["closed_loop_experiment_happened"] is False
    assert manifest["confirmatory_test_tuning_happened"] is False
    assert manifest["stage_a_outcome_used_only_for_preregistered_escalation"] is True
    assert manifest["stage_a_result"]["final_decision"] == "DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"
    assert manifest["planned_episode_count"] == 200
    assert manifest["stage_b_pair_count_per_policy"] == 40
    assert manifest["stage_b_reset_seeds"] == STAGE_B_RESET_SEEDS
    assert len(manifest["tasks"]) == 20
    assert manifest["canonical_payload_sha256"] == "2A14FA11271EC8FAD9BD91A1251952E9039A5BD297105BEBB78E27EFC4470A3B"
    assert manifest["identity_overlap_verification"]["overlap_with_stage_a_reset_seeds"] == 0
    keys = {(row["policy"], row["suite"], row["task_id"], row["reset_seed"]) for row in manifest["episodes"]}
    assert len(keys) == len(manifest["episodes"])
    pair_sets = {
        policy: {
            (row["suite"], row["task_id"], row["reset_seed"])
            for row in manifest["episodes"]
            if row["policy"] == policy
        }
        for policy in manifest["policy_order"]
    }
    assert len({tuple(sorted(values)) for values in pair_sets.values()}) == 1


def test_dagr_stage_b_result_records_valid_simple_baseline_kill() -> None:
    result = json.loads((DAGR / "stage_b_result.json").read_text(encoding="utf-8-sig"))

    assert result["final_decision"] == "DAGR_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD"
    assert result["closed_loop_experiment_happened"] is True
    assert result["confirmatory_test_tuning_happened"] is False
    assert result["completed_episode_count"] == 200
    assert result["summary"]["exception_count"] == 0

    by_policy = result["summary"]["by_policy"]
    assert by_policy["frozen_smolvla"]["successes"] == 28
    assert by_policy["dam_static_component_proxy"]["successes"] == 5
    assert by_policy["dagr_full"]["successes"] == 18
    assert by_policy["dagr_no_dynamic_route_ablation"]["successes"] == 16
    assert by_policy["gripper_transition_heuristic"]["successes"] == 24
    assert by_policy["dagr_full"]["mean_activation_fraction"] == 0.999952
    assert by_policy["dagr_full"]["action_validity_all_finite"] is True
    assert by_policy["dagr_full"]["action_validity_all_shape_ok"] is True

    paired = result["paired_vs_dagr_full"]
    assert paired["frozen_smolvla"]["paired_success_delta"] == -0.25
    assert paired["frozen_smolvla"]["paired_bootstrap_ci"] == [-0.4, -0.1]
    assert paired["gripper_transition_heuristic"]["paired_success_delta"] == -0.15
    assert paired["gripper_transition_heuristic"]["paired_bootstrap_ci"] == [-0.3, 0.0]
    assert paired["dagr_no_dynamic_route_ablation"]["paired_success_delta"] == 0.05
    assert paired["dam_static_component_proxy"]["paired_success_delta"] == 0.325


def test_dagr_stage_a_decision_uses_catastrophic_gate_only() -> None:
    noncat = {
        "exception_count": 0,
        "by_policy": {
            policy: {
                "successes": 2,
                "task_balanced_success_rate": 0.2,
            }
            for policy in STAGE_A_POLICY_ORDER
        },
    }
    noncat["by_policy"]["dagr_full"] = {"successes": 1, "task_balanced_success_rate": 0.1}
    assert _stage_a_decision(noncat) == "DAGR_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"

    catastrophic = {
        "exception_count": 0,
        "by_policy": {
            policy: {
                "successes": 0,
                "task_balanced_success_rate": 0.0,
            }
            for policy in STAGE_A_POLICY_ORDER
        },
    }
    catastrophic["by_policy"]["frozen_smolvla"] = {"successes": 4, "task_balanced_success_rate": 0.4}
    assert _stage_a_decision(catastrophic) == "DAGR_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE"
