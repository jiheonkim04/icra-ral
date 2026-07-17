"""Frozen no-training spec for the first R2P-XVLA task-5 adaptation attempt.

This module records a deterministic configuration lock only. It does not load
models, train, run an optimizer step, write checkpoints, launch simulators, or
evaluate Ours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPEC_ARTIFACT = Path("runs/xvla_prior/epoch5_r2p_xvla_task5_training_spec_v1.json")
XVLA_SOURCE = "/mnt/c/assets/repos/X-VLA"
XVLA_SOURCE_HEAD = "6bc2513f5f1cbec715cc668b414392a6cae5c671"
MODEL_ID = "2toINF/X-VLA-Libero"
MODEL_REVISION = "129e71460678b7236cee6fc9707f09d9fa0c3590"
TASK5_HDF5_WSL = (
    "/mnt/c/assets/data/libero/libero_spatial/"
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
RESIDUAL_IDENTITY = 20260727
RESIDUAL_INITIAL_STATE_INDEX = 16
RESIDUAL_INITIAL_STATE_SHA256 = "7230223d3b36c289be0dc4cfbfe916bfe65e2b20c4755b123504b97f9db19e76"


def build_r2p_xvla_training_spec() -> dict[str, Any]:
    """Return the frozen bounded R2P-XVLA training specification."""

    shared_training = {
        "seed": 20260718,
        "device": "single_local_cuda_16gb",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "source_checkout": XVLA_SOURCE,
        "source_head": XVLA_SOURCE_HEAD,
        "official_entrypoint": "peft_train.py",
        "official_lora_config": {
            "r": 8,
            "lora_alpha": 16,
            "bias": "none",
            "target_modules": "all-linear",
            "modules_to_save": [
                "transformer.soft_prompt_hub",
                "transformer.action_encoder",
                "transformer.action_decoder",
            ],
        },
        "action_mode": "ee6d",
        "domain_id": 3,
        "num_actions": 30,
        "batch_size": 1,
        "learning_rate": 0.0001,
        "learning_coef": 1.0,
        "weight_decay": 0.0,
        "max_grad_norm": 1.0,
        "max_optimizer_steps": 64,
        "save_steps": [16, 32, 64],
        "eval_steps": [0, 16, 32, 64],
        "max_cuda_peak_mib": 14500,
        "max_wall_clock_minutes_per_arm": 90,
        "trainable_components": [
            "peft_lora_adapter",
            "transformer.soft_prompt_hub",
            "transformer.action_encoder",
            "transformer.action_decoder",
        ],
        "frozen_or_zero_lr_components_during_local_freeze": [
            "base_florence_vlm_weights",
            "base_transformer_core_weights",
        ],
    }

    sampler = {
        "kind": "deterministic_phase_cycle",
        "phase_names": ["source_on_ramekin", "transit", "target_on_plate"],
        "cycle_phase_counts": {
            "source_on_ramekin": 1,
            "transit": 2,
            "target_on_plate": 2,
        },
        "replacement": True,
        "source_split": "train_demos_0_to_39_only",
        "confirmatory_residual_reset_used_for_sampling": False,
    }

    arms = [
        {
            "arm_id": "r2p_xvla_rank8_phase_weights_lr1e4_steps64",
            "role": "primary_selected_method",
            "method": "R2P-XVLA",
            "phase_weight_lambda": 2.0,
            "phase_loss_weights": {
                "source_on_ramekin": 1.0,
                "transit": 2.0,
                "target_on_plate": 1.5,
            },
            "sampler": sampler,
            "loss": (
                "X-VLA supervised action loss with deterministic phase sampling "
                "and per-chunk source/transit/target weights"
            ),
            "claim_if_successful": (
                "phase-balanced source-to-target supervision improves the matched "
                "black-bowl-on-ramekin to plate residual of X-VLA-Libero"
            ),
        },
        {
            "arm_id": "uniform_task5_xvla_rank8_lambda0_lr1e4_steps64",
            "role": "uniform_weight_ablation",
            "method": "uniform_task5_xvla_ablation",
            "phase_weight_lambda": 0.0,
            "phase_loss_weights": {
                "source_on_ramekin": 1.0,
                "transit": 1.0,
                "target_on_plate": 1.0,
            },
            "sampler": sampler,
            "loss": "same X-VLA supervised action loss and sampler with uniform phase weights",
            "claim_if_successful": "generic task-5 adaptation, not R2P phase balancing, may explain any gain",
        },
    ]

    return {
        "schema_version": "2026-07-18.epoch5_r2p_xvla_task5_training_spec.v1",
        "freeze_id": "epoch5_r2p_xvla_task5_training_spec_v1",
        "stage": "epoch_5_r2p_xvla_task5_training_config_frozen_no_training",
        "date_kst": "2026-07-18",
        "training_happened_at_freeze": False,
        "optimizer_step_happened_at_freeze": False,
        "checkpoint_written_at_freeze": False,
        "closed_loop_ours_evaluation_happened_at_freeze": False,
        "method": "R2P-XVLA",
        "selected_prior": {
            "ecosystem": "X-VLA-Libero on LIBERO",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "official_code_checkout": XVLA_SOURCE,
            "official_code_head": XVLA_SOURCE_HEAD,
            "official_peft_training_available": True,
        },
        "prerequisites": {
            "xvla_first_prior_gate": "POST_SECONDPRIOR_LIBERO_SPATIAL_IDENTITY20260727_XVLA_PRIOR_RESIDUAL_TASK5",
            "smolvla_base_gate": "POST_SECONDPRIOR_LIBERO_SPATIAL_IDENTITY20260727_SMOLVLA_BASE_CLEAN_FAILURE_TASK5",
            "headroom_gate": "TASK5_TASK_LEVEL_EXPERT_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE",
            "second_prior_gate": "POST_SECONDPRIOR_LIBERO_SPATIAL_IDENTITY20260727_SECOND_PRIOR_CLEAN_FAILURE_TARGET_REMAINS",
            "data_health_gate": "POST_SECONDPRIOR_LIBERO_SPATIAL_IDENTITY20260727_DATA_AUDIT_PASS_CANDIDATE_READY",
            "candidate_gate": "EXACTLY_TWO_CANDIDATES_GENERATED_ONE_SELECTED",
            "xvla_prior_report": "reports/post_secondprior_libero_spatial_20260727_prior_scan_result.json",
            "smolvla_base_report": "reports/post_secondprior_libero_spatial_20260727_base_gate_result.json",
            "headroom_report": "reports/post_secondprior_libero_spatial_20260727_headroom_result.json",
            "second_prior_report": "reports/post_secondprior_libero_spatial_20260727_second_prior_result.json",
            "data_audit_report": "reports/post_secondprior_libero_spatial_20260727_data_audit_result.json",
            "candidate_generation_report": "reports/post_secondprior_libero_spatial_20260727_candidate_generation_result.json",
        },
        "single_identity_safeguard": {
            "residual_identity_count": 1,
            "same_reset_expert_headroom_available": False,
            "paper_claim_from_single_identity_allowed": False,
            "closed_loop_positive_result_interpretation": (
                "diagnostic mechanism evidence only; a later independent identity or "
                "second condition is required before any prototype-go or paper-scale claim"
            ),
            "retuning_after_residual_rollout_allowed": False,
        },
        "interface_audit": {
            "official_training_script": "C:/assets/repos/X-VLA/peft_train.py",
            "official_training_script_has_peft_lora": True,
            "raw_libero_hdf5_is_not_direct_official_xvla_training_input": True,
            "local_xvla_format_data_adapter_required_before_gradient_smoke": True,
            "optimizer_step_allowed_before_data_adapter_and_gradient_smoke": False,
        },
        "deployment_input_policy": {
            "privileged_state_at_inference": False,
            "phase_labels_at_inference": False,
            "allowed_inputs": ["RGB", "wrist RGB", "proprioception", "instruction"],
        },
        "data": {
            "suite": "libero_spatial",
            "task_id": 5,
            "instruction": "pick up the black bowl on the ramekin and place it on the plate",
            "hdf5_path": TASK5_HDF5_WSL,
            "train_demo_indices": list(range(40)),
            "validation_demo_indices": list(range(40, 50)),
            "train_chunk_count": 4325,
            "validation_chunk_count": 1121,
            "train_phase_chunk_counts": {
                "source_on_ramekin": 2627,
                "transit": 650,
                "target_on_plate": 1048,
            },
            "validation_phase_chunk_counts": {
                "source_on_ramekin": 711,
                "transit": 164,
                "target_on_plate": 246,
            },
            "residual_failure_reset_identities": [RESIDUAL_IDENTITY],
            "residual_failure_initial_state_indices": [RESIDUAL_INITIAL_STATE_INDEX],
            "residual_initial_state_sha256": [RESIDUAL_INITIAL_STATE_SHA256],
            "residual_initial_state_overlap_count": 0,
            "phase_label_source": "HDF5 simulator state for training/validation labels only",
            "privileged_state_at_inference": False,
            "phase_state_layout": {
                "target_black_bowl_pos_slice": [10, 13],
                "ramekin_pos_slice": [31, 34],
                "plate_pos_slice": [38, 41],
                "source_xy_threshold": 0.05,
                "target_xy_threshold": 0.05,
                "source_target_separation_min": 0.10,
            },
        },
        "shared_training": shared_training,
        "arms": arms,
        "matrix_limits": {
            "max_total_training_arms": 2,
            "max_total_configurations_in_epoch5_task5": 2,
            "new_configs_after_residual_rollout_allowed": False,
            "second_backbone_training_before_primary_gate": False,
        },
        "pre_optimizer_required_gates": {
            "xvla_format_data_adapter_materialized": False,
            "xvla_format_data_adapter_smoke_passed": False,
            "one_batch_gradient_smoke_passed": False,
            "optimizer_step_allowed_before_all_gates": False,
        },
        "validation_selection": {
            "selection_source": "offline_validation_only_before_closed_loop",
            "closed_loop_residual_reset_used_for_model_selection": False,
            "primary_offline_metrics": [
                "validation_phase_weighted_action_loss",
                "validation_transit_action_loss",
                "validation_target_on_plate_action_loss",
                "validation_source_on_ramekin_action_loss",
                "fixed_validation_action_delta_vs_xvla_prior",
                "cuda_peak_mib",
            ],
            "offline_pass_criteria": {
                "finite_losses_required": True,
                "primary_must_beat_uniform_on_phase_weighted_validation_loss": True,
                "primary_must_not_worsen_source_phase_loss_vs_uniform_by_more_than": 0.05,
                "fixed_chunk_mean_abs_action_delta_max": 0.25,
                "fixed_chunk_max_abs_action_delta_max": 1.0,
                "cuda_peak_mib_max": 14500,
            },
            "closed_loop_eligibility": (
                "After offline selection only, run at most one frozen target-residual "
                "closed-loop diagnostic on identity 20260727 and one clean-retention "
                "mini-manifest; if it fails, do not retune on the residual identity."
            ),
            "clean_retention_manifest": {
                "source": "X-VLA prior-success tasks from the 20260727 spatial scan",
                "task_ids": [0, 1, 2, 3, 4, 6, 7, 8, 9],
                "selection_fixed_before_ours_rollout": True,
                "noninferiority_rule": "no more than one new failure in the fixed mini-manifest before any expansion",
            },
        },
        "kill_rules": [
            "missing prerequisite artifact",
            "attempted network download",
            "optimizer step before data-adapter and gradient-smoke gates pass",
            "two identical infrastructure failures for the same arm",
            "nonfinite loss or gradient",
            "cuda peak memory above 14500 MiB",
            "validation source-phase drift beyond frozen bound",
            "uniform ablation matches or beats primary on phase-weighted validation loss",
            "proposal of any third Epoch 5 task-5 R2P-XVLA training configuration",
            "any retuning based on residual identity 20260727",
            "using simulator object state or phase labels as inference inputs",
        ],
        "next_action_after_freeze": (
            "Materialize a tiny X-VLA-format data-adapter smoke artifact and run a "
            "one-batch no-optimizer gradient smoke; do not run optimizer.step yet."
        ),
    }


def validate_r2p_xvla_training_spec(spec: dict[str, Any]) -> list[str]:
    """Return validation errors for a spec. Empty means pass."""

    errors: list[str] = []
    arms = spec.get("arms", [])
    if len(arms) != 2:
        errors.append("spec must contain exactly two training arms")
    if spec.get("training_happened_at_freeze") is not False:
        errors.append("training_happened_at_freeze must be false")
    if spec.get("optimizer_step_happened_at_freeze") is not False:
        errors.append("optimizer_step_happened_at_freeze must be false")
    if spec.get("checkpoint_written_at_freeze") is not False:
        errors.append("checkpoint_written_at_freeze must be false")
    if spec.get("closed_loop_ours_evaluation_happened_at_freeze") is not False:
        errors.append("closed_loop_ours_evaluation_happened_at_freeze must be false")
    if spec.get("method") != "R2P-XVLA":
        errors.append("method must be R2P-XVLA")
    prerequisites = spec.get("prerequisites", {})
    if prerequisites.get("candidate_gate") != "EXACTLY_TWO_CANDIDATES_GENERATED_ONE_SELECTED":
        errors.append("candidate gate must be recorded")
    if prerequisites.get("second_prior_gate") != (
        "POST_SECONDPRIOR_LIBERO_SPATIAL_IDENTITY20260727_SECOND_PRIOR_CLEAN_FAILURE_TARGET_REMAINS"
    ):
        errors.append("second-prior clean-failure gate must be recorded")
    if spec.get("deployment_input_policy", {}).get("privileged_state_at_inference") is not False:
        errors.append("privileged state at inference must be false")
    if spec.get("deployment_input_policy", {}).get("phase_labels_at_inference") is not False:
        errors.append("phase labels at inference must be false")
    if spec.get("pre_optimizer_required_gates", {}).get("optimizer_step_allowed_before_all_gates") is not False:
        errors.append("optimizer step must be disallowed before gates")
    data = spec.get("data", {})
    if data.get("residual_failure_reset_identities") != [RESIDUAL_IDENTITY]:
        errors.append("task5 residual identity must remain 20260727")
    if data.get("residual_initial_state_overlap_count") != 0:
        errors.append("residual initial states must not overlap train/validation demos")
    if data.get("privileged_state_at_inference") is not False:
        errors.append("data privileged_state_at_inference must be false")
    roles = {arm.get("role") for arm in arms}
    if roles != {"primary_selected_method", "uniform_weight_ablation"}:
        errors.append("arms must be primary plus uniform ablation")
    lambdas = {arm.get("role"): float(arm.get("phase_weight_lambda", -1)) for arm in arms}
    if lambdas.get("primary_selected_method") != 2.0:
        errors.append("primary arm lambda must be 2.0")
    if lambdas.get("uniform_weight_ablation") != 0.0:
        errors.append("uniform ablation lambda must be 0.0")
    matrix = spec.get("matrix_limits", {})
    if len(arms) > int(matrix.get("max_total_training_arms", 0)):
        errors.append("arms exceed max_total_training_arms")
    if matrix.get("new_configs_after_residual_rollout_allowed") is not False:
        errors.append("new configs after residual rollout must be disallowed")
    if spec.get("single_identity_safeguard", {}).get("paper_claim_from_single_identity_allowed") is not False:
        errors.append("single identity must not be allowed to support a paper claim")
    return errors


def write_r2p_xvla_training_spec(output_path: Path) -> dict[str, Any]:
    """Build, validate, and write the deterministic training-spec JSON."""

    spec = build_r2p_xvla_training_spec()
    errors = validate_r2p_xvla_training_spec(spec)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SPEC_ARTIFACT)
    args = parser.parse_args()
    spec = write_r2p_xvla_training_spec(args.output)
    print(json.dumps({"output": str(args.output), "freeze_id": spec["freeze_id"], "arms": len(spec["arms"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
