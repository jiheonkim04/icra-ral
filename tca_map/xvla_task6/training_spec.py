"""Frozen no-training spec for the first MPR-XVLA adaptation attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SPEC_ARTIFACT = Path("runs/xvla_prior/epoch5_mpr_xvla_training_spec_v1.json")
XVLA_SOURCE = "/mnt/c/assets/repos/X-VLA"
XVLA_SOURCE_HEAD = "6bc2513f5f1cbec715cc668b414392a6cae5c671"
MODEL_ID = "2toINF/X-VLA-Libero"
MODEL_REVISION = "129e71460678b7236cee6fc9707f09d9fa0c3590"
TASK6_HDF5_WSL = (
    "/mnt/c/assets/data/libero/libero_10/"
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5"
)
TASK6_HEADROOM_ARTIFACTS = [
    "runs/xvla_prior/diagnostic_task6_expert_headroom_20260725_20260717T2050KST/result.json",
    "runs/xvla_prior/diagnostic_task6_expert_headroom_20260731_20260717T2055KST/result.json",
]
TASK6_DATA_AUDIT_ARTIFACT = (
    "runs/xvla_prior/diagnostic_task6_spatial_data_audit_20260717T2115KST/result.json"
)
TASK6_OPENVLA_INT4_SCREEN_ARTIFACT = (
    "runs/openvla_oft_int4/"
    "diagnostic_task6_residual_openvla_int4_20260725_20260731_openvlaenv_20260717T2114KST/result.json"
)
TASK6_CANDIDATE_ARTIFACT = "reports/epoch5_task6_ours_candidate_design.md"
TASK6_RESIDUAL_IDENTITIES = [20260725, 20260731]
TASK6_RESIDUAL_INITIAL_STATE_INDICES = [14, 20]
TASK6_RESIDUAL_SHA256 = [
    "47a0a589a343a89446f23421036719e5afd5bfd6fb1fc975c9a3546d867c3c82",
    "4f63fc206bad261b4721178ee1859e47c3111c119b2ef428e8d296ae7c0069e3",
]


def build_mpr_xvla_training_spec() -> dict[str, Any]:
    shared_training = {
        "seed": 20260717,
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
        "cycle_phase_counts": [1, 0, 1, 2],
        "phase_count_meaning": {
            "0": "neither mug-on-plate nor pudding-right relation is complete",
            "1": "white mug is on the plate and chocolate pudding still needs right-of-plate placement",
            "2": "both mug-on-plate and pudding-right relations are complete",
        },
        "replacement": True,
        "source_split": "train_demos_0_to_39_only",
        "confirmatory_residual_resets_used_for_sampling": False,
    }

    arms = [
        {
            "arm_id": "mpr_xvla_rank8_lambda2_lr1e4_steps64",
            "role": "primary_selected_method",
            "method": "MPR-XVLA",
            "phase_weight_lambda": 2.0,
            "sampler": sampler,
            "loss": (
                "X-VLA supervised action loss with sample/chunk weight "
                "(1 + lambda * mug_done_pudding_remaining_phase)"
            ),
            "claim_if_successful": (
                "mug-placed/pudding-right phase-balanced supervision improves "
                "the shared task-6 residual of the stronger X-VLA prior"
            ),
        },
        {
            "arm_id": "uniform_task6_xvla_rank8_lambda0_lr1e4_steps64",
            "role": "uniform_weight_ablation",
            "method": "uniform_task6_xvla_ablation",
            "phase_weight_lambda": 0.0,
            "sampler": sampler,
            "loss": "same X-VLA supervised action loss and sampler with uniform weights",
            "claim_if_successful": "generic task-6 adaptation, not MPR-XVLA weighting, may explain any gain",
        },
    ]

    return {
        "schema_version": "2026-07-17.epoch5_mpr_xvla_training_spec.v1",
        "freeze_id": "epoch5_mpr_xvla_training_spec_v1",
        "stage": "epoch_5_mpr_xvla_training_config_frozen_no_training",
        "date_kst": "2026-07-17",
        "training_happened_at_freeze": False,
        "optimizer_step_happened_at_freeze": False,
        "checkpoint_written_at_freeze": False,
        "closed_loop_ours_evaluation_happened_at_freeze": False,
        "method": "MPR-XVLA",
        "selected_prior": {
            "ecosystem": "X-VLA-Libero on LIBERO",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "official_code_checkout": XVLA_SOURCE,
            "official_code_head": XVLA_SOURCE_HEAD,
            "official_peft_training_available": True,
        },
        "prerequisites": {
            "matched_base_prior_gate": "TASK6_MATCHED_BASE_PRIOR_RESIDUAL_CONFIRMED",
            "headroom_gate": "TASK6_TASK_LEVEL_EXPERT_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE",
            "data_health_gate": "TASK6_SPATIAL_DATA_HEALTH_PASS_PREDESIGN_READY",
            "second_prior_gate": "TASK6_NOT_SOLVED_BY_OPENVLA_OFT_INT4",
            "candidate_design_gate": "TASK6_MPR_XVLA_SELECTED_AFTER_SECOND_PRIOR_RESIDUAL_SURVIVED",
            "headroom_artifacts": TASK6_HEADROOM_ARTIFACTS,
            "data_audit_artifact": TASK6_DATA_AUDIT_ARTIFACT,
            "openvla_int4_second_prior_artifact": TASK6_OPENVLA_INT4_SCREEN_ARTIFACT,
            "candidate_design_artifact": TASK6_CANDIDATE_ARTIFACT,
        },
        "interface_audit": {
            "official_training_script": "C:\\assets\\repos\\X-VLA\\peft_train.py",
            "official_training_script_has_peft_lora": True,
            "official_model_forward_returns_loss_dict": True,
            "official_dataloader_contract": {
                "meta_json_required": True,
                "dataset_name": "libero",
                "handler": "datasets.domain_handler.simulations.LiberoHandler",
                "required_hdf5_keys": [
                    "abs_action_6d",
                    "configured observation_key image datasets",
                    "configured language_instruction_key",
                ],
                "action_space": "ee6d, 20D padded single-arm trajectory, model action dim from config",
                "num_actions_from_model": 30,
            },
            "local_data_adapter_required_before_gradient_smoke": True,
            "raw_libero_hdf5_is_not_direct_official_xvla_training_input": True,
        },
        "data": {
            "suite": "libero_10",
            "task_id": 6,
            "instruction": "put the white mug on the plate and put the chocolate pudding to the right of the plate",
            "hdf5_path": TASK6_HDF5_WSL,
            "train_demo_indices": list(range(40)),
            "validation_demo_indices": list(range(40, 50)),
            "residual_failure_reset_identities": TASK6_RESIDUAL_IDENTITIES,
            "residual_failure_initial_state_indices": TASK6_RESIDUAL_INITIAL_STATE_INDICES,
            "residual_initial_state_sha256": TASK6_RESIDUAL_SHA256,
            "residual_initial_state_overlap_count": 0,
            "phase_label_source": "HDF5 simulator state for training/validation labels only",
            "privileged_state_at_inference": False,
            "phase_state_layout": {
                "porcelain_mug_pos_slice": [10, 13],
                "red_mug_pos_slice": [17, 20],
                "plate_pos_slice": [24, 27],
                "chocolate_pudding_pos_slice": [31, 34],
                "mug_plate_xy_threshold": 0.05,
                "pudding_abs_dx_threshold": 0.07,
                "pudding_dy_min": 0.08,
                "pudding_dy_max": 0.16,
            },
        },
        "shared_training": shared_training,
        "arms": arms,
        "matrix_limits": {
            "max_total_training_arms": 2,
            "max_total_configurations_in_epoch5_task6": 2,
            "new_configs_after_confirmatory_rollout_allowed": False,
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
            "closed_loop_residual_resets_used_for_model_selection": False,
            "primary_offline_metrics": [
                "validation_mug_done_pudding_remaining_weighted_loss",
                "validation_all_phase_loss",
                "validation_clean_phase_loss",
                "fixed_validation_action_delta_vs_xvla_prior",
                "cuda_peak_mib",
            ],
            "offline_pass_criteria": {
                "finite_losses_required": True,
                "mug_done_pudding_remaining_validation_loss_not_worse_than_prior": True,
                "clean_phase_validation_loss_relative_degradation_max": 0.05,
                "fixed_chunk_mean_abs_action_delta_max": 0.25,
                "fixed_chunk_max_abs_action_delta_max": 1.0,
                "cuda_peak_mib_max": 14500,
            },
            "primary_vs_ablation_rule": (
                "Prefer MPR-XVLA only if it passes all offline gates and beats the "
                "uniform ablation on mug-done/pudding-remaining validation loss."
            ),
            "closed_loop_eligibility": (
                "After offline selection, run at most one frozen residual-manifest "
                "closed-loop evaluation on 20260725 and 20260731; if it fails, do not retune."
            ),
        },
        "kill_rules": [
            "missing prerequisite artifact",
            "attempted network download",
            "optimizer step before data-adapter and gradient-smoke gates pass",
            "two identical infrastructure failures for the same arm",
            "nonfinite loss or gradient",
            "cuda peak memory above 14500 MiB",
            "validation clean-phase drift beyond frozen bound",
            "proposal of any third Epoch 5 task-6 MPR-XVLA training configuration",
            "any retuning based on residual reset identities 20260725 or 20260731",
            "using simulator object state as an inference input",
        ],
        "next_action_after_freeze": (
            "Materialize a tiny X-VLA-format data-adapter smoke artifact and run a "
            "one-batch no-optimizer gradient smoke; do not run optimizer.step yet."
        ),
    }


def validate_mpr_xvla_training_spec(spec: dict[str, Any]) -> list[str]:
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
    if spec.get("method") != "MPR-XVLA":
        errors.append("method must be MPR-XVLA")
    prerequisites = spec.get("prerequisites", {})
    if prerequisites.get("second_prior_gate") != "TASK6_NOT_SOLVED_BY_OPENVLA_OFT_INT4":
        errors.append("second-prior no-solve gate must be recorded")
    if spec.get("interface_audit", {}).get("local_data_adapter_required_before_gradient_smoke") is not True:
        errors.append("data adapter gate must be required")
    if spec.get("pre_optimizer_required_gates", {}).get("optimizer_step_allowed_before_all_gates") is not False:
        errors.append("optimizer step must be disallowed before gates")
    data = spec.get("data", {})
    if data.get("residual_failure_reset_identities") != TASK6_RESIDUAL_IDENTITIES:
        errors.append("task6 residual identities must remain 20260725 and 20260731")
    if data.get("residual_initial_state_overlap_count") != 0:
        errors.append("residual initial states must not overlap train/validation demos")
    if data.get("privileged_state_at_inference") is not False:
        errors.append("privileged state at inference must be false")
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
    if matrix.get("new_configs_after_confirmatory_rollout_allowed") is not False:
        errors.append("new configs after confirmatory rollout must be disallowed")
    return errors


def write_mpr_xvla_training_spec(output_path: Path) -> dict[str, Any]:
    spec = build_mpr_xvla_training_spec()
    errors = validate_mpr_xvla_training_spec(spec)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SPEC_ARTIFACT)
    args = parser.parse_args()
    spec = write_mpr_xvla_training_spec(args.output)
    print(json.dumps({"output": str(args.output), "freeze_id": spec["freeze_id"], "arms": len(spec["arms"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
