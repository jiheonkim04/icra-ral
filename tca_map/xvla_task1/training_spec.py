"""Frozen no-training spec for the first BR-XVLA adaptation attempt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

SPEC_ARTIFACT = Path("runs/xvla_prior/epoch5_br_xvla_training_spec_v1.json")
XVLA_SOURCE = "/mnt/c/assets/repos/X-VLA"
XVLA_SOURCE_HEAD = "6bc2513f5f1cbec715cc668b414392a6cae5c671"
MODEL_ID = "2toINF/X-VLA-Libero"
MODEL_REVISION = "129e71460678b7236cee6fc9707f09d9fa0c3590"
TASK1_HDF5_WSL = (
    "/mnt/c/assets/data/libero/libero_10/"
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5"
)
TASK1_HEADROOM_ARTIFACT = (
    "runs/xvla_prior/diagnostic_task1_expert_headroom_20260727_20260717T180914KST/result.json"
)
TASK1_DATA_AUDIT_ARTIFACT = (
    "runs/xvla_prior/diagnostic_task1_basket_data_audit_20260727_20260717T181823KST/result.json"
)
TASK1_RESIDUAL_SHA256 = "bb8073f96294281b7008501d0b6ebdec3668f90448421c5937b58f57c1b8c5e2"


def build_br_xvla_training_spec() -> dict[str, Any]:
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
            "0": "neither cream cheese nor butter in the basket region",
            "1": "exactly one target object in the basket region; one target remains",
            "2": "both target objects in the basket region",
        },
        "replacement": True,
        "source_split": "train_demos_0_to_39_only",
        "confirmatory_residual_resets_used_for_sampling": False,
    }

    arms = [
        {
            "arm_id": "br_xvla_rank8_lambda2_lr1e4_steps64",
            "role": "primary_selected_method",
            "method": "BR-XVLA",
            "phase_weight_lambda": 2.0,
            "sampler": sampler,
            "loss": (
                "X-VLA supervised action loss with sample/chunk weight "
                "(1 + lambda * one_target_remaining_phase)"
            ),
            "claim_if_successful": (
                "basket-remaining phase-balanced supervision improves the shared "
                "task-1 residual of the stronger X-VLA prior"
            ),
        },
        {
            "arm_id": "uniform_xvla_rank8_lambda0_lr1e4_steps64",
            "role": "uniform_weight_ablation",
            "method": "uniform_task1_xvla_ablation",
            "phase_weight_lambda": 0.0,
            "sampler": sampler,
            "loss": "same X-VLA supervised action loss and sampler with uniform weights",
            "claim_if_successful": "generic task-1 adaptation, not BR-XVLA weighting, may explain any gain",
        },
    ]

    return {
        "schema_version": "2026-07-17.epoch5_br_xvla_training_spec.v1",
        "freeze_id": "epoch5_br_xvla_training_spec_v1",
        "stage": "epoch_5_br_xvla_training_config_frozen_no_training",
        "date_kst": "2026-07-17",
        "training_happened_at_freeze": False,
        "optimizer_step_happened_at_freeze": False,
        "checkpoint_written_at_freeze": False,
        "closed_loop_ours_evaluation_happened_at_freeze": False,
        "method": "BR-XVLA",
        "selected_prior": {
            "ecosystem": "X-VLA-Libero on LIBERO",
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "official_code_checkout": XVLA_SOURCE,
            "official_code_head": XVLA_SOURCE_HEAD,
            "official_peft_training_available": True,
        },
        "prerequisites": {
            "matched_base_prior_gate": "TASK1_MATCHED_BASE_PRIOR_RESIDUAL_CONFIRMED",
            "headroom_gate": "TASK1_TASK_LEVEL_EXPERT_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE",
            "data_health_gate": "TASK1_BASKET_DATA_HEALTH_PASS_PREDESIGN_READY",
            "headroom_artifact": TASK1_HEADROOM_ARTIFACT,
            "data_audit_artifact": TASK1_DATA_AUDIT_ARTIFACT,
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
            "task_id": 1,
            "instruction": "put both the cream cheese box and the butter in the basket",
            "hdf5_path": TASK1_HDF5_WSL,
            "train_demo_indices": list(range(40)),
            "validation_demo_indices": list(range(40, 50)),
            "residual_failure_reset_identity": 20260727,
            "excluded_xvla_regression_identity": 20260725,
            "residual_failure_initial_state_index": 16,
            "residual_initial_state_sha256": TASK1_RESIDUAL_SHA256,
            "residual_initial_state_overlap_count": 0,
            "phase_label_source": "HDF5 simulator state for training/validation labels only",
            "privileged_state_at_inference": False,
            "phase_state_layout": {
                "cream_cheese_pos_slice": [17, 20],
                "butter_pos_slice": [52, 55],
                "basket_pos_slice": [59, 62],
                "basket_xy_threshold": 0.08,
            },
        },
        "shared_training": shared_training,
        "arms": arms,
        "matrix_limits": {
            "max_total_training_arms": 2,
            "max_total_configurations_in_epoch5_task1": 2,
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
                "validation_one_target_remaining_weighted_loss",
                "validation_all_phase_loss",
                "validation_clean_phase_loss",
                "fixed_validation_action_delta_vs_xvla_prior",
                "cuda_peak_mib",
            ],
            "offline_pass_criteria": {
                "finite_losses_required": True,
                "one_target_remaining_validation_loss_not_worse_than_prior": True,
                "clean_phase_validation_loss_relative_degradation_max": 0.05,
                "fixed_chunk_mean_abs_action_delta_max": 0.25,
                "fixed_chunk_max_abs_action_delta_max": 1.0,
                "cuda_peak_mib_max": 14500,
            },
            "primary_vs_ablation_rule": (
                "Prefer BR-XVLA only if it passes all offline gates and beats the "
                "uniform ablation on one-target-remaining validation loss."
            ),
            "closed_loop_eligibility": (
                "After offline selection, run at most one frozen residual-manifest "
                "closed-loop evaluation on 20260727; if it fails, do not retune."
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
            "proposal of any third Epoch 5 task-1 BR-XVLA training configuration",
            "any retuning based on residual reset identity 20260727",
            "using X-VLA regression identity 20260725 as an Ours target",
        ],
        "next_action_after_freeze": (
            "Materialize a tiny X-VLA-format data-adapter smoke artifact and run a "
            "one-batch no-optimizer gradient smoke; do not run optimizer.step yet."
        ),
    }


def validate_br_xvla_training_spec(spec: dict[str, Any]) -> list[str]:
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
    if spec.get("method") != "BR-XVLA":
        errors.append("method must be BR-XVLA")
    if spec.get("interface_audit", {}).get("local_data_adapter_required_before_gradient_smoke") is not True:
        errors.append("data adapter gate must be required")
    if spec.get("pre_optimizer_required_gates", {}).get("optimizer_step_allowed_before_all_gates") is not False:
        errors.append("optimizer step must be disallowed before gates")
    if spec.get("data", {}).get("excluded_xvla_regression_identity") != 20260725:
        errors.append("20260725 must remain excluded")
    if spec.get("data", {}).get("residual_failure_reset_identity") != 20260727:
        errors.append("20260727 must be the only residual target")
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


def write_br_xvla_training_spec(output_path: Path) -> dict[str, Any]:
    spec = build_br_xvla_training_spec()
    errors = validate_br_xvla_training_spec(spec)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SPEC_ARTIFACT)
    args = parser.parse_args()
    spec = write_br_xvla_training_spec(args.output)
    print(json.dumps({"output": str(args.output), "freeze_id": spec["freeze_id"], "arms": len(spec["arms"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
