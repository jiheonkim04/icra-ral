"""Frozen bounded training specification for Epoch 5 R2R-OFT.

This module records the configuration gate that must pass before any
optimizer-step training.  It intentionally emits a small, deterministic JSON
lock rather than launching training.  The lock constrains the first R2R-OFT
training attempt to two arms: the selected phase-weighted method and its
uniform-weight ablation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


SPEC_ARTIFACT = Path("runs/openvla_oft_int4/epoch5_r2r_oft_training_spec_v1.json")
PRIOR_CHECKPOINT_WSL = (
    "/home/jiheon/assets/checkpoints/openvla-oft/"
    "moojink_openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
)
TASK8_HDF5_WSL = "/mnt/c/assets/data/libero/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5"


def build_epoch5_training_spec() -> dict[str, Any]:
    """Return the frozen bounded R2R-OFT training specification."""

    shared_training = {
        "seed": 20260717,
        "device": "single_local_cuda_16gb",
        "load_in_4bit": True,
        "full_bf16_attempted": False,
        "num_images_in_input": 2,
        "use_proprio": True,
        "use_l1_regression": True,
        "use_diffusion": False,
        "use_film": False,
        "unnorm_key": "libero_10_no_noops",
        "action_dim": 7,
        "action_chunk_size": 8,
        "batch_size": 1,
        "gradient_accumulation_steps": 1,
        "optimizer": "AdamW",
        "learning_rate": 0.0002,
        "weight_decay": 0.0,
        "max_optimizer_steps": 64,
        "save_steps": [16, 32, 64],
        "eval_steps": [0, 16, 32, 64],
        "max_wall_clock_minutes_per_arm": 90,
        "max_cuda_peak_mib": 14500,
        "lora_target_modules": "all-linear",
        "lora_dropout": 0.0,
        "merge_lora_during_training": False,
        "trainable_components": ["vla_lora_adapters"],
        "frozen_components": ["prior_action_head", "prior_proprio_projector"],
        "checkpoint_contents": [
            "peft_lora_adapter",
            "optimizer_state",
            "training_metrics_jsonl",
            "frozen_spec_snapshot",
            "git_commit",
        ],
    }

    sampler = {
        "kind": "deterministic_phase_cycle",
        "cycle_phase_counts": [1, 0, 1, 2],
        "phase_count_meaning": {
            "0": "no moka pot on/near inferred stove target",
            "1": "exactly one moka pot on/near inferred stove target",
            "2": "both moka pots on/near inferred stove target",
        },
        "replacement": True,
        "source_split": "train_demos_0_to_39_only",
        "confirmatory_residual_resets_used_for_sampling": False,
    }

    arms = [
        {
            "arm_id": "r2r_oft_rank4_lambda2_lr2e4_steps64",
            "role": "primary_selected_method",
            "method": "R2R-OFT",
            "lora_rank": 4,
            "lora_alpha": 8,
            "phase_weight_lambda": 2.0,
            "loss": "mean((1 + lambda * one_pot_remaining_phase) * L1(predicted_chunk, expert_chunk))",
            "sampler": sampler,
            "claim_if_successful": (
                "phase-weighted remaining-object supervision improves the exact task-8 "
                "second-object residual of the selected OpenVLA-OFT prior"
            ),
        },
        {
            "arm_id": "uniform_oft_rank4_lambda0_lr2e4_steps64",
            "role": "uniform_weight_ablation",
            "method": "uniform_task8_oft_ablation",
            "lora_rank": 4,
            "lora_alpha": 8,
            "phase_weight_lambda": 0.0,
            "loss": "mean(L1(predicted_chunk, expert_chunk)) with the same sampler and infrastructure",
            "sampler": sampler,
            "claim_if_successful": (
                "generic task-8 QLoRA adaptation or phase exposure, not the R2R weighting, "
                "may explain any gain"
            ),
        },
    ]

    return {
        "schema_version": 1,
        "freeze_id": "epoch5_r2r_oft_training_spec_v1",
        "stage": "epoch_5_r2r_oft_training_config_frozen",
        "date_kst": "2026-07-17",
        "training_happened_at_freeze": False,
        "optimizer_step_happened_at_freeze": False,
        "checkpoint_written_at_freeze": False,
        "method": "R2R-OFT",
        "selected_prior": {
            "ecosystem": "OpenVLA-OFT on LIBERO",
            "local_checkpoint": PRIOR_CHECKPOINT_WSL,
            "official_code_checkout": "/mnt/c/assets/repos/openvla-oft",
            "official_oft_reference": {
                "lora_rank": 32,
                "batch_size_per_gpu": 8,
                "paper_scale_max_steps": 150005,
                "local_development_deviation": (
                    "rank-4 QLoRA bounded development on a 16GB local GPU; not a "
                    "full-precision or paper-scale reproduction"
                ),
            },
        },
        "prerequisites": {
            "base_prior_residual_gate": "RESIDUAL_FOUND_PRIOR_POSITIVE_TASK_LEVEL_HEADROOM_POSITIVE",
            "data_health_gate": "R2R_OFT_DATA_HEALTH_PASS_PRETRAINING_READY",
            "qlora_gradient_gate": "R2R_OFT_QLORA_GRADIENT_SMOKE_PASS",
            "data_audit_artifact": "runs/openvla_oft_int4/epoch5_r2r_oft_pretraining_data_audit.json",
            "qlora_smoke_artifact": "runs/openvla_oft_int4/epoch5_r2r_oft_qlora_gradient_smoke.json",
        },
        "data": {
            "suite": "libero_10",
            "task_id": 8,
            "instruction": "put both moka pots on the stove",
            "hdf5_path": TASK8_HDF5_WSL,
            "train_demo_indices": list(range(40)),
            "validation_demo_indices": list(range(40, 50)),
            "residual_failure_reset_identities": [20260721, 20260722],
            "residual_failure_initial_state_indices": [10, 11],
            "residual_initial_state_overlap_count": 0,
            "privileged_state_at_inference": False,
            "phase_label_source": "HDF5 simulator state for training/validation labels only",
            "phase_state_layout": {
                "moka_pot_1_pos_slice": [10, 13],
                "moka_pot_2_pos_slice": [17, 20],
                "target_xy_source": "median final pot xy over train demos only",
            },
        },
        "shared_training": shared_training,
        "arms": arms,
        "matrix_limits": {
            "max_total_training_arms": 2,
            "max_total_configurations_in_epoch5": 2,
            "new_configs_after_confirmatory_rollout_allowed": False,
            "second_backbone_training_before_primary_gate": False,
        },
        "validation_selection": {
            "selection_source": "offline_validation_only_before_closed_loop",
            "closed_loop_residual_resets_used_for_model_selection": False,
            "primary_offline_metrics": [
                "validation_one_pot_weighted_l1",
                "validation_all_phase_l1",
                "validation_clean_phase_l1",
                "fixed_validation_action_delta_vs_prior",
                "cuda_peak_mib",
            ],
            "offline_pass_criteria": {
                "finite_losses_required": True,
                "one_pot_validation_l1_not_worse_than_prior": True,
                "clean_phase_validation_l1_relative_degradation_max": 0.05,
                "fixed_chunk_mean_abs_action_delta_max": 0.25,
                "fixed_chunk_max_abs_action_delta_max": 1.0,
                "cuda_peak_mib_max": 14500,
            },
            "primary_vs_ablation_rule": (
                "Prefer the primary arm only if it passes all offline gates and beats the "
                "uniform ablation on one-pot validation L1; otherwise report ambiguity or kill."
            ),
            "closed_loop_eligibility": (
                "After offline selection, run at most one frozen residual-manifest closed-loop "
                "evaluation; if it fails, do not retune on those reset identities."
            ),
        },
        "kill_rules": [
            "missing prerequisite artifact",
            "attempted network download",
            "full-BF16 OpenVLA-OFT load attempted",
            "two identical infrastructure failures for the same arm",
            "nonfinite loss or gradient",
            "cuda peak memory above 14500 MiB",
            "validation clean-phase drift beyond frozen bound",
            "proposal of any third Epoch 5 R2R-OFT training configuration",
            "any retuning based on residual reset identities 20260721 or 20260722",
        ],
        "next_action_after_freeze": (
            "Implement or launch only this frozen two-arm training plan in a detached WSL job; "
            "record PID/log/heartbeat before the first optimizer step."
        ),
    }


def validate_training_spec(spec: dict[str, Any]) -> list[str]:
    """Return validation errors for a training spec.  Empty means pass."""

    errors: list[str] = []
    arms = spec.get("arms", [])
    if len(arms) != 2:
        errors.append("spec must contain exactly two training arms")
    if len(arms) > int(spec.get("matrix_limits", {}).get("max_total_training_arms", 0)):
        errors.append("training arms exceed max_total_training_arms")
    if spec.get("training_happened_at_freeze") is not False:
        errors.append("training_happened_at_freeze must be false")
    if spec.get("optimizer_step_happened_at_freeze") is not False:
        errors.append("optimizer_step_happened_at_freeze must be false")
    if spec.get("checkpoint_written_at_freeze") is not False:
        errors.append("checkpoint_written_at_freeze must be false")
    shared = spec.get("shared_training", {})
    if shared.get("load_in_4bit") is not True:
        errors.append("load_in_4bit must be true")
    if shared.get("full_bf16_attempted") is not False:
        errors.append("full_bf16_attempted must be false")
    if int(shared.get("max_optimizer_steps", 10**9)) > 64:
        errors.append("max_optimizer_steps must be <= 64 for the first bounded run")
    if spec.get("validation_selection", {}).get("closed_loop_residual_resets_used_for_model_selection") is not False:
        errors.append("closed-loop residual resets must not be used for model selection")
    if spec.get("matrix_limits", {}).get("new_configs_after_confirmatory_rollout_allowed") is not False:
        errors.append("new configs after confirmatory rollout must be disallowed")

    arm_ids = {arm.get("arm_id") for arm in arms}
    if len(arm_ids) != len(arms):
        errors.append("arm_id values must be unique")
    roles = {arm.get("role") for arm in arms}
    if roles != {"primary_selected_method", "uniform_weight_ablation"}:
        errors.append("arms must be exactly primary_selected_method and uniform_weight_ablation")
    lambdas = {arm.get("role"): float(arm.get("phase_weight_lambda", -1)) for arm in arms}
    if lambdas.get("primary_selected_method") != 2.0:
        errors.append("primary arm lambda must be 2.0")
    if lambdas.get("uniform_weight_ablation") != 0.0:
        errors.append("uniform ablation lambda must be 0.0")
    return errors


def write_training_spec(output_path: Path) -> dict[str, Any]:
    """Build, validate, and write the deterministic training spec JSON."""

    spec = build_epoch5_training_spec()
    errors = validate_training_spec(spec)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SPEC_ARTIFACT)
    args = parser.parse_args()
    spec = write_training_spec(args.output)
    print(json.dumps({"output": str(args.output), "freeze_id": spec["freeze_id"], "arms": len(spec["arms"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
