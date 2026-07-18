"""Load and enforce the frozen method-level specification."""

from __future__ import annotations

import json
import pathlib
from typing import Any

from .adapter import (
    ActionConsistentMissingViewAdapter,
    adapter_parameter_count,
    state_dict_parameter_count,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "configs" / "action_consistent_missing_view_distillation_xvla_frozen_spec.json"


def load_frozen_method_spec(path: pathlib.Path = DEFAULT_SPEC) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        spec = json.load(handle)
    validate_frozen_method_spec(spec)
    return spec


def validate_frozen_method_spec(spec: dict[str, Any]) -> None:
    if spec.get("method") != "ACTION_CONSISTENT_MISSING_VIEW_DISTILLATION_XVLA":
        raise ValueError("method identity drift")
    if spec.get("novelty_decision") != "INCREMENTAL_BUT_POTENTIALLY_PUBLISHABLE":
        raise ValueError("overlap-audit decision drift")

    deployment = spec.get("deployment_graph") or {}
    forbidden = deployment.get("forbidden_inputs") or []
    required_forbidden = {
        "clean_view_teacher",
        "future_frame",
        "expert_action",
        "demonstration_action_oracle",
        "reward",
        "done_or_success_flag",
        "simulator_object_contact_or_pose_state",
        "privileged_reset_identity",
        "retrieval_library_or_nearest_demonstration_search",
        "reconstructed_wrist_token_or_image_insertion",
    }
    if not required_forbidden.issubset(set(forbidden)):
        raise ValueError("privileged deployment prohibition drift")
    if deployment.get("reconstruction_decoder_executed") is not False:
        raise ValueError("reconstruction decoder may not execute at deployment")
    if deployment.get("clean_view_hook_state") != "DEACTIVATED_EXACT_XVLA_PATH":
        raise ValueError("clean bypass drift")

    xvla = spec.get("xvla") or {}
    if xvla.get("full_model_finetuning") is not False or xvla.get("all_official_parameters_frozen") is not True:
        raise ValueError("frozen X-VLA scope drift")
    if xvla.get("cpu_or_disk_model_offload") is not False:
        raise ValueError("model offload is prohibited")
    paired_forward = spec.get("paired_training_forward") or {}
    condition = str(paired_forward.get("wrist_dropout_implementation", ""))
    if "do not change image_mask" not in condition or "black-frame tensor" not in condition:
        raise ValueError("frozen black-pixel wrist-dropout wiring drift")

    module = spec.get("trainable_module") or {}
    adapter = ActionConsistentMissingViewAdapter(
        hidden_size=int(module["hidden_size"]),
        bottleneck_dim=int(module["bottleneck_dim"]),
        wrist_token_count=int(module["wrist_token_count"]),
        wrist_token_dim=int(module["wrist_token_dim"]),
        residual_scale=float(module["residual_scale"]),
    )
    count = adapter_parameter_count(adapter)
    if count != int(module.get("trainable_parameter_count_exact", -1)):
        raise ValueError(f"trainable parameter count drift: {count}")
    inference_count = state_dict_parameter_count(adapter.inference_state_dict())
    if inference_count != int(module.get("inference_parameter_count_exact", -1)):
        raise ValueError(f"inference parameter count drift: {inference_count}")

    roles = spec.get("comparator_roles") or {}
    required_roles = {"BASE", "EXTERNAL_PRIOR", "OURS", "KEY_ABLATION", "MECHANISM_ABLATION", "GENERIC_CONTROL"}
    if not required_roles.issubset(set(roles)):
        raise ValueError("comparator role drift")
    if roles["EXTERNAL_PRIOR"].get("implementation_label") != "MECHANISM_FAITHFUL_RL4IL_LOCAL_PORT":
        raise ValueError("RL4IL fidelity label drift")

    arms = spec.get("stage0_training_arms") or []
    counts = {int(arm.get("trainable_parameter_count", -1)) for arm in arms}
    if len(arms) != 4 or counts != {count}:
        raise ValueError("Stage 0 arm capacity matching drift")

    splits = spec.get("data_splits") or {}
    if bool(splits.get("confirmatory_outcomes_accessed")):
        raise ValueError("confirmatory outcomes may not be accessed")
    if set(splits) < {"discovery", "validation", "fixed_confirmation_reserve", "stage_a", "stage_b"}:
        raise ValueError("discovery/validation/confirmatory split missing")
    stage_a = splits["stage_a"]
    if len(stage_a.get("initial_reset_identities_per_task") or []) != 3:
        raise ValueError("Stage A must begin with three identities per task")
    if len(stage_a.get("single_expansion_reset_identities_per_task") or []) != 2:
        raise ValueError("Stage A may add exactly two identities per task once")
    if int(stage_a.get("maximum_expansion_count", -1)) != 1:
        raise ValueError("Stage A expansion-count drift")
    stage_b = splits["stage_b"]
    if len(stage_b.get("task_keys") or []) < 4:
        raise ValueError("simulation-only Stage B requires at least four tasks")
    if len(stage_b.get("failure_conditions") or []) < 3:
        raise ValueError("simulation-only Stage B requires three wrist-failure conditions")
    if int(stage_b.get("initial_paired_failure_episode_rows_per_policy", 0)) < 60:
        raise ValueError("Stage B initial paired sample is under 60")
    if int(stage_b.get("single_expansion_paired_failure_episode_rows_per_policy", 0)) != 80:
        raise ValueError("Stage B fixed expansion drift")
    if int(stage_b.get("maximum_expansion_count", -1)) != 1:
        raise ValueError("Stage B expansion-count drift")

    budget = spec.get("training_budget") or {}
    if int(budget.get("optimizer_steps_per_arm", 0)) <= 0:
        raise ValueError("optimizer budget must be positive")
    if int(budget.get("effective_batch", 0)) != 8:
        raise ValueError("effective batch drift")
    if budget.get("checkpoint_selection") != "FINAL_STEP_ONLY_NO_VALIDATION_SELECTION":
        raise ValueError("checkpoint-selection drift")

    effect = spec.get("practical_effect_rule") or {}
    if float(effect.get("relative_improvement_min", 0.0)) != 0.05:
        raise ValueError("practical-effect relative threshold drift")
    if int(effect.get("noise_multiplier", 0)) != 10:
        raise ValueError("numerical-noise multiplier drift")

    repair = spec.get("bounded_repair") or {}
    if int(repair.get("maximum_count", -1)) != 1 or int(repair.get("current_count", -1)) != 1:
        raise ValueError("the single preflight path repair must remain consumed")
    if repair.get("consumed_by") != "PREFLIGHT_OFFICIAL_READER_IMPORT_INITIALIZATION_ORDER_ERROR":
        raise ValueError("bounded repair identity drift")
    if len(repair.get("failed_attempts") or []) != 2:
        raise ValueError("both layers of the single reader-import repair must remain preserved")
    if repair.get("scientific_protocol_changed") is not False:
        raise ValueError("bounded path repair may not change the scientific protocol")
    if repair.get("additional_repairs_authorized") is not False:
        raise ValueError("no additional implementation repair is authorized")

    boundaries = spec.get("execution_boundaries") or {}
    prohibited_true = (
        "closed_loop_rollout_authorized_by_method_spec",
        "confirmatory_tuning_authorized",
        "new_method_generation_authorized",
        "physical_robot_manipulation_authorized",
    )
    if any(bool(boundaries.get(key)) for key in prohibited_true):
        raise ValueError("method specification exceeded its execution boundary")
    if boundaries.get("simulation_only_paper_candidate_path_valid") is not True:
        raise ValueError("simulation-only paper path drift")
    if boundaries.get("second_backbone_required_for_paper_candidate_go") is not False:
        raise ValueError("second backbone may not be a universal paper gate")
    if boundaries.get("camera_only_validation_required_for_paper_candidate_go") is not False:
        raise ValueError("camera-only validation may not be a universal paper gate")
    erratum = spec.get("pre_execution_specification_erratum") or {}
    if erratum.get("scientific_condition_changed") is not False:
        raise ValueError("pre-execution condition correction may not change the science")
    if erratum.get("bounded_implementation_repair_consumed") is not False:
        raise ValueError("description correction may not consume the implementation repair")
