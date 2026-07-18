"""Report-only action-bias bounds gate for SGL-XVLA Stage 0.

This gate freezes conservative action-bias limits for the SGL-XVLA candidate
using only existing expert-action statistics and X-VLA action-range metadata.
It does not load a VLA model, launch a simulator, train, run an optimizer, write
checkpoints, or evaluate Ours.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tca_map.xvla_spatial_task5.sgl_observability_audit import (
    OBSERVABILITY_ARTIFACT,
    build_sgl_observability_audit,
    validate_sgl_observability_audit,
)
from tca_map.xvla_spatial_task5.sgl_stage0_gate import (
    CANDIDATE_ID,
    CLEAN_RETENTION_IDENTITIES,
    HELD_OUT_CONFIRMATORY_IDENTITY_POOL,
    RESIDUAL_IDENTITIES,
    STAGE0_ARTIFACT,
    TARGET_INSTRUCTION,
    TARGET_SUITE,
    TARGET_TASK_ID,
    build_sgl_xvla_stage0_gate,
    validate_sgl_xvla_stage0_gate,
)


ACTION_BOUNDS_ARTIFACT = Path("runs/xvla_prior/epoch5_sgl_xvla_task5_action_bounds_gate_v1.json")

EXPERT_HEADROOM_RESULT_PATHS = [
    "runs/xvla_prior/diagnostic_libero_spatial_task5_expert_headroom_20260730_20260718T1105KST/result.json",
    "runs/xvla_prior/diagnostic_libero_spatial_task5_expert_headroom_20260733_20260718T1121KST/result.json",
]
XVLA_ACTION_RANGE_RESULT_PATHS = [
    "runs/xvla_prior/failure_scan_libero_spatial_identity20260730_post_r2p_archive_20260718T0645KST/task_5/result.json",
    "runs/xvla_prior/repeated_residual_spatial_task5_id20260731_33_xvla_prior_20260718T1115KST/result.json",
]

EXPERT_SELECTED_DEMO_ACTION_STATS: dict[str, Any] = {
    "selected_demo": "demo_9",
    "finite": True,
    "shape": [118, 7],
    "range": {"min": -1.0, "max": 1.0, "max_abs": 1.0, "mean": 0.031709},
    "translation_range": {"min": -0.9375, "max": 0.9375, "max_abs": 0.9375, "mean": 0.045316},
    "rotation_range": {"min": -0.207857, "max": 0.133929, "max_abs": 0.207857, "mean": -0.022176},
    "gripper_range": {"min": -1.0, "max": 1.0, "max_abs": 1.0, "mean": 0.152542},
    "clip_rate_if_env_adapter_applied": 0.0,
}

EXPERT_HEADROOM_EVIDENCE = [
    {
        "reset_identity": 20260730,
        "path": EXPERT_HEADROOM_RESULT_PATHS[0],
        "decision": "TASK5_TASK_LEVEL_EXPERT_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE",
        "selected_demo": "demo_9",
        "exact_replay_success": True,
        "same_reset_headroom_available": False,
        "task_level_expert_headroom_positive": True,
        "action_stats_match_frozen_stats": True,
    },
    {
        "reset_identity": 20260733,
        "path": EXPERT_HEADROOM_RESULT_PATHS[1],
        "decision": "TASK5_TASK_LEVEL_EXPERT_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE",
        "selected_demo": "demo_9",
        "exact_replay_success": True,
        "same_reset_headroom_available": False,
        "task_level_expert_headroom_positive": True,
        "action_stats_match_frozen_stats": True,
    },
]

XVLA_ACTION_RANGE_EVIDENCE = [
    {
        "reset_identity": 20260730,
        "identity_role": "residual_failure",
        "path": XVLA_ACTION_RANGE_RESULT_PATHS[0],
        "success": False,
        "steps": 900,
        "action_chunk_count": 30,
        "first_three_action_chunk_ranges": [
            {"finite": True, "max": 1.1772905588150024, "min": -0.1951722502708435},
            {"finite": True, "max": 1.0731316804885864, "min": -0.4875808358192444},
            {"finite": True, "max": 1.0529303550720215, "min": -0.5744004249572754},
        ],
        "first_two_max_abs_upper": 1.1772905588150024,
    },
    {
        "reset_identity": 20260731,
        "identity_role": "clean_retention_success",
        "path": XVLA_ACTION_RANGE_RESULT_PATHS[1],
        "success": True,
        "steps": 88,
        "action_chunk_count": 3,
        "first_three_action_chunk_ranges": [
            {"finite": True, "max": 1.1748055219650269, "min": -0.19885407388210297},
            {"finite": True, "max": 1.064784288406372, "min": -0.44312745332717896},
            {"finite": True, "max": 1.0412840843200684, "min": -0.5786740779876709},
        ],
        "first_two_max_abs_upper": 1.1748055219650269,
    },
    {
        "reset_identity": 20260732,
        "identity_role": "clean_retention_success",
        "path": XVLA_ACTION_RANGE_RESULT_PATHS[1],
        "success": True,
        "steps": 134,
        "action_chunk_count": 5,
        "first_three_action_chunk_ranges": [
            {"finite": True, "max": 1.1748055219650269, "min": -0.19885407388210297},
            {"finite": True, "max": 1.064784288406372, "min": -0.44312745332717896},
            {"finite": True, "max": 1.0412840843200684, "min": -0.5786740779876709},
        ],
        "first_two_max_abs_upper": 1.1748055219650269,
    },
    {
        "reset_identity": 20260733,
        "identity_role": "residual_failure",
        "path": XVLA_ACTION_RANGE_RESULT_PATHS[1],
        "success": False,
        "steps": 900,
        "action_chunk_count": 30,
        "first_three_action_chunk_ranges": [
            {"finite": True, "max": 1.1748055219650269, "min": -0.19885407388210297},
            {"finite": True, "max": 1.064784288406372, "min": -0.44312745332717896},
            {"finite": True, "max": 1.0412840843200684, "min": -0.5786740779876709},
        ],
        "first_two_max_abs_upper": 1.1748055219650269,
    },
]


def _is_finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _range_max_abs(action_range: dict[str, Any]) -> float:
    return max(abs(float(action_range["min"])), abs(float(action_range["max"])))


def _all_numbers_finite(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_all_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_all_numbers_finite(item) for item in value)
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    return True


def _max_observed_first_two_abs() -> float:
    return max(float(item["first_two_max_abs_upper"]) for item in XVLA_ACTION_RANGE_EVIDENCE)


def build_sgl_action_bounds_gate() -> dict[str, Any]:
    """Build the deterministic action-bias bounds/no-optimizer gate."""

    stage0 = build_sgl_xvla_stage0_gate()
    stage0_errors = validate_sgl_xvla_stage0_gate(stage0)
    observability = build_sgl_observability_audit()
    observability_errors = validate_sgl_observability_audit(observability)
    expert_stats = dict(EXPERT_SELECTED_DEMO_ACTION_STATS)
    expert_translation_max_abs = float(expert_stats["translation_range"]["max_abs"])
    expert_rotation_max_abs = float(expert_stats["rotation_range"]["max_abs"])
    expert_gripper_max_abs = float(expert_stats["gripper_range"]["max_abs"])
    max_observed_first_two_abs = _max_observed_first_two_abs()

    action_bounds = {
        "action_dim": 7,
        "bias_type": "support-gated no-optimizer residual action bias",
        "max_activated_chunks": 2,
        "assumed_chunk_size_steps": 30,
        "max_activated_steps": 60,
        "activation_window": "first one or two X-VLA chunks only",
        "enabled_bias_dimensions": [
            "single lift-axis translation component",
            "gripper open/close component",
        ],
        "disabled_bias_dimensions": [
            "lateral translation components",
            "roll/pitch/yaw rotation components",
        ],
        "lift_axis_translation_bias_max_abs_per_step": 0.20,
        "lateral_translation_bias_max_abs_per_step": 0.0,
        "rotation_bias_max_abs_per_step": 0.0,
        "gripper_bias_max_abs_per_step": 0.25,
        "post_bias_action_clamp_min": -1.0,
        "post_bias_action_clamp_max": 1.0,
        "post_bias_action_clamp_abs": 1.0,
        "finite_bias_required": True,
        "nan_or_inf_policy": "fail_closed_before_any_rollout",
        "no_learned_parameters": True,
        "no_optimizer": True,
    }

    lift_bound = float(action_bounds["lift_axis_translation_bias_max_abs_per_step"])
    lateral_bound = float(action_bounds["lateral_translation_bias_max_abs_per_step"])
    rotation_bound = float(action_bounds["rotation_bias_max_abs_per_step"])
    gripper_bound = float(action_bounds["gripper_bias_max_abs_per_step"])
    bounds_pass_expert_envelope = (
        lift_bound <= expert_translation_max_abs * 0.25
        and lateral_bound <= expert_translation_max_abs * 0.25
        and rotation_bound <= expert_rotation_max_abs * 0.25
        and gripper_bound <= expert_gripper_max_abs * 0.25
    )
    known_raw_xvla_exceeds_env_bound = max_observed_first_two_abs > 1.0

    decision = (
        "SGL_ACTION_BIAS_BOUNDS_FROZEN_POST_CLAMP_NO_OPTIMIZER_NO_TRAINING"
        if not stage0_errors
        and not observability_errors
        and bounds_pass_expert_envelope
        and expert_stats["finite"]
        and _all_numbers_finite(action_bounds)
        else "SGL_ACTION_BIAS_BOUNDS_NOT_VERIFIED"
    )

    return {
        "schema_version": "2026-07-18.epoch5_sgl_xvla_action_bounds_gate.v1",
        "stage": "epoch_5_sgl_xvla_task5_stage0_action_bounds_report_only",
        "candidate_id": CANDIDATE_ID,
        "decision": decision,
        "target": {
            "suite": TARGET_SUITE,
            "task_id": TARGET_TASK_ID,
            "instruction": TARGET_INSTRUCTION,
            "residual_identities": list(RESIDUAL_IDENTITIES),
            "clean_retention_identities": list(CLEAN_RETENTION_IDENTITIES),
            "held_out_confirmatory_identity_pool": list(HELD_OUT_CONFIRMATORY_IDENTITY_POOL),
        },
        "upstream_gates": {
            "stage0_artifact": str(STAGE0_ARTIFACT),
            "stage0_validation_errors": stage0_errors,
            "stage0_valid": not stage0_errors,
            "observability_artifact": str(OBSERVABILITY_ARTIFACT),
            "observability_validation_errors": observability_errors,
            "observability_valid": not observability_errors,
            "observability_decision": observability["decision"],
        },
        "expert_action_evidence": {
            "result_paths": list(EXPERT_HEADROOM_RESULT_PATHS),
            "headroom_records": list(EXPERT_HEADROOM_EVIDENCE),
            "selected_demo_action_stats": expert_stats,
            "same_reset_headroom_available": False,
            "task_level_expert_headroom_positive": True,
            "clip_rate_if_env_adapter_applied": expert_stats["clip_rate_if_env_adapter_applied"],
        },
        "xvla_action_range_evidence": {
            "result_paths": list(XVLA_ACTION_RANGE_RESULT_PATHS),
            "records": list(XVLA_ACTION_RANGE_EVIDENCE),
            "max_observed_first_two_abs": max_observed_first_two_abs,
            "known_raw_xvla_first_two_chunks_exceed_env_bound": known_raw_xvla_exceeds_env_bound,
            "interpretation": (
                "Existing X-VLA chunk metadata already exceeds +1.0 in the first "
                "two chunks, so any future executable SGL policy must clamp after "
                "bias composition and record pre/post-clamp saturation."
            ),
        },
        "frozen_action_bias_bounds": action_bounds,
        "saturation_guard": {
            "post_bias_action_clamp_required": True,
            "post_bias_action_clamp_abs": 1.0,
            "suppress_bias_if_component_already_saturated": True,
            "component_saturation_abs_threshold": 0.98,
            "forbid_bias_that_increases_existing_saturation": True,
            "record_pre_bias_component_actions_before_any_rollout": True,
            "record_post_bias_component_actions_before_any_rollout": True,
            "record_added_clip_count_by_identity_before_any_rollout": True,
            "fail_if_clean_retention_added_clip_count_positive": True,
        },
        "bounds_audit": {
            "expert_translation_max_abs": expert_translation_max_abs,
            "expert_rotation_max_abs": expert_rotation_max_abs,
            "expert_gripper_max_abs": expert_gripper_max_abs,
            "lift_bound_fraction_of_expert_translation_max_abs": round(lift_bound / expert_translation_max_abs, 9),
            "rotation_bound_fraction_of_expert_rotation_max_abs": 0.0,
            "gripper_bound_fraction_of_expert_gripper_max_abs": round(gripper_bound / expert_gripper_max_abs, 9),
            "bounds_pass_expert_envelope": bounds_pass_expert_envelope,
            "all_bounds_finite": _all_numbers_finite(action_bounds),
            "max_observed_first_two_abs": max_observed_first_two_abs,
            "known_raw_xvla_exceeds_env_bound": known_raw_xvla_exceeds_env_bound,
        },
        "comparator_role_implications": [
            {
                "comparator_role": "BASE",
                "scientific_question": "Would bounded SGL improve residual task5 success over unmodified X-VLA/Base?",
                "claim_metric": "paired residual and held-out success after a future frozen rollout protocol",
                "blocking_condition": "No residual gain or clean-retention degradation outside the frozen margin.",
                "does_this_gate_answer_it": False,
            },
            {
                "comparator_role": "SIMPLE_CONTROL",
                "scientific_question": "Can a fixed lift/regrasp template explain any future gain?",
                "claim_metric": "residual success, clean retention, latency, and intervention count",
                "blocking_condition": "The fixed-lift control matches SGL at equal or lower cost.",
                "does_this_gate_answer_it": False,
            },
            {
                "comparator_role": "CLEAN_RETENTION",
                "scientific_question": "Does the support-gated action bias preserve identities X-VLA already solves?",
                "claim_metric": "success and added-clipping count on identities 20260731 and 20260732",
                "blocking_condition": "Any clean-retention failure or positive added clipping under the frozen pre-rollout guard.",
                "does_this_gate_answer_it": False,
            },
        ],
        "execution_classification": {
            "execution_type": "REPORT_ONLY",
            "evidence_role": "OURS_CANDIDATE_STAGE0_ACTION_BOUNDS_AUDIT",
            "artifact_status": "NOT_APPLICABLE",
            "simulator_episode_count": 0,
            "vla_model_loaded": False,
            "training_happened": False,
            "optimizer_step_happened": False,
            "checkpoint_written": False,
            "closed_loop_ours_evaluation_happened": False,
            "lora_or_qlora_training_happened": False,
        },
        "no_training_no_ours_booleans": {
            "training_happened": False,
            "optimizer_step_happened": False,
            "checkpoint_written": False,
            "closed_loop_ours_evaluation_happened": False,
            "ours_rollout_happened": False,
            "lora_or_qlora_training_happened": False,
        },
        "bounded_conclusion": {
            "action_bounds_frozen": decision
            == "SGL_ACTION_BIAS_BOUNDS_FROZEN_POST_CLAMP_NO_OPTIMIZER_NO_TRAINING",
            "bounds_pass_expert_envelope": bounds_pass_expert_envelope,
            "known_xvla_saturation_requires_post_clamp": known_raw_xvla_exceeds_env_bound,
            "candidate_can_advance_to_simple_fixed_lift_control_gate": decision
            == "SGL_ACTION_BIAS_BOUNDS_FROZEN_POST_CLAMP_NO_OPTIMIZER_NO_TRAINING",
            "candidate_can_train": False,
            "candidate_can_roll_out_ours": False,
        },
        "next_action": (
            "Freeze the simple fixed-lift/regrasp control before any SGL-XVLA "
            "method execution. Do not train, write checkpoints, or run Ours."
        ),
    }


def validate_sgl_action_bounds_gate(gate: dict[str, Any]) -> list[str]:
    """Return validation errors for an action-bias bounds gate."""

    errors: list[str] = []
    if gate.get("candidate_id") != CANDIDATE_ID:
        errors.append("candidate_id must be SGL-XVLA")
    upstream = gate.get("upstream_gates", {})
    if upstream.get("stage0_valid") is not True:
        errors.append("Stage 0 gate must validate")
    if upstream.get("observability_valid") is not True:
        errors.append("observability audit must validate")
    target = gate.get("target", {})
    if target.get("residual_identities") != RESIDUAL_IDENTITIES:
        errors.append("residual identities must remain frozen")
    if target.get("clean_retention_identities") != CLEAN_RETENTION_IDENTITIES:
        errors.append("clean-retention identities must remain required")
    bounds = gate.get("frozen_action_bias_bounds", {})
    required_bound_keys = [
        "max_activated_chunks",
        "max_activated_steps",
        "lift_axis_translation_bias_max_abs_per_step",
        "lateral_translation_bias_max_abs_per_step",
        "rotation_bias_max_abs_per_step",
        "gripper_bias_max_abs_per_step",
        "post_bias_action_clamp_abs",
    ]
    for key in required_bound_keys:
        if not _is_finite_number(bounds.get(key)):
            errors.append(f"{key} must be finite")
    if bounds.get("max_activated_chunks") != 2:
        errors.append("max_activated_chunks must be exactly two")
    if bounds.get("max_activated_steps") != 60:
        errors.append("max_activated_steps must be exactly sixty")
    if bounds.get("post_bias_action_clamp_abs") != 1.0:
        errors.append("post-bias clamp must be exactly abs 1.0")
    if bounds.get("no_optimizer") is not True:
        errors.append("bounds gate must freeze no_optimizer true")
    if bounds.get("no_learned_parameters") is not True:
        errors.append("bounds gate must not introduce learned parameters")
    expert = gate.get("expert_action_evidence", {}).get("selected_demo_action_stats", {})
    if expert.get("finite") is not True:
        errors.append("expert action stats must be finite")
    if expert.get("clip_rate_if_env_adapter_applied") != 0.0:
        errors.append("expert env-adapter clip rate must be zero")
    try:
        translation_max_abs = _range_max_abs(expert["translation_range"])
        rotation_max_abs = _range_max_abs(expert["rotation_range"])
        gripper_max_abs = _range_max_abs(expert["gripper_range"])
        if bounds["lift_axis_translation_bias_max_abs_per_step"] > translation_max_abs * 0.25:
            errors.append("lift-axis bias exceeds 25% of expert translation range")
        if bounds["lateral_translation_bias_max_abs_per_step"] > translation_max_abs * 0.25:
            errors.append("lateral bias exceeds 25% of expert translation range")
        if bounds["rotation_bias_max_abs_per_step"] > rotation_max_abs * 0.25:
            errors.append("rotation bias exceeds 25% of expert rotation range")
        if bounds["gripper_bias_max_abs_per_step"] > gripper_max_abs * 0.25:
            errors.append("gripper bias exceeds 25% of expert gripper range")
    except (KeyError, TypeError, ValueError):
        errors.append("expert range fields are malformed")
    action_records = gate.get("xvla_action_range_evidence", {}).get("records", [])
    identity_set = {item.get("reset_identity") for item in action_records}
    if not set(CLEAN_RETENTION_IDENTITIES).issubset(identity_set):
        errors.append("clean-retention action-range evidence is missing")
    if not {20260730, 20260733}.issubset(identity_set):
        errors.append("residual failure action-range evidence is missing")
    if gate.get("xvla_action_range_evidence", {}).get("known_raw_xvla_first_two_chunks_exceed_env_bound") is not True:
        errors.append("known raw X-VLA saturation must be recorded")
    saturation = gate.get("saturation_guard", {})
    for key in [
        "post_bias_action_clamp_required",
        "suppress_bias_if_component_already_saturated",
        "forbid_bias_that_increases_existing_saturation",
        "record_pre_bias_component_actions_before_any_rollout",
        "record_post_bias_component_actions_before_any_rollout",
        "record_added_clip_count_by_identity_before_any_rollout",
        "fail_if_clean_retention_added_clip_count_positive",
    ]:
        if saturation.get(key) is not True:
            errors.append(f"{key} must be true")
    execution = gate.get("execution_classification", {})
    for key in [
        "vla_model_loaded",
        "training_happened",
        "optimizer_step_happened",
        "checkpoint_written",
        "closed_loop_ours_evaluation_happened",
        "lora_or_qlora_training_happened",
    ]:
        if execution.get(key) is not False:
            errors.append(f"{key} must be false")
    if execution.get("simulator_episode_count") != 0:
        errors.append("simulator_episode_count must be zero")
    conclusion = gate.get("bounded_conclusion", {})
    if conclusion.get("action_bounds_frozen") is not True:
        errors.append("action bounds must be frozen")
    if conclusion.get("candidate_can_train") is not False:
        errors.append("candidate cannot train after action-bounds gate")
    if conclusion.get("candidate_can_roll_out_ours") is not False:
        errors.append("candidate cannot roll out Ours after action-bounds gate")
    return errors


def write_sgl_action_bounds_gate(output_path: Path) -> dict[str, Any]:
    """Build, validate, and write the action-bounds gate JSON."""

    gate = build_sgl_action_bounds_gate()
    errors = validate_sgl_action_bounds_gate(gate)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ACTION_BOUNDS_ARTIFACT)
    args = parser.parse_args()
    gate = write_sgl_action_bounds_gate(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": gate["decision"],
                "action_bounds_frozen": gate["bounded_conclusion"]["action_bounds_frozen"],
                "candidate_can_train": gate["bounded_conclusion"]["candidate_can_train"],
                "candidate_can_roll_out_ours": gate["bounded_conclusion"]["candidate_can_roll_out_ours"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
