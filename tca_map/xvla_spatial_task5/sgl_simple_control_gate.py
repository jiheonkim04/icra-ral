"""Report-only simple-control freeze for SGL-XVLA Stage 0.

This module preregisters exactly one strongest simple explanation for any future
SGL-XVLA gain: a nonadaptive, language-gated fixed lift/regrasp template that
uses the same conservative action bounds frozen in the action-bounds gate. It
does not train, load a model, launch a simulator, write checkpoints, or evaluate
Ours or the control.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

from tca_map.xvla_spatial_task5.sgl_action_bounds_gate import (
    ACTION_BOUNDS_ARTIFACT,
    build_sgl_action_bounds_gate,
    validate_sgl_action_bounds_gate,
)
from tca_map.xvla_spatial_task5.sgl_stage0_gate import (
    CANDIDATE_ID,
    CLEAN_RETENTION_IDENTITIES,
    HELD_OUT_CONFIRMATORY_IDENTITY_POOL,
    RESIDUAL_IDENTITIES,
    TARGET_INSTRUCTION,
    TARGET_SUITE,
    TARGET_TASK_ID,
)


SIMPLE_CONTROL_ARTIFACT = Path("runs/xvla_prior/epoch5_sgl_xvla_task5_simple_control_gate_v1.json")
CONTROL_ID = "FIXED-LIFT-REGRASP-CONTROL"


def _finite_abs_leq(value: Any, upper: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value)) and abs(float(value)) <= upper


def build_sgl_simple_control_gate() -> dict[str, Any]:
    """Build the deterministic fixed-lift/regrasp simple-control freeze."""

    action_gate = build_sgl_action_bounds_gate()
    action_gate_errors = validate_sgl_action_bounds_gate(action_gate)
    bounds = action_gate["frozen_action_bias_bounds"]
    lift_bound = float(bounds["lift_axis_translation_bias_max_abs_per_step"])
    gripper_bound = float(bounds["gripper_bias_max_abs_per_step"])
    lateral_bound = float(bounds["lateral_translation_bias_max_abs_per_step"])
    rotation_bound = float(bounds["rotation_bias_max_abs_per_step"])

    template_schedule = [
        {
            "chunk_index": 0,
            "lift_axis_translation_bias_abs": lift_bound,
            "gripper_close_bias_abs": gripper_bound,
            "lateral_translation_bias_abs": lateral_bound,
            "rotation_bias_abs": rotation_bound,
        },
        {
            "chunk_index": 1,
            "lift_axis_translation_bias_abs": lift_bound,
            "gripper_close_bias_abs": gripper_bound,
            "lateral_translation_bias_abs": lateral_bound,
            "rotation_bias_abs": rotation_bound,
        },
    ]
    schedule_within_bounds = all(
        _finite_abs_leq(item["lift_axis_translation_bias_abs"], lift_bound)
        and _finite_abs_leq(item["gripper_close_bias_abs"], gripper_bound)
        and _finite_abs_leq(item["lateral_translation_bias_abs"], lateral_bound)
        and _finite_abs_leq(item["rotation_bias_abs"], rotation_bound)
        for item in template_schedule
    )
    decision = (
        "SGL_SIMPLE_FIXED_LIFT_REGRASP_CONTROL_FROZEN_NO_TRAINING_NO_OURS"
        if not action_gate_errors and schedule_within_bounds
        else "SGL_SIMPLE_FIXED_LIFT_REGRASP_CONTROL_NOT_VERIFIED"
    )

    return {
        "schema_version": "2026-07-18.epoch5_sgl_xvla_simple_control_gate.v1",
        "stage": "epoch_5_sgl_xvla_task5_stage0_simple_control_report_only",
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
            "action_bounds_artifact": str(ACTION_BOUNDS_ARTIFACT),
            "action_bounds_validation_errors": action_gate_errors,
            "action_bounds_valid": not action_gate_errors,
            "action_bounds_decision": action_gate["decision"],
        },
        "simple_control": {
            "control_id": CONTROL_ID,
            "comparator_role": "SIMPLE_CONTROL",
            "is_primary_simple_control": True,
            "other_simple_controls_frozen_for_same_objection": [],
            "strongest_simple_explanation_tested": (
                "A nonadaptive fixed lift/regrasp template, activated by the same "
                "language-level support condition, might account for any future "
                "support-separation gain without SGL's claimed mechanism."
            ),
            "activation_condition": {
                "condition": "task instruction contains the support token `ramekin`",
                "allowed_activation_sources": ["language instruction"],
                "uses_visual_progress_feedback": False,
                "uses_simulator_state": False,
                "uses_reward_or_success": False,
                "uses_reset_identity": False,
                "activates_on_residual_identities": list(RESIDUAL_IDENTITIES),
                "activates_on_clean_retention_identities": list(CLEAN_RETENTION_IDENTITIES),
            },
            "template": {
                "template_type": "fixed_open_loop_lift_regrasp_bias",
                "nonadaptive": True,
                "max_activated_chunks": bounds["max_activated_chunks"],
                "max_activated_steps": bounds["max_activated_steps"],
                "schedule": template_schedule,
                "zero_bias_after_chunk_index": 1,
                "post_bias_action_clamp_abs": bounds["post_bias_action_clamp_abs"],
                "saturation_guard_inherited": True,
                "axis_and_sign_binding_rule": (
                    "Bind lift-axis and gripper-close directions from the official "
                    "LIBERO/X-VLA action adapter before any rollout. If the binding "
                    "cannot be verified from source semantics, fail closed; never "
                    "choose signs from residual or clean-retention outcomes."
                ),
            },
            "cost_accounting_fields_required_if_later_executed": [
                "activation_count",
                "intervention_step_count",
                "pre_bias_component_actions",
                "post_bias_component_actions",
                "added_clip_count_by_identity",
                "latency_overhead_seconds",
            ],
        },
        "comparator_role_calibration": {
            "scientific_question": "Can a trivial fixed lift/regrasp explain substantially all future SGL gain?",
            "claim_metric": "paired residual success, clean retention, added clipping, latency, and intervention count",
            "blocking_condition": (
                "The simple control blocks the SGL novelty claim only if it matches "
                "or exceeds SGL on the primary residual claim at equal/lower cost, "
                "without meaningful clean-retention or generalization loss."
            ),
            "nonblocking_tradeoffs": (
                "A control that wins one isolated development identity but loses "
                "the matched aggregate, clean retention, held-out identities, or "
                "efficiency-adjusted result does not automatically explain SGL."
            ),
            "universal_beat_all_rule_applied": False,
        },
        "execution_classification": {
            "execution_type": "REPORT_ONLY",
            "evidence_role": "SIMPLE_CONTROL_PREREGISTRATION",
            "artifact_status": "NOT_APPLICABLE",
            "simulator_episode_count": 0,
            "vla_model_loaded": False,
            "training_happened": False,
            "optimizer_step_happened": False,
            "checkpoint_written": False,
            "control_rollout_happened": False,
            "closed_loop_ours_evaluation_happened": False,
            "lora_or_qlora_training_happened": False,
        },
        "no_training_no_ours_booleans": {
            "training_happened": False,
            "optimizer_step_happened": False,
            "checkpoint_written": False,
            "control_rollout_happened": False,
            "closed_loop_ours_evaluation_happened": False,
            "ours_rollout_happened": False,
            "lora_or_qlora_training_happened": False,
        },
        "bounded_conclusion": {
            "simple_control_frozen": decision
            == "SGL_SIMPLE_FIXED_LIFT_REGRASP_CONTROL_FROZEN_NO_TRAINING_NO_OURS",
            "exactly_one_simple_control_for_this_objection": True,
            "template_within_action_bounds": schedule_within_bounds,
            "candidate_can_advance_to_held_out_identity_manifest_gate": decision
            == "SGL_SIMPLE_FIXED_LIFT_REGRASP_CONTROL_FROZEN_NO_TRAINING_NO_OURS",
            "candidate_can_train": False,
            "candidate_can_roll_out_ours": False,
            "control_can_roll_out_now": False,
        },
        "next_action": (
            "Freeze the held-out identity/development manifest for SGL-XVLA before "
            "any control or Ours rollout. Do not train, write checkpoints, or run Ours."
        ),
    }


def validate_sgl_simple_control_gate(gate: dict[str, Any]) -> list[str]:
    """Return validation errors for the fixed-lift/regrasp control freeze."""

    errors: list[str] = []
    if gate.get("candidate_id") != CANDIDATE_ID:
        errors.append("candidate_id must be SGL-XVLA")
    upstream = gate.get("upstream_gates", {})
    if upstream.get("action_bounds_valid") is not True:
        errors.append("action-bounds gate must validate")
    control = gate.get("simple_control", {})
    if control.get("control_id") != CONTROL_ID:
        errors.append("control_id must be the fixed lift/regrasp control")
    if control.get("is_primary_simple_control") is not True:
        errors.append("the fixed lift/regrasp control must be primary")
    if control.get("other_simple_controls_frozen_for_same_objection") != []:
        errors.append("do not freeze redundant simple controls for the same objection")
    activation = control.get("activation_condition", {})
    for key in [
        "uses_visual_progress_feedback",
        "uses_simulator_state",
        "uses_reward_or_success",
        "uses_reset_identity",
    ]:
        if activation.get(key) is not False:
            errors.append(f"{key} must be false")
    if activation.get("activates_on_clean_retention_identities") != CLEAN_RETENTION_IDENTITIES:
        errors.append("simple control must activate on clean-retention identities")
    target = gate.get("target", {})
    if target.get("residual_identities") != RESIDUAL_IDENTITIES:
        errors.append("residual identities must remain frozen")
    if target.get("clean_retention_identities") != CLEAN_RETENTION_IDENTITIES:
        errors.append("clean-retention identities must remain required")
    template = control.get("template", {})
    if template.get("nonadaptive") is not True:
        errors.append("simple control must be nonadaptive")
    if template.get("max_activated_chunks") != 2:
        errors.append("simple control must activate for at most two chunks")
    if template.get("max_activated_steps") != 60:
        errors.append("simple control must activate for at most sixty steps")
    if template.get("zero_bias_after_chunk_index") != 1:
        errors.append("simple control must zero bias after chunk index 1")
    if template.get("post_bias_action_clamp_abs") != 1.0:
        errors.append("simple control must inherit post-bias clamp abs 1.0")
    if template.get("saturation_guard_inherited") is not True:
        errors.append("simple control must inherit saturation guard")
    schedule = template.get("schedule", [])
    if [item.get("chunk_index") for item in schedule] != [0, 1]:
        errors.append("simple control schedule must cover exactly chunks 0 and 1")
    for item in schedule:
        if not _finite_abs_leq(item.get("lift_axis_translation_bias_abs"), 0.20):
            errors.append("lift-axis template bias exceeds frozen bound")
        if not _finite_abs_leq(item.get("gripper_close_bias_abs"), 0.25):
            errors.append("gripper template bias exceeds frozen bound")
        if item.get("lateral_translation_bias_abs") != 0.0:
            errors.append("lateral template bias must be zero")
        if item.get("rotation_bias_abs") != 0.0:
            errors.append("rotation template bias must be zero")
    calibration = gate.get("comparator_role_calibration", {})
    if calibration.get("universal_beat_all_rule_applied") is not False:
        errors.append("simple-control calibration must not apply a universal beat-all rule")
    execution = gate.get("execution_classification", {})
    for key in [
        "vla_model_loaded",
        "training_happened",
        "optimizer_step_happened",
        "checkpoint_written",
        "control_rollout_happened",
        "closed_loop_ours_evaluation_happened",
        "lora_or_qlora_training_happened",
    ]:
        if execution.get(key) is not False:
            errors.append(f"{key} must be false")
    if execution.get("simulator_episode_count") != 0:
        errors.append("simulator_episode_count must be zero")
    conclusion = gate.get("bounded_conclusion", {})
    if conclusion.get("simple_control_frozen") is not True:
        errors.append("simple control must be frozen")
    if conclusion.get("exactly_one_simple_control_for_this_objection") is not True:
        errors.append("exactly one simple control must be frozen")
    if conclusion.get("candidate_can_train") is not False:
        errors.append("candidate cannot train after simple-control gate")
    if conclusion.get("candidate_can_roll_out_ours") is not False:
        errors.append("candidate cannot roll out Ours after simple-control gate")
    if conclusion.get("control_can_roll_out_now") is not False:
        errors.append("simple-control gate does not authorize control rollout")
    return errors


def write_sgl_simple_control_gate(output_path: Path) -> dict[str, Any]:
    """Build, validate, and write the simple-control gate JSON."""

    gate = build_sgl_simple_control_gate()
    errors = validate_sgl_simple_control_gate(gate)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=SIMPLE_CONTROL_ARTIFACT)
    args = parser.parse_args()
    gate = write_sgl_simple_control_gate(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": gate["decision"],
                "simple_control_frozen": gate["bounded_conclusion"]["simple_control_frozen"],
                "candidate_can_train": gate["bounded_conclusion"]["candidate_can_train"],
                "candidate_can_roll_out_ours": gate["bounded_conclusion"]["candidate_can_roll_out_ours"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
