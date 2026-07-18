"""Report-only runner preflight for the frozen SGL-XVLA rollout protocol.

The preflight checks whether the currently frozen SGL-XVLA executable behavior
would be meaningfully distinct from the fixed lift/regrasp simple control before
any simulator episode is launched. It does not load a model, train, write
checkpoints, run the control, or run Ours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tca_map.xvla_spatial_task5.sgl_rollout_protocol import (
    ROLLOUT_PROTOCOL_ARTIFACT,
    build_sgl_rollout_protocol,
    validate_sgl_rollout_protocol,
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


RUNNER_PREFLIGHT_ARTIFACT = Path("runs/xvla_prior/epoch5_sgl_xvla_task5_runner_preflight_v1.json")

ACTION_BINDING_EVIDENCE = [
    {
        "path": "scripts/run_phase_barrier_vla_prototype.py",
        "line": 201,
        "evidence": "contact_z_boost adds to action dimension 2 before clipping",
        "supports": "lift_axis_dimension_index_2",
    },
    {
        "path": "scripts/run_s2c_vla_stage0.py",
        "line": 304,
        "evidence": "gripper convention records LIBERO/SmolVLA checkpoint 7D action dimension 6",
        "supports": "gripper_dimension_index_6",
    },
    {
        "path": "scripts/run_tsc_vla_stage0.py",
        "line": 765,
        "evidence": "gripper convention records LIBERO/SmolVLA checkpoint action dimension 6",
        "supports": "gripper_dimension_index_6",
    },
]


def _canonical_template() -> dict[str, Any]:
    return {
        "activation_condition": "language_instruction_contains_ramekin",
        "allowed_activation_sources": ["language instruction"],
        "uses_visual_progress_feedback": False,
        "uses_simulator_state": False,
        "uses_reward_or_success": False,
        "uses_reset_identity": False,
        "max_activated_chunks": 2,
        "max_activated_steps": 60,
        "schedule": [
            {
                "chunk_index": 0,
                "lift_axis_translation_bias_abs": 0.2,
                "gripper_close_bias_abs": 0.25,
                "lateral_translation_bias_abs": 0.0,
                "rotation_bias_abs": 0.0,
            },
            {
                "chunk_index": 1,
                "lift_axis_translation_bias_abs": 0.2,
                "gripper_close_bias_abs": 0.25,
                "lateral_translation_bias_abs": 0.0,
                "rotation_bias_abs": 0.0,
            },
        ],
        "zero_bias_after_chunk_index": 1,
        "post_bias_action_clamp_abs": 1.0,
        "saturation_guard": "suppress_bias_if_component_already_saturated_and_forbid_added_clean_clipping",
    }


def templates_equivalent(left: dict[str, Any], right: dict[str, Any]) -> bool:
    """Return whether two frozen behavior templates are exactly equivalent."""

    return left == right


def build_sgl_runner_preflight() -> dict[str, Any]:
    """Build the deterministic report-only runner preflight."""

    protocol = build_sgl_rollout_protocol()
    protocol_errors = validate_sgl_rollout_protocol(protocol)
    sgl_template = _canonical_template()
    simple_control_template = _canonical_template()
    equivalent_to_simple_control = templates_equivalent(sgl_template, simple_control_template)
    missing_distinct_component_reasons = [
        "visual/progress detector remains unverified",
        "only language-level support activation is frozen",
        "no learned, adaptive, or observation-conditioned residual component is authorized",
        "fixed lift/regrasp schedule equals the primary simple-control schedule",
        "same saturation guard and post-bias clamp are inherited",
    ]
    decision = (
        "SGL_RUNNER_PREFLIGHT_BLOCKED_SIMPLE_CONTROL_EQUIVALENCE_NO_ROLLOUT"
        if not protocol_errors and equivalent_to_simple_control
        else "SGL_RUNNER_PREFLIGHT_NOT_VERIFIED"
    )

    return {
        "schema_version": "2026-07-18.epoch5_sgl_xvla_runner_preflight.v1",
        "stage": "epoch_5_sgl_xvla_task5_runner_preflight_report_only",
        "candidate_id": CANDIDATE_ID,
        "decision": decision,
        "target": {
            "suite": TARGET_SUITE,
            "task_id": TARGET_TASK_ID,
            "instruction": TARGET_INSTRUCTION,
            "development_residual_identities": list(RESIDUAL_IDENTITIES),
            "clean_retention_identities": list(CLEAN_RETENTION_IDENTITIES),
            "held_out_confirmatory_identity_pool": list(HELD_OUT_CONFIRMATORY_IDENTITY_POOL),
        },
        "upstream_protocol": {
            "artifact": str(ROLLOUT_PROTOCOL_ARTIFACT),
            "validation_errors": protocol_errors,
            "valid": not protocol_errors,
            "decision": protocol["decision"],
        },
        "action_binding_preflight": {
            "lift_axis_dimension_index": 2,
            "gripper_dimension_index": 6,
            "binding_evidence": list(ACTION_BINDING_EVIDENCE),
            "binding_source_semantics_required_before_any_future_runner": True,
            "outcome_tuned_binding_allowed": False,
        },
        "static_behavior_equivalence": {
            "sgl_frozen_template": sgl_template,
            "fixed_control_template": simple_control_template,
            "templates_equivalent_under_current_frozen_stage0": equivalent_to_simple_control,
            "missing_distinct_sgl_component_reasons": missing_distinct_component_reasons,
            "scientific_meaning": (
                "Under the currently frozen language-only activation and fixed "
                "bounded lift/regrasp schedule, SGL-XVLA has no executable behavior "
                "not already represented by the fixed lift/regrasp simple control."
            ),
        },
        "comparator_role_statuses": {
            "BASE_CLAIM_STATUS": "NOT_TESTED_PREFLIGHT_BLOCK",
            "PRIOR_ADVANCE_STATUS": "NOT_TESTED_PREFLIGHT_BLOCK",
            "ABLATION_COMPONENT_STATUS": "KEY_COMPONENT_NOT_SUPPORTED",
            "SIMPLE_EXPLANATION_STATUS": "SIMPLE_CONTROL_EXPLAINS_GAIN",
            "CLEAN_RETENTION_STATUS": "NOT_TESTED_PREFLIGHT_BLOCK",
            "GENERALIZATION_STATUS": "NOT_TESTED_PREFLIGHT_BLOCK",
            "OVERALL_PAPER_CANDIDATE_STATUS": "SIMPLE_CONTROL_EXPLAINS_GAIN",
            "status_scope": (
                "Static preflight block only: no simulator success result exists. "
                "The simple control explains the currently frozen executable "
                "behavior before any rollout."
            ),
        },
        "execution_classification": {
            "execution_type": "REPORT_ONLY",
            "evidence_role": "RUNNER_PREFLIGHT_STATIC_EQUIVALENCE_AUDIT",
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
        "authorization_boundary": {
            "runner_preflight_completed": decision
            == "SGL_RUNNER_PREFLIGHT_BLOCKED_SIMPLE_CONTROL_EQUIVALENCE_NO_ROLLOUT",
            "primary_sgl_xvla_current_frozen_executable_blocked": equivalent_to_simple_control,
            "simulator_episode_authorized": False,
            "control_rollout_authorized": False,
            "ours_rollout_authorized": False,
            "held_out_rollout_authorized": False,
            "training_authorized": False,
            "checkpoint_write_authorized": False,
            "paper_candidate_go": False,
            "prototype_go": False,
            "backup_candidate_stage0_authorized_next": decision
            == "SGL_RUNNER_PREFLIGHT_BLOCKED_SIMPLE_CONTROL_EQUIVALENCE_NO_ROLLOUT",
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
            "preflight_passed_for_rollout": False,
            "blocked_before_simulator_episode": True,
            "blocking_reason": "SGL frozen executable is behaviorally equivalent to the fixed simple control",
            "current_sgl_candidate_killed": True,
            "kill_scope": "current frozen SGL-XVLA executable only; does not reopen archived R2P-XVLA",
            "candidate_can_train": False,
            "candidate_can_roll_out_ours": False,
            "control_can_roll_out_now": False,
            "backup_ocr_xvla_can_start_stage0": decision
            == "SGL_RUNNER_PREFLIGHT_BLOCKED_SIMPLE_CONTROL_EQUIVALENCE_NO_ROLLOUT",
        },
        "next_action": (
            "Do not run SGL-XVLA. Start Stage 0 report-only gating for the backup "
            "OCR-XVLA candidate generated in the frozen candidate report, with no "
            "training, checkpoints, simulator episode, or Ours rollout until its "
            "own gates authorize a protocol."
        ),
    }


def validate_sgl_runner_preflight(preflight: dict[str, Any]) -> list[str]:
    """Return validation errors for the runner preflight."""

    errors: list[str] = []
    if preflight.get("candidate_id") != CANDIDATE_ID:
        errors.append("candidate_id must be SGL-XVLA")
    upstream = preflight.get("upstream_protocol", {})
    if upstream.get("valid") is not True:
        errors.append("upstream rollout protocol must validate")
    equivalence = preflight.get("static_behavior_equivalence", {})
    if equivalence.get("templates_equivalent_under_current_frozen_stage0") is not True:
        errors.append("preflight should detect simple-control equivalence")
    if equivalence.get("sgl_frozen_template") != equivalence.get("fixed_control_template"):
        errors.append("frozen SGL and fixed-control templates must match for this block")
    if len(equivalence.get("missing_distinct_sgl_component_reasons", [])) < 3:
        errors.append("missing distinct component reasons are incomplete")
    binding = preflight.get("action_binding_preflight", {})
    if binding.get("lift_axis_dimension_index") != 2:
        errors.append("lift axis must be dimension index 2 under local source evidence")
    if binding.get("gripper_dimension_index") != 6:
        errors.append("gripper must be dimension index 6 under local source evidence")
    if binding.get("outcome_tuned_binding_allowed") is not False:
        errors.append("outcome-tuned binding must be forbidden")
    statuses = preflight.get("comparator_role_statuses", {})
    if statuses.get("SIMPLE_EXPLANATION_STATUS") != "SIMPLE_CONTROL_EXPLAINS_GAIN":
        errors.append("simple-control status must block the current SGL executable")
    if statuses.get("OVERALL_PAPER_CANDIDATE_STATUS") != "SIMPLE_CONTROL_EXPLAINS_GAIN":
        errors.append("overall status must not be GO after static equivalence block")
    auth = preflight.get("authorization_boundary", {})
    if auth.get("primary_sgl_xvla_current_frozen_executable_blocked") is not True:
        errors.append("primary SGL executable must be blocked")
    if auth.get("backup_candidate_stage0_authorized_next") is not True:
        errors.append("backup candidate Stage0 should be authorized next")
    for key in [
        "simulator_episode_authorized",
        "control_rollout_authorized",
        "ours_rollout_authorized",
        "held_out_rollout_authorized",
        "training_authorized",
        "checkpoint_write_authorized",
        "paper_candidate_go",
        "prototype_go",
    ]:
        if auth.get(key) is not False:
            errors.append(f"{key} must be false")
    execution = preflight.get("execution_classification", {})
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
    conclusion = preflight.get("bounded_conclusion", {})
    if conclusion.get("preflight_passed_for_rollout") is not False:
        errors.append("preflight must not pass for rollout")
    if conclusion.get("blocked_before_simulator_episode") is not True:
        errors.append("preflight must block before simulator episode")
    if conclusion.get("current_sgl_candidate_killed") is not True:
        errors.append("current frozen SGL candidate must be killed")
    if conclusion.get("candidate_can_train") is not False:
        errors.append("candidate cannot train")
    if conclusion.get("candidate_can_roll_out_ours") is not False:
        errors.append("candidate cannot roll out Ours")
    if conclusion.get("control_can_roll_out_now") is not False:
        errors.append("control cannot roll out now")
    if conclusion.get("backup_ocr_xvla_can_start_stage0") is not True:
        errors.append("backup OCR-XVLA should be the next candidate")
    return errors


def write_sgl_runner_preflight(output_path: Path) -> dict[str, Any]:
    """Build, validate, and write the runner preflight JSON."""

    preflight = build_sgl_runner_preflight()
    errors = validate_sgl_runner_preflight(preflight)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(preflight, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return preflight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RUNNER_PREFLIGHT_ARTIFACT)
    args = parser.parse_args()
    preflight = write_sgl_runner_preflight(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": preflight["decision"],
                "blocked_before_simulator_episode": preflight["bounded_conclusion"][
                    "blocked_before_simulator_episode"
                ],
                "current_sgl_candidate_killed": preflight["bounded_conclusion"]["current_sgl_candidate_killed"],
                "backup_ocr_xvla_can_start_stage0": preflight["bounded_conclusion"][
                    "backup_ocr_xvla_can_start_stage0"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
