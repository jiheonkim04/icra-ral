"""Report-only Stage 0 completion adjudication for SGL-XVLA.

The adjudication aggregates the frozen SGL-XVLA Stage 0 gates and decides only
whether a separate no-training rollout protocol may be frozen next. It does not
train, load a model, launch a simulator, write checkpoints, run a control, or
evaluate Ours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tca_map.xvla_spatial_task5.sgl_action_bounds_gate import (
    ACTION_BOUNDS_ARTIFACT,
    build_sgl_action_bounds_gate,
    validate_sgl_action_bounds_gate,
)
from tca_map.xvla_spatial_task5.sgl_identity_manifest_gate import (
    IDENTITY_MANIFEST_ARTIFACT,
    build_sgl_identity_manifest_gate,
    validate_sgl_identity_manifest_gate,
)
from tca_map.xvla_spatial_task5.sgl_observability_audit import (
    OBSERVABILITY_ARTIFACT,
    build_sgl_observability_audit,
    validate_sgl_observability_audit,
)
from tca_map.xvla_spatial_task5.sgl_simple_control_gate import (
    SIMPLE_CONTROL_ARTIFACT,
    build_sgl_simple_control_gate,
    validate_sgl_simple_control_gate,
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


STAGE0_ADJUDICATION_ARTIFACT = Path("runs/xvla_prior/epoch5_sgl_xvla_task5_stage0_adjudication_v1.json")


def build_sgl_stage0_adjudication() -> dict[str, Any]:
    """Build the deterministic report-only Stage 0 adjudication."""

    stage0 = build_sgl_xvla_stage0_gate()
    observability = build_sgl_observability_audit()
    action_bounds = build_sgl_action_bounds_gate()
    simple_control = build_sgl_simple_control_gate()
    identity_manifest = build_sgl_identity_manifest_gate()
    validation_errors = {
        "stage0_gate": validate_sgl_xvla_stage0_gate(stage0),
        "observability_audit": validate_sgl_observability_audit(observability),
        "action_bounds_gate": validate_sgl_action_bounds_gate(action_bounds),
        "simple_control_gate": validate_sgl_simple_control_gate(simple_control),
        "identity_manifest_gate": validate_sgl_identity_manifest_gate(identity_manifest),
    }
    all_gates_valid = all(not errors for errors in validation_errors.values())
    required_checks = [
        "support_observability_no_training",
        "action_bias_bounds_no_optimizer",
        "simple_fixed_lift_control_frozen",
        "held_out_identity_manifest_frozen",
    ]
    visual_progress_verified = observability["observability"]["visual_progress_observability_verified"]
    decision = (
        "SGL_STAGE0_COMPLETE_PROTOCOL_FREEZE_AUTHORIZED_NO_TRAINING_NO_OURS_ROLLOUT"
        if all_gates_valid
        else "SGL_STAGE0_INCOMPLETE_OR_INVALID"
    )

    return {
        "schema_version": "2026-07-18.epoch5_sgl_xvla_stage0_adjudication.v1",
        "stage": "epoch_5_sgl_xvla_task5_stage0_completion_adjudication_report_only",
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
        "gate_artifacts": {
            "stage0_gate": str(STAGE0_ARTIFACT),
            "observability_audit": str(OBSERVABILITY_ARTIFACT),
            "action_bounds_gate": str(ACTION_BOUNDS_ARTIFACT),
            "simple_control_gate": str(SIMPLE_CONTROL_ARTIFACT),
            "identity_manifest_gate": str(IDENTITY_MANIFEST_ARTIFACT),
        },
        "stage0_check_status": {
            "support_observability_no_training": {
                "status": "PASS_LANGUAGE_LEVEL_ONLY",
                "decision": observability["decision"],
                "support_gate_observability_passed": observability["bounded_conclusion"][
                    "support_gate_observability_passed"
                ],
                "visual_progress_observability_verified": visual_progress_verified,
                "scope_limit": (
                    "This supports only the language-level support gate. A visual "
                    "or progress detector remains unverified and would require a "
                    "separate gate if introduced."
                ),
            },
            "action_bias_bounds_no_optimizer": {
                "status": "PASS_FROZEN_POST_CLAMP_BOUNDS",
                "decision": action_bounds["decision"],
                "max_activated_chunks": action_bounds["frozen_action_bias_bounds"]["max_activated_chunks"],
                "post_bias_action_clamp_abs": action_bounds["frozen_action_bias_bounds"][
                    "post_bias_action_clamp_abs"
                ],
                "known_xvla_saturation_requires_post_clamp": action_bounds["bounded_conclusion"][
                    "known_xvla_saturation_requires_post_clamp"
                ],
            },
            "simple_fixed_lift_control_frozen": {
                "status": "PASS_PRIMARY_SIMPLE_CONTROL_FROZEN",
                "decision": simple_control["decision"],
                "control_id": simple_control["simple_control"]["control_id"],
                "exactly_one_simple_control": simple_control["bounded_conclusion"][
                    "exactly_one_simple_control_for_this_objection"
                ],
            },
            "held_out_identity_manifest_frozen": {
                "status": "PASS_IDENTITY_ROLES_FROZEN",
                "decision": identity_manifest["decision"],
                "development_residual_identities": list(RESIDUAL_IDENTITIES),
                "clean_retention_identities": list(CLEAN_RETENTION_IDENTITIES),
                "held_out_confirmatory_identity_pool": list(HELD_OUT_CONFIRMATORY_IDENTITY_POOL),
            },
        },
        "validation_errors": validation_errors,
        "all_required_stage0_checks_frozen": all_gates_valid,
        "comparator_role_statuses": {
            "BASE_CLAIM_STATUS": "NOT_TESTED_STAGE0_ONLY",
            "PRIOR_ADVANCE_STATUS": "NOT_TESTED_STAGE0_ONLY",
            "ABLATION_COMPONENT_STATUS": "NOT_TESTED_STAGE0_ONLY",
            "SIMPLE_EXPLANATION_STATUS": "CONTROL_FROZEN_NOT_TESTED",
            "CLEAN_RETENTION_STATUS": "REQUIRED_NOT_TESTED",
            "GENERALIZATION_STATUS": "HELD_OUT_MANIFEST_FROZEN_NOT_TESTED",
            "OVERALL_PAPER_CANDIDATE_STATUS": "PRIOR_ADVANCE_NOT_ESTABLISHED",
            "status_scope": (
                "Comparator-role statuses are Stage0-only. SGL-XVLA has not shown "
                "a task-success result, prior advance, clean retention, ablation, "
                "or held-out generalization."
            ),
        },
        "authorization_boundary": {
            "stage0_complete": all_gates_valid,
            "candidate_generation_completed": True,
            "candidate_can_advance_to_no_training_rollout_protocol_freeze": all_gates_valid,
            "training_authorized": False,
            "lora_or_qlora_training_authorized": False,
            "checkpoint_write_authorized": False,
            "control_rollout_authorized": False,
            "ours_rollout_authorized": False,
            "paper_candidate_go": False,
            "prototype_go": False,
        },
        "execution_classification": {
            "execution_type": "REPORT_ONLY",
            "evidence_role": "STAGE0_COMPLETION_ADJUDICATION",
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
            "stage0_complete_for_protocol_freeze": all_gates_valid,
            "candidate_can_advance_to_no_training_rollout_protocol_freeze": all_gates_valid,
            "candidate_can_train": False,
            "candidate_can_roll_out_ours": False,
            "control_can_roll_out_now": False,
            "paper_candidate_go": False,
            "prototype_go": False,
        },
        "next_action": (
            "Freeze a separate no-training rollout protocol for SGL-XVLA/control "
            "comparison before any simulator episode. Do not train, write "
            "checkpoints, or run Ours until that protocol is frozen."
        ),
    }


def validate_sgl_stage0_adjudication(adjudication: dict[str, Any]) -> list[str]:
    """Return validation errors for the Stage 0 adjudication."""

    errors: list[str] = []
    if adjudication.get("candidate_id") != CANDIDATE_ID:
        errors.append("candidate_id must be SGL-XVLA")
    if adjudication.get("all_required_stage0_checks_frozen") is not True:
        errors.append("all required Stage0 checks must be frozen")
    validation_errors = adjudication.get("validation_errors", {})
    for gate_name, gate_errors in validation_errors.items():
        if gate_errors:
            errors.append(f"{gate_name} has validation errors")
    statuses = adjudication.get("stage0_check_status", {})
    required_status_keys = {
        "support_observability_no_training",
        "action_bias_bounds_no_optimizer",
        "simple_fixed_lift_control_frozen",
        "held_out_identity_manifest_frozen",
    }
    if set(statuses) != required_status_keys:
        errors.append("Stage0 check status keys must exactly match required checks")
    if statuses.get("support_observability_no_training", {}).get("visual_progress_observability_verified") is not False:
        errors.append("visual progress must remain explicitly unverified")
    target = adjudication.get("target", {})
    if target.get("residual_identities") != RESIDUAL_IDENTITIES:
        errors.append("residual identities must remain frozen")
    if target.get("clean_retention_identities") != CLEAN_RETENTION_IDENTITIES:
        errors.append("clean-retention identities must remain required")
    if target.get("held_out_confirmatory_identity_pool") != HELD_OUT_CONFIRMATORY_IDENTITY_POOL:
        errors.append("held-out pool must remain frozen")
    auth = adjudication.get("authorization_boundary", {})
    if auth.get("candidate_can_advance_to_no_training_rollout_protocol_freeze") is not True:
        errors.append("candidate should advance only to no-training rollout protocol freeze")
    for key in [
        "training_authorized",
        "lora_or_qlora_training_authorized",
        "checkpoint_write_authorized",
        "control_rollout_authorized",
        "ours_rollout_authorized",
        "paper_candidate_go",
        "prototype_go",
    ]:
        if auth.get(key) is not False:
            errors.append(f"{key} must be false")
    comparator_statuses = adjudication.get("comparator_role_statuses", {})
    if comparator_statuses.get("OVERALL_PAPER_CANDIDATE_STATUS") != "PRIOR_ADVANCE_NOT_ESTABLISHED":
        errors.append("paper-candidate status must not be GO at Stage0")
    execution = adjudication.get("execution_classification", {})
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
    conclusion = adjudication.get("bounded_conclusion", {})
    if conclusion.get("stage0_complete_for_protocol_freeze") is not True:
        errors.append("Stage0 must be complete for protocol freeze")
    if conclusion.get("candidate_can_train") is not False:
        errors.append("candidate cannot train after Stage0 adjudication")
    if conclusion.get("candidate_can_roll_out_ours") is not False:
        errors.append("candidate cannot roll out Ours after Stage0 adjudication")
    if conclusion.get("control_can_roll_out_now") is not False:
        errors.append("Stage0 adjudication does not authorize control rollout")
    return errors


def write_sgl_stage0_adjudication(output_path: Path) -> dict[str, Any]:
    """Build, validate, and write the Stage 0 adjudication JSON."""

    adjudication = build_sgl_stage0_adjudication()
    errors = validate_sgl_stage0_adjudication(adjudication)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(adjudication, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return adjudication


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STAGE0_ADJUDICATION_ARTIFACT)
    args = parser.parse_args()
    adjudication = write_sgl_stage0_adjudication(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": adjudication["decision"],
                "stage0_complete_for_protocol_freeze": adjudication["bounded_conclusion"][
                    "stage0_complete_for_protocol_freeze"
                ],
                "candidate_can_train": adjudication["bounded_conclusion"]["candidate_can_train"],
                "candidate_can_roll_out_ours": adjudication["bounded_conclusion"]["candidate_can_roll_out_ours"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
