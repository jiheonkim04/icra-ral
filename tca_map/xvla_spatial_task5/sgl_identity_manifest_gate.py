"""Report-only identity-role manifest freeze for SGL-XVLA Stage 0.

This module freezes development, clean-retention, and held-out identity roles
before any SGL-XVLA control rollout, Ours rollout, training, optimizer step, or
checkpoint write. It is a preregistration artifact only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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
    TARGET_INSTRUCTION,
    TARGET_SUITE,
    TARGET_TASK_ID,
)


IDENTITY_MANIFEST_ARTIFACT = Path("runs/xvla_prior/epoch5_sgl_xvla_task5_identity_manifest_gate_v1.json")
IDENTITY_MAPPING_BASE = 20260711


def initial_state_index(reset_identity: int) -> int:
    """Return the task5 reset-identity to initial-state-index mapping."""

    return reset_identity - IDENTITY_MAPPING_BASE


def _identity_records(identities: list[int], role: str) -> list[dict[str, Any]]:
    return [
        {
            "reset_identity": identity,
            "initial_state_index": initial_state_index(identity),
            "identity_role": role,
        }
        for identity in identities
    ]


def build_sgl_identity_manifest_gate() -> dict[str, Any]:
    """Build the deterministic held-out identity/development manifest."""

    simple_control_gate = build_sgl_simple_control_gate()
    simple_control_errors = validate_sgl_simple_control_gate(simple_control_gate)
    dev_residual_identities = list(RESIDUAL_IDENTITIES)
    clean_retention_identities = list(CLEAN_RETENTION_IDENTITIES)
    held_out_identities = list(HELD_OUT_CONFIRMATORY_IDENTITY_POOL)
    all_role_identities = dev_residual_identities + clean_retention_identities + held_out_identities
    role_sets_disjoint = len(all_role_identities) == len(set(all_role_identities))
    role_lists_individually_sorted = all(
        identities == sorted(identities)
        for identities in [dev_residual_identities, clean_retention_identities, held_out_identities]
    )
    decision = (
        "SGL_HELDOUT_IDENTITY_MANIFEST_FROZEN_NO_TRAINING_NO_OURS"
        if not simple_control_errors and role_sets_disjoint and role_lists_individually_sorted
        else "SGL_HELDOUT_IDENTITY_MANIFEST_NOT_VERIFIED"
    )

    return {
        "schema_version": "2026-07-18.epoch5_sgl_xvla_identity_manifest_gate.v1",
        "stage": "epoch_5_sgl_xvla_task5_stage0_identity_manifest_report_only",
        "candidate_id": CANDIDATE_ID,
        "decision": decision,
        "target": {
            "suite": TARGET_SUITE,
            "task_id": TARGET_TASK_ID,
            "instruction": TARGET_INSTRUCTION,
            "identity_mapping_rule": "initial_state_index = reset_identity - 20260711 for task5",
            "identity_mapping_base": IDENTITY_MAPPING_BASE,
        },
        "upstream_gates": {
            "simple_control_artifact": str(SIMPLE_CONTROL_ARTIFACT),
            "simple_control_validation_errors": simple_control_errors,
            "simple_control_valid": not simple_control_errors,
            "simple_control_decision": simple_control_gate["decision"],
        },
        "identity_roles": {
            "development_residual_identities": _identity_records(dev_residual_identities, "development_residual"),
            "clean_retention_identities": _identity_records(clean_retention_identities, "clean_retention"),
            "held_out_confirmatory_identities": _identity_records(held_out_identities, "held_out_confirmatory"),
            "all_frozen_identities_sorted": sorted(all_role_identities),
            "role_sets_disjoint": role_sets_disjoint,
            "role_lists_individually_sorted": role_lists_individually_sorted,
            "frozen_before_any_sgl_ours_result": True,
            "frozen_before_any_simple_control_rollout": True,
            "frozen_before_any_training": True,
        },
        "role_usage_policy": {
            "development_residual": {
                "allowed_use": [
                    "pre-registered implementation smoke checks after a separate rollout protocol is frozen",
                    "paired development residual evaluation only if later explicitly authorized",
                    "debugging protocol integrity without checkpoint or threshold selection",
                ],
                "forbidden_use": [
                    "held-out confirmation",
                    "checkpoint selection",
                    "post-hoc candidate redesign after observing Ours success",
                ],
            },
            "clean_retention": {
                "allowed_use": [
                    "detecting degradation on identities X-VLA already solved",
                    "added-clipping guard",
                    "noninferiority/retention accounting",
                ],
                "forbidden_use": [
                    "claiming residual improvement",
                    "selecting a favorable action-bias sign",
                    "dropping identities after failures",
                ],
            },
            "held_out_confirmatory": {
                "allowed_use": [
                    "confirmatory evaluation only after implementation, control, metrics, and stopping rules are frozen",
                    "paired held-out residual/retention reporting under a later explicit protocol",
                ],
                "forbidden_use": [
                    "development tuning",
                    "debugging threshold or sign choices",
                    "rerunning/cherry-picking identities",
                    "checkpoint selection",
                ],
            },
        },
        "anti_cherry_pick_rules": [
            "Do not add, drop, reorder, or relabel identities after any SGL, control, or Ours result.",
            "Do not inspect held-out outcomes before the development/control/Ours protocol is frozen.",
            "Do not choose action-bias signs, thresholds, or clipping rules from held-out outcomes.",
            "Do not treat clean-retention identities as optional because the language support gate activates on them.",
            "If an infrastructure failure invalidates an identity, preserve the invalid artifact and freeze any replacement before seeing a valid outcome.",
        ],
        "comparator_role_calibration": {
            "base_claim_status_role": "development and held-out identities test the residual claim against Base/Prior only under a later matched rollout protocol",
            "simple_explanation_status_role": "the simple fixed-lift control must use the same identity roles and cannot be added or removed post hoc",
            "clean_retention_status_role": "clean-retention identities block only unacceptable degradation or added clipping, not the residual-gain claim by themselves",
            "universal_beat_all_rule_applied": False,
        },
        "execution_classification": {
            "execution_type": "REPORT_ONLY",
            "evidence_role": "IDENTITY_MANIFEST_PREREGISTRATION",
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
            "identity_manifest_frozen": decision
            == "SGL_HELDOUT_IDENTITY_MANIFEST_FROZEN_NO_TRAINING_NO_OURS",
            "stage0_required_checks_now_frozen": [
                "support_observability_no_training",
                "action_bias_bounds_no_optimizer",
                "simple_fixed_lift_control_frozen",
                "held_out_identity_manifest_frozen",
            ],
            "candidate_can_advance_to_stage0_completion_adjudication": decision
            == "SGL_HELDOUT_IDENTITY_MANIFEST_FROZEN_NO_TRAINING_NO_OURS",
            "candidate_can_train": False,
            "candidate_can_roll_out_ours": False,
            "control_can_roll_out_now": False,
        },
        "next_action": (
            "Write a Stage 0 completion/adjudication report for SGL-XVLA, then "
            "decide whether a separate no-training rollout protocol may be frozen. "
            "Do not train, write checkpoints, or run Ours."
        ),
    }


def validate_sgl_identity_manifest_gate(gate: dict[str, Any]) -> list[str]:
    """Return validation errors for the SGL identity manifest gate."""

    errors: list[str] = []
    if gate.get("candidate_id") != CANDIDATE_ID:
        errors.append("candidate_id must be SGL-XVLA")
    upstream = gate.get("upstream_gates", {})
    if upstream.get("simple_control_valid") is not True:
        errors.append("simple-control gate must validate")
    roles = gate.get("identity_roles", {})
    dev = roles.get("development_residual_identities", [])
    clean = roles.get("clean_retention_identities", [])
    held_out = roles.get("held_out_confirmatory_identities", [])
    if [item.get("reset_identity") for item in dev] != RESIDUAL_IDENTITIES:
        errors.append("development residual identities must match frozen residuals")
    if [item.get("reset_identity") for item in clean] != CLEAN_RETENTION_IDENTITIES:
        errors.append("clean-retention identities must match frozen clean-retention list")
    if [item.get("reset_identity") for item in held_out] != HELD_OUT_CONFIRMATORY_IDENTITY_POOL:
        errors.append("held-out confirmatory pool must match frozen list")
    if roles.get("role_sets_disjoint") is not True:
        errors.append("identity roles must be disjoint")
    if roles.get("role_lists_individually_sorted") is not True:
        errors.append("identity records must remain sorted within each role")
    if roles.get("all_frozen_identities_sorted") != sorted(RESIDUAL_IDENTITIES + CLEAN_RETENTION_IDENTITIES + HELD_OUT_CONFIRMATORY_IDENTITY_POOL):
        errors.append("flattened sorted identity list must match all frozen identities")
    for key in [
        "frozen_before_any_sgl_ours_result",
        "frozen_before_any_simple_control_rollout",
        "frozen_before_any_training",
    ]:
        if roles.get(key) is not True:
            errors.append(f"{key} must be true")
    for record in dev + clean + held_out:
        identity = record.get("reset_identity")
        if record.get("initial_state_index") != initial_state_index(identity):
            errors.append(f"initial_state_index mismatch for {identity}")
    policy = gate.get("role_usage_policy", {})
    if "checkpoint selection" not in policy.get("held_out_confirmatory", {}).get("forbidden_use", []):
        errors.append("held-out policy must forbid checkpoint selection")
    if "dropping identities after failures" not in policy.get("clean_retention", {}).get("forbidden_use", []):
        errors.append("clean-retention policy must forbid dropping failures")
    anti_cherry_pick_rules = gate.get("anti_cherry_pick_rules", [])
    if not any("Do not add, drop, reorder, or relabel identities" in rule for rule in anti_cherry_pick_rules):
        errors.append("anti-cherry-pick rules must freeze identity membership")
    calibration = gate.get("comparator_role_calibration", {})
    if calibration.get("universal_beat_all_rule_applied") is not False:
        errors.append("manifest calibration must not apply a universal beat-all rule")
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
    if conclusion.get("identity_manifest_frozen") is not True:
        errors.append("identity manifest must be frozen")
    required_checks = {
        "support_observability_no_training",
        "action_bias_bounds_no_optimizer",
        "simple_fixed_lift_control_frozen",
        "held_out_identity_manifest_frozen",
    }
    if set(conclusion.get("stage0_required_checks_now_frozen", [])) != required_checks:
        errors.append("all required Stage0 checks must be recorded as frozen")
    if conclusion.get("candidate_can_train") is not False:
        errors.append("candidate cannot train after identity manifest gate")
    if conclusion.get("candidate_can_roll_out_ours") is not False:
        errors.append("candidate cannot roll out Ours after identity manifest gate")
    if conclusion.get("control_can_roll_out_now") is not False:
        errors.append("identity manifest gate does not authorize control rollout")
    return errors


def write_sgl_identity_manifest_gate(output_path: Path) -> dict[str, Any]:
    """Build, validate, and write the identity manifest gate JSON."""

    gate = build_sgl_identity_manifest_gate()
    errors = validate_sgl_identity_manifest_gate(gate)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=IDENTITY_MANIFEST_ARTIFACT)
    args = parser.parse_args()
    gate = write_sgl_identity_manifest_gate(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": gate["decision"],
                "identity_manifest_frozen": gate["bounded_conclusion"]["identity_manifest_frozen"],
                "candidate_can_train": gate["bounded_conclusion"]["candidate_can_train"],
                "candidate_can_roll_out_ours": gate["bounded_conclusion"]["candidate_can_roll_out_ours"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
