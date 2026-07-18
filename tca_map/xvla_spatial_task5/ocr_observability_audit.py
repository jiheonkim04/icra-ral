"""Report-only trigger observability audit for OCR-XVLA Stage 0.

The audit checks whether OCR's no-progress retry trigger can be specified from
existing allowed artifacts only. It does not load models, launch simulators,
train, write checkpoints, inspect privileged simulator state, or run Ours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tca_map.xvla_spatial_task5.ocr_stage0_gate import (
    OCR_CANDIDATE_ID,
    OCR_STAGE0_ARTIFACT,
    build_ocr_xvla_stage0_gate,
    validate_ocr_xvla_stage0_gate,
)
from tca_map.xvla_spatial_task5.sgl_stage0_gate import (
    CLEAN_RETENTION_IDENTITIES,
    HELD_OUT_CONFIRMATORY_IDENTITY_POOL,
    RESIDUAL_IDENTITIES,
    TARGET_INSTRUCTION,
    TARGET_SUITE,
    TARGET_TASK_ID,
)


OCR_OBSERVABILITY_ARTIFACT = Path("runs/xvla_prior/epoch5_ocr_xvla_task5_observability_audit_v1.json")

EXISTING_XVLA_TRACE_ARTIFACTS = [
    {
        "path": "runs/xvla_prior/failure_scan_libero_spatial_identity20260730_post_r2p_archive_20260718T0645KST/task_5/result.json",
        "reset_identities": [20260730],
        "roles": ["residual_failure"],
        "length_bytes": 8683,
        "episode_count": 1,
        "available_episode_fields": [
            "action_chunk_count",
            "action_chunk_ranges",
            "action_chunk_shapes",
            "completed",
            "cuda_memory",
            "done",
            "elapsed_seconds",
            "environment_latency_seconds",
            "exception",
            "final_reward",
            "initial_state_index",
            "policy_latency_seconds",
            "reset_identity",
            "steps",
            "success",
        ],
        "has_per_step_rgb_or_video": False,
        "has_per_step_proprio": False,
        "has_per_step_action_history": False,
        "has_object_separation_signal_from_allowed_observation": False,
    },
    {
        "path": "runs/xvla_prior/repeated_residual_spatial_task5_id20260731_33_xvla_prior_20260718T1115KST/result.json",
        "reset_identities": [20260731, 20260732, 20260733],
        "roles": ["clean_retention_success", "clean_retention_success", "residual_failure"],
        "length_bytes": 20477,
        "episode_count": 3,
        "available_episode_fields": [
            "action_chunk_count",
            "action_chunk_ranges",
            "action_chunk_shapes",
            "completed",
            "cuda_memory",
            "done",
            "elapsed_seconds",
            "environment_latency_seconds",
            "exception",
            "final_reward",
            "initial_state_index",
            "policy_latency_seconds",
            "reset_identity",
            "steps",
            "success",
        ],
        "has_per_step_rgb_or_video": False,
        "has_per_step_proprio": False,
        "has_per_step_action_history": False,
        "has_object_separation_signal_from_allowed_observation": False,
    },
]

VIDEO_SEARCH_RESULT = {
    "searched_dirs": [
        "runs/xvla_prior/failure_scan_libero_spatial_identity20260730_post_r2p_archive_20260718T0645KST/task_5",
        "runs/xvla_prior/repeated_residual_spatial_task5_id20260731_33_xvla_prior_20260718T1115KST",
    ],
    "video_or_image_files_found": [],
    "extensions_checked": [".mp4", ".avi", ".gif", ".png", ".jpg", ".jpeg"],
}


def build_ocr_observability_audit() -> dict[str, Any]:
    """Build the deterministic OCR trigger observability audit."""

    stage0 = build_ocr_xvla_stage0_gate()
    stage0_errors = validate_ocr_xvla_stage0_gate(stage0)
    required_trace_fields = [
        "per-step RGB or video frames",
        "per-step proprio/eef/gripper trace",
        "per-step executed or proposed action history",
        "a frozen observation-only object-separation/progress signal",
        "timestamps or step indices linking observations to first grasp/lift attempt",
    ]
    available_fields = sorted(
        {
            field
            for artifact in EXISTING_XVLA_TRACE_ARTIFACTS
            for field in artifact["available_episode_fields"]
        }
    )
    missing_required_trace_fields = list(required_trace_fields)
    trigger_observable_from_existing_artifacts = False
    decision = (
        "OCR_TRIGGER_OBSERVABILITY_BLOCKED_NO_ALLOWED_PROGRESS_TRACE_NO_ROLLOUT"
        if not stage0_errors and not trigger_observable_from_existing_artifacts
        else "OCR_TRIGGER_OBSERVABILITY_NOT_VERIFIED"
    )

    return {
        "schema_version": "2026-07-18.epoch5_ocr_xvla_observability_audit.v1",
        "stage": "epoch_5_ocr_xvla_task5_trigger_observability_report_only",
        "candidate_id": OCR_CANDIDATE_ID,
        "decision": decision,
        "target": {
            "suite": TARGET_SUITE,
            "task_id": TARGET_TASK_ID,
            "instruction": TARGET_INSTRUCTION,
            "residual_identities": list(RESIDUAL_IDENTITIES),
            "clean_retention_identities": list(CLEAN_RETENTION_IDENTITIES),
            "held_out_confirmatory_identity_pool": list(HELD_OUT_CONFIRMATORY_IDENTITY_POOL),
        },
        "stage0_gate": {
            "artifact": str(OCR_STAGE0_ARTIFACT),
            "validation_errors": stage0_errors,
            "valid": not stage0_errors,
        },
        "existing_artifact_inventory": {
            "xvla_trace_artifacts": list(EXISTING_XVLA_TRACE_ARTIFACTS),
            "video_search_result": dict(VIDEO_SEARCH_RESULT),
            "available_episode_summary_fields": available_fields,
        },
        "trigger_observability": {
            "proposed_trigger": "no visual/proprio/action-history progress after first grasp/lift attempt",
            "required_trace_fields": required_trace_fields,
            "missing_required_trace_fields": missing_required_trace_fields,
            "trigger_observable_from_existing_artifacts": trigger_observable_from_existing_artifacts,
            "deterministic_trigger_can_be_frozen_now": False,
            "reason": (
                "Existing X-VLA task5 artifacts contain episode summaries and action "
                "chunk range metadata, but not per-step RGB/video, proprio, or "
                "action-history traces needed to define an observation-consistency "
                "no-progress trigger without privileged state."
            ),
        },
        "forbidden_proxy_fields_not_used": [
            "success",
            "final_reward",
            "done",
            "reset_identity",
            "initial_state_index",
            "infrastructure failure/success flags",
        ],
        "comparator_role_statuses": {
            "BASE_CLAIM_STATUS": "NOT_TESTED_STAGE0_TRIGGER_BLOCK",
            "PRIOR_ADVANCE_STATUS": "NOT_TESTED_STAGE0_TRIGGER_BLOCK",
            "ABLATION_COMPONENT_STATUS": "KEY_COMPONENT_NOT_SUPPORTED",
            "SIMPLE_EXPLANATION_STATUS": "NOT_TESTED_STAGE0_TRIGGER_BLOCK",
            "CLEAN_RETENTION_STATUS": "NOT_TESTED_STAGE0_TRIGGER_BLOCK",
            "GENERALIZATION_STATUS": "NOT_TESTED_STAGE0_TRIGGER_BLOCK",
            "OVERALL_PAPER_CANDIDATE_STATUS": "IMPLEMENTATION_DATA_OR_RESOURCE_FAILURE",
            "status_scope": (
                "OCR is blocked because the current existing artifacts do not expose "
                "the allowed trace fields needed to freeze its trigger. This is not "
                "evidence that retry mechanisms are scientifically impossible."
            ),
        },
        "execution_classification": {
            "execution_type": "REPORT_ONLY",
            "evidence_role": "OURS_CANDIDATE_STAGE0_OBSERVABILITY_AUDIT",
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
            "trigger_observability_passed": False,
            "ocr_candidate_can_advance_to_action_bounds": False,
            "ocr_candidate_can_train": False,
            "ocr_candidate_can_roll_out_ours": False,
            "ocr_candidate_blocked_from_existing_artifacts": True,
            "task5_candidate_set_exhausted": True,
            "candidate_exhaustion_reason": (
                "SGL is blocked by simple-control equivalence; OCR trigger cannot "
                "be frozen from existing allowed traces."
            ),
        },
        "next_action": (
            "Do not run OCR-XVLA. Mark the current task5 candidate set exhausted "
            "and resume official-prior-first residual search elsewhere; no "
            "training, checkpoints, simulator episode, or Ours rollout."
        ),
    }


def validate_ocr_observability_audit(audit: dict[str, Any]) -> list[str]:
    """Return validation errors for the OCR observability audit."""

    errors: list[str] = []
    if audit.get("candidate_id") != OCR_CANDIDATE_ID:
        errors.append("candidate_id must be OCR-XVLA")
    if audit.get("stage0_gate", {}).get("valid") is not True:
        errors.append("OCR Stage0 gate must validate before observability audit")
    trigger = audit.get("trigger_observability", {})
    if trigger.get("trigger_observable_from_existing_artifacts") is not False:
        errors.append("trigger must not be observable from current existing artifacts")
    if trigger.get("deterministic_trigger_can_be_frozen_now") is not False:
        errors.append("deterministic OCR trigger must not be frozen now")
    missing = set(trigger.get("missing_required_trace_fields", []))
    required_missing = {
        "per-step RGB or video frames",
        "per-step proprio/eef/gripper trace",
        "per-step executed or proposed action history",
        "a frozen observation-only object-separation/progress signal",
        "timestamps or step indices linking observations to first grasp/lift attempt",
    }
    if missing != required_missing:
        errors.append("missing trace fields must exactly match the required trigger inputs")
    inventory = audit.get("existing_artifact_inventory", {})
    if inventory.get("video_search_result", {}).get("video_or_image_files_found") != []:
        errors.append("video/image search must be empty for this audit")
    for artifact in inventory.get("xvla_trace_artifacts", []):
        for key in [
            "has_per_step_rgb_or_video",
            "has_per_step_proprio",
            "has_per_step_action_history",
            "has_object_separation_signal_from_allowed_observation",
        ]:
            if artifact.get(key) is not False:
                errors.append(f"{key} must be false in existing artifact inventory")
    forbidden_proxy_fields = set(audit.get("forbidden_proxy_fields_not_used", []))
    for field in ["success", "final_reward", "done", "reset_identity", "initial_state_index"]:
        if field not in forbidden_proxy_fields:
            errors.append(f"forbidden proxy field not recorded: {field}")
    statuses = audit.get("comparator_role_statuses", {})
    if statuses.get("OVERALL_PAPER_CANDIDATE_STATUS") != "IMPLEMENTATION_DATA_OR_RESOURCE_FAILURE":
        errors.append("overall status must be implementation/data/resource failure")
    execution = audit.get("execution_classification", {})
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
    conclusion = audit.get("bounded_conclusion", {})
    if conclusion.get("trigger_observability_passed") is not False:
        errors.append("trigger observability must fail")
    if conclusion.get("ocr_candidate_can_roll_out_ours") is not False:
        errors.append("OCR cannot roll out Ours")
    if conclusion.get("ocr_candidate_blocked_from_existing_artifacts") is not True:
        errors.append("OCR must be blocked from existing artifacts")
    if conclusion.get("task5_candidate_set_exhausted") is not True:
        errors.append("task5 candidate set should be exhausted")
    return errors


def write_ocr_observability_audit(output_path: Path) -> dict[str, Any]:
    """Build, validate, and write the OCR observability audit JSON."""

    audit = build_ocr_observability_audit()
    errors = validate_ocr_observability_audit(audit)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OCR_OBSERVABILITY_ARTIFACT)
    args = parser.parse_args()
    audit = write_ocr_observability_audit(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": audit["decision"],
                "trigger_observability_passed": audit["bounded_conclusion"]["trigger_observability_passed"],
                "ocr_candidate_blocked": audit["bounded_conclusion"]["ocr_candidate_blocked_from_existing_artifacts"],
                "task5_candidate_set_exhausted": audit["bounded_conclusion"]["task5_candidate_set_exhausted"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
