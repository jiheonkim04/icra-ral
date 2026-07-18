"""Frozen no-training Stage 0 gate for the OCR-XVLA backup candidate.

OCR-XVLA is the backup candidate after SGL-XVLA's current frozen executable was
blocked by simple-control equivalence. This module records a deterministic
candidate-gate specification only. It does not load a model, train, run an
optimizer, write checkpoints, launch a simulator, or evaluate Ours.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tca_map.xvla_spatial_task5.sgl_runner_preflight import (
    RUNNER_PREFLIGHT_ARTIFACT,
    build_sgl_runner_preflight,
    validate_sgl_runner_preflight,
)
from tca_map.xvla_spatial_task5.sgl_stage0_gate import (
    CLEAN_RETENTION_IDENTITIES,
    HELD_OUT_CONFIRMATORY_IDENTITY_POOL,
    RESIDUAL_IDENTITIES,
    TARGET_INSTRUCTION,
    TARGET_SUITE,
    TARGET_TASK_ID,
)


OCR_STAGE0_ARTIFACT = Path("runs/xvla_prior/epoch5_ocr_xvla_task5_stage0_gate_v1.json")
OCR_CANDIDATE_ID = "OCR-XVLA"
CANDIDATE_GENERATION_REPORT = "reports/post_r2p_archive_libero_spatial_task5_candidate_generation_result.json"
SGL_PREFLIGHT_REPORT = "reports/post_r2p_archive_libero_spatial_task5_sgl_runner_preflight_result.json"
R2P_ARCHIVE_REPORT = "reports/post_secondprior_libero_spatial_20260727_r2p_xvla_archive_decision.json"


def build_ocr_xvla_stage0_gate() -> dict[str, Any]:
    """Return the frozen no-training Stage 0 gate for OCR-XVLA."""

    sgl_preflight = build_sgl_runner_preflight()
    sgl_preflight_errors = validate_sgl_runner_preflight(sgl_preflight)
    return {
        "schema_version": "2026-07-18.epoch5_ocr_xvla_stage0_gate.v1",
        "freeze_id": "epoch5_ocr_xvla_task5_stage0_gate_v1",
        "stage": "epoch_5_ocr_xvla_task5_stage0_spec_frozen_no_training",
        "date_kst": "2026-07-18",
        "candidate": {
            "candidate_id": OCR_CANDIDATE_ID,
            "name": "Observation-Consistency Retry for X-VLA",
            "status": "FROZEN_STAGE0_SPEC_ONLY_BACKUP_CANDIDATE",
            "candidate_generation_report": CANDIDATE_GENERATION_REPORT,
            "sgl_preflight_artifact": str(RUNNER_PREFLIGHT_ARTIFACT),
            "sgl_preflight_report": SGL_PREFLIGHT_REPORT,
            "archived_r2p_xvla_report": R2P_ARCHIVE_REPORT,
        },
        "upstream_primary_candidate": {
            "candidate_id": "SGL-XVLA",
            "decision": sgl_preflight["decision"],
            "validation_errors": sgl_preflight_errors,
            "valid": not sgl_preflight_errors,
            "blocked_before_simulator_episode": sgl_preflight["bounded_conclusion"][
                "blocked_before_simulator_episode"
            ],
            "backup_candidate_stage0_authorized_next": sgl_preflight["bounded_conclusion"][
                "backup_ocr_xvla_can_start_stage0"
            ],
        },
        "target": {
            "suite": TARGET_SUITE,
            "task_id": TARGET_TASK_ID,
            "instruction": TARGET_INSTRUCTION,
            "repeated_shared_failure_identities": list(RESIDUAL_IDENTITIES),
            "xvla_solved_clean_retention_identities": list(CLEAN_RETENTION_IDENTITIES),
            "held_out_confirmatory_identity_pool": list(HELD_OUT_CONFIRMATORY_IDENTITY_POOL),
            "held_out_pool_frozen_before_any_ocr_result": True,
        },
        "authorization_boundary": {
            "stage0_spec_created": True,
            "backup_candidate_activation_authorized_by_sgl_block": True,
            "training_happened_at_freeze": False,
            "optimizer_step_happened_at_freeze": False,
            "checkpoint_written_at_freeze": False,
            "closed_loop_ours_evaluation_happened_at_freeze": False,
            "model_loaded_at_freeze": False,
            "simulator_episode_count_at_freeze": 0,
            "training_authorized_by_this_gate": False,
            "ours_rollout_authorized_by_this_gate": False,
            "control_rollout_authorized_by_this_gate": False,
            "max_candidates_in_current_generation": 2,
            "selected_backup_candidate": OCR_CANDIDATE_ID,
        },
        "closed_methods_not_reopened": [
            "R2R-OFT",
            "BR-XVLA",
            "MPR-XVLA",
            "PRC-XVLA",
            "CR-LightVLA",
            "ATCD",
            "MCI-VLA",
            "CSPR-VLA",
            "R2P-XVLA",
            "SGL-XVLA-current-frozen-executable",
        ],
        "inference_input_policy": {
            "allowed_inputs": [
                "RGB observations",
                "wrist RGB observations when already part of the policy input",
                "proprioception already available to X-VLA",
                "language instruction",
                "recent action history already produced by the policy",
            ],
            "forbidden_inputs": [
                "simulator object pose",
                "simulator contact state",
                "reward",
                "success flag",
                "HDF5 demo identity",
                "reset identity label",
                "phase label",
                "task success oracle",
            ],
            "privileged_state_at_inference": False,
            "reward_or_success_at_inference": False,
            "reset_identity_at_inference": False,
        },
        "mechanism_hypothesis": {
            "problem_frame": (
                "Repeated task5 failures may include a recoverable first-attempt "
                "grasp/lift no-progress condition after the bowl fails to separate "
                "from the ramekin."
            ),
            "candidate_component": (
                "A no-progress observation-consistency monitor checks allowed "
                "RGB/proprio/action-history signals after the first grasp/lift "
                "attempt. If progress is absent under a frozen criterion, it "
                "triggers one bounded re-center/regrasp retry and then returns "
                "control to X-VLA."
            ),
            "claim_axis_if_future_rollout_is_authorized": (
                "Improve paired success on the repeated task5 retry-recoverable "
                "residual while adding at most one retry and preserving X-VLA "
                "clean-retention identities."
            ),
        },
        "stage0_required_checks": [
            {
                "check_id": "observation_consistency_trigger_observability_no_training",
                "purpose": "Verify no-progress can be detected from existing allowed traces only.",
                "allowed_evidence": [
                    "existing X-VLA failure videos and result metadata",
                    "existing successful X-VLA clean-retention videos and metadata",
                    "RGB/proprio/action-history traces already produced by frozen diagnostics",
                ],
                "forbidden_evidence": [
                    "simulator object pose at inference",
                    "reward at inference",
                    "success flag at inference",
                    "any OCR rollout",
                ],
                "pass_condition": (
                    "A deterministic observation/action-history criterion can be "
                    "specified before any OCR rollout and without privileged state."
                ),
            },
            {
                "check_id": "retry_action_bounds_no_optimizer",
                "purpose": "Specify one bounded retry action template without optimization.",
                "allowed_evidence": [
                    "expert HDF5 action statistics",
                    "X-VLA action range diagnostics",
                    "existing failure/clean-retention metadata",
                ],
                "forbidden_evidence": [
                    "optimizer.step",
                    "checkpoint selection",
                    "closed-loop OCR success",
                ],
                "pass_condition": "Retry action modifications are finite, bounded, and independent of outcome tuning.",
            },
            {
                "check_id": "simple_timeout_retry_control_frozen",
                "purpose": "Freeze the strongest simple timeout-retry explanation before OCR execution.",
                "allowed_evidence": [
                    "hand-authored timeout retry template specification",
                    "no-training action-range sanity checks",
                ],
                "forbidden_evidence": [
                    "choosing the control after seeing OCR results",
                    "multiple redundant controls for the same objection",
                ],
                "pass_condition": "A single fixed timeout-retry control is specified as the primary simple-control comparator.",
            },
            {
                "check_id": "held_out_identity_manifest_frozen",
                "purpose": "Protect against test-set tuning.",
                "allowed_evidence": [
                    "frozen held-out identity list",
                    "frozen clean-retention identity list",
                ],
                "forbidden_evidence": [
                    "changing held-out identities after OCR results",
                    "selecting checkpoints or thresholds on held-out success",
                ],
                "pass_condition": "Development, clean-retention, and held-out identity roles are written before any OCR rollout.",
            },
        ],
        "future_comparator_roles": {
            "base": {
                "comparator_role": "BASE",
                "scientific_question": "Does OCR improve the frozen X-VLA backbone on retry-recoverable task5 residuals?",
                "claim_metric": "paired residual success and intervention count",
                "blocking_condition": "No residual success gain or excessive intervention frequency.",
                "nonblocking_tradeoffs": "A small latency increase may be acceptable if retry count remains bounded.",
            },
            "closest_prior": {
                "comparator_role": "FIRST_PRIOR_AND_SECOND_PRIOR",
                "scientific_question": "Does OCR advance beyond X-VLA/OpenVLA on the same local residual protocol?",
                "claim_metric": "paired residual success and clean-retention breakdown",
                "blocking_condition": "Official priors match OCR on residual success without the retry mechanism.",
                "nonblocking_tradeoffs": "Comparable success with lower intervention frequency may be relevant if preregistered.",
            },
            "key_ablation": {
                "comparator_role": "ABLATION",
                "scientific_question": "Is the observation-consistency trigger necessary?",
                "claim_metric": "OCR vs always-retry and never-retry residual/clean-retention outcomes",
                "blocking_condition": "Always-retry or never-retry matches OCR with lower complexity and no clean cost.",
                "nonblocking_tradeoffs": "Always-retry may win one identity but fail aggregate clean retention or efficiency.",
            },
            "simple_control": {
                "comparator_role": "CONTROL",
                "scientific_question": "Can a fixed timeout retry explain the gain?",
                "claim_metric": "residual success, clean retention, latency, and intervention count",
                "blocking_condition": "Timeout retry matches OCR at equal/lower cost without missing mechanism capability.",
                "nonblocking_tradeoffs": "Timeout retry success on a single development identity does not automatically block held-out OCR benefit.",
            },
        },
        "kill_rules": [
            "Any simulator object state, contact state, reward, success flag, HDF5 identity, or reset identity is used as an inference input.",
            "Any training, optimizer step, checkpoint write, model load, simulator episode, control rollout, or Ours rollout happens during Stage 0 freeze.",
            "R2P-XVLA or any closed method is reopened, retuned, renamed as OCR, or used as the implementation substrate.",
            "SGL-XVLA current frozen executable is reopened instead of treating it as blocked by simple-control equivalence.",
            "The no-progress trigger cannot be specified from allowed observations/action history before rollout.",
            "The fixed timeout-retry simple control is not frozen before OCR execution.",
            "Held-out identities change after any OCR result.",
            "Clean-retention identities 20260731 and 20260732 are omitted from the next gate.",
        ],
        "next_action_after_stage0_freeze": (
            "Create a report-only OCR observability/trigger audit using existing "
            "failure and clean-retention artifacts only. Do not train, write "
            "checkpoints, run a simulator episode, or run Ours."
        ),
    }


def validate_ocr_xvla_stage0_gate(spec: dict[str, Any]) -> list[str]:
    """Return validation errors for the OCR Stage 0 gate."""

    errors: list[str] = []
    if spec.get("candidate", {}).get("candidate_id") != OCR_CANDIDATE_ID:
        errors.append("candidate_id must be OCR-XVLA")
    upstream = spec.get("upstream_primary_candidate", {})
    if upstream.get("valid") is not True:
        errors.append("SGL preflight must validate before OCR starts")
    if upstream.get("blocked_before_simulator_episode") is not True:
        errors.append("SGL must be blocked before OCR starts")
    if upstream.get("backup_candidate_stage0_authorized_next") is not True:
        errors.append("OCR backup Stage0 must be authorized by SGL block")
    target = spec.get("target", {})
    if target.get("repeated_shared_failure_identities") != RESIDUAL_IDENTITIES:
        errors.append("residual identities must be frozen")
    if target.get("xvla_solved_clean_retention_identities") != CLEAN_RETENTION_IDENTITIES:
        errors.append("clean-retention identities must be 20260731/20260732")
    if target.get("held_out_confirmatory_identity_pool") != HELD_OUT_CONFIRMATORY_IDENTITY_POOL:
        errors.append("held-out pool must remain frozen")
    boundary = spec.get("authorization_boundary", {})
    for key in [
        "training_happened_at_freeze",
        "optimizer_step_happened_at_freeze",
        "checkpoint_written_at_freeze",
        "closed_loop_ours_evaluation_happened_at_freeze",
        "model_loaded_at_freeze",
        "training_authorized_by_this_gate",
        "ours_rollout_authorized_by_this_gate",
        "control_rollout_authorized_by_this_gate",
    ]:
        if boundary.get(key) is not False:
            errors.append(f"{key} must be false")
    if boundary.get("simulator_episode_count_at_freeze") != 0:
        errors.append("simulator_episode_count_at_freeze must be zero")
    if boundary.get("max_candidates_in_current_generation") != 2:
        errors.append("max candidate count must remain two")
    closed = set(spec.get("closed_methods_not_reopened", []))
    if "R2P-XVLA" not in closed:
        errors.append("R2P-XVLA must remain closed")
    if "SGL-XVLA-current-frozen-executable" not in closed:
        errors.append("blocked SGL executable must remain closed")
    policy = spec.get("inference_input_policy", {})
    if policy.get("privileged_state_at_inference") is not False:
        errors.append("privileged state at inference must be false")
    if policy.get("reward_or_success_at_inference") is not False:
        errors.append("reward/success at inference must be false")
    if policy.get("reset_identity_at_inference") is not False:
        errors.append("reset identity at inference must be false")
    required_allowed = {
        "RGB observations",
        "proprioception already available to X-VLA",
        "language instruction",
        "recent action history already produced by the policy",
    }
    if not required_allowed.issubset(set(policy.get("allowed_inputs", []))):
        errors.append("allowed OCR inputs are incomplete")
    required_forbidden = {
        "simulator object pose",
        "simulator contact state",
        "reward",
        "success flag",
        "HDF5 demo identity",
        "reset identity label",
        "phase label",
        "task success oracle",
    }
    if not required_forbidden.issubset(set(policy.get("forbidden_inputs", []))):
        errors.append("forbidden OCR inference inputs are incomplete")
    check_ids = {item.get("check_id") for item in spec.get("stage0_required_checks", [])}
    required_checks = {
        "observation_consistency_trigger_observability_no_training",
        "retry_action_bounds_no_optimizer",
        "simple_timeout_retry_control_frozen",
        "held_out_identity_manifest_frozen",
    }
    if check_ids != required_checks:
        errors.append("OCR Stage0 checks must exactly match the frozen required checks")
    if not any("SGL-XVLA current frozen executable is reopened" in rule for rule in spec.get("kill_rules", [])):
        errors.append("kill rules must block reopening SGL current executable")
    return errors


def write_ocr_xvla_stage0_gate(output_path: Path) -> dict[str, Any]:
    """Build, validate, and write the OCR Stage 0 gate JSON."""

    spec = build_ocr_xvla_stage0_gate()
    errors = validate_ocr_xvla_stage0_gate(spec)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OCR_STAGE0_ARTIFACT)
    args = parser.parse_args()
    spec = write_ocr_xvla_stage0_gate(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "freeze_id": spec["freeze_id"],
                "candidate_id": spec["candidate"]["candidate_id"],
                "training_authorized": spec["authorization_boundary"]["training_authorized_by_this_gate"],
                "ours_rollout_authorized": spec["authorization_boundary"]["ours_rollout_authorized_by_this_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
