"""Frozen no-training Stage 0 gate for the SGL-XVLA candidate.

This module records a deterministic candidate-gate specification only. It does
not load a VLA model, train, run an optimizer step, write checkpoints, launch a
simulator, or evaluate Ours. The gate exists because task5 now has repeated
Base/X-VLA/OpenVLA failures after the R2P-XVLA archive; it does not reopen
R2P-XVLA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGE0_ARTIFACT = Path("runs/xvla_prior/epoch5_sgl_xvla_task5_stage0_gate_v1.json")
CANDIDATE_ID = "SGL-XVLA"
TARGET_SUITE = "libero_spatial"
TARGET_TASK_ID = 5
TARGET_INSTRUCTION = "pick up the black bowl on the ramekin and place it on the plate"
RESIDUAL_IDENTITIES = [20260727, 20260730, 20260733]
CLEAN_RETENTION_IDENTITIES = [20260731, 20260732]
HELD_OUT_CONFIRMATORY_IDENTITY_POOL = [20260734, 20260735, 20260736, 20260737]
CANDIDATE_GENERATION_REPORT = "reports/post_r2p_archive_libero_spatial_task5_candidate_generation_result.json"
RESIDUAL_CONFIRMATION_REPORT = "reports/post_r2p_archive_libero_spatial_task5_residual_confirmation_result.json"
R2P_ARCHIVE_REPORT = "reports/post_secondprior_libero_spatial_20260727_r2p_xvla_archive_decision.json"


def build_sgl_xvla_stage0_gate() -> dict[str, Any]:
    """Return the frozen no-training Stage 0 gate for SGL-XVLA."""

    return {
        "schema_version": "2026-07-18.epoch5_sgl_xvla_stage0_gate.v1",
        "freeze_id": "epoch5_sgl_xvla_task5_stage0_gate_v1",
        "stage": "epoch_5_sgl_xvla_task5_stage0_spec_frozen_no_training",
        "date_kst": "2026-07-18",
        "candidate": {
            "candidate_id": CANDIDATE_ID,
            "name": "Support-Gated Lift for X-VLA",
            "status": "FROZEN_STAGE0_SPEC_ONLY",
            "candidate_generation_report": CANDIDATE_GENERATION_REPORT,
            "residual_confirmation_report": RESIDUAL_CONFIRMATION_REPORT,
            "archived_r2p_xvla_report": R2P_ARCHIVE_REPORT,
        },
        "target": {
            "suite": TARGET_SUITE,
            "task_id": TARGET_TASK_ID,
            "instruction": TARGET_INSTRUCTION,
            "repeated_shared_failure_identities": list(RESIDUAL_IDENTITIES),
            "xvla_solved_clean_retention_identities": list(CLEAN_RETENTION_IDENTITIES),
            "held_out_confirmatory_identity_pool": list(HELD_OUT_CONFIRMATORY_IDENTITY_POOL),
            "held_out_pool_frozen_before_any_ours_result": True,
        },
        "authorization_boundary": {
            "stage0_spec_created": True,
            "training_happened_at_freeze": False,
            "optimizer_step_happened_at_freeze": False,
            "checkpoint_written_at_freeze": False,
            "closed_loop_ours_evaluation_happened_at_freeze": False,
            "model_loaded_at_freeze": False,
            "simulator_episode_count_at_freeze": 0,
            "candidate_generation_was_authorized": True,
            "training_authorized_by_this_gate": False,
            "ours_rollout_authorized_by_this_gate": False,
            "max_candidates_in_current_generation": 2,
            "selected_primary_candidate": CANDIDATE_ID,
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
        ],
        "inference_input_policy": {
            "allowed_inputs": [
                "RGB observations",
                "wrist RGB observations when already part of the policy input",
                "proprioception already available to X-VLA",
                "language instruction",
                "recent action history if already produced by the policy",
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
                "Repeated failures occur when the black bowl begins on the ramekin; "
                "the plausible mechanism is insufficient support separation before "
                "lateral transport to the plate."
            ),
            "candidate_component": (
                "A support-condition detector gates a bounded lift/regrasp action "
                "bias for the first one or two chunks, then returns control to the "
                "base X-VLA policy."
            ),
            "claim_axis_if_future_training_is_authorized": (
                "Improve paired success on the repeated task5 support-separation "
                "residual while preserving X-VLA successes on clean-retention "
                "identities and non-task5 spatial controls."
            ),
        },
        "stage0_required_checks": [
            {
                "check_id": "support_observability_no_training",
                "purpose": "Verify that bowl-on-ramekin support context is observable from allowed inputs.",
                "allowed_evidence": [
                    "existing failure videos",
                    "existing successful prior videos",
                    "RGB/proprio traces already produced by frozen diagnostics",
                    "HDF5 demonstrations for offline labeling only",
                ],
                "forbidden_evidence": [
                    "simulator object pose at inference",
                    "reward at inference",
                    "success flag at inference",
                    "any Ours rollout",
                ],
                "pass_condition": (
                    "A deterministic observation-only spec can identify when the "
                    "support-gated lift may activate, and it is frozen before "
                    "training or Ours rollout."
                ),
            },
            {
                "check_id": "action_bias_bounds_no_optimizer",
                "purpose": "Specify finite bounded action-bias limits without optimization.",
                "allowed_evidence": [
                    "expert HDF5 action statistics",
                    "X-VLA action range diagnostics",
                    "Base/OpenVLA failure action chunk metadata",
                ],
                "forbidden_evidence": [
                    "optimizer.step",
                    "checkpoint selection",
                    "closed-loop Ours success",
                ],
                "pass_condition": "All proposed action biases remain within a frozen safe range and are independent of residual outcome tuning.",
            },
            {
                "check_id": "simple_fixed_lift_control_frozen",
                "purpose": "Freeze the strongest simple explanation before method execution.",
                "allowed_evidence": [
                    "hand-authored fixed lift template specification",
                    "no-training action-range sanity checks",
                ],
                "forbidden_evidence": [
                    "choosing the control after seeing Ours results",
                    "multiple redundant controls for the same objection",
                ],
                "pass_condition": "A single fixed lift/regrasp control is specified as the primary simple-control comparator.",
            },
            {
                "check_id": "held_out_identity_manifest_frozen",
                "purpose": "Protect against test-set tuning.",
                "allowed_evidence": [
                    "frozen held-out identity list",
                    "frozen clean-retention identity list",
                ],
                "forbidden_evidence": [
                    "changing held-out identities after Ours results",
                    "selecting checkpoints on held-out closed-loop success",
                ],
                "pass_condition": "Development, clean-retention, and held-out identity roles are written before any SGL rollout.",
            },
        ],
        "future_comparator_roles": {
            "base": {
                "comparator_role": "BASE",
                "scientific_question": "Does SGL improve the frozen backbone on the repeated support-separation residual?",
                "claim_metric": "paired success on residual and held-out identities",
                "blocking_condition": "No practically meaningful residual-success gain or unacceptable clean degradation.",
                "nonblocking_tradeoffs": "Small latency increases are acceptable only inside frozen bounds.",
            },
            "closest_prior": {
                "comparator_role": "FIRST_PRIOR_AND_SECOND_PRIOR",
                "scientific_question": "Does SGL advance beyond X-VLA and OpenVLA-OFT INT4 on the same local protocol?",
                "claim_metric": "paired residual success and clean-retention breakdown",
                "blocking_condition": "Official priors solve held-out residuals or SGL fails to improve the claim axis.",
                "nonblocking_tradeoffs": "Comparable success with lower intervention count may be considered if preregistered.",
            },
            "key_ablation": {
                "comparator_role": "ABLATION",
                "scientific_question": "Is the support gate necessary?",
                "claim_metric": "gated vs ungated residual success and clean retention",
                "blocking_condition": "Ungated/uniform adaptation matches or exceeds SGL with no mechanism or efficiency loss.",
                "nonblocking_tradeoffs": "Ungated clean advantage is nonblocking only if residual claim gain is stronger and retention is within margin.",
            },
            "simple_control": {
                "comparator_role": "CONTROL",
                "scientific_question": "Can a fixed lift/regrasp template explain the gain?",
                "claim_metric": "residual success, clean retention, latency, and intervention count",
                "blocking_condition": "Fixed lift matches SGL at equal or lower cost without generalization loss.",
                "nonblocking_tradeoffs": "A fixed lift that works on one development reset but fails held-out residuals does not explain SGL.",
            },
        },
        "kill_rules": [
            "Any simulator object state, contact state, reward, success flag, HDF5 identity, or reset identity is used as an inference input.",
            "Any training, optimizer step, checkpoint write, model load, simulator episode, or Ours rollout happens during Stage 0 freeze.",
            "R2P-XVLA or any closed method is reopened, retuned, renamed as SGL, or used as the implementation substrate.",
            "The support condition cannot be specified from allowed observations before training.",
            "The fixed lift simple control is not frozen before any method execution.",
            "Held-out identities change after any Ours result.",
            "Clean-retention identities 20260731 and 20260732 are omitted from the next gate.",
        ],
        "next_action_after_stage0_freeze": (
            "Create a Stage 0 observability/report-only runner or report that checks "
            "support-condition observability from existing artifacts only. Do not train, "
            "write checkpoints, or run Ours."
        ),
    }


def validate_sgl_xvla_stage0_gate(spec: dict[str, Any]) -> list[str]:
    """Return validation errors for the SGL Stage 0 gate."""

    errors: list[str] = []
    if spec.get("candidate", {}).get("candidate_id") != CANDIDATE_ID:
        errors.append("candidate_id must be SGL-XVLA")
    if spec.get("target", {}).get("repeated_shared_failure_identities") != RESIDUAL_IDENTITIES:
        errors.append("residual identities must be frozen to 20260727/20260730/20260733")
    if spec.get("target", {}).get("xvla_solved_clean_retention_identities") != CLEAN_RETENTION_IDENTITIES:
        errors.append("clean-retention identities must be 20260731/20260732")
    if spec.get("target", {}).get("held_out_confirmatory_identity_pool") != HELD_OUT_CONFIRMATORY_IDENTITY_POOL:
        errors.append("held-out identity pool must be frozen before Ours results")
    boundary = spec.get("authorization_boundary", {})
    for key in [
        "training_happened_at_freeze",
        "optimizer_step_happened_at_freeze",
        "checkpoint_written_at_freeze",
        "closed_loop_ours_evaluation_happened_at_freeze",
        "model_loaded_at_freeze",
        "training_authorized_by_this_gate",
        "ours_rollout_authorized_by_this_gate",
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
    policy = spec.get("inference_input_policy", {})
    if policy.get("privileged_state_at_inference") is not False:
        errors.append("privileged state at inference must be false")
    if policy.get("reward_or_success_at_inference") is not False:
        errors.append("reward/success at inference must be false")
    if policy.get("reset_identity_at_inference") is not False:
        errors.append("reset identity at inference must be false")
    forbidden = set(policy.get("forbidden_inputs", []))
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
    if not required_forbidden.issubset(forbidden):
        errors.append("forbidden inference inputs are incomplete")
    check_ids = {item.get("check_id") for item in spec.get("stage0_required_checks", [])}
    required_checks = {
        "support_observability_no_training",
        "action_bias_bounds_no_optimizer",
        "simple_fixed_lift_control_frozen",
        "held_out_identity_manifest_frozen",
    }
    if check_ids != required_checks:
        errors.append("stage0 checks must exactly match the frozen required checks")
    if any("R2P-XVLA" in rule and "reopened" not in rule and "retuned" not in rule for rule in spec.get("kill_rules", [])):
        errors.append("R2P kill rule must explicitly block reopen/retune")
    return errors


def write_sgl_xvla_stage0_gate(output_path: Path) -> dict[str, Any]:
    """Build, validate, and write the deterministic Stage 0 gate JSON."""

    spec = build_sgl_xvla_stage0_gate()
    errors = validate_sgl_xvla_stage0_gate(spec)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(spec, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=STAGE0_ARTIFACT)
    args = parser.parse_args()
    spec = write_sgl_xvla_stage0_gate(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "freeze_id": spec["freeze_id"],
                "candidate_id": spec["candidate"]["candidate_id"],
                "training_authorized": spec["authorization_boundary"]["training_authorized_by_this_gate"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
