"""Report-only no-training rollout protocol freeze for SGL-XVLA.

This preregisters the first executable protocol that may later compare SGL-XVLA
against the fixed lift/regrasp simple control and frozen X-VLA evidence. It does
not implement the runner, load a model, launch a simulator, train, write
checkpoints, or run Ours/control episodes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tca_map.xvla_spatial_task5.sgl_stage0_adjudication import (
    STAGE0_ADJUDICATION_ARTIFACT,
    build_sgl_stage0_adjudication,
    validate_sgl_stage0_adjudication,
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


ROLLOUT_PROTOCOL_ARTIFACT = Path("runs/xvla_prior/epoch5_sgl_xvla_task5_no_training_rollout_protocol_v1.json")


DEVELOPMENT_EVALUATION_IDENTITIES = list(RESIDUAL_IDENTITIES) + list(CLEAN_RETENTION_IDENTITIES)
ROLLOUT_ARMS = [
    "X-VLA-FROZEN-EXISTING-EVIDENCE",
    "FIXED-LIFT-REGRASP-CONTROL",
    "SGL-XVLA",
]


def build_sgl_rollout_protocol() -> dict[str, Any]:
    """Build the deterministic no-training SGL/control rollout protocol."""

    adjudication = build_sgl_stage0_adjudication()
    adjudication_errors = validate_sgl_stage0_adjudication(adjudication)
    protocol_valid = not adjudication_errors
    decision = (
        "SGL_NO_TRAINING_ROLLOUT_PROTOCOL_FROZEN_NO_EPISODES_RUN"
        if protocol_valid
        else "SGL_NO_TRAINING_ROLLOUT_PROTOCOL_NOT_VERIFIED"
    )

    return {
        "schema_version": "2026-07-18.epoch5_sgl_xvla_rollout_protocol.v1",
        "stage": "epoch_5_sgl_xvla_task5_no_training_rollout_protocol_freeze_report_only",
        "candidate_id": CANDIDATE_ID,
        "decision": decision,
        "target": {
            "suite": TARGET_SUITE,
            "task_id": TARGET_TASK_ID,
            "instruction": TARGET_INSTRUCTION,
            "development_residual_identities": list(RESIDUAL_IDENTITIES),
            "clean_retention_identities": list(CLEAN_RETENTION_IDENTITIES),
            "development_evaluation_identities": list(DEVELOPMENT_EVALUATION_IDENTITIES),
            "held_out_confirmatory_identity_pool": list(HELD_OUT_CONFIRMATORY_IDENTITY_POOL),
            "held_out_used_in_this_protocol": False,
        },
        "upstream_adjudication": {
            "artifact": str(STAGE0_ADJUDICATION_ARTIFACT),
            "validation_errors": adjudication_errors,
            "valid": protocol_valid,
            "decision": adjudication["decision"],
        },
        "rollout_arms": [
            {
                "arm_id": "X-VLA-FROZEN-EXISTING-EVIDENCE",
                "comparator_role": "BASE_AND_FIRST_PRIOR_REFERENCE",
                "execution_plan": "reuse_existing_valid_XVLA_identity_results_when_hashes_match",
                "rerun_policy": "do_not_rerun_unless_integrity_or_identity_hash_is_missing",
                "training_happened": False,
                "checkpoint_selection_allowed": False,
            },
            {
                "arm_id": "FIXED-LIFT-REGRASP-CONTROL",
                "comparator_role": "SIMPLE_CONTROL",
                "execution_plan": "future_no_training_control_rollout_only_after_runner_preflight",
                "episode_identities": list(DEVELOPMENT_EVALUATION_IDENTITIES),
                "held_out_episode_identities": [],
                "training_happened": False,
                "checkpoint_selection_allowed": False,
            },
            {
                "arm_id": "SGL-XVLA",
                "comparator_role": "OURS_CANDIDATE",
                "execution_plan": "future_no_training_sgl_rollout_only_after_runner_preflight",
                "episode_identities": list(DEVELOPMENT_EVALUATION_IDENTITIES),
                "held_out_episode_identities": [],
                "training_happened": False,
                "checkpoint_selection_allowed": False,
            },
        ],
        "episode_budget_if_later_authorized": {
            "control_development_episodes": len(DEVELOPMENT_EVALUATION_IDENTITIES),
            "sgl_development_episodes": len(DEVELOPMENT_EVALUATION_IDENTITIES),
            "xvla_reference_rerun_episodes": 0,
            "held_out_episodes": 0,
            "total_new_simulator_episodes": len(DEVELOPMENT_EVALUATION_IDENTITIES) * 2,
            "episode_budget_is_frozen_before_any_new_simulator_episode": True,
        },
        "metrics": [
            "success",
            "final_reward",
            "steps",
            "action_chunk_count",
            "activation_count",
            "intervention_step_count",
            "pre_bias_component_action_range",
            "post_bias_component_action_range",
            "added_clip_count_by_identity",
            "policy_latency_seconds",
            "environment_latency_seconds",
            "exception",
            "video_sha256",
            "result_json_sha256",
        ],
        "development_decision_rules": {
            "primary_residual_pass_condition": (
                "SGL-XVLA succeeds on at least 2 of 3 development residual identities "
                "and improves over the frozen X-VLA residual reference."
            ),
            "clean_retention_pass_condition": (
                "SGL-XVLA succeeds on both clean-retention identities 20260731 and "
                "20260732 with zero positive added clipping."
            ),
            "simple_control_blocking_condition": (
                "The fixed lift/regrasp control blocks SGL novelty if it matches or "
                "exceeds SGL residual success at equal/lower intervention cost with "
                "no worse clean retention or added clipping."
            ),
            "held_out_advancement_condition": (
                "Only if residual and clean-retention conditions pass and the simple "
                "control does not explain the gain may a separate held-out protocol "
                "be frozen. Held-out identities are not run in this protocol."
            ),
            "failure_statuses": [
                "KEY_COMPONENT_NOT_SUPPORTED",
                "SIMPLE_CONTROL_EXPLAINS_GAIN",
                "CLEAN_RETENTION_FAILURE",
                "UNDERPOWERED_ONE_EXPANSION_ALLOWED",
                "IMPLEMENTATION_DATA_OR_RESOURCE_FAILURE",
            ],
        },
        "durability_requirements_for_future_worker": {
            "required_files": [
                "windows_launcher_pid.txt",
                "wsl_worker_pid.txt",
                "heartbeat.txt",
                "worker_started_at.txt",
                "worker_finished_at.txt",
                "exit_code.txt",
                "stdout.log",
                "stderr.log",
                "result.json",
                "result.md",
            ],
            "required_hashes": [
                "result_json_sha256",
                "video_sha256_by_episode_when_video_exists",
                "source_file_sha256",
            ],
            "long_running_worker_policy": "must have PID, heartbeat, logs, exit code, hashes, and report",
        },
        "forbidden_actions": [
            "No training, LoRA, QLoRA, optimizer step, or checkpoint write.",
            "No held-out identity rollout in this development protocol.",
            "No simulator object state, contact state, reward, success flag, reset identity, or HDF5 identity as an inference input.",
            "No changing identities, arms, metrics, thresholds, or stopping rules after the first new simulator episode.",
            "No reopening R2P-XVLA or any archived method.",
        ],
        "comparator_role_calibration": {
            "base_and_first_prior": "X-VLA reference answers whether SGL improves the frozen prior/backbone on the residual claim axis.",
            "simple_control": "The fixed lift/regrasp control answers whether a trivial explanation accounts for substantially all gain.",
            "clean_retention": "Clean identities test degradation and added clipping, not residual gain by themselves.",
            "held_out": "Held-out identities are reserved for a later confirmatory protocol and are not part of this development screen.",
            "universal_beat_all_rule_applied": False,
        },
        "execution_classification": {
            "execution_type": "REPORT_ONLY",
            "evidence_role": "NO_TRAINING_ROLLOUT_PROTOCOL_PREREGISTRATION",
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
            "protocol_frozen": protocol_valid,
            "runner_implementation_preflight_authorized_next": protocol_valid,
            "simulator_episode_authorized_by_this_artifact": False,
            "training_authorized": False,
            "checkpoint_write_authorized": False,
            "control_rollout_authorized_now": False,
            "ours_rollout_authorized_now": False,
            "held_out_rollout_authorized": False,
            "paper_candidate_go": False,
            "prototype_go": False,
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
            "rollout_protocol_frozen": protocol_valid,
            "candidate_can_advance_to_runner_preflight": protocol_valid,
            "candidate_can_train": False,
            "candidate_can_roll_out_ours_now": False,
            "control_can_roll_out_now": False,
            "held_out_can_roll_out_now": False,
        },
        "next_action": (
            "Implement a report-only runner preflight for the frozen SGL-XVLA and "
            "fixed-control rollout protocol. Do not launch simulator episodes until "
            "the preflight passes."
        ),
    }


def validate_sgl_rollout_protocol(protocol: dict[str, Any]) -> list[str]:
    """Return validation errors for the frozen no-training rollout protocol."""

    errors: list[str] = []
    if protocol.get("candidate_id") != CANDIDATE_ID:
        errors.append("candidate_id must be SGL-XVLA")
    upstream = protocol.get("upstream_adjudication", {})
    if upstream.get("valid") is not True:
        errors.append("upstream Stage0 adjudication must validate")
    target = protocol.get("target", {})
    if target.get("development_residual_identities") != RESIDUAL_IDENTITIES:
        errors.append("development residual identities must remain frozen")
    if target.get("clean_retention_identities") != CLEAN_RETENTION_IDENTITIES:
        errors.append("clean-retention identities must remain frozen")
    if target.get("held_out_confirmatory_identity_pool") != HELD_OUT_CONFIRMATORY_IDENTITY_POOL:
        errors.append("held-out identity pool must remain frozen")
    if target.get("held_out_used_in_this_protocol") is not False:
        errors.append("held-out identities must not be used in this development protocol")
    arms = protocol.get("rollout_arms", [])
    if [arm.get("arm_id") for arm in arms] != ROLLOUT_ARMS:
        errors.append("rollout arms must match the frozen arm order")
    budget = protocol.get("episode_budget_if_later_authorized", {})
    expected_new_episodes = len(DEVELOPMENT_EVALUATION_IDENTITIES) * 2
    if budget.get("total_new_simulator_episodes") != expected_new_episodes:
        errors.append("new simulator episode budget must be exactly control+SGL over development identities")
    if budget.get("held_out_episodes") != 0:
        errors.append("held-out episode budget must be zero")
    if budget.get("episode_budget_is_frozen_before_any_new_simulator_episode") is not True:
        errors.append("episode budget must be frozen before any new simulator episode")
    rules = protocol.get("development_decision_rules", {})
    if "at least 2 of 3" not in rules.get("primary_residual_pass_condition", ""):
        errors.append("primary residual pass condition must be explicit")
    if "both clean-retention identities" not in rules.get("clean_retention_pass_condition", ""):
        errors.append("clean-retention pass condition must require both identities")
    if "fixed lift/regrasp control blocks" not in rules.get("simple_control_blocking_condition", ""):
        errors.append("simple-control blocking condition must be explicit")
    durability = protocol.get("durability_requirements_for_future_worker", {})
    required_files = set(durability.get("required_files", []))
    for required in ["heartbeat.txt", "exit_code.txt", "result.json", "result.md"]:
        if required not in required_files:
            errors.append(f"missing durability file requirement: {required}")
    forbidden = " ".join(protocol.get("forbidden_actions", []))
    for phrase in [
        "No training",
        "No held-out identity rollout",
        "No simulator object state",
        "No changing identities",
        "No reopening R2P-XVLA",
    ]:
        if phrase not in forbidden:
            errors.append(f"forbidden action missing phrase: {phrase}")
    calibration = protocol.get("comparator_role_calibration", {})
    if calibration.get("universal_beat_all_rule_applied") is not False:
        errors.append("rollout protocol must not use a universal beat-all rule")
    auth = protocol.get("authorization_boundary", {})
    if auth.get("runner_implementation_preflight_authorized_next") is not True:
        errors.append("runner preflight should be authorized next")
    for key in [
        "simulator_episode_authorized_by_this_artifact",
        "training_authorized",
        "checkpoint_write_authorized",
        "control_rollout_authorized_now",
        "ours_rollout_authorized_now",
        "held_out_rollout_authorized",
        "paper_candidate_go",
        "prototype_go",
    ]:
        if auth.get(key) is not False:
            errors.append(f"{key} must be false")
    execution = protocol.get("execution_classification", {})
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
    conclusion = protocol.get("bounded_conclusion", {})
    if conclusion.get("rollout_protocol_frozen") is not True:
        errors.append("rollout protocol must be frozen")
    if conclusion.get("candidate_can_train") is not False:
        errors.append("candidate cannot train after protocol freeze")
    if conclusion.get("candidate_can_roll_out_ours_now") is not False:
        errors.append("candidate cannot roll out Ours now")
    if conclusion.get("control_can_roll_out_now") is not False:
        errors.append("control cannot roll out now")
    if conclusion.get("held_out_can_roll_out_now") is not False:
        errors.append("held-out cannot roll out now")
    return errors


def write_sgl_rollout_protocol(output_path: Path) -> dict[str, Any]:
    """Build, validate, and write the rollout protocol JSON."""

    protocol = build_sgl_rollout_protocol()
    errors = validate_sgl_rollout_protocol(protocol)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return protocol


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROLLOUT_PROTOCOL_ARTIFACT)
    args = parser.parse_args()
    protocol = write_sgl_rollout_protocol(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": protocol["decision"],
                "rollout_protocol_frozen": protocol["bounded_conclusion"]["rollout_protocol_frozen"],
                "simulator_episode_authorized_by_this_artifact": protocol["authorization_boundary"][
                    "simulator_episode_authorized_by_this_artifact"
                ],
                "candidate_can_train": protocol["bounded_conclusion"]["candidate_can_train"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
