"""Report-only observability audit for SGL-XVLA Stage 0.

The audit checks whether the SGL support gate can be specified from allowed
inputs before any training or Ours rollout. It intentionally does not load
models, inspect simulator state, run image classifiers, launch simulators,
train, or write checkpoints.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

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


OBSERVABILITY_ARTIFACT = Path("runs/xvla_prior/epoch5_sgl_xvla_task5_observability_audit_v1.json")


def build_sgl_observability_audit() -> dict[str, Any]:
    """Build the deterministic report-only observability audit."""

    stage0 = build_sgl_xvla_stage0_gate()
    stage0_errors = validate_sgl_xvla_stage0_gate(stage0)
    instruction = TARGET_INSTRUCTION.lower()
    support_tokens = ["ramekin"]
    language_support_observable = all(token in instruction for token in support_tokens)
    allowed_activation_sources = ["language instruction"]
    forbidden_sources_consulted: list[str] = []
    visual_progress_observability_verified = False
    decision = (
        "SGL_STAGE0_SUPPORT_OBSERVABILITY_LANGUAGE_PASS_VISUAL_PROGRESS_UNVERIFIED_NO_TRAINING"
        if not stage0_errors and language_support_observable
        else "SGL_STAGE0_SUPPORT_OBSERVABILITY_NOT_VERIFIED"
    )
    return {
        "schema_version": "2026-07-18.epoch5_sgl_xvla_observability_audit.v1",
        "stage": "epoch_5_sgl_xvla_task5_stage0_observability_report_only",
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
        "stage0_gate": {
            "artifact": str(STAGE0_ARTIFACT),
            "validation_errors": stage0_errors,
            "valid": not stage0_errors,
        },
        "observability": {
            "support_condition": "black bowl starts on the ramekin",
            "support_tokens_required_in_instruction": support_tokens,
            "language_support_observable": language_support_observable,
            "allowed_activation_sources": allowed_activation_sources,
            "forbidden_sources_consulted": forbidden_sources_consulted,
            "privileged_state_consulted": False,
            "reward_or_success_consulted": False,
            "reset_identity_consulted_for_activation": False,
            "visual_support_detector_trained_or_run": False,
            "visual_progress_observability_verified": visual_progress_observability_verified,
            "visual_progress_evidence_gap": (
                "No observation-only progress detector is verified here. This audit "
                "only supports a frozen language-level support gate. Any future "
                "visual/progress monitor must pass a separate no-training gate."
            ),
        },
        "clean_retention_implication": {
            "same_instruction_for_clean_retention_identities": True,
            "support_gate_would_activate_on_clean_retention_identities": True,
            "reason": (
                "The support condition is instruction-level, so clean-retention "
                "identities 20260731 and 20260732 must be included before any "
                "method execution."
            ),
            "clean_retention_identities_required_in_next_gate": list(CLEAN_RETENTION_IDENTITIES),
        },
        "bounded_conclusion": {
            "support_gate_observability_passed": bool(language_support_observable and not stage0_errors),
            "progress_detector_observability_passed": False,
            "candidate_can_advance_to_action_bias_bounds_gate": bool(language_support_observable and not stage0_errors),
            "candidate_can_train": False,
            "candidate_can_roll_out_ours": False,
        },
        "execution_classification": {
            "execution_type": "REPORT_ONLY",
            "evidence_role": "OURS_CANDIDATE_STAGE0_OBSERVABILITY_AUDIT",
            "artifact_status": "NOT_APPLICABLE",
            "simulator_episode_count": 0,
            "vla_model_loaded": False,
            "visual_model_loaded": False,
            "training_happened": False,
            "optimizer_step_happened": False,
            "checkpoint_written": False,
            "closed_loop_ours_evaluation_happened": False,
        },
        "no_training_no_ours_booleans": {
            "training_happened": False,
            "optimizer_step_happened": False,
            "checkpoint_written": False,
            "closed_loop_ours_evaluation_happened": False,
            "ours_rollout_happened": False,
            "lora_or_qlora_training_happened": False,
        },
        "next_action": (
            "Freeze an action-bias bounds/no-optimizer gate for SGL-XVLA using "
            "expert action statistics and existing failure metadata only."
        ),
    }


def validate_sgl_observability_audit(audit: dict[str, Any]) -> list[str]:
    """Return validation errors for an observability audit."""

    errors: list[str] = []
    if audit.get("candidate_id") != CANDIDATE_ID:
        errors.append("candidate_id must be SGL-XVLA")
    if not audit.get("stage0_gate", {}).get("valid"):
        errors.append("stage0 gate must validate before observability audit")
    obs = audit.get("observability", {})
    if obs.get("language_support_observable") is not True:
        errors.append("language support condition must be observable")
    if obs.get("privileged_state_consulted") is not False:
        errors.append("privileged state must not be consulted")
    if obs.get("reward_or_success_consulted") is not False:
        errors.append("reward/success must not be consulted")
    if obs.get("reset_identity_consulted_for_activation") is not False:
        errors.append("reset identity must not be consulted for activation")
    if obs.get("visual_support_detector_trained_or_run") is not False:
        errors.append("visual detector must not be trained or run")
    if obs.get("visual_progress_observability_verified") is not False:
        errors.append("visual progress observability must remain unverified in this audit")
    if obs.get("forbidden_sources_consulted") != []:
        errors.append("forbidden sources consulted must be empty")
    conclusion = audit.get("bounded_conclusion", {})
    if conclusion.get("support_gate_observability_passed") is not True:
        errors.append("support-gate observability must pass")
    if conclusion.get("progress_detector_observability_passed") is not False:
        errors.append("progress-detector observability must not pass in this audit")
    if conclusion.get("candidate_can_train") is not False:
        errors.append("candidate cannot train after this audit")
    if conclusion.get("candidate_can_roll_out_ours") is not False:
        errors.append("candidate cannot roll out Ours after this audit")
    execution = audit.get("execution_classification", {})
    for key in [
        "vla_model_loaded",
        "visual_model_loaded",
        "training_happened",
        "optimizer_step_happened",
        "checkpoint_written",
        "closed_loop_ours_evaluation_happened",
    ]:
        if execution.get(key) is not False:
            errors.append(f"{key} must be false")
    if execution.get("simulator_episode_count") != 0:
        errors.append("simulator_episode_count must be zero")
    retention = audit.get("clean_retention_implication", {})
    if retention.get("clean_retention_identities_required_in_next_gate") != CLEAN_RETENTION_IDENTITIES:
        errors.append("clean-retention identities must remain required")
    return errors


def write_sgl_observability_audit(output_path: Path) -> dict[str, Any]:
    """Build, validate, and write the observability audit JSON."""

    audit = build_sgl_observability_audit()
    errors = validate_sgl_observability_audit(audit)
    if errors:
        raise ValueError("; ".join(errors))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OBSERVABILITY_ARTIFACT)
    args = parser.parse_args()
    audit = write_sgl_observability_audit(args.output)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "decision": audit["decision"],
                "candidate_can_train": audit["bounded_conclusion"]["candidate_can_train"],
                "support_gate_observability_passed": audit["bounded_conclusion"]["support_gate_observability_passed"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
