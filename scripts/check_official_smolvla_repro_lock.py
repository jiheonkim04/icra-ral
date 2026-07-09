from __future__ import annotations

import argparse
import copy
import re
from pathlib import Path
from typing import Any

import yaml


ALLOWED_FINAL_DECISIONS = {
    "ROLLOUT_PROTOCOL_READY",
    "LORA_CHECKPOINTS_MISSING_REGENERATION_REQUIRED",
    "REVISION_LOCK_INCOMPLETE",
    "NEEDS_WSL_OR_LINUX_OFFICIAL_EVAL",
    "OFFICIAL_EVAL_ENV_BLOCKED",
    "PROTOCOL_FIX_INCOMPLETE",
}

ALLOWED_ENV_STATUSES = {
    "OFFICIAL_EVAL_ENV_READY",
    "NEEDS_WSL_OR_LINUX",
    "MISSING_OFFICIAL_EVAL_DEPENDENCY",
    "OFFICIAL_EVAL_ENTRYPOINT_UNCLEAR",
}

ALLOWED_CHECKPOINT_STATUSES = {
    "CHECKPOINT_COMPLETE",
    "CHECKPOINT_PARTIAL",
    "CHECKPOINT_MISSING",
    "IDENTITY_UNPROVEN",
}

REQUIRED_BASELINES = {
    "frozen_base",
    "rank4_lora",
    "validation_selected_action_space_static_mix",
    "task_or_instruction_router_proxy",
    "frame_oracle_upper_bound",
    "task_oracle_upper_bound",
}

HEX40 = re.compile(r"^[0-9a-f]{40}$")


class ReproLockError(ValueError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReproLockError(message)


def _require_keys(mapping: dict[str, Any], keys: set[str], prefix: str) -> None:
    missing = sorted(keys - set(mapping))
    _require(not missing, f"{prefix} missing required keys: {missing}")


def load_lock(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    _require(isinstance(data, dict), "lock file must parse to a YAML mapping")
    return data


def validate_lock(data: dict[str, Any]) -> dict[str, Any]:
    data = copy.deepcopy(data)
    _require_keys(
        data,
        {
            "audit_boundary",
            "model",
            "dataset",
            "environment",
            "baseline_names",
            "lora_checkpoint_policy",
            "rollout_action_semantics",
            "official_eval_readiness",
            "closed_loop_rollout_protocol",
            "final_decision",
        },
        "root",
    )

    final_decision = data["final_decision"]
    _require(final_decision in ALLOWED_FINAL_DECISIONS, f"invalid final_decision: {final_decision}")

    boundary = data["audit_boundary"]
    _require(boundary.get("no_experiment_protocol_fix") is True, "protocol fix must be marked no-experiment")
    for key in [
        "experiments_run",
        "training_run",
        "gpu_used",
        "simulator_rollout_run",
        "downloads_run",
        "openvla_oft_run",
        "fcar_run",
        "lora_seeds_regenerated",
        "historical_metrics_modified",
    ]:
        _require(boundary.get(key) is False, f"audit boundary expected {key}: false")

    for section_name in ["model", "dataset"]:
        section = data[section_name]
        _require(section.get("revision_status") == "REVISION_LOCKED", f"{section_name} revision not locked")
        revision = section.get("hf_revision")
        _require(isinstance(revision, str) and bool(HEX40.match(revision)), f"{section_name} hf_revision is not 40 hex")
        proof = section.get("revision_proof", {})
        unique_revisions = proof.get("unique_metadata_revisions", [])
        _require(unique_revisions == [revision], f"{section_name} metadata revisions do not match locked revision")

    baselines = data["baseline_names"]
    _require_keys(baselines, REQUIRED_BASELINES, "baseline_names")
    static_mix = baselines["validation_selected_action_space_static_mix"]
    _require(static_mix.get("adapter_weight_merge") is False, "static mix must not be marked as weight merge")
    _require(static_mix.get("adapter_soup") is False, "static mix must not be marked as adapter soup")
    _require(static_mix.get("alpha_source") == "validation_only", "static mix alpha must be validation-only")
    _require(baselines["task_or_instruction_router_proxy"].get("official_moira") is False, "router proxy is not official MoIRA")

    checkpoint_policy = data["lora_checkpoint_policy"]
    _require(checkpoint_policy.get("required_for_official_rollout_or_final_report") is True, "LoRA checkpoints must be required")
    _require(checkpoint_policy.get("required_seeds") == [11, 22, 33], "LoRA required seeds must be [11, 22, 33]")
    seed_statuses = checkpoint_policy.get("seed_statuses", {})
    _require_keys(seed_statuses, {"seed_11", "seed_22", "seed_33"}, "lora_checkpoint_policy.seed_statuses")
    statuses = []
    for seed_name, seed_info in seed_statuses.items():
        status = seed_info.get("status")
        _require(status in ALLOWED_CHECKPOINT_STATUSES, f"{seed_name} has invalid checkpoint status {status}")
        statuses.append(status)
    missing_or_unproven = any(status != "CHECKPOINT_COMPLETE" for status in statuses)

    action_semantics = data["rollout_action_semantics"]
    _require(action_semantics.get("official_entrypoint_required") == "lerobot-eval --env.type=libero", "official eval entrypoint changed")
    _require(action_semantics.get("static_mix_single_action_after_independent_queues_allowed") is False, "static mix queue rule changed")

    readiness = data["official_eval_readiness"]
    env_status = readiness.get("env_status")
    _require(env_status in ALLOWED_ENV_STATUSES, f"invalid official eval env_status: {env_status}")

    if final_decision == "ROLLOUT_PROTOCOL_READY":
        _require(not missing_or_unproven, "rollout ready cannot have missing or unproven LoRA checkpoints")
        _require(env_status == "OFFICIAL_EVAL_ENV_READY", "rollout ready requires official eval env ready")
    if final_decision == "LORA_CHECKPOINTS_MISSING_REGENERATION_REQUIRED":
        _require(missing_or_unproven, "LoRA checkpoint regeneration decision requires missing or unproven checkpoints")

    return data


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the official SmolVLA reproducibility lock.")
    parser.add_argument("path", nargs="?", default="configs/official_smolvla_repro_lock.yaml")
    args = parser.parse_args()
    data = validate_lock(load_lock(args.path))
    print(f"OFFICIAL_SMOLVLA_REPRO_LOCK_OK final_decision={data['final_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
