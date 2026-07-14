"""MTF-VLA Stage A manifest and official rollout runner.

This module freezes the first closed-loop comparison for MTF-VLA after the
verified adapter checkpoints exist.  It reuses the official LeRobot/LIBERO
policy path and changes only experiment bookkeeping: the five policy specs,
paired task/reset manifest, resume-safe partial writes, and Stage A
adjudication rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tca_map.smolvla.official_closed_loop_scaleup import (
    _cuda_memory,
    _episode_base_record,
    _extract_single_env,
    _json_default,
    _make_env_cfg,
    _round,
    _rss_mb,
    _set_runtime_env,
    _sha256_file,
    trace_one_episode,
)
from tca_map.smolvla.official_wsl_libero_rollout import PolicySpec, _load_policy_and_processors


DATE_KST = "2026-07-14"
BRANCH = "codex/autonomous-until-paper-governance-v2"
METHOD = "MTF-VLA"
CONFIG_ID = "mtf_r20_ret100"
PROPOSAL_HASH = "11DC94A2B75CD8605577AB044E5743DFDA4131A4FA7F6C6A7390519B9F995B31"
STAGE_A_RESET_SEEDS = [20261201, 20261202]
STAGE_B_RESET_SEEDS = [20261203, 20261204]
STAGE_A_TASK_COUNT = 5
STAGE_B_TASK_COUNT = 20
STAGE_A_EPISODES_PER_POLICY = STAGE_A_TASK_COUNT * len(STAGE_A_RESET_SEEDS)
STAGE_B_EPISODES_PER_POLICY = STAGE_B_TASK_COUNT * len(STAGE_B_RESET_SEEDS)
STAGE_A_POLICY_ORDER = [
    "frozen_smolvla",
    "frameskip_proxy_lora",
    "uniform_retained_ratio_lora",
    "mtf_no_retention_ablation",
    "mtf_full",
]
TRAINABLE_POLICY_ROLES = {
    "frameskip_proxy_lora": "closest_external_prior_proxy",
    "mtf_full": "ours",
    "mtf_no_retention_ablation": "key_ablation",
    "uniform_retained_ratio_lora": "strongest_simple_reviewer_killer",
}
TRAINABLE_ADAPTER_DIRS = {
    "frameskip_proxy_lora": "frameskip_proxy_lora/seed_101",
    "mtf_full": "mtf_full/seed_101",
    "mtf_no_retention_ablation": "mtf_no_retention_ablation/seed_101",
    "uniform_retained_ratio_lora": "uniform_retained_ratio_lora/seed_101",
}
FINAL_DECISIONS = {
    "MTF_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT",
    "MTF_STAGE_A_PREFLIGHT_PASS_READY_FOR_ROLLOUT",
    "MTF_STAGE_A_PREFLIGHT_BLOCKED_REPAIR_LOADING_OR_MAPPING",
    "MTF_STAGE_A_MEASUREMENT_INVALID_REPAIR_REQUIRED",
    "MTF_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE",
    "MTF_STAGE_A_CATASTROPHIC_KILL_CLEARLY_WORSE_THAN_BASELINE_OR_ABLATION",
    "MTF_STAGE_A_POSITIVE_TO_STAGE_B_REQUIRED",
    "MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED",
    "MTF_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT",
    "MTF_STAGE_B_MEASUREMENT_INVALID_REPAIR_REQUIRED",
    "MTF_STAGE_B_PROTOTYPE_GO",
    "MTF_STAGE_B_KILL_BASE_NOT_IMPROVED",
    "MTF_STAGE_B_KILL_CLOSEST_PRIOR_EXPLAINS_METHOD",
    "MTF_STAGE_B_KILL_KEY_COMPONENT_NOT_USEFUL",
    "MTF_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD",
    "MTF_STAGE_B_USEFUL_IMPROVEMENT_EXCLUDED",
    "MTF_STAGE_B_UNRESOLVED_EXPANSION_REQUIRED",
}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest().upper()


def _relative_posix(path_text: str) -> str:
    return path_text.replace("\\", "/")


def _repo_relative(path_text: str) -> Path:
    return Path(_relative_posix(path_text))


def _wsl_repo_path(wsl_repo_root: str, relative_path: str) -> str:
    root = str(wsl_repo_root).rstrip("/")
    return f"{root}/{_relative_posix(relative_path).lstrip('/')}"


def mtf_policy_specs() -> list[PolicySpec]:
    return [
        PolicySpec("frozen_smolvla"),
        PolicySpec("frameskip_proxy_lora", TRAINABLE_ADAPTER_DIRS["frameskip_proxy_lora"]),
        PolicySpec("uniform_retained_ratio_lora", TRAINABLE_ADAPTER_DIRS["uniform_retained_ratio_lora"]),
        PolicySpec("mtf_no_retention_ablation", TRAINABLE_ADAPTER_DIRS["mtf_no_retention_ablation"]),
        PolicySpec("mtf_full", TRAINABLE_ADAPTER_DIRS["mtf_full"]),
    ]


def _select_stage_a_tasks(task_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = list(task_manifest.get("tasks") or [])
    if len(tasks) < STAGE_A_TASK_COUNT:
        raise ValueError(f"official task manifest has only {len(tasks)} tasks")
    indices = [(index * len(tasks)) // STAGE_A_TASK_COUNT for index in range(STAGE_A_TASK_COUNT)]
    selected = []
    for stage_index, task_index in enumerate(indices):
        task = dict(tasks[task_index])
        task["stage_a_task_index"] = int(stage_index)
        task["source_official_task_manifest_index"] = int(task_index)
        task["stage_a_selection_rule"] = f"floor(k * {len(tasks)} / {STAGE_A_TASK_COUNT})"
        selected.append(task)
    return selected


def _select_stage_b_tasks(task_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = list(task_manifest.get("tasks") or [])
    if len(tasks) != STAGE_B_TASK_COUNT:
        raise ValueError(f"Stage B expects the frozen official 20-task manifest, found {len(tasks)} tasks")
    selected = []
    for stage_index, task_item in enumerate(tasks):
        task = dict(task_item)
        task["stage_b_task_index"] = int(stage_index)
        task["source_official_task_manifest_index"] = int(stage_index)
        task["stage_b_selection_rule"] = "use all tasks from the frozen official 20-task manifest"
        selected.append(task)
    return selected


def _checkpoint_variant_map(checkpoint_manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["variant"]): dict(item) for item in checkpoint_manifest.get("variants") or []}


def _build_policy_records(args: argparse.Namespace, checkpoint_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    variants = _checkpoint_variant_map(checkpoint_manifest)
    records = []
    for policy in STAGE_A_POLICY_ORDER:
        if policy == "frozen_smolvla":
            records.append(
                {
                    "policy": policy,
                    "role": "unmodified_backbone",
                    "adapter_dir": None,
                    "checkpoint_path": None,
                    "wsl_adapter_path": None,
                    "adapter_model_sha256": None,
                    "adapter_config_sha256": None,
                    "disk_reload": None,
                }
            )
            continue
        if policy not in variants:
            raise ValueError(f"checkpoint manifest missing variant {policy}")
        variant = variants[policy]
        checkpoint_path = _relative_posix(str(variant["checkpoint_path"]))
        records.append(
            {
                "policy": policy,
                "role": TRAINABLE_POLICY_ROLES[policy],
                "proxy_or_reproduction_label": "faithful_local_proxy_not_official_reproduction" if policy == "frameskip_proxy_lora" else None,
                "adapter_dir": TRAINABLE_ADAPTER_DIRS[policy],
                "checkpoint_path": checkpoint_path,
                "wsl_adapter_path": _wsl_repo_path(str(args.wsl_repo_root), checkpoint_path),
                "adapter_model_sha256": str(variant["adapter_model_sha256"]),
                "adapter_config_sha256": str(variant["adapter_config_sha256"]),
                "disk_reload": bool(variant.get("disk_reload")),
                "seed": int(variant.get("seed", 0)),
                "training_event_count": int(variant.get("training_event_count", 0)),
                "validation_action_l2_mean": variant.get("validation_action_l2_mean"),
                "adapter_minus_base_action_l2_p95": variant.get("adapter_minus_base_action_l2_p95"),
            }
        )
    return records


def build_stage_a_manifest(args: argparse.Namespace) -> dict[str, Any]:
    task_manifest = _load_json(Path(args.official_task_manifest))
    checkpoint_manifest = _load_json(Path(args.checkpoint_manifest))
    if checkpoint_manifest.get("final_decision") != "MTF_ALL_ADAPTER_CHECKPOINTS_VERIFIED_STAGE_A_READY":
        raise ValueError("MTF adapter checkpoint manifest is not Stage-A ready")
    if not checkpoint_manifest.get("stage_a_allowed"):
        raise ValueError("MTF adapter checkpoint manifest does not allow Stage A")

    tasks = _select_stage_a_tasks(task_manifest)
    policies = _build_policy_records(args, checkpoint_manifest)
    pairs = []
    for task in tasks:
        for seed in STAGE_A_RESET_SEEDS:
            pairs.append(
                {
                    "pair_id": f"{task['suite']}|task_{task['task_id']}|seed_{seed}",
                    "suite": task["suite"],
                    "task_id": int(task["task_id"]),
                    "instruction": task["instruction"],
                    "reset_seed": int(seed),
                    "stage_a_task_index": int(task["stage_a_task_index"]),
                    "source_official_task_manifest_index": int(task["source_official_task_manifest_index"]),
                }
            )

    episodes = []
    planned_index = 0
    for policy in STAGE_A_POLICY_ORDER:
        for pair in pairs:
            episodes.append(
                {
                    "planned_episode_index": int(planned_index),
                    "episode_id": f"{policy}|{pair['suite']}|task_{pair['task_id']}|seed_{pair['reset_seed']}",
                    "policy": policy,
                    "pair_id": pair["pair_id"],
                    "suite": pair["suite"],
                    "task_id": int(pair["task_id"]),
                    "instruction": pair["instruction"],
                    "reset_seed": int(pair["reset_seed"]),
                }
            )
            planned_index += 1

    payload = {
        "schema_version": 1,
        "method": METHOD,
        "config_id": CONFIG_ID,
        "proposal_hash": PROPOSAL_HASH,
        "branch": BRANCH,
        "date": f"{args.date} KST",
        "mode": "stage_a_manifest",
        "final_decision": "MTF_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT",
        "closed_loop_experiment_happened": False,
        "confirmatory_test_tuning_happened": False,
        "confirmatory_test_identities_used_for_training_or_validation": False,
        "policy_order": list(STAGE_A_POLICY_ORDER),
        "policies": policies,
        "stage_a_reset_seeds": list(STAGE_A_RESET_SEEDS),
        "stage_a_pair_count_per_policy": len(pairs),
        "planned_episode_count": len(episodes),
        "tasks": tasks,
        "pairs": pairs,
        "episodes": episodes,
        "identity_overlap_verification": {
            "stage_a_rollout_reset_seeds": list(STAGE_A_RESET_SEEDS),
            "overlap_with_adapter_training_rollout_identities": 0,
            "overlap_with_validation_search_rollout_identities": 0,
            "overlap_with_previous_known_allocated_rollout_identities": 0,
            "duplicate_evaluation_keys": 0,
            "identical_task_reset_pairs_across_policies": True,
            "note": "MTF adapter training and validation used offline dataset frame splits; these Stage A rollout reset seeds were selected only after checkpoint freeze.",
        },
        "task_balanced_allocation": {
            "task_count": len(tasks),
            "reset_count_per_task": len(STAGE_A_RESET_SEEDS),
            "episodes_per_task_per_policy": len(STAGE_A_RESET_SEEDS),
            "paired_cases_per_policy": len(pairs),
            "fixed_before_rollout": True,
        },
        "task_selection": {
            "source_manifest": str(args.official_task_manifest),
            "source_manifest_sha256": _sha256_file(Path(args.official_task_manifest)),
            "rule": "select 5 global evenly spaced tasks from the frozen official 20-task manifest: floor(k * n / 5)",
            "outcome_dependent": False,
        },
        "reset_identity_selection": {
            "rule": "fresh unused MTF Stage A block after RAC/FANG/EvoState allocated ranges",
            "reset_seeds": list(STAGE_A_RESET_SEEDS),
            "previous_known_allocations_avoided": [
                "official baseline scale-up reset seeds 20260711..20260715",
                "CBFD/SCVC/PSE reset identities 20260716..20260760",
                "CAVM/FANG/RAC/EvoState reset identity blocks through 20261145",
            ],
        },
        "partition_separation": {
            "offline_training_splits": ["train"],
            "offline_validation_splits": ["val"],
            "offline_reserved_confirmatory_splits": ["test"],
            "stage_a_rollout_resets_are_frozen_after_checkpoint_selection": True,
            "stage_a_rollout_resets_used_for_adapter_training": False,
            "stage_a_rollout_resets_used_for_validation_search": False,
        },
        "frozen_stage_a_rules": {
            "permanent_kill_zero_vs_baseline": "mtf_full has 0/10 while any paired baseline has at least 4/10",
            "permanent_kill_clear_degradation": "mtf_full is at least 30 absolute points below a baseline or ablation",
            "small_difference_rule": "small differences, ties, and one- or two-episode gaps advance to Stage B",
            "next_stage_count": "Stage B requires at least 40 paired episodes per key policy",
        },
        "execution": {
            "official_path": "LeRobot SmolVLA/LIBERO policy, processors, action queue, relative 7D control, and official LIBERO success condition",
            "policy_order_affects_environment_initialization": False,
            "environment_initialization_rule": "each episode calls env.reset(seed=[reset_seed]) after constructing the task env; the same task/reset pairs are executed for every policy",
            "base_path_default": str(args.base_path),
            "lora_root_default": str(args.lora_root),
            "libero_config_dir_default": str(args.libero_config_dir),
            "partial_result_path": str(args.stage_a_partial_output),
            "result_path": str(args.stage_a_output),
            "preflight_result_path": str(args.stage_a_preflight_output),
            "resume_rule": "resume only missing (policy, suite, task_id, reset_seed) episode keys",
        },
        "checkpoint_manifest": {
            "path": str(args.checkpoint_manifest),
            "sha256": _sha256_file(Path(args.checkpoint_manifest)),
            "checkpoint_root": _relative_posix(str(checkpoint_manifest.get("checkpoint_root"))),
            "variant_count": int(checkpoint_manifest.get("variant_count", 0)),
        },
    }
    payload["canonical_payload_sha256"] = _sha256_payload({key: value for key, value in payload.items() if key != "canonical_payload_sha256"})
    validate_stage_a_manifest(payload)
    return payload


def build_stage_b_manifest(args: argparse.Namespace) -> dict[str, Any]:
    task_manifest = _load_json(Path(args.official_task_manifest))
    checkpoint_manifest = _load_json(Path(args.checkpoint_manifest))
    stage_a_result = _load_json(Path(args.stage_a_output))
    if checkpoint_manifest.get("final_decision") != "MTF_ALL_ADAPTER_CHECKPOINTS_VERIFIED_STAGE_A_READY":
        raise ValueError("MTF adapter checkpoint manifest is not Stage-A ready")
    if stage_a_result.get("final_decision") not in {
        "MTF_STAGE_A_POSITIVE_TO_STAGE_B_REQUIRED",
        "MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED",
    }:
        raise ValueError("MTF Stage B requires a completed Stage A decision requiring Stage B")
    if int(stage_a_result.get("completed_episode_count", -1)) != len(STAGE_A_POLICY_ORDER) * STAGE_A_EPISODES_PER_POLICY:
        raise ValueError("MTF Stage A result is incomplete")
    if int((stage_a_result.get("summary") or {}).get("exception_count") or 0) != 0:
        raise ValueError("MTF Stage A result has exceptions; repair/adjudication required before Stage B")

    tasks = _select_stage_b_tasks(task_manifest)
    policies = _build_policy_records(args, checkpoint_manifest)
    pairs = []
    for task in tasks:
        for seed in STAGE_B_RESET_SEEDS:
            pairs.append(
                {
                    "pair_id": f"{task['suite']}|task_{task['task_id']}|seed_{seed}",
                    "suite": task["suite"],
                    "task_id": int(task["task_id"]),
                    "instruction": task["instruction"],
                    "reset_seed": int(seed),
                    "stage_b_task_index": int(task["stage_b_task_index"]),
                    "source_official_task_manifest_index": int(task["source_official_task_manifest_index"]),
                }
            )

    episodes = []
    planned_index = 0
    for policy in STAGE_A_POLICY_ORDER:
        for pair in pairs:
            episodes.append(
                {
                    "planned_episode_index": int(planned_index),
                    "episode_id": f"{policy}|{pair['suite']}|task_{pair['task_id']}|seed_{pair['reset_seed']}",
                    "policy": policy,
                    "pair_id": pair["pair_id"],
                    "suite": pair["suite"],
                    "task_id": int(pair["task_id"]),
                    "instruction": pair["instruction"],
                    "reset_seed": int(pair["reset_seed"]),
                }
            )
            planned_index += 1

    payload = {
        "schema_version": 1,
        "method": METHOD,
        "config_id": CONFIG_ID,
        "proposal_hash": PROPOSAL_HASH,
        "branch": BRANCH,
        "date": f"{args.date} KST",
        "mode": "stage_b_manifest",
        "final_decision": "MTF_STAGE_B_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT",
        "closed_loop_experiment_happened": False,
        "confirmatory_test_tuning_happened": False,
        "stage_a_outcome_used_only_for_preregistered_escalation": True,
        "policy_order": list(STAGE_A_POLICY_ORDER),
        "policies": policies,
        "stage_b_reset_seeds": list(STAGE_B_RESET_SEEDS),
        "stage_b_pair_count_per_policy": len(pairs),
        "planned_episode_count": len(episodes),
        "tasks": tasks,
        "pairs": pairs,
        "episodes": episodes,
        "identity_overlap_verification": {
            "stage_a_rollout_reset_seeds": list(STAGE_A_RESET_SEEDS),
            "stage_b_rollout_reset_seeds": list(STAGE_B_RESET_SEEDS),
            "overlap_with_stage_a_reset_seeds": len(set(STAGE_A_RESET_SEEDS) & set(STAGE_B_RESET_SEEDS)),
            "overlap_with_adapter_training_rollout_identities": 0,
            "overlap_with_validation_search_rollout_identities": 0,
            "overlap_with_previous_known_allocated_rollout_identities": 0,
            "duplicate_evaluation_keys": 0,
            "identical_task_reset_pairs_across_policies": True,
            "note": "Stage B uses fresh reset seeds and all official tasks after the frozen Stage A decision required Stage B.",
        },
        "task_balanced_allocation": {
            "task_count": len(tasks),
            "reset_count_per_task": len(STAGE_B_RESET_SEEDS),
            "episodes_per_task_per_policy": len(STAGE_B_RESET_SEEDS),
            "paired_cases_per_policy": len(pairs),
            "fixed_before_rollout": True,
        },
        "task_selection": {
            "source_manifest": str(args.official_task_manifest),
            "source_manifest_sha256": _sha256_file(Path(args.official_task_manifest)),
            "rule": "use all 20 tasks from the frozen official task manifest",
            "outcome_dependent": False,
        },
        "reset_identity_selection": {
            "rule": "fresh unused MTF Stage B block immediately after the Stage A block",
            "reset_seeds": list(STAGE_B_RESET_SEEDS),
            "previous_known_allocations_avoided": [
                "official baseline scale-up reset seeds 20260711..20260715",
                "CBFD/SCVC/PSE reset identities 20260716..20260760",
                "CAVM/FANG/RAC/EvoState reset identity blocks through 20261145",
                "MTF Stage A reset seeds 20261201..20261202",
            ],
        },
        "partition_separation": {
            "offline_training_splits": ["train"],
            "offline_validation_splits": ["val"],
            "offline_reserved_confirmatory_splits": ["test"],
            "stage_b_rollout_resets_are_frozen_after_stage_a_adjudication": True,
            "stage_b_rollout_resets_used_for_adapter_training": False,
            "stage_b_rollout_resets_used_for_validation_search": False,
        },
        "frozen_stage_b_rules": {
            "prototype_go": "mtf_full beats base, FrameSkip proxy, no-retention ablation, and uniform retained-ratio LoRA with at least 10 absolute points prototype gain or consistently positive paired evidence",
            "base_not_improved": "frozen_smolvla matches or beats mtf_full",
            "closest_prior_explains": "frameskip_proxy_lora matches or beats mtf_full",
            "key_component_not_useful": "mtf_no_retention_ablation matches or beats mtf_full",
            "simple_baseline_explains": "uniform_retained_ratio_lora matches or beats mtf_full",
            "expansion_rule": "one expansion to 80 paired episodes is allowed only if Stage B is genuinely unresolved",
        },
        "execution": {
            "official_path": "LeRobot SmolVLA/LIBERO policy, processors, action queue, relative 7D control, and official LIBERO success condition",
            "policy_order_affects_environment_initialization": False,
            "environment_initialization_rule": "each episode calls env.reset(seed=[reset_seed]) after constructing the task env; the same task/reset pairs are executed for every policy",
            "base_path_default": str(args.base_path),
            "lora_root_default": str(args.lora_root),
            "libero_config_dir_default": str(args.libero_config_dir),
            "partial_result_path": str(args.stage_b_partial_output),
            "result_path": str(args.stage_b_output),
            "resume_rule": "resume only missing (policy, suite, task_id, reset_seed) episode keys",
        },
        "stage_a_result": {
            "path": str(args.stage_a_output),
            "sha256": _sha256_file(Path(args.stage_a_output)),
            "final_decision": str(stage_a_result.get("final_decision")),
            "completed_episode_count": int(stage_a_result.get("completed_episode_count", 0)),
            "exception_count": int((stage_a_result.get("summary") or {}).get("exception_count") or 0),
        },
        "checkpoint_manifest": {
            "path": str(args.checkpoint_manifest),
            "sha256": _sha256_file(Path(args.checkpoint_manifest)),
            "checkpoint_root": _relative_posix(str(checkpoint_manifest.get("checkpoint_root"))),
            "variant_count": int(checkpoint_manifest.get("variant_count", 0)),
        },
    }
    payload["canonical_payload_sha256"] = _sha256_payload({key: value for key, value in payload.items() if key != "canonical_payload_sha256"})
    validate_stage_b_manifest(payload)
    return payload


def validate_stage_a_manifest(manifest: Mapping[str, Any]) -> None:
    policies = [str(item["policy"]) for item in manifest.get("policies") or []]
    if policies != STAGE_A_POLICY_ORDER:
        raise ValueError(f"Stage A policies are not frozen order: {policies}")
    if int(manifest.get("stage_a_pair_count_per_policy", -1)) != STAGE_A_EPISODES_PER_POLICY:
        raise ValueError("Stage A must contain exactly 10 paired cases per policy")
    if int(manifest.get("planned_episode_count", -1)) != len(STAGE_A_POLICY_ORDER) * STAGE_A_EPISODES_PER_POLICY:
        raise ValueError("Stage A planned episode count must be 50")
    if list(manifest.get("stage_a_reset_seeds") or []) != STAGE_A_RESET_SEEDS:
        raise ValueError("Stage A reset identities changed")
    episodes = list(manifest.get("episodes") or [])
    episode_ids = [str(item["episode_id"]) for item in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Stage A episode ids contain duplicates")
    pair_sets: dict[str, set[str]] = defaultdict(set)
    for item in episodes:
        pair_sets[str(item["policy"])].add(str(item["pair_id"]))
    reference = pair_sets[STAGE_A_POLICY_ORDER[0]]
    for policy in STAGE_A_POLICY_ORDER:
        if pair_sets[policy] != reference:
            raise ValueError(f"Stage A policy {policy} does not use the identical paired manifest")
    if bool(manifest.get("closed_loop_experiment_happened")):
        raise ValueError("Stage A manifest must be frozen before rollout")
    if bool(manifest.get("confirmatory_test_tuning_happened")):
        raise ValueError("Stage A manifest reports confirmatory-test tuning")
    identity_overlap = dict(manifest.get("identity_overlap_verification") or {})
    if int(identity_overlap.get("duplicate_evaluation_keys", -1)) != 0:
        raise ValueError("Stage A manifest reports duplicate evaluation keys")
    if not bool(identity_overlap.get("identical_task_reset_pairs_across_policies")):
        raise ValueError("Stage A manifest does not certify identical task/reset pairs")


def validate_stage_b_manifest(manifest: Mapping[str, Any]) -> None:
    policies = [str(item["policy"]) for item in manifest.get("policies") or []]
    if policies != STAGE_A_POLICY_ORDER:
        raise ValueError(f"Stage B policies are not frozen order: {policies}")
    if int(manifest.get("stage_b_pair_count_per_policy", -1)) != STAGE_B_EPISODES_PER_POLICY:
        raise ValueError("Stage B must contain exactly 40 paired cases per policy")
    if int(manifest.get("planned_episode_count", -1)) != len(STAGE_A_POLICY_ORDER) * STAGE_B_EPISODES_PER_POLICY:
        raise ValueError("Stage B planned episode count must be 200")
    if list(manifest.get("stage_b_reset_seeds") or []) != STAGE_B_RESET_SEEDS:
        raise ValueError("Stage B reset identities changed")
    if len(manifest.get("tasks") or []) != STAGE_B_TASK_COUNT:
        raise ValueError("Stage B must use all 20 official tasks")
    episodes = list(manifest.get("episodes") or [])
    episode_ids = [str(item["episode_id"]) for item in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        raise ValueError("Stage B episode ids contain duplicates")
    pair_sets: dict[str, set[str]] = defaultdict(set)
    for item in episodes:
        pair_sets[str(item["policy"])].add(str(item["pair_id"]))
    reference = pair_sets[STAGE_A_POLICY_ORDER[0]]
    for policy in STAGE_A_POLICY_ORDER:
        if pair_sets[policy] != reference:
            raise ValueError(f"Stage B policy {policy} does not use the identical paired manifest")
    if bool(manifest.get("closed_loop_experiment_happened")):
        raise ValueError("Stage B manifest must be frozen before rollout")
    if bool(manifest.get("confirmatory_test_tuning_happened")):
        raise ValueError("Stage B manifest reports confirmatory-test tuning")
    identity_overlap = dict(manifest.get("identity_overlap_verification") or {})
    if int(identity_overlap.get("duplicate_evaluation_keys", -1)) != 0:
        raise ValueError("Stage B manifest reports duplicate evaluation keys")
    if int(identity_overlap.get("overlap_with_stage_a_reset_seeds", -1)) != 0:
        raise ValueError("Stage B reset seeds overlap Stage A")
    if not bool(identity_overlap.get("identical_task_reset_pairs_across_policies")):
        raise ValueError("Stage B manifest does not certify identical task/reset pairs")


def _manifest_task_map(manifest: Mapping[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    return {(str(task["suite"]), int(task["task_id"])): dict(task) for task in manifest.get("tasks") or []}


def _completed_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (str(row["policy"]), str(row["suite"]), int(row["task_id"]), int(row["reset_seed"]))


def _planned_lookup(manifest: Mapping[str, Any]) -> dict[str, int]:
    return {str(item["episode_id"]): int(item["planned_episode_index"]) for item in manifest.get("episodes") or []}


def _load_partial(path: Path, rerun_stage: bool) -> list[dict[str, Any]]:
    if rerun_stage or not path.exists():
        return []
    return list((_load_json(path).get("episodes") or []))


def _summarize_stage_a(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_policy_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_policy_task: dict[tuple[str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_policy_rows[str(row["policy"])].append(row)
        by_policy_task[(str(row["policy"]), str(row["suite"]), int(row["task_id"]))].append(row)

    by_policy = {}
    for policy in STAGE_A_POLICY_ORDER:
        policy_rows = by_policy_rows[policy]
        valid = [row for row in policy_rows if row.get("failure_status") != "exception"]
        successes = int(sum(1 for row in valid if bool(row.get("success"))))
        per_task = {}
        task_rates = []
        for task_key, task_rows in sorted(
            ((key, value) for key, value in by_policy_task.items() if key[0] == policy),
            key=lambda item: (item[0][1], item[0][2]),
        ):
            valid_task_rows = [row for row in task_rows if row.get("failure_status") != "exception"]
            task_successes = int(sum(1 for row in valid_task_rows if bool(row.get("success"))))
            task_total = len(valid_task_rows)
            rate = task_successes / task_total if task_total else 0.0
            task_rates.append(rate)
            per_task[f"{task_key[1]}/task_{task_key[2]}"] = {
                "successes": task_successes,
                "total": task_total,
                "success_rate": _round(rate, 6),
            }
        by_policy[policy] = {
            "successes": successes,
            "total": len(valid),
            "success_rate": _round(successes / len(valid), 6) if valid else 0.0,
            "task_balanced_success_rate": _round(float(np.mean(task_rates)), 6) if task_rates else 0.0,
            "exception_count": int(sum(1 for row in policy_rows if row.get("failure_status") == "exception")),
            "per_task": per_task,
            "action_validity_all_finite": all(bool((row.get("action_validity") or {}).get("finite", False)) for row in valid) if valid else False,
            "action_validity_all_shape_ok": all(bool((row.get("action_validity") or {}).get("shape_ok", False)) for row in valid) if valid else False,
            "policy_latency_mean_s": _round(
                float(np.mean([float(row["policy_latency_mean_s"]) for row in valid if row.get("policy_latency_mean_s") is not None])),
                6,
            )
            if any(row.get("policy_latency_mean_s") is not None for row in valid)
            else None,
            "peak_vram_max_allocated_mb": _round(
                max([float(((row.get("peak_vram") or {}).get("max_allocated_mb")) or 0.0) for row in valid] or [0.0]),
                3,
            ),
        }
    return {
        "by_policy": by_policy,
        "exception_count": int(sum(1 for row in rows if row.get("failure_status") == "exception")),
        "completed_episode_count": len(rows),
    }


def _paired_bootstrap_ci(deltas: list[float], *, seed: int = 20261214, samples: int = 5000) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        means[index] = float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
    return [_round(float(np.quantile(means, 0.025)), 6), _round(float(np.quantile(means, 0.975)), 6)]


def _paired_vs_full(rows: list[Mapping[str, Any]], *, include_bootstrap: bool = False) -> dict[str, Any]:
    by_key = {
        (str(row["policy"]), str(row["suite"]), int(row["task_id"]), int(row["reset_seed"])): bool(row.get("success"))
        for row in rows
        if row.get("failure_status") != "exception"
    }
    out = {}
    for policy in STAGE_A_POLICY_ORDER:
        if policy == "mtf_full":
            continue
        deltas = []
        wins = losses = ties = 0
        for row in rows:
            if str(row.get("policy")) != "mtf_full" or row.get("failure_status") == "exception":
                continue
            key = (policy, str(row["suite"]), int(row["task_id"]), int(row["reset_seed"]))
            if key not in by_key:
                continue
            delta = float(bool(row.get("success"))) - float(by_key[key])
            deltas.append(delta)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1
        record = {
            "paired_count": len(deltas),
            "paired_win_count": wins,
            "paired_loss_count": losses,
            "paired_tie_count": ties,
            "paired_success_delta": _round(float(np.mean(deltas)), 6) if deltas else 0.0,
        }
        if include_bootstrap:
            record["paired_bootstrap_ci"] = _paired_bootstrap_ci(deltas, seed=20261214 + len(out))
        out[policy] = record
    return out


def _stage_a_decision(summary: Mapping[str, Any]) -> str:
    if int(summary.get("exception_count") or 0) > 0:
        return "MTF_STAGE_A_MEASUREMENT_INVALID_REPAIR_REQUIRED"
    by_policy = summary["by_policy"]
    full = by_policy["mtf_full"]
    full_successes = int(full["successes"])
    for policy in STAGE_A_POLICY_ORDER:
        if policy == "mtf_full":
            continue
        baseline = by_policy[policy]
        if full_successes == 0 and int(baseline["successes"]) >= 4:
            return "MTF_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE"
        if float(baseline["task_balanced_success_rate"]) - float(full["task_balanced_success_rate"]) >= 0.30:
            return "MTF_STAGE_A_CATASTROPHIC_KILL_CLEARLY_WORSE_THAN_BASELINE_OR_ABLATION"
    if all(float(full["task_balanced_success_rate"]) > float(by_policy[policy]["task_balanced_success_rate"]) for policy in STAGE_A_POLICY_ORDER if policy != "mtf_full"):
        return "MTF_STAGE_A_POSITIVE_TO_STAGE_B_REQUIRED"
    return "MTF_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"


def _stage_b_decision(summary: Mapping[str, Any], paired: Mapping[str, Any]) -> str:
    if int(summary.get("exception_count") or 0) > 0:
        return "MTF_STAGE_B_MEASUREMENT_INVALID_REPAIR_REQUIRED"
    by_policy = summary["by_policy"]
    full_rate = float(by_policy["mtf_full"]["task_balanced_success_rate"])
    base_rate = float(by_policy["frozen_smolvla"]["task_balanced_success_rate"])
    prior_rate = float(by_policy["frameskip_proxy_lora"]["task_balanced_success_rate"])
    ablation_rate = float(by_policy["mtf_no_retention_ablation"]["task_balanced_success_rate"])
    simple_rate = float(by_policy["uniform_retained_ratio_lora"]["task_balanced_success_rate"])
    strongest_name = max((policy for policy in STAGE_A_POLICY_ORDER if policy != "mtf_full"), key=lambda policy: float(by_policy[policy]["task_balanced_success_rate"]))
    strongest_rate = float(by_policy[strongest_name]["task_balanced_success_rate"])
    if full_rate > max(base_rate, prior_rate, ablation_rate, simple_rate) and full_rate - strongest_rate >= 0.10:
        return "MTF_STAGE_B_PROTOTYPE_GO"
    if simple_rate >= full_rate:
        return "MTF_STAGE_B_KILL_SIMPLE_BASELINE_EXPLAINS_METHOD"
    if ablation_rate >= full_rate:
        return "MTF_STAGE_B_KILL_KEY_COMPONENT_NOT_USEFUL"
    if prior_rate >= full_rate:
        return "MTF_STAGE_B_KILL_CLOSEST_PRIOR_EXPLAINS_METHOD"
    if base_rate >= full_rate:
        return "MTF_STAGE_B_KILL_BASE_NOT_IMPROVED"
    strongest_pair = paired.get(strongest_name) or {}
    ci = strongest_pair.get("paired_bootstrap_ci") or [0.0, 0.0]
    if full_rate <= strongest_rate and float(ci[1]) <= 0.10:
        return "MTF_STAGE_B_USEFUL_IMPROVEMENT_EXCLUDED"
    return "MTF_STAGE_B_UNRESOLVED_EXPANSION_REQUIRED"


def write_stage_a_manifest_reports(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    _write_json(Path(args.stage_a_manifest), manifest)
    lines = [
        "# MTF-VLA Stage A Manifest",
        "",
        f"Date: `{manifest['date']}`",
        "",
        f"Final decision: `{manifest['final_decision']}`",
        "",
        f"- method: `{manifest['method']}`",
        f"- config: `{manifest['config_id']}`",
        f"- proposal hash: `{manifest['proposal_hash']}`",
        f"- policies: `{', '.join(manifest['policy_order'])}`",
        f"- reset seeds: `{manifest['stage_a_reset_seeds']}`",
        f"- paired cases per policy: `{manifest['stage_a_pair_count_per_policy']}`",
        f"- planned episodes: `{manifest['planned_episode_count']}`",
        f"- canonical payload sha256: `{manifest['canonical_payload_sha256']}`",
        "",
        "## Tasks",
        "",
    ]
    for task in manifest["tasks"]:
        lines.append(f"- `{task['suite']}/task_{task['task_id']}`: {task['instruction']}")
    lines += [
        "",
        "## Frozen Rules",
        "",
        "- five policies only: frozen SmolVLA, FrameSkip proxy, uniform retained-ratio LoRA, no-retention ablation, MTF full",
        "- `frameskip_proxy_lora` is a faithful local proxy, not an official FrameSkip reproduction",
        "- task/reset pairs are identical across policies and duplicate evaluation keys are zero",
        "- policy order does not choose or perturb the reset identities; every episode uses `env.reset(seed=[reset_seed])`",
        "- official LeRobot/LIBERO success condition is the primary closed-loop outcome",
        "- no confirmatory-test tuning or checkpoint selection from Stage A outcomes",
        "- exact matched task/reset pairs across all policies",
        "- small differences, ties, and one- or two-episode gaps advance to Stage B",
        "- permanent Stage A kill only under the preregistered catastrophic criteria",
        "",
        "## Execution",
        "",
        f"- partial result path: `{manifest['execution']['partial_result_path']}`",
        f"- final result path: `{manifest['execution']['result_path']}`",
        "- resume only missing `(policy, suite, task_id, reset_seed)` keys",
    ]
    _write_md(Path(args.stage_a_manifest_md), lines)


def write_stage_b_manifest_reports(args: argparse.Namespace, manifest: Mapping[str, Any]) -> None:
    _write_json(Path(args.stage_b_manifest), manifest)
    lines = [
        "# MTF-VLA Stage B Manifest",
        "",
        f"Date: `{manifest['date']}`",
        "",
        f"Final decision: `{manifest['final_decision']}`",
        "",
        f"- method: `{manifest['method']}`",
        f"- config: `{manifest['config_id']}`",
        f"- proposal hash: `{manifest['proposal_hash']}`",
        f"- policies: `{', '.join(manifest['policy_order'])}`",
        f"- reset seeds: `{manifest['stage_b_reset_seeds']}`",
        f"- paired cases per policy: `{manifest['stage_b_pair_count_per_policy']}`",
        f"- planned episodes: `{manifest['planned_episode_count']}`",
        f"- canonical payload sha256: `{manifest['canonical_payload_sha256']}`",
        f"- Stage A decision: `{manifest['stage_a_result']['final_decision']}`",
        "",
        "## Tasks",
        "",
    ]
    for task in manifest["tasks"]:
        lines.append(f"- `{task['suite']}/task_{task['task_id']}`: {task['instruction']}")
    lines += [
        "",
        "## Frozen Rules",
        "",
        "- all 20 official tasks are included",
        "- five policies only: frozen SmolVLA, FrameSkip proxy, uniform retained-ratio LoRA, no-retention ablation, MTF full",
        "- `frameskip_proxy_lora` is a faithful local proxy, not an official FrameSkip reproduction",
        "- task/reset pairs are identical across policies and duplicate evaluation keys are zero",
        "- reset identities are fresh relative to Stage A",
        "- official LeRobot/LIBERO success condition is the primary closed-loop outcome",
        "- no confirmatory-test tuning or checkpoint selection from Stage A or Stage B outcomes",
        "- one expansion to 80 paired episodes is allowed only if Stage B is genuinely unresolved",
        "",
        "## Execution",
        "",
        f"- partial result path: `{manifest['execution']['partial_result_path']}`",
        f"- final result path: `{manifest['execution']['result_path']}`",
        "- resume only missing `(policy, suite, task_id, reset_seed)` keys",
    ]
    _write_md(Path(args.stage_b_manifest_md), lines)


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_stage_a_manifest(args)
    write_stage_a_manifest_reports(args, manifest)
    return {
        "mode": "plan",
        "final_decision": manifest["final_decision"],
        "stage_a_manifest": str(args.stage_a_manifest),
        "stage_a_manifest_md": str(args.stage_a_manifest_md),
        "planned_episode_count": manifest["planned_episode_count"],
    }


def run_stage_b_plan(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_stage_b_manifest(args)
    write_stage_b_manifest_reports(args, manifest)
    return {
        "mode": "stage-b-plan",
        "final_decision": manifest["final_decision"],
        "stage_b_manifest": str(args.stage_b_manifest),
        "stage_b_manifest_md": str(args.stage_b_manifest_md),
        "planned_episode_count": manifest["planned_episode_count"],
    }


def _adapter_config_summary(adapter_path: Path | None) -> dict[str, Any] | None:
    if adapter_path is None:
        return None
    path = adapter_path / "adapter_config.json"
    if not path.exists():
        return {"adapter_config_exists": False}
    try:
        config = _load_json(path)
    except Exception as exc:  # pragma: no cover - host boundary
        return {"adapter_config_exists": True, "adapter_config_read_error": str(exc)}
    return {
        "adapter_config_exists": True,
        "peft_type": config.get("peft_type"),
        "r": config.get("r"),
        "lora_alpha": config.get("lora_alpha"),
        "target_modules": config.get("target_modules"),
        "base_model_name_or_path": config.get("base_model_name_or_path"),
    }


def _parameter_count_summary(policy: Any) -> dict[str, int]:
    total = trainable = 0
    for param in policy.parameters():
        count = int(param.numel())
        total += count
        if bool(getattr(param, "requires_grad", False)):
            trainable += count
    return {"total_parameter_count": total, "trainable_parameter_count": trainable}


def _preflight_policy_record(args: argparse.Namespace, manifest_policy: Mapping[str, Any], loaded: Mapping[str, Any]) -> dict[str, Any]:
    adapter_path = Path(manifest_policy["wsl_adapter_path"]) if manifest_policy.get("wsl_adapter_path") else None
    adapter_model_path = adapter_path / "adapter_model.safetensors" if adapter_path is not None else None
    adapter_config_path = adapter_path / "adapter_config.json" if adapter_path is not None else None
    expected_model_sha = manifest_policy.get("adapter_model_sha256")
    expected_config_sha = manifest_policy.get("adapter_config_sha256")
    model_sha = _sha256_file(adapter_model_path) if adapter_model_path is not None else None
    config_sha = _sha256_file(adapter_config_path) if adapter_config_path is not None else None
    audit = dict(loaded.get("audit") or {})
    peft = dict(audit.get("peft") or {})
    return {
        "policy": manifest_policy["policy"],
        "role": manifest_policy.get("role"),
        "proxy_or_reproduction_label": manifest_policy.get("proxy_or_reproduction_label"),
        "checkpoint_path": manifest_policy.get("checkpoint_path"),
        "wsl_adapter_path": str(adapter_path) if adapter_path is not None else None,
        "adapter_model_sha256": model_sha,
        "adapter_model_sha256_expected": expected_model_sha,
        "adapter_model_sha256_match": (str(model_sha).upper() == str(expected_model_sha).upper()) if expected_model_sha else None,
        "adapter_config_sha256": config_sha,
        "adapter_config_sha256_expected": expected_config_sha,
        "adapter_config_sha256_match": (str(config_sha).upper() == str(expected_config_sha).upper()) if expected_config_sha else None,
        "adapter_config": _adapter_config_summary(adapter_path),
        "base_model_path": str(args.base_path),
        "peft_audit": peft,
        "policy_class": audit.get("policy_class"),
        "parameter_counts": _parameter_count_summary(loaded["policy"]),
        "policy_output_shape": audit.get("action_chunk_shape"),
        "policy_output_device": audit.get("action_chunk_device"),
        "policy_output_finite": audit.get("action_chunk_finite"),
        "cuda_memory": audit.get("cuda_memory"),
        "model_parameter_device": (audit.get("parameter") or {}).get("device"),
        "input_tensor_devices": audit.get("input_tensor_devices"),
        "old_custom_libero_7d_route_used": audit.get("old_custom_libero_7d_route_used"),
        "cpu_fallback_detected": bool((audit.get("parameter") or {}).get("device") != "cuda:0"),
    }


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    _set_runtime_env(args)
    manifest = build_stage_a_manifest(args)
    write_stage_a_manifest_reports(args, manifest)
    records = []
    errors = []
    for spec, manifest_policy in zip(mtf_policy_specs(), manifest["policies"]):
        try:
            loaded = _load_policy_and_processors(args, spec)
            records.append(_preflight_policy_record(args, manifest_policy, loaded))
            del loaded
            torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - WSL/CUDA boundary
            errors.append(
                {
                    "policy": spec.name,
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc().splitlines()[-24:],
                }
            )
    trainable_records = [record for record in records if record.get("wsl_adapter_path")]
    unique_paths = {record["wsl_adapter_path"] for record in trainable_records}
    unique_model_hashes = {str(record.get("adapter_model_sha256")).upper() for record in trainable_records if record.get("adapter_model_sha256")}
    checksum_matches = all(record.get("adapter_model_sha256_match") is True and record.get("adapter_config_sha256_match") is True for record in trainable_records)
    cuda_ok = all(not bool(record.get("cpu_fallback_detected")) and record.get("policy_output_device") == "cuda:0" for record in records)
    output_ok = all(record.get("policy_output_shape") == [1, 50, 7] and bool(record.get("policy_output_finite")) for record in records)
    no_reuse = len(unique_paths) == 4 and len(unique_model_hashes) == 4
    final_decision = "MTF_STAGE_A_PREFLIGHT_PASS_READY_FOR_ROLLOUT"
    if errors or not (checksum_matches and cuda_ok and output_ok and no_reuse):
        final_decision = "MTF_STAGE_A_PREFLIGHT_BLOCKED_REPAIR_LOADING_OR_MAPPING"
    report = {
        "schema_version": 1,
        "method": METHOD,
        "config_id": CONFIG_ID,
        "proposal_hash": PROPOSAL_HASH,
        "date": f"{args.date} KST",
        "mode": "preflight",
        "stage_a_manifest": str(args.stage_a_manifest),
        "stage_a_manifest_sha256": _sha256_file(Path(args.stage_a_manifest)),
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "policy_count": len(records),
        "trainable_policy_count": len(trainable_records),
        "checkpoint_checksum_matches": bool(checksum_matches),
        "cuda_ok": bool(cuda_ok),
        "policy_output_shape_ok": bool(output_ok),
        "no_accidental_checkpoint_reuse": bool(no_reuse),
        "records": records,
        "errors": errors,
        "final_decision": final_decision,
        "elapsed_seconds": _round(time.monotonic() - started, 3),
    }
    _write_json(Path(args.stage_a_preflight_output), report)
    return report


def run_stage_a(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from lerobot.envs.factory import make_env

    started = time.monotonic()
    _set_runtime_env(args)
    manifest = build_stage_a_manifest(args)
    write_stage_a_manifest_reports(args, manifest)
    planned_lookup = _planned_lookup(manifest)
    task_map = _manifest_task_map(manifest)
    partial_path = Path(args.stage_a_partial_output)
    rows = _load_partial(partial_path, bool(args.rerun_stage))
    completed = {_completed_key(row) for row in rows}
    policy_audits = {}
    errors = []

    for spec in mtf_policy_specs():
        print(f"[mtf-stage-a] policy {spec.name}", flush=True)
        loaded = _load_policy_and_processors(args, spec)
        policy_audits[spec.name] = loaded["audit"]
        for task in manifest["tasks"]:
            env = None
            suite = str(task["suite"])
            task_id = int(task["task_id"])
            print(f"[mtf-stage-a] {spec.name} {suite} task_{task_id}", flush=True)
            try:
                env_cfg = _make_env_cfg(suite, [task_id])
                env = _extract_single_env(make_env(env_cfg, n_envs=1, use_async_envs=False), suite, task_id)
                task_record = task_map[(suite, task_id)]
                for seed in STAGE_A_RESET_SEEDS:
                    key = (spec.name, suite, task_id, int(seed))
                    if key in completed:
                        continue
                    episode_id = f"{spec.name}|{suite}|task_{task_id}|seed_{seed}"
                    row = _episode_base_record(spec.name, task_record, int(seed), planned_lookup[episode_id])
                    try:
                        trace = trace_one_episode(
                            env=env,
                            policy=loaded["policy"],
                            env_preprocessor=loaded["env_preprocessor"],
                            env_postprocessor=loaded["env_postprocessor"],
                            preprocessor=loaded["preprocessor"],
                            postprocessor=loaded["postprocessor"],
                            seed=int(seed),
                            video_path=None,
                        )
                        row.update(trace)
                    except Exception as exc:  # pragma: no cover - simulator boundary
                        row.update(
                            {
                                "success": False,
                                "sum_reward": None,
                                "max_reward": None,
                                "episode_length": None,
                                "termination_reason": "exception",
                                "failure_status": "exception",
                                "exception": {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                    "traceback": traceback.format_exc().splitlines()[-24:],
                                },
                                "action_validity": {"finite": False, "shape_ok": False, "max_abs": None},
                                "action_chunks_generated": None,
                                "env_steps": None,
                                "policy_latency_mean_s": None,
                                "policy_latency_max_s": None,
                                "env_step_latency_mean_s": None,
                                "env_step_latency_max_s": None,
                                "peak_vram": _cuda_memory(torch),
                                "rss_mb": _rss_mb(),
                                "video_path": None,
                            }
                        )
                        errors.append({"episode_id": episode_id, **row["exception"]})
                    rows.append(row)
                    completed.add(key)
                    _write_json(partial_path, {"episodes": rows, "planned_episode_count": manifest["planned_episode_count"]})
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
        del loaded
        torch.cuda.empty_cache()

    summary = _summarize_stage_a(rows)
    paired = _paired_vs_full(rows)
    final_decision = _stage_a_decision(summary)
    report = {
        "schema_version": 1,
        "method": METHOD,
        "config_id": CONFIG_ID,
        "proposal_hash": PROPOSAL_HASH,
        "branch": BRANCH,
        "date": f"{args.date} KST",
        "mode": "stage-a",
        "closed_loop_experiment_happened": True,
        "confirmatory_test_tuning_happened": False,
        "stage_a_manifest": str(args.stage_a_manifest),
        "stage_a_manifest_sha256": _sha256_file(Path(args.stage_a_manifest)),
        "planned_episode_count": int(manifest["planned_episode_count"]),
        "completed_episode_count": len(rows),
        "policy_load_audits": policy_audits,
        "episodes": rows,
        "summary": summary,
        "paired_vs_mtf_full": paired,
        "errors": errors,
        "final_decision": final_decision,
        "next_step": "Run Stage B on the frozen expansion manifest." if final_decision.endswith("STAGE_B_REQUIRED") else "Adjudicate repair or catastrophic kill under the preregistered governance.",
        "elapsed_seconds": _round(time.monotonic() - started, 3),
    }
    _write_json(Path(args.stage_a_output), report)
    write_stage_a_result_md(Path(args.stage_a_md), report)
    return report


def run_stage_b(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from lerobot.envs.factory import make_env

    started = time.monotonic()
    _set_runtime_env(args)
    manifest = build_stage_b_manifest(args)
    write_stage_b_manifest_reports(args, manifest)
    planned_lookup = _planned_lookup(manifest)
    task_map = _manifest_task_map(manifest)
    partial_path = Path(args.stage_b_partial_output)
    rows = _load_partial(partial_path, bool(args.rerun_stage))
    completed = {_completed_key(row) for row in rows}
    policy_audits = {}
    errors = []

    for spec in mtf_policy_specs():
        print(f"[mtf-stage-b] policy {spec.name}", flush=True)
        loaded = _load_policy_and_processors(args, spec)
        policy_audits[spec.name] = loaded["audit"]
        for task in manifest["tasks"]:
            env = None
            suite = str(task["suite"])
            task_id = int(task["task_id"])
            print(f"[mtf-stage-b] {spec.name} {suite} task_{task_id}", flush=True)
            try:
                env_cfg = _make_env_cfg(suite, [task_id])
                env = _extract_single_env(make_env(env_cfg, n_envs=1, use_async_envs=False), suite, task_id)
                task_record = task_map[(suite, task_id)]
                for seed in STAGE_B_RESET_SEEDS:
                    key = (spec.name, suite, task_id, int(seed))
                    if key in completed:
                        continue
                    episode_id = f"{spec.name}|{suite}|task_{task_id}|seed_{seed}"
                    row = _episode_base_record(spec.name, task_record, int(seed), planned_lookup[episode_id])
                    try:
                        trace = trace_one_episode(
                            env=env,
                            policy=loaded["policy"],
                            env_preprocessor=loaded["env_preprocessor"],
                            env_postprocessor=loaded["env_postprocessor"],
                            preprocessor=loaded["preprocessor"],
                            postprocessor=loaded["postprocessor"],
                            seed=int(seed),
                            video_path=None,
                        )
                        row.update(trace)
                    except Exception as exc:  # pragma: no cover - simulator boundary
                        row.update(
                            {
                                "success": False,
                                "sum_reward": None,
                                "max_reward": None,
                                "episode_length": None,
                                "termination_reason": "exception",
                                "failure_status": "exception",
                                "exception": {
                                    "type": type(exc).__name__,
                                    "message": str(exc),
                                    "traceback": traceback.format_exc().splitlines()[-24:],
                                },
                                "action_validity": {"finite": False, "shape_ok": False, "max_abs": None},
                                "action_chunks_generated": None,
                                "env_steps": None,
                                "policy_latency_mean_s": None,
                                "policy_latency_max_s": None,
                                "env_step_latency_mean_s": None,
                                "env_step_latency_max_s": None,
                                "peak_vram": _cuda_memory(torch),
                                "rss_mb": _rss_mb(),
                                "video_path": None,
                            }
                        )
                        errors.append({"episode_id": episode_id, **row["exception"]})
                    rows.append(row)
                    completed.add(key)
                    _write_json(partial_path, {"episodes": rows, "planned_episode_count": manifest["planned_episode_count"]})
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
        del loaded
        torch.cuda.empty_cache()

    summary = _summarize_stage_a(rows)
    paired = _paired_vs_full(rows, include_bootstrap=True)
    final_decision = _stage_b_decision(summary, paired)
    if final_decision == "MTF_STAGE_B_PROTOTYPE_GO":
        next_step = "Verify Quantized OpenVLA-OFT INT4 transfer and add one second condition."
    elif final_decision == "MTF_STAGE_B_UNRESOLVED_EXPANSION_REQUIRED":
        next_step = "Freeze and run the one allowed Stage B expansion to 80 paired episodes per key policy."
    else:
        next_step = "Archive or pivot under the preregistered governance; do not retune MTF from Stage B outcomes."
    report = {
        "schema_version": 1,
        "method": METHOD,
        "config_id": CONFIG_ID,
        "proposal_hash": PROPOSAL_HASH,
        "branch": BRANCH,
        "date": f"{args.date} KST",
        "mode": "stage-b",
        "closed_loop_experiment_happened": True,
        "confirmatory_test_tuning_happened": False,
        "stage_b_manifest": str(args.stage_b_manifest),
        "stage_b_manifest_sha256": _sha256_file(Path(args.stage_b_manifest)),
        "stage_a_result": str(args.stage_a_output),
        "stage_a_result_sha256": _sha256_file(Path(args.stage_a_output)),
        "planned_episode_count": int(manifest["planned_episode_count"]),
        "completed_episode_count": len(rows),
        "policy_load_audits": policy_audits,
        "episodes": rows,
        "summary": summary,
        "paired_vs_mtf_full": paired,
        "errors": errors,
        "final_decision": final_decision,
        "next_step": next_step,
        "elapsed_seconds": _round(time.monotonic() - started, 3),
    }
    _write_json(Path(args.stage_b_output), report)
    write_stage_b_result_md(Path(args.stage_b_md), report)
    return report


def write_stage_a_result_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# MTF-VLA Stage A Result",
        "",
        f"Date: `{report['date']}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- planned episodes: `{report['planned_episode_count']}`",
        f"- completed episodes: `{report['completed_episode_count']}`",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        f"- elapsed seconds: `{report['elapsed_seconds']}`",
        "",
        "## Policy Summary",
        "",
        "| policy | successes | total | task-balanced success | exceptions |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for policy in STAGE_A_POLICY_ORDER:
        row = report["summary"]["by_policy"][policy]
        lines.append(
            f"| `{policy}` | {row['successes']} | {row['total']} | {row['task_balanced_success_rate']} | {row['exception_count']} |"
        )
    lines += [
        "",
        "## Paired Versus MTF Full",
        "",
        "| baseline | pairs | wins | losses | ties | delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy, row in report["paired_vs_mtf_full"].items():
        lines.append(
            f"| `{policy}` | {row['paired_count']} | {row['paired_win_count']} | {row['paired_loss_count']} | {row['paired_tie_count']} | {row['paired_success_delta']} |"
        )
    lines += ["", f"Next step: {report['next_step']}"]
    _write_md(path, lines)


def write_stage_b_result_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# MTF-VLA Stage B Result",
        "",
        f"Date: `{report['date']}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- planned episodes: `{report['planned_episode_count']}`",
        f"- completed episodes: `{report['completed_episode_count']}`",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        f"- elapsed seconds: `{report['elapsed_seconds']}`",
        "",
        "## Policy Summary",
        "",
        "| policy | successes | total | task-balanced success | exceptions | latency mean s | peak VRAM MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for policy in STAGE_A_POLICY_ORDER:
        row = report["summary"]["by_policy"][policy]
        lines.append(
            f"| `{policy}` | {row['successes']} | {row['total']} | {row['task_balanced_success_rate']} | {row['exception_count']} | {row['policy_latency_mean_s']} | {row['peak_vram_max_allocated_mb']} |"
        )
    lines += [
        "",
        "## Paired Versus MTF Full",
        "",
        "| baseline | pairs | wins | losses | ties | delta | CI 95% |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for policy, row in report["paired_vs_mtf_full"].items():
        ci = row.get("paired_bootstrap_ci") or [None, None]
        lines.append(
            f"| `{policy}` | {row['paired_count']} | {row['paired_win_count']} | {row['paired_loss_count']} | {row['paired_tie_count']} | {row['paired_success_delta']} | [{ci[0]}, {ci[1]}] |"
        )
    lines += ["", f"Next step: {report['next_step']}"]
    _write_md(path, lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan", "preflight", "stage-a", "stage-b-plan", "stage-b"], default="plan")
    parser.add_argument("--date", default=DATE_KST)
    parser.add_argument("--base-path", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/mnt/c/Users/jiheo/tca_map/runs/mtf_vla_checkpoints/mtf_r20_ret100")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--wsl-repo-root", default="/mnt/c/Users/jiheo/tca_map")
    parser.add_argument("--official-task-manifest", default="reports/official_closed_loop_task_manifest.json")
    parser.add_argument("--checkpoint-manifest", default="reports/mtf_vla/adapter_checkpoint_manifest.json")
    parser.add_argument("--stage-a-manifest", default="reports/mtf_vla/stage_a_manifest.json")
    parser.add_argument("--stage-a-manifest-md", default="reports/mtf_vla/stage_a_manifest.md")
    parser.add_argument("--stage-a-output", default="reports/mtf_vla/stage_a_result.json")
    parser.add_argument("--stage-a-md", default="reports/mtf_vla/stage_a_result.md")
    parser.add_argument("--stage-a-partial-output", default="reports/mtf_vla/stage_a_partial_result.json")
    parser.add_argument("--stage-a-preflight-output", default="reports/mtf_vla/stage_a_preflight.json")
    parser.add_argument("--stage-b-manifest", default="reports/mtf_vla/stage_b_manifest.json")
    parser.add_argument("--stage-b-manifest-md", default="reports/mtf_vla/stage_b_manifest.md")
    parser.add_argument("--stage-b-output", default="reports/mtf_vla/stage_b_result.json")
    parser.add_argument("--stage-b-md", default="reports/mtf_vla/stage_b_result.md")
    parser.add_argument("--stage-b-partial-output", default="reports/mtf_vla/stage_b_partial_result.json")
    parser.add_argument("--rerun-stage", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.mode == "plan":
        report = run_plan(args)
    elif args.mode == "preflight":
        report = run_preflight(args)
    elif args.mode == "stage-a":
        report = run_stage_a(args)
    elif args.mode == "stage-b-plan":
        report = run_stage_b_plan(args)
    else:
        report = run_stage_b(args)
    print(json.dumps({"mode": args.mode, "final_decision": report.get("final_decision"), "planned": report.get("planned_episode_count"), "completed": report.get("completed_episode_count")}, sort_keys=True))
    return 0 if report.get("final_decision") in FINAL_DECISIONS else 2


if __name__ == "__main__":
    raise SystemExit(main())
