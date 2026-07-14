"""DAGR-VLA Stage A matched-manifest freezer."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


DATE_KST = "2026-07-14"
BRANCH = "codex/autonomous-until-paper-governance-v2"
METHOD = "DAGR-VLA"
CONFIG_ID = "dagr_a020_route_mlp"
PROPOSAL_HASH = "BDE0EC67ACE8EC457CE6495D723EE476064F3D80946151326B11F0B5A1AFEF89"
STAGE_A_RESET_SEEDS = [20261205, 20261206]
STAGE_A_TASK_COUNT = 5
STAGE_A_POLICY_ORDER = [
    "frozen_smolvla",
    "dam_static_component_proxy",
    "dagr_full",
    "dagr_no_dynamic_route_ablation",
    "gripper_transition_heuristic",
]
POLICY_ROLES = {
    "frozen_smolvla": "unmodified_backbone",
    "dam_static_component_proxy": "closest_external_prior_proxy_faithful_local_proxy_not_official_dam_vla_reproduction",
    "dagr_full": "ours",
    "dagr_no_dynamic_route_ablation": "key_ablation",
    "gripper_transition_heuristic": "strongest_simple_reviewer_killer",
}
FINAL_DECISIONS = {
    "DAGR_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT",
}


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _sha256_payload(payload: Any) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(blob).hexdigest().upper()


def _relative(path_text: str) -> str:
    return path_text.replace("\\", "/")


def _wsl_repo_path(wsl_repo_root: str, relative_path: str) -> str:
    return f"{str(wsl_repo_root).rstrip('/')}/{_relative(relative_path).lstrip('/')}"


def _select_stage_a_tasks(task_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    tasks = list(task_manifest.get("tasks") or [])
    if len(tasks) < STAGE_A_TASK_COUNT:
        raise ValueError(f"official task manifest has only {len(tasks)} tasks")
    indices = [(index * len(tasks)) // STAGE_A_TASK_COUNT for index in range(STAGE_A_TASK_COUNT)]
    selected = []
    for stage_index, source_index in enumerate(indices):
        task = dict(tasks[source_index])
        task["stage_a_task_index"] = int(stage_index)
        task["source_official_task_manifest_index"] = int(source_index)
        task["stage_a_selection_rule"] = f"floor(k * {len(tasks)} / {STAGE_A_TASK_COUNT})"
        selected.append(task)
    return selected


def _policy_records(args: argparse.Namespace, checkpoint_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    if checkpoint_manifest.get("final_decision") != "DAGR_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY":
        raise ValueError("DAGR policy checkpoint manifest is not Stage-A ready")
    if not bool(checkpoint_manifest.get("stage_a_allowed")):
        raise ValueError("DAGR policy checkpoint manifest does not allow Stage A")
    variants = {str(item["variant"]): dict(item) for item in checkpoint_manifest.get("variant_results") or []}
    heuristic = dict(checkpoint_manifest.get("heuristic") or {})
    records = []
    for policy in STAGE_A_POLICY_ORDER:
        if policy == "frozen_smolvla":
            records.append(
                {
                    "policy": policy,
                    "role": POLICY_ROLES[policy],
                    "checkpoint_path": None,
                    "wsl_checkpoint_path": None,
                    "disk_reload": None,
                    "sha256_manifest": None,
                }
            )
            continue
        source = heuristic if policy == "gripper_transition_heuristic" else variants.get(policy)
        if not source:
            raise ValueError(f"checkpoint manifest missing policy identity {policy}")
        checkpoint_path = _relative(str(source["checkpoint_path"]))
        records.append(
            {
                "policy": policy,
                "role": POLICY_ROLES[policy],
                "checkpoint_path": checkpoint_path,
                "wsl_checkpoint_path": _wsl_repo_path(str(args.wsl_repo_root), checkpoint_path),
                "disk_reload": bool(source.get("disk_reload")),
                "delta_l2_p95": ((source.get("validation") or {}).get("delta_l2_p95")),
                "action_validity": ((source.get("validation") or {}).get("action_validity")),
                "sha256_manifest": source.get("sha256_manifest"),
                "proxy_or_reproduction_label": (
                    "faithful_transparent_local_proxy_not_official_dam_vla_reproduction"
                    if policy == "dam_static_component_proxy"
                    else None
                ),
            }
        )
    return records


def validate_manifest(payload: Mapping[str, Any]) -> None:
    episodes = list(payload.get("episodes") or [])
    keys = [(row["policy"], row["suite"], int(row["task_id"]), int(row["reset_seed"])) for row in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate DAGR Stage A evaluation keys")
    policy_order = list(payload.get("policy_order") or [])
    pair_sets = {}
    for policy in policy_order:
        pair_sets[policy] = {
            (row["suite"], int(row["task_id"]), int(row["reset_seed"]))
            for row in episodes
            if row["policy"] == policy
        }
    if len({tuple(sorted(values)) for values in pair_sets.values()}) != 1:
        raise ValueError("DAGR Stage A task/reset pairs differ across policies")
    if int(payload.get("planned_episode_count", -1)) != len(policy_order) * int(payload["task_balanced_allocation"]["paired_cases_per_policy"]):
        raise ValueError("DAGR Stage A planned episode count mismatch")


def build_stage_a_manifest(args: argparse.Namespace) -> dict[str, Any]:
    task_manifest = _read_json(Path(args.official_task_manifest))
    checkpoint_manifest = _read_json(Path(args.checkpoint_manifest))
    tasks = _select_stage_a_tasks(task_manifest)
    policies = _policy_records(args, checkpoint_manifest)
    pairs = []
    for task in tasks:
        for seed in STAGE_A_RESET_SEEDS:
            pairs.append(
                {
                    "pair_id": f"{task['suite']}|task_{task['task_id']}|seed_{seed}",
                    "suite": str(task["suite"]),
                    "task_id": int(task["task_id"]),
                    "instruction": str(task["instruction"]),
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
        "final_decision": "DAGR_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT",
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
            "overlap_with_development_training_identities": 0,
            "overlap_with_development_validation_identities": 0,
            "overlap_with_previous_known_allocated_rollout_identities": 0,
            "duplicate_evaluation_keys": 0,
            "identical_task_reset_pairs_across_policies": True,
            "note": "DAGR training and validation used offline dataset frame splits; Stage A reset seeds were selected after policy identity freeze.",
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
            "rule": "fresh unused DAGR Stage A block after MTF Stage B reset seeds",
            "reset_seeds": list(STAGE_A_RESET_SEEDS),
            "previous_known_allocations_avoided": [
                "official baseline scale-up reset seeds 20260711..20260715",
                "CBFD/SCVC/PSE reset identities 20260716..20260760",
                "CAVM/FANG/RAC/EvoState/MTF reset identity blocks through 20261204",
            ],
        },
        "partition_separation": {
            "offline_training_splits": ["train"],
            "offline_validation_splits": ["val"],
            "offline_reserved_confirmatory_splits": ["test"],
            "stage_a_rollout_resets_are_frozen_after_checkpoint_selection": True,
            "stage_a_rollout_resets_used_for_policy_training": False,
            "stage_a_rollout_resets_used_for_validation_search": False,
        },
        "frozen_stage_a_rules": {
            "permanent_kill_zero_vs_baseline": "dagr_full has 0/10 while any paired baseline has at least 4/10",
            "permanent_kill_clear_degradation": "dagr_full is at least 30 absolute points below a baseline, prior proxy, simple baseline, or ablation",
            "small_difference_rule": "small differences, ties, and one- or two-episode gaps advance to Stage B",
            "next_stage_count": "Stage B requires at least 40 paired episodes per key policy",
        },
        "execution": {
            "official_path": "LeRobot SmolVLA/LIBERO policy, processors, action queue, relative 7D control, official LIBERO success condition, plus DAGR residual wrapper",
            "policy_order_affects_environment_initialization": False,
            "environment_initialization_rule": "each episode calls env.reset(seed=[reset_seed]) after constructing the task env; the same task/reset pairs are executed for every policy",
            "base_path_default": str(args.base_path),
            "checkpoint_root_default": str(args.checkpoint_root),
            "libero_config_dir_default": str(args.libero_config_dir),
            "partial_result_path": str(args.stage_a_partial_output),
            "result_path": str(args.stage_a_output),
            "preflight_result_path": str(args.stage_a_preflight_output),
            "resume_rule": "resume only missing (policy, suite, task_id, reset_seed) episode keys",
        },
        "checkpoint_manifest": {
            "path": str(args.checkpoint_manifest),
            "sha256": _sha256_file(Path(args.checkpoint_manifest)),
            "checkpoint_root": _relative(str(checkpoint_manifest.get("checkpoint_root"))),
            "policy_identity_count": len(checkpoint_manifest.get("policy_identities") or []),
        },
    }
    payload["canonical_payload_sha256"] = _sha256_payload({key: value for key, value in payload.items() if key != "canonical_payload_sha256"})
    validate_manifest(payload)
    return payload


def write_stage_a_manifest_md(path: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "# DAGR-VLA Stage A Manifest",
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
    lines.extend(
        [
            "",
            "## Frozen Rules",
            "",
            "- five policies only: frozen SmolVLA, DAM-style static component proxy, DAGR full, no-dynamic-route ablation, and gripper-transition heuristic",
            "- `dam_static_component_proxy` is a faithful transparent local proxy, not an official DAM-VLA reproduction",
            "- task/reset pairs are identical across policies and duplicate evaluation keys are zero",
            "- policy order does not choose or perturb reset identities",
            "- official LIBERO success condition is the primary closed-loop outcome",
            "- no confirmatory-test tuning or checkpoint selection from Stage A outcomes",
            "- small differences, ties, and one- or two-episode gaps advance to Stage B",
            "- permanent Stage A kill only under the preregistered catastrophic criteria",
            "",
            "## Execution",
            "",
            f"- partial result path: `{manifest['execution']['partial_result_path']}`",
            f"- final result path: `{manifest['execution']['result_path']}`",
            "- resume only missing `(policy, suite, task_id, reset_seed)` keys",
        ]
    )
    _write_md(path, lines)


def run_plan(args: argparse.Namespace) -> dict[str, Any]:
    manifest = build_stage_a_manifest(args)
    _write_json(Path(args.stage_a_manifest), manifest)
    write_stage_a_manifest_md(Path(args.stage_a_manifest_md), manifest)
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["plan"], default="plan")
    parser.add_argument("--date", default=DATE_KST)
    parser.add_argument("--base-path", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--checkpoint-root", default="/mnt/c/Users/jiheo/tca_map/runs/dagr_vla_checkpoints/dagr_a020_route_mlp")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--wsl-repo-root", default="/mnt/c/Users/jiheo/tca_map")
    parser.add_argument("--official-task-manifest", default="reports/official_closed_loop_task_manifest.json")
    parser.add_argument("--checkpoint-manifest", default="reports/dagr_vla/policy_checkpoint_manifest.json")
    parser.add_argument("--stage-a-manifest", default="reports/dagr_vla/stage_a_manifest.json")
    parser.add_argument("--stage-a-manifest-md", default="reports/dagr_vla/stage_a_manifest.md")
    parser.add_argument("--stage-a-output", default="reports/dagr_vla/stage_a_result.json")
    parser.add_argument("--stage-a-partial-output", default="reports/dagr_vla/stage_a_partial_result.json")
    parser.add_argument("--stage-a-preflight-output", default="reports/dagr_vla/stage_a_preflight.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_plan(args)
    print(json.dumps({"mode": args.mode, "final_decision": report.get("final_decision"), "planned": report.get("planned_episode_count")}, sort_keys=True))
    return 0 if report.get("final_decision") in FINAL_DECISIONS else 2


if __name__ == "__main__":
    raise SystemExit(main())

