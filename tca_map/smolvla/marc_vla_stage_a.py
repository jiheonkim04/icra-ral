"""MARC-VLA Stage A matched-manifest freezer."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any


DATE_KST = "2026-07-15"
BRANCH = "codex/autonomous-until-paper-governance-v2"
METHOD = "MARC-VLA"
CONFIG_ID = "marc_a020_gate_mlp"
PROPOSAL_HASH = "D1F910465D4E415C996B3F8C7CE2B2CF47339EA94D697B06A9DCED49AC1E585A"
STAGE_A_RESET_SEEDS = [20261209, 20261210]
STAGE_B_RESET_SEEDS = [20261211, 20261212]
STAGE_A_TASK_COUNT = 5
STAGE_A_EPISODES_PER_POLICY = STAGE_A_TASK_COUNT * len(STAGE_A_RESET_SEEDS)
MARC_FULL_POLICY = "marc_full"
STAGE_A_POLICY_ORDER = [
    "frozen_smolvla",
    "openvla_oft_l1_proxy",
    "marc_full",
    "marc_no_disagreement_gate_ablation",
    "static_l1_mixture_baseline",
]
POLICY_ROLES = {
    "frozen_smolvla": "unmodified_backbone",
    "openvla_oft_l1_proxy": "closest_external_prior_proxy_faithful_transparent_local_proxy_not_official_openvla_oft_reproduction",
    "marc_full": "ours",
    "marc_no_disagreement_gate_ablation": "key_ablation",
    "static_l1_mixture_baseline": "strongest_simple_reviewer_killer",
}
FINAL_DECISIONS = {
    "MARC_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT",
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
    if checkpoint_manifest.get("final_decision") != "MARC_POLICY_IDENTITIES_VERIFIED_STAGE_A_MANIFEST_READY":
        raise ValueError("MARC policy checkpoint manifest is not Stage-A ready")
    if not bool(checkpoint_manifest.get("stage_a_allowed")):
        raise ValueError("MARC policy checkpoint manifest does not allow Stage A")
    if list(checkpoint_manifest.get("policy_identities") or []) != STAGE_A_POLICY_ORDER:
        raise ValueError("MARC policy checkpoint manifest does not match the frozen five-policy order")

    variants = {str(item["variant"]): dict(item) for item in checkpoint_manifest.get("variant_results") or []}
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
                    "proxy_or_reproduction_label": None,
                }
            )
            continue

        source = variants.get(policy)
        if not source:
            raise ValueError(f"checkpoint manifest missing policy identity {policy}")
        validation = dict(source.get("validation") or {})
        checkpoint_path = _relative(str(source["checkpoint_path"]))
        records.append(
            {
                "policy": policy,
                "role": POLICY_ROLES[policy],
                "checkpoint_path": checkpoint_path,
                "wsl_checkpoint_path": _wsl_repo_path(str(args.wsl_repo_root), checkpoint_path),
                "disk_reload": bool(source.get("disk_reload")),
                "checkpoint_reload_max_abs_diff": source.get("checkpoint_reload_max_abs_diff"),
                "initial_delta_p95": source.get("initial_delta_p95"),
                "delta_l2_p95": validation.get("delta_l2_p95"),
                "clean_delta_l2_p95": validation.get("clean_delta_l2_p95"),
                "translation_delta_l2_p95": validation.get("translation_delta_l2_p95"),
                "rotation_delta_l2_p95": validation.get("rotation_delta_l2_p95"),
                "gripper_delta_abs_p95": validation.get("gripper_delta_abs_p95"),
                "action_validity": validation.get("action_validity"),
                "gate_metrics": source.get("gate_metrics"),
                "sha256_manifest": source.get("sha256_manifest"),
                "proxy_or_reproduction_label": (
                    "faithful_transparent_local_proxy_not_official_openvla_oft_reproduction"
                    if policy == "openvla_oft_l1_proxy"
                    else None
                ),
            }
        )
    return records


def validate_manifest(payload: Mapping[str, Any]) -> None:
    episodes = list(payload.get("episodes") or [])
    keys = [(row["policy"], row["suite"], int(row["task_id"]), int(row["reset_seed"])) for row in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate MARC Stage A evaluation keys")

    policy_order = list(payload.get("policy_order") or [])
    pair_sets = {}
    for policy in policy_order:
        pair_sets[policy] = {
            (row["suite"], int(row["task_id"]), int(row["reset_seed"]))
            for row in episodes
            if row["policy"] == policy
        }
    if len({tuple(sorted(values)) for values in pair_sets.values()}) != 1:
        raise ValueError("MARC Stage A task/reset pairs differ across policies")

    allocation = dict(payload.get("task_balanced_allocation") or {})
    expected = len(policy_order) * int(allocation.get("paired_cases_per_policy", -1))
    if int(payload.get("planned_episode_count", -1)) != expected:
        raise ValueError("MARC Stage A planned episode count mismatch")


def validate_stage_a_manifest(manifest: Mapping[str, Any]) -> None:
    policies = [str(item["policy"]) for item in manifest.get("policies") or []]
    if policies != STAGE_A_POLICY_ORDER:
        raise ValueError(f"MARC Stage A policies are not frozen order: {policies}")
    if int(manifest.get("stage_a_pair_count_per_policy", -1)) != STAGE_A_EPISODES_PER_POLICY:
        raise ValueError("MARC Stage A must contain exactly 10 paired cases per policy")
    if int(manifest.get("planned_episode_count", -1)) != len(STAGE_A_POLICY_ORDER) * STAGE_A_EPISODES_PER_POLICY:
        raise ValueError("MARC Stage A planned episode count must be 50")
    if list(manifest.get("stage_a_reset_seeds") or []) != STAGE_A_RESET_SEEDS:
        raise ValueError("MARC Stage A reset identities changed")
    if bool(manifest.get("closed_loop_experiment_happened")):
        raise ValueError("MARC Stage A manifest must be frozen before rollout")
    if bool(manifest.get("confirmatory_test_tuning_happened")):
        raise ValueError("MARC Stage A manifest reports confirmatory-test tuning")
    if len(manifest.get("tasks") or []) != STAGE_A_TASK_COUNT:
        raise ValueError("MARC Stage A must use five task-balanced tasks")
    validate_manifest(manifest)
    identity_overlap = dict(manifest.get("identity_overlap_verification") or {})
    if int(identity_overlap.get("duplicate_evaluation_keys", -1)) != 0:
        raise ValueError("MARC Stage A manifest reports duplicate evaluation keys")
    if not bool(identity_overlap.get("identical_task_reset_pairs_across_policies")):
        raise ValueError("MARC Stage A manifest does not certify identical task/reset pairs")

    proxy_records = [row for row in manifest.get("policies") or [] if row.get("policy") == "openvla_oft_l1_proxy"]
    if not proxy_records:
        raise ValueError("MARC Stage A manifest is missing the OpenVLA-OFT-style proxy")
    proxy_label = str(proxy_records[0].get("proxy_or_reproduction_label") or "")
    if "not_official_openvla_oft_reproduction" not in proxy_label:
        raise ValueError("MARC Stage A proxy is not explicitly labeled as non-official OpenVLA-OFT reproduction")


def _task_index_map_from_artifact(path: Path) -> dict[str, int]:
    artifact = _read_json(path)
    mapping: dict[str, int] = {}
    for record in artifact.get("records") or []:
        task = str(record.get("task") or "").strip()
        if not task:
            continue
        task_index = int(record.get("task_index", -1))
        previous = mapping.get(task)
        if previous is not None and previous != task_index:
            raise ValueError(f"MARC stable artifact has inconsistent task index for {task!r}: {previous} vs {task_index}")
        mapping[task] = task_index
    return mapping


def _task_index_for_task(task: Mapping[str, Any], task_index_map: Mapping[str, int]) -> int:
    instruction = str(task["instruction"]).strip()
    if instruction not in task_index_map:
        raise ValueError(f"MARC task instruction missing from stable artifact task-index map: {instruction}")
    return int(task_index_map[instruction])


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
        "final_decision": "MARC_STAGE_A_PLAN_FROZEN_READY_FOR_OFFICIAL_ROLLOUT",
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
            "overlap_with_reserved_test_identities": 0,
            "overlap_with_previous_known_allocated_rollout_identities": 0,
            "duplicate_evaluation_keys": 0,
            "identical_task_reset_pairs_across_policies": True,
            "note": "MARC policy training and validation used offline dataset frame splits; Stage A rollout reset seeds were selected after policy identity freeze.",
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
            "rule": "fresh unused MARC Stage A block after DAGR Stage B reset seeds",
            "reset_seeds": list(STAGE_A_RESET_SEEDS),
            "previous_known_allocations_avoided": [
                "official baseline scale-up reset seeds 20260711..20260715",
                "CBFD/SCVC/PSE reset identities 20260716..20260760",
                "CAVM/FANG/RAC/EvoState/MTF reset identity blocks through 20261204",
                "DAGR Stage A reset seeds 20261205..20261206",
                "DAGR Stage B reset seeds 20261207..20261208",
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
            "permanent_kill_mechanism_invalidity": "MARC full mechanism is invalid or nonacting under real rollout",
            "permanent_kill_clear_prior_or_ablation_dominance": "marc_full is clearly dominated by the L1 proxy, no-gate ablation, or static mixture under Stage A",
            "permanent_kill_catastrophic_degradation": "marc_full is catastrophically worse than a paired baseline or ablation",
            "small_difference_rule": "small differences, ties, and one- or two-episode gaps advance to Stage B",
            "next_stage_count": "Stage B requires at least 40 paired episodes per key policy",
        },
        "execution": {
            "official_path": "LeRobot SmolVLA/LIBERO policy, processors, action queue, relative 7D control, official LIBERO success condition, plus MARC action adapter",
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
            "variant_count": len(checkpoint_manifest.get("variant_results") or []),
        },
    }
    payload["canonical_payload_sha256"] = _sha256_payload(
        {key: value for key, value in payload.items() if key != "canonical_payload_sha256"}
    )
    validate_stage_a_manifest(payload)
    return payload


def write_stage_a_manifest_md(path: Path, manifest: Mapping[str, Any]) -> None:
    lines = [
        "# MARC-VLA Stage A Manifest",
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
            "- five policies only: frozen SmolVLA, OpenVLA-OFT-style L1 proxy, MARC full, no-disagreement-gate ablation, and static L1 mixture",
            "- `openvla_oft_l1_proxy` is a faithful transparent local proxy, not an official OpenVLA-OFT reproduction",
            "- task/reset pairs are identical across policies and duplicate evaluation keys are zero",
            "- policy order does not choose or perturb reset identities",
            "- official LIBERO success condition is the primary closed-loop outcome",
            "- no confirmatory-test tuning or checkpoint selection from Stage A outcomes",
            "- small differences, ties, and one- or two-episode gaps advance to Stage B",
            "- permanent Stage A kill only under the preregistered catastrophic, invalid-mechanism, or clear-dominance criteria",
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
    parser.add_argument("--mode", choices=["plan", "freeze-stage-a-manifest"], default="plan")
    parser.add_argument("--date", default=DATE_KST)
    parser.add_argument("--base-path", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--checkpoint-root", default="/mnt/c/Users/jiheo/tca_map/runs/marc_vla_checkpoints/marc_a020_gate_mlp")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--wsl-repo-root", default="/mnt/c/Users/jiheo/tca_map")
    parser.add_argument("--official-task-manifest", default="reports/official_closed_loop_task_manifest.json")
    parser.add_argument("--checkpoint-manifest", default="reports/marc_vla/policy_checkpoint_manifest.json")
    parser.add_argument("--stable-artifact", default="reports/official_smolvla_stable_prediction_artifact.json")
    parser.add_argument("--stage-a-manifest", default="reports/marc_vla/stage_a_manifest.json")
    parser.add_argument("--stage-a-manifest-md", default="reports/marc_vla/stage_a_manifest.md")
    parser.add_argument("--stage-a-output", default="reports/marc_vla/stage_a_result.json")
    parser.add_argument("--stage-a-md", default="reports/marc_vla/stage_a_result.md")
    parser.add_argument("--stage-a-partial-output", default="reports/marc_vla/stage_a_partial_result.json")
    parser.add_argument("--stage-a-preflight-output", default="reports/marc_vla/stage_a_preflight.json")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = run_plan(args)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "final_decision": report.get("final_decision"),
                "planned": report.get("planned_episode_count"),
            },
            sort_keys=True,
        )
    )
    return 0 if report.get("final_decision") in FINAL_DECISIONS else 2


if __name__ == "__main__":
    raise SystemExit(main())
