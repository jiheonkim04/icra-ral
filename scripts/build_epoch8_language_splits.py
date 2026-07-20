#!/usr/bin/env python3
"""Freeze candidate-independent Epoch 8 language data/evaluation roles.

This script reads only the already-frozen Epoch 7 partition, released language
metadata, and static benchmark files. It never loads a model/simulator or reads
an Ours outcome.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "epoch8_language_splits"
E7_MANIFEST = ROOT / "reports" / "epoch7_selective_language_grounding" / "method_partition_manifest.json"
PARA_CSV = Path("/mnt/c/assets/repos/LIBERO-Para/metrics/libero_para_metadata.csv")
CF_ROOT = Path("/mnt/c/assets/repos/LIBERO-CF/libero/libero")
SALT = "epoch8-language-grounding-splits-v1"
FROZEN_AT = "2026-07-20T19:55:00+09:00"
FAMILIES = ("act", "obj", "comp")
TRAIN_TASKS = (0, 1, 2, 3, 4, 5, 7, 8)
PRIMARY_TARGET_TASKS = (3, 4, 5, 7, 8)
GOAL_GENERALIZATION_TASKS = (6, 9)
MANUAL_AUDIT_REJECTIONS = {
    "4a74defd2da3967a4722ac16874ef037eb7201943213e51c40f4771fcbc00b1b": (
        "The question does not entail pushing or moving the plate to the requested goal state."
    ),
    "8c1113dab269a9ae412d4ae10d9cc9ea9fcec8f065874156b212b74edd88f592": (
        "The paraphrase substitutes pot for bowl and burner for stove, so the physical referents are not reliably equivalent."
    ),
    "ba14d4a6c9307a8e40136c9b9f4b2c6ab434c8622dfabf727072db88f0504e10": (
        "The paraphrase adds a heating intent and substitutes pan for bowl, changing both object identity and requested state."
    ),
}
MANUAL_AUDIT_CRITERIA = [
    "The wording must be grammatical enough to determine an executable request.",
    "The request or pragmatically licensed hint must entail the canonical task goal in the released benchmark scene.",
    "Object aliases may vary, but may not introduce a plausible different physical referent or state predicate.",
    "No extra action, target, or intent may change the benchmark success condition.",
]


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_id(row: dict[str, str]) -> str:
    fields = (
        row["eval"],
        row["high"],
        row["mid"],
        row["low"],
        row["batch_idx"],
        row["original_instruction"],
        row["new_instruction"],
    )
    return sha256_bytes("|".join(fields).encode("utf-8"))


def stable_key(*parts: object) -> str:
    return sha256_bytes("|".join([SALT, *(str(part) for part in parts)]).encode("utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_language_audit_md(path: Path, audit: dict[str, Any]) -> None:
    lines = [
        "# Epoch 8 Validation Language-Pair Audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "This audit was completed from instruction text and frozen benchmark metadata only. No model, simulator, or Ours outcome was loaded. All 30 final validation pairs were reviewed before candidate formulation.",
        "",
        "## Criteria",
        "",
    ]
    lines.extend(f"- {criterion}" for criterion in audit["criteria"])
    lines.extend(
        [
            "",
            "## Summary",
            "",
            f"- Final accepted pairs: {audit['counts']['accepted_final_pairs']} / {audit['counts']['required_final_pairs']}",
            f"- Rejected ranked attempts: {audit['counts']['rejected_attempts']}",
            f"- Reviewed attempts: {audit['counts']['reviewed_attempts']}",
            "- Selection after rejection: advance to the next row under the original frozen SHA256 ranking; no manual replacement choice.",
            "",
            "## Decisions",
            "",
            "| Task | Family | Decision | Row ID | Paraphrase | Reason |",
            "|---:|---|---|---|---|---|",
        ]
    )
    for row in audit["decisions"]:
        paraphrase = row["paraphrase_instruction"].replace("|", "\\|")
        reason = row["reason"].replace("|", "\\|")
        lines.append(
            f"| {row['eval_id']} | {row['family']} | {row['decision']} | `{row['row_id']}` | {paraphrase} | {reason} |"
        )
    lines.extend(
        [
            "",
            "## Leakage Check",
            "",
            "`ours_outcomes_observed=false`; `model_loaded=false`; `simulator_episode_count=0`. Confirmation language text remains sealed.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def common(role: str) -> dict[str, Any]:
    return {
        "schema_version": f"epoch8.language_{role}_manifest.v1",
        "frozen_at": FROZEN_AT,
        "role": role,
        "salt": SALT,
        "candidate_independent": True,
        "ours_outcomes_observed": False,
        "model_loaded": False,
        "simulator_episode_count": 0,
        "source_partition": {
            "path": E7_MANIFEST.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(E7_MANIFEST),
            "assignment_sha256": "041e3cb49f2daf72fe5dd55e71cc7c5dd0bfddc1e11d2dc2707181457110bfd1",
        },
    }


def select_rows(
    all_rows: list[dict[str, str]],
    permitted: set[str],
    excluded: set[str],
    tasks: tuple[int, ...],
    purpose: str,
    include_text: bool,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in all_rows:
        rid = row_id(row)
        key = (int(row["eval"]), row["high"])
        if rid in permitted and rid not in excluded and key[0] in tasks and key[1] in FAMILIES:
            grouped[key].append(row)
    selected: list[dict[str, Any]] = []
    for eval_id in tasks:
        for family in FAMILIES:
            candidates = sorted(grouped[(eval_id, family)], key=lambda row: stable_key(purpose, row_id(row)))
            if not candidates:
                raise ValueError(f"no {purpose} row for eval={eval_id}, family={family}")
            row = candidates[0]
            record: dict[str, Any] = {
                "row_id": row_id(row),
                "eval_id": eval_id,
                "family": family,
                "high": row["high"],
                "mid": row["mid"],
                "low": row["low"],
                "batch_idx": int(row["batch_idx"]),
                "keyword_similarity": float(row["keyword_similarity"]),
                "structural_similarity": float(row["structural_similarity"]),
            }
            if include_text:
                record["canonical_instruction"] = row["original_instruction"]
                record["paraphrase_instruction"] = row["new_instruction"]
            selected.append(record)
    return selected


def select_demos(
    records: list[dict[str, Any]],
    tasks: tuple[int, ...],
    purpose: str,
    excluded_paths: set[str] | None = None,
) -> list[dict[str, Any]]:
    excluded_paths = excluded_paths or set()
    selected: list[dict[str, Any]] = []
    for eval_id in tasks:
        candidates = sorted(
            (row for row in records if row["eval_id"] == eval_id and row["relative_path"] not in excluded_paths),
            key=lambda row: stable_key(purpose, row["relative_path"]),
        )
        if not candidates:
            raise ValueError(f"no {purpose} demonstration for eval={eval_id}")
        row = candidates[0]
        selected.append(
            {
                "eval_id": eval_id,
                "relative_path": row["relative_path"],
                "sha256": row["sha256"],
                "bytes": row["bytes"],
                "frames": row["frames"],
                "first_dynamic_index": row["first_dynamic_index"],
            }
        )
    return selected


def cf_generalization_tasks() -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    suites = ("libero_cf_spatial", "libero_cf_object", "libero_cf_long", "libero_cf_ood")
    for suite in suites:
        bddl_dir = CF_ROOT / "bddl_files" / suite
        init_dir = CF_ROOT / "init_files" / suite
        ranked = sorted(bddl_dir.glob("*.bddl"), key=lambda path: stable_key("cf-task", suite, path.name))
        for path in ranked[:2]:
            init_path = init_dir / f"{path.stem}.pruned_init"
            if not init_path.exists():
                raise FileNotFoundError(init_path)
            output.append(
                {
                    "suite": suite,
                    "task_stem": path.stem,
                    "bddl_relative_path": path.relative_to(CF_ROOT).as_posix(),
                    "bddl_sha256": sha256_file(path),
                    "init_relative_path": init_path.relative_to(CF_ROOT).as_posix(),
                    "init_sha256": sha256_file(init_path),
                    "initial_state_indices": [0, 1, 2],
                    "policy_query_seeds": [211, 212, 213],
                }
            )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview-audit", action="store_true")
    args = parser.parse_args()

    inherited = json.loads(E7_MANIFEST.read_text(encoding="utf-8"))
    if inherited["assignment_sha256"] != "041e3cb49f2daf72fe5dd55e71cc7c5dd0bfddc1e11d2dc2707181457110bfd1":
        raise ValueError("inherited partition assignment changed")
    with PARA_CSV.open("r", encoding="utf-8", newline="") as stream:
        all_rows = list(csv.DictReader(stream))
    if sha256_file(PARA_CSV) != inherited["artifacts"]["libero_para"]["metadata_sha256"]:
        raise ValueError("LIBERO-Para metadata hash changed")

    stage0_used_rows = {row["row_id"] for row in inherited["stage0_base_energy_samples"]["paraphrases"]}
    stage0_used_demos = {row["relative_path"] for row in inherited["stage0_base_energy_samples"]["demonstrations"]}
    validation_permitted = set(inherited["paraphrase_partition_row_ids"]["validation"])
    if not set(MANUAL_AUDIT_REJECTIONS).issubset(validation_permitted - stage0_used_rows):
        raise ValueError("a manually rejected row is not an unused validation row")
    validation_rows = select_rows(
        all_rows,
        validation_permitted,
        stage0_used_rows | set(MANUAL_AUDIT_REJECTIONS),
        tuple(range(10)),
        "epoch8-validation-pair",
        True,
    )
    if args.preview_audit:
        for row in validation_rows:
            print(
                f"eval{row['eval_id']} {row['family']} {row['row_id']}\n"
                f"  canonical:  {row['canonical_instruction']}\n"
                f"  paraphrase: {row['paraphrase_instruction']}\n"
                f"  type: {row['mid']}/{row['low']}"
            )
        return 0

    canonical = {int(key): value for key, value in inherited["canonical_instructions"].items()}
    discovery = common("discovery")
    discovery.update(
        {
            "status": "CONSUMED_DISCOVERY_EVIDENCE_ONLY",
            "closed_loop": {
                "protocol": "reports/epoch7_selective_language_grounding/problem_verification_protocol.json",
                "initial_state_indices": [0, 1, 2],
                "policy_query_seeds": [7, 8, 9],
                "pairs": 30,
                "episodes": 60,
                "results": {"canonical": "30/30", "paraphrase": "19/30"},
                "may_be_reused_as_confirmation": False,
            },
            "epoch7_stage0": {
                "sample_count": 30,
                "demo_paths": sorted(stage0_used_demos),
                "paraphrase_row_ids": sorted(stage0_used_rows),
                "decision": "STAGE0_MECHANISM_NOT_SUPPORTED_FOR_FROZEN_SCALAR_ACTION_ENERGY",
                "may_be_reused_for_epoch8_threshold_or_confirmation": False,
            },
        }
    )

    train_demo_records = [row for row in inherited["demo_partitions"]["train"] if row["eval_id"] in TRAIN_TASKS]
    train_row_ids = []
    train_allowed = set(inherited["paraphrase_partition_row_ids"]["train"])
    for row in all_rows:
        if int(row["eval"]) in TRAIN_TASKS and row_id(row) in train_allowed:
            train_row_ids.append(row_id(row))
    training = common("training")
    training.update(
        {
            "status": "FROZEN_TRAINING_POOL",
            "task_eval_ids": list(TRAIN_TASKS),
            "held_out_task_eval_ids": list(GOAL_GENERALIZATION_TASKS),
            "demonstrations": [
                {key: row[key] for key in ("eval_id", "relative_path", "sha256", "bytes", "frames", "first_dynamic_index")}
                for row in train_demo_records
            ],
            "paraphrase_row_ids": sorted(train_row_ids),
            "counts": {"demonstrations": len(train_demo_records), "paraphrases": len(train_row_ids)},
            "legal_privileged_supervision": [
                "simulator object identity, segmentation, contact, and task metadata may be derived for these training rows only",
                "no privileged field may enter inference",
            ],
        }
    )

    validation = common("validation")
    validation.update(
        {
            "status": "FROZEN_MANUALLY_AUDITED",
            "selection": "one unused Epoch 7 validation paraphrase per task/family, SHA256-ranked with the Epoch 8 salt; Candidate A and immutable manually rejected rows excluded",
            "language_pair_audit": "reports/epoch8_language_splits/language_pair_audit.json",
            "demonstrations": select_demos(
                inherited["demo_partitions"]["validation"],
                tuple(range(10)),
                "epoch8-validation-demo",
                stage0_used_demos,
            ),
            "language_pairs": validation_rows,
            "primary_stage0_task_eval_ids": list(PRIMARY_TARGET_TASKS),
            "target_swap_groups": [
                {"name": "bowl_destination", "eval_ids": [3, 4, 5], "directed_pairs": [[3, 4], [4, 3], [4, 5], [5, 4], [3, 5], [5, 3]]},
                {"name": "wine_destination", "eval_ids": [7, 8], "directed_pairs": [[7, 8], [8, 7]]},
            ],
            "closed_loop_screen": {
                "initial_state_indices": [3, 4],
                "policy_query_seeds": [13, 14],
                "outcomes_may_select_only_predeclared_checkpoint_alternatives": True,
                "may_change_task_or_metric_definition": False,
            },
        }
    )

    confirmation_rows = select_rows(
        all_rows,
        set(inherited["paraphrase_partition_row_ids"]["confirmatory"]),
        set(),
        TRAIN_TASKS,
        "epoch8-confirmation-pair",
        False,
    )
    directed_swaps = [[3, 4], [4, 3], [4, 5], [5, 4], [3, 5], [5, 3], [7, 8], [8, 7]]
    episodes: list[dict[str, Any]] = []
    for eval_id in TRAIN_TASKS:
        episodes.append({"condition": "canonical", "eval_id": eval_id, "instruction_eval_id": eval_id, "initial_state_index": 10, "policy_query_seed": 101})
    family_reset = {"act": (11, 102), "obj": (12, 103), "comp": (13, 104)}
    for row in confirmation_rows:
        reset, seed = family_reset[row["family"]]
        episodes.append({"condition": "paraphrase", "eval_id": row["eval_id"], "instruction_eval_id": row["eval_id"], "row_id": row["row_id"], "family": row["family"], "initial_state_index": reset, "policy_query_seed": seed})
    for source_eval_id, instruction_eval_id in directed_swaps:
        episodes.append({"condition": "target_swap", "eval_id": instruction_eval_id, "state_source_eval_id": source_eval_id, "instruction_eval_id": instruction_eval_id, "instruction": canonical[instruction_eval_id], "initial_state_index": 14, "policy_query_seed": 105, "requires_world_signature_identity": True})
    if len(episodes) != 40:
        raise AssertionError(len(episodes))
    confirmation = common("confirmation")
    confirmation.update(
        {
            "status": "SEALED_UNTIL_STAGE_B_AUTHORIZATION",
            "task_eval_ids": list(TRAIN_TASKS),
            "demonstrations": select_demos(inherited["demo_partitions"]["confirmatory"], TRAIN_TASKS, "epoch8-confirmation-demo"),
            "language_pairs": confirmation_rows,
            "language_text_sealed": True,
            "episodes_per_policy": episodes,
            "episode_count_per_policy": len(episodes),
            "policy_roles": ["Base", "strongest competent Prior if available", "canonicalization control", "plain paraphrase augmentation control", "Ours", "mechanism ablation", "capacity-matched control"],
            "selection_or_tuning_on_outcomes": False,
            "sealed_rule": inherited["sealed_confirmatory_rule"],
            "target_swap_compatibility_audit": "reports/epoch8_language_splits/target_swap_compatibility_audit.json",
        }
    )

    generalization_rows = select_rows(
        all_rows,
        set(inherited["paraphrase_partition_row_ids"]["confirmatory"]),
        set(),
        GOAL_GENERALIZATION_TASKS,
        "epoch8-goal-generalization-pair",
        False,
    )
    generalization = common("generalization")
    generalization.update(
        {
            "status": "SEALED_HELD_OUT_TASK_AND_SUITE_GENERALIZATION",
            "goal_task_holdout": {
                "task_eval_ids": list(GOAL_GENERALIZATION_TASKS),
                "canonical_instructions": {str(eval_id): canonical[eval_id] for eval_id in GOAL_GENERALIZATION_TASKS},
                "training_exclusion": True,
                "confirmatory_paraphrase_rows": generalization_rows,
                "language_text_sealed": True,
                "initial_state_indices": [15, 16, 17, 18, 19],
                "policy_query_seeds": [115, 116, 117, 118, 119],
            },
            "libero_cf_holdout": {
                "artifact_revision": "8460457bfca6e0ef2e856bc104e2c60b023ef2a7",
                "selection": "two BDDLs per released suite, SHA256-ranked by salt/suite/path before any X-VLA/Ours outcome",
                "tasks": cf_generalization_tasks(),
            },
            "may_tune_on_outcomes": False,
        }
    )

    manifests = {
        "discovery_manifest.json": discovery,
        "training_manifest.json": training,
        "validation_manifest.json": validation,
        "confirmation_manifest.json": confirmation,
        "generalization_manifest.json": generalization,
    }
    for name, payload in manifests.items():
        write_json(OUT / name, payload)

    row_by_id = {row_id(row): row for row in all_rows}
    decisions: list[dict[str, Any]] = []
    for rid, reason in MANUAL_AUDIT_REJECTIONS.items():
        row = row_by_id[rid]
        decisions.append(
            {
                "row_id": rid,
                "eval_id": int(row["eval"]),
                "family": row["high"],
                "canonical_instruction": row["original_instruction"],
                "paraphrase_instruction": row["new_instruction"],
                "decision": "REJECT",
                "reason": reason,
            }
        )
    for row in validation_rows:
        decisions.append(
            {
                "row_id": row["row_id"],
                "eval_id": row["eval_id"],
                "family": row["family"],
                "canonical_instruction": row["canonical_instruction"],
                "paraphrase_instruction": row["paraphrase_instruction"],
                "decision": "ACCEPT",
                "reason": "Grammatical request or licensed hint that preserves the benchmark-scene object referents and success condition.",
            }
        )
    decisions.sort(key=lambda row: (row["eval_id"], FAMILIES.index(row["family"]), row["decision"], row["row_id"]))
    audit = {
        "schema_version": "epoch8.language_pair_manual_audit.v1",
        "audited_at": FROZEN_AT,
        "status": "COMPLETE_BEFORE_CANDIDATE_FORMULATION",
        "review_mode": "manual semantic review of text plus released task identity; no policy or simulator outcomes",
        "selection_policy": "advance to the next unused validation row under the original frozen SHA256 ranking after an explicit rejection",
        "candidate_independent": True,
        "ours_outcomes_observed": False,
        "model_loaded": False,
        "simulator_episode_count": 0,
        "criteria": MANUAL_AUDIT_CRITERIA,
        "counts": {
            "required_final_pairs": 30,
            "accepted_final_pairs": len(validation_rows),
            "rejected_attempts": len(MANUAL_AUDIT_REJECTIONS),
            "reviewed_attempts": len(decisions),
        },
        "decisions": decisions,
    }
    if len(validation_rows) != 30:
        raise AssertionError(len(validation_rows))
    write_json(OUT / "language_pair_audit.json", audit)
    write_language_audit_md(OUT / "language_pair_audit.md", audit)

    indexed_supplements = ["language_pair_audit.json", "language_pair_audit.md"]
    if (OUT / "target_swap_compatibility_audit.json").exists():
        indexed_supplements.append("target_swap_compatibility_audit.json")
    index = {
        "schema_version": "epoch8.language_split_index.v1",
        "frozen_at": FROZEN_AT,
        "status": "FROZEN_BEFORE_CANDIDATE_FORMULATION",
        "salt": SALT,
        "candidate_independent": True,
        "ours_outcomes_observed": False,
        "files": [
            {"path": f"reports/epoch8_language_splits/{name}", "sha256": sha256_file(OUT / name), "bytes": (OUT / name).stat().st_size}
            for name in manifests
        ] + [
            {"path": f"reports/epoch8_language_splits/{name}", "sha256": sha256_file(OUT / name), "bytes": (OUT / name).stat().st_size}
            for name in indexed_supplements
        ],
        "role_integrity": {
            "discovery_never_confirmation": True,
            "candidate_a_rows_excluded_from_validation": True,
            "goal_generalization_tasks_excluded_from_training": True,
            "confirmation_text_and_outcomes_sealed": True,
            "confirmation_may_not_select_threshold_task_or_checkpoint": True,
        },
        "next_gate": "Formulate and compare two causally distinct mechanisms without reading sealed confirmation text or outcomes.",
    }
    write_json(OUT / "split_index.json", index)
    print(json.dumps({"output": str(OUT), "validation_pairs": len(validation_rows), "manual_rejections": len(MANUAL_AUDIT_REJECTIONS), "confirmation_episodes_per_policy": len(episodes), "generalization_cf_tasks": len(generalization["libero_cf_holdout"]["tasks"])}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
