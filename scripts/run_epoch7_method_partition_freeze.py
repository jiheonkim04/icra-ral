#!/usr/bin/env python3
"""Freeze outcome-independent Epoch 7 method-development partitions.

This script reads the official X-VLA-format LIBERO-Goal artifact and released
LIBERO-Para metadata.  It does not load a model, run a simulator, train, or
inspect any policy outcome.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np


SALT = "epoch7-equivalence-selectivity-partitions-v1"
DATASET_ID = "2toINF/Libero-XVLA-format"
DATASET_REVISION = "27ddd36538ee4812bd31fd8b494f8d7c6a11ef9d"
MODEL_ID = "2toINF/X-VLA-Libero"
MODEL_REVISION = "129e71460678b7236cee6fc9707f09d9fa0c3590"
SOURCE_REVISION = "6bc2513f5f1cbec715cc668b414392a6cae5c671"
PARA_REVISION = "5a2198299a6d7a49bdb3cd519c7e92ed803adf5f"
FAMILIES = ("act", "obj", "comp")


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def digest_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stable_key(*parts: object) -> str:
    return digest_text("|".join([SALT, *(str(part) for part in parts)]))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def split_demo_paths(paths: Iterable[Path]) -> dict[str, list[Path]]:
    ranked = sorted(paths, key=lambda path: stable_key("demo", path.parent.name, path.name))
    if len(ranked) < 12:
        raise ValueError("each task needs at least 12 official demonstrations")
    return {
        "train": ranked[:-8],
        "validation": ranked[-8:-4],
        "confirmatory": ranked[-4:],
    }


def split_paraphrase_rows(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    ranked = sorted(rows, key=lambda row: stable_key("para", paraphrase_row_id(row)))
    count = len(ranked)
    if count < 6:
        raise ValueError("each task/family needs at least six paraphrases")
    validation_count = max(1, int(math.floor(count * 0.15)))
    confirmatory_count = max(1, int(math.floor(count * 0.15)))
    train_end = count - validation_count - confirmatory_count
    return {
        "train": ranked[:train_end],
        "validation": ranked[train_end : train_end + validation_count],
        "confirmatory": ranked[train_end + validation_count :],
    }


def paraphrase_row_id(row: dict[str, str]) -> str:
    fields = (
        row["eval"],
        row["high"],
        row["mid"],
        row["low"],
        row["batch_idx"],
        row["original_instruction"],
        row["new_instruction"],
    )
    return digest_text("|".join(fields))


def token_jaccard(left: str, right: str) -> float:
    lset = set(left.lower().split())
    rset = set(right.lower().split())
    return len(lset & rset) / len(lset | rset)


def hard_negative_map(canonical: dict[int, str]) -> dict[int, dict[str, Any]]:
    output: dict[int, dict[str, Any]] = {}
    for eval_id, instruction in canonical.items():
        candidates = [
            (token_jaccard(instruction, other), other_id, other)
            for other_id, other in canonical.items()
            if other_id != eval_id
        ]
        similarity, other_id, other = sorted(candidates, key=lambda item: (-item[0], item[1]))[0]
        output[eval_id] = {
            "eval_id": other_id,
            "instruction": other,
            "token_jaccard": similarity,
            "selection": "maximum lowercase-whitespace-token Jaccard; lowest eval_id tie break",
        }
    return output


def first_dynamic_index(abs_action_6d: np.ndarray, image_count: int, horizon: int = 30) -> int:
    upper = min(10, len(abs_action_6d) - horizon - 1, image_count - 2)
    for index in range(max(0, upper + 1)):
        if float(np.max(np.abs(abs_action_6d[index + 1] - abs_action_6d[index]))) >= 1e-5:
            return index
    raise ValueError("no dynamic early window compatible with the official X-VLA handler")


def audit_demo(path: Path, root: Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        required = {
            "abs_action_6d",
            "language_instruction",
            "observation/third_image",
            "observation/wrist_image",
            "proprio",
        }
        missing = sorted(required - set(name for name in required if name in handle))
        if missing:
            raise ValueError(f"{path} missing {missing}")
        actions = np.asarray(handle["abs_action_6d"], dtype=np.float32)
        proprio = np.asarray(handle["proprio"], dtype=np.float32)
        third_count = len(handle["observation/third_image"])
        wrist_count = len(handle["observation/wrist_image"])
        instruction_value = handle["language_instruction"][()]
        instruction = (
            instruction_value.decode("utf-8")
            if isinstance(instruction_value, (bytes, np.bytes_))
            else str(instruction_value)
        )
        if actions.ndim != 2 or actions.shape[1] != 10:
            raise ValueError(f"{path} abs_action_6d shape {actions.shape}")
        if proprio.ndim != 2 or proprio.shape[1] != 9:
            raise ValueError(f"{path} proprio shape {proprio.shape}")
        if not (len(actions) == len(proprio) == third_count == wrist_count):
            raise ValueError(f"{path} modality length mismatch")
        if not (np.isfinite(actions).all() and np.isfinite(proprio).all()):
            raise ValueError(f"{path} contains nonfinite action/proprio values")
        dynamic_index = first_dynamic_index(actions, third_count)
    return {
        "relative_path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "frames": int(actions.shape[0]),
        "instruction": instruction,
        "abs_action_6d_shape": [int(value) for value in actions.shape],
        "proprio_shape": [int(value) for value in proprio.shape],
        "finite_action_and_proprio": True,
        "first_dynamic_index": dynamic_index,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--xvla-data-root",
        type=Path,
        default=Path("/mnt/c/assets/datasets/Libero-XVLA-format/libero_goal"),
    )
    parser.add_argument(
        "--para-metadata",
        type=Path,
        default=Path("/mnt/c/assets/repos/LIBERO-Para/metrics/libero_para_metadata.csv"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    task_dirs = sorted(path for path in args.xvla_data_root.iterdir() if path.is_dir())
    if len(task_dirs) != 10:
        raise ValueError(f"expected ten Goal task directories, found {len(task_dirs)}")

    canonical: dict[int, str] = {}
    demo_records: dict[str, list[dict[str, Any]]] = {name: [] for name in ("train", "validation", "confirmatory")}
    stage0_demo_samples: list[dict[str, Any]] = []
    task_counts: dict[str, dict[str, int]] = {}
    all_demo_bytes = 0
    eval_by_instruction: dict[str, int] = {}

    # LIBERO-Para eval ids define the canonical task order.
    with args.para_metadata.open("r", encoding="utf-8", newline="") as stream:
        para_rows = list(csv.DictReader(stream))
    for row in para_rows:
        eval_id = int(row["eval"])
        canonical.setdefault(eval_id, row["original_instruction"])
    if sorted(canonical) != list(range(10)):
        raise ValueError("LIBERO-Para canonical eval ids are not 0..9")
    eval_by_instruction = {instruction: eval_id for eval_id, instruction in canonical.items()}

    for task_dir in task_dirs:
        paths = sorted(task_dir.glob("*.hdf5"))
        split = split_demo_paths(paths)
        task_counts[task_dir.name] = {name: len(values) for name, values in split.items()}
        audited_by_path: dict[Path, dict[str, Any]] = {}
        for split_name, split_paths in split.items():
            for path in split_paths:
                record = audit_demo(path, args.xvla_data_root)
                audited_by_path[path] = record
                all_demo_bytes += int(record["bytes"])
                instruction = str(record["instruction"])
                if instruction not in eval_by_instruction:
                    raise ValueError(f"unmapped canonical instruction {instruction!r}")
                demo_records[split_name].append(
                    {
                        **record,
                        "eval_id": eval_by_instruction[instruction],
                        "split": split_name,
                    }
                )
        selected_path = sorted(
            split["validation"],
            key=lambda path: stable_key("stage0-demo", task_dir.name, path.name),
        )[0]
        selected = audited_by_path[selected_path]
        index = int(selected["first_dynamic_index"])
        stage0_demo_samples.append(
            {
                "eval_id": eval_by_instruction[str(selected["instruction"])],
                "relative_path": selected["relative_path"],
                "file_sha256": selected["sha256"],
                "frame_index": index,
                "image_frame_index": index + 1,
                "proprio_frame_index": index,
                "action_frame_indices": [index + 1, index + 30],
                "handler_semantics": "official LiberoHandler: drop first image; 31-point 30 Hz/1 s trajectory; action_slice drops proprio point",
            }
        )

    grouped_rows: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in para_rows:
        key = (int(row["eval"]), row["high"])
        if key[1] not in FAMILIES:
            raise ValueError(f"unknown family {key[1]}")
        grouped_rows[key].append(row)
    if set(grouped_rows) != {(eval_id, family) for eval_id in range(10) for family in FAMILIES}:
        raise ValueError("missing task/family paraphrase group")

    para_ids: dict[str, list[str]] = {name: [] for name in ("train", "validation", "confirmatory")}
    para_counts: dict[str, dict[str, dict[str, int]]] = {str(i): {} for i in range(10)}
    stage0_paraphrases: list[dict[str, Any]] = []
    for (eval_id, family), rows in sorted(grouped_rows.items()):
        split = split_paraphrase_rows(rows)
        para_counts[str(eval_id)][family] = {name: len(values) for name, values in split.items()}
        for split_name, split_rows in split.items():
            para_ids[split_name].extend(paraphrase_row_id(row) for row in split_rows)
        selected = sorted(
            split["validation"],
            key=lambda row: stable_key("stage0-para", paraphrase_row_id(row)),
        )[0]
        stage0_paraphrases.append(
            {
                "row_id": paraphrase_row_id(selected),
                "eval_id": eval_id,
                "family": family,
                "instruction": selected["new_instruction"],
                "original_instruction": selected["original_instruction"],
                "high": selected["high"],
                "mid": selected["mid"],
                "low": selected["low"],
                "batch_idx": int(selected["batch_idx"]),
                "structural_similarity": float(selected["structural_similarity"]),
                "keyword_similarity": float(selected["keyword_similarity"]),
            }
        )

    for values in demo_records.values():
        values.sort(key=lambda row: (row["eval_id"], row["relative_path"]))
    for values in para_ids.values():
        values.sort()
    stage0_demo_samples.sort(key=lambda row: row["eval_id"])
    stage0_paraphrases.sort(key=lambda row: (row["eval_id"], FAMILIES.index(row["family"])))

    assignments = {
        "demo": {
            name: [row["relative_path"] for row in rows]
            for name, rows in demo_records.items()
        },
        "paraphrase": para_ids,
    }
    assignment_sha256 = digest_text(json.dumps(assignments, sort_keys=True, separators=(",", ":")))
    metadata_sha256 = sha256_file(args.para_metadata)
    negatives = hard_negative_map(canonical)

    payload = {
        "schema_version": "epoch7.method_partition_manifest.v1",
        "created_at": timestamp(),
        "status": "FROZEN_BEFORE_OURS_EXECUTION",
        "execution_type": "OFFLINE_DIAGNOSTIC_NO_MODEL_NO_SIMULATOR_NO_OUTCOMES",
        "salt": SALT,
        "assignment_sha256": assignment_sha256,
        "policy": {
            "model_loaded": False,
            "training_happened": False,
            "optimizer_step_happened": False,
            "simulator_episode_count": 0,
            "closed_loop_outcome_read": False,
            "ours_outcome_read": False,
            "confirmatory_content_used_for_tuning": False,
        },
        "artifacts": {
            "xvla_format_dataset": {
                "dataset_id": DATASET_ID,
                "revision": DATASET_REVISION,
                "license": "Apache-2.0",
                "root": str(args.xvla_data_root),
                "files": sum(len(rows) for rows in demo_records.values()),
                "bytes": all_demo_bytes,
            },
            "libero_para": {
                "revision": PARA_REVISION,
                "metadata_path": str(args.para_metadata),
                "metadata_sha256": metadata_sha256,
                "rows": len(para_rows),
            },
            "xvla_model": {
                "model_id": MODEL_ID,
                "revision": MODEL_REVISION,
                "source_revision": SOURCE_REVISION,
            },
        },
        "canonical_instructions": {str(key): value for key, value in sorted(canonical.items())},
        "demo_partition_rule": "Within each task, SHA256-rank by salt/task/file; last four confirmatory, previous four validation, remainder train.",
        "demo_partition_counts": task_counts,
        "demo_partitions": demo_records,
        "paraphrase_partition_rule": "Within each eval_id/family, SHA256-rank by immutable row id; floor(15%) validation, floor(15%) confirmatory, remainder train.",
        "paraphrase_partition_counts": para_counts,
        "paraphrase_partition_row_ids": para_ids,
        "hard_negative_instructions": {str(key): value for key, value in sorted(negatives.items())},
        "stage0_base_energy_samples": {
            "demonstrations": stage0_demo_samples,
            "paraphrases": stage0_paraphrases,
            "cross_product_rule": "one frozen validation demo per task crossed with that task's one frozen validation paraphrase per family",
            "sample_count": 30,
            "negative_rule": "one frozen maximum-token-Jaccard distinct canonical instruction per task",
            "diffusion_time_noise_seeds": [1701, 1702],
        },
        "future_closed_loop_partitions": {
            "prior_problem_discovery_consumed": {
                "initial_state_indices": [0, 1, 2],
                "seeds": [7, 8, 9],
            },
            "stage_a_discovery": {
                "task_eval_ids": list(range(10)),
                "initial_state_indices": [3, 4],
                "seeds": [13, 14],
                "paraphrase_split": "validation",
                "outcomes_may_debug_or_select_only_prespecified_checkpoint_alternatives": True,
            },
            "stage_b_confirmatory": {
                "task_eval_ids": list(range(10)),
                "initial_state_indices": [10, 11, 12, 13, 14],
                "seeds": [101, 102, 103, 104, 105],
                "paraphrase_split": "confirmatory",
                "sealed_until_stage_b_authorization": True,
                "may_tune_on_outcomes": False,
            },
            "libero_cf_generalization": {
                "artifact_revision": "8460457bfca6e0ef2e856bc104e2c60b023ef2a7",
                "task_selection_rule": "SHA256-rank within each released suite before any X-VLA outcome",
                "initial_state_indices": [0, 1, 2],
                "seeds": [211, 212, 213],
                "status": "ROUTE_FROZEN_TASK_MANIFEST_DEFERRED_UNTIL_STAGE_A_GO",
            },
        },
        "sealed_confirmatory_rule": "Paths and hashes may be integrity-audited; images, actions, paraphrase text, and outcomes may not be used for training, checkpoint selection, threshold selection, or debugging before Stage B authorization.",
    }
    atomic_write_json(args.output, payload)
    print(json.dumps({
        "status": payload["status"],
        "assignment_sha256": assignment_sha256,
        "demo_counts": {name: len(rows) for name, rows in demo_records.items()},
        "paraphrase_counts": {name: len(rows) for name, rows in para_ids.items()},
        "stage0_samples": 30,
        "output": str(args.output),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
