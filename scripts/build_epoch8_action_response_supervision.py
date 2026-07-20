#!/usr/bin/env python3
"""Freeze real-demo supervision for Epoch 8 paired action response.

This is an outcome-independent data audit. It reads only the frozen split
manifests, released paraphrase metadata, and official X-VLA-format training
demonstrations. It loads no model and runs no simulator.
"""

from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = ROOT / "reports/epoch8_language_splits"
TRAIN_MANIFEST = SPLIT_ROOT / "training_manifest.json"
VALIDATION_MANIFEST = SPLIT_ROOT / "validation_manifest.json"
LANGUAGE_AUDIT = SPLIT_ROOT / "language_pair_audit.json"
DATA_ROOT = Path("/mnt/c/assets/datasets/Libero-XVLA-format/libero_goal")
PARA_CSV = Path("/mnt/c/assets/repos/LIBERO-Para/metrics/libero_para_metadata.csv")
OUTPUT = ROOT / "reports/epoch8_action_response_supervision.json"
FROZEN_AT = "2026-07-20T20:10:00+09:00"
SALT = "epoch8-pcat-action-response-supervision-v1"
TASKS = (3, 4, 5, 7, 8)
FAMILIES = ("act", "obj", "comp")
PAIR_GROUPS = (
    ("bowl_destination_plate_stove", 3, 4),
    ("bowl_destination_plate_cabinet", 3, 5),
    ("bowl_destination_stove_cabinet", 4, 5),
    ("wine_destination_rack_cabinet", 7, 8),
)
TRAIN_PAIRS_PER_GROUP = 12
FRAME_INDEX = 0
CHUNK_LENGTH = 30
# Fixed physical tolerances: 5 cm per translation coordinate and 0.2 per
# continuous 6D rotation coordinate. These are not estimated from outcomes.
MATCH_SCALES = np.asarray([0.05, 0.05, 0.05, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2], dtype=np.float64)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
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


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_action_sample(record: dict[str, Any]) -> dict[str, Any]:
    path = DATA_ROOT / record["relative_path"]
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"demonstration hash drift: {path}")
    with h5py.File(path, "r") as handle:
        raw = np.asarray(handle["abs_action_6d"], dtype=np.float32)
        if len(raw) < FRAME_INDEX + CHUNK_LENGTH + 1 or raw.shape[1] != 10:
            raise ValueError(f"invalid action shape: {path} {raw.shape}")
        left = np.concatenate(
            [raw[:, :9], (raw[:, 9:] > 0.0).astype(np.float32)], axis=-1
        )
        canonical_value = handle["language_instruction"][()]
        canonical = (
            canonical_value.decode("utf-8")
            if isinstance(canonical_value, bytes)
            else str(canonical_value)
        )
    proprio = left[FRAME_INDEX]
    action = left[FRAME_INDEX + 1 : FRAME_INDEX + CHUNK_LENGTH + 1]
    if not np.isfinite(proprio).all() or not np.isfinite(action).all():
        raise ValueError(f"nonfinite data: {path}")
    return {
        "eval_id": int(record["eval_id"]),
        "relative_path": record["relative_path"],
        "file_sha256": record["sha256"],
        "canonical_instruction": canonical,
        "frame_index": FRAME_INDEX,
        "proprio_left": proprio,
        "action_left": action,
        "proprio_sha256": array_sha256(proprio),
        "action_sha256": array_sha256(action),
    }


def match_distance(left: dict[str, Any], right: dict[str, Any]) -> float:
    delta = (left["proprio_left"][:9] - right["proprio_left"][:9]) / MATCH_SCALES
    return float(np.sqrt(np.mean(delta**2)))


def action_delta_rms(left: dict[str, Any], right: dict[str, Any], scales: np.ndarray) -> float:
    delta = (right["action_left"][:, :9] - left["action_left"][:, :9]) / scales
    return float(np.sqrt(np.mean(delta**2)))


def public_sample(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        key: sample[key]
        for key in (
            "eval_id",
            "relative_path",
            "file_sha256",
            "canonical_instruction",
            "frame_index",
            "proprio_sha256",
            "action_sha256",
        )
    }


def pair_samples(
    name: str,
    left_task: int,
    right_task: int,
    samples: dict[int, list[dict[str, Any]]],
    action_scales: np.ndarray,
    count: int,
) -> list[dict[str, Any]]:
    candidates = []
    for left_index, left in enumerate(samples[left_task]):
        for right_index, right in enumerate(samples[right_task]):
            candidates.append(
                (
                    match_distance(left, right),
                    stable_key(name, left["relative_path"], right["relative_path"]),
                    left_index,
                    right_index,
                )
            )
    used_left: set[int] = set()
    used_right: set[int] = set()
    output: list[dict[str, Any]] = []
    for distance, _, left_index, right_index in sorted(candidates):
        if left_index in used_left or right_index in used_right:
            continue
        used_left.add(left_index)
        used_right.add(right_index)
        left = samples[left_task][left_index]
        right = samples[right_task][right_index]
        pair_id = stable_key(name, left["relative_path"], right["relative_path"])
        output.append(
            {
                "pair_id": pair_id,
                "pair_group": name,
                "left": public_sample(left),
                "right": public_sample(right),
                "initial_proprio_match_rms": distance,
                "real_action_delta_normalized_rms": action_delta_rms(left, right, action_scales),
                "orientation": "right_minus_left",
            }
        )
        if len(output) == count:
            break
    if len(output) != count:
        raise ValueError(f"only {len(output)} unique matches for {name}")
    return output


def select_training_paraphrases(train: dict[str, Any]) -> list[dict[str, Any]]:
    permitted = set(train["paraphrase_row_ids"])
    with PARA_CSV.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    grouped: dict[tuple[int, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rid = row_id(row)
        key = (int(row["eval"]), row["high"])
        if rid in permitted and key[0] in TASKS and key[1] in FAMILIES:
            grouped[key].append(row)
    output = []
    for eval_id in TASKS:
        for family in FAMILIES:
            ranked = sorted(
                grouped[(eval_id, family)],
                key=lambda row: stable_key("training-paraphrase", eval_id, family, row_id(row)),
            )
            if not ranked:
                raise ValueError(f"no training paraphrase for eval={eval_id} family={family}")
            row = ranked[0]
            output.append(
                {
                    "row_id": row_id(row),
                    "eval_id": eval_id,
                    "family": family,
                    "canonical_instruction": row["original_instruction"],
                    "paraphrase_instruction": row["new_instruction"],
                    "mid": row["mid"],
                    "low": row["low"],
                }
            )
    return output


def main() -> int:
    train = load_json(TRAIN_MANIFEST)
    validation = load_json(VALIDATION_MANIFEST)
    language_audit = load_json(LANGUAGE_AUDIT)
    if train["status"] != "FROZEN_TRAINING_POOL":
        raise ValueError("training split not frozen")
    if validation["status"] != "FROZEN_MANUALLY_AUDITED":
        raise ValueError("validation split not audited")
    if language_audit["status"] != "COMPLETE_BEFORE_CANDIDATE_FORMULATION":
        raise ValueError("language audit status changed")

    train_samples: dict[int, list[dict[str, Any]]] = {eval_id: [] for eval_id in TASKS}
    for record in train["demonstrations"]:
        eval_id = int(record["eval_id"])
        if eval_id in train_samples:
            train_samples[eval_id].append(load_action_sample(record))
    for rows in train_samples.values():
        rows.sort(key=lambda row: row["relative_path"])

    continuous_actions = np.concatenate(
        [sample["action_left"][:, :9] for rows in train_samples.values() for sample in rows], axis=0
    ).astype(np.float64)
    raw_scales = np.std(continuous_actions, axis=0)
    action_scales = np.maximum(raw_scales, 0.02)
    if not np.isfinite(action_scales).all() or np.any(action_scales <= 0):
        raise ValueError("invalid action scales")

    train_pairs: list[dict[str, Any]] = []
    for name, left_task, right_task in PAIR_GROUPS:
        train_pairs.extend(
            pair_samples(
                name,
                left_task,
                right_task,
                train_samples,
                action_scales,
                TRAIN_PAIRS_PER_GROUP,
            )
        )

    validation_samples: dict[int, list[dict[str, Any]]] = {eval_id: [] for eval_id in TASKS}
    for record in validation["demonstrations"]:
        eval_id = int(record["eval_id"])
        if eval_id in validation_samples:
            validation_samples[eval_id].append(load_action_sample(record))
    if any(len(rows) != 1 for rows in validation_samples.values()):
        raise ValueError("expected exactly one validation demo per primary task")
    validation_pairs: list[dict[str, Any]] = []
    for name, left_task, right_task in PAIR_GROUPS:
        validation_pairs.extend(
            pair_samples(name, left_task, right_task, validation_samples, action_scales, 1)
        )

    group_summary: dict[str, Any] = {}
    for name, _, _ in PAIR_GROUPS:
        rows = [row for row in train_pairs if row["pair_group"] == name]
        group_summary[name] = {
            "training_pairs": len(rows),
            "initial_proprio_match_rms_median": float(
                np.median([row["initial_proprio_match_rms"] for row in rows])
            ),
            "initial_proprio_match_rms_max": max(row["initial_proprio_match_rms"] for row in rows),
            "real_action_delta_normalized_rms_median": float(
                np.median([row["real_action_delta_normalized_rms"] for row in rows])
            ),
            "real_action_delta_normalized_rms_min": min(
                row["real_action_delta_normalized_rms"] for row in rows
            ),
        }

    validation_language = [
        row for row in validation["language_pairs"] if int(row["eval_id"]) in TASKS
    ]
    training_language = select_training_paraphrases(train)
    noncollapsed = all(
        summary["real_action_delta_normalized_rms_median"] >= 0.25
        for summary in group_summary.values()
    ) and all(row["real_action_delta_normalized_rms"] >= 0.25 for row in validation_pairs)
    close_matches = all(
        summary["initial_proprio_match_rms_max"] <= 0.25 for summary in group_summary.values()
    ) and all(row["initial_proprio_match_rms"] <= 0.30 for row in validation_pairs)
    pass_gate = (
        len(train_pairs) == len(PAIR_GROUPS) * TRAIN_PAIRS_PER_GROUP
        and len(validation_pairs) == len(PAIR_GROUPS)
        and len(training_language) == len(TASKS) * len(FAMILIES)
        and len(validation_language) == len(TASKS) * len(FAMILIES)
        and noncollapsed
        and close_matches
    )
    payload = {
        "schema_version": "epoch8.action_response_supervision.v1",
        "frozen_at": FROZEN_AT,
        "status": "ACTION_RESPONSE_SUPERVISION_PREFLIGHT_PASS" if pass_gate else "ACTION_RESPONSE_SUPERVISION_PREFLIGHT_FAIL",
        "candidate": "PCAT",
        "classification": "OUTCOME_INDEPENDENT_REAL_DEMONSTRATION_SUPERVISION_AUDIT",
        "candidate_independent_splits_unchanged": True,
        "ours_outcomes_observed": False,
        "model_loaded": False,
        "simulator_episode_count": 0,
        "optimizer_step_count": 0,
        "sources": {
            "training_manifest": {"path": str(TRAIN_MANIFEST.relative_to(ROOT)), "sha256": sha256_file(TRAIN_MANIFEST)},
            "validation_manifest": {"path": str(VALIDATION_MANIFEST.relative_to(ROOT)), "sha256": sha256_file(VALIDATION_MANIFEST)},
            "language_audit": {"path": str(LANGUAGE_AUDIT.relative_to(ROOT)), "sha256": sha256_file(LANGUAGE_AUDIT)},
            "xvla_format_root": str(DATA_ROOT),
            "paraphrase_metadata": {"path": str(PARA_CSV), "sha256": sha256_file(PARA_CSV)},
        },
        "construction": {
            "frame_index": FRAME_INDEX,
            "action_chunk": [FRAME_INDEX + 1, FRAME_INDEX + CHUNK_LENGTH],
            "pre_interaction": True,
            "pairing": "one-to-one greedy minimum initial-proprio RMS with deterministic SHA256 tie break",
            "pair_match_scales": MATCH_SCALES.tolist(),
            "action_delta": "right real clean 30x9 continuous action chunk minus left real clean chunk",
            "action_delta_scales": action_scales.tolist(),
            "action_delta_raw_standard_deviations": raw_scales.tolist(),
            "gripper_excluded_from_transport_vector": True,
            "gripper_retained_in_factual_clean_action_loss": True,
            "no_synthetic_action_label": True,
        },
        "counts": {
            "eligible_training_demonstrations_per_task": {
                str(eval_id): len(rows) for eval_id, rows in train_samples.items()
            },
            "training_action_pairs": len(train_pairs),
            "validation_action_pairs": len(validation_pairs),
            "training_equivalence_pairs": len(training_language),
            "validation_equivalence_pairs": len(validation_language),
            "tasks": len(TASKS),
            "object_families": 2,
        },
        "gates": {
            "minimum_group_median_action_delta_normalized_rms": 0.25,
            "minimum_validation_action_delta_normalized_rms": 0.25,
            "maximum_training_initial_proprio_match_rms": 0.25,
            "maximum_validation_initial_proprio_match_rms": 0.30,
            "noncollapsed": noncollapsed,
            "close_initial_state_matches": close_matches,
        },
        "group_summary": group_summary,
        "training_action_pairs": train_pairs,
        "validation_action_pairs": validation_pairs,
        "training_equivalence_pairs": training_language,
        "validation_equivalence_pairs": validation_language,
        "limits": [
            "Initial end-effector pose is matched, but full simulator state is not present in the released X-VLA-format HDF5 files.",
            "Real action transport is therefore a matched-demonstration causal proxy; official paired simulator rollouts remain the primary endpoint.",
        ],
    }
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(OUTPUT),
                "status": payload["status"],
                "training_action_pairs": len(train_pairs),
                "validation_action_pairs": len(validation_pairs),
                "noncollapsed": noncollapsed,
                "close_initial_state_matches": close_matches,
            },
            indent=2,
        )
    )
    return 0 if pass_gate else 1


if __name__ == "__main__":
    raise SystemExit(main())
