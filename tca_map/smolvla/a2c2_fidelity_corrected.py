"""Frozen helpers for the one corrected A2C2 local-port verification.

This module contains no training path and no Ours method.  It isolates the
outcome-independent queue, preprocessing, artifact, identity, and decision
contracts so they can be tested without importing the large VLA runtime.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


FIDELITY_LABEL = "A2C2_FIDELITY_CORRECTED_LOCAL_PORT"
OFFICIAL_COMMIT = "54dd088302a0ef3f50c4add3ec927ab94d76a406"
# The public prior was uploaded before the author's next architecture commit.
# Its 2-D image projection strict-loads against this immediately preceding
# author commit; later source constructs a 4-D 1x1 convolution instead.
CHECKPOINT_COMPATIBLE_COMMIT = "c197a011aabf070cf2c0b2b0705be5f33d178ad7"
BASE_REVISION = "caa0efcb24e261574c824366526c5775d3664cac"
BASE_MODEL_SHA256 = "45F3B6FC1B8AE0B7CF3AB8EBD22336AB23EB3798A8BFEF027F5D45596C45A9BE"
PRIOR_REVISION = "9c89cca4aae8eecc42a20084ef414ff74f94ba05"
PRIOR_MODEL_SHA256 = "85D00523E8273A4141E288E4F6692224D50AAF8DF99AD8CCF7E72EE7BF3AB712"
CHUNK_SIZE = 50
ROOT_SEED = 7
EVAL_TASK_IDS = (0, 4, 8)
OLD_V1_INIT_STATE_IDS = (0, 1, 2, 3, 4)
VERIFICATION_INIT_STATE_IDS = (5, 6, 7, 8, 9)
DEVELOPMENT_SMOKE_IDENTITY = (2, 10)

CONDITIONS: dict[str, dict[str, int | bool]] = {
    "BASE_STANDARD_E10_D0": {"execution_horizon": 10, "inference_delay": 0, "with_prior": False},
    "BASE_DELAYED_E40_D10": {"execution_horizon": 40, "inference_delay": 10, "with_prior": False},
    "PRIOR_DELAYED_E40_D10": {"execution_horizon": 40, "inference_delay": 10, "with_prior": True},
}

ALLOWED_FINAL_DECISIONS = (
    "CORRECTED_A2C2_PRIOR_IMPROVES_AND_LEAVES_RESIDUAL",
    "CORRECTED_A2C2_PRIOR_SATURATES_DELAY",
    "CORRECTED_A2C2_PRIOR_NO_IMPROVEMENT",
    "CORRECTED_A2C2_BASE_NOT_COMPETENT",
    "CORRECTED_A2C2_EVALUATION_INVALID",
    "CORRECTED_A2C2_IMPLEMENTATION_OR_RESOURCE_FAILURE",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def rotate_live_rgb_180(image: np.ndarray) -> np.ndarray:
    """Apply the exact official evaluator's H/W reversal and make it contiguous."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"expected HxWx3 RGB image, got {array.shape}")
    return np.ascontiguousarray(array[::-1, ::-1])


def phase_feature(offset: int, chunk_size: int = CHUNK_SIZE) -> np.ndarray:
    """Return the released-code sin/cos phase using its H-1 denominator."""

    if not 0 <= int(offset) < int(chunk_size):
        raise ValueError(f"offset {offset} outside chunk size {chunk_size}")
    denominator = max(int(chunk_size) - 1, 1)
    phase = 2.0 * math.pi * float(offset) / float(denominator)
    return np.asarray([math.sin(phase), math.cos(phase)], dtype=np.float32)


def noise_seed(task_id: int, init_state_id: int, chunk_index: int, root_seed: int = ROOT_SEED) -> int:
    payload = f"{root_seed}:{task_id}:{init_state_id}:{chunk_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def refresh_action_plan(
    *,
    new_entries: Sequence[Any],
    pending_entries: Sequence[Any],
    execution_horizon: int,
    inference_delay: int,
    first_chunk: bool,
) -> tuple[list[Any], list[Any]]:
    """Pure official A2C2 queue refresh used by both runner and tests."""

    chunk = list(new_entries)
    pending = list(pending_entries)
    e = int(execution_horizon)
    d = int(inference_delay)
    if len(chunk) != CHUNK_SIZE:
        raise ValueError(f"expected {CHUNK_SIZE} new entries, got {len(chunk)}")
    if d < 0 or e < d or e + d > CHUNK_SIZE:
        raise ValueError(f"invalid A2C2 queue condition e={e}, d={d}, H={CHUNK_SIZE}")

    if first_chunk:
        plan = chunk[:e]
    else:
        if len(pending) < d:
            raise RuntimeError(f"pending queue underflow: have {len(pending)}, need {d}")
        plan = pending[:d] + chunk[d:e]
        pending = pending[d:]
    pending.extend(chunk[e : e + d])
    return plan, pending


def episode_key(row: Mapping[str, Any]) -> tuple[str, int, int]:
    return str(row["condition"]), int(row["task_id"]), int(row["official_init_state_id"])


def expected_panel_keys() -> set[tuple[str, int, int]]:
    return {
        (condition, task_id, init_state_id)
        for condition in CONDITIONS
        for task_id in EVAL_TASK_IDS
        for init_state_id in VERIFICATION_INIT_STATE_IDS
    }


def verify_artifact_configs(base_root: Path, prior_root: Path) -> dict[str, Any]:
    """Verify frozen artifact identity/config without loading either large model."""

    base_model = base_root / "model.safetensors"
    prior_model = prior_root / "model.safetensors"
    base_config = json.loads((base_root / "config.json").read_text(encoding="utf-8"))
    prior_config = json.loads((prior_root / "config.json").read_text(encoding="utf-8"))
    errors: list[str] = []
    if sha256_file(base_model) != BASE_MODEL_SHA256:
        errors.append("base model SHA256 mismatch")
    if sha256_file(prior_model) != PRIOR_MODEL_SHA256:
        errors.append("prior model SHA256 mismatch")
    if int(base_config.get("chunk_size", -1)) != CHUNK_SIZE:
        errors.append("base chunk_size mismatch")
    if base_config.get("input_features", {}).get("observation.state", {}).get("shape") != [8]:
        errors.append("base state feature mismatch")
    if base_config.get("output_features", {}).get("action", {}).get("shape") != [7]:
        errors.append("base action feature mismatch")
    if int(prior_config.get("n_encoder_layers", -1)) != 6:
        errors.append("prior encoder layer mismatch")
    if prior_config.get("input_features", {}).get("vlm_hidden", {}).get("shape") != [960]:
        errors.append("prior vlm_hidden mismatch")
    if bool(prior_config.get("freeze_vision_backbone", True)):
        errors.append("prior freeze_vision_backbone mismatch")
    if prior_config.get("input_features", {}).get("observation.state", {}).get("shape") != [8]:
        errors.append("prior state feature mismatch")
    if prior_config.get("output_features", {}).get("action", {}).get("shape") != [7]:
        errors.append("prior action feature mismatch")
    return {
        "base_root": str(base_root),
        "prior_root": str(prior_root),
        "base_model_sha256": sha256_file(base_model),
        "prior_model_sha256": sha256_file(prior_model),
        "base_config": base_config,
        "prior_config": prior_config,
        "errors": errors,
        "valid": not errors,
    }


def _rows_by_condition(rows: Sequence[Mapping[str, Any]], condition: str) -> list[Mapping[str, Any]]:
    return [row for row in rows if str(row.get("condition")) == condition]


def adjudicate_panel(
    rows: Sequence[Mapping[str, Any]],
    *,
    infrastructure_failure: bool = False,
) -> dict[str, Any]:
    """Apply the hash-frozen corrected decision rules to completed rows."""

    if infrastructure_failure:
        return {
            "final_decision": "CORRECTED_A2C2_IMPLEMENTATION_OR_RESOURCE_FAILURE",
            "valid": False,
        }

    keys = [episode_key(row) for row in rows]
    expected = expected_panel_keys()
    unique = set(keys)
    arms = {condition: _rows_by_condition(rows, condition) for condition in CONDITIONS}
    all_actions_finite = all(bool(row.get("action_finite")) for row in rows)
    all_actions_legal = all(bool(row.get("action_legal")) for row in rows)
    no_exceptions = all(row.get("exception") is None for row in rows)
    base_forwards = all(int(row.get("base_model_forward_count", 0)) > 0 for row in rows)
    prior_forwards = all(
        int(row.get("prior_module_forward_count", 0)) > 0
        for row in arms["PRIOR_DELAYED_E40_D10"]
    )
    prior_correction = any(
        float(row.get("prior_mean_abs_correction", 0.0)) > 0.0
        for row in arms["PRIOR_DELAYED_E40_D10"]
    )
    manifest_valid = bool(
        len(rows) == len(expected) == 45
        and len(keys) == len(unique)
        and unique == expected
        and all_actions_finite
        and all_actions_legal
        and no_exceptions
        and base_forwards
        and prior_forwards
        and prior_correction
    )

    successes = {
        condition: sum(bool(row.get("success")) for row in condition_rows)
        for condition, condition_rows in arms.items()
    }
    task_clean = {
        task_id: sum(
            bool(row.get("success"))
            for row in arms["BASE_STANDARD_E10_D0"]
            if int(row["task_id"]) == task_id
        )
        for task_id in EVAL_TASK_IDS
    }
    clean_map = {
        (int(row["task_id"]), int(row["official_init_state_id"])): bool(row["success"])
        for row in arms["BASE_STANDARD_E10_D0"]
    }
    delayed_map = {
        (int(row["task_id"]), int(row["official_init_state_id"])): bool(row["success"])
        for row in arms["BASE_DELAYED_E40_D10"]
    }
    prior_map = {
        (int(row["task_id"]), int(row["official_init_state_id"])): bool(row["success"])
        for row in arms["PRIOR_DELAYED_E40_D10"]
    }
    clean_to_delayed = [key for key in clean_map if clean_map[key] and not delayed_map.get(key, False)]
    recoveries = [key for key in delayed_map if not delayed_map[key] and prior_map.get(key, False)]
    regressions = [key for key in delayed_map if delayed_map[key] and not prior_map.get(key, False)]
    residuals = [key for key in clean_map if clean_map[key] and not prior_map.get(key, False)]

    base_competent = bool(successes["BASE_STANDARD_E10_D0"] >= 8 and all(task_clean.values()))
    repeatable_gap = bool(
        successes["BASE_STANDARD_E10_D0"] - successes["BASE_DELAYED_E40_D10"] >= 3
        and len(clean_to_delayed) >= 3
        and len({key[0] for key in clean_to_delayed}) >= 2
    )
    improves = bool(
        successes["PRIOR_DELAYED_E40_D10"] - successes["BASE_DELAYED_E40_D10"] >= 2
        and len(recoveries) >= 2
        and len(regressions) <= 1
        and prior_correction
    )
    saturates = bool(
        improves
        and (
            successes["PRIOR_DELAYED_E40_D10"] >= successes["BASE_STANDARD_E10_D0"] - 1
            or len(residuals) <= 1
        )
    )
    residual_remains = bool(
        improves
        and successes["BASE_STANDARD_E10_D0"] - successes["PRIOR_DELAYED_E40_D10"] >= 2
        and len(residuals) >= 2
        and len({key[0] for key in residuals}) >= 2
    )

    if not manifest_valid or not repeatable_gap:
        decision = "CORRECTED_A2C2_EVALUATION_INVALID"
    elif not base_competent:
        decision = "CORRECTED_A2C2_BASE_NOT_COMPETENT"
    elif saturates:
        decision = "CORRECTED_A2C2_PRIOR_SATURATES_DELAY"
    elif improves and residual_remains:
        decision = "CORRECTED_A2C2_PRIOR_IMPROVES_AND_LEAVES_RESIDUAL"
    else:
        decision = "CORRECTED_A2C2_PRIOR_NO_IMPROVEMENT"

    return {
        "final_decision": decision,
        "valid": manifest_valid and repeatable_gap and base_competent,
        "successes": successes,
        "task_clean_successes": task_clean,
        "paired_counts": {
            "clean_to_delayed_failures": len(clean_to_delayed),
            "prior_recoveries": len(recoveries),
            "prior_regressions": len(regressions),
            "clean_to_prior_residuals": len(residuals),
        },
        "paired_identities": {
            "clean_to_delayed_failures": clean_to_delayed,
            "prior_recoveries": recoveries,
            "prior_regressions": regressions,
            "clean_to_prior_residuals": residuals,
        },
        "gates": {
            "manifest_valid": manifest_valid,
            "base_competent": base_competent,
            "repeatable_delay_gap": repeatable_gap,
            "prior_improves": improves,
            "prior_saturates": saturates,
            "residual_remains": residual_remains,
        },
    }
