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
OFFICIAL_SEMANTICS_SMOKE_IDENTITIES = ((2, 11), (6, 11))
OFFICIAL_SEMANTICS_SMOKE_STEPS = 80
RAW_ACTION_HARD_DIAGNOSTIC_CAP = 2.0
PRIOR_INSTABILITY_MAX_EXCEEDANCE_MARGIN = 0.05
PRIOR_INSTABILITY_EXCEEDANCE_FRACTION_MARGIN = 0.02
PRIOR_INSTABILITY_NATIVE_CLIP_FRACTION_MARGIN = 0.10

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
    "CORRECTED_A2C2_NO_REPEATABLE_DELAY_GAP",
    "CORRECTED_A2C2_EVALUATION_INVALID",
    "CORRECTED_A2C2_IMPLEMENTATION_OR_RESOURCE_FAILURE",
)

OFFICIAL_SEMANTICS_SMOKE_DECISIONS = (
    "CORRECTED_A2C2_OFFICIAL_SEMANTICS_SMOKE_PASS",
    "CORRECTED_A2C2_NONFINITE_ACTION_FAILURE",
    "CORRECTED_A2C2_CONTROLLER_REJECTION",
    "CORRECTED_A2C2_PRIOR_SPECIFIC_ACTION_INSTABILITY",
    "CORRECTED_A2C2_ACTION_SEMANTICS_INVALID",
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


def summarize_action_path(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize passive raw/controller telemetry without modifying actions."""

    if not records:
        return {"valid": False, "errors": ["no native action records"], "step_count": 0}
    raw = np.asarray([record["raw_action"] for record in records], dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != 7:
        return {"valid": False, "errors": [f"raw action shape {raw.shape} is not Nx7"], "step_count": len(records)}
    absolute = np.abs(raw)
    exceed = absolute > 1.0
    exceedance_events = []
    for record, flags in zip(records, exceed, strict=True):
        dimensions = np.flatnonzero(flags).tolist()
        if dimensions:
            exceedance_events.append(
                {
                    "condition": record.get("condition"),
                    "task_id": int(record["task_id"]),
                    "official_init_state_id": int(record["official_init_state_id"]),
                    "step": int(record["step"]),
                    "source_chunk_index": int(record["source_chunk_index"]),
                    "source_action_offset": int(record["source_action_offset"]),
                    "dimensions": dimensions,
                    "raw_action": [round(float(value), 9) for value in record["raw_action"]],
                }
            )

    arm_effective = np.asarray([record["arm_effective"] for record in records], dtype=np.float64)
    gripper_effective = np.asarray([record["gripper_effective"] for record in records], dtype=np.float64)
    gripper_actuator = np.asarray([record["gripper_actuator"] for record in records], dtype=np.float64)
    torques = np.asarray([record["torques"] for record in records], dtype=np.float64)
    arm_low = np.asarray(records[0]["arm_output_low"], dtype=np.float64)
    arm_high = np.asarray(records[0]["arm_output_high"], dtype=np.float64)
    actuator_low = np.asarray(records[0]["gripper_actuator_low"], dtype=np.float64)
    actuator_high = np.asarray(records[0]["gripper_actuator_high"], dtype=np.float64)
    torque_low = np.asarray(records[0]["torque_low"], dtype=np.float64)
    torque_high = np.asarray(records[0]["torque_high"], dtype=np.float64)
    tolerance = 1e-7
    finite = bool(
        np.isfinite(raw).all()
        and np.isfinite(arm_effective).all()
        and np.isfinite(gripper_effective).all()
        and np.isfinite(gripper_actuator).all()
        and np.isfinite(torques).all()
    )
    arm_in_bounds = bool(np.all(arm_effective >= arm_low - tolerance) and np.all(arm_effective <= arm_high + tolerance))
    gripper_in_bounds = bool(np.max(np.abs(gripper_effective)) <= 1.0 + tolerance)
    actuator_in_bounds = bool(
        np.all(gripper_actuator >= actuator_low - tolerance)
        and np.all(gripper_actuator <= actuator_high + tolerance)
    )
    torque_in_bounds = bool(np.all(torques >= torque_low - tolerance) and np.all(torques <= torque_high + tolerance))
    simulator_state_finite = all(bool(record["simulator_state_finite"]) for record in records)
    controller_accepted = all(bool(record["controller_accepted"]) for record in records)
    raw_hard_cap_pass = bool(np.max(absolute) <= RAW_ACTION_HARD_DIAGNOSTIC_CAP)
    errors = []
    for passed, message in (
        (finite, "nonfinite raw or effective action"),
        (arm_in_bounds, "native arm effective action outside controller output bounds"),
        (gripper_in_bounds, "native gripper effective value outside [-1,1]"),
        (actuator_in_bounds, "native gripper actuator command outside actuator control range"),
        (torque_in_bounds, "native torque outside robot torque limits"),
        (simulator_state_finite, "nonfinite simulator state"),
        (controller_accepted, "controller rejected an action"),
        (raw_hard_cap_pass, "raw action exceeded the frozen 2x-nominal diagnostic cap"),
    ):
        if not passed:
            errors.append(message)
    step_count = int(len(records))
    element_count = int(raw.size)
    return {
        "valid": not errors,
        "errors": errors,
        "step_count": step_count,
        "raw_action_dimension": 7,
        "raw_action_finite": bool(np.isfinite(raw).all()),
        "raw_max_abs_by_dimension": [round(float(value), 9) for value in absolute.max(axis=0)],
        "raw_p99_abs_by_dimension": [round(float(value), 9) for value in np.quantile(absolute, 0.99, axis=0)],
        "raw_max_abs": round(float(absolute.max()), 9),
        "raw_p99_abs": round(float(np.quantile(absolute, 0.99)), 9),
        "raw_abs_values_by_dimension": [
            [round(float(value), 9) for value in absolute[:, dimension]] for dimension in range(7)
        ],
        "above_nominal_count_by_dimension": [int(value) for value in exceed.sum(axis=0)],
        "above_nominal_fraction_by_dimension": [round(float(value), 9) for value in exceed.mean(axis=0)],
        "above_nominal_element_count": int(exceed.sum()),
        "above_nominal_element_fraction": round(float(exceed.sum() / element_count), 9),
        "steps_with_any_nominal_exceedance": int(exceed.any(axis=1).sum()),
        "steps_with_any_nominal_exceedance_fraction": round(float(exceed.any(axis=1).mean()), 9),
        "max_exceedance_by_dimension": [round(float(max(value - 1.0, 0.0)), 9) for value in absolute.max(axis=0)],
        "max_exceedance_magnitude": round(float(max(absolute.max() - 1.0, 0.0)), 9),
        "raw_hard_diagnostic_cap": RAW_ACTION_HARD_DIAGNOSTIC_CAP,
        "raw_hard_diagnostic_cap_pass": raw_hard_cap_pass,
        "exceedance_events": exceedance_events,
        "native_arm_clip_step_count": sum(bool(record["arm_input_clipped"]) for record in records),
        "native_arm_clip_step_fraction": round(
            float(sum(bool(record["arm_input_clipped"]) for record in records) / step_count), 9
        ),
        "native_gripper_saturation_call_count": sum(int(record["gripper_saturation_calls"]) for record in records),
        "native_gripper_saturation_step_count": sum(
            int(record["gripper_saturation_calls"]) > 0 for record in records
        ),
        "arm_effective_max_abs_by_dimension": [round(float(value), 9) for value in np.abs(arm_effective).max(axis=0)],
        "arm_effective_p99_abs_by_dimension": [
            round(float(value), 9) for value in np.quantile(np.abs(arm_effective), 0.99, axis=0)
        ],
        "arm_output_low": arm_low.tolist(),
        "arm_output_high": arm_high.tolist(),
        "arm_effective_within_bounds": arm_in_bounds,
        "gripper_raw_max_abs": round(float(absolute[:, 6].max()), 9),
        "gripper_raw_p99_abs": round(float(np.quantile(absolute[:, 6], 0.99)), 9),
        "gripper_effective_max_abs": round(float(np.abs(gripper_effective).max()), 9),
        "gripper_effective_within_bounds": gripper_in_bounds,
        "gripper_actuator_max_abs": round(float(np.abs(gripper_actuator).max()), 9),
        "gripper_actuator_within_control_range": actuator_in_bounds,
        "torque_max_abs_by_dimension": [round(float(value), 9) for value in np.abs(torques).max(axis=0)],
        "torque_within_limits": torque_in_bounds,
        "simulator_state_finite": simulator_state_finite,
        "controller_accepted_every_action": controller_accepted,
    }


def adjudicate_official_semantics_smoke(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected = {
        (condition, task_id, init_state_id)
        for condition in ("BASE_DELAYED_E40_D10", "PRIOR_DELAYED_E40_D10")
        for task_id, init_state_id in OFFICIAL_SEMANTICS_SMOKE_IDENTITIES
    }
    keys = {episode_key(row) for row in rows}
    if any(not bool(row.get("action_finite")) for row in rows):
        decision = "CORRECTED_A2C2_NONFINITE_ACTION_FAILURE"
    elif any(int(row.get("controller_rejection_count", 0)) > 0 for row in rows):
        decision = "CORRECTED_A2C2_CONTROLLER_REJECTION"
    else:
        comparative = []
        for identity in OFFICIAL_SEMANTICS_SMOKE_IDENTITIES:
            base = next(
                (row for row in rows if episode_key(row) == ("BASE_DELAYED_E40_D10", *identity)), None
            )
            prior = next(
                (row for row in rows if episode_key(row) == ("PRIOR_DELAYED_E40_D10", *identity)), None
            )
            if base is None or prior is None:
                continue
            base_diag = base.get("raw_action_diagnostics", {})
            prior_diag = prior.get("raw_action_diagnostics", {})
            max_margin = float(prior_diag.get("max_exceedance_magnitude", 0.0)) - float(
                base_diag.get("max_exceedance_magnitude", 0.0)
            )
            fraction_margin = float(prior_diag.get("above_nominal_element_fraction", 0.0)) - float(
                base_diag.get("above_nominal_element_fraction", 0.0)
            )
            clip_margin = float(prior_diag.get("native_arm_clip_step_fraction", 0.0)) - float(
                base_diag.get("native_arm_clip_step_fraction", 0.0)
            )
            comparative.append(
                {
                    "task_id": identity[0],
                    "official_init_state_id": identity[1],
                    "prior_minus_base_max_exceedance": round(max_margin, 9),
                    "prior_minus_base_exceedance_fraction": round(fraction_margin, 9),
                    "prior_minus_base_native_clip_fraction": round(clip_margin, 9),
                    "substantial": bool(
                        max_margin >= PRIOR_INSTABILITY_MAX_EXCEEDANCE_MARGIN
                        and (
                            fraction_margin >= PRIOR_INSTABILITY_EXCEEDANCE_FRACTION_MARGIN
                            or clip_margin >= PRIOR_INSTABILITY_NATIVE_CLIP_FRACTION_MARGIN
                        )
                    ),
                }
            )
        reproducible_instability = bool(
            len(comparative) == len(OFFICIAL_SEMANTICS_SMOKE_IDENTITIES)
            and all(item["substantial"] for item in comparative)
        )
        structure_valid = bool(
            keys == expected
            and len(rows) == len(expected)
            and all(row.get("exception") is None for row in rows)
            and all(bool(row.get("action_semantics_valid")) for row in rows)
            and all(int(row.get("base_model_forward_count", 0)) > 0 for row in rows)
            and all(
                int(row.get("prior_module_forward_count", 0)) > 0
                for row in rows
                if row.get("condition") == "PRIOR_DELAYED_E40_D10"
            )
            and all(not bool(row.get("task_success_persisted", True)) for row in rows)
            and all(not bool(row.get("task_success_counted", True)) for row in rows)
        )
        if reproducible_instability:
            decision = "CORRECTED_A2C2_PRIOR_SPECIFIC_ACTION_INSTABILITY"
        elif not structure_valid:
            decision = "CORRECTED_A2C2_ACTION_SEMANTICS_INVALID"
        else:
            decision = "CORRECTED_A2C2_OFFICIAL_SEMANTICS_SMOKE_PASS"
        return {
            "final_decision": decision,
            "valid": decision == "CORRECTED_A2C2_OFFICIAL_SEMANTICS_SMOKE_PASS",
            "expected_keys": sorted(expected),
            "observed_keys": sorted(keys),
            "comparative_diagnostics": comparative,
            "reproducible_prior_specific_instability": reproducible_instability,
            "practical_rule": {
                "max_exceedance_margin": PRIOR_INSTABILITY_MAX_EXCEEDANCE_MARGIN,
                "exceedance_fraction_margin": PRIOR_INSTABILITY_EXCEEDANCE_FRACTION_MARGIN,
                "native_clip_fraction_margin": PRIOR_INSTABILITY_NATIVE_CLIP_FRACTION_MARGIN,
                "requires_both_development_identities": True,
            },
        }
    return {"final_decision": decision, "valid": False}


def aggregate_panel_action_diagnostics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate exact per-step raw diagnostics by frozen panel arm."""

    arms: dict[str, Any] = {}
    for condition in CONDITIONS:
        condition_rows = [row for row in rows if row.get("condition") == condition]
        by_dimension: list[list[float]] = [[] for _ in range(7)]
        exceedance_events = []
        native_arm_clip_steps = 0
        total_steps = 0
        for row in condition_rows:
            diagnostic = row.get("raw_action_diagnostics", {})
            values = diagnostic.get("raw_abs_values_by_dimension", [[] for _ in range(7)])
            for dimension in range(7):
                by_dimension[dimension].extend(float(value) for value in values[dimension])
            exceedance_events.extend(diagnostic.get("exceedance_events", []))
            native_arm_clip_steps += int(diagnostic.get("native_arm_clip_step_count", 0))
            total_steps += int(diagnostic.get("step_count", 0))
        arrays = [np.asarray(values, dtype=np.float64) for values in by_dimension]
        if any(array.size == 0 for array in arrays):
            arms[condition] = {"valid": False, "errors": ["missing raw action values"]}
            continue
        counts = [int(np.sum(array > 1.0)) for array in arrays]
        element_count = sum(int(array.size) for array in arrays)
        arms[condition] = {
            "valid": True,
            "episode_count": len(condition_rows),
            "step_count": total_steps,
            "raw_max_abs_by_dimension": [round(float(array.max()), 9) for array in arrays],
            "raw_p99_abs_by_dimension": [round(float(np.quantile(array, 0.99)), 9) for array in arrays],
            "above_nominal_count_by_dimension": counts,
            "above_nominal_fraction_by_dimension": [
                round(float(count / arrays[index].size), 9) for index, count in enumerate(counts)
            ],
            "above_nominal_element_count": sum(counts),
            "above_nominal_element_fraction": round(float(sum(counts) / element_count), 9),
            "max_exceedance_by_dimension": [round(float(max(array.max() - 1.0, 0.0)), 9) for array in arrays],
            "max_exceedance_magnitude": round(float(max(max(array.max() for array in arrays) - 1.0, 0.0)), 9),
            "native_arm_clip_step_count": native_arm_clip_steps,
            "native_arm_clip_step_fraction": round(float(native_arm_clip_steps / max(total_steps, 1)), 9),
            "gripper_raw_max_abs": round(float(arrays[6].max()), 9),
            "gripper_raw_p99_abs": round(float(np.quantile(arrays[6], 0.99)), 9),
            "all_action_semantics_valid": all(bool(row.get("action_semantics_valid")) for row in condition_rows),
            "exceedance_events": exceedance_events,
        }
    paired_prior_minus_delayed_base = []
    delayed = {
        (int(row["task_id"]), int(row["official_init_state_id"])): row
        for row in rows
        if row.get("condition") == "BASE_DELAYED_E40_D10"
    }
    prior = {
        (int(row["task_id"]), int(row["official_init_state_id"])): row
        for row in rows
        if row.get("condition") == "PRIOR_DELAYED_E40_D10"
    }
    for identity in sorted(set(delayed) & set(prior)):
        base_diag = delayed[identity]["raw_action_diagnostics"]
        prior_diag = prior[identity]["raw_action_diagnostics"]
        paired_prior_minus_delayed_base.append(
            {
                "task_id": identity[0],
                "official_init_state_id": identity[1],
                "max_exceedance_magnitude_delta": round(
                    float(prior_diag["max_exceedance_magnitude"])
                    - float(base_diag["max_exceedance_magnitude"]),
                    9,
                ),
                "above_nominal_element_fraction_delta": round(
                    float(prior_diag["above_nominal_element_fraction"])
                    - float(base_diag["above_nominal_element_fraction"]),
                    9,
                ),
                "native_arm_clip_step_fraction_delta": round(
                    float(prior_diag["native_arm_clip_step_fraction"])
                    - float(base_diag["native_arm_clip_step_fraction"]),
                    9,
                ),
            }
        )
    return {"arms": arms, "paired_prior_minus_delayed_base": paired_prior_minus_delayed_base}


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
    all_action_semantics_valid = all(bool(row.get("action_semantics_valid")) for row in rows)
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
        and all_action_semantics_valid
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

    if not manifest_valid:
        decision = "CORRECTED_A2C2_EVALUATION_INVALID"
    elif not base_competent:
        decision = "CORRECTED_A2C2_BASE_NOT_COMPETENT"
    elif not repeatable_gap:
        decision = "CORRECTED_A2C2_NO_REPEATABLE_DELAY_GAP"
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
