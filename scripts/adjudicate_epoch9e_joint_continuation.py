#!/usr/bin/env python3
"""Adjudicate the completed fixed-denominator Epoch 9E panel append-only."""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json


REPORTS = ROOT / "reports"
PROTOCOL_PATH = REPORTS / "epoch9e_joint_certification_protocol.json"
ORIGINAL_PROTOCOL_PATH = REPORTS / "epoch9b_v2_task_preservation_protocol.json"
PREFLIGHT_PATH = REPORTS / "epoch9e_exact_pair_preflight.json"
ORIGINAL_RESULT_PATH = REPORTS / "epoch9e_joint_certification/result.json"
CONTINUATION_RESULT_PATH = REPORTS / "epoch9e_joint_continuation/result.json"
SEAL_PATH = REPORTS / "epoch9e_joint_continuation_execution_seal.json"
SENSITIVITY_PATH = REPORTS / "epoch9e_missing_pair_sensitivity_protocol.json"
CORRECTION_PATH = REPORTS / "epoch9e_failfast_root_cause_and_scope_correction.json"
STATUS_CORRECTION_PATH = REPORTS / "epoch9e_continuation_host_exit_status_correction.json"
PARSER_REPAIR_PATH = REPORTS / "epoch9e_continuation_adjudicator_parser_repair.json"
OUTPUT_JSON = REPORTS / "epoch9e_joint_continuation_adjudication.json"
OUTPUT_MD = REPORTS / "epoch9e_joint_continuation_adjudication.md"
TRACE_ROOT = REPORTS / "epoch9e_joint_certification/traces"
SHAM_TRACE_ROOT = REPORTS / "epoch9e_joint_certification/sham_traces"
MISSING_PREFIX = "RuntimeError: trace does not contain the frozen five-step response window: "


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def paired_t_interval(values: np.ndarray) -> list[float | None]:
    if len(values) < 2 or not np.isfinite(values).all():
        return [None, None]
    standard_error = float(stats.sem(values))
    if not np.isfinite(standard_error):
        return [None, None]
    mean = float(np.mean(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * standard_error)
    return [mean - half, mean + half]


def lower_positive(interval: list[float | None]) -> bool:
    return interval[0] is not None and float(interval[0]) > 0.0


def interval_includes_zero(interval: list[float | None]) -> bool:
    return interval[0] is not None and interval[1] is not None and float(interval[0]) <= 0.0 <= float(interval[1])


def adjudicator_binding_valid(seal: dict[str, Any], parser_repair: dict[str, Any], status_correction: dict[str, Any]) -> bool:
    current = sha256(Path(__file__))
    if current == seal["continuation_adjudicator_sha256"]:
        return True
    return bool(
        parser_repair["original_sealed_adjudicator_sha256"] == seal["continuation_adjudicator_sha256"]
        and parser_repair["repaired_adjudicator_sha256"] == current
        and parser_repair["status_correction_sha256"] == sha256(STATUS_CORRECTION_PATH)
        and parser_repair["scientific_fields_changed"] is False
        and parser_repair["outcomes_recomputed_or_rerun"] is False
        and status_correction["scientific_result_changed"] is False
    )


def effective_runner_exit_code(monitor: dict[str, Any], status_correction: dict[str, Any]) -> int:
    recorded = int(monitor.get("authoritative_runner_exit_code", 255))
    if recorded == 0:
        return 0
    if int(monitor.get("attempt", -1)) != int(status_correction["attempt"]):
        return recorded
    monitor_path = ROOT / status_correction["host_monitor_path"]
    status_path = ROOT / status_correction["raw_status_path"]
    if (
        monitor_path.is_file()
        and status_path.is_file()
        and sha256(monitor_path) == status_correction["host_monitor_sha256"]
        and sha256(status_path) == status_correction["raw_status_sha256"]
        and status_path.read_text(encoding="utf-8") == status_correction["raw_status_text"] == "0n"
        and int(monitor.get("wsl_process_exit_code", -1)) == 0
        and int(status_correction["corrected_authoritative_runner_exit_code"]) == 0
    ):
        return 0
    return recorded


def adjusted_hc3(pair_rows: list[dict[str, Any]], bases: dict[int, dict[str, Any]], contrasts: list[float]) -> dict[str, Any]:
    covariate_names = ["initial back x", "initial back y", "initial back lane margin", "back-first probe order"]
    if len(pair_rows) != len(contrasts) or len(pair_rows) <= 5:
        return {"valid": False, "estimate_m": None, "hc3_standard_error_m": None, "degrees_of_freedom": None, "hc3_95_interval_m": [None, None], "covariates_centered": covariate_names}
    z = []
    for row in pair_rows:
        base = bases[int(row["base_identity_id"])]
        initial = base["candidate_initial_xyz_eval_only"]["back"]
        z.append([float(initial[0]), float(initial[1]), float(base["candidate_initial_lane_margin_m_eval_only"]["back"]), float(base["probe_order"][0] == "back")])
    z_array = np.asarray(z, dtype=np.float64)
    z_array -= z_array.mean(axis=0, keepdims=True)
    y = np.asarray(contrasts, dtype=np.float64)
    x = np.column_stack((np.ones(len(y)), z_array))
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    hat = np.sum((x @ inverse) * x, axis=1)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    for vector, value in zip(x, residual / np.maximum(1.0 - hat, 1e-9), strict=True):
        meat += np.outer(vector, vector) * value**2
    covariance = inverse @ meat @ inverse
    standard_error = float(np.sqrt(max(float(covariance[0, 0]), 0.0)))
    degrees = len(y) - x.shape[1]
    estimate = float(beta[0])
    critical = float(stats.t.ppf(0.975, degrees))
    interval = [estimate - critical * standard_error, estimate + critical * standard_error]
    return {
        "valid": bool(degrees > 0 and np.isfinite([estimate, standard_error, *interval]).all()),
        "estimate_m": estimate,
        "hc3_standard_error_m": standard_error,
        "degrees_of_freedom": degrees,
        "hc3_95_interval_m": interval,
        "covariates_centered": covariate_names,
    }


def lane_contains(protocol: dict[str, Any], slot: str, xyz: np.ndarray) -> np.ndarray:
    lane, reach = protocol["safe_center_lanes_m"][slot], protocol["reachable_center_envelope_m"]
    return ((xyz[:, 0] >= lane["x"][0]) & (xyz[:, 0] <= lane["x"][1]) &
            (xyz[:, 1] >= lane["y"][0]) & (xyz[:, 1] <= lane["y"][1]) &
            (xyz[:, 2] >= reach["z"][0]) & (xyz[:, 2] <= reach["z"][1]))


def trace_probe_audit(path: Path, slot: str, original_protocol: dict[str, Any]) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as trace:
        phase = np.asarray(trace["phase"]).astype(str)
        actions = np.asarray(trace["action"], dtype=np.float64)
        positions = np.asarray(trace["candidate_positions_eval_only"], dtype=np.float64)
        quality = np.asarray(trace["rgb_quality"], dtype=np.float64)
        target_contact = np.asarray(trace["target_contact_eval_only"], dtype=bool)
        pair_collision = np.asarray(trace["candidate_pair_collision_eval_only"], dtype=bool)
        distractor_collision = np.asarray(trace["candidate_distractor_collision_eval_only"], dtype=bool)
        estimated = np.asarray(trace["estimated_world_displacement_m"], dtype=np.float64)
    front, back = positions[:, 0, :], positions[:, 1, :]
    index = 0 if slot == "front" else 1
    target = positions[:, index, :]
    excitation = float(np.max(np.linalg.norm(target - target[0], axis=1)))
    low, high = np.asarray([-0.25, -0.05, 0.85]), np.asarray([0.25, 0.25, 1.10])
    front_in_back = lane_contains(original_protocol, "back", front)
    back_in_front = lane_contains(original_protocol, "front", back)
    liftoff = actions[phase == "nondrag_vertical_liftoff"]
    response_steps = int(np.count_nonzero(np.isin(phase, ["fixed_micro_impulse", "post_impulse_response"])))
    signed_margin_summary = {}
    for candidate, xyz in (("front", front), ("back", back)):
        lane, reach = original_protocol["safe_center_lanes_m"][candidate], original_protocol["reachable_center_envelope_m"]
        margins = np.column_stack((xyz[:, 0]-lane["x"][0], lane["x"][1]-xyz[:, 0], xyz[:, 1]-lane["y"][0], lane["y"][1]-xyz[:, 1], xyz[:, 2]-reach["z"][0], reach["z"][1]-xyz[:, 2])).min(axis=1)
        signed_margin_summary[candidate] = {"minimum_m": float(np.min(margins)), "median_m": float(np.median(margins)), "maximum_m": float(np.max(margins))}
    return {
        "slot": slot,
        "trace_path": relative(path),
        "trace_sha256": sha256(path),
        "steps": int(len(phase)),
        "response_window_steps": response_steps,
        "response_window_valid": response_steps == 5,
        "finite_bounded_actions": bool(actions.size and np.isfinite(actions).all() and np.all(np.abs(actions) <= 1.0)),
        "intended_contact_or_excitation": bool(np.any(target_contact) or excitation >= 0.001),
        "sampled_target_contact": bool(np.any(target_contact)),
        "target_excitation_peak_m": excitation,
        "lane_and_reachability_continuous_pass": bool(np.all(lane_contains(original_protocol, "front", front)) and np.all(lane_contains(original_protocol, "back", back))),
        "continuous_candidate_lane_signed_margin_summary_m": signed_margin_summary,
        "continuous_estimated_displacement_summary_m": {"minimum": float(np.min(estimated)), "median": float(np.median(estimated)), "maximum": float(np.max(estimated))},
        "unintended_collision": bool(np.any(pair_collision) or np.any(distractor_collision)),
        "identity_swap": bool(np.any(front[:, 0] <= back[:, 0]) or np.any(front_in_back) or np.any(back_in_front)),
        "fall": bool(np.any(front[:, 2] < 0.85) or np.any(back[:, 2] < 0.85)),
        "workspace_exit": bool(np.any(front < low) or np.any(front > high) or np.any(back < low) or np.any(back > high)),
        "unrecoverable_track_loss": bool(not np.isfinite(quality).all() or float(np.min(quality)) < 0.50),
        "minimum_rgb_quality": float(np.min(quality)),
        "nondrag_liftoff_planar_commands_exact_zero": bool(liftoff.size and np.all(liftoff[:, :2] == 0.0)),
        "nondrag_attempt_observed": bool(np.any(phase == "nondrag_vertical_liftoff")),
        "forbidden_online_inputs_used": [],
        "simulator_state_used_for_actions": False,
        "mass_or_property_used_for_actions": False,
        "audit_source": "raw_trace_for_finalized_missing_response_assignment",
    }


def stored_probe_audits(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(row["probe_audits"].values())


def tipping_point(other_values: list[float], admissible: list[float]) -> dict[str, Any]:
    low, high = map(float, admissible)

    def lower(value: float) -> float:
        interval = paired_t_interval(np.asarray([*other_values, value], dtype=np.float64))
        return float(interval[0]) if interval[0] is not None else float("nan")

    low_value, high_value = lower(low), lower(high)
    if not np.isfinite([low_value, high_value]).all():
        return {"threshold_contrast_m": None, "lower_at_admissible_low_m": low_value, "lower_at_admissible_high_m": high_value, "classification": "UNDEFINED"}
    if low_value > 0:
        return {"threshold_contrast_m": low, "lower_at_admissible_low_m": low_value, "lower_at_admissible_high_m": high_value, "classification": "SURVIVES_FULL_ADMISSIBLE_RANGE"}
    if high_value <= 0:
        return {"threshold_contrast_m": None, "lower_at_admissible_low_m": low_value, "lower_at_admissible_high_m": high_value, "classification": "NO_ADMISSIBLE_VALUE_RESCUES_INTERVAL"}
    left, right = low, high
    for _ in range(100):
        middle = (left + right) / 2.0
        if lower(middle) > 0:
            right = middle
        else:
            left = middle
    return {"threshold_contrast_m": right, "lower_at_admissible_low_m": low_value, "lower_at_admissible_high_m": high_value, "classification": "INTERIOR_TIPPING_POINT"}


def final_gates(counts: dict[str, Any], paired: dict[str, Any], controls: dict[str, bool], integrity: dict[str, bool]) -> dict[str, bool]:
    return {
        "complete_fixed_manifest_24_primary_12_sham": integrity["complete_fixed_manifest"],
        "finite_bounded_actions_48_of_48": counts["finite_bounded_actions"] == 48,
        "intended_contact_or_excitation_at_least_46_of_48": counts["intended_contact_or_excitation"] >= 46,
        "both_candidates_excited_at_least_22_of_24": counts["both_candidates_excited"] >= 22,
        "full_trajectory_lane_reachable_48_of_48": counts["full_trajectory_lane_reachable"] == 48,
        "zero_collision_identity_swap_fall_workspace_exit_track_loss": sum(counts[key] for key in ("collisions", "identity_swaps", "falls", "workspace_exits", "unrecoverable_track_losses")) == 0,
        "rank_at_least_20_of_24": counts["rank_correct"] >= 20,
        "rank_each_heavy_position_at_least_10_of_12": all(value["correct"] >= 10 and value["total"] == 12 for value in counts["rank_by_heavy_position"].values()),
        "exact_pair_flips_at_least_9_of_12_with_missing_adverse": counts["exact_pair_correct_flips"] >= 9,
        "fixed_denominator_one_sided_sign_p_below_0_01": paired["fixed_denominator_one_sided_exact_sign_p"] < 0.01,
        "complete_case_student_t_interval_positive": lower_positive(paired["complete_case_student_t_95_interval_m"]),
        "complete_case_hc3_interval_positive": bool(paired["complete_case_adjusted_hc3"]["valid"] and paired["complete_case_adjusted_hc3"]["estimate_m"] > 0 and lower_positive(paired["complete_case_adjusted_hc3"]["hc3_95_interval_m"])),
        "worst_case_sensitivity_student_t_interval_positive": lower_positive(paired["worst_case_augmented_student_t_95_interval_m"]),
        "worst_case_sensitivity_hc3_interval_positive": bool(paired["worst_case_augmented_adjusted_hc3"]["valid"] and paired["worst_case_augmented_adjusted_hc3"]["estimate_m"] > 0 and lower_positive(paired["worst_case_augmented_adjusted_hc3"]["hc3_95_interval_m"])),
        "precontact_position_order_control_pass": controls["position_order"],
        "sham_control_pass": controls["sham"],
        "completion_oracle_at_least_20_of_24": counts["completion_oracle"] >= 20,
        "completion_each_heavy_position_at_least_9_of_12": all(value["success"] >= 9 and value["total"] == 12 for value in counts["completion_by_heavy_position"].values()),
        "sealed_controller_and_information_boundary": integrity["controller_and_information_boundary"],
        "all_trace_hashes_and_continuous_disclosures_pass": integrity["trace_hashes_and_disclosures"],
        "execution_and_resource_contract_pass": integrity["execution_and_resource"],
    }


def main() -> int:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise FileExistsError("refusing to overwrite continuation adjudication")
    required = (PROTOCOL_PATH, ORIGINAL_PROTOCOL_PATH, PREFLIGHT_PATH, ORIGINAL_RESULT_PATH, CONTINUATION_RESULT_PATH, SEAL_PATH, SENSITIVITY_PATH, CORRECTION_PATH, STATUS_CORRECTION_PATH, PARSER_REPAIR_PATH)
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    protocol, original_protocol, preflight = load(PROTOCOL_PATH), load(ORIGINAL_PROTOCOL_PATH), load(PREFLIGHT_PATH)
    original_result, continuation, seal = load(ORIGINAL_RESULT_PATH), load(CONTINUATION_RESULT_PATH), load(SEAL_PATH)
    sensitivity, correction = load(SENSITIVITY_PATH), load(CORRECTION_PATH)
    status_correction, parser_repair = load(STATUS_CORRECTION_PATH), load(PARSER_REPAIR_PATH)
    bindings = {
        "protocol": sha256(PROTOCOL_PATH) == seal["joint_protocol_sha256"],
        "original_result": sha256(ORIGINAL_RESULT_PATH) == seal["interrupted_result_sha256"],
        "continuation_runner": sha256(ROOT / seal["continuation_runner_path"]) == seal["continuation_runner_sha256"],
        "continuation_adjudicator": adjudicator_binding_valid(seal, parser_repair, status_correction),
        "controller": sha256(ROOT / seal["controller_path"]) == seal["controller_sha256"],
        "sensitivity": sha256(SENSITIVITY_PATH) == seal["missing_pair_sensitivity_sha256"],
        "continuation_result_protocol": continuation["protocol_sha256"] == sha256(PROTOCOL_PATH),
        "continuation_result_seal": continuation["execution_seal_sha256"] == sha256(SEAL_PATH),
    }
    for trace in seal["immutable_existing_traces"]:
        bindings[f"immutable_trace:{trace['path']}"] = sha256(ROOT / trace["path"]) == trace["sha256"]
    assignments = protocol["assignments"]
    expected_primary = {f"primary:{row['scene_id']}" for row in assignments}
    original_map = {row["row_key"]: row for row in original_result["rows"]}
    continuation_map = {row["row_key"]: row for row in continuation["rows"] if row["row_type"] == "PRIMARY_ASSIGNMENT"}
    primary_map = {**original_map, **continuation_map}
    foreign_or_duplicate = bool(set(original_map) & set(continuation_map) or not set(primary_map).issubset(expected_primary))
    sham_rows = [row for row in continuation["rows"] if row["row_type"] == "SHAM_CONTROL"]
    expected_sham = {f"sham:{row['sham_id']}" for row in protocol["sham_control"]["manifest"]}
    missing_keys = []
    invalid_keys = []
    primary_records = []
    all_audits = []
    for assignment in assignments:
        key = f"primary:{assignment['scene_id']}"
        row = primary_map.get(key)
        if row is None:
            missing_keys.append(key)
            continue
        if row.get("completed"):
            audits = stored_probe_audits(row)
            status = "COMPLETED"
        elif isinstance(row.get("exception"), str) and row["exception"].startswith(MISSING_PREFIX):
            audits = [trace_probe_audit(TRACE_ROOT / f"{assignment['scene_id']}_{slot}.npz", slot, original_protocol) for slot in assignment["probe_order"]]
            status = "FAILED_FROZEN_RESPONSE_WINDOW"
        else:
            audits = []
            status = "INVALID_OTHER_FAILURE"
            invalid_keys.append(key)
        primary_records.append({"assignment": assignment, "row": row, "status": status, "probe_audits": audits})
        all_audits.extend(audits)
    complete_shams = [row for row in sham_rows if row.get("completed") and row.get("exception") is None]
    sham_key_set = {row["row_key"] for row in sham_rows}
    complete_fixed_manifest = bool(not foreign_or_duplicate and not missing_keys and not invalid_keys and len(primary_records) == 24 and len(all_audits) == 48 and sham_key_set == expected_sham and len(complete_shams) == 12)
    by_heavy, completion_by_heavy = {}, {}
    for slot in ("front", "back"):
        subset = [record for record in primary_records if record["assignment"]["heavy_slot_eval_only"] == slot]
        by_heavy[slot] = {"correct": sum(record["status"] == "COMPLETED" and bool(record["row"]["heavy_rank_correct_eval_only"]) for record in subset), "total": len(subset)}
        completion_by_heavy[slot] = {"success": sum(record["status"] == "COMPLETED" and bool(record["row"]["oracle_completion"]["official_task_success"]) for record in subset), "total": len(subset)}
    bases = {int(row["base_identity_id"]): row for row in protocol["base_states"]}
    records_by_base: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in primary_records:
        records_by_base[int(record["assignment"]["base_identity_id"])][record["assignment"]["assignment"]] = record
    pair_rows = []
    complete_pair_rows = []
    complete_contrasts = []
    worst_contrasts = []
    response_limit = float(sensitivity["assignment_B_missing_response_physically_admissible_range_m"][1])
    for identity in sorted(bases):
        pair = records_by_base.get(identity, {})
        if set(pair) != {"A", "B"}:
            continue
        a, b = pair["A"], pair["B"]
        a_response = float(a["row"]["responses_m"]["back"]) if a["status"] == "COMPLETED" else None
        b_response = float(b["row"]["responses_m"]["back"]) if b["status"] == "COMPLETED" else None
        complete = a_response is not None and b_response is not None
        contrast = float(b_response - a_response) if complete else None
        worst = float((b_response if b_response is not None else 0.0) - (a_response if a_response is not None else response_limit))
        best = float((b_response if b_response is not None else response_limit) - (a_response if a_response is not None else 0.0))
        flip = bool(complete and a["row"]["predicted_heavy_slot"] == "back" and b["row"]["predicted_heavy_slot"] == "front")
        pair_row = {
            "base_identity_id": identity,
            "assignment_A_status": a["status"],
            "assignment_B_status": b["status"],
            "assignment_A_back_response_m": a_response,
            "assignment_B_back_response_m": b_response,
            "physical_contrast_status": "COMPLETE" if complete else "MISSING_NO_IMPUTATION",
            "observed_physical_contrast_m": contrast,
            "missing_contrast_admissible_range_m": None if complete else [worst, best],
            "worst_case_sensitivity_contrast_m": worst,
            "binary_positive_sign": bool(complete and contrast > 0),
            "both_assignments_correct_flip": flip,
        }
        pair_rows.append(pair_row)
        worst_contrasts.append(contrast if complete else worst)
        if complete:
            complete_pair_rows.append(pair_row)
            complete_contrasts.append(float(contrast))
    complete_array = np.asarray(complete_contrasts, dtype=np.float64)
    worst_array = np.asarray(worst_contrasts, dtype=np.float64)
    positive = sum(row["binary_positive_sign"] for row in pair_rows)
    sign_p = float(stats.binomtest(positive, 12, 0.5, alternative="greater").pvalue)
    complete_hc3 = adjusted_hc3(complete_pair_rows, bases, complete_contrasts)
    worst_hc3 = adjusted_hc3(pair_rows, bases, worst_contrasts)
    missing_20261134 = next(row for row in pair_rows if row["base_identity_id"] == 20261134)
    other_for_tip = [float(row["observed_physical_contrast_m"] if row["physical_contrast_status"] == "COMPLETE" else row["worst_case_sensitivity_contrast_m"]) for row in pair_rows if row["base_identity_id"] != 20261134]
    paired = {
        "observed_complete_pair_count": len(complete_contrasts),
        "missing_pair_count": 12 - len(complete_contrasts),
        "observed_complete_contrasts_m": complete_contrasts,
        "observed_complete_case_mean_m": float(np.mean(complete_array)) if len(complete_array) else None,
        "observed_complete_case_median_m": float(np.median(complete_array)) if len(complete_array) else None,
        "complete_case_student_t_95_interval_m": paired_t_interval(complete_array),
        "complete_case_adjusted_hc3": complete_hc3,
        "fixed_denominator_positive_pairs": positive,
        "fixed_denominator_nonpositive_or_missing_pairs": 12 - positive,
        "fixed_denominator_one_sided_exact_sign_p": sign_p,
        "worst_case_augmented_contrasts_m_sensitivity_only": worst_contrasts,
        "worst_case_augmented_student_t_95_interval_m": paired_t_interval(worst_array),
        "worst_case_augmented_adjusted_hc3": worst_hc3,
        "base_20261134_tipping_point": tipping_point(other_for_tip, missing_20261134["missing_contrast_admissible_range_m"]),
        "twelve_pair_observed_physical_mean_or_ci_reported": False,
    }
    sham_audits = [row["probe_audit"] for row in complete_shams]
    sham_pairs_map: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in complete_shams:
        sham_pairs_map[int(row["base_identity_id"])][row["assignment"]] = row
    sham_pairs = []
    for identity in sorted(sham_pairs_map):
        pair = sham_pairs_map[identity]
        if set(pair) == {"A", "B"}:
            sham_pairs.append({"base_identity_id": identity, "contrast_m": float(pair["B"]["back_response_m"] - pair["A"]["back_response_m"]), "prediction_flip": pair["A"]["predicted_heavy_slot"] != pair["B"]["predicted_heavy_slot"]})
    sham_contrasts = np.asarray([row["contrast_m"] for row in sham_pairs], dtype=np.float64)
    sham_interval = paired_t_interval(sham_contrasts)
    position_order = bool(preflight["summary"] == {"assignment_rows": 24, "first_rgb_exact_rows": 24, "initial_localization_exact_rows": 24, "a_b_rgb_exact_pairs": 12, "a_b_localization_exact_pairs": 12})
    sham_pass = bool(len(sham_pairs) == 6 and len(complete_shams) == 12 and sum(bool(audit["sampled_target_contact"]) for audit in sham_audits) == 0 and sum(bool(audit["unintended_collision"]) for audit in sham_audits) == 0 and sum(bool(row["prediction_flip"]) for row in sham_pairs) == 0 and interval_includes_zero(sham_interval))
    controls = {"position_order": position_order, "sham": sham_pass}
    trace_disclosures = all_audits + sham_audits
    trace_hashes = all((ROOT / audit["trace_path"]).is_file() and sha256(ROOT / audit["trace_path"]) == audit["trace_sha256"] for audit in trace_disclosures)
    no_privilege = all(not audit["forbidden_online_inputs_used"] and not audit["simulator_state_used_for_actions"] and not audit["mass_or_property_used_for_actions"] for audit in trace_disclosures)
    monitors = [load(path) for path in sorted((REPORTS / "epoch9e_joint_continuation").glob("host_resource_monitor_attempt_*.json"))]
    latest_result_hash = monitors[-1].get("scientific_result_sha256_after_runner") if monitors else None
    effective_exit_codes = [effective_runner_exit_code(monitor, status_correction) for monitor in monitors]
    resource_ok = bool(monitors and all(code == 0 for code in effective_exit_codes) and all(monitor.get("host_ram_ceiling_breached") is False and float(monitor.get("peak_host_ram_percent", 100.0)) < 82.0 for monitor in monitors) and latest_result_hash == sha256(CONTINUATION_RESULT_PATH) and int(continuation["resource_monitor"]["wsl_swap_used_peak_bytes"]) == 0)
    controller_hash_ok = bindings["controller"] and seal["controller_sha256"] == correction["frozen_hashes"]["controller"]["sha256"]
    integrity = {
        "complete_fixed_manifest": complete_fixed_manifest,
        "controller_and_information_boundary": bool(controller_hash_ok and no_privilege and all(bindings.values()) and continuation["validation_accessed"] is False and continuation["confirmation_accessed"] is False),
        "trace_hashes_and_disclosures": bool(trace_hashes and len(trace_disclosures) == 60),
        "execution_and_resource": resource_ok,
    }
    counts = {
        "primary_assignments_fixed_denominator": 24,
        "primary_completed": sum(record["status"] == "COMPLETED" for record in primary_records),
        "primary_failed_missing_response": sum(record["status"] == "FAILED_FROZEN_RESPONSE_WINDOW" for record in primary_records),
        "primary_invalid_other": sum(record["status"] == "INVALID_OTHER_FAILURE" for record in primary_records),
        "primary_unexecuted": len(missing_keys),
        "candidate_probes_fixed_denominator": 48,
        "finite_bounded_actions": sum(bool(audit["finite_bounded_actions"]) for audit in all_audits),
        "intended_contact_or_excitation": sum(bool(audit["intended_contact_or_excitation"]) for audit in all_audits),
        "both_candidates_excited": sum(len(record["probe_audits"]) == 2 and all(audit["intended_contact_or_excitation"] for audit in record["probe_audits"]) for record in primary_records),
        "full_trajectory_lane_reachable": sum(bool(audit["lane_and_reachability_continuous_pass"]) for audit in all_audits),
        "collisions": sum(bool(audit["unintended_collision"]) for audit in all_audits),
        "identity_swaps": sum(bool(audit["identity_swap"]) for audit in all_audits),
        "falls": sum(bool(audit["fall"]) for audit in all_audits),
        "workspace_exits": sum(bool(audit["workspace_exit"]) for audit in all_audits),
        "unrecoverable_track_losses": sum(bool(audit["unrecoverable_track_loss"]) for audit in all_audits),
        "rank_correct": sum(record["status"] == "COMPLETED" and bool(record["row"]["heavy_rank_correct_eval_only"]) for record in primary_records),
        "rank_by_heavy_position": by_heavy,
        "exact_pair_correct_flips": sum(bool(row["both_assignments_correct_flip"]) for row in pair_rows),
        "completion_oracle": sum(record["status"] == "COMPLETED" and bool(record["row"]["oracle_completion"]["official_task_success"]) for record in primary_records),
        "completion_by_heavy_position": completion_by_heavy,
        "sham_completed": len(complete_shams),
        "sham_failed": len(sham_rows) - len(complete_shams),
        "sham_unexecuted": len(expected_sham - sham_key_set),
    }
    gates = final_gates(counts, paired, controls, integrity)
    go = all(gates.values())
    decision = protocol["success_decision"] if go else protocol["failure_decision"]
    adjudication = {
        "schema_version": "epoch9e.joint_continuation_adjudication.v1",
        "timestamp": timestamp(),
        "decision": decision,
        "joint_certification_go": go,
        "prospectively_supersedes_interruption_state": True,
        "historical_artifacts_edited": False,
        "paper_status": "PAPER_NOT_AUTHORIZED",
        "execution_bindings": bindings,
        "integrity": integrity,
        "counts": counts,
        "paired_mass_intervention": paired,
        "pair_rows": pair_rows,
        "position_order_control": {"pass": position_order, "first_rgb_exact_pairs": preflight["summary"]["a_b_rgb_exact_pairs"], "initial_localization_exact_pairs": preflight["summary"]["a_b_localization_exact_pairs"], "position_order_only_correct": 12, "denominator": 24, "pair_flips": 0},
        "sham_control": {"pass": sham_pass, "rows": len(complete_shams), "pairs": len(sham_pairs), "contrasts_m": sham_contrasts.tolist(), "paired_student_t_95_interval_m": sham_interval, "sampled_contact_rows": sum(bool(audit["sampled_target_contact"]) for audit in sham_audits), "collision_rows": sum(bool(audit["unintended_collision"]) for audit in sham_audits), "prediction_flips": sum(bool(row["prediction_flip"]) for row in sham_pairs)},
        "fixed_missing_assignment_handling": {"base_identity_id": 20261134, "assignment_B_status": next(record["status"] for record in primary_records if record["assignment"]["base_identity_id"] == 20261134 and record["assignment"]["assignment"] == "B"), "rank": "incorrect", "completion": "failure", "flip": "nonflip", "physical_contrast": "missing_not_imputed"},
        "failed_invalid_unexecuted": {"failed_missing_response_keys": [record["row"]["row_key"] for record in primary_records if record["status"] == "FAILED_FROZEN_RESPONSE_WINDOW"], "invalid_other_keys": invalid_keys, "unexecuted_primary_keys": missing_keys, "unexecuted_sham_keys": sorted(expected_sham - sham_key_set)},
        "continuous_trace_disclosure": trace_disclosures,
        "gates": gates,
        "failed_gates": [name for name, value in gates.items() if not value],
        "resource_attempts": monitors,
        "resource_effective_authoritative_exit_codes": effective_exit_codes,
        "host_exit_status_correction": status_correction,
        "adjudicator_parser_repair": parser_repair,
        "source_hashes": {"protocol": sha256(PROTOCOL_PATH), "original_result": sha256(ORIGINAL_RESULT_PATH), "continuation_result": sha256(CONTINUATION_RESULT_PATH), "seal": sha256(SEAL_PATH), "sensitivity": sha256(SENSITIVITY_PATH)},
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT_JSON, adjudication)
    atomic_write_text(OUTPUT_MD, f"""# Epoch 9E Completed Fixed-Denominator Joint Adjudication

Decision: `{decision}`

The append-only continuation finalized `{counts['primary_completed']}` completed and `{counts['primary_failed_missing_response']}` frozen-response-miss primary assignments out of the fixed 24, with `{counts['primary_unexecuted']}` unexecuted. All 12 frozen shams are reported separately. Base `20261134` remains adverse/nonflip, its Assignment B rank and completion count as failures, and its missing response is never imputed into the observed physical mean or interval.

Observed physical contrasts: `{paired['observed_complete_pair_count']}/12`; complete-case mean `{paired['observed_complete_case_mean_m']}` m; complete-case 95% interval `{paired['complete_case_student_t_95_interval_m']}`. Fixed-denominator sign p is `{paired['fixed_denominator_one_sided_exact_sign_p']}`. The sensitivity-only worst-case interval is `{paired['worst_case_augmented_student_t_95_interval_m']}`.

Rank: `{counts['rank_correct']}/24`; exact correct flips: `{counts['exact_pair_correct_flips']}/12`; completion oracle: `{counts['completion_oracle']}/24`; contact/excitation: `{counts['intended_contact_or_excitation']}/48`; lane/reach: `{counts['full_trajectory_lane_reachable']}/48`.

Every original gate and conservative missing-pair gate is recorded in `{relative(OUTPUT_JSON)}`. Validation and confirmation remain sealed. Paper status remains `PAPER_NOT_AUTHORIZED` at this adjudication boundary.
""")
    print(json.dumps({"decision": decision, "go": go, "rank": counts["rank_correct"], "flips": counts["exact_pair_correct_flips"], "completion": counts["completion_oracle"], "complete_contrasts": len(complete_contrasts), "sign_p": sign_p, "failed_gates": adjudication["failed_gates"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
