#!/usr/bin/env python3
"""Adjudicate the sole Epoch 9E joint certification without repair or rerun."""

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
SEAL_PATH = REPORTS / "epoch9e_joint_execution_seal.json"
RESULT_PATH = REPORTS / "epoch9e_joint_certification/result.json"
HOST_PATH = REPORTS / "epoch9e_joint_certification/host_resource_monitor.json"
OUTPUT_JSON = REPORTS / "epoch9e_joint_certification_adjudication.json"
OUTPUT_MD = REPORTS / "epoch9e_joint_certification_adjudication.md"


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
    mean = float(np.mean(values))
    standard_error = float(stats.sem(values))
    if not np.isfinite(standard_error):
        return [None, None]
    half = float(stats.t.ppf(0.975, len(values) - 1) * standard_error)
    return [mean - half, mean + half]


def adjusted_hc3(pair_rows: list[dict[str, Any]], bases: dict[int, dict[str, Any]]) -> dict[str, Any]:
    if len(pair_rows) != 12:
        return {
            "valid": False,
            "estimate_m": None,
            "hc3_standard_error_m": None,
            "degrees_of_freedom": None,
            "hc3_95_interval_m": [None, None],
            "covariates_centered": ["initial back x", "initial back y", "initial back lane margin", "back-first probe order"],
        }
    covariates, outcome = [], []
    for row in pair_rows:
        base = bases[int(row["base_identity_id"])]
        initial = base["candidate_initial_xyz_eval_only"]["back"]
        margin = base["candidate_initial_lane_margin_m_eval_only"]["back"]
        covariates.append([float(initial[0]), float(initial[1]), float(margin), float(base["probe_order"][0] == "back")])
        outcome.append(float(row["mass_contrast_m"]))
    z = np.asarray(covariates, dtype=np.float64)
    z -= z.mean(axis=0, keepdims=True)
    y = np.asarray(outcome, dtype=np.float64)
    x = np.column_stack((np.ones(len(y)), z))
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
    critical = float(stats.t.ppf(0.975, degrees))
    estimate = float(beta[0])
    interval = [estimate - critical * standard_error, estimate + critical * standard_error]
    return {
        "valid": bool(np.isfinite([estimate, standard_error, *interval]).all() and degrees == 7),
        "estimate_m": estimate,
        "hc3_standard_error_m": standard_error,
        "degrees_of_freedom": degrees,
        "hc3_95_interval_m": interval,
        "covariates_centered": ["initial back x", "initial back y", "initial back lane margin", "back-first probe order"],
    }


def interval_lower_positive(interval: list[float | None]) -> bool:
    return interval[0] is not None and float(interval[0]) > 0.0


def interval_includes_zero(interval: list[float | None]) -> bool:
    return interval[0] is not None and interval[1] is not None and float(interval[0]) <= 0.0 <= float(interval[1])


def joint_gates(counts: dict[str, Any], statistics: dict[str, Any], controls: dict[str, Any], integrity: dict[str, bool]) -> dict[str, bool]:
    return {
        "complete_unique_manifest_36_rows": bool(integrity["complete_unique_manifest"]),
        "finite_bounded_actions_48_of_48": counts["finite_bounded_actions"] == 48,
        "intended_contact_or_excitation_at_least_46_of_48": counts["intended_contact_or_excitation"] >= 46,
        "both_candidates_excited_at_least_22_of_24": counts["both_candidates_excited"] >= 22,
        "full_trajectory_lane_reachable_48_of_48": counts["full_trajectory_lane_reachable"] == 48,
        "zero_collision_identity_swap_fall_workspace_exit_track_loss": sum(counts[key] for key in ("collisions", "identity_swaps", "falls", "workspace_exits", "unrecoverable_track_losses")) == 0,
        "rank_at_least_20_of_24": counts["rank_correct"] >= 20,
        "rank_each_heavy_position_at_least_10_of_12": all(value["correct"] >= 10 and value["total"] == 12 for value in counts["rank_by_heavy_position"].values()),
        "exact_pair_flips_at_least_9_of_12": counts["exact_pair_correct_flips"] >= 9,
        "one_sided_sign_test_p_strictly_below_0_01": statistics["one_sided_exact_sign_test_p"] < 0.01,
        "paired_95_interval_excludes_zero_positive": interval_lower_positive(statistics["paired_student_t_95_interval_m"]),
        "adjusted_hc3_interval_excludes_zero_positive": bool(statistics["adjusted_position_lane_order"]["valid"] and statistics["adjusted_position_lane_order"]["estimate_m"] > 0 and interval_lower_positive(statistics["adjusted_position_lane_order"]["hc3_95_interval_m"])),
        "precontact_position_order_control_pass": bool(controls["position_order_pass"]),
        "sham_control_pass": bool(controls["sham_pass"]),
        "completion_oracle_at_least_20_of_24": counts["completion_oracle"] >= 20,
        "completion_each_heavy_position_at_least_9_of_12": all(value["success"] >= 9 and value["total"] == 12 for value in counts["completion_by_heavy_position"].values()),
        "sealed_nondrag_controller_audit_pass": bool(integrity["nondrag_controller"]),
        "admissible_observation_only_method": bool(integrity["information_boundary"]),
        "trace_hashes_pass": bool(integrity["trace_hashes"]),
        "execution_and_resource_contract_pass": bool(integrity["execution_and_resource"]),
    }


def main() -> int:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise FileExistsError("refusing to overwrite the one-shot joint adjudication")
    for path in (PROTOCOL_PATH, SEAL_PATH, RESULT_PATH, HOST_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    protocol, seal, result, host = load(PROTOCOL_PATH), load(SEAL_PATH), load(RESULT_PATH), load(HOST_PATH)
    bindings = {
        "protocol": sha256(PROTOCOL_PATH) == seal["joint_protocol_sha256"],
        "runner": sha256(ROOT / seal["runner_path"]) == seal["runner_sha256"],
        "adjudicator": sha256(Path(__file__)) == seal["adjudicator_sha256"],
        "host": sha256(ROOT / seal["host_wrapper_path"]) == seal["host_wrapper_sha256"],
        "controller": sha256(ROOT / seal["controller_path"]) == seal["controller_sha256"],
        "original_runner": sha256(ROOT / seal["original_runner_path"]) == seal["original_runner_sha256"],
        "mechanics_adjudication": sha256(ROOT / seal["mechanics_adjudication_path"]) == seal["mechanics_adjudication_sha256"],
        "result_host_hash": sha256(RESULT_PATH) == host.get("scientific_result_sha256_after_runner"),
        "result_protocol": result.get("protocol_sha256") == sha256(PROTOCOL_PATH),
        "result_seal": result.get("execution_seal_sha256") == sha256(SEAL_PATH),
    }
    rows = result.get("rows", [])
    keys = [row.get("row_key") for row in rows]
    primary = [row for row in rows if row.get("row_type") == "PRIMARY_ASSIGNMENT" and row.get("completed")]
    sham = [row for row in rows if row.get("row_type") == "SHAM_CONTROL" and row.get("completed")]
    expected_primary = {f"primary:{row['scene_id']}" for row in protocol["assignments"]}
    expected_sham = {f"sham:{row['sham_id']}" for row in protocol["sham_control"]["manifest"]}
    complete_unique_manifest = bool(len(keys) == len(set(keys)) and set(keys) == expected_primary | expected_sham and len(primary) == 24 and len(sham) == 12 and all(row.get("exception") is None for row in rows))
    audits = [audit for row in primary for audit in row.get("probe_audits", {}).values()]
    sham_audits = [row["probe_audit"] for row in sham if "probe_audit" in row]

    by_heavy, completion_by_heavy = {}, {}
    for slot in ("front", "back"):
        subset = [row for row in primary if row["heavy_slot_eval_only"] == slot]
        by_heavy[slot] = {"correct": sum(bool(row["heavy_rank_correct_eval_only"]) for row in subset), "total": len(subset)}
        completion_by_heavy[slot] = {"success": sum(bool(row["oracle_completion"]["official_task_success"]) for row in subset), "total": len(subset)}

    bases = {int(row["base_identity_id"]): row for row in protocol["base_states"]}
    pair_map: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in primary:
        pair_map[int(row["base_identity_id"])][row["assignment"]] = row
    pair_rows = []
    for identity in sorted(pair_map):
        pair = pair_map[identity]
        if set(pair) != {"A", "B"}:
            continue
        a, b = pair["A"], pair["B"]
        pair_rows.append({
            "base_identity_id": identity,
            "assignment_A_predicted": a["predicted_heavy_slot"],
            "assignment_B_predicted": b["predicted_heavy_slot"],
            "both_assignments_correct_flip": bool(a["predicted_heavy_slot"] == "back" and b["predicted_heavy_slot"] == "front"),
            "back_response_heavy_assignment_A_m": float(a["responses_m"]["back"]),
            "back_response_light_assignment_B_m": float(b["responses_m"]["back"]),
            "mass_contrast_m": float(b["responses_m"]["back"] - a["responses_m"]["back"]),
            "first_rgb_hash_exact": bool(a["exact_state_audit"]["first_rgb_after_mass_sha256"] == b["exact_state_audit"]["first_rgb_after_mass_sha256"] == bases[identity]["first_agentview_rgb_sha256"]),
            "initial_localization_exact": bool(a["exact_state_audit"]["initial_rgb_localization_audit"] == b["exact_state_audit"]["initial_rgb_localization_audit"] == bases[identity]["initial_rgb_localization_audit"]),
        })
    contrasts = np.asarray([row["mass_contrast_m"] for row in pair_rows], dtype=np.float64)
    positive = int(np.count_nonzero(contrasts > 0))
    negative = int(np.count_nonzero(contrasts < 0))
    zeros = int(np.count_nonzero(contrasts == 0))
    sign_p = float(stats.binomtest(positive, positive + negative, 0.5, alternative="greater").pvalue) if positive + negative else 1.0
    contrast_interval = paired_t_interval(contrasts)
    adjusted = adjusted_hc3(pair_rows, bases)

    sham_map: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in sham:
        sham_map[int(row["base_identity_id"])][row["assignment"]] = row
    sham_pairs = []
    for identity in sorted(sham_map):
        pair = sham_map[identity]
        if set(pair) != {"A", "B"}:
            continue
        a, b = pair["A"], pair["B"]
        sham_pairs.append({
            "base_identity_id": identity,
            "mass_contrast_m": float(b["back_response_m"] - a["back_response_m"]),
            "prediction_flip": a["predicted_heavy_slot"] != b["predicted_heavy_slot"],
        })
    sham_contrasts = np.asarray([row["mass_contrast_m"] for row in sham_pairs], dtype=np.float64)
    sham_interval = paired_t_interval(sham_contrasts)
    sham_contacts = sum(bool(row["sampled_target_contact"]) for row in sham_audits)
    sham_collisions = sum(bool(row["unintended_collision"]) for row in sham_audits)
    sham_flips = sum(bool(row["prediction_flip"]) for row in sham_pairs)
    first_rgb_pairs = sum(bool(row["first_rgb_hash_exact"]) for row in pair_rows)
    localization_pairs = sum(bool(row["initial_localization_exact"]) for row in pair_rows)
    position_order_correct = len(pair_rows)
    position_order_flips = 0
    controls = {
        "position_order_pass": bool(len(pair_rows) == 12 and first_rgb_pairs == 12 and localization_pairs == 12 and position_order_correct <= 12 and position_order_flips == 0),
        "sham_pass": bool(len(sham_pairs) == 6 and len(sham) == 12 and sham_contacts == 0 and sham_collisions == 0 and sham_flips == 0 and interval_includes_zero(sham_interval)),
    }

    trace_hashes = True
    for audit in audits + sham_audits:
        path = ROOT / audit["trace_path"]
        trace_hashes = bool(trace_hashes and path.is_file() and sha256(path) == audit["trace_sha256"])
    no_privilege = all(not audit["forbidden_online_inputs_used"] and not audit["simulator_state_used_for_actions"] and not audit["mass_or_property_used_for_actions"] for audit in audits + sham_audits)
    nondrag_integrity = bool(len(audits) == 48 and all(audit["nondrag_attempt_count"] >= 1 and audit["nondrag_liftoff_planar_commands_exact_zero"] and audit["nondrag_all_separations_verified"] and not audit["nondrag_forbidden_inputs_used"] for audit in audits))
    execution_resource = bool(all(bindings.values()) and result.get("one_shot_no_resume") is True and result.get("validation_accessed") is False and result.get("confirmation_accessed") is False and host.get("runner_exit_code") == 0 and host.get("host_ram_ceiling_breached") is False and float(host.get("peak_host_ram_percent", 100.0)) < 82.0 and int(result.get("resource_monitor", {}).get("wsl_swap_used_peak_bytes", -1)) == 0)
    integrity = {
        "complete_unique_manifest": complete_unique_manifest,
        "nondrag_controller": nondrag_integrity,
        "information_boundary": no_privilege,
        "trace_hashes": trace_hashes,
        "execution_and_resource": execution_resource,
    }
    counts = {
        "primary_scenes": len(primary),
        "candidate_probes": len(audits),
        "sham_rows": len(sham),
        "finite_bounded_actions": sum(bool(row["finite_bounded_actions"]) for row in audits),
        "intended_contact_or_excitation": sum(bool(row["intended_contact_or_excitation"]) for row in audits),
        "both_candidates_excited": sum(bool(row["both_candidates_excited_eval_only"]) for row in primary),
        "full_trajectory_lane_reachable": sum(bool(row["lane_and_reachability_continuous_pass"]) for row in audits),
        "collisions": sum(bool(row["unintended_collision"]) for row in audits),
        "identity_swaps": sum(bool(row["identity_swap"]) for row in audits),
        "falls": sum(bool(row["fall"]) for row in audits),
        "workspace_exits": sum(bool(row["workspace_exit"]) for row in audits),
        "unrecoverable_track_losses": sum(bool(row["unrecoverable_track_loss"]) for row in audits),
        "rank_correct": sum(bool(row["heavy_rank_correct_eval_only"]) for row in primary),
        "rank_by_heavy_position": by_heavy,
        "exact_pair_correct_flips": sum(bool(row["both_assignments_correct_flip"]) for row in pair_rows),
        "completion_oracle": sum(bool(row["oracle_completion"]["official_task_success"]) for row in primary),
        "completion_by_heavy_position": completion_by_heavy,
    }
    statistics = {
        "definition": "back response light (assignment B) minus back response heavy (assignment A)",
        "contrasts_m": contrasts.tolist(),
        "mean_m": float(np.mean(contrasts)) if len(contrasts) else None,
        "median_m": float(np.median(contrasts)) if len(contrasts) else None,
        "standard_deviation_m": float(np.std(contrasts, ddof=1)) if len(contrasts) > 1 else None,
        "paired_student_t_95_interval_m": contrast_interval,
        "positive_pairs": positive,
        "negative_pairs": negative,
        "zero_pairs": zeros,
        "one_sided_exact_sign_test_p": sign_p,
        "adjusted_position_lane_order": adjusted,
    }
    gates = joint_gates(counts, statistics, controls, integrity)
    joint_go = all(gates.values())
    decision = protocol["success_decision"] if joint_go else protocol["failure_decision"]
    adjudication = {
        "schema_version": "epoch9e.joint_certification_adjudication.v1",
        "timestamp": timestamp(),
        "decision": decision,
        "joint_certification_go": joint_go,
        "paper_status": "PAPER_NOT_AUTHORIZED",
        "near_miss_rerun_authorized": False,
        "implementation_or_scientific_repair_used": False,
        "execution_bindings": bindings,
        "integrity": integrity,
        "counts": counts,
        "paired_mass_intervention": statistics,
        "precontact_position_order_control": {
            "theoretical_correct": position_order_correct,
            "denominator": 24,
            "pair_flips": position_order_flips,
            "first_rgb_exact_pairs": first_rgb_pairs,
            "initial_localization_exact_pairs": localization_pairs,
            "pass": controls["position_order_pass"],
        },
        "sham_control": {
            "base_pairs": len(sham_pairs),
            "rows": len(sham),
            "contrasts_m": sham_contrasts.tolist(),
            "paired_student_t_95_interval_m": sham_interval,
            "sampled_contact_rows": sham_contacts,
            "collision_rows": sham_collisions,
            "prediction_flips": sham_flips,
            "pass": controls["sham_pass"],
        },
        "pair_rows": pair_rows,
        "misses": [
            {"row_key": row.get("row_key"), "exception": row.get("exception")}
            for row in rows if not row.get("completed") or row.get("exception") is not None
        ] + [{"gate": name} for name, passed in gates.items() if not passed],
        "gates": gates,
        "resource_summary": {
            "runner_exit_code": host.get("runner_exit_code"),
            "peak_host_ram_percent": host.get("peak_host_ram_percent"),
            "peak_host_used_physical_bytes": host.get("peak_host_used_physical_bytes"),
            "peak_gpu_used_mib": host.get("peak_gpu_used_mib"),
            "process_max_rss_bytes": result.get("resource_monitor", {}).get("process_max_rss_bytes"),
            "wsl_mem_used_peak_bytes": result.get("resource_monitor", {}).get("wsl_mem_used_peak_bytes"),
            "wsl_swap_used_peak_bytes": result.get("resource_monitor", {}).get("wsl_swap_used_peak_bytes"),
        },
        "source_hashes": {
            "protocol": sha256(PROTOCOL_PATH),
            "seal": sha256(SEAL_PATH),
            "result": sha256(RESULT_PATH),
            "host": sha256(HOST_PATH),
        },
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT_JSON, adjudication)
    interval = statistics["paired_student_t_95_interval_m"]
    markdown = f"""# Epoch 9E Joint Certification Adjudication

Decision: `{decision}`

Paper: `PAPER_NOT_AUTHORIZED`

The sole sealed panel retained {len(primary)}/24 complete primary assignments, {len(audits)}/48 candidate probes, and {len(sham)}/12 complete sham rows. No near-miss rerun or controller repair is authorized.

## Primary record

- finite bounded actions: {counts['finite_bounded_actions']}/48
- intended contact or excitation: {counts['intended_contact_or_excitation']}/48
- both candidates excited: {counts['both_candidates_excited']}/24
- full-trajectory lane/reachability: {counts['full_trajectory_lane_reachable']}/48
- heavy/light rank: {counts['rank_correct']}/24
- exact paired assignment flips: {counts['exact_pair_correct_flips']}/12
- completion oracle: {counts['completion_oracle']}/24
- collisions / identity swaps / falls / workspace exits / track losses: {counts['collisions']} / {counts['identity_swaps']} / {counts['falls']} / {counts['workspace_exits']} / {counts['unrecoverable_track_losses']}

## Paired effect and controls

The preregistered mean exact-pair contrast is `{statistics['mean_m']}` m with paired 95% Student-t interval `{interval}` and one-sided exact sign-test `p = {sign_p}`. Position/order control pass is `{str(controls['position_order_pass']).lower()}`; sham control pass is `{str(controls['sham_pass']).lower()}`.

Every gate, pair contrast, miss, continuous trace hash, and resource boundary is recorded in `{relative(OUTPUT_JSON)}`. Validation and confirmation identities remain sealed.
"""
    atomic_write_text(OUTPUT_MD, markdown)
    print(json.dumps({"decision": decision, "joint_certification_go": joint_go, "counts": counts, "sign_p": sign_p, "mean_contrast_m": statistics["mean_m"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
