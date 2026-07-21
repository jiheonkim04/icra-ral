#!/usr/bin/env python3
"""Adjudicate the frozen Epoch 9D causal panel without tuning."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
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
PROTOCOL_PATH = REPORTS / "epoch9d_causal_panel_protocol.json"
SEAL_PATH = REPORTS / "epoch9d_causal_execution_seal.json"
RESULT_PATH = REPORTS / "epoch9d_causal_panel/result.json"
HOST_RESOURCE_PATH = REPORTS / "epoch9d_causal_panel/host_resource_monitor.json"
PARSER_REPAIR_PATH = REPORTS / "epoch9d_causal_adjudication_parser_repair.json"
OUTPUT_JSON = REPORTS / "epoch9d_causal_panel_adjudication.json"
OUTPUT_MD = REPORTS / "epoch9d_causal_panel_adjudication.md"


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


def paired_t_interval(values: np.ndarray) -> list[float]:
    mean = float(np.mean(values))
    standard_error = float(stats.sem(values))
    half = float(stats.t.ppf(0.975, len(values) - 1) * standard_error)
    return [mean - half, mean + half]


def adjusted_hc3(pair_rows: list[dict[str, Any]], base_lookup: dict[int, dict[str, Any]]) -> dict[str, Any]:
    covariates = []
    outcome = []
    strata = sorted({base_lookup[row["base_identity_id"]]["spatial_stratum"] for row in pair_rows})
    for row in pair_rows:
        base = base_lookup[row["base_identity_id"]]
        initial = base["candidate_initial_xyz_eval_only"]["back"]
        margin = base["candidate_initial_lane_margin_m_eval_only"]["back"]
        order = float(base["probe_order"][0] == "back")
        dummies = [float(base["spatial_stratum"] == value) for value in strata[1:]]
        covariates.append([float(initial[0]), float(initial[1]), float(margin), order, *dummies])
        outcome.append(float(row["mass_contrast_m"]))
    z = np.asarray(covariates, dtype=np.float64)
    z = z - z.mean(axis=0, keepdims=True)
    y = np.asarray(outcome, dtype=np.float64)
    x = np.column_stack((np.ones(len(y)), z))
    inverse = np.linalg.pinv(x.T @ x)
    beta = inverse @ x.T @ y
    residual = y - x @ beta
    hat = np.sum((x @ inverse) * x, axis=1)
    adjusted_residual = residual / np.maximum(1.0 - hat, 1e-9)
    meat = np.zeros((x.shape[1], x.shape[1]), dtype=np.float64)
    for vector, value in zip(x, adjusted_residual, strict=True):
        meat += np.outer(vector, vector) * value**2
    covariance = inverse @ meat @ inverse
    standard_error = float(np.sqrt(max(covariance[0, 0], 0.0)))
    degrees = len(y) - x.shape[1]
    critical = float(stats.t.ppf(0.975, degrees))
    estimate = float(beta[0])
    return {
        "estimate_m": estimate,
        "hc3_standard_error_m": standard_error,
        "degrees_of_freedom": degrees,
        "hc3_95_interval_m": [estimate - critical * standard_error, estimate + critical * standard_error],
        "covariates_centered": [
            "initial back x",
            "initial back y",
            "initial back lane margin",
            "back-first probe order",
            *[f"spatial stratum {value}" for value in strata[1:]],
        ],
    }


def all_probe_audits(primary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [audit for row in primary for audit in row["probe_audits"].values()]


def safety_count(audits: list[dict[str, Any]], key: str) -> int:
    return sum(bool(row[key]) for row in audits)


def main() -> int:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise FileExistsError("refusing to overwrite causal adjudication")
    for path in (PROTOCOL_PATH, SEAL_PATH, RESULT_PATH, HOST_RESOURCE_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    protocol = load(PROTOCOL_PATH)
    seal = load(SEAL_PATH)
    current_adjudicator_sha256 = sha256(Path(__file__))
    if current_adjudicator_sha256 != seal["adjudicator_sha256"]:
        if not PARSER_REPAIR_PATH.exists():
            raise RuntimeError("unsealed adjudicator change")
        repair = load(PARSER_REPAIR_PATH)
        if (
            repair["original_sealed_adjudicator_sha256"] != seal["adjudicator_sha256"]
            or repair["repaired_adjudicator_sha256"] != current_adjudicator_sha256
            or repair["scientific_fields_changed"]
        ):
            raise RuntimeError("invalid adjudicator parser repair seal")
    result = load(RESULT_PATH)
    host = load(HOST_RESOURCE_PATH)
    if sha256(PROTOCOL_PATH) != seal["causal_protocol_sha256"]:
        raise RuntimeError("protocol seal mismatch")
    if sha256(RESULT_PATH) != host["scientific_result_sha256_after_runner"]:
        raise RuntimeError("host monitor result hash mismatch")
    if result["protocol_sha256"] != sha256(PROTOCOL_PATH):
        raise RuntimeError("result protocol mismatch")
    if result["execution_seal_sha256"] != sha256(SEAL_PATH):
        raise RuntimeError("result execution seal mismatch")
    if result["validation_accessed"] or result["confirmation_accessed"]:
        raise RuntimeError("sealed stage contamination")

    rows = result["rows"]
    keys = [row["row_key"] for row in rows]
    unique_rows = len(keys) == len(set(keys))
    primary = [row for row in rows if row["row_type"] == "PRIMARY_ASSIGNMENT"]
    sham = [row for row in rows if row["row_type"] == "SHAM_CONTROL"]
    expected_primary = {f"primary:{row['scene_id']}" for row in protocol["assignments"]}
    expected_sham = {f"sham:{row['sham_id']}" for row in protocol["sham_control"]["manifest"]}
    complete_manifest = set(keys) == expected_primary | expected_sham
    all_complete = all(bool(row.get("completed")) and row.get("exception") is None for row in rows)
    audits = all_probe_audits(primary)

    finite_actions = sum(bool(row["finite_bounded_actions"]) for row in audits)
    intended = sum(bool(row["intended_contact_or_excitation"]) for row in audits)
    both_excited = sum(bool(row["both_candidates_excited_eval_only"]) for row in primary)
    collisions = safety_count(audits, "unintended_collision")
    identity_swaps = safety_count(audits, "identity_swap")
    falls = safety_count(audits, "fall")
    workspace_exits = safety_count(audits, "workspace_exit")
    track_losses = safety_count(audits, "unrecoverable_track_loss")
    rank_correct = sum(bool(row["heavy_rank_correct_eval_only"]) for row in primary)
    by_heavy = {}
    for slot in ("front", "back"):
        subset = [row for row in primary if row["heavy_slot_eval_only"] == slot]
        by_heavy[slot] = {
            "correct": sum(bool(row["heavy_rank_correct_eval_only"]) for row in subset),
            "total": len(subset),
        }

    base_lookup = {int(row["base_identity_id"]): row for row in protocol["base_states"]}
    pair_map: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in primary:
        pair_map[int(row["base_identity_id"])][row["assignment"]] = row
    pair_rows = []
    for identity in sorted(pair_map):
        pair = pair_map[identity]
        if set(pair) != {"A", "B"}:
            continue
        a = pair["A"]
        b = pair["B"]
        pair_rows.append(
            {
                "base_identity_id": identity,
                "assignment_A_predicted": a["predicted_heavy_slot"],
                "assignment_B_predicted": b["predicted_heavy_slot"],
                "both_assignments_correct_flip": bool(
                    a["predicted_heavy_slot"] == "back" and b["predicted_heavy_slot"] == "front"
                ),
                "back_response_heavy_assignment_A_m": float(a["responses_m"]["back"]),
                "back_response_light_assignment_B_m": float(b["responses_m"]["back"]),
                "mass_contrast_m": float(b["responses_m"]["back"] - a["responses_m"]["back"]),
                "first_rgb_hash_exact": bool(
                    a["exact_state_audit"]["first_rgb_after_mass_sha256"]
                    == b["exact_state_audit"]["first_rgb_after_mass_sha256"]
                    == a["exact_state_audit"]["expected_first_rgb_sha256"]
                ),
                "initial_localization_exact": bool(
                    a["exact_state_audit"]["initial_rgb_localization_audit"]
                    == b["exact_state_audit"]["initial_rgb_localization_audit"]
                    == base_lookup[int(identity)]["initial_rgb_localization_audit"]
                ),
            }
        )
    pair_flips = sum(row["both_assignments_correct_flip"] for row in pair_rows)
    contrasts = np.asarray([row["mass_contrast_m"] for row in pair_rows], dtype=np.float64)
    positive = int(np.count_nonzero(contrasts > 0))
    negative = int(np.count_nonzero(contrasts < 0))
    zeros = int(np.count_nonzero(contrasts == 0))
    sign_test = stats.binomtest(positive, positive + negative, 0.5, alternative="greater")
    contrast_interval = paired_t_interval(contrasts)
    adjusted = adjusted_hc3(pair_rows, base_lookup)

    sham_pair_map: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in sham:
        sham_pair_map[int(row["base_identity_id"])][row["assignment"]] = row
    sham_pairs = []
    for identity in sorted(sham_pair_map):
        pair = sham_pair_map[identity]
        if set(pair) != {"A", "B"}:
            continue
        a, b = pair["A"], pair["B"]
        sham_pairs.append(
            {
                "base_identity_id": identity,
                "mass_contrast_m": float(b["back_response_m"] - a["back_response_m"]),
                "prediction_flip": a["predicted_heavy_slot"] != b["predicted_heavy_slot"],
            }
        )
    sham_contrasts = np.asarray([row["mass_contrast_m"] for row in sham_pairs], dtype=np.float64)
    sham_interval = paired_t_interval(sham_contrasts)
    sham_audits = [row["probe_audit"] for row in sham]
    sham_contacts = sum(bool(row["sampled_target_contact"]) for row in sham_audits)
    sham_collisions = sum(bool(row["unintended_collision"]) for row in sham_audits)
    sham_flips = sum(bool(row["prediction_flip"]) for row in sham_pairs)

    precontact_hash_pairs = sum(bool(row["first_rgb_hash_exact"]) for row in pair_rows)
    precontact_localization_pairs = sum(bool(row["initial_localization_exact"]) for row in pair_rows)
    position_order_control_correct = len(pair_rows)  # exactly one correct assignment per exact swapped pair
    position_order_control_pair_flips = 0
    no_privilege = all(
        not audit["forbidden_online_inputs_used"]
        and not audit["simulator_state_used_for_actions"]
        and not audit["mass_or_property_used_for_actions"]
        for audit in audits + sham_audits
    )

    gates = {
        "complete_unique_manifest_48_rows": unique_rows and complete_manifest and all_complete and len(primary) == 32 and len(sham) == 16,
        "finite_bounded_actions_64_of_64": finite_actions == 64,
        "intended_contact_or_excitation_at_least_61_of_64": intended >= 61,
        "both_candidates_excited_at_least_29_of_32": both_excited >= 29,
        "zero_collision_identity_swap_fall_workspace_exit_track_loss": (
            collisions + identity_swaps + falls + workspace_exits + track_losses == 0
        ),
        "rank_at_least_26_of_32": rank_correct >= 26,
        "rank_each_heavy_position_at_least_12_of_16": all(value["correct"] >= 12 for value in by_heavy.values()),
        "exact_pair_flips_at_least_12_of_16": pair_flips >= 12,
        "one_sided_sign_test_p_below_0_01": float(sign_test.pvalue) < 0.01,
        "paired_95_interval_excludes_zero_positive": contrast_interval[0] > 0,
        "adjusted_hc3_interval_excludes_zero_positive": adjusted["estimate_m"] > 0 and adjusted["hc3_95_interval_m"][0] > 0,
        "precontact_position_order_controls_cannot_reproduce": (
            precontact_hash_pairs == 16
            and precontact_localization_pairs == 16
            and position_order_control_correct <= 16
            and position_order_control_pair_flips == 0
        ),
        "sham_does_not_reproduce": (
            len(sham_pairs) == 8
            and sham_contacts == 0
            and sham_collisions == 0
            and sham_flips == 0
            and sham_interval[0] <= 0 <= sham_interval[1]
        ),
        "admissible_observation_only_score": no_privilege,
        "resource_contract": (
            float(host["peak_host_ram_percent"]) < 82.0
            and int(result["resource_monitor"]["wsl_swap_used_peak_bytes"]) == 0
            and int(host["runner_exit_code"]) == 0
        ),
    }
    causal_go = all(gates.values())

    count_misses = []
    count_checks = {
        "intended_contact_or_excitation": (intended, 61, 2, "probe"),
        "both_candidates_excited": (both_excited, 29, 1, "scene"),
        "rank_overall": (rank_correct, 26, 1, "scene"),
        "rank_front": (by_heavy["front"]["correct"], 12, 1, "scene"),
        "rank_back": (by_heavy["back"]["correct"], 12, 1, "scene"),
        "pair_flips": (pair_flips, 12, 1, "scene"),
    }
    for name, (observed, required, allowance, unit) in count_checks.items():
        if observed < required:
            count_misses.append(
                {"gate": name, "observed": observed, "required": required, "shortfall": required - observed, "allowance": allowance, "unit": unit}
            )
    non_count_gate_failure = any(
        not value
        for name, value in gates.items()
        if name
        not in {
            "intended_contact_or_excitation_at_least_61_of_64",
            "both_candidates_excited_at_least_29_of_32",
            "rank_at_least_26_of_32",
            "rank_each_heavy_position_at_least_12_of_16",
            "exact_pair_flips_at_least_12_of_16",
        }
    )
    disqualifying = bool(
        not unique_rows
        or not complete_manifest
        or not all_complete
        or collisions
        or identity_swaps
        or falls
        or workspace_exits
        or not no_privilege
    )
    near_miss_eligible = bool(
        not causal_go
        and not disqualifying
        and not non_count_gate_failure
        and len(count_misses) == 1
        and count_misses[0]["shortfall"] <= count_misses[0]["allowance"]
    )
    decision = (
        "CAUSAL_SIGNAL_GO"
        if causal_go
        else "CAUSAL_SIGNAL_NEAR_MISS_REPLICATION_AUTHORIZED"
        if near_miss_eligible
        else "ACTIVE_DYNAMIC_PROBE_CAUSAL_SIGNAL_NOT_CONFIRMED"
    )
    adjudication = {
        "schema_version": "epoch9d.causal_mass_swap_adjudication.v1",
        "timestamp": timestamp(),
        "decision": decision,
        "paper_status": "PAPER_NOT_AUTHORIZED",
        "causal_signal_go": causal_go,
        "near_miss_replication_eligible": near_miss_eligible,
        "near_miss_count_gate_failures": count_misses,
        "protocol": {"path": relative(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
        "execution_seal": {"path": relative(SEAL_PATH), "sha256": sha256(SEAL_PATH)},
        "raw_result": {"path": relative(RESULT_PATH), "sha256": sha256(RESULT_PATH)},
        "host_resource": {"path": relative(HOST_RESOURCE_PATH), "sha256": sha256(HOST_RESOURCE_PATH)},
        "counts": {
            "primary_scenes": len(primary),
            "candidate_probes": len(audits),
            "sham_rows": len(sham),
            "finite_bounded_actions": finite_actions,
            "intended_contact_or_excitation": intended,
            "both_candidates_excited": both_excited,
            "collisions": collisions,
            "identity_swaps": identity_swaps,
            "falls": falls,
            "workspace_exits": workspace_exits,
            "unrecoverable_track_losses": track_losses,
            "rank_correct": rank_correct,
            "rank_by_heavy_position": by_heavy,
            "exact_pair_correct_flips": pair_flips,
            "first_rgb_exact_pairs": precontact_hash_pairs,
            "initial_localization_exact_pairs": precontact_localization_pairs,
        },
        "paired_mass_intervention": {
            "definition": "back response light (assignment B) minus back response heavy (assignment A)",
            "contrasts_m": contrasts.tolist(),
            "mean_m": float(np.mean(contrasts)),
            "median_m": float(np.median(contrasts)),
            "standard_deviation_m": float(np.std(contrasts, ddof=1)),
            "paired_student_t_95_interval_m": contrast_interval,
            "positive_pairs": positive,
            "negative_pairs": negative,
            "zero_pairs": zeros,
            "one_sided_exact_sign_test_p": float(sign_test.pvalue),
            "adjusted_position_lane_order": adjusted,
        },
        "precontact_position_order_control": {
            "theoretical_correct": position_order_control_correct,
            "denominator": 32,
            "pair_flips": position_order_control_pair_flips,
            "first_rgb_exact_pairs": precontact_hash_pairs,
            "initial_localization_exact_pairs": precontact_localization_pairs,
        },
        "sham_control": {
            "pairs": len(sham_pairs),
            "contrasts_m": sham_contrasts.tolist(),
            "mean_m": float(np.mean(sham_contrasts)),
            "paired_student_t_95_interval_m": sham_interval,
            "sampled_contact_rows": sham_contacts,
            "collision_rows": sham_collisions,
            "prediction_flips": sham_flips,
        },
        "pair_rows": pair_rows,
        "gates": gates,
        "resource_summary": {
            "peak_host_ram_percent": host["peak_host_ram_percent"],
            "peak_host_used_physical_bytes": host["peak_host_used_physical_bytes"],
            "peak_gpu_used_mib": host["peak_gpu_used_mib"],
            "process_max_rss_bytes": result["resource_monitor"]["process_max_rss_bytes"],
            "wsl_mem_used_peak_bytes": result["resource_monitor"]["wsl_mem_used_peak_bytes"],
            "wsl_swap_used_peak_bytes": result["resource_monitor"]["wsl_swap_used_peak_bytes"],
        },
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT_JSON, adjudication)
    markdown = f"""# Epoch 9D Exact-State Mass-Swap Causal Adjudication

Decision: `{decision}`

Paper: `PAPER_NOT_AUTHORIZED`

The sealed panel completed {len(primary)}/32 primary assignments, {len(audits)}/64 candidate probes, and {len(sham)}/16 sham rows. All rows, including failures, are retained in `{relative(RESULT_PATH)}`.

## Primary counts

- finite bounded actions: {finite_actions}/64
- intended contact or excitation: {intended}/64
- both candidates excited: {both_excited}/32
- correct heavy/light ranking: {rank_correct}/32
- front-heavy ranking: {by_heavy['front']['correct']}/16
- back-heavy ranking: {by_heavy['back']['correct']}/16
- exact pairs with both assignments correctly flipped: {pair_flips}/16
- collisions / identity swaps / falls / workspace exits / unrecoverable track losses: {collisions} / {identity_swaps} / {falls} / {workspace_exits} / {track_losses}

## Paired causal effect

The preregistered exact-pair contrast (back-light response minus back-heavy response) is `{float(np.mean(contrasts)):.6f} m` on average. Its paired 95% Student-t interval is `[{contrast_interval[0]:.6f}, {contrast_interval[1]:.6f}] m`; {positive}/{positive + negative} nonzero pairs have the expected sign, giving one-sided exact sign-test `p = {float(sign_test.pvalue):.6g}`. The centered position/lane/order HC3 estimate is `{adjusted['estimate_m']:.6f} m` with interval `[{adjusted['hc3_95_interval_m'][0]:.6f}, {adjusted['hc3_95_interval_m'][1]:.6f}] m`.

## Shortcut and sham controls

Exact paired first RGB matched in {precontact_hash_pairs}/16 pairs and initial localization fields matched in {precontact_localization_pairs}/16. A deterministic position/order/pre-contact-only rule cannot flip within an exact pair and therefore scores exactly {position_order_control_correct}/32 at best under swapped labels. The sham ran {len(sham_pairs)} paired base states with {sham_contacts} sampled contacts, {sham_collisions} collisions, and {sham_flips} prediction flips; its paired contrast interval is `[{sham_interval[0]:.6f}, {sham_interval[1]:.6f}] m`.

## Gate record

`CAUSAL_SIGNAL_GO` is {str(causal_go).lower()}. Near-miss replication eligibility is {str(near_miss_eligible).lower()}. Every individual gate and exact-pair row is recorded in `{relative(OUTPUT_JSON)}`. No validation or confirmation identity was accessed.
"""
    atomic_write_text(OUTPUT_MD, markdown)
    print(json.dumps({
        "decision": decision,
        "causal_signal_go": causal_go,
        "rank_correct": rank_correct,
        "pair_flips": pair_flips,
        "sign_p": float(sign_test.pvalue),
        "mean_contrast_m": float(np.mean(contrasts)),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
