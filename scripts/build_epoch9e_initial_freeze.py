#!/usr/bin/env python3
"""Build the append-only Epoch 9E authority, endpoint, and identity freeze."""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import stats


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.adjudicate_epoch9d_causal_panel import adjusted_hc3, paired_t_interval
from tca_map.epoch7_latent_dynamics import atomic_write_json


REPORTS = ROOT / "reports"
PROMPT = (
    Path("/mnt/c/Users/jiheo/Downloads/epoch9e_nondrag_disengagement_ral_convergence_prompt.md")
    if Path("/mnt/c/Users/jiheo/Downloads").exists()
    else Path("C:/Users/jiheo/Downloads/epoch9e_nondrag_disengagement_ral_convergence_prompt.md")
)
SCOPE_JSON = REPORTS / "epoch9e_scope_and_authority_correction.json"
SCOPE_MD = REPORTS / "epoch9e_scope_and_authority_correction.md"
ENDPOINT_JSON = REPORTS / "epoch9e_endpoint_construct_audit.json"
ENDPOINT_MD = REPORTS / "epoch9e_endpoint_construct_audit.md"
IDENTITY_OUTPUT = REPORTS / "epoch9e_fresh_identity_manifest.json"
SOURCE_COMMIT = "74dd66c32a8b8595e187b13d3ccafe05cae6753b"
SOURCE_BRANCH = "codex/epoch9d-causal-probe-bounded-convergence"
BRANCH = "codex/epoch9e-nondrag-disengagement-convergence"
TEXT_SUFFIXES = {".bib", ".csv", ".json", ".jsonl", ".log", ".md", ".toml", ".tsv", ".txt", ".yaml", ".yml"}
PATH_ID_PATTERN = re.compile(r"(?:identity|pilot|base|scene|demo)[_= -]?(\d{1,4})(?!\d)", re.IGNORECASE)
PATH_SEED_PATTERN = re.compile(r"seed[= _-]?(\d+)", re.IGNORECASE)


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


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def protected_snapshot(path: Path) -> dict[str, Any]:
    files = sorted(value for value in path.rglob("*") if value.is_file())
    lines = [
        f"{relative(value)}\t{value.stat().st_size}\t{sha256(value)}"
        for value in files
    ]
    return {
        "path": relative(path) + "/",
        "file_count": len(files),
        "total_bytes": sum(value.stat().st_size for value in files),
        "manifest_sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest().upper(),
        "touched_by_epoch9e": False,
    }


def integers(value: Any) -> Iterable[int]:
    if isinstance(value, bool):
        return
    if isinstance(value, int):
        yield int(value)
    elif isinstance(value, list):
        for item in value:
            yield from integers(item)


def identity_key(key: str) -> bool:
    value = key.lower()
    exact = {
        "identity",
        "identities",
        "identity_id",
        "identity_ids",
        "base_identity_id",
        "base_identity_ids",
        "generated_identity_id",
        "generated_identity_ids",
        "demo_index",
        "demo_indices",
        "demo_identity",
        "demo_identities",
        "source_state_demo_index",
        "source_demo_identity",
        "source_demo_identities",
        "init_state_id",
        "init_state_ids",
        "init_state_index",
        "init_state_indices",
        "reset_id",
        "reset_ids",
        "reset_index",
        "reset_indices",
    }
    return (
        value in exact
        or value.endswith("_generated_identity_id")
        or value.endswith("_generated_identity_ids")
        or value.endswith("_demo_index")
        or value.endswith("_demo_indices")
        or value.endswith("_demo_identity")
        or value.endswith("_demo_identities")
    )


def development_context(path: Path, key_path: tuple[str, ...], enclosing: dict[str, Any]) -> bool:
    tokens = [relative(path).lower(), *(item.lower() for item in key_path)]
    for key in ("partition", "evidence_class", "split", "stage"):
        value = enclosing.get(key)
        if isinstance(value, str):
            tokens.append(value.lower())
    joined = " ".join(tokens)
    return any(token in joined for token in ("development", "discovery", "training", "pilot", "smoke", "calibration"))


def expand_seed_value(key: str, value: Any) -> list[int]:
    found = list(integers(value))
    if "range" in key.lower() and isinstance(value, list) and len(found) == 2 and found[1] >= found[0]:
        if found[1] - found[0] <= 100_000:
            return list(range(found[0], found[1] + 1))
    return found


def inventory() -> dict[str, Any]:
    identity_by_key: dict[str, set[int]] = defaultdict(set)
    identity_occurrences: Counter[str] = Counter()
    seeds: set[int] = set()
    development_ids: set[int] = set()
    epoch9_ids: set[int] = set()
    scanned = 0
    parsed = 0
    parse_failures: list[str] = []

    def walk(value: Any, path: Path, key_path: tuple[str, ...] = (), enclosing: dict[str, Any] | None = None) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = (*key_path, str(key))
                key_lower = str(key).lower()
                if "seed" in key_lower:
                    seeds.update(expand_seed_value(key_lower, child))
                if identity_key(key_lower):
                    found = list(integers(child))
                    identity_by_key[key_lower].update(found)
                    identity_occurrences[key_lower] += len(found)
                    if "epoch9" in relative(path).lower():
                        epoch9_ids.update(found)
                    if development_context(path, child_path, value):
                        development_ids.update(found)
                walk(child, path, child_path, value)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, path, (*key_path, f"[{index}]"), enclosing)

    for scan_root in (REPORTS, ROOT / "rollouts"):
        for path in sorted(value for value in scan_root.rglob("*") if value.is_file()):
            scanned += 1
            path_text = relative(path)
            for match in PATH_ID_PATTERN.finditer(path_text):
                number = int(match.group(1))
                identity_by_key["path_identity"].add(number)
                identity_occurrences["path_identity"] += 1
                if "epoch9" in path_text.lower():
                    epoch9_ids.add(number)
                if any(token in path_text.lower() for token in ("development", "pilot", "smoke", "training")):
                    development_ids.add(number)
            for match in PATH_SEED_PATTERN.finditer(path_text):
                seeds.add(int(match.group(1)))
            if path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="strict")
            except UnicodeDecodeError:
                parse_failures.append(path_text)
                continue
            if path.suffix.lower() == ".json":
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    parse_failures.append(path_text)
                else:
                    parsed += 1
                    walk(payload, path)

    if not development_ids or not epoch9_ids:
        raise RuntimeError("identity namespace scan was empty")
    development_m = max(development_ids)
    maximum_epoch9_reference = max(epoch9_ids)
    allocation_floor = max(development_m, maximum_epoch9_reference) + 1
    allocations = {
        "mechanics_smoke": {"identity_ids": list(range(allocation_floor, allocation_floor + 8)), "generator_seeds": list(range(915000, 915008))},
        "joint_certification_base_pairs": {"identity_ids": list(range(allocation_floor + 8, allocation_floor + 20)), "generator_seeds": list(range(915100, 915112))},
        "estimator_training": {"identity_ids": list(range(allocation_floor + 20, allocation_floor + 68)), "generator_seeds": list(range(915200, 915248))},
        "estimator_development": {"identity_ids": list(range(allocation_floor + 68, allocation_floor + 92)), "generator_seeds": list(range(915300, 915324))},
        "validation_fresh": {"identity_ids": list(range(allocation_floor + 92, allocation_floor + 132)), "generator_seeds": list(range(915400, 915440))},
        "official_closed_loop": {"identity_ids": list(range(allocation_floor + 132, allocation_floor + 212)), "generator_seeds": list(range(915500, 915580))},
        "confirmation_fresh": {"identity_ids": list(range(allocation_floor + 212, allocation_floor + 252)), "generator_seeds": list(range(915600, 915640))},
        "robustness_generalization": {"identity_ids": list(range(allocation_floor + 252, allocation_floor + 292)), "generator_seeds": list(range(915700, 915740))},
        "mechanics_replacement_only_if_unit_defect": {"identity_ids": list(range(allocation_floor + 292, allocation_floor + 300)), "generator_seeds": list(range(915800, 915808))},
    }
    all_new_ids = [number for group in allocations.values() for number in group["identity_ids"]]
    all_new_seeds = [number for group in allocations.values() for number in group["generator_seeds"]]
    if len(all_new_ids) != len(set(all_new_ids)) or set(all_new_ids) & epoch9_ids:
        raise RuntimeError("new identity collision")
    if len(all_new_seeds) != len(set(all_new_seeds)) or set(all_new_seeds) & seeds:
        raise RuntimeError("new seed collision")
    if min(all_new_ids) <= development_m:
        raise RuntimeError("fresh identities are not above M")
    return {
        "schema_version": "epoch9e.fresh_identity_manifest.v1",
        "generated_at": timestamp(),
        "scan_scope": ["reports/**", "rollouts/**", "evidence ledgers under reports/**"],
        "scanned_file_count": scanned,
        "parsed_json_file_count": parsed,
        "text_parse_failures": sorted(set(parse_failures)),
        "identity_values_by_concrete_reference_key": {key: sorted(values) for key, values in sorted(identity_by_key.items())},
        "identity_occurrence_count_by_key": dict(sorted(identity_occurrences.items())),
        "epoch9_identity_values": sorted(epoch9_ids),
        "development_identity_values": sorted(development_ids),
        "maximum_used_numeric_development_identity_M": development_m,
        "maximum_prior_epoch9_identity_reference": maximum_epoch9_reference,
        "allocation_floor": allocation_floor,
        "previous_seed_values": sorted(seeds),
        "previous_seed_count": len(seeds),
        "allocations": allocations,
        "sealed_source_demo_identities": {"validation": list(range(40, 45)), "confirmation": list(range(45, 50))},
        "disjointness_audit": {
            "all_new_identity_ids_unique": len(all_new_ids) == len(set(all_new_ids)),
            "all_new_identity_ids_above_M": min(all_new_ids) > development_m,
            "all_new_identity_ids_disjoint_from_prior_epoch9_references": not bool(set(all_new_ids) & epoch9_ids),
            "all_new_seeds_unique": len(all_new_seeds) == len(set(all_new_seeds)),
            "all_new_seeds_disjoint_from_every_scanned_seed": not bool(set(all_new_seeds) & seeds),
            "sealed_40_49_not_reallocated": not bool(set(all_new_ids) & set(range(40, 50))),
        },
        "stage_access": {
            "mechanics_smoke": False,
            "joint_certification": False,
            "estimator_training": False,
            "estimator_development": False,
            "validation": False,
            "official_closed_loop": False,
            "confirmation": False,
            "robustness_generalization": False,
        },
    }


def causal_reproduction() -> dict[str, Any]:
    protocol_path = REPORTS / "epoch9d_causal_panel_protocol.json"
    result_path = REPORTS / "epoch9d_causal_panel/result.json"
    adjudication_path = REPORTS / "epoch9d_causal_panel_adjudication.json"
    host_path = REPORTS / "epoch9d_causal_panel/host_resource_monitor.json"
    seal_path = REPORTS / "epoch9d_causal_execution_seal.json"
    protocol = load(protocol_path)
    result = load(result_path)
    original = load(adjudication_path)
    host = load(host_path)
    seal = load(seal_path)
    if sha256(protocol_path) != seal["causal_protocol_sha256"]:
        raise RuntimeError("Epoch 9D causal protocol seal mismatch")
    if sha256(result_path) != host["scientific_result_sha256_after_runner"]:
        raise RuntimeError("Epoch 9D host/result mismatch")
    rows = result["rows"]
    primary = [row for row in rows if row["row_type"] == "PRIMARY_ASSIGNMENT"]
    sham = [row for row in rows if row["row_type"] == "SHAM_CONTROL"]
    trace_bindings = []
    for row in primary:
        for slot, audit in row["probe_audits"].items():
            path = ROOT / audit["trace_path"]
            trace_bindings.append({"row_key": row["row_key"], "slot": slot, "path": relative(path), "sha256": sha256(path), "matches_row": sha256(path) == audit["trace_sha256"]})
    for row in sham:
        audit = row["probe_audit"]
        path = ROOT / audit["trace_path"]
        trace_bindings.append({"row_key": row["row_key"], "slot": row["slot"], "path": relative(path), "sha256": sha256(path), "matches_row": sha256(path) == audit["trace_sha256"]})
    if len(trace_bindings) != 80 or not all(row["matches_row"] for row in trace_bindings):
        raise RuntimeError("Epoch 9D raw trace binding failed")
    audits = [audit for row in primary for audit in row["probe_audits"].values()]
    finite = sum(bool(row["finite_bounded_actions"]) for row in audits)
    intended = sum(bool(row["intended_contact_or_excitation"]) for row in audits)
    both = sum(bool(row["both_candidates_excited_eval_only"]) for row in primary)
    safety = {key: sum(bool(row[key]) for row in audits) for key in ("unintended_collision", "identity_swap", "fall", "workspace_exit", "unrecoverable_track_loss")}
    rank = sum(bool(row["heavy_rank_correct_eval_only"]) for row in primary)
    by_heavy = {
        slot: {
            "correct": sum(bool(row["heavy_rank_correct_eval_only"]) for row in primary if row["heavy_slot_eval_only"] == slot),
            "total": sum(row["heavy_slot_eval_only"] == slot for row in primary),
        }
        for slot in ("front", "back")
    }
    base_lookup = {int(row["base_identity_id"]): row for row in protocol["base_states"]}
    pair_map: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in primary:
        pair_map[int(row["base_identity_id"])][row["assignment"]] = row
    pair_rows = []
    for identity in sorted(pair_map):
        a, b = pair_map[identity]["A"], pair_map[identity]["B"]
        pair_rows.append({
            "base_identity_id": identity,
            "both_assignments_correct_flip": a["predicted_heavy_slot"] == "back" and b["predicted_heavy_slot"] == "front",
            "mass_contrast_m": float(b["responses_m"]["back"] - a["responses_m"]["back"]),
            "first_rgb_hash_exact": a["exact_state_audit"]["first_rgb_after_mass_sha256"] == b["exact_state_audit"]["first_rgb_after_mass_sha256"] == a["exact_state_audit"]["expected_first_rgb_sha256"],
            "initial_localization_exact": a["exact_state_audit"]["initial_rgb_localization_audit"] == b["exact_state_audit"]["initial_rgb_localization_audit"] == base_lookup[identity]["initial_rgb_localization_audit"],
        })
    pair_flips = sum(row["both_assignments_correct_flip"] for row in pair_rows)
    contrasts = np.asarray([row["mass_contrast_m"] for row in pair_rows], dtype=np.float64)
    positive = int(np.count_nonzero(contrasts > 0))
    negative = int(np.count_nonzero(contrasts < 0))
    sign_p = float(stats.binomtest(positive, positive + negative, 0.5, alternative="greater").pvalue)
    interval = paired_t_interval(contrasts)
    adjusted = adjusted_hc3(pair_rows, base_lookup)
    sham_map: dict[int, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in sham:
        sham_map[int(row["base_identity_id"])][row["assignment"]] = row
    sham_contrasts = np.asarray([pair["B"]["back_response_m"] - pair["A"]["back_response_m"] for pair in sham_map.values()], dtype=np.float64)
    sham_interval = paired_t_interval(sham_contrasts)
    sham_audits = [row["probe_audit"] for row in sham]
    sham_contacts = sum(bool(row["sampled_target_contact"]) for row in sham_audits)
    sham_collisions = sum(bool(row["unintended_collision"]) for row in sham_audits)
    sham_flips = sum(pair["A"]["predicted_heavy_slot"] != pair["B"]["predicted_heavy_slot"] for pair in sham_map.values())
    no_privilege = all(not audit["forbidden_online_inputs_used"] and not audit["simulator_state_used_for_actions"] and not audit["mass_or_property_used_for_actions"] for audit in audits + sham_audits)
    expected_primary = {f"primary:{row['scene_id']}" for row in protocol["assignments"]}
    expected_sham = {f"sham:{row['sham_id']}" for row in protocol["sham_control"]["manifest"]}
    keys = [row["row_key"] for row in rows]
    gates = {
        "complete_unique_manifest_48_rows": len(keys) == len(set(keys)) and set(keys) == expected_primary | expected_sham and all(row.get("completed") and row.get("exception") is None for row in rows) and len(primary) == 32 and len(sham) == 16,
        "finite_bounded_actions_64_of_64": finite == 64,
        "intended_contact_or_excitation_at_least_61_of_64": intended >= 61,
        "both_candidates_excited_at_least_29_of_32": both >= 29,
        "zero_collision_identity_swap_fall_workspace_exit_track_loss": sum(safety.values()) == 0,
        "rank_at_least_26_of_32": rank >= 26,
        "rank_each_heavy_position_at_least_12_of_16": all(value["correct"] >= 12 for value in by_heavy.values()),
        "exact_pair_flips_at_least_12_of_16": pair_flips >= 12,
        "one_sided_sign_test_p_below_0_01": sign_p < 0.01,
        "paired_95_interval_excludes_zero_positive": interval[0] > 0,
        "adjusted_hc3_interval_excludes_zero_positive": adjusted["estimate_m"] > 0 and adjusted["hc3_95_interval_m"][0] > 0,
        "precontact_position_order_controls_cannot_reproduce": sum(row["first_rgb_hash_exact"] for row in pair_rows) == 16 and sum(row["initial_localization_exact"] for row in pair_rows) == 16,
        "sham_does_not_reproduce": len(sham_map) == 8 and sham_contacts == 0 and sham_collisions == 0 and sham_flips == 0 and sham_interval[0] <= 0 <= sham_interval[1],
        "admissible_observation_only_score": no_privilege,
        "resource_contract": float(host["peak_host_ram_percent"]) < 82.0 and int(result["resource_monitor"]["wsl_swap_used_peak_bytes"]) == 0 and int(host["runner_exit_code"]) == 0,
    }
    if gates != original["gates"] or not all(gates.values()):
        raise RuntimeError("raw Epoch 9D gate reproduction differs from frozen adjudication")
    reproduced_counts = {
        "primary_scenes": len(primary), "candidate_probes": len(audits), "sham_rows": len(sham),
        "finite_bounded_actions": finite, "intended_contact_or_excitation": intended,
        "both_candidates_excited": both, "collisions": safety["unintended_collision"],
        "identity_swaps": safety["identity_swap"], "falls": safety["fall"],
        "workspace_exits": safety["workspace_exit"], "unrecoverable_track_losses": safety["unrecoverable_track_loss"],
        "rank_correct": rank, "rank_by_heavy_position": by_heavy, "exact_pair_correct_flips": pair_flips,
        "first_rgb_exact_pairs": sum(row["first_rgb_hash_exact"] for row in pair_rows),
        "initial_localization_exact_pairs": sum(row["initial_localization_exact"] for row in pair_rows),
    }
    if reproduced_counts != original["counts"]:
        raise RuntimeError("raw Epoch 9D count reproduction differs")
    return {
        "decision": "CAUSAL_SIGNAL_GO",
        "protocol": {"path": relative(protocol_path), "sha256": sha256(protocol_path)},
        "execution_seal": {"path": relative(seal_path), "sha256": sha256(seal_path)},
        "raw_result": {"path": relative(result_path), "sha256": sha256(result_path)},
        "adjudication": {"path": relative(adjudication_path), "sha256": sha256(adjudication_path)},
        "host_resource": {"path": relative(host_path), "sha256": sha256(host_path)},
        "trace_binding_count": len(trace_bindings),
        "all_trace_hashes_match_raw_rows": all(row["matches_row"] for row in trace_bindings),
        "trace_bindings": trace_bindings,
        "counts": reproduced_counts,
        "paired_mass_intervention": {
            "contrasts_m": contrasts.tolist(), "mean_m": float(np.mean(contrasts)), "median_m": float(np.median(contrasts)),
            "paired_student_t_95_interval_m": interval, "positive_pairs": positive, "negative_pairs": negative,
            "one_sided_exact_sign_test_p": sign_p, "adjusted_position_lane_order": adjusted,
        },
        "position_order_precontact_control": {"theoretical_correct": 16, "denominator": 32, "pair_flips": 0, "first_rgb_exact_pairs": 16, "initial_localization_exact_pairs": 16},
        "sham_control": {"pairs": len(sham_map), "mean_m": float(np.mean(sham_contrasts)), "paired_student_t_95_interval_m": sham_interval, "sampled_contact_rows": sham_contacts, "collision_rows": sham_collisions, "prediction_flips": sham_flips},
        "gates": gates,
        "matches_frozen_adjudication": True,
    }


def lane_endpoint_audit() -> dict[str, Any]:
    protocol_path = REPORTS / "epoch9b_v2_task_preservation_protocol.json"
    variant_path = REPORTS / "epoch9d_controller_development_protocol.json"
    result_path = REPORTS / "epoch9d_controller_development/variant1_pilot_result.json"
    failure_path = REPORTS / "epoch9d_controller_bounded_failure.json"
    protocol = load(protocol_path)
    variant = load(variant_path)
    result = load(result_path)

    def inside(position: np.ndarray, slot: str) -> bool:
        lane = protocol["safe_center_lanes_m"][slot]
        reach = protocol["reachable_center_envelope_m"]
        return bool(lane["x"][0] <= position[0] <= lane["x"][1] and lane["y"][0] <= position[1] <= lane["y"][1] and reach["z"][0] <= position[2] <= reach["z"][1])

    failures = []
    for row in result["rows"]:
        for probe in row["probes"]:
            if probe["lane_and_reachability_continuous_pass"]:
                continue
            trace_path = ROOT / probe["trace_path"]
            with np.load(trace_path, allow_pickle=False) as trace:
                phases = np.asarray(trace["phase"]).astype(str)
                positions = np.asarray(trace["candidate_positions_eval_only"], dtype=np.float64)
            indices = []
            for index, pair in enumerate(positions):
                failed_slots = [slot for slot_index, slot in enumerate(("front", "back")) if not inside(pair[slot_index], slot)]
                if failed_slots:
                    indices.append((index, failed_slots))
            first, slots = indices[0]
            failures.append({
                "row_key": row["row_key"], "probe_slot": probe["slot"], "probe_order": row["scene"]["probe_order"],
                "source": "inherited_from_prior_probe" if first == 0 else "causal_exit_in_this_probe",
                "first_failure_index": int(first), "first_failure_phase": str(phases[first]), "failed_slots": slots,
                "first_failure_positions_m_eval_only": {slot: positions[first, slot_index].tolist() for slot_index, slot in enumerate(("front", "back")) if slot in slots},
                "guard_events": probe["predictive_lane_guard"]["events"], "trace_path": relative(trace_path), "trace_sha256": sha256(trace_path),
            })
    causal = [row for row in failures if row["source"] == "causal_exit_in_this_probe"]
    inherited = [row for row in failures if row["source"] == "inherited_from_prior_probe"]
    if len(failures) != 4 or len(causal) != 2 or len(inherited) != 2 or {row["first_failure_phase"] for row in causal} != {"contact_verify_retract"}:
        raise RuntimeError("Epoch 9D lane failure reconstruction mismatch")
    failure = load(failure_path)
    if failure["new_lane_exit_phases"] != ["contact_verify_retract", "contact_verify_retract"]:
        raise RuntimeError("historical bounded failure report mismatch")
    return {
        "frozen_variant1_protocol": {"path": relative(variant_path), "sha256": sha256(variant_path)},
        "frozen_conditional_adjustment": variant["variant1_conditional_adjustment"],
        "pilot_result": {"path": relative(result_path), "sha256": sha256(result_path)},
        "pilot_counts": result["summary"],
        "failed_probe_rows": failures,
        "causal_exit_count": len(causal),
        "inherited_failure_count": len(inherited),
        "causal_exit_phases": [row["first_failure_phase"] for row in causal],
        "conditional_adjustment_was_ineligible": True,
        "post_response_pose_variant_was_inapplicable": True,
        "construct": {
            "source_protocol": {"path": relative(protocol_path), "sha256": sha256(protocol_path)},
            "geometric_basis": protocol["geometric_basis"],
            "clean_reset_basis": protocol["clean_reset_basis"],
            "safe_center_lanes_m": protocol["safe_center_lanes_m"],
            "reachable_center_envelope_m": protocol["reachable_center_envelope_m"],
            "absolute_displacement_rule": protocol["v2_absolute_displacement_rule"],
            "classification": "CONSERVATIVE_SURROGATE_RETAINED_AS_FROZEN_HARD_CERTIFICATION_GATE",
            "classification_basis": [
                "lane rectangles were built from clean-reset extrema expanded by a fixed 0.01 m cross-lane margin, not from an exact collision boundary",
                "the collision radius supports the separate displacement cap but does not make each lane edge a physical collision surface",
                "Epoch 9D achieved 12/12 oracle completion and zero actual safety events despite two causal lane exits, so lane violation did not imply observed task impossibility",
                "the rectangle still protects identity separation, fixture-side trackability, and a consistent reachable workspace construct",
            ],
            "epoch9e_rule": "retain the original full-trajectory lane/reachability requirement at 48/48; adopt no replacement endpoint",
            "actual_validity_endpoints_also_required": ["zero actual safety events", "post-probe completion oracle >=20/24 overall and >=9/12 per heavy-position stratum"],
            "continuous_lane_margin_reporting": "primary hard gate plus full disclosed distribution",
            "replacement_selected_from_old_rows": False,
        },
    }


def main() -> int:
    outputs = (SCOPE_JSON, SCOPE_MD, ENDPOINT_JSON, ENDPOINT_MD, IDENTITY_OUTPUT)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite Epoch 9E initial freeze")
    if git("rev-parse", "HEAD") != SOURCE_COMMIT:
        raise RuntimeError("Epoch 9E must start exactly at 74dd66c")
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("wrong Epoch 9E branch")
    if not PROMPT.exists():
        raise FileNotFoundError(PROMPT)
    protected = [protected_snapshot(ROOT / "rollouts/2026_07_17"), protected_snapshot(ROOT / "rollouts/2026_07_18")]
    expected = {
        "rollouts/2026_07_17/": (27, 5_143_751, "25DE8FF5AA6112D7EFF8BCF38D3A4C3F0F3C8C8EE0458E5FA83D17438719EC54"),
        "rollouts/2026_07_18/": (10, 924_633, "CF701D6F73D4783F016E48A72C093DC9FD6D940B7081DA8FBEC128DB94C24A00"),
    }
    for row in protected:
        if (row["file_count"], row["total_bytes"], row["manifest_sha256"]) != expected[row["path"]]:
            raise RuntimeError(f"protected rollout drift: {row['path']}")
    causal = causal_reproduction()
    endpoint = lane_endpoint_audit()
    identities = inventory()
    scope = {
        "schema_version": "epoch9e.scope_and_authority_correction.v1",
        "created_at": timestamp(),
        "authority_source": {"path": str(PROMPT).replace("\\", "/"), "sha256": sha256(PROMPT), "role": "newest controlling user instruction"},
        "repository": str(ROOT).replace("\\", "/"),
        "source_branch": SOURCE_BRANCH,
        "source_commit_full": SOURCE_COMMIT,
        "epoch9e_branch": BRANCH,
        "append_only": True,
        "epoch9d_files_modified": [],
        "epoch9d_scientific_scope_correction": {
            "exact_interpretation": ["CAUSAL_MASS_SIGNAL_CONFIRMED", "VARIANT1_PILOT_FROZEN_NO_GO", "PRE_RESPONSE_DISENGAGEMENT_UNRESOLVED", "PAPER_NOT_AUTHORIZED"],
            "causal_phase_b": {"rank_correct": 28, "rank_total": 32, "exact_pair_correct_flips": 12, "exact_pair_total": 16, "paired_mass_response_contrast_m": causal["paired_mass_intervention"]["mean_m"], "paired_95_interval_m": causal["paired_mass_intervention"]["paired_student_t_95_interval_m"]},
            "variant1_pilot": {"contact_excitation": 24, "probe_total": 24, "oracle_completion": 12, "scene_total": 12, "actual_safety_events": 0, "lane_reach": 20, "rank_correct": 9, "back_heavy_rank_correct": 3},
            "joint_gate_meaning": "task-preservation-not-achieved was a frozen joint gate failure, not 0/12 completion or physical impossibility",
        },
        "historical_evidence": {
            "causal_adjudication": causal["adjudication"],
            "controller_failure": {"path": relative(REPORTS / "epoch9d_controller_bounded_failure.json"), "sha256": sha256(REPORTS / "epoch9d_controller_bounded_failure.json")},
            "campaign_state": {"path": relative(REPORTS / "epoch9d_campaign_state.json"), "sha256": sha256(REPORTS / "epoch9d_campaign_state.json")},
        },
        "sealed_source_demo_identities": {"validation": list(range(40, 45)), "confirmation": list(range(45, 50)), "accessed_by_epoch9e": []},
        "protected_rollouts": protected,
        "resource_contract": {"host_physical_bytes": 24_871_014_400, "host_ram_ceiling_percent": 82.0, "simulator_environments_at_once": 1, "resident_models_at_once": 1, "wsl_swap_use_allowed_bytes": 0, "model_offload_allowed": False, "two_four_shard_schedule_invariance_required": False},
        "one_shot_authority": {"controller_family": "NONDRAG_DISENGAGEMENT_ONLY", "joint_certification_panels": 1, "near_miss_rerun": False, "controller_rotation_after_failure": False},
        "paper_status": "PAPER_NOT_AUTHORIZED",
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    endpoint_report = {
        "schema_version": "epoch9e.endpoint_construct_audit.v1",
        "created_at": timestamp(),
        "phase_b_raw_reproduction": causal,
        "variant1_lane_failure_reconstruction": endpoint,
        "joint_certification_endpoint_decision": {"full_trajectory_lane_reachability_gate": "48/48 RETAINED", "replacement_endpoint": None, "score_window_sign_changed": False, "chosen_using_new_outcomes": False},
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(SCOPE_JSON, scope)
    atomic_write_json(ENDPOINT_JSON, endpoint_report)
    atomic_write_json(IDENTITY_OUTPUT, identities)
    atomic_write_text(SCOPE_MD, f"""# Epoch 9E scope and authority correction

Epoch 9E starts from full commit `{SOURCE_COMMIT}` on `{BRANCH}` under authority SHA-256 `{scope['authority_source']['sha256']}`. This report is append-only: no Epoch 9D file or decision is rewritten.

The precise preserved interpretation is `CAUSAL_MASS_SIGNAL_CONFIRMED`, `VARIANT1_PILOT_FROZEN_NO_GO`, `PRE_RESPONSE_DISENGAGEMENT_UNRESOLVED`, and `PAPER_NOT_AUTHORIZED`. Phase B remains a valid causal GO. Variant 1 remains a frozen NO-GO. Its 12/12 oracle completion and zero safety events show that the failed joint gate was not evidence of physical impossibility.

Only one non-drag disengagement controller and one joint certification are authorized. Identities 40--44 and 45--49 remain sealed. The two protected rollout directories remain untracked and byte-identical.
""")
    atomic_write_text(ENDPOINT_MD, f"""# Epoch 9E endpoint construct audit

All {causal['trace_binding_count']} Epoch 9D causal traces match their raw-row hashes. Recomputed Phase B gates all pass: ranking is {causal['counts']['rank_correct']}/32, exact correct flips are {causal['counts']['exact_pair_correct_flips']}/16, mean paired contrast is {causal['paired_mass_intervention']['mean_m']:.9f} m with 95% interval {causal['paired_mass_intervention']['paired_student_t_95_interval_m']}, and one-sided exact sign-test p is {causal['paired_mass_intervention']['one_sided_exact_sign_test_p']:.9g}. Pre-contact, position/order, sham, admissibility, and resource gates reproduce the frozen adjudication.

The four Variant 1 failed probe rows reconstruct as two causal exits and two inherited failures. Both causal exits first cross during `contact_verify_retract`, before the response window. The frozen `(10,14] mm` adjustment remains ineligible and the post-response recovery variant remains inapplicable.

The lane rectangle is a conservative workspace/identity surrogate derived from clean-reset extrema plus a 10 mm cross-lane margin, not an exact collision surface. Epoch 9E nevertheless retains the original full-trajectory lane/reachability gate at 48/48 and adopts no replacement. Zero actual safety events and downstream completion remain additional mandatory validity endpoints, and the full continuous margin distribution must be disclosed.
""")
    print(json.dumps({"source_commit": SOURCE_COMMIT, "causal_go_reproduced": all(causal["gates"].values()), "M": identities["maximum_used_numeric_development_identity_M"], "allocation_floor": identities["allocation_floor"], "protected": protected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
