#!/usr/bin/env python3
"""Adjudicate the sealed Epoch 9D controller pilot without tuning."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json


REPORTS = ROOT / "reports"
PROTOCOL_PATH = REPORTS / "epoch9d_controller_development_protocol.json"
SEAL_PATH = REPORTS / "epoch9d_controller_pilot_execution_seal.json"
ORIGINAL_PROTOCOL_PATH = REPORTS / "epoch9b_v2_task_preservation_protocol.json"
RESULT_PATH = REPORTS / "epoch9d_controller_development/variant1_pilot_result.json"
HOST_RESOURCE_PATH = REPORTS / "epoch9d_controller_development/variant1_pilot_host_resource_monitor.json"
OUTPUT_JSON = REPORTS / "epoch9d_controller_variant1_pilot_adjudication.json"
OUTPUT_MD = REPORTS / "epoch9d_controller_variant1_pilot_adjudication.md"


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


def pilot_gate_from_counts(counts: dict[str, Any]) -> dict[str, bool]:
    by_heavy = counts["rank_by_heavy_position"]
    return {
        "complete_scenes_12_of_12": counts["complete_scenes"] == 12,
        "finite_bounded_actions_24_of_24": counts["finite_bounded_actions"] == 24,
        "intended_contact_or_excitation_at_least_23_of_24": counts["intended_contact_or_excitation"] >= 23,
        "lane_and_reachability_24_of_24": counts["lane_and_reachability"] == 24,
        "zero_collision_identity_swap_fall_workspace_exit": (
            counts["collisions"]
            + counts["identity_swaps"]
            + counts["falls"]
            + counts["workspace_exits"]
            == 0
        ),
        "rank_correct_at_least_10_of_12": counts["rank_correct"] >= 10,
        "rank_front_heavy_at_least_5_of_6": by_heavy["front"]["correct"] >= 5,
        "rank_back_heavy_at_least_5_of_6": by_heavy["back"]["correct"] >= 5,
        "oracle_completion_at_least_10_of_12": counts["oracle_completion"] >= 10,
    }


def counts_from_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    complete = [row for row in rows if row.get("completed")]
    audits = [audit for row in complete for audit in row["probe_audits"].values()]
    by_heavy = {}
    for slot in ("front", "back"):
        subset = [row for row in complete if row["scene"]["heavy_slot"] == slot]
        by_heavy[slot] = {
            "correct": sum(bool(row["heavy_rank_correct"]) for row in subset),
            "total": len(subset),
        }
    return {
        "scenes": len(rows),
        "complete_scenes": len(complete),
        "probes": len(audits),
        "finite_bounded_actions": sum(bool(audit["finite_bounded_actions"]) for audit in audits),
        "intended_contact_or_excitation": sum(
            bool(audit["intended_contact_or_excitation"]) for audit in audits
        ),
        "both_candidates_contacted_or_excited_scenes": sum(
            all(bool(audit["intended_contact_or_excitation"]) for audit in row["probe_audits"].values())
            for row in complete
        ),
        "lane_and_reachability": sum(
            bool(probe["lane_and_reachability_continuous_pass"])
            for row in complete
            for probe in row["probes"]
        ),
        "collisions": sum(bool(audit["unintended_collision"]) for audit in audits),
        "identity_swaps": sum(bool(audit["identity_swap"]) for audit in audits),
        "falls": sum(bool(audit["fall"]) for audit in audits),
        "workspace_exits": sum(bool(audit["workspace_exit"]) for audit in audits),
        "track_losses": sum(bool(audit["unrecoverable_track_loss"]) for audit in audits),
        "rank_correct": sum(bool(row["heavy_rank_correct"]) for row in complete),
        "rank_by_heavy_position": by_heavy,
        "oracle_completion": sum(
            bool(row["oracle_completion"]["official_task_success"]) for row in complete
        ),
        "guard_triggered_probes": sum(
            bool(probe["predictive_lane_guard"]["triggered"])
            for row in complete
            for probe in row["probes"]
        ),
    }


def precontact_rgb_margin(
    probe: dict[str, Any], protocol: dict[str, Any], slot: str
) -> float:
    estimated_y = float(probe["estimated_initial_target_xyz_m"][1])
    lane = protocol["safe_center_lanes_m"][slot]["y"]
    return min(estimated_y - float(lane[0]), float(lane[1]) - estimated_y)


def validate_protocol_and_rows(
    protocol: dict[str, Any], result: dict[str, Any], host: dict[str, Any]
) -> tuple[list[str], list[dict[str, Any]]]:
    violations: list[str] = []
    manifest = protocol["variant1_pilot_manifest"]
    expected = {scene["scene_id"]: scene for scene in manifest}
    rows = result.get("rows", [])
    keys = [row.get("row_key") for row in rows]
    if len(rows) != 12 or set(keys) != set(expected) or len(keys) != len(set(keys)):
        violations.append("pilot rows do not exactly match the frozen 12-scene manifest")
    for row in rows:
        key = row.get("row_key")
        if key not in expected:
            continue
        if row.get("scene") != expected[key]:
            violations.append(f"{key}: embedded scene differs from frozen manifest")
        if not row.get("completed"):
            violations.append(f"{key}: incomplete row")
            continue
        if not row["exact_state_audit"].get("first_rgb_exact"):
            violations.append(f"{key}: exact first RGB audit failed")
        boundary = row.get("method_information_boundary", {})
        if boundary.get("mass_or_property_passed_to_guard_probe_or_score") is not False:
            violations.append(f"{key}: mass/property leakage")
        if boundary.get("simulator_pose_passed_to_guard_probe_or_score") is not False:
            violations.append(f"{key}: simulator-pose leakage")
        if boundary.get("oracle_privilege_evaluation_only") is not True:
            violations.append(f"{key}: oracle boundary missing")
        if set(row["probe_audits"]) != {"front", "back"} or len(row["probes"]) != 2:
            violations.append(f"{key}: probe cardinality mismatch")
        for probe in row["probes"]:
            if probe.get("forbidden_online_inputs_used"):
                violations.append(f"{key}/{probe.get('slot')}: forbidden online input")
            if probe.get("simulator_state_used_for_actions"):
                violations.append(f"{key}/{probe.get('slot')}: simulator state controlled actions")
            if probe.get("mass_or_property_used_for_actions"):
                violations.append(f"{key}/{probe.get('slot')}: mass/property controlled actions")
            guard = probe.get("predictive_lane_guard", {})
            if guard.get("admissible_rgb_geometry_only") is not True or guard.get("mass_or_property_input") is not False:
                violations.append(f"{key}/{probe.get('slot')}: guard information boundary failed")
    if result.get("validation_accessed") or result.get("confirmation_accessed"):
        violations.append("sealed validation or confirmation accessed")
    if result.get("protocol_sha256") != sha256(PROTOCOL_PATH):
        violations.append("result protocol hash mismatch")
    if result.get("execution_seal_sha256") != sha256(SEAL_PATH):
        violations.append("result execution-seal hash mismatch")
    if result.get("runner_sha256") != sha256(ROOT / "scripts/run_epoch9d_controller_pilot.py"):
        violations.append("result runner hash mismatch")
    if result.get("controller") != protocol["variant1"]:
        violations.append("result controller differs from frozen Variant 1")
    for row in rows:
        if not row.get("completed"):
            continue
        for audit in row["probe_audits"].values():
            trace_path = ROOT / audit["trace_path"]
            if not trace_path.exists() or sha256(trace_path) != audit["trace_sha256"]:
                violations.append(f"{row['row_key']}: trace hash mismatch")
    if host.get("runner_exit_code") != 0:
        violations.append("runner exit code was nonzero")
    if host.get("host_ram_ceiling_breached") is not False:
        violations.append("host RAM ceiling breached")
    if float(host.get("peak_host_ram_percent", math.inf)) >= float(host.get("host_ram_ceiling_percent", 82.0)):
        violations.append("host RAM peak reached the ceiling")
    if result.get("resource_monitor", {}).get("wsl_swap_used_peak_bytes") != 0:
        violations.append("WSL swap use detected")
    if host.get("scientific_result_sha256_after_runner") != sha256(RESULT_PATH):
        violations.append("host monitor result hash mismatch")
    return violations, rows


def main() -> int:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise FileExistsError("refusing to overwrite controller pilot adjudication")
    for path in (PROTOCOL_PATH, SEAL_PATH, RESULT_PATH, HOST_RESOURCE_PATH):
        if not path.exists():
            raise FileNotFoundError(path)
    protocol = load(PROTOCOL_PATH)
    seal = load(SEAL_PATH)
    result = load(RESULT_PATH)
    host = load(HOST_RESOURCE_PATH)
    bindings = {
        "protocol": sha256(PROTOCOL_PATH) == seal["protocol_sha256"],
        "runner": sha256(ROOT / seal["runner_path"]) == seal["runner_sha256"],
        "adjudicator": sha256(Path(__file__)) == seal["adjudicator_sha256"],
        "host_wrapper": sha256(ROOT / seal["host_wrapper_path"]) == seal["host_wrapper_sha256"],
        "original_runner": sha256(ROOT / seal["original_epoch9b_runner_path"]) == seal["original_epoch9b_runner_sha256"],
        "controller_freeze": sha256(ROOT / seal["original_controller_freeze_path"]) == seal["original_controller_freeze_sha256"],
        "calibration": sha256(ROOT / seal["calibration_path"]) == seal["calibration_sha256"],
    }
    violations, rows = validate_protocol_and_rows(protocol, result, host)
    if not all(bindings.values()):
        violations.extend(f"execution binding failed: {name}" for name, passed in bindings.items() if not passed)
    counts = counts_from_rows(rows)
    gates = pilot_gate_from_counts(counts)
    protocol_clean = not violations
    pilot_go = protocol_clean and all(gates.values())

    failed_lane_probes: list[dict[str, Any]] = []
    for row in rows:
        if not row.get("completed"):
            continue
        for probe in row["probes"]:
            if not probe["lane_and_reachability_continuous_pass"]:
                margin = precontact_rgb_margin(probe, load(ORIGINAL_PROTOCOL_PATH), probe["slot"])
                failed_lane_probes.append(
                    {
                        "row_key": row["row_key"],
                        "slot": probe["slot"],
                        "precontact_rgb_signed_y_margin_m": margin,
                        "within_frozen_adjustment_interval": 0.010 < margin <= 0.014,
                    }
                )
    non_lane_gates = {name: passed for name, passed in gates.items() if name != "lane_and_reachability_24_of_24"}
    adjustment_eligible = bool(
        protocol_clean
        and not pilot_go
        and failed_lane_probes
        and all(row["within_frozen_adjustment_interval"] for row in failed_lane_probes)
        and all(non_lane_gates.values())
    )
    if pilot_go:
        decision = "VARIANT1_SELECTED_FINAL_PANEL_AUTHORIZED"
    elif adjustment_eligible:
        decision = "VARIANT1_CONDITIONAL_ADJUSTMENT_AUTHORIZED"
    else:
        decision = "VARIANT1_PILOT_NOT_SELECTED"
    adjudication = {
        "schema_version": "epoch9d.controller_variant1_pilot_adjudication.v1",
        "adjudicated_at": timestamp(),
        "decision": decision,
        "pilot_selection_go": pilot_go,
        "conditional_adjustment_eligible": adjustment_eligible,
        "execution_bindings": bindings,
        "protocol_violations": violations,
        "counts": counts,
        "gates": gates,
        "failed_lane_probe_adjustment_audit": failed_lane_probes,
        "controller": protocol["variant1"],
        "controller_protocol_path": relative(PROTOCOL_PATH),
        "controller_protocol_sha256": sha256(PROTOCOL_PATH),
        "execution_seal_path": relative(SEAL_PATH),
        "execution_seal_sha256": sha256(SEAL_PATH),
        "result_path": relative(RESULT_PATH),
        "result_sha256": sha256(RESULT_PATH),
        "host_resource_path": relative(HOST_RESOURCE_PATH),
        "host_resource_sha256": sha256(HOST_RESOURCE_PATH),
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT_JSON, adjudication)
    markdown = f"""# Epoch 9D controller Variant 1 pilot adjudication

Decision: `{decision}`.

The sealed 12-scene pilot recorded {counts['finite_bounded_actions']}/24 finite bounded probes, {counts['intended_contact_or_excitation']}/24 intended contacts or excitations, {counts['lane_and_reachability']}/24 lane-safe probes, {counts['rank_correct']}/12 correct heavy rankings, and {counts['oracle_completion']}/12 oracle completions. Safety events (collision, identity swap, fall, workspace exit) total {counts['collisions'] + counts['identity_swaps'] + counts['falls'] + counts['workspace_exits']}.

Pilot selection is {str(pilot_go).lower()}. The single preregistered 10-to-14 mm guard adjustment is {str(adjustment_eligible).lower()}. No validation or confirmation identity was accessed.
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(json.dumps({"decision": decision, "counts": counts, "gates": gates}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
