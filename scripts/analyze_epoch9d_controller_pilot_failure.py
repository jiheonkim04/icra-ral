#!/usr/bin/env python3
"""Localize the sealed controller-pilot failures and close the bounded branch."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json


REPORTS = ROOT / "reports"
RESULT_PATH = REPORTS / "epoch9d_controller_development/variant1_pilot_result.json"
ADJUDICATION_PATH = REPORTS / "epoch9d_controller_variant1_pilot_adjudication.json"
PROTOCOL_PATH = REPORTS / "epoch9d_controller_development_protocol.json"
ORIGINAL_PROTOCOL_PATH = REPORTS / "epoch9b_v2_task_preservation_protocol.json"
OUTPUT_JSON = REPORTS / "epoch9d_controller_bounded_failure.json"
OUTPUT_MD = REPORTS / "epoch9d_controller_bounded_failure.md"
POST_RESPONSE_PHASES = {"retreat_side", "retreat_high", "return_neutral"}


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


def inside(protocol: dict[str, Any], position: np.ndarray, slot: str) -> bool:
    lane = protocol["safe_center_lanes_m"][slot]
    reach = protocol["reachable_center_envelope_m"]
    return bool(
        lane["x"][0] <= position[0] <= lane["x"][1]
        and lane["y"][0] <= position[1] <= lane["y"][1]
        and reach["z"][0] <= position[2] <= reach["z"][1]
    )


def lane_failure_audit(
    row: dict[str, Any], probe: dict[str, Any], protocol: dict[str, Any]
) -> dict[str, Any]:
    trace_path = ROOT / probe["trace_path"]
    with np.load(trace_path, allow_pickle=False) as trace:
        phases = np.asarray(trace["phase"]).astype(str)
        positions = np.asarray(trace["candidate_positions_eval_only"], dtype=np.float64)
    failures = []
    for index, pair in enumerate(positions):
        failed_slots = [
            slot for slot_index, slot in enumerate(("front", "back")) if not inside(protocol, pair[slot_index], slot)
        ]
        if failed_slots:
            failures.append((index, failed_slots))
    first_index, first_slots = failures[0]
    source = "inherited_from_prior_probe" if first_index == 0 else "new_exit_in_this_probe"
    events = probe["predictive_lane_guard"]["events"]
    return {
        "row_key": row["row_key"],
        "heavy_slot": row["scene"]["heavy_slot"],
        "spatial_stratum": row["scene"]["spatial_stratum"],
        "probe_slot": probe["slot"],
        "probe_order": row["scene"]["probe_order"],
        "source": source,
        "first_failure_index": int(first_index),
        "first_failure_phase": str(phases[first_index]),
        "first_failure_slots": first_slots,
        "first_failure_positions_m_eval_only": {
            slot: positions[first_index, slot_index].tolist()
            for slot_index, slot in enumerate(("front", "back"))
            if slot in first_slots
        },
        "failure_sample_count": len(failures),
        "guard_events": events,
        "trace_path": relative(trace_path),
        "trace_sha256": sha256(trace_path),
        "response_estimated_displacement_m": probe["response_estimated_displacement_m"],
    }


def main() -> int:
    if OUTPUT_JSON.exists() or OUTPUT_MD.exists():
        raise FileExistsError("refusing to overwrite bounded controller failure report")
    result = load(RESULT_PATH)
    adjudication = load(ADJUDICATION_PATH)
    development = load(PROTOCOL_PATH)
    original = load(ORIGINAL_PROTOCOL_PATH)
    if adjudication["pilot_selection_go"] or adjudication["conditional_adjustment_eligible"]:
        raise RuntimeError("failure closure is not authorized by the pilot adjudication")

    lane_rows = []
    rank_failures = []
    for row in result["rows"]:
        if not row["heavy_rank_correct"]:
            rank_failures.append(
                {
                    "row_key": row["row_key"],
                    "heavy_slot": row["scene"]["heavy_slot"],
                    "spatial_stratum": row["scene"]["spatial_stratum"],
                    "responses_m": row["responses_m"],
                    "back_response_minus_threshold_m": (
                        row["responses_m"]["back"]
                        - development["variant1"]["base_controller_config"]["back_heavy_threshold_m"]
                    ),
                }
            )
        for probe in row["probes"]:
            if not probe["lane_and_reachability_continuous_pass"]:
                lane_rows.append(lane_failure_audit(row, probe, original))
    new_exits = [row for row in lane_rows if row["source"] == "new_exit_in_this_probe"]
    inherited = [row for row in lane_rows if row["source"] == "inherited_from_prior_probe"]
    post_probe_pose_only = bool(
        new_exits and all(row["first_failure_phase"] in POST_RESPONSE_PHASES for row in new_exits)
    )
    variant2_authorized = bool(
        post_probe_pose_only
        and adjudication["counts"]["oracle_completion"] >= 10
        and adjudication["counts"]["collisions"]
        + adjudication["counts"]["identity_swaps"]
        + adjudication["counts"]["falls"]
        + adjudication["counts"]["workspace_exits"]
        == 0
    )
    if variant2_authorized:
        raise RuntimeError("Variant 2 remains authorized; refusing terminal closure")
    report = {
        "schema_version": "epoch9d.bounded_controller_failure.v1",
        "closed_at": timestamp(),
        "terminal_program_status": "ACTIVE_DYNAMIC_PROBE_SIGNAL_CONFIRMED_TASK_PRESERVATION_NOT_ACHIEVED",
        "causal_signal_status": "CAUSAL_SIGNAL_GO_REMAINS_ESTABLISHED",
        "task_preserving_controller_go": False,
        "variant1_pilot_decision": adjudication["decision"],
        "variant1_pilot_counts": adjudication["counts"],
        "failed_selection_gates": [name for name, passed in adjudication["gates"].items() if not passed],
        "conditional_adjustment_eligible": adjudication["conditional_adjustment_eligible"],
        "lane_failure_probe_rows": lane_rows,
        "new_lane_exit_count": len(new_exits),
        "inherited_lane_failure_probe_count": len(inherited),
        "new_lane_exit_phases": [row["first_failure_phase"] for row in new_exits],
        "post_probe_pose_is_sole_limiting_cause": post_probe_pose_only,
        "variant2_authorized": variant2_authorized,
        "variant2_rejection_reason": (
            "Both causal lane exits began during contact_verify_retract before the fixed five-step response window; "
            "they are not post-probe pose/restoration failures. The pilot also missed the overall and back-heavy "
            "ranking gates, which a later recovery stage cannot change without violating the frozen response window."
        ),
        "rank_failure_rows": rank_failures,
        "search_budget_audit": {
            "maximum_variants_beyond_original": 2,
            "variants_executed": 1,
            "maximum_fresh_pilot_scenes": 24,
            "fresh_pilot_scenes_executed": 12,
            "remaining_budget_is_not_authority": True,
            "reason_no_further_pilot": (
                "the only frozen Variant 1 adjustment condition failed and the frozen Variant 2 authorization condition failed"
            ),
        },
        "downstream_stage_authority": {
            "fresh_24_scene_controller_feasibility": False,
            "estimator_training": False,
            "validation": False,
            "official_closed_loop": False,
            "confirmation": False,
            "paper_package": False,
        },
        "protocol_violations": adjudication["protocol_violations"],
        "validation_accessed": False,
        "confirmation_accessed": False,
        "source_evidence": {
            "protocol": {"path": relative(PROTOCOL_PATH), "sha256": sha256(PROTOCOL_PATH)},
            "result": {"path": relative(RESULT_PATH), "sha256": sha256(RESULT_PATH)},
            "adjudication": {"path": relative(ADJUDICATION_PATH), "sha256": sha256(ADJUDICATION_PATH)},
        },
    }
    atomic_write_json(OUTPUT_JSON, report)
    markdown = f"""# Epoch 9D bounded controller failure

Terminal state: `ACTIVE_DYNAMIC_PROBE_SIGNAL_CONFIRMED_TASK_PRESERVATION_NOT_ACHIEVED`.

The causal mass signal remains established, but Variant 1 did not preserve the task under its sealed pilot gate. It achieved 24/24 finite bounded probes, 24/24 intended contacts or excitations, 12/12 scenes with both candidates excited, 12/12 oracle completions, and zero collision, identity-swap, fall, or workspace-exit events. It failed lane/reachability at 20/24, overall ranking at 9/12, and back-heavy ranking at 3/6.

The four failed lane audit rows reduce to two causal exits and two inherited failures. Both causal exits first crossed a lane boundary during `contact_verify_retract`, before the fixed response window. Their triggering RGB signed margins were below 10 mm, not in the frozen `(10, 14]` mm adjustment interval. The adjustment is therefore ineligible.

Variant 2 is also unauthorized. The evidence does not show post-probe pose/restoration as the limiting cause, and a recovery stage after the response window cannot repair the two failed ranking gates without changing the frozen score path. Unused numerical search budget does not create authority to rotate controllers.

No fresh 24-scene feasibility panel, estimator training, validation, official closed loop, confirmation, or paper package is authorized. Validation and confirmation identities remain untouched.
"""
    OUTPUT_MD.write_text(markdown, encoding="utf-8")
    print(json.dumps({
        "terminal_program_status": report["terminal_program_status"],
        "new_lane_exit_phases": report["new_lane_exit_phases"],
        "variant2_authorized": report["variant2_authorized"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
