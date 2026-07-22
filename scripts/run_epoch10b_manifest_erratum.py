"""Choice-free Epoch 10B manifest completion and superseding re-adjudication.

The original 1,287-row JSONL, preregistration, mechanics report, and terminal
adjudication are immutable inputs.  This runner can execute only the nine
primary keys frozen by the hashed erratum decision, and it writes them to a
separate JSONL before logically unioning the two logs for re-adjudication.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Mapping

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPO_ROOT))

from scripts.run_epoch10b_fresh_controller_assay import (
    HORIZONS,
    _append_jsonl,
    _array_sha256,
    _canonical_sha256,
    _sha256_file,
    _target_for_demo,
    _write_json,
    adjudicate_certification,
    control_actions,
    execute_fresh_branch,
)


DECISION_PATH = REPO_ROOT / "reports/epoch10b_erratum_frozen_decision.json"
DECISION_HASH_PATH = REPO_ROOT / "reports/epoch10b_erratum_frozen_decision.sha256"
EXPECTED_DIFF_PATH = REPO_ROOT / "reports/epoch10b_erratum_expected_key_diff.json"
ORIGINAL_PREREG_PATH = REPO_ROOT / "reports/epoch10b_assay_preregistration.json"
CORRECTED_PREREG_PATH = (
    REPO_ROOT
    / "runs/epoch10b_probation_recovery_archive_20260722/collision_corrected_preregistration_pre_probation.json"
)
ORIGINAL_RAW_PATH = REPO_ROOT / "runs/epoch10b_mechanics_certification/branches.jsonl"
RUN_DIR = REPO_ROOT / "runs/epoch10b_manifest_erratum"
ERRATUM_RAW_PATH = RUN_DIR / "frame60_primary_panel.jsonl"
ACTIVE_PATH = RUN_DIR / "active_branch.json"
INFRASTRUCTURE_ATTEMPTS_PATH = RUN_DIR / "infrastructure_attempts.jsonl"
PANEL_STATE_PATH = RUN_DIR / "panel_state.json"
HOST_MONITOR_PATH = RUN_DIR / "host_monitor.json"
UNION_MANIFEST_PATH = REPO_ROOT / "reports/epoch10b_erratum_union_analysis_manifest.json"
MECHANICS_PATH = REPO_ROOT / "reports/epoch10b_erratum_mechanics_reanalysis.json"
SUPERSEDING_PATH = REPO_ROOT / "reports/epoch10b_erratum_superseding_adjudication.json"

ORIGINAL_PREREG_SHA256 = "5be914187a1374cfa527dea8bde110e121c54c27e4aea6f7d78c43afe94072c4"
CORRECTED_PREREG_SHA256 = "2e5f2fe58153ed7dad17a88a11c41377275b7ebd56e7f71d989b30d073b8b6b6"
ORIGINAL_RAW_SHA256 = "a2f2992d03fae52177408f057ba311b4f522b3955d91ba78dcd74a165e55ced7"
FROZEN_DECISION_SHA256 = "cab840da177eaf99a9fa9f34b9814adc7464a273f512215bb7566ea7468a64a0"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        try:
            rows.append(json.loads(line))
        except Exception as exc:
            raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {type(exc).__name__}: {exc}") from exc
    return rows


def _is_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_is_finite(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return all(_is_finite(item) for item in value)
    return True


def _key_ledger_sha256(keys: list[str]) -> str:
    payload = "".join(f"{key}\n" for key in sorted(set(keys))).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _verify_frozen_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    expected_sidecar = DECISION_HASH_PATH.read_text(encoding="utf-8").split()[0]
    if expected_sidecar != FROZEN_DECISION_SHA256 or _sha256_file(DECISION_PATH) != FROZEN_DECISION_SHA256:
        raise RuntimeError("frozen erratum decision hash mismatch")
    if _sha256_file(ORIGINAL_PREREG_PATH) != ORIGINAL_PREREG_SHA256:
        raise RuntimeError("original frozen preregistration changed")
    if _sha256_file(CORRECTED_PREREG_PATH) != CORRECTED_PREREG_SHA256:
        raise RuntimeError("collision-corrected preregistration archive changed")
    if _sha256_file(ORIGINAL_RAW_PATH) != ORIGINAL_RAW_SHA256:
        raise RuntimeError("original 1,287-row raw log changed")

    decision = _load_json(DECISION_PATH)
    expected = _load_json(EXPECTED_DIFF_PATH)
    original = _load_json(ORIGINAL_PREREG_PATH)
    corrected = _load_json(CORRECTED_PREREG_PATH)
    if decision["selected_path"] != "PATH_B_CHOICE_FREE_COMPLETION_OF_ONE_PRE_REGISTERED_PRIMARY_PANEL":
        raise RuntimeError("frozen decision does not authorize Path B")
    if int(decision["authorized_panel"]["reverse_rows_authorized"]) != 0:
        raise RuntimeError("the frozen decision unexpectedly authorizes reverse rows")
    if corrected.get("canonical_payload_sha256") != "439e25f4ad0502aa39faa73ac51001fb728f5669527231b0ea66c84f4b5fe5da":
        raise RuntimeError("corrected preregistration canonical hash mismatch")
    return decision, expected, original, corrected


def _authorized_state(corrected: Mapping[str, Any], state_id: str) -> dict[str, Any]:
    matches = [dict(row) for row in corrected["states"] if str(row["state_id"]) == state_id]
    if len(matches) != 1:
        raise RuntimeError(f"authorized state must occur exactly once, got {len(matches)}")
    return matches[0]


def run_panel(camera_size: int) -> dict[str, Any]:
    decision, expected_diff, _original, corrected = _verify_frozen_inputs()
    panel = decision["authorized_panel"]
    state_id = str(panel["state_id"])
    state = _authorized_state(corrected, state_id)
    for key in ("frame", "phase", "phase_index", "registered_seed", "state_sha256", "expert_action_sha256"):
        if state[key] != panel[key]:
            raise RuntimeError(f"authorized state field mismatch: {key}")

    task_matches = [
        row
        for row in corrected["tasks"]
        if row["suite"] == state["suite"] and int(row["task_id"]) == int(state["task_id"])
    ]
    if len(task_matches) != 1:
        raise RuntimeError("authorized task source is not unique")
    task = task_matches[0]
    if _sha256_file(Path(task["hdf5_path"])) != task["hdf5_sha256"]:
        raise RuntimeError("authorized HDF5 source hash mismatch")
    if _sha256_file(Path(task["bddl_file"])) != task["bddl_sha256"]:
        raise RuntimeError("authorized BDDL source hash mismatch")

    expected_keys = list(panel["frozen_control_order"])
    expected_branch_keys = [f"certification|primary|{state_id}|{control}" for control in expected_keys]
    if expected_branch_keys != expected_diff["uniquely_identified_missing_primary_panel"]["expected_keys_in_execution_order"]:
        raise RuntimeError("decision and expected-key ledger disagree")
    if hashlib.sha256(("\n".join(expected_branch_keys) + "\n").encode("utf-8")).hexdigest() != panel[
        "expected_keys_sha256"
    ]:
        raise RuntimeError("authorized expected-key order hash mismatch")

    existing = _load_jsonl(ERRATUM_RAW_PATH)
    completed = {str(row["branch_key"]): row for row in existing}
    if len(completed) != len(existing):
        raise RuntimeError("erratum log contains duplicate keys")
    unexpected = sorted(set(completed).difference(expected_branch_keys))
    if unexpected:
        raise RuntimeError(f"erratum log contains unexpected keys: {unexpected}")
    if ACTIVE_PATH.is_file():
        pending_bytes = ACTIVE_PATH.read_bytes()
        try:
            pending_record: Any = json.loads(pending_bytes.decode("utf-8"))
        except Exception as exc:
            pending_record = {
                "unparseable": True,
                "bytes": len(pending_bytes),
                "sha256": hashlib.sha256(pending_bytes).hexdigest(),
                "error": f"{type(exc).__name__}: {exc}",
            }
        interrupted = {
            "classification": "CONTROLLED_HOST_MONITOR_INTERRUPTION_WITHOUT_MATERIALIZED_ROW",
            "pending_record": pending_record,
            "completed_erratum_rows": len(existing),
            "missing_key_only_resume": True,
            "scientific_zero_assigned": False,
        }
        interrupted["attempt_sha256"] = _canonical_sha256(interrupted)
        prior_attempts = _load_jsonl(INFRASTRUCTURE_ATTEMPTS_PATH)
        if interrupted["attempt_sha256"] not in {row.get("attempt_sha256") for row in prior_attempts}:
            _append_jsonl(INFRASTRUCTURE_ATTEMPTS_PATH, interrupted)

    with h5py.File(task["hdf5_path"], "r") as handle:
        demo = handle["data"][state["demo_name"]]
        states = np.asarray(demo["states"], dtype=np.float64)
        actions = np.asarray(demo["actions"], dtype=np.float64)
    frame = int(state["frame"])
    if _array_sha256(states[frame]) != state["state_sha256"]:
        raise RuntimeError("authorized registered state hash mismatch")
    if _array_sha256(actions[frame]) != state["expert_action_sha256"]:
        raise RuntimeError("authorized expert action hash mismatch")

    target = _target_for_demo(task, states, actions, int(camera_size))
    controls = control_actions(actions[frame], state["phase"], state_id)
    started = time.monotonic()
    new_rows = 0
    for control_name in expected_keys:
        branch_key = f"certification|primary|{state_id}|{control_name}"
        if branch_key in completed:
            continue
        _write_json(
            ACTIVE_PATH,
            {
                "branch_key": branch_key,
                "control": control_name,
                "execution_pass": "primary",
                "started_at_unix": time.time(),
                "frozen_decision_sha256": FROZEN_DECISION_SHA256,
            },
        )
        row = execute_fresh_branch(
            task=task,
            states=states,
            actions=actions,
            target=target,
            frame=frame,
            first_action=controls[control_name],
            design=str(corrected["constructor"]["selected_design"]),
            branch_key=branch_key,
            camera_size=int(camera_size),
            horizon=max(HORIZONS),
            registered_seed=int(state["registered_seed"]),
        )
        row.update(
            {
                "state_id": state_id,
                "suite": state["suite"],
                "task_id": int(state["task_id"]),
                "demo_name": state["demo_name"],
                "demo_cluster": state["demo_cluster"],
                "phase": state["phase"],
                "control": control_name,
                "execution_pass": "primary",
                "nominal_first_action": controls["nominal_a"],
                "delivered_action_delta_l2": float(np.linalg.norm(controls[control_name] - controls["nominal_a"])),
                "delivered_action_delta_linf": float(np.max(np.abs(controls[control_name] - controls["nominal_a"]))),
                "erratum_decision_sha256": FROZEN_DECISION_SHA256,
                "erratum_source_preregistration_sha256": CORRECTED_PREREG_SHA256,
            }
        )
        _append_jsonl(ERRATUM_RAW_PATH, row)
        completed[branch_key] = row
        new_rows += 1
        ACTIVE_PATH.unlink(missing_ok=True)
        print(json.dumps({"completed": len(completed), "branch_key": branch_key, "valid": row.get("valid")}), flush=True)

    rows = _load_jsonl(ERRATUM_RAW_PATH)
    actual_keys = [str(row["branch_key"]) for row in rows]
    complete = bool(
        len(rows) == 9
        and actual_keys == expected_branch_keys
        and len(set(actual_keys)) == 9
        and all(row.get("valid") is True and _is_finite(row) for row in rows)
    )
    result = {
        "schema_version": 1,
        "campaign": "epoch10b_manifest_adjudicator_erratum",
        "status": "ERRATUM_PANEL_COMPLETE" if complete else "ERRATUM_PANEL_INCOMPLETE_RESUMABLE",
        "row_count": len(rows),
        "new_row_count": new_rows,
        "expected_row_count": 9,
        "exact_expected_key_order": actual_keys == expected_branch_keys,
        "all_valid_finite_unique": complete,
        "active_branch_present": ACTIVE_PATH.is_file(),
        "raw_path": str(ERRATUM_RAW_PATH.relative_to(REPO_ROOT)),
        "raw_sha256": _sha256_file(ERRATUM_RAW_PATH) if ERRATUM_RAW_PATH.is_file() else None,
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "original_raw_sha256_unchanged": _sha256_file(ORIGINAL_RAW_PATH) == ORIGINAL_RAW_SHA256,
    }
    result["canonical_payload_sha256"] = _canonical_sha256(result)
    _write_json(PANEL_STATE_PATH, result)
    return result


def _union_preregistration(original: Mapping[str, Any], corrected: Mapping[str, Any]) -> dict[str, Any]:
    union_prereg = copy.deepcopy(corrected)
    original_reverse_ids = {
        str(row["state_id"]) for row in original["states"] if bool(row.get("reverse_order_duplicate"))
    }
    if len(original_reverse_ids) != 16:
        raise RuntimeError(f"original reverse registration must name 16 state IDs, got {len(original_reverse_ids)}")
    for row in union_prereg["states"]:
        row["reverse_order_duplicate"] = str(row["state_id"]) in original_reverse_ids
    state_ids = [str(row["state_id"]) for row in union_prereg["states"]]
    if len(state_ids) != 128 or len(set(state_ids)) != 128:
        raise RuntimeError("union preregistration does not contain 128 distinct primary states")
    if sum(bool(row["reverse_order_duplicate"]) for row in union_prereg["states"]) != 16:
        raise RuntimeError("union preregistration does not retain exactly 16 original reverse panels")
    return union_prereg


def analyze_union() -> dict[str, Any]:
    decision, expected_diff, original, corrected = _verify_frozen_inputs()
    original_rows = _load_jsonl(ORIGINAL_RAW_PATH)
    erratum_rows = _load_jsonl(ERRATUM_RAW_PATH)
    expected_keys = decision["authorized_panel"]["frozen_control_order"]
    state_id = decision["authorized_panel"]["state_id"]
    expected_branch_keys = [f"certification|primary|{state_id}|{control}" for control in expected_keys]
    actual_erratum_keys = [str(row["branch_key"]) for row in erratum_rows]
    execution_valid = bool(
        len(erratum_rows) == 9
        and actual_erratum_keys == expected_branch_keys
        and len(set(actual_erratum_keys)) == 9
        and all(row.get("valid") is True and _is_finite(row) for row in erratum_rows)
        and not ACTIVE_PATH.is_file()
    )
    if not execution_valid:
        raise RuntimeError("erratum panel is not nine valid, finite, unique rows in frozen order")
    if not HOST_MONITOR_PATH.is_file():
        raise RuntimeError("host monitor is missing")
    monitor = _load_json(HOST_MONITOR_PATH)
    if not bool(monitor.get("pass")):
        raise RuntimeError("host monitor did not pass")

    rows = original_rows + erratum_rows
    branch_keys = [str(row["branch_key"]) for row in rows]
    if len(rows) != 1296 or len(set(branch_keys)) != 1296:
        raise RuntimeError("logical union does not contain 1,296 unique rows")
    expected_ledger = expected_diff["erratum_expected_set_rule"]["expected_sorted_key_ledger_sha256"]
    if _key_ledger_sha256(branch_keys) != expected_ledger:
        raise RuntimeError("logical union key ledger does not match the frozen erratum ledger")

    union_prereg = _union_preregistration(original, corrected)
    adjudication = adjudicate_certification(rows, union_prereg)
    mechanics = {
        "schema_version": 1,
        "campaign": "epoch10b_manifest_adjudicator_erratum",
        "status": "ASSAY_CERTIFIED" if adjudication["certified"] else "ASSAY_INVALID",
        "original_raw_path": str(ORIGINAL_RAW_PATH.relative_to(REPO_ROOT)),
        "original_raw_sha256": _sha256_file(ORIGINAL_RAW_PATH),
        "original_raw_rows": len(original_rows),
        "erratum_raw_path": str(ERRATUM_RAW_PATH.relative_to(REPO_ROOT)),
        "erratum_raw_sha256": _sha256_file(ERRATUM_RAW_PATH),
        "erratum_raw_rows": len(erratum_rows),
        "logical_union_rows": len(rows),
        "logical_union_unique_keys": len(set(branch_keys)),
        "logical_union_sorted_key_ledger_sha256": _key_ledger_sha256(branch_keys),
        "manifest_primary_state_count": len(union_prereg["states"]),
        "manifest_distinct_primary_state_count": len({row["state_id"] for row in union_prereg["states"]}),
        "manifest_reverse_state_count": sum(bool(row["reverse_order_duplicate"]) for row in union_prereg["states"]),
        "host_monitor_path": str(HOST_MONITOR_PATH.relative_to(REPO_ROOT)),
        "host_monitor_sha256": _sha256_file(HOST_MONITOR_PATH),
        "checkpoint_actions_queried": 0,
        "checkpoint_outcomes_opened": False,
        "closed_loop_success_labels_opened": False,
        **adjudication,
    }
    mechanics["canonical_payload_sha256"] = _canonical_sha256(mechanics)
    _write_json(MECHANICS_PATH, mechanics)

    union_manifest = {
        "schema_version": 1,
        "campaign": "epoch10b_manifest_adjudicator_erratum",
        "union_is_logical_only": True,
        "original_raw_immutable": True,
        "original_raw_path": str(ORIGINAL_RAW_PATH.relative_to(REPO_ROOT)),
        "original_raw_bytes": ORIGINAL_RAW_PATH.stat().st_size,
        "original_raw_rows": len(original_rows),
        "original_raw_sha256": _sha256_file(ORIGINAL_RAW_PATH),
        "erratum_raw_path": str(ERRATUM_RAW_PATH.relative_to(REPO_ROOT)),
        "erratum_raw_bytes": ERRATUM_RAW_PATH.stat().st_size,
        "erratum_raw_rows": len(erratum_rows),
        "erratum_raw_sha256": _sha256_file(ERRATUM_RAW_PATH),
        "logical_union_rows": len(rows),
        "logical_union_unique_keys": len(set(branch_keys)),
        "logical_union_sorted_key_ledger_sha256": _key_ledger_sha256(branch_keys),
        "expected_sorted_key_ledger_sha256": expected_ledger,
        "missing_keys": [],
        "extra_keys": [],
        "primary_states": 128,
        "primary_rows": 1152,
        "reverse_states": 16,
        "reverse_rows": 144,
        "analysis_preregistration_rule": "128 primary states from the collision-corrected pre-probation preregistration; 16 reverse registrations retained from the original frozen preregistration, as required by the erratum decision.",
        "original_preregistration_sha256": ORIGINAL_PREREG_SHA256,
        "corrected_preregistration_archive_sha256": CORRECTED_PREREG_SHA256,
        "frozen_decision_sha256": FROZEN_DECISION_SHA256,
    }
    union_manifest["canonical_payload_sha256"] = _canonical_sha256(union_manifest)
    _write_json(UNION_MANIFEST_PATH, union_manifest)

    selected = next(
        (row for row in adjudication["horizon_audits"] if row["horizon"] == adjudication["selected_horizon"]),
        None,
    )
    horizon_summaries = []
    for audit in adjudication["horizon_audits"]:
        endpoint_summaries = {}
        for endpoint_name, endpoint_audit in audit["endpoint_audits"].items():
            endpoint_summaries[endpoint_name] = {
                "eligible": endpoint_audit["eligible"],
                "twin_noise_floor": endpoint_audit["twin_noise_floor"],
                "harmful_minus_nominal_grouped": endpoint_audit["harmful_minus_nominal_grouped"],
                "medium_minus_small_descriptive_stress": endpoint_audit[
                    "medium_minus_small_descriptive_stress"
                ],
                "responsive_state_count": endpoint_audit["responsive_state_count"],
                "responsive_counts_by_suite": endpoint_audit["responsive_counts_by_suite"],
                "nominal_duplicate_score_spearman": endpoint_audit["nominal_duplicate_score_spearman"],
                "nominal_duplicate_score_icc": endpoint_audit["nominal_duplicate_score_icc"],
            }
        horizon_summaries.append(
            {
                "horizon": audit["horizon"],
                "complete_state_count": audit["complete_state_count"],
                "twin_subset_count": audit["twin_subset_count"],
                "pairs_within_1e_8": audit["pairs_within_1e_8"],
                "maximum_twin_state_l2": audit["maximum_twin_state_l2"],
                "suite_pairs_within_1e_8": audit["suite_pairs_within_1e_8"],
                "reverse_order_duplicate_count": audit["reverse_order_duplicate_count"],
                "reverse_order_max_state_l2": audit["reverse_order_max_state_l2"],
                "twin_gate_pass": audit["twin_gate_pass"],
                "selected_endpoint": audit["selected_endpoint"],
                "responsiveness_gate_pass": audit["responsiveness_gate_pass"],
                "pass": audit["pass"],
                "endpoint_summaries": endpoint_summaries,
            }
        )
    superseding = {
        "schema_version": 1,
        "campaign": "epoch10b_manifest_adjudicator_erratum",
        "status": "SUPERSEDING_ERRATUM_ADJUDICATION",
        "supersedes_without_replacing": "reports/epoch10b_assay_adjudication.json",
        "original_terminal_state_preserved": "EPOCH10B_ICAE_ASSAY_INVALID_ROUTE_CLOSED",
        "terminal_state": (
            "EPOCH10B_ERRATUM_MANIFEST_COMPLETED_CERTIFIED_STAGE0_AUTHORIZED"
            if adjudication["certified"]
            else "EPOCH10B_ERRATUM_PROTOCOL_DEFECT_ROUTE_CLOSED"
        ),
        "path": "PATH_B_CHOICE_FREE_COMPLETION_OF_ONE_PRE_REGISTERED_PRIMARY_PANEL",
        "questions": {
            "mechanics_rows_executed_validly": {
                "pass": all(row.get("valid") is True and _is_finite(row) for row in rows),
                "valid_finite_rows": sum(row.get("valid") is True and _is_finite(row) for row in rows),
                "total_rows": len(rows),
                "environment_close_called_rows": sum(bool(row.get("cleanup", {}).get("close_called")) for row in rows),
                "erratum_environment_close_called_rows": sum(
                    bool(row.get("cleanup", {}).get("close_called")) for row in erratum_rows
                ),
                "erratum_error_rows": sum(bool(row.get("error")) for row in erratum_rows),
            },
            "frozen_sample_manifest_complete": {
                "pass": len(union_prereg["states"]) == len({row["state_id"] for row in union_prereg["states"]}) == 128,
                "distinct_primary_states": 128,
                "required_primary_states": 128,
                "primary_rows": 1152,
                "reverse_states": 16,
                "reverse_rows": 144,
            },
            "twin_fidelity_gate": {
                "pass": bool(selected and selected["twin_gate_pass"]),
                "nominal_pairs": selected["twin_subset_count"] if selected else None,
                "pairs_within_1e_8": selected["pairs_within_1e_8"] if selected else None,
                "maximum_state_l2": selected["maximum_twin_state_l2"] if selected else None,
                "suite_pairs_within_1e_8": selected["suite_pairs_within_1e_8"] if selected else None,
                "reverse_order_panels": selected["reverse_order_duplicate_count"] if selected else None,
                "reverse_order_maximum_l2": selected["reverse_order_max_state_l2"] if selected else None,
            },
            "original_endpoint_horizon_gate": {
                "pass": bool(selected and selected["responsiveness_gate_pass"]),
                "selected_horizon": adjudication["selected_horizon"],
                "selected_endpoint": adjudication["selected_endpoint"],
                "selected_endpoint_audit": (
                    selected["endpoint_audits"][adjudication["selected_endpoint"]]
                    if selected and adjudication["selected_endpoint"]
                    else None
                ),
            },
        },
        "expected_observed_keys": {
            "expected": 1296,
            "observed": len(set(branch_keys)),
            "missing": 0,
            "extra": 0,
        },
        "original_values": {
            "distinct_primary_states": 127,
            "primary_rows": 1143,
            "reverse_states_in_raw": 16,
            "reverse_rows_in_raw": 144,
            "reported_reverse_order_duplicate_count": 15,
        },
        "corrected_values": {
            "distinct_primary_states": 128,
            "primary_rows": 1152,
            "reverse_states": 16,
            "reverse_rows": 144,
            "reverse_order_duplicate_count": selected["reverse_order_duplicate_count"] if selected else None,
        },
        "all_horizon_gate_summary": horizon_summaries,
        "erratum_resource_integrity": {
            "host_monitor_pass": monitor["pass"],
            "peak_host_ram_percent": monitor["peak_host_ram_percent"],
            "peak_wsl_memory_used_bytes": monitor["peak_wsl_memory_used_bytes"],
            "peak_wsl_swap_used_bytes": monitor["peak_wsl_swap_used_bytes"],
            "peak_gpu_vram_used_mib": monitor["peak_gpu_vram_used_mib"],
            "protected_manifests_unchanged": monitor["protected_manifests_unchanged"],
            "material_system_event_count": monitor["material_system_event_count"],
            "wsl_shutdown_unconditional": monitor["wsl_shutdown_unconditional"],
        },
        "engineering_only_rule_change": "Manifest indexing now joins duplicate reverse registration by logical OR instead of silently taking the later false flag. No scientific definition, score, threshold, endpoint, horizon, action, or existing row changed.",
        "new_empirical_support_claimed": False,
        "checkpoint_outcomes_remained_sealed": True,
        "frozen_decision_sha256": FROZEN_DECISION_SHA256,
        "mechanics_reanalysis_path": str(MECHANICS_PATH.relative_to(REPO_ROOT)),
        "mechanics_reanalysis_sha256": _sha256_file(MECHANICS_PATH),
        "union_manifest_path": str(UNION_MANIFEST_PATH.relative_to(REPO_ROOT)),
        "union_manifest_sha256": _sha256_file(UNION_MANIFEST_PATH),
    }
    superseding["canonical_payload_sha256"] = _canonical_sha256(superseding)
    _write_json(SUPERSEDING_PATH, superseding)
    return superseding


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("preflight", "run-panel", "analyze"))
    parser.add_argument("--camera-size", type=int, default=256)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.mode == "preflight":
        decision, expected, original, corrected = _verify_frozen_inputs()
        result = {
            "status": "ERRATUM_PREFLIGHT_PASS",
            "decision_path": decision["selected_path"],
            "missing_key_count": expected["diff_before_erratum"]["missing_key_count"],
            "original_states": len(original["states"]),
            "corrected_distinct_states": len({row["state_id"] for row in corrected["states"]}),
        }
    elif args.mode == "run-panel":
        result = run_panel(args.camera_size)
    else:
        result = analyze_union()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
