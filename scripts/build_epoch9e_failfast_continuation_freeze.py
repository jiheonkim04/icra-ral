#!/usr/bin/env python3
"""Freeze the append-only Epoch 9E fail-fast correction and continuation scope."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json


REPORTS = ROOT / "reports"
AUTHORITY = Path("/mnt/c/Users/jiheo/Downloads/epoch9e_failfast_runner_continuation_authorization.md")
START = "4f57ecb94a3c84e0a5889bc0bd60cbd53ad415e8"
PROTOCOL = REPORTS / "epoch9e_joint_certification_protocol.json"
ORIGINAL = REPORTS / "epoch9b_v2_task_preservation_protocol.json"
RESULT = REPORTS / "epoch9e_joint_certification/result.json"
ADJUDICATION = REPORTS / "epoch9e_joint_certification_adjudication.json"
TRACE_ROOT = REPORTS / "epoch9e_joint_certification/traces"
CONTROLLER = ROOT / "scripts/epoch9e_nondrag_controller.py"
RUNNER = ROOT / "scripts/run_epoch9e_joint_certification.py"
OLD_ADJUDICATOR = ROOT / "scripts/adjudicate_epoch9e_joint_certification.py"
OLD_HOST = ROOT / "scripts/run_epoch9e_joint_certification_host.ps1"
OLD_SEAL = REPORTS / "epoch9e_joint_execution_seal.json"
PREFLIGHT = REPORTS / "epoch9e_exact_pair_preflight.json"
OUTPUT_AUTH = REPORTS / "epoch9e_failfast_continuation_authorization.json"
OUTPUT_CORRECTION = REPORTS / "epoch9e_failfast_root_cause_and_scope_correction.json"
OUTPUT_CORRECTION_MD = REPORTS / "epoch9e_failfast_root_cause_and_scope_correction.md"
OUTPUT_SENSITIVITY = REPORTS / "epoch9e_missing_pair_sensitivity_protocol.json"
OUTPUT_SENSITIVITY_MD = REPORTS / "epoch9e_missing_pair_sensitivity_protocol.md"
OUTPUT_SCHEDULE = REPORTS / "epoch9e_continuation_schedule_audit.json"
OUTPUT_SCHEDULE_MD = REPORTS / "epoch9e_continuation_schedule_audit.md"


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


def protected_snapshot(root: Path) -> dict[str, Any]:
    files = sorted(path for path in root.rglob("*") if path.is_file())
    lines = [f"{relative(path)}\t{path.stat().st_size}\t{sha256(path)}" for path in files]
    return {
        "path": relative(root) + "/",
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "manifest_sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest().upper(),
    }


def trace_audit(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as trace:
        phase = np.asarray(trace["phase"]).astype(str)
        verify = np.asarray(trace["rgb_displacement_pixels"], dtype=np.float64)[phase == "contact_verify_observe"]
        actions = np.asarray(trace["action"], dtype=np.float64)
        contact = np.asarray(trace["target_contact_eval_only"], dtype=bool)
    response_steps = int(np.count_nonzero(np.isin(phase, ["fixed_micro_impulse", "post_impulse_response"])))
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "steps": int(len(phase)),
        "phase_counts": dict(sorted(Counter(phase.tolist()).items())),
        "response_window_steps": response_steps,
        "response_window_valid": response_steps == 5,
        "contact_verify_rgb_displacement_pixels_min": float(np.min(verify)) if len(verify) else None,
        "contact_verify_rgb_displacement_pixels_max": float(np.max(verify)) if len(verify) else None,
        "sampled_physical_contact": bool(np.any(contact)),
        "actions_finite_and_bounded": bool(np.isfinite(actions).all() and np.all(np.abs(actions) <= 1.0)),
    }


def main() -> int:
    outputs = (OUTPUT_AUTH, OUTPUT_CORRECTION, OUTPUT_CORRECTION_MD, OUTPUT_SENSITIVITY, OUTPUT_SENSITIVITY_MD, OUTPUT_SCHEDULE, OUTPUT_SCHEDULE_MD)
    if any(path.exists() for path in outputs):
        raise FileExistsError("refusing to overwrite fail-fast continuation freeze")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != START:
        raise RuntimeError(f"continuation freeze must start at {START}, found {head}")
    authority_hash = sha256(AUTHORITY)
    if authority_hash != "A70E137D2C92E0395F47E13CF99692702F0FDB86E2CC714B5C219D66618EC9E7":
        raise RuntimeError("continuation authority hash mismatch")
    protocol, original, result, adjudication, preflight = load(PROTOCOL), load(ORIGINAL), load(RESULT), load(ADJUDICATION), load(PREFLIGHT)
    if [row["row_key"] for row in result["rows"]] != [
        "primary:epoch9e_joint_base_20261134_assignment_A",
        "primary:epoch9e_joint_base_20261134_assignment_B",
    ]:
        raise RuntimeError("historical interrupted result shape changed")
    if preflight["scientific_outcomes_accessed"] or any(row["actions_executed"] != 0 or row["reward_done_success_accessed"] for row in preflight["rows"]):
        raise RuntimeError("pending schedule preflight crossed the outcome boundary")
    traces = [trace_audit(path) for path in sorted(TRACE_ROOT.glob("*.npz"))]
    missing = next(row for row in traces if row["path"].endswith("assignment_B_back.npz"))
    if len(traces) != 4 or missing["response_window_valid"] or not missing["sampled_physical_contact"]:
        raise RuntimeError("immutable missing-window trace classification changed")
    existing_scientific_names = [path.name for path in TRACE_ROOT.glob("*.npz")]
    if any(str(identity) in name for identity in range(20261135, 20261146) for name in existing_scientific_names):
        raise RuntimeError("pending scientific trace already exists")
    assignments = protocol["assignments"]
    pending_primary = [row for row in assignments if int(row["base_identity_id"]) != 20261134]
    pending_shams = protocol["sham_control"]["manifest"]
    protected = [protected_snapshot(ROOT / "rollouts/2026_07_17"), protected_snapshot(ROOT / "rollouts/2026_07_18")]
    expected = {
        "rollouts/2026_07_17/": (27, 5_143_751, "25DE8FF5AA6112D7EFF8BCF38D3A4C3F0F3C8C8EE0458E5FA83D17438719EC54"),
        "rollouts/2026_07_18/": (10, 924_633, "CF701D6F73D4783F016E48A72C093DC9FD6D940B7081DA8FBEC128DB94C24A00"),
    }
    if any((row["file_count"], row["total_bytes"], row["manifest_sha256"]) != expected[row["path"]] for row in protected):
        raise RuntimeError("protected manifest changed")
    frozen_hashes = {
        "controller": {"path": relative(CONTROLLER), "sha256": sha256(CONTROLLER)},
        "joint_protocol": {"path": relative(PROTOCOL), "sha256": sha256(PROTOCOL)},
        "original_runner": {"path": relative(RUNNER), "sha256": sha256(RUNNER)},
        "original_adjudicator": {"path": relative(OLD_ADJUDICATOR), "sha256": sha256(OLD_ADJUDICATOR)},
        "original_host_wrapper": {"path": relative(OLD_HOST), "sha256": sha256(OLD_HOST)},
        "original_execution_seal": {"path": relative(OLD_SEAL), "sha256": sha256(OLD_SEAL)},
        "interrupted_result": {"path": relative(RESULT), "sha256": sha256(RESULT)},
        "interrupted_adjudication": {"path": relative(ADJUDICATION), "sha256": sha256(ADJUDICATION)},
        "existing_traces": traces,
    }
    authority = {
        "schema_version": "epoch9e.failfast_continuation_authorization.v1",
        "recorded_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "authority_path": str(AUTHORITY),
        "authority_sha256": authority_hash,
        "starting_checkpoint": START,
        "authorized_change": "one integrity-only fail-fast runner/adjudicator repair",
        "controller_or_scientific_change_authorized": False,
        "rerun_20261134_authorized": False,
        "replacement_pair_authorized": False,
        "allowance_consumed": False,
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT_AUTH, authority)
    correction = {
        "schema_version": "epoch9e.failfast_root_cause_scope_correction.v1",
        "recorded_at": authority["recorded_at"],
        "prospective_status": [
            "EPOCH9E_JOINT_EXECUTION_INTERRUPTED_BY_FAIL_FAST_RUNNER_DEFECT",
            "ACTIVE_ROUTE_NOT_YET_ADJUDICABLE",
            "PAPER_NOT_AUTHORIZED",
        ],
        "historical_terminal_artifacts_edited": False,
        "scientific_miss": {
            "row_key": "primary:epoch9e_joint_base_20261134_assignment_B",
            "probe": "back",
            "physical_contact": True,
            "frozen_transition_threshold_pixels": 0.55,
            "observed_verification_range_pixels": [missing["contact_verify_rgb_displacement_pixels_min"], missing["contact_verify_rgb_displacement_pixels_max"]],
            "frozen_response_window_steps": 0,
            "immutable": True,
        },
        "implementation_defect": {
            "missing_window_detection_line": {"path": relative(RUNNER), "line": 207},
            "whole_batch_abort_lines": {"path": relative(RUNNER), "lines": [489, 490]},
            "description": "run_primary converted the missing response window into a failed row, then main raised for every failed row instead of retaining the classified scientific miss and continuing untouched keys.",
        },
        "fixed_handling_20261134": {
            "assignment_A": "completed; observed rank miss; successful oracle completion; never rerun",
            "assignment_B_front": "complete mechanics trace retained; never rerun",
            "assignment_B_back": "contacted but invalid/missing response window; never scored or rerun",
            "assignment_B": "failed/incomplete primary assignment in fixed intention-to-treat denominators",
            "exact_pair": "fixed adverse/nonflip for binary flip and sign evidence; excluded from physical-contrast mean/CI",
        },
        "frozen_hashes": frozen_hashes,
        "protected_untracked_manifests": protected,
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT_CORRECTION, correction)
    a_back_response = float(result["rows"][0]["responses_m"]["back"])
    response_limit = float(original["v2_absolute_displacement_rule"]["limit_m"])
    sensitivity = {
        "schema_version": "epoch9e.missing_pair_sensitivity_protocol.v1",
        "frozen_at": authority["recorded_at"],
        "missing_base_identity_id": 20261134,
        "contrast_definition": "assignment B back-light response minus assignment A back-heavy response",
        "observed_assignment_A_back_heavy_response_m": a_back_response,
        "assignment_B_missing_response_physically_admissible_range_m": [0.0, response_limit],
        "missing_contrast_physically_admissible_range_m": [-a_back_response, response_limit - a_back_response],
        "worst_case_missing_contrast_m": -a_back_response,
        "rules": {
            "complete_case": "Report mean, median, Student-t 95% interval, exact sample count, sign counts, and centered position/lane/order HC3 interval using complete physical contrasts only; maximum n is 11.",
            "binary_fixed_denominator": "Count base 20261134 as nonpositive and nonflip in the fixed 12-pair exact sign and flip evidence.",
            "worst_case_augmented": "Append the preregistered worst-case missing contrast (B response 0 minus observed A response) to the complete contrasts; require its 12-pair Student-t interval lower endpoint and centered HC3 interval lower endpoint to remain strictly positive.",
            "significance": "Require a one-sided exact binomial sign-test p < 0.01 with positives counted over fixed n=12, so the missing pair can never help.",
            "tipping_point": "Solve and disclose the missing contrast at which the augmented 12-pair Student-t lower endpoint first reaches zero over the admissible range; PASS requires the worst-case bound to be above that tipping point.",
            "reporting": "Never label the sensitivity-augmented estimate as an observed 12-pair physical mean or CI.",
        },
        "paired_gate_pass_requires": [
            "complete-case mean positive and 95% Student-t lower endpoint > 0",
            "complete-case centered HC3 estimate positive and lower endpoint > 0",
            "fixed-denominator one-sided exact sign p < 0.01 with missing pair nonpositive",
            "worst-case augmented Student-t lower endpoint > 0",
            "worst-case augmented centered HC3 estimate positive and lower endpoint > 0",
        ],
        "outcomes_after_20261134_accessed_before_freeze": False,
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT_SENSITIVITY, sensitivity)
    schedule = {
        "schema_version": "epoch9e.failfast_continuation_schedule_audit.v1",
        "frozen_at": authority["recorded_at"],
        "starting_checkpoint": START,
        "original_result_rows_preserved": [row["row_key"] for row in result["rows"]],
        "original_trace_count": len(traces),
        "pending_primary_rows": [{"row_key": f"primary:{row['scene_id']}", "base_identity_id": row["base_identity_id"], "assignment": row["assignment"]} for row in pending_primary],
        "pending_sham_rows": [{"row_key": f"sham:{row['sham_id']}", "base_identity_id": row["base_identity_id"], "assignment": row["assignment"]} for row in pending_shams],
        "pending_primary_count": len(pending_primary),
        "pending_sham_count": len(pending_shams),
        "primary_identity_order": [row["base_identity_id"] for row in pending_primary],
        "primary_assignment_order": [row["assignment"] for row in pending_primary],
        "sham_identity_order": [row["base_identity_id"] for row in pending_shams],
        "sham_assignment_order": [row["assignment"] for row in pending_shams],
        "pending_scientific_outcomes_opened": False,
        "state_rgb_only_preflight_disclosed": True,
        "preflight_actions_executed": sum(row["actions_executed"] for row in preflight["rows"]),
        "preflight_reward_done_success_accessed": any(row["reward_done_success_accessed"] for row in preflight["rows"]),
        "never_recompute_keys": [row["row_key"] for row in result["rows"]],
        "serial": True,
        "host_ram_ceiling_percent": 82.0,
        "wsl_swap_used_peak_bytes": 0,
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    if len(pending_primary) != 22 or len(pending_shams) != 12:
        raise RuntimeError("frozen continuation schedule size mismatch")
    atomic_write_json(OUTPUT_SCHEDULE, schedule)
    atomic_write_text(OUTPUT_CORRECTION_MD, f"""# Epoch 9E Fail-Fast Root Cause and Scope Correction

Prospective state: `EPOCH9E_JOINT_EXECUTION_INTERRUPTED_BY_FAIL_FAST_RUNNER_DEFECT`; `ACTIVE_ROUTE_NOT_YET_ADJUDICABLE`; `PAPER_NOT_AUTHORIZED`.

The immutable scientific miss is `primary:epoch9e_joint_base_20261134_assignment_B`, back probe. Physical contact occurred, but ordinary RGB verification stayed at `{missing['contact_verify_rgb_displacement_pixels_min']:.6f}--{missing['contact_verify_rgb_displacement_pixels_max']:.6f} px`, below the frozen `0.55 px` transition threshold, so the trace contains zero frozen response-window steps.

The separate implementation defect is at `{relative(RUNNER)}:489--490`: every failed row raised and aborted the batch even though the aggregate gates had remaining miss allowance. Historical artifacts remain untouched. Base `20261134` is never rerun or replaced; Assignment B stays failed/incomplete, and the pair is adverse/nonflip for binary evidence but absent from physical-contrast means and intervals.
""")
    atomic_write_text(OUTPUT_SENSITIVITY_MD, f"""# Epoch 9E Missing-Pair Sensitivity Protocol

This rule was frozen before opening outcomes for bases `20261135--20261145`.

The observed Assignment A back-heavy response is `{a_back_response:.12f} m`. The frozen admissible response range is `[0, {response_limit:.3f}] m`, so the missing contrast range is `[{(-a_back_response):.12f}, {(response_limit-a_back_response):.12f}] m`. The worst-case value is `{(-a_back_response):.12f} m`.

Only complete physical contrasts enter the reported physical mean, Student-t interval, and complete-case HC3 model. Base `20261134` is nonpositive/nonflip in fixed 12-pair binary evidence. A paired PASS additionally requires the fixed-denominator sign test and both worst-case augmented Student-t and HC3 lower endpoints to remain positive. The augmented sensitivity result is never described as an observed 12-pair physical estimate.
""")
    atomic_write_text(OUTPUT_SCHEDULE_MD, f"""# Epoch 9E Frozen Continuation Schedule Audit

The append-only continuation contains exactly `{len(pending_primary)}` untouched primary assignments from base IDs `20261135--20261145`, followed by the original `{len(pending_shams)}` sham rows, all in the committed protocol order. Base `20261134` and its four traces are excluded from execution and retained unchanged.

The exact-pair preflight used zero actions and accessed no reward, done, or success outcome. No scientific trace for the pending identities exists. Validation `40--44` and confirmation `45--49` remain sealed.
""")
    print(json.dumps({"authority_sha256": authority_hash, "pending_primary": len(pending_primary), "pending_shams": len(pending_shams), "missing_verify_range": correction["scientific_miss"]["observed_verification_range_pixels"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
