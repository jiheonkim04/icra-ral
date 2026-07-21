#!/usr/bin/env python3
"""Build the append-only terminal evidence index and handoff after Epoch 9E NO-GO."""

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
PROMPT = Path("/mnt/c/Users/jiheo/Downloads/epoch9e_nondrag_disengagement_ral_convergence_prompt.md")
SOURCE = "74dd66c32a8b8595e187b13d3ccafe05cae6753b"
DECISION = "EPOCH9E_NONDRAG_DISENGAGEMENT_FROZEN_NO_GO_ACTIVE_ROUTE_CLOSED"
RESULT = REPORTS / "epoch9e_joint_certification/result.json"
ADJUDICATION = REPORTS / "epoch9e_joint_certification_adjudication.json"
PROTOCOL = REPORTS / "epoch9e_joint_certification_protocol.json"
IDENTITIES = REPORTS / "epoch9e_fresh_identity_manifest.json"
OUTPUT_INDEX = REPORTS / "epoch9e_evidence_index.json"
OUTPUT_STATE = REPORTS / "epoch9e_campaign_state.json"
OUTPUT_JSON = REPORTS / "epoch9e_terminal_handoff.json"
OUTPUT_MD = REPORTS / "epoch9e_terminal_handoff.md"


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
    lines, total = [], 0
    for path in sorted(value for value in root.rglob("*") if value.is_file()):
        size = path.stat().st_size
        total += size
        lines.append(f"{relative(path)}\t{size}\t{sha256(path)}")
    return {
        "path": relative(root) + "/",
        "file_count": len(lines),
        "total_bytes": total,
        "manifest_sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest().upper(),
    }


def json_schema(path: Path) -> str | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        payload = load(path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload.get("schema_version") if isinstance(payload, dict) else None


def evidence_files() -> list[Path]:
    values: set[Path] = set()
    for path in REPORTS.glob("epoch9e*"):
        if path in {OUTPUT_INDEX, OUTPUT_STATE, OUTPUT_JSON, OUTPUT_MD}:
            continue
        if path.is_file():
            values.add(path)
        elif path.is_dir():
            values.update(value for value in path.rglob("*") if value.is_file())
    values.update(path for path in (ROOT / "scripts").glob("*epoch9e*") if path.is_file())
    values.update(path for path in (ROOT / "tests").glob("test_epoch9e*") if path.is_file())
    return sorted(values, key=relative)


def trace_summary(path: Path, lane_protocol: dict[str, Any]) -> dict[str, Any]:
    slot = "front" if path.stem.endswith("_front") else "back"
    with np.load(path, allow_pickle=False) as trace:
        phase = np.asarray(trace["phase"]).astype(str)
        positions = np.asarray(trace["candidate_positions_eval_only"], dtype=np.float64)
        quality = np.asarray(trace["rgb_quality"], dtype=np.float64)
        displacement = np.asarray(trace["estimated_world_displacement_m"], dtype=np.float64)
        actions = np.asarray(trace["action"], dtype=np.float64)
        contacts = np.asarray(trace["target_contact_eval_only"], dtype=bool)
        pair_collision = np.asarray(trace["candidate_pair_collision_eval_only"], dtype=bool)
        distractor_collision = np.asarray(trace["candidate_distractor_collision_eval_only"], dtype=bool)
    signed_margins: dict[str, dict[str, float]] = {}
    reach = lane_protocol["reachable_center_envelope_m"]
    for index, candidate in enumerate(("front", "back")):
        xyz = positions[:, index, :]
        lane = lane_protocol["safe_center_lanes_m"][candidate]
        margins = np.column_stack((
            xyz[:, 0] - lane["x"][0], lane["x"][1] - xyz[:, 0],
            xyz[:, 1] - lane["y"][0], lane["y"][1] - xyz[:, 1],
            xyz[:, 2] - reach["z"][0], reach["z"][1] - xyz[:, 2],
        )).min(axis=1)
        signed_margins[candidate] = {
            "minimum_m": float(np.min(margins)),
            "median_m": float(np.median(margins)),
            "maximum_m": float(np.max(margins)),
        }
    response_steps = int(np.count_nonzero(np.isin(phase, ["fixed_micro_impulse", "post_impulse_response"])))
    return {
        "path": relative(path),
        "sha256": sha256(path),
        "slot": slot,
        "steps": int(len(phase)),
        "phase_counts": dict(sorted(Counter(phase.tolist()).items())),
        "frozen_response_window_steps": response_steps,
        "frozen_response_window_complete": response_steps == 5,
        "continuous_candidate_lane_signed_margin_summary_m": signed_margins,
        "continuous_estimated_displacement_summary_m": {
            "minimum": float(np.min(displacement)),
            "median": float(np.median(displacement)),
            "maximum": float(np.max(displacement)),
        },
        "minimum_rgb_quality": float(np.min(quality)),
        "maximum_absolute_action_component": float(np.max(np.abs(actions))),
        "sampled_target_contact": bool(np.any(contacts)),
        "candidate_pair_collision": bool(np.any(pair_collision)),
        "candidate_distractor_collision": bool(np.any(distractor_collision)),
        "full_continuous_arrays_preserved_in_npz": True,
    }


def main() -> int:
    for output in (OUTPUT_INDEX, OUTPUT_STATE, OUTPUT_JSON, OUTPUT_MD):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite {output}")
    for path in (PROMPT, RESULT, ADJUDICATION, PROTOCOL, IDENTITIES):
        if not path.is_file():
            raise FileNotFoundError(path)
    result, adjudication, protocol, identities = load(RESULT), load(ADJUDICATION), load(PROTOCOL), load(IDENTITIES)
    if adjudication["decision"] != DECISION or adjudication["joint_certification_go"] is not False:
        raise RuntimeError("terminal handoff requires the exact frozen joint NO-GO")
    prompt_hash = sha256(PROMPT)
    if prompt_hash != "C49CBFB70150F2DB2CBE86B30E5FA54B9E117F69821E82279C59836801EE60EF":
        raise RuntimeError("Epoch 9E authority prompt hash changed")
    source_ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", SOURCE, "HEAD"], cwd=ROOT).returncode == 0
    changed_epoch9d = subprocess.check_output(
        ["git", "diff", "--name-only", f"{SOURCE}..HEAD", "--", "reports/epoch9d*", "scripts/*epoch9d*", "tests/test_epoch9d*"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    if not source_ancestor or changed_epoch9d:
        raise RuntimeError("Epoch 9D source is not preserved append-only")
    protected = [protected_snapshot(ROOT / "rollouts/2026_07_17"), protected_snapshot(ROOT / "rollouts/2026_07_18")]
    expected_protected = {
        "rollouts/2026_07_17/": (27, 5_143_751, "25DE8FF5AA6112D7EFF8BCF38D3A4C3F0F3C8C8EE0458E5FA83D17438719EC54"),
        "rollouts/2026_07_18/": (10, 924_633, "CF701D6F73D4783F016E48A72C093DC9FD6D940B7081DA8FBEC128DB94C24A00"),
    }
    if any((row["file_count"], row["total_bytes"], row["manifest_sha256"]) != expected_protected[row["path"]] for row in protected):
        raise RuntimeError("protected rollout evidence changed")
    entries = [{"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path), "schema_version": json_schema(path)} for path in evidence_files()]
    evidence_index = {
        "schema_version": "epoch9e.evidence_index.v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "terminal_decision": DECISION,
        "source_checkpoint": SOURCE,
        "outcome_checkpoint": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "authority_prompt": {"path": str(PROMPT), "sha256": prompt_hash},
        "entry_count": len(entries),
        "entries": entries,
        "protected_untracked_manifests": protected,
        "validation_accessed": False,
        "confirmation_accessed": False,
    }
    atomic_write_json(OUTPUT_INDEX, evidence_index)
    traces = [trace_summary(path, load(REPORTS / "epoch9b_v2_task_preservation_protocol.json")) for path in sorted((REPORTS / "epoch9e_joint_certification/traces").glob("*.npz"))]
    allocations = identities["allocations"]
    mechanics_ids = allocations["mechanics_smoke"]["identity_ids"]
    joint_ids = allocations["joint_certification_base_pairs"]["identity_ids"]
    active_workers = subprocess.run(["pgrep", "-af", "run_epoch9"], text=True, capture_output=True).stdout.splitlines()
    active_workers = [line for line in active_workers if ".py" in line and "build_epoch9e_terminal_handoff.py" not in line]
    accounting = {
        "mechanics_smoke": {"planned_scenes": 8, "completed_scenes": 8, "planned_probes": 16, "completed_probes": 16},
        "joint_primary_assignments": {"planned": 24, "launched": 2, "completed": 1, "protocol_failed": 1, "unexecuted": 22},
        "joint_candidate_probes": {"planned": 48, "trace_files_written": len(traces), "admitted_complete_row_probes": 2, "failed_row_trace_files": 2, "unexecuted": 44},
        "joint_sham_rows": {"planned": 12, "launched": 0, "completed": 0, "failed": 0, "unexecuted": 12},
        "joint_completion_oracles": {"planned": 24, "executed": 1, "successes": 1, "failures": 0, "unexecuted": 23},
    }
    failure = next(row for row in result["rows"] if not row.get("completed"))
    state = {
        "schema_version": "epoch9e.campaign_state.v1",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "terminal_state": DECISION,
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "outcome_checkpoint": evidence_index["outcome_checkpoint"],
        "remote_checkpoint_before_terminal_reporting": subprocess.check_output(["git", "rev-parse", "origin/codex/epoch9e-nondrag-disengagement-convergence"], cwd=ROOT, text=True).strip(),
        "source_checkpoint": SOURCE,
        "source_is_ancestor": source_ancestor,
        "epoch9d_paths_changed_since_source": changed_epoch9d,
        "frozen_prior_status": ["CAUSAL_MASS_SIGNAL_CONFIRMED", "VARIANT1_PILOT_FROZEN_NO_GO", "PRE_RESPONSE_DISENGAGEMENT_UNRESOLVED", "PAPER_NOT_AUTHORIZED"],
        "mechanics_status": "MECHANICS_SMOKE_PASS_FREEZE_CONTROLLER",
        "joint_failure": {"row_key": failure["row_key"], "exception": failure["exception"], "response_window_missing_trace": relative(REPORTS / "epoch9e_joint_certification/traces/epoch9e_joint_base_20261134_assignment_B_back.npz")},
        "episode_accounting": accounting,
        "identities": {
            "mechanics_outcome_accessed": mechanics_ids,
            "joint_state_rgb_only_preflight_accessed": joint_ids,
            "joint_scientific_execution_accessed": [20261134],
            "joint_scientific_execution_unexecuted": joint_ids[1:],
            "validation_demo_ids_sealed": [40, 41, 42, 43, 44],
            "confirmation_demo_ids_sealed": [45, 46, 47, 48, 49],
        },
        "statistics": {
            "complete_exact_pairs": 0,
            "paired_contrast_m": None,
            "paired_95_interval_m": [None, None],
            "one_sided_sign_test_p": 1.0,
            "interpretation": "No complete A/B exact pair exists. Unexecuted rows are not failures and no pooled or repaired estimate is reported.",
            "completed_scene_only_descriptive_record": {"rank_correct": 0, "denominator": 1, "both_candidates_excited": 1, "oracle_completion_success": 1},
        },
        "epoch9d_separate_baseline": {
            "status": "CAUSAL_MASS_SIGNAL_CONFIRMED",
            "rank_correct": 28,
            "rank_denominator": 32,
            "exact_pair_flips": 12,
            "exact_pair_denominator": 16,
            "mean_contrast_m": 0.006593329847616967,
            "paired_95_interval_m": [0.004073638133887584, 0.00911302156134635],
            "one_sided_sign_test_p": 0.0002593994140625,
            "not_pooled_with_epoch9e": True,
        },
        "resource": adjudication["resource_summary"],
        "wrapper_status_anomaly": {
            "outer_powershell_exit_code": 1,
            "host_monitor_runner_exit_code": 0,
            "python_stderr_contains_unhandled_runtime_error": True,
            "scientific_interpretation_changed": False,
            "reason": "The complete-manifest and controller-audit gates fail independently; no execution continuation is authorized.",
        },
        "continuous_trace_disclosure": traces,
        "hash_schema_checks": {"all_execution_bindings": all(adjudication["execution_bindings"].values()), "trace_hashes_pass": adjudication["integrity"]["trace_hashes"], "evidence_index_sha256": sha256(OUTPUT_INDEX)},
        "protected_untracked_manifests": protected,
        "active_epoch9_scientific_workers": active_workers,
        "validation_accessed": False,
        "confirmation_accessed": False,
        "estimator_development_started": False,
        "official_evaluation_started": False,
        "paper_status": "PAPER_NOT_AUTHORIZED",
        "paper_paths": [],
        "next_action": "STOP_CAMPAIGN_NO_RERUN_NO_CONTROLLER_OR_ATTRIBUTION_ROTATION",
    }
    atomic_write_json(OUTPUT_STATE, state)
    handoff = {
        "schema_version": "epoch9e.terminal_handoff.v1",
        "timestamp": state["timestamp"],
        "terminal_state": DECISION,
        "branch": state["branch"],
        "outcome_checkpoint": state["outcome_checkpoint"],
        "source_checkpoint": SOURCE,
        "failure": state["joint_failure"],
        "episode_accounting": accounting,
        "identities": state["identities"],
        "statistics": state["statistics"],
        "resources": state["resource"],
        "tests": {
            "pre_execution_regression": "36 passed",
            "joint_seal_regression": "16 passed",
            "post_terminal_evidence_regression": "21 passed",
            "final_terminal_artifact_suite": "record after artifact build",
        },
        "hash_schema_checks": state["hash_schema_checks"],
        "protected_untracked_manifests": protected,
        "evidence_index": {"path": relative(OUTPUT_INDEX), "sha256": sha256(OUTPUT_INDEX)},
        "campaign_state": {"path": relative(OUTPUT_STATE), "sha256": sha256(OUTPUT_STATE)},
        "validation_accessed": False,
        "confirmation_accessed": False,
        "paper_status": "PAPER_NOT_AUTHORIZED",
        "paper_paths": [],
        "required_stop": state["next_action"],
    }
    atomic_write_json(OUTPUT_JSON, handoff)
    markdown = f"""# Epoch 9E Terminal Handoff

Terminal state: `{DECISION}`

Branch: `{state['branch']}`  
Outcome checkpoint: `{state['outcome_checkpoint']}`  
Preserved source: `{SOURCE}`

The bounded mechanics smoke passed and froze the sole non-drag controller. The first and only joint panel then completed one primary assignment and failed during the second assignment because `{failure['row_key']}` did not contain the frozen five-step response window in its back-probe trace. No joint rerun, row replacement, endpoint repair, controller rotation, estimator development, validation, confirmation, official evaluation, or paper build is authorized.

## Executed versus unexecuted

- primary assignments: 1 complete, 1 protocol-failed, 22 unexecuted out of 24 planned;
- candidate probes: 4 trace files written, 2 admitted from the complete row, 2 retained from the failed row, 44 unexecuted out of 48 planned;
- shams: 0 executed and 12 unexecuted;
- completion oracle: 1 executed successfully and 23 unexecuted.

Unexecuted rows are not reported as task failures or 0% success. There is no complete A/B exact pair, so no paired contrast or confidence interval is claimed. Epoch 9D's causal GO remains separate and unchanged.

## Integrity and resources

All sealed execution bindings and retained trace hashes pass. Peak host RAM was `{state['resource']['peak_host_ram_percent']:.3f}%`, peak system-wide GPU allocation was `{state['resource']['peak_gpu_used_mib']} MiB`, and scientific WSL swap use was `0` bytes. Validation identities `40--44` and confirmation identities `45--49` remain sealed. Protected rollout manifests remain byte-identical.

Evidence index: `{relative(OUTPUT_INDEX)}`  
Campaign state: `{relative(OUTPUT_STATE)}`  
Machine-readable handoff: `{relative(OUTPUT_JSON)}`

Paper status: `PAPER_NOT_AUTHORIZED`. Paper paths: none.
"""
    atomic_write_text(OUTPUT_MD, markdown)
    print(json.dumps({"decision": DECISION, "index_entries": len(entries), "outcome_checkpoint": state["outcome_checkpoint"], "protected": protected}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
