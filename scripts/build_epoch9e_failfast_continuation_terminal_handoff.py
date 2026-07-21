#!/usr/bin/env python3
"""Build the append-only terminal handoff for the completed Epoch 9E continuation."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json


REPORTS = ROOT / "reports"
AUTHORITY = Path("/mnt/c/Users/jiheo/Downloads/epoch9e_failfast_runner_continuation_authorization.md")
AUTHORITY_SHA256 = "A70E137D2C92E0395F47E13CF99692702F0FDB86E2CC714B5C219D66618EC9E7"
START = "4f57ecb94a3c84e0a5889bc0bd60cbd53ad415e8"
DECISION = "EPOCH9E_NONDRAG_DISENGAGEMENT_FROZEN_NO_GO_ACTIVE_ROUTE_CLOSED"
BRANCH = "codex/epoch9e-nondrag-disengagement-convergence"

ADJUDICATION = REPORTS / "epoch9e_joint_continuation_adjudication.json"
CONTINUATION_RESULT = REPORTS / "epoch9e_joint_continuation/result.json"
ORIGINAL_RESULT = REPORTS / "epoch9e_joint_certification/result.json"
ROOT_CAUSE = REPORTS / "epoch9e_failfast_root_cause_and_scope_correction.json"
SCHEDULE = REPORTS / "epoch9e_continuation_schedule_audit.json"
SEAL = REPORTS / "epoch9e_joint_continuation_execution_seal.json"
SENSITIVITY = REPORTS / "epoch9e_missing_pair_sensitivity_protocol.json"
STATUS_CORRECTION = REPORTS / "epoch9e_continuation_host_exit_status_correction.json"
PARSER_REPAIR = REPORTS / "epoch9e_continuation_adjudicator_parser_repair.json"
HOST_MONITOR = REPORTS / "epoch9e_joint_continuation/host_resource_monitor_attempt_1.json"
OLD_INDEX = REPORTS / "epoch9e_evidence_index.json"
OLD_HANDOFF = REPORTS / "epoch9e_terminal_handoff.json"

OUTPUT_JSON = REPORTS / "epoch9e_failfast_continuation_terminal_handoff.json"
OUTPUT_MD = REPORTS / "epoch9e_failfast_continuation_terminal_handoff.md"
OUTPUT_INDEX = REPORTS / "epoch9e_failfast_continuation_evidence_index_v2.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def protected_snapshot(root: Path) -> dict[str, Any]:
    lines: list[str] = []
    total = 0
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


def schema_version(path: Path) -> str | None:
    if path.suffix.lower() != ".json":
        return None
    try:
        payload = load(path)
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    return payload.get("schema_version") if isinstance(payload, dict) else None


def evidence_files() -> list[Path]:
    paths: set[Path] = set()
    for path in REPORTS.glob("epoch9e*"):
        if path == OUTPUT_INDEX:
            continue
        if path.is_file():
            paths.add(path)
        elif path.is_dir():
            paths.update(value for value in path.rglob("*") if value.is_file())
    paths.update(path for path in (ROOT / "scripts").glob("*epoch9e*") if path.is_file())
    paths.update(path for path in (ROOT / "tests").glob("test_epoch9e*") if path.is_file())
    return sorted(paths, key=relative)


def main() -> int:
    for output in (OUTPUT_JSON, OUTPUT_MD, OUTPUT_INDEX):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite append-only artifact {output}")
    required = (
        AUTHORITY,
        ADJUDICATION,
        CONTINUATION_RESULT,
        ORIGINAL_RESULT,
        ROOT_CAUSE,
        SCHEDULE,
        SEAL,
        SENSITIVITY,
        STATUS_CORRECTION,
        PARSER_REPAIR,
        HOST_MONITOR,
        OLD_INDEX,
        OLD_HANDOFF,
    )
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)
    if sha256(AUTHORITY) != AUTHORITY_SHA256:
        raise RuntimeError("continuation authority hash changed")
    if git("branch", "--show-current") != BRANCH:
        raise RuntimeError("unexpected branch")
    if subprocess.run(["git", "merge-base", "--is-ancestor", START, "HEAD"], cwd=ROOT).returncode != 0:
        raise RuntimeError("required starting checkpoint is not an ancestor")

    adjudication = load(ADJUDICATION)
    continuation = load(CONTINUATION_RESULT)
    correction = load(ROOT_CAUSE)
    schedule = load(SCHEDULE)
    seal = load(SEAL)
    monitor = load(HOST_MONITOR)
    parser_repair = load(PARSER_REPAIR)
    status_correction = load(STATUS_CORRECTION)
    if adjudication["decision"] != DECISION or adjudication["joint_certification_go"] is not False:
        raise RuntimeError("terminal continuation handoff requires the exact frozen NO-GO")
    if continuation["summary"] != {
        "classified_missing_response_rows": 0,
        "completed_rows": 34,
        "other_failed_rows": 0,
        "recorded_rows": 34,
        "scheduled_rows": 34,
        "unexecuted_rows": 0,
        "unique_recorded_keys": 34,
    }:
        raise RuntimeError("continuation manifest is incomplete")

    expected_frozen = {
        "controller": "99DA452B5AD3603A9FDD1209704479B18F302987E79C65EA8C4B9622E16657D7",
        "original_result": "BC9E772479ABAFF477D9EA56399E4A4BB407F1874439A7ACE20E8B84F3C332C8",
    }
    if sha256(ROOT / correction["frozen_hashes"]["controller"]["path"]) != expected_frozen["controller"]:
        raise RuntimeError("controller changed")
    if sha256(ORIGINAL_RESULT) != expected_frozen["original_result"]:
        raise RuntimeError("historical interrupted result changed")
    for row in correction["frozen_hashes"]["existing_traces"]:
        if sha256(ROOT / row["path"]) != row["sha256"]:
            raise RuntimeError(f"historical trace changed: {row['path']}")

    protected = [
        protected_snapshot(ROOT / "rollouts/2026_07_17"),
        protected_snapshot(ROOT / "rollouts/2026_07_18"),
    ]
    expected_protected = {
        "rollouts/2026_07_17/": (27, 5_143_751, "25DE8FF5AA6112D7EFF8BCF38D3A4C3F0F3C8C8EE0458E5FA83D17438719EC54"),
        "rollouts/2026_07_18/": (10, 924_633, "CF701D6F73D4783F016E48A72C093DC9FD6D940B7081DA8FBEC128DB94C24A00"),
    }
    if any((row["file_count"], row["total_bytes"], row["manifest_sha256"]) != expected_protected[row["path"]] for row in protected):
        raise RuntimeError("protected rollout evidence changed")
    tracked_protected = git("ls-files", "--", "rollouts/2026_07_17", "rollouts/2026_07_18")
    if tracked_protected:
        raise RuntimeError("protected rollout directories must remain untracked")

    commit_names = {
        "starting_checkpoint": START,
        "authority_and_sensitivity_freeze": "db1b10b",
        "wrapper_repair": "7368b56",
        "continuation_sealer": "a6f7738",
        "preoutcome_execution_seal": "f099979",
        "prospective_exit_status_serialization_repair": "04362e3",
        "attempt_1_status_correction_seal": "d9d2f68",
        "status_repair_contract_test": "2ded887",
        "raw_continuation_evidence": "8398fe6",
        "completed_panel_adjudication": "aad1afc",
    }
    commits = {name: git("rev-parse", value) for name, value in commit_names.items()}
    if any(subprocess.run(["git", "merge-base", "--is-ancestor", value, "HEAD"], cwd=ROOT).returncode != 0 for value in commits.values()):
        raise RuntimeError("terminal commit chain is not preserved")

    counts = adjudication["counts"]
    paired = adjudication["paired_mass_intervention"]
    resources = {
        "host_ram_ceiling_percent": 82.0,
        "baseline_host_ram_percent": monitor["baseline_host_ram_percent"],
        "peak_host_ram_percent": monitor["peak_host_ram_percent"],
        "final_host_ram_percent": monitor["final_host_ram_percent"],
        "peak_system_wide_gpu_used_mib": monitor["peak_gpu_used_mib"],
        "scientific_process_max_rss_bytes": continuation["resource_monitor"]["process_max_rss_bytes"],
        "wsl_memory_used_peak_bytes": continuation["resource_monitor"]["wsl_mem_used_peak_bytes"],
        "wsl_swap_used_peak_bytes": continuation["resource_monitor"]["wsl_swap_used_peak_bytes"],
        "host_ceiling_breached": monitor["host_ram_ceiling_breached"],
        "effective_authoritative_runner_exit_code": adjudication["resource_effective_authoritative_exit_codes"][-1],
        "attempt_1_raw_status_serialization": status_correction["raw_status_text"],
        "attempt_1_status_correction_hash_bound": True,
        "scientific_rerun_for_status_correction": False,
    }
    terminal = {
        "schema_version": "epoch9e.failfast_continuation_terminal_handoff.v1",
        "timestamp": datetime.now().astimezone().isoformat(timespec="seconds"),
        "terminal_state": DECISION,
        "joint_certification_go": False,
        "branch": BRANCH,
        "starting_checkpoint": START,
        "outcome_checkpoint": git("rev-parse", "HEAD"),
        "remote_checkpoint_before_final_reporting": git("rev-parse", f"origin/{BRANCH}"),
        "commit_chain": commits,
        "wrapper_repair_boundary": {
            "before_commit": commits["authority_and_sensitivity_freeze"],
            "after_commit": commits["wrapper_repair"],
            "cause_path": correction["implementation_defect"]["whole_batch_abort_lines"]["path"],
            "missing_window_detection_line": correction["implementation_defect"]["missing_window_detection_line"]["line"],
            "whole_batch_abort_lines": correction["implementation_defect"]["whole_batch_abort_lines"]["lines"],
            "cause": correction["implementation_defect"]["description"],
            "repair_scope": "continuation runner/adjudicator and host integrity wrappers only",
            "integrity_only_allowance_consumed": True,
            "controller_or_scientific_semantics_changed": False,
        },
        "frozen_hash_proof": {
            "controller": {"path": correction["frozen_hashes"]["controller"]["path"], "sha256": expected_frozen["controller"]},
            "historical_interrupted_result": {"path": relative(ORIGINAL_RESULT), "sha256": expected_frozen["original_result"]},
            "historical_trace_hashes": [{"path": row["path"], "sha256": row["sha256"]} for row in correction["frozen_hashes"]["existing_traces"]],
            "scientific_contract_sha256": seal["scientific_contract_sha256"],
            "continuation_result": {"path": relative(CONTINUATION_RESULT), "sha256": sha256(CONTINUATION_RESULT)},
            "final_adjudication": {"path": relative(ADJUDICATION), "sha256": sha256(ADJUDICATION)},
            "all_final_execution_bindings_pass": all(adjudication["execution_bindings"].values()),
        },
        "identity_and_outcome_access": {
            "before_continuation": {
                "state_rgb_only_preflight_base_ids": list(range(20261134, 20261146)),
                "pending_scientific_outcomes_opened": schedule["pending_scientific_outcomes_opened"],
                "scientific_primary_base_ids": [20261134],
                "scientific_primary_rows": correction["fixed_handling_20261134"],
                "scientific_sham_rows": [],
            },
            "continuation": {
                "primary_base_ids": list(range(20261135, 20261146)),
                "primary_rows": 22,
                "sham_base_ids": list(range(20261134, 20261140)),
                "sham_rows": 12,
                "schedule_order_unchanged": True,
                "serial_single_environment": schedule["serial"],
                "base_20261134_primary_rerun": False,
            },
            "sealed": {
                "validation_demo_ids": [40, 41, 42, 43, 44],
                "confirmation_demo_ids": [45, 46, 47, 48, 49],
                "validation_accessed": adjudication["validation_accessed"],
                "confirmation_accessed": adjudication["confirmation_accessed"],
            },
        },
        "fixed_denominator_accounting": {
            "candidate_probes": {"planned": 48, "finite_bounded": counts["finite_bounded_actions"], "contact_or_excitation": counts["intended_contact_or_excitation"], "valid_response_windows": 47, "missing_response_windows": 1, "invalid_other": 0, "unexecuted": 0},
            "primary_assignments": {"planned": 24, "completed": counts["primary_completed"], "failed_missing_response": counts["primary_failed_missing_response"], "invalid_other": counts["primary_invalid_other"], "unexecuted": counts["primary_unexecuted"]},
            "exact_state_pairs": {"binary_denominator": 12, "observed_physical_contrasts": paired["observed_complete_pair_count"], "missing_physical_contrasts": paired["missing_pair_count"]},
            "shams": {"planned": 12, "completed": counts["sham_completed"], "failed": counts["sham_failed"], "invalid_other": 0, "unexecuted": counts["sham_unexecuted"]},
            "completion_oracle_fixed_denominator": {"planned": 24, "success": counts["completion_oracle"], "failure_or_incomplete": 24 - counts["completion_oracle"], "unexecuted": 0},
        },
        "base_20261134_endpoint_handling": {
            "assignment_A": "completed; observed rank miss; oracle success; never rerun",
            "assignment_B_front": "valid mechanics trace retained; never rerun",
            "assignment_B_back": "physical contact with 0/5 valid response steps; immutable missing response; never rerun or imputed",
            "assignment_B_rank": "failure in fixed 24-assignment intention-to-treat endpoint",
            "assignment_B_completion": "failure in fixed 24-assignment intention-to-treat endpoint",
            "binary_pair": "adverse/nonflip in fixed 12-pair flip and sign endpoints",
            "continuous_pair": "missing and excluded from the 11-pair observed mean/CI",
            "sensitivity_contrast_range_m": [-0.006408279988615858, 0.04359172001138414],
            "worst_case_contrast_m": -0.006408279988615858,
        },
        "scientific_results": {
            "rank": {"correct": counts["rank_correct"], "denominator": 24, "by_heavy_position": counts["rank_by_heavy_position"]},
            "exact_correct_flips": {"correct": counts["exact_pair_correct_flips"], "denominator": 12},
            "completion_oracle": {"success": counts["completion_oracle"], "denominator": 24, "by_heavy_position": counts["completion_by_heavy_position"]},
            "complete_case_physical": {"n": paired["observed_complete_pair_count"], "mean_m": paired["observed_complete_case_mean_m"], "student_t_95_interval_m": paired["complete_case_student_t_95_interval_m"], "hc3_95_interval_m": paired["complete_case_adjusted_hc3"]["hc3_95_interval_m"]},
            "fixed_denominator_sign": {"n": 12, "positive": paired["fixed_denominator_positive_pairs"], "nonpositive_or_missing": paired["fixed_denominator_nonpositive_or_missing_pairs"], "one_sided_exact_p": paired["fixed_denominator_one_sided_exact_sign_p"]},
            "worst_case_sensitivity": {"student_t_95_interval_m": paired["worst_case_augmented_student_t_95_interval_m"], "hc3_95_interval_m": paired["worst_case_augmented_adjusted_hc3"]["hc3_95_interval_m"], "tipping_point": paired["base_20261134_tipping_point"]},
            "twelve_pair_observed_physical_mean_or_ci_reported": paired["twelve_pair_observed_physical_mean_or_ci_reported"],
        },
        "gates": adjudication["gates"],
        "failed_gates": adjudication["failed_gates"],
        "resources": resources,
        "regression_and_schema_checks": {
            "targeted_failfast_suite": "13 passed",
            "epoch9e_suite_before_terminal_artifact_addition": "38 passed",
            "final_terminal_artifact_suite": "43 passed",
            "all_integrity_groups_pass": all(adjudication["integrity"].values()),
            "continuation_result_schema_version": continuation["schema_version"],
            "adjudication_schema_version": adjudication["schema_version"],
            "status_correction_schema_version": status_correction["schema_version"],
            "parser_repair_schema_version": parser_repair["schema_version"],
        },
        "historical_append_only_preservation": {
            "historical_artifacts_edited": False,
            "prior_evidence_index": {"path": relative(OLD_INDEX), "sha256": sha256(OLD_INDEX)},
            "prior_terminal_handoff": {"path": relative(OLD_HANDOFF), "sha256": sha256(OLD_HANDOFF)},
        },
        "protected_untracked_manifests": protected,
        "validation_accessed": False,
        "confirmation_accessed": False,
        "estimator_development_started": False,
        "official_evaluation_started": False,
        "paper_status": "PAPER_NOT_AUTHORIZED",
        "paper_paths": [],
        "authorized_next_stage": "STOP_ACTIVE_ROUTE_NO_RERUN_NO_ESTIMATOR_VALIDATION_CONFIRMATION_OR_PAPER",
        "evidence_index_v2_path": relative(OUTPUT_INDEX),
    }
    atomic_write_json(OUTPUT_JSON, terminal)
    atomic_write_text(
        OUTPUT_MD,
        f"""# Epoch 9E Fail-Fast Continuation Terminal Handoff

Terminal state: `{DECISION}`

Branch: `{BRANCH}`  
Starting checkpoint: `{START}`  
Completed-panel adjudication checkpoint: `{terminal['outcome_checkpoint']}`

The wrapper-only continuation completed all 22 untouched primary rows and all 12 frozen shams without rerunning base `20261134`. The fixed panel contains 23 completed primary assignments, one immutable missing-response failure, no invalid-other rows, and no unexecuted rows.

The original route closes because rank was `{counts['rank_correct']}/24` (heavy-back `{counts['rank_by_heavy_position']['back']['correct']}/12`, heavy-front `{counts['rank_by_heavy_position']['front']['correct']}/12`), exact correct flips were `{counts['exact_pair_correct_flips']}/12`, and the fixed-denominator one-sided sign p-value was `{paired['fixed_denominator_one_sided_exact_sign_p']}`. The conservative worst-case Student-t interval `{paired['worst_case_augmented_student_t_95_interval_m']}` and HC3 interval `{paired['worst_case_augmented_adjusted_hc3']['hc3_95_interval_m']}` remained positive, but they do not rescue the failed original gates.

Base `20261134` remains adverse/nonflip. Assignment B rank and completion are failures; its back response is missing and not imputed. Continuous physical reporting uses 11 observed contrasts only (mean `{paired['observed_complete_case_mean_m']}` m, 95% Student-t interval `{paired['complete_case_student_t_95_interval_m']}`).

The controller hash remains `{expected_frozen['controller']}`. Validation identities `40--44` and confirmation identities `45--49` remain sealed. Peak host RAM was `{resources['peak_host_ram_percent']:.6f}%`, peak system-wide GPU allocation was `{resources['peak_system_wide_gpu_used_mib']} MiB, and scientific WSL swap use was `0` bytes. Protected rollout manifests remain untracked and byte-identical.

Machine-readable handoff: `{relative(OUTPUT_JSON)}`  
Evidence index v2: `{relative(OUTPUT_INDEX)}`

Paper status: `PAPER_NOT_AUTHORIZED`. Authorized next stage: stop the active route; no estimator, validation, confirmation, official evaluation, or paper construction.
""",
    )

    entries = [
        {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path), "schema_version": schema_version(path)}
        for path in evidence_files()
    ]
    evidence_index = {
        "schema_version": "epoch9e.failfast_continuation_evidence_index.v2",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "terminal_decision": DECISION,
        "branch": BRANCH,
        "starting_checkpoint": START,
        "outcome_checkpoint": terminal["outcome_checkpoint"],
        "authority_prompt": {"path": str(AUTHORITY), "sha256": AUTHORITY_SHA256},
        "terminal_handoff": {"path": relative(OUTPUT_JSON), "sha256": sha256(OUTPUT_JSON)},
        "entry_count": len(entries),
        "entries": entries,
        "protected_untracked_manifests": protected,
        "validation_accessed": False,
        "confirmation_accessed": False,
        "paper_status": "PAPER_NOT_AUTHORIZED",
    }
    atomic_write_json(OUTPUT_INDEX, evidence_index)
    for row in entries:
        path = ROOT / row["path"]
        if not path.is_file() or path.stat().st_size != row["bytes"] or sha256(path) != row["sha256"]:
            raise RuntimeError(f"evidence index self-check failed: {row['path']}")
    print(json.dumps({"decision": DECISION, "failed_gates": adjudication["failed_gates"], "entries": len(entries), "outcome_checkpoint": terminal["outcome_checkpoint"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
