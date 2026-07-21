#!/usr/bin/env python3
"""Create the Epoch 9D scope correction and initial campaign state."""

from __future__ import annotations

import hashlib
import json
import shutil
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
SCOPE_MD = REPORTS / "epoch9d_scope_correction.md"
STATE_JSON = REPORTS / "epoch9d_campaign_state.json"
INVENTORY = REPORTS / "epoch9d_identity_seed_inventory.json"
HISTORICAL_TERMINAL_JSON = REPORTS / "epoch9b_terminal_adjudication.json"
HISTORICAL_TERMINAL_MD = REPORTS / "epoch9b_terminal_adjudication.md"
EXPECTED_START = "fee058a022e0af347f413f7a8b0361f3cd9074c9"
BRANCH = "codex/epoch9d-causal-probe-bounded-convergence"


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


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def protected_snapshot(path: Path) -> dict[str, Any]:
    files = sorted(value for value in path.rglob("*") if value.is_file())
    lines = [f"{relative(value)}\t{value.stat().st_size}\t{sha256(value)}" for value in files]
    return {
        "path": relative(path) + "/",
        "file_count": len(files),
        "total_bytes": sum(value.stat().st_size for value in files),
        "manifest_sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest().upper(),
        "touched_by_epoch9d": False,
    }


def host_preflight() -> dict[str, Any]:
    try:
        import psutil

        memory = psutil.virtual_memory()
        disk = psutil.disk_usage(str(ROOT))
        memory_record: dict[str, Any] = {
            "physical_total_bytes": int(memory.total),
            "physical_available_bytes_at_audit": int(memory.available),
            "physical_used_percent_at_audit": float(memory.percent),
            "ceiling_percent": 82.0,
        }
        disk_record: dict[str, Any] = {
            "free_bytes_at_audit": int(disk.free),
            "used_percent_at_audit": float(disk.percent),
        }
    except (ImportError, OSError):
        memory_record = {"status": "PSUTIL_UNAVAILABLE", "ceiling_percent": 82.0}
        usage = shutil.disk_usage(ROOT)
        disk_record = {"free_bytes_at_audit": int(usage.free)}
    try:
        gpu_text = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
        gpu = {"status": "AVAILABLE", "query": gpu_text}
    except (FileNotFoundError, subprocess.SubprocessError):
        gpu = {"status": "NOT_REQUIRED_OR_UNAVAILABLE_FOR_SIMULATOR_ONLY_PHASES_A_TO_C"}
    return {
        "host_memory": memory_record,
        "disk": disk_record,
        "gpu": gpu,
        "execution": {
            "simulator_environments_at_once": 1,
            "resident_vla_models_at_once": 1,
            "phase_a": "offline trace analysis; no simulator or model",
            "phase_b": "one off-screen LIBERO simulator; no resident VLA",
            "phase_c": "one off-screen LIBERO simulator; no resident VLA during controller feasibility",
            "parallel_heavy_execution": False,
            "wsl_swap_policy": "swap may exist on the host but scientific runs must record zero use and may not rely on it",
            "precision_or_offload_change_authorized": False,
        },
        "dependency_source": "existing official-smolvla-libero environment and already-present LIBERO assets",
        "download_required": False,
        "external_license_or_token_required": False,
        "bounded_runtime_basis": "historical dynamic-probe scenes took approximately 20 seconds each; Phase B is serialized and resumable",
        "risk_decision": "PROCEED_SERIAL_WITH_RESOURCE_MONITORING",
    }


def main() -> int:
    for path in (INVENTORY, HISTORICAL_TERMINAL_JSON, HISTORICAL_TERMINAL_MD):
        if not path.exists():
            raise FileNotFoundError(path)
    if SCOPE_MD.exists() or STATE_JSON.exists():
        raise FileExistsError("refusing to overwrite Epoch 9D scope artifacts")
    head = git("rev-parse", "HEAD")
    branch = git("branch", "--show-current")
    if head != EXPECTED_START:
        raise RuntimeError(f"unexpected start checkpoint: {head}")
    if branch != BRANCH:
        raise RuntimeError(f"unexpected branch: {branch}")
    inventory = json.loads(INVENTORY.read_text(encoding="utf-8"))
    if inventory["epoch9_largest_previously_used_numeric_development_identity_M"] != 39:
        raise RuntimeError("Epoch 9 development identity boundary drifted")
    protected = [
        protected_snapshot(ROOT / "rollouts/2026_07_17"),
        protected_snapshot(ROOT / "rollouts/2026_07_18"),
    ]
    expected = {
        "rollouts/2026_07_17/": (27, 5143751, "25DE8FF5AA6112D7EFF8BCF38D3A4C3F0F3C8C8EE0458E5FA83D17438719EC54"),
        "rollouts/2026_07_18/": (10, 924633, "CF701D6F73D4783F016E48A72C093DC9FD6D940B7081DA8FBEC128DB94C24A00"),
    }
    for row in protected:
        if (row["file_count"], row["total_bytes"], row["manifest_sha256"]) != expected[row["path"]]:
            raise RuntimeError(f"protected rollout drift: {row}")

    allocations = {
        "historical_development_identity_max_M": 39,
        "sealed_validation_source_demo_identities": [40, 41, 42, 43, 44],
        "sealed_confirmation_source_demo_identities": [45, 46, 47, 48, 49],
        "mechanical_debugging_generated_identity_ids": [50, 51],
        "labeled_development_generated_identity_ids": [52, 53, 54, 55],
        "causal_panel_generated_base_identity_ids": list(range(56, 72)),
        "controller_pilot_generated_identity_ids": list(range(72, 96)),
        "controller_feasibility_generated_identity_ids": list(range(96, 120)),
        "training_development_generated_identity_ids": list(range(120, 160)),
        "validation_fresh_generated_identity_ids": list(range(160, 170)),
        "official_closed_loop_generated_identity_ids": list(range(170, 210)),
        "confirmation_fresh_generated_identity_ids": list(range(210, 220)),
        "generator_seed_ranges": {
            "mechanical_debugging": [914000, 914001],
            "labeled_development": [914010, 914013],
            "causal_panel": [914100, 914115],
            "controller_pilots": [914200, 914223],
            "controller_feasibility": [914300, 914323],
            "training_development": [914400, 914439],
            "validation_fresh": [914500, 914509],
            "official_closed_loop": [914600, 914639],
            "confirmation_fresh": [914700, 914709],
        },
        "allocation_rule": (
            "Generated identity IDs are campaign identities backed by frozen simulator reset seeds and exact flattened "
            "base states. They are not aliases for source demos. IDs 40--49 remain untouched until their sealed stage."
        ),
    }
    used_seeds = set(inventory["seed_values"])
    for low, high in allocations["generator_seed_ranges"].values():
        overlap = sorted(used_seeds.intersection(range(low, high + 1)))
        if overlap:
            raise RuntimeError(f"allocated generator seed overlap: {overlap}")

    scope = f"""# Epoch 9D Scope Correction

State: `ACTIVE_DYNAMIC_PROBE_CAUSAL_SIGNAL_AND_TASK_HEADROOM_UNRESOLVED`

Paper: `PAPER_NOT_AUTHORIZED`

Authority date: {timestamp()}

This record narrows the program-wide wording in the historical Epoch 9B/9C terminal adjudication. It does not delete, edit, relabel, or supersede any historical row, threshold, hash, or frozen NO-GO decision. The historical records remain immutable evidence at `{relative(HISTORICAL_TERMINAL_JSON)}` and `{relative(HISTORICAL_TERMINAL_MD)}`.

## Supported historical conclusions

- The original frozen panel remains a NO-GO under its frozen gate: 48/48 contacts, 47/48 lane/reachability, 22/24 property rankings, 21/24 oracle completions, and zero collisions.
- Repair3 remains a NO-GO: 48/48 mechanics/lane, 19/24 rankings (12/12 front-heavy, 7/12 back-heavy), and 16/24 oracle completions.
- Those results do not authorize validation, official Ours, confirmation, or a paper.
- They also do not establish absence of a useful mass signal. The original result is strong positive development evidence and a near-pass, while its front/back asymmetry makes shortcut or contact-geometry confounding plausible.
- The old observability diagnostic evaluated old fixed-probe traces, not the later dynamic-nudge raw traces.
- Epoch 9C's 0/2 result is one paired planar-push initial condition under two dynamics settings. The standard condition itself lacked oracle headroom and no policy ran. It invalidates that oracle/controller instance, not the whole semantic-versus-physical attribution thesis.

## Corrected active scope

One bounded continuation will first test whether the original frozen dynamic-nudge response contains an observation-available causal mass signal under exact-state mass swaps. Only a first-score `CAUSAL_SIGNAL_GO` may open the bounded task-preserving controller stage. Validation, official closed loop, confirmation, and paper construction remain sealed behind their stated gates.

The current state is therefore `ACTIVE_DYNAMIC_PROBE_CAUSAL_SIGNAL_AND_TASK_HEADROOM_UNRESOLVED`, with `PAPER_NOT_AUTHORIZED` unchanged.

## Preservation and identity boundary

- Start checkpoint: `{EXPECTED_START}`.
- Continuation branch: `{BRANCH}`.
- Largest prior Epoch 9 development identity: `M = 39`.
- Validation source identities 40--44 and confirmation source identities 45--49 remain sealed and unaccessed.
- Fresh Epoch 9D development identities use generated exact reset records beginning at 50; they are not aliases for sealed or historical demos.
- Full identity and seed enumeration: `{relative(INVENTORY)}` (`{sha256(INVENTORY)}`).
- `rollouts/2026_07_17/` and `rollouts/2026_07_18/` retain their historical protected hashes and remain untracked.
"""
    atomic_write_text(SCOPE_MD, scope)

    state = {
        "schema_version": "epoch9d.campaign_state.v1",
        "timestamp": timestamp(),
        "branch": BRANCH,
        "starting_checkpoint": EXPECTED_START,
        "head_at_scope_correction": head,
        "program_status": "ACTIVE_DYNAMIC_PROBE_CAUSAL_SIGNAL_AND_TASK_HEADROOM_UNRESOLVED",
        "paper_status": "PAPER_NOT_AUTHORIZED",
        "phase_status": {
            "A_existing_trace_causal_diagnosis": "IN_PROGRESS",
            "B_mass_swap_causal_panel": "SEALED_PENDING_PHASE_A_FREEZE",
            "C_task_preserving_controller": "LOCKED_PENDING_CAUSAL_SIGNAL_GO",
            "D_model_validation": "LOCKED_PENDING_CONTROLLER_GO",
            "E_official_and_confirmation": "LOCKED_PENDING_MODEL_VALIDATION_GO",
            "paper_package": "LOCKED_PENDING_POSITIVE_CONFIRMATION",
        },
        "validation_accessed": False,
        "confirmation_accessed": False,
        "active_research_workers_at_audit": 0,
        "change_scope_justification": {
            "expected_initial_checkpoint_file_count": 5,
            "generated_identity_inventory_line_count": 69018,
            "reason": (
                "The value-complete preregistered identity/seed inventory spans 4,584 historical evidence files "
                "and intentionally exceeds 5,000 generated JSON lines. It is a bounded 2.4 MB audit artifact, "
                "not source-code expansion, copied rollout data, or regenerated historical evidence."
            ),
        },
        "historical_terminal_records_preserved": [
            {"path": relative(HISTORICAL_TERMINAL_JSON), "sha256": sha256(HISTORICAL_TERMINAL_JSON)},
            {"path": relative(HISTORICAL_TERMINAL_MD), "sha256": sha256(HISTORICAL_TERMINAL_MD)},
        ],
        "scope_correction": {"path": relative(SCOPE_MD), "sha256": sha256(SCOPE_MD)},
        "identity_inventory": {"path": relative(INVENTORY), "sha256": sha256(INVENTORY)},
        "identity_and_seed_allocations": allocations,
        "protected_rollouts": protected,
        "resource_preflight": host_preflight(),
        "execution_policy": {
            "one_simulator_environment_at_a_time": True,
            "one_resident_vla_at_a_time": True,
            "host_ram_ceiling_percent": 82.0,
            "precision_change_authorized": False,
            "hidden_offload_authorized": False,
            "physical_robot_authorized": False,
            "external_submission_authorized": False,
        },
    }
    atomic_write_json(STATE_JSON, state)
    print(json.dumps({
        "scope": relative(SCOPE_MD),
        "state": relative(STATE_JSON),
        "status": state["program_status"],
        "M": allocations["historical_development_identity_max_M"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
