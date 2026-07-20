#!/usr/bin/env python3
"""Build the auditable Epoch 9 terminal state and resumable handoff."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tca_map.epoch7_latent_dynamics import atomic_write_json

REPORTS = ROOT / "reports"
ADJUDICATION = REPORTS / "epoch9_terminal_adjudication.json"
STATE = REPORTS / "epoch9_campaign_state.json"
HANDOFF = REPORTS / "epoch9_terminal_handoff.md"
INDEX = REPORTS / "epoch9_evidence_index.json"

RESULT_PATHS = {
    "paired_balanced_order": REPORTS / "epoch9_relational_probe_dataset/development/paired_v1/result.json",
    "paired_fixed_order": REPORTS
    / "epoch9_relational_probe_dataset/development/repair1_front_first_v1/result.json",
    "paired_short_back_push": REPORTS
    / "epoch9_relational_probe_dataset/development/repair2_full_v1/result.json",
    "paired_open_gripper": REPORTS
    / "epoch9_relational_probe_dataset/development/repair3_demo31_33_diagnostic/result.json",
    "paired_open_contact_calibration": REPORTS
    / "epoch9_relational_probe_dataset/development/repair4_demo31_33_diagnostic/result.json",
    "front_reference": REPORTS
    / "epoch9_relational_probe_dataset/development/rotation1_front_reference_v1/result.json",
    "front_reference_short_push": REPORTS
    / "epoch9_relational_probe_dataset/development/rotation1_repair1_demo37_diagnostic/result.json",
    "clearance_first_return": REPORTS
    / "epoch9_relational_probe_dataset/development/rotation2_demo37_diagnostic/result.json",
    "zero_travel_contact_hold": REPORTS
    / "epoch9_relational_probe_dataset/development/rotation3_demo37_diagnostic/result.json",
}


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
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def protected_snapshot(path: Path) -> dict[str, Any]:
    files = sorted(value for value in path.rglob("*") if value.is_file())
    manifest_lines = [f"{relative(value)}\t{value.stat().st_size}\t{sha256(value)}" for value in files]
    manifest = "\n".join(manifest_lines).encode("utf-8")
    return {
        "path": relative(path) + "/",
        "file_count": len(files),
        "total_bytes": sum(value.stat().st_size for value in files),
        "manifest_sha256": hashlib.sha256(manifest).hexdigest().upper(),
        "touched_by_epoch9": False,
    }


def main() -> None:
    results = {
        name: {
            "path": relative(path),
            "sha256": sha256(path),
            "summary": json.loads(path.read_text(encoding="utf-8"))["summary"],
        }
        for name, path in RESULT_PATHS.items()
    }
    controller = json.loads(
        (REPORTS / "epoch9_controller_development/v12_visual_tolerance_calibration/result.json").read_text(
            encoding="utf-8"
        )
    )
    protected = [
        protected_snapshot(ROOT / "rollouts/2026_07_17"),
        protected_snapshot(ROOT / "rollouts/2026_07_18"),
    ]
    terminal = {
        "schema_version": "epoch9.active_physical_grounding.terminal_adjudication.v1",
        "timestamp": timestamp(),
        "branch": "codex/epoch9-active-physical-grounding",
        "source_checkpoint": "f8314938e440aaefb6399d80594c78431043aa7f",
        "epoch8_scientific_checkpoint": "a1d0d7f262038d43cbedf7640373c411b8f2e383",
        "evidence_class": "DEVELOPMENT_ONLY",
        "controller_development": {
            "path": "reports/epoch9_controller_development/v12_visual_tolerance_calibration/result.json",
            "summary": controller["summary"],
            "adjudication": (
                "The independent-task v12 mechanics passed, but later same-scene evidence exposed wrong gripper-action "
                "semantics and non-generalizing candidate disturbance; it is not a deployable controller."
            ),
        },
        "same_scene_attempts": results,
        "terminal_finding": (
            "The final zero-travel contact/hold probe retained 100% contact and action legality but moved the light bowl "
            "0.04194 m on the offending development identity, above the unchanged 0.03 m reversibility gate. The same "
            "identity remained unsafe after shortening inward travel and changing to clearance-first return, localizing "
            "the disturbance to reaching the fixed contact boundary itself."
        ),
        "decision": "ACTIVE_FIXED_SLOT_CONTACT_ROUTE_FALSIFIED_CONTINUATION_REQUIRED",
        "closed_scope": [
            "sequential paired fixed-slot Cartesian probing with this gripper geometry",
            "front-reference fixed-contact probing under the 3 cm reversibility gate",
            "shortened inward travel, clearance-first return, and zero-travel contact/hold variants",
            "reset-specific waypoint repair on development identities",
        ],
        "open_scope": [
            "learned RGB/proprio contact-aware exploration and explicit object-restoration control",
            "non-contact or purpose-built non-grasping probe geometry",
            "new tasks whose safe interaction envelope is validated before property-model fitting",
            "other active physical properties and real-robot studies under separate authorization",
        ],
        "model_fit_executed": False,
        "model_fit_reason": "No same-scene controller passed the frozen execution gate, so response-model fitting was unauthorized.",
        "validation_accessed": False,
        "confirmation_accessed": False,
        "official_closed_loop_ours_episodes": 0,
        "paper_status": "PAPER_NOT_AUTHORIZED",
        "program_status": "CONTINUATION_REQUIRED",
        "epoch8_status_correction": "EPOCH8_EXACT_METHODS_FINISHED_CONTINUATION_REQUIRED",
        "protected_rollouts": protected,
        "claim_boundary": (
            "These results falsify the enumerated local controller family, not active physical grounding in general. "
            "No temporal-property, VLA improvement, benchmark, real-robot, or paper-level positive claim is authorized."
        ),
    }
    atomic_write_json(ADJUDICATION, terminal)

    state = {
        "schema_version": "epoch9.campaign_state.v1",
        "timestamp": timestamp(),
        "branch": terminal["branch"],
        "source_checkpoint": terminal["source_checkpoint"],
        "epoch8_scientific_checkpoint": terminal["epoch8_scientific_checkpoint"],
        "epoch8_status": terminal["epoch8_status_correction"],
        "epoch9_status": terminal["decision"],
        "paper_status": terminal["paper_status"],
        "program_status": terminal["program_status"],
        "validation_accessed": False,
        "confirmation_accessed": False,
        "active_workers": 0,
        "safe_resume": (
            "Start a new method cycle with a preregistered learned RGB/proprio contact-aware exploration and object-"
            "restoration controller, or a purpose-built non-grasping probe geometry. Do not reopen the sealed 40..49 "
            "identities or fit the prepared temporal model until a new same-scene development controller passes every "
            "contact, action, 3 cm object, 5 cm EEF, and 5 px visual gate."
        ),
        "terminal_adjudication": relative(ADJUDICATION),
        "protected_rollouts": protected,
    }
    atomic_write_json(STATE, state)

    handoff = f"""# Epoch 9 Terminal Handoff

State: `ACTIVE_FIXED_SLOT_CONTACT_ROUTE_FALSIFIED_CONTINUATION_REQUIRED`

Paper: `PAPER_NOT_AUTHORIZED`

Program: `CONTINUATION_REQUIRED`

Epoch 8 correction: `EPOCH8_EXACT_METHODS_FINISHED_CONTINUATION_REQUIRED`

Branch: `codex/epoch9-active-physical-grounding`

Source checkpoint: `f8314938e440aaefb6399d80594c78431043aa7f`

## Outcome

Epoch 9 completed a primary-source overlap audit, froze fresh development/validation/confirmation identities, and produced legal RGB/proprio/action-history trajectory infrastructure. The independent v12 probe mechanics passed 20/20 fresh development episodes, but same-scene deployment exposed a wrong gripper-sign assumption, unsafe sequential interactions, missed open-tool contacts, and a persistent fixed-contact disturbance.

The terminal zero-travel contact/hold diagnostic contacted in 4/4 episodes with bounded actions but displaced the light front bowl `{results['zero_travel_contact_hold']['summary']['max_candidate_final_displacement_m']:.5f}` m, above the frozen 0.03 m limit. Shorter travel and clearance-first return produced essentially the same displacement. This local fixed-slot contact family is therefore closed.

## Evidence boundary

- Development only; validation indices 40..44 and confirmation indices 45..49 were never accessed.
- No temporal response model was fit because no same-scene controller passed the execution gate.
- No official closed-loop Ours VLA episode, physical-robot trial, external submission, or paper was produced.
- The protected rollout directories were read only for inventory and remain untracked.

## Safe resume

Open a new method cycle, not another waypoint repair. The highest-value residual is learned RGB/proprio contact-aware exploration with explicit object restoration, or purpose-built non-grasping probe geometry. Freeze that controller on development identities before fitting the prepared temporal reference model. Keep 40..49 sealed until a complete model freeze exists.

Machine-readable adjudication: `reports/epoch9_terminal_adjudication.json`

Campaign state: `reports/epoch9_campaign_state.json`
"""
    atomic_write_text(HANDOFF, handoff)

    indexed_paths = [
        REPORTS / "epoch9_active_property_overlap_delta.md",
        REPORTS / "epoch9_resource_preflight.json",
        REPORTS / "epoch9_active_grounding_protocol.json",
        REPORTS / "epoch9_active_grounding_protocol_rotation3.json",
        REPORTS / "epoch9_paired_probe_route_adjudication.json",
        ADJUDICATION,
        STATE,
        HANDOFF,
        ROOT / "scripts/run_epoch9_probe_controller_development.py",
        ROOT / "scripts/run_epoch9_relational_probe_dataset.py",
        ROOT / "scripts/run_epoch9_temporal_relational_model.py",
        ROOT / "tca_map/epoch9_active_grounding.py",
        ROOT / "tests/test_epoch9_active_grounding.py",
        *RESULT_PATHS.values(),
    ]
    index = {
        "schema_version": "epoch9.evidence_index.v1",
        "timestamp": timestamp(),
        "artifacts": [
            {"path": relative(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in indexed_paths
        ],
        "protected_rollouts": protected,
    }
    atomic_write_json(INDEX, index)
    print(json.dumps({"decision": terminal["decision"], "indexed_artifacts": len(indexed_paths)}, indent=2))


if __name__ == "__main__":
    main()
