#!/usr/bin/env python3
"""Build the Epoch 9B/9C terminal adjudication and evidence index."""

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
OUTPUT_JSON = REPORTS / "epoch9b_terminal_adjudication.json"
OUTPUT_MD = REPORTS / "epoch9b_terminal_adjudication.md"
STATE = REPORTS / "epoch9b_campaign_state.json"
INDEX = REPORTS / "epoch9b_evidence_index.json"

EVIDENCE = {
    "metric_integrity": REPORTS / "epoch9b_metric_integrity_result.json",
    "v2_protocol": REPORTS / "epoch9b_v2_task_preservation_protocol.json",
    "observability": REPORTS / "epoch9b_observability_diagnostic.json",
    "score_calibration": REPORTS / "epoch9b_dynamic_nudge/back_response_threshold_calibration.json",
    "original_controller_freeze": REPORTS / "epoch9b_dynamic_nudge/controller_freeze.json",
    "original_panel": REPORTS / "epoch9b_dynamic_nudge/feasibility_panel_result.json",
    "repair1_protocol": REPORTS / "epoch9b_v2_task_preservation_protocol_repair1.json",
    "repair1_edge_development": REPORTS / "epoch9b_dynamic_nudge/development/d17_centered_contact_edge_stress/result.json",
    "repair2_protocol": REPORTS / "epoch9b_v2_task_preservation_protocol_repair2.json",
    "repair2_edge_development": REPORTS / "epoch9b_dynamic_nudge/development/d18_inward_contact_edge_stress/result.json",
    "repair2_balanced_development": REPORTS / "epoch9b_dynamic_nudge/development/d19_inward_contact_balanced/result.json",
    "repair3_protocol": REPORTS / "epoch9b_v2_task_preservation_protocol_repair3.json",
    "repair3_controller_freeze": REPORTS / "epoch9b_dynamic_nudge/controller_freeze_repair3.json",
    "repair3_panel": REPORTS / "epoch9b_dynamic_nudge/feasibility_panel_repair3_result.json",
    "attribution_protocol": REPORTS / "epoch9c_attribution_feedback_protocol.json",
    "attribution_result": REPORTS / "epoch9c_attribution_feedback_result.json",
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


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_write_text(path: Path, value: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def protected_snapshot(path: Path) -> dict[str, Any]:
    files = sorted(value for value in path.rglob("*") if value.is_file())
    lines = [f"{relative(value)}\t{value.stat().st_size}\t{sha256(value)}" for value in files]
    return {
        "path": relative(path) + "/",
        "file_count": len(files),
        "total_bytes": sum(value.stat().st_size for value in files),
        "manifest_sha256": hashlib.sha256("\n".join(lines).encode("utf-8")).hexdigest().upper(),
        "touched_by_epoch9b": False,
    }


def main() -> int:
    for name, path in EVIDENCE.items():
        if not path.exists():
            raise FileNotFoundError(f"missing terminal evidence {name}: {path}")
    data = {name: load(path) for name, path in EVIDENCE.items()}
    original = data["original_panel"]["summary"]
    repair1 = data["repair1_edge_development"]["summary"]
    repair2_edge = data["repair2_edge_development"]["summary"]
    repair2 = data["repair2_balanced_development"]["summary"]
    repair3 = data["repair3_panel"]["summary"]
    attribution = data["attribution_result"]
    protected = [
        protected_snapshot(ROOT / "rollouts/2026_07_17"),
        protected_snapshot(ROOT / "rollouts/2026_07_18"),
    ]
    expected_protected = {
        "rollouts/2026_07_17/": (27, 5143751, "25DE8FF5AA6112D7EFF8BCF38D3A4C3F0F3C8C8EE0458E5FA83D17438719EC54"),
        "rollouts/2026_07_18/": (10, 924633, "CF701D6F73D4783F016E48A72C093DC9FD6D940B7081DA8FBEC128DB94C24A00"),
    }
    for value in protected:
        expected = expected_protected[value["path"]]
        if (value["file_count"], value["total_bytes"], value["manifest_sha256"]) != expected:
            raise RuntimeError(f"protected rollout drift detected: {value}")

    terminal = {
        "schema_version": "epoch9b.empirical_rotations.terminal_adjudication.v1",
        "timestamp": timestamp(),
        "branch": "codex/epoch9b-adaptive-probe-paper-continuation",
        "source_checkpoint": "d805e0a84fc8bcb720ecfd0cafeeda153aa603b4",
        "decision": "NO_DEFENSIBLE_LOCAL_PATH_AFTER_EMPIRICAL_ROTATIONS",
        "active_property_status": "ACTIVE_PROPERTY_THESIS_EMPIRICALLY_INFEASIBLE_ROTATING",
        "attribution_rotation_status": attribution["decision"],
        "paper_status": "PAPER_NOT_AUTHORIZED",
        "official_closed_loop_ours_episodes": 0,
        "final_temporal_model_fit": False,
        "validation_accessed": False,
        "confirmation_accessed": False,
        "metric_integrity_decision": data["metric_integrity"].get("decision"),
        "observability_decision": data["observability"].get("decision"),
        "active_property_adjudication": {
            "original_frozen_panel": original,
            "original_panel_finding": "substantive thresholds passed, but one real 0.004117 m front-y lane excursion made the exact frozen panel NO-GO",
            "repair1_centered_edge_stress": repair1,
            "repair2_inward_edge_stress": repair2_edge,
            "repair2_balanced_development": repair2,
            "repair3_frozen_panel": repair3,
            "repair3_finding": "all 48 probes were mechanically valid, but ranking was 19/24 with 7/12 back-heavy and oracle completion was 16/24; both missed frozen GO thresholds",
            "closure_basis": [
                "the audited old fixed-probe trajectories carried no reliable mass signal after grouped nuisance-controlled evaluation",
                "adaptive RGB localization, subpixel tracking, retract-and-reobserve contact verification, and a bounded ballistic response were implemented",
                "fixture-clear mirrored, centered, and inward-biased contact geometries were tested on distinct development evidence",
                "two complete 24-scene frozen panels exposed a non-rescuable task-preservation versus ranking/headroom tradeoff",
                "no post-panel threshold retuning, temporal-model fitting, or sealed-stage access was performed",
            ],
        },
        "attribution_adjudication": {
            "purpose": "independent semantic-versus-physical causal-attribution rotation required after active-property closure",
            "already_available_families": ["articulated_drawer", "pick_transport_place"],
            "missing_family": "planar_push",
            "final_feedback_result": {
                "decision": attribution["decision"],
                "paired_initial_state_exact": attribution["paired_initial_state_exact"],
                "first_observation_exact": attribution["first_observation_exact"],
                "conditions": [
                    {
                        "condition": row["condition"],
                        "target_contact": row["target_contact_any"],
                        "official_success": row["official_success"],
                        "minimum_goal_distance_m": min(row["target_goal_distance_trace_m_eval_only"]),
                    }
                    for row in attribution["rows"]
                ],
            },
            "closure_basis": "the final pose-preserving, low-gain feedback oracle contacted the target in both exact-init conditions but completed neither, so a third legal manipulation family could not be established and VLA attribution rollout remained unauthorized",
        },
        "claim_boundary": "This is an auditable local empirical exhaustion result for the two routes mandated by the continuation prompt. It is not a claim that active physical grounding, causal attribution, or VLA research is impossible in general.",
        "protected_rollouts": protected,
    }
    atomic_write_json(OUTPUT_JSON, terminal)

    index = {
        "schema_version": "epoch9b.evidence_index.v1",
        "generated_at": timestamp(),
        "terminal_decision": terminal["decision"],
        "artifacts": {
            name: {
                "path": relative(path),
                "sha256": sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in EVIDENCE.items()
        },
        "terminal_adjudication": {
            "path": relative(OUTPUT_JSON),
            "sha256": sha256(OUTPUT_JSON),
            "bytes": OUTPUT_JSON.stat().st_size,
        },
        "sealed_validation_or_confirmation_accessed": False,
    }
    atomic_write_json(INDEX, index)

    state = {
        "schema_version": "epoch9b.campaign_state.v1",
        "timestamp": timestamp(),
        "branch": terminal["branch"],
        "source_checkpoint": terminal["source_checkpoint"],
        "epoch9b_status": terminal["active_property_status"],
        "epoch9c_status": terminal["attribution_rotation_status"],
        "paper_status": terminal["paper_status"],
        "program_status": terminal["decision"],
        "active_workers": 0,
        "validation_accessed": False,
        "confirmation_accessed": False,
        "terminal_adjudication": relative(OUTPUT_JSON),
        "evidence_index": relative(INDEX),
        "protected_rollouts": protected,
        "safe_resume": "No paper or sealed-stage continuation is authorized from these outcomes. A future cycle requires a materially new task/data/robot capability and a fresh preregistered protocol; do not retune the closed response threshold or reuse the frozen panels as development.",
    }
    atomic_write_json(STATE, state)

    standard_distance = terminal["attribution_adjudication"]["final_feedback_result"]["conditions"][0]["minimum_goal_distance_m"]
    altered_distance = terminal["attribution_adjudication"]["final_feedback_result"]["conditions"][1]["minimum_goal_distance_m"]
    markdown = f"""# Epoch 9B/9C Terminal Adjudication

Decision: `NO_DEFENSIBLE_LOCAL_PATH_AFTER_EMPIRICAL_ROTATIONS`

Paper: `PAPER_NOT_AUTHORIZED`

Validation identities 40–44 accessed: **no**. Confirmation identities 45–49 accessed: **no**.

## Active-property route

The Epoch 9 visual-return metric was defective: its legacy front crop tracked a static region rather than the manipulated bowl. The corrected object-centered metric responded monotonically to deliberate simulator translations, while the historical simulator displacement values remained valid and unchanged.

The grouped development-only observability diagnostic found no reliable mass signal in the old fixed-probe trajectories. Epoch 9B therefore built a genuinely different adaptive probe with RGB localization, subpixel response tracking, retract-and-reobserve contact verification, a fixture-clear paddle path, and a bounded ballistic micro-impulse.

| Frozen evidence | Contact/excitation | Lane/reach | Rank | Oracle | Collisions | Decision |
|---|---:|---:|---:|---:|---:|---|
| Original 24-scene panel | {original['intended_contact_or_excitation_probes']}/48 | {original['lane_reachability_pass_probes']}/48 | {original['heavy_rank_correct_scenes']}/24 ({original['heavy_rank_by_position']['front']['rank_correct']}/12 front, {original['heavy_rank_by_position']['back']['rank_correct']}/12 back) | {original['oracle_completion_success_scenes']}/24 | {original['candidate_pair_collision_probes'] + original['candidate_distractor_collision_probes']} | NO-GO: one real lane excursion |
| Repair3 24-scene panel | {repair3['intended_contact_or_excitation_probes']}/48 | {repair3['lane_reachability_pass_probes']}/48 | {repair3['heavy_rank_correct_scenes']}/24 ({repair3['heavy_rank_by_position']['front']['rank_correct']}/12 front, {repair3['heavy_rank_by_position']['back']['rank_correct']}/12 back) | {repair3['oracle_completion_success_scenes']}/24 | {repair3['candidate_pair_collision_probes'] + repair3['candidate_distractor_collision_probes']} | NO-GO: rank and oracle thresholds missed |

Centered and inward-biased front-contact repairs were also preserved as development failures. They fixed some mechanical failure modes but did not jointly recover strict lane robustness and oracle headroom. The final temporal model, validation, official Ours closed loop, and confirmation were therefore not authorized.

## Attribution rotation

The independent semantic-versus-physical attribution route already had valid altered-dynamics expert headroom for drawer actuation and heavy-bowl placement, but not planar push. The final frozen pose-adaptive push oracle used exact paired initial state and identical first observation, and contacted the plate in both conditions. Neither completed: minimum plate-to-goal distance was {standard_distance:.6f} m in standard dynamics and {altered_distance:.6f} m under low friction. The required third family was not restored, so no attribution VLA rollout was legal.

## Supported conclusion

The mandated active-property and independent attribution routes have both reached auditable local empirical infeasibility under distinct frozen tests. This supports `NO_DEFENSIBLE_LOCAL_PATH_AFTER_EMPIRICAL_ROTATIONS`, not a universal impossibility claim. No manuscript is authorized. Every numerical claim above is hash-linked in `reports/epoch9b_evidence_index.json`.
"""
    atomic_write_text(OUTPUT_MD, markdown)
    print(json.dumps({"decision": terminal["decision"], "artifacts": len(index["artifacts"])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
