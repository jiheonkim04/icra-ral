#!/usr/bin/env python3
"""Hash the critical Epoch 8 protocols, raw results, and adjudications."""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports/epoch8_evidence_index.json"

PATHS = [
    "reports/epoch8_route_evidence_ledger.json",
    "reports/epoch8_route_evidence_audit.md",
    "reports/epoch8_language_artifact_matrix.json",
    "reports/epoch8_language_primary_overlap_audit.md",
    "reports/epoch8_language_splits/split_index.json",
    "reports/epoch8_candidate_mechanism_comparison.md",
    "reports/epoch8_action_response_supervision.json",
    "reports/epoch8_pcat_stage0_protocol.json",
    "scripts/run_epoch8_pcat_stage0.py",
    "reports/epoch8_pcat_stage0/result.json",
    "reports/epoch8_pcat_stage0/train_log.jsonl",
    "reports/epoch8_pcat_stage0_adjudication.json",
    "reports/epoch8_rotation_ranking.md",
    "reports/epoch8_latent_dynamics_feedback_development_protocol.json",
    "scripts/run_epoch8_latent_dynamics_feedback_development.py",
    "reports/epoch8_latent_dynamics_feedback_development.json",
    "reports/epoch8_latent_dynamics_feedback_adjudication.md",
    "reports/epoch8_two_shard_actual_arrival_protocol.json",
    "scripts/run_epoch8_two_shard_actual_arrival.py",
    "scripts/monitor_epoch8_two_shard_resource_smoke.ps1",
    "reports/epoch8_two_shard_actual_arrival/run_20260720_2039_kst_resource_repair1/epoch8_two_shard_preflight.json",
    "reports/epoch8_two_shard_actual_arrival/run_20260720_2039_kst_resource_repair1/two_shard_resource_smoke_host.json",
    "reports/epoch8_two_shard_actual_arrival/run_20260720_2039_kst_resource_repair1/two_shard_resource_heartbeat.json",
    "reports/epoch8_active_latent_property_protocol.json",
    "scripts/run_epoch8_active_latent_property_stage_minus1.py",
    "reports/epoch8_active_latent_property/preflight.json",
    "reports/epoch8_active_latent_property/stage_minus1_result.json",
    "reports/epoch8_active_property_probe_belief_protocol.json",
    "scripts/run_epoch8_active_property_probe_belief_stage0.py",
    "reports/epoch8_active_property_probe_belief_stage0.json",
    "reports/epoch8_active_property_probe_belief_stage0_repair1.json",
    "reports/epoch8_active_property_probe_return_protocol.json",
    "scripts/run_epoch8_active_property_probe_return.py",
    "reports/epoch8_active_property_probe_return_result.json",
    "reports/epoch8_terminal_audit.md",
    "reports/epoch8_campaign_state.json",
    "reports/epoch8_terminal_handoff.md",
    "reports/epoch8_change_scope_justification.md",
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def main() -> None:
    rows = []
    for relative in PATHS:
        path = ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(relative)
        rows.append({"path": relative, "bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = {
        "schema_version": "epoch8.evidence_index.v1",
        "generated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "terminal_state": "LOCAL_PROGRAM_BLOCKED_WITH_AUDITABLE_EXHAUSTION",
        "critical_artifact_count": len(rows),
        "all_present": True,
        "artifacts": rows,
        "protected_evidence": {
            "rollouts/2026_07_17": {
                "files": 27,
                "bytes": 5143751,
                "tree_sha256": "6BC7D7A37A2AD98E4A4EE1B20926B923CDFC88CF7D6CACB1352363657E29DE4F",
            },
            "rollouts/2026_07_18": {
                "files": 10,
                "bytes": 924633,
                "tree_sha256": "7045345B51D54D7BD25DCA991D1E66D9D32AB4E577BEC3A4A46EC2101D1DD981",
            },
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"artifacts": len(rows), "all_present": True}, sort_keys=True))


if __name__ == "__main__":
    main()
