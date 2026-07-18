#!/usr/bin/env python
"""Summarize the preregistered X-VLA wrist-camera-dropout condition check."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


BASELINE_EXPECTED = {
    20260731: {"success": True, "artifact": "reports/ocr_xvla_trace_observability_result.json"},
    20260732: {"success": True, "artifact": "reports/ocr_xvla_trace_observability_result.json"},
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(run_dir.glob("identity_*/result.json")):
        result = load_json(result_path)
        episode = (result.get("episodes") or [{}])[0]
        identity = int(episode["reset_identity"])
        baseline = BASELINE_EXPECTED.get(identity, {"success": None, "artifact": None})
        rows.append(
            {
                "identity": identity,
                "baseline_success": baseline["success"],
                "baseline_artifact": baseline["artifact"],
                "dropout_success": bool(episode.get("success")),
                "completed": bool(episode.get("completed")),
                "dropout_steps": int(episode.get("steps", 0)),
                "dropout_action_chunk_count": int(episode.get("action_chunk_count", 0)),
                "result_path": str(result_path),
                "input_perturbation": result.get("input_perturbation"),
            }
        )
    completed = len(rows) == len(BASELINE_EXPECTED) and all(row["completed"] for row in rows)
    baseline_success_count = sum(1 for row in rows if row["baseline_success"])
    dropout_success_count = sum(1 for row in rows if row["dropout_success"])
    success_drop = baseline_success_count - dropout_success_count
    condition_verified = bool(completed and baseline_success_count == 2 and success_drop >= 1)
    return {
        "schema_version": "2026-07-18.epoch5_wrist_dropout_condition_result.v1",
        "stage": "epoch_5_claim_specific_condition_wrist_camera_dropout_prior_degradation_check",
        "decision": "CLAIM_CONDITION_OFFICIAL_PRIOR_DEGRADATION_VERIFIED"
        if condition_verified
        else "CLAIM_CONDITION_OFFICIAL_PRIOR_DEGRADATION_NOT_VERIFIED",
        "axis": "claim_specific_controlled_condition",
        "condition": "wrist_camera_dropout_partial_observation",
        "policy": "frozen official X-VLA-Libero",
        "task_suite": "libero_spatial",
        "task_id": 5,
        "instruction": "pick up the black bowl on the ramekin and place it on the plate",
        "run_dir": str(run_dir),
        "preregistered_discovery_identities": sorted(BASELINE_EXPECTED),
        "baseline_success_count": baseline_success_count,
        "dropout_success_count": dropout_success_count,
        "success_drop": success_drop,
        "condition_verified": condition_verified,
        "no_ours_method_selected_before_condition": True,
        "training_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "ours_rollout_happened": False,
        "control_rollout_happened": False,
        "broad_identity_sweep_happened": False,
        "matched_future_protocol": {
            "same_perturbation_for_prior_and_prior_plus_ours": True,
            "simulator_state_unchanged": True,
            "rgb_policy_input_only": "robot0_eye_in_hand_image zeroed before policy action generation",
        },
        "rows": rows,
        "next_action": "Select one method candidate within the steer budget."
        if condition_verified
        else "Archive this condition and select a different official-prior ecosystem or a separately preregistered condition.",
    }


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Wrist-Camera Dropout Condition Verification",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Condition: `{report['condition']}`",
        f"- Run: `{report['run_dir']}`",
        "- Policy: frozen official X-VLA-Libero only.",
        "- No Ours method, optimizer, checkpoint, or control rollout was selected or executed before this condition check.",
        "",
        "## Result",
        "",
        f"- Clean baseline successes: `{report['baseline_success_count']}/2`",
        f"- Wrist-dropout successes: `{report['dropout_success_count']}/2`",
        f"- Success drop: `{report['success_drop']}`",
        f"- Condition verified: `{report['condition_verified']}`",
        "",
        "| identity | clean baseline success | dropout success | dropout steps | chunks | result |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['identity']} | {row['baseline_success']} | {row['dropout_success']} | "
            f"{row['dropout_steps']} | {row['dropout_action_chunk_count']} | `{row['result_path']}` |"
        )
    lines.extend(["", f"Next action: {report['next_action']}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.run_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.output_md)
    print(json.dumps({"decision": report["decision"], "output_json": str(args.output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
