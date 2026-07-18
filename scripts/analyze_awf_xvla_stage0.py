#!/usr/bin/env python
"""Summarize AWF-XVLA Stage 0 under the verified wrist-dropout condition."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DISCOVERY_IDENTITIES = [20260731, 20260732]
BASELINE_CLEAN_SUCCESS_COUNT = 2
BASELINE_DROPOUT_SUCCESS_COUNT = 0


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_report(run_dir: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for result_path in sorted(run_dir.glob("identity_*/result.json")):
        result = load_json(result_path)
        episode = (result.get("episodes") or [{}])[0]
        rows.append(
            {
                "identity": int(episode["reset_identity"]),
                "awf_success": bool(episode.get("success")),
                "completed": bool(episode.get("completed")),
                "steps": int(episode.get("steps", 0)),
                "action_chunk_count": int(episode.get("action_chunk_count", 0)),
                "mitigation_triggered_step_count": int(
                    episode.get("wrist_dropout_mitigation_triggered_step_count", 0)
                ),
                "result_path": str(result_path),
            }
        )
    completed = len(rows) == len(DISCOVERY_IDENTITIES) and all(row["completed"] for row in rows)
    awf_success_count = sum(1 for row in rows if row["awf_success"])
    stage0_go = bool(completed and awf_success_count >= 1)
    return {
        "schema_version": "2026-07-18.epoch5_awf_xvla_stage0_result.v1",
        "stage": "epoch_5_awf_xvla_stage0_wrist_dropout_discovery",
        "decision": "AWF_XVLA_STAGE0_GO" if stage0_go else "AWF_XVLA_STAGE0_NO_GO",
        "method": "AWF-XVLA",
        "method_name": "Agentview-Wrist Fill for X-VLA",
        "condition": "wrist_camera_dropout_partial_observation",
        "run_dir": str(run_dir),
        "task_suite": "libero_spatial",
        "task_id": 5,
        "instruction": "pick up the black bowl on the ramekin and place it on the plate",
        "discovery_identities": list(DISCOVERY_IDENTITIES),
        "clean_baseline_success_count": BASELINE_CLEAN_SUCCESS_COUNT,
        "dropout_prior_success_count": BASELINE_DROPOUT_SUCCESS_COUNT,
        "awf_success_count": awf_success_count,
        "completed": completed,
        "stage0_go": stage0_go,
        "training_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "ours_rollout_happened": True,
        "control_rollout_happened": False,
        "broad_identity_sweep_happened": False,
        "mechanism": {
            "dropout_detector": "mean wrist RGB <= 1.0",
            "intervention": "replace wrist RGB slot with the same flipped agentview RGB that X-VLA receives as its policy-input agentview image",
            "legal_inputs": ["current agentview RGB", "current wrist RGB dropout status"],
            "forbidden_inputs_not_used": ["reward", "done/success", "simulator object/contact state", "future observation"],
        },
        "preregistered_pass_condition": "at least 1/2 successes under the verified wrist-dropout condition, improving over the 0/2 frozen-prior dropout baseline",
        "rows": rows,
        "next_action": (
            "Freeze comparator roles, a simple fill/blackout control, and held-out/clean-retention protocol before Stage A."
            if stage0_go
            else "Archive AWF-XVLA and do not tune it on the discovery failures."
        ),
    }


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# AWF-XVLA Stage 0 Result",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Method: `{report['method_name']}`",
        f"- Condition: `{report['condition']}`",
        f"- Run: `{report['run_dir']}`",
        "- No training, optimizer step, checkpoint, privileged state, reward/done/success trigger, or broad sweep was used.",
        "",
        "## Result",
        "",
        f"- Clean baseline: `{report['clean_baseline_success_count']}/2`",
        f"- Frozen-prior wrist dropout: `{report['dropout_prior_success_count']}/2`",
        f"- AWF-XVLA wrist dropout: `{report['awf_success_count']}/2`",
        f"- Stage 0 GO: `{report['stage0_go']}`",
        "",
        "| identity | AWF success | steps | chunks | mitigation-triggered steps | result |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in report["rows"]:
        lines.append(
            f"| {row['identity']} | {row['awf_success']} | {row['steps']} | {row['action_chunk_count']} | "
            f"{row['mitigation_triggered_step_count']} | `{row['result_path']}` |"
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
