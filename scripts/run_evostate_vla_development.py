"""Run EvoState-VLA development-only audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.evostate_vla import PROPOSAL_HASH, audit_evostate_records  # noqa: E402


DATE_KST = "2026-07-14"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = dict(payload)
    serializable.pop("model_metadata", None)
    path.write_text(json.dumps(serializable, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# EvoState-VLA Development Audit",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- training happened: `{report['training_happened']}`",
        f"- transition pairs: `{report['transition_pairs']}`",
        f"- train transition pairs: `{report['train_transition_pairs']}`",
        f"- validation transition pairs: `{report['validation_transition_pairs']}`",
        f"- duplicate transition keys: `{report['duplicate_transition_keys']}`",
        f"- transition improvement vs constant: `{report.get('transition_improvement_vs_constant')}`",
        f"- transition improvement vs actionless: `{report.get('transition_improvement_vs_actionless')}`",
        f"- controllability effective rank: `{report.get('controllability_effective_rank')}`",
        f"- gate positive fraction: `{report.get('gate_positive_fraction')}`",
        f"- validation action delta p95: `{report.get('validation_action_delta_p95')}`",
        f"- validation action validity: `{report.get('validation_action_validity')}`",
        "",
        "Task transition counts:",
        "",
        "```json",
        json.dumps(report.get("task_transition_counts"), indent=2, sort_keys=True),
        "```",
        "",
        "Hard stop reasons:",
    ]
    reasons = list(report.get("hard_stop_reasons") or [])
    if reasons:
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- none")
    lines.extend(
        [
            "",
            f"Next step: {report['next_step']}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="reports/cavm_vla/acquisition_records.jsonl")
    parser.add_argument("--json-output", default="reports/evostate_vla/development_audit.json")
    parser.add_argument("--md-output", default="reports/evostate_vla/development_audit.md")
    args = parser.parse_args()

    records = _read_jsonl(Path(args.records))
    report = audit_evostate_records(records)
    _write_json(Path(args.json_output), report)
    _write_md(Path(args.md_output), report)
    print(json.dumps({"final_decision": report["final_decision"], "transition_pairs": report["transition_pairs"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
