"""Run RAC-VLA development-only audits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.rac_vla import PROPOSAL_HASH, audit_rac_records, run_validation_search  # noqa: E402


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
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# RAC-VLA Development Audit",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- training happened: `{report['training_happened']}`",
        f"- consequence pairs: `{report['consequence_pairs']}`",
        f"- labeled examples: `{report['labeled_examples']}`",
        f"- train examples: `{report['train_examples']}`",
        f"- validation examples: `{report['validation_examples']}`",
        f"- duplicate perturbation keys: `{report['duplicate_perturbation_keys']}`",
        f"- full validation accuracy: `{report.get('full_validation_accuracy')}`",
        f"- action-only validation accuracy: `{report.get('action_only_validation_accuracy')}`",
        f"- no-consequence validation accuracy: `{report.get('no_consequence_validation_accuracy')}`",
        f"- full-vs-best-baseline accuracy margin: `{report.get('full_vs_best_baseline_accuracy_margin')}`",
        f"- gate positive fraction: `{report.get('gate_positive_fraction')}`",
        f"- clean gate positive fraction: `{report.get('clean_gate_positive_fraction')}`",
        f"- clean action delta p95: `{report.get('clean_action_delta_p95')}`",
        f"- shifted action delta p95: `{report.get('shifted_action_delta_p95')}`",
        f"- validation action validity: `{report.get('validation_action_validity')}`",
        "",
        "Task consequence counts:",
        "",
        "```json",
        json.dumps(report.get("task_consequence_counts"), indent=2, sort_keys=True),
        "```",
        "",
        "Perturbation label counts:",
        "",
        "```json",
        json.dumps(report.get("label_counts"), indent=2, sort_keys=True),
        "```",
        "",
        "Hard stop reasons:",
    ]
    reasons = list(report.get("hard_stop_reasons") or [])
    if reasons:
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- none")
    lines.extend(["", f"Next step: {report['next_step']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _write_validation_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# RAC-VLA Validation Search",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"Search budget: `{report['search_budget']}`",
        "",
        "## Selected Config",
        "",
        "```json",
        json.dumps(report.get("selected_config"), indent=2, sort_keys=True),
        "```",
        "",
        "## Tried Configs",
        "",
        "| config | decision | full acc | margin | shifted margin | gate | clean p95 | shifted p95 | score |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.get("tried_configs", []):
        score = ((item.get("score_terms") or {}).get("total"))
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.get('config_id')}`",
                    f"`{item.get('final_decision')}`",
                    f"{item.get('full_validation_accuracy')}",
                    f"{item.get('full_vs_best_baseline_accuracy_margin')}",
                    f"{item.get('full_vs_best_baseline_shifted_accuracy_margin')}",
                    f"{item.get('gate_positive_fraction')}",
                    f"{item.get('clean_action_delta_p95')}",
                    f"{item.get('shifted_action_delta_p95')}",
                    f"{score}",
                ]
            )
            + " |"
        )
    lines.extend(["", f"Next step: {report['next_step']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="reports/cavm_vla/acquisition_records.jsonl")
    parser.add_argument("--json-output", default="reports/rac_vla/development_audit.json")
    parser.add_argument("--md-output", default="reports/rac_vla/development_audit.md")
    parser.add_argument("--validation-json-output", default="reports/rac_vla/validation_search.json")
    parser.add_argument("--validation-md-output", default="reports/rac_vla/validation_search.md")
    args = parser.parse_args()

    records = _read_jsonl(Path(args.records))
    report = audit_rac_records(records)
    _write_json(Path(args.json_output), report)
    _write_md(Path(args.md_output), report)
    validation = run_validation_search(records)
    _write_json(Path(args.validation_json_output), validation)
    _write_validation_md(Path(args.validation_md_output), validation)
    print(
        json.dumps(
            {
                "final_decision": report["final_decision"],
                "consequence_pairs": report["consequence_pairs"],
                "full_vs_best_baseline_accuracy_margin": report.get("full_vs_best_baseline_accuracy_margin"),
                "validation_decision": validation["final_decision"],
                "selected_config": validation["selected_config"]["config_id"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
