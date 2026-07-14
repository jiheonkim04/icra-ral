"""Run MARC-VLA development-only Stage 0 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.marc_vla import PROPOSAL_HASH, audit_marc_records, run_validation_search  # noqa: E402


DATE_KST = "2026-07-15"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# MARC-VLA Development Audit",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- training happened: `{report['training_happened']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        f"- scoreable development records: `{report['scoreable_development_records']}`",
        f"- train records: `{report['train_records']}`",
        f"- validation records: `{report['validation_records']}`",
        f"- reserved records not used: `{report['reserved_records_not_used']}`",
        f"- selected task count: `{report['selected_task_count']}`",
        f"- duplicate sample keys: `{report['duplicate_sample_keys']}`",
        f"- duplicate frame keys: `{report['duplicate_frame_keys']}`",
        f"- train disagreement positive fraction: `{report['train_disagreement_label_summary']['positive_fraction']}`",
        f"- validation disagreement positive fraction: `{report['validation_disagreement_label_summary']['positive_fraction']}`",
        f"- gate probe margin: `{report['gate_probe_summary']['accuracy_margin']}`",
        f"- full-vs-L1 target mean L2: `{report['target_distinction_metrics_validation']['full_vs_l1_proxy_target_mean_l2']}`",
        f"- full-vs-no-gate target mean L2: `{report['target_distinction_metrics_validation']['full_vs_no_gate_target_mean_l2']}`",
        f"- full-vs-static target mean L2: `{report['target_distinction_metrics_validation']['full_vs_static_target_mean_l2']}`",
        f"- base action L2 validation: `{report['base_action_l2_validation']}`",
        f"- mean action L2 validation: `{report['mean_action_l2_validation']}`",
        f"- preexisting LoRA action L2 validation: `{report['lora_action_l2_validation']}`",
        f"- initial action delta p95: `{report['initial_action_delta_p95']}`",
        f"- base action validity: `{report['base_action_validity']}`",
        "",
        "Disagreement thresholds:",
        "",
        "```json",
        json.dumps(report.get("disagreement_thresholds"), indent=2, sort_keys=True),
        "```",
        "",
        "Gate probe summary:",
        "",
        "```json",
        json.dumps(report.get("gate_probe_summary"), indent=2, sort_keys=True),
        "```",
        "",
        "Split manifest:",
        "",
        "```json",
        json.dumps(report.get("split_manifest"), indent=2, sort_keys=True),
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
    selected = report.get("selected_config") or {}
    selected_score = selected.get("score_terms") or {}
    selected_metrics = selected.get("validation_metrics") or {}
    lines = [
        "# MARC-VLA Validation Search",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- lightweight validation training happened: `{report['training_happened']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        f"- audit final decision: `{report['audit_final_decision']}`",
        f"- search budget: `{report['search_budget']}`",
        f"- tried configs: `{report['tried_config_count']}`",
        f"- selected config: `{selected.get('config_id')}`",
        f"- selected correction alpha: `{selected.get('correction_alpha')}`",
        f"- selected gate architecture: `{selected.get('gate_architecture')}`",
        f"- selected score: `{selected_score.get('total')}`",
        f"- selected delta L2 p95: `{selected_metrics.get('delta_l2_p95')}`",
        f"- selected clean delta L2 p95: `{selected_metrics.get('clean_delta_l2_p95')}`",
        f"- selected action validity: `{selected_metrics.get('action_validity')}`",
        f"- selected MARC action L2: `{selected_metrics.get('marc_full_action_l2')}`",
        f"- selected L1 proxy action L2: `{selected_metrics.get('l1_proxy_action_l2')}`",
        "",
        "Score weights:",
        "",
        "```json",
        json.dumps(report.get("score_weights"), indent=2, sort_keys=True),
        "```",
        "",
        "Selected config:",
        "",
        "```json",
        json.dumps(selected, indent=2, sort_keys=True),
        "```",
        "",
        "Tried configurations:",
        "",
        "| config | decision | alpha | arch | proxy | gate | clean | distinction | validity | compute | total |",
        "| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.get("tried_configs", []):
        score = item.get("score_terms") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.get('config_id')}`",
                    f"`{item.get('final_decision')}`",
                    f"{item.get('correction_alpha')}",
                    f"`{item.get('gate_architecture')}`",
                    f"{score.get('l1_proxy_validity_and_full_proxy_distinction')}",
                    f"{score.get('gate_predictability')}",
                    f"{score.get('clean_retention_and_bounded_delta')}",
                    f"{score.get('full_ablation_static_distinction')}",
                    f"{score.get('action_validity')}",
                    f"{score.get('compute_overhead')}",
                    f"{score.get('total')}",
                ]
            )
            + " |"
        )
    lines.extend(["", f"Next step: {report['next_step']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit", "validation", "all"], default="audit")
    parser.add_argument("--prediction-artifact", default="reports/official_smolvla_stable_prediction_artifact.json")
    parser.add_argument("--json-output", default="reports/marc_vla/development_audit.json")
    parser.add_argument("--md-output", default="reports/marc_vla/development_audit.md")
    parser.add_argument("--label-output", default="reports/marc_vla/disagreement_label_manifest.json")
    parser.add_argument("--split-output", default="reports/marc_vla/split_manifest.json")
    parser.add_argument("--validation-json-output", default="reports/marc_vla/validation_search.json")
    parser.add_argument("--validation-md-output", default="reports/marc_vla/validation_search.md")
    parser.add_argument("--selected-config-output", default="reports/marc_vla/selected_config.json")
    parser.add_argument("--checkpoint-dir", default="reports/marc_vla/validation_checkpoints")
    args = parser.parse_args()

    artifact = _read_json(Path(args.prediction_artifact))
    summary: dict[str, Any] = {"mode": args.mode, "source_prediction_artifact": str(args.prediction_artifact)}
    if args.mode in {"audit", "all"}:
        report = audit_marc_records(artifact["records"])
        report = {
            **report,
            "date_kst": DATE_KST,
            "mode": "audit",
            "source_prediction_artifact": str(args.prediction_artifact),
        }
        _write_json(Path(args.json_output), report)
        _write_md(Path(args.md_output), report)
        _write_json(Path(args.label_output), report["disagreement_label_manifest"])
        _write_json(Path(args.split_output), report["split_manifest"])
        summary.update(
            {
                "audit_decision": report["final_decision"],
                "audit_hard_stop_count": len(report["hard_stop_reasons"]),
                "json_output": args.json_output,
                "md_output": args.md_output,
            }
        )
    if args.mode in {"validation", "all"}:
        validation = run_validation_search(artifact["records"], output_dir=args.checkpoint_dir)
        validation = {
            **validation,
            "date_kst": DATE_KST,
            "mode": "validation",
            "source_prediction_artifact": str(args.prediction_artifact),
        }
        _write_json(Path(args.validation_json_output), validation)
        _write_validation_md(Path(args.validation_md_output), validation)
        if validation.get("selected_config"):
            _write_json(Path(args.selected_config_output), validation["selected_config"])
        summary.update(
            {
                "validation_decision": validation["final_decision"],
                "selected_config": (validation.get("selected_config") or {}).get("config_id"),
                "tried_config_count": validation["tried_config_count"],
                "validation_json_output": args.validation_json_output,
                "validation_md_output": args.validation_md_output,
            }
        )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
