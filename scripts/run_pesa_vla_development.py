"""Run PESA-VLA development-only Stage 0 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.pesa_vla import PROPOSAL_HASH, audit_pesa_records  # noqa: E402


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
    target_metrics = report.get("target_distinction_metrics_validation") or {}
    gradient = report.get("gradient_audit") or {}
    simple = target_metrics.get("simple_killer") or {}
    lines = [
        "# PESA-VLA Development Audit",
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
        f"- train query positive fraction: `{report['train_query_label_summary']['positive_fraction']}`",
        f"- validation query positive fraction: `{report['validation_query_label_summary']['positive_fraction']}`",
        f"- query probe margin: `{report['query_probe_summary']['accuracy_margin']}`",
        f"- standard LoRA headroom L1 validation: `{report['standard_lora_headroom_l1_validation']}`",
        f"- validation spectral active-rank mean: `{report['validation_spectral_summary']['active_rank_mean']}`",
        f"- validation spectral active-fraction mean: `{report['validation_spectral_summary']['active_fraction_mean']}`",
        f"- full-vs-prior proxy mean L2: `{target_metrics.get('full_vs_priorvla_proxy_mean_l2')}`",
        f"- full-vs-ablation mean L2: `{target_metrics.get('full_vs_no_spectral_no_query_mean_l2')}`",
        f"- full-vs-simple killer mean L2: `{target_metrics.get('full_vs_selected_simple_killer_mean_l2')}`",
        f"- selected simple killer: `{simple.get('selected_simple_killer')}`",
        f"- gradient norm ratio: `{gradient.get('gradient_norm_ratio_largest_to_smallest')}`",
        f"- initial action delta p95: `{report['initial_action_delta_p95']}`",
        f"- base action validity: `{report['base_action_validity']}`",
        "",
        "Query thresholds:",
        "",
        "```json",
        json.dumps(report.get("query_thresholds"), indent=2, sort_keys=True),
        "```",
        "",
        "Query probe summary:",
        "",
        "```json",
        json.dumps(report.get("query_probe_summary"), indent=2, sort_keys=True),
        "```",
        "",
        "Spectral summary:",
        "",
        "```json",
        json.dumps(report.get("validation_spectral_summary"), indent=2, sort_keys=True),
        "```",
        "",
        "Gradient audit:",
        "",
        "```json",
        json.dumps(gradient, indent=2, sort_keys=True),
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit"], default="audit")
    parser.add_argument("--prediction-artifact", default="reports/official_smolvla_stable_prediction_artifact.json")
    parser.add_argument("--json-output", default="reports/pesa_vla/development_audit.json")
    parser.add_argument("--md-output", default="reports/pesa_vla/development_audit.md")
    parser.add_argument("--query-label-output", default="reports/pesa_vla/query_label_manifest.json")
    parser.add_argument("--spectral-output", default="reports/pesa_vla/spectral_activation_manifest.json")
    parser.add_argument("--split-output", default="reports/pesa_vla/split_manifest.json")
    args = parser.parse_args()

    artifact = _read_json(Path(args.prediction_artifact))
    report = audit_pesa_records(artifact["records"])
    report = {
        **report,
        "date_kst": DATE_KST,
        "mode": "audit",
        "source_prediction_artifact": str(args.prediction_artifact),
    }
    _write_json(Path(args.json_output), report)
    _write_md(Path(args.md_output), report)
    _write_json(Path(args.query_label_output), report["query_label_manifest"])
    _write_json(Path(args.spectral_output), report["spectral_activation_manifest"])
    _write_json(Path(args.split_output), report["split_manifest"])
    summary = {
        "mode": args.mode,
        "source_prediction_artifact": str(args.prediction_artifact),
        "audit_decision": report["final_decision"],
        "audit_hard_stop_count": len(report["hard_stop_reasons"]),
        "json_output": args.json_output,
        "md_output": args.md_output,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
