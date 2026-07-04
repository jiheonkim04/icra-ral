"""Report-only pivot plan after checkpoint/LIBERO rollout provenance no-go."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SIMULATOR_IMPORT_SMOKE",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_TINY_ROLLOUT",
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _metric(report: dict[str, Any], arm_name: str, metric_name: str) -> float | None:
    arms = report.get("arms")
    if isinstance(arms, dict):
        value = ((arms.get(arm_name) or {}).get("metrics") or {}).get(metric_name)
        return None if value is None else float(value)
    if isinstance(arms, list):
        for arm in arms:
            if arm.get("arm") == arm_name:
                value = (arm.get("metrics") or {}).get(metric_name)
                return None if value is None else float(value)
    return None


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    evidence = report.get("offline_evidence_summary") or {}
    lines = [
        "# Offline TCA-Map / LoRA Pivot Plan Report",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- plan passed: `{report.get('offline_tca_map_lora_pivot_plan_passed')}`",
        f"- selected next step: `{report.get('selected_next_step')}`",
        f"- ready for offline evidence table: `{report.get('ready_for_offline_evidence_table')}`",
        f"- ready for learned-policy rollout scaling: `{report.get('ready_for_learned_policy_rollout_scaling')}`",
        f"- ready for paper claim: `{report.get('ready_for_paper_claim')}`",
        "",
        "Current offline evidence:",
        "",
        f"- head comparison passed: `{evidence.get('head_comparison_passed')}`",
        f"- LoRA comparison passed: `{evidence.get('lora_comparison_passed')}`",
        f"- bounded pilot report passed: `{evidence.get('bounded_pilot_report_passed')}`",
        f"- TCA head vs ActionMap action L1 delta: `{evidence.get('tca_head_action_l1_delta_vs_actionmap')}`",
        f"- TCA LoRA vs ActionMap LoRA action L1 delta: `{evidence.get('tca_lora_action_l1_delta_vs_actionmap_lora')}`",
        f"- wrong-target proxy delta, TCA LoRA vs ActionMap LoRA: `{evidence.get('tca_lora_wrong_target_delta_vs_actionmap_lora')}`",
        "",
        "Blocked:",
        "",
    ]
    for blocker in report.get("blocked_by") or []:
        lines.append(f"- {blocker}")
    lines.extend(["", f"Recommended next step: {report.get('recommended_next_step')}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    provenance_path = Path(args.provenance_report)
    head_path = Path(args.head_report)
    lora_path = Path(args.lora_report)
    bounded_path = Path(args.bounded_pilot_report)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "evidence_label": "offline_tca_map_lora_pivot_plan",
        "offline_tca_map_lora_pivot_plan_passed": False,
        "decision": "stop",
        "selected_next_step": None,
        "ready_for_offline_evidence_table": False,
        "ready_for_lora_scaleup_plan": False,
        "ready_for_libero_aligned_checkpoint_source_plan": False,
        "ready_for_learned_policy_rollout_scaling": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "policy": {
            "report_only": True,
            "downloads_performed": False,
            "installs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "simulator_environment_created": False,
            "rollouts_performed": False,
            "benchmark_rollouts_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
            "policy_behavior_changed": False,
            "forbidden_gates_set": forbidden,
        },
        "paths": {
            "provenance_report": str(provenance_path),
            "head_report": str(head_path),
            "lora_report": str(lora_path),
            "bounded_pilot_report": str(bounded_path),
        },
        "offline_evidence_summary": {},
        "blocked_by": [],
        "recommended_next_step": None,
        "error": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["decision"] = "stop"
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        return report, code

    if forbidden:
        return block("Forbidden gate(s) set for report-only pivot plan: " + ", ".join(forbidden), 2)
    for required_path in [provenance_path, head_path, lora_path, bounded_path]:
        if not required_path.exists():
            return block(f"Required report is missing: {required_path}", 3)

    provenance = _read_json(provenance_path)
    head = _read_json(head_path)
    lora = _read_json(lora_path)
    bounded = _read_json(bounded_path)
    if not provenance.get("checkpoint_task_provenance_resolution_passed"):
        return block("Checkpoint/task provenance resolution did not pass.", 4)

    learned_policy_no_go = provenance.get("decision") == "no_go_learned_policy_rollout_scaling"
    head_passed = bool(head.get("libero_offline_head_comparison_passed"))
    lora_passed = bool(lora.get("libero_offline_lora_comparison_passed"))
    bounded_passed = bool(bounded.get("libero_offline_bounded_pilot_report_passed"))
    tca_head_delta = ((head.get("comparison") or {}).get("tca_map_vs_actionmap") or {}).get("action_l1_delta")
    tca_lora_delta = ((lora.get("comparison") or {}).get("tca_lora_vs_actionmap_lora") or {}).get("action_l1_delta")
    tca_lora_wrong_delta = ((lora.get("comparison") or {}).get("tca_lora_vs_actionmap_lora") or {}).get(
        "wrong_target_proxy_rate_delta"
    )
    report["offline_evidence_summary"] = {
        "head_comparison_passed": head_passed,
        "lora_comparison_passed": lora_passed,
        "bounded_pilot_report_passed": bounded_passed,
        "learned_policy_rollout_no_go_from_provenance": learned_policy_no_go,
        "tca_head_action_l1_delta_vs_actionmap": tca_head_delta,
        "tca_lora_action_l1_delta_vs_actionmap_lora": tca_lora_delta,
        "tca_lora_wrong_target_delta_vs_actionmap_lora": tca_lora_wrong_delta,
        "tca_head_wrong_target_proxy_rate": _metric(head, "tca_map_head_only_proxy", "wrong_target_proxy_rate"),
        "actionmap_head_wrong_target_proxy_rate": _metric(head, "actionmap_head_only_proxy", "wrong_target_proxy_rate"),
        "tca_lora_wrong_target_proxy_rate": _metric(lora, "tca_map_lora", "wrong_target_proxy_rate"),
        "actionmap_lora_wrong_target_proxy_rate": _metric(lora, "actionmap_lora", "wrong_target_proxy_rate"),
    }
    blockers = [
        "current base checkpoint is not valid LIBERO learned-policy rollout evidence",
        "offline proxy metrics are not standard success",
        "no paper-grade simulator rollout success is available",
        "OpenVLA-OFT remains outside the current local path",
    ]
    report["blocked_by"] = blockers

    if learned_policy_no_go and head_passed and lora_passed and bounded_passed:
        report["decision"] = "pivot_offline_evidence_ladder"
        report["offline_tca_map_lora_pivot_plan_passed"] = True
        report["selected_next_step"] = "consolidate_offline_tca_lora_evidence_table_and_gap_report"
        report["ready_for_offline_evidence_table"] = True
        report["ready_for_lora_scaleup_plan"] = True
        report["ready_for_libero_aligned_checkpoint_source_plan"] = True
        report["recommended_next_step"] = (
            "Create a report-only offline evidence table and gap report that consolidates ActionMap, TCA-Map, Distributional TCA-Select, "
            "required LoRA, and remaining rollout/checkpoint blockers. Keep learned-policy rollout scaling blocked until a LIBERO-action-aligned checkpoint or bounded training path is validated."
        )
    elif not learned_policy_no_go:
        report["decision"] = "review_required"
        report["offline_tca_map_lora_pivot_plan_passed"] = True
        report["selected_next_step"] = "review_rollout_provenance_before_pivot"
        report["recommended_next_step"] = "Provenance did not block learned-policy rollout scaling; review source reports before pivoting."
    else:
        report["decision"] = "regenerate_offline_reports"
        report["selected_next_step"] = "rerun_missing_offline_head_or_lora_reports"
        report["recommended_next_step"] = "Regenerate missing LIBERO offline head, LoRA, or bounded pilot reports before building an evidence table."
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provenance-report", default="reports/checkpoint_task_provenance_resolution_report.json")
    parser.add_argument("--head-report", default="reports/libero_offline_actionmap_tca_comparison_report.json")
    parser.add_argument("--lora-report", default="reports/libero_offline_lora_comparison_report.json")
    parser.add_argument("--bounded-pilot-report", default="reports/libero_offline_bounded_pilot_report.json")
    parser.add_argument("--report-path", default="reports/offline_tca_map_lora_pivot_plan_report.json")
    parser.add_argument("--markdown-report-path", default="reports/offline_tca_map_lora_pivot_plan_report.md")
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_path)
    markdown_path = Path(args.markdown_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, markdown_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
