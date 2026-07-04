"""Planning-only action-stat provenance correction path."""

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


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Action-Stat Provenance Correction Plan Report",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- plan passed: `{report.get('action_stat_provenance_correction_plan_passed')}`",
        f"- rollout scaling ready: `{report.get('ready_for_rollout_scaling')}`",
        f"- ready for LIBERO action-stat audit: `{report.get('ready_for_libero_action_stat_audit')}`",
        f"- selected next step: `{report.get('selected_next_step')}`",
        "",
        "Correction options:",
        "",
    ]
    for option in report.get("correction_options") or []:
        lines.append(f"- `{option.get('id')}`: {option.get('description')} Decision: `{option.get('decision')}`.")
    lines.extend(["", f"Recommended next step: {report.get('recommended_next_step')}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    audit_path = Path(args.audit_report)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "evidence_label": "action_stat_provenance_correction_plan",
        "action_stat_provenance_correction_plan_passed": False,
        "decision": "stop",
        "ready_for_rollout_scaling": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "ready_for_libero_action_stat_audit": False,
        "selected_next_step": None,
        "policy": {
            "planning_only": True,
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
            "forbidden_gates_set": forbidden,
        },
        "paths": {"audit_report": str(audit_path)},
        "audit_summary": {},
        "correction_options": [],
        "recommended_next_step": None,
        "error": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["decision"] = "stop"
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        return report, code

    if forbidden:
        return block("Forbidden gate(s) set for planning-only task: " + ", ".join(forbidden), 2)
    if not audit_path.exists():
        return block(f"Action normalization provenance audit report is missing: {audit_path}", 3)

    audit = _read_json(audit_path)
    if not audit.get("action_normalization_provenance_audit_passed"):
        return block("Action normalization provenance audit did not pass.", 4)

    diagnosis = audit.get("diagnosis") or {}
    action_stats = audit.get("action_stats") or {}
    sample_ranges = audit.get("sample_action_ranges") or {}
    report["audit_summary"] = {
        "audit_decision": audit.get("decision"),
        "checkpoint_action_stats_appear_non_libero_scale": diagnosis.get("checkpoint_action_stats_appear_non_libero_scale"),
        "checkpoint_action_stats_prefix_mismatch_risk": diagnosis.get("checkpoint_action_stats_prefix_mismatch_risk"),
        "libero_expert_actions_appear_unit_scaled": diagnosis.get("libero_expert_actions_appear_unit_scaled"),
        "policy_action_shape": diagnosis.get("policy_action_shape"),
        "config_action_normalization": diagnosis.get("config_action_normalization"),
        "action_stat_prefixes": action_stats.get("action_stat_prefixes"),
        "action_mean_range": action_stats.get("action_mean_range"),
        "action_std_range": action_stats.get("action_std_range"),
        "expert_action_preview_range": sample_ranges.get("expert_action_preview_range"),
        "clipped_values_total": sample_ranges.get("clipped_values_total"),
    }
    mismatch = bool(
        diagnosis.get("checkpoint_action_stats_appear_non_libero_scale")
        or diagnosis.get("checkpoint_action_stats_prefix_mismatch_risk")
    )
    options = [
        {
            "id": "libero_action_stat_subset_audit",
            "description": "Compute action mean/std/range directly from a bounded local LIBERO HDF5 subset and compare against checkpoint processor stats.",
            "decision": "selected",
            "risk": "low",
            "downloads": False,
            "model_load": False,
            "rollout": False,
            "training": False,
        },
        {
            "id": "normalized_action_space_probe",
            "description": "Plan a future bounded offline diagnostic that compares normalized policy/action spaces before postprocessor-scale assumptions are used.",
            "decision": "defer_until_libero_stats_audit",
            "risk": "moderate",
            "downloads": False,
            "model_load": "future_gated",
            "rollout": False,
            "training": False,
        },
        {
            "id": "postprocessor_bypass_or_replacement",
            "description": "Do not modify runtime behavior yet; only consider after LIBERO action stats confirm the mismatch and a separate implementation plan exists.",
            "decision": "blocked_until_stats_audit",
            "risk": "moderate",
            "downloads": False,
            "model_load": "future_gated",
            "rollout": False,
            "training": False,
        },
        {
            "id": "checkpoint_task_provenance_resolution",
            "description": "Resolve whether the local SmolVLA checkpoint is intended for LIBERO or SO100-style action stats before treating rollout failures as policy failures.",
            "decision": "defer_to_documented_source_check",
            "risk": "low",
            "downloads": False,
            "model_load": False,
            "rollout": False,
            "training": False,
        },
    ]
    report["correction_options"] = options
    report["ready_for_libero_action_stat_audit"] = bool(mismatch)
    report["selected_next_step"] = "libero_action_stat_subset_audit" if mismatch else "manual_review"
    report["decision"] = "reduce_scope" if mismatch else "stop"
    report["action_stat_provenance_correction_plan_passed"] = True
    report["recommended_next_step"] = (
        "Implement a report-only LIBERO action-stat subset audit over local HDF5 files. Do not change policy behavior or run rollouts yet."
        if mismatch
        else "No strong action-stat mismatch was detected; review audit inputs before proceeding."
    )
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit-report", default="reports/action_normalization_provenance_audit_report.json")
    parser.add_argument("--report-path", default="reports/action_stat_provenance_correction_plan_report.json")
    parser.add_argument("--markdown-report-path", default="reports/action_stat_provenance_correction_plan_report.md")
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
