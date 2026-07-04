"""Planning-only normalized-action-space probe / provenance resolution gate."""

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
    "ALLOW_WSL_SMOLVLA_SINGLE_ACTION",
    "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
    "ALLOW_REPEATED_OFFLINE_DEMO_DECODING",
    "ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING",
    "ALLOW_NORMALIZED_ACTION_SPACE_PROBE",
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    evidence = report.get("evidence_summary") or {}
    lines = [
        "# Normalized Action-Space Probe Plan Report",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- plan passed: `{report.get('normalized_action_space_probe_plan_passed')}`",
        f"- selected next step: `{report.get('selected_next_step')}`",
        f"- ready for bounded normalized probe runner: `{report.get('ready_for_bounded_normalized_action_space_probe_runner')}`",
        f"- ready for checkpoint provenance resolver: `{report.get('ready_for_checkpoint_task_provenance_resolution')}`",
        f"- rollout scaling ready: `{report.get('ready_for_rollout_scaling')}`",
        "",
        "Evidence summary:",
        "",
        f"- LIBERO action dim: `{evidence.get('libero_action_dim')}`",
        f"- LIBERO action max abs: `{evidence.get('libero_action_max_abs')}`",
        f"- checkpoint action-stat prefixes: `{evidence.get('checkpoint_action_stat_prefixes')}`",
        f"- checkpoint action mean max abs: `{evidence.get('checkpoint_action_mean_max_abs')}`",
        f"- checkpoint action std max: `{evidence.get('checkpoint_action_std_max')}`",
        f"- scale mismatch confirmed: `{evidence.get('scale_mismatch_confirmed')}`",
        f"- dimension mismatch confirmed: `{evidence.get('dimension_mismatch_confirmed')}`",
        "",
        "Options:",
        "",
    ]
    for option in report.get("options") or []:
        lines.append(f"- `{option.get('id')}`: decision `{option.get('decision')}`. {option.get('reason')}")
    lines.extend(["", f"Recommended next step: {report.get('recommended_next_step')}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def _prefixes_look_so100(prefixes: Any) -> bool:
    if not isinstance(prefixes, list):
        return False
    lowered = [str(item).lower() for item in prefixes]
    return any(item.startswith("so100") for item in lowered)


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    audit_path = Path(args.libero_action_stat_report)
    vlm_summary_path = Path(args.vlm_summary_report)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "evidence_label": "normalized_action_space_probe_plan",
        "normalized_action_space_probe_plan_passed": False,
        "decision": "stop",
        "selected_next_step": None,
        "ready_for_bounded_normalized_action_space_probe_runner": False,
        "ready_for_checkpoint_task_provenance_resolution": False,
        "ready_for_offline_head_tca_pivot": False,
        "ready_for_rollout_scaling": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
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
            "policy_behavior_changed": False,
            "forbidden_gates_set": forbidden,
        },
        "paths": {
            "libero_action_stat_report": str(audit_path),
            "vlm_summary_report": str(vlm_summary_path),
        },
        "evidence_summary": {},
        "options": [],
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
        return block(f"LIBERO action-stat subset audit report is missing: {audit_path}", 3)

    audit = _read_json(audit_path)
    if not audit.get("libero_action_stat_subset_audit_passed"):
        return block("LIBERO action-stat subset audit did not pass.", 4)

    comparison = audit.get("comparison_to_checkpoint") or {}
    stats = audit.get("libero_action_stats") or {}
    prefixes = comparison.get("checkpoint_action_stat_prefixes") or []
    scale_mismatch = bool(comparison.get("scale_mismatch_confirmed"))
    dimension_mismatch = bool(comparison.get("dimension_mismatch_confirmed"))
    so100_prefix = _prefixes_look_so100(prefixes)
    vlm_summary_present = vlm_summary_path.exists()
    vlm_alignment_signal = None
    if vlm_summary_present:
        vlm_summary = _read_json(vlm_summary_path)
        vlm_alignment_signal = (
            (vlm_summary.get("comparison") or {}).get("vlm_enabled_alignment_signal")
            or vlm_summary.get("vlm_enabled_alignment_signal")
        )

    report["evidence_summary"] = {
        "audit_decision": audit.get("decision"),
        "libero_action_dim": stats.get("dim"),
        "libero_action_max_abs": stats.get("max_abs"),
        "libero_action_count": stats.get("count"),
        "checkpoint_action_stat_prefixes": prefixes,
        "checkpoint_action_mean_max_abs": comparison.get("checkpoint_action_mean_max_abs"),
        "checkpoint_action_std_max": comparison.get("checkpoint_action_std_max"),
        "policy_action_shape": comparison.get("policy_action_shape"),
        "scale_mismatch_confirmed": scale_mismatch,
        "dimension_mismatch_confirmed": dimension_mismatch,
        "checkpoint_prefixes_look_so100": so100_prefix,
        "vlm_summary_present": vlm_summary_present,
        "vlm_enabled_alignment_signal": vlm_alignment_signal,
    }

    strong_provenance_mismatch = bool(so100_prefix and scale_mismatch and dimension_mismatch)
    mismatch = bool(scale_mismatch or dimension_mismatch)
    options = [
        {
            "id": "checkpoint_task_provenance_resolution",
            "decision": "selected" if strong_provenance_mismatch else "available",
            "risk": "low",
            "reason": "SO100-prefixed checkpoint processor stats and unit-scale 7D LIBERO actions indicate task/action provenance must be resolved before behavior changes.",
            "future_gate_required": False,
            "downloads": False,
            "model_load": False,
            "rollout": False,
            "training": False,
        },
        {
            "id": "bounded_normalized_action_space_probe",
            "decision": "defer_until_provenance_resolved" if strong_provenance_mismatch else ("selected" if mismatch else "not_needed"),
            "risk": "moderate",
            "reason": "A future probe may compare normalized action-space quantities, but it should be separately gated because it may require CPU model inference or postprocessor bypass instrumentation.",
            "future_gate_required": True,
            "downloads": False,
            "model_load": "future_gated",
            "rollout": False,
            "training": False,
        },
        {
            "id": "postprocessor_bypass_or_replacement",
            "decision": "blocked",
            "risk": "moderate",
            "reason": "Do not change action unnormalization or policy behavior until provenance is resolved and a separate probe explains what quantity is being bypassed.",
            "future_gate_required": True,
            "downloads": False,
            "model_load": "future_gated",
            "rollout": False,
            "training": False,
        },
        {
            "id": "offline_head_tca_map_pivot",
            "decision": "fallback_if_no_libero_aligned_checkpoint",
            "risk": "low",
            "reason": "If this checkpoint is not LIBERO-action aligned, learned-policy rollout work should be deprioritized while offline TCA-Map/ActionMap/LoRA evidence continues.",
            "future_gate_required": False,
            "downloads": False,
            "model_load": False,
            "rollout": False,
            "training": "future_bounded_head_or_lora_only",
        },
    ]
    report["options"] = options

    if strong_provenance_mismatch:
        report["decision"] = "reduce_scope"
        report["selected_next_step"] = "checkpoint_task_provenance_resolution"
        report["ready_for_checkpoint_task_provenance_resolution"] = True
        report["ready_for_bounded_normalized_action_space_probe_runner"] = False
        report["ready_for_offline_head_tca_pivot"] = True
        report["recommended_next_step"] = (
            "Create a report-only checkpoint/task provenance resolution audit before any normalized-action probe, policy behavior change, or rollout scaling."
        )
    elif mismatch:
        report["decision"] = "reduce_scope"
        report["selected_next_step"] = "bounded_normalized_action_space_probe_plan"
        report["ready_for_checkpoint_task_provenance_resolution"] = True
        report["ready_for_bounded_normalized_action_space_probe_runner"] = False
        report["recommended_next_step"] = (
            "Create a separately gated normalized-action-space probe plan; keep the future runner offline, CPU-first, and non-rollout."
        )
    else:
        report["decision"] = "review_required"
        report["selected_next_step"] = "manual_review"
        report["recommended_next_step"] = "No confirmed action-stat mismatch was found; review diagnostics before proceeding."

    report["normalized_action_space_probe_plan_passed"] = True
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--libero-action-stat-report", default="reports/libero_action_stat_subset_audit_report.json")
    parser.add_argument("--vlm-summary-report", default="reports/vlm_enabled_offline_decoding_summary_report.json")
    parser.add_argument("--report-path", default="reports/normalized_action_space_probe_plan_report.json")
    parser.add_argument("--markdown-report-path", default="reports/normalized_action_space_probe_plan_report.md")
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
