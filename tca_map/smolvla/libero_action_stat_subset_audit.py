"""Report-only LIBERO HDF5 action-stat subset audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


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
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _range(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "max_abs": None, "mean": None, "std": None}
    return {
        "count": int(values.shape[0]),
        "dim": int(values.shape[1]) if values.ndim == 2 else None,
        "min": [round(float(x), 6) for x in values.min(axis=0)],
        "max": [round(float(x), 6) for x in values.max(axis=0)],
        "mean": [round(float(x), 6) for x in values.mean(axis=0)],
        "std": [round(float(x), 6) for x in values.std(axis=0)],
        "max_abs": round(float(np.max(np.abs(values))), 6),
    }


def _collect_actions(data_root: Path, max_files: int, max_actions_per_file: int) -> tuple[np.ndarray, list[dict[str, Any]]]:
    import h5py

    files = sorted(data_root.rglob("*.hdf5"))[:max_files]
    chunks: list[np.ndarray] = []
    file_summaries: list[dict[str, Any]] = []
    for path in files:
        with h5py.File(path, "r") as handle:
            demo_names = sorted((handle.get("data") or {}).keys())
            file_count = 0
            file_dim = None
            for demo_name in demo_names:
                actions = np.asarray(handle["data"][demo_name]["actions"], dtype=np.float32)
                if actions.ndim != 2:
                    continue
                selected = actions[: max(0, max_actions_per_file - file_count)]
                if selected.size:
                    chunks.append(selected)
                    file_count += int(selected.shape[0])
                    file_dim = int(selected.shape[1])
                if file_count >= max_actions_per_file:
                    break
            file_summaries.append(
                {
                    "path": str(path),
                    "demo_count_seen": len(demo_names),
                    "actions_sampled": file_count,
                    "action_dim": file_dim,
                }
            )
    if not chunks:
        return np.zeros((0, 0), dtype=np.float32), file_summaries
    return np.concatenate(chunks, axis=0), file_summaries


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    stats = report.get("libero_action_stats") or {}
    cmp_ = report.get("comparison_to_checkpoint") or {}
    lines = [
        "# LIBERO Action-Stat Subset Audit Report",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- audit passed: `{report.get('libero_action_stat_subset_audit_passed')}`",
        f"- sampled files: `{report.get('sampled_file_count')}`",
        f"- sampled actions: `{stats.get('count')}`",
        f"- action dim: `{stats.get('dim')}`",
        f"- LIBERO max abs: `{stats.get('max_abs')}`",
        f"- checkpoint action mean max abs: `{cmp_.get('checkpoint_action_mean_max_abs')}`",
        f"- checkpoint action std max: `{cmp_.get('checkpoint_action_std_max')}`",
        f"- mismatch confirmed: `{cmp_.get('scale_mismatch_confirmed')}`",
        f"- rollout scaling ready: `{report.get('ready_for_rollout_scaling')}`",
        "",
        f"Recommended next step: {report.get('recommended_next_step')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    plan_path = Path(args.plan_report)
    data_root = Path(os.environ.get("LIBERO_DATA_ROOT") or args.libero_data_root)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "evidence_label": "libero_action_stat_subset_audit",
        "libero_action_stat_subset_audit_passed": False,
        "decision": "stop",
        "ready_for_rollout_scaling": False,
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
            "gpu_jobs_performed": False,
            "training_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
            "forbidden_gates_set": forbidden,
        },
        "paths": {"plan_report": str(plan_path), "libero_data_root": str(data_root)},
        "sampled_file_count": 0,
        "sampled_files": [],
        "libero_action_stats": {},
        "comparison_to_checkpoint": {},
        "recommended_next_step": None,
        "error": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["decision"] = "stop"
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        return report, code

    if forbidden:
        return block("Forbidden gate(s) set for report-only audit: " + ", ".join(forbidden), 2)
    if not plan_path.exists():
        return block(f"Action-stat provenance correction plan is missing: {plan_path}", 3)
    if not data_root.exists():
        return block(f"LIBERO data root is missing: {data_root}", 4)
    plan = _read_json(plan_path)
    if not plan.get("ready_for_libero_action_stat_audit"):
        return block("Correction plan did not authorize LIBERO action-stat audit.", 5)

    actions, files = _collect_actions(data_root, args.max_files, args.max_actions_per_file)
    if actions.size == 0:
        return block("No HDF5 action samples were found.", 6)
    stats = _range(actions)
    audit_summary = plan.get("audit_summary") or {}
    checkpoint_mean_max_abs = ((audit_summary.get("action_mean_range") or {}).get("max_abs")) or 0.0
    checkpoint_std_max = ((audit_summary.get("action_std_range") or {}).get("max")) or 0.0
    scale_mismatch = bool(stats.get("max_abs") is not None and stats["max_abs"] <= 1.1 and (checkpoint_mean_max_abs > 10 or checkpoint_std_max > 10))
    dim_mismatch = bool(stats.get("dim") != (audit_summary.get("policy_action_shape") or [None])[0])
    report["sampled_file_count"] = len(files)
    report["sampled_files"] = files
    report["libero_action_stats"] = stats
    report["comparison_to_checkpoint"] = {
        "checkpoint_action_mean_max_abs": checkpoint_mean_max_abs,
        "checkpoint_action_std_max": checkpoint_std_max,
        "checkpoint_action_stat_prefixes": audit_summary.get("action_stat_prefixes"),
        "scale_mismatch_confirmed": scale_mismatch,
        "dimension_mismatch_confirmed": dim_mismatch,
        "policy_action_shape": audit_summary.get("policy_action_shape"),
        "libero_action_dim": stats.get("dim"),
    }
    report["libero_action_stat_subset_audit_passed"] = True
    report["decision"] = "no_go_rollout_scaling" if (scale_mismatch or dim_mismatch) else "review_required"
    report["recommended_next_step"] = (
        "Plan a normalized-action-space probe or checkpoint/task provenance resolution before any learned-policy rollout."
        if scale_mismatch or dim_mismatch
        else "Review action-stat audit manually before choosing the next step."
    )
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-report", default="reports/action_stat_provenance_correction_plan_report.json")
    parser.add_argument("--libero-data-root", default="C:/assets/data/libero")
    parser.add_argument("--max-files", type=int, default=5)
    parser.add_argument("--max-actions-per-file", type=int, default=500)
    parser.add_argument("--report-path", default="reports/libero_action_stat_subset_audit_report.json")
    parser.add_argument("--markdown-report-path", default="reports/libero_action_stat_subset_audit_report.md")
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
