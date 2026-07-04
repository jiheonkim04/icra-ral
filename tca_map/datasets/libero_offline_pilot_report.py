"""Summary-only bounded LIBERO offline pilot report.

This module consolidates local runtime reports for the LIBERO offline proxy
ladder. It reads reports only; it does not download, train, use GPU, load
models, run inference, execute simulators, run rollouts, or make paper claims.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

SCHEMA_VERSION = "tca-map-libero-offline-bounded-pilot-report-v0"


def _load_json(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(report: dict | None, arm_name: str, metric_name: str) -> float | None:
    if not isinstance(report, dict):
        return None
    arms = report.get("arms")
    if isinstance(arms, dict):
        payload = arms.get(arm_name) or {}
        value = (payload.get("metrics") or {}).get(metric_name)
        return None if value is None else float(value)
    if isinstance(arms, list):
        for payload in arms:
            if payload.get("arm") == arm_name:
                value = (payload.get("metrics") or {}).get(metric_name)
                return None if value is None else float(value)
    return None


def _delta(report: dict | None, left: str, right: str, metric_name: str) -> float | None:
    left_value = _metric(report, left, metric_name)
    right_value = _metric(report, right, metric_name)
    if left_value is None or right_value is None:
        return None
    return round(left_value - right_value, 6)


def build_libero_offline_bounded_pilot_report(
    interface_report_path: Path,
    split_report_path: Path,
    head_report_path: Path,
    lora_report_path: Path,
) -> dict:
    interface = _load_json(interface_report_path)
    split = _load_json(split_report_path)
    head = _load_json(head_report_path)
    lora = _load_json(lora_report_path)

    required_reports = {
        "interface": interface is not None,
        "counterfactual_split": split is not None,
        "head_comparison": head is not None,
        "lora_comparison": lora is not None,
    }
    gates = {
        "ready_for_offline_interface_smoke": bool((interface or {}).get("ready_for_offline_interface_smoke")),
        "ready_for_tiny_offline_counterfactual_split": bool((split or {}).get("ready_for_tiny_offline_counterfactual_split")),
        "libero_offline_head_comparison_passed": bool((head or {}).get("libero_offline_head_comparison_passed")),
        "libero_offline_lora_comparison_passed": bool((lora or {}).get("libero_offline_lora_comparison_passed")),
        "ready_for_rollout": bool((interface or {}).get("ready_for_rollout")),
    }
    passed = bool(
        all(required_reports.values())
        and gates["ready_for_offline_interface_smoke"]
        and gates["ready_for_tiny_offline_counterfactual_split"]
        and gates["libero_offline_head_comparison_passed"]
        and gates["libero_offline_lora_comparison_passed"]
        and not gates["ready_for_rollout"]
    )

    head_summary = {
        "pair_count": (head or {}).get("pair_count"),
        "tca_map_vs_actionmap": (head or {}).get("comparison", {}).get("tca_map_vs_actionmap"),
        "tca_select_vs_tca_map": (head or {}).get("comparison", {}).get("tca_select_vs_tca_map"),
        "actionmap_action_l1": _metric(head, "actionmap_head_only_proxy", "action_l1"),
        "tca_map_action_l1": _metric(head, "tca_map_head_only_proxy", "action_l1"),
        "tca_select_action_l1": _metric(head, "tca_map_distributional_select_proxy", "action_l1"),
    }
    lora_summary = {
        "record_count": (lora or {}).get("record_count"),
        "action_prefix_dim": (lora or {}).get("action_prefix_dim"),
        "max_steps": (lora or {}).get("max_steps"),
        "lora_rank": (lora or {}).get("lora_rank"),
        "tca_lora_vs_actionmap_lora": (lora or {}).get("comparison", {}).get("tca_lora_vs_actionmap_lora"),
        "tca_select_lora_vs_tca_lora": (lora or {}).get("comparison", {}).get("tca_select_lora_vs_tca_lora"),
        "action_l1_delta_tca_lora_minus_actionmap_lora": _delta(lora, "tca_map_lora", "actionmap_lora", "action_l1"),
        "wrong_target_delta_tca_lora_minus_actionmap_lora": _delta(
            lora, "tca_map_lora", "actionmap_lora", "wrong_target_proxy_rate"
        ),
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "summary_only": True,
            "bounded_local_pilot": True,
            "local_libero_hdf5_used_by_source_reports": True,
            "offline_proxy_only": True,
            "not_standard_success": True,
            "not_rollout_success": True,
            "not_paper_grade": True,
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "gpu_training_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "training_performed_by_this_report": False,
            "rollouts_performed": False,
            "simulator_executed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
        },
        "source_reports": {
            "interface": str(interface_report_path),
            "counterfactual_split": str(split_report_path),
            "head_comparison": str(head_report_path),
            "lora_comparison": str(lora_report_path),
        },
        "required_reports_present": required_reports,
        "gates": gates,
        "dataset_summary": {
            "hdf5_inventory_count": (split or {}).get("hdf5_inventory_count"),
            "counterfactual_pair_count": (split or {}).get("counterfactual_pair_count"),
            "matched_task_count": (split or {}).get("matched_task_count"),
            "suites": (split or {}).get("suites", []),
        },
        "head_only_summary": head_summary,
        "required_lora_summary": lora_summary,
        "libero_offline_bounded_pilot_report_passed": passed,
        "ready_for_simulator_readiness_risk_assessment": passed,
        "ready_for_rollout": False,
        "blocked_for_paper_grade_claims": True,
        "blocked_by": [
            "offline proxy metrics are not standard success",
            "no simulator rollout success is available",
            "no paper-grade empirical claim is permitted from this report",
        ],
        "recommended_next_step": (
            "Run a simulator readiness/import-render risk assessment if installed locally; stop before rollout unless the risk assessment is green."
            if passed
            else "Regenerate missing LIBERO offline interface, split, head comparison, or LoRA comparison reports."
        ),
    }


def write_reports(report: dict, report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LIBERO Offline Bounded Pilot Report",
        "",
        "This report is summary-only. It is not standard success, not rollout success, and not paper-grade evidence.",
        "",
        f"- passed: `{report['libero_offline_bounded_pilot_report_passed']}`",
        f"- ready for simulator readiness risk assessment: `{report['ready_for_simulator_readiness_risk_assessment']}`",
        f"- ready for rollout: `{report['ready_for_rollout']}`",
        "",
        "## Dataset",
    ]
    for key, value in report["dataset_summary"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Gates"])
    for key, value in report["gates"].items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Next Step", report["recommended_next_step"], ""])
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interface-report", default="reports/libero_offline_interface_smoke_report.json")
    parser.add_argument("--split-report", default="reports/libero_offline_counterfactual_split_report.json")
    parser.add_argument("--head-report", default="reports/libero_offline_actionmap_tca_comparison_report.json")
    parser.add_argument("--lora-report", default="reports/libero_offline_lora_comparison_report.json")
    parser.add_argument("--report-json", default="reports/libero_offline_bounded_pilot_report.json")
    parser.add_argument("--report-md", default="reports/libero_offline_bounded_pilot_report.md")
    args = parser.parse_args()

    report = build_libero_offline_bounded_pilot_report(
        interface_report_path=Path(args.interface_report),
        split_report_path=Path(args.split_report),
        head_report_path=Path(args.head_report),
        lora_report_path=Path(args.lora_report),
    )
    write_reports(report, report_json=Path(args.report_json), report_md=Path(args.report_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
