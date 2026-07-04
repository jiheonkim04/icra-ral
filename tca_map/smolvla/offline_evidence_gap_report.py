"""Report-only offline evidence table and gap report."""

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


def _metric_from_dict_arms(report: dict[str, Any], arm: str, metric: str) -> float | int | None:
    payload = ((report.get("arms") or {}).get(arm) or {})
    value = (payload.get("metrics") or {}).get(metric)
    return value


def _params_from_dict_arms(report: dict[str, Any], arm: str) -> int | None:
    value = ((report.get("arms") or {}).get(arm) or {}).get("trainable_parameter_count")
    return None if value is None else int(value)


def _metric_from_list_arms(report: dict[str, Any], arm: str, metric: str) -> float | int | None:
    for payload in report.get("arms") or []:
        if payload.get("arm") == arm:
            return (payload.get("metrics") or {}).get(metric)
    return None


def _metric(report: dict[str, Any], metric: str) -> float | int | None:
    return (report.get("metrics") or {}).get(metric)


def _params_from_list_arms(report: dict[str, Any], arm: str) -> int | None:
    for payload in report.get("arms") or []:
        if payload.get("arm") == arm:
            value = payload.get("trainable_lora_parameter_count")
            return None if value is None else int(value)
    return None


def _row(
    *,
    arm: str,
    evidence_type: str,
    action_l1: float | int | None,
    wrong_target_proxy_rate: float | int | None,
    counterfactual_margin: float | int | None,
    offline_proxy_score: float | int | None,
    target_top1: float | int | None,
    trainable_params: int | None,
) -> dict[str, Any]:
    return {
        "arm": arm,
        "evidence_type": evidence_type,
        "action_l1": action_l1,
        "wrong_target_proxy_rate": wrong_target_proxy_rate,
        "counterfactual_separation_margin": counterfactual_margin,
        "offline_standard_proxy": offline_proxy_score,
        "target_top1_accuracy": target_top1,
        "trainable_parameter_count": trainable_params,
        "not_standard_success": True,
        "not_paper_grade": True,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Offline TCA-Map / LoRA Evidence Gap Report",
        "",
        "This report consolidates real-LIBERO offline proxy evidence only. It is not standard success, not rollout success, and not paper-grade evidence.",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- report passed: `{report.get('offline_evidence_gap_report_passed')}`",
        f"- ready for LoRA scale-up plan: `{report.get('ready_for_lora_scaleup_plan')}`",
        f"- ready for learned-policy rollout scaling: `{report.get('ready_for_learned_policy_rollout_scaling')}`",
        f"- ready for paper claim: `{report.get('ready_for_paper_claim')}`",
        "",
        "| Arm | Evidence | Action L1 | Wrong-target proxy | CF margin | Offline proxy | Target top-1 | Trainable params |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report.get("evidence_table") or []:
        lines.append(
            "| {arm} | {evidence_type} | {action_l1} | {wrong_target_proxy_rate} | {counterfactual_separation_margin} | {offline_standard_proxy} | {target_top1_accuracy} | {trainable_parameter_count} |".format(
                **{key: row.get(key) for key in row}
            )
        )
    lines.extend(["", "## Gaps", ""])
    for gap in report.get("gap_table") or []:
        lines.append(f"- `{gap.get('id')}`: {gap.get('status')} - {gap.get('next_step')}")
    lines.extend(["", f"Recommended next step: {report.get('recommended_next_step')}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    pivot_path = Path(args.pivot_report)
    head_path = Path(args.head_report)
    lora_path = Path(args.lora_report)
    scaleup_path = Path(args.bounded_lora_scaleup_report)
    stress_path = Path(args.tca_select_stress_report)
    provenance_path = Path(args.provenance_report)
    bounded_path = Path(args.bounded_pilot_report)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "evidence_label": "offline_tca_lora_evidence_gap_report",
        "offline_evidence_gap_report_passed": False,
        "decision": "stop",
        "ready_for_lora_scaleup_plan": False,
        "ready_for_offline_proxy_extension": False,
        "ready_for_libero_aligned_checkpoint_source_plan": False,
        "ready_for_learned_policy_rollout_scaling": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "policy": {
            "report_only": True,
            "offline_proxy_only": True,
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
            "pivot_report": str(pivot_path),
            "head_report": str(head_path),
            "lora_report": str(lora_path),
            "bounded_lora_scaleup_report": str(scaleup_path),
            "tca_select_stress_report": str(stress_path),
            "provenance_report": str(provenance_path),
            "bounded_pilot_report": str(bounded_path),
        },
        "bounded_lora_scaleup_included": False,
        "bounded_lora_scaleup_record_count": 0,
        "tca_select_ambiguity_stress_included": False,
        "tca_select_ambiguity_stress_record_count": 0,
        "evidence_table": [],
        "gap_table": [],
        "deltas": {},
        "recommended_next_step": None,
        "error": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["decision"] = "stop"
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        return report, code

    if forbidden:
        return block("Forbidden gate(s) set for report-only evidence table: " + ", ".join(forbidden), 2)
    for required_path in [pivot_path, head_path, lora_path, provenance_path, bounded_path]:
        if not required_path.exists():
            return block(f"Required report is missing: {required_path}", 3)

    pivot = _read_json(pivot_path)
    head = _read_json(head_path)
    lora = _read_json(lora_path)
    scaleup = _read_json(scaleup_path) if scaleup_path.exists() else {}
    stress = _read_json(stress_path) if stress_path.exists() else {}
    provenance = _read_json(provenance_path)
    bounded = _read_json(bounded_path)
    if not pivot.get("ready_for_offline_evidence_table"):
        return block("Offline pivot plan did not authorize the evidence table.", 4)

    rows = [
        _row(
            arm="ActionMap head-only",
            evidence_type="real-LIBERO offline proxy",
            action_l1=_metric_from_dict_arms(head, "actionmap_head_only_proxy", "action_l1"),
            wrong_target_proxy_rate=_metric_from_dict_arms(head, "actionmap_head_only_proxy", "wrong_target_proxy_rate"),
            counterfactual_margin=_metric_from_dict_arms(
                head, "actionmap_head_only_proxy", "counterfactual_separation_margin"
            ),
            offline_proxy_score=_metric_from_dict_arms(head, "actionmap_head_only_proxy", "offline_standard_proxy"),
            target_top1=_metric_from_dict_arms(head, "actionmap_head_only_proxy", "target_top1_accuracy"),
            trainable_params=_params_from_dict_arms(head, "actionmap_head_only_proxy"),
        ),
        _row(
            arm="TCA-Map head-only",
            evidence_type="real-LIBERO offline proxy",
            action_l1=_metric_from_dict_arms(head, "tca_map_head_only_proxy", "action_l1"),
            wrong_target_proxy_rate=_metric_from_dict_arms(head, "tca_map_head_only_proxy", "wrong_target_proxy_rate"),
            counterfactual_margin=_metric_from_dict_arms(
                head, "tca_map_head_only_proxy", "counterfactual_separation_margin"
            ),
            offline_proxy_score=_metric_from_dict_arms(head, "tca_map_head_only_proxy", "offline_standard_proxy"),
            target_top1=_metric_from_dict_arms(head, "tca_map_head_only_proxy", "target_top1_accuracy"),
            trainable_params=_params_from_dict_arms(head, "tca_map_head_only_proxy"),
        ),
        _row(
            arm="TCA-Map + Distributional TCA-Select",
            evidence_type="real-LIBERO offline proxy",
            action_l1=_metric_from_dict_arms(head, "tca_map_distributional_select_proxy", "action_l1"),
            wrong_target_proxy_rate=_metric_from_dict_arms(
                head, "tca_map_distributional_select_proxy", "wrong_target_proxy_rate"
            ),
            counterfactual_margin=_metric_from_dict_arms(
                head, "tca_map_distributional_select_proxy", "counterfactual_separation_margin"
            ),
            offline_proxy_score=_metric_from_dict_arms(
                head, "tca_map_distributional_select_proxy", "offline_standard_proxy"
            ),
            target_top1=_metric_from_dict_arms(head, "tca_map_distributional_select_proxy", "target_top1_accuracy"),
            trainable_params=_params_from_dict_arms(head, "tca_map_distributional_select_proxy"),
        ),
        _row(
            arm="ActionMap + LoRA",
            evidence_type="real-LIBERO tiny offline LoRA proxy",
            action_l1=_metric_from_list_arms(lora, "actionmap_lora", "action_l1"),
            wrong_target_proxy_rate=_metric_from_list_arms(lora, "actionmap_lora", "wrong_target_proxy_rate"),
            counterfactual_margin=_metric_from_list_arms(lora, "actionmap_lora", "counterfactual_separation_margin"),
            offline_proxy_score=_metric_from_list_arms(lora, "actionmap_lora", "offline_standard_proxy"),
            target_top1=_metric_from_list_arms(lora, "actionmap_lora", "target_top1_accuracy"),
            trainable_params=_params_from_list_arms(lora, "actionmap_lora"),
        ),
        _row(
            arm="TCA-Map + LoRA",
            evidence_type="real-LIBERO tiny offline LoRA proxy",
            action_l1=_metric_from_list_arms(lora, "tca_map_lora", "action_l1"),
            wrong_target_proxy_rate=_metric_from_list_arms(lora, "tca_map_lora", "wrong_target_proxy_rate"),
            counterfactual_margin=_metric_from_list_arms(lora, "tca_map_lora", "counterfactual_separation_margin"),
            offline_proxy_score=_metric_from_list_arms(lora, "tca_map_lora", "offline_standard_proxy"),
            target_top1=_metric_from_list_arms(lora, "tca_map_lora", "target_top1_accuracy"),
            trainable_params=_params_from_list_arms(lora, "tca_map_lora"),
        ),
        _row(
            arm="TCA-Map + LoRA + Distributional TCA-Select",
            evidence_type="real-LIBERO tiny offline LoRA proxy",
            action_l1=_metric_from_list_arms(lora, "tca_map_lora_distributional_select", "action_l1"),
            wrong_target_proxy_rate=_metric_from_list_arms(
                lora, "tca_map_lora_distributional_select", "wrong_target_proxy_rate"
            ),
            counterfactual_margin=_metric_from_list_arms(
                lora, "tca_map_lora_distributional_select", "counterfactual_separation_margin"
            ),
            offline_proxy_score=_metric_from_list_arms(
                lora, "tca_map_lora_distributional_select", "offline_standard_proxy"
            ),
            target_top1=_metric_from_list_arms(lora, "tca_map_lora_distributional_select", "target_top1_accuracy"),
            trainable_params=_params_from_list_arms(lora, "tca_map_lora_distributional_select"),
        ),
    ]
    if scaleup and scaleup.get("bounded_lora_offline_scaleup_passed"):
        rows.extend(
            [
                _row(
                    arm="ActionMap + LoRA",
                    evidence_type="real-LIBERO bounded offline LoRA proxy",
                    action_l1=_metric_from_list_arms(scaleup, "actionmap_lora", "action_l1"),
                    wrong_target_proxy_rate=_metric_from_list_arms(
                        scaleup, "actionmap_lora", "wrong_target_proxy_rate"
                    ),
                    counterfactual_margin=_metric_from_list_arms(
                        scaleup, "actionmap_lora", "counterfactual_separation_margin"
                    ),
                    offline_proxy_score=_metric_from_list_arms(scaleup, "actionmap_lora", "offline_standard_proxy"),
                    target_top1=_metric_from_list_arms(scaleup, "actionmap_lora", "target_top1_accuracy"),
                    trainable_params=_params_from_list_arms(scaleup, "actionmap_lora"),
                ),
                _row(
                    arm="TCA-Map + LoRA",
                    evidence_type="real-LIBERO bounded offline LoRA proxy",
                    action_l1=_metric_from_list_arms(scaleup, "tca_map_lora", "action_l1"),
                    wrong_target_proxy_rate=_metric_from_list_arms(scaleup, "tca_map_lora", "wrong_target_proxy_rate"),
                    counterfactual_margin=_metric_from_list_arms(
                        scaleup, "tca_map_lora", "counterfactual_separation_margin"
                    ),
                    offline_proxy_score=_metric_from_list_arms(scaleup, "tca_map_lora", "offline_standard_proxy"),
                    target_top1=_metric_from_list_arms(scaleup, "tca_map_lora", "target_top1_accuracy"),
                    trainable_params=_params_from_list_arms(scaleup, "tca_map_lora"),
                ),
                _row(
                    arm="TCA-Map + LoRA + Distributional TCA-Select",
                    evidence_type="real-LIBERO bounded offline LoRA proxy",
                    action_l1=_metric_from_list_arms(scaleup, "tca_map_lora_distributional_select", "action_l1"),
                    wrong_target_proxy_rate=_metric_from_list_arms(
                        scaleup, "tca_map_lora_distributional_select", "wrong_target_proxy_rate"
                    ),
                    counterfactual_margin=_metric_from_list_arms(
                        scaleup, "tca_map_lora_distributional_select", "counterfactual_separation_margin"
                    ),
                    offline_proxy_score=_metric_from_list_arms(
                        scaleup, "tca_map_lora_distributional_select", "offline_standard_proxy"
                    ),
                    target_top1=_metric_from_list_arms(
                        scaleup, "tca_map_lora_distributional_select", "target_top1_accuracy"
                    ),
                    trainable_params=_params_from_list_arms(scaleup, "tca_map_lora_distributional_select"),
                ),
            ]
        )
        report["bounded_lora_scaleup_included"] = True
        report["bounded_lora_scaleup_record_count"] = int(scaleup.get("record_count") or 0)

    if stress and stress.get("tca_select_ambiguity_stress_passed"):
        rows.append(
            _row(
                arm="Distributional TCA-Select ambiguity stress",
                evidence_type="real-LIBERO offline ambiguity stress proxy",
                action_l1=_metric(stress, "selected_action_l1"),
                wrong_target_proxy_rate=_metric(stress, "selected_wrong_target_proxy_rate"),
                counterfactual_margin=_metric(stress, "condition_sensitivity_margin"),
                offline_proxy_score=None,
                target_top1=None,
                trainable_params=0,
            )
        )
        report["tca_select_ambiguity_stress_included"] = True
        report["tca_select_ambiguity_stress_record_count"] = int(stress.get("record_count") or 0)

    report["evidence_table"] = rows
    report["deltas"] = {
        "head_tca_vs_actionmap": (head.get("comparison") or {}).get("tca_map_vs_actionmap"),
        "head_tca_select_vs_tca": (head.get("comparison") or {}).get("tca_select_vs_tca_map"),
        "lora_tca_vs_actionmap_lora": (lora.get("comparison") or {}).get("tca_lora_vs_actionmap_lora"),
        "lora_tca_select_vs_tca_lora": (lora.get("comparison") or {}).get("tca_select_lora_vs_tca_lora"),
        "bounded_lora_tca_vs_actionmap_lora": (scaleup.get("comparison") or {}).get(
            "tca_lora_vs_actionmap_lora"
        )
        if scaleup
        else None,
        "bounded_lora_tca_select_vs_tca_lora": (scaleup.get("comparison") or {}).get(
            "tca_select_lora_vs_tca_lora"
        )
        if scaleup
        else None,
        "tca_select_ambiguity_stress_vs_top_heatmap": {
            "wrong_target_proxy_rate_delta": _metric(stress, "selection_wrong_target_proxy_delta_vs_top_heatmap"),
            "action_l1_delta": _metric(stress, "selection_action_l1_delta_vs_top_heatmap"),
            "top_heatmap_wrong_target_proxy_rate": _metric(stress, "top_heatmap_wrong_target_proxy_rate"),
            "selected_wrong_target_proxy_rate": _metric(stress, "selected_wrong_target_proxy_rate"),
            "top_heatmap_action_l1": _metric(stress, "top_heatmap_action_l1"),
            "selected_action_l1": _metric(stress, "selected_action_l1"),
        }
        if report["tca_select_ambiguity_stress_included"]
        else None,
    }
    report["gap_table"] = [
        {
            "id": "standard_success",
            "status": "blocked",
            "next_step": "Requires simulator benchmark rollout with a valid LIBERO-action-aligned policy path.",
        },
        {
            "id": "learned_policy_rollout",
            "status": "blocked_for_current_checkpoint",
            "next_step": provenance.get("recommended_next_step"),
        },
        {
            "id": "offline_proxy_scale",
            "status": "ready_for_extension_plan",
            "next_step": "Plan a larger offline proxy subset only after keeping labels clear that it is not standard success.",
        },
        {
            "id": "required_lora_track",
            "status": "bounded_proxy_present" if report["bounded_lora_scaleup_included"] else "tiny_proxy_present",
            "next_step": (
                "Use bounded LoRA scale-up as offline proxy only; next evidence step should improve attribution or resolve a LIBERO-aligned learned-policy path."
                if report["bounded_lora_scaleup_included"]
                else "Plan bounded LoRA scale-up under the local compute budget; no full fine-tuning."
            ),
        },
        {
            "id": "tca_select_inference_attribution",
            "status": "offline_ambiguity_stress_proxy_present"
            if report["tca_select_ambiguity_stress_included"]
            else "blocked",
            "next_step": (
                "Use ambiguity-stress evidence as offline proxy only; next attribution step must preserve no-paper-claim labels."
                if report["tca_select_ambiguity_stress_included"]
                else "Run the offline TCA-Select ambiguity stress test before selection-specific attribution claims."
            ),
        },
        {
            "id": "paper_claim",
            "status": "blocked",
            "next_step": "Needs rollout evidence, baselines, compute table, and no privileged inference.",
        },
    ]
    report["offline_evidence_gap_report_passed"] = True
    report["decision"] = "offline_evidence_table_ready"
    report["ready_for_lora_scaleup_plan"] = bool(pivot.get("ready_for_lora_scaleup_plan"))
    report["ready_for_offline_proxy_extension"] = True
    report["ready_for_libero_aligned_checkpoint_source_plan"] = bool(
        pivot.get("ready_for_libero_aligned_checkpoint_source_plan")
    )
    report["ready_for_learned_policy_rollout_scaling"] = False
    report["recommended_next_step"] = (
        "Regenerate the stress-aware attribution synthesis; keep current-checkpoint learned-policy rollout and paper claims blocked."
        if report["tca_select_ambiguity_stress_included"]
        else
        "Generate a scale-up-aware offline evidence synthesis or attribution-gap report; keep current-checkpoint learned-policy rollout and paper claims blocked."
        if report["bounded_lora_scaleup_included"]
        else "Plan a bounded LoRA/offline-proxy scale-up on real LIBERO HDF5 subsets, while keeping current-checkpoint learned-policy rollout and paper claims blocked."
        if bool((bounded or {}).get("libero_offline_bounded_pilot_report_passed"))
        else "Regenerate the bounded pilot report before scaling any offline proxy."
    )
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pivot-report", default="reports/offline_tca_map_lora_pivot_plan_report.json")
    parser.add_argument("--head-report", default="reports/libero_offline_actionmap_tca_comparison_report.json")
    parser.add_argument("--lora-report", default="reports/libero_offline_lora_comparison_report.json")
    parser.add_argument("--bounded-lora-scaleup-report", default="reports/bounded_lora_offline_scaleup_report.json")
    parser.add_argument("--tca-select-stress-report", default="reports/tca_select_ambiguity_stress_report.json")
    parser.add_argument("--provenance-report", default="reports/checkpoint_task_provenance_resolution_report.json")
    parser.add_argument("--bounded-pilot-report", default="reports/libero_offline_bounded_pilot_report.json")
    parser.add_argument("--report-path", default="reports/offline_tca_lora_evidence_gap_report_runtime.json")
    parser.add_argument("--markdown-report-path", default="reports/offline_tca_lora_evidence_gap_report_runtime.md")
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
