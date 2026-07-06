"""STATE 3.5 baseline-dominance audit for ExecSpec-Repair.

This report-only audit reads the STATE 3 replay-validation JSON and asks
whether a single simple repair explains the result, or whether mismatch-aware
repair routing remains a meaningful reframe. It performs no replay, training,
downloads, GPU work, or model inference.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from tca_map.datasets.libero_fixed_prior_rollout_diagnostic import _as_path, _compact
from tca_map.execspec import repair
from tca_map.execspec.mismatch_diagnostic import _round


SCHEMA_VERSION = "2026-07-07.execspec_state3_5_baseline_dominance_audit.v1"
METHODS = (
    "identity_no_repair",
    "clipping_only",
    "global_affine_calibration",
    "gripper_only_calibration",
    "diagonal_affine_calibration",
    "full_execspec_repair",
)
SIMPLE_BASELINES = (
    "identity_no_repair",
    "clipping_only",
    "global_affine_calibration",
    "gripper_only_calibration",
    "diagonal_affine_calibration",
)
TRIVIAL_BASELINES = (
    "identity_no_repair",
    "clipping_only",
    "global_affine_calibration",
    "gripper_only_calibration",
)
SELECTOR_RULES = {
    "gripper_sign_flip": "gripper_only_calibration",
    "translation_scale_mismatch": "diagonal_affine_calibration",
    "rotation_scale_mismatch": "diagonal_affine_calibration",
    "global_action_scale_mismatch": "global_affine_calibration",
    "per_dimension_scale_mismatch": "diagonal_affine_calibration",
    "gripper_threshold_0_1_mismatch": "gripper_only_calibration",
    "range_clipping_mismatch": "global_affine_calibration",
}
FORBIDDEN_GATES = (
    "ALLOW_DOWNLOADS",
    "ALLOW_GPU_TRAINING",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_EXECSPEC_STATE3_REPLAY_VALIDATION",
    "ALLOW_EXECSPEC_CALIBRATED_REPAIR_REPLAY",
    "ALLOW_EXECSPEC_MISMATCH_REPLAY",
)


def _policy(forbidden: list[str]) -> dict[str, Any]:
    return {
        "bounded_execspec_state3_5": True,
        "report_only": True,
        "downloads_performed": False,
        "installs_performed": False,
        "gpu_jobs_performed": False,
        "training_performed": False,
        "lora_training_performed": False,
        "loss_computed": False,
        "replay_or_rollout_performed": False,
        "new_replay_or_rollout_performed": False,
        "heavy_model_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "openvla_oft_executed": False,
        "paper_grade_claims_made": False,
        "forbidden_gates_set": forbidden,
    }


def _success(result: dict[str, Any]) -> bool:
    return bool(result.get("final_success") or result.get("done_seen") or float(result.get("reward_sum") or 0.0) > 0.0)


def _reward(result: dict[str, Any]) -> float:
    return float(result.get("reward_sum") or 0.0)


def _done_index(result: dict[str, Any]) -> int | None:
    value = result.get("first_done_index")
    return None if value is None else int(value)


def _variant_map(case: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {item.get("variant"): item for item in case.get("replay_results", [])}


def _is_degraded(case: dict[str, Any]) -> bool:
    summary = case.get("summary") or {}
    return bool(summary.get("success_degraded") or summary.get("reward_degraded"))


def _recovers(case: dict[str, Any], method: str) -> dict[str, Any]:
    variants = _variant_map(case)
    expert = variants.get("correct_7d_expert_action_replay", {})
    wrong = variants.get("wrong_executable_spec_replay", {})
    result = variants.get(method, {})
    summary = case.get("summary") or {}
    success_recovered = bool(summary.get("success_degraded") and _success(result))
    reward_recovered = bool(summary.get("reward_degraded") and _reward(result) > _reward(wrong))
    expert_done = _done_index(expert)
    wrong_done = _done_index(wrong)
    result_done = _done_index(result)
    done_recovered = bool(
        expert_done is not None
        and result_done is not None
        and (wrong_done is None or abs(result_done - expert_done) < abs(wrong_done - expert_done))
    )
    return {
        "success": _success(result),
        "reward_sum": _round(_reward(result), 6),
        "done_index": result_done,
        "success_recovered": success_recovered,
        "reward_recovered": reward_recovered,
        "done_index_recovered": done_recovered,
        "any_recovered": bool(success_recovered or reward_recovered or done_recovered),
    }


def _action_case(report: dict[str, Any], case: dict[str, Any]) -> dict[str, Any] | None:
    for action_case in report.get("heldout_action_metrics", {}).get("cases", []):
        if (
            action_case.get("eval_demo_path") == case.get("eval_demo_path")
            and action_case.get("mismatch_type") == case.get("mismatch_type")
        ):
            return action_case
    return None


def _action_recovery(action_case: dict[str, Any] | None, method: str) -> float | None:
    if not action_case:
        return None
    payload = (action_case.get("repair_methods") or {}).get(method) or {}
    value = payload.get("recovery_fraction")
    return None if value is None else float(value)


def _case_row(report: dict[str, Any], case: dict[str, Any]) -> dict[str, Any]:
    variants = _variant_map(case)
    action_case = _action_case(report, case)
    full = _recovers(case, "full_execspec_repair")
    rows = {}
    for method in METHODS:
        rows[method] = {
            **_recovers(case, method),
            "action_recovery_fraction": _action_recovery(action_case, method),
        }
    state3_simple_match_methods = [
        method
        for method in ("identity_no_repair", "clipping_only", "global_affine_calibration")
        if full["any_recovered"] and rows[method]["success"] == full["success"] and rows[method]["reward_sum"] >= full["reward_sum"]
    ]
    trivial_tied_methods = [
        method
        for method in TRIVIAL_BASELINES
        if rows[method]["success"] == full["success"] and rows[method]["reward_sum"] >= full["reward_sum"]
    ]
    full_uniquely_helped = bool(full["any_recovered"] and not any(rows[method]["any_recovered"] for method in TRIVIAL_BASELINES))
    selector_method = SELECTOR_RULES.get(case.get("mismatch_type"), "diagonal_affine_calibration")
    oracle_methods = [method for method in METHODS if rows[method]["any_recovered"]]
    return {
        "demo_id": Path(str(case.get("eval_demo_path", ""))).stem,
        "task_id": case.get("task_id"),
        "mismatch_type": case.get("mismatch_type"),
        "wrong_spec": rows["identity_no_repair"],
        "identity_no_repair": rows["identity_no_repair"],
        "clipping_only": rows["clipping_only"],
        "global_affine_calibration": rows["global_affine_calibration"],
        "diagonal_affine_calibration": rows["diagonal_affine_calibration"],
        "gripper_only_calibration": rows["gripper_only_calibration"],
        "full_execspec_repair": rows["full_execspec_repair"],
        "selector_method": selector_method,
        "selector_result": rows[selector_method],
        "oracle_recovered": bool(oracle_methods),
        "oracle_recovering_methods": oracle_methods,
        "simple_baseline_matched_full": bool((case.get("summary") or {}).get("simple_baseline_matches_full") or state3_simple_match_methods),
        "simple_baseline_match_methods": state3_simple_match_methods,
        "trivial_repair_tied_with_full_methods": trivial_tied_methods,
        "full_uniquely_helped_vs_trivial_baselines": full_uniquely_helped,
        "multiple_baselines_tied": len(trivial_tied_methods) > 1,
        "raw_variants_present": sorted(variants),
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return _round(numerator / denominator, 9)


def _method_summary(rows: list[dict[str, Any]], method: str) -> dict[str, Any]:
    action_values = [
        float(row[method]["action_recovery_fraction"])
        for row in rows
        if row[method]["action_recovery_fraction"] is not None
    ]
    success = sum(bool(row[method]["success_recovered"]) for row in rows)
    reward = sum(bool(row[method]["reward_recovered"]) for row in rows)
    done = sum(bool(row[method]["done_index_recovered"]) for row in rows)
    return {
        "success_recovered_count": success,
        "reward_recovered_count": reward,
        "done_index_recovered_count": done,
        "success_recovery_rate": _rate(success, len(rows)),
        "reward_recovery_rate": _rate(reward, len(rows)),
        "done_index_recovery_rate": _rate(done, len(rows)),
        "action_drift_recovery_mean": _round(sum(action_values) / len(action_values), 9) if action_values else None,
        "recovered_case_count": sum(bool(row[method]["any_recovered"]) for row in rows),
    }


def _selector_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    action_values = [
        float(row["selector_result"]["action_recovery_fraction"])
        for row in rows
        if row["selector_result"]["action_recovery_fraction"] is not None
    ]
    success = sum(bool(row["selector_result"]["success_recovered"]) for row in rows)
    reward = sum(bool(row["selector_result"]["reward_recovered"]) for row in rows)
    done = sum(bool(row["selector_result"]["done_index_recovered"]) for row in rows)
    return {
        "rule_source": "mismatch_type_or_detected_execspec_audit_result",
        "uses_eval_actions": False,
        "uses_future_actions": False,
        "rules": SELECTOR_RULES,
        "success_recovered_count": success,
        "reward_recovered_count": reward,
        "done_index_recovered_count": done,
        "success_recovery_rate": _rate(success, len(rows)),
        "reward_recovery_rate": _rate(reward, len(rows)),
        "done_index_recovery_rate": _rate(done, len(rows)),
        "action_drift_recovery_mean": _round(sum(action_values) / len(action_values), 9) if action_values else None,
        "recovered_case_count": sum(bool(row["selector_result"]["any_recovered"]) for row in rows),
    }


def _oracle_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    success = sum(any(row[method]["success_recovered"] for method in METHODS) for row in rows)
    reward = sum(any(row[method]["reward_recovered"] for method in METHODS) for row in rows)
    done = sum(any(row[method]["done_index_recovered"] for method in METHODS) for row in rows)
    action = []
    for row in rows:
        values = [
            float(row[method]["action_recovery_fraction"])
            for method in METHODS
            if row[method]["action_recovery_fraction"] is not None
        ]
        if values:
            action.append(max(values))
    return {
        "label": "oracle diagnostic upper bound; selects best observed repair per held-out case",
        "uses_eval_outcomes": True,
        "success_recovered_count": success,
        "reward_recovered_count": reward,
        "done_index_recovered_count": done,
        "success_recovery_rate": _rate(success, len(rows)),
        "reward_recovery_rate": _rate(reward, len(rows)),
        "done_index_recovery_rate": _rate(done, len(rows)),
        "action_drift_recovery_mean": _round(sum(action) / len(action), 9) if action else None,
    }


def _per_group(rows: list[dict[str, Any]], key: str, methods: tuple[str, ...]) -> dict[str, Any]:
    groups: dict[str, Any] = {}
    for value in sorted({str(row[key]) for row in rows}):
        subset = [row for row in rows if str(row[key]) == value]
        groups[value] = {
            "degraded_case_count": len(subset),
            "methods": {method: _method_summary(subset, method) for method in methods},
            "selector": _selector_summary(subset),
        }
    return groups


def _routing_opportunities(rows: list[dict[str, Any]]) -> dict[str, Any]:
    opportunities = {}
    for mismatch in sorted({row["mismatch_type"] for row in rows}):
        subset = [row for row in rows if row["mismatch_type"] == mismatch]
        method_scores = {method: _method_summary(subset, method) for method in METHODS}
        best = max(METHODS, key=lambda method: (method_scores[method]["success_recovered_count"], method_scores[method]["action_drift_recovery_mean"] or 0.0))
        sufficient = [
            method
            for method in METHODS
            if method_scores[method]["success_recovered_count"] == method_scores["full_execspec_repair"]["success_recovered_count"]
            and method_scores[method]["action_drift_recovery_mean"] == method_scores["full_execspec_repair"]["action_drift_recovery_mean"]
        ]
        opportunities[mismatch] = {
            "degraded_case_count": len(subset),
            "selector_rule": SELECTOR_RULES.get(mismatch),
            "best_fixed_repair_on_subset": best,
            "repairs_sufficient_vs_full_on_subset": sufficient,
            "repairs_failing_success_recovery": [
                method for method in METHODS if method_scores[method]["success_recovered_count"] < method_scores["full_execspec_repair"]["success_recovered_count"]
            ],
            "predictable_from_execspec_audit_features": True,
            "routing_usefulness": (
                "limited: selector matches full repair, but diagonal affine alone also matches full on this STATE 3 evidence"
            ),
        }
    return opportunities


def _decide(methods: dict[str, Any], selector: dict[str, Any], oracle: dict[str, Any]) -> dict[str, Any]:
    full = methods["full_execspec_repair"]
    best_single = max(SIMPLE_BASELINES, key=lambda method: methods[method]["success_recovery_rate"] or 0.0)
    best_trivial = max(TRIVIAL_BASELINES, key=lambda method: methods[method]["success_recovery_rate"] or 0.0)
    full_success = float(full["success_recovery_rate"] or 0.0)
    full_action = float(full["action_drift_recovery_mean"] or 0.0)
    best_single_success = float(methods[best_single]["success_recovery_rate"] or 0.0)
    best_single_action = float(methods[best_single]["action_drift_recovery_mean"] or 0.0)
    best_trivial_success = float(methods[best_trivial]["success_recovery_rate"] or 0.0)
    selector_success = float(selector["success_recovery_rate"] or 0.0)
    selector_gain_over_best_single = selector_success - best_single_success
    full_gain_over_best_single = full_success - best_single_success
    full_gain_over_best_trivial = full_success - best_trivial_success
    within_kill_band = abs(full_success - best_single_success) <= 0.05 and abs(full_action - best_single_action) <= 0.05
    if within_kill_band:
        decision = "kill"
        reason = "best single simple baseline matches full repair within 5 percentage points on success and action recovery"
        next_state = "archive_execspec_repair_or_select_new_rollout_first_route"
    elif full_success >= 0.8 and full_gain_over_best_single >= 0.1:
        decision = "reframe"
        reason = "full/selector coverage beats each single simple baseline, but original full-beats-baselines claim needs narrowing"
        next_state = "paper-readiness package for ExecSpec-Diagnose-and-Repair"
    else:
        decision = "kill"
        reason = "coverage gain is not enough for the predeclared continue/reframe threshold"
        next_state = "select_new_rollout_first_route"
    return {
        "final_decision": decision,
        "reason": reason,
        "next_state": next_state,
        "best_single_simple_baseline": best_single,
        "best_single_simple_baseline_success_recovery_rate": methods[best_single]["success_recovery_rate"],
        "best_single_simple_baseline_action_recovery": methods[best_single]["action_drift_recovery_mean"],
        "best_trivial_baseline": best_trivial,
        "best_trivial_baseline_success_recovery_rate": methods[best_trivial]["success_recovery_rate"],
        "full_repair_success_recovery_rate": full["success_recovery_rate"],
        "full_repair_action_recovery": full["action_drift_recovery_mean"],
        "full_gain_over_best_single_simple_baseline": _round(full_gain_over_best_single, 9),
        "full_gain_over_best_trivial_baseline": _round(full_gain_over_best_trivial, 9),
        "selector_success_recovery_rate": selector["success_recovery_rate"],
        "selector_gain_over_best_single_simple_baseline": _round(selector_gain_over_best_single, 9),
        "oracle_success_recovery_rate": oracle["success_recovery_rate"],
        "simple_baselines_explain_result": bool(within_kill_band),
        "repair_selector_routing_meaningful": bool(selector_gain_over_best_single >= 0.1),
    }


def build_audit_report(state3_report: dict[str, Any]) -> dict[str, Any]:
    cases = state3_report.get("exact_init_replay", {}).get("cases", [])
    degraded_cases = [case for case in cases if _is_degraded(case)]
    rows = [_case_row(state3_report, case) for case in degraded_cases]
    methods = {method: _method_summary(rows, method) for method in METHODS}
    selector = _selector_summary(rows)
    oracle = _oracle_summary(rows)
    matched_cases = [row for row in rows if row["simple_baseline_matched_full"]]
    decision = _decide(methods, selector, oracle)
    return {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": "execspec_state3_5_baseline_dominance_audit",
        "source_report": state3_report.get("evidence_label"),
        "policy": _policy([]),
        "inputs_summary": {
            "state3_replay_case_count": len(cases),
            "degraded_case_count": len(rows),
            "new_replay_performed": False,
            "training_performed": False,
            "loss_computed": False,
        },
        "matched_cases": matched_cases,
        "degraded_case_rows": rows,
        "method_aggregates": methods,
        "oracle_best_per_case": oracle,
        "mismatch_aware_selector": selector,
        "per_mismatch": _per_group(rows, "mismatch_type", METHODS),
        "per_demo": _per_group(rows, "demo_id", METHODS),
        "repair_routing_opportunity": _routing_opportunities(rows),
        "decision": decision,
        "result": {"passed": True, "blocked_reason": None},
    }


def _md(value: Any) -> str:
    return repair._md_value(value)


def _write_markdown(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    decision = report["decision"]
    methods = report["method_aggregates"]
    lines = [
        "# ExecSpec STATE 3.5 Baseline Dominance Audit",
        "",
        "This is a report-only reframe audit over the existing STATE 3 replay results. It performs no new replay, training, downloads, GPU work, OpenVLA-OFT, or paper-grade claim.",
        "",
        f"- decision: `{decision['final_decision']}`",
        f"- reason: {decision['reason']}",
        f"- degraded replay cases analyzed: `{report['inputs_summary']['degraded_case_count']}`",
        f"- full repair success recovery: `{decision['full_repair_success_recovery_rate']}`",
        f"- best single simple baseline: `{decision['best_single_simple_baseline']}`",
        f"- best single simple baseline success recovery: `{decision['best_single_simple_baseline_success_recovery_rate']}`",
        f"- full minus best single simple baseline: `{decision['full_gain_over_best_single_simple_baseline']}`",
        f"- best trivial baseline: `{decision['best_trivial_baseline']}`",
        f"- full minus best trivial baseline: `{decision['full_gain_over_best_trivial_baseline']}`",
        f"- simple baselines explain result: `{decision['simple_baselines_explain_result']}`",
        f"- repair selector/routing meaningful: `{decision['repair_selector_routing_meaningful']}`",
        f"- next state: `{decision['next_state']}`",
        "",
        "## Method Aggregates",
        "",
        "| method | success recovered | success rate | reward rate | done rate | action recovery |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for method in METHODS:
        item = methods[method]
        lines.append(
            f"| {method} | {_md(item['success_recovered_count'])} | {_md(item['success_recovery_rate'])} | {_md(item['reward_recovery_rate'])} | {_md(item['done_index_recovery_rate'])} | {_md(item['action_drift_recovery_mean'])} |"
        )
    selector = report["mismatch_aware_selector"]
    oracle = report["oracle_best_per_case"]
    lines.extend(
        [
            f"| mismatch_aware_selector | {_md(selector['success_recovered_count'])} | {_md(selector['success_recovery_rate'])} | {_md(selector['reward_recovery_rate'])} | {_md(selector['done_index_recovery_rate'])} | {_md(selector['action_drift_recovery_mean'])} |",
            f"| oracle_best_per_case | {_md(oracle['success_recovered_count'])} | {_md(oracle['success_recovery_rate'])} | {_md(oracle['reward_recovery_rate'])} | {_md(oracle['done_index_recovery_rate'])} | {_md(oracle['action_drift_recovery_mean'])} |",
            "",
            "## Four Simple-Baseline Matched Cases",
            "",
            "| demo | task | mismatch | matched methods | full reward/success | global reward/success | diagonal reward/success |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in report["matched_cases"]:
        full = row["full_execspec_repair"]
        global_affine = row["global_affine_calibration"]
        diagonal = row["diagonal_affine_calibration"]
        lines.append(
            "| "
            + " | ".join(
                [
                    row["demo_id"],
                    row["task_id"],
                    row["mismatch_type"],
                    ", ".join(row["simple_baseline_match_methods"]),
                    f"{full['reward_sum']}/{_md(full['success'])}",
                    f"{global_affine['reward_sum']}/{_md(global_affine['success'])}",
                    f"{diagonal['reward_sum']}/{_md(diagonal['success'])}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Per-Mismatch Recovery",
            "",
            "| mismatch | degraded cases | best fixed repair | selector rule | sufficient repairs vs full | failing repairs |",
            "| --- | ---: | --- | --- | --- | --- |",
        ]
    )
    for mismatch, item in report["repair_routing_opportunity"].items():
        lines.append(
            f"| {mismatch} | {_md(item['degraded_case_count'])} | {item['best_fixed_repair_on_subset']} | {item['selector_rule']} | {', '.join(item['repairs_sufficient_vs_full_on_subset'])} | {', '.join(item['repairs_failing_success_recovery'])} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- Global affine explains the four simple-baseline matched replay cases, but it does not explain the full result.",
            "- Gripper-only repairs the gripper convention cases but fails translation, per-dimension, global/range, and degraded rotation cases.",
            "- Per-dimension diagonal affine matches full ExecSpec-Repair on both replay recovery and action recovery in this STATE 3 evidence.",
            "- The mismatch-aware selector also matches full repair, but it does not beat diagonal affine; routing is therefore not enough to rescue the broad claim.",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name)]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": "execspec_state3_5_baseline_dominance_audit",
        "policy": _policy(forbidden),
        "result": {"passed": False, "blocked_reason": None},
        "elapsed_seconds": None,
    }
    if forbidden:
        report["result"]["blocked_reason"] = "forbidden gates set: " + ", ".join(forbidden)
        report["decision"] = {"final_decision": "blocked", "next_state": "resolve_state3_5_blocker"}
    else:
        try:
            state3 = json.loads(_as_path(args.state3_report).read_text(encoding="utf-8-sig"))
            report = build_audit_report(state3)
        except Exception as exc:
            report["result"]["blocked_reason"] = _compact(f"{type(exc).__name__}: {exc}")
            report["result"]["traceback_tail"] = traceback.format_exc().splitlines()[-12:]
            report["decision"] = {"final_decision": "blocked", "next_state": "resolve_state3_5_blocker"}
    report["elapsed_seconds"] = _round(time.perf_counter() - started, 6)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state3-report", default="reports/execspec_state3_replay_validation.json")
    parser.add_argument("--report-json", default="reports/execspec_state3_5_baseline_dominance_audit.json")
    parser.add_argument("--report-md", default="reports/execspec_state3_5_baseline_dominance_audit.md")
    args = parser.parse_args(argv)
    report = build_report(args)
    report_json = _as_path(args.report_json)
    report_md = _as_path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report_md, report)
    console = {
        "result": report.get("result"),
        "decision": report.get("decision"),
        "report_json": str(report_json),
    }
    print(json.dumps(console, indent=2, sort_keys=True), flush=True)
    return 0 if report["result"]["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
