"""Bounded CSS-Shield Phase 2 state machine.

Phase 2 strengthens the first CSS-Shield diagnostic package without overwriting
the original STATE 1-5 evidence. It remains diagnostic-only: no training,
downloads, GPU jobs, OpenVLA-OFT, external submission, or paper-grade claim.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from statistics import mean, pstdev
from typing import Any


STATE_SCHEMA = "2026-07-07.css_shield_phase2_state.v1"


def _load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: str | Path, data: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: str | Path, lines: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _num(value: Any, default: float = 0.0) -> float:
    if value is None:
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _variant(report: dict[str, Any], name: str) -> dict[str, Any]:
    for item in report.get("variants") or []:
        if item.get("shield_variant") == name:
            return item
    return {}


def _phase_state(
    *,
    main_commit: str,
    current_state: str,
    last_completed: str,
    decision: str,
    next_milestone: str,
    reason: str,
    metrics: dict[str, Any],
    policy: dict[str, Any] | None = None,
    hard_blockers: list[str] | None = None,
) -> dict[str, Any]:
    policy = policy or {}
    return {
        "schema_version": STATE_SCHEMA,
        "current_main_commit": main_commit,
        "current_state": current_state,
        "last_completed_state": last_completed,
        "continue_kill_decision": decision,
        "next_milestone": next_milestone,
        "why_next": reason,
        "hard_blockers": hard_blockers or [],
        "diagnostic_only": True,
        "paper_grade_claim": False,
        "rollout_happened": bool(policy.get("rollout_happened")),
        "native_smolvla_inference_happened": bool(policy.get("model_inference_performed")),
        "training_happened": bool(policy.get("training_performed")),
        "loss_computed": bool(policy.get("loss_computed")),
        "gpu_happened": bool(policy.get("gpu_jobs_performed")),
        "download_happened": bool(policy.get("downloads_performed")),
        "openvla_oft_happened": bool(policy.get("openvla_oft_executed")),
        "last_result_summary": metrics,
        "exact_resume_command": "powershell -ExecutionPolicy Bypass -File scripts\\162_css_shield_phase2_autopilot.ps1 -Continuous",
    }


def audit_diagnostic_package() -> dict[str, Any]:
    ral = _load_json("reports/css_shield_ral_strength_check.json")
    scale = _load_json("reports/css_shield_state4_scale_summary.json")
    state = _load_json("reports/css_shield_autopilot_state.json")
    full_vs_safety = _num(scale.get("full_vs_safety_wrong_target_delta"))
    full_vs_clipping = _num(scale.get("full_vs_clipping_wrong_target_delta"))
    unsafe_vs_clipping = _num(scale.get("full_vs_clipping_unsafe_delta"))
    intervention = _num(scale.get("full_intervention_rate"), 1.0)
    false_positive = _num(scale.get("full_false_positive_rate"), 1.0)
    continue_signal = bool(
        ral.get("continue")
        and scale.get("continue")
        and full_vs_safety > 0.0
        and full_vs_clipping > 0.0
        and intervention < 0.95
        and false_positive <= 0.25
    )
    audit = {
        "schema_version": "2026-07-07.css_shield_phase2_package_audit.v1",
        "decision": "continue" if continue_signal else "kill_or_reframe",
        "continue": continue_signal,
        "real_simulator_rollout_evidence": bool(state.get("rollout_happened")),
        "controlled_proposal_diagnostic_evidence": True,
        "native_smolvla_action_evidence": bool(state.get("native_smolvla_inference_happened")),
        "synthetic_or_oracle_state_diagnostic_evidence": True,
        "full_shield_beats_safety_only": full_vs_safety > 0.0,
        "full_shield_beats_clipping_only": full_vs_clipping > 0.0 or unsafe_vs_clipping > 0.0,
        "utility_degradation_acceptable": intervention < 0.95,
        "false_positive_rate_low": false_positive <= 0.25,
        "full_vs_safety_wrong_target_delta": full_vs_safety,
        "full_vs_clipping_wrong_target_delta": full_vs_clipping,
        "full_vs_clipping_unsafe_delta": unsafe_vs_clipping,
        "full_intervention_rate": intervention,
        "full_false_positive_rate": false_positive,
        "missing_for_ral": [
            "native-action diagnostic with enough steps",
            "multi-task or randomized diagnostic beyond a single scene",
            "utility preservation under native proposals",
            "comparison showing semantic component is more than safety-only",
            "paper-grade rollout benchmark, which is not claimed here",
        ],
        "reason": "First diagnostic package shows nontrivial behavior beyond clipping/safety-only, so Phase 2 native-action testing is justified."
        if continue_signal
        else "First diagnostic package does not clear the Phase 2 novelty/utility gate.",
    }
    _write_json("reports/css_shield_phase2_package_audit.json", audit)
    _write_md(
        "reports/css_shield_phase2_package_audit.md",
        [
            "# CSS-Shield Phase 2 Package Audit",
            "",
            "Diagnostic-only audit of the first CSS-Shield package.",
            "",
            f"- decision: `{audit['decision']}`",
            f"- real simulator rollout evidence: `{audit['real_simulator_rollout_evidence']}`",
            f"- controlled proposal evidence: `{audit['controlled_proposal_diagnostic_evidence']}`",
            f"- native SmolVLA action evidence: `{audit['native_smolvla_action_evidence']}`",
            f"- synthetic/oracle-state diagnostic evidence: `{audit['synthetic_or_oracle_state_diagnostic_evidence']}`",
            f"- full beats safety-only: `{audit['full_shield_beats_safety_only']}`",
            f"- full beats clipping-only: `{audit['full_shield_beats_clipping_only']}`",
            f"- false positive rate low: `{audit['false_positive_rate_low']}`",
            f"- reason: {audit['reason']}",
            "",
            "## Missing For RA-L",
            "",
            *[f"- {item}" for item in audit["missing_for_ral"]],
        ],
    )
    return audit


def assess_native_action_report(path: str | Path = "reports/css_shield_phase2_native_action_diagnostic_report.json") -> dict[str, Any]:
    report = _load_json(path)
    policy = report.get("policy") or {}
    comparison = report.get("comparison") or {}
    full = _variant(report, "full_css_shield")
    no_shield = _variant(report, "no_shield")
    used_native = (report.get("proposal_source") or {}).get("used") == "native_smolvla"
    full_vs_safety_wrong = _num(comparison.get("full_vs_safety_only_wrong_target_rate_reduction"))
    full_vs_safety_unsafe = _num(comparison.get("full_vs_safety_only_unsafe_rate_reduction"))
    full_vs_clipping_wrong = _num(comparison.get("full_vs_clipping_wrong_target_rate_reduction"))
    full_vs_clipping_unsafe = _num(comparison.get("full_vs_clipping_unsafe_rate_reduction"))
    full_steps = len(full.get("step_records") or [])
    shield_variant_steps = sum(len(item.get("step_records") or []) for item in report.get("variants") or [])
    intervention = _num(full.get("intervention_rate"), 1.0)
    false_positive = _num(full.get("false_positive_intervention_rate"), 1.0)
    utility_drop = _num(comparison.get("utility_drop_vs_no_shield"))
    full_stop_all = intervention >= 0.95
    wrong_metric_computable = full.get("wrong_target_action_rate_after") is not None and no_shield.get("wrong_target_action_rate_after") is not None
    continue_signal = bool(
        report.get("result", {}).get("passed")
        and used_native
        and bool(policy.get("model_inference_performed"))
        and bool(policy.get("rollout_happened"))
        and full_steps >= 20
        and wrong_metric_computable
        and full_vs_safety_wrong > 0.0
        and (full_vs_clipping_wrong > 0.0 or full_vs_clipping_unsafe > 0.0)
        and not full_stop_all
        and false_positive <= 0.25
        and utility_drop <= 0.25
    )
    if not wrong_metric_computable:
        reason = "Native-action wrong-target metric is not computable for this task, so Phase 2 cannot claim semantic value."
    elif full_vs_safety_wrong <= 0.0:
        reason = "Full CSS-Shield does not beat safety-only on native-action wrong-target reduction."
    elif full_stop_all:
        reason = "Full CSS-Shield behaves like stop-all under native actions."
    elif utility_drop > 0.25:
        reason = "Native-action utility drop exceeds the bounded Phase 2 threshold."
    else:
        reason = "Native-action diagnostic shows full CSS-Shield adds wrong-target value beyond safety-only and clipping-only."
    result = {
        "schema_version": "2026-07-07.css_shield_phase2_native_action_assessment.v1",
        "decision": "continue" if continue_signal else "kill_or_reframe",
        "continue": continue_signal,
        "report_path": str(path),
        "proposal_source_used": (report.get("proposal_source") or {}).get("used"),
        "native_smolvla_inference_happened": bool(policy.get("model_inference_performed")),
        "rollout_happened": bool(policy.get("rollout_happened")),
        "native_policy_available": bool(((report.get("proposal_source") or {}).get("native") or {}).get("available")),
        "full_steps": full_steps,
        "shield_variant_steps": shield_variant_steps,
        "wrong_target_metric_computable": wrong_metric_computable,
        "full_vs_safety_wrong_target_delta": full_vs_safety_wrong,
        "full_vs_safety_unsafe_delta": full_vs_safety_unsafe,
        "full_vs_clipping_wrong_target_delta": full_vs_clipping_wrong,
        "full_vs_clipping_unsafe_delta": full_vs_clipping_unsafe,
        "full_intervention_rate": intervention,
        "full_false_positive_rate": false_positive,
        "full_false_negative_rate": full.get("false_negative_unsafe_or_wrong_rate"),
        "utility_drop_vs_no_shield": utility_drop,
        "reward_full": full.get("reward_sum"),
        "success_full": full.get("final_success"),
        "action_modification_l2_mean": full.get("action_modification_l2_mean"),
        "reason": reason,
    }
    _write_json("reports/css_shield_phase2_native_action_assessment.json", result)
    _write_md(
        "reports/css_shield_phase2_native_action_assessment.md",
        [
            "# CSS-Shield Phase 2 Native-Action Assessment",
            "",
            "Diagnostic-only. This is not a benchmark or paper-grade claim.",
            "",
            f"- decision: `{result['decision']}`",
            f"- native SmolVLA inference happened: `{result['native_smolvla_inference_happened']}`",
            f"- rollout happened: `{result['rollout_happened']}`",
            f"- full native steps: `{result['full_steps']}`",
            f"- shield-variant simulator steps: `{result['shield_variant_steps']}`",
            f"- wrong-target metric computable: `{result['wrong_target_metric_computable']}`",
            f"- full vs safety-only wrong-target delta: `{result['full_vs_safety_wrong_target_delta']}`",
            f"- full vs clipping-only wrong-target delta: `{result['full_vs_clipping_wrong_target_delta']}`",
            f"- full vs clipping-only unsafe delta: `{result['full_vs_clipping_unsafe_delta']}`",
            f"- utility drop vs no shield: `{result['utility_drop_vs_no_shield']}`",
            f"- false positive rate: `{result['full_false_positive_rate']}`",
            f"- reason: {result['reason']}",
        ],
    )
    return result


def assess_multitask_reports(pattern: str = "reports/css_shield_phase2_multitask_task*_report.json") -> dict[str, Any]:
    reports = [_load_json(path) for path in sorted(glob.glob(pattern))]
    reports = [report for report in reports if report]
    assessments: list[dict[str, Any]] = []
    for report in reports:
        temp_path = report.get("_path")
        comparison = report.get("comparison") or {}
        full = _variant(report, "full_css_shield")
        policy = report.get("policy") or {}
        assessments.append(
            {
                "task_id": (report.get("case") or {}).get("task_id"),
                "case_index": (report.get("case") or {}).get("case_index"),
                "used_native": (report.get("proposal_source") or {}).get("used") == "native_smolvla",
                "rollout_happened": bool(policy.get("rollout_happened")),
                "model_inference_performed": bool(policy.get("model_inference_performed")),
                "full_vs_safety_wrong_target_delta": _num(comparison.get("full_vs_safety_only_wrong_target_rate_reduction")),
                "full_vs_clipping_wrong_target_delta": _num(comparison.get("full_vs_clipping_wrong_target_rate_reduction")),
                "full_vs_clipping_unsafe_delta": _num(comparison.get("full_vs_clipping_unsafe_rate_reduction")),
                "utility_drop_vs_no_shield": _num(comparison.get("utility_drop_vs_no_shield")),
                "full_intervention_rate": _num(full.get("intervention_rate"), 1.0),
                "full_false_positive_rate": _num(full.get("false_positive_intervention_rate"), 1.0),
                "full_steps": len(full.get("step_records") or []),
                "report_path": temp_path,
            }
        )
    wins_safety = sum(1 for item in assessments if item["full_vs_safety_wrong_target_delta"] > 0.0)
    wins_clipping = sum(1 for item in assessments if item["full_vs_clipping_wrong_target_delta"] > 0.0 or item["full_vs_clipping_unsafe_delta"] > 0.0)
    utility_ok = sum(1 for item in assessments if item["utility_drop_vs_no_shield"] <= 0.25)
    count = len(assessments)
    deltas = [item["full_vs_safety_wrong_target_delta"] for item in assessments]
    continue_signal = bool(count >= 3 and wins_safety >= max(2, count // 2 + 1) and wins_clipping >= max(2, count // 2 + 1) and utility_ok == count)
    result = {
        "schema_version": "2026-07-07.css_shield_phase2_multitask_assessment.v1",
        "decision": "continue" if continue_signal else "kill_or_reframe",
        "continue": continue_signal,
        "task_count": count,
        "wins_vs_safety_only": wins_safety,
        "wins_vs_clipping_only": wins_clipping,
        "utility_ok_count": utility_ok,
        "full_vs_safety_wrong_target_delta_mean": mean(deltas) if deltas else None,
        "full_vs_safety_wrong_target_delta_std": pstdev(deltas) if len(deltas) > 1 else 0.0 if deltas else None,
        "per_task": assessments,
        "reason": "Full CSS-Shield wins on most Phase 2 tasks without severe utility collapse."
        if continue_signal
        else "Full CSS-Shield did not generalize across enough Phase 2 tasks or utility collapsed.",
    }
    _write_json("reports/css_shield_phase2_multitask_assessment.json", result)
    _write_md(
        "reports/css_shield_phase2_multitask_assessment.md",
        [
            "# CSS-Shield Phase 2 Multi-Task Assessment",
            "",
            f"- decision: `{result['decision']}`",
            f"- task count: `{result['task_count']}`",
            f"- wins vs safety-only: `{result['wins_vs_safety_only']}`",
            f"- wins vs clipping-only: `{result['wins_vs_clipping_only']}`",
            f"- mean full vs safety wrong-target delta: `{result['full_vs_safety_wrong_target_delta_mean']}`",
            f"- reason: {result['reason']}",
        ],
    )
    return result


def assess_novelty(native: dict[str, Any], multi: dict[str, Any] | None = None) -> dict[str, Any]:
    multi = multi or {}
    if multi:
        continue_signal = bool(native.get("continue") and multi.get("continue"))
        reason = "Native-action and multi-task diagnostics both support CSS-Shield novelty." if continue_signal else "Phase 2 baseline/novelty check fails after multi-task assessment."
    else:
        continue_signal = bool(native.get("continue"))
        reason = "Native-action diagnostic supports CSS-Shield novelty; multi-task check has not run." if continue_signal else native.get("reason", "Native-action diagnostic failed.")
    result = {
        "schema_version": "2026-07-07.css_shield_phase2_novelty_check.v1",
        "decision": "continue" if continue_signal else "kill_or_reframe",
        "continue": continue_signal,
        "more_than_clipping": bool(native.get("full_vs_clipping_wrong_target_delta", 0.0) > 0.0 or native.get("full_vs_clipping_unsafe_delta", 0.0) > 0.0),
        "more_than_safety_only": bool(native.get("full_vs_safety_wrong_target_delta", 0.0) > 0.0),
        "semantic_component_useful": bool(native.get("full_vs_safety_wrong_target_delta", 0.0) > 0.0),
        "intervention_rate_acceptable": bool(_num(native.get("full_intervention_rate"), 1.0) < 0.95),
        "native_action_evidence_sufficient": bool(native.get("continue")),
        "multitask_evidence_sufficient": bool(multi.get("continue")) if multi else False,
        "ral_plausible": continue_signal,
        "reason": reason,
    }
    _write_json("reports/css_shield_phase2_novelty_check.json", result)
    _write_md(
        "reports/css_shield_phase2_novelty_check.md",
        [
            "# CSS-Shield Phase 2 Baseline And Novelty Check",
            "",
            f"- decision: `{result['decision']}`",
            f"- more than clipping: `{result['more_than_clipping']}`",
            f"- more than safety-only: `{result['more_than_safety_only']}`",
            f"- semantic component useful: `{result['semantic_component_useful']}`",
            f"- intervention rate acceptable: `{result['intervention_rate_acceptable']}`",
            f"- native-action evidence sufficient: `{result['native_action_evidence_sufficient']}`",
            f"- multi-task evidence sufficient: `{result['multitask_evidence_sufficient']}`",
            f"- RA-L plausible: `{result['ral_plausible']}`",
            f"- reason: {result['reason']}",
        ],
    )
    return result


def write_phase2_package(native: dict[str, Any], novelty: dict[str, Any]) -> None:
    promising = bool(native.get("continue") and novelty.get("continue"))
    if promising:
        _write_md(
            "reports/css_shield_phase2_results.md",
            [
                "# CSS-Shield Phase 2 Results",
                "",
                "Diagnostic-only Phase 2 evidence. This is not a paper-grade claim.",
                "",
                f"- native-action full vs safety wrong-target delta: `{native.get('full_vs_safety_wrong_target_delta')}`",
                f"- native-action full vs clipping wrong-target delta: `{native.get('full_vs_clipping_wrong_target_delta')}`",
                f"- utility drop: `{native.get('utility_drop_vs_no_shield')}`",
                f"- decision: `{novelty.get('decision')}`",
            ],
        )
        _write_md(
            "reports/css_shield_ral_go_no_go.md",
            [
                "# CSS-Shield RA-L Go/No-Go",
                "",
                "Decision: continue diagnostics, not RA-L ready.",
                "",
                "CSS-Shield remains diagnostic-only until broader task coverage and paper-grade rollout evidence exist.",
            ],
        )
    else:
        _write_md(
            "reports/css_shield_phase2_kill_report.md",
            [
                "# CSS-Shield Phase 2 Kill/Reframe Report",
                "",
                "Decision: kill or reframe the current RA-L route.",
                "",
                f"- reason: {native.get('reason') or novelty.get('reason')}",
                f"- native-action wrong-target metric computable: `{native.get('wrong_target_metric_computable')}`",
                f"- full vs safety wrong-target delta: `{native.get('full_vs_safety_wrong_target_delta')}`",
                f"- full vs clipping wrong-target delta: `{native.get('full_vs_clipping_wrong_target_delta')}`",
                "",
                "If CSS-Shield is continued, it should be reframed around diagnostics or a stronger native-action semantic metric.",
            ],
        )
        _write_md(
            "reports/css_shield_failure_tree.md",
            [
                "# CSS-Shield Phase 2 Failure Tree",
                "",
                "1. First diagnostic package looked promising under controlled proposals.",
                "2. Phase 2 package audit allowed native-action testing.",
                "3. Native SmolVLA rollout diagnostic ran under bounded conditions.",
                "4. Native-action semantic/wrong-target evidence did not clear the Phase 2 gate.",
                "5. Current RA-L route should be killed or reframed unless the native semantic metric is redesigned.",
            ],
        )
        _write_md(
            "reports/css_shield_pivot_options.md",
            [
                "# CSS-Shield Phase 2 Pivot Options",
                "",
                "A. Reframe CSS-Shield as a diagnostic tool for semantic safety failures.",
                "B. Build a stronger native-action wrong-target metric before more experiments.",
                "C. Evaluate on tasks where native policy produces measurable semantic target confusion.",
                "D. Abandon CSS-Shield as a main RA-L control-improvement route if native-action gains remain absent.",
            ],
        )
    readiness = [
        "# CSS-Shield RA-L Readiness",
        "",
        f"Phase 2 decision: `{novelty.get('decision')}`",
        "",
        "Status: diagnostic-only, not RA-L ready.",
        "",
        f"- more than safety-only: `{novelty.get('more_than_safety_only')}`",
        f"- more than clipping-only: `{novelty.get('more_than_clipping')}`",
        f"- native-action evidence sufficient: `{novelty.get('native_action_evidence_sufficient')}`",
        f"- RA-L plausible: `{novelty.get('ral_plausible')}`",
    ]
    _write_md("reports/css_shield_ral_readiness.md", readiness)


def write_state_files(state: dict[str, Any]) -> None:
    _write_json("reports/css_shield_phase2_state.json", state)
    _write_md(
        "reports/css_shield_phase2_state.md",
        [
            "# CSS-Shield Phase 2 State",
            "",
            f"- current main commit: `{state.get('current_main_commit')}`",
            f"- current state: `{state.get('current_state')}`",
            f"- last completed state: `{state.get('last_completed_state')}`",
            f"- decision: `{state.get('continue_kill_decision')}`",
            f"- next milestone: `{state.get('next_milestone')}`",
            f"- reason: {state.get('why_next')}",
            f"- rollout happened: `{state.get('rollout_happened')}`",
            f"- native SmolVLA inference happened: `{state.get('native_smolvla_inference_happened')}`",
            f"- training/loss: `{state.get('training_happened')}` / `{state.get('loss_computed')}`",
            f"- GPU/download/OpenVLA-OFT: `{state.get('gpu_happened')}` / `{state.get('download_happened')}` / `{state.get('openvla_oft_happened')}`",
            f"- resume command: `{state.get('exact_resume_command')}`",
            "",
            "## Last Result Summary",
            "",
            "```json",
            json.dumps(state.get("last_result_summary"), indent=2, sort_keys=True),
            "```",
        ],
    )
    existing = Path("reports/css_shield_phase2_decision_log.md").read_text(encoding="utf-8") if Path("reports/css_shield_phase2_decision_log.md").exists() else "# CSS-Shield Phase 2 Decision Log\n"
    _write_md(
        "reports/css_shield_phase2_decision_log.md",
        [
            existing.rstrip(),
            "",
            "## Phase 2 Update",
            "",
            f"- completed: `{state.get('last_completed_state')}`",
            f"- decision: `{state.get('continue_kill_decision')}`",
            f"- next: `{state.get('current_state')}`",
            f"- reason: {state.get('why_next')}",
        ],
    )
    _write_md(
        "reports/css_shield_phase2_risk_register.md",
        [
            "# CSS-Shield Phase 2 Risk Register",
            "",
            "- Native SmolVLA actions may not expose a computable wrong-target metric; mitigation: kill/reframe rather than rely on synthetic proposals.",
            "- Full shield may equal safety-only under native actions; mitigation: require full-vs-safety wrong-target improvement.",
            "- Utility collapse may hide behind safety improvements; mitigation: bound utility drop and intervention rate.",
            "- Multi-task support may be insufficient; mitigation: stop before paper claims and report concrete runner gaps.",
            "- Evidence is diagnostic-only; mitigation: keep RA-L readiness separate from paper-grade claims.",
        ],
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-commit", required=True)
    parser.add_argument("--complete-state", required=True, choices=["STATE1", "STATE2", "STATE3", "STATE4", "STATE5"])
    args = parser.parse_args(argv)

    if args.complete_state == "STATE1":
        audit = audit_diagnostic_package()
        state = _phase_state(
            main_commit=args.main_commit,
            current_state="PHASE2_STATE 2" if audit["continue"] else "PHASE2_STATE 5",
            last_completed="PHASE2_STATE 1",
            decision=audit["decision"],
            next_milestone="More realistic native-action diagnostic" if audit["continue"] else "Phase 2 kill/reframe package",
            reason=audit["reason"],
            metrics=audit,
            policy={
                "rollout_happened": audit["real_simulator_rollout_evidence"],
                "model_inference_performed": audit["native_smolvla_action_evidence"],
            },
            hard_blockers=[] if audit["continue"] else [audit["reason"]],
        )
    elif args.complete_state == "STATE2":
        native = assess_native_action_report()
        state = _phase_state(
            main_commit=args.main_commit,
            current_state="PHASE2_STATE 3" if native["continue"] else "PHASE2_STATE 5",
            last_completed="PHASE2_STATE 2",
            decision=native["decision"],
            next_milestone="Multi-task/randomized diagnostic" if native["continue"] else "Phase 2 kill/reframe package",
            reason=native["reason"],
            metrics=native,
            policy={
                "rollout_happened": native["rollout_happened"],
                "model_inference_performed": native["native_smolvla_inference_happened"],
            },
            hard_blockers=[] if native["continue"] else [native["reason"]],
        )
    elif args.complete_state == "STATE3":
        multi = assess_multitask_reports()
        policy = {"rollout_happened": bool(multi.get("task_count")), "model_inference_performed": True}
        state = _phase_state(
            main_commit=args.main_commit,
            current_state="PHASE2_STATE 4" if multi["continue"] else "PHASE2_STATE 5",
            last_completed="PHASE2_STATE 3",
            decision=multi["decision"],
            next_milestone="Baseline and novelty check" if multi["continue"] else "Phase 2 kill/reframe package",
            reason=multi["reason"],
            metrics=multi,
            policy=policy,
            hard_blockers=[] if multi["continue"] else [multi["reason"]],
        )
    elif args.complete_state == "STATE4":
        native = _load_json("reports/css_shield_phase2_native_action_assessment.json")
        multi = _load_json("reports/css_shield_phase2_multitask_assessment.json")
        novelty = assess_novelty(native, multi if multi else None)
        state = _phase_state(
            main_commit=args.main_commit,
            current_state="PHASE2_STATE 5",
            last_completed="PHASE2_STATE 4",
            decision=novelty["decision"],
            next_milestone="RA-L go/no-go package",
            reason=novelty["reason"],
            metrics=novelty,
            policy={"rollout_happened": True, "model_inference_performed": True},
            hard_blockers=[] if novelty["continue"] else [novelty["reason"]],
        )
    else:
        native = _load_json("reports/css_shield_phase2_native_action_assessment.json")
        novelty = _load_json("reports/css_shield_phase2_novelty_check.json")
        if not novelty:
            novelty = assess_novelty(native)
        write_phase2_package(native, novelty)
        state = _phase_state(
            main_commit=args.main_commit,
            current_state="COMPLETE",
            last_completed="PHASE2_STATE 5",
            decision=novelty["decision"],
            next_milestone="No executable next state" if novelty["continue"] else "CSS-Shield kill/reframe review",
            reason="Phase 2 go/no-go package created." if novelty["continue"] else "Phase 2 kill/reframe package created.",
            metrics=novelty,
            policy={"rollout_happened": True, "model_inference_performed": True},
            hard_blockers=[] if novelty["continue"] else [novelty.get("reason", "Phase 2 gate failed.")],
        )

    write_state_files(state)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
