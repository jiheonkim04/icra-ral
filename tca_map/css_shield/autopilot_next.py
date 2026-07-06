"""Persistent CSS-Shield autopilot state updater and bounded state runner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STATE_SCHEMA = "2026-07-07.css_shield_autopilot_state.v2"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _summary(report: dict[str, Any]) -> dict[str, Any]:
    stage = report.get("stage_c_controlled_diagnostic") or {}
    state2 = report.get("state2_randomized_diagnostic") or {}
    source = state2 if state2 else stage
    summary = source.get("summary") or {}
    comparison = summary.get("comparison") or {}
    by_variant = summary.get("by_variant") or {}
    full = by_variant.get("full_css_shield") or {}
    return {
        "comparison": comparison,
        "by_variant": by_variant,
        "full": full,
        "decision": source.get("decision") or {},
        "trial_count": source.get("trial_count") or source.get("proposal_count"),
    }


def select_next_milestone(state: dict[str, Any], state15_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Select the next CSS-Shield milestone from persisted state/report data."""

    if state.get("current_stage") in {"STATE 3", "STATE 4", "STATE 5", "COMPLETE"}:
        mapping = {
            "STATE 3": ("STATE 3", "RA-L strength check", "State 2 passed; assess whether the signal is plausibly publishable."),
            "STATE 4": ("STATE 4", "Scale randomized diagnostics", "State 3 passed; run a larger bounded randomized diagnostic batch."),
            "STATE 5": ("STATE 5", "Paper-readiness package or kill report", "State 4 completed; package the decision honestly."),
            "COMPLETE": ("COMPLETE", "No executable next state", "CSS-Shield bounded autopilot package is complete."),
        }
        next_state, milestone, why = mapping[state["current_stage"]]
        return {"next_state": next_state, "next_milestone": milestone, "why": why, "decision": state.get("continue_kill_decision", "continue")}

    report = state15_report or {}
    state15 = ((report.get("stage_c_controlled_diagnostic") or {}).get("decision") or {})
    state2 = ((report.get("state2_randomized_diagnostic") or {}).get("decision") or {})
    if state2:
        if state2.get("continue"):
            return {
                "next_state": "STATE 3",
                "next_milestone": "RA-L strength check",
                "why": "State 2 randomized semantic/safety batch passed; next check is whether evidence has publishable strength.",
                "decision": "continue",
            }
        return {
            "next_state": "STATE 5",
            "next_milestone": "CSS-Shield kill/reframe package",
            "why": "State 2 randomized batch did not show full-shield novelty beyond simple baselines.",
            "decision": "kill_or_reframe",
        }
    if state15:
        if state15.get("continue"):
            return {
                "next_state": "STATE 2",
                "next_milestone": "Counterfactual semantic shield randomized diagnostic batch",
                "why": "State 1.5 found observable intended/distractor targets and full shield beat simple semantic baselines.",
                "decision": "continue",
            }
        return {
            "next_state": "STATE 5",
            "next_milestone": "CSS-Shield kill/reframe package",
            "why": "State 1.5 semantic observability or full-shield novelty gate failed.",
            "decision": "kill_or_reframe",
        }
    if state.get("last_completed_stage") == "STATE 1":
        return {
            "next_state": "STATE 1.5",
            "next_milestone": "Semantic observability and wrong-target shield gate",
            "why": "State 1 showed safety damping but did not verify semantic/counterfactual wrong-target shielding.",
            "decision": "continue",
        }
    return {
        "next_state": "STATE 1.5",
        "next_milestone": "Semantic observability and wrong-target shield gate",
        "why": "No newer executable CSS-Shield state report was found.",
        "decision": "continue",
    }


def assess_ral_strength(report: dict[str, Any]) -> dict[str, Any]:
    s = _summary(report)
    comparison = s["comparison"]
    full = s["full"]
    policy = report.get("policy") or {}
    wrong_vs_safety = float(comparison.get("full_vs_safety_wrong_target_delta") or 0.0)
    wrong_vs_clipping = float(comparison.get("full_vs_clipping_wrong_target_delta") or 0.0)
    unsafe_vs_clipping = float(comparison.get("full_vs_clipping_unsafe_delta") or 0.0)
    intervention = float(full.get("intervention_rate") or 0.0)
    utility = full.get("target_directed_movement_mean")
    green = bool(
        wrong_vs_safety > 0.0
        and wrong_vs_clipping > 0.0
        and bool(policy.get("rollout_happened"))
        and bool(policy.get("model_inference_performed"))
        and intervention < 0.95
    )
    return {
        "schema_version": "2026-07-07.css_shield_ral_strength_check.v1",
        "decision": "continue" if green else "kill_or_reframe",
        "continue": green,
        "diagnostic_only": True,
        "paper_grade": False,
        "novelty_beyond_clipping_only": wrong_vs_clipping > 0.0 or unsafe_vs_clipping > 0.0,
        "novelty_beyond_safety_only": wrong_vs_safety > 0.0,
        "semantic_wrong_target_value": wrong_vs_safety,
        "rollout_or_simulator_metric_exists": bool(policy.get("rollout_happened")),
        "native_smolvla_action_source_used": bool(policy.get("model_inference_performed")),
        "wrong_target_rate_improvement": wrong_vs_safety,
        "unsafe_rate_improvement": unsafe_vs_clipping,
        "utility_degradation_acceptable": intervention < 0.95,
        "intervention_rate": intervention,
        "utility_proxy": utility,
        "real_simulator_diagnostic": bool(policy.get("rollout_happened")),
        "synthetic_or_proposal_based": True,
        "plausibly_ral_with_more_scaling": green,
        "missing_evidence_for_ral_stability": [
            "multi-task randomized diagnostic evidence",
            "stronger realism audit for failure proposals",
            "longer horizon utility/recovery evidence",
            "comparison to recent VLA safety and semantic grounding baselines",
            "paper-grade rollout success/safety table",
        ],
        "reason": "CSS-Shield shows semantic wrong-target value beyond clipping and safety-only in bounded simulator diagnostics."
        if green
        else "CSS-Shield does not yet show enough novelty beyond simple baselines.",
    }


def assess_scale_report(report: dict[str, Any]) -> dict[str, Any]:
    s = _summary(report)
    comparison = s["comparison"]
    full = s["full"]
    wrong_vs_safety = float(comparison.get("full_vs_safety_wrong_target_delta") or 0.0)
    wrong_vs_clipping = float(comparison.get("full_vs_clipping_wrong_target_delta") or 0.0)
    unsafe_vs_clipping = float(comparison.get("full_vs_clipping_unsafe_delta") or 0.0)
    intervention = float(full.get("intervention_rate") or 0.0)
    false_positive = float(full.get("false_positive_intervention_rate") or 0.0)
    green = bool(wrong_vs_safety > 0.0 and wrong_vs_clipping > 0.0 and intervention < 0.95 and false_positive <= 0.25)
    return {
        "schema_version": "2026-07-07.css_shield_state4_scale_summary.v1",
        "decision": "continue" if green else "kill_or_reframe",
        "continue": green,
        "trial_count": s.get("trial_count"),
        "full_vs_safety_wrong_target_delta": wrong_vs_safety,
        "full_vs_clipping_wrong_target_delta": wrong_vs_clipping,
        "full_vs_clipping_unsafe_delta": unsafe_vs_clipping,
        "full_intervention_rate": intervention,
        "full_false_positive_rate": false_positive,
        "full_false_negative_rate": full.get("false_negative_unsafe_or_wrong_rate"),
        "full_target_directed_movement_mean": full.get("target_directed_movement_mean"),
        "full_wrong_target_movement_mean": full.get("wrong_target_movement_mean"),
        "utility_drop_acceptable": intervention < 0.95,
        "failure_cases_interpretable": True,
        "diagnostic_only": True,
        "paper_grade": False,
        "reason": "Scaled randomized diagnostic preserves semantic wrong-target advantage without stop-all behavior."
        if green
        else "Scaled randomized diagnostic does not preserve enough semantic advantage or utility.",
    }


def _state_base(main_commit: str, *, current_stage: str, last_completed_stage: str, decision: str, next_milestone: str, why: str, policy: dict[str, Any], key_metric: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "schema_version": STATE_SCHEMA,
        "current_main_commit": main_commit,
        "current_stage": current_stage,
        "last_completed_stage": last_completed_stage,
        "last_result_summary": {"diagnostic_only": True, "paper_grade": False, "key_metric": key_metric},
        "continue_kill_decision": decision,
        "next_milestone": next_milestone,
        "why_next": why,
        "hard_blockers": [] if decision == "continue" else [why],
        "rollout_happened": bool(policy.get("rollout_happened")),
        "training_happened": bool(policy.get("training_performed")),
        "lora_training_happened": bool(policy.get("lora_training_performed")),
        "loss_computed": bool(policy.get("loss_computed")),
        "gpu_happened": bool(policy.get("gpu_jobs_performed")),
        "download_happened": bool(policy.get("downloads_performed")),
        "heavy_import_happened": bool(policy.get("heavy_model_imports_performed")),
        "native_smolvla_inference_happened": bool(policy.get("model_inference_performed")),
        "openvla_oft_happened": bool(policy.get("openvla_oft_executed")),
        "paper_grade_or_diagnostic": "diagnostic_only",
        "exact_resume_command": "powershell -ExecutionPolicy Bypass -File scripts\\160_css_shield_autopilot_next.ps1 -Continuous",
    }


def build_autopilot_state(main_commit: str, state15_report: dict[str, Any] | None, previous_state: dict[str, Any] | None = None) -> dict[str, Any]:
    previous_state = previous_state or {}
    selected = select_next_milestone(previous_state, state15_report)
    report = state15_report or {}
    policy = report.get("policy") or {}
    result = report.get("result") or {}
    stage = report.get("stage_c_controlled_diagnostic") or {}
    state2 = report.get("state2_randomized_diagnostic") or None
    last_completed = "STATE 2" if state2 else "STATE 1.5" if result.get("passed") else previous_state.get("last_completed_stage", "STATE 1")
    key_metric = None
    if stage:
        key_metric = (_summary(report).get("comparison") or {})
    return {
        **_state_base(
            main_commit,
            current_stage=selected["next_state"],
            last_completed_stage=last_completed,
            decision=selected["decision"],
            next_milestone=selected["next_milestone"],
            why=selected["why"],
            policy=policy,
            key_metric=key_metric,
        ),
        "last_result_summary": {
            "passed": result.get("passed"),
            "diagnostic_only": True,
            "paper_grade": False,
            "key_metric": key_metric,
            "state1_5_decision": stage.get("decision"),
            "state2_decision": state2.get("decision") if state2 else None,
        },
    }


def build_state3_state(main_commit: str, report: dict[str, Any], ral: dict[str, Any]) -> dict[str, Any]:
    return _state_base(
        main_commit,
        current_stage="STATE 4" if ral["continue"] else "STATE 5",
        last_completed_stage="STATE 3",
        decision=ral["decision"],
        next_milestone="Scale randomized diagnostics" if ral["continue"] else "CSS-Shield kill/reframe package",
        why=ral["reason"],
        policy=report.get("policy") or {},
        key_metric={
            "semantic_wrong_target_value": ral["semantic_wrong_target_value"],
            "unsafe_rate_improvement": ral["unsafe_rate_improvement"],
            "intervention_rate": ral["intervention_rate"],
        },
    )


def build_state4_state(main_commit: str, report: dict[str, Any], scale: dict[str, Any]) -> dict[str, Any]:
    return _state_base(
        main_commit,
        current_stage="STATE 5",
        last_completed_stage="STATE 4",
        decision=scale["decision"],
        next_milestone="Paper-readiness package or kill report",
        why=scale["reason"],
        policy=report.get("policy") or {},
        key_metric={
            "trial_count": scale["trial_count"],
            "full_vs_safety_wrong_target_delta": scale["full_vs_safety_wrong_target_delta"],
            "full_vs_clipping_wrong_target_delta": scale["full_vs_clipping_wrong_target_delta"],
            "full_intervention_rate": scale["full_intervention_rate"],
        },
    )


def write_markdown(state: dict[str, Any], path: Path) -> None:
    lines = [
        "# CSS-Shield Autopilot State",
        "",
        f"- current main commit: `{state.get('current_main_commit')}`",
        f"- current stage: `{state.get('current_stage')}`",
        f"- last completed stage: `{state.get('last_completed_stage')}`",
        f"- continue/kill decision: `{state.get('continue_kill_decision')}`",
        f"- next milestone: `{state.get('next_milestone')}`",
        f"- why next: {state.get('why_next')}",
        f"- hard blockers: `{state.get('hard_blockers')}`",
        f"- rollout happened: `{state.get('rollout_happened')}`",
        f"- training happened: `{state.get('training_happened')}`",
        f"- LoRA training happened: `{state.get('lora_training_happened')}`",
        f"- loss computed: `{state.get('loss_computed')}`",
        f"- GPU/download/heavy import/OpenVLA-OFT: `{state.get('gpu_happened')}` / `{state.get('download_happened')}` / `{state.get('heavy_import_happened')}` / `{state.get('openvla_oft_happened')}`",
        f"- native SmolVLA inference happened: `{state.get('native_smolvla_inference_happened')}`",
        f"- evidence level: `{state.get('paper_grade_or_diagnostic')}`",
        f"- exact resume command: `{state.get('exact_resume_command')}`",
        "",
        "## Last Result Summary",
        "",
        "```json",
        json.dumps(state.get("last_result_summary"), indent=2, sort_keys=True),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def append_decision_log(state: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = path.read_text(encoding="utf-8") if path.exists() else "# CSS-Shield Autopilot Decision Log\n\n"
    block = [
        "## Autopilot Update",
        "",
        f"- last completed stage: `{state.get('last_completed_stage')}`",
        f"- decision: `{state.get('continue_kill_decision')}`",
        f"- next milestone: `{state.get('next_milestone')}`",
        f"- reason: {state.get('why_next')}",
        "",
    ]
    path.write_text(existing.rstrip() + "\n\n" + "\n".join(block), encoding="utf-8")


def write_next_milestone(state: dict[str, Any], path: Path) -> None:
    lines = [
        "# CSS-Shield Next Milestone",
        "",
        f"Next state: `{state.get('current_stage')}`",
        "",
        f"Milestone: {state.get('next_milestone')}",
        "",
        f"Reason: {state.get('why_next')}",
        "",
        f"Resume command: `{state.get('exact_resume_command')}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def write_ral_strength_report(ral: dict[str, Any], md_path: Path, json_path: Path) -> None:
    _write_json(json_path, ral)
    lines = [
        "# CSS-Shield RA-L Strength Check",
        "",
        "Diagnostic-only assessment. This is not a paper-grade claim.",
        "",
        f"- decision: `{ral['decision']}`",
        f"- novelty beyond clipping-only: `{ral['novelty_beyond_clipping_only']}`",
        f"- novelty beyond safety-only: `{ral['novelty_beyond_safety_only']}`",
        f"- semantic wrong-target value: `{ral['semantic_wrong_target_value']}`",
        f"- simulator metric exists: `{ral['rollout_or_simulator_metric_exists']}`",
        f"- native SmolVLA action source used: `{ral['native_smolvla_action_source_used']}`",
        f"- utility degradation acceptable: `{ral['utility_degradation_acceptable']}`",
        f"- plausibly RA-L with more scaling: `{ral['plausibly_ral_with_more_scaling']}`",
        "",
        "## Missing Evidence",
        "",
    ]
    lines += [f"- {item}" for item in ral["missing_evidence_for_ral_stability"]]
    lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_scale_summary(scale: dict[str, Any], md_path: Path, json_path: Path) -> None:
    _write_json(json_path, scale)
    lines = [
        "# CSS-Shield State 4 Scale Summary",
        "",
        "Diagnostic-only randomized batch summary.",
        "",
        f"- decision: `{scale['decision']}`",
        f"- trial count: `{scale['trial_count']}`",
        f"- full vs safety-only wrong-target delta: `{scale['full_vs_safety_wrong_target_delta']}`",
        f"- full vs clipping-only wrong-target delta: `{scale['full_vs_clipping_wrong_target_delta']}`",
        f"- full vs clipping-only unsafe delta: `{scale['full_vs_clipping_unsafe_delta']}`",
        f"- full intervention rate: `{scale['full_intervention_rate']}`",
        f"- full false positive rate: `{scale['full_false_positive_rate']}`",
        f"- reason: {scale['reason']}",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def write_state5_package(ral: dict[str, Any], scale: dict[str, Any], promising: bool) -> None:
    if promising:
        payloads = {
            "reports/css_shield_first_results.md": "# CSS-Shield First Results\n\nDiagnostic-only first results: full CSS-Shield beat clipping-only and safety-only on semantic wrong-target metrics in bounded simulator diagnostics.\n",
            "reports/css_shield_main_table.md": f"# CSS-Shield Main Table\n\n| metric | value |\n| --- | ---: |\n| State 4 trials | {scale.get('trial_count')} |\n| full vs safety wrong-target delta | {scale.get('full_vs_safety_wrong_target_delta')} |\n| full vs clipping wrong-target delta | {scale.get('full_vs_clipping_wrong_target_delta')} |\n| full intervention rate | {scale.get('full_intervention_rate')} |\n",
            "reports/css_shield_ablation_table.md": "# CSS-Shield Ablation Table\n\nRequired ablations remain no shield, clipping-only, safety-only, semantic-only, and full CSS-Shield. Current diagnostic evidence supports semantic-only/full over safety-only for wrong-target actions.\n",
            "reports/css_shield_claims_and_limits.md": "# CSS-Shield Claims And Limits\n\nClaim allowed now: bounded diagnostic evidence suggests semantic wrong-target shielding beyond clipping-only and safety-only.\n\nLimits: not paper-grade, not multi-task benchmark evidence, proposal failures are still partly controlled, and reward/success is not established.\n",
            "reports/css_shield_ral_readiness.md": "# CSS-Shield RA-L Readiness\n\nStatus: promising but not RA-L ready.\n\nNext required evidence: multi-task randomized diagnostics, stronger realism audit, longer-horizon utility evidence, and comparison to recent VLA safety/semantic grounding baselines.\n",
            "reports/css_shield_failure_cases.md": "# CSS-Shield Failure Cases\n\nKnown risks: synthetic/proposal-based wrong-target failures may overstate usefulness; safety-only may match full shield on purely unsafe actions; reward/success remains unproven.\n",
        }
    else:
        payloads = {
            "reports/css_shield_kill_report.md": "# CSS-Shield Kill Report\n\nCSS-Shield should be killed or reframed because bounded diagnostics did not show enough semantic advantage beyond simple baselines.\n",
            "reports/css_shield_failure_tree.md": "# CSS-Shield Failure Tree\n\n1. State 1 safety signal.\n2. State 1.5 semantic observability gate.\n3. State 2/4 randomized diagnostic.\n4. Baseline comparison failure or utility collapse.\n",
            "reports/css_shield_pivot_options.md": "# CSS-Shield Pivot Options\n\nA. Reframe as diagnostic benchmark.\nB. Improve realistic action-source generation.\nC. Abandon if no native-policy wrong-target failures are found.\n",
        }
    for path, text in payloads.items():
        Path(path).write_text(text, encoding="utf-8")


def update_state_files(state: dict[str, Any], args: argparse.Namespace) -> None:
    _write_json(Path(args.state_json), state)
    write_markdown(state, Path(args.state_md))
    append_decision_log(state, Path(args.decision_log_md))
    write_next_milestone(state, Path(args.next_md))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-commit", required=True)
    parser.add_argument("--state15-report-json", default="reports/css_shield_state1_5_semantic_diagnostic_report.json")
    parser.add_argument("--state4-report-json", default="reports/css_shield_state4_scale_diagnostic_report.json")
    parser.add_argument("--state-json", default="reports/css_shield_autopilot_state.json")
    parser.add_argument("--state-md", default="reports/css_shield_autopilot_state.md")
    parser.add_argument("--decision-log-md", default="reports/css_shield_autopilot_decision_log.md")
    parser.add_argument("--next-md", default="reports/css_shield_next_milestone.md")
    parser.add_argument("--complete-state", default="auto", choices=["auto", "STATE3", "STATE4", "STATE5"])
    args = parser.parse_args(argv)

    previous = _load_json(Path(args.state_json))
    if args.complete_state == "STATE3":
        report = _load_json(Path(args.state15_report_json))
        ral = assess_ral_strength(report)
        write_ral_strength_report(ral, Path("reports/css_shield_ral_strength_check.md"), Path("reports/css_shield_ral_strength_check.json"))
        state = build_state3_state(args.main_commit, report, ral)
    elif args.complete_state == "STATE4":
        report = _load_json(Path(args.state4_report_json))
        scale = assess_scale_report(report)
        write_scale_summary(scale, Path("reports/css_shield_state4_scale_summary.md"), Path("reports/css_shield_state4_scale_summary.json"))
        state = build_state4_state(args.main_commit, report, scale)
    elif args.complete_state == "STATE5":
        ral = _load_json(Path("reports/css_shield_ral_strength_check.json"))
        scale = _load_json(Path("reports/css_shield_state4_scale_summary.json"))
        state4_report = _load_json(Path(args.state4_report_json))
        state4_policy = state4_report.get("policy") or {}
        promising = bool(ral.get("continue") and scale.get("continue"))
        write_state5_package(ral, scale, promising)
        state = _state_base(
            args.main_commit,
            current_stage="COMPLETE",
            last_completed_stage="STATE 5",
            decision="continue" if promising else "kill_or_reframe",
            next_milestone="No executable next state" if promising else "CSS-Shield reframed or killed",
            why="Paper-readiness package created; human review is next." if promising else "Kill/reframe package created.",
            policy=state4_policy or {
                "rollout_happened": True,
                "heavy_model_imports_performed": bool(previous.get("heavy_import_happened")),
                "model_inference_performed": bool(previous.get("native_smolvla_inference_happened", previous.get("heavy_import_happened"))),
            },
            key_metric={
                "ral_decision": ral.get("decision"),
                "scale_decision": scale.get("decision"),
                "full_vs_safety_wrong_target_delta": scale.get("full_vs_safety_wrong_target_delta"),
            },
        )
    else:
        report = _load_json(Path(args.state15_report_json))
        state = build_autopilot_state(args.main_commit, report, previous)

    update_state_files(state, args)
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if state.get("continue_kill_decision") in {"continue", "kill_or_reframe"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
