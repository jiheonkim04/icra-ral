"""Persistent CSS-Shield autopilot state updater."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def select_next_milestone(state: dict[str, Any], state15_report: dict[str, Any] | None = None) -> dict[str, Any]:
    """Select the next CSS-Shield milestone from persisted state/report data."""

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
        comparison = ((stage.get("summary") or {}).get("comparison") or {})
        key_metric = {
            "full_vs_safety_wrong_target_delta": comparison.get("full_vs_safety_wrong_target_delta"),
            "full_vs_clipping_wrong_target_delta": comparison.get("full_vs_clipping_wrong_target_delta"),
            "full_vs_clipping_unsafe_delta": comparison.get("full_vs_clipping_unsafe_delta"),
        }
    if state2:
        comparison = (((state2.get("summary") or {}).get("comparison") or {}))
        key_metric = {
            "state2_full_vs_safety_wrong_target_delta": comparison.get("full_vs_safety_wrong_target_delta"),
            "state2_full_vs_clipping_wrong_target_delta": comparison.get("full_vs_clipping_wrong_target_delta"),
            "state2_full_vs_clipping_unsafe_delta": comparison.get("full_vs_clipping_unsafe_delta"),
        }
    return {
        "schema_version": "2026-07-07.css_shield_autopilot_state.v1",
        "current_main_commit": main_commit,
        "current_stage": selected["next_state"],
        "last_completed_stage": last_completed,
        "last_result_summary": {
            "passed": result.get("passed"),
            "diagnostic_only": True,
            "paper_grade": False,
            "key_metric": key_metric,
            "state1_5_decision": stage.get("decision"),
            "state2_decision": state2.get("decision") if state2 else None,
        },
        "continue_kill_decision": selected["decision"],
        "next_milestone": selected["next_milestone"],
        "why_next": selected["why"],
        "hard_blockers": [] if selected["decision"] == "continue" else [selected["why"]],
        "rollout_happened": bool(policy.get("rollout_happened")),
        "training_happened": bool(policy.get("training_performed")),
        "lora_training_happened": bool(policy.get("lora_training_performed")),
        "loss_computed": bool(policy.get("loss_computed")),
        "gpu_happened": bool(policy.get("gpu_jobs_performed")),
        "download_happened": bool(policy.get("downloads_performed")),
        "heavy_import_happened": bool(policy.get("heavy_model_imports_performed")),
        "openvla_oft_happened": bool(policy.get("openvla_oft_executed")),
        "paper_grade_or_diagnostic": "diagnostic_only",
        "exact_resume_command": "powershell -ExecutionPolicy Bypass -File scripts\\160_css_shield_autopilot_next.ps1",
    }


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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--main-commit", required=True)
    parser.add_argument("--state15-report-json", default="reports/css_shield_state1_5_semantic_diagnostic_report.json")
    parser.add_argument("--state-json", default="reports/css_shield_autopilot_state.json")
    parser.add_argument("--state-md", default="reports/css_shield_autopilot_state.md")
    parser.add_argument("--decision-log-md", default="reports/css_shield_autopilot_decision_log.md")
    parser.add_argument("--next-md", default="reports/css_shield_next_milestone.md")
    args = parser.parse_args(argv)

    previous = _load_json(Path(args.state_json))
    report = _load_json(Path(args.state15_report_json))
    state = build_autopilot_state(args.main_commit, report, previous)
    Path(args.state_json).write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(state, Path(args.state_md))
    append_decision_log(state, Path(args.decision_log_md))
    write_next_milestone(state, Path(args.next_md))
    print(json.dumps(state, indent=2, sort_keys=True))
    return 0 if state.get("continue_kill_decision") in {"continue", "kill_or_reframe"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
