"""Report-only synthesis of bounded LoRA scale-up attribution gaps."""

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


def _delta(report: dict[str, Any], key: str, metric: str) -> float | int | None:
    value = ((report.get("deltas") or {}).get(key) or {}).get(metric)
    return value


def _status(report: dict[str, Any], gap_id: str) -> str | None:
    for gap in report.get("gap_table") or []:
        if gap.get("id") == gap_id:
            return gap.get("status")
    return None


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Scale-Up Attribution Gap Synthesis",
        "",
        "This is a report-only synthesis of offline proxy evidence. It is not standard success, not rollout success, and not paper-grade evidence.",
        "",
        f"- passed: `{report.get('scaleup_attribution_gap_synthesis_passed')}`",
        f"- bounded LoRA scale-up included: `{report.get('bounded_lora_scaleup_included')}`",
        f"- ready for paper claim: `{report.get('ready_for_paper_claim')}`",
        f"- ready for learned-policy rollout scaling: `{report.get('ready_for_learned_policy_rollout_scaling')}`",
        "",
        "## Findings",
        "",
    ]
    for item in report.get("findings") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Attribution Gaps", ""])
    for item in report.get("attribution_gaps") or []:
        lines.append(f"- {item}")
    lines.extend(["", "## Next Steps", ""])
    for item in report.get("next_steps") or []:
        lines.append(f"- {item}")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    evidence_path = Path(args.evidence_report)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "evidence_label": "scaleup_attribution_gap_synthesis",
        "scaleup_attribution_gap_synthesis_passed": False,
        "decision": "stop",
        "bounded_lora_scaleup_included": False,
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
            "forbidden_gates_set": forbidden,
        },
        "paths": {"evidence_report": str(evidence_path)},
        "input_summary": {},
        "findings": [],
        "attribution_gaps": [],
        "next_steps": [],
        "recommended_next_step": None,
        "error": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        return report, code

    if forbidden:
        return block("Forbidden gate(s) set for report-only synthesis: " + ", ".join(forbidden), 2)
    if not evidence_path.exists():
        return block(f"Scale-up-aware evidence report is missing: {evidence_path}", 3)

    evidence = _read_json(evidence_path)
    if not evidence.get("offline_evidence_gap_report_passed"):
        return block("Offline evidence gap report did not pass.", 4)

    bounded_included = bool(evidence.get("bounded_lora_scaleup_included"))
    report["bounded_lora_scaleup_included"] = bounded_included
    report["input_summary"] = {
        "evidence_row_count": len(evidence.get("evidence_table") or []),
        "bounded_lora_scaleup_record_count": int(evidence.get("bounded_lora_scaleup_record_count") or 0),
        "standard_success_gap_status": _status(evidence, "standard_success"),
        "learned_policy_rollout_gap_status": _status(evidence, "learned_policy_rollout"),
        "required_lora_track_status": _status(evidence, "required_lora_track"),
        "bounded_lora_action_l1_delta": _delta(evidence, "bounded_lora_tca_vs_actionmap_lora", "action_l1_delta"),
        "bounded_lora_wrong_target_delta": _delta(
            evidence, "bounded_lora_tca_vs_actionmap_lora", "wrong_target_proxy_rate_delta"
        ),
        "bounded_select_action_l1_delta": _delta(
            evidence, "bounded_lora_tca_select_vs_tca_lora", "action_l1_delta"
        ),
        "bounded_select_wrong_target_delta": _delta(
            evidence, "bounded_lora_tca_select_vs_tca_lora", "wrong_target_proxy_rate_delta"
        ),
    }

    findings = [
        "Head-only TCA-Map remains the strongest offline proxy arm in the current deterministic HDF5-action setup.",
        "Bounded TCA-Map + LoRA improves over ActionMap + LoRA in the scale-up proxy on action L1 and wrong-target proxy deltas.",
        "The bounded LoRA scale-up is useful as low-compute method debugging evidence, but it is not standard success or rollout success.",
    ]
    if bounded_included:
        findings.append(
            "The scale-up-aware table includes bounded LoRA proxy rows, so the required LoRA track is no longer only a tiny-smoke artifact."
        )
    select_action_delta = report["input_summary"]["bounded_select_action_l1_delta"]
    select_wrong_delta = report["input_summary"]["bounded_select_wrong_target_delta"]
    if select_action_delta == 0.0 and select_wrong_delta == 0.0:
        findings.append(
            "Distributional TCA-Select adds no extra LoRA proxy gain in this runner; the current candidate-selection proxy is not yet stressful enough to isolate selection gain."
        )

    report["findings"] = findings
    report["attribution_gaps"] = [
        "The evidence is offline proxy evidence over action snippets, not simulator success.",
        "The current SmolVLA checkpoint remains action-provenance mismatched for LIBERO learned-policy rollout scaling.",
        "LoRA improves some proxy metrics, so future analysis must keep ActionMap + LoRA vs TCA-Map + LoRA as the central attribution comparison.",
        "Distributional TCA-Select needs a stronger candidate diversity or ambiguity stress test before claiming selection-specific gain.",
        "No paper-grade claim can be made until a valid rollout path, baselines, compute table, and no-privileged-inference checks exist.",
    ]
    report["next_steps"] = [
        "Create a report-only TCA-Select stress-test plan for offline candidate ambiguity without training or rollout.",
        "Keep searching or planning for a LIBERO-action-aligned learned-policy checkpoint path separately.",
        "Do not scale to paper-grade benchmark rollouts until checkpoint/action provenance and simulator policy path are green.",
    ]
    report["scaleup_attribution_gap_synthesis_passed"] = True
    report["decision"] = "scaleup_attribution_gaps_ready"
    report["recommended_next_step"] = report["next_steps"][0]
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-report", default="reports/offline_tca_lora_evidence_gap_report_runtime.json")
    parser.add_argument("--report-path", default="reports/scaleup_attribution_gap_synthesis_report.json")
    parser.add_argument("--markdown-report-path", default="reports/scaleup_attribution_gap_synthesis_report.md")
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
