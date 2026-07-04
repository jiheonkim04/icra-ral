"""Planning-only TCA-Select ambiguity stress-test gate."""

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


DEFAULT_LIMITS = {
    "max_pairs": 16,
    "max_records": 64,
    "candidate_count": 8,
    "temperature": 0.5,
    "max_runtime_seconds": 300,
}


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# TCA-Select Ambiguity Stress-Test Plan Report",
        "",
        "This is planning-only. It does not train, rollout, load models, or make paper claims.",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- passed: `{report.get('tca_select_ambiguity_stress_plan_passed')}`",
        f"- ready for offline runner: `{report.get('ready_for_offline_tca_select_ambiguity_stress_runner')}`",
        f"- ready for paper claim: `{report.get('ready_for_paper_claim')}`",
        "",
        "## Planned Metrics",
        "",
    ]
    for metric in report.get("planned_metrics") or []:
        lines.append(f"- `{metric}`")
    lines.extend(["", "## Next Step", "", str(report.get("recommended_next_step")), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    synthesis_path = Path(args.synthesis_report)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    limits = {
        "max_pairs": min(int(args.max_pairs), DEFAULT_LIMITS["max_pairs"]),
        "max_records": min(int(args.max_records), DEFAULT_LIMITS["max_records"]),
        "candidate_count": min(int(args.candidate_count), DEFAULT_LIMITS["candidate_count"]),
        "temperature": float(args.temperature),
        "max_runtime_seconds": min(int(args.max_runtime_seconds), DEFAULT_LIMITS["max_runtime_seconds"]),
        "device": "cpu",
        "training_allowed": False,
        "rollout_allowed": False,
        "privileged_inference_allowed": False,
    }
    report: dict[str, Any] = {
        "evidence_label": "tca_select_ambiguity_stress_plan",
        "tca_select_ambiguity_stress_plan_passed": False,
        "decision": "stop",
        "ready_for_offline_tca_select_ambiguity_stress_runner": False,
        "ready_for_learned_policy_rollout_scaling": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "policy": {
            "planning_only": True,
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
        "paths": {"synthesis_report": str(synthesis_path)},
        "limits": limits,
        "source_summary": {},
        "stress_test_design": {},
        "planned_metrics": [],
        "pass_fail_criteria": [],
        "recommended_next_step": None,
        "error": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        return report, code

    if forbidden:
        return block("Forbidden gate(s) set for planning-only stress plan: " + ", ".join(forbidden), 2)
    if not synthesis_path.exists():
        return block(f"Scale-up attribution synthesis report is missing: {synthesis_path}", 3)
    synthesis = _read_json(synthesis_path)
    if not synthesis.get("scaleup_attribution_gap_synthesis_passed"):
        return block("Scale-up attribution synthesis did not pass.", 4)
    if synthesis.get("ready_for_paper_claim"):
        return block("Unexpected paper-claim readiness in synthesis report; refusing stress-test plan.", 5)

    report["source_summary"] = {
        "bounded_lora_scaleup_included": bool(synthesis.get("bounded_lora_scaleup_included")),
        "selection_delta_action_l1": (synthesis.get("input_summary") or {}).get("bounded_select_action_l1_delta"),
        "selection_delta_wrong_target": (synthesis.get("input_summary") or {}).get(
            "bounded_select_wrong_target_delta"
        ),
        "recommended_next_step": synthesis.get("recommended_next_step"),
    }
    report["stress_test_design"] = {
        "data_source": "existing local LIBERO counterfactual split report and HDF5 action snippets",
        "candidate_generation": [
            "sample action candidates around expert and counterfactual action prefixes",
            "include near-tie candidates that have similar action distance but disagree in target consistency",
            "include nuisance perturbations that should not change the selected target-consistent action",
        ],
        "scoring": [
            "target-conditioned action-heatmap consistency",
            "full-vs-masked condition sensitivity",
            "distance-to-expert voxel/action as offline proxy only",
        ],
        "forbidden_inputs": [
            "privileged simulator state",
            "external verifier model",
            "rollout outcome",
            "paper-grade success labels",
        ],
    }
    report["planned_metrics"] = [
        "selection_action_l1_delta_vs_top_heatmap",
        "selection_wrong_target_proxy_delta_vs_top_heatmap",
        "target_consistency_margin",
        "condition_sensitivity_margin",
        "candidate_diversity_score",
        "nuisance_stability_score",
        "latency_ms",
        "max_gpu_memory_mb_zero_expected",
    ]
    report["pass_fail_criteria"] = [
        "runner must be offline proxy only and CPU-only",
        "no model loading, inference, training, rollout, simulator execution, GPU job, downloads, or OpenVLA-OFT",
        "selection should reduce wrong-target proxy versus top-heatmap baseline on ambiguous cases",
        "report must preserve not_standard_success and not_paper_grade labels",
    ]
    report["tca_select_ambiguity_stress_plan_passed"] = True
    report["decision"] = "proceed_offline_tca_select_ambiguity_stress_runner"
    report["ready_for_offline_tca_select_ambiguity_stress_runner"] = True
    report["recommended_next_step"] = (
        "Implement a CPU-only offline TCA-Select ambiguity stress-test runner using existing local LIBERO "
        "counterfactual split artifacts. Keep it offline proxy only and no paper claim."
    )
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesis-report", default="reports/scaleup_attribution_gap_synthesis_report.json")
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_LIMITS["max_pairs"])
    parser.add_argument("--max-records", type=int, default=DEFAULT_LIMITS["max_records"])
    parser.add_argument("--candidate-count", type=int, default=DEFAULT_LIMITS["candidate_count"])
    parser.add_argument("--temperature", type=float, default=DEFAULT_LIMITS["temperature"])
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_LIMITS["max_runtime_seconds"])
    parser.add_argument("--report-path", default="reports/tca_select_ambiguity_stress_plan_report.json")
    parser.add_argument("--markdown-report-path", default="reports/tca_select_ambiguity_stress_plan_report.md")
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
