"""Report-only summary for bounded real candidate-generation smoke."""

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
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_REAL_CANDIDATE_GENERATION_SMOKE",
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


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    metrics = report.get("metrics") or {}
    lines = [
        "# Real Candidate-Generation Smoke Summary Report",
        "",
        "This is report-only synthesis of an engineering smoke. It is not standard success, rollout success, or paper-grade evidence.",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- summary passed: `{report.get('real_candidate_generation_smoke_summary_passed')}`",
        f"- smoke passed: `{metrics.get('smoke_passed')}`",
        f"- elapsed seconds: `{metrics.get('elapsed_sec')}`",
        f"- single-sample inference seconds: `{metrics.get('single_sample_inference_elapsed_sec')}`",
        f"- selection latency ms: `{metrics.get('selection_latency_ms')}`",
        f"- candidate count: `{metrics.get('candidate_count')}`",
        f"- heatmap grid: `{metrics.get('heatmap_grid')}`",
        f"- selected candidate index: `{metrics.get('selected_candidate_index')}`",
        f"- selected target index: `{metrics.get('selected_target_index')}`",
        f"- wrong-target proxy: `{metrics.get('wrong_target_proxy')}`",
        f"- CUDA max allocated MB: `{metrics.get('cuda_max_allocated_mb')}`",
        f"- ready for paper claim: `{report.get('ready_for_paper_claim')}`",
        "",
        "## Next Step",
        "",
        str(report.get("recommended_next_step")),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    smoke_path = Path(args.smoke_report)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "evidence_label": "real_candidate_generation_smoke_summary",
        "real_candidate_generation_smoke_summary_passed": False,
        "decision": "stop",
        "policy": {
            "summary_only": True,
            "downloads_performed": False,
            "installs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "candidate_generation_performed": False,
            "training_performed": False,
            "rollouts_performed": False,
            "simulator_environment_created": False,
            "gpu_jobs_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
            "forbidden_gates_set": forbidden,
        },
        "claims": {
            "standard_success_claimed": False,
            "rollout_success_claimed": False,
            "benchmark_success_claimed": False,
            "sota_claimed": False,
            "paper_grade_claim_made": False,
        },
        "paths": {"smoke_report": str(smoke_path)},
        "metrics": {},
        "gaps": [],
        "ready_for_candidate_generation_comparison_plan": False,
        "ready_for_rollout": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "recommended_next_step": None,
        "error": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        return report, code

    if forbidden:
        return block("Forbidden gate(s) set for summary-only task: " + ", ".join(forbidden), 2)
    if not smoke_path.exists():
        return block(f"Real candidate-generation smoke report is missing: {smoke_path}", 3)

    smoke = _read_json(smoke_path)
    policy = smoke.get("policy") or {}
    result = smoke.get("result") or {}
    runtime = smoke.get("runtime") or {}
    generation = smoke.get("generation") or {}
    selection = smoke.get("selection") or {}
    smoke_passed = bool(smoke.get("real_candidate_generation_smoke_passed") and result.get("passed"))
    forbidden_behavior = [
        name
        for name in [
            "downloads_performed",
            "training_performed",
            "rollouts_performed",
            "simulator_environment_created",
            "openvla_oft_executed",
            "tokens_read_or_written",
            "paper_grade_claims_made",
            "external_verifier_used",
            "privileged_inference_used",
        ]
        if policy.get(name)
    ]
    gaps: list[str] = []
    if not smoke_passed:
        gaps.append("The bounded real candidate-generation smoke has not passed.")
    if forbidden_behavior:
        gaps.append("Forbidden behavior appeared in smoke policy: " + ", ".join(forbidden_behavior))
    if generation.get("candidate_count") != 4:
        gaps.append("Candidate count differs from the planned K=4 default.")
    if generation.get("heatmap_grid") != 8:
        gaps.append("Heatmap grid differs from the planned low-resolution grid 8 default.")
    if selection.get("wrong_target_proxy"):
        gaps.append("TCA-Select selected a wrong-target candidate in the smoke.")
    if policy.get("model_inference_performed") and not policy.get("bounded_single_sample_only"):
        gaps.append("Model inference was not marked as bounded single-sample only.")

    report["metrics"] = {
        "smoke_passed": smoke_passed,
        "elapsed_sec": result.get("elapsed_sec"),
        "device": runtime.get("device"),
        "single_sample_inference_elapsed_sec": runtime.get("single_sample_inference_elapsed_sec"),
        "selection_latency_ms": runtime.get("selection_latency_ms"),
        "cuda_max_allocated_mb": runtime.get("cuda_max_allocated_mb"),
        "candidate_count": generation.get("candidate_count"),
        "heatmap_grid": generation.get("heatmap_grid"),
        "action_dim": generation.get("action_dim"),
        "selected_candidate_index": selection.get("selected_candidate_index"),
        "selected_target_index": selection.get("selected_target_index"),
        "selected_action_l1_to_seed": selection.get("selected_action_l1_to_seed"),
        "wrong_target_proxy": selection.get("wrong_target_proxy"),
        "model_inference_performed_in_source_smoke": bool(policy.get("model_inference_performed")),
        "candidate_generation_performed_in_source_smoke": bool(policy.get("candidate_generation_performed")),
        "forbidden_behavior_in_source_smoke": forbidden_behavior,
    }
    report["gaps"] = gaps
    report["real_candidate_generation_smoke_summary_passed"] = True
    report["decision"] = "real_candidate_generation_smoke_engineering_evidence_ready" if not gaps else "reduce_scope"
    report["ready_for_candidate_generation_comparison_plan"] = bool(smoke_passed and not gaps)
    report["recommended_next_step"] = (
        "Plan a bounded offline candidate-generation comparison that contrasts learned seed candidates with existing offline TCA-Select stress proxies, still without rollout or paper claims."
        if report["ready_for_candidate_generation_comparison_plan"]
        else "Resolve the listed smoke gaps before using this as candidate-generation engineering evidence."
    )
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-report", default="reports/real_candidate_generation_smoke_report.json")
    parser.add_argument("--report-path", default="reports/real_candidate_generation_smoke_summary_report.json")
    parser.add_argument("--markdown-report-path", default="reports/real_candidate_generation_smoke_summary_report.md")
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
