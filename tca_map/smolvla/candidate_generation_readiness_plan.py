"""Report-only learned-policy candidate-generation readiness plan.

This module plans the next bounded interface step after offline TCA-Select
ambiguity evidence. It reads existing reports only; it does not import heavy VLA
models, load SmolVLA, run inference, train, rollout, use GPU jobs, download
assets, execute simulators, execute OpenVLA-OFT, or make paper claims.
"""

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


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _nested_bool(payload: dict[str, Any], *keys: str) -> bool:
    current: Any = payload
    for key in keys:
        if not isinstance(current, dict):
            return False
        current = current.get(key)
    return bool(current)


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Learned-Policy Candidate-Generation Readiness Plan Report",
        "",
        "This is report-only planning. It does not load models, infer, train, rollout, use GPU jobs, or make paper claims.",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- passed: `{report.get('candidate_generation_readiness_plan_passed')}`",
        f"- ready for contract checker: `{report.get('ready_for_candidate_generation_contract_checker')}`",
        f"- ready for real candidate-generation smoke execution: `{report.get('ready_for_real_candidate_generation_smoke_execution')}`",
        f"- ready for paper claim: `{report.get('ready_for_paper_claim')}`",
        "",
        "## Required Future Gates",
        "",
    ]
    for item in report.get("required_future_gates") or []:
        lines.append(f"- `{item}`")
    lines.extend(["", "## Next Step", "", str(report.get("recommended_next_step")), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    synthesis_path = Path(args.synthesis_report)
    load_path = Path(args.load_only_report)
    single_sample_path = Path(args.single_sample_report)
    feature_cache_path = Path(args.feature_cache_report)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]

    report: dict[str, Any] = {
        "evidence_label": "learned_policy_candidate_generation_readiness_plan",
        "candidate_generation_readiness_plan_passed": False,
        "decision": "stop",
        "ready_for_candidate_generation_contract_checker": False,
        "ready_for_real_candidate_generation_smoke_plan": False,
        "ready_for_real_candidate_generation_smoke_execution": False,
        "ready_for_learned_policy_rollout_scaling": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "policy": {
            "report_only": True,
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
        "paths": {
            "synthesis_report": str(synthesis_path),
            "load_only_report": str(load_path),
            "single_sample_report": str(single_sample_path),
            "feature_cache_report": str(feature_cache_path),
        },
        "input_summary": {},
        "planned_contract": {},
        "required_future_gates": [],
        "candidate_generation_blockers": [],
        "recommended_next_step": None,
        "error": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        return report, code

    if forbidden:
        return block("Forbidden gate(s) set for planning-only candidate-generation readiness: " + ", ".join(forbidden), 2)
    if not synthesis_path.exists():
        return block(f"Stress-aware attribution synthesis report is missing: {synthesis_path}", 3)

    synthesis = _read_json_if_exists(synthesis_path)
    load_only = _read_json_if_exists(load_path)
    single_sample = _read_json_if_exists(single_sample_path)
    feature_cache = _read_json_if_exists(feature_cache_path)

    if not synthesis.get("scaleup_attribution_gap_synthesis_passed"):
        return block("Stress-aware attribution synthesis did not pass.", 4)
    if synthesis.get("ready_for_paper_claim"):
        return block("Unexpected paper-claim readiness in synthesis; refusing candidate-generation plan.", 5)

    stress_included = bool(synthesis.get("tca_select_ambiguity_stress_included"))
    evidence_rows = int((synthesis.get("input_summary") or {}).get("evidence_row_count") or 0)
    load_only_passed = _nested_bool(load_only, "result", "passed")
    single_sample_passed = _nested_bool(single_sample, "result", "passed")
    feature_cache_valid = bool(feature_cache.get("cache_valid") or feature_cache.get("feature_cache_eval_smoke_passed"))
    report["input_summary"] = {
        "stress_aware_synthesis_passed": True,
        "tca_select_ambiguity_stress_included": stress_included,
        "evidence_row_count": evidence_rows,
        "prior_smolvla_load_only_passed": load_only_passed,
        "prior_single_sample_interface_passed": single_sample_passed,
        "prior_feature_cache_eval_valid": feature_cache_valid,
        "synthesis_recommended_next_step": synthesis.get("recommended_next_step"),
    }
    report["planned_contract"] = {
        "purpose": "define a future bounded path from learned policy outputs to low-resolution action heatmap candidates",
        "future_inputs": [
            "one local SmolVLA-compatible checkpoint",
            "one local tokenizer/processor dependency",
            "one synthetic or local offline observation sample",
            "one target text instruction",
            "TCA-Map action-head projection or adapter output contract",
        ],
        "future_outputs": [
            "continuous action candidate list",
            "low-resolution action heatmap",
            "masked or counterfactual target-conditioned heatmap for sensitivity scoring",
            "metadata proving no privileged simulator state or external verifier was used",
        ],
        "limits_for_future_smoke": {
            "candidate_count_max": 8,
            "heatmap_grid_max": 8,
            "batch_size": 1,
            "max_runtime_minutes": 10,
            "max_vram_mb": 14336,
            "rollouts_allowed": False,
            "training_allowed": False,
            "openvla_oft_allowed": False,
        },
        "forbidden_future_inputs": [
            "rollout outcome",
            "privileged simulator state",
            "external verifier model",
            "paper-grade success labels",
            "OpenVLA-OFT execution",
        ],
    }
    report["required_future_gates"] = [
        "separate candidate-generation contract checker with synthetic tensors only",
        "separate risk assessment before any heavy import or model inference",
        "ALLOW_HEAVY_IMPORT=1 only inside a future bounded load/inference task",
        "ALLOW_SINGLE_SAMPLE_INFERENCE=1 only inside a future bounded inference task",
        "no rollout until a separate learned-policy rollout risk gate passes",
    ]

    blockers = []
    if not stress_included:
        blockers.append("TCA-Select ambiguity stress evidence is missing from the synthesis report.")
    if not load_only_passed:
        blockers.append("Prior SmolVLA load-only report is missing or did not pass.")
    if not single_sample_passed:
        blockers.append("Prior single-sample interface report is missing or did not pass.")
    if not feature_cache_valid:
        blockers.append("Prior feature-cache eval report is missing or invalid.")
    report["candidate_generation_blockers"] = blockers

    report["candidate_generation_readiness_plan_passed"] = True
    report["decision"] = "candidate_generation_readiness_plan_ready"
    report["ready_for_candidate_generation_contract_checker"] = stress_included
    report["ready_for_real_candidate_generation_smoke_plan"] = stress_included and load_only_passed
    report["ready_for_real_candidate_generation_smoke_execution"] = False
    report["recommended_next_step"] = (
        "Implement a report-only or synthetic-tensor candidate-generation contract checker before any model inference."
        if stress_included
        else "Repair stress-aware synthesis before planning learned-policy candidate generation."
    )
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--synthesis-report", default="reports/scaleup_attribution_gap_synthesis_report.json")
    parser.add_argument("--load-only-report", default="reports/smolvla_load_only_smoke_report.json")
    parser.add_argument("--single-sample-report", default="reports/smolvla_single_sample_interface_report.json")
    parser.add_argument("--feature-cache-report", default="reports/feature_cache_eval_report.json")
    parser.add_argument("--report-path", default="reports/candidate_generation_readiness_plan_report.json")
    parser.add_argument("--markdown-report-path", default="reports/candidate_generation_readiness_plan_report.md")
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
