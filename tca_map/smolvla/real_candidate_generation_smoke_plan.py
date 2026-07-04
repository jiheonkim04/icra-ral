"""Planning-only risk gate for real candidate-generation smoke.

This module decides whether a future bounded real candidate-generation smoke may
be implemented. It reads existing reports only. It does not import heavy VLA
models, load SmolVLA, run model inference, train, rollout, use GPU jobs, execute
simulators, download assets, execute OpenVLA-OFT, or make paper claims.
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
MAX_VRAM_MB = 14336
MAX_RUNTIME_MINUTES = 10
MAX_CANDIDATES = 4
MAX_HEATMAP_GRID = 8


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
        "# Real Candidate-Generation Smoke Plan Report",
        "",
        "This is planning-only. It does not load models, infer, train, rollout, use GPU jobs, or make paper claims.",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- passed: `{report.get('real_candidate_generation_smoke_plan_passed')}`",
        f"- ready for future smoke implementation: `{report.get('ready_for_real_candidate_generation_smoke_implementation')}`",
        f"- ready for future smoke execution: `{report.get('ready_for_real_candidate_generation_smoke_execution')}`",
        f"- ready for paper claim: `{report.get('ready_for_paper_claim')}`",
        "",
        "## Required Future Gates",
        "",
    ]
    for gate in report.get("required_future_gates") or []:
        lines.append(f"- `{gate}`")
    lines.extend(["", "## Next Step", "", str(report.get("recommended_next_step")), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    contract_path = Path(args.contract_report)
    runtime_path = Path(args.runtime_deps_report)
    load_path = Path(args.load_only_report)
    single_path = Path(args.single_sample_report)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "evidence_label": "real_candidate_generation_smoke_plan",
        "real_candidate_generation_smoke_plan_passed": False,
        "decision": "stop",
        "ready_for_real_candidate_generation_smoke_implementation": False,
        "ready_for_real_candidate_generation_smoke_execution": False,
        "ready_for_rollout": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "policy": {
            "planning_only": True,
            "report_only": True,
            "downloads_performed": False,
            "installs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "training_performed": False,
            "rollouts_performed": False,
            "simulator_environment_created": False,
            "gpu_jobs_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
            "forbidden_gates_set": forbidden,
        },
        "paths": {
            "contract_report": str(contract_path),
            "runtime_deps_report": str(runtime_path),
            "load_only_report": str(load_path),
            "single_sample_report": str(single_path),
        },
        "input_summary": {},
        "risk_assessment": {},
        "required_future_gates": [],
        "implementation_constraints": {},
        "blockers": [],
        "recommended_next_step": None,
        "error": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        return report, code

    if forbidden:
        return block("Forbidden gate(s) set for planning-only real candidate-generation smoke plan: " + ", ".join(forbidden), 2)
    if not contract_path.exists():
        return block(f"Candidate-generation contract report is missing: {contract_path}", 3)

    contract = _read_json_if_exists(contract_path)
    runtime = _read_json_if_exists(runtime_path)
    load = _read_json_if_exists(load_path)
    single = _read_json_if_exists(single_path)

    contract_passed = bool(contract.get("candidate_generation_contract_check_passed"))
    runtime_ready = _nested_bool(runtime, "runtime_dependencies", "ready_for_load_only_runtime")
    gpu_total_mb = ((runtime.get("gpu") or {}).get("memory_total_mb"))
    load_passed = _nested_bool(load, "result", "passed")
    single_passed = _nested_bool(single, "result", "passed")
    prior_vram_mb = ((single.get("interface") or {}).get("cuda_max_allocated_mb") or 0)
    report["input_summary"] = {
        "contract_passed": contract_passed,
        "runtime_ready": runtime_ready,
        "gpu_total_mb": gpu_total_mb,
        "prior_load_only_passed": load_passed,
        "prior_single_sample_passed": single_passed,
        "prior_single_sample_vram_mb": prior_vram_mb,
        "contract_ready_for_real_smoke_plan": bool(contract.get("ready_for_real_candidate_generation_smoke_plan")),
    }
    blockers = []
    if not contract_passed:
        blockers.append("Candidate-generation synthetic contract check did not pass.")
    if not runtime_ready:
        blockers.append("SmolVLA runtime dependency report is missing or not ready.")
    if not load_passed:
        blockers.append("Prior SmolVLA load-only smoke is missing or did not pass.")
    if not single_passed:
        blockers.append("Prior single-sample interface smoke is missing or did not pass.")
    if prior_vram_mb and prior_vram_mb > MAX_VRAM_MB:
        blockers.append(f"Prior single-sample VRAM {prior_vram_mb} MB exceeds {MAX_VRAM_MB} MB.")
    report["blockers"] = blockers
    green = not blockers
    report["risk_assessment"] = {
        "task": "future bounded real candidate-generation smoke",
        "expected_runtime_minutes": f"<= {MAX_RUNTIME_MINUTES}",
        "expected_vram_mb": f"<= {MAX_VRAM_MB}",
        "candidate_count": MAX_CANDIDATES,
        "heatmap_grid": MAX_HEATMAP_GRID,
        "device_policy": "CPU first; CUDA only if separately selected by future smoke and VRAM remains <= 14GB",
        "downloads_required": False,
        "training_required": False,
        "rollout_required": False,
        "simulator_required": False,
        "openvla_oft_required": False,
        "token_or_secret_required": False,
        "decision": "green_for_implementation_plan" if green else "blocked",
    }
    report["required_future_gates"] = [
        "ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1",
        "ALLOW_HEAVY_IMPORT=1",
        "ALLOW_SINGLE_SAMPLE_INFERENCE=1",
    ]
    report["implementation_constraints"] = {
        "single_sample_only": True,
        "candidate_count_max": MAX_CANDIDATES,
        "heatmap_grid_max": MAX_HEATMAP_GRID,
        "max_runtime_minutes": MAX_RUNTIME_MINUTES,
        "max_vram_mb": MAX_VRAM_MB,
        "batch_size": 1,
        "training_allowed": False,
        "rollout_allowed": False,
        "simulator_allowed": False,
        "openvla_oft_allowed": False,
        "external_verifier_allowed": False,
        "privileged_state_allowed": False,
        "paper_claim_allowed": False,
    }
    report["real_candidate_generation_smoke_plan_passed"] = True
    report["decision"] = "proceed_bounded_real_candidate_generation_smoke_implementation" if green else "blocked"
    report["ready_for_real_candidate_generation_smoke_implementation"] = green
    report["ready_for_real_candidate_generation_smoke_execution"] = False
    report["recommended_next_step"] = (
        "Implement a separately gated bounded real candidate-generation smoke; do not execute it in the implementation branch until its own gates are set task-locally."
        if green
        else "Resolve blockers before implementing real candidate-generation smoke."
    )
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract-report", default="reports/candidate_generation_contract_check_report.json")
    parser.add_argument("--runtime-deps-report", default="reports/smolvla_runtime_deps_report.json")
    parser.add_argument("--load-only-report", default="reports/smolvla_load_only_smoke_report.json")
    parser.add_argument("--single-sample-report", default="reports/smolvla_single_sample_interface_report.json")
    parser.add_argument("--report-path", default="reports/real_candidate_generation_smoke_plan_report.json")
    parser.add_argument("--markdown-report-path", default="reports/real_candidate_generation_smoke_plan_report.md")
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
