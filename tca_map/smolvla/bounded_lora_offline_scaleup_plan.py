"""Planning-only bounded LoRA/offline proxy scale-up gate."""

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
    "max_samples": 64,
    "max_steps": 64,
    "max_runtime_minutes": 20,
    "max_vram_gb": 0,
    "max_cpu_ram_gb": 8,
    "lora_rank": 4,
}


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    limits = report.get("limits") or {}
    lines = [
        "# Bounded LoRA / Offline Proxy Scale-Up Plan Report",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- plan passed: `{report.get('bounded_lora_offline_scaleup_plan_passed')}`",
        f"- ready for runner: `{report.get('ready_for_bounded_lora_offline_scaleup_runner')}`",
        f"- ready for rollout scaling: `{report.get('ready_for_learned_policy_rollout_scaling')}`",
        f"- ready for paper claim: `{report.get('ready_for_paper_claim')}`",
        "",
        "Limits:",
        "",
    ]
    for key, value in limits.items():
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "Required future runner gates:", ""])
    for gate in report.get("required_future_gates") or []:
        lines.append(f"- `{gate}`")
    lines.extend(["", f"Recommended next step: {report.get('recommended_next_step')}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    evidence_path = Path(args.evidence_gap_report)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    limits = {
        "max_pairs": min(int(args.max_pairs), DEFAULT_LIMITS["max_pairs"]),
        "max_samples": min(int(args.max_samples), DEFAULT_LIMITS["max_samples"]),
        "max_steps": min(int(args.max_steps), DEFAULT_LIMITS["max_steps"]),
        "max_runtime_minutes": min(int(args.max_runtime_minutes), DEFAULT_LIMITS["max_runtime_minutes"]),
        "max_vram_gb": DEFAULT_LIMITS["max_vram_gb"],
        "max_cpu_ram_gb": DEFAULT_LIMITS["max_cpu_ram_gb"],
        "lora_rank": min(int(args.lora_rank), DEFAULT_LIMITS["lora_rank"]),
        "device": "cpu",
        "full_finetuning_allowed": False,
        "backbone_frozen": True,
    }
    report: dict[str, Any] = {
        "evidence_label": "bounded_lora_offline_scaleup_plan",
        "bounded_lora_offline_scaleup_plan_passed": False,
        "decision": "stop",
        "ready_for_bounded_lora_offline_scaleup_runner": False,
        "ready_for_offline_proxy_extension": False,
        "ready_for_learned_policy_rollout_scaling": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "limits": limits,
        "required_future_gates": [],
        "policy": {
            "planning_only": True,
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
        "paths": {"evidence_gap_report": str(evidence_path)},
        "source_summary": {},
        "stop_conditions": [
            "missing evidence gap report",
            "offline evidence table not passed",
            "future runner would exceed max_steps or runtime budget",
            "future runner requires GPU, heavy model import, rollout, OpenVLA-OFT, full fine-tuning, package install, token access, or paper claim",
        ],
        "recommended_next_step": None,
        "error": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["decision"] = "stop"
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        return report, code

    if forbidden:
        return block("Forbidden gate(s) set for planning-only scale-up plan: " + ", ".join(forbidden), 2)
    if not evidence_path.exists():
        return block(f"Offline evidence gap report is missing: {evidence_path}", 3)

    evidence = _read_json(evidence_path)
    if not evidence.get("offline_evidence_gap_report_passed"):
        return block("Offline evidence gap report did not pass.", 4)

    evidence_rows = evidence.get("evidence_table") or []
    lora_rows = [row for row in evidence_rows if "LoRA" in str(row.get("arm"))]
    report["source_summary"] = {
        "evidence_decision": evidence.get("decision"),
        "evidence_row_count": len(evidence_rows),
        "lora_row_count": len(lora_rows),
        "ready_for_lora_scaleup_plan": evidence.get("ready_for_lora_scaleup_plan"),
        "ready_for_offline_proxy_extension": evidence.get("ready_for_offline_proxy_extension"),
        "current_checkpoint_rollout_scaling_ready": evidence.get("ready_for_learned_policy_rollout_scaling"),
    }
    if not evidence.get("ready_for_lora_scaleup_plan"):
        return block("Evidence gap report did not authorize LoRA scale-up planning.", 5)

    report["bounded_lora_offline_scaleup_plan_passed"] = True
    report["decision"] = "proceed_bounded_offline_lora_scaleup_runner"
    report["ready_for_bounded_lora_offline_scaleup_runner"] = True
    report["ready_for_offline_proxy_extension"] = True
    report["required_future_gates"] = ["ALLOW_TINY_TRAINING=1"]
    report["recommended_next_step"] = (
        "Implement a separately gated CPU-only offline LoRA scale-up runner over real LIBERO HDF5 subsets. "
        "Use at most 16 pairs, 64 samples, 64 steps, LoRA rank 4, frozen base weights, no full fine-tuning, no rollout, no model load, no GPU job, no OpenVLA-OFT, and no paper claim."
    )
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-gap-report", default="reports/offline_tca_lora_evidence_gap_report_runtime.json")
    parser.add_argument("--max-pairs", type=int, default=DEFAULT_LIMITS["max_pairs"])
    parser.add_argument("--max-samples", type=int, default=DEFAULT_LIMITS["max_samples"])
    parser.add_argument("--max-steps", type=int, default=DEFAULT_LIMITS["max_steps"])
    parser.add_argument("--max-runtime-minutes", type=int, default=DEFAULT_LIMITS["max_runtime_minutes"])
    parser.add_argument("--lora-rank", type=int, default=DEFAULT_LIMITS["lora_rank"])
    parser.add_argument("--report-path", default="reports/bounded_lora_offline_scaleup_plan_report.json")
    parser.add_argument("--markdown-report-path", default="reports/bounded_lora_offline_scaleup_plan_report.md")
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
