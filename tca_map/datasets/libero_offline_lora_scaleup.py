"""Bounded CPU-only LIBERO offline LoRA scale-up runner.

This runner reuses the tiny NumPy LoRA machinery from the offline comparison
path. It reads local LIBERO HDF5 action snippets only; it does not load
SmolVLA, import heavy VLA models, use GPU, run simulators, run rollouts,
download assets, execute OpenVLA-OFT, or make paper claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from tca_map.adapters.tiny_lora_smoke import TinyLoraSmokeError
from tca_map.datasets.libero_offline_lora_comparison import run_libero_offline_lora_comparison


SCHEMA_VERSION = "tca-map-libero-offline-lora-scaleup-v0"
MAX_SCALEUP_PAIRS = 16
MAX_SCALEUP_ACTION_STEPS = 32
MAX_SCALEUP_SAMPLES = 64
MAX_SCALEUP_STEPS = 64
MAX_SCALEUP_RUNTIME_SECONDS = 15 * 60
MAX_PLANNED_RUNTIME_SECONDS = 20 * 60
MAX_SCALEUP_RANK = 4

FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
    "ALLOW_CLOUD_HANDOFF",
    "ALLOW_SIMULATOR_IMPORT_SMOKE",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_TINY_ROLLOUT",
]


def _forbidden_gates_set() -> list[str]:
    return [name for name in FORBIDDEN_GATES if os.environ.get(name)]


def validate_scaleup_bounds(
    max_pairs: int,
    max_action_steps: int,
    max_samples: int,
    max_steps: int,
    max_runtime_seconds: int,
    rank: int,
) -> None:
    if max_pairs < 1 or max_pairs > MAX_SCALEUP_PAIRS:
        raise TinyLoraSmokeError(f"max_pairs must be between 1 and {MAX_SCALEUP_PAIRS}")
    if max_action_steps < 1 or max_action_steps > MAX_SCALEUP_ACTION_STEPS:
        raise TinyLoraSmokeError(f"max_action_steps must be between 1 and {MAX_SCALEUP_ACTION_STEPS}")
    if max_samples < 1 or max_samples > MAX_SCALEUP_SAMPLES:
        raise TinyLoraSmokeError(f"max_samples must be between 1 and {MAX_SCALEUP_SAMPLES}")
    if max_steps < 1 or max_steps > MAX_SCALEUP_STEPS:
        raise TinyLoraSmokeError(f"max_steps must be between 1 and {MAX_SCALEUP_STEPS}")
    if max_runtime_seconds < 1 or max_runtime_seconds > MAX_SCALEUP_RUNTIME_SECONDS:
        raise TinyLoraSmokeError(f"max_runtime_seconds must be between 1 and {MAX_SCALEUP_RUNTIME_SECONDS}")
    if rank < 1 or rank > MAX_SCALEUP_RANK:
        raise TinyLoraSmokeError(f"rank must be between 1 and {MAX_SCALEUP_RANK}")


def _write_markdown(report: dict[str, Any], report_md: Path) -> None:
    comparison = report.get("comparison") or {}
    tca_vs_actionmap = comparison.get("tca_lora_vs_actionmap_lora") or {}
    select_vs_tca = comparison.get("tca_select_lora_vs_tca_lora") or {}
    lines = [
        "# Bounded LIBERO Offline LoRA Scale-Up Report",
        "",
        "This is an offline proxy diagnostic only. It is not standard success, not rollout success, and not paper-grade evidence.",
        "",
        f"- passed: `{report.get('bounded_lora_offline_scaleup_passed')}`",
        f"- record count: `{report.get('record_count')}`",
        f"- max pairs: `{report.get('max_pairs')}`",
        f"- max samples: `{report.get('max_samples')}`",
        f"- max steps: `{report.get('max_steps')}`",
        f"- LoRA rank: `{report.get('lora_rank')}`",
        f"- ready for offline evidence refresh: `{report.get('ready_for_offline_evidence_refresh')}`",
        f"- ready for rollout: `{report.get('ready_for_rollout')}`",
        f"- ready for paper claim: `{report.get('ready_for_paper_claim')}`",
        "",
        "## Key Deltas",
        "",
        f"- TCA-Map + LoRA vs ActionMap + LoRA action L1 delta: `{tca_vs_actionmap.get('action_l1_delta')}`",
        f"- TCA-Map + LoRA vs ActionMap + LoRA wrong-target delta: `{tca_vs_actionmap.get('wrong_target_proxy_rate_delta')}`",
        f"- TCA-Select + LoRA vs TCA-Map + LoRA action L1 delta: `{select_vs_tca.get('action_l1_delta')}`",
        f"- TCA-Select + LoRA vs TCA-Map + LoRA wrong-target delta: `{select_vs_tca.get('wrong_target_proxy_rate_delta')}`",
        "",
        "## Next Step",
        "",
        str(report.get("recommended_next_step")),
        "",
    ]
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_md.write_text("\n".join(lines), encoding="utf-8")


def write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, report_md)


def run_bounded_libero_offline_lora_scaleup(
    manifest_path: Path,
    report_json: Path,
    report_md: Path,
    max_pairs: int = MAX_SCALEUP_PAIRS,
    max_action_steps: int = 16,
    max_samples: int = MAX_SCALEUP_SAMPLES,
    max_steps: int = MAX_SCALEUP_STEPS,
    max_runtime_seconds: int = MAX_SCALEUP_RUNTIME_SECONDS,
    rank: int = MAX_SCALEUP_RANK,
    require_training_gate: bool = True,
) -> dict[str, Any]:
    forbidden = _forbidden_gates_set()
    if forbidden:
        raise TinyLoraSmokeError("dangerous gates are set: " + ", ".join(forbidden))
    if require_training_gate and os.environ.get("ALLOW_TINY_TRAINING") != "1":
        raise TinyLoraSmokeError("ALLOW_TINY_TRAINING=1 is required for bounded LIBERO offline LoRA scale-up")
    validate_scaleup_bounds(
        max_pairs=max_pairs,
        max_action_steps=max_action_steps,
        max_samples=max_samples,
        max_steps=max_steps,
        max_runtime_seconds=max_runtime_seconds,
        rank=rank,
    )

    report = run_libero_offline_lora_comparison(
        manifest_path=manifest_path,
        report_json=report_json,
        report_md=report_md,
        max_pairs=max_pairs,
        max_action_steps=max_action_steps,
        max_steps=max_steps,
        max_runtime_seconds=max_runtime_seconds,
        max_samples=max_samples,
        rank=rank,
        require_training_gate=require_training_gate,
    )
    passed = bool(
        report.get("libero_offline_lora_comparison_passed")
        and report.get("runtime_within_cap")
        and int(report.get("record_count", 0)) <= MAX_SCALEUP_SAMPLES
        and int(report.get("max_steps", 0)) <= MAX_SCALEUP_STEPS
    )
    policy = dict(report.get("policy") or {})
    policy.update(
        {
            "bounded_lora_offline_scaleup": True,
            "task_local_training_gate_required": require_training_gate,
            "task_local_training_gate_used": require_training_gate,
            "device": "cpu",
            "full_finetuning_allowed": False,
            "smolvla_loaded": False,
            "openvla_oft_downloaded": False,
            "libero_dataset_downloaded": False,
            "standard_success_claimed": False,
        }
    )
    report.update(
        {
            "schema_version": SCHEMA_VERSION,
            "policy": policy,
            "scaleup_limits": {
                "max_pairs_cap": MAX_SCALEUP_PAIRS,
                "max_action_steps_cap": MAX_SCALEUP_ACTION_STEPS,
                "max_samples_cap": MAX_SCALEUP_SAMPLES,
                "max_steps_cap": MAX_SCALEUP_STEPS,
                "max_runtime_seconds_cap": MAX_SCALEUP_RUNTIME_SECONDS,
                "planning_budget_seconds": MAX_PLANNED_RUNTIME_SECONDS,
                "lora_rank_cap": MAX_SCALEUP_RANK,
                "device": "cpu",
            },
            "bounded_lora_offline_scaleup_passed": passed,
            "ready_for_offline_evidence_refresh": passed,
            "ready_for_bounded_local_pilot_report": passed,
            "ready_for_rollout": False,
            "ready_for_learned_policy_rollout_scaling": False,
            "ready_for_benchmark_claim": False,
            "ready_for_paper_claim": False,
            "interpretation": (
                "Offline proxy diagnostic only. This scale-up trains tiny NumPy low-rank matrices over local "
                "LIBERO HDF5 action-prefix snippets. It is not standard success, not rollout success, not a "
                "SmolVLA model-load result, and not paper-grade evidence."
            ),
            "recommended_next_step": (
                "Refresh the offline TCA-Map/LoRA evidence table to include this bounded scale-up, while keeping "
                "current-checkpoint learned-policy rollout scaling and paper claims blocked."
                if passed
                else "Fix the bounded offline LoRA scale-up before refreshing evidence tables."
            ),
        }
    )
    write_reports(report, report_json=report_json, report_md=report_md)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_report.json")
    parser.add_argument("--report-json", default="reports/bounded_lora_offline_scaleup_report.json")
    parser.add_argument("--report-md", default="reports/bounded_lora_offline_scaleup_report.md")
    parser.add_argument("--max-pairs", type=int, default=MAX_SCALEUP_PAIRS)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=MAX_SCALEUP_STEPS)
    parser.add_argument("--max-runtime-seconds", type=int, default=MAX_SCALEUP_RUNTIME_SECONDS)
    parser.add_argument("--max-samples", type=int, default=MAX_SCALEUP_SAMPLES)
    parser.add_argument("--rank", type=int, default=MAX_SCALEUP_RANK)
    args = parser.parse_args(argv)

    try:
        report = run_bounded_libero_offline_lora_scaleup(
            manifest_path=Path(args.manifest),
            report_json=Path(args.report_json),
            report_md=Path(args.report_md),
            max_pairs=args.max_pairs,
            max_action_steps=args.max_action_steps,
            max_steps=args.max_steps,
            max_runtime_seconds=args.max_runtime_seconds,
            max_samples=args.max_samples,
            rank=args.rank,
            require_training_gate=True,
        )
    except TinyLoraSmokeError as exc:
        raise SystemExit(str(exc))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
