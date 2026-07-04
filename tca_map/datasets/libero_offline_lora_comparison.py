"""Tiny offline LoRA comparison over local LIBERO HDF5 action snippets.

This is a bounded local pilot diagnostic. It reads a tiny counterfactual
manifest, converts local HDF5 action snippets into lightweight feature records,
and trains only tiny NumPy LoRA adapter matrices. It does not load SmolVLA,
import heavy VLA models, use GPU, run simulators, run rollouts, or make paper
claims.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from tca_map.adapters.tiny_lora_smoke import (
    DEFAULT_LORA_RANK,
    DEFAULT_MAX_RUNTIME_SECONDS,
    DEFAULT_MAX_STEPS,
    TinyLoraSmokeError,
    _arm_report,
    ensure_safe_environment,
    validate_smoke_bounds,
)

SCHEMA_VERSION = "tca-map-libero-offline-lora-comparison-v0"
ACTION_PREFIX_DIM = 4


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing input manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _read_first_action_block(path: Path, max_steps: int = 16) -> list[list[float]]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        data_group = handle.get("data")
        if data_group is None:
            raise ValueError(f"{path} has no data group")
        for demo_name in sorted(data_group.keys()):
            demo = data_group[demo_name]
            if "actions" not in demo:
                continue
            actions = demo["actions"][:max_steps]
            return [[float(value) for value in row.tolist()] for row in actions]
    raise ValueError(f"{path} has no demo actions dataset")


def _mean_action(actions: list[list[float]]) -> list[float]:
    if not actions:
        return []
    width = len(actions[0])
    return [sum(row[index] for row in actions) / len(actions) for index in range(width)]


def _text_features(text: str, width: int = 16) -> list[float]:
    words = [word for word in text.lower().replace("_", " ").split() if word]
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    hashed = [((digest[index] / 255.0) * 2.0) - 1.0 for index in range(width - 4)]
    scalar = [
        min(len(text), 240) / 240.0,
        min(len(words), 40) / 40.0,
        sum(char in "aeiou" for char in text.lower()) / max(1, len(text)),
        sum(char.isdigit() for char in text) / max(1, len(text)),
    ]
    return scalar + hashed


def _record(sample_id: str, instruction: str, action: list[float], target_id: int, candidate_objects: list[str]) -> dict:
    return {
        "sample_id": sample_id,
        "hidden_tokens": _text_features(instruction),
        "expert_action": action[:ACTION_PREFIX_DIM],
        "target": {"object_id": target_id, "instruction": instruction},
        "candidate_objects": candidate_objects,
    }


def build_libero_lora_records(manifest_path: Path, max_pairs: int = 4, max_action_steps: int = 16) -> list[dict]:
    manifest = _load_json(manifest_path)
    if not manifest.get("ready_for_tiny_offline_counterfactual_split"):
        raise ValueError("counterfactual split manifest is not ready")

    records: list[dict[str, Any]] = []
    for pair in manifest.get("counterfactual_pairs", [])[:max_pairs]:
        positive_instruction = pair.get("positive_instruction") or "positive target"
        counter_instruction = pair.get("counterfactual_instruction") or "counterfactual target"
        candidates = [positive_instruction, counter_instruction]

        positive_action = _mean_action(
            _read_first_action_block(Path(pair["positive_demo_file"]), max_steps=max_action_steps)
        )
        counter_action = _mean_action(
            _read_first_action_block(Path(pair["counterfactual_demo_file"]), max_steps=max_action_steps)
        )

        records.append(
            _record(
                sample_id=f"{pair['pair_id']}::positive",
                instruction=positive_instruction,
                action=positive_action,
                target_id=0,
                candidate_objects=candidates,
            )
        )
        records.append(
            _record(
                sample_id=f"{pair['pair_id']}::counterfactual",
                instruction=counter_instruction,
                action=counter_action,
                target_id=1,
                candidate_objects=candidates,
            )
        )
    return records


def _policy(training_performed: bool) -> dict:
    return {
        "bounded_tiny_lora_smoke": True,
        "risk_assessed_autonomy_for_tiny_training_smoke": True,
        "local_libero_hdf5_used": True,
        "real_dataset_used": True,
        "offline_proxy_only": True,
        "not_standard_success": True,
        "not_paper_grade": True,
        "backbone_frozen": True,
        "trainable_lora_adapter_weights_only": True,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "gpu_training_performed": False,
        "heavy_model_imports_performed": False,
        "adapter_construction_performed": training_performed,
        "model_load_performed": False,
        "model_inference_performed": False,
        "training_performed": training_performed,
        "rollouts_performed": False,
        "simulator_executed": False,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
        "paper_grade_claims_made": False,
    }


def _comparison(arms: dict[str, dict]) -> dict:
    def metric(arm_name: str, metric_name: str) -> float | None:
        value = arms[arm_name].get("metrics", {}).get(metric_name)
        return None if value is None else float(value)

    def delta(left: str, right: str, metric_name: str) -> float | None:
        left_value = metric(left, metric_name)
        right_value = metric(right, metric_name)
        if left_value is None or right_value is None:
            return None
        return round(left_value - right_value, 6)

    return {
        "tca_lora_vs_actionmap_lora": {
            "offline_standard_proxy_delta": delta("tca_map_lora", "actionmap_lora", "offline_standard_proxy"),
            "action_l1_delta": delta("tca_map_lora", "actionmap_lora", "action_l1"),
            "wrong_target_proxy_rate_delta": delta("tca_map_lora", "actionmap_lora", "wrong_target_proxy_rate"),
            "counterfactual_margin_delta": delta(
                "tca_map_lora", "actionmap_lora", "counterfactual_separation_margin"
            ),
            "trainable_lora_parameter_delta": int(
                arms["tca_map_lora"].get("trainable_lora_parameter_count", 0)
                - arms["actionmap_lora"].get("trainable_lora_parameter_count", 0)
            ),
        },
        "tca_select_lora_vs_tca_lora": {
            "offline_standard_proxy_delta": delta(
                "tca_map_lora_distributional_select", "tca_map_lora", "offline_standard_proxy"
            ),
            "action_l1_delta": delta("tca_map_lora_distributional_select", "tca_map_lora", "action_l1"),
            "wrong_target_proxy_rate_delta": delta(
                "tca_map_lora_distributional_select", "tca_map_lora", "wrong_target_proxy_rate"
            ),
            "counterfactual_margin_delta": delta(
                "tca_map_lora_distributional_select", "tca_map_lora", "counterfactual_separation_margin"
            ),
        },
    }


def run_libero_offline_lora_comparison(
    manifest_path: Path,
    report_json: Path,
    report_md: Path,
    max_pairs: int = 4,
    max_action_steps: int = 16,
    max_steps: int = DEFAULT_MAX_STEPS,
    max_runtime_seconds: int = DEFAULT_MAX_RUNTIME_SECONDS,
    max_samples: int = 16,
    rank: int = DEFAULT_LORA_RANK,
    require_training_gate: bool = True,
) -> dict:
    ensure_safe_environment(require_training_gate=require_training_gate)
    validate_smoke_bounds(max_steps=max_steps, max_runtime_seconds=max_runtime_seconds, max_samples=max_samples, rank=rank)

    started = time.perf_counter()
    records = build_libero_lora_records(manifest_path, max_pairs=max_pairs, max_action_steps=max_action_steps)[:max_samples]
    if not records:
        raise TinyLoraSmokeError("no LIBERO HDF5 records were built for tiny LoRA comparison")

    arm_names = ["actionmap_lora", "tca_map_lora", "tca_map_lora_distributional_select"]
    arm_reports = []
    for arm_name in arm_names:
        if time.perf_counter() - started > max_runtime_seconds:
            raise TinyLoraSmokeError("LIBERO offline LoRA comparison exceeded max_runtime_seconds")
        arm_reports.append(
            _arm_report(
                records=records,
                arm_name=arm_name,
                max_steps=max_steps,
                lr=0.05,
                rank=rank,
                grid_size=8,
            )
        )

    total_elapsed = time.perf_counter() - started
    arms = {arm["arm"]: arm for arm in arm_reports}
    passed = bool(
        total_elapsed <= max_runtime_seconds
        and max_steps <= 100
        and len(records) <= 200
        and all(arm.get("finite_losses") for arm in arm_reports)
    )

    report = {
        "schema_version": SCHEMA_VERSION,
        "policy": _policy(training_performed=True),
        "source_manifest": str(manifest_path),
        "max_pairs": max_pairs,
        "max_action_steps": max_action_steps,
        "max_samples": max_samples,
        "max_steps": max_steps,
        "max_runtime_seconds": max_runtime_seconds,
        "lora_rank": rank,
        "action_prefix_dim": ACTION_PREFIX_DIM,
        "record_count": len(records),
        "elapsed_seconds": round(total_elapsed, 6),
        "runtime_within_cap": total_elapsed <= max_runtime_seconds,
        "arms": arm_reports,
        "comparison": _comparison(arms),
        "libero_offline_lora_comparison_passed": passed,
        "ready_for_bounded_local_pilot_report": passed,
        "ready_for_rollout": False,
        "interpretation": (
            "Offline proxy diagnostic only. The LoRA arms train tiny NumPy low-rank matrices on local LIBERO HDF5 "
            "action-prefix snippets. This is not standard success, not rollout success, not a full SmolVLA adapter "
            "result, and not paper-grade evidence."
        ),
        "recommended_next_step": (
            "Generate a bounded local pilot report that includes LIBERO offline head-only and required LoRA proxy gates; "
            "stop before simulator execution, rollout, heavy VLA imports, OpenVLA-OFT, or paper claims."
            if passed
            else "Fix the LIBERO offline LoRA comparison before local pilot reporting."
        ),
    }
    write_reports(report, report_json=report_json, report_md=report_md)
    return report


def write_reports(report: dict, report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    lines = [
        "# LIBERO Offline LoRA Comparison",
        "",
        "This is an offline proxy diagnostic only. It is not standard success, not rollout success, and not paper-grade evidence.",
        "",
        f"- passed: `{report['libero_offline_lora_comparison_passed']}`",
        f"- record count: `{report['record_count']}`",
        f"- max steps: `{report['max_steps']}`",
        f"- ready for bounded local pilot report: `{report['ready_for_bounded_local_pilot_report']}`",
        f"- ready for rollout: `{report['ready_for_rollout']}`",
        "",
        "## Arms",
    ]
    for arm in report["arms"]:
        metrics = arm.get("metrics", {})
        lines.extend(
            [
                f"- `{arm['arm']}` action L1: `{metrics.get('action_l1')}`",
                f"- `{arm['arm']}` wrong-target proxy: `{metrics.get('wrong_target_proxy_rate')}`",
                f"- `{arm['arm']}` trainable LoRA params: `{arm.get('trainable_lora_parameter_count')}`",
            ]
        )
    lines.extend(["", "## Next Step", report["recommended_next_step"], ""])
    report_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_report.json")
    parser.add_argument("--report-json", default="reports/libero_offline_lora_comparison_report.json")
    parser.add_argument("--report-md", default="reports/libero_offline_lora_comparison_report.md")
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-runtime-seconds", type=int, default=DEFAULT_MAX_RUNTIME_SECONDS)
    parser.add_argument("--max-samples", type=int, default=16)
    parser.add_argument("--rank", type=int, default=DEFAULT_LORA_RANK)
    args = parser.parse_args()

    try:
        report = run_libero_offline_lora_comparison(
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


if __name__ == "__main__":
    main()
