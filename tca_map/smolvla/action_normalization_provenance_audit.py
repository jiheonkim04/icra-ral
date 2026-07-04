"""Report-only audit of SmolVLA action normalization provenance."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


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


def _flatten_numbers(items: list[Any]) -> np.ndarray:
    values: list[float] = []
    for item in items:
        if isinstance(item, (list, tuple)):
            for value in item:
                try:
                    values.append(float(value))
                except (TypeError, ValueError):
                    pass
    return np.asarray(values, dtype=np.float32)


def _range(values: np.ndarray) -> dict[str, Any]:
    if values.size == 0:
        return {"count": 0, "min": None, "max": None, "max_abs": None}
    return {
        "count": int(values.size),
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "max_abs": round(float(np.max(np.abs(values))), 6),
    }


def _load_config_files(smolvla_ckpt: Path) -> dict[str, Any]:
    config_path = smolvla_ckpt / "config.json"
    preprocessor_path = smolvla_ckpt / "policy_preprocessor.json"
    postprocessor_path = smolvla_ckpt / "policy_postprocessor.json"
    report: dict[str, Any] = {
        "config_path": str(config_path),
        "preprocessor_path": str(preprocessor_path),
        "postprocessor_path": str(postprocessor_path),
        "config_exists": config_path.exists(),
        "preprocessor_exists": preprocessor_path.exists(),
        "postprocessor_exists": postprocessor_path.exists(),
        "config": {},
        "preprocessor": {},
        "postprocessor": {},
    }
    if config_path.exists():
        config = _read_json(config_path)
        report["config"] = {
            "load_vlm_weights": config.get("load_vlm_weights"),
            "vlm_model_name": config.get("vlm_model_name"),
            "normalization_mapping": config.get("normalization_mapping") or {},
            "output_features": config.get("output_features") or {},
            "input_features": config.get("input_features") or {},
        }
    if preprocessor_path.exists():
        report["preprocessor"] = _read_json(preprocessor_path)
    if postprocessor_path.exists():
        report["postprocessor"] = _read_json(postprocessor_path)
    return report


def _processor_state_files(config_files: dict[str, Any], smolvla_ckpt: Path) -> list[Path]:
    paths: list[Path] = []
    for root_key in ("preprocessor", "postprocessor"):
        for step in (config_files.get(root_key) or {}).get("steps") or []:
            state_file = (step.get("state_file") or "").strip()
            if state_file:
                paths.append(smolvla_ckpt / state_file)
    return paths


def _load_action_stats(paths: list[Path]) -> dict[str, Any]:
    from safetensors import safe_open

    tensors: dict[str, Any] = {}
    prefixes: set[str] = set()
    missing = [str(path) for path in paths if not path.exists()]
    for path in paths:
        if not path.exists():
            continue
        with safe_open(path, framework="np") as handle:
            for key in handle.keys():
                if ".buffer.action." not in key:
                    continue
                arr = np.asarray(handle.get_tensor(key), dtype=np.float32).reshape(-1)
                prefix = key.split(".buffer.action.", 1)[0]
                prefixes.add(prefix)
                tensors[key] = {
                    "path": str(path),
                    "shape": list(arr.shape),
                    "values": [round(float(x), 6) for x in arr],
                    "min": round(float(arr.min()), 6),
                    "max": round(float(arr.max()), 6),
                    "max_abs": round(float(np.max(np.abs(arr))), 6),
                    "mean_abs": round(float(np.mean(np.abs(arr))), 6),
                }
    means = [value for key, info in tensors.items() if key.endswith(".mean") for value in info["values"]]
    stds = [value for key, info in tensors.items() if key.endswith(".std") for value in info["values"]]
    mean_arr = np.asarray(means, dtype=np.float32)
    std_arr = np.asarray(stds, dtype=np.float32)
    return {
        "state_files": [str(path) for path in paths],
        "missing_state_files": missing,
        "action_stat_keys": sorted(tensors),
        "action_stat_prefixes": sorted(prefixes),
        "tensor_summaries": tensors,
        "action_mean_range": _range(mean_arr),
        "action_std_range": _range(std_arr),
        "action_stats_present": bool(tensors),
    }


def _sample_ranges(vlm_report: dict[str, Any]) -> dict[str, Any]:
    samples = vlm_report.get("samples") or []
    expert = _flatten_numbers([sample.get("expert_action_preview") for sample in samples])
    adapted = _flatten_numbers([sample.get("adapted_action_preview") for sample in samples])
    policy = _flatten_numbers([sample.get("policy_action_preview") for sample in samples])
    clipped_total = int((vlm_report.get("metrics") or {}).get("clipped_values_total") or 0)
    return {
        "sample_count": len(samples),
        "expert_action_preview_range": _range(expert),
        "adapted_action_preview_range": _range(adapted),
        "policy_action_preview_range": _range(policy),
        "clipped_values_total": clipped_total,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    diagnosis = report.get("diagnosis") or {}
    stats = report.get("action_stats") or {}
    sample = report.get("sample_action_ranges") or {}
    lines = [
        "# Action Normalization Provenance Audit Report",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- audit passed: `{report.get('action_normalization_provenance_audit_passed')}`",
        f"- rollout scaling ready: `{report.get('ready_for_rollout_scaling')}`",
        f"- action stat prefixes: `{stats.get('action_stat_prefixes')}`",
        f"- action mean range: `{stats.get('action_mean_range')}`",
        f"- action std range: `{stats.get('action_std_range')}`",
        f"- expert action preview range: `{sample.get('expert_action_preview_range')}`",
        f"- adapted action preview range: `{sample.get('adapted_action_preview_range')}`",
        f"- clipped values total: `{sample.get('clipped_values_total')}`",
        "",
        "Blockers:",
        "",
    ]
    for item in diagnosis.get("blockers") or []:
        lines.append(f"- {item}")
    lines.extend(["", f"Recommended next step: {report.get('recommended_next_step')}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    summary_path = Path(args.summary_report)
    vlm_path = Path(args.vlm_enabled_report)
    smolvla_ckpt = Path(os.environ.get("SMOLVLA_CKPT") or args.smolvla_ckpt)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "evidence_label": "action_normalization_provenance_audit",
        "action_normalization_provenance_audit_passed": False,
        "decision": "stop",
        "ready_for_rollout_scaling": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "policy": {
            "report_only": True,
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
            "summary_report": str(summary_path),
            "vlm_enabled_report": str(vlm_path),
            "smolvla_ckpt": str(smolvla_ckpt),
        },
        "config_files": {},
        "action_stats": {},
        "sample_action_ranges": {},
        "diagnosis": {},
        "recommended_next_step": None,
        "error": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["decision"] = "stop"
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        return report, code

    if forbidden:
        return block("Forbidden gate(s) set for report-only audit: " + ", ".join(forbidden), 2)
    if not summary_path.exists():
        return block(f"VLM-on/off summary report is missing: {summary_path}", 3)
    if not vlm_path.exists():
        return block(f"VLM-enabled repeated offline report is missing: {vlm_path}", 4)
    if not smolvla_ckpt.exists():
        return block(f"SmolVLA checkpoint path is missing: {smolvla_ckpt}", 5)

    summary_report = _read_json(summary_path)
    vlm_report = _read_json(vlm_path)
    if not summary_report.get("vlm_enabled_offline_decoding_summary_passed"):
        return block("VLM-on/off summary did not pass; resolve it before provenance audit.", 6)

    config_files = _load_config_files(smolvla_ckpt)
    state_files = _processor_state_files(config_files, smolvla_ckpt)
    action_stats = _load_action_stats(state_files)
    sample_ranges = _sample_ranges(vlm_report)
    report["config_files"] = config_files
    report["action_stats"] = action_stats
    report["sample_action_ranges"] = sample_ranges

    config = config_files.get("config") or {}
    output_features = config.get("output_features") or {}
    policy_action_shape = (output_features.get("action") or {}).get("shape")
    normalization_mapping = config.get("normalization_mapping") or {}
    action_mean_max_abs = (action_stats.get("action_mean_range") or {}).get("max_abs") or 0.0
    action_std_max = (action_stats.get("action_std_range") or {}).get("max") or 0.0
    expert_max_abs = (sample_ranges.get("expert_action_preview_range") or {}).get("max_abs") or 0.0
    prefixes = action_stats.get("action_stat_prefixes") or []
    blockers: list[str] = []
    if any(str(prefix).startswith("so100") for prefix in prefixes):
        blockers.append("Processor action statistics are keyed by SO100 prefixes, which is a strong checkpoint/action-provenance mismatch risk for LIBERO diagnostics.")
    if action_mean_max_abs > 10 or action_std_max > 10:
        blockers.append("Action mean/std magnitudes are much larger than the local LIBERO expert-action preview range.")
    if expert_max_abs <= 1.1 and (action_mean_max_abs > 10 or action_std_max > 10):
        blockers.append("LIBERO expert actions appear normalized near [-1, 1], while checkpoint action stats are large robot-scale values.")
    if normalization_mapping.get("ACTION") == "MEAN_STD":
        blockers.append("ACTION MEAN_STD unnormalization is active, so mismatched action statistics can distort decoded actions before 6D-to-7D adaptation.")
    if policy_action_shape and policy_action_shape != [7]:
        blockers.append("Policy emits a 6D action while the local LIBERO expert/action adapter path uses a 7D convention.")
    if sample_ranges.get("clipped_values_total"):
        blockers.append("Adapted decoded actions still clip values, consistent with action-scale or provenance mismatch.")
    if not action_stats.get("action_stats_present"):
        blockers.append("No action mean/std tensors were found in processor safetensors.")

    report["diagnosis"] = {
        "blockers": blockers,
        "checkpoint_action_stats_appear_non_libero_scale": bool(action_mean_max_abs > 10 or action_std_max > 10),
        "checkpoint_action_stats_prefix_mismatch_risk": bool(any(str(prefix).startswith("so100") for prefix in prefixes)),
        "libero_expert_actions_appear_unit_scaled": bool(expert_max_abs <= 1.1 and expert_max_abs > 0),
        "policy_action_shape": policy_action_shape,
        "config_action_normalization": normalization_mapping.get("ACTION"),
        "ready_for_action_stat_mapping_plan": True,
        "rollout_scaling_blocked_reason": "Action stat provenance and 6D-to-7D action adaptation remain unresolved.",
    }
    report["action_normalization_provenance_audit_passed"] = True
    report["decision"] = "no_go_rollout_scaling"
    report["recommended_next_step"] = (
        "Create a planning-only action-stat mapping or checkpoint/task-provenance correction plan before any learned-policy rollout scaling."
    )
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-report", default="reports/vlm_enabled_offline_decoding_summary_report.json")
    parser.add_argument("--vlm-enabled-report", default="reports/vlm_enabled_repeated_offline_decoding_report.json")
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--report-path", default="reports/action_normalization_provenance_audit_report.json")
    parser.add_argument("--markdown-report-path", default="reports/action_normalization_provenance_audit_report.md")
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
