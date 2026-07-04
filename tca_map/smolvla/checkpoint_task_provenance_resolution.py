"""Report-only checkpoint/task provenance resolution audit."""

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
    "ALLOW_WSL_SMOLVLA_SINGLE_ACTION",
    "ALLOW_OFFLINE_DEMO_ACTION_DECODING",
    "ALLOW_REPEATED_OFFLINE_DEMO_DECODING",
    "ALLOW_VLM_ENABLED_REPEATED_OFFLINE_DECODING",
    "ALLOW_NORMALIZED_ACTION_SPACE_PROBE",
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8-sig") if path.exists() else ""


def _shape(features: dict[str, Any], key: str) -> list[int] | None:
    value = features.get(key) or {}
    shape = value.get("shape")
    return list(shape) if isinstance(shape, list) else None


def _processor_feature_shape(processor: dict[str, Any], feature: str) -> list[int] | None:
    for step in processor.get("steps") or []:
        features = ((step.get("config") or {}).get("features") or {})
        if feature in features:
            shape = features[feature].get("shape")
            return list(shape) if isinstance(shape, list) else None
    return None


def _tokenizer_name(processor: dict[str, Any]) -> str | None:
    for step in processor.get("steps") or []:
        if step.get("registry_name") == "tokenizer_processor":
            return (step.get("config") or {}).get("tokenizer_name")
    return None


def _readme_signals(text: str) -> dict[str, Any]:
    lowered = text.lower()
    return {
        "mentions_lerobot_libero_quickstart": 'lerobotdataset("lerobot/libero")' in lowered,
        "mentions_base_model_to_fine_tune": "base model to fine tune" in lowered,
        "mentions_so100_robot": "so100" in lowered,
        "mentions_real_world_inference": "real-world inference" in lowered,
    }


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    evidence = report.get("evidence_summary") or {}
    lines = [
        "# Checkpoint / Task Provenance Resolution Report",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- audit passed: `{report.get('checkpoint_task_provenance_resolution_passed')}`",
        f"- selected next step: `{report.get('selected_next_step')}`",
        f"- current checkpoint valid for LIBERO learned-policy rollout evidence: `{report.get('current_checkpoint_libero_rollout_evidence_valid')}`",
        f"- ready for rollout scaling: `{report.get('ready_for_rollout_scaling')}`",
        f"- ready for offline TCA-Map pivot: `{report.get('ready_for_offline_head_tca_pivot')}`",
        f"- ready for LIBERO-aligned checkpoint source plan: `{report.get('ready_for_libero_aligned_checkpoint_source_plan')}`",
        "",
        "Evidence:",
        "",
        f"- checkpoint action shape: `{evidence.get('checkpoint_action_shape')}`",
        f"- checkpoint state shape: `{evidence.get('checkpoint_state_shape')}`",
        f"- checkpoint image input count: `{evidence.get('checkpoint_image_input_count')}`",
        f"- LIBERO action dim: `{evidence.get('libero_action_dim')}`",
        f"- LIBERO action max abs: `{evidence.get('libero_action_max_abs')}`",
        f"- processor stat prefixes: `{evidence.get('checkpoint_action_stat_prefixes')}`",
        f"- SO100 prefix detected: `{evidence.get('checkpoint_prefixes_look_so100')}`",
        f"- README mentions base-model fine-tuning: `{evidence.get('readme_mentions_base_model_to_fine_tune')}`",
        f"- README mentions SO100: `{evidence.get('readme_mentions_so100_robot')}`",
        "",
        f"Recommended next step: {report.get('recommended_next_step')}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _looks_so100(prefixes: Any, readme: dict[str, Any]) -> bool:
    prefix_hit = any(str(item).lower().startswith("so100") for item in (prefixes or []))
    return bool(prefix_hit or readme.get("mentions_so100_robot"))


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    checkpoint_root = Path(os.environ.get("SMOLVLA_CKPT") or args.checkpoint_root)
    plan_path = Path(args.normalized_plan_report)
    stat_audit_path = Path(args.libero_action_stat_report)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "evidence_label": "checkpoint_task_provenance_resolution",
        "checkpoint_task_provenance_resolution_passed": False,
        "decision": "stop",
        "selected_next_step": None,
        "current_checkpoint_libero_rollout_evidence_valid": False,
        "ready_for_rollout_scaling": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "ready_for_offline_head_tca_pivot": False,
        "ready_for_libero_aligned_checkpoint_source_plan": False,
        "ready_for_normalized_action_space_probe_runner": False,
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
            "policy_behavior_changed": False,
            "forbidden_gates_set": forbidden,
        },
        "paths": {
            "checkpoint_root": str(checkpoint_root),
            "normalized_plan_report": str(plan_path),
            "libero_action_stat_report": str(stat_audit_path),
        },
        "evidence_summary": {},
        "risks": [],
        "recommended_next_step": None,
        "error": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["decision"] = "stop"
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        return report, code

    if forbidden:
        return block("Forbidden gate(s) set for report-only provenance audit: " + ", ".join(forbidden), 2)
    if not checkpoint_root.exists():
        return block(f"SmolVLA checkpoint root is missing: {checkpoint_root}", 3)
    if not plan_path.exists():
        return block(f"Normalized action-space probe plan report is missing: {plan_path}", 4)
    if not stat_audit_path.exists():
        return block(f"LIBERO action-stat subset audit report is missing: {stat_audit_path}", 5)

    plan = _read_json(plan_path)
    if not plan.get("ready_for_checkpoint_task_provenance_resolution"):
        return block("Normalized action-space plan did not authorize checkpoint/task provenance resolution.", 6)

    stat_audit = _read_json(stat_audit_path)
    if not stat_audit.get("libero_action_stat_subset_audit_passed"):
        return block("LIBERO action-stat subset audit did not pass.", 7)

    config_path = checkpoint_root / "config.json"
    preprocessor_path = checkpoint_root / "policy_preprocessor.json"
    postprocessor_path = checkpoint_root / "policy_postprocessor.json"
    readme_path = checkpoint_root / "README.md"
    for required_path in [config_path, preprocessor_path, postprocessor_path, readme_path]:
        if not required_path.exists():
            return block(f"Required checkpoint metadata file is missing: {required_path}", 8)

    config = _read_json(config_path)
    preprocessor = _read_json(preprocessor_path)
    postprocessor = _read_json(postprocessor_path)
    readme = _readme_signals(_read_text(readme_path))
    comparison = stat_audit.get("comparison_to_checkpoint") or {}
    stats = stat_audit.get("libero_action_stats") or {}
    checkpoint_action_shape = _shape(config.get("output_features") or {}, "action")
    checkpoint_state_shape = _shape(config.get("input_features") or {}, "observation.state")
    image_inputs = [
        key
        for key, value in (config.get("input_features") or {}).items()
        if (value or {}).get("type") == "VISUAL"
    ]
    prefixes = comparison.get("checkpoint_action_stat_prefixes") or []
    scale_mismatch = bool(comparison.get("scale_mismatch_confirmed"))
    dimension_mismatch = bool(comparison.get("dimension_mismatch_confirmed"))
    so100_signal = _looks_so100(prefixes, readme)
    base_model_signal = bool(readme.get("mentions_base_model_to_fine_tune"))
    libero_quickstart_signal = bool(readme.get("mentions_lerobot_libero_quickstart"))
    checkpoint_action_dim = checkpoint_action_shape[0] if checkpoint_action_shape else None
    libero_action_dim = stats.get("dim")
    action_shape_matches = bool(checkpoint_action_dim == libero_action_dim)

    report["evidence_summary"] = {
        "checkpoint_action_shape": checkpoint_action_shape,
        "checkpoint_state_shape": checkpoint_state_shape,
        "checkpoint_image_inputs": image_inputs,
        "checkpoint_image_input_count": len(image_inputs),
        "checkpoint_normalization_mapping": config.get("normalization_mapping"),
        "checkpoint_vlm_model_name": config.get("vlm_model_name"),
        "checkpoint_load_vlm_weights": config.get("load_vlm_weights"),
        "preprocessor_action_shape": _processor_feature_shape(preprocessor, "action"),
        "preprocessor_state_shape": _processor_feature_shape(preprocessor, "observation.state"),
        "postprocessor_action_shape": _processor_feature_shape(postprocessor, "action"),
        "preprocessor_tokenizer_name": _tokenizer_name(preprocessor),
        "libero_action_dim": libero_action_dim,
        "libero_action_max_abs": stats.get("max_abs"),
        "checkpoint_action_stat_prefixes": prefixes,
        "checkpoint_action_mean_max_abs": comparison.get("checkpoint_action_mean_max_abs"),
        "checkpoint_action_std_max": comparison.get("checkpoint_action_std_max"),
        "checkpoint_prefixes_look_so100": so100_signal,
        "scale_mismatch_confirmed": scale_mismatch,
        "dimension_mismatch_confirmed": dimension_mismatch,
        "action_shape_matches_libero": action_shape_matches,
        "readme_mentions_lerobot_libero_quickstart": libero_quickstart_signal,
        "readme_mentions_base_model_to_fine_tune": base_model_signal,
        "readme_mentions_so100_robot": readme.get("mentions_so100_robot"),
        "readme_mentions_real_world_inference": readme.get("mentions_real_world_inference"),
    }

    risks: list[str] = []
    if so100_signal:
        risks.append("checkpoint_or_model_card_contains_so100_provenance_signal")
    if base_model_signal:
        risks.append("model_card_describes_checkpoint_as_base_model_to_fine_tune")
    if libero_quickstart_signal:
        risks.append("model_card_libero_quickstart_is_not_sufficient_to_prove_libero_action_stats")
    if scale_mismatch:
        risks.append("checkpoint_action_stat_scale_mismatches_local_libero_unit_scale_actions")
    if dimension_mismatch or not action_shape_matches:
        risks.append("checkpoint_action_dimension_mismatches_local_libero_7d_actions")
    report["risks"] = risks

    severe_mismatch = bool(so100_signal and base_model_signal and scale_mismatch and (dimension_mismatch or not action_shape_matches))
    if severe_mismatch:
        report["decision"] = "no_go_learned_policy_rollout_scaling"
        report["checkpoint_task_provenance_resolution_passed"] = True
        report["selected_next_step"] = "pivot_to_offline_head_tca_map_and_lora_or_find_libero_aligned_checkpoint"
        report["current_checkpoint_libero_rollout_evidence_valid"] = False
        report["ready_for_offline_head_tca_pivot"] = True
        report["ready_for_libero_aligned_checkpoint_source_plan"] = True
        report["ready_for_normalized_action_space_probe_runner"] = False
        report["recommended_next_step"] = (
            "Do not scale learned-policy LIBERO rollouts with this checkpoint. Continue paper work through offline/head TCA-Map and required LoRA tracks, "
            "or create a separate source-resolution plan for a LIBERO-action-aligned SmolVLA checkpoint before any further learned-policy rollout evidence."
        )
    elif scale_mismatch or dimension_mismatch:
        report["decision"] = "reduce_scope"
        report["checkpoint_task_provenance_resolution_passed"] = True
        report["selected_next_step"] = "plan_bounded_normalized_action_space_probe"
        report["ready_for_libero_aligned_checkpoint_source_plan"] = True
        report["recommended_next_step"] = (
            "Mismatch remains but SO100/base-model provenance is not conclusive. Plan a separate offline normalized-action-space probe before any rollout."
        )
    else:
        report["decision"] = "review_required"
        report["checkpoint_task_provenance_resolution_passed"] = True
        report["selected_next_step"] = "manual_review"
        report["current_checkpoint_libero_rollout_evidence_valid"] = False
        report["recommended_next_step"] = "No direct action-stat mismatch was confirmed; review provenance evidence before proceeding."

    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--normalized-plan-report", default="reports/normalized_action_space_probe_plan_report.json")
    parser.add_argument("--libero-action-stat-report", default="reports/libero_action_stat_subset_audit_report.json")
    parser.add_argument("--report-path", default="reports/checkpoint_task_provenance_resolution_report.json")
    parser.add_argument("--markdown-report-path", default="reports/checkpoint_task_provenance_resolution_report.md")
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
