"""Report-only VLM-enabled versus no-VLM offline decoding summary."""

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


def _safe_float(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _delta(new: float | None, old: float | None) -> dict[str, float | None]:
    if new is None or old is None:
        return {"absolute": None, "relative": None, "percent_reduction": None}
    relative = (old - new) / old if old else None
    return {
        "absolute": round(new - old, 6),
        "relative": round(relative, 6) if relative is not None else None,
        "percent_reduction": round(relative * 100.0, 3) if relative is not None else None,
    }


def _load_config_summary(smolvla_ckpt: Path) -> dict[str, Any]:
    config_path = smolvla_ckpt / "config.json"
    preprocessor_path = smolvla_ckpt / "policy_preprocessor.json"
    postprocessor_path = smolvla_ckpt / "policy_postprocessor.json"
    summary: dict[str, Any] = {
        "config_path": str(config_path),
        "config_exists": config_path.exists(),
        "policy_preprocessor_exists": preprocessor_path.exists(),
        "policy_postprocessor_exists": postprocessor_path.exists(),
        "config_load_vlm_weights": None,
        "vlm_model_name": None,
        "normalization_mapping": {},
        "policy_action_shape": None,
        "policy_state_shape": None,
        "policy_image_inputs": [],
        "postprocessor_action_norm": None,
        "preprocessor_device": None,
        "postprocessor_device": None,
    }
    if config_path.exists():
        config = _read_json(config_path)
        summary["config_load_vlm_weights"] = config.get("load_vlm_weights")
        summary["vlm_model_name"] = config.get("vlm_model_name")
        summary["normalization_mapping"] = config.get("normalization_mapping") or {}
        output_features = config.get("output_features") or {}
        input_features = config.get("input_features") or {}
        summary["policy_action_shape"] = (output_features.get("action") or {}).get("shape")
        summary["policy_state_shape"] = (input_features.get("observation.state") or {}).get("shape")
        summary["policy_image_inputs"] = sorted(
            key for key, spec in input_features.items() if (spec or {}).get("type") == "VISUAL"
        )
    if preprocessor_path.exists():
        pre = _read_json(preprocessor_path)
        for step in pre.get("steps") or []:
            if step.get("registry_name") == "device_processor":
                summary["preprocessor_device"] = (step.get("config") or {}).get("device")
    if postprocessor_path.exists():
        post = _read_json(postprocessor_path)
        for step in post.get("steps") or []:
            if step.get("registry_name") == "unnormalizer_processor":
                summary["postprocessor_action_norm"] = ((step.get("config") or {}).get("norm_map") or {}).get("ACTION")
            if step.get("registry_name") == "device_processor":
                summary["postprocessor_device"] = (step.get("config") or {}).get("device")
    return summary


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    comparison = report.get("comparison") or {}
    diagnosis = report.get("diagnosis") or {}
    lines = [
        "# VLM-Enabled Offline Decoding Summary Report",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- summary passed: `{report.get('vlm_enabled_offline_decoding_summary_passed')}`",
        f"- no-VLM mean action L1/MSE: `{comparison.get('no_vlm_mean_action_l1_to_expert')}` / `{comparison.get('no_vlm_mean_action_mse_to_expert')}`",
        f"- VLM-enabled mean action L1/MSE: `{comparison.get('vlm_enabled_mean_action_l1_to_expert')}` / `{comparison.get('vlm_enabled_mean_action_mse_to_expert')}`",
        f"- L1 percent reduction: `{(comparison.get('l1_delta') or {}).get('percent_reduction')}`",
        f"- MSE percent reduction: `{(comparison.get('mse_delta') or {}).get('percent_reduction')}`",
        f"- no-VLM signal: `{comparison.get('no_vlm_alignment_signal')}`",
        f"- VLM-enabled signal: `{comparison.get('vlm_enabled_alignment_signal')}`",
        f"- rollout scaling ready: `{report.get('ready_for_rollout_scaling')}`",
        f"- benchmark claim ready: `{report.get('ready_for_benchmark_claim')}`",
        f"- paper claim ready: `{report.get('ready_for_paper_claim')}`",
        "",
        "Diagnosis:",
        "",
    ]
    for item in diagnosis.get("blockers") or []:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            f"Recommended next step: {report.get('recommended_next_step')}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    no_vlm_path = Path(args.no_vlm_report)
    vlm_path = Path(args.vlm_enabled_report)
    smolvla_ckpt = Path(os.environ.get("SMOLVLA_CKPT") or args.smolvla_ckpt)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "evidence_label": "vlm_enabled_offline_decoding_summary",
        "vlm_enabled_offline_decoding_summary_passed": False,
        "decision": "stop",
        "ready_for_rollout_scaling": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "policy": {
            "summary_only": True,
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
        "claims": {
            "standard_success_claimed": False,
            "benchmark_success_claimed": False,
            "counterfactual_robustness_claimed": False,
            "sota_claimed": False,
            "paper_grade_claim_made": False,
        },
        "paths": {
            "no_vlm_report": str(no_vlm_path),
            "vlm_enabled_report": str(vlm_path),
            "smolvla_ckpt": str(smolvla_ckpt),
        },
        "comparison": {},
        "config_summary": {},
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
        return block("Forbidden gate(s) set for summary-only task: " + ", ".join(forbidden), 2)
    if not no_vlm_path.exists():
        return block(f"No-VLM repeated offline report is missing: {no_vlm_path}", 3)
    if not vlm_path.exists():
        return block(f"VLM-enabled repeated offline report is missing: {vlm_path}", 4)

    no_vlm = _read_json(no_vlm_path)
    vlm = _read_json(vlm_path)
    no_metrics = no_vlm.get("metrics") or {}
    vlm_metrics = vlm.get("metrics") or {}
    no_l1 = _safe_float(no_metrics.get("mean_action_l1_to_expert"))
    vlm_l1 = _safe_float(vlm_metrics.get("mean_action_l1_to_expert"))
    no_mse = _safe_float(no_metrics.get("mean_action_mse_to_expert"))
    vlm_mse = _safe_float(vlm_metrics.get("mean_action_mse_to_expert"))
    no_clipped = int(no_metrics.get("clipped_values_total") or 0)
    vlm_clipped = int(vlm_metrics.get("clipped_values_total") or 0)
    same_timesteps = list(no_metrics.get("timesteps") or []) == list(vlm_metrics.get("timesteps") or [])
    same_sample_count = no_metrics.get("sample_count") == vlm_metrics.get("sample_count")
    l1_delta = _delta(vlm_l1, no_l1)
    mse_delta = _delta(vlm_mse, no_mse)
    config_summary = _load_config_summary(smolvla_ckpt)

    blockers: list[str] = []
    if vlm_metrics.get("offline_alignment_signal") != "strong":
        blockers.append("VLM-enabled alignment is not strong; it remains an offline diagnostic, not rollout evidence.")
    if vlm_metrics.get("offline_alignment_signal") == "weak":
        blockers.append("VLM-enabled alignment signal is still weak despite lower action-distance metrics.")
    if vlm_clipped:
        blockers.append("Adapted actions still clip values; action normalization or 6D-to-7D adaptation remains a likely source of error.")
    if config_summary.get("postprocessor_action_norm") == "MEAN_STD":
        blockers.append("Policy postprocessor uses ACTION MEAN_STD unnormalization; provenance of action scale/statistics should be audited before rollout scaling.")
    if config_summary.get("policy_action_shape") and config_summary.get("policy_action_shape") != [7]:
        blockers.append("Policy action shape is not the 7D LIBERO expert-action convention; adapter interpretation remains a blocker.")
    if not same_timesteps or not same_sample_count:
        blockers.append("The VLM and no-VLM reports do not compare identical timesteps/sample counts.")

    report["comparison"] = {
        "no_vlm_load_vlm_weights": no_metrics.get("load_vlm_weights"),
        "vlm_enabled_load_vlm_weights": vlm_metrics.get("load_vlm_weights"),
        "same_timesteps": same_timesteps,
        "same_sample_count": same_sample_count,
        "timesteps": vlm_metrics.get("timesteps"),
        "sample_count": vlm_metrics.get("sample_count"),
        "no_vlm_mean_action_l1_to_expert": no_l1,
        "vlm_enabled_mean_action_l1_to_expert": vlm_l1,
        "l1_delta": l1_delta,
        "no_vlm_mean_action_mse_to_expert": no_mse,
        "vlm_enabled_mean_action_mse_to_expert": vlm_mse,
        "mse_delta": mse_delta,
        "no_vlm_mean_policy6_l1_to_expert_first6": no_metrics.get("mean_policy6_l1_to_expert_first6"),
        "vlm_enabled_mean_policy6_l1_to_expert_first6": vlm_metrics.get("mean_policy6_l1_to_expert_first6"),
        "no_vlm_alignment_signal": no_metrics.get("offline_alignment_signal"),
        "vlm_enabled_alignment_signal": vlm_metrics.get("offline_alignment_signal"),
        "no_vlm_clipped_values_total": no_clipped,
        "vlm_enabled_clipped_values_total": vlm_clipped,
        "vlm_enabled_improved_l1": bool(vlm_l1 is not None and no_l1 is not None and vlm_l1 < no_l1),
        "vlm_enabled_improved_mse": bool(vlm_mse is not None and no_mse is not None and vlm_mse < no_mse),
    }
    report["config_summary"] = config_summary
    report["diagnosis"] = {
        "blockers": blockers,
        "vlm_enabled_behaviorally_relevant": bool(report["comparison"]["vlm_enabled_improved_l1"] or report["comparison"]["vlm_enabled_improved_mse"]),
        "rollout_scaling_blocked_reason": "Offline alignment remains weak and adapter/action-normalization blockers remain unresolved.",
        "ready_for_action_normalization_provenance_probe": True,
    }
    report["vlm_enabled_offline_decoding_summary_passed"] = True
    report["decision"] = "summary_complete"
    report["recommended_next_step"] = (
        "Create a report-only action-normalization/provenance audit for the SmolVLA action stats, 6D-to-7D adapter, "
        "and clipping behavior before any learned-policy rollout scaling."
    )
    return report, 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-vlm-report", default="reports/repeated_offline_demo_action_decoding_report.json")
    parser.add_argument("--vlm-enabled-report", default="reports/vlm_enabled_repeated_offline_decoding_report.json")
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--report-path", default="reports/vlm_enabled_offline_decoding_summary_report.json")
    parser.add_argument("--markdown-report-path", default="reports/vlm_enabled_offline_decoding_summary_report.md")
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
