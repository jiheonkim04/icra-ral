"""Limited fixed-prior rollout readiness gate for the LIBERO offline proxy stack.

The gate checks whether the current ActionMap/TCA offline proxy outputs can be
used safely in a tiny LIBERO/RoboSuite rollout diagnostic. It performs no
training, no rollout, no simulator creation, no model loading, no GPU work, and
no downloads. If the gate is red, it writes a concrete blocker report rather
than attempting rollout.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.datasets.libero_offline_lora_comparison import ACTION_PREFIX_DIM, build_libero_lora_records
from tca_map.datasets.libero_publishability_gate_audit import _prior_source_leakage_audit
from tca_map.smolvla.interface_adapters import adapt_policy_action_to_env_action


SCHEMA_VERSION = "2026-07-05.libero_fixed_prior_rollout_readiness.v1"
FORBIDDEN_GATES = (
    "ALLOW_DOWNLOADS",
    "ALLOW_GPU_TRAINING",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_TINY_TRAINING",
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_TINY_ROLLOUT",
    "ALLOW_TINY_LEARNED_POLICY_ROLLOUT",
    "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP",
)


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"_file_exists": False, "_path": str(path), "_read_success": False}
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except Exception as exc:
        return {"_file_exists": True, "_path": str(path), "_read_success": False, "_read_error": str(exc)}
    if isinstance(data, dict):
        data["_file_exists"] = True
        data["_path"] = str(path)
        data["_read_success"] = True
        return data
    return {"_file_exists": True, "_path": str(path), "_read_success": False, "_read_error": "JSON root is not an object"}


def _truthy_path(data: dict[str, Any], *keys: str) -> bool:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict) or key not in cur:
            return False
        cur = cur[key]
    return bool(cur)


def _simulator_status(paths: dict[str, Path]) -> dict[str, Any]:
    reports = {name: _load_json(path) for name, path in paths.items()}
    import_ok = _truthy_path(reports["import"], "bounded_simulator_import_smoke_passed")
    render_ok = _truthy_path(reports["render"], "bounded_simulator_render_smoke_passed")
    reset_ok = _truthy_path(reports["reset_step"], "bounded_simulator_reset_step_smoke_passed")
    zero_rollout_ok = _truthy_path(reports["zero_rollout"], "bounded_libero_robosuite_diagnostic_rollout_passed")
    return {
        "reports": {
            name: {
                "path": str(path),
                "exists": bool(reports[name].get("_file_exists")),
                "readable": bool(reports[name].get("_read_success")),
            }
            for name, path in paths.items()
        },
        "import_smoke_passed": import_ok,
        "render_smoke_passed": render_ok,
        "reset_step_smoke_passed": reset_ok,
        "zero_action_libero_rollout_smoke_passed": zero_rollout_ok,
        "environment_plumbing_ready_for_tiny_diagnostic": bool(import_ok and render_ok and reset_ok and zero_rollout_ok),
    }


def _target_prior_status(records: list[dict[str, Any]]) -> dict[str, Any]:
    leakage = _prior_source_leakage_audit()
    fixed = leakage.get("fixed_learned_text_fusion", {})
    candidate_text_ok = all(
        isinstance(text, str) and text.strip()
        for record in records
        for text in (record.get("candidate_objects") or [])
    )
    return {
        "prior_source_leakage_audit": leakage,
        "fixed_prior_classification": fixed.get("classification"),
        "uses_bddl_metadata_at_inference": bool(fixed.get("uses_bddl_metadata")),
        "uses_eval_labels_at_inference": bool(fixed.get("uses_eval_labels")),
        "uses_dataset_target_labels_at_inference": bool(fixed.get("uses_dataset_target_labels")),
        "uses_task_id_filename_or_manifest_target_field_as_target_proxy": bool(
            fixed.get("uses_task_id_filename_or_manifest_target_field_as_target_proxy")
        ),
        "candidate_task_natural_language_text_available": candidate_text_ok,
        "available_at_test_time_under_current_assumption": bool(
            candidate_text_ok and fixed.get("classification") == "A_valid_test_time_semantic_prior"
        ),
        "caveat": "Valid only if equivalent candidate/task natural-language target text is available at rollout test time without BDDL, eval labels, dataset target labels, task ids, filenames, or manifest target fields.",
    }


def _action_bridge_status(records: list[dict[str, Any]], env_action_dim: int) -> dict[str, Any]:
    action_dims = sorted({len(record.get("expert_action") or []) for record in records})
    candidate_dims = sorted(
        {
            len(candidate)
            for record in records
            for candidate in (record.get("candidate_actions") or [])
            if isinstance(candidate, list)
        }
    )
    sample_action = records[0].get("expert_action") or []
    status: dict[str, Any] = {
        "offline_action_prefix_dim_constant": ACTION_PREFIX_DIM,
        "offline_record_action_dims": action_dims,
        "offline_candidate_action_dims": candidate_dims,
        "env_action_dim": env_action_dim,
        "existing_adapter_tested": True,
        "existing_adapter_supports_current_proxy_action": False,
        "adapter_error": None,
        "action_scale_safe": bool(max((abs(float(v)) for v in sample_action), default=0.0) <= 1.0),
        "clipping_expected_from_sample": False,
        "gripper_mapping_resolved": False,
        "rotation_mapping_resolved": False,
        "coordinate_convention_resolved": False,
    }
    try:
        adapted = adapt_policy_action_to_env_action(sample_action, env_action_dim)
        values = adapted.values
        status.update(
            {
                "existing_adapter_supports_current_proxy_action": True,
                "adapter_metadata": adapted.metadata,
                "adapted_sample_action_dim": len(values),
                "adapted_sample_max_abs": round(float(np.max(np.abs(values))), 6) if values else 0.0,
                "clipping_expected_from_sample": bool(adapted.metadata.get("clipped_values", 0) > 0),
                "gripper_mapping_resolved": bool(adapted.metadata.get("gripper_value") is not None or len(sample_action) == env_action_dim),
                "rotation_mapping_resolved": len(sample_action) == env_action_dim,
                "coordinate_convention_resolved": len(sample_action) == env_action_dim,
            }
        )
    except Exception as exc:
        status["adapter_error"] = str(exc)
    if action_dims != [env_action_dim]:
        status["blocker"] = (
            f"Current offline proxy records use {action_dims}D actions from ACTION_PREFIX_DIM={ACTION_PREFIX_DIM}, "
            f"but the LIBERO env action dimension is {env_action_dim}. The validated adapter does not define a 4D->7D rotation/gripper bridge."
        )
    else:
        status["blocker"] = None
    return status


def _previous_blockers(paths: dict[str, Path]) -> dict[str, Any]:
    reports = {name: _load_json(path) for name, path in paths.items()}
    action_stat_decision = reports["action_stats"].get("decision")
    metadata_findings = reports["metadata"].get("high_priority_findings") or reports["metadata"].get("findings") or []
    return {
        "reports": {
            name: {
                "path": str(path),
                "exists": bool(reports[name].get("_file_exists")),
                "readable": bool(reports[name].get("_read_success")),
            }
            for name, path in paths.items()
        },
        "checkpoint_task_action_stat_provenance_blocker_applies_to_learned_smolvla": action_stat_decision == "no_go_rollout_scaling",
        "action_bridge_mismatch_risk_present": True,
        "horizon_mismatch_risk_present": True,
        "gripper_scaling_risk_present": True,
        "prompt_camera_state_mismatch_risk_present": True,
        "metadata_findings": metadata_findings,
        "note": "These prior no-go issues are strongest for learned SmolVLA rollout scaling, but action bridge, gripper, horizon, prompt/camera/state risks still matter for fixed-prior proxy rollout.",
    }


def _policy() -> dict[str, Any]:
    return {
        "limited_fixed_prior_rollout_readiness_gate": True,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "gpu_training_performed": False,
        "training_performed": False,
        "lora_training_performed": False,
        "loss_computed": False,
        "heavy_model_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "simulator_environment_created": False,
        "rollout_attempted": False,
        "rollouts_performed": False,
        "openvla_oft_executed": False,
        "paper_grade_claims_made": False,
    }


def _write_reports(report: dict[str, Any], report_json: Path, report_md: Path) -> None:
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# Fixed-Prior Rollout Readiness Gate",
        "",
        "This is a readiness gate only. It performs no rollout, training, model loading, GPU job, download, OpenVLA-OFT execution, or paper-grade claim.",
        "",
        f"- gate status: `{report['risk_gate_status']}`",
        f"- rollout diagnostic authorized: `{report['rollout_diagnostic_authorized']}`",
        f"- simulator plumbing ready: `{report['simulator_status']['environment_plumbing_ready_for_tiny_diagnostic']}`",
        f"- action bridge ready: `{report['action_bridge_status']['existing_adapter_supports_current_proxy_action']}`",
        f"- target prior non-leaking under assumption: `{report['target_prior_status']['available_at_test_time_under_current_assumption']}`",
        f"- blocker count: `{len(report['blockers'])}`",
        "",
        "## Blockers",
        "",
    ]
    if report["blockers"]:
        lines.extend(f"- {item}" for item in report["blockers"])
    else:
        lines.append("- none")
    lines.extend(["", "## Recommended Next Step", "", report["recommended_next_step"], ""])
    report_md.write_text("\n".join(lines), encoding="utf-8")


def run_fixed_prior_rollout_readiness_gate(
    *,
    manifest_path: Path,
    report_json: Path,
    report_md: Path,
    max_pairs: int = 32,
    max_action_steps: int = 16,
    env_action_dim: int = 7,
    simulator_reports: dict[str, Path] | None = None,
    previous_reports: dict[str, Path] | None = None,
) -> dict[str, Any]:
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name)]
    records = build_libero_lora_records(manifest_path, max_pairs=max_pairs, max_action_steps=max_action_steps)
    simulator_reports = simulator_reports or {
        "import": Path("reports/bounded_simulator_import_smoke_report.json"),
        "render": Path("reports/bounded_simulator_render_smoke_report.json"),
        "reset_step": Path("reports/bounded_simulator_reset_step_smoke_report.json"),
        "zero_rollout": Path("reports/bounded_libero_robosuite_diagnostic_rollout_report.json"),
    }
    previous_reports = previous_reports or {
        "action_stats": Path("reports/libero_action_stat_subset_audit_report.json"),
        "metadata": Path("reports/action_interface_metadata_audit_report.json"),
        "alignment": Path("reports/hdf5_rollout_alignment_audit_report.json"),
        "vlm_summary": Path("reports/vlm_enabled_offline_decoding_summary_report.json"),
    }
    simulator = _simulator_status(simulator_reports)
    target_prior = _target_prior_status(records)
    action_bridge = _action_bridge_status(records, env_action_dim)
    previous = _previous_blockers(previous_reports)
    blockers: list[str] = []
    warnings: list[str] = []
    if forbidden:
        blockers.append("forbidden execution gates are set: " + ", ".join(forbidden))
    if not simulator["environment_plumbing_ready_for_tiny_diagnostic"]:
        blockers.append("existing LIBERO/RoboSuite import/render/reset/zero-action diagnostic evidence is incomplete")
    if not target_prior["available_at_test_time_under_current_assumption"]:
        blockers.append("non-leaking candidate/task natural-language target prior is not available under the current audit")
    if not action_bridge["existing_adapter_supports_current_proxy_action"]:
        blockers.append(str(action_bridge["blocker"] or action_bridge["adapter_error"] or "current action bridge is not ready"))
    if not action_bridge["gripper_mapping_resolved"]:
        blockers.append("gripper mapping is unresolved for current offline proxy actions")
    if not action_bridge["rotation_mapping_resolved"]:
        blockers.append("rotation/coordinate mapping is unresolved for current offline proxy actions")
    warnings.append("Current offline fixed-prior proxy uses text-derived cached features and mean HDF5 action snippets, not live camera/state-conditioned policy inference.")
    warnings.append("Previous learned-policy no-go issues still block learned SmolVLA rollout scaling; this gate only concerns fixed-prior proxy rollout readiness.")
    status = "green" if not blockers else "red"
    if status == "green" and warnings:
        status = "yellow"
    authorized = status == "green"
    recommended = (
        "Run the bounded fixed-prior rollout diagnostic with 1 task, 1 episode, and 10-25 steps."
        if authorized
        else "Do not run rollout. First add a validated 7D rollout-action bridge for fixed-prior proxy outputs, or rebuild the offline ActionMap/TCA rollout records to preserve all 7 LIBERO action dimensions and explicitly validate gripper/rotation/coordinate conventions on HDF5 before simulator stepping."
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "policy": _policy(),
        "source_manifest": str(manifest_path),
        "record_count": len(records),
        "max_pairs": max_pairs,
        "max_action_steps": max_action_steps,
        "env_action_dim": env_action_dim,
        "compared_variants_planned_if_green": [
            "ActionMap-style baseline",
            "fixed semantic target-prior TCA",
            "hard learned-target TCA if cheap",
            "oracle target upper bound only if non-invasive",
        ],
        "simulator_status": simulator,
        "target_prior_status": target_prior,
        "action_bridge_status": action_bridge,
        "camera_state_mapping_status": {
            "offline_proxy_uses_live_camera_or_state": False,
            "camera_mapping_matches_offline_assumption": False,
            "state_mapping_matches_offline_assumption": False,
            "note": "The current fixed-prior offline proxy is instruction/action-snippet based; a rollout diagnostic would test action bridge behavior, not full visual policy competence.",
        },
        "previous_learned_policy_no_go_status": previous,
        "blockers": blockers,
        "warnings": warnings,
        "risk_gate_status": status,
        "rollout_diagnostic_authorized": authorized,
        "rollout_happened": False,
        "training_happened": False,
        "lora_training_happened": False,
        "loss_computed": False,
        "ready_for_paper_claim": False,
        "recommended_next_step": recommended,
    }
    _write_reports(report, report_json, report_md)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--report-json", default="reports/libero_fixed_prior_rollout_readiness_gate_report.json")
    parser.add_argument("--report-md", default="reports/libero_fixed_prior_rollout_readiness_gate_report.md")
    parser.add_argument("--max-pairs", type=int, default=32)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--env-action-dim", type=int, default=7)
    args = parser.parse_args()
    report = run_fixed_prior_rollout_readiness_gate(
        manifest_path=Path(args.manifest),
        report_json=Path(args.report_json),
        report_md=Path(args.report_md),
        max_pairs=args.max_pairs,
        max_action_steps=args.max_action_steps,
        env_action_dim=args.env_action_dim,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
