"""Run G3P-VLA development-only Stage 0 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.g3p_vla import PROPOSAL_HASH, audit_g3p_records  # noqa: E402


DATE_KST = "2026-07-15"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, report: Mapping[str, Any]) -> None:
    source_gate = report.get("source_gate_manifest") or {}
    train_labels = report.get("train_point_label_summary") or {}
    validation_labels = report.get("validation_point_label_summary") or {}
    predictability = report.get("point_predictability_summary") or {}
    gradient = report.get("gradient_audit") or {}
    lines = [
        "# G3P-VLA Development Audit",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- training happened: `{report['training_happened']}`",
        f"- validation search happened: `{report['validation_search_happened']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        f"- scoreable development records: `{report['scoreable_development_records']}`",
        f"- train records: `{report['train_records']}`",
        f"- validation records: `{report['validation_records']}`",
        f"- reserved records not used: `{report['reserved_records_not_used']}`",
        f"- selected task count: `{report['selected_task_count']}`",
        f"- duplicate sample keys: `{report['duplicate_sample_keys']}`",
        f"- duplicate frame keys: `{report['duplicate_frame_keys']}`",
        f"- source gate passed: `{source_gate.get('source_gate_passed')}`",
        f"- RGB video available in dataset: `{source_gate.get('rgb_video_available_in_dataset')}`",
        f"- privileged object/pose feature available: `{source_gate.get('privileged_object_pose_available_as_dataset_feature')}`",
        f"- train valid point fraction: `{train_labels.get('valid_point_fraction')}`",
        f"- validation valid point fraction: `{validation_labels.get('valid_point_fraction')}`",
        f"- validation material point fraction: `{validation_labels.get('material_point_fraction_of_valid')}`",
        f"- point predictability margin: `{predictability.get('accuracy_margin')}`",
        f"- best trivial baseline: `{predictability.get('best_trivial_baseline')}`",
        f"- oracle action headroom L2 validation: `{report['oracle_action_headroom_l2_validation']}`",
        f"- initial action delta p95: `{report['initial_action_delta_p95']}`",
        f"- base action validity: `{report['base_action_validity']}`",
        f"- point gradient norm: `{gradient.get('point_probe_gradient_norm')}`",
        f"- adapter surrogate gradient norm: `{gradient.get('adapter_surrogate_gradient_norm')}`",
        "",
        "Source gate manifest:",
        "",
        "```json",
        json.dumps(source_gate, indent=2, sort_keys=True),
        "```",
        "",
        "Point label manifest:",
        "",
        "```json",
        json.dumps(report.get("point_label_manifest"), indent=2, sort_keys=True),
        "```",
        "",
        "Predictability summary:",
        "",
        "```json",
        json.dumps(predictability, indent=2, sort_keys=True),
        "```",
        "",
        "Gradient audit:",
        "",
        "```json",
        json.dumps(gradient, indent=2, sort_keys=True),
        "```",
        "",
        "Split manifest:",
        "",
        "```json",
        json.dumps(report.get("split_manifest"), indent=2, sort_keys=True),
        "```",
        "",
        "Hard stop reasons:",
    ]
    reasons = list(report.get("hard_stop_reasons") or [])
    if reasons:
        lines.extend(f"- `{reason}`" for reason in reasons)
    else:
        lines.append("- none")
    lines.extend(["", f"Next step: {report['next_step']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _source_metadata(split_manifest_path: Path) -> dict[str, Any]:
    if not split_manifest_path.exists():
        return {}
    manifest = _read_json(split_manifest_path)
    dataset_root = Path(str((manifest.get("paths") or {}).get("dataset_root", "")))
    info_path = dataset_root / "meta" / "info.json"
    if not info_path.exists():
        return {"dataset_root": str(dataset_root), "features": {}}
    info = _read_json(info_path)
    return {
        "dataset_root": str(dataset_root),
        "features": info.get("features") or {},
        "total_frames": info.get("total_frames"),
        "total_episodes": info.get("total_episodes"),
        "total_tasks": info.get("total_tasks"),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit"], default="audit")
    parser.add_argument("--prediction-artifact", default="reports/official_smolvla_stable_prediction_artifact.json")
    parser.add_argument("--split-manifest", default="reports/official_smolvla_split_manifest.json")
    parser.add_argument("--json-output", default="reports/g3p_vla/development_audit.json")
    parser.add_argument("--md-output", default="reports/g3p_vla/development_audit.md")
    parser.add_argument("--source-gate-output", default="reports/g3p_vla/source_gate_manifest.json")
    parser.add_argument("--point-label-output", default="reports/g3p_vla/point_label_manifest.json")
    parser.add_argument("--split-output", default="reports/g3p_vla/split_manifest.json")
    args = parser.parse_args()

    artifact = _read_json(Path(args.prediction_artifact))
    report = audit_g3p_records(
        artifact["records"],
        source_metadata=_source_metadata(Path(args.split_manifest)),
    )
    report = {
        **report,
        "date_kst": DATE_KST,
        "mode": "audit",
        "source_prediction_artifact": str(args.prediction_artifact),
        "source_split_manifest": str(args.split_manifest),
    }
    _write_json(Path(args.json_output), report)
    _write_md(Path(args.md_output), report)
    _write_json(Path(args.source_gate_output), report["source_gate_manifest"])
    _write_json(Path(args.point_label_output), report["point_label_manifest"])
    _write_json(Path(args.split_output), report["split_manifest"])
    summary = {
        "mode": args.mode,
        "source_prediction_artifact": str(args.prediction_artifact),
        "audit_decision": report["final_decision"],
        "audit_hard_stop_count": len(report["hard_stop_reasons"]),
        "json_output": args.json_output,
        "md_output": args.md_output,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
