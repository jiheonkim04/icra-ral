"""Run MTF-VLA development-only Stage 0 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.mtf_vla import PROPOSAL_HASH, audit_mtf_records, run_validation_search  # noqa: E402


DATE_KST = "2026-07-14"


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, set):
        return sorted(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _load_state_by_index(dataset_root: Path) -> dict[int, list[float]]:
    import pandas as pd

    state_by_index: dict[int, list[float]] = {}
    for parquet_path in sorted((dataset_root / "data" / "chunk-000").glob("*.parquet")):
        frame = pd.read_parquet(parquet_path, columns=["index", "observation.state"])
        for index_value, state_value in zip(frame["index"], frame["observation.state"]):
            state_by_index[int(index_value)] = [float(value) for value in state_value]
    return state_by_index


def _write_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# MTF-VLA Development Audit",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- training happened: `{report['training_happened']}`",
        f"- scoreable development records: `{report['scoreable_records']}`",
        f"- raw prediction records: `{report['raw_prediction_records']}`",
        f"- train records: `{report['train_records']}`",
        f"- validation records: `{report['validation_records']}`",
        f"- reserved records not used: `{report['reserved_records_not_used']}`",
        f"- selected task count: `{report['selected_task_count']}`",
        f"- duplicate sample keys: `{report['duplicate_sample_keys']}`",
        f"- duplicate frame keys: `{report['duplicate_frame_keys']}`",
        f"- high milestone count: `{report['high_milestone_count']}`",
        f"- retention frame count: `{report['retention_frame_count']}`",
        f"- high milestone fraction: `{report['high_milestone_fraction']}`",
        f"- retention frame fraction: `{report['retention_frame_fraction']}`",
        f"- high-low score gap: `{report['high_low_score_gap']}`",
        f"- gripper transition fraction: `{report['gripper_transition_fraction']}`",
        f"- state joined fraction: `{report['state_joined_fraction']}`",
        f"- uniform overlap fraction: `{report['uniform_overlap_fraction']}`",
        f"- adapter init action delta p95: `{report['adapter_init_action_delta_p95']}`",
        "",
        "Base headroom:",
        "",
        "```json",
        json.dumps(report.get("base_headroom"), indent=2, sort_keys=True),
        "```",
        "",
        "Frame score summary:",
        "",
        "```json",
        json.dumps(report.get("frame_score_summary"), indent=2, sort_keys=True),
        "```",
        "",
        "Base-retention target manifest:",
        "",
        "```json",
        json.dumps(report.get("base_retention_target_manifest"), indent=2, sort_keys=True),
        "```",
        "",
        "Split manifest:",
        "",
        "```json",
        json.dumps(report.get("split_manifest"), indent=2, sort_keys=True),
        "```",
        "",
        "FrameSkip proxy:",
        "",
        "```json",
        json.dumps(report.get("frameskip_proxy"), indent=2, sort_keys=True),
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


def _write_validation_md(path: Path, report: Mapping[str, Any]) -> None:
    selected = report.get("selected_config") or {}
    selected_score = selected.get("score_terms") or {}
    manifest = report.get("selected_training_manifest") or {}
    counts = manifest.get("counts") or {}
    lines = [
        "# MTF-VLA Validation Search",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- training happened: `{report['training_happened']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        f"- search budget: `{report['search_budget']}`",
        f"- tried configs: `{report['tried_config_count']}`",
        f"- selected config: `{selected.get('config_id')}`",
        f"- selected score: `{selected_score.get('total')}`",
        f"- selected retained ratio: `{selected.get('retained_high_frame_ratio')}`",
        f"- selected retention coefficient: `{selected.get('retention_coefficient')}`",
        f"- selected train records: `{counts.get('train_records')}`",
        f"- selected MTF high frames: `{counts.get('mtf_high_frames')}`",
        f"- selected MTF retention frames: `{counts.get('mtf_retention_frames')}`",
        "",
        "Score weights:",
        "",
        "```json",
        json.dumps(report.get("score_weights"), indent=2, sort_keys=True),
        "```",
        "",
        "Selected config:",
        "",
        "```json",
        json.dumps(selected, indent=2, sort_keys=True),
        "```",
        "",
        "Tried configurations:",
        "",
        "| config | decision | ratio | retention | train high | train retention | proxy | clean | mechanism | validity | compute | total |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in report.get("tried_configs", []):
        score = item.get("score_terms") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{item.get('config_id')}`",
                    f"`{item.get('final_decision')}`",
                    f"{item.get('retained_high_frame_ratio')}",
                    f"{item.get('retention_coefficient')}",
                    f"{item.get('high_train_frames')}",
                    f"{item.get('retention_train_frames')}",
                    f"{score.get('validation_closed_loop_proxy')}",
                    f"{score.get('clean_retention')}",
                    f"{score.get('mechanism_activation')}",
                    f"{score.get('action_validity_and_bounded_delta')}",
                    f"{score.get('compute_overhead')}",
                    f"{score.get('total')}",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Checkpoint status:",
            "",
            "- no adapter checkpoint was trained in this validation-search step;",
            "- selected training manifest is frozen for the next adapter-training step;",
            "- Stage A must not start before disk-reloadable checkpoints exist.",
            "",
            f"Next step: {report['next_step']}",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit", "validation", "all"], default="audit")
    parser.add_argument("--prediction-artifact", default="reports/official_smolvla_stable_prediction_artifact.json")
    parser.add_argument("--dataset-root", default=r"C:\assets\datasets\lerobot_libero")
    parser.add_argument("--headroom-json", default="reports/official_closed_loop_scaleup_result.json")
    parser.add_argument("--json-output", default="reports/mtf_vla/development_audit.json")
    parser.add_argument("--md-output", default="reports/mtf_vla/development_audit.md")
    parser.add_argument("--split-output", default="reports/mtf_vla/split_manifest.json")
    parser.add_argument("--score-output", default="reports/mtf_vla/frame_score_summary.json")
    parser.add_argument("--retention-output", default="reports/mtf_vla/base_retention_manifest.json")
    parser.add_argument("--validation-json-output", default="reports/mtf_vla/validation_search.json")
    parser.add_argument("--validation-md-output", default="reports/mtf_vla/validation_search.md")
    parser.add_argument("--selected-config-output", default="reports/mtf_vla/selected_config.json")
    parser.add_argument("--selected-training-manifest-output", default="reports/mtf_vla/selected_training_manifest.json")
    args = parser.parse_args()

    artifact = _read_json(Path(args.prediction_artifact))
    headroom = _read_json(Path(args.headroom_json)) if Path(args.headroom_json).exists() else {}
    state_by_index = _load_state_by_index(Path(args.dataset_root))
    summary: dict[str, Any] = {"mode": args.mode, "prediction_artifact": str(args.prediction_artifact)}
    if args.mode in {"audit", "all"}:
        report = audit_mtf_records(
            artifact["records"],
            state_by_index=state_by_index,
            base_headroom_summary=headroom,
        )
        report = {**report, "source_prediction_artifact": str(args.prediction_artifact)}
        _write_json(Path(args.json_output), report)
        _write_md(Path(args.md_output), report)
        _write_json(Path(args.split_output), report["split_manifest"])
        _write_json(Path(args.score_output), report["frame_score_summary"])
        _write_json(Path(args.retention_output), report["base_retention_target_manifest"])
        summary.update(
            {
                "audit_decision": report["final_decision"],
                "scoreable_records": report["scoreable_records"],
                "selected_task_count": report["selected_task_count"],
                "high_low_score_gap": report["high_low_score_gap"],
                "gripper_transition_fraction": report["gripper_transition_fraction"],
                "hard_stop_count": len(report["hard_stop_reasons"]),
            }
        )
    if args.mode in {"validation", "all"}:
        validation = run_validation_search(
            artifact["records"],
            state_by_index=state_by_index,
            base_headroom_summary=headroom,
        )
        validation = {**validation, "source_prediction_artifact": str(args.prediction_artifact)}
        _write_json(Path(args.validation_json_output), validation)
        _write_validation_md(Path(args.validation_md_output), validation)
        if validation.get("selected_config"):
            _write_json(Path(args.selected_config_output), validation["selected_config"])
        if validation.get("selected_training_manifest"):
            _write_json(Path(args.selected_training_manifest_output), validation["selected_training_manifest"])
        summary.update(
            {
                "validation_decision": validation["final_decision"],
                "selected_config": (validation.get("selected_config") or {}).get("config_id"),
                "tried_config_count": validation["tried_config_count"],
            }
        )
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
