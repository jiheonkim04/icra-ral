"""Run EAC-VLA development-only Stage 0 audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.eac_vla import PROPOSAL_HASH, audit_eac_stage0  # noqa: E402


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
    dispersion = report["dispersion_manifest"]
    queue = report["queue_surface_manifest"]
    split = report["split_manifest"]
    passthrough = report["action_value_passthrough_summary"]
    lines = [
        "# EAC-VLA Stage 0 Audit",
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
        f"- scoreable validation records: `{report['scoreable_validation_records']}`",
        f"- validation unique frames: `{report['validation_unique_frames']}`",
        f"- reserved records not used for tuning: `{report['reserved_records_not_used_for_tuning']}`",
        f"- queue helper present: `{queue['queue_helper_present']}`",
        f"- expected chunk shape recorded: `{queue['chunk_shape_ok']}`",
        f"- full chunk values available in artifact: `{queue['full_chunk_values_available_in_artifact']}`",
        f"- runtime full-chunk check required before validation search: `{queue['runtime_full_chunk_check_required_before_validation_search']}`",
        f"- first-two dispersion p95: `{dispersion['first_two_dispersion_summary']['p95']}`",
        f"- first-two dispersion nonzero fraction: `{dispersion['first_two_dispersion_summary']['nonzero_fraction']}`",
        f"- commitment counts: `{dispersion['commitment_counts']}`",
        f"- max commitment share: `{dispersion['max_commitment_share']}`",
        f"- passthrough max abs error: `{passthrough['max']}`",
        "",
        "Split manifest:",
        "",
        "```json",
        json.dumps(split, indent=2, sort_keys=True),
        "```",
        "",
        "Queue surface manifest:",
        "",
        "```json",
        json.dumps(queue, indent=2, sort_keys=True),
        "```",
        "",
        "Dispersion manifest summary:",
        "",
        "```json",
        json.dumps({k: v for k, v in dispersion.items() if k != "frame_metric_preview"}, indent=2, sort_keys=True),
        "```",
        "",
        "Stage 0 limitations:",
    ]
    lines.extend(f"- `{item}`" for item in report.get("stage_0_limitations", []))
    lines.extend(["", "Hard stop reasons:"])
    hard_stops = list(report.get("hard_stop_reasons") or [])
    if hard_stops:
        lines.extend(f"- `{reason}`" for reason in hard_stops)
    else:
        lines.append("- none")
    lines.extend(["", f"Next step: {report['next_step']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _queue_helper_present(path: Path) -> bool:
    text = path.read_text(encoding="utf-8")
    return "def _action_queue_len" in text and "def _queue_owner" in text


def _previous_preflight_chunk_shape(path: Path) -> list[int] | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    for record in payload.get("records") or []:
        shape = record.get("base_action_chunk_shape") or record.get("policy_output_shape")
        if shape:
            if len(shape) == 3 and int(shape[0]) == 1:
                return [int(shape[1]), int(shape[2])]
            return [int(item) for item in shape]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit"], default="audit")
    parser.add_argument("--canonical-base-artifact", default="reports/canonical_frozen_base_prediction_artifact.json")
    parser.add_argument("--official-queue-source", default="tca_map/smolvla/official_closed_loop_scaleup.py")
    parser.add_argument("--previous-preflight", default="reports/marc_vla/stage_a_preflight.json")
    parser.add_argument("--json-output", default="reports/eac_vla/stage_0_audit.json")
    parser.add_argument("--md-output", default="reports/eac_vla/stage_0_audit.md")
    parser.add_argument("--queue-output", default="reports/eac_vla/queue_surface_manifest.json")
    parser.add_argument("--dispersion-output", default="reports/eac_vla/dispersion_manifest.json")
    parser.add_argument("--split-output", default="reports/eac_vla/split_manifest.json")
    args = parser.parse_args()

    artifact = _read_json(Path(args.canonical_base_artifact))
    report = audit_eac_stage0(
        artifact["records"],
        queue_helper_present=_queue_helper_present(Path(args.official_queue_source)),
        previous_preflight_chunk_shape=_previous_preflight_chunk_shape(Path(args.previous_preflight)),
    )
    report = {
        **report,
        "date_kst": DATE_KST,
        "mode": args.mode,
        "source_canonical_base_artifact": str(args.canonical_base_artifact),
        "source_official_queue_source": str(args.official_queue_source),
        "source_previous_preflight": str(args.previous_preflight),
    }
    _write_json(Path(args.json_output), report)
    _write_md(Path(args.md_output), report)
    _write_json(Path(args.queue_output), report["queue_surface_manifest"])
    _write_json(Path(args.dispersion_output), report["dispersion_manifest"])
    _write_json(Path(args.split_output), report["split_manifest"])
    summary = {
        "mode": args.mode,
        "audit_decision": report["final_decision"],
        "audit_hard_stop_count": len(report["hard_stop_reasons"]),
        "json_output": args.json_output,
        "md_output": args.md_output,
    }
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
