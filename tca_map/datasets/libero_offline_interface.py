"""Safe offline LIBERO-style dataset interface inspection.

The functions here inspect tiny local files only. They do not import LIBERO,
RoboSuite, MuJoCo, VLA models, or run training/rollouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tca_map.datasets.libero_metadata_subset import DATASET_EXTENSIONS, read_asset_paths

SCHEMA_VERSION = "tca-map-libero-offline-interface-smoke-v0"


def find_dataset_files(root: Path | None, max_files: int = 20) -> list[Path]:
    if root is None or not root.exists():
        return []
    files: list[Path] = []
    for path in root.rglob("*"):
        if path.is_file() and path.suffix.lower() in DATASET_EXTENSIONS:
            files.append(path)
            if len(files) >= max_files:
                break
    return files


def _summarize_record(record: dict[str, Any]) -> dict:
    action = record.get("expert_action", record.get("action", []))
    if not isinstance(action, list):
        action = []
    return {
        "has_instruction": isinstance(record.get("instruction") or record.get("language"), str),
        "has_target": "target" in record or "target_object" in record,
        "has_action": bool(action),
        "action_dim": len(action),
    }


def inspect_jsonl(path: Path, max_records: int = 4) -> dict:
    records: list[dict] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"line {line_number}: {exc}")
            continue
        if isinstance(payload, dict):
            records.append(payload)
        if len(records) >= max_records:
            break
    summaries = [_summarize_record(record) for record in records]
    return {
        "reader": "jsonl",
        "record_count_sampled": len(records),
        "record_summaries": summaries,
        "errors": errors,
        "interface_ready": bool(records and not errors and all(item["has_instruction"] and item["has_action"] for item in summaries)),
    }


def inspect_json(path: Path, max_records: int = 4) -> dict:
    errors: list[str] = []
    try:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
    except json.JSONDecodeError as exc:
        return {"reader": "json", "record_count_sampled": 0, "record_summaries": [], "errors": [str(exc)], "interface_ready": False}
    if isinstance(payload, dict) and isinstance(payload.get("samples"), list):
        records = [item for item in payload["samples"] if isinstance(item, dict)]
    elif isinstance(payload, list):
        records = [item for item in payload if isinstance(item, dict)]
    elif isinstance(payload, dict):
        records = [payload]
    else:
        records = []
        errors.append("JSON payload is not a dict, list, or {'samples': [...]}")
    records = records[:max_records]
    summaries = [_summarize_record(record) for record in records]
    return {
        "reader": "json",
        "record_count_sampled": len(records),
        "record_summaries": summaries,
        "errors": errors,
        "interface_ready": bool(records and not errors and all(item["has_instruction"] and item["has_action"] for item in summaries)),
    }


def inspect_npz(path: Path) -> dict:
    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        return {"reader": "npz", "errors": [f"numpy unavailable: {exc}"], "interface_ready": False}
    try:
        data = np.load(path, allow_pickle=False)
        arrays = {key: list(data[key].shape) for key in data.files}
    except Exception as exc:
        return {"reader": "npz", "errors": [str(exc)], "interface_ready": False}
    has_action = any(key in arrays for key in ("action", "actions", "expert_action", "expert_actions"))
    return {"reader": "npz", "arrays": arrays, "errors": [], "interface_ready": has_action}


def inspect_hdf5(path: Path, max_dataset_entries: int = 12) -> dict:
    try:
        import h5py  # type: ignore
    except Exception as exc:
        return {
            "reader": "hdf5",
            "errors": [f"h5py unavailable for HDF5 inspection: {exc}"],
            "interface_ready": False,
            "requires_optional_reader": "h5py",
        }
    keys_sample: list[str] = []
    datasets_sample: dict[str, list[int]] = {}
    dataset_count = 0
    action_dataset_paths: list[str] = []
    try:
        with h5py.File(path, "r") as handle:
            def visitor(name: str, obj: Any) -> None:
                nonlocal dataset_count
                if len(keys_sample) < 12:
                    keys_sample.append(name)
                if hasattr(obj, "shape"):
                    dataset_count += 1
                    if len(datasets_sample) < max_dataset_entries:
                        datasets_sample[name] = list(obj.shape)
                    if "action" in name.lower():
                        action_dataset_paths.append(name)

            handle.visititems(visitor)
    except Exception as exc:
        return {"reader": "hdf5", "errors": [str(exc)], "interface_ready": False}
    has_action = bool(action_dataset_paths)
    return {
        "reader": "hdf5",
        "keys_sample": keys_sample,
        "dataset_count": dataset_count,
        "dataset_sample_limit": max_dataset_entries,
        "datasets_sample": datasets_sample,
        "action_dataset_paths_sample": action_dataset_paths[:5],
        "errors": [],
        "interface_ready": has_action,
    }


def inspect_dataset_file(path: Path) -> dict:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        details = inspect_jsonl(path)
    elif suffix == ".json":
        details = inspect_json(path)
    elif suffix == ".npz":
        details = inspect_npz(path)
    elif suffix in {".hdf5", ".h5"}:
        details = inspect_hdf5(path)
    else:
        details = {"reader": "unsupported", "errors": [f"unsupported extension: {suffix}"], "interface_ready": False}
    return {
        "path": str(path),
        "extension": suffix,
        **details,
    }


def build_offline_interface_report(data_root: Path, max_files: int = 1) -> dict:
    files = find_dataset_files(data_root, max_files=max_files)
    inspections = [inspect_dataset_file(path) for path in files]
    ready = bool(inspections and any(item.get("interface_ready") for item in inspections))
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "check_only": True,
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "simulator_executed": False,
            "rollouts_performed": False,
            "heavy_model_imports_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
        },
        "data_root": str(data_root),
        "data_root_exists": data_root.exists(),
        "dataset_files_detected": bool(files),
        "dataset_file_count_sampled": len(files),
        "file_inspections": inspections,
        "ready_for_offline_interface_smoke": ready,
        "ready_for_rollout": False,
        "decision": "proceed" if ready else "stop",
        "reason": (
            "at least one tiny local file has instruction/action fields readable by the offline interface"
            if ready
            else "no local tiny dataset file with readable instruction/action fields was found"
        ),
        "recommended_next_step": (
            "Run a bounded offline interface smoke on the tiny local file only; do not train, rollout, or make paper claims."
            if ready
            else "Place a documented tiny LIBERO-style demo file under LIBERO_DATA_ROOT or keep using metadata-only split plumbing."
        ),
    }


def write_reports(report: dict, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LIBERO Offline Interface Smoke Gate",
        "",
        f"- decision: `{report['decision']}`",
        f"- data root exists: `{report['data_root_exists']}`",
        f"- dataset files detected: `{report['dataset_files_detected']}`",
        f"- ready for offline interface smoke: `{report['ready_for_offline_interface_smoke']}`",
        f"- ready for rollout: `{report['ready_for_rollout']}`",
        "",
        report["reason"],
        "",
        "This gate performs no downloads, GPU jobs, training, rollouts, simulator execution, heavy VLA imports, token access, OpenVLA-OFT execution, or paper-grade claims.",
        "",
        "## Next Step",
        report["recommended_next_step"],
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-file", default="configs/paths.local.yaml")
    parser.add_argument("--data-root", default="")
    parser.add_argument("--max-files", type=int, default=1)
    parser.add_argument("--report-json", default="reports/libero_offline_interface_smoke_report.json")
    parser.add_argument("--report-md", default="reports/libero_offline_interface_smoke_report.md")
    args = parser.parse_args()

    paths = read_asset_paths(Path(args.paths_file))
    data_root = Path(args.data_root or paths.get("libero_data_root", "C:/assets/data/libero"))
    report = build_offline_interface_report(data_root=data_root, max_files=args.max_files)
    write_reports(report, json_path=Path(args.report_json), markdown_path=Path(args.report_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
