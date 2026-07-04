"""Tiny offline LIBERO counterfactual split manifest builder.

This module links local LIBERO BDDL task metadata to already-acquired HDF5
demo files. It inspects file structure only and never imports simulators,
VLA models, or runs training/rollouts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

from tca_map.datasets.libero_metadata_subset import (
    DEFAULT_SUITES,
    discover_bddl_tasks,
    make_counterfactual_metadata_pairs,
    read_asset_paths,
    read_subset_config,
)
from tca_map.datasets.libero_offline_interface import inspect_hdf5

SCHEMA_VERSION = "tca-map-libero-offline-counterfactual-split-v0"


def _safe_relative(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root)).replace("\\", "/")
    except ValueError:
        return str(path).replace("\\", "/")


def _task_id_from_demo_path(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_demo"):
        return stem[: -len("_demo")]
    return stem


def _first_action_shape(inspection: dict) -> list[int] | None:
    for key, shape in inspection.get("datasets_sample", {}).items():
        if "action" in key.lower() and isinstance(shape, list):
            return [int(item) for item in shape]
    return None


def build_hdf5_inventory(data_root: Path, max_files: int = 64) -> list[dict]:
    if not data_root.exists():
        return []
    inventory: list[dict] = []
    for path in sorted(data_root.rglob("*.hdf5")):
        inspection = inspect_hdf5(path)
        inventory.append(
            {
                "task_id": _task_id_from_demo_path(path),
                "suite": path.parent.name,
                "path": str(path),
                "relative_path": _safe_relative(path, data_root),
                "reader": inspection.get("reader"),
                "interface_ready": bool(inspection.get("interface_ready")),
                "dataset_count": inspection.get("dataset_count", 0),
                "action_shape_sample": _first_action_shape(inspection),
                "action_dataset_paths_sample": inspection.get("action_dataset_paths_sample", []),
                "errors": inspection.get("errors", []),
            }
        )
        if len(inventory) >= max_files:
            break
    return inventory


def _index_inventory_by_task(inventory: Iterable[dict]) -> dict[str, dict]:
    indexed: dict[str, dict] = {}
    for item in inventory:
        if not item.get("interface_ready"):
            continue
        task_id = str(item.get("task_id", "")).lower()
        if task_id and task_id not in indexed:
            indexed[task_id] = item
    return indexed


def _attach_demo(task: dict, demo: dict) -> dict:
    return {
        **task,
        "metadata_only": False,
        "demo_file": demo["path"],
        "demo_relative_path": demo["relative_path"],
        "demo_suite": demo["suite"],
        "action_shape_sample": demo.get("action_shape_sample"),
        "requires_demo_file": True,
        "requires_simulator": False,
    }


def build_offline_counterfactual_split(
    libero_root: Path,
    libero_data_root: Path,
    suites: Iterable[str] = DEFAULT_SUITES,
    max_tasks_per_suite: int = 4,
    max_demo_files: int = 64,
    max_counterfactual_pairs: int = 8,
) -> dict:
    try:
        import h5py  # noqa: F401

        hdf5_reader_available = True
        hdf5_reader_error = None
    except Exception as exc:  # pragma: no cover - depends on local optional runtime
        hdf5_reader_available = False
        hdf5_reader_error = str(exc)

    tasks = discover_bddl_tasks(
        libero_root=libero_root,
        suites=suites,
        max_tasks_per_suite=max_tasks_per_suite,
    )
    inventory = build_hdf5_inventory(libero_data_root, max_files=max_demo_files) if hdf5_reader_available else []
    inventory_by_task = _index_inventory_by_task(inventory)

    matched_tasks: list[dict] = []
    for task in tasks:
        demo = inventory_by_task.get(str(task["task_id"]).lower())
        if demo:
            matched_tasks.append(_attach_demo(task, demo))

    raw_pairs = make_counterfactual_metadata_pairs(matched_tasks, max_pairs=max_counterfactual_pairs)
    pairs: list[dict] = []
    task_by_id = {task["task_id"]: task for task in matched_tasks}
    for pair in raw_pairs:
        positive = task_by_id[pair["positive_task_id"]]
        counterfactual = task_by_id[pair["counterfactual_task_id"]]
        pairs.append(
            {
                **pair,
                "positive_demo_file": positive["demo_file"],
                "positive_demo_relative_path": positive["demo_relative_path"],
                "counterfactual_demo_file": counterfactual["demo_file"],
                "counterfactual_demo_relative_path": counterfactual["demo_relative_path"],
                "positive_action_shape_sample": positive.get("action_shape_sample"),
                "counterfactual_action_shape_sample": counterfactual.get("action_shape_sample"),
                "offline_proxy_only": True,
                "requires_simulator": False,
                "requires_rollout": False,
                "requires_training": False,
            }
        )

    ready = bool(hdf5_reader_available and matched_tasks and pairs)
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
            "offline_proxy_only": True,
        },
        "paths": {
            "libero_root": str(libero_root),
            "libero_root_exists": libero_root.exists(),
            "libero_data_root": str(libero_data_root),
            "libero_data_root_exists": libero_data_root.exists(),
        },
        "suites": list(suites),
        "hdf5_reader_available": hdf5_reader_available,
        "hdf5_reader_error": hdf5_reader_error,
        "metadata_task_count": len(tasks),
        "hdf5_inventory_count": len(inventory),
        "matched_task_count": len(matched_tasks),
        "standard_examples": matched_tasks,
        "counterfactual_pair_count": len(pairs),
        "counterfactual_pairs": pairs,
        "ready_for_tiny_offline_counterfactual_split": ready,
        "ready_for_tiny_offline_actionmap_tca_comparison": ready,
        "ready_for_rollout": False,
        "decision": "proceed" if ready else "stop",
        "reason": (
            "local LIBERO BDDL tasks and HDF5 demo files were matched into tiny counterfactual pairs"
            if ready
            else "no matched local LIBERO HDF5 counterfactual pairs are ready"
        ),
        "recommended_next_step": (
            "Run a tiny offline ActionMap vs TCA-Map comparison on this manifest only; do not train, rollout, execute simulators, or make paper claims."
            if ready
            else "Keep using metadata-only split plumbing until local HDF5 demo files and h5py are available."
        ),
    }


def write_reports(report: dict, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LIBERO Offline Counterfactual Split Report",
        "",
        f"- decision: `{report['decision']}`",
        f"- HDF5 reader available: `{report['hdf5_reader_available']}`",
        f"- metadata tasks: `{report['metadata_task_count']}`",
        f"- HDF5 inventory files: `{report['hdf5_inventory_count']}`",
        f"- matched tasks: `{report['matched_task_count']}`",
        f"- counterfactual pairs: `{report['counterfactual_pair_count']}`",
        f"- ready for tiny offline counterfactual split: `{report['ready_for_tiny_offline_counterfactual_split']}`",
        f"- ready for tiny offline ActionMap/TCA-Map comparison: `{report['ready_for_tiny_offline_actionmap_tca_comparison']}`",
        f"- ready for rollout: `{report['ready_for_rollout']}`",
        "",
        report["reason"],
        "",
        "This report reads local metadata and HDF5 structure only. It performs no downloads, GPU jobs, training, rollouts, simulator execution, heavy VLA imports, token access, OpenVLA-OFT execution, or paper-grade claims.",
        "",
        "## Next Step",
        report["recommended_next_step"],
        "",
    ]
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paths-file", default="configs/paths.local.yaml")
    parser.add_argument("--config", default="configs/libero_offline_counterfactual_split.yaml")
    parser.add_argument("--libero-root", default="")
    parser.add_argument("--libero-data-root", default="")
    parser.add_argument("--report-json", default="reports/libero_offline_counterfactual_split_report.json")
    parser.add_argument("--report-md", default="reports/libero_offline_counterfactual_split_report.md")
    args = parser.parse_args()

    paths = read_asset_paths(Path(args.paths_file))
    config = read_subset_config(Path(args.config))
    suites = config.get("suites") or list(DEFAULT_SUITES)
    libero_root = Path(args.libero_root or paths.get("libero_root", "C:/assets/repos/LIBERO"))
    libero_data_root = Path(args.libero_data_root or paths.get("libero_data_root", "C:/assets/data/libero"))

    report = build_offline_counterfactual_split(
        libero_root=libero_root,
        libero_data_root=libero_data_root,
        suites=[str(suite) for suite in suites],
        max_tasks_per_suite=int(config.get("max_tasks_per_suite", 4)),
        max_demo_files=int(config.get("max_demo_files", 64)),
        max_counterfactual_pairs=int(config.get("max_counterfactual_pairs", 8)),
    )
    write_reports(report, json_path=Path(args.report_json), markdown_path=Path(args.report_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
