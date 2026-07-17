"""Tiny X-VLA-format data-adapter smoke for R2P-XVLA task 5.

This module materializes a small, local X-VLA-compatible LIBERO meta/HDF5
adapter from the task-5 HDF5 demos and pulls one sample through X-VLA's official
data reader. It performs no model loading, no training, no backward pass, no
optimizer step, no checkpoint write, and no simulator rollout.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tca_map.xvla_spatial_task5.data_audit import SpatialTask5DataAuditConfig, _phase_labels
from tca_map.xvla_task1.data_adapter_smoke import (
    _demo_sort_key,
    _write_encoded_rgb_frames,
    build_abs_action_6d,
    smoke_xvla_reader,
)


DEFAULT_HDF5_PATH = Path(
    "C:/assets/data/libero/libero_spatial/"
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
DEFAULT_XVLA_ROOT = Path("C:/assets/repos/X-VLA")
TASK_DESCRIPTION = "pick up the black bowl on the ramekin and place it on the plate"


def _count_phases(states: np.ndarray) -> dict[str, int]:
    labels = _phase_labels(np.asarray(states, dtype=np.float64), SpatialTask5DataAuditConfig(hdf5_path=DEFAULT_HDF5_PATH))
    phases, counts = np.unique(labels["phase"], return_counts=True)
    out = {str(phase): int(count) for phase, count in zip(phases, counts)}
    for name in ("source_on_ramekin", "transit", "target_on_plate"):
        out.setdefault(name, 0)
    return out


def materialize_xvla_demo(source_hdf5: Path, demo_name: str, output_hdf5: Path) -> dict[str, Any]:
    """Materialize one source demo into X-VLA's LIBERO reader format."""

    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_hdf5, "r") as source:
        demo = source["data"][demo_name]
        actions = np.asarray(demo["actions"], dtype=np.float64)
        robot_states = np.asarray(demo["robot_states"], dtype=np.float64)
        states = np.asarray(demo["states"], dtype=np.float64)
        abs_action_6d = build_abs_action_6d(robot_states, actions)
        agentview = np.asarray(demo["obs"]["agentview_rgb"], dtype=np.uint8)
        wrist = np.asarray(demo["obs"]["eye_in_hand_rgb"], dtype=np.uint8)
    with h5py.File(output_hdf5, "w") as target:
        target.create_dataset("abs_action_6d", data=abs_action_6d, compression="gzip")
        _write_encoded_rgb_frames(target, "agentview_rgb", agentview)
        _write_encoded_rgb_frames(target, "eye_in_hand_rgb", wrist)
        target.create_dataset("language_instruction", data=np.bytes_(TASK_DESCRIPTION))
    return {
        "source_demo": demo_name,
        "output_hdf5": str(output_hdf5),
        "steps": int(abs_action_6d.shape[0]),
        "abs_action_6d_shape": [int(x) for x in abs_action_6d.shape],
        "agentview_shape": [int(x) for x in agentview.shape],
        "eye_in_hand_shape": [int(x) for x in wrist.shape],
        "abs_action_6d_finite": bool(np.isfinite(abs_action_6d).all()),
        "abs_action_6d_min": float(np.min(abs_action_6d)),
        "abs_action_6d_max": float(np.max(abs_action_6d)),
        "phase_step_counts": _count_phases(states),
    }


def materialize_adapter(source_hdf5: Path, output_dir: Path, demo_names: list[str]) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    converted_dir = output_dir / "converted_hdf5"
    converted_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    datalist = []
    for demo_name in demo_names:
        output_hdf5 = converted_dir / f"{demo_name}.hdf5"
        rows.append(materialize_xvla_demo(source_hdf5, demo_name, output_hdf5))
        datalist.append(str(output_hdf5))
    meta = {
        "dataset_name": "libero",
        "datalist": datalist,
        "observation_key": ["agentview_rgb", "eye_in_hand_rgb"],
        "language_instruction_key": "language_instruction",
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {"meta_path": str(meta_path), "converted_dir": str(converted_dir), "converted_demos": rows}


def _default_demo_names(source_hdf5: Path) -> list[str]:
    with h5py.File(source_hdf5, "r") as handle:
        names = sorted([str(name) for name in handle["data"].keys()], key=_demo_sort_key)
    if not names:
        raise ValueError("source HDF5 contains no demos")
    selected = [names[0]]
    if len(names) > 40:
        selected.append(names[40])
    elif len(names) > 1:
        selected.append(names[-1])
    return list(dict.fromkeys(selected))


def run_data_adapter_smoke(source_hdf5: Path, output_dir: Path, xvla_root: Path, demo_names: list[str]) -> dict[str, Any]:
    if not demo_names:
        demo_names = _default_demo_names(source_hdf5)
    materialized = materialize_adapter(source_hdf5, output_dir, demo_names)
    reader_summary = smoke_xvla_reader(xvla_root, Path(materialized["meta_path"]))
    converted = materialized["converted_demos"]
    phase_coverage = {
        name: int(sum(int(row["phase_step_counts"].get(name, 0)) for row in converted))
        for name in ("source_on_ramekin", "transit", "target_on_plate")
    }
    report = {
        "schema_version": "2026-07-18.epoch5_r2p_xvla_task5_data_adapter_smoke.v1",
        "stage": "epoch_5_r2p_xvla_task5_data_adapter_smoke",
        "method": "R2P-XVLA",
        "source_hdf5": str(source_hdf5),
        "output_dir": str(output_dir),
        "xvla_root": str(xvla_root),
        "demo_names": demo_names,
        "policy": {
            "training_happened": False,
            "optimizer_step_happened": False,
            "checkpoint_written": False,
            "model_loaded": False,
            "backward_happened": False,
            "simulator_rollout_happened": False,
            "dataset_reader_instantiated": True,
        },
        "materialized": materialized,
        "phase_coverage": phase_coverage,
        "reader_smoke": reader_summary,
    }
    action_shape = reader_summary.get("action", {}).get("shape")
    proprio_shape = reader_summary.get("proprio", {}).get("shape")
    image_shape = reader_summary.get("image_input", {}).get("shape")
    domain_id = reader_summary.get("domain_id", {})
    report["passed"] = bool(
        action_shape == [30, 20]
        and proprio_shape == [20]
        and image_shape == [3, 3, 224, 224]
        and str(domain_id.get("dtype")) == "torch.int64"
        and all(value > 0 for value in phase_coverage.values())
    )
    report["decision"] = "R2P_XVLA_DATA_ADAPTER_SMOKE_PASS" if report["passed"] else "R2P_XVLA_DATA_ADAPTER_SMOKE_FAIL"
    (output_dir / "result.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-hdf5", type=Path, default=DEFAULT_HDF5_PATH)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--xvla-root", type=Path, default=DEFAULT_XVLA_ROOT)
    parser.add_argument("--demo-name", action="append", default=[])
    args = parser.parse_args(argv)
    report = run_data_adapter_smoke(
        source_hdf5=Path(args.source_hdf5),
        output_dir=Path(args.output_dir),
        xvla_root=Path(args.xvla_root),
        demo_names=[str(name) for name in args.demo_name],
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
