"""Tiny X-VLA-format data-adapter smoke for MPR-XVLA.

This module materializes a small, local X-VLA-compatible LIBERO meta/HDF5
adapter from the task-6 HDF5 demos and pulls one sample through X-VLA's official
data reader. It performs no model loading, no training, no backward pass, and no
optimizer step.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from tca_map.xvla_task1.data_adapter_smoke import (
    _demo_sort_key,
    _install_mmengine_fileio_shim_if_needed,
    _write_encoded_rgb_frames,
    build_abs_action_6d,
    smoke_xvla_reader,
)

DEFAULT_HDF5_PATH = Path(
    "/mnt/c/assets/data/libero/libero_10/"
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5"
)
DEFAULT_XVLA_ROOT = Path("/mnt/c/assets/repos/X-VLA")
TASK_DESCRIPTION = "put the white mug on the plate and put the chocolate pudding to the right of the plate"


def materialize_xvla_demo(source_hdf5: Path, demo_name: str, output_hdf5: Path) -> dict[str, Any]:
    output_hdf5.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(source_hdf5, "r") as source:
        demo = source["data"][demo_name]
        actions = np.asarray(demo["actions"], dtype=np.float64)
        robot_states = np.asarray(demo["robot_states"], dtype=np.float64)
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


def run_data_adapter_smoke(source_hdf5: Path, output_dir: Path, xvla_root: Path, demo_names: list[str]) -> dict[str, Any]:
    if not demo_names:
        with h5py.File(source_hdf5, "r") as handle:
            names = sorted([str(name) for name in handle["data"].keys()], key=_demo_sort_key)
        demo_names = [names[0]]
    materialized = materialize_adapter(source_hdf5, output_dir, demo_names)
    reader_summary = smoke_xvla_reader(xvla_root, Path(materialized["meta_path"]))
    report = {
        "schema_version": "2026-07-17.epoch5_mpr_xvla_data_adapter_smoke.v1",
        "stage": "epoch_5_mpr_xvla_data_adapter_smoke",
        "method": "MPR-XVLA",
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
            "dataset_reader_instantiated": True,
        },
        "materialized": materialized,
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
    )
    report["decision"] = "MPR_XVLA_DATA_ADAPTER_SMOKE_PASS" if report["passed"] else "MPR_XVLA_DATA_ADAPTER_SMOKE_FAIL"
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
