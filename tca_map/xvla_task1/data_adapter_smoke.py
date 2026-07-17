"""Tiny X-VLA-format data-adapter smoke for BR-XVLA.

This module materializes a small, local X-VLA-compatible LIBERO meta/HDF5
adapter from the task-1 HDF5 demos and pulls one sample through X-VLA's official
data reader. It performs no model loading, no training, no backward pass, and no
optimizer step.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import cv2
from scipy.spatial.transform import Rotation as R

DEFAULT_HDF5_PATH = Path(
    "/mnt/c/assets/data/libero/libero_10/"
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5"
)
DEFAULT_XVLA_ROOT = Path("/mnt/c/assets/repos/X-VLA")
TASK_DESCRIPTION = "put both the cream cheese box and the butter in the basket"


def _demo_sort_key(name: str) -> tuple[int, str]:
    prefix, _, suffix = str(name).rpartition("_")
    if prefix == "demo" and suffix.isdigit():
        return (int(suffix), str(name))
    return (10**9, str(name))


def _rot6d_from_scalar_first_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    if quat.ndim != 2 or quat.shape[1] != 4:
        raise ValueError(f"quat must be [T, 4], got {quat.shape}")
    matrices = R.from_quat(quat, scalar_first=True).as_matrix()
    return matrices[:, :, :2].reshape(quat.shape[0], 6)


def build_abs_action_6d(robot_states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Build X-VLA LIBERO `abs_action_6d` rows from LIBERO robot state/actions."""

    robot_states = np.asarray(robot_states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if robot_states.ndim != 2 or robot_states.shape[1] < 9:
        raise ValueError(f"robot_states must be [T, >=9], got {robot_states.shape}")
    if actions.ndim != 2 or actions.shape[0] != robot_states.shape[0] or actions.shape[1] < 7:
        raise ValueError(f"actions must be [T, >=7] and align with robot_states, got {actions.shape}")
    pos = robot_states[:, 2:5]
    rot6d = _rot6d_from_scalar_first_quat(robot_states[:, 5:9])
    gripper_raw = actions[:, 6:7]
    return np.concatenate([pos, rot6d, gripper_raw], axis=1).astype(np.float32)


def _write_encoded_rgb_frames(handle: h5py.File, name: str, frames: np.ndarray) -> None:
    frames = np.asarray(frames, dtype=np.uint8)
    if frames.ndim != 4 or frames.shape[-1] != 3:
        raise ValueError(f"{name} frames must be [T, H, W, 3], got {frames.shape}")
    dtype = h5py.vlen_dtype(np.dtype("uint8"))
    dataset = handle.create_dataset(name, shape=(frames.shape[0],), dtype=dtype)
    for index, frame in enumerate(frames):
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise ValueError(f"failed to PNG-encode {name}[{index}]")
        dataset[index] = np.asarray(encoded, dtype=np.uint8).reshape(-1)


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


def _install_mmengine_fileio_shim_if_needed() -> bool:
    if importlib.util.find_spec("mmengine") is not None:
        return False

    def _get(path: str) -> bytes:
        return Path(path).read_bytes()

    def _isdir(path: str) -> bool:
        return Path(path).is_dir()

    def _list_dir_or_file(path: str, suffix: str = "", recursive: bool = False, list_dir: bool = False) -> list[str]:
        root = Path(path)
        iterator = root.rglob("*") if recursive else root.iterdir()
        out: list[str] = []
        for item in iterator:
            if item.is_dir() and not list_dir:
                continue
            if suffix and not str(item).endswith(suffix):
                continue
            out.append(str(item.relative_to(root)))
        return out

    def _join_path(*parts: str) -> str:
        return str(Path(parts[0]).joinpath(*parts[1:]))

    mmengine = types.ModuleType("mmengine")
    mmengine.fileio = types.SimpleNamespace(  # type: ignore[attr-defined]
        get=_get,
        isdir=_isdir,
        list_dir_or_file=_list_dir_or_file,
        join_path=_join_path,
    )
    sys.modules["mmengine"] = mmengine
    return True


def smoke_xvla_reader(xvla_root: Path, meta_path: Path) -> dict[str, Any]:
    shim_used = _install_mmengine_fileio_shim_if_needed()
    root = str(xvla_root)
    if root in sys.path:
        sys.path.remove(root)
    sys.path.insert(0, root)
    from datasets.dataset import InfiniteDataReader  # type: ignore

    reader = InfiniteDataReader(
        str(meta_path),
        num_actions=30,
        num_views=3,
        training=False,
        action_mode="ee6d",
    )
    sample = next(iter(reader))
    summary: dict[str, Any] = {"sample_keys": sorted(str(key) for key in sample.keys()), "mmengine_fileio_shim_used": shim_used}
    for key, value in sample.items():
        if hasattr(value, "shape"):
            summary[key] = {
                "shape": [int(x) for x in value.shape],
                "dtype": str(getattr(value, "dtype", None)),
                "finite": bool(np.isfinite(value.detach().cpu().numpy()).all()) if hasattr(value, "detach") else True,
            }
        else:
            summary[key] = str(value)
    return summary


def run_data_adapter_smoke(source_hdf5: Path, output_dir: Path, xvla_root: Path, demo_names: list[str]) -> dict[str, Any]:
    if not demo_names:
        with h5py.File(source_hdf5, "r") as handle:
            names = sorted([str(name) for name in handle["data"].keys()], key=_demo_sort_key)
        demo_names = [names[0]]
    materialized = materialize_adapter(source_hdf5, output_dir, demo_names)
    reader_summary = smoke_xvla_reader(xvla_root, Path(materialized["meta_path"]))
    report = {
        "schema_version": "2026-07-17.epoch5_br_xvla_data_adapter_smoke.v1",
        "stage": "epoch_5_br_xvla_data_adapter_smoke",
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
    report["decision"] = "BR_XVLA_DATA_ADAPTER_SMOKE_PASS" if report["passed"] else "BR_XVLA_DATA_ADAPTER_SMOKE_FAIL"
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
