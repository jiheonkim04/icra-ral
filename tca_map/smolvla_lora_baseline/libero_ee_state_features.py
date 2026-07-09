"""Canonical LIBERO EEF state features for fixed SmolVLA 7D adapters.

The local LIBERO HDF5 files expose ``obs/ee_states`` as 6D
``ee_pos + ee_ori``. The live RoboSuite/LIBERO env exposes
``robot0_eef_pos`` and ``robot0_eef_quat`` instead. This module keeps both
paths in one place and converts live quaternions to the HDF5-compatible
orientation convention.
"""

from __future__ import annotations

from typing import Any

import numpy as np


ORIENTATION_CONVENTION = "xyzw_quaternion_axis_angle_0_to_2pi"
FEATURE_SCHEMA = "LIBERO_EE_STATES_6D_PLUS_TIMESTEP_FRACTION"


def _require_shape(name: str, values: np.ndarray, width: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size < int(width):
        raise ValueError(f"{name} must contain at least {width} values, got {arr.size}")
    arr = arr[: int(width)].astype(np.float32)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains non-finite values")
    return arr


def quat_xyzw_to_hdf5_axis_angle(quat_xyzw: Any) -> np.ndarray:
    """Convert live RoboSuite XYZW quaternion to HDF5 ``ee_ori``.

    HDF5 ``ee_ori`` matches the noncanonical axis-angle branch produced from an
    XYZW quaternion with angle ``2 * atan2(norm(vector), w)``. The angle is not
    folded back to ``[-pi, pi]``; values just above pi are preserved, which is
    why SciPy's canonical ``as_rotvec`` branch does not match these files.
    """

    quat = _require_shape("robot0_eef_quat", np.asarray(quat_xyzw, dtype=np.float64), 4).astype(np.float64)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-12:
        raise ValueError("robot0_eef_quat has near-zero norm")
    quat = quat / norm
    vector = quat[:3]
    scalar = float(quat[3])
    vector_norm = float(np.linalg.norm(vector))
    if vector_norm <= 1e-12:
        return np.zeros((3,), dtype=np.float32)
    angle = 2.0 * np.arctan2(vector_norm, scalar)
    axis_angle = (vector / vector_norm) * angle
    return axis_angle.astype(np.float32)


def build_hdf5_ee_states(obs_group: Any, timestep: int) -> tuple[np.ndarray, dict[str, Any]]:
    if "ee_states" in obs_group:
        ee = _require_shape("hdf5 obs/ee_states", obs_group["ee_states"][int(timestep)], 6)
        source = "hdf5_obs_ee_states"
    elif "ee_pos" in obs_group and "ee_ori" in obs_group:
        pos = _require_shape("hdf5 obs/ee_pos", obs_group["ee_pos"][int(timestep)], 3)
        ori = _require_shape("hdf5 obs/ee_ori", obs_group["ee_ori"][int(timestep)], 3)
        ee = np.concatenate([pos, ori], axis=0).astype(np.float32)
        source = "hdf5_obs_ee_pos_plus_ee_ori"
    else:
        raise ValueError("HDF5 obs group lacks ee_states and ee_pos/ee_ori")
    return ee, {
        "source": source,
        "feature_shape": list(ee.shape),
        "orientation_convention": ORIENTATION_CONVENTION,
        "uses_quat_first3_fallback": False,
    }


def build_live_ee_states(obs: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    if "ee_states" in obs:
        ee = _require_shape("live obs ee_states", obs["ee_states"], 6)
        source = "live_obs_ee_states"
    elif "ee_pos" in obs and "ee_ori" in obs:
        pos = _require_shape("live obs ee_pos", obs["ee_pos"], 3)
        ori = _require_shape("live obs ee_ori", obs["ee_ori"], 3)
        ee = np.concatenate([pos, ori], axis=0).astype(np.float32)
        source = "live_obs_ee_pos_plus_ee_ori"
    elif "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
        pos = _require_shape("live obs robot0_eef_pos", obs["robot0_eef_pos"], 3)
        ori = quat_xyzw_to_hdf5_axis_angle(obs["robot0_eef_quat"])
        ee = np.concatenate([pos, ori], axis=0).astype(np.float32)
        source = "live_obs_robot0_eef_pos_plus_xyzw_axis_angle_0_to_2pi"
    else:
        raise ValueError("live observation lacks ee_states, ee_pos/ee_ori, or robot0_eef_pos/robot0_eef_quat")
    return ee, {
        "source": source,
        "feature_shape": list(ee.shape),
        "orientation_convention": ORIENTATION_CONVENTION,
        "uses_quat_first3_fallback": False,
    }


def build_hdf5_feature(obs_group: Any, timestep: int, action_length: int) -> tuple[np.ndarray, dict[str, Any]]:
    ee, meta = build_hdf5_ee_states(obs_group, int(timestep))
    fraction = np.asarray([float(timestep) / max(1, int(action_length) - 1)], dtype=np.float32)
    feature = np.concatenate([ee, fraction], axis=0).astype(np.float32)
    meta = {**meta, "schema": FEATURE_SCHEMA, "feature_shape": list(feature.shape), "timestep_fraction": float(fraction[0])}
    return feature, meta


def build_live_feature(obs: dict[str, Any], timestep_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    ee, meta = build_live_ee_states(obs)
    fraction = np.asarray([float(timestep_fraction)], dtype=np.float32)
    feature = np.concatenate([ee, fraction], axis=0).astype(np.float32)
    meta = {**meta, "schema": FEATURE_SCHEMA, "feature_shape": list(feature.shape), "timestep_fraction": float(fraction[0])}
    return feature, meta


def old_quat_first3_feature(obs: dict[str, Any], timestep_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    if "robot0_eef_pos" not in obs or "robot0_eef_quat" not in obs:
        raise ValueError("old quat[:3] feature requires robot0_eef_pos and robot0_eef_quat")
    pos = _require_shape("live obs robot0_eef_pos", obs["robot0_eef_pos"], 3)
    quat = _require_shape("live obs robot0_eef_quat", obs["robot0_eef_quat"], 4)
    ee = np.concatenate([pos, quat[:3]], axis=0).astype(np.float32)
    feature = np.concatenate([ee, np.asarray([float(timestep_fraction)], dtype=np.float32)], axis=0).astype(np.float32)
    return feature, {
        "source": "live_obs_robot0_eef_pos_plus_quat_first3_legacy_bad",
        "schema": FEATURE_SCHEMA,
        "feature_shape": list(feature.shape),
        "orientation_convention": "legacy_quat_first3_not_hdf5_compatible",
        "uses_quat_first3_fallback": True,
        "timestep_fraction": float(timestep_fraction),
    }
