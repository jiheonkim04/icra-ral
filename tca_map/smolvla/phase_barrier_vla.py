"""PhaseBarrier-VLA utilities.

The heavy runner owns SmolVLA and LIBERO imports.  This module keeps the
method's learned feasibility field and action projection small and testable.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


PHASES = ("approach", "contact", "transport", "placement")
BASE_FEATURES = (
    "bias",
    "translation_norm",
    "xy_norm",
    "z_action_mean",
    "rotation_norm",
    "gripper_mean",
    "eef_z",
    "eef_xy_norm",
    "step_fraction",
)


def _as_action(action: Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2:
        raise ValueError(f"action must be 1D or 2D, got shape {arr.shape}")
    if arr.shape[1] < 7:
        raise ValueError(f"expected at least 7 action dimensions, got {arr.shape[1]}")
    return arr


def normalize_phase(phase: str) -> str:
    phase = str(phase).lower()
    aliases = {
        "grasp": "contact",
        "grasp_contact": "contact",
        "lift": "transport",
        "release": "placement",
    }
    phase = aliases.get(phase, phase)
    if phase not in PHASES:
        raise ValueError(f"unknown phase: {phase}")
    return phase


def action_feature_dict(
    action: Sequence[Sequence[float]] | np.ndarray,
    *,
    eef: Sequence[float] | None = None,
    step_fraction: float = 0.0,
) -> dict[str, float]:
    arr = _as_action(action)
    translation = arr[:, :3]
    rotation = arr[:, 3:6]
    grip = arr[:, 6]
    if eef is None:
        eef_arr = np.zeros(3, dtype=np.float64)
    else:
        eef_arr = np.asarray(eef, dtype=np.float64).reshape(-1)
        if eef_arr.size < 3:
            raise ValueError("eef must contain at least three values")
        eef_arr = eef_arr[:3]
    return {
        "bias": 1.0,
        "translation_norm": float(np.linalg.norm(translation)),
        "xy_norm": float(np.linalg.norm(translation[:, :2])),
        "z_action_mean": float(np.mean(translation[:, 2])),
        "rotation_norm": float(np.linalg.norm(rotation)),
        "gripper_mean": float(np.mean(grip)),
        "eef_z": float(eef_arr[2]),
        "eef_xy_norm": float(np.linalg.norm(eef_arr[:2])),
        "step_fraction": float(np.clip(step_fraction, 0.0, 1.0)),
    }


def vectorize_features(features: Mapping[str, float], phase: str, *, use_phase: bool = True) -> np.ndarray:
    values = [float(features.get(name, 0.0)) for name in BASE_FEATURES]
    if use_phase:
        normalized = normalize_phase(phase)
        values.extend(1.0 if normalized == item else 0.0 for item in PHASES)
    return np.asarray(values, dtype=np.float64)


@dataclass(frozen=True)
class BarrierRecord:
    phase: str
    features: dict[str, float]
    label: float
    weight: float = 1.0


@dataclass
class PhaseBarrierModel:
    weights: list[float]
    use_phase: bool = True
    threshold: float = 0.0

    def score(self, features: Mapping[str, float], phase: str) -> float:
        x = vectorize_features(features, phase, use_phase=self.use_phase)
        w = np.asarray(self.weights, dtype=np.float64)
        if x.shape[0] != w.shape[0]:
            raise ValueError(f"feature width {x.shape[0]} does not match weights {w.shape[0]}")
        return float(x @ w)

    def to_json(self) -> dict[str, Any]:
        return {
            "weights": [float(value) for value in self.weights],
            "use_phase": bool(self.use_phase),
            "threshold": float(self.threshold),
        }

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "PhaseBarrierModel":
        return cls(
            weights=[float(value) for value in payload["weights"]],
            use_phase=bool(payload.get("use_phase", True)),
            threshold=float(payload.get("threshold", 0.0)),
        )


def fit_phase_barrier(
    records: Iterable[BarrierRecord],
    *,
    use_phase: bool = True,
    l2: float = 1e-3,
) -> PhaseBarrierModel:
    rows = list(records)
    if not rows:
        raise ValueError("cannot fit PhaseBarrierModel with no records")
    x = np.stack([vectorize_features(row.features, row.phase, use_phase=use_phase) for row in rows], axis=0)
    y = np.asarray([1.0 if float(row.label) > 0.0 else -1.0 for row in rows], dtype=np.float64)
    weights = np.asarray([max(float(row.weight), 1e-6) for row in rows], dtype=np.float64)
    sqrt_w = np.sqrt(weights)
    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w
    reg = float(l2) * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    solved = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)
    return PhaseBarrierModel(weights=[float(value) for value in solved], use_phase=use_phase)


def project_action_with_barrier(
    action: Sequence[Sequence[float]] | np.ndarray,
    *,
    margin: float,
    phase: str,
    threshold: float = 0.0,
    strength: float = 0.35,
    max_scale_reduction: float = 0.55,
    z_boost: float = 0.015,
    clip: float = 1.0,
) -> np.ndarray:
    arr = _as_action(action).copy()
    risk = max(0.0, float(threshold) - float(margin))
    if risk <= 0.0:
        return np.clip(arr, -float(clip), float(clip))
    phase = normalize_phase(phase)
    scale = 1.0 - min(float(max_scale_reduction), float(strength) * risk)
    arr[:, :2] *= scale
    arr[:, 3:6] *= scale
    if phase in {"contact", "transport"}:
        arr[:, 2] += min(0.04, float(z_boost) * risk)
    elif phase == "placement":
        arr[:, 2] *= max(scale, 0.25)
    else:
        arr[:, 2] *= scale
    return np.clip(arr, -float(clip), float(clip))


def simple_global_damping(action: Sequence[Sequence[float]] | np.ndarray, *, scale: float = 0.8) -> np.ndarray:
    arr = _as_action(action).copy()
    arr[:, :6] *= float(scale)
    return np.clip(arr, -1.0, 1.0)


def pre_vla_style_halt_proxy(
    action: Sequence[Sequence[float]] | np.ndarray,
    *,
    margin: float,
    threshold: float = 0.0,
) -> np.ndarray:
    arr = _as_action(action).copy()
    if float(margin) < float(threshold):
        arr[:, :6] = 0.0
    return np.clip(arr, -1.0, 1.0)


def infer_phase_from_step(step_index: int, max_steps: int) -> str:
    frac = 0.0 if max_steps <= 1 else float(step_index) / float(max_steps - 1)
    if frac < 0.25:
        return "approach"
    if frac < 0.45:
        return "contact"
    if frac < 0.78:
        return "transport"
    return "placement"
