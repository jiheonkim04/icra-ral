"""CensorCredit-VLA utilities.

This module implements the light temporal-credit component for the second
implementation cycle.  It learns whether the current action should be trusted
before giving later recovery credit to the prefix that caused the need for
recovery.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


FEATURES = (
    "bias",
    "action_norm",
    "translation_norm",
    "rotation_norm",
    "gripper",
    "prev_action_norm",
    "action_delta_norm",
    "step_fraction",
)


def _as_action(action: Sequence[float] | Sequence[Sequence[float]] | np.ndarray) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr.reshape(-1)
    if arr.ndim != 1:
        raise ValueError(f"action must flatten to 1D, got shape {arr.shape}")
    if arr.shape[0] < 7:
        raise ValueError(f"expected at least 7 action dims, got {arr.shape[0]}")
    return arr[:7]


def temporal_feature_dict(
    action: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    previous_action: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None = None,
    step_fraction: float = 0.0,
) -> dict[str, float]:
    current = _as_action(action)
    previous = np.zeros_like(current) if previous_action is None else _as_action(previous_action)
    delta = current - previous
    return {
        "bias": 1.0,
        "action_norm": float(np.linalg.norm(current)),
        "translation_norm": float(np.linalg.norm(current[:3])),
        "rotation_norm": float(np.linalg.norm(current[3:6])),
        "gripper": float(current[6]),
        "prev_action_norm": float(np.linalg.norm(previous)),
        "action_delta_norm": float(np.linalg.norm(delta)),
        "step_fraction": float(np.clip(step_fraction, 0.0, 1.0)),
    }


def vectorize_temporal_features(features: Mapping[str, float]) -> np.ndarray:
    return np.asarray([float(features.get(name, 0.0)) for name in FEATURES], dtype=np.float64)


@dataclass(frozen=True)
class CensorRecord:
    features: dict[str, float]
    label: float
    weight: float = 1.0


@dataclass
class CensorCreditModel:
    weights: list[float]
    threshold: float = 0.0

    def score(self, features: Mapping[str, float]) -> float:
        x = vectorize_temporal_features(features)
        w = np.asarray(self.weights, dtype=np.float64)
        if x.shape[0] != w.shape[0]:
            raise ValueError(f"feature width {x.shape[0]} does not match weights {w.shape[0]}")
        return float(x @ w)

    def to_json(self) -> dict[str, Any]:
        return {"weights": [float(value) for value in self.weights], "threshold": float(self.threshold)}

    @classmethod
    def from_json(cls, payload: Mapping[str, Any]) -> "CensorCreditModel":
        return cls(weights=[float(value) for value in payload["weights"]], threshold=float(payload.get("threshold", 0.0)))


def fit_censor_credit(records: Iterable[CensorRecord], *, l2: float = 1e-3) -> CensorCreditModel:
    rows = list(records)
    if not rows:
        raise ValueError("cannot fit CensorCreditModel with no records")
    x = np.stack([vectorize_temporal_features(row.features) for row in rows], axis=0)
    y = np.asarray([1.0 if float(row.label) > 0.0 else -1.0 for row in rows], dtype=np.float64)
    weights = np.asarray([max(float(row.weight), 1e-6) for row in rows], dtype=np.float64)
    sqrt_w = np.sqrt(weights)
    xw = x * sqrt_w[:, None]
    yw = y * sqrt_w
    reg = float(l2) * np.eye(x.shape[1], dtype=np.float64)
    reg[0, 0] = 0.0
    solved = np.linalg.solve(xw.T @ xw + reg, xw.T @ yw)
    return CensorCreditModel(weights=[float(value) for value in solved])


def temporal_hold_blend(
    action: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    previous_action: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None,
    margin: float,
    threshold: float = 0.0,
    hold_strength: float = 0.70,
) -> np.ndarray:
    current = _as_action(action)
    if previous_action is None or float(margin) >= float(threshold):
        return np.clip(current.reshape(1, -1), -1.0, 1.0)
    previous = _as_action(previous_action)
    alpha = float(np.clip(hold_strength, 0.0, 1.0))
    blended = (1.0 - alpha) * current + alpha * previous
    return np.clip(blended.reshape(1, -1), -1.0, 1.0)


def simple_temporal_ema(
    action: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    previous_action: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None,
    ema_strength: float = 0.35,
) -> np.ndarray:
    current = _as_action(action)
    if previous_action is None:
        return np.clip(current.reshape(1, -1), -1.0, 1.0)
    previous = _as_action(previous_action)
    alpha = float(np.clip(ema_strength, 0.0, 1.0))
    return np.clip(((1.0 - alpha) * current + alpha * previous).reshape(1, -1), -1.0, 1.0)


def vla_corrector_jump_proxy(
    action: Sequence[float] | Sequence[Sequence[float]] | np.ndarray,
    *,
    previous_action: Sequence[float] | Sequence[Sequence[float]] | np.ndarray | None,
    jump_threshold: float = 0.45,
) -> np.ndarray:
    current = _as_action(action)
    if previous_action is None:
        return np.clip(current.reshape(1, -1), -1.0, 1.0)
    previous = _as_action(previous_action)
    if float(np.linalg.norm(current - previous)) > float(jump_threshold):
        return np.clip(previous.reshape(1, -1), -1.0, 1.0)
    return np.clip(current.reshape(1, -1), -1.0, 1.0)
