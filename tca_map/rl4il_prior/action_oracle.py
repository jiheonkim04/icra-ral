"""Action-sequence oracle utilities for the RL4IL prior port.

The released RL4IL scripts use a scalar `label` pathway. For the IL setting,
the paper states that a demonstration's recorded action sequence is the
supervision signal. These helpers implement that action-sequence loss without
touching policy training or rollout code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class ActionOracleConfig:
    """Frozen action-distance settings for the RL4IL prior port."""

    resample_steps: int = 64
    length_penalty_weight: float = 0.01


def as_action_array(actions: np.ndarray | Sequence[Sequence[float]]) -> np.ndarray:
    """Return a finite two-dimensional float32 action array."""

    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError(f"actions must be rank-2, got shape {arr.shape}")
    if arr.shape[0] <= 0 or arr.shape[1] <= 0:
        raise ValueError(f"actions must be non-empty, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError("actions contain non-finite values")
    return arr


def resample_action_sequence(
    actions: np.ndarray | Sequence[Sequence[float]],
    *,
    steps: int,
) -> np.ndarray:
    """Linearly resample a T x D action sequence to `steps` x D."""

    arr = as_action_array(actions)
    if int(steps) <= 0:
        raise ValueError("steps must be positive")
    steps = int(steps)
    if arr.shape[0] == steps:
        return arr.astype(np.float32, copy=True)
    if arr.shape[0] == 1:
        return np.repeat(arr.astype(np.float32), repeats=steps, axis=0)

    old_x = np.linspace(0.0, 1.0, num=arr.shape[0], dtype=np.float32)
    new_x = np.linspace(0.0, 1.0, num=steps, dtype=np.float32)
    out = np.empty((steps, arr.shape[1]), dtype=np.float32)
    for dim in range(arr.shape[1]):
        out[:, dim] = np.interp(new_x, old_x, arr[:, dim]).astype(np.float32)
    return out


def action_sequence_distance(
    a: np.ndarray | Sequence[Sequence[float]],
    b: np.ndarray | Sequence[Sequence[float]],
    config: ActionOracleConfig = ActionOracleConfig(),
) -> float:
    """Distance used to define the RL4IL action-sequence oracle.

    The dominant term is MSE between fixed-length resampled trajectories. A
    small normalized length penalty breaks otherwise equal action-shape ties
    without overwhelming action geometry.
    """

    arr_a = as_action_array(a)
    arr_b = as_action_array(b)
    if arr_a.shape[1] != arr_b.shape[1]:
        raise ValueError(f"action dimensions differ: {arr_a.shape[1]} vs {arr_b.shape[1]}")
    ra = resample_action_sequence(arr_a, steps=config.resample_steps)
    rb = resample_action_sequence(arr_b, steps=config.resample_steps)
    mse = float(np.mean((ra - rb) ** 2))
    denom = max(arr_a.shape[0], arr_b.shape[0], 1)
    length_penalty = float(abs(arr_a.shape[0] - arr_b.shape[0]) / denom)
    return mse + float(config.length_penalty_weight) * length_penalty


def pairwise_action_distance_matrix(
    actions: Sequence[np.ndarray],
    config: ActionOracleConfig = ActionOracleConfig(),
) -> np.ndarray:
    """Return an N x N symmetric action-distance matrix."""

    arrays = [as_action_array(action) for action in actions]
    n = len(arrays)
    out = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(i + 1, n):
            dist = action_sequence_distance(arrays[i], arrays[j], config)
            out[i, j] = dist
            out[j, i] = dist
    return out


def oracle_index_for_candidates(
    source_index: int,
    candidate_indices: Iterable[int],
    distance_matrix: np.ndarray,
) -> int:
    """Return the candidate index with minimum action-sequence distance."""

    matrix = np.asarray(distance_matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("distance_matrix must be square")
    source_index = int(source_index)
    candidates = [int(idx) for idx in candidate_indices if int(idx) != source_index]
    if not candidates:
        raise ValueError("candidate set is empty after excluding self")
    for idx in candidates:
        if idx < 0 or idx >= matrix.shape[0]:
            raise IndexError(f"candidate index {idx} outside matrix size {matrix.shape[0]}")
    return min(candidates, key=lambda idx: (float(matrix[source_index, idx]), idx))

