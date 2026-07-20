"""Target-aware visual and distribution metrics for Epoch 9B.

The visual helpers intentionally accept an explicit image-space object center.
This makes the observed crop auditable and prevents a high-quality match on a
static distractor from being mistaken for object return.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable
from typing import Any

import numpy as np


def rgb_sha256(frame: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(frame, dtype=np.uint8))
    return hashlib.sha256(value.tobytes()).hexdigest().upper()


def _gray(frame: np.ndarray) -> np.ndarray:
    import cv2

    value = np.asarray(frame)
    if value.ndim == 2:
        return value.astype(np.uint8, copy=False)
    if value.ndim != 3 or value.shape[2] != 3:
        raise ValueError(f"expected HxW or HxWx3 frame, received {value.shape}")
    return cv2.cvtColor(value.astype(np.uint8, copy=False), cv2.COLOR_RGB2GRAY)


def template_shift_at_center(
    initial_frame: np.ndarray,
    current_frame: np.ndarray,
    center_xy: tuple[int, int],
    *,
    radius: int = 8,
    search: int = 18,
) -> dict[str, Any]:
    """Track one object-centered template in RGB with explicit crop telemetry."""

    import cv2

    initial = _gray(initial_frame)
    current = _gray(current_frame)
    if initial.shape != current.shape:
        raise ValueError(f"frame shape mismatch: {initial.shape} versus {current.shape}")
    height, width = initial.shape
    x, y = (int(center_xy[0]), int(center_xy[1]))
    if not (radius <= x < width - radius and radius <= y < height - radius):
        raise ValueError(f"center {center_xy} with radius {radius} is outside {width}x{height}")
    bounded_search = min(search, x - radius, y - radius, width - 1 - x - radius, height - 1 - y - radius)
    if bounded_search < 1:
        raise ValueError("search region has no positive margin")

    template = initial[y - radius : y + radius + 1, x - radius : x + radius + 1]
    region = current[
        y - radius - bounded_search : y + radius + bounded_search + 1,
        x - radius - bounded_search : x + radius + bounded_search + 1,
    ]
    if float(np.std(template)) < 1e-6:
        raise ValueError("object template is constant and cannot be tracked")
    score = cv2.matchTemplate(region, template, cv2.TM_CCOEFF_NORMED)
    _, quality, _, location = cv2.minMaxLoc(score)
    dx = int(location[0]) - bounded_search
    dy = int(location[1]) - bounded_search

    def _parabolic_offset(left: float, middle: float, right: float) -> float:
        """Return the bounded vertex offset of a three-sample parabola."""

        denominator = left - 2.0 * middle + right
        if abs(denominator) < 1e-12:
            return 0.0
        return float(np.clip(0.5 * (left - right) / denominator, -1.0, 1.0))

    peak_x, peak_y = (int(location[0]), int(location[1]))
    subpixel_x = float(peak_x)
    subpixel_y = float(peak_y)
    if 0 < peak_x < score.shape[1] - 1:
        subpixel_x += _parabolic_offset(
            float(score[peak_y, peak_x - 1]),
            float(score[peak_y, peak_x]),
            float(score[peak_y, peak_x + 1]),
        )
    if 0 < peak_y < score.shape[0] - 1:
        subpixel_y += _parabolic_offset(
            float(score[peak_y - 1, peak_x]),
            float(score[peak_y, peak_x]),
            float(score[peak_y + 1, peak_x]),
        )
    subpixel_dx = subpixel_x - float(bounded_search)
    subpixel_dy = subpixel_y - float(bounded_search)
    return {
        "dx": float(dx),
        "dy": float(dy),
        "magnitude_pixels": float(np.hypot(dx, dy)),
        "subpixel_dx": subpixel_dx,
        "subpixel_dy": subpixel_dy,
        "subpixel_magnitude_pixels": float(np.hypot(subpixel_dx, subpixel_dy)),
        "quality": float(quality),
        "center_xy": [x, y],
        "template_bounds_xyxy": [x - radius, y - radius, x + radius, y + radius],
        "search_pixels_requested": int(search),
        "search_pixels_effective": int(bounded_search),
        "template_std_gray": float(np.std(template)),
    }


def changed_pixel_support(
    initial_frame: np.ndarray,
    current_frame: np.ndarray,
    *,
    threshold: int = 10,
    workspace_y_max: int | None = None,
) -> dict[str, Any]:
    """Describe visible frame change without claiming instance displacement."""

    import cv2

    initial = np.asarray(initial_frame, dtype=np.int16)
    current = np.asarray(current_frame, dtype=np.int16)
    if initial.shape != current.shape:
        raise ValueError(f"frame shape mismatch: {initial.shape} versus {current.shape}")
    difference = np.max(np.abs(current - initial), axis=2) if initial.ndim == 3 else np.abs(current - initial)
    mask = difference > int(threshold)
    if workspace_y_max is not None:
        mask[int(workspace_y_max) :, :] = False
    ys, xs = np.where(mask)
    bbox = None if xs.size == 0 else [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]
    component_count, _, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if component_count <= 1:
        largest_bbox = None
        largest_area = 0
        largest_centroid = None
    else:
        largest_index = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        left, top, component_width, component_height, largest_area = (
            int(value) for value in stats[largest_index]
        )
        largest_bbox = [left, top, left + component_width - 1, top + component_height - 1]
        largest_centroid = [float(value) for value in centroids[largest_index]]
    return {
        "threshold_uint8": int(threshold),
        "changed_pixel_count": int(xs.size),
        "changed_fraction": float(xs.size / mask.size),
        "changed_bbox_xyxy": bbox,
        "connected_component_count": int(max(0, component_count - 1)),
        "largest_component_area": int(largest_area),
        "largest_component_bbox_xyxy": largest_bbox,
        "largest_component_centroid_xy": largest_centroid,
        "mean_abs_channel_change": float(np.mean(np.abs(current - initial))),
        "max_abs_channel_change": int(np.max(np.abs(current - initial))),
    }


def bounds_overlap(first: Iterable[int], second: Iterable[int]) -> bool:
    ax0, ay0, ax1, ay1 = (int(value) for value in first)
    bx0, by0, bx1, by1 = (int(value) for value in second)
    return max(ax0, bx0) <= min(ax1, bx1) and max(ay0, by0) <= min(ay1, by1)


def distribution_summary(
    values: Iterable[float],
    *,
    threshold: float,
    bootstrap_seed: int = 9009,
    bootstrap_draws: int = 5000,
) -> dict[str, Any]:
    """Summarize continuous displacement values and uncertainty of the mean."""

    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {
            "count": 0,
            "threshold": float(threshold),
            "count_above_threshold": 0,
            "fraction_above_threshold": None,
            "mean": None,
            "mean_bootstrap_ci95": [None, None],
            "median": None,
            "std": None,
            "min": None,
            "max": None,
            "quantiles": {},
        }
    rng = np.random.default_rng(int(bootstrap_seed))
    indices = rng.integers(0, array.size, size=(int(bootstrap_draws), array.size))
    boot_means = np.mean(array[indices], axis=1)
    quantile_levels = (0.05, 0.25, 0.5, 0.75, 0.95)
    return {
        "count": int(array.size),
        "threshold": float(threshold),
        "count_above_threshold": int(np.sum(array > threshold)),
        "fraction_above_threshold": float(np.mean(array > threshold)),
        "mean": float(np.mean(array)),
        "mean_bootstrap_ci95": [float(value) for value in np.quantile(boot_means, [0.025, 0.975])],
        "median": float(np.median(array)),
        "std": float(np.std(array, ddof=1)) if array.size > 1 else 0.0,
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "quantiles": {f"q{int(level * 100):02d}": float(np.quantile(array, level)) for level in quantile_levels},
    }


def nondecreasing(values: Iterable[float], *, tolerance: float = 1e-9) -> bool:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    return bool(array.size > 0 and np.all(np.diff(array) >= -float(tolerance)))
