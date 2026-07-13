"""Photometric sensor-ensemble VLA helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch


CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2")
TRANSFORMS = ("identity", "bright_low_contrast", "dark_high_contrast")
VARIANTS = (
    "clean_frozen_smolvla",
    "bright_single",
    "dark_single",
    "pse_duplicate_clean",
    "pse_full",
)
VARIANT_TRANSFORMS = {
    "clean_frozen_smolvla": ("identity",),
    "bright_single": ("bright_low_contrast",),
    "dark_single": ("dark_high_contrast",),
    "pse_duplicate_clean": ("identity", "identity", "identity"),
    "pse_full": ("identity", "bright_low_contrast", "dark_high_contrast"),
}
PRIVILEGED_INFERENCE_FIELDS = {
    "success",
    "reward",
    "task_outcome",
    "future_observation",
    "future_action",
    "sim_state",
    "mujoco_state",
    "object_pose",
    "bddl_predicate",
}


@dataclass(frozen=True)
class PSEConfig:
    bright_gain: float = 0.42
    bright_bias: float = 0.28
    dark_gain: float = 1.25
    dark_bias: float = -0.10

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def assert_no_privileged_inference_fields(fields: Iterable[str]) -> None:
    present = {str(field) for field in fields}
    forbidden = sorted(present & PRIVILEGED_INFERENCE_FIELDS)
    if forbidden:
        raise ValueError(f"privileged PSE inference fields: {forbidden}")


def transforms_for_variant(variant: str) -> tuple[str, ...]:
    try:
        return tuple(VARIANT_TRANSFORMS[str(variant)])
    except KeyError as exc:
        raise ValueError(f"unknown PSE variant: {variant}") from exc


def transform_image(image: torch.Tensor, transform: str, config: PSEConfig) -> torch.Tensor:
    if str(transform) == "identity":
        return image
    if str(transform) == "bright_low_contrast":
        return torch.clamp(float(config.bright_gain) * image + float(config.bright_bias), 0.0, 1.0)
    if str(transform) == "dark_high_contrast":
        return torch.clamp(float(config.dark_gain) * image + float(config.dark_bias), 0.0, 1.0)
    raise ValueError(f"unknown PSE transform: {transform}")


def transform_batch_images(
    batch: Mapping[str, Any],
    *,
    transform: str,
    config: PSEConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert_no_privileged_inference_fields(["current_image"])
    out = dict(batch)
    diagnostics: dict[str, Any] = {"image_mse_vs_identity": {}, "image_mean_abs_delta": {}}
    for key in CAMERA_KEYS:
        if key not in out:
            continue
        clean = out[key]
        transformed = transform_image(clean, str(transform), config)
        out[key] = transformed
        diagnostics["image_mse_vs_identity"][key] = float(torch.mean((transformed.detach().float() - clean.detach().float()) ** 2).cpu().item())
        diagnostics["image_mean_abs_delta"][key] = float(torch.mean(torch.abs(transformed.detach().float() - clean.detach().float())).cpu().item())
    return out, diagnostics


def average_action_arrays(actions: Sequence[np.ndarray]) -> np.ndarray:
    arrays = [np.asarray(action, dtype=np.float64).reshape(-1) for action in actions]
    if not arrays:
        raise ValueError("cannot average zero actions")
    shape = arrays[0].shape
    if any(array.shape != shape for array in arrays):
        raise ValueError(f"action shape mismatch: {[list(array.shape) for array in arrays]}")
    return np.mean(np.stack(arrays, axis=0), axis=0).reshape(1, -1)


def action_l2_delta(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(left, dtype=np.float64).reshape(-1) - np.asarray(right, dtype=np.float64).reshape(-1)))


def mechanism_active(summary: Mapping[str, Any]) -> bool:
    full = ((summary.get("by_variant") or {}).get("pse_full") or {})
    return float(full.get("mean_delta_vs_clean", 0.0) or 0.0) > 1e-4
