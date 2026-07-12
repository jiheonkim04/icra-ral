"""Sensor-canonicalized VLA control helpers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch


CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2")
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
class SCVCConfig:
    gain: float = 0.42
    bias: float = 0.28
    eps: float = 1e-4
    temporal_blend: float = 0.80

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def assert_no_privileged_inference_fields(fields: Iterable[str]) -> None:
    present = {str(field) for field in fields}
    forbidden = sorted(present & PRIVILEGED_INFERENCE_FIELDS)
    if forbidden:
        raise ValueError(f"privileged SCVC inference fields: {forbidden}")


def tensor_stats(image: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    mean = image.mean(dim=(0, 2, 3), keepdim=True)
    std = image.std(dim=(0, 2, 3), keepdim=True).clamp_min(1e-6)
    return mean, std


def stats_to_json(mean: torch.Tensor, std: torch.Tensor) -> dict[str, list[float]]:
    return {
        "mean": [float(x) for x in mean.detach().cpu().reshape(-1).tolist()],
        "std": [float(x) for x in std.detach().cpu().reshape(-1).tolist()],
    }


def stats_from_json(payload: Mapping[str, Any], *, dtype: torch.dtype, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    mean = torch.tensor(payload["mean"], dtype=dtype, device=device).reshape(1, -1, 1, 1)
    std = torch.tensor(payload["std"], dtype=dtype, device=device).reshape(1, -1, 1, 1).clamp_min(1e-6)
    return mean, std


def apply_sensor_shift(image: torch.Tensor, config: SCVCConfig) -> torch.Tensor:
    return torch.clamp(float(config.gain) * image + float(config.bias), 0.0, 1.0)


def known_inverse_affine(image: torch.Tensor, config: SCVCConfig) -> torch.Tensor:
    return torch.clamp((image - float(config.bias)) / max(float(config.gain), float(config.eps)), 0.0, 1.0)


def canonicalize_image(
    shifted: torch.Tensor,
    *,
    target_mean: torch.Tensor,
    target_std: torch.Tensor,
    memory: dict[str, torch.Tensor],
    memory_key: str,
    use_temporal: bool,
    config: SCVCConfig,
) -> torch.Tensor:
    current_mean, current_std = tensor_stats(shifted)
    if use_temporal:
        mean_key = f"{memory_key}:mean"
        std_key = f"{memory_key}:std"
        if mean_key in memory and std_key in memory:
            blend = float(np.clip(config.temporal_blend, 0.0, 1.0))
            current_mean = blend * memory[mean_key] + (1.0 - blend) * current_mean
            current_std = blend * memory[std_key] + (1.0 - blend) * current_std
        memory[mean_key] = current_mean.detach()
        memory[std_key] = current_std.detach()
    out = (shifted - current_mean) / current_std.clamp_min(float(config.eps)) * target_std + target_mean
    return torch.clamp(out, 0.0, 1.0)


def transform_batch_images(
    batch: Mapping[str, Any],
    *,
    variant: str,
    calibration: Mapping[str, Any],
    memory: dict[str, torch.Tensor],
    config: SCVCConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert_no_privileged_inference_fields(["current_image", "calibration_stats", "running_sensor_stats"])
    out = dict(batch)
    diagnostics: dict[str, Any] = {"image_mse_shifted_vs_clean": {}, "image_mse_output_vs_clean": {}, "image_delta_vs_shifted": {}}
    for key in CAMERA_KEYS:
        if key not in out:
            continue
        clean = out[key]
        if str(variant) == "clean_frozen_smolvla":
            output = clean
            shifted = clean
        else:
            shifted = apply_sensor_shift(clean, config)
            if str(variant) == "shifted_frozen_smolvla":
                output = shifted
            elif str(variant) == "known_inverse_affine":
                output = known_inverse_affine(shifted, config)
            elif str(variant) in {"scvc_no_temporal", "scvc_full"}:
                target = (calibration.get("camera_stats") or {}).get(key)
                if target is None:
                    raise RuntimeError(f"missing calibration stats for {key}")
                target_mean, target_std = stats_from_json(target, dtype=shifted.dtype, device=shifted.device)
                output = canonicalize_image(
                    shifted,
                    target_mean=target_mean,
                    target_std=target_std,
                    memory=memory,
                    memory_key=key,
                    use_temporal=str(variant) == "scvc_full",
                    config=config,
                )
            else:
                raise ValueError(f"unknown SCVC variant: {variant}")
        out[key] = output
        diagnostics["image_mse_shifted_vs_clean"][key] = float(torch.mean((shifted.detach().float() - clean.detach().float()) ** 2).cpu().item())
        diagnostics["image_mse_output_vs_clean"][key] = float(torch.mean((output.detach().float() - clean.detach().float()) ** 2).cpu().item())
        diagnostics["image_delta_vs_shifted"][key] = float(torch.mean(torch.abs(output.detach().float() - shifted.detach().float())).cpu().item())
    return out, diagnostics


def merge_camera_stats(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for key in CAMERA_KEYS:
        means = [np.asarray(row["camera_stats"][key]["mean"], dtype=np.float64) for row in rows if key in row.get("camera_stats", {})]
        stds = [np.asarray(row["camera_stats"][key]["std"], dtype=np.float64) for row in rows if key in row.get("camera_stats", {})]
        if not means:
            continue
        merged[key] = {
            "mean": [float(x) for x in np.mean(np.stack(means, axis=0), axis=0).tolist()],
            "std": [float(max(x, 1e-6)) for x in np.mean(np.stack(stds, axis=0), axis=0).tolist()],
        }
    return merged


def mechanism_active(summary: Mapping[str, Any]) -> bool:
    full = ((summary.get("by_variant") or {}).get("scvc_full") or {})
    return float(full.get("mean_image_delta_vs_shifted", 0.0) or 0.0) > 1e-4
