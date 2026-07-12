"""Geometric-continuity anchored perception for VLA observation corruption."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping

import numpy as np
import torch
import torch.nn.functional as F


CAMERA_KEYS = ("observation.images.camera1", "observation.images.camera2")
PRIVILEGED_INFERENCE_FIELDS = {
    "sim_state",
    "mujoco_state",
    "success",
    "reward",
    "future_observation",
    "future_action_target",
    "object_pose",
    "bddl_predicate",
    "reset_identity",
    "task_outcome",
}


@dataclass(frozen=True)
class GCAPConfig:
    fill_value: float = 0.0
    edge_gain: float = 0.08
    temporal_blend: float = 1.0
    mask_dilate: int = 5
    min_active_fraction: float = 0.08
    max_active_fraction: float = 0.78

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


def assert_no_privileged_inference_fields(fields: Iterable[str]) -> None:
    present = {str(field) for field in fields}
    forbidden = sorted(present & PRIVILEGED_INFERENCE_FIELDS)
    if forbidden:
        raise ValueError(f"privileged GCAP inference fields: {forbidden}")


def occlusion_box(
    *,
    height: int,
    width: int,
    identity: int,
    step_fraction: float,
    camera_key: str,
    config: GCAPConfig,
) -> tuple[int, int, int, int] | None:
    frac = float(np.clip(step_fraction, 0.0, 1.0))
    if frac < float(config.min_active_fraction) or frac > float(config.max_active_fraction):
        return None
    identity_shift = float((int(identity) % 3) - 1)
    phase_shift = float(np.sin(2.0 * np.pi * frac))
    if str(camera_key).endswith("camera1"):
        box_w = max(8, int(round(0.36 * int(width))))
        box_h = max(8, int(round(0.34 * int(height))))
        center_x = 0.50 + 0.055 * identity_shift + 0.035 * phase_shift
        center_y = 0.54 + 0.045 * np.cos(np.pi * frac)
    else:
        box_w = max(8, int(round(0.28 * int(width))))
        box_h = max(8, int(round(0.28 * int(height))))
        center_x = 0.46 - 0.045 * identity_shift
        center_y = 0.50 + 0.035 * phase_shift
    x0 = int(round(center_x * int(width) - box_w / 2))
    y0 = int(round(center_y * int(height) - box_h / 2))
    x0 = max(0, min(int(width) - box_w, x0))
    y0 = max(0, min(int(height) - box_h, y0))
    return x0, y0, x0 + box_w, y0 + box_h


def apply_rect_occlusion(
    image: torch.Tensor,
    *,
    box: tuple[int, int, int, int] | None,
    fill_value: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    if image.ndim != 4:
        raise ValueError(f"expected BCHW image tensor, got rank {image.ndim}")
    mask = torch.zeros((image.shape[0], 1, image.shape[2], image.shape[3]), dtype=image.dtype, device=image.device)
    if box is None:
        return image.clone(), mask
    x0, y0, x1, y1 = [int(value) for value in box]
    occluded = image.clone()
    occluded[:, :, y0:y1, x0:x1] = float(fill_value)
    mask[:, :, y0:y1, x0:x1] = 1.0
    return occluded, mask


def dilate_mask(mask: torch.Tensor, radius: int) -> torch.Tensor:
    radius = int(max(0, radius))
    if radius == 0:
        return mask
    kernel = 2 * radius + 1
    return F.max_pool2d(mask, kernel_size=kernel, stride=1, padding=radius)


def sobel_edges(image: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4:
        raise ValueError(f"expected BCHW image tensor, got rank {image.ndim}")
    gray = image.mean(dim=1, keepdim=True)
    kx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], dtype=image.dtype, device=image.device)
    ky = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]], dtype=image.dtype, device=image.device)
    gx = F.conv2d(gray, kx.reshape(1, 1, 3, 3), padding=1)
    gy = F.conv2d(gray, ky.reshape(1, 1, 3, 3), padding=1)
    edge = torch.sqrt(gx * gx + gy * gy + 1e-12)
    denom = edge.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
    return edge / denom


def edge_enhance(image: torch.Tensor, *, mask: torch.Tensor, config: GCAPConfig) -> torch.Tensor:
    if float(config.edge_gain) <= 0.0 or float(mask.sum().detach().cpu().item()) <= 0.0:
        return image
    dilated = dilate_mask(mask, int(config.mask_dilate))
    boundary = torch.clamp(dilated - mask, min=0.0, max=1.0)
    support = torch.clamp(dilated + 0.35 * boundary, min=0.0, max=1.0)
    edge = sobel_edges(image).repeat(1, image.shape[1], 1, 1)
    lo = image.amin(dim=(1, 2, 3), keepdim=True)
    hi = image.amax(dim=(1, 2, 3), keepdim=True)
    enhanced = image + float(config.edge_gain) * edge * support
    return torch.minimum(torch.maximum(enhanced, lo), hi)


def repair_camera_tensor(
    current: torch.Tensor,
    *,
    previous: torch.Tensor | None,
    mask: torch.Tensor,
    variant: str,
    config: GCAPConfig,
) -> torch.Tensor:
    variant = str(variant)
    if variant in {"clean_frozen_smolvla", "occluded_frozen_smolvla"}:
        return current
    if variant == "full_frame_hold_last":
        if previous is None or float(mask.sum().detach().cpu().item()) <= 0.0:
            return current
        return previous.detach().clone()
    if variant == "sobel_edge_boost":
        return edge_enhance(current, mask=mask, config=config)
    if variant == "gcap_no_temporal_ablation":
        return edge_enhance(current, mask=mask, config=config)
    if variant in {"gcap_full", "clean_gcap_full"}:
        repaired = current
        if previous is not None and float(mask.sum().detach().cpu().item()) > 0.0:
            blend = float(np.clip(config.temporal_blend, 0.0, 1.0))
            temporal = blend * previous.detach() + (1.0 - blend) * current
            repaired = current * (1.0 - mask) + temporal * mask
        return edge_enhance(repaired, mask=mask, config=config)
    raise ValueError(f"unknown GCAP variant: {variant}")


def transform_batch_images(
    batch: Mapping[str, Any],
    *,
    variant: str,
    condition: str,
    identity: int,
    step_fraction: float,
    memory: dict[str, torch.Tensor],
    config: GCAPConfig,
) -> tuple[dict[str, Any], dict[str, Any]]:
    assert_no_privileged_inference_fields(
        [
            "current_observation_image",
            "previous_repaired_image",
            "detected_occlusion_mask",
            "step_fraction",
            "camera_key",
        ]
    )
    out = dict(batch)
    diagnostics: dict[str, Any] = {"camera_masks": {}, "camera_boxes": {}, "mean_mask_fraction": {}}
    use_occlusion = str(condition) == "occluded"
    for key in CAMERA_KEYS:
        if key not in out:
            continue
        image = out[key]
        _, _, height, width = image.shape
        box = occlusion_box(height=height, width=width, identity=identity, step_fraction=step_fraction, camera_key=key, config=config) if use_occlusion else None
        current, mask = apply_rect_occlusion(image, box=box, fill_value=float(config.fill_value))
        repaired = repair_camera_tensor(current, previous=memory.get(key), mask=mask, variant=str(variant), config=config)
        out[key] = repaired
        memory[key] = repaired.detach().clone()
        diagnostics["camera_masks"][key] = float(mask.mean().detach().cpu().item())
        diagnostics["camera_boxes"][key] = None if box is None else [int(value) for value in box]
        diagnostics["mean_mask_fraction"][key] = float(mask.mean().detach().cpu().item())
    return out, diagnostics


def image_mse(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.mean((a.detach().float().cpu() - b.detach().float().cpu()) ** 2).item())
