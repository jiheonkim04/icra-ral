"""COVI-VLA executable Stage 0 components."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


PROPOSAL_HASH = "338430D2C6CF1D82410C036D79102ED3F38B2367BB35B9AE2811161698A3E621"
LEGAL_INFERENCE_FEATURES = (
    "observation.images.image",
    "observation.images.image2",
    "observation.state",
    "language_or_task_instruction",
    "base_action",
    "predicted_occlusion_context",
)
FORBIDDEN_INFERENCE_KEYS = {
    "clean_complementary_image",
    "confirmatory_outcome",
    "dataset_global_index",
    "episode_index",
    "future_action",
    "future_observation",
    "ground_truth_mask_context",
    "object_pose",
    "reset_identity",
    "reward",
    "segmentation_mask",
    "success",
    "target_action",
}


@dataclass(frozen=True)
class COVIStage0Config:
    seed: int = 20260715
    feature_dim: int = 960
    tokens_per_stream: int = 64
    available_streams: int = 2
    state_dim: int = 8
    action_dim: int = 7
    task_dim: int = 40
    context_dim: int = 6
    hidden_dim: int = 256
    context_hidden_dim: int = 128
    gate_max: float = 0.10
    feature_clip: float = 0.25
    init_gate: float = 1e-4
    mask_target_fraction: float = 0.18
    mask_fraction_min: float = 0.14
    mask_fraction_max: float = 0.22
    batch_size: int = 32
    epochs: int = 40
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    lambda_view: float = 1.0
    lambda_clean: float = 1.0
    lambda_delta: float = 0.10
    lambda_gate: float = 0.01
    lambda_action: float = 0.25
    practical_margin: float = 0.02
    bootstrap_iterations: int = 5000
    bootstrap_seed: int = 20260716
    init_action_delta_p95_max: float = 1e-6
    clean_action_delta_p95_max: float = 0.02
    trained_action_delta_p95_max: float = 0.10
    translation_delta_p95_max: float = 0.05
    rotation_delta_p95_max: float = 0.05
    gripper_delta_p95_max: float = 0.25

    @property
    def source_without_context_dim(self) -> int:
        return 2 * self.feature_dim + self.state_dim + self.action_dim + self.task_dim

    @property
    def predictor_input_dim(self) -> int:
        return self.source_without_context_dim + self.context_dim

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sample_id(record: Mapping[str, Any]) -> str:
    return str(
        record.get("sample_id")
        or f"{record.get('split')}|{record.get('task_index')}|{record.get('episode_index')}|{record.get('frame_index')}"
    )


def partition_stage0_manifest(manifest: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Freeze one train episode per task for fit and one for the unresolved check."""

    splits = manifest.get("splits") or {}
    train = [dict(row) for row in splits.get("train", [])]
    validation = [dict(row) for row in splits.get("val", [])]
    confirmatory = [dict(row) for row in splits.get("test", [])]
    episodes_by_task: dict[int, list[int]] = {}
    for row in train:
        episodes_by_task.setdefault(int(row["task_index"]), []).append(int(row["episode_index"]))
    selected: dict[int, tuple[int, int]] = {}
    for task, episodes in episodes_by_task.items():
        unique = sorted(set(episodes))
        if len(unique) != 2:
            raise ValueError(f"task {task} expected exactly two train episodes, got {unique}")
        selected[task] = (unique[0], unique[1])
    fit = [row for row in train if int(row["episode_index"]) == selected[int(row["task_index"])][0]]
    one_check = [row for row in train if int(row["episode_index"]) == selected[int(row["task_index"])][1]]
    output = {
        "discovery_fit": sorted(fit, key=_sample_id),
        "discovery_one_check": sorted(one_check, key=_sample_id),
        "validation": sorted(validation, key=_sample_id),
        "confirmatory_reserved": sorted(confirmatory, key=_sample_id),
    }
    keys = {name: {_sample_id(row) for row in rows} for name, rows in output.items()}
    names = list(keys)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            overlap = keys[left] & keys[right]
            if overlap:
                raise ValueError(f"partition overlap {left}/{right}: {sorted(overlap)[:3]}")
    return output


def partition_summary(partitions: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for name, rows in partitions.items():
        sample_ids = [_sample_id(row) for row in rows]
        frame_keys = [
            (int(row["task_index"]), int(row["episode_index"]), int(row["frame_index"]))
            for row in rows
        ]
        summary[name] = {
            "records": len(rows),
            "episodes": len({int(row["episode_index"]) for row in rows}),
            "tasks": len({int(row["task_index"]) for row in rows}),
            "duplicate_sample_ids": len(sample_ids) - len(set(sample_ids)),
            "duplicate_frame_keys": len(frame_keys) - len(set(frame_keys)),
        }
    return summary


def _seed_for(sample_id: str, stream: int, variant: str) -> int:
    payload = f"{PROPOSAL_HASH}|{sample_id}|{stream}|{variant}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def irregular_occlusion_mask(
    height: int,
    width: int,
    *,
    sample_id: str,
    stream: int,
    target_fraction: float = 0.18,
) -> np.ndarray:
    """Create a deterministic connected, non-rectangular radial mask."""

    rng = np.random.default_rng(_seed_for(sample_id, stream, "irregular"))
    cy = float(rng.uniform(0.30, 0.70)) * (height - 1)
    cx = float(rng.uniform(0.30, 0.70)) * (width - 1)
    aspect = float(rng.uniform(0.70, 1.35))
    phase_3 = float(rng.uniform(0.0, 2.0 * math.pi))
    phase_5 = float(rng.uniform(0.0, 2.0 * math.pi))
    yy, xx = np.mgrid[0:height, 0:width]
    dy = yy - cy
    dx = xx - cx
    theta = np.arctan2(dy, dx)
    radial_shape = 1.0 + 0.22 * np.sin(3.0 * theta + phase_3) + 0.12 * np.sin(5.0 * theta + phase_5)

    def make(radius: float) -> np.ndarray:
        ellipse_radius = np.sqrt((dx / aspect) ** 2 + (dy * aspect) ** 2)
        return ellipse_radius <= radius * radial_shape

    low, high = 1.0, float(max(height, width))
    for _ in range(24):
        middle = (low + high) / 2.0
        if float(make(middle).mean()) < target_fraction:
            low = middle
        else:
            high = middle
    return make((low + high) / 2.0)


def equal_area_rectangle_mask(
    height: int,
    width: int,
    *,
    sample_id: str,
    stream: int,
    area_fraction: float,
) -> np.ndarray:
    rng = np.random.default_rng(_seed_for(sample_id, stream, "rectangle"))
    aspect = float(rng.uniform(0.65, 1.55))
    area = max(1.0, area_fraction * height * width)
    rect_h = min(height, max(1, int(round(math.sqrt(area / aspect)))))
    rect_w = min(width, max(1, int(round(area / rect_h))))
    top = int(rng.integers(0, max(1, height - rect_h + 1)))
    left = int(rng.integers(0, max(1, width - rect_w + 1)))
    mask = np.zeros((height, width), dtype=bool)
    mask[top : top + rect_h, left : left + rect_w] = True
    return mask


def apply_scene_obstruction(image: torch.Tensor, mask: np.ndarray, *, sample_id: str, stream: int) -> torch.Tensor:
    if image.ndim != 3:
        raise ValueError(f"image must be CHW, got {tuple(image.shape)}")
    height, width = int(image.shape[-2]), int(image.shape[-1])
    if mask.shape != (height, width):
        raise ValueError(f"mask shape {mask.shape} does not match image {(height, width)}")
    rng = np.random.default_rng(_seed_for(sample_id, stream, "texture"))
    shift_y = int(rng.integers(max(1, height // 5), max(2, 2 * height // 5)))
    shift_x = int(rng.integers(max(1, width // 5), max(2, 2 * width // 5)))
    texture = torch.roll(image, shifts=(shift_y, shift_x), dims=(-2, -1))
    mask_tensor = torch.as_tensor(mask, device=image.device, dtype=torch.bool).unsqueeze(0)
    return torch.where(mask_tensor, texture, image)


def mask_context(mask_1: np.ndarray, mask_2: np.ndarray) -> np.ndarray:
    values: list[float] = [float(mask_1.mean()), float(mask_2.mean())]
    for mask in (mask_1, mask_2):
        yy, xx = np.nonzero(mask)
        values.extend(
            [
                float(xx.mean() / max(1, mask.shape[1] - 1)) if xx.size else 0.0,
                float(yy.mean() / max(1, mask.shape[0] - 1)) if yy.size else 0.0,
            ]
        )
    return np.asarray(values, dtype=np.float32)


class COVIStage0Adapter(nn.Module):
    """Small complementary-feature predictor with an exact identity residual init."""

    def __init__(self, config: COVIStage0Config | None = None) -> None:
        super().__init__()
        self.config = config or COVIStage0Config()
        cfg = self.config
        self.context_head = nn.Sequential(
            nn.Linear(2 * cfg.feature_dim, cfg.context_hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.context_hidden_dim, cfg.context_dim),
            nn.Sigmoid(),
        )
        self.predictor = nn.Sequential(
            nn.Linear(cfg.predictor_input_dim, cfg.hidden_dim),
            nn.GELU(),
            nn.Linear(cfg.hidden_dim, cfg.feature_dim),
        )
        self.residual_projection = nn.Linear(cfg.feature_dim, cfg.feature_dim)
        self.gate_head = nn.Linear(cfg.predictor_input_dim, 1)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)
        nn.init.zeros_(self.gate_head.weight)
        probability = min(1.0 - 1e-6, max(1e-6, cfg.init_gate / cfg.gate_max))
        nn.init.constant_(self.gate_head.bias, math.log(probability / (1.0 - probability)))

    def forward(self, source_without_context: torch.Tensor, camera2_summary: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.config
        visual = source_without_context[:, : 2 * cfg.feature_dim]
        context = self.context_head(visual)
        predictor_input = torch.cat([source_without_context, context], dim=-1)
        imagined = F.normalize(self.predictor(predictor_input), dim=-1)
        residual = torch.clamp(self.residual_projection(imagined), -cfg.feature_clip, cfg.feature_clip)
        gate = cfg.gate_max * torch.sigmoid(self.gate_head(predictor_input))
        adapted = camera2_summary + gate * residual
        return {
            "context": context,
            "predictor_input": predictor_input,
            "imagined": imagined,
            "residual": residual,
            "gate": gate,
            "adapted": adapted,
        }

    def injection(self, source_without_context: torch.Tensor) -> dict[str, torch.Tensor]:
        cfg = self.config
        visual = source_without_context[:, : 2 * cfg.feature_dim]
        context = self.context_head(visual)
        predictor_input = torch.cat([source_without_context, context], dim=-1)
        imagined = F.normalize(self.predictor(predictor_input), dim=-1)
        residual = torch.clamp(self.residual_projection(imagined), -cfg.feature_clip, cfg.feature_clip)
        gate = cfg.gate_max * torch.sigmoid(self.gate_head(predictor_input))
        return {"context": context, "imagined": imagined, "residual": residual, "gate": gate}


def covi_stage0_loss(
    model: COVIStage0Adapter,
    *,
    occluded_source: torch.Tensor,
    occluded_camera2: torch.Tensor,
    target: torch.Tensor,
    context_target: torch.Tensor,
    clean_source: torch.Tensor,
    clean_camera2: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    cfg = model.config
    target = F.normalize(target, dim=-1)
    clean_camera2 = F.normalize(clean_camera2, dim=-1)
    occluded_camera2 = F.normalize(occluded_camera2, dim=-1)
    occ = model(occluded_source, occluded_camera2)
    clean = model(clean_source, clean_camera2)
    view = (
        F.smooth_l1_loss(occ["imagined"], target)
        + F.smooth_l1_loss(F.normalize(occ["adapted"], dim=-1), target)
        + 0.25 * F.smooth_l1_loss(occ["context"], context_target)
    )
    clean_loss = F.mse_loss(clean["adapted"], clean_camera2)
    residual_rms = torch.sqrt(torch.mean((occ["gate"] * occ["residual"]) ** 2, dim=-1) + 1e-12)
    delta = torch.mean(torch.relu(residual_rms - cfg.feature_clip) ** 2)
    gate = torch.mean(clean["gate"]) + 0.25 * torch.mean(occ["gate"])
    action = torch.mean((clean["adapted"] - clean_camera2) ** 2)
    terms = {"view": view, "clean": clean_loss, "delta": delta, "gate": gate, "action": action}
    total = (
        cfg.lambda_view * view
        + cfg.lambda_clean * clean_loss
        + cfg.lambda_delta * delta
        + cfg.lambda_gate * gate
        + cfg.lambda_action * action
    )
    return total, terms


def parameter_gradient_norms(model: nn.Module) -> dict[str, float]:
    groups: dict[str, list[torch.Tensor]] = {
        "context_head": [],
        "predictor": [],
        "residual_projection": [],
        "gate_head": [],
    }
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        prefix = name.split(".", 1)[0]
        if prefix in groups:
            groups[prefix].append(parameter.grad.detach().float().reshape(-1))
    result: dict[str, float] = {}
    for name, tensors in groups.items():
        result[name] = float(torch.linalg.vector_norm(torch.cat(tensors)).item()) if tensors else 0.0
    return result


def objective_gradient_audit(
    model: COVIStage0Adapter,
    *,
    occluded_source: torch.Tensor,
    occluded_camera2: torch.Tensor,
    target: torch.Tensor,
    context_target: torch.Tensor,
    clean_source: torch.Tensor,
    clean_camera2: torch.Tensor,
) -> dict[str, Any]:
    """Measure frozen weighted-objective gradients before optimization."""

    cfg = model.config
    weights = {
        "view": cfg.lambda_view,
        "clean": cfg.lambda_clean,
        "delta": cfg.lambda_delta,
        "gate": cfg.lambda_gate,
        "action": cfg.lambda_action,
    }
    model.zero_grad(set_to_none=True)
    _, terms = covi_stage0_loss(
        model,
        occluded_source=occluded_source,
        occluded_camera2=occluded_camera2,
        target=target,
        context_target=context_target,
        clean_source=clean_source,
        clean_camera2=clean_camera2,
    )
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    gradient_norms: dict[str, float] = {}
    finite: dict[str, bool] = {}
    for name, term in terms.items():
        gradients = torch.autograd.grad(
            weights[name] * term,
            parameters,
            retain_graph=True,
            allow_unused=True,
        )
        squared = torch.zeros((), device=term.device, dtype=torch.float64)
        term_finite = bool(torch.isfinite(term).item())
        for gradient in gradients:
            if gradient is None:
                continue
            term_finite = term_finite and bool(torch.isfinite(gradient).all().item())
            squared = squared + torch.sum(gradient.detach().double() ** 2)
        gradient_norms[name] = float(torch.sqrt(squared).item())
        finite[name] = term_finite
    model.zero_grad(set_to_none=True)
    nonzero = [value for value in gradient_norms.values() if value > 0.0 and math.isfinite(value)]
    ratio = max(nonzero) / min(nonzero) if nonzero else math.inf
    return {
        "term_values": {name: float(term.detach().item()) for name, term in terms.items()},
        "weights": weights,
        "weighted_gradient_norms": gradient_norms,
        "finite_by_objective": finite,
        "nonzero_objective_count": len(nonzero),
        "largest_to_smallest_nonzero_gradient_ratio": float(ratio),
        "threshold": 100.0,
        "passed": bool(nonzero and all(finite.values()) and ratio <= 100.0),
    }


def prediction_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    diff = prediction - target
    rmse = float(np.sqrt(np.mean(diff**2)))
    pred_norm = np.linalg.norm(prediction, axis=1)
    target_norm = np.linalg.norm(target, axis=1)
    cosine = np.sum(prediction * target, axis=1) / np.maximum(1e-12, pred_norm * target_norm)
    return {"rmse": rmse, "cosine_mean": float(np.mean(cosine))}


def normalized_rmse_margin(
    candidate: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    train_target_mean: np.ndarray,
) -> float:
    candidate_rmse = prediction_metrics(candidate, target)["rmse"]
    baseline_rmse = prediction_metrics(baseline, target)["rmse"]
    mean_prediction = np.repeat(np.asarray(train_target_mean)[None, :], len(target), axis=0)
    target_scale = max(1e-12, prediction_metrics(mean_prediction, target)["rmse"])
    return float((baseline_rmse - candidate_rmse) / target_scale)


def episode_cluster_bootstrap_margin(
    *,
    candidate: np.ndarray,
    baseline: np.ndarray,
    target: np.ndarray,
    train_target_mean: np.ndarray,
    episode_ids: Sequence[int],
    iterations: int = 5000,
    seed: int = 20260716,
) -> dict[str, Any]:
    episode_ids = np.asarray(episode_ids, dtype=np.int64)
    unique = np.unique(episode_ids)
    if unique.size == 0:
        return {"episode_count": 0, "iterations": iterations, "low": None, "high": None, "mean": None}
    indices = {episode: np.flatnonzero(episode_ids == episode) for episode in unique}
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for draw in range(iterations):
        sampled = rng.choice(unique, size=unique.size, replace=True)
        row_indices = np.concatenate([indices[int(episode)] for episode in sampled])
        draws[draw] = normalized_rmse_margin(
            candidate[row_indices], baseline[row_indices], target[row_indices], train_target_mean
        )
    return {
        "episode_count": int(unique.size),
        "record_count": int(len(episode_ids)),
        "iterations": int(iterations),
        "low": float(np.quantile(draws, 0.025)),
        "high": float(np.quantile(draws, 0.975)),
        "mean": float(np.mean(draws)),
    }


def classify_stage0(report: Mapping[str, Any], config: COVIStage0Config | None = None) -> str:
    cfg = config or COVIStage0Config()
    if report.get("fatal_preimplementation"):
        return "FATAL_PREIMPLEMENTATION"
    if not report.get("implementation_and_data_valid", False):
        return "IMPLEMENTATION_OR_DATA_FAILURE"
    if not report.get("diagnostic_headroom_exists", False):
        return "CONDITION_TOO_SEVERE_OR_NO_HEADROOM"
    if not report.get("identity_and_safety_passed", False):
        return "IMPLEMENTATION_OR_DATA_FAILURE"
    margin = float(report.get("candidate_margin", -math.inf))
    prior_margin = float(report.get("candidate_margin_vs_vim_proxy", -math.inf))
    random_margin = float(report.get("candidate_margin_vs_random_cutout", -math.inf))
    interval = report.get("bootstrap_interval") or {}
    low = interval.get("low")
    high = interval.get("high")
    if low is not None and margin >= cfg.practical_margin and float(low) > 0.0 and prior_margin > 0.0 and random_margin > 0.0:
        return "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH"
    robust = bool(
        int(interval.get("episode_count", 0)) >= 40
        and high is not None
        and float(high) < cfg.practical_margin
        and margin <= 0.0
        and prior_margin <= 0.0
        and random_margin <= 0.0
        and report.get("normalization_sensitivity_resolved", False)
    )
    if robust:
        return "ROBUST_EMPIRICAL_DESIGN_FAILURE"
    return "COVI_STAGE_0_UNDERPOWERED_ONE_CHECK_ALLOWED"
