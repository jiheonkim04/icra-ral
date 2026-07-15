"""NICE-VLA normalized-innovation math and Stage 0A validation helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


PROPOSAL_HASH = "898BA577B38966D877E3EEC724EB98751BD8C2685CCD0BBA620EB6B6B9598C0A"
K_STEP = 10
ACTION_DIM = 7
CONDITION_DIM = 18
LOW_RANK = 8
VARIANCE_FLOOR = 1e-6
VARIANCE_CEILING = 1e2
CHOLESKY_JITTER = 1e-8


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest().upper()


def pair_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(row["suite"]),
            str(row["task_identity"]),
            str(row["source_path"]),
            f"demo_{int(row['demo_id'])}",
            str(int(row["frame_t"])),
            str(int(row["frame_t_plus_10"])),
        )
    )


def condition_vector(
    action: torch.Tensor,
    previous_action: torch.Tensor,
    gripper_deadband: float,
) -> torch.Tensor:
    """Build the frozen continuous action-regime condition with one transition bit."""

    if action.shape != previous_action.shape or action.ndim != 2 or action.shape[-1] != ACTION_DIM:
        raise ValueError("actions must have matching [B,7] shapes")
    if not math.isfinite(float(gripper_deadband)) or gripper_deadband <= 0.0:
        raise ValueError("gripper_deadband must be finite and positive")
    translation = torch.linalg.vector_norm(action[:, 0:3], dim=-1, keepdim=True)
    rotation = torch.linalg.vector_norm(action[:, 3:6], dim=-1, keepdim=True)
    gripper = action[:, 6:7].abs()
    transition = ((action[:, 6:7] - previous_action[:, 6:7]).abs() >= gripper_deadband).to(action.dtype)
    result = torch.cat((action, previous_action, translation, rotation, gripper, transition), dim=-1)
    if result.shape[-1] != CONDITION_DIM:
        raise AssertionError("condition construction produced the wrong width")
    return result


def discovery_gripper_deadband(actions_by_episode: Sequence[np.ndarray]) -> float:
    nonzero = []
    for raw in actions_by_episode:
        actions = np.asarray(raw, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError("each action episode must be [T,7]")
        if len(actions) > 1:
            values = np.abs(np.diff(actions[:, 6]))
            nonzero.extend(values[values > 0.0].tolist())
    if not nonzero:
        raise ValueError("discovery gripper differences collapsed")
    return float(np.median(np.asarray(nonzero, dtype=np.float64)))


class _ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return value + self.fc2(F.silu(self.fc1(self.norm(value))))


class TinyResidualMean(nn.Module):
    """Frozen Stage 0A shape/gradient proxy for the source residual MLP."""

    def __init__(self, token_dim: int, hidden_width: int = 64, action_embed_dim: int = 16) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.action_proj = nn.Linear(ACTION_DIM, action_embed_dim)
        self.in_proj = nn.Linear(self.token_dim + action_embed_dim, hidden_width)
        self.blocks = nn.ModuleList((_ResidualBlock(hidden_width), _ResidualBlock(hidden_width)))
        self.out_norm = nn.LayerNorm(hidden_width)
        self.out_proj = nn.Linear(hidden_width, self.token_dim)

    def forward(self, visual: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if visual.ndim != 3 or visual.shape[-1] != self.token_dim:
            raise ValueError("visual must be [B,L,D] with configured D")
        if action.ndim != 2 or action.shape != (visual.shape[0], ACTION_DIM):
            raise ValueError("action must be [B,7]")
        embedded = self.action_proj(action)[:, None, :].expand(-1, visual.shape[1], -1)
        value = self.in_proj(torch.cat((visual, embedded), dim=-1))
        for block in self.blocks:
            value = block(value)
        return self.out_proj(self.out_norm(value))


def mean_cosine_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("prediction and target must have matching [B,L,D] shapes")
    pred_flat = prediction.reshape(prediction.shape[0], -1)
    target_flat = target.reshape(target.shape[0], -1)
    return 1.0 - F.cosine_similarity(pred_flat, target_flat, dim=-1, eps=1e-8).mean()


class TinyCovariance(nn.Module):
    """Token-wise diagonal scale model with an optional global rank-8 head."""

    def __init__(self, token_dim: int, rank: int = 0) -> None:
        super().__init__()
        if rank not in (0, LOW_RANK):
            raise ValueError(f"rank must be 0 or {LOW_RANK}")
        self.token_dim = int(token_dim)
        self.rank = int(rank)
        self.condition_proj = nn.Linear(CONDITION_DIM, 32)
        self.scale_1 = nn.Linear(self.token_dim + 32, 128)
        self.scale_2 = nn.Linear(128, 128)
        self.scale_out = nn.Linear(128, self.token_dim)
        if self.rank:
            self.rank_1 = nn.Linear(self.token_dim + CONDITION_DIM, 128)
            self.rank_2 = nn.Linear(128, 128)
            self.rank_out = nn.Linear(128, self.rank)

    @staticmethod
    def _bounded_variance(raw: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.softplus(raw) + VARIANCE_FLOOR, max=VARIANCE_CEILING)

    def forward(self, visual: torch.Tensor, condition: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        if visual.ndim != 3 or visual.shape[-1] != self.token_dim:
            raise ValueError("visual must be [B,L,D] with configured D")
        if condition.shape != (visual.shape[0], CONDITION_DIM):
            raise ValueError("condition must be [B,18]")
        cond = self.condition_proj(condition)[:, None, :].expand(-1, visual.shape[1], -1)
        hidden = F.silu(self.scale_1(torch.cat((visual, cond), dim=-1)))
        diagonal = self._bounded_variance(self.scale_out(F.silu(self.scale_2(hidden))))
        if not self.rank:
            return diagonal.reshape(visual.shape[0], -1), None
        pooled = visual.mean(dim=1)
        rank_hidden = F.silu(self.rank_1(torch.cat((pooled, condition), dim=-1)))
        rank_values = self._bounded_variance(self.rank_out(F.silu(self.rank_2(rank_hidden))))
        return diagonal.reshape(visual.shape[0], -1), rank_values


def deterministic_pca_basis(residuals: torch.Tensor, rank: int = LOW_RANK) -> torch.Tensor:
    if residuals.ndim < 2:
        raise ValueError("residuals need a sample dimension")
    flat = residuals.detach().to(dtype=torch.float64).reshape(residuals.shape[0], -1)
    centered = flat - flat.mean(dim=0, keepdim=True)
    if centered.shape[0] <= rank:
        raise ValueError("rank-8 basis requires at least nine residual samples")
    gram = centered @ centered.T
    eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    order = torch.argsort(eigenvalues, descending=True)[:rank]
    selected = eigenvalues[order]
    if bool(torch.any(selected <= 1e-12)):
        raise ValueError("fewer than eight positive residual singular values")
    basis = centered.T @ eigenvectors[:, order]
    basis = basis / torch.sqrt(selected).unsqueeze(0)
    for column in range(rank):
        pivot = torch.argmax(torch.abs(basis[:, column]))
        if basis[pivot, column] < 0:
            basis[:, column] *= -1
    return basis.to(dtype=residuals.dtype, device=residuals.device)


def innovation_terms(
    residual: torch.Tensor,
    diagonal_variance: torch.Tensor,
    *,
    basis: torch.Tensor | None = None,
    rank_variance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized innovation, Mahalanobis term, and log determinant."""

    flat = residual.reshape(residual.shape[0], -1)
    if diagonal_variance.shape != flat.shape:
        raise ValueError("diagonal variance must match flattened residual")
    if bool(torch.any(diagonal_variance < VARIANCE_FLOOR)):
        raise ValueError("diagonal variance is below the frozen floor")
    inverse_residual = flat / diagonal_variance
    mahalanobis = torch.sum(flat * inverse_residual, dim=-1)
    logdet = torch.sum(torch.log(diagonal_variance), dim=-1)
    if basis is not None or rank_variance is not None:
        if basis is None or rank_variance is None:
            raise ValueError("basis and rank variance must be supplied together")
        if basis.shape != (flat.shape[1], rank_variance.shape[1]) or rank_variance.shape[0] != flat.shape[0]:
            raise ValueError("low-rank shapes do not match residual")
        if bool(torch.any(rank_variance < VARIANCE_FLOOR)):
            raise ValueError("rank variance is below the frozen floor")
        weighted_basis = basis.unsqueeze(0) / diagonal_variance.unsqueeze(-1)
        small = torch.einsum("nr,bns->brs", basis, weighted_basis)
        small = small + torch.diag_embed(1.0 / rank_variance)
        projection = torch.einsum("nr,bn->br", basis, inverse_residual)
        solved = torch.linalg.solve(small, projection.unsqueeze(-1)).squeeze(-1)
        mahalanobis = mahalanobis - torch.sum(projection * solved, dim=-1)
        _, small_logdet = torch.linalg.slogdet(small)
        logdet = logdet + torch.sum(torch.log(rank_variance), dim=-1) + small_logdet
    normalized = torch.clamp(mahalanobis, min=0.0) / flat.shape[1]
    return normalized, mahalanobis, logdet


def covariance_nll(
    residual: torch.Tensor,
    diagonal_variance: torch.Tensor,
    *,
    basis: torch.Tensor | None = None,
    rank_variance: torch.Tensor | None = None,
) -> torch.Tensor:
    _, mahalanobis, logdet = innovation_terms(
        residual,
        diagonal_variance,
        basis=basis,
        rank_variance=rank_variance,
    )
    width = residual[0].numel()
    return 0.5 * torch.mean((mahalanobis + logdet) / width)


def dense_innovation_reference(
    residual: torch.Tensor,
    diagonal_variance: torch.Tensor,
    *,
    basis: torch.Tensor | None = None,
    rank_variance: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = residual.reshape(residual.shape[0], -1)
    if flat.shape[1] > 32:
        raise ValueError("dense references are restricted to n<=32")
    mahalanobis = []
    logdet = []
    for index in range(flat.shape[0]):
        covariance = torch.diag(diagonal_variance[index])
        if basis is not None and rank_variance is not None:
            covariance = covariance + basis @ torch.diag(rank_variance[index]) @ basis.T
        solved = torch.linalg.solve(covariance, flat[index])
        mahalanobis.append(flat[index] @ solved)
        logdet.append(torch.linalg.slogdet(covariance).logabsdet)
    return torch.stack(mahalanobis), torch.stack(logdet)


def nearest_rank_quantile(values: Sequence[float], probability: float) -> float:
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a nonempty finite vector")
    if not 0.0 < probability <= 1.0:
        raise ValueError("probability must be in (0,1]")
    index = min(array.size, int(math.ceil(probability * array.size))) - 1
    return float(array[index])


def episode_cluster_score(values: Sequence[float]) -> float:
    if len(values) < 16:
        raise ValueError("an episode cluster requires at least 16 scores")
    return nearest_rank_quantile(values, 0.90)


def conformal_threshold(episode_scores_by_task: Mapping[str, Sequence[float]], coverage: float) -> dict[str, Any]:
    if not episode_scores_by_task:
        raise ValueError("calibration tasks are required")
    counts = {task: len(values) for task, values in episode_scores_by_task.items()}
    if not counts or min(counts.values()) == 0 or len(set(counts.values())) != 1:
        raise ValueError("calibration episodes must have an equal positive task quota")
    values = np.sort(np.asarray([score for scores in episode_scores_by_task.values() for score in scores], dtype=np.float64))
    if not np.all(np.isfinite(values)):
        raise ValueError("episode scores must be finite")
    rank = min(len(values), int(math.ceil((len(values) + 1) * coverage)))
    return {
        "coverage": float(coverage),
        "episode_count": int(len(values)),
        "episodes_per_task": int(next(iter(counts.values()))),
        "one_indexed_rank": int(rank),
        "threshold": float(values[rank - 1]),
    }


def validate_manifest(
    manifest_rows: Sequence[Mapping[str, Any]],
    result_rows: Sequence[Mapping[str, Any]],
    *,
    allowed_partitions: Sequence[str] = ("discovery",),
) -> dict[str, Any]:
    planned = [pair_key(row) for row in manifest_rows]
    completed = [str(row.get("pair_key") or pair_key(row)) for row in result_rows]
    planned_set = set(planned)
    completed_set = set(completed)
    allowed = set(allowed_partitions)
    split_overlap = sum(str(row.get("partition", "discovery")) not in allowed for row in manifest_rows)
    return {
        "planned_count": len(planned),
        "completed_count": len(completed),
        "duplicate_manifest_key_count": len(planned) - len(planned_set),
        "duplicate_result_key_count": len(completed) - len(completed_set),
        "missing_manifest_key_count": len(planned_set - completed_set),
        "extra_result_key_count": len(completed_set - planned_set),
        "split_overlap_count": int(split_overlap),
        "passed": bool(
            len(planned) == len(planned_set)
            and len(completed) == len(completed_set)
            and planned_set == completed_set
            and split_overlap == 0
        ),
    }


def passthrough_queue_action(queue: np.ndarray, monitor_enabled: bool = False) -> tuple[np.ndarray, np.ndarray]:
    value = np.asarray(queue)
    if value.ndim != 2 or value.shape[1] != ACTION_DIM:
        raise ValueError("queue must be [H,7]")
    if monitor_enabled:
        raise ValueError("Stage 0A passthrough supports monitor-disabled identity only")
    copied = value.copy()
    return copied, copied[0].copy()


def action_validity(actions: np.ndarray, lower: float = -1.0, upper: float = 1.0) -> dict[str, Any]:
    value = np.asarray(actions, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != ACTION_DIM:
        raise ValueError("actions must be [N,7]")
    finite = np.isfinite(value)
    inside = finite & (value >= lower) & (value <= upper)
    return {
        "finite_fraction": float(np.mean(finite)),
        "inside_fraction": float(np.mean(inside)),
        "translation_inside_fraction": float(np.mean(inside[:, 0:3])),
        "rotation_inside_fraction": float(np.mean(inside[:, 3:6])),
        "gripper_inside_fraction": float(np.mean(inside[:, 6:7])),
        "minimum": float(np.min(value)),
        "maximum": float(np.max(value)),
    }


def auroc_average_ranks(negative_scores: Sequence[float], positive_scores: Sequence[float]) -> float:
    """Compute AUROC with average ranks for ties and higher scores as positive."""

    negative = np.asarray(negative_scores, dtype=np.float64)
    positive = np.asarray(positive_scores, dtype=np.float64)
    if negative.ndim != 1 or positive.ndim != 1 or not len(negative) or not len(positive):
        raise ValueError("both score groups must be nonempty vectors")
    values = np.concatenate((negative, positive))
    if not np.all(np.isfinite(values)):
        raise ValueError("scores must be finite")
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and values[order[stop]] == values[order[start]]:
            stop += 1
        average_rank = 0.5 * ((start + 1) + stop)
        ranks[order[start:stop]] = average_rank
        start = stop
    positive_rank_sum = float(np.sum(ranks[len(negative) :]))
    return (positive_rank_sum - len(positive) * (len(positive) + 1) / 2.0) / (
        len(negative) * len(positive)
    )


@dataclass(frozen=True)
class Stage0DecisionInputs:
    completed_pairs: int
    planned_pairs: int
    exception_count: int
    manifest_passed: bool
    source_passed: bool
    latent_passed: bool
    gradient_passed: bool
    algebra_passed: bool
    calibration_passed: bool
    passthrough_passed: bool
    reload_passed: bool
    action_validity_passed: bool
    forbidden_reads_zero: bool


def classify_stage0a(inputs: Stage0DecisionInputs) -> str:
    if not inputs.source_passed or not inputs.latent_passed:
        return "NICE_STAGE_0A_DATA_FAILURE"
    passed = bool(
        inputs.completed_pairs == inputs.planned_pairs == 128
        and inputs.exception_count == 0
        and inputs.manifest_passed
        and inputs.gradient_passed
        and inputs.algebra_passed
        and inputs.calibration_passed
        and inputs.passthrough_passed
        and inputs.reload_passed
        and inputs.action_validity_passed
        and inputs.forbidden_reads_zero
    )
    return "NICE_STAGE_0A_PASS_STAGE_0B_ALLOWED" if passed else "NICE_STAGE_0A_IMPLEMENTATION_FAILURE"
