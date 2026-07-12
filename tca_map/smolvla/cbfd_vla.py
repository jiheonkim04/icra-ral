"""Cross-backbone failure-set distillation helpers."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


PRIVILEGED_INFERENCE_FIELDS = {
    "success",
    "reward",
    "task_outcome",
    "teacher_action",
    "teacher_success",
    "future_action",
    "future_observation",
    "sim_state",
    "mujoco_state",
    "reset_identity",
    "bddl_predicate",
    "object_pose",
}

PHASES = ("approach", "contact", "transport", "release")


@dataclass(frozen=True)
class CBFDConfig:
    state_dim: int = 8
    action_dim: int = 7
    hidden_dim: int = 96
    failure_weight: float = 3.0
    retention_weight: float = 1.0
    delta_l2: float = 0.01

    @property
    def input_dim(self) -> int:
        return self.state_dim + len(PHASES) + 2

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CBFDExample:
    state: list[float]
    action: list[float]
    task_key: str
    step_fraction: float
    source: str
    failure_weight: float = 1.0


class CBFDPolicyHead(nn.Module):
    def __init__(self, config: CBFDConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features)


def assert_no_privileged_inference_fields(fields: Iterable[str]) -> None:
    present = {str(field) for field in fields}
    forbidden = sorted(present & PRIVILEGED_INFERENCE_FIELDS)
    if forbidden:
        raise ValueError(f"privileged CBFD inference fields: {forbidden}")


def _as_vector(values: Sequence[float] | np.ndarray | None, dim: int) -> np.ndarray:
    if values is None:
        return np.zeros(int(dim), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size < int(dim):
        arr = np.pad(arr, (0, int(dim) - arr.size))
    if arr.size > int(dim):
        arr = arr[: int(dim)]
    return arr.astype(np.float32)


def phase_from_fraction(step_fraction: float) -> str:
    frac = float(np.clip(step_fraction, 0.0, 1.0))
    if frac < 0.25:
        return "approach"
    if frac < 0.50:
        return "contact"
    if frac < 0.80:
        return "transport"
    return "release"


def task_code(task_key: str | None) -> float:
    key = str(task_key or "")
    if "libero_10" in key:
        return 1.0
    if "libero_spatial" in key:
        return -1.0
    return 0.0


def task_key(suite: str, task_id: int) -> str:
    return f"{suite}/task_{int(task_id)}"


def make_cbfd_features(
    state: Sequence[float] | np.ndarray,
    *,
    step_fraction: float,
    task_key_value: str | None,
    config: CBFDConfig,
) -> list[float]:
    state_vec = _as_vector(state, config.state_dim)
    phase = phase_from_fraction(step_fraction)
    phase_one_hot = np.asarray([1.0 if phase == item else 0.0 for item in PHASES], dtype=np.float32)
    meta = np.asarray([float(np.clip(step_fraction, 0.0, 1.0)), task_code(task_key_value)], dtype=np.float32)
    features = np.concatenate([state_vec, phase_one_hot, meta], axis=0)
    if features.size != int(config.input_dim):
        raise RuntimeError(f"CBFD feature width mismatch: {features.size} != {config.input_dim}")
    return [float(value) for value in features]


def train_cbfd_policy(
    examples: Sequence[CBFDExample],
    *,
    config: CBFDConfig,
    epochs: int = 220,
    lr: float = 1e-3,
    seed: int = 0,
    include_retention: bool = True,
    use_failure_weights: bool = True,
) -> tuple[CBFDPolicyHead, dict[str, Any]]:
    if not examples:
        raise ValueError("cannot train CBFD without examples")
    selected = [row for row in examples if include_retention or row.source != "retention"]
    if not selected:
        raise ValueError("CBFD selected examples are empty")
    torch.manual_seed(int(seed))
    model = CBFDPolicyHead(config)
    features = torch.tensor(
        [
            make_cbfd_features(row.state, step_fraction=row.step_fraction, task_key_value=row.task_key, config=config)
            for row in selected
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(np.asarray([_as_vector(row.action, config.action_dim) for row in selected], dtype=np.float32), dtype=torch.float32)
    weights_np: list[float] = []
    for row in selected:
        if row.source == "retention":
            weights_np.append(float(config.retention_weight))
        elif use_failure_weights:
            weights_np.append(float(config.failure_weight) * float(row.failure_weight))
        else:
            weights_np.append(1.0)
    weights = torch.tensor(np.asarray(weights_np, dtype=np.float32).reshape(-1, 1), dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: list[float] = []
    grad_norms: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        pred = model(features)
        abs_err = torch.abs(pred - targets)
        imitation = torch.mean(weights * abs_err)
        l2 = torch.mean(pred**2)
        loss = imitation + float(config.delta_l2) * l2
        loss.backward()
        grad_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_sq += float(torch.sum(param.grad.detach() ** 2).item())
        grad_norms.append(float(grad_sq**0.5))
        optimizer.step()
        losses.append(float(loss.detach().item()))
    stats = {
        "example_count": int(len(selected)),
        "teacher_example_count": int(sum(1 for row in selected if row.source == "teacher")),
        "retention_example_count": int(sum(1 for row in selected if row.source == "retention")),
        "include_retention": bool(include_retention),
        "use_failure_weights": bool(use_failure_weights),
        "epochs": int(epochs),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "loss_decreased": bool(losses[-1] < losses[0]),
        "finite_gradients": bool(np.isfinite(grad_norms).all()),
        "max_grad_norm": float(max(grad_norms)),
        "config": config.to_json(),
    }
    return model, stats


def predict_cbfd_action(
    model: CBFDPolicyHead,
    *,
    state: Sequence[float] | np.ndarray,
    step_fraction: float,
    task_key_value: str | None,
) -> np.ndarray:
    model.eval()
    features = make_cbfd_features(state, step_fraction=step_fraction, task_key_value=task_key_value, config=model.config)
    with torch.inference_mode():
        tensor = torch.tensor(np.asarray(features, dtype=np.float32).reshape(1, -1), dtype=torch.float32)
        action = model(tensor)
    return torch.clamp(action, -1.0, 1.0).detach().cpu().numpy().reshape(-1).astype(np.float32)


def memory_action(
    examples: Sequence[CBFDExample],
    *,
    state: Sequence[float] | np.ndarray,
    step_fraction: float,
    task_key_value: str | None,
    config: CBFDConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    teacher = [row for row in examples if row.source == "teacher"]
    if not teacher:
        raise ValueError("teacher memory has no teacher examples")
    query = _as_vector(state, config.state_dim)
    phase = phase_from_fraction(step_fraction)
    best_index = 0
    best_score = float("inf")
    for index, row in enumerate(teacher):
        row_state = _as_vector(row.state, config.state_dim)
        state_dist = float(np.linalg.norm(query - row_state))
        phase_penalty = 0.25 if phase_from_fraction(row.step_fraction) != phase else 0.0
        task_penalty = 0.5 if str(row.task_key) != str(task_key_value) else 0.0
        step_penalty = abs(float(row.step_fraction) - float(step_fraction))
        score = state_dist + phase_penalty + task_penalty + step_penalty
        if score < best_score:
            best_index = index
            best_score = score
    chosen = teacher[best_index]
    return _as_vector(chosen.action, config.action_dim), {
        "memory_score": float(best_score),
        "memory_task_key": chosen.task_key,
        "memory_step_fraction": float(chosen.step_fraction),
    }


def stage_a_decision(summary: Mapping[str, Any], *, strongest_baseline: str = "direct_distill_proxy") -> str:
    by_variant = summary.get("by_variant") or {}
    full = by_variant.get("cbfd_full") or {}
    strongest = by_variant.get(strongest_baseline) or {}
    full_total = int(full.get("total", 0) or 0)
    full_successes = int(full.get("successes", 0) or 0)
    strongest_successes = int(strongest.get("successes", 0) or 0)
    full_rate = float(full.get("task_balanced_success_rate", 0.0) or 0.0)
    strongest_rate = float(strongest.get("task_balanced_success_rate", 0.0) or 0.0)
    mechanism_active = bool(summary.get("mechanism_active"))
    if not mechanism_active:
        return "STAGE_A_MECHANISM_INVALID_KILL"
    if full_total >= 10 and full_successes == 0 and strongest_successes >= 4:
        return "STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE"
    if strongest_rate - full_rate >= 0.30:
        return "STAGE_A_PERMANENT_KILL_CLEARLY_WORSE"
    if full_rate > strongest_rate:
        return "STAGE_A_POSITIVE_TO_STAGE_B_REQUIRED"
    return "STAGE_A_NON_GO_TO_STAGE_B_REQUIRED"


def save_cbfd_checkpoint(path: Path | str, model: CBFDPolicyHead, stats: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config.to_json(), "state_dict": model.state_dict(), "stats": dict(stats)}, target)


def load_cbfd_checkpoint(path: Path | str, *, map_location: str = "cpu") -> tuple[CBFDPolicyHead, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    config = CBFDConfig(**payload["config"])
    model = CBFDPolicyHead(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload.get("stats") or {})


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
