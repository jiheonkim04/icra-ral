"""Posterior-transition conservative policy head for local VLA prototypes."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import nn


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

PHASES = ("approach", "contact", "transport", "release")


@dataclass(frozen=True)
class PTCConfig:
    state_dim: int = 6
    action_dim: int = 7
    hidden_dim: int = 64
    transition_blend: float = 0.65
    log_std_min: float = -4.0
    log_std_max: float = 1.0
    beta_log_std: float = 0.01
    conservative_lambda: float = 0.05

    @property
    def input_dim(self) -> int:
        return self.state_dim * 2 + len(PHASES) + 2

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PTCExample:
    state: list[float]
    transition: list[float]
    action: list[float]
    task_key: str
    step_fraction: float
    phase: str
    uses_transition: bool = True


class PTCPolicyHead(nn.Module):
    def __init__(self, config: PTCConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim * 2),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        raw = self.net(features)
        mean, log_std = torch.split(raw, self.config.action_dim, dim=-1)
        log_std = torch.clamp(log_std, float(self.config.log_std_min), float(self.config.log_std_max))
        return mean, log_std


def assert_no_privileged_inference_fields(fields: Iterable[str]) -> None:
    present = {str(field) for field in fields}
    forbidden = sorted(present & PRIVILEGED_INFERENCE_FIELDS)
    if forbidden:
        raise ValueError(f"privileged PTC inference fields: {forbidden}")


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


def _as_vector(values: Sequence[float] | np.ndarray | None, dim: int) -> np.ndarray:
    if values is None:
        return np.zeros(int(dim), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size < int(dim):
        arr = np.pad(arr, (0, int(dim) - arr.size))
    if arr.size > int(dim):
        arr = arr[: int(dim)]
    return arr.astype(np.float32)


def mean_action_key(phase: str, task_key: str | None) -> str:
    return f"{phase}|{task_code(task_key):.1f}"


def make_ptc_features(
    state: Sequence[float] | np.ndarray,
    transition: Sequence[float] | np.ndarray | None,
    *,
    step_fraction: float,
    task_key: str | None,
    config: PTCConfig,
    use_transition: bool = True,
) -> list[float]:
    state_vec = _as_vector(state, config.state_dim)
    transition_vec = _as_vector(transition, config.state_dim)
    if not use_transition:
        transition_vec = np.zeros_like(transition_vec)
    phase = phase_from_fraction(step_fraction)
    phase_one_hot = np.asarray([1.0 if phase == item else 0.0 for item in PHASES], dtype=np.float32)
    meta = np.asarray([float(np.clip(step_fraction, 0.0, 1.0)), task_code(task_key)], dtype=np.float32)
    features = np.concatenate([state_vec, transition_vec, phase_one_hot, meta], axis=0)
    if features.size != int(config.input_dim):
        raise RuntimeError(f"PTC feature width mismatch: {features.size} != {config.input_dim}")
    return [float(value) for value in features]


def transition_context(
    *,
    current_state: Sequence[float] | np.ndarray,
    previous_state: Sequence[float] | np.ndarray | None,
    prior_transition: Sequence[float] | np.ndarray | None,
    config: PTCConfig,
) -> np.ndarray:
    current = _as_vector(current_state, config.state_dim)
    if previous_state is None:
        recent = np.zeros(config.state_dim, dtype=np.float32)
    else:
        recent = current - _as_vector(previous_state, config.state_dim)
    prior = _as_vector(prior_transition, config.state_dim)
    blend = float(np.clip(config.transition_blend, 0.0, 1.0))
    return (blend * recent + (1.0 - blend) * prior).astype(np.float32)


def _stats_from_examples(examples: Sequence[PTCExample], config: PTCConfig) -> dict[str, Any]:
    if not examples:
        raise ValueError("PTC examples are empty")
    actions = np.asarray([_as_vector(row.action, config.action_dim) for row in examples], dtype=np.float32)
    transitions = np.asarray([_as_vector(row.transition, config.state_dim) for row in examples], dtype=np.float32)
    phase_task_mean_actions: dict[str, list[float]] = {}
    phase_task_mean_transitions: dict[str, list[float]] = {}
    for phase in PHASES:
        for code in (-1.0, 0.0, 1.0):
            key = f"{phase}|{code:.1f}"
            rows = [i for i, row in enumerate(examples) if mean_action_key(row.phase, row.task_key) == key]
            if rows:
                phase_task_mean_actions[key] = [float(value) for value in np.mean(actions[rows], axis=0)]
                phase_task_mean_transitions[key] = [float(value) for value in np.mean(transitions[rows], axis=0)]
    return {
        "global_mean_action": [float(value) for value in np.mean(actions, axis=0)],
        "phase_task_mean_actions": phase_task_mean_actions,
        "phase_task_mean_transitions": phase_task_mean_transitions,
    }


def mean_action_from_stats(stats: Mapping[str, Any], *, phase: str, task_key: str | None, config: PTCConfig) -> np.ndarray:
    key = mean_action_key(phase, task_key)
    phase_actions = stats.get("phase_task_mean_actions") or {}
    if key in phase_actions:
        return _as_vector(phase_actions[key], config.action_dim)
    return _as_vector(stats.get("global_mean_action"), config.action_dim)


def transition_prior_from_stats(stats: Mapping[str, Any], *, phase: str, task_key: str | None, config: PTCConfig) -> np.ndarray:
    key = mean_action_key(phase, task_key)
    transitions = stats.get("phase_task_mean_transitions") or {}
    if key in transitions:
        return _as_vector(transitions[key], config.state_dim)
    return np.zeros(config.state_dim, dtype=np.float32)


def train_ptc_policy(
    examples: Sequence[PTCExample],
    *,
    config: PTCConfig,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
    use_transition: bool = True,
) -> tuple[PTCPolicyHead, dict[str, Any]]:
    if not examples:
        raise ValueError("cannot train PTC without examples")
    torch.manual_seed(int(seed))
    stats = _stats_from_examples(examples, config)
    model = PTCPolicyHead(config)
    features = torch.tensor(
        [
            make_ptc_features(
                row.state,
                row.transition,
                step_fraction=row.step_fraction,
                task_key=row.task_key,
                config=config,
                use_transition=use_transition,
            )
            for row in examples
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor(np.asarray([_as_vector(row.action, config.action_dim) for row in examples], dtype=np.float32), dtype=torch.float32)
    conservative = torch.tensor(
        np.asarray(
        [
            mean_action_from_stats(stats, phase=row.phase, task_key=row.task_key, config=config)
            for row in examples
        ],
        dtype=np.float32,
        ),
        dtype=torch.float32,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: list[float] = []
    grad_norms: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        mean, log_std = model(features)
        mse = torch.nn.functional.mse_loss(mean, targets)
        std_penalty = torch.mean(log_std**2)
        conservative_loss = torch.nn.functional.mse_loss(mean, conservative)
        loss = mse + float(config.beta_log_std) * std_penalty + float(config.conservative_lambda) * conservative_loss
        loss.backward()
        grad_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_sq += float(torch.sum(param.grad.detach() ** 2).item())
        grad_norms.append(float(grad_sq**0.5))
        optimizer.step()
        losses.append(float(loss.detach().item()))
    stats.update(
        {
            "example_count": int(len(examples)),
            "epochs": int(epochs),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "loss_decreased": bool(losses[-1] < losses[0]),
            "max_grad_norm": float(max(grad_norms)),
            "finite_gradients": bool(np.isfinite(grad_norms).all()),
            "uses_transition": bool(use_transition),
        }
    )
    return model, stats


def predict_ptc_action(
    model: PTCPolicyHead,
    *,
    state: Sequence[float] | np.ndarray,
    transition: Sequence[float] | np.ndarray | None,
    step_fraction: float,
    task_key: str | None,
    use_transition: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    features = make_ptc_features(
        state,
        transition,
        step_fraction=step_fraction,
        task_key=task_key,
        config=model.config,
        use_transition=use_transition,
    )
    with torch.inference_mode():
        tensor = torch.tensor(np.asarray(features, dtype=np.float32).reshape(1, -1), dtype=torch.float32)
        mean, log_std = model(tensor)
    action = torch.clamp(mean, -1.0, 1.0).detach().cpu().numpy().reshape(-1)
    scale = torch.exp(log_std).detach().cpu().numpy().reshape(-1)
    return action.astype(np.float32), scale.astype(np.float32)


def save_ptc_checkpoint(path: Path | str, model: PTCPolicyHead, stats: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"config": model.config.to_json(), "state_dict": model.state_dict(), "stats": dict(stats)}, target)


def load_ptc_checkpoint(path: Path | str, *, map_location: str = "cpu") -> tuple[PTCPolicyHead, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    config = PTCConfig(**payload["config"])
    model = PTCPolicyHead(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload.get("stats") or {})


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
