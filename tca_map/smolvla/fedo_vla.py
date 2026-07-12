"""Feedback execution-disturbance observer for VLA action deployment.

FEDO-VLA operates between a frozen policy and a low-level action interface. It
uses command/realized-action feedback, not simulator state or success labels, at
inference time.
"""

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
class FEDOConfig:
    action_dim: int = 7
    hidden_dim: int = 64
    max_residual_norm: float = 0.75

    @property
    def input_dim(self) -> int:
        # current action, previous command, previous realized action, previous
        # execution error, step_fraction/norm metadata, phase one-hot, task code
        return self.action_dim * 4 + 6 + len(PHASES) + 1


@dataclass(frozen=True)
class FEDOExample:
    features: list[float]
    target_residual: list[float]
    step_fraction: float
    phase: str
    uses_feedback: bool
    uses_phase: bool


class FEDOCompensator(nn.Module):
    def __init__(self, config: FEDOConfig):
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
        residual = self.net(features)
        max_norm = float(self.config.max_residual_norm)
        if max_norm <= 0.0:
            return residual
        norm = torch.linalg.norm(residual, dim=-1, keepdim=True).clamp_min(1e-6)
        scale = torch.clamp(max_norm / norm, max=1.0)
        return residual * scale


def phase_from_fraction(step_fraction: float) -> str:
    frac = float(np.clip(step_fraction, 0.0, 1.0))
    if frac < 0.25:
        return "approach"
    if frac < 0.48:
        return "contact"
    if frac < 0.80:
        return "transport"
    return "release"


def _as_action(action: Sequence[float] | np.ndarray | None, action_dim: int) -> np.ndarray:
    if action is None:
        return np.zeros(int(action_dim), dtype=np.float32)
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size != int(action_dim):
        raise ValueError(f"expected action_dim={action_dim}, got {arr.size}")
    return arr


def _task_code(task_key: str | None) -> float:
    key = str(task_key or "")
    if "libero_10" in key:
        return 1.0
    if "libero_spatial" in key:
        return -1.0
    return 0.0


def fault_gain_bias(*, identity: int, step_fraction: float, action_dim: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic controlled action-realization fault for prototypes."""

    phase = phase_from_fraction(step_fraction)
    identity_band = (int(identity) % 3) - 1
    severity = 0.08 * float(identity_band)
    gain = np.ones(int(action_dim), dtype=np.float32)
    bias = np.zeros(int(action_dim), dtype=np.float32)

    if phase == "approach":
        gain[:3] = np.asarray([0.72 + severity, 0.80 + severity * 0.5, 0.88], dtype=np.float32)
        bias[:3] = np.asarray([0.018, -0.010, 0.0], dtype=np.float32)
    elif phase == "contact":
        gain[:3] = np.asarray([0.64 + severity, 0.70 + severity * 0.5, 0.78], dtype=np.float32)
        gain[6] = 0.72 + severity * 0.5
        bias[:3] = np.asarray([0.012, -0.018, -0.006], dtype=np.float32)
    elif phase == "transport":
        gain[:3] = np.asarray([0.78 + severity * 0.5, 0.68 + severity, 0.82], dtype=np.float32)
        bias[:3] = np.asarray([-0.010, 0.016, 0.004], dtype=np.float32)
    else:
        gain[:3] = np.asarray([0.82, 0.76 + severity, 0.68 + severity * 0.5], dtype=np.float32)
        gain[6] = 0.80
        bias[:3] = np.asarray([0.0, 0.010, -0.012], dtype=np.float32)

    return np.clip(gain, 0.45, 1.20), bias


def mean_fault_gain_bias(*, step_fraction: float, action_dim: int = 7) -> tuple[np.ndarray, np.ndarray]:
    gains = []
    biases = []
    for identity in (20260713, 20260714, 20260715):
        gain, bias = fault_gain_bias(identity=identity, step_fraction=step_fraction, action_dim=action_dim)
        gains.append(gain)
        biases.append(bias)
    return np.mean(gains, axis=0).astype(np.float32), np.mean(biases, axis=0).astype(np.float32)


def apply_control_fault(
    command: Sequence[float] | np.ndarray,
    *,
    identity: int,
    step_fraction: float,
    action_dim: int = 7,
) -> np.ndarray:
    cmd = _as_action(command, action_dim)
    gain, bias = fault_gain_bias(identity=int(identity), step_fraction=float(step_fraction), action_dim=action_dim)
    realized = gain * cmd + bias
    return np.clip(realized, -1.0, 1.0).astype(np.float32)


def inverse_fault_command(
    intended_action: Sequence[float] | np.ndarray,
    *,
    identity: int | None,
    step_fraction: float,
    action_dim: int = 7,
) -> np.ndarray:
    intended = _as_action(intended_action, action_dim)
    if identity is None:
        gain, bias = mean_fault_gain_bias(step_fraction=float(step_fraction), action_dim=action_dim)
    else:
        gain, bias = fault_gain_bias(identity=int(identity), step_fraction=float(step_fraction), action_dim=action_dim)
    command = (intended - bias) / np.clip(gain, 1e-4, None)
    return np.clip(command, -1.0, 1.0).astype(np.float32)


def static_inverse_gain_action(
    intended_action: Sequence[float] | np.ndarray,
    *,
    step_fraction: float,
    action_dim: int = 7,
) -> np.ndarray:
    return inverse_fault_command(intended_action, identity=None, step_fraction=step_fraction, action_dim=action_dim)


def apex_feedback_proxy_action(
    intended_action: Sequence[float] | np.ndarray,
    *,
    previous_command: Sequence[float] | np.ndarray | None,
    previous_realized: Sequence[float] | np.ndarray | None,
    feedback_gain: float = 0.70,
    smoothing: float = 0.15,
    action_dim: int = 7,
) -> np.ndarray:
    intended = _as_action(intended_action, action_dim)
    prev_cmd = _as_action(previous_command, action_dim)
    prev_realized = _as_action(previous_realized, action_dim)
    error = prev_cmd - prev_realized
    smooth_term = np.zeros_like(intended) if previous_command is None else prev_cmd - intended
    command = intended + float(feedback_gain) * error + float(smoothing) * smooth_term
    return np.clip(command, -1.0, 1.0).astype(np.float32)


def make_fedo_features(
    intended_action: Sequence[float] | np.ndarray,
    *,
    previous_command: Sequence[float] | np.ndarray | None,
    previous_realized: Sequence[float] | np.ndarray | None,
    step_fraction: float,
    task_key: str | None,
    config: FEDOConfig,
    use_feedback: bool = True,
    use_phase: bool = True,
) -> list[float]:
    action = _as_action(intended_action, config.action_dim)
    prev_cmd = _as_action(previous_command, config.action_dim)
    prev_realized = _as_action(previous_realized, config.action_dim)
    if not use_feedback:
        prev_cmd = np.zeros_like(prev_cmd)
        prev_realized = np.zeros_like(prev_realized)
    prev_error = prev_cmd - prev_realized if use_feedback else np.zeros_like(prev_cmd)
    phase = phase_from_fraction(step_fraction) if use_phase else ""
    phase_one_hot = np.asarray([1.0 if phase == item else 0.0 for item in PHASES], dtype=np.float32)
    if not use_phase:
        phase_one_hot[:] = 0.0
    meta = np.asarray(
        [
            float(np.clip(step_fraction, 0.0, 1.0)),
            float(np.linalg.norm(action)),
            float(np.linalg.norm(prev_cmd)),
            float(np.linalg.norm(prev_realized)),
            float(np.linalg.norm(prev_error)),
            float(np.linalg.norm(action - prev_realized)),
        ],
        dtype=np.float32,
    )
    features = np.concatenate(
        [
            action,
            prev_cmd,
            prev_realized,
            prev_error,
            meta,
            phase_one_hot,
            np.asarray([_task_code(task_key)], dtype=np.float32),
        ],
        axis=0,
    )
    if features.size != int(config.input_dim):
        raise RuntimeError(f"feature width mismatch: {features.size} != {config.input_dim}")
    return [float(value) for value in features]


def build_fedo_examples(
    intended_actions: Sequence[Sequence[float] | np.ndarray],
    *,
    identities: Sequence[int],
    task_keys: Sequence[str],
    config: FEDOConfig,
    use_feedback: bool = True,
    use_phase: bool = True,
) -> list[FEDOExample]:
    if not intended_actions:
        raise ValueError("intended_actions is empty")
    examples: list[FEDOExample] = []
    previous_command: np.ndarray | None = None
    previous_realized: np.ndarray | None = None
    total = max(1, len(intended_actions) - 1)
    for index, raw_action in enumerate(intended_actions):
        identity = int(identities[index % len(identities)])
        task_key = str(task_keys[index % len(task_keys)])
        step_fraction = float(index) / float(total)
        intended = _as_action(raw_action, config.action_dim)
        target_command = inverse_fault_command(
            intended,
            identity=identity,
            step_fraction=step_fraction,
            action_dim=config.action_dim,
        )
        target_residual = np.clip(target_command - intended, -float(config.max_residual_norm), float(config.max_residual_norm))
        features = make_fedo_features(
            intended,
            previous_command=previous_command,
            previous_realized=previous_realized,
            step_fraction=step_fraction,
            task_key=task_key,
            config=config,
            use_feedback=use_feedback,
            use_phase=use_phase,
        )
        examples.append(
            FEDOExample(
                features=features,
                target_residual=[float(value) for value in target_residual],
                step_fraction=step_fraction,
                phase=phase_from_fraction(step_fraction),
                uses_feedback=bool(use_feedback),
                uses_phase=bool(use_phase),
            )
        )
        previous_command = target_command
        previous_realized = apply_control_fault(
            target_command,
            identity=identity,
            step_fraction=step_fraction,
            action_dim=config.action_dim,
        )
    return examples


def train_fedo_compensator(
    examples: Sequence[FEDOExample],
    *,
    config: FEDOConfig,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[FEDOCompensator, dict[str, Any]]:
    if not examples:
        raise ValueError("cannot train FEDO without examples")
    torch.manual_seed(int(seed))
    model = FEDOCompensator(config)
    features = torch.tensor([row.features for row in examples], dtype=torch.float32)
    targets = torch.tensor([row.target_residual for row in examples], dtype=torch.float32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: list[float] = []
    grad_norms: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(features)
        loss = torch.nn.functional.smooth_l1_loss(prediction, targets)
        loss.backward()
        total_grad_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_sq += float(torch.sum(param.grad.detach() ** 2).item())
        grad_norm = float(total_grad_sq**0.5)
        optimizer.step()
        losses.append(float(loss.detach().item()))
        grad_norms.append(grad_norm)
    stats = {
        "example_count": int(len(examples)),
        "epochs": int(epochs),
        "initial_loss": float(losses[0]),
        "final_loss": float(losses[-1]),
        "loss_decreased": bool(losses[-1] < losses[0]),
        "max_grad_norm": float(max(grad_norms)),
        "finite_gradients": bool(np.isfinite(grad_norms).all()),
        "uses_feedback": bool(any(row.uses_feedback for row in examples)),
        "uses_phase": bool(any(row.uses_phase for row in examples)),
    }
    return model, stats


def predict_fedo_command(
    model: FEDOCompensator,
    features: Sequence[float] | np.ndarray,
    intended_action: Sequence[float] | np.ndarray,
) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        tensor = torch.tensor(np.asarray(features, dtype=np.float32).reshape(1, -1), dtype=torch.float32)
        residual = model(tensor).detach().cpu().numpy().reshape(-1)
    intended = _as_action(intended_action, model.config.action_dim)
    return np.clip(intended + residual.astype(np.float32), -1.0, 1.0).astype(np.float32)


def save_fedo_checkpoint(path: Path | str, model: FEDOCompensator, stats: Mapping[str, Any] | None = None) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(model.config),
            "state_dict": model.state_dict(),
            "stats": dict(stats or {}),
        },
        target,
    )


def load_fedo_checkpoint(path: Path | str, *, map_location: str = "cpu") -> tuple[FEDOCompensator, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    config = FEDOConfig(**payload["config"])
    model = FEDOCompensator(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload.get("stats") or {})


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024, ), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_no_privileged_inference_fields(fields: Iterable[str]) -> None:
    present = {str(field) for field in fields}
    forbidden = sorted(present & PRIVILEGED_INFERENCE_FIELDS)
    if forbidden:
        raise ValueError(f"privileged FEDO inference fields: {forbidden}")
