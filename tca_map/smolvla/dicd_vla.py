"""Delay-indexed action-chunk adapter for VLA policies.

DICD-VLA is intentionally small: it operates on already postprocessed action
chunks and recent executed actions.  It does not inspect simulator state or
task outcome at inference time.
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
    "reset_identity",
    "task_outcome",
}


@dataclass(frozen=True)
class DICDConfig:
    action_dim: int = 7
    chunk_len: int = 8
    history_len: int = 2
    hidden_dim: int = 64

    @property
    def input_dim(self) -> int:
        return self.action_dim * (self.chunk_len + self.history_len) + 4


@dataclass(frozen=True)
class DICDExample:
    features: list[float]
    target: list[float]
    delay: int
    step_index: int
    uses_history: bool


class DelayIndexedChunkAdapter(nn.Module):
    def __init__(self, config: DICDConfig):
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


def _as_action_vector(action: Sequence[float] | np.ndarray, action_dim: int) -> np.ndarray:
    arr = np.asarray(action, dtype=np.float32).reshape(-1)
    if arr.size != int(action_dim):
        raise ValueError(f"expected action_dim={action_dim}, got {arr.size}")
    return arr


def _as_chunk(chunk: Sequence[Sequence[float]] | np.ndarray, config: DICDConfig) -> np.ndarray:
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != int(config.action_dim):
        raise ValueError(f"expected chunk shape (*, {config.action_dim}), got {arr.shape}")
    out = np.zeros((config.chunk_len, config.action_dim), dtype=np.float32)
    keep = min(int(config.chunk_len), int(arr.shape[0]))
    out[:keep] = arr[:keep]
    return out


def direct_chunk_index_action(chunk: Sequence[Sequence[float]] | np.ndarray, delay: int, config: DICDConfig) -> np.ndarray:
    arr = np.asarray(chunk, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    if arr.ndim != 2 or arr.shape[1] != int(config.action_dim):
        raise ValueError(f"expected chunk shape (*, {config.action_dim}), got {arr.shape}")
    index = int(np.clip(int(delay), 0, max(0, arr.shape[0] - 1)))
    return np.asarray(arr[index], dtype=np.float32).reshape(1, -1)


def make_dicd_features(
    chunk: Sequence[Sequence[float]] | np.ndarray,
    *,
    history: Iterable[Sequence[float] | np.ndarray] = (),
    delay: int,
    step_fraction: float,
    config: DICDConfig,
    use_history: bool = True,
) -> list[float]:
    chunk_arr = _as_chunk(chunk, config)
    hist = np.zeros((config.history_len, config.action_dim), dtype=np.float32)
    if use_history:
        history_rows = [_as_action_vector(row, config.action_dim) for row in history]
        history_rows = history_rows[-int(config.history_len) :]
        if history_rows:
            hist[-len(history_rows) :] = np.stack(history_rows, axis=0)

    first = chunk_arr[0]
    indexed = direct_chunk_index_action(chunk_arr, delay, config).reshape(-1)
    previous = hist[-1] if use_history and int(config.history_len) > 0 else np.zeros(config.action_dim, dtype=np.float32)
    meta = np.asarray(
        [
            float(delay) / max(1.0, float(config.chunk_len - 1)),
            float(np.clip(step_fraction, 0.0, 1.0)),
            float(np.linalg.norm(indexed - first)),
            float(np.linalg.norm(first - previous)),
        ],
        dtype=np.float32,
    )
    features = np.concatenate([chunk_arr.reshape(-1), hist.reshape(-1), meta], axis=0)
    if features.size != config.input_dim:
        raise RuntimeError(f"feature width mismatch: {features.size} != {config.input_dim}")
    return [float(value) for value in features]


def build_dicd_examples(
    chunks: Sequence[Sequence[Sequence[float]] | np.ndarray],
    executed_actions: Sequence[Sequence[float] | np.ndarray],
    *,
    delay: int,
    config: DICDConfig,
    include_clean: bool = True,
    use_history: bool = True,
) -> list[DICDExample]:
    if len(chunks) != len(executed_actions):
        raise ValueError("chunks and executed_actions must have the same length")
    examples: list[DICDExample] = []
    max_index = len(chunks) - max(0, int(delay))
    for index in range(max(0, max_index)):
        history = executed_actions[max(0, index - config.history_len) : index]
        step_fraction = 0.0 if len(chunks) <= 1 else index / float(len(chunks) - 1)
        examples.append(
            DICDExample(
                features=make_dicd_features(
                    chunks[index],
                    history=history,
                    delay=int(delay),
                    step_fraction=step_fraction,
                    config=config,
                    use_history=use_history,
                ),
                target=[float(value) for value in _as_action_vector(executed_actions[index + int(delay)], config.action_dim)],
                delay=int(delay),
                step_index=int(index),
                uses_history=bool(use_history),
            )
        )
        if include_clean:
            examples.append(
                DICDExample(
                    features=make_dicd_features(
                        chunks[index],
                        history=history,
                        delay=0,
                        step_fraction=step_fraction,
                        config=config,
                        use_history=use_history,
                    ),
                    target=[float(value) for value in direct_chunk_index_action(chunks[index], 0, config).reshape(-1)],
                    delay=0,
                    step_index=int(index),
                    uses_history=bool(use_history),
                )
            )
    return examples


def train_dicd_adapter(
    examples: Sequence[DICDExample],
    *,
    config: DICDConfig,
    epochs: int = 200,
    lr: float = 1e-3,
    seed: int = 0,
) -> tuple[DelayIndexedChunkAdapter, dict[str, Any]]:
    if not examples:
        raise ValueError("cannot train DICD adapter without examples")
    torch.manual_seed(int(seed))
    model = DelayIndexedChunkAdapter(config)
    features = torch.tensor([row.features for row in examples], dtype=torch.float32)
    targets = torch.tensor([row.target for row in examples], dtype=torch.float32)
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
    }
    return model, stats


def predict_dicd_action(model: DelayIndexedChunkAdapter, features: Sequence[float] | np.ndarray) -> np.ndarray:
    model.eval()
    with torch.inference_mode():
        tensor = torch.tensor(np.asarray(features, dtype=np.float32).reshape(1, -1), dtype=torch.float32)
        action = model(tensor).detach().cpu().numpy()
    return np.asarray(action, dtype=np.float32).reshape(1, -1)


def save_dicd_checkpoint(path: Path | str, model: DelayIndexedChunkAdapter, stats: Mapping[str, Any] | None = None) -> None:
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


def load_dicd_checkpoint(path: Path | str, *, map_location: str = "cpu") -> tuple[DelayIndexedChunkAdapter, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    config = DICDConfig(**payload["config"])
    model = DelayIndexedChunkAdapter(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload.get("stats") or {})


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_no_privileged_inference_fields(fields: Iterable[str]) -> None:
    present = {str(field) for field in fields}
    forbidden = sorted(present & PRIVILEGED_INFERENCE_FIELDS)
    if forbidden:
        raise ValueError(f"privileged DICD inference fields: {forbidden}")
