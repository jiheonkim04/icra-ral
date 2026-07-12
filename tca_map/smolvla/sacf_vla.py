"""Same-scene action counterfactual factorization for local VLA prototypes."""

from __future__ import annotations

import hashlib
import re
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
    "states",
}

FAMILIES = ("libero_spatial", "libero_object")

STOPWORDS = {
    "a",
    "an",
    "and",
    "in",
    "it",
    "of",
    "on",
    "place",
    "pick",
    "put",
    "the",
    "to",
    "up",
}


@dataclass(frozen=True)
class SACFConfig:
    state_dim: int = 8
    action_dim: int = 7
    semantic_width: int = 16
    phase_bins: int = 8
    hidden_dim: int = 64
    factor_loss_weight: float = 0.35
    shared_invariance_weight: float = 0.05
    prefix_fraction: float = 0.35

    @property
    def shared_input_dim(self) -> int:
        return self.state_dim + self.phase_bins + len(FAMILIES)

    @property
    def semantic_input_dim(self) -> int:
        return self.shared_input_dim + self.semantic_width

    def to_json(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SACFExample:
    state: list[float]
    action: list[float]
    instruction: str
    family: str
    task_key: str
    step_fraction: float
    phase_index: int


class SACFPolicyHead(nn.Module):
    def __init__(self, config: SACFConfig):
        super().__init__()
        self.config = config
        self.shared_net = nn.Sequential(
            nn.Linear(config.shared_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )
        self.semantic_net = nn.Sequential(
            nn.Linear(config.semantic_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(self, shared_features: torch.Tensor, semantic_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        shared = self.shared_net(shared_features)
        semantic = self.semantic_net(semantic_features)
        return torch.clamp(shared + semantic, -1.0, 1.0), shared, semantic


class PlainBCPrefixHead(nn.Module):
    def __init__(self, config: SACFConfig):
        super().__init__()
        self.config = config
        self.net = nn.Sequential(
            nn.Linear(config.semantic_input_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(),
            nn.Linear(config.hidden_dim, config.action_dim),
        )

    def forward(self, semantic_features: torch.Tensor) -> torch.Tensor:
        return torch.clamp(self.net(semantic_features), -1.0, 1.0)


def assert_no_privileged_inference_fields(fields: Iterable[str]) -> None:
    present = {str(field) for field in fields}
    forbidden = sorted(present & PRIVILEGED_INFERENCE_FIELDS)
    if forbidden:
        raise ValueError(f"privileged SACF inference fields: {forbidden}")


def normalize_instruction(instruction: str) -> str:
    text = str(instruction).lower().replace("_", " ")
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]+", " ", text)).strip()


def instruction_to_demo_filename(instruction: str) -> str:
    return normalize_instruction(instruction).replace(" ", "_") + "_demo.hdf5"


def phase_index_from_fraction(step_fraction: float, config: SACFConfig) -> int:
    frac = float(np.clip(step_fraction, 0.0, 0.999999))
    return int(frac * int(config.phase_bins))


def semantic_hash_features(instruction: str, width: int = 16) -> np.ndarray:
    tokens = [token for token in normalize_instruction(instruction).split() if token and token not in STOPWORDS]
    out = np.zeros(int(width), dtype=np.float32)
    if not tokens:
        return out
    for token in tokens:
        digest = hashlib.sha256(token.encode("utf-8")).digest()
        slot = int.from_bytes(digest[:4], "little") % int(width)
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        out[slot] += sign
    norm = float(np.linalg.norm(out))
    if norm > 1e-8:
        out /= norm
    return out.astype(np.float32)


def _as_vector(values: Sequence[float] | np.ndarray | None, dim: int) -> np.ndarray:
    if values is None:
        return np.zeros(int(dim), dtype=np.float32)
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size < int(dim):
        arr = np.pad(arr, (0, int(dim) - arr.size))
    if arr.size > int(dim):
        arr = arr[: int(dim)]
    return arr.astype(np.float32)


def _family_one_hot(family: str) -> np.ndarray:
    return np.asarray([1.0 if str(family) == item else 0.0 for item in FAMILIES], dtype=np.float32)


def _phase_one_hot(phase_index: int, config: SACFConfig) -> np.ndarray:
    out = np.zeros(int(config.phase_bins), dtype=np.float32)
    out[int(np.clip(phase_index, 0, int(config.phase_bins) - 1))] = 1.0
    return out


def make_sacf_features(
    state: Sequence[float] | np.ndarray,
    *,
    instruction: str,
    family: str,
    step_fraction: float,
    config: SACFConfig,
) -> tuple[list[float], list[float]]:
    phase_index = phase_index_from_fraction(step_fraction, config)
    shared = np.concatenate(
        [
            _as_vector(state, config.state_dim),
            _phase_one_hot(phase_index, config),
            _family_one_hot(family),
        ],
        axis=0,
    ).astype(np.float32)
    semantic = np.concatenate([shared, semantic_hash_features(instruction, config.semantic_width)], axis=0).astype(np.float32)
    if shared.size != int(config.shared_input_dim):
        raise RuntimeError(f"SACF shared feature width mismatch: {shared.size} != {config.shared_input_dim}")
    if semantic.size != int(config.semantic_input_dim):
        raise RuntimeError(f"SACF semantic feature width mismatch: {semantic.size} != {config.semantic_input_dim}")
    return [float(value) for value in shared], [float(value) for value in semantic]


def _stats_from_examples(examples: Sequence[SACFExample], config: SACFConfig) -> dict[str, Any]:
    if not examples:
        raise ValueError("SACF examples are empty")
    actions = np.asarray([_as_vector(row.action, config.action_dim) for row in examples], dtype=np.float32)
    phase_task_mean_actions: dict[str, list[float]] = {}
    for task_key in sorted({row.task_key for row in examples}):
        for phase_index in range(int(config.phase_bins)):
            rows = [i for i, row in enumerate(examples) if row.task_key == task_key and int(row.phase_index) == phase_index]
            if rows:
                phase_task_mean_actions[f"{task_key}|phase_{phase_index}"] = [float(value) for value in np.mean(actions[rows], axis=0)]
    return {
        "global_mean_action": [float(value) for value in np.mean(actions, axis=0)],
        "phase_task_mean_actions": phase_task_mean_actions,
    }


def task_phase_mean_action(
    stats: Mapping[str, Any],
    *,
    task_key: str,
    step_fraction: float,
    config: SACFConfig,
) -> np.ndarray:
    phase_index = phase_index_from_fraction(step_fraction, config)
    key = f"{task_key}|phase_{phase_index}"
    phase_actions = stats.get("phase_task_mean_actions") or {}
    if key in phase_actions:
        return _as_vector(phase_actions[key], config.action_dim)
    return _as_vector(stats.get("global_mean_action"), config.action_dim)


def _tensor_features(examples: Sequence[SACFExample], config: SACFConfig) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    shared_rows: list[list[float]] = []
    semantic_rows: list[list[float]] = []
    target_rows: list[np.ndarray] = []
    for row in examples:
        shared, semantic = make_sacf_features(
            row.state,
            instruction=row.instruction,
            family=row.family,
            step_fraction=row.step_fraction,
            config=config,
        )
        shared_rows.append(shared)
        semantic_rows.append(semantic)
        target_rows.append(_as_vector(row.action, config.action_dim))
    return (
        torch.tensor(np.asarray(shared_rows, dtype=np.float32), dtype=torch.float32),
        torch.tensor(np.asarray(semantic_rows, dtype=np.float32), dtype=torch.float32),
        torch.tensor(np.asarray(target_rows, dtype=np.float32), dtype=torch.float32),
    )


def _counterfactual_pairs(examples: Sequence[SACFExample], *, max_pairs: int = 2048, seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    groups: dict[tuple[str, int], list[int]] = {}
    for index, row in enumerate(examples):
        groups.setdefault((row.family, int(row.phase_index)), []).append(index)
    pairs: list[tuple[int, int]] = []
    for indices in groups.values():
        by_instruction: dict[str, list[int]] = {}
        for index in indices:
            by_instruction.setdefault(normalize_instruction(examples[index].instruction), []).append(index)
        keys = sorted(key for key, value in by_instruction.items() if value)
        for left, right in zip(keys, keys[1:]):
            pairs.append((by_instruction[left][0], by_instruction[right][0]))
        if len(keys) > 2:
            pairs.append((by_instruction[keys[0]][0], by_instruction[keys[-1]][0]))
    if not pairs:
        return torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long)
    rng = np.random.default_rng(int(seed))
    if len(pairs) > int(max_pairs):
        selected = rng.choice(len(pairs), size=int(max_pairs), replace=False)
        pairs = [pairs[int(i)] for i in selected]
    left = torch.tensor([pair[0] for pair in pairs], dtype=torch.long)
    right = torch.tensor([pair[1] for pair in pairs], dtype=torch.long)
    return left, right


def train_sacf_policy(
    examples: Sequence[SACFExample],
    *,
    config: SACFConfig,
    epochs: int = 160,
    lr: float = 3e-3,
    seed: int = 0,
) -> tuple[SACFPolicyHead, dict[str, Any]]:
    if not examples:
        raise ValueError("cannot train SACF without examples")
    torch.manual_seed(int(seed))
    model = SACFPolicyHead(config)
    shared_features, semantic_features, targets = _tensor_features(examples, config)
    left_idx, right_idx = _counterfactual_pairs(examples, seed=seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: list[float] = []
    bc_losses: list[float] = []
    factor_losses: list[float] = []
    inv_losses: list[float] = []
    grad_norms: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        pred, shared, semantic = model(shared_features, semantic_features)
        bc_loss = torch.nn.functional.mse_loss(pred, targets)
        if left_idx.numel() > 0:
            factor_loss = torch.nn.functional.mse_loss(semantic[left_idx] - semantic[right_idx], targets[left_idx] - targets[right_idx])
            inv_loss = torch.nn.functional.mse_loss(shared[left_idx], shared[right_idx])
        else:
            factor_loss = pred.sum() * 0.0
            inv_loss = pred.sum() * 0.0
        loss = bc_loss + float(config.factor_loss_weight) * factor_loss + float(config.shared_invariance_weight) * inv_loss
        loss.backward()
        grad_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_sq += float(torch.sum(param.grad.detach() ** 2).item())
        grad_norms.append(float(grad_sq**0.5))
        optimizer.step()
        losses.append(float(loss.detach().item()))
        bc_losses.append(float(bc_loss.detach().item()))
        factor_losses.append(float(factor_loss.detach().item()))
        inv_losses.append(float(inv_loss.detach().item()))
    with torch.inference_mode():
        pred, shared, semantic = model(shared_features, semantic_features)
    stats = _stats_from_examples(examples, config)
    stats.update(
        {
            "example_count": int(len(examples)),
            "counterfactual_pair_count": int(left_idx.numel()),
            "epochs": int(epochs),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "initial_bc_loss": float(bc_losses[0]),
            "final_bc_loss": float(bc_losses[-1]),
            "initial_factor_loss": float(factor_losses[0]),
            "final_factor_loss": float(factor_losses[-1]),
            "initial_shared_invariance_loss": float(inv_losses[0]),
            "final_shared_invariance_loss": float(inv_losses[-1]),
            "loss_decreased": bool(losses[-1] < losses[0]),
            "bc_loss_decreased": bool(bc_losses[-1] < bc_losses[0]),
            "factor_loss_decreased": bool(factor_losses[-1] <= factor_losses[0]),
            "max_grad_norm": float(max(grad_norms)),
            "finite_gradients": bool(np.isfinite(grad_norms).all()),
            "mean_semantic_component_norm": float(torch.mean(torch.linalg.norm(semantic, dim=-1)).item()),
            "mean_shared_component_norm": float(torch.mean(torch.linalg.norm(shared, dim=-1)).item()),
            "mean_prediction_norm": float(torch.mean(torch.linalg.norm(pred, dim=-1)).item()),
        }
    )
    return model, stats


def train_plain_bc_prefix(
    examples: Sequence[SACFExample],
    *,
    config: SACFConfig,
    epochs: int = 160,
    lr: float = 3e-3,
    seed: int = 0,
) -> tuple[PlainBCPrefixHead, dict[str, Any]]:
    if not examples:
        raise ValueError("cannot train plain BC prefix without examples")
    torch.manual_seed(int(seed))
    model = PlainBCPrefixHead(config)
    _shared_features, semantic_features, targets = _tensor_features(examples, config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=1e-4)
    losses: list[float] = []
    grad_norms: list[float] = []
    for _ in range(int(epochs)):
        optimizer.zero_grad(set_to_none=True)
        pred = model(semantic_features)
        loss = torch.nn.functional.mse_loss(pred, targets)
        loss.backward()
        grad_sq = 0.0
        for param in model.parameters():
            if param.grad is not None:
                grad_sq += float(torch.sum(param.grad.detach() ** 2).item())
        grad_norms.append(float(grad_sq**0.5))
        optimizer.step()
        losses.append(float(loss.detach().item()))
    stats = _stats_from_examples(examples, config)
    stats.update(
        {
            "example_count": int(len(examples)),
            "epochs": int(epochs),
            "initial_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "loss_decreased": bool(losses[-1] < losses[0]),
            "max_grad_norm": float(max(grad_norms)),
            "finite_gradients": bool(np.isfinite(grad_norms).all()),
        }
    )
    return model, stats


def predict_sacf_action(
    model: SACFPolicyHead,
    *,
    state: Sequence[float] | np.ndarray,
    instruction: str,
    family: str,
    step_fraction: float,
) -> tuple[np.ndarray, dict[str, float]]:
    model.eval()
    shared_features, semantic_features = make_sacf_features(
        state,
        instruction=instruction,
        family=family,
        step_fraction=step_fraction,
        config=model.config,
    )
    with torch.inference_mode():
        shared_tensor = torch.tensor(np.asarray(shared_features, dtype=np.float32).reshape(1, -1), dtype=torch.float32)
        semantic_tensor = torch.tensor(np.asarray(semantic_features, dtype=np.float32).reshape(1, -1), dtype=torch.float32)
        action, shared, semantic = model(shared_tensor, semantic_tensor)
    action_np = action.detach().cpu().numpy().reshape(-1).astype(np.float32)
    return action_np, {
        "semantic_component_norm": float(torch.linalg.norm(semantic).item()),
        "shared_component_norm": float(torch.linalg.norm(shared).item()),
    }


def predict_plain_action(
    model: PlainBCPrefixHead,
    *,
    state: Sequence[float] | np.ndarray,
    instruction: str,
    family: str,
    step_fraction: float,
) -> np.ndarray:
    model.eval()
    _shared_features, semantic_features = make_sacf_features(
        state,
        instruction=instruction,
        family=family,
        step_fraction=step_fraction,
        config=model.config,
    )
    with torch.inference_mode():
        semantic_tensor = torch.tensor(np.asarray(semantic_features, dtype=np.float32).reshape(1, -1), dtype=torch.float32)
        action = model(semantic_tensor)
    return action.detach().cpu().numpy().reshape(-1).astype(np.float32)


def save_sacf_checkpoint(path: Path | str, model: SACFPolicyHead, stats: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "sacf_full", "config": model.config.to_json(), "state_dict": model.state_dict(), "stats": dict(stats)}, target)


def load_sacf_checkpoint(path: Path | str, *, map_location: str = "cpu") -> tuple[SACFPolicyHead, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    config = SACFConfig(**payload["config"])
    model = SACFPolicyHead(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload.get("stats") or {})


def save_plain_checkpoint(path: Path | str, model: PlainBCPrefixHead, stats: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"kind": "plain_bc_prefix", "config": model.config.to_json(), "state_dict": model.state_dict(), "stats": dict(stats)}, target)


def load_plain_checkpoint(path: Path | str, *, map_location: str = "cpu") -> tuple[PlainBCPrefixHead, dict[str, Any]]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=False)
    config = SACFConfig(**payload["config"])
    model = PlainBCPrefixHead(config)
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return model, dict(payload.get("stats") or {})


def file_sha256(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
