"""Pure SPARC-VLA conceptor, manifest, action, and hook helpers."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "CC2F9ACCE2A26EC438C58F2854ADC95134354C245CAD8ED961D29A895DBC697D"
HIDDEN_WIDTH = 720
ACTION_TOKENS = 50
DENOISING_STEPS = 10
CANDIDATE_RESIDUAL_SITES = (0, 5, 11, 14)
APERTURES = (0.1, 0.5, 1.0, 2.0, 10.0)
BETAS = (0.1, 0.3, 0.5)


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest().upper()


def tensor_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    return hashlib.sha256(array.tobytes(order="C")).hexdigest().upper()


def episode_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(row["partition"]),
            str(row["policy"]),
            str(row["suite"]),
            f"task={int(row['task_id'])}",
            f"seed={int(row['reset_seed'])}",
        )
    )


def activation_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        (
            str(row["episode_key"]),
            f"replan={int(row['replan_index'])}",
            f"site={int(row['residual_site'])}",
            f"denoise={int(row['denoising_step'])}",
        )
    )


def manifest_audit(expected_keys: Sequence[str], observed_keys: Sequence[str]) -> dict[str, Any]:
    expected = list(expected_keys)
    observed = list(observed_keys)
    duplicate_expected = len(expected) - len(set(expected))
    duplicate_observed = len(observed) - len(set(observed))
    missing = sorted(set(expected) - set(observed))
    extra = sorted(set(observed) - set(expected))
    return {
        "expected_count": len(expected),
        "observed_count": len(observed),
        "duplicate_expected_count": duplicate_expected,
        "duplicate_observed_count": duplicate_observed,
        "missing_keys": missing,
        "extra_keys": extra,
        "passed": not duplicate_expected and not duplicate_observed and not missing and not extra,
    }


def _activation_matrix(value: Any) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != HIDDEN_WIDTH or array.shape[0] < 1:
        raise ValueError(f"expected [N,{HIDDEN_WIDTH}] activation matrix, received {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("activation matrix contains nonfinite values")
    return array


def equal_episode_covariance(episodes: Sequence[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return the equal-episode mean and covariance from [N_e, 720] arrays."""

    arrays = [_activation_matrix(value) for value in episodes]
    if not arrays:
        raise ValueError("at least one episode is required")
    episode_means = [array.mean(axis=0) for array in arrays]
    mean = np.mean(episode_means, axis=0)
    covariance = np.zeros((HIDDEN_WIDTH, HIDDEN_WIDTH), dtype=np.float64)
    for array in arrays:
        centered = array - mean
        covariance += (centered.T @ centered) / array.shape[0]
    covariance /= len(arrays)
    covariance = (covariance + covariance.T) / 2.0
    return mean, covariance


def aggregate_covariances(covariances: Sequence[Any]) -> np.ndarray:
    arrays = [np.asarray(value, dtype=np.float64) for value in covariances]
    if not arrays or any(array.shape != (HIDDEN_WIDTH, HIDDEN_WIDTH) for array in arrays):
        raise ValueError(f"covariances must be nonempty [{HIDDEN_WIDTH},{HIDDEN_WIDTH}] matrices")
    if any(not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("covariance contains nonfinite values")
    result = np.mean(arrays, axis=0)
    return (result + result.T) / 2.0


def compute_conceptor(covariance: Any, alpha: float) -> np.ndarray:
    covariance_array = np.asarray(covariance, dtype=np.float64)
    if covariance_array.shape != (HIDDEN_WIDTH, HIDDEN_WIDTH):
        raise ValueError(f"expected [{HIDDEN_WIDTH},{HIDDEN_WIDTH}] covariance")
    if not np.all(np.isfinite(covariance_array)) or float(alpha) <= 0.0:
        raise ValueError("conceptor requires finite covariance and positive aperture")
    covariance_array = (covariance_array + covariance_array.T) / 2.0
    regularized = covariance_array + float(alpha) ** -2 * np.eye(HIDDEN_WIDTH, dtype=np.float64)
    condition_number = float(np.linalg.cond(regularized))
    if not np.isfinite(condition_number) or condition_number > 1e12:
        raise ValueError(f"regularized conceptor solve is ill-conditioned: {condition_number}")
    conceptor = np.linalg.solve(regularized.T, covariance_array.T).T
    conceptor = (conceptor + conceptor.T) / 2.0
    validate_conceptor(conceptor)
    return conceptor


def validate_conceptor(conceptor: Any, *, tolerance: float = 1e-8) -> dict[str, Any]:
    array = np.asarray(conceptor, dtype=np.float64)
    if array.shape != (HIDDEN_WIDTH, HIDDEN_WIDTH) or not np.all(np.isfinite(array)):
        raise ValueError("conceptor must be a finite [720,720] matrix")
    symmetry_error = float(np.max(np.abs(array - array.T)))
    eigenvalues = np.linalg.eigvalsh((array + array.T) / 2.0)
    minimum = float(eigenvalues[0])
    maximum = float(eigenvalues[-1])
    if symmetry_error > tolerance or minimum < -tolerance or maximum > 1.0 + tolerance:
        raise ValueError(
            f"invalid conceptor: symmetry={symmetry_error}, eigenvalue_range=[{minimum},{maximum}]"
        )
    return {
        "symmetry_max_abs_error": symmetry_error,
        "eigenvalue_min": minimum,
        "eigenvalue_max": maximum,
        "quota": quota(array),
        "effective_rank_ge_0_1": int(np.sum(eigenvalues >= 0.1)),
    }


def conceptor_and_not(success: Any, failure: Any, *, rcond: float = 1e-12) -> np.ndarray:
    success_array = np.asarray(success, dtype=np.float64)
    failure_array = np.asarray(failure, dtype=np.float64)
    validate_conceptor(success_array)
    validate_conceptor(failure_array)
    identity = np.eye(HIDDEN_WIDTH, dtype=np.float64)
    complement = identity - failure_array
    result = np.linalg.pinv(
        np.linalg.pinv(success_array, rcond=rcond) + np.linalg.pinv(complement, rcond=rcond) - identity,
        rcond=rcond,
    )
    result = (result + result.T) / 2.0
    validate_conceptor(result)
    return result


def aggregate_mean_conceptors(covariances: Sequence[Any], alpha: float) -> np.ndarray:
    result = np.mean([compute_conceptor(value, alpha) for value in covariances], axis=0)
    result = (result + result.T) / 2.0
    validate_conceptor(result)
    return result


def quota(conceptor: Any) -> float:
    array = np.asarray(conceptor, dtype=np.float64)
    if array.shape != (HIDDEN_WIDTH, HIDDEN_WIDTH):
        raise ValueError("quota requires a [720,720] matrix")
    return float(np.trace(array) / HIDDEN_WIDTH)


def normalized_frobenius_similarity(left: Any, right: Any) -> float:
    left_array = np.asarray(left, dtype=np.float64)
    right_array = np.asarray(right, dtype=np.float64)
    if left_array.shape != right_array.shape:
        raise ValueError("similarity matrices must have equal shapes")
    denominator = float(np.linalg.norm(left_array, ord="fro") * np.linalg.norm(right_array, ord="fro"))
    if denominator <= 0.0:
        return 0.0
    return float(np.sum(left_array * right_array) / denominator)


def conceptor_overlap(success: Any, failure: Any) -> float:
    return normalized_frobenius_similarity(success, failure)


def retained_energy(operator: Any, covariance: Any) -> float:
    operator_array = np.asarray(operator, dtype=np.float64)
    covariance_array = np.asarray(covariance, dtype=np.float64)
    if operator_array.shape != (HIDDEN_WIDTH, HIDDEN_WIDTH) or covariance_array.shape != operator_array.shape:
        raise ValueError("retained energy requires [720,720] matrices")
    numerator = float(np.trace(operator_array @ covariance_array @ operator_array.T))
    denominator = max(float(np.trace(covariance_array)), 1e-12)
    return numerator / denominator


def select_residual_site(operators_by_site: Mapping[int, Sequence[Any]]) -> dict[str, Any]:
    if set(operators_by_site) != set(CANDIDATE_RESIDUAL_SITES):
        raise ValueError("residual-site candidates do not match the preregistration")
    means = {
        int(site): float(np.mean([quota(operator) for operator in operators]))
        for site, operators in operators_by_site.items()
    }
    selected = min(means, key=lambda site: (-round(means[site], 12), site))
    return {"selected_site": selected, "mean_quotas": means}


def select_aperture(overlaps_by_alpha: Mapping[float, float]) -> dict[str, Any]:
    if {float(value) for value in overlaps_by_alpha} != set(APERTURES):
        raise ValueError("aperture candidates do not match the preregistration")
    inside = [alpha for alpha, value in overlaps_by_alpha.items() if 0.80 <= float(value) <= 0.90]
    candidates = inside or list(overlaps_by_alpha)
    selected = min(candidates, key=lambda alpha: (round(abs(float(overlaps_by_alpha[alpha]) - 0.85), 12), float(alpha)))
    return {
        "selected_aperture": float(selected),
        "overlaps": {str(float(key)): float(value) for key, value in sorted(overlaps_by_alpha.items())},
        "inside_band": bool(inside),
    }


def interpolation_gate(conceptor: Any, beta: float) -> np.ndarray:
    array = np.asarray(conceptor, dtype=np.float64)
    validate_conceptor(array)
    if not 0.0 <= float(beta) <= 1.0:
        raise ValueError("beta must lie in [0,1]")
    return (1.0 - float(beta)) * np.eye(HIDDEN_WIDTH, dtype=np.float64) + float(beta) * array


def apply_token_operator(hidden: Any, conceptor: Any, beta: float) -> np.ndarray:
    hidden_array = np.asarray(hidden)
    if hidden_array.shape[-2:] != (ACTION_TOKENS, HIDDEN_WIDTH):
        raise ValueError(f"hidden tensor must end in [{ACTION_TOKENS},{HIDDEN_WIDTH}]")
    gate = interpolation_gate(conceptor, beta).astype(hidden_array.dtype, copy=False)
    return hidden_array @ gate.T


def action_safety(base: Any, ours: Any) -> dict[str, Any]:
    base_array = np.asarray(base, dtype=np.float64)
    ours_array = np.asarray(ours, dtype=np.float64)
    if base_array.shape != (50, 7) or ours_array.shape != (50, 7):
        raise ValueError("action safety requires Base and Ours [50,7] chunks")
    delta = ours_array - base_array
    absolute = np.abs(ours_array)
    base_absolute = np.abs(base_array)
    exceedance = np.maximum(absolute - 1.0, 0.0)
    base_exceedance = np.maximum(base_absolute - 1.0, 0.0)
    first = delta[:10]
    metrics = {
        "finite_fraction": float(np.mean(np.isfinite(ours_array))),
        "absolute_max": float(np.nanmax(absolute)),
        "outside_fraction": float(np.mean(absolute > 1.0)),
        "base_outside_fraction": float(np.mean(base_absolute > 1.0)),
        "p99_exceedance": float(np.nanpercentile(exceedance, 99)),
        "base_p99_exceedance": float(np.nanpercentile(base_exceedance, 99)),
        "translation_delta_l2_p95": float(np.percentile(np.linalg.norm(first[:, :3], axis=1), 95)),
        "rotation_delta_l2_p95": float(np.percentile(np.linalg.norm(first[:, 3:6], axis=1), 95)),
        "gripper_abs_delta_p95": float(np.percentile(np.abs(first[:, 6]), 95)),
        "full_chunk_7d_delta_l2_p95": float(np.percentile(np.linalg.norm(delta, axis=1), 95)),
        "mean_full_chunk_7d_delta_l2": float(np.mean(np.linalg.norm(delta, axis=1))),
        "changed_dimension_count": int(np.sum(np.max(np.abs(delta), axis=0) > 0.0)),
    }
    metrics["passed"] = bool(
        metrics["finite_fraction"] == 1.0
        and metrics["absolute_max"] <= 1.25
        and metrics["outside_fraction"] <= metrics["base_outside_fraction"] + 0.01
        and metrics["p99_exceedance"] <= metrics["base_p99_exceedance"] + 0.02
        and metrics["translation_delta_l2_p95"] <= 0.20
        and metrics["rotation_delta_l2_p95"] <= 0.20
        and metrics["gripper_abs_delta_p95"] <= 0.20
        and metrics["full_chunk_7d_delta_l2_p95"] <= 0.30
    )
    return metrics


def _expert_layers(policy: Any) -> Any:
    roots = [policy]
    seen: set[int] = set()
    while roots:
        root = roots.pop(0)
        if id(root) in seen:
            continue
        seen.add(id(root))
        try:
            return root.model.vlm_with_expert.lm_expert.layers
        except AttributeError:
            pass
        for name in ("base_model", "model", "module"):
            child = getattr(root, name, None)
            if child is not None and child is not root:
                roots.append(child)
    raise AttributeError("could not locate SmolVLA action-expert layers")


@dataclass(frozen=True)
class HookCapture:
    denoising_step: int
    shape: tuple[int, ...]
    tensor_sha256: str
    input_norm: float
    output_norm: float
    delta_norm: float
    max_token_delta_norm: float


class SparcPostResidualAdapter:
    """Inference-only pre-hook that gates the full residual between expert layers."""

    def __init__(self, policy: Any, residual_site: int, *, expected_steps: int = DENOISING_STEPS):
        if residual_site not in CANDIDATE_RESIDUAL_SITES:
            raise ValueError("residual site is not preregistered")
        layers = _expert_layers(policy)
        if residual_site + 1 >= len(layers):
            raise ValueError("residual site lacks a following layer")
        self.policy = policy
        self.residual_site = int(residual_site)
        self.expected_steps = int(expected_steps)
        self._layernorm = layers[residual_site + 1].input_layernorm
        self._handle: Any | None = None
        self._configured = False
        self._strategy = "global"
        self._beta = 0.0
        self._operators: list[np.ndarray] = []
        self.captures: list[HookCapture] = []
        self.full_tensors: list[np.ndarray] = []

    @property
    def configured(self) -> bool:
        return self._configured

    def register(self) -> None:
        if self._handle is not None:
            raise RuntimeError("SPARC hook is already registered")
        self._handle = self._layernorm.register_forward_pre_hook(self._hook)

    def remove(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def reset_capture(self) -> None:
        self.captures.clear()
        self.full_tensors.clear()

    def configure(self, operators: Any, *, beta: float, strategy: str = "global") -> None:
        if strategy not in {"global", "per_step"}:
            raise ValueError("strategy must be global or per_step")
        values = [operators] if np.asarray(operators).ndim == 2 else list(operators)
        required = 1 if strategy == "global" else self.expected_steps
        if len(values) != required:
            raise ValueError(f"{strategy} requires {required} operators")
        checked = []
        for value in values:
            array = np.asarray(value, dtype=np.float64)
            validate_conceptor(array)
            checked.append(array.copy())
        if not 0.0 <= float(beta) <= 1.0:
            raise ValueError("beta must lie in [0,1]")
        self._operators = checked
        self._beta = float(beta)
        self._strategy = strategy
        self._configured = True

    def clear_configuration(self) -> None:
        self._configured = False
        self._strategy = "global"
        self._beta = 0.0
        self._operators = []

    def _hook(self, _module: Any, args: tuple[Any, ...]) -> None:
        import torch

        if torch.is_grad_enabled():
            raise RuntimeError("SPARC post-residual hook is inference-only")
        if len(args) != 1:
            raise RuntimeError("unexpected action-expert layernorm inputs")
        hidden = args[0]
        if tuple(hidden.shape) != (1, ACTION_TOKENS, HIDDEN_WIDTH):
            raise RuntimeError(f"unexpected SPARC tensor shape: {tuple(hidden.shape)}")
        step = len(self.captures)
        if step >= self.expected_steps:
            raise RuntimeError("SPARC hook fired more than the expected denoising steps")
        before = hidden.detach().float().cpu().numpy().copy()
        after = hidden
        if self._configured and self._beta > 0.0:
            operator = self._operators[0] if self._strategy == "global" else self._operators[step]
            if not np.array_equal(operator, np.eye(HIDDEN_WIDTH, dtype=np.float64)):
                gate = interpolation_gate(operator, self._beta)
                gate_tensor = torch.as_tensor(gate, device=hidden.device, dtype=hidden.dtype)
                steered = hidden @ gate_tensor.T
                if not bool(torch.isfinite(steered).all()):
                    raise RuntimeError("SPARC hook produced nonfinite activations")
                hidden.copy_(steered)
                after = hidden
        after_cpu = after.detach().float().cpu().numpy().copy()
        delta = after_cpu - before
        token_deltas = np.linalg.norm(delta[0], axis=1)
        self.full_tensors.append(before)
        self.captures.append(
            HookCapture(
                denoising_step=step,
                shape=tuple(before.shape),
                tensor_sha256=tensor_sha256(before),
                input_norm=float(np.linalg.norm(before)),
                output_norm=float(np.linalg.norm(after_cpu)),
                delta_norm=float(np.linalg.norm(delta)),
                max_token_delta_norm=float(np.max(token_deltas)),
            )
        )

    def assert_complete(self) -> None:
        if len(self.captures) != self.expected_steps:
            raise RuntimeError(f"expected {self.expected_steps} captures, observed {len(self.captures)}")
        if [capture.denoising_step for capture in self.captures] != list(range(self.expected_steps)):
            raise RuntimeError("denoising capture order is invalid")

    def save(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        metadata = {
            "proposal_hash": PROPOSAL_HASH,
            "residual_site": self.residual_site,
            "expected_steps": self.expected_steps,
            "configured": self._configured,
            "strategy": self._strategy,
            "beta": self._beta,
            "operator_hashes": [tensor_sha256(value) for value in self._operators],
        }
        temporary = destination.with_suffix(destination.suffix + ".tmp.npz")
        np.savez_compressed(
            temporary,
            metadata=np.asarray(json.dumps(metadata, sort_keys=True)),
            operators=np.asarray(self._operators, dtype=np.float32),
        )
        temporary.replace(destination)

    @classmethod
    def load(cls, policy: Any, path: str | Path) -> "SparcPostResidualAdapter":
        with np.load(Path(path), allow_pickle=False) as payload:
            metadata = json.loads(str(payload["metadata"].item()))
            operators = np.asarray(payload["operators"], dtype=np.float64)
        if metadata.get("proposal_hash") != PROPOSAL_HASH:
            raise ValueError("SPARC adapter proposal hash mismatch")
        adapter = cls(
            policy,
            int(metadata["residual_site"]),
            expected_steps=int(metadata["expected_steps"]),
        )
        if bool(metadata["configured"]):
            values: Any = operators[0] if str(metadata["strategy"]) == "global" else list(operators)
            adapter.configure(values, beta=float(metadata["beta"]), strategy=str(metadata["strategy"]))
            observed = [tensor_sha256(value) for value in adapter._operators]
            if observed != list(metadata["operator_hashes"]):
                raise ValueError("SPARC adapter operator hash mismatch")
        return adapter

    def __enter__(self) -> "SparcPostResidualAdapter":
        self.register()
        return self

    def __exit__(self, _exc_type: Any, _exc: Any, _traceback: Any) -> None:
        self.remove()
