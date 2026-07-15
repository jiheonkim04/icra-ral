"""Pure IARC-VLA Stage 0A contracts and mechanism helpers."""

from __future__ import annotations

from collections import defaultdict
import copy
from dataclasses import dataclass
import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np


PROPOSAL_HASH = "A1B0CF8BCBCF6A88F27B31EF5E38BAF408A3E62BB34206A1AC9F051EA6B57408"
ROBUST_NORM_SQUARED_FLOOR = 1e-12
CONFLICT_COSINE_THRESHOLD = -0.01
PROJECTION_TOLERANCE = 1e-6
CONTEXT_PREFIX = "Context note: the workspace contains several objects. Task:"
IMAGE_KEYS = ("observation.images.image", "observation.images.image2")
PERTURBATION_FAMILIES = (
    "gaussian_sensor_noise",
    "image_translation",
    "instruction_repetition",
    "context_wrapper",
)
SEVERITIES: dict[str, tuple[float | int, ...]] = {
    "gaussian_sensor_noise": (0.02, 0.05, 0.10),
    "image_translation": (4, 8, 16),
    "instruction_repetition": (1, 2, 3),
    "context_wrapper": (1, 2, 3),
}
TRANSLATION_DIRECTIONS = ("up", "down", "left", "right")


@dataclass(frozen=True)
class PerturbationSpec:
    family: str
    severity_index: int
    severity: float | int
    seed: int
    direction: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "severity_index": self.severity_index,
            "severity": self.severity,
            "seed": self.seed,
            "direction": self.direction,
        }


def stable_seed(*parts: Any) -> int:
    payload = "|".join([PROPOSAL_HASH, *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def sample_id(row: Mapping[str, Any]) -> str:
    return str(
        row.get("sample_id")
        or f"{row.get('split')}|{row.get('task_index')}|{row.get('episode_index')}|{row.get('frame_index')}"
    )


def frame_key(row: Mapping[str, Any]) -> tuple[int, int, int]:
    return (int(row["task_index"]), int(row["episode_index"]), int(row["frame_index"]))


def _stable_row_key(row: Mapping[str, Any]) -> tuple[int, int, int, str]:
    return (
        int(row["task_index"]),
        int(row["episode_index"]),
        int(row["frame_index"]),
        sample_id(row),
    )


def _rank_rows_by_midphase(rows: Sequence[Mapping[str, Any]]) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in sorted((dict(item) for item in rows), key=_stable_row_key):
        grouped[int(row["task_index"])].append(row)
    for task_index in grouped:
        grouped[task_index].sort(
            key=lambda row: (
                abs(float(row.get("normalized_phase", 0.0)) - 0.5),
                int(row["episode_index"]),
                int(row["frame_index"]),
                sample_id(row),
            )
        )
    return grouped


def partition_stage0_manifest(manifest: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Select the frozen per-task midphase ranks without decoding test rows."""

    splits = manifest.get("splits") or {}
    train = _rank_rows_by_midphase(splits.get("train") or [])
    validation = _rank_rows_by_midphase(splits.get("val") or [])
    tasks = sorted(train)
    if tasks != sorted(validation):
        raise ValueError("train and validation task identities differ")
    too_short = [task for task in tasks if len(train[task]) < 3 or len(validation[task]) < 1]
    if too_short:
        raise ValueError(f"insufficient Stage 0 ranks for tasks: {too_short}")

    output = {
        "micro_fit": [train[task][0] for task in tasks],
        "conflict_audit": [train[task][1] for task in tasks],
        "one_check": [train[task][2] for task in tasks],
        "validation": [validation[task][0] for task in tasks],
        "confirmatory_reserved": [dict(row) for row in splits.get("test") or []],
    }
    for rows in output.values():
        rows.sort(key=_stable_row_key)
    assert_zero_partition_overlap(output)
    return output


def assert_zero_partition_overlap(partitions: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    identities: dict[str, set[str]] = {}
    frames: dict[str, set[tuple[int, int, int]]] = {}
    for name, rows in partitions.items():
        sample_ids = [sample_id(row) for row in rows]
        frame_keys = [frame_key(row) for row in rows]
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError(f"duplicate sample identity in {name}")
        if len(frame_keys) != len(set(frame_keys)):
            raise ValueError(f"duplicate frame identity in {name}")
        identities[name] = set(sample_ids)
        frames[name] = set(frame_keys)
    names = list(partitions)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if identities[left] & identities[right]:
                raise ValueError(f"sample overlap between {left} and {right}")
            if frames[left] & frames[right]:
                raise ValueError(f"frame overlap between {left} and {right}")


def partition_summary(partitions: Mapping[str, Sequence[Mapping[str, Any]]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, rows in partitions.items():
        ids = [sample_id(row) for row in rows]
        keys = [frame_key(row) for row in rows]
        result[name] = {
            "records": len(rows),
            "tasks": len({int(row["task_index"]) for row in rows}),
            "episodes": len({int(row["episode_index"]) for row in rows}),
            "duplicate_sample_ids": len(ids) - len(set(ids)),
            "duplicate_frame_keys": len(keys) - len(set(keys)),
        }
    names = list(partitions)
    result["pairwise_sample_overlap"] = {
        f"{left}__{right}": len(
            {sample_id(row) for row in partitions[left]} & {sample_id(row) for row in partitions[right]}
        )
        for index, left in enumerate(names)
        for right in names[index + 1 :]
    }
    return result


def perturbation_spec(
    row: Mapping[str, Any],
    *,
    partition: str,
    sorted_task_indices: Sequence[int] | None = None,
) -> PerturbationSpec:
    tasks = sorted(set(int(value) for value in (sorted_task_indices or range(40))))
    task_index = int(row["task_index"])
    if task_index not in tasks:
        raise ValueError(f"task {task_index} is not in the frozen task list")
    position = tasks.index(task_index)
    family = PERTURBATION_FAMILIES[position % len(PERTURBATION_FAMILIES)]
    within_family_index = position // len(PERTURBATION_FAMILIES)
    severity_index = within_family_index % 3
    seed = stable_seed(partition, sample_id(row), task_index, severity_index)
    direction = None
    if family == "image_translation":
        direction = TRANSLATION_DIRECTIONS[stable_seed(partition, sample_id(row), "direction") % 4]
    return PerturbationSpec(
        family=family,
        severity_index=severity_index,
        severity=SEVERITIES[family][severity_index],
        seed=seed,
        direction=direction,
    )


def perturb_instruction(instruction: str, spec: PerturbationSpec) -> str:
    if spec.family == "instruction_repetition":
        copies = int(spec.severity) + 1
        return " ; ".join([instruction] * copies)
    if spec.family == "context_wrapper":
        prefixes = " ".join([CONTEXT_PREFIX] * int(spec.severity))
        return f"{prefixes} {instruction}"
    return instruction


def _translate_numpy(image: np.ndarray, *, pixels: int, direction: str) -> np.ndarray:
    if image.ndim < 2:
        raise ValueError(f"image needs spatial dimensions, got {image.shape}")
    height, width = image.shape[-2:]
    if pixels <= 0 or pixels >= min(height, width):
        raise ValueError(f"invalid translation {pixels} for image {image.shape}")
    pad = [(0, 0)] * image.ndim
    pad[-2] = (pixels, pixels)
    pad[-1] = (pixels, pixels)
    padded = np.pad(image, pad, mode="edge")
    top = pixels
    left = pixels
    if direction == "up":
        top += pixels
    elif direction == "down":
        top -= pixels
    elif direction == "left":
        left += pixels
    elif direction == "right":
        left -= pixels
    else:
        raise ValueError(f"unknown translation direction: {direction}")
    return padded[..., top : top + height, left : left + width].copy()


def perturb_image(image: Any, spec: PerturbationSpec, *, camera_key: str) -> Any:
    is_torch = hasattr(image, "detach") and hasattr(image, "device")
    if is_torch:
        array = image.detach().cpu().numpy()
    else:
        array = np.asarray(image)
    if not np.all(np.isfinite(array)) or float(array.min()) < 0.0 or float(array.max()) > 1.0:
        raise ValueError("raw RGB image must be finite and in [0,1]")

    if spec.family == "gaussian_sensor_noise":
        rng = np.random.default_rng(stable_seed(spec.seed, camera_key, "gaussian"))
        changed = np.clip(
            array.astype(np.float32) + rng.standard_normal(array.shape, dtype=np.float32) * float(spec.severity),
            0.0,
            1.0,
        )
    elif spec.family == "image_translation":
        changed = _translate_numpy(array, pixels=int(spec.severity), direction=str(spec.direction))
    else:
        return image.clone() if is_torch else np.array(array, copy=True)

    if is_torch:
        import torch

        return torch.as_tensor(changed, dtype=image.dtype, device=image.device)
    return changed.astype(array.dtype, copy=False)


def perturb_raw_sample(sample: Mapping[str, Any], spec: PerturbationSpec) -> dict[str, Any]:
    """Clone one sample and alter only the allowlisted raw image or task fields."""

    result = dict(sample)
    if spec.family in {"gaussian_sensor_noise", "image_translation"}:
        missing = [key for key in IMAGE_KEYS if key not in sample]
        if missing:
            raise KeyError(f"missing frozen image keys: {missing}")
        for key in IMAGE_KEYS:
            result[key] = perturb_image(sample[key], spec, camera_key=key)
    elif spec.family in {"instruction_repetition", "context_wrapper"}:
        if not isinstance(sample.get("task"), str):
            raise TypeError("raw sample task must be a string")
        result["task"] = perturb_instruction(str(sample["task"]), spec)
    else:
        raise ValueError(f"unknown perturbation family: {spec.family}")
    return result


def value_hash(value: Any) -> str:
    digest = hashlib.sha256()
    if hasattr(value, "detach"):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(tuple(array.shape)).encode("ascii"))
        digest.update(array.tobytes())
    elif isinstance(value, str):
        digest.update(value.encode("utf-8"))
    elif isinstance(value, Mapping):
        for key in sorted(value):
            digest.update(str(key).encode("utf-8"))
            digest.update(value_hash(value[key]).encode("ascii"))
    else:
        digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest().upper()


def sorted_trainable_parameters(model: Any) -> list[tuple[str, Any]]:
    return sorted(
        [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad],
        key=lambda item: item[0],
    )


def parameter_manifest(named_parameters: Sequence[tuple[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "shape": [int(value) for value in parameter.shape],
            "numel": int(parameter.numel()),
            "dtype": str(parameter.dtype),
            "module_group": name.rsplit(".", 1)[0],
        }
        for name, parameter in named_parameters
    ]


def flatten_gradients(named_parameters: Sequence[tuple[str, Any]]) -> Any:
    import torch

    pieces = []
    for _name, parameter in named_parameters:
        gradient = parameter.grad
        if gradient is None:
            pieces.append(torch.zeros(parameter.shape, dtype=torch.float32, device=parameter.device).reshape(-1))
        else:
            pieces.append(gradient.detach().to(dtype=torch.float32).reshape(-1).clone())
    if not pieces:
        return torch.empty(0, dtype=torch.float32)
    vector = torch.cat(pieces)
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("gradient vector contains nonfinite values")
    return vector


def unflatten_vector(vector: Any, manifest: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    offset = 0
    for item in manifest:
        count = int(item["numel"])
        shape = tuple(int(value) for value in item["shape"])
        result[str(item["name"])] = vector[offset : offset + count].reshape(shape).clone()
        offset += count
    if offset != int(vector.numel()):
        raise ValueError(f"vector has {int(vector.numel())} values but manifest consumes {offset}")
    return result


def assign_vector_to_gradients(vector: Any, named_parameters: Sequence[tuple[str, Any]]) -> None:
    offset = 0
    for _name, parameter in named_parameters:
        count = int(parameter.numel())
        value = vector[offset : offset + count].reshape(parameter.shape).to(parameter.device, parameter.dtype)
        parameter.grad = value.clone()
        offset += count
    if offset != int(vector.numel()):
        raise ValueError(f"vector has {int(vector.numel())} values but parameters consume {offset}")


def module_norms(vector: Any, manifest: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    squared: dict[str, float] = defaultdict(float)
    offset = 0
    for item in manifest:
        count = int(item["numel"])
        group = str(item["module_group"])
        value = vector[offset : offset + count].float()
        squared[group] += float(value.dot(value).item())
        offset += count
    return {group: math.sqrt(value) for group, value in sorted(squared.items())}


def project_clean_gradient(
    clean_gradient: Any,
    robust_gradient: Any,
    *,
    robust_norm_squared_floor: float = ROBUST_NORM_SQUARED_FLOOR,
) -> dict[str, Any]:
    import torch

    clean = clean_gradient.detach().to(dtype=torch.float32).reshape(-1)
    robust = robust_gradient.detach().to(dtype=torch.float32).reshape(-1)
    if clean.shape != robust.shape:
        raise ValueError(f"gradient shape mismatch: {tuple(clean.shape)} != {tuple(robust.shape)}")
    if not bool(torch.isfinite(clean).all()) or not bool(torch.isfinite(robust).all()):
        raise ValueError("projection input contains nonfinite values")
    dot = float(torch.dot(clean, robust).item())
    clean_norm = float(torch.linalg.vector_norm(clean).item())
    robust_norm = float(torch.linalg.vector_norm(robust).item())
    robust_norm_squared = float(torch.dot(robust, robust).item())
    denominator = clean_norm * robust_norm
    cosine = dot / denominator if denominator > 0.0 else None
    if robust_norm_squared < robust_norm_squared_floor:
        return {
            "status": "robust_gradient_below_floor",
            "projected_gradient": None,
            "projection_coefficient": None,
            "dot_before": dot,
            "dot_after": None,
            "clean_norm": clean_norm,
            "robust_norm": robust_norm,
            "robust_norm_squared": robust_norm_squared,
            "projected_norm": None,
            "cosine": cosine,
            "gate_conflict": False,
            "constraint_tolerance": None,
            "constraint_passed": False,
        }
    if dot < 0.0:
        coefficient = -dot / robust_norm_squared
        projected = clean + coefficient * robust
        status = "projected_conflict"
    else:
        coefficient = 0.0
        projected = clean.clone()
        status = "agreeing_or_orthogonal"
    dot_after = float(torch.dot(robust, projected).item())
    projected_norm = float(torch.linalg.vector_norm(projected).item())
    tolerance = PROJECTION_TOLERANCE * max(1.0, robust_norm * projected_norm)
    return {
        "status": status,
        "projected_gradient": projected,
        "projection_coefficient": coefficient,
        "dot_before": dot,
        "dot_after": dot_after,
        "clean_norm": clean_norm,
        "robust_norm": robust_norm,
        "robust_norm_squared": robust_norm_squared,
        "projected_norm": projected_norm,
        "cosine": cosine,
        "gate_conflict": bool(cosine is not None and cosine < CONFLICT_COSINE_THRESHOLD),
        "constraint_tolerance": tolerance,
        "constraint_passed": bool(dot_after >= -tolerance),
    }


def classify_stage0(summary: Mapping[str, Any]) -> str:
    data_checks = ("partition_health", "perturbation_health", "action_target_health")
    implementation_checks = (
        "preflight_passed",
        "shared_draw_health",
        "lora_only_trainable",
        "identity_passed",
        "checkpoint_reload_passed",
        "base_unchanged",
        "mechanism_invariants_passed",
        "action_validity_passed",
        "memory_passed",
        "confirmatory_sealed",
    )
    capacity_checks = ("gradient_health", "subset_fit_passed")
    if not all(bool(summary.get(name)) for name in data_checks):
        return "IARC_DATA_OR_SUPERVISION_FAILURE"
    if not all(bool(summary.get(name)) for name in implementation_checks):
        return "IARC_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if not all(bool(summary.get(name)) for name in capacity_checks):
        return "IARC_LOW_COMPUTE_PARAMETERIZATION_INSUFFICIENT"

    conflict_count = int(summary.get("conflict_count") or 0)
    family_count = int(summary.get("conflict_family_count") or 0)
    if conflict_count >= 4 and family_count >= 2:
        return "IARC_STAGE_0A_PASS_HEADROOM_PENDING"
    if conflict_count > 0 and (conflict_count <= 3 or family_count == 1):
        return "IARC_STAGE_0A_UNDERPOWERED_ONE_CHECK_ALLOWED"
    return "IARC_DESIGN_FAILURE_NONACTING_MECHANISM"


def numeric_summary(values: Sequence[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "iqr": None, "p05": None, "p95": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "iqr": float(np.percentile(array, 75) - np.percentile(array, 25)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def clone_parameter_values(named_parameters: Sequence[tuple[str, Any]]) -> dict[str, Any]:
    return {name: parameter.detach().clone() for name, parameter in named_parameters}


def restore_parameter_values(named_parameters: Sequence[tuple[str, Any]], snapshot: Mapping[str, Any]) -> None:
    import torch

    with torch.no_grad():
        for name, parameter in named_parameters:
            parameter.copy_(snapshot[name])


def clone_sample(sample: Mapping[str, Any]) -> dict[str, Any]:
    return copy.deepcopy(dict(sample))
