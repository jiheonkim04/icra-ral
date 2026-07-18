"""Frozen RIFA-XVLA Stage 0 mechanism-smoke execution.

The runner keeps X-VLA and the trained RL4IL local prior frozen.  It attaches
one zero-initialized reliability-conditioned residual adapter at X-VLA's real
action-hidden normalization point, trains the full and no-reliability arms on
the same fixed development samples, and evaluates only preregistered mechanism
gates.  It does not perform a closed-loop rollout or use privileged inference
state.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
import traceback
import types
from dataclasses import dataclass
from typing import Any, Iterable

import h5py
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from scipy.spatial.transform import Rotation

try:  # Windows validation does not expose the POSIX resource module.
    import resource
except ImportError:  # pragma: no cover - exercised only by Windows-side tests
    resource = None  # type: ignore[assignment]

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = REPO_ROOT / "configs" / "rifa_xvla_stage0_frozen_contract.json"
DEFAULT_REPORT_JSON = REPO_ROOT / "reports" / "rifa_xvla_stage0_result.json"
DEFAULT_REPORT_MD = REPO_ROOT / "reports" / "rifa_xvla_stage0_result.md"
DEFAULT_TRACKED_CHECKPOINT_DIR = REPO_ROOT / "reports" / "checkpoints" / "rifa_xvla_stage0"
IMPLEMENTATION_LABEL = "RIFA_XVLA_FROZEN_STAGE0_LOCAL_IMPLEMENTATION"
CAM_DIM = 512


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def utcish_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def memory_report() -> dict[str, Any]:
    report: dict[str, Any] = {}
    if resource is not None:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        report["ru_maxrss_kib"] = int(usage.ru_maxrss)
    status = pathlib.Path("/proc/self/status")
    if status.exists():
        for line in status.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(("VmRSS:", "VmHWM:", "VmSize:")):
                key, value = line.split(":", 1)
                report[key.lower()] = value.strip()
    meminfo = pathlib.Path("/proc/meminfo")
    if meminfo.exists():
        for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
            if line.startswith(("MemTotal:", "MemAvailable:")):
                key, value = line.split(":", 1)
                report[key.lower()] = value.strip()
    return report


def cuda_report() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False, "pid": int(os.getpid())}
    return {
        "available": True,
        "pid": int(os.getpid()),
        "device": torch.cuda.get_device_name(0),
        "allocated_mib": float(torch.cuda.memory_allocated() / 2**20),
        "max_allocated_mib": float(torch.cuda.max_memory_allocated() / 2**20),
        "reserved_mib": float(torch.cuda.memory_reserved() / 2**20),
        "max_reserved_mib": float(torch.cuda.max_memory_reserved() / 2**20),
    }


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,driver_version",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except Exception as exc:  # pragma: no cover - runtime boundary
        return f"nvidia_smi_failed: {type(exc).__name__}: {exc}"


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_frozen_contract(path: pathlib.Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("method") != "RIFA_XVLA":
        raise ValueError("frozen contract method must be RIFA_XVLA")
    if contract.get("execution_classification") != "OURS_VLA_TRAINING":
        raise ValueError("frozen execution classification drift")
    panel = contract.get("panel") or []
    expected = [
        ("libero_goal", 0, [20260733, 20260734, 20260735]),
        ("libero_object", 0, [20260733, 20260734, 20260735]),
        ("libero_spatial", 5, [20260731, 20260732, 20260735]),
    ]
    actual = [(row.get("suite"), int(row.get("task_id")), list(row.get("identities") or [])) for row in panel]
    if actual != expected:
        raise ValueError(f"frozen panel drift: {actual!r}")
    split = contract.get("data_split") or {}
    if split.get("training_demo_indices") != "0..39" or split.get("validation_demo_indices") != "40..49":
        raise ValueError("frozen data split drift")
    budget = contract.get("training_budget") or {}
    if int(budget.get("configuration_count", 0)) != 1 or int(budget.get("batch_size", 0)) != 1:
        raise ValueError("Stage 0 must remain one configuration with batch size one")
    if int(budget.get("optimizer_steps_per_arm", 0)) <= 0:
        raise ValueError("optimizer step budget must be positive")
    if bool(budget.get("downloads_allowed")) or bool(budget.get("validation_selection_or_tuning")):
        raise ValueError("downloads and validation tuning are frozen off")
    boundary = contract.get("execution_boundary") or {}
    if bool(boundary.get("closed_loop_rollout_authorized")):
        raise ValueError("Stage 0 contract may not authorize closed-loop rollout")
    return contract


class RIFAAdapter(nn.Module):
    """Zero-initialized action-hidden residual with a reliability gate."""

    def __init__(
        self,
        hidden_size: int,
        *,
        imputed_dim: int = CAM_DIM,
        reliability_dim: int = 3,
        bottleneck_dim: int = 128,
        residual_scale: float = 0.05,
        no_reliability: bool = False,
    ) -> None:
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.imputed_dim = int(imputed_dim)
        self.reliability_dim = int(reliability_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.residual_scale = float(residual_scale)
        self.no_reliability = bool(no_reliability)
        self.imputed_projection = nn.Linear(self.imputed_dim, self.bottleneck_dim)
        self.missing_projection = nn.Linear(1, self.bottleneck_dim, bias=False)
        self.adapter_core = nn.Sequential(
            nn.LayerNorm(self.bottleneck_dim),
            nn.GELU(),
            nn.Linear(self.bottleneck_dim, self.bottleneck_dim),
            nn.GELU(),
        )
        self.residual_projection = nn.Linear(self.bottleneck_dim, self.hidden_size)
        self.reliability_gate = nn.Linear(self.reliability_dim, 1)
        nn.init.zeros_(self.residual_projection.weight)
        nn.init.zeros_(self.residual_projection.bias)
        nn.init.zeros_(self.reliability_gate.weight)
        nn.init.constant_(self.reliability_gate.bias, -2.0)

    def forward(
        self,
        hidden: torch.Tensor,
        imputed_feature: torch.Tensor,
        reliability: torch.Tensor,
        missing_indicator: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        if hidden.ndim != 3:
            raise ValueError(f"hidden must be [B,T,H], got {tuple(hidden.shape)}")
        if imputed_feature.ndim != 2 or imputed_feature.shape[-1] != self.imputed_dim:
            raise ValueError("imputed feature shape drift")
        if reliability.ndim != 2 or reliability.shape[-1] != self.reliability_dim:
            raise ValueError("reliability feature shape drift")
        if missing_indicator.ndim != 2 or missing_indicator.shape[-1] != 1:
            raise ValueError("missing indicator shape drift")
        gate_features = torch.zeros_like(reliability) if self.no_reliability else reliability
        gate = torch.sigmoid(self.reliability_gate(gate_features)) * missing_indicator
        latent = self.imputed_projection(imputed_feature) + self.missing_projection(missing_indicator)
        residual = torch.tanh(self.residual_projection(self.adapter_core(latent)))
        residual = residual * gate * self.residual_scale
        output = hidden + residual.unsqueeze(1)
        return output, {"gate": gate, "residual": residual}


class ActionHiddenHook:
    """Inject an adapter at the real X-VLA transformer.norm action output."""

    def __init__(self) -> None:
        self.adapter: RIFAAdapter | None = None
        self.context: dict[str, torch.Tensor] | None = None
        self.forward_count = 0
        self.last_gate = 0.0
        self.last_residual_norm = 0.0

    def activate(self, adapter: RIFAAdapter, context: dict[str, torch.Tensor]) -> None:
        self.adapter = adapter
        self.context = context

    def deactivate(self) -> None:
        self.adapter = None
        self.context = None

    def __call__(self, _module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        if self.adapter is None or self.context is None:
            return output
        modified, telemetry = self.adapter(
            output,
            self.context["imputed_feature"],
            self.context["reliability"],
            self.context["missing_indicator"],
        )
        self.forward_count += 1
        self.last_gate = float(telemetry["gate"].detach().float().mean().cpu())
        self.last_residual_norm = float(telemetry["residual"].detach().float().norm().cpu())
        return modified


def parameter_vector(module: nn.Module) -> torch.Tensor:
    params = [parameter.detach().float().reshape(-1).cpu() for parameter in module.parameters()]
    return torch.cat(params) if params else torch.empty(0)


def trainable_parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def gradient_global_norm(module: nn.Module) -> tuple[float, int, int]:
    total = 0.0
    tensor_count = 0
    nonzero_count = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        tensor_count += 1
        if not bool(torch.isfinite(grad).all().item()):
            return float("nan"), tensor_count, nonzero_count
        if bool(torch.count_nonzero(grad).item()):
            nonzero_count += 1
        total += float(torch.sum(grad * grad).item())
    return float(math.sqrt(total)), int(tensor_count), int(nonzero_count)


def _module_is_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError:
        module = sys.modules.get(name)
        return bool(module is not None and getattr(module, "__spec__", None) is not None)


def _make_import_shim(name: str, *, package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=package)
    if package:
        module.__path__ = []  # type: ignore[attr-defined]
    return module


def install_optional_xvla_shims() -> list[str]:
    used: list[str] = []
    if not _module_is_available("mmengine"):
        mmengine = _make_import_shim("mmengine", package=True)
        fileio = _make_import_shim("mmengine.fileio")
        fileio.get = lambda path: pathlib.Path(path).read_bytes()  # type: ignore[attr-defined]
        fileio.isdir = lambda path: pathlib.Path(path).is_dir()  # type: ignore[attr-defined]

        def list_dir_or_file(path: str, suffix: str = "", recursive: bool = False, list_dir: bool = False) -> list[str]:
            root = pathlib.Path(path)
            iterator: Iterable[pathlib.Path] = root.rglob("*") if recursive else root.iterdir()
            return [
                str(item.relative_to(root))
                for item in iterator
                if (list_dir or item.is_file()) and (not suffix or str(item).endswith(suffix))
            ]

        fileio.list_dir_or_file = list_dir_or_file  # type: ignore[attr-defined]
        fileio.join_path = lambda *parts: str(pathlib.Path(parts[0]).joinpath(*parts[1:]))  # type: ignore[attr-defined]
        mmengine.fileio = fileio  # type: ignore[attr-defined]
        sys.modules["mmengine"] = mmengine
        sys.modules["mmengine.fileio"] = fileio
        used.append("mmengine.fileio")
    if not _module_is_available("fastapi"):
        fastapi = _make_import_shim("fastapi", package=True)

        class FastAPI:
            def post(self, *_args: Any, **_kwargs: Any) -> Any:
                return lambda function: function

        fastapi.FastAPI = FastAPI  # type: ignore[attr-defined]
        responses = _make_import_shim("fastapi.responses")

        class JSONResponse(dict):
            def __init__(self, content: Any = None, status_code: int = 200, **kwargs: Any) -> None:
                super().__init__(content=content, status_code=status_code, **kwargs)

        responses.JSONResponse = JSONResponse  # type: ignore[attr-defined]
        fastapi.responses = responses  # type: ignore[attr-defined]
        sys.modules["fastapi"] = fastapi
        sys.modules["fastapi.responses"] = responses
        used.append("fastapi")
    if not _module_is_available("uvicorn"):
        uvicorn = _make_import_shim("uvicorn")
        uvicorn.run = lambda *_args, **_kwargs: None  # type: ignore[attr-defined]
        sys.modules["uvicorn"] = uvicorn
        used.append("uvicorn")
    if not _module_is_available("json_numpy"):
        json_numpy = _make_import_shim("json_numpy")
        json_numpy.loads = json.loads  # type: ignore[attr-defined]
        json_numpy.dumps = json.dumps  # type: ignore[attr-defined]
        sys.modules["json_numpy"] = json_numpy
        used.append("json_numpy")
    return used


def install_xvla_transformers_compat_patches() -> list[str]:
    from models import modeling_florence2  # type: ignore

    patches: list[str] = []
    florence = modeling_florence2.Florence2ForConditionalGeneration
    if "_supports_sdpa" not in getattr(florence, "__dict__", {}):
        florence._supports_sdpa = False
        patches.append("Florence2ForConditionalGeneration._supports_sdpa=False")
    language = modeling_florence2.Florence2LanguageForConditionalGeneration
    if not getattr(language.get_output_embeddings, "_rifa_missing_lm_head_safe", False):
        original = language.get_output_embeddings

        def safe_get_output_embeddings(self: Any) -> Any:
            return None if not hasattr(self, "lm_head") else original(self)

        safe_get_output_embeddings._rifa_missing_lm_head_safe = True  # type: ignore[attr-defined]
        language.get_output_embeddings = safe_get_output_embeddings
        patches.append("Florence2LanguageForConditionalGeneration.missing_lm_head_safe")
    return patches


def _rot6d_from_scalar_first_quat(quaternion: np.ndarray) -> np.ndarray:
    matrix = Rotation.from_quat(np.asarray(quaternion), scalar_first=True).as_matrix()
    return matrix[:, :, :2].reshape(matrix.shape[0], 6)


def build_abs_action_6d(robot_states: np.ndarray, actions: np.ndarray) -> np.ndarray:
    robot_states = np.asarray(robot_states, dtype=np.float64)
    actions = np.asarray(actions, dtype=np.float64)
    if robot_states.ndim != 2 or robot_states.shape[1] < 9:
        raise ValueError(f"robot_states shape invalid: {robot_states.shape}")
    if actions.ndim != 2 or actions.shape[0] != robot_states.shape[0] or actions.shape[1] < 7:
        raise ValueError(f"actions shape invalid: {actions.shape}")
    return np.concatenate(
        [robot_states[:, 2:5], _rot6d_from_scalar_first_quat(robot_states[:, 5:9]), actions[:, 6:7]],
        axis=1,
    ).astype(np.float32)


def _write_encoded_frames(handle: h5py.File, name: str, frames: np.ndarray) -> None:
    import cv2

    frames = np.asarray(frames, dtype=np.uint8)
    dataset = handle.create_dataset(name, shape=(frames.shape[0],), dtype=h5py.vlen_dtype(np.dtype("uint8")))
    for index, frame in enumerate(frames):
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if not ok:
            raise RuntimeError(f"failed to encode {name}[{index}]")
        dataset[index] = np.asarray(encoded, dtype=np.uint8).reshape(-1)


def materialize_xvla_clip(
    source: pathlib.Path,
    output_dir: pathlib.Path,
    *,
    demo_index: int,
    instruction: str,
    clip_steps: int,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    output_hdf5 = output_dir / f"demo_{int(demo_index)}.hdf5"
    with h5py.File(source, "r") as handle:
        group = handle["data"][f"demo_{int(demo_index)}"]
        length = min(int(clip_steps), int(group["actions"].shape[0]))
        if length < 41:
            raise ValueError(f"demo_{demo_index} has only {length} frames; need at least 41")
        actions = np.asarray(group["actions"][:length], dtype=np.float32)
        robot_states = np.asarray(group["robot_states"][:length], dtype=np.float32)
        agent = np.asarray(group["obs"]["agentview_rgb"][:length], dtype=np.uint8)
        wrist = np.asarray(group["obs"]["eye_in_hand_rgb"][:length], dtype=np.uint8)
    with h5py.File(output_hdf5, "w") as target:
        target.create_dataset("abs_action_6d", data=build_abs_action_6d(robot_states, actions), compression="gzip")
        _write_encoded_frames(target, "agentview_rgb", agent)
        _write_encoded_frames(target, "eye_in_hand_rgb", wrist)
        target.create_dataset("language_instruction", data=np.bytes_(instruction))
    meta = {
        "dataset_name": "libero",
        "datalist": [str(output_hdf5)],
        "observation_key": ["agentview_rgb", "eye_in_hand_rgb"],
        "language_instruction_key": "language_instruction",
    }
    meta_path = output_dir / "meta.json"
    atomic_write_json(meta_path, meta)
    return {
        "meta_path": str(meta_path),
        "hdf5_path": str(output_hdf5),
        "hdf5_sha256": sha256_file(output_hdf5),
        "source_hdf5": str(source),
        "source_demo_index": int(demo_index),
        "source_frame_index_for_policy_observation": 1,
        "clip_steps": int(length),
        "agent_frame": agent[1].copy(),
        "wrist_frame": wrist[1].copy(),
    }


def read_official_xvla_sample(meta_path: pathlib.Path) -> dict[str, Any]:
    from datasets.dataset import InfiniteDataReader  # type: ignore

    reader = InfiniteDataReader(str(meta_path), num_actions=30, num_views=3, training=False, action_mode="ee6d")
    return next(iter(reader))


def task_key(task: dict[str, Any]) -> str:
    return f"{task['suite']}_task{int(task['task_id'])}"


def matching_rl4il_task(task: dict[str, Any]) -> dict[str, Any]:
    from tca_map.rl4il_prior.mechanism_port import PANEL as rl4il_panel

    matches = [
        row
        for row in rl4il_panel
        if row["suite"] == task["suite"] and int(row["task_id"]) == int(task["task_id"])
    ]
    if len(matches) != 1:
        raise ValueError(f"no unique frozen RL4IL task for {task_key(task)}")
    return matches[0]


def freeze_module(module: nn.Module) -> int:
    for parameter in module.parameters():
        parameter.requires_grad = False
    module.eval()
    return trainable_parameter_count(module)


def extract_rl4il_context(
    clip: Any,
    port: dict[str, Any],
    obs: dict[str, Any],
    instruction: str,
    condition: str,
) -> dict[str, Any]:
    from tca_map.rl4il_prior.mechanism_port import (
        bfs,
        encode_live_query,
        policy_features,
        seed_neighbors,
        top_policy_candidates,
    )

    counts: dict[str, int] = {}
    q, query_meta = encode_live_query(clip, port, obs, instruction, condition, counts)
    if condition == "clean":
        train_emb = port["train_clean_emb"]
        graph = port["clean_graph"]
        policy = port["clean_policy"]
        fusion = port["clean_fusion"]
        donor_dispersion = 0.0
        missing = 0.0
    elif condition == "mask_1_in_hand_dropout":
        train_emb = port["train_mask1_emb"]
        graph = port["mask1_graph"]
        policy = port["mask1_policy"]
        fusion = port["mask1_fusion"]
        donor_ids = [int(value) for value in query_meta.get("imputation_donor_indices", [])]
        donors = port["train_raw_cam1"][donor_ids]
        donor_center = donors.mean(axis=0, keepdims=True)
        donor_dispersion = float(np.linalg.norm(donors - donor_center, axis=1).mean() / math.sqrt(CAM_DIM))
        missing = 1.0
    else:
        raise ValueError(f"unknown condition {condition!r}")

    sequence = bfs(seed_neighbors(q, train_emb), graph)
    if len(sequence) < 2:
        raise RuntimeError("RL4IL reliability extraction found fewer than two candidates")
    state_np, candidate_np = policy_features(q, sequence, train_emb, port["action_lengths"])
    device = next(policy.parameters()).device
    with torch.no_grad():
        scores = policy(
            torch.tensor(state_np, dtype=torch.float32, device=device),
            torch.tensor(candidate_np, dtype=torch.float32, device=device),
        )
        probabilities = torch.softmax(scores, dim=-1)
        entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-12))).sum()
        normalized_entropy = float((entropy / math.log(max(2, probabilities.numel()))).cpu())
    counts["retrieval_policy_forward_count"] = counts.get("retrieval_policy_forward_count", 0) + 1
    top_ids = top_policy_candidates(policy, q, sequence, train_emb, port["action_lengths"])
    with torch.no_grad():
        predicted_descriptor = fusion(
            torch.tensor(q, dtype=torch.float32, device=device),
            torch.tensor(train_emb[top_ids], dtype=torch.float32, device=device),
            torch.tensor(port["action_desc"][top_ids], dtype=torch.float32, device=device),
        )
    counts["action_fusion_forward_count"] = counts.get("action_fusion_forward_count", 0) + 1
    predicted_np = predicted_descriptor.detach().float().cpu().numpy()
    descriptor_mse = np.mean((port["action_desc"][top_ids] - predicted_np.reshape(1, -1)) ** 2, axis=1)
    ordered = np.sort(descriptor_mse)
    margin = float(ordered[1] - ordered[0])
    return {
        "imputed_feature": np.asarray(q[CAM_DIM : 2 * CAM_DIM], dtype=np.float32),
        "reliability_raw": np.asarray([donor_dispersion, margin, normalized_entropy], dtype=np.float32),
        "missing_indicator": float(missing),
        "query_metadata": query_meta,
        "module_forward_counts": counts,
        "candidate_count": int(len(sequence)),
        "descriptor_best_mse": float(ordered[0]),
        "descriptor_margin": margin,
        "policy_normalized_entropy": normalized_entropy,
        "donor_dispersion": donor_dispersion,
        "legal_deployment_inputs_only": True,
    }


def fit_reliability_normalizer(contexts: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    values = np.stack([np.asarray(context["reliability_raw"], dtype=np.float32) for context in contexts])
    mean = values.mean(axis=0)
    std = values.std(axis=0, ddof=0)
    safe_std = np.where(std < 1e-8, np.ones_like(std), std)
    return {"mean": mean.astype(np.float32), "std": safe_std.astype(np.float32), "raw_std": std.astype(np.float32)}


def normalized_context(context: dict[str, Any], normalizer: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    reliability = (np.asarray(context["reliability_raw"], dtype=np.float32) - normalizer["mean"]) / normalizer["std"]
    return {
        "imputed_feature": np.asarray(context["imputed_feature"], dtype=np.float32),
        "reliability": reliability.astype(np.float32),
        "missing_indicator": np.asarray([float(context["missing_indicator"])], dtype=np.float32),
    }


def tensor_context(context: dict[str, np.ndarray], device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "imputed_feature": torch.tensor(context["imputed_feature"], dtype=torch.float32, device=device).view(1, -1),
        "reliability": torch.tensor(context["reliability"], dtype=torch.float32, device=device).view(1, -1),
        "missing_indicator": torch.tensor(context["missing_indicator"], dtype=torch.float32, device=device).view(1, 1),
    }


def prepare_offline_inputs(
    sample: dict[str, Any],
    processor: Any,
    device: torch.device,
    *,
    condition: str,
) -> dict[str, torch.Tensor]:
    language = processor.encode_language(str(sample["language_instruction"]))["input_ids"]
    image_input = sample["image_input"].clone()
    if condition == "mask_1_in_hand_dropout":
        black = Image.fromarray(np.zeros((128, 128, 3), dtype=np.uint8))
        processed_black = processor.image_processor([black], return_tensors="pt")["pixel_values"][0]
        image_input[1] = processed_black.to(dtype=image_input.dtype)
    elif condition != "clean":
        raise ValueError(f"unknown condition {condition!r}")
    inputs = {
        "input_ids": language,
        "image_input": image_input.unsqueeze(0),
        "image_mask": sample["image_mask"].unsqueeze(0),
        "domain_id": sample["domain_id"].view(1),
        "proprio": sample["proprio"].unsqueeze(0),
        "action": sample["action"].unsqueeze(0),
    }
    return {
        key: value.to(device=device, dtype=torch.float32) if value.is_floating_point() else value.to(device=device)
        for key, value in inputs.items()
    }


def flip_agentview(image: np.ndarray) -> np.ndarray:
    return np.flip(np.flip(np.asarray(image, dtype=np.uint8), 0), 1).copy()


def matrix_to_rotate6d(matrix: np.ndarray) -> np.ndarray:
    return np.concatenate([matrix[:3, 0], matrix[:3, 1]], axis=-1).astype(np.float32)


def prepare_live_inputs(
    obs: dict[str, Any],
    instruction: str,
    processor: Any,
    device: torch.device,
    *,
    condition: str,
) -> dict[str, torch.Tensor]:
    wrist = np.asarray(obs["robot0_eye_in_hand_image"], dtype=np.uint8)
    if condition == "mask_1_in_hand_dropout":
        wrist = np.zeros_like(wrist)
    elif condition != "clean":
        raise ValueError(f"unknown condition {condition!r}")
    encoded = processor([flip_agentview(obs["agentview_image"]), wrist], instruction)
    current = np.concatenate(
        [
            np.asarray(obs["robo_pos"], dtype=np.float32),
            np.asarray(obs["robo_ori"], dtype=np.float32),
            np.asarray([0.0], dtype=np.float32),
        ]
    )
    proprio = np.concatenate([current, np.zeros_like(current)]).astype(np.float32)
    encoded["proprio"] = torch.tensor(proprio).view(1, -1)
    encoded["domain_id"] = torch.tensor([3], dtype=torch.long)
    return {
        key: value.to(device=device, dtype=torch.float32) if value.is_floating_point() else value.to(device=device)
        for key, value in encoded.items()
    }


def generate_plan(
    model: nn.Module,
    hook: ActionHiddenHook,
    inputs: dict[str, torch.Tensor],
    *,
    adapter: RIFAAdapter | None,
    context: dict[str, torch.Tensor] | None,
    denoise_steps: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    if adapter is None:
        hook.deactivate()
    else:
        if context is None:
            raise ValueError("adapter generation requires context")
        hook.activate(adapter, context)
    generation_inputs = {key: value for key, value in inputs.items() if key != "action"}
    try:
        with torch.no_grad():
            output = model.generate_actions(**generation_inputs, steps=int(denoise_steps))
        return output.detach().float().cpu().numpy().squeeze(0)
    finally:
        hook.deactivate()


def rotate6d_to_rotvec(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    a1 = values[:, :3]
    a2 = values[:, 3:6]
    b1 = a1 / (np.linalg.norm(a1, axis=1, keepdims=True) + 1e-9)
    b2 = a2 - np.sum(b1 * a2, axis=1, keepdims=True) * b1
    b2 = b2 / (np.linalg.norm(b2, axis=1, keepdims=True) + 1e-9)
    b3 = np.cross(b1, b2)
    matrices = np.stack([b1, b2, b3], axis=-1)
    return Rotation.from_matrix(matrices).as_rotvec().astype(np.float32)


def plan_to_libero_actions(plan: np.ndarray) -> np.ndarray:
    plan = np.asarray(plan, dtype=np.float32)
    gripper = np.where(plan[:, 9:10] > 0.5, 1.0, -1.0).astype(np.float32)
    return np.concatenate([plan[:, :3], rotate6d_to_rotvec(plan[:, 3:9]), gripper], axis=-1).astype(np.float32)


def action_delta(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_actions = plan_to_libero_actions(left)
    right_actions = plan_to_libero_actions(right)
    delta = left_actions - right_actions
    return {
        "rms": float(np.sqrt(np.mean(delta.astype(np.float64) ** 2))),
        "max_abs": float(np.max(np.abs(delta))),
        "translation_rms": float(np.sqrt(np.mean(delta[:, :3].astype(np.float64) ** 2))),
        "rotation_rms": float(np.sqrt(np.mean(delta[:, 3:6].astype(np.float64) ** 2))),
        "gripper_flip_count": int(np.count_nonzero(delta[:, 6])),
        "finite": bool(np.isfinite(left_actions).all() and np.isfinite(right_actions).all()),
    }


def train_arm(
    name: str,
    adapter: RIFAAdapter,
    model: nn.Module,
    hook: ActionHiddenHook,
    samples: list[dict[str, Any]],
    device: torch.device,
    budget: dict[str, Any],
    checkpoint_path: pathlib.Path,
    heartbeat: callable,
) -> dict[str, Any]:
    adapter.train()
    before = parameter_vector(adapter)
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(budget["learning_rate"]),
        weight_decay=float(budget["weight_decay"]),
    )
    requested_steps = int(budget["optimizer_steps_per_arm"])
    losses: list[float] = []
    gradient_norms: list[float] = []
    grad_tensor_counts: list[int] = []
    nonzero_grad_tensor_counts: list[int] = []
    gates: list[float] = []
    residual_norms: list[float] = []
    for step in range(requested_steps):
        sample = samples[step % len(samples)]
        inputs = sample["model_inputs"]
        context = tensor_context(sample["normalized_context"], device)
        optimizer.zero_grad(set_to_none=True)
        hook.activate(adapter, context)
        torch.manual_seed(int(budget["seed"]) + step)
        torch.cuda.manual_seed_all(int(budget["seed"]) + step)
        try:
            loss_dict = model(**inputs)
        finally:
            hook.deactivate()
        loss = sum(loss_dict.values())
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"{name} produced nonfinite loss at step {step + 1}")
        loss.backward()
        norm, grad_tensors, nonzero_tensors = gradient_global_norm(adapter)
        if not np.isfinite(norm):
            raise RuntimeError(f"{name} produced nonfinite gradients at step {step + 1}")
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(budget["max_grad_norm"]))
        optimizer.step()
        losses.append(float(loss.detach().float().cpu()))
        gradient_norms.append(float(norm))
        grad_tensor_counts.append(int(grad_tensors))
        nonzero_grad_tensor_counts.append(int(nonzero_tensors))
        gates.append(float(hook.last_gate))
        residual_norms.append(float(hook.last_residual_norm))
        heartbeat(f"train_{name}_step_{step + 1}_of_{requested_steps}")
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(adapter.state_dict(), checkpoint_path)
    after = parameter_vector(adapter)
    reloaded = RIFAAdapter(
        adapter.hidden_size,
        imputed_dim=adapter.imputed_dim,
        reliability_dim=adapter.reliability_dim,
        bottleneck_dim=adapter.bottleneck_dim,
        residual_scale=adapter.residual_scale,
        no_reliability=adapter.no_reliability,
    ).to(device)
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    reload_delta = float(torch.norm(parameter_vector(reloaded) - after).item())
    reloaded.eval()
    return {
        "name": name,
        "adapter": reloaded,
        "metrics": {
            "trainable_parameter_count": trainable_parameter_count(adapter),
            "optimizer_steps": int(len(losses)),
            "first_loss": float(losses[0]),
            "final_loss": float(losses[-1]),
            "losses": losses,
            "gradient_norms": gradient_norms,
            "gradient_tensor_counts": grad_tensor_counts,
            "nonzero_gradient_tensor_counts": nonzero_grad_tensor_counts,
            "finite_nonzero_gradients": bool(
                gradient_norms
                and all(np.isfinite(value) for value in gradient_norms)
                and any(value > 0.0 for value in gradient_norms)
                and any(value > 0 for value in nonzero_grad_tensor_counts)
            ),
            "weight_change_l2": float(torch.norm(after - before).item()),
            "weights_changed": bool(torch.norm(after - before).item() > 0.0),
            "gate_values": gates,
            "residual_norms": residual_norms,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "bytes": int(checkpoint_path.stat().st_size),
                "disk_reload_ok": bool(reload_delta == 0.0),
                "reload_parameter_delta_l2": reload_delta,
            },
        },
    }


def summarize_delta_rows(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row[key] for row in rows]
    return {
        "count": len(values),
        "mean_rms": float(np.mean([value["rms"] for value in values])),
        "max_rms": float(np.max([value["rms"] for value in values])),
        "max_abs": float(np.max([value["max_abs"] for value in values])),
        "all_finite": bool(all(bool(value["finite"]) for value in values)),
        "total_gripper_flips": int(sum(int(value["gripper_flip_count"]) for value in values)),
    }


def apply_stage0_decision(gates: dict[str, bool]) -> str:
    if gates and all(bool(value) for value in gates.values()):
        return "RIFA_XVLA_STAGE0_PASS"
    data_gates = {"missing_modality_signal_observable", "rl4il_reliability_features_nonconstant"}
    implementation_gates = {
        "real_xvla_forward_path",
        "cuda_execution",
        "trainable_parameters_nonzero_and_matched",
        "finite_nonzero_gradients",
        "optimizer_steps_exact",
        "weights_changed",
        "checkpoint_write_and_disk_reload",
        "base_preserving_initialization",
    }
    design_gates = {"bounded_action_delta", "clean_validation_retained", "action_outputs_finite"}
    if any(not bool(gates.get(name, False)) for name in data_gates):
        return "RIFA_XVLA_STAGE0_DATA_OR_SUPERVISION_FAILURE"
    if any(not bool(gates.get(name, False)) for name in implementation_gates):
        return "RIFA_XVLA_STAGE0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if any(not bool(gates.get(name, False)) for name in design_gates):
        return "RIFA_XVLA_STAGE0_DESIGN_FAILURE"
    if not bool(gates.get("full_vs_no_reliability_difference", False)):
        return "RIFA_XVLA_STAGE0_UNDERPOWERED_OR_UNRESOLVED"
    return "RIFA_XVLA_STAGE0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"


def write_markdown(path: pathlib.Path, result: dict[str, Any]) -> None:
    gates = result.get("gates") or {}
    full = ((result.get("training") or {}).get("RIFA_XVLA") or {})
    ablation = ((result.get("training") or {}).get("RIFA_XVLA_NO_RELIABILITY") or {})
    validation = result.get("validation") or {}
    lines = [
        "# RIFA-XVLA Stage 0 Result",
        "",
        f"- Decision: `{result.get('decision')}`",
        f"- Execution classification: `{result.get('execution_classification')}`",
        f"- Execution valid: `{result.get('execution_valid')}`",
        f"- CUDA PID: `{result.get('cuda_pid')}`",
        f"- Peak VRAM MiB: `{(result.get('cuda') or {}).get('max_allocated_mib')}`",
        f"- X-VLA forward calls: `{(result.get('forward_counts') or {}).get('xvla_model_forward_calls')}`",
        f"- Full trainable parameters: `{full.get('trainable_parameter_count')}`",
        f"- Full / ablation optimizer steps: `{full.get('optimizer_steps')} / {ablation.get('optimizer_steps')}`",
        f"- Dropout full-vs-Base max RMS: `{((validation.get('dropout_full_vs_base') or {}).get('max_rms'))}`",
        f"- Dropout full-vs-ablation mean RMS: `{((validation.get('dropout_full_vs_ablation') or {}).get('mean_rms'))}`",
        f"- Clean full-vs-Base max RMS: `{((validation.get('clean_full_vs_base') or {}).get('max_rms'))}`",
        "",
        "## Frozen gates",
        "",
        "| gate | pass |",
        "|---|---|",
    ]
    for name, value in gates.items():
        lines.append(f"| `{name}` | `{value}` |")
    lines += [
        "",
        "No closed-loop Ours rollout or official success measurement occurred in Stage 0. "
        "The frozen X-VLA base, RL4IL checkpoints, and CLIP encoders remained frozen.",
    ]
    if result.get("exceptions"):
        lines += ["", "## Exceptions", "", "```json", json.dumps(result["exceptions"], indent=2), "```"]
    write_text(path, "\n".join(lines) + "\n")


@dataclass
class RuntimePaths:
    run_dir: pathlib.Path
    status: pathlib.Path
    heartbeat: pathlib.Path
    partial: pathlib.Path
    result: pathlib.Path
    result_md: pathlib.Path
    exit_code: pathlib.Path


def run_stage0(run_dir: pathlib.Path, contract_path: pathlib.Path = DEFAULT_CONTRACT) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = RuntimePaths(
        run_dir=run_dir,
        status=run_dir / "status.json",
        heartbeat=run_dir / "heartbeat.json",
        partial=run_dir / "partial_result.json",
        result=run_dir / "result.json",
        result_md=run_dir / "result.md",
        exit_code=run_dir / "exit_code.txt",
    )
    contract = load_frozen_contract(contract_path)
    contract_hash = sha256_file(contract_path)
    result: dict[str, Any] = {
        "schema_version": "2026-07-18.epoch5_rifa_xvla_stage0_result.v1",
        "execution_classification": "OURS_VLA_TRAINING",
        "implementation_label": IMPLEMENTATION_LABEL,
        "method": "RIFA_XVLA",
        "stage": "frozen_stage0_mechanism_smoke",
        "run_dir": str(run_dir),
        "pid": int(os.getpid()),
        "cuda_pid": int(os.getpid()) if torch.cuda.is_available() else None,
        "contract": {"path": str(contract_path), "sha256": contract_hash},
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_at": utcish_timestamp(),
        "execution_valid": False,
        "decision": "RIFA_XVLA_STAGE0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE",
        "exceptions": [],
        "repair_count": 0,
        "no_training_beyond_frozen_budget": True,
        "no_ours_closed_loop_rollout": True,
        "official_closed_loop_success_measured": False,
        "no_confirmatory_tuning": True,
        "no_new_method_generated": True,
        "no_prior_search_reopened": True,
        "no_natural_reset_mining": True,
        "no_privileged_inference_input": True,
        "model_offload_used": False,
        "downloads_used": False,
        "nvidia_smi_before": nvidia_smi(),
    }

    def heartbeat(stage: str) -> None:
        payload = {"timestamp": utcish_timestamp(), "stage": stage, "pid": int(os.getpid())}
        atomic_write_json(paths.heartbeat, payload)
        atomic_write_json(paths.status, {**payload, "state": "running"})
        partial = {
            "schema_version": result["schema_version"],
            "execution_classification": result["execution_classification"],
            "method": result["method"],
            "stage": stage,
            "pid": int(os.getpid()),
            "contract_sha256": contract_hash,
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        atomic_write_json(paths.partial, partial)

    atomic_write_json(
        run_dir / "launch_manifest.json",
        {
            "schema_version": "2026-07-18.epoch5_rifa_xvla_stage0_launch_manifest.v1",
            "execution_classification": "OURS_VLA_TRAINING",
            "contract_path": str(contract_path),
            "contract_sha256": contract_hash,
            "source_head": result["source_head"],
            "python": sys.executable,
            "argv": sys.argv,
            "pid": int(os.getpid()),
            "frozen_panel": contract["panel"],
            "training_budget": contract["training_budget"],
            "decision_thresholds": contract["decision_thresholds"],
            "closed_loop_rollout_authorized": False,
        },
    )
    write_text(run_dir / "worker_pid.txt", f"{os.getpid()}\n")
    write_text(
        run_dir / "exact_resume_command.txt",
        f"{sys.executable} scripts/run_rifa_xvla_stage0.py --run-dir {run_dir} --contract {contract_path}\n",
    )

    clip: Any | None = None
    model: nn.Module | None = None
    ports: dict[str, dict[str, Any]] = {}
    hook_handle: Any = None
    try:
        heartbeat("risk_and_contract_validation")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for frozen RIFA Stage 0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda:0")
        seed = int(contract["training_budget"]["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        result["risk_assessment"] = {
            "source": "local tracked code and already-present checkpoints/data only",
            "downloads": "disabled",
            "xvla_checkpoint_present": True,
            "rl4il_training_dir_present": pathlib.Path(contract["source_prior_training_dir"]).is_dir(),
            "dataset_files_present": all(pathlib.Path(task["hdf5"]).is_file() for task in contract["panel"]),
            "cuda_available": True,
            "system_ram_before": memory_report(),
            "cuda_before": cuda_report(),
            "no_cpu_or_disk_model_offload": True,
        }
        if not result["risk_assessment"]["rl4il_training_dir_present"]:
            raise FileNotFoundError(contract["source_prior_training_dir"])
        if not result["risk_assessment"]["dataset_files_present"]:
            raise FileNotFoundError("one or more frozen LIBERO HDF5 files are missing")

        heartbeat("load_frozen_rl4il_prior")
        from tca_map.rl4il_prior.mechanism_port import FrozenCLIPEncoder, load_task_port

        clip = FrozenCLIPEncoder(device)
        freeze_module(clip)
        prior_training_dir = pathlib.Path(contract["source_prior_training_dir"])
        prior_hashes: dict[str, Any] = {}
        frozen_prior_trainable = 0
        for task in contract["panel"]:
            frozen_task = matching_rl4il_task(task)
            port = load_task_port(prior_training_dir, frozen_task, device)
            for key in ("clean_policy", "clean_fusion", "mask1_policy", "mask1_fusion", "imp_policy", "soft_imp"):
                frozen_prior_trainable += freeze_module(port[key])
            ports[task_key(task)] = port
            prior_hashes[task_key(task)] = {
                "bundle": port["bundle_sha256"],
                "checkpoints": port["checkpoint_hashes"],
            }
        result["frozen_prior"] = {
            "implementation_label": "MECHANISM_FAITHFUL_RL4IL_LOCAL_PORT",
            "training_dir": str(prior_training_dir),
            "trainable_parameter_count": int(frozen_prior_trainable),
            "clip_trainable_parameter_count": trainable_parameter_count(clip),
            "artifact_hashes": prior_hashes,
        }

        heartbeat("materialize_fixed_offline_samples")
        xvla_root = pathlib.Path(contract["xvla"]["source_root"])
        if str(xvla_root) in sys.path:
            sys.path.remove(str(xvla_root))
        sys.path.insert(0, str(xvla_root))
        shims = install_optional_xvla_shims()
        materialized_root = run_dir / "materialized_fixed_samples"
        materialized_root.mkdir(parents=True, exist_ok=False)
        train_samples: list[dict[str, Any]] = []
        validation_samples: list[dict[str, Any]] = []
        materialized_rows: list[dict[str, Any]] = []
        split = contract["data_split"]
        for task in contract["panel"]:
            port = ports[task_key(task)]
            for split_name, indices in (
                ("training", split["stage0_training_demo_indices_per_task"]),
                ("validation", split["stage0_validation_demo_indices_per_task"]),
            ):
                for demo_index in indices:
                    sample_dir = materialized_root / f"{task_key(task)}_{split_name}_demo{int(demo_index)}"
                    materialized = materialize_xvla_clip(
                        pathlib.Path(task["hdf5"]),
                        sample_dir,
                        demo_index=int(demo_index),
                        instruction=str(task["instruction"]),
                        clip_steps=int(split["materialized_clip_steps"]),
                    )
                    sample = read_official_xvla_sample(pathlib.Path(materialized["meta_path"]))
                    raw_obs = {
                        "agentview_image": materialized.pop("agent_frame"),
                        "robot0_eye_in_hand_image": materialized.pop("wrist_frame"),
                    }
                    conditions = ["mask_1_in_hand_dropout"] if split_name == "training" else list(split["validation_conditions"])
                    for condition in conditions:
                        context_raw = extract_rl4il_context(
                            clip,
                            port,
                            raw_obs,
                            str(task["instruction"]),
                            condition,
                        )
                        row = {
                            "sample_key": f"{task_key(task)}_{split_name}_demo{int(demo_index)}_{condition}",
                            "task": task,
                            "split": split_name,
                            "demo_index": int(demo_index),
                            "condition": condition,
                            "sample": sample,
                            "context_raw": context_raw,
                        }
                        (train_samples if split_name == "training" else validation_samples).append(row)
                    materialized_rows.append(materialized)
        result["data"] = {
            "training_demo_indices": split["stage0_training_demo_indices_per_task"],
            "validation_demo_indices": split["stage0_validation_demo_indices_per_task"],
            "training_sample_count": len(train_samples),
            "validation_sample_count": len(validation_samples),
            "materialized": materialized_rows,
            "split_overlap": bool(
                set(split["stage0_training_demo_indices_per_task"])
                & set(split["stage0_validation_demo_indices_per_task"])
            ),
            "confirmatory_test_data_used": False,
        }
        normalizer = fit_reliability_normalizer([sample["context_raw"] for sample in train_samples])
        for sample in train_samples + validation_samples:
            sample["normalized_context"] = normalized_context(sample["context_raw"], normalizer)
        raw_train_reliability = np.stack([sample["context_raw"]["reliability_raw"] for sample in train_samples])
        result["reliability"] = {
            "feature_names": contract["mechanism"]["reliability_features"],
            "training_raw_mean": normalizer["mean"].tolist(),
            "training_raw_std": normalizer["raw_std"].tolist(),
            "training_raw_range": np.ptp(raw_train_reliability, axis=0).tolist(),
            "nonconstant_by_feature": [bool(value > 1e-8) for value in np.ptp(raw_train_reliability, axis=0)],
            "deployment_inputs_only": True,
        }

        heartbeat("load_frozen_xvla")
        result["optional_import_shims_used"] = shims
        result["transformers_compat_patches"] = install_xvla_transformers_compat_patches()
        from models.modeling_xvla import XVLA  # type: ignore
        from models.processing_xvla import XVLAProcessor  # type: ignore

        xvla_config = contract["xvla"]
        model = XVLA.from_pretrained(
            xvla_config["model_id"],
            revision=xvla_config["model_revision"],
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
            cache_dir=xvla_config["cache_dir"],
        )
        processor = XVLAProcessor.from_pretrained(
            xvla_config["model_id"],
            revision=xvla_config["model_revision"],
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=xvla_config["cache_dir"],
        )
        freeze_module(model)
        model.to(device=device, dtype=torch.float32)
        model.eval()
        hidden_size = int(model.transformer.hidden_size)
        mechanism = contract["mechanism"]
        full_adapter = RIFAAdapter(
            hidden_size,
            imputed_dim=int(mechanism["imputed_feature_dim"]),
            bottleneck_dim=int(mechanism["adapter_bottleneck_dim"]),
            residual_scale=float(mechanism["maximum_hidden_residual_scale"]),
            no_reliability=False,
        ).to(device)
        ablation_adapter = RIFAAdapter(
            hidden_size,
            imputed_dim=int(mechanism["imputed_feature_dim"]),
            bottleneck_dim=int(mechanism["adapter_bottleneck_dim"]),
            residual_scale=float(mechanism["maximum_hidden_residual_scale"]),
            no_reliability=True,
        ).to(device)
        full_adapter.load_state_dict(ablation_adapter.state_dict())
        hook = ActionHiddenHook()
        hook_handle = model.transformer.norm.register_forward_hook(hook)
        result["xvla"] = {
            "model_id": xvla_config["model_id"],
            "model_revision": xvla_config["model_revision"],
            "source_root": xvla_config["source_root"],
            "source_revision": xvla_config["source_revision"],
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "base_trainable_parameter_count": trainable_parameter_count(model),
            "hidden_size": hidden_size,
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
            "actual_action_hidden_hook_module": "model.transformer.norm",
        }
        result["runtime_dependencies"] = {
            name: package_version(name) for name in ["torch", "transformers", "h5py", "scikit-learn", "scipy"]
        }

        heartbeat("prepare_real_xvla_training_inputs")
        for sample in train_samples + validation_samples:
            sample["model_inputs"] = prepare_offline_inputs(
                sample["sample"],
                processor,
                device,
                condition=sample["condition"],
            )
        cuda_tensor_count = sum(
            int(value.is_cuda)
            for sample in train_samples
            for value in sample["model_inputs"].values()
        )
        result["cuda_tensor_count"] = int(cuda_tensor_count)

        heartbeat("verify_base_preserving_initialization")
        first = train_samples[0]
        first_context = tensor_context(first["normalized_context"], device)
        denoise_steps = int(xvla_config["denoise_steps"])
        base_initial = generate_plan(
            model,
            hook,
            first["model_inputs"],
            adapter=None,
            context=None,
            denoise_steps=denoise_steps,
            seed=seed + 1000,
        )
        full_initial = generate_plan(
            model,
            hook,
            first["model_inputs"],
            adapter=full_adapter,
            context=first_context,
            denoise_steps=denoise_steps,
            seed=seed + 1000,
        )
        ablation_initial = generate_plan(
            model,
            hook,
            first["model_inputs"],
            adapter=ablation_adapter,
            context=first_context,
            denoise_steps=denoise_steps,
            seed=seed + 1000,
        )
        initial_full_delta = action_delta(full_initial, base_initial)
        initial_ablation_delta = action_delta(ablation_initial, base_initial)
        result["base_preserving_initialization"] = {
            "residual_projection_exact_zero_full": bool(
                torch.count_nonzero(full_adapter.residual_projection.weight).item() == 0
                and torch.count_nonzero(full_adapter.residual_projection.bias).item() == 0
            ),
            "residual_projection_exact_zero_ablation": bool(
                torch.count_nonzero(ablation_adapter.residual_projection.weight).item() == 0
                and torch.count_nonzero(ablation_adapter.residual_projection.bias).item() == 0
            ),
            "full_vs_base": initial_full_delta,
            "ablation_vs_base": initial_ablation_delta,
        }

        heartbeat("train_frozen_full_and_ablation")
        checkpoint_dir = run_dir / "checkpoints"
        full_run = train_arm(
            "RIFA_XVLA",
            full_adapter,
            model,
            hook,
            train_samples,
            device,
            contract["training_budget"],
            checkpoint_dir / "rifa_xvla_full.pt",
            heartbeat,
        )
        ablation_run = train_arm(
            "RIFA_XVLA_NO_RELIABILITY",
            ablation_adapter,
            model,
            hook,
            train_samples,
            device,
            contract["training_budget"],
            checkpoint_dir / "rifa_xvla_no_reliability.pt",
            heartbeat,
        )
        full_adapter = full_run.pop("adapter")
        ablation_adapter = ablation_run.pop("adapter")
        result["training"] = {
            "RIFA_XVLA": full_run["metrics"],
            "RIFA_XVLA_NO_RELIABILITY": ablation_run["metrics"],
        }

        heartbeat("evaluate_fixed_offline_validation")
        evaluation_rows: list[dict[str, Any]] = []
        generation_calls = 3
        for row_index, sample in enumerate(validation_samples):
            context = tensor_context(sample["normalized_context"], device)
            eval_seed = seed + 2000 + row_index
            base = generate_plan(
                model,
                hook,
                sample["model_inputs"],
                adapter=None,
                context=None,
                denoise_steps=denoise_steps,
                seed=eval_seed,
            )
            full = generate_plan(
                model,
                hook,
                sample["model_inputs"],
                adapter=full_adapter,
                context=context,
                denoise_steps=denoise_steps,
                seed=eval_seed,
            )
            generation_calls += 2
            output = {
                "source": "fixed_validation_demo",
                "sample_key": sample["sample_key"],
                "suite": sample["task"]["suite"],
                "task_id": int(sample["task"]["task_id"]),
                "demo_index": int(sample["demo_index"]),
                "condition": sample["condition"],
                "full_vs_base": action_delta(full, base),
                "reliability_raw": sample["context_raw"]["reliability_raw"].tolist(),
                "missing_indicator": float(sample["context_raw"]["missing_indicator"]),
            }
            if sample["condition"] == "mask_1_in_hand_dropout":
                ablation = generate_plan(
                    model,
                    hook,
                    sample["model_inputs"],
                    adapter=ablation_adapter,
                    context=context,
                    denoise_steps=denoise_steps,
                    seed=eval_seed,
                )
                generation_calls += 1
                output["full_vs_ablation"] = action_delta(full, ablation)
            evaluation_rows.append(output)

        heartbeat("evaluate_frozen_live_initial_observations")
        os.environ.setdefault("MUJOCO_GL", "egl")
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        live_context_rows: list[dict[str, Any]] = []
        for task in contract["panel"]:
            suite = benchmark.get_benchmark_dict()[str(task["suite"])]()
            libero_task = suite.get_task(int(task["task_id"]))
            bddl_file = pathlib.Path(get_libero_path("bddl_files")) / libero_task.problem_folder / libero_task.bddl_file
            initial_states = suite.get_task_init_states(int(task["task_id"]))
            port = ports[task_key(task)]
            for identity in task["identities"]:
                identity = int(identity)
                index = identity - 20260711
                env = None
                try:
                    env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=128, camera_widths=128)
                    env.seed(identity)
                    env.reset()
                    obs = env.set_init_state(np.asarray(initial_states[index], dtype=np.float64))
                    for _ in range(10):
                        obs, _reward, _done, _info = env.step(
                            np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
                        )
                    obs["robo_ori"] = matrix_to_rotate6d(env.env.robots[0].controller.ee_ori_mat)
                    obs["robo_pos"] = np.asarray(env.env.robots[0].controller.ee_pos, dtype=np.float32)
                    for condition in contract["conditions"]:
                        context_raw = extract_rl4il_context(
                            clip,
                            port,
                            obs,
                            str(task["instruction"]),
                            condition,
                        )
                        context_np = normalized_context(context_raw, normalizer)
                        context = tensor_context(context_np, device)
                        inputs = prepare_live_inputs(
                            obs,
                            str(task["instruction"]),
                            processor,
                            device,
                            condition=condition,
                        )
                        eval_seed = seed + 3000 + len(live_context_rows)
                        base = generate_plan(
                            model,
                            hook,
                            inputs,
                            adapter=None,
                            context=None,
                            denoise_steps=denoise_steps,
                            seed=eval_seed,
                        )
                        full = generate_plan(
                            model,
                            hook,
                            inputs,
                            adapter=full_adapter,
                            context=context,
                            denoise_steps=denoise_steps,
                            seed=eval_seed,
                        )
                        generation_calls += 2
                        output = {
                            "source": "official_live_initial_observation",
                            "suite": task["suite"],
                            "task_id": int(task["task_id"]),
                            "reset_identity": identity,
                            "initial_state_index": int(index),
                            "condition": condition,
                            "full_vs_base": action_delta(full, base),
                            "reliability_raw": context_raw["reliability_raw"].tolist(),
                            "reliability_normalized": context_np["reliability"].tolist(),
                            "missing_indicator": float(context_raw["missing_indicator"]),
                            "legal_deployment_inputs_only": True,
                        }
                        if condition == "mask_1_in_hand_dropout":
                            ablation = generate_plan(
                                model,
                                hook,
                                inputs,
                                adapter=ablation_adapter,
                                context=context,
                                denoise_steps=denoise_steps,
                                seed=eval_seed,
                            )
                            generation_calls += 1
                            output["full_vs_ablation"] = action_delta(full, ablation)
                        live_context_rows.append(output)
                        heartbeat(f"live_probe_{task_key(task)}_{identity}_{condition}")
                finally:
                    if env is not None:
                        env.close()
        evaluation_rows.extend(live_context_rows)

        clean_rows = [row for row in evaluation_rows if row["condition"] == "clean"]
        dropout_rows = [row for row in evaluation_rows if row["condition"] == "mask_1_in_hand_dropout"]
        clean_summary = summarize_delta_rows(clean_rows, "full_vs_base")
        dropout_summary = summarize_delta_rows(dropout_rows, "full_vs_base")
        ablation_summary = summarize_delta_rows(dropout_rows, "full_vs_ablation")
        all_live_reliability = np.asarray(
            [row["reliability_raw"] for row in live_context_rows if row["condition"] == "mask_1_in_hand_dropout"],
            dtype=np.float32,
        )
        result["validation"] = {
            "rows": evaluation_rows,
            "offline_validation_row_count": len([row for row in evaluation_rows if row["source"] == "fixed_validation_demo"]),
            "live_initial_observation_row_count": len(live_context_rows),
            "live_simulator_reset_probe_count": 9,
            "simulator_episode_count": 0,
            "closed_loop_action_step_count": 0,
            "clean_full_vs_base": clean_summary,
            "dropout_full_vs_base": dropout_summary,
            "dropout_full_vs_ablation": ablation_summary,
            "live_dropout_reliability_raw_range": np.ptp(all_live_reliability, axis=0).tolist(),
        }
        result["forward_counts"] = {
            "xvla_model_forward_calls": int(
                2 * int(contract["training_budget"]["optimizer_steps_per_arm"]) + generation_calls
            ),
            "xvla_training_forward_calls": int(
                2 * int(contract["training_budget"]["optimizer_steps_per_arm"])
            ),
            "xvla_generate_actions_calls": int(generation_calls),
            "xvla_action_transformer_calls": int(
                2 * int(contract["training_budget"]["optimizer_steps_per_arm"])
                + generation_calls * denoise_steps
            ),
            "rifa_action_hidden_hook_calls": int(hook.forward_count),
            "rl4il_module_forward_count": int(
                sum(
                    sum(int(value) for value in sample["context_raw"]["module_forward_counts"].values())
                    for sample in train_samples + validation_samples
                )
                + sum(
                    1 for _row in live_context_rows
                )
            ),
        }

        thresholds = contract["decision_thresholds"]
        full_metrics = result["training"]["RIFA_XVLA"]
        ablation_metrics = result["training"]["RIFA_XVLA_NO_RELIABILITY"]
        train_ranges = np.asarray(result["reliability"]["training_raw_range"], dtype=np.float64)
        live_ranges = np.asarray(result["validation"]["live_dropout_reliability_raw_range"], dtype=np.float64)
        reliability_nonconstant = bool(
            np.all(train_ranges > float(thresholds["reliability_raw_range_min_exclusive"]))
            and np.all(live_ranges > float(thresholds["reliability_raw_range_min_exclusive"]))
        )
        all_missing = [float(sample["context_raw"]["missing_indicator"]) for sample in train_samples + validation_samples]
        all_missing.extend(float(row["missing_indicator"]) for row in live_context_rows)
        result["gates"] = {
            "real_xvla_forward_path": bool(
                result["xvla"]["model_class"] == "XVLA"
                and result["forward_counts"]["xvla_model_forward_calls"] > 0
                and result["forward_counts"]["rifa_action_hidden_hook_calls"] > 0
            ),
            "cuda_execution": bool(
                torch.cuda.is_available()
                and result["cuda_tensor_count"] > 0
                and str(result["xvla"]["device"]).startswith("cuda")
            ),
            "trainable_parameters_nonzero_and_matched": bool(
                int(full_metrics["trainable_parameter_count"]) >= int(thresholds["trainable_parameter_count_min"])
                and int(full_metrics["trainable_parameter_count"])
                == int(ablation_metrics["trainable_parameter_count"])
                and result["xvla"]["base_trainable_parameter_count"] == 0
                and result["frozen_prior"]["trainable_parameter_count"] == 0
                and result["frozen_prior"]["clip_trainable_parameter_count"] == 0
            ),
            "finite_nonzero_gradients": bool(
                full_metrics["finite_nonzero_gradients"] and ablation_metrics["finite_nonzero_gradients"]
            ),
            "optimizer_steps_exact": bool(
                int(full_metrics["optimizer_steps"]) == int(thresholds["optimizer_steps_per_arm_exact"])
                and int(ablation_metrics["optimizer_steps"]) == int(thresholds["optimizer_steps_per_arm_exact"])
            ),
            "weights_changed": bool(full_metrics["weights_changed"] and ablation_metrics["weights_changed"]),
            "checkpoint_write_and_disk_reload": bool(
                full_metrics["checkpoint"]["disk_reload_ok"]
                and ablation_metrics["checkpoint"]["disk_reload_ok"]
            ),
            "base_preserving_initialization": bool(
                result["base_preserving_initialization"]["residual_projection_exact_zero_full"]
                and result["base_preserving_initialization"]["residual_projection_exact_zero_ablation"]
                and initial_full_delta["rms"] <= float(thresholds["base_preserving_initial_action_rms_max"])
                and initial_ablation_delta["rms"] <= float(thresholds["base_preserving_initial_action_rms_max"])
            ),
            "missing_modality_signal_observable": bool(
                max(all_missing) - min(all_missing) >= float(thresholds["missing_indicator_range_min"])
            ),
            "rl4il_reliability_features_nonconstant": reliability_nonconstant,
            "full_vs_no_reliability_difference": bool(
                ablation_summary["mean_rms"]
                > float(thresholds["full_vs_ablation_dropout_action_rms_min_exclusive"])
            ),
            "bounded_action_delta": bool(
                dropout_summary["max_rms"] <= float(thresholds["dropout_full_vs_base_action_rms_max"])
                and dropout_summary["max_abs"] <= float(thresholds["dropout_full_vs_base_action_max_abs_max"])
            ),
            "clean_validation_retained": bool(
                clean_summary["max_rms"] <= float(thresholds["clean_full_vs_base_action_rms_max"])
                and clean_summary["max_abs"] <= float(thresholds["clean_full_vs_base_action_max_abs_max"])
            ),
            "action_outputs_finite": bool(
                clean_summary["all_finite"]
                and dropout_summary["all_finite"]
                and ablation_summary["all_finite"]
            ),
        }
        result["decision"] = apply_stage0_decision(result["gates"])
        result["execution_valid"] = True
        result["frozen_decision_rule_applied"] = True
        result["next_empirical_stage"] = {
            "automatically_authorized": False,
            "executed": False,
            "reason": (
                "The Stage 0 contract explicitly authorizes no closed-loop rollout and names no automatic follow-on empirical stage. "
                "A pass permits a separately frozen Stage A protocol; it does not itself preregister one."
            ),
        }

        heartbeat("persist_tracked_checkpoints_and_result")
        DEFAULT_TRACKED_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        tracked_checkpoints = {}
        for source, name in (
            (checkpoint_dir / "rifa_xvla_full.pt", "rifa_xvla_full.pt"),
            (checkpoint_dir / "rifa_xvla_no_reliability.pt", "rifa_xvla_no_reliability.pt"),
        ):
            target = DEFAULT_TRACKED_CHECKPOINT_DIR / name
            shutil.copy2(source, target)
            tracked_checkpoints[name] = {
                "path": str(target),
                "sha256": sha256_file(target),
                "bytes": int(target.stat().st_size),
            }
        result["tracked_checkpoints"] = tracked_checkpoints
    except Exception as exc:  # pragma: no cover - empirical boundary
        result["exceptions"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-120:],
            }
        )
        result["execution_valid"] = False
        result["decision"] = "RIFA_XVLA_STAGE0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
        result["next_empirical_stage"] = {
            "automatically_authorized": False,
            "executed": False,
            "reason": "Stage 0 execution was invalid; no follow-on empirical stage is allowed.",
        }
    finally:
        if hook_handle is not None:
            try:
                hook_handle.remove()
            except Exception:
                pass
        try:
            del model, clip, ports
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                result["cuda"] = cuda_report()
                torch.cuda.empty_cache()
        except Exception as cleanup_exc:
            result.setdefault("cleanup_exceptions", []).append(
                {"type": type(cleanup_exc).__name__, "message": str(cleanup_exc)}
            )
        result["system_ram"] = memory_report()
        result["nvidia_smi_after"] = nvidia_smi()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["finished_at"] = utcish_timestamp()
        process_exit_code = 0 if result.get("execution_valid") else 2
        result["process_exit_code"] = int(process_exit_code)
        atomic_write_json(paths.result, result)
        write_markdown(paths.result_md, result)
        shutil.copy2(paths.result, DEFAULT_REPORT_JSON)
        shutil.copy2(paths.result_md, DEFAULT_REPORT_MD)
        atomic_write_json(
            paths.status,
            {
                "timestamp": utcish_timestamp(),
                "stage": "finished",
                "state": "complete" if result.get("execution_valid") else "failed",
                "pid": int(os.getpid()),
                "decision": result.get("decision"),
            },
        )
        atomic_write_json(
            paths.heartbeat,
            {"timestamp": utcish_timestamp(), "stage": "finished", "pid": int(os.getpid())},
        )
        write_text(paths.exit_code, f"{process_exit_code}\n")
    return int(result["process_exit_code"]), result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--contract", type=pathlib.Path, default=DEFAULT_CONTRACT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code, result = run_stage0(pathlib.Path(args.run_dir), pathlib.Path(args.contract))
    print(json.dumps({"decision": result.get("decision"), "execution_valid": result.get("execution_valid")}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
