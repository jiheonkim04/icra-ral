#!/usr/bin/env python
"""Run the frozen official-prior-first A2C2 problem verification.

Empirical modes are exactly classified as SETUP_PREFLIGHT,
CACHED_FEATURE_PROBE, PRIOR_MODULE_TRAINING, or VLA_CLOSED_LOOP_ROLLOUT.
This script never trains the frozen SmolVLA and never executes an Ours method.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.a2c2_local_prior import (  # noqa: E402
    A2C2LocalConfig,
    A2C2ResidualTransformer,
    FIDELITY_LABEL,
    OFFICIAL_COMMIT,
    SmolVLAHiddenCapture,
    parameter_counts,
    phase_feature,
    tensor_sha256,
)
from tca_map.smolvla.official_canonical_eval import (  # noqa: E402
    _load_base_policy_and_processors,
    _make_noise,
    _postprocess_chunk,
)
from tca_map.smolvla.official_closed_loop_scaleup import (  # noqa: E402
    _extract_single_env,
    _rss_mb,
    _successes_from_info,
)
from tca_map.smolvla.official_libero_baseline_scaleup import _add_training_batch_dims  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import (  # noqa: E402
    PolicySpec,
    _cuda_memory,
    _load_policy_and_processors,
    _make_env_cfg,
    _set_runtime_env,
)


DATE_KST = "2026-07-19"
SCHEMA_VERSION = 1
SUITE = "libero_spatial"
EVAL_TASKS = [
    {
        "task_id": 0,
        "global_task_index": 34,
        "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    },
    {
        "task_id": 4,
        "global_task_index": 31,
        "instruction": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    },
    {
        "task_id": 8,
        "global_task_index": 36,
        "instruction": "pick up the black bowl next to the plate and place it on the plate",
    },
]
INIT_STATE_IDS = [0, 1, 2, 3, 4]
TRAIN_EPISODES_BY_TASK = {
    30: [1261, 1274, 1277, 1291],
    31: [1262, 1263, 1268, 1276],
    32: [1264, 1265, 1266, 1271],
    33: [1267, 1269, 1270, 1308],
    34: [1272, 1273, 1275, 1282],
    35: [1278, 1279, 1303, 1319],
    36: [1280, 1285, 1293, 1307],
    37: [1281, 1286, 1294, 1297],
    38: [1283, 1287, 1289, 1299],
    39: [1290, 1295, 1296, 1306],
}
TRAIN_EPISODES = [episode for task in sorted(TRAIN_EPISODES_BY_TASK) for episode in TRAIN_EPISODES_BY_TASK[task]]
BASE_CONDITIONS = {
    "BASE_STANDARD_E10_D0": {"execution_horizon": 10, "inference_delay": 0},
    "BASE_DELAYED_E40_D10": {"execution_horizon": 40, "inference_delay": 10},
}
PRIOR_CONDITION = "PRIOR_DELAYED_E40_D10"
CHUNK_SIZE = 50
MAX_STEPS = 220
BASE_SEED = 2026071901
FEATURE_SEED = 2026071902
TRAIN_SEED = 2026071903
RAM_LIMIT_FRACTION = 0.82
VRAM_RESERVED_LIMIT_FRACTION = 0.88


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(type(value).__name__)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    temporary.replace(path)


def _write_md(path: Path, title: str, payload: Mapping[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"Date: `{DATE_KST} KST`",
        "",
        f"Fidelity label: `{FIDELITY_LABEL}`",
        "",
        f"Final decision: `{payload.get('final_decision')}`",
        "",
        "```json",
        json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _round(value: Any, digits: int = 6) -> Any:
    if value is None:
        return None
    return round(float(value), digits)


def _runtime_environment(args: argparse.Namespace) -> None:
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def _resource_snapshot(torch_mod: Any | None = None) -> dict[str, Any]:
    import psutil

    memory = psutil.virtual_memory()
    result: dict[str, Any] = {
        "pid": os.getpid(),
        "rss_mb": _round(_rss_mb(), 3),
        "system_ram_total_gib": _round(memory.total / 1024**3, 3),
        "system_ram_used_gib": _round(memory.used / 1024**3, 3),
        "system_ram_used_fraction": _round(memory.percent / 100.0, 6),
        "system_ram_limit_fraction": RAM_LIMIT_FRACTION,
    }
    if torch_mod is not None and torch_mod.cuda.is_available():
        props = torch_mod.cuda.get_device_properties(0)
        total = int(props.total_memory)
        reserved = int(torch_mod.cuda.memory_reserved(0))
        allocated = int(torch_mod.cuda.memory_allocated(0))
        result.update(
            {
                "cuda_pid": os.getpid(),
                "gpu_name": props.name,
                "vram_total_mib": _round(total / 1024**2, 3),
                "vram_allocated_mib": _round(allocated / 1024**2, 3),
                "vram_reserved_mib": _round(reserved / 1024**2, 3),
                "vram_reserved_fraction": _round(reserved / total, 6),
                "vram_reserved_limit_fraction": VRAM_RESERVED_LIMIT_FRACTION,
            }
        )
    return result


def _enforce_resources(torch_mod: Any | None = None) -> dict[str, Any]:
    snapshot = _resource_snapshot(torch_mod)
    if float(snapshot["system_ram_used_fraction"]) > RAM_LIMIT_FRACTION:
        raise RuntimeError(f"RESOURCE_LIMIT_SYSTEM_RAM: {snapshot}")
    if snapshot.get("vram_reserved_fraction") is not None and float(snapshot["vram_reserved_fraction"]) > VRAM_RESERVED_LIMIT_FRACTION:
        raise RuntimeError(f"RESOURCE_LIMIT_VRAM_RESERVED: {snapshot}")
    return snapshot


def _set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _stats(args: argparse.Namespace) -> dict[str, Any]:
    return json.loads((Path(args.dataset_root) / "meta" / "stats.json").read_text(encoding="utf-8"))


def _normalization_tensors(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    stats = _stats(args)
    image_means = []
    image_stds = []
    for key in ("observation.images.image", "observation.images.image2"):
        image_means.append(np.asarray(stats[key]["mean"], dtype=np.float32).reshape(3))
        image_stds.append(np.asarray(stats[key]["std"], dtype=np.float32).reshape(3))
    return {
        "image_mean": torch.tensor(np.stack(image_means)),
        "image_std": torch.tensor(np.stack(image_stds)),
        "state_mean": torch.tensor(stats["observation.state"]["mean"], dtype=torch.float32),
        "state_std": torch.tensor(stats["observation.state"]["std"], dtype=torch.float32),
        "action_mean": torch.tensor(stats["action"]["mean"], dtype=torch.float32),
        "action_std": torch.tensor(stats["action"]["std"], dtype=torch.float32),
    }


def _make_prior(args: argparse.Namespace, hidden_dim: int, *, pretrained: bool = True) -> A2C2ResidualTransformer:
    values = _normalization_tensors(args)
    config = A2C2LocalConfig(
        vlm_hidden_dim=int(hidden_dim),
        pretrained_backbone_weights=("ResNet18_Weights.IMAGENET1K_V1" if pretrained else None),
    )
    return A2C2ResidualTransformer(config, **values)


def _dataset(args: argparse.Namespace) -> Any:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    return LeRobotDataset(
        "lerobot/libero",
        root=Path(args.dataset_root),
        episodes=TRAIN_EPISODES,
        video_backend=args.video_backend,
    )


def _sample_images(sample: Mapping[str, Any]) -> Any:
    import torch

    images = torch.stack(
        (
            sample["observation.images.image"].to(torch.float32),
            sample["observation.images.image2"].to(torch.float32),
        ),
        dim=0,
    )
    return images


def _base_prediction(
    *,
    policy: Any,
    capture: SmolVLAHiddenCapture,
    preprocessor: Any,
    postprocessor: Any,
    raw_sample: Mapping[str, Any],
    seed: int,
) -> tuple[np.ndarray, Any]:
    import torch

    batch = _add_training_batch_dims(preprocessor(dict(raw_sample)))
    noise = _make_noise(policy, int(seed), torch)
    policy.reset()
    with torch.inference_mode():
        raw_chunk = policy.predict_action_chunk(batch, noise=noise)
    chunk, _ = _postprocess_chunk(raw_chunk, postprocessor, 7)
    hidden = capture.pop().detach().cpu()
    if chunk.shape != (CHUNK_SIZE, 7):
        raise RuntimeError(f"unexpected SmolVLA chunk shape {chunk.shape}")
    if hidden.ndim != 2 or hidden.shape[0] != 1:
        raise RuntimeError(f"unexpected SmolVLA hidden shape {tuple(hidden.shape)}")
    return chunk.astype(np.float32), hidden[0]


def _base_loader_args(args: argparse.Namespace) -> argparse.Namespace:
    values = vars(args).copy()
    values.update(
        {
            "checkpoint_path": str(args.base_path),
            "chunk_size": CHUNK_SIZE,
            "hf_home": str(args.hf_home),
            "vlm_root": str(args.vlm_root),
        }
    )
    return argparse.Namespace(**values)


def run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    _runtime_environment(args)
    _set_seed(BASE_SEED)
    started = time.monotonic()
    before = _enforce_resources(torch)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "date": f"{DATE_KST} KST",
        "job_classification": "SETUP_PREFLIGHT",
        "fidelity_label": FIDELITY_LABEL,
        "official_commit": OFFICIAL_COMMIT,
        "model_forward_count": 0,
        "training_happened": False,
        "closed_loop_rollout_happened": False,
        "before_resources": before,
        "exceptions": [],
    }
    try:
        policy, cfg, preprocessor, postprocessor, load_info = _load_base_policy_and_processors(_base_loader_args(args))
        dataset = _dataset(args)
        sample = dataset[0]
        with SmolVLAHiddenCapture(policy) as capture:
            chunk, hidden = _base_prediction(
                policy=policy,
                capture=capture,
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                raw_sample=sample,
                seed=BASE_SEED,
            )
        report["model_forward_count"] = 1
        prior = _make_prior(args, int(hidden.shape[-1])).to("cuda").eval()
        images = _sample_images(sample).unsqueeze(0).to("cuda")
        with torch.inference_mode():
            features = prior.encode_images(images)
            batch_size = int(args.microbatch)
            loss, _ = prior.training_loss(
                image_features=features.expand(batch_size, -1, -1, -1, -1).contiguous(),
                state=sample["observation.state"].reshape(1, 8).to("cuda").expand(batch_size, -1),
                base_action=torch.tensor(chunk[0], device="cuda").reshape(1, 7).expand(batch_size, -1),
                target_action=sample["action"].reshape(1, 7).to("cuda").expand(batch_size, -1),
                base_chunk=torch.tensor(chunk, device="cuda").unsqueeze(0).expand(batch_size, -1, -1),
                time_feature=phase_feature(0).to("cuda").reshape(1, 2).expand(batch_size, -1),
                vlm_hidden=hidden.to("cuda").reshape(1, -1).expand(batch_size, -1),
                tasks=[str(sample["task"])] * batch_size,
            )
        after = _enforce_resources(torch)
        report.update(
            {
                "base_load_info": load_info,
                "base_config_class": type(cfg).__name__,
                "base_checkpoint_sha256": _sha256_file(Path(args.base_path) / "model.safetensors"),
                "chunk_shape": list(chunk.shape),
                "chunk_finite": bool(np.isfinite(chunk).all()),
                "vlm_hidden_shape": list(hidden.shape),
                "vlm_hidden_finite": bool(torch.isfinite(hidden).all()),
                "prior_parameter_counts": parameter_counts(prior),
                "frozen_vision_backbone": all(not parameter.requires_grad for parameter in prior.image_encoder.parameters()),
                "cached_feature_shape": list(features.shape),
                "microbatch": int(args.microbatch),
                "microbatch_forward_loss": float(loss.detach().cpu()),
                "after_resources": after,
                "elapsed_seconds": _round(time.monotonic() - started, 3),
                "final_decision": "A2C2_SETUP_PREFLIGHT_ACCEPTED",
            }
        )
    except Exception as exc:
        report["exceptions"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc().splitlines()[-30:],
            }
        )
        report["elapsed_seconds"] = _round(time.monotonic() - started, 3)
        report["final_decision"] = "A2C2_SETUP_PREFLIGHT_FAILED"
    finally:
        if torch.cuda.is_available():
            report["peak_vram"] = _cuda_memory(torch)
            torch.cuda.empty_cache()
    _write_json(Path(args.preflight_output), report)
    _write_md(Path(args.preflight_md), "A2C2 Prior Setup Preflight", report)
    return report


def _episode_bounds(dataset: Any) -> list[tuple[int, int]]:
    from_values = dataset.episode_data_index["from"].tolist()
    to_values = dataset.episode_data_index["to"].tolist()
    return [(int(start), int(stop)) for start, stop in zip(from_values, to_values, strict=True)]


def _offsets(max_offset: int) -> list[int]:
    if max_offset <= 0:
        return [0]
    return sorted({0, int(max_offset // 3), int((2 * max_offset) // 3), int(max_offset)})


def _cache_datasets(handle: Any, *, hidden_dim: int, feature_shape: Sequence[int]) -> dict[str, Any]:
    import h5py

    text_dtype = h5py.string_dtype(encoding="utf-8")
    specifications = {
        "image_features": ((0, 2, *feature_shape), (None, 2, *feature_shape), np.float16),
        "state": ((0, 8), (None, 8), np.float32),
        "base_action": ((0, 7), (None, 7), np.float32),
        "target_action": ((0, 7), (None, 7), np.float32),
        "base_chunk": ((0, 50, 7), (None, 50, 7), np.float32),
        "time_feature": ((0, 2), (None, 2), np.float32),
        "vlm_hidden": ((0, hidden_dim), (None, hidden_dim), np.float32),
        "episode_index": ((0,), (None,), np.int64),
        "frame_index": ((0,), (None,), np.int64),
        "anchor_local_index": ((0,), (None,), np.int64),
        "offset": ((0,), (None,), np.int64),
        "task_index": ((0,), (None,), np.int64),
    }
    datasets = {}
    for name, (shape, maxshape, dtype) in specifications.items():
        if name in handle:
            datasets[name] = handle[name]
        else:
            chunks = (1, *shape[1:])
            datasets[name] = handle.create_dataset(
                name,
                shape=shape,
                maxshape=maxshape,
                chunks=chunks,
                dtype=dtype,
                compression="gzip" if name == "image_features" else None,
                compression_opts=1 if name == "image_features" else None,
            )
    if "task" not in handle:
        datasets["task"] = handle.create_dataset("task", shape=(0,), maxshape=(None,), dtype=text_dtype)
    else:
        datasets["task"] = handle["task"]
    return datasets


def _append_cache(datasets: Mapping[str, Any], batch: Mapping[str, Any]) -> None:
    count = int(len(batch["episode_index"]))
    if count <= 0:
        return
    start = int(datasets["episode_index"].shape[0])
    stop = start + count
    for name, dataset in datasets.items():
        dataset.resize((stop, *dataset.shape[1:]))
        dataset[start:stop] = batch[name]


def run_cache(args: argparse.Namespace) -> dict[str, Any]:
    import h5py
    import torch

    _runtime_environment(args)
    _set_seed(FEATURE_SEED)
    started = time.monotonic()
    before = _enforce_resources(torch)
    preflight = json.loads(Path(args.preflight_output).read_text(encoding="utf-8"))
    if preflight.get("final_decision") != "A2C2_SETUP_PREFLIGHT_ACCEPTED":
        raise RuntimeError("accepted setup preflight is required before cache generation")
    dataset = _dataset(args)
    policy, _cfg, preprocessor, postprocessor, load_info = _load_base_policy_and_processors(_base_loader_args(args))
    cache_path = Path(args.cache_path)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    rows_before = 0
    forward_count = 0
    completed_anchor_count = 0
    peak_rss = _rss_mb()
    peak_snapshot = before
    exceptions: list[dict[str, Any]] = []
    prior = None
    with h5py.File(cache_path, "a") as handle, SmolVLAHiddenCapture(policy) as capture:
        completed_anchors = set(handle["anchor_local_index"][:].tolist()) if "anchor_local_index" in handle else set()
        rows_before = int(handle["episode_index"].shape[0]) if "episode_index" in handle else 0
        datasets = None
        for start, stop in _episode_bounds(dataset):
            for anchor in range(start, stop, int(args.anchor_stride)):
                max_offset = min(CHUNK_SIZE - 1, (stop - 1) - anchor)
                if max_offset < 0 or anchor in completed_anchors:
                    continue
                try:
                    anchor_sample = dataset[anchor]
                    episode_index = int(anchor_sample["episode_index"].item())
                    frame_index = int(anchor_sample["frame_index"].item())
                    seed = FEATURE_SEED + episode_index * 10000 + frame_index
                    chunk, hidden = _base_prediction(
                        policy=policy,
                        capture=capture,
                        preprocessor=preprocessor,
                        postprocessor=postprocessor,
                        raw_sample=anchor_sample,
                        seed=seed,
                    )
                    forward_count += 1
                    if prior is None:
                        prior = _make_prior(args, int(hidden.shape[-1])).to("cuda").eval()
                    samples = [dataset[anchor + offset] for offset in _offsets(max_offset)]
                    images = torch.stack([_sample_images(sample) for sample in samples], dim=0).to("cuda")
                    with torch.inference_mode():
                        features = prior.encode_images(images).detach().cpu().to(torch.float16).numpy()
                    offsets = _offsets(max_offset)
                    task_values = [str(sample["task"]) for sample in samples]
                    batch = {
                        "image_features": features,
                        "state": np.stack([sample["observation.state"].numpy() for sample in samples]).astype(np.float32),
                        "base_action": np.stack([chunk[offset] for offset in offsets]).astype(np.float32),
                        "target_action": np.stack([sample["action"].numpy() for sample in samples]).astype(np.float32),
                        "base_chunk": np.repeat(chunk[None, :, :], len(samples), axis=0).astype(np.float32),
                        "time_feature": phase_feature(torch.tensor(offsets)).numpy().astype(np.float32),
                        "vlm_hidden": np.repeat(hidden.numpy()[None, :], len(samples), axis=0).astype(np.float32),
                        "episode_index": np.asarray([episode_index] * len(samples), dtype=np.int64),
                        "frame_index": np.asarray([frame_index + offset for offset in offsets], dtype=np.int64),
                        "anchor_local_index": np.asarray([anchor] * len(samples), dtype=np.int64),
                        "offset": np.asarray(offsets, dtype=np.int64),
                        "task_index": np.asarray([int(sample["task_index"].item()) for sample in samples], dtype=np.int64),
                        "task": np.asarray(task_values, dtype=object),
                    }
                    if datasets is None:
                        datasets = _cache_datasets(
                            handle,
                            hidden_dim=int(hidden.shape[-1]),
                            feature_shape=features.shape[2:],
                        )
                        handle.attrs["fidelity_label"] = FIDELITY_LABEL
                        handle.attrs["official_commit"] = OFFICIAL_COMMIT
                        handle.attrs["hidden_dim"] = int(hidden.shape[-1])
                        handle.attrs["anchor_stride"] = int(args.anchor_stride)
                    _append_cache(datasets, batch)
                    handle.flush()
                    completed_anchors.add(anchor)
                    completed_anchor_count += 1
                    if completed_anchor_count % 10 == 0:
                        snapshot = _enforce_resources(torch)
                        peak_snapshot = snapshot if float(snapshot["rss_mb"]) >= float(peak_snapshot["rss_mb"]) else peak_snapshot
                        peak_rss = max(peak_rss, _rss_mb())
                        status = {
                            "status": "running",
                            "job_classification": "CACHED_FEATURE_PROBE",
                            "completed_anchors_this_run": completed_anchor_count,
                            "model_forward_count_this_run": forward_count,
                            "cache_rows": int(handle["episode_index"].shape[0]),
                            "elapsed_seconds": _round(time.monotonic() - started, 3),
                            "resource_snapshot": snapshot,
                        }
                        _write_json(Path(args.cache_status), status)
                        print("[a2c2-cache] " + json.dumps(status, sort_keys=True), flush=True)
                except Exception as exc:
                    exceptions.append(
                        {
                            "anchor_local_index": anchor,
                            "type": type(exc).__name__,
                            "message": str(exc),
                            "traceback": traceback.format_exc().splitlines()[-24:],
                        }
                    )
                    raise
        row_count = int(handle["episode_index"].shape[0]) if "episode_index" in handle else 0
        unique_episodes = len(set(handle["episode_index"][:].tolist())) if row_count else 0
        task_counts = Counter(int(value) for value in handle["task_index"][:].tolist()) if row_count else Counter()
        hidden_dim = int(handle.attrs.get("hidden_dim", 0))
    report = {
        "schema_version": SCHEMA_VERSION,
        "date": f"{DATE_KST} KST",
        "job_classification": "CACHED_FEATURE_PROBE",
        "fidelity_label": FIDELITY_LABEL,
        "official_commit": OFFICIAL_COMMIT,
        "training_happened": False,
        "closed_loop_rollout_happened": False,
        "cache_path": str(cache_path),
        "cache_sha256": _sha256_file(cache_path),
        "rows_before_resume": rows_before,
        "row_count": row_count,
        "unique_episode_count": unique_episodes,
        "task_counts": {str(key): int(value) for key, value in sorted(task_counts.items())},
        "frozen_training_episode_ids": TRAIN_EPISODES,
        "anchor_stride": int(args.anchor_stride),
        "offset_rule": "sorted unique {0, floor(max/3), floor(2max/3), max}, max=min(49, remaining episode steps)",
        "base_model_forward_count_this_run": forward_count,
        "completed_anchor_count_this_run": completed_anchor_count,
        "vlm_hidden_dim": hidden_dim,
        "base_load_info": load_info,
        "peak_rss_mb": _round(peak_rss, 3),
        "peak_resource_snapshot": peak_snapshot,
        "peak_vram": _cuda_memory(torch),
        "exceptions": exceptions,
        "elapsed_seconds": _round(time.monotonic() - started, 3),
        "final_decision": "A2C2_CACHED_FEATURES_ACCEPTED" if row_count > 0 and unique_episodes == 40 and not exceptions else "A2C2_CACHED_FEATURES_INVALID",
    }
    _write_json(Path(args.cache_output), report)
    _write_md(Path(args.cache_md), "A2C2 Cached Feature Probe", report)
    _write_json(Path(args.cache_status), {"status": "completed", **report})
    torch.cuda.empty_cache()
    return report


class _H5TrainingDataset:
    def __init__(self, path: Path) -> None:
        import h5py

        self.path = path
        self.handle = h5py.File(path, "r")

    def __len__(self) -> int:
        return int(self.handle["episode_index"].shape[0])

    def batch(self, indices: Sequence[int], device: str) -> dict[str, Any]:
        import torch

        ordered = np.asarray(indices, dtype=np.int64)
        # h5py requires monotonic unique fancy indices, so small frozen batches are read row-wise.
        def rows(name: str) -> np.ndarray:
            return np.stack([self.handle[name][int(index)] for index in ordered], axis=0)

        tasks = []
        for index in ordered:
            value = self.handle["task"][int(index)]
            tasks.append(value.decode("utf-8") if isinstance(value, bytes) else str(value))
        return {
            "image_features": torch.tensor(rows("image_features"), device=device, dtype=torch.float32),
            "state": torch.tensor(rows("state"), device=device, dtype=torch.float32),
            "base_action": torch.tensor(rows("base_action"), device=device, dtype=torch.float32),
            "target_action": torch.tensor(rows("target_action"), device=device, dtype=torch.float32),
            "base_chunk": torch.tensor(rows("base_chunk"), device=device, dtype=torch.float32),
            "time_feature": torch.tensor(rows("time_feature"), device=device, dtype=torch.float32),
            "vlm_hidden": torch.tensor(rows("vlm_hidden"), device=device, dtype=torch.float32),
            "tasks": tasks,
        }

    @property
    def hidden_dim(self) -> int:
        return int(self.handle.attrs["hidden_dim"])

    def close(self) -> None:
        self.handle.close()


def _save_prior_checkpoint(path: Path, *, model: Any, optimizer: Any, step: int, args: argparse.Namespace) -> None:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "fidelity_label": FIDELITY_LABEL,
        "official_commit": OFFICIAL_COMMIT,
        "step": int(step),
        "config": model.config.to_dict(),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "training_contract": {
            "steps": int(args.training_steps),
            "microbatch": int(args.microbatch),
            "learning_rate": float(args.learning_rate),
            "weight_decay": float(args.weight_decay),
            "grad_clip_norm": float(args.grad_clip_norm),
            "seed": TRAIN_SEED,
        },
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _load_prior_checkpoint(path: Path, args: argparse.Namespace, *, device: str) -> tuple[Any, dict[str, Any]]:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=False)
    config = A2C2LocalConfig(**payload["config"])
    values = _normalization_tensors(args)
    model = A2C2ResidualTransformer(config, **values)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device)
    return model, payload


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    _runtime_environment(args)
    _set_seed(TRAIN_SEED)
    started = time.monotonic()
    before = _enforce_resources(torch)
    cache_report = json.loads(Path(args.cache_output).read_text(encoding="utf-8"))
    if cache_report.get("final_decision") != "A2C2_CACHED_FEATURES_ACCEPTED":
        raise RuntimeError("accepted cached features are required before prior training")
    dataset = _H5TrainingDataset(Path(args.cache_path))
    model = _make_prior(args, dataset.hidden_dim).to("cuda").train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    counts = parameter_counts(model)
    initial_trainable_hash = tensor_sha256(parameter for parameter in model.parameters() if parameter.requires_grad)
    initial_backbone_hash = tensor_sha256(model.image_encoder.parameters())
    sentinel_name, sentinel_parameter = next(
        (name, parameter) for name, parameter in model.named_parameters() if name.startswith("residual_head") and parameter.requires_grad
    )
    sentinel_initial = sentinel_parameter.detach().cpu().clone()
    losses: list[float] = []
    grad_norms: list[float] = []
    exceptions: list[dict[str, Any]] = []
    checkpoint_records: list[dict[str, Any]] = []
    peak_rss = _rss_mb()
    peak_snapshot = before
    generator = torch.Generator(device="cpu")
    generator.manual_seed(TRAIN_SEED)
    optimizer_steps = 0
    first_loss = None
    final_loss = None
    try:
        for step in range(1, int(args.training_steps) + 1):
            indices = torch.randint(0, len(dataset), (int(args.microbatch),), generator=generator).tolist()
            batch = dataset.batch(indices, "cuda")
            optimizer.zero_grad(set_to_none=True)
            loss, _metrics = model.training_loss(
                image_features=batch["image_features"],
                state=batch["state"],
                base_action=batch["base_action"],
                target_action=batch["target_action"],
                base_chunk=batch["base_chunk"],
                time_feature=batch["time_feature"],
                vlm_hidden=batch["vlm_hidden"],
                tasks=batch["tasks"],
            )
            if not torch.isfinite(loss):
                raise RuntimeError(f"nonfinite loss at step {step}: {loss}")
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                float(args.grad_clip_norm),
                error_if_nonfinite=True,
            )
            optimizer.step()
            optimizer_steps += 1
            value = float(loss.detach().cpu())
            gradient = float(grad_norm.detach().cpu())
            if first_loss is None:
                first_loss = value
            final_loss = value
            losses.append(value)
            grad_norms.append(gradient)
            if step % int(args.save_every) == 0 or step == int(args.training_steps):
                checkpoint_path = Path(args.checkpoint_dir) / f"step_{step:06d}.pt"
                _save_prior_checkpoint(checkpoint_path, model=model, optimizer=optimizer, step=step, args=args)
                checkpoint_records.append(
                    {
                        "step": step,
                        "path": str(checkpoint_path),
                        "sha256": _sha256_file(checkpoint_path),
                        "bytes": checkpoint_path.stat().st_size,
                    }
                )
            if step % int(args.log_every) == 0:
                snapshot = _enforce_resources(torch)
                peak_rss = max(peak_rss, _rss_mb())
                peak_snapshot = snapshot if float(snapshot["rss_mb"]) >= float(peak_snapshot["rss_mb"]) else peak_snapshot
                status = {
                    "status": "running",
                    "job_classification": "PRIOR_MODULE_TRAINING",
                    "step": step,
                    "optimizer_steps": optimizer_steps,
                    "loss_last": value,
                    "loss_last_100_mean": _round(np.mean(losses[-100:]), 9),
                    "grad_norm_last": _round(gradient, 9),
                    "elapsed_seconds": _round(time.monotonic() - started, 3),
                    "resource_snapshot": snapshot,
                }
                _write_json(Path(args.training_status), status)
                print("[a2c2-train] " + json.dumps(status, sort_keys=True), flush=True)
    except Exception as exc:
        exceptions.append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc().splitlines()[-30:],
            }
        )
    final_trainable_hash = tensor_sha256(parameter for parameter in model.parameters() if parameter.requires_grad)
    final_backbone_hash = tensor_sha256(model.image_encoder.parameters())
    sentinel_change = float(torch.max(torch.abs(sentinel_parameter.detach().cpu() - sentinel_initial)))
    disk_reload = {
        "attempted": False,
        "succeeded": False,
        "max_abs_output_diff": None,
        "exception": None,
    }
    if checkpoint_records and not exceptions:
        disk_reload["attempted"] = True
        try:
            model.eval()
            probe = dataset.batch(list(range(min(int(args.microbatch), len(dataset)))), "cuda")
            with torch.inference_mode():
                original = model.forward_normalized(
                    image_features=probe["image_features"],
                    state_normalized=model.normalize_state(probe["state"]),
                    base_action_normalized=model.normalize_action(probe["base_action"]),
                    base_chunk_normalized=model.normalize_action(probe["base_chunk"]),
                    time_feature=probe["time_feature"],
                    vlm_hidden=probe["vlm_hidden"],
                    tasks=probe["tasks"],
                )
            del model
            torch.cuda.empty_cache()
            reloaded, payload = _load_prior_checkpoint(Path(checkpoint_records[-1]["path"]), args, device="cuda")
            reloaded.eval()
            with torch.inference_mode():
                repeated = reloaded.forward_normalized(
                    image_features=probe["image_features"],
                    state_normalized=reloaded.normalize_state(probe["state"]),
                    base_action_normalized=reloaded.normalize_action(probe["base_action"]),
                    base_chunk_normalized=reloaded.normalize_action(probe["base_chunk"]),
                    time_feature=probe["time_feature"],
                    vlm_hidden=probe["vlm_hidden"],
                    tasks=probe["tasks"],
                )
            disk_reload.update(
                {
                    "succeeded": True,
                    "reloaded_step": int(payload["step"]),
                    "max_abs_output_diff": float(torch.max(torch.abs(original - repeated)).detach().cpu()),
                }
            )
        except Exception as exc:
            disk_reload["exception"] = {"type": type(exc).__name__, "message": str(exc)}
    accepted = bool(
        not exceptions
        and counts["trainable"] > 0
        and optimizer_steps == int(args.training_steps)
        and first_loss is not None
        and final_loss is not None
        and grad_norms
        and all(math.isfinite(value) and value > 0.0 for value in grad_norms)
        and initial_trainable_hash != final_trainable_hash
        and sentinel_change > 0.0
        and initial_backbone_hash == final_backbone_hash
        and checkpoint_records
        and disk_reload["succeeded"]
        and float(disk_reload["max_abs_output_diff"]) == 0.0
    )
    report = {
        "schema_version": SCHEMA_VERSION,
        "date": f"{DATE_KST} KST",
        "job_classification": "PRIOR_MODULE_TRAINING",
        "fidelity_label": FIDELITY_LABEL,
        "official_commit": OFFICIAL_COMMIT,
        "vla_training_happened": False,
        "ours_training_happened": False,
        "prior_module_training_happened": True,
        "cuda_pid": os.getpid(),
        "dataset_rows": len(dataset),
        "trainable_parameter_count": counts["trainable"],
        "total_parameter_count": counts["total"],
        "frozen_parameter_count": counts["frozen"],
        "optimizer": "AdamW",
        "optimizer_steps": optimizer_steps,
        "microbatch": int(args.microbatch),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "gradient_clip_norm": float(args.grad_clip_norm),
        "first_loss": first_loss,
        "final_loss": final_loss,
        "last_100_loss_mean": _round(np.mean(losses[-100:]), 9) if losses else None,
        "gradient_norm": {
            "first": grad_norms[0] if grad_norms else None,
            "final": grad_norms[-1] if grad_norms else None,
            "min": min(grad_norms) if grad_norms else None,
            "max": max(grad_norms) if grad_norms else None,
            "all_finite_nonzero": bool(grad_norms and all(math.isfinite(value) and value > 0.0 for value in grad_norms)),
        },
        "initial_trainable_hash": initial_trainable_hash,
        "final_trainable_hash": final_trainable_hash,
        "sentinel_parameter": sentinel_name,
        "sentinel_max_abs_change": sentinel_change,
        "initial_frozen_backbone_hash": initial_backbone_hash,
        "final_frozen_backbone_hash": final_backbone_hash,
        "frozen_backbone_unchanged": initial_backbone_hash == final_backbone_hash,
        "checkpoints": checkpoint_records,
        "disk_reload": disk_reload,
        "before_resources": before,
        "peak_rss_mb": _round(peak_rss, 3),
        "peak_resource_snapshot": peak_snapshot,
        "peak_vram": _cuda_memory(torch),
        "exceptions": exceptions,
        "elapsed_seconds": _round(time.monotonic() - started, 3),
        "final_decision": "PRIOR_MODULE_TRAINING_ACCEPTED" if accepted else "PRIOR_MODULE_TRAINING_INVALID",
    }
    dataset.close()
    _write_json(Path(args.training_output), report)
    _write_md(Path(args.training_md), "A2C2 Prior Module Training", report)
    _write_json(Path(args.training_status), {"status": "completed", **report})
    torch.cuda.empty_cache()
    return report


def _noise_seed(task_id: int, init_state_id: int, chunk_index: int) -> int:
    payload = f"{BASE_SEED}:{task_id}:{init_state_id}:{chunk_index}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**63 - 1)


def _latest_observation(env: Any, observation: Any, env_preprocessor: Any) -> dict[str, Any]:
    from lerobot.envs.utils import add_envs_task, preprocess_observation

    latest = preprocess_observation(observation)
    latest = add_envs_task(env, latest)
    return env_preprocessor(latest)


def _prior_observation(latest: Mapping[str, Any]) -> tuple[Any, Any, list[str]]:
    import torch

    images = torch.stack(
        (
            latest["observation.images.image"],
            latest["observation.images.image2"],
        ),
        dim=1,
    ).to("cuda", dtype=torch.float32)
    state = latest["observation.state"].to("cuda", dtype=torch.float32)
    tasks = latest.get("task")
    if isinstance(tasks, str):
        tasks = [tasks]
    return images, state, list(tasks)


def _trace_episode(
    *,
    env: Any,
    policy: Any,
    capture: SmolVLAHiddenCapture,
    preprocessor: Any,
    postprocessor: Any,
    env_preprocessor: Any,
    condition: Mapping[str, int],
    task_id: int,
    init_state_id: int,
    prior: Any | None,
) -> dict[str, Any]:
    import torch

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    env.envs[0].init_state_id = int(init_state_id)
    policy.reset()
    observation, _ = env.reset(seed=[BASE_SEED + task_id * 100 + init_state_id])
    execution_horizon = int(condition["execution_horizon"])
    inference_delay = int(condition["inference_delay"])
    action_plan: list[dict[str, Any]] = []
    pending_actions: list[dict[str, Any]] = []
    first_chunk = True
    chunk_index = 0
    base_forward_count = 0
    prior_forward_count = 0
    corrections: list[float] = []
    successes: list[bool] = []
    rewards: list[float] = []
    action_finite = True
    started = time.monotonic()
    step = 0
    while step < MAX_STEPS:
        if not action_plan:
            latest = _latest_observation(env, observation, env_preprocessor)
            batch = preprocessor(dict(latest))
            noise = _make_noise(policy, _noise_seed(task_id, init_state_id, chunk_index), torch)
            with torch.inference_mode():
                raw_chunk = policy.predict_action_chunk(batch, noise=noise)
            chunk, _ = _postprocess_chunk(raw_chunk, postprocessor, 7)
            hidden = capture.pop().detach()
            base_forward_count += 1
            entries = [
                {
                    "action": chunk[index].copy(),
                    "time_offset": index,
                    "chunk": chunk.copy(),
                    "vlm_hidden": hidden.clone(),
                }
                for index in range(CHUNK_SIZE)
            ]
            if first_chunk:
                first_chunk = False
                action_plan = entries[:execution_horizon]
                pending_actions.extend(entries[execution_horizon : execution_horizon + inference_delay])
            else:
                if len(pending_actions) < inference_delay:
                    raise RuntimeError(
                        f"pending queue underflow: have {len(pending_actions)}, need {inference_delay}"
                    )
                action_plan = [pending_actions.pop(0) for _ in range(inference_delay)]
                action_plan.extend(entries[inference_delay:execution_horizon])
                pending_actions.extend(entries[execution_horizon : execution_horizon + inference_delay])
            chunk_index += 1
        entry = action_plan.pop(0)
        action = np.asarray(entry["action"], dtype=np.float32)
        if prior is not None:
            latest = _latest_observation(env, observation, env_preprocessor)
            images, state, tasks = _prior_observation(latest)
            base_tensor = torch.tensor(action, device="cuda", dtype=torch.float32).reshape(1, 7)
            chunk_tensor = torch.tensor(entry["chunk"], device="cuda", dtype=torch.float32).unsqueeze(0)
            time_tensor = phase_feature(int(entry["time_offset"]), CHUNK_SIZE).to("cuda").unsqueeze(0)
            with torch.inference_mode():
                corrected = prior.predict_action(
                    images=images,
                    state=state,
                    base_action=base_tensor,
                    base_chunk=chunk_tensor,
                    time_feature=time_tensor,
                    vlm_hidden=entry["vlm_hidden"].to("cuda"),
                    tasks=tasks,
                )
            corrected_np = corrected[0].detach().cpu().numpy().astype(np.float32)
            corrections.append(float(np.mean(np.abs(corrected_np - action))))
            action = corrected_np
            prior_forward_count += 1
        action_finite = action_finite and bool(np.isfinite(action).all())
        observation, reward, terminated, truncated, info = env.step(action.reshape(1, 7))
        step_success = bool(_successes_from_info(info, 1)[0])
        successes.append(step_success)
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))
        step += 1
        if step_success or bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
            break
    return {
        "success": bool(any(successes)),
        "episode_length": int(step),
        "sum_reward": _round(np.sum(rewards) if rewards else 0.0, 6),
        "max_reward": _round(np.max(rewards) if rewards else 0.0, 6),
        "action_finite": action_finite,
        "base_model_forward_count": base_forward_count,
        "prior_module_forward_count": prior_forward_count,
        "prior_mean_abs_correction": _round(np.mean(corrections), 9) if corrections else 0.0,
        "prior_max_mean_abs_correction": _round(np.max(corrections), 9) if corrections else 0.0,
        "official_init_state_id": int(init_state_id),
        "max_steps": MAX_STEPS,
        "elapsed_seconds": _round(time.monotonic() - started, 3),
        "peak_vram": _cuda_memory(torch),
        "rss_mb": _round(_rss_mb(), 3),
        "exception": None,
    }


def _rollout_mode(args: argparse.Namespace, *, with_prior: bool) -> dict[str, Any]:
    import torch
    from lerobot.envs.factory import make_env

    _runtime_environment(args)
    _set_runtime_env(args)
    _set_seed(BASE_SEED)
    started = time.monotonic()
    before = _enforce_resources(torch)
    if with_prior:
        training = json.loads(Path(args.training_output).read_text(encoding="utf-8"))
        if training.get("final_decision") != "PRIOR_MODULE_TRAINING_ACCEPTED":
            raise RuntimeError("accepted prior module training is required for prior rollout")
        checkpoint_path = Path(training["checkpoints"][-1]["path"])
        prior, checkpoint_payload = _load_prior_checkpoint(checkpoint_path, args, device="cuda")
        prior.eval()
        conditions = {PRIOR_CONDITION: BASE_CONDITIONS["BASE_DELAYED_E40_D10"]}
        output_path = Path(args.prior_rollout_output)
        partial_path = Path(args.prior_rollout_partial)
        md_path = Path(args.prior_rollout_md)
    else:
        prior = None
        checkpoint_path = None
        checkpoint_payload = None
        conditions = BASE_CONDITIONS
        output_path = Path(args.base_rollout_output)
        partial_path = Path(args.base_rollout_partial)
        md_path = Path(args.base_rollout_md)
    loaded = _load_policy_and_processors(args, PolicySpec("frozen_base"))
    policy = loaded["policy"]
    rows = []
    exceptions: list[dict[str, Any]] = []
    if partial_path.exists():
        partial = json.loads(partial_path.read_text(encoding="utf-8"))
        rows = list(partial.get("episodes") or [])
        exceptions = list(partial.get("exceptions") or [])
    completed = {
        (row["condition"], int(row["task_id"]), int(row["official_init_state_id"]))
        for row in rows
    }
    launched = 0
    with SmolVLAHiddenCapture(policy) as capture:
        for condition_name, condition in conditions.items():
            for task in EVAL_TASKS:
                env = None
                try:
                    env_cfg = _make_env_cfg(SUITE, [int(task["task_id"])])
                    env = _extract_single_env(
                        make_env(env_cfg, n_envs=1, use_async_envs=False),
                        SUITE,
                        int(task["task_id"]),
                    )
                    for init_state_id in INIT_STATE_IDS:
                        key = (condition_name, int(task["task_id"]), int(init_state_id))
                        if key in completed:
                            continue
                        row = {
                            "condition": condition_name,
                            "suite": SUITE,
                            "task_id": int(task["task_id"]),
                            "global_task_index": int(task["global_task_index"]),
                            "instruction": task["instruction"],
                            "official_init_state_id": int(init_state_id),
                            "execution_horizon": int(condition["execution_horizon"]),
                            "inference_delay": int(condition["inference_delay"]),
                            "uses_prior": with_prior,
                            "uses_expert_action_at_live_inference": False,
                        }
                        print(
                            f"[a2c2-rollout] {condition_name} task={task['task_id']} init={init_state_id}",
                            flush=True,
                        )
                        try:
                            trace = _trace_episode(
                                env=env,
                                policy=policy,
                                capture=capture,
                                preprocessor=loaded["preprocessor"],
                                postprocessor=loaded["postprocessor"],
                                env_preprocessor=loaded["env_preprocessor"],
                                condition=condition,
                                task_id=int(task["task_id"]),
                                init_state_id=int(init_state_id),
                                prior=prior,
                            )
                            row.update(trace)
                        except Exception as exc:
                            error = {
                                "type": type(exc).__name__,
                                "message": str(exc),
                                "traceback": traceback.format_exc().splitlines()[-30:],
                            }
                            row.update(
                                {
                                    "success": False,
                                    "episode_length": None,
                                    "action_finite": False,
                                    "base_model_forward_count": 0,
                                    "prior_module_forward_count": 0,
                                    "exception": error,
                                }
                            )
                            exceptions.append({"condition": condition_name, "task_id": task["task_id"], "init_state_id": init_state_id, **error})
                        rows.append(row)
                        completed.add(key)
                        launched += 1
                        snapshot = _enforce_resources(torch)
                        partial = {
                            "status": "running",
                            "job_classification": "VLA_CLOSED_LOOP_ROLLOUT",
                            "fidelity_label": FIDELITY_LABEL,
                            "with_prior": with_prior,
                            "episodes": rows,
                            "exceptions": exceptions,
                            "completed_episode_rows": len(rows),
                            "planned_episode_rows": len(conditions) * len(EVAL_TASKS) * len(INIT_STATE_IDS),
                            "resource_snapshot": snapshot,
                            "elapsed_seconds": _round(time.monotonic() - started, 3),
                        }
                        _write_json(partial_path, partial)
                        if args.limit_episodes and launched >= int(args.limit_episodes):
                            break
                    if args.limit_episodes and launched >= int(args.limit_episodes):
                        break
                finally:
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                if args.limit_episodes and launched >= int(args.limit_episodes):
                    break
            if args.limit_episodes and launched >= int(args.limit_episodes):
                break
    planned = len(conditions) * len(EVAL_TASKS) * len(INIT_STATE_IDS)
    complete = len(rows) == planned
    valid = bool(
        complete
        and not exceptions
        and all(row.get("action_finite") for row in rows)
        and all(int(row.get("base_model_forward_count") or 0) > 0 for row in rows)
        and (not with_prior or all(int(row.get("prior_module_forward_count") or 0) > 0 for row in rows))
    )
    summary = {}
    for name in conditions:
        condition_rows = [row for row in rows if row["condition"] == name]
        summary[name] = {
            "episodes": len(condition_rows),
            "successes": sum(bool(row.get("success")) for row in condition_rows),
            "success_rate": _round(np.mean([bool(row.get("success")) for row in condition_rows]), 6) if condition_rows else None,
            "base_model_forward_count": sum(int(row.get("base_model_forward_count") or 0) for row in condition_rows),
            "prior_module_forward_count": sum(int(row.get("prior_module_forward_count") or 0) for row in condition_rows),
            "per_task_successes": {
                str(task["task_id"]): sum(bool(row.get("success")) for row in condition_rows if int(row["task_id"]) == int(task["task_id"]))
                for task in EVAL_TASKS
            },
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "date": f"{DATE_KST} KST",
        "job_classification": "VLA_CLOSED_LOOP_ROLLOUT",
        "fidelity_label": FIDELITY_LABEL,
        "official_commit": OFFICIAL_COMMIT,
        "with_prior": with_prior,
        "ours_executed": False,
        "expert_action_replay_counted_as_success": False,
        "official_reset_states": True,
        "official_init_state_ids": INIT_STATE_IDS,
        "conditions": conditions,
        "tasks": EVAL_TASKS,
        "planned_episode_rows": planned,
        "completed_episode_rows": len(rows),
        "successful_episode_rows": sum(bool(row.get("success")) for row in rows),
        "summary": summary,
        "episodes": rows,
        "exceptions": exceptions,
        "base_policy_load_audit": loaded["audit"],
        "prior_checkpoint": str(checkpoint_path) if checkpoint_path else None,
        "prior_checkpoint_step": int(checkpoint_payload["step"]) if checkpoint_payload else None,
        "before_resources": before,
        "peak_vram": _cuda_memory(torch),
        "rss_mb": _round(_rss_mb(), 3),
        "elapsed_seconds": _round(time.monotonic() - started, 3),
        "final_decision": (
            "A2C2_PRIOR_CLOSED_LOOP_ACCEPTED" if with_prior and valid else
            "A2C2_BASE_CLOSED_LOOP_ACCEPTED" if not with_prior and valid else
            "A2C2_CLOSED_LOOP_INCOMPLETE_RESUME_REQUIRED" if not complete else
            "A2C2_CLOSED_LOOP_INVALID"
        ),
    }
    _write_json(output_path, report)
    _write_md(md_path, "A2C2 Prior Closed Loop" if with_prior else "A2C2 Base Closed Loop", report)
    _write_json(partial_path, {"status": "completed" if complete else "incomplete", **report})
    torch.cuda.empty_cache()
    return report


def _identity(row: Mapping[str, Any]) -> tuple[int, int]:
    return int(row["task_id"]), int(row["official_init_state_id"])


def run_adjudication(args: argparse.Namespace) -> dict[str, Any]:
    base = json.loads(Path(args.base_rollout_output).read_text(encoding="utf-8"))
    prior = json.loads(Path(args.prior_rollout_output).read_text(encoding="utf-8"))
    training = json.loads(Path(args.training_output).read_text(encoding="utf-8"))
    cache = json.loads(Path(args.cache_output).read_text(encoding="utf-8"))
    clean_rows = [row for row in base.get("episodes", []) if row.get("condition") == "BASE_STANDARD_E10_D0"]
    delayed_rows = [row for row in base.get("episodes", []) if row.get("condition") == "BASE_DELAYED_E40_D10"]
    prior_rows = list(prior.get("episodes", []))
    clean = {_identity(row): bool(row.get("success")) for row in clean_rows}
    delayed = {_identity(row): bool(row.get("success")) for row in delayed_rows}
    corrected = {_identity(row): bool(row.get("success")) for row in prior_rows}
    expected = {(task["task_id"], init_id) for task in EVAL_TASKS for init_id in INIT_STATE_IDS}
    manifest_valid = set(clean) == expected and set(delayed) == expected and set(corrected) == expected
    infrastructure_valid = bool(
        base.get("final_decision") == "A2C2_BASE_CLOSED_LOOP_ACCEPTED"
        and prior.get("final_decision") == "A2C2_PRIOR_CLOSED_LOOP_ACCEPTED"
        and training.get("final_decision") == "PRIOR_MODULE_TRAINING_ACCEPTED"
        and cache.get("final_decision") == "A2C2_CACHED_FEATURES_ACCEPTED"
    )
    clean_successes = sum(clean.values())
    delayed_successes = sum(delayed.values())
    prior_successes = sum(corrected.values())
    clean_to_delayed_failures = [key for key in expected if clean.get(key) and not delayed.get(key)]
    recovered = [key for key in expected if not delayed.get(key) and corrected.get(key)]
    prior_regressions = [key for key in expected if delayed.get(key) and not corrected.get(key)]
    clean_to_prior_residual = [key for key in expected if clean.get(key) and not corrected.get(key)]
    repeated_failure_tasks = sorted({key[0] for key in clean_to_delayed_failures})
    residual_tasks = sorted({key[0] for key in clean_to_prior_residual})
    base_competent = bool(
        clean_successes >= 8
        and all(sum(value for (task_id, _), value in clean.items() if task_id == task["task_id"]) >= 1 for task in EVAL_TASKS)
    )
    repeatable_problem = bool(
        clean_successes - delayed_successes >= 3
        and len(clean_to_delayed_failures) >= 3
        and len(repeated_failure_tasks) >= 2
    )
    prior_improves = bool(
        prior_successes - delayed_successes >= 2
        and len(recovered) >= 2
        and len(prior_regressions) <= 1
        and sum(int(row.get("prior_module_forward_count") or 0) for row in prior_rows) > 0
        and any(float(row.get("prior_mean_abs_correction") or 0.0) > 1e-6 for row in prior_rows)
    )
    saturates = bool(prior_improves and (prior_successes >= clean_successes - 1 or len(clean_to_prior_residual) <= 1))
    residual = bool(
        prior_improves
        and clean_successes - prior_successes >= 2
        and len(clean_to_prior_residual) >= 2
        and len(residual_tasks) >= 2
    )
    if not manifest_valid:
        decision = "EVALUATION_INVALID"
    elif not infrastructure_valid:
        decision = "PRIOR_INFRASTRUCTURE_BLOCKED"
    elif not base_competent:
        decision = "BASE_NOT_COMPETENT"
    elif not repeatable_problem:
        decision = "NO_REPEATABLE_PROBLEM"
    elif not prior_improves:
        decision = "NO_DIAGNOSTIC_HEADROOM"
    elif saturates:
        decision = "PRIOR_SATURATES_PROBLEM"
    elif residual:
        decision = "VERIFIED_PRIOR_RESIDUAL"
    else:
        decision = "NO_DIAGNOSTIC_HEADROOM"
    report = {
        "schema_version": SCHEMA_VERSION,
        "date": f"{DATE_KST} KST",
        "job_classification": "REPORT_ONLY",
        "fidelity_label": FIDELITY_LABEL,
        "official_commit": OFFICIAL_COMMIT,
        "ours_designed_or_executed": False,
        "frozen_panel": {"suite": SUITE, "tasks": EVAL_TASKS, "official_init_state_ids": INIT_STATE_IDS},
        "conditions": {**BASE_CONDITIONS, PRIOR_CONDITION: BASE_CONDITIONS["BASE_DELAYED_E40_D10"]},
        "thresholds": {
            "base_competence": "clean successes >=8/15 and >=1 success on every task",
            "repeatable_problem": "clean-delayed >=3/15, >=3 matched clean-to-delayed failures, spanning >=2 tasks",
            "prior_improvement": "prior-delayed >=2/15, >=2 delayed failures recovered, <=1 delayed success regressed, nonzero live prior forwards and correction",
            "prior_saturation": "prior improves and prior >= clean-1 or <=1 matched clean-success/prior-failure remains",
            "residual": "prior improves and clean-prior >=2/15 with >=2 matched residuals spanning >=2 tasks",
        },
        "validity": {"manifest_valid": manifest_valid, "infrastructure_valid": infrastructure_valid},
        "counts": {
            "clean_successes": clean_successes,
            "delayed_base_successes": delayed_successes,
            "delayed_prior_successes": prior_successes,
            "clean_to_delayed_failure_count": len(clean_to_delayed_failures),
            "prior_recovery_count": len(recovered),
            "prior_regression_count": len(prior_regressions),
            "clean_to_prior_residual_count": len(clean_to_prior_residual),
        },
        "identity_lists": {
            "clean_to_delayed_failures": sorted(clean_to_delayed_failures),
            "prior_recoveries": sorted(recovered),
            "prior_regressions": sorted(prior_regressions),
            "clean_to_prior_residuals": sorted(clean_to_prior_residual),
        },
        "gates": {
            "base_competent": base_competent,
            "repeatable_problem": repeatable_problem,
            "prior_improves": prior_improves,
            "prior_saturates": saturates,
            "residual_remains": residual,
            "diagnostic_or_task_headroom": residual,
            "residual_is_not_infrastructure_artifact": infrastructure_valid,
            "residual_is_not_single_accidental_episode": len(clean_to_prior_residual) >= 2 and len(residual_tasks) >= 2,
        },
        "source_reports": {
            "cache": str(args.cache_output),
            "training": str(args.training_output),
            "base_rollout": str(args.base_rollout_output),
            "prior_rollout": str(args.prior_rollout_output),
        },
        "final_decision": decision,
        "next_step": (
            "Generate at most two Ours candidates around the verified residual."
            if decision == "VERIFIED_PRIOR_RESIDUAL"
            else "Do not design or execute Ours for this thesis; follow the frozen pivot-closure rule."
        ),
    }
    _write_json(Path(args.adjudication_output), report)
    _write_md(Path(args.adjudication_md), "A2C2 Official-Prior-First Problem Verification", report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["preflight", "cache", "train", "rollout-base", "rollout-prior", "adjudicate"],
        required=True,
    )
    parser.add_argument("--base-path", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--dataset-root", default="/mnt/c/assets/datasets/lerobot_libero")
    parser.add_argument("--hf-home", default="/mnt/c/assets/hf_home")
    parser.add_argument("--vlm-root", default="/mnt/c/assets/hf_home/HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--lora-root", default="/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--microbatch", type=int, choices=[1, 2, 4, 8], default=4)
    parser.add_argument("--anchor-stride", type=int, default=8)
    parser.add_argument("--training-steps", type=int, default=40000)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--save-every", type=int, default=20000)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--limit-episodes", type=int, default=0)
    parser.add_argument("--preflight-output", default="reports/a2c2_prior/preflight_result.json")
    parser.add_argument("--preflight-md", default="reports/a2c2_prior/preflight_result.md")
    parser.add_argument("--cache-path", default="runs/a2c2_prior/a2c2_cached_features.h5")
    parser.add_argument("--cache-status", default="runs/a2c2_prior/cache_status.json")
    parser.add_argument("--cache-output", default="reports/a2c2_prior/cached_feature_result.json")
    parser.add_argument("--cache-md", default="reports/a2c2_prior/cached_feature_result.md")
    parser.add_argument("--checkpoint-dir", default="runs/a2c2_prior/checkpoints")
    parser.add_argument("--training-status", default="runs/a2c2_prior/training_status.json")
    parser.add_argument("--training-output", default="reports/a2c2_prior/prior_module_training_result.json")
    parser.add_argument("--training-md", default="reports/a2c2_prior/prior_module_training_result.md")
    parser.add_argument("--base-rollout-partial", default="runs/a2c2_prior/base_rollout_partial.json")
    parser.add_argument("--base-rollout-output", default="reports/a2c2_prior/base_closed_loop_result.json")
    parser.add_argument("--base-rollout-md", default="reports/a2c2_prior/base_closed_loop_result.md")
    parser.add_argument("--prior-rollout-partial", default="runs/a2c2_prior/prior_rollout_partial.json")
    parser.add_argument("--prior-rollout-output", default="reports/a2c2_prior/prior_closed_loop_result.json")
    parser.add_argument("--prior-rollout-md", default="reports/a2c2_prior/prior_closed_loop_result.md")
    parser.add_argument("--adjudication-output", default="reports/a2c2_prior/problem_verification_result.json")
    parser.add_argument("--adjudication-md", default="reports/a2c2_prior/problem_verification_result.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "preflight":
        report = run_preflight(args)
    elif args.mode == "cache":
        report = run_cache(args)
    elif args.mode == "train":
        report = run_training(args)
    elif args.mode == "rollout-base":
        report = _rollout_mode(args, with_prior=False)
    elif args.mode == "rollout-prior":
        report = _rollout_mode(args, with_prior=True)
    else:
        report = run_adjudication(args)
    print(json.dumps({"mode": args.mode, "final_decision": report.get("final_decision")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
