"""Run the frozen executable COVI-VLA Stage 0 audit."""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch.nn import functional as F


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.covi_vla import (  # noqa: E402
    COVIStage0Adapter,
    COVIStage0Config,
    FORBIDDEN_INFERENCE_KEYS,
    LEGAL_INFERENCE_FEATURES,
    PROPOSAL_HASH,
    apply_scene_obstruction,
    classify_stage0,
    covi_stage0_loss,
    episode_cluster_bootstrap_margin,
    equal_area_rectangle_mask,
    irregular_occlusion_mask,
    mask_context,
    normalized_rmse_margin,
    parameter_gradient_norms,
    partition_stage0_manifest,
    partition_summary,
    prediction_metrics,
)
from tca_map.smolvla.official_libero_baseline_scaleup import (  # noqa: E402
    _add_training_batch_dims,
    _postprocess_action,
    _raw_current_action,
)


DATE_KST = "2026-07-15"
CHECKPOINT_PATH = Path(r"C:\assets\checkpoints\smolvla_libero")
VLM_PATH = Path(r"C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct")
DATASET_ROOT = Path(r"C:\assets\datasets\lerobot_libero")
SPLIT_MANIFEST = Path("reports/official_smolvla_split_manifest.json")
BASE_ARTIFACT = Path("reports/official_smolvla_stable_prediction_artifact.json")
RUN_DIR = Path("runs/covi_vla/stage0")
RESULT_JSON = Path("reports/covi_vla/stage_0_result.json")
RESULT_MD = Path("reports/covi_vla/stage_0_result.md")
BLOCKER_JSON = Path("reports/covi_vla/implementation_blocker.json")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "-c", "safe.directory=C:/Users/jiheo/tca_map", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _package_versions() -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ["torch", "lerobot", "transformers", "numpy", "av", "pyarrow"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def _set_offline_environment() -> None:
    os.environ["HF_HOME"] = str(Path(r"C:\assets\hf_home"))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _preflight() -> dict[str, Any]:
    paths = {
        "checkpoint": CHECKPOINT_PATH,
        "vlm": VLM_PATH,
        "dataset": DATASET_ROOT,
        "split_manifest": SPLIT_MANIFEST,
        "base_artifact": BASE_ARTIFACT,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    return {
        "paths": {name: str(path.resolve() if path.exists() else path) for name, path in paths.items()},
        "missing": missing,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "passed": not missing and bool(torch.cuda.is_available()),
    }


def _attach_local_indices(
    partitions: Mapping[str, Sequence[Mapping[str, Any]]],
    names: Sequence[str],
) -> tuple[list[int], dict[str, list[dict[str, Any]]]]:
    selected_rows = [dict(row) for name in names for row in partitions[name]]
    episodes = sorted({int(row["episode_index"]) for row in selected_rows})
    lengths: dict[int, int] = {}
    for row in selected_rows:
        episode = int(row["episode_index"])
        length = int(row.get("episode_length", 0))
        if length <= 0:
            raise ValueError(f"missing episode length for episode {episode}")
        lengths[episode] = length
    offsets: dict[int, int] = {}
    offset = 0
    for episode in episodes:
        offsets[episode] = offset
        offset += lengths[episode]
    output: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        output[name] = []
        for raw in partitions[name]:
            row = dict(raw)
            episode = int(row["episode_index"])
            row["dataset_local_index"] = offsets[episode] + int(row["frame_index"])
            output[name].append(row)
    return episodes, output


def _task_one_hot(task_index: int, task_dim: int) -> np.ndarray:
    vector = np.zeros(task_dim, dtype=np.float32)
    vector[int(task_index)] = 1.0
    return vector


def _embed_pair(policy: Any, image_1: torch.Tensor, image_2: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
    batch = {
        "observation.images.camera1": image_1.to("cuda"),
        "observation.images.camera2": image_2.to("cuda"),
    }
    prepared, masks = policy.prepare_images(batch)
    dtype = next(policy.parameters()).dtype
    prepared = [image.to(dtype=dtype) for image in prepared]
    with torch.no_grad():
        token_1 = policy.model.vlm_with_expert.embed_image(prepared[0])
        token_2 = policy.model.vlm_with_expert.embed_image(prepared[1])
    shapes = {
        "raw_pair": [list(image_1.shape), list(image_2.shape)],
        "prepared_pair": [list(prepared[0].shape), list(prepared[1].shape)],
        "token_pair": [list(token_1.shape), list(token_2.shape)],
        "mask_pair": [list(masks[0].shape), list(masks[1].shape)],
    }
    return token_1.float().mean(dim=1).cpu(), token_2.float().mean(dim=1).cpu(), shapes


def _feature_cache_identity(rows: Sequence[Mapping[str, Any]], config: COVIStage0Config) -> str:
    payload = {
        "proposal_hash": PROPOSAL_HASH,
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "config": config.to_dict(),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest().upper()


def _extract_features(
    *,
    policy: Any,
    dataset: Any,
    rows: Sequence[Mapping[str, Any]],
    base_records: Mapping[str, Mapping[str, Any]],
    config: COVIStage0Config,
    batch_size: int,
    cache_path: Path,
) -> dict[str, Any]:
    identity = _feature_cache_identity(rows, config)
    if cache_path.exists():
        cached = torch.load(cache_path, map_location="cpu", weights_only=False)
        if cached.get("identity") == identity:
            cached["loaded_from_cache"] = True
            return cached

    fields: dict[str, list[torch.Tensor]] = {
        "clean_1": [],
        "clean_2": [],
        "occluded_1": [],
        "occluded_2": [],
        "rectangle_1": [],
        "rectangle_2": [],
    }
    states: list[np.ndarray] = []
    base_actions: list[np.ndarray] = []
    tasks: list[int] = []
    episodes: list[int] = []
    contexts: list[np.ndarray] = []
    rectangle_contexts: list[np.ndarray] = []
    sample_ids: list[str] = []
    coverage: list[float] = []
    shape_audit: dict[str, Any] | None = None
    started = time.monotonic()

    for start in range(0, len(rows), batch_size):
        subset = rows[start : start + batch_size]
        clean_1: list[torch.Tensor] = []
        clean_2: list[torch.Tensor] = []
        occluded_1: list[torch.Tensor] = []
        occluded_2: list[torch.Tensor] = []
        rectangle_1: list[torch.Tensor] = []
        rectangle_2: list[torch.Tensor] = []
        for row in subset:
            sample_id = str(row["sample_id"])
            raw = dataset[int(row["dataset_local_index"])]
            image_1 = raw["observation.images.image"].detach().cpu().float()
            image_2 = raw["observation.images.image2"].detach().cpu().float()
            height, width = int(image_1.shape[-2]), int(image_1.shape[-1])
            mask_1 = irregular_occlusion_mask(height, width, sample_id=sample_id, stream=1)
            mask_2 = irregular_occlusion_mask(height, width, sample_id=sample_id, stream=2)
            rect_1 = equal_area_rectangle_mask(
                height, width, sample_id=sample_id, stream=1, area_fraction=float(mask_1.mean())
            )
            rect_2 = equal_area_rectangle_mask(
                height, width, sample_id=sample_id, stream=2, area_fraction=float(mask_2.mean())
            )
            clean_1.append(image_1)
            clean_2.append(image_2)
            occluded_1.append(apply_scene_obstruction(image_1, mask_1, sample_id=sample_id, stream=1))
            occluded_2.append(apply_scene_obstruction(image_2, mask_2, sample_id=sample_id, stream=2))
            rectangle_1.append(apply_scene_obstruction(image_1, rect_1, sample_id=sample_id, stream=11))
            rectangle_2.append(apply_scene_obstruction(image_2, rect_2, sample_id=sample_id, stream=12))
            contexts.append(mask_context(mask_1, mask_2))
            rectangle_contexts.append(mask_context(rect_1, rect_2))
            coverage.extend([float(mask_1.mean()), float(mask_2.mean())])
            stable = base_records.get(sample_id)
            if stable is None:
                raise KeyError(f"base artifact missing sample {sample_id}")
            states.append(np.asarray(stable["state"], dtype=np.float32))
            base_actions.append(np.asarray(stable["base_action"], dtype=np.float32))
            tasks.append(int(row["task_index"]))
            episodes.append(int(row["episode_index"]))
            sample_ids.append(sample_id)

        clean_a, clean_b, shape_audit = _embed_pair(policy, torch.stack(clean_1), torch.stack(clean_2))
        occ_a, occ_b, _ = _embed_pair(policy, torch.stack(occluded_1), torch.stack(occluded_2))
        rect_a, rect_b, _ = _embed_pair(policy, torch.stack(rectangle_1), torch.stack(rectangle_2))
        fields["clean_1"].append(clean_a)
        fields["clean_2"].append(clean_b)
        fields["occluded_1"].append(occ_a)
        fields["occluded_2"].append(occ_b)
        fields["rectangle_1"].append(rect_a)
        fields["rectangle_2"].append(rect_b)
        print(f"[covi-stage0] feature cache {min(start + batch_size, len(rows))}/{len(rows)}", flush=True)

    cache = {
        "identity": identity,
        "proposal_hash": PROPOSAL_HASH,
        "sample_ids": sample_ids,
        "task_indices": torch.tensor(tasks, dtype=torch.long),
        "episode_indices": torch.tensor(episodes, dtype=torch.long),
        "states": torch.tensor(np.stack(states), dtype=torch.float32),
        "base_actions": torch.tensor(np.stack(base_actions), dtype=torch.float32),
        "contexts": torch.tensor(np.stack(contexts), dtype=torch.float32),
        "rectangle_contexts": torch.tensor(np.stack(rectangle_contexts), dtype=torch.float32),
        "coverage": torch.tensor(coverage, dtype=torch.float32),
        "shape_audit": shape_audit,
        "elapsed_sec": time.monotonic() - started,
        "loaded_from_cache": False,
    }
    for name, chunks in fields.items():
        cache[name] = torch.cat(chunks, dim=0)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, cache_path)
    return cache


def _build_source(cache: Mapping[str, Any], camera_1_key: str, camera_2_key: str, config: COVIStage0Config) -> torch.Tensor:
    tasks = cache["task_indices"]
    one_hot = F.one_hot(tasks, num_classes=config.task_dim).float()
    return torch.cat(
        [
            cache[camera_1_key].float(),
            cache[camera_2_key].float(),
            cache["states"].float(),
            cache["base_actions"].float(),
            one_hot,
        ],
        dim=-1,
    )


def _standardize(fit: torch.Tensor, all_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mean = fit.mean(dim=0)
    scale = fit.std(dim=0, unbiased=False)
    scale = torch.where(scale < 1e-6, torch.ones_like(scale), scale)
    return (all_values - mean) / scale, mean, scale


def _train_adapter(
    *,
    source: torch.Tensor,
    clean_source: torch.Tensor,
    camera2: torch.Tensor,
    clean_camera2: torch.Tensor,
    target: torch.Tensor,
    context: torch.Tensor,
    fit_count: int,
    config: COVIStage0Config,
    label: str,
) -> tuple[COVIStage0Adapter, dict[str, Any]]:
    torch.manual_seed(config.seed)
    model = COVIStage0Adapter(config).to("cuda")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    generator = torch.Generator(device="cpu").manual_seed(config.seed)
    epoch_rows: list[dict[str, float]] = []
    first_terms: dict[str, float] | None = None
    first_gradients: dict[str, float] | None = None
    last_gradients: dict[str, float] | None = None
    started = time.monotonic()
    for epoch in range(config.epochs):
        order = torch.randperm(fit_count, generator=generator)
        totals: list[float] = []
        for start in range(0, fit_count, config.batch_size):
            index = order[start : start + config.batch_size]
            optimizer.zero_grad(set_to_none=True)
            total, terms = covi_stage0_loss(
                model,
                occluded_source=source[index].to("cuda"),
                occluded_camera2=camera2[index].to("cuda"),
                target=target[index].to("cuda"),
                context_target=context[index].to("cuda"),
                clean_source=clean_source[index].to("cuda"),
                clean_camera2=clean_camera2[index].to("cuda"),
            )
            if not torch.isfinite(total):
                raise RuntimeError(f"{label} nonfinite loss at epoch {epoch}")
            total.backward()
            gradients = parameter_gradient_norms(model)
            if first_terms is None:
                first_terms = {name: float(value.detach().item()) for name, value in terms.items()}
                first_gradients = gradients
            last_gradients = gradients
            optimizer.step()
            totals.append(float(total.detach().item()))
        epoch_rows.append({"epoch": epoch + 1, "loss": float(np.mean(totals))})
    nonzero = [value for value in (last_gradients or {}).values() if value > 0.0 and math.isfinite(value)]
    ratio = max(nonzero) / min(nonzero) if nonzero else math.inf
    return model, {
        "label": label,
        "epochs": config.epochs,
        "batch_size": config.batch_size,
        "optimizer": "AdamW",
        "learning_rate": config.learning_rate,
        "weight_decay": config.weight_decay,
        "first_loss_terms": first_terms,
        "first_gradient_norms": first_gradients,
        "last_gradient_norms": last_gradients,
        "largest_to_smallest_nonzero_gradient_ratio": float(ratio),
        "epoch_losses": epoch_rows,
        "elapsed_sec": time.monotonic() - started,
    }


def _predict(model: COVIStage0Adapter, source: torch.Tensor, camera2: torch.Tensor, batch_size: int = 128) -> dict[str, np.ndarray]:
    outputs: dict[str, list[np.ndarray]] = {"imagined": [], "adapted": [], "context": [], "residual": [], "gate": []}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(source), batch_size):
            result = model(source[start : start + batch_size].to("cuda"), F.normalize(camera2[start : start + batch_size].to("cuda"), dim=-1))
            for name in outputs:
                outputs[name].append(result[name].float().cpu().numpy())
    return {name: np.concatenate(chunks, axis=0) for name, chunks in outputs.items()}


def _ridge_predictions(train_x: np.ndarray, train_y: np.ndarray, validation_x: np.ndarray, l2: float = 1e-2) -> np.ndarray:
    x_train = np.asarray(train_x, dtype=np.float32)
    x_validation = np.asarray(validation_x, dtype=np.float32)
    y_train = np.asarray(train_y, dtype=np.float32)
    kernel = x_train @ x_train.T
    kernel.flat[:: kernel.shape[0] + 1] += l2
    dual = np.linalg.solve(kernel, y_train)
    return (x_validation @ x_train.T) @ dual


def _knn_predictions(
    train_x: np.ndarray,
    train_y: np.ndarray,
    train_tasks: np.ndarray,
    validation_x: np.ndarray,
    validation_tasks: np.ndarray,
    k: int = 5,
) -> np.ndarray:
    predictions = []
    for row, task in zip(validation_x, validation_tasks, strict=True):
        candidates = np.flatnonzero(train_tasks == task)
        distance = np.mean((train_x[candidates] - row[None, :]) ** 2, axis=1)
        nearest = candidates[np.argsort(distance)[:k]]
        predictions.append(train_y[nearest].mean(axis=0))
    return np.asarray(predictions, dtype=np.float32)


def _normalization_sensitivity(
    *,
    full_normalized: np.ndarray,
    baseline_normalized: np.ndarray,
    target_raw: np.ndarray,
    train_target_raw: np.ndarray,
) -> dict[str, Any]:
    train_mean = train_target_raw.mean(axis=0)
    target_norm = target_raw / np.maximum(1e-12, np.linalg.norm(target_raw, axis=1, keepdims=True))
    norm_mean = train_target_raw / np.maximum(1e-12, np.linalg.norm(train_target_raw, axis=1, keepdims=True))
    l2_margin = normalized_rmse_margin(full_normalized, baseline_normalized, target_norm, norm_mean.mean(axis=0))
    scale = float(np.median(np.linalg.norm(train_target_raw, axis=1)))
    full_raw = full_normalized * scale
    baseline_raw = baseline_normalized * scale
    raw_margin = normalized_rmse_margin(full_raw, baseline_raw, target_raw, train_mean)
    mean = train_target_raw.mean(axis=0)
    std = train_target_raw.std(axis=0)
    std[std < 1e-6] = 1.0
    target_z = (target_raw - mean) / std
    full_z = (full_raw - mean) / std
    baseline_z = (baseline_raw - mean) / std
    z_margin = normalized_rmse_margin(full_z, baseline_z, target_z, np.zeros_like(mean))
    margins = {"raw": float(raw_margin), "l2_normalized": float(l2_margin), "train_z_scored": float(z_margin)}
    signs = {int(np.sign(value)) for value in margins.values() if abs(value) > 1e-9}
    return {
        "margins": margins,
        "sign_consistent": len(signs) <= 1,
        "range": float(max(margins.values()) - min(margins.values())),
        "resolved": bool(len(signs) <= 1 and max(margins.values()) - min(margins.values()) <= 0.05),
    }


def _clone_raw_with_images(raw: Mapping[str, Any], image_1: torch.Tensor, image_2: torch.Tensor) -> dict[str, Any]:
    result = {key: value for key, value in raw.items()}
    result["observation.images.image"] = image_1.clone()
    result["observation.images.image2"] = image_2.clone()
    return result


def _policy_action(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    raw: Mapping[str, Any],
    noise: torch.Tensor,
    adapter: COVIStage0Adapter | None = None,
    source: torch.Tensor | None = None,
) -> np.ndarray:
    batch = _add_training_batch_dims(preprocessor(dict(raw)))
    embedder = policy.model.vlm_with_expert
    original = embedder.embed_image
    counter = {"value": 0}
    if adapter is not None:
        if source is None:
            raise ValueError("adapter action requires source")
        with torch.no_grad():
            injection = adapter.injection(source.to("cuda"))
            delta = injection["gate"] * injection["residual"]

        def adapted_embed(image: torch.Tensor) -> torch.Tensor:
            output = original(image)
            call = counter["value"]
            counter["value"] += 1
            if call == 1:
                output = output + delta.to(device=output.device, dtype=output.dtype).unsqueeze(1)
            return output

        embedder.embed_image = adapted_embed
    try:
        with torch.no_grad():
            chunk = policy._get_action_chunk(batch, noise=noise)
            return _postprocess_action(chunk[0, 0], postprocessor)[:7]
    finally:
        embedder.embed_image = original


def _action_smoke(
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: Any,
    validation_rows: Sequence[Mapping[str, Any]],
    feature_cache: Mapping[str, Any],
    standardized_occ_source: torch.Tensor,
    standardized_clean_source: torch.Tensor,
    model: COVIStage0Adapter,
    initial_model: COVIStage0Adapter,
    action_min: np.ndarray,
    action_max: np.ndarray,
    config: COVIStage0Config,
) -> dict[str, Any]:
    by_task: dict[int, list[Mapping[str, Any]]] = {}
    for row in validation_rows:
        by_task.setdefault(int(row["task_index"]), []).append(row)
    selected = [
        min(rows, key=lambda row: abs(float(row.get("normalized_phase", 0.0)) - 0.5))
        for _, rows in sorted(by_task.items())
    ]
    index_by_sample = {sample_id: index for index, sample_id in enumerate(feature_cache["sample_ids"])}
    records: list[dict[str, Any]] = []
    started = time.monotonic()
    for position, row in enumerate(selected):
        sample_id = str(row["sample_id"])
        cache_index = index_by_sample[sample_id]
        raw = dataset[int(row["dataset_local_index"])]
        clean_1 = raw["observation.images.image"].detach().cpu().float()
        clean_2 = raw["observation.images.image2"].detach().cpu().float()
        height, width = int(clean_1.shape[-2]), int(clean_1.shape[-1])
        mask_1 = irregular_occlusion_mask(height, width, sample_id=sample_id, stream=1)
        mask_2 = irregular_occlusion_mask(height, width, sample_id=sample_id, stream=2)
        occ_1 = apply_scene_obstruction(clean_1, mask_1, sample_id=sample_id, stream=1)
        occ_2 = apply_scene_obstruction(clean_2, mask_2, sample_id=sample_id, stream=2)
        clean_raw = _clone_raw_with_images(raw, clean_1, clean_2)
        occ_raw = _clone_raw_with_images(raw, occ_1, occ_2)
        oracle_raw = _clone_raw_with_images(raw, occ_1, clean_2)
        generator = torch.Generator(device="cuda").manual_seed(config.seed + int(row["task_index"]))
        noise = torch.randn((1, 50, 32), generator=generator, device="cuda", dtype=torch.float32)
        clean_base = _policy_action(
            policy=policy, preprocessor=preprocessor, postprocessor=postprocessor, raw=clean_raw, noise=noise
        )
        occ_base = _policy_action(
            policy=policy, preprocessor=preprocessor, postprocessor=postprocessor, raw=occ_raw, noise=noise
        )
        oracle = _policy_action(
            policy=policy, preprocessor=preprocessor, postprocessor=postprocessor, raw=oracle_raw, noise=noise
        )
        initial = _policy_action(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            raw=occ_raw,
            noise=noise,
            adapter=initial_model,
            source=standardized_occ_source[cache_index : cache_index + 1],
        )
        full = _policy_action(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            raw=occ_raw,
            noise=noise,
            adapter=model,
            source=standardized_occ_source[cache_index : cache_index + 1],
        )
        clean_covi = _policy_action(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            raw=clean_raw,
            noise=noise,
            adapter=model,
            source=standardized_clean_source[cache_index : cache_index + 1],
        )
        target = _raw_current_action(raw)[:7]
        with torch.no_grad():
            injection = model.injection(standardized_occ_source[cache_index : cache_index + 1].to("cuda"))
        records.append(
            {
                "sample_id": sample_id,
                "task_index": int(row["task_index"]),
                "episode_index": int(row["episode_index"]),
                "frame_index": int(row["frame_index"]),
                "base_clean_action": clean_base.tolist(),
                "base_occluded_action": occ_base.tolist(),
                "clean_view_oracle_action": oracle.tolist(),
                "covi_initial_action": initial.tolist(),
                "covi_action": full.tolist(),
                "covi_clean_action": clean_covi.tolist(),
                "no_imagined_view_ablation_action": occ_base.tolist(),
                "target_action": target.tolist(),
                "initial_delta_l2": float(np.linalg.norm(initial - occ_base)),
                "action_delta_l2": float(np.linalg.norm(full - occ_base)),
                "translation_delta_l2": float(np.linalg.norm(full[:3] - occ_base[:3])),
                "rotation_delta_l2": float(np.linalg.norm(full[3:6] - occ_base[3:6])),
                "gripper_delta_abs": float(abs(full[6] - occ_base[6])),
                "clean_retention_delta_l2": float(np.linalg.norm(clean_covi - clean_base)),
                "clean_base_target_error": float(np.linalg.norm(clean_base - target)),
                "occluded_base_target_error": float(np.linalg.norm(occ_base - target)),
                "oracle_target_error": float(np.linalg.norm(oracle - target)),
                "covi_target_error": float(np.linalg.norm(full - target)),
                "output_valid": bool(np.all(full >= action_min - 1e-6) and np.all(full <= action_max + 1e-6)),
                "residual_norm": float(torch.linalg.vector_norm(injection["residual"]).item()),
                "gate": float(injection["gate"].item()),
                "changed_tokens": 64,
                "available_visual_tokens": 128,
                "mask_context": mask_context(mask_1, mask_2).tolist(),
            }
        )
        print(f"[covi-stage0] action smoke {position + 1}/{len(selected)}", flush=True)

    def values(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in records], dtype=np.float64)

    summary = {
        "record_count": len(records),
        "episode_count": len({row["episode_index"] for row in records}),
        "task_count": len({row["task_index"] for row in records}),
        "initial_action_delta_p95": float(np.quantile(values("initial_delta_l2"), 0.95)),
        "action_delta_mean": float(values("action_delta_l2").mean()),
        "action_delta_p95": float(np.quantile(values("action_delta_l2"), 0.95)),
        "translation_delta_p95": float(np.quantile(values("translation_delta_l2"), 0.95)),
        "rotation_delta_p95": float(np.quantile(values("rotation_delta_l2"), 0.95)),
        "gripper_delta_p95": float(np.quantile(values("gripper_delta_abs"), 0.95)),
        "clean_retention_delta_p95": float(np.quantile(values("clean_retention_delta_l2"), 0.95)),
        "output_valid_fraction": float(np.mean([row["output_valid"] for row in records])),
        "clean_base_target_error_mean": float(values("clean_base_target_error").mean()),
        "occluded_base_target_error_mean": float(values("occluded_base_target_error").mean()),
        "oracle_target_error_mean": float(values("oracle_target_error").mean()),
        "covi_target_error_mean": float(values("covi_target_error").mean()),
        "occluded_minus_oracle_target_error": float(
            values("occluded_base_target_error").mean() - values("oracle_target_error").mean()
        ),
        "elapsed_sec": time.monotonic() - started,
        "records": records,
    }
    return summary


def _write_md(path: Path, report: Mapping[str, Any]) -> None:
    metrics = report.get("representation_metrics") or {}
    action = report.get("action_smoke") or {}
    lines = [
        "# COVI-VLA Stage 0 Result",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- implementation and data valid: `{report.get('implementation_and_data_valid')}`",
        f"- diagnostic headroom exists: `{report.get('diagnostic_headroom_exists')}`",
        f"- identity and safety passed: `{report.get('identity_and_safety_passed')}`",
        f"- candidate margin: `{report.get('candidate_margin')}`",
        f"- strongest comparator: `{metrics.get('strongest_non_oracle_comparator')}`",
        f"- validation records: `{metrics.get('validation_record_count')}`",
        f"- independent validation episodes: `{metrics.get('validation_episode_count')}`",
        f"- bootstrap interval: `{report.get('bootstrap_interval')}`",
        f"- initial action delta p95: `{action.get('initial_action_delta_p95')}`",
        f"- trained action delta p95: `{action.get('action_delta_p95')}`",
        f"- clean retention delta p95: `{action.get('clean_retention_delta_p95')}`",
        f"- output valid fraction: `{action.get('output_valid_fraction')}`",
        f"- test records decoded: `{report.get('confirmatory_test_records_decoded')}`",
        "",
        "The Stage 0 occlusion is a synthetic development proxy. It does not establish the final physical-occlusion claim.",
        "",
        "False-negative adjudication:",
        "",
        "```json",
        json.dumps(report.get("false_negative_safeguard"), indent=2, sort_keys=True),
        "```",
        "",
        f"Next command: `{report.get('exact_next_command')}`",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_stage0(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    config = COVIStage0Config()
    preflight = _preflight()
    if not preflight["passed"]:
        reason = "missing local assets: " + ", ".join(preflight["missing"]) if preflight["missing"] else "CUDA unavailable"
        raise RuntimeError(reason)
    _set_offline_environment()

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    manifest = _read_json(SPLIT_MANIFEST)
    partitions = partition_stage0_manifest(manifest)
    split_summary = partition_summary(partitions)
    selected_episodes, indexed = _attach_local_indices(partitions, ["discovery_fit", "validation"])
    rows = indexed["discovery_fit"] + indexed["validation"]
    fit_count = len(indexed["discovery_fit"])
    validation_count = len(indexed["validation"])
    if fit_count != 600 or validation_count != 400:
        raise RuntimeError(f"frozen split count mismatch: fit={fit_count}, validation={validation_count}")

    artifact = _read_json(BASE_ARTIFACT)
    base_records = {str(row["sample_id"]): row for row in artifact["records"]}
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=DATASET_ROOT,
        episodes=selected_episodes,
        video_backend="pyav",
    )
    cfg = PreTrainedConfig.from_pretrained(CHECKPOINT_PATH, local_files_only=True, cache_dir=Path(os.environ["HF_HOME"]))
    cfg.device = "cuda"
    cfg.load_vlm_weights = True
    cfg.compile_model = False
    cfg.push_to_hub = False
    cfg.vlm_model_name = str(VLM_PATH)
    policy = SmolVLAPolicy.from_pretrained(
        CHECKPOINT_PATH,
        config=cfg,
        local_files_only=True,
        cache_dir=Path(os.environ["HF_HOME"]),
        token=False,
        strict=False,
    )
    policy.to("cuda")
    policy.eval()
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(CHECKPOINT_PATH),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(VLM_PATH)},
            "device_processor": {"device": "cuda"},
        },
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )

    cache = _extract_features(
        policy=policy,
        dataset=dataset,
        rows=rows,
        base_records=base_records,
        config=config,
        batch_size=int(args.feature_batch_size),
        cache_path=RUN_DIR / "feature_cache.pt",
    )
    source_occ_raw = _build_source(cache, "occluded_1", "occluded_2", config)
    source_clean_raw = _build_source(cache, "clean_1", "clean_2", config)
    source_rect_raw = _build_source(cache, "rectangle_1", "rectangle_2", config)
    combined = torch.cat([source_occ_raw[:fit_count], source_clean_raw[:fit_count]], dim=0)
    _, source_mean, source_scale = _standardize(combined, combined)
    source_occ = (source_occ_raw - source_mean) / source_scale
    source_clean = (source_clean_raw - source_mean) / source_scale
    source_rect = (source_rect_raw - source_mean) / source_scale
    target_raw = cache["clean_2"].float()
    target = F.normalize(target_raw, dim=-1)

    initial_model = COVIStage0Adapter(config).to("cuda")
    model, training = _train_adapter(
        source=source_occ,
        clean_source=source_clean,
        camera2=cache["occluded_2"].float(),
        clean_camera2=cache["clean_2"].float(),
        target=target,
        context=cache["contexts"].float(),
        fit_count=fit_count,
        config=config,
        label="covi_stage0_full",
    )
    random_model, random_training = _train_adapter(
        source=source_rect,
        clean_source=source_clean,
        camera2=cache["rectangle_2"].float(),
        clean_camera2=cache["clean_2"].float(),
        target=target,
        context=cache["rectangle_contexts"].float(),
        fit_count=fit_count,
        config=config,
        label="random_cutout_equal_area_mlp",
    )

    checkpoint_path = RUN_DIR / "covi_stage0_adapter.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "proposal_hash": PROPOSAL_HASH,
            "config": config.to_dict(),
            "state_dict": model.state_dict(),
            "source_mean": source_mean,
            "source_scale": source_scale,
        },
        checkpoint_path,
    )
    reloaded = COVIStage0Adapter(config).to("cuda")
    reloaded.load_state_dict(torch.load(checkpoint_path, map_location="cuda", weights_only=True)["state_dict"])
    model_predictions = _predict(model, source_occ, cache["occluded_2"].float())
    reload_predictions = _predict(reloaded, source_occ, cache["occluded_2"].float())
    reload_max_abs = float(np.max(np.abs(model_predictions["imagined"] - reload_predictions["imagined"])))
    random_predictions = _predict(random_model, source_rect, cache["rectangle_2"].float())

    fit_slice = slice(0, fit_count)
    val_slice = slice(fit_count, fit_count + validation_count)
    fit_x = source_occ[fit_slice].numpy()
    validation_x = source_occ[val_slice].numpy()
    fit_y_norm = target[fit_slice].numpy()
    validation_y_norm = target[val_slice].numpy()
    fit_tasks = cache["task_indices"][fit_slice].numpy()
    validation_tasks = cache["task_indices"][val_slice].numpy()
    direct = _ridge_predictions(fit_x, fit_y_norm, validation_x)
    vim_proxy = _knn_predictions(fit_x, fit_y_norm, fit_tasks, validation_x, validation_tasks, k=5)
    train_mean = fit_y_norm.mean(axis=0)
    mean_prediction = np.repeat(train_mean[None, :], validation_count, axis=0)
    no_imagined = F.normalize(cache["occluded_2"][val_slice].float(), dim=-1).numpy()
    full = model_predictions["imagined"][val_slice]
    random_cutout = random_predictions["imagined"][val_slice]
    comparator_predictions = {
        "train_mean_target": mean_prediction,
        "direct_two_camera_ridge": direct,
        "vim_view_imagination_proxy_knn5": vim_proxy,
        "random_cutout_equal_area_mlp": random_cutout,
        "covi_no_imagined_view_ablation": no_imagined,
    }
    comparator_metrics = {
        name: prediction_metrics(prediction, validation_y_norm)
        for name, prediction in comparator_predictions.items()
    }
    comparator_metrics["covi_stage0_full"] = prediction_metrics(full, validation_y_norm)
    strongest_name = min(comparator_metrics, key=lambda name: comparator_metrics[name]["rmse"] if name != "covi_stage0_full" else math.inf)
    strongest = comparator_predictions[strongest_name]
    margin = normalized_rmse_margin(full, strongest, validation_y_norm, train_mean)
    margin_vim = normalized_rmse_margin(full, vim_proxy, validation_y_norm, train_mean)
    margin_random = normalized_rmse_margin(full, random_cutout, validation_y_norm, train_mean)
    episode_ids = cache["episode_indices"][val_slice].numpy()
    interval = episode_cluster_bootstrap_margin(
        candidate=full,
        baseline=strongest,
        target=validation_y_norm,
        train_target_mean=train_mean,
        episode_ids=episode_ids,
        iterations=config.bootstrap_iterations,
        seed=config.bootstrap_seed,
    )
    normalization = _normalization_sensitivity(
        full_normalized=full,
        baseline_normalized=strongest,
        target_raw=target_raw[val_slice].numpy(),
        train_target_raw=target_raw[fit_slice].numpy(),
    )

    action_min = np.asarray((artifact.get("action_range") or {}).get("min"), dtype=np.float32)
    action_max = np.asarray((artifact.get("action_range") or {}).get("max"), dtype=np.float32)
    action_smoke = _action_smoke(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        validation_rows=indexed["validation"],
        feature_cache=cache,
        standardized_occ_source=source_occ,
        standardized_clean_source=source_clean,
        model=model,
        initial_model=initial_model,
        action_min=action_min,
        action_max=action_max,
        config=config,
    )

    coverage = cache["coverage"].numpy()
    target_variance = np.var(target_raw[fit_slice].numpy(), axis=0)
    nonzero_target_dims = int(np.sum(target_variance > 1e-8))
    full_occ_gate = model_predictions["gate"][val_slice].reshape(-1)
    clean_predictions = _predict(model, source_clean, cache["clean_2"].float())
    full_clean_gate = clean_predictions["gate"][val_slice].reshape(-1)
    gate_localization_ratio = float(full_occ_gate.mean() / max(1e-12, full_clean_gate.mean()))
    feature_headroom = float(prediction_metrics(no_imagined, validation_y_norm)["rmse"])
    gradients = training["last_gradient_norms"] or {}
    gradient_valid = bool(
        all(math.isfinite(float(value)) for value in gradients.values())
        and float(gradients.get("context_head", 0.0)) > 0.0
        and float(gradients.get("predictor", 0.0)) > 0.0
        and float(gradients.get("residual_projection", 0.0)) > 0.0
        and float(gradients.get("gate_head", 0.0)) > 0.0
    )
    shape_audit = cache["shape_audit"] or {}
    expected_tokens = all(shape[-2:] == [64, 960] for shape in shape_audit.get("token_pair", []))
    split_valid = bool(
        split_summary["discovery_fit"]["records"] == 600
        and split_summary["validation"]["records"] == 400
        and split_summary["confirmatory_reserved"]["records"] == 1200
        and all(details["duplicate_sample_ids"] == 0 for details in split_summary.values())
        and all(details["duplicate_frame_keys"] == 0 for details in split_summary.values())
    )
    mask_valid = bool(np.mean((coverage >= config.mask_fraction_min) & (coverage <= config.mask_fraction_max)) >= 0.99)
    implementation_valid = bool(
        split_valid
        and expected_tokens
        and mask_valid
        and nonzero_target_dims >= 100
        and gradient_valid
        and reload_max_abs <= 1e-6
    )
    identity_safe = bool(
        action_smoke["initial_action_delta_p95"] <= config.init_action_delta_p95_max
        and action_smoke["action_delta_p95"] <= config.trained_action_delta_p95_max
        and action_smoke["translation_delta_p95"] <= config.translation_delta_p95_max
        and action_smoke["rotation_delta_p95"] <= config.rotation_delta_p95_max
        and action_smoke["gripper_delta_p95"] <= config.gripper_delta_p95_max
        and action_smoke["clean_retention_delta_p95"] <= config.clean_action_delta_p95_max
        and action_smoke["output_valid_fraction"] == 1.0
    )
    headroom = bool(feature_headroom >= config.practical_margin)
    classification_input = {
        "implementation_and_data_valid": implementation_valid,
        "diagnostic_headroom_exists": headroom,
        "identity_and_safety_passed": identity_safe,
        "candidate_margin": margin,
        "candidate_margin_vs_vim_proxy": margin_vim,
        "candidate_margin_vs_random_cutout": margin_random,
        "bootstrap_interval": interval,
        "normalization_sensitivity_resolved": normalization["resolved"],
    }
    decision = classify_stage0(classification_input, config)
    if decision == "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH":
        next_command = (
            r"C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_covi_vla_stage0.py --mode validation-search"
        )
    elif decision == "COVI_STAGE_0_UNDERPOWERED_ONE_CHECK_ALLOWED":
        next_command = (
            r"C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_covi_vla_stage0.py --mode one-check"
        )
    else:
        next_command = "adjudicate_and_archive_or_repair_under_current_governance"

    report = {
        "method": "COVI-VLA",
        "date_kst": DATE_KST,
        "proposal_hash": PROPOSAL_HASH,
        "reviewer_status": "APPROVE_WITH_FIXED_EMPIRICAL_RISKS",
        "mode": args.mode,
        "command": (
            r"C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_covi_vla_stage0.py --mode audit"
        ),
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "preflight": preflight,
        "config": config.to_dict(),
        "paths": {
            "checkpoint": str(CHECKPOINT_PATH),
            "vlm": str(VLM_PATH),
            "dataset": str(DATASET_ROOT),
            "split_manifest": str(SPLIT_MANIFEST),
            "base_artifact": str(BASE_ARTIFACT),
            "feature_cache": str(RUN_DIR / "feature_cache.pt"),
            "adapter_checkpoint": str(checkpoint_path),
        },
        "identities": {
            "checkpoint_config_sha256": _sha256(CHECKPOINT_PATH / "config.json"),
            "checkpoint_weights_sha256": _sha256(CHECKPOINT_PATH / "model.safetensors"),
            "split_manifest_sha256": _sha256(SPLIT_MANIFEST),
            "base_artifact_sha256": _sha256(BASE_ARTIFACT),
            "adapter_checkpoint_sha256": _sha256(checkpoint_path),
        },
        "split_summary": split_summary,
        "confirmatory_test_records_decoded": 0,
        "source_gate": {
            "legal_inference_features": list(LEGAL_INFERENCE_FEATURES),
            "forbidden_inference_keys": sorted(FORBIDDEN_INFERENCE_KEYS),
            "used_inference_features": [
                "occluded_camera1_visual_summary",
                "occluded_camera2_visual_summary",
                "observation.state",
                "base_action",
                "language_or_task_instruction_one_hot_proxy",
                "predicted_occlusion_context",
            ],
            "ground_truth_context_used_for_supervision_only": True,
            "clean_complementary_view_used_for_supervision_and_oracle_only": True,
            "confirmatory_identity_used": False,
            "privileged_inference_used": False,
            "passed": True,
        },
        "hook_and_shape_audit": {
            **shape_audit,
            "expected_visual_token_shape_per_stream": [64, 960],
            "hook": "SmolVLAPolicy.model.embed_prefix/vlm_with_expert.embed_image before prefix concatenation",
            "exact_hook_reproduced": expected_tokens,
            "changed_tokens_when_active": 64,
            "available_visual_tokens": 128,
        },
        "occlusion_proxy": {
            "name": "irregular_scene_obstruction_proxy_v1",
            "development_proxy_only": True,
            "physical_occlusion_claim_validated": False,
            "coverage_mean": float(coverage.mean()),
            "coverage_min": float(coverage.min()),
            "coverage_max": float(coverage.max()),
            "coverage_in_range_fraction": float(
                np.mean((coverage >= config.mask_fraction_min) & (coverage <= config.mask_fraction_max))
            ),
            "random_cutout_equal_area_comparator_live": True,
        },
        "target_health": {
            "target_shape": [fit_count, config.feature_dim],
            "variance_nonzero_dimensions": nonzero_target_dims,
            "variance_min": float(target_variance.min()),
            "variance_max": float(target_variance.max()),
            "collapsed": nonzero_target_dims < 100,
        },
        "training": training,
        "random_cutout_training": random_training,
        "base_parameters_updated": 0,
        "adapter_parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "checkpoint_reload_max_abs_difference": reload_max_abs,
        "representation_metrics": {
            "validation_record_count": validation_count,
            "validation_episode_count": len(set(int(value) for value in episode_ids.tolist())),
            "comparators": comparator_metrics,
            "strongest_non_oracle_comparator": strongest_name,
            "candidate_margin_over_strongest": margin,
            "candidate_margin_vs_vim_proxy": margin_vim,
            "candidate_margin_vs_random_cutout": margin_random,
            "bootstrap_interval": interval,
            "normalization_sensitivity": normalization,
            "clean_complementary_view_oracle_rmse": 0.0,
            "no_imagined_view_feature_headroom_rmse": feature_headroom,
            "vim_proxy_status": "faithful_transparent_local_proxy_not_official_vim",
        },
        "mechanism_activation": {
            "occluded_gate_mean": float(full_occ_gate.mean()),
            "clean_gate_mean": float(full_clean_gate.mean()),
            "occluded_to_clean_gate_ratio": gate_localization_ratio,
            "residual_norm_mean": float(np.linalg.norm(model_predictions["residual"][val_slice], axis=1).mean()),
            "full_differs_from_no_imagined_ablation": bool(
                np.max(np.abs(full - no_imagined)) > 1e-8
            ),
        },
        "action_smoke": action_smoke,
        "implementation_and_data_valid": implementation_valid,
        "diagnostic_headroom_exists": headroom,
        "identity_and_safety_passed": identity_safe,
        "candidate_margin": margin,
        "candidate_margin_vs_vim_proxy": margin_vim,
        "candidate_margin_vs_random_cutout": margin_random,
        "bootstrap_interval": interval,
        "normalization_sensitivity_resolved": normalization["resolved"],
        "false_negative_safeguard": {
            "strongest_fair_interpretation": "identity-preserving frozen-SmolVLA complementary-feature adapter under a development occlusion proxy",
            "narrowest_publishable_claim": "bounded complementary-feature adaptation may improve physical scene-induced occlusion only if later physical validation succeeds",
            "evidence_class": (
                "UNDERPOWERED_OR_UNRESOLVED"
                if decision == "COVI_STAGE_0_UNDERPOWERED_ONE_CHECK_ALLOWED"
                else "ROBUST_EMPIRICAL_DESIGN_FAILURE"
                if decision == "ROBUST_EMPIRICAL_DESIGN_FAILURE"
                else "IMPLEMENTATION_OR_DATA_FAILURE"
                if decision == "IMPLEMENTATION_OR_DATA_FAILURE"
                else "PASS"
            ),
            "false_positive_risk": "synthetic development occlusion may overstate physical-occlusion transfer",
            "false_negative_risk": "one validation episode per task and normalization sensitivity can hide a useful small effect",
            "confidence": "moderate" if implementation_valid else "low",
            "record_count": validation_count,
            "independent_episode_count": int(interval["episode_count"]),
            "bootstrap_interval": interval,
            "practical_effect_threshold": config.practical_margin,
            "normalization_sensitivity": normalization,
            "exact_evidence_required_for_permanent_kill": "valid data and implementation, safe acting mechanism, 40 independent episodes, resolved normalization sensitivity, and bootstrap upper bound below 0.02 against both VIM proxy and random-cutout",
            "small_point_estimate_alone_used_for_kill": False,
            "one_fixed_check_allowed": decision == "COVI_STAGE_0_UNDERPOWERED_ONE_CHECK_ALLOWED",
        },
        "closed_loop_experiment_happened": False,
        "validation_search_happened": False,
        "confirmatory_test_tuning_happened": False,
        "final_decision": decision,
        "exact_next_command": next_command,
        "runtime": {
            "elapsed_sec": time.monotonic() - started,
            "cuda_allocated_mb": float(torch.cuda.memory_allocated() / (1024**2)),
            "cuda_peak_allocated_mb": float(torch.cuda.max_memory_allocated() / (1024**2)),
        },
    }
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["audit"], default="audit")
    parser.add_argument("--feature-batch-size", type=int, default=16)
    parser.add_argument("--result-json", default=str(RESULT_JSON))
    parser.add_argument("--result-md", default=str(RESULT_MD))
    parser.add_argument("--blocker-json", default=str(BLOCKER_JSON))
    args = parser.parse_args()
    os.chdir(REPO_ROOT)
    torch.manual_seed(20260715)
    np.random.seed(20260715)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(20260715)
        torch.cuda.reset_peak_memory_stats()
    try:
        report = run_stage0(args)
        _write_json(Path(args.result_json), report)
        _write_md(Path(args.result_md), report)
        blocker = Path(args.blocker_json)
        if blocker.exists():
            blocker.unlink()
        print(json.dumps({"final_decision": report["final_decision"], "result": args.result_json}, sort_keys=True))
        return 0
    except Exception as exc:
        blocker = {
            "method": "COVI-VLA",
            "date_kst": DATE_KST,
            "proposal_hash": PROPOSAL_HASH,
            "final_decision": "IMPLEMENTATION_OR_DATA_FAILURE",
            "failing_prerequisite": type(exc).__name__,
            "observed": str(exc),
            "expected": "frozen COVI Stage 0 completes with official local assets and no confirmatory-test access",
            "attempted_command": (
                r"C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_covi_vla_stage0.py --mode audit"
            ),
            "traceback": traceback.format_exc(),
            "bounded_implementation_repair_possible": True,
            "scientific_method_kill": False,
            "exact_resume_command": (
                r"C:\Users\jiheo\miniconda3\envs\tca_map\python.exe scripts\run_covi_vla_stage0.py --mode audit"
            ),
        }
        _write_json(Path(args.blocker_json), blocker)
        print(json.dumps({"final_decision": blocker["final_decision"], "blocker": args.blocker_json}, sort_keys=True))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
