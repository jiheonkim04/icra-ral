"""Run the frozen NICE-VLA Stage 0B1 offline development audit."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_famr_vla_stage0 import (  # noqa: E402
    VLM_PATH,
    _active_linux_workers,
    _apply_official_env_image_processor,
    _load_policy_and_processors,
    _resource_evidence,
    _set_offline_environment,
)
from scripts.run_nice_vla_stage0 import (  # noqa: E402
    SOURCE_COMMIT,
    _base_passthrough,
    _default_source_root,
    _edge_hash,
    _feature_path,
    _load_feature,
    _problem_language,
    _proposal_hash_text,
    _resolve_task,
    _save_feature,
    _sha256,
    _source_map,
    _utc_now,
    _write_json,
    _write_text,
)
from tca_map.smolvla.nice_vla import (  # noqa: E402
    ACTION_DIM,
    K_STEP,
    LOW_RANK,
    PROPOSAL_HASH,
    VARIANCE_CEILING,
    VARIANCE_FLOOR,
    TinyCovariance,
    auroc_average_ranks,
    canonical_json_sha256,
    condition_vector,
    conformal_threshold,
    covariance_nll,
    deterministic_pca_basis,
    discovery_gripper_deadband,
    episode_cluster_score,
    innovation_terms,
    mean_cosine_loss,
    pair_key,
    validate_manifest,
)


SEED = 20262011
DISCOVERY_PAIRS = 1152
CALIBRATION_PAIRS = 320
EVALUATION_PAIRS = 320
PLANNED_PAIRS = 1792
FRAMES_PER_DEMO = 16
MEAN_STEPS = 400
COVARIANCE_STEPS = 300
BATCH_SIZE = 8
PROPOSAL_FILE = REPO_ROOT / "reports" / "nice_vla" / "researcher_proposal.md"
RESOURCE_REGISTRY = REPO_ROOT / "reports" / "resource_contention_intervals.json"
STAGE0A_RESULT = REPO_ROOT / "reports" / "nice_vla" / "stage_0a_result.json"
STAGE0B_PROTOCOL = REPO_ROOT / "reports" / "nice_vla" / "stage_0b1_execution_protocol.md"
STAGE0B_CLARIFICATION = REPO_ROOT / "reports" / "nice_vla" / "stage_0b1_protocol_clarification.md"

DISCOVERY_TASKS = (
    ("libero_10", 1),
    ("libero_10", 3),
    ("libero_goal", 1),
    ("libero_goal", 3),
    ("libero_object", 1),
    ("libero_spatial", 1),
)
VALIDATION_TASKS = (
    ("libero_10", 5),
    ("libero_goal", 5),
    ("libero_object", 3),
    ("libero_spatial", 3),
)


def _asset_path(*parts: str) -> Path:
    root = Path("C:/assets") if os.name == "nt" else Path("/mnt/c/assets")
    return root.joinpath(*parts)


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    report = Path(args.report_root)
    run = Path(args.run_root)
    return {
        "report": report,
        "run": run,
        "feature_dir": run / "features",
        "checkpoint": Path(args.checkpoint),
        "data_root": Path(args.libero_data_root),
        "source_root": Path(args.vla_corrector_source),
        "pid": report / "stage_0b1_pid.txt",
        "heartbeat": report / "stage_0b1_heartbeat.json",
        "status": report / "stage_0b1_status.json",
        "partial": report / "stage_0b1_partial.json",
        "manifest": report / "stage_0b1_pair_manifest.json",
        "preflight": report / "stage_0b1_preflight.json",
        "result_json": report / "stage_0b1_result.json",
        "result_md": report / "stage_0b1_result.md",
        "validation": report / "stage_0b1_validation.json",
        "blocker": report / "stage_0b1_implementation_blocker.json",
        "mean_checkpoint": run / "vla_corrector_shared_mean.pt",
        "covariance_checkpoint": run / "nice_reference_diagonal.pt",
        "basis": run / "nice_rank8_basis.npy",
        "z_memmap": run / "z_t.npy",
        "delta_memmap": run / "delta_z.npy",
        "action_memmap": run / "actions.npy",
        "previous_memmap": run / "previous_actions.npy",
    }


def _heartbeat_loop(path: Path, state: dict[str, Any], stop: threading.Event) -> None:
    while not stop.wait(10.0):
        _write_json(path, {**state, "updated_at": _utc_now()})


def _sample_frames(length: int) -> list[int]:
    valid = length - K_STEP
    if valid < FRAMES_PER_DEMO:
        return list(range(max(0, valid)))
    values = np.floor(np.linspace(0, valid - 1, FRAMES_PER_DEMO)).astype(int).tolist()
    if len(values) != len(set(values)):
        raise RuntimeError("fixed Stage 0B1 sampler produced duplicate starts")
    return values


def _add_partition(
    rows: list[dict[str, Any]],
    task_reports: list[dict[str, Any]],
    *,
    data_root: Path,
    tasks: Sequence[tuple[str, int]],
    role: str,
    demos: Sequence[int],
    task_offset: int,
) -> None:
    import h5py

    for local_task_index, (suite, task_id) in enumerate(tasks):
        source = _resolve_task(data_root, suite, task_id)
        with h5py.File(source, "r") as handle:
            data = handle["data"]
            language = _problem_language(data)
            demo_reports = []
            for demo_id in demos:
                demo = data[f"demo_{demo_id}"]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                    raise RuntimeError(f"unexpected actions {actions.shape} for {source} demo_{demo_id}")
                observations = demo["obs"]
                if "agentview_rgb" not in observations or "eye_in_hand_rgb" not in observations:
                    raise RuntimeError(f"missing images for {source} demo_{demo_id}")
                frames = _sample_frames(len(actions))
                demo_reports.append({"demo_id": demo_id, "length": len(actions), "sampled_frames": frames})
                for slot, frame in enumerate(frames):
                    row = {
                        "partition": role,
                        "role": role,
                        "suite": suite,
                        "task_id": task_id,
                        "task_identity": f"{suite}/task_{task_id}",
                        "task_index": task_offset + local_task_index,
                        "task_language": language,
                        "source_path": str(source),
                        "demo_id": int(demo_id),
                        "episode": int(demo_id),
                        "frame_t": int(frame),
                        "frame_t_plus_10": int(frame + K_STEP),
                        "frame": int(frame),
                        "sample_slot": int(slot),
                        "episode_length": int(len(actions)),
                        "temporal_t_plus_20_eligible": bool(role == "validation_evaluation" and frame + 20 < len(actions)),
                    }
                    row["pair_key"] = pair_key(row)
                    rows.append(row)
        stat = source.stat()
        task_reports.append(
            {
                "role": role,
                "suite": suite,
                "task_id": task_id,
                "task_identity": f"{suite}/task_{task_id}",
                "source_path": str(source),
                "resolved_filename": source.name,
                "source_size_bytes": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "source_edge_sha256": _edge_hash(source),
                "task_language": language,
                "demonstrations": demo_reports,
            }
        )


def _build_manifest(data_root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reports: list[dict[str, Any]] = []
    _add_partition(
        rows,
        reports,
        data_root=data_root,
        tasks=DISCOVERY_TASKS,
        role="discovery",
        demos=tuple(range(0, 12)),
        task_offset=0,
    )
    _add_partition(
        rows,
        reports,
        data_root=data_root,
        tasks=VALIDATION_TASKS,
        role="validation_calibration",
        demos=tuple(range(30, 35)),
        task_offset=len(DISCOVERY_TASKS),
    )
    _add_partition(
        rows,
        reports,
        data_root=data_root,
        tasks=VALIDATION_TASKS,
        role="validation_evaluation",
        demos=tuple(range(35, 40)),
        task_offset=len(DISCOVERY_TASKS),
    )
    counts = {role: sum(row["role"] == role for row in rows) for role in (
        "discovery", "validation_calibration", "validation_evaluation"
    )}
    expected = {
        "discovery": DISCOVERY_PAIRS,
        "validation_calibration": CALIBRATION_PAIRS,
        "validation_evaluation": EVALUATION_PAIRS,
    }
    if counts != expected:
        raise RuntimeError(f"fixed manifest count mismatch: {counts} != {expected}")
    return rows, {"task_reports": reports, "role_counts": counts}


def _prepare_images(policy: Any, images: Sequence[Any]) -> Any:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

    tensors = []
    for image in images:
        value = image if torch.is_tensor(image) else torch.as_tensor(np.asarray(image).copy())
        if value.shape[-1] in (1, 3):
            value = value.permute(2, 0, 1)
        value = value.float()
        if float(value.max()) > 1.0:
            value = value / 255.0
        tensors.append(value)
    batch = torch.stack(tensors).to("cuda")
    resize_cfg = getattr(policy.config, "resize_imgs_with_padding", None)
    if resize_cfg is not None:
        batch = resize_with_pad(batch, *resize_cfg, pad_value=0)
    return (batch * 2.0 - 1.0).to(dtype=next(policy.model.parameters()).dtype)


def _extract_pair(policy: Any, row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    import h5py
    import torch

    with h5py.File(str(row["source_path"]), "r") as handle:
        demo = handle["data"][f"demo_{int(row['demo_id'])}"]
        observations = demo["obs"]
        current = int(row["frame_t"])
        future = int(row["frame_t_plus_10"])
        current_pair = _apply_official_env_image_processor(
            observations["agentview_rgb"][current], observations["eye_in_hand_rgb"][current]
        )
        future_pair = _apply_official_env_image_processor(
            observations["agentview_rgb"][future], observations["eye_in_hand_rgb"][future]
        )
        images = [*current_pair, *future_pair]
        if bool(row["temporal_t_plus_20_eligible"]):
            temporal_pair = _apply_official_env_image_processor(
                observations["agentview_rgb"][current + 20], observations["eye_in_hand_rgb"][current + 20]
            )
            images.extend(temporal_pair)
        actions = np.asarray(demo["actions"], dtype=np.float32)
        action = actions[current]
        previous = actions[current - 1] if current > 0 else action
    prepared = _prepare_images(policy, images)
    with torch.no_grad():
        tokens = policy.model.vlm_with_expert.embed_image(prepared).float().cpu()
    expected_views = 6 if bool(row["temporal_t_plus_20_eligible"]) else 4
    if tokens.shape[:2] != (expected_views, 64) or tokens.shape[2] != 960:
        raise RuntimeError(f"unexpected token tensor {tuple(tokens.shape)}")
    z_t = torch.cat((tokens[0], tokens[1]), dim=0).numpy()
    z_future = torch.cat((tokens[2], tokens[3]), dim=0).numpy()
    values = {
        "z_t": z_t.astype(np.float16),
        "delta_z": (z_future - z_t).astype(np.float16),
        "action": action.astype(np.float32),
        "previous_action": previous.astype(np.float32),
    }
    if expected_views == 6:
        z_temporal = torch.cat((tokens[4], tokens[5]), dim=0).numpy()
        values["delta_z_t_plus_20"] = (z_temporal - z_t).astype(np.float16)
    return values


def _row_summary(row: Mapping[str, Any], feature_path: Path, values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    delta = values["delta_z"].astype(np.float32)
    return {
        **dict(row),
        "feature_path": str(feature_path),
        "feature_sha256": _sha256(feature_path),
        "latent_shape": list(values["z_t"].shape),
        "latent_finite_fraction": float(np.mean(np.isfinite(values["z_t"]))),
        "delta_finite_fraction": float(np.mean(np.isfinite(delta))),
        "delta_variance": float(np.var(delta)),
        "action_finite_fraction": float(np.mean(np.isfinite(values["action"]))),
        "temporal_target_present": "delta_z_t_plus_20" in values,
    }


def _partial_payload(manifest_hash: str, rows: Sequence[Mapping[str, Any]], exception_count: int = 0) -> dict[str, Any]:
    return {
        "method": "NICE-VLA",
        "stage": "0B1",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "planned_pair_count": PLANNED_PAIRS,
        "completed_pair_count": len(rows),
        "completed_pair_keys": [row["pair_key"] for row in rows],
        "rows": list(rows),
        "exception_count": int(exception_count),
        "updated_at": _utc_now(),
    }


def _load_resume(paths: Mapping[str, Path], manifest: Sequence[Mapping[str, Any]], manifest_hash: str) -> list[dict[str, Any]]:
    if not paths["partial"].is_file():
        return []
    partial = json.loads(paths["partial"].read_text(encoding="utf-8-sig"))
    if partial.get("proposal_hash") != PROPOSAL_HASH or partial.get("manifest_hash") != manifest_hash:
        raise RuntimeError("Stage 0B1 partial identity mismatch")
    rows = list(partial.get("rows") or [])
    audit = validate_manifest(manifest, rows)
    if audit["duplicate_result_key_count"] or audit["extra_result_key_count"]:
        raise RuntimeError(f"invalid Stage 0B1 partial keys: {audit}")
    for row in rows:
        feature = Path(str(row["feature_path"]))
        if not feature.is_file() or _sha256(feature) != row["feature_sha256"]:
            raise RuntimeError(f"missing/changed Stage 0B1 feature {row['pair_key']}")
    return rows


def _materialize_memmaps(paths: Mapping[str, Path], manifest: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> None:
    ordered = {str(row["pair_key"]): row for row in rows}
    shapes = ((paths["z_memmap"], np.float16, (PLANNED_PAIRS, 128, 960)),
              (paths["delta_memmap"], np.float16, (PLANNED_PAIRS, 128, 960)),
              (paths["action_memmap"], np.float32, (PLANNED_PAIRS, 7)),
              (paths["previous_memmap"], np.float32, (PLANNED_PAIRS, 7)))
    arrays = [np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape) for path, dtype, shape in shapes]
    for index, manifest_row in enumerate(manifest):
        feature = _load_feature(Path(ordered[str(manifest_row["pair_key"])]["feature_path"]))
        arrays[0][index] = feature["z_t"]
        arrays[1][index] = feature["delta_z"]
        arrays[2][index] = feature["action"]
        arrays[3][index] = feature["previous_action"]
    for array in arrays:
        array.flush()


def _batch(array: np.ndarray, indices: np.ndarray, device: Any) -> Any:
    import torch

    return torch.from_numpy(np.asarray(array[indices], dtype=np.float32)).to(device)


def _gradient_norm(parameters: Sequence[Any]) -> float:
    total = 0.0
    for parameter in parameters:
        if parameter.grad is not None:
            total += float(torch_square_sum(parameter.grad))
    return math.sqrt(total)


def torch_square_sum(value: Any) -> float:
    return float((value.detach().float() ** 2).sum().item())


def _load_mean_class(source_root: Path, token_dim: int) -> Any:
    source_path = str(source_root / "src")
    if source_path not in sys.path:
        sys.path.insert(0, source_path)
    from siglip_dynamics.MLP import SiglipResidualMLP
    from siglip_dynamics.config import ModelScale, SiglipMLPConfig

    config = SiglipMLPConfig(
        token_dim=token_dim,
        action_dim=ACTION_DIM,
        action_embed_dim=256,
        dropout=0.0,
        scale=ModelScale.M20,
    )
    return SiglipResidualMLP(config)


def _fit_mean(paths: Mapping[str, Path], z: np.ndarray, delta: np.ndarray, actions: np.ndarray, indices: np.ndarray) -> tuple[Any, dict[str, Any]]:
    import torch

    device = torch.device("cuda")
    model = _load_mean_class(paths["source_root"], 960).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    permutation = indices[torch.randperm(len(indices), generator=generator).numpy()]
    cursor = 0
    trace = []
    first_gradient = None
    for step in range(1, MEAN_STEPS + 1):
        if cursor + BATCH_SIZE > len(permutation):
            permutation = indices[torch.randperm(len(indices), generator=generator).numpy()]
            cursor = 0
        selected = permutation[cursor : cursor + BATCH_SIZE]
        cursor += BATCH_SIZE
        visual = _batch(z, selected, device)
        target = _batch(delta, selected, device)
        action = _batch(actions, selected, device)
        optimizer.zero_grad(set_to_none=True)
        loss = mean_cosine_loss(model(visual, action), target)
        loss.backward()
        if first_gradient is None:
            first_gradient = _gradient_norm(list(model.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            trace.append({"step": step, "loss": float(loss.item())})
    torch.save({"proposal_hash": PROPOSAL_HASH, "state_dict": model.state_dict()}, paths["mean_checkpoint"])
    reloaded = _load_mean_class(paths["source_root"], 960).to(device)
    reloaded.load_state_dict(torch.load(paths["mean_checkpoint"], map_location=device, weights_only=True)["state_dict"])
    probe = indices[:BATCH_SIZE]
    with torch.no_grad():
        reference = model(_batch(z, probe, device), _batch(actions, probe, device))
        loaded = reloaded(_batch(z, probe, device), _batch(actions, probe, device))
    reload_error = float(torch.max(torch.abs(reference - loaded)).item())
    for parameter in model.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    del reloaded
    return model, {
        "steps": MEAN_STEPS,
        "batch_size": BATCH_SIZE,
        "trace": trace,
        "first_gradient_norm": float(first_gradient or 0.0),
        "checkpoint_path": str(paths["mean_checkpoint"]),
        "checkpoint_sha256": _sha256(paths["mean_checkpoint"]),
        "reload_max_abs_error": reload_error,
    }


def _fit_covariance(
    paths: Mapping[str, Path],
    mean: Any,
    z: np.ndarray,
    delta: np.ndarray,
    actions: np.ndarray,
    previous: np.ndarray,
    indices: np.ndarray,
    deadband: float,
) -> tuple[Any, dict[str, Any]]:
    import torch

    device = torch.device("cuda")
    model = TinyCovariance(960, rank=0).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    permutation = indices[torch.randperm(len(indices), generator=generator).numpy()]
    cursor = 0
    trace = []
    first_gradient = None
    frozen_mean_gradient = None
    for step in range(1, COVARIANCE_STEPS + 1):
        if cursor + BATCH_SIZE > len(permutation):
            permutation = indices[torch.randperm(len(indices), generator=generator).numpy()]
            cursor = 0
        selected = permutation[cursor : cursor + BATCH_SIZE]
        cursor += BATCH_SIZE
        visual = _batch(z, selected, device)
        target = _batch(delta, selected, device)
        action = _batch(actions, selected, device)
        prior = _batch(previous, selected, device)
        with torch.no_grad():
            residual = target - mean(visual, action)
        condition = condition_vector(action, prior, deadband)
        optimizer.zero_grad(set_to_none=True)
        diagonal, _ = model(visual.detach(), condition.detach())
        loss = covariance_nll(residual.detach(), diagonal)
        loss.backward()
        if first_gradient is None:
            first_gradient = _gradient_norm(list(model.parameters()))
            frozen_mean_gradient = _gradient_norm(list(mean.parameters()))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step == 1 or step % 100 == 0:
            trace.append({"step": step, "loss": float(loss.item())})
    torch.save({"proposal_hash": PROPOSAL_HASH, "state_dict": model.state_dict()}, paths["covariance_checkpoint"])
    reloaded = TinyCovariance(960, rank=0).to(device)
    reloaded.load_state_dict(torch.load(paths["covariance_checkpoint"], map_location=device, weights_only=True)["state_dict"])
    probe = indices[:BATCH_SIZE]
    with torch.no_grad():
        visual = _batch(z, probe, device)
        action = _batch(actions, probe, device)
        prior = _batch(previous, probe, device)
        condition = condition_vector(action, prior, deadband)
        reference, _ = model(visual, condition)
        loaded, _ = reloaded(visual, condition)
    reload_error = float(torch.max(torch.abs(reference - loaded)).item())
    del reloaded
    return model, {
        "steps": COVARIANCE_STEPS,
        "batch_size": BATCH_SIZE,
        "trace": trace,
        "first_gradient_norm": float(first_gradient or 0.0),
        "frozen_mean_gradient_norm": float(frozen_mean_gradient or 0.0),
        "checkpoint_path": str(paths["covariance_checkpoint"]),
        "checkpoint_sha256": _sha256(paths["covariance_checkpoint"]),
        "reload_max_abs_error": reload_error,
    }


def _cosine_loss_rows(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred = prediction.reshape(len(prediction), -1).astype(np.float64)
    truth = target.reshape(len(target), -1).astype(np.float64)
    numerator = np.sum(pred * truth, axis=1)
    denominator = np.linalg.norm(pred, axis=1) * np.linalg.norm(truth, axis=1)
    cosine = np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 1e-12)
    return 1.0 - cosine


def _predict_mean(mean: Any, z: np.ndarray, actions: np.ndarray, indices: Sequence[int], batch_size: int = 8) -> np.ndarray:
    import torch

    output = np.empty((len(indices), 128, 960), dtype=np.float16)
    device = torch.device("cuda")
    with torch.no_grad():
        for start in range(0, len(indices), batch_size):
            selected = np.asarray(indices[start : start + batch_size], dtype=np.int64)
            value = mean(_batch(z, selected, device), _batch(actions, selected, device))
            output[start : start + len(selected)] = value.float().cpu().numpy().astype(np.float16)
    return output


def _natural_scores(
    mean: Any,
    covariance: Any,
    z: np.ndarray,
    delta: np.ndarray,
    actions: np.ndarray,
    previous: np.ndarray,
    indices: Sequence[int],
    deadband: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    import torch

    predictions = np.empty((len(indices), 128, 960), dtype=np.float16)
    variances = np.empty((len(indices), 128, 960), dtype=np.float16)
    nice_scores = np.empty(len(indices), dtype=np.float64)
    prior_scores = np.empty(len(indices), dtype=np.float64)
    scale_values = []
    device = torch.device("cuda")
    with torch.no_grad():
        for start in range(0, len(indices), BATCH_SIZE):
            selected = np.asarray(indices[start : start + BATCH_SIZE], dtype=np.int64)
            visual = _batch(z, selected, device)
            target = _batch(delta, selected, device)
            action = _batch(actions, selected, device)
            prior_action = _batch(previous, selected, device)
            prediction = mean(visual, action)
            condition = condition_vector(action, prior_action, deadband)
            diagonal, _ = covariance(visual, condition)
            residual = target - prediction
            score, _, _ = innovation_terms(residual, diagonal)
            cosine = mean_cosine_per_row(prediction, target)
            count = len(selected)
            predictions[start : start + count] = prediction.float().cpu().numpy().astype(np.float16)
            variances[start : start + count] = diagonal.reshape(count, 128, 960).float().cpu().numpy().astype(np.float16)
            nice_scores[start : start + count] = score.double().cpu().numpy()
            prior_scores[start : start + count] = cosine.double().cpu().numpy()
            scale_values.append(diagonal.float().cpu().numpy())
    scales = np.concatenate(scale_values, axis=0)
    summary = {
        "minimum": float(np.min(scales)),
        "median": float(np.median(scales)),
        "maximum": float(np.max(scales)),
        "clamped_fraction": float(np.mean((scales <= VARIANCE_FLOOR) | (scales >= VARIANCE_CEILING))),
    }
    return predictions, variances, nice_scores, prior_scores, summary


def mean_cosine_per_row(prediction: Any, target: Any) -> Any:
    import torch.nn.functional as functional

    pred = prediction.reshape(prediction.shape[0], -1)
    truth = target.reshape(target.shape[0], -1)
    return 1.0 - functional.cosine_similarity(pred, truth, dim=-1, eps=1e-8)


def _mean_headroom(
    mean: Any,
    manifest: Sequence[Mapping[str, Any]],
    z: np.ndarray,
    delta: np.ndarray,
    actions: np.ndarray,
    discovery_indices: Sequence[int],
    evaluation_indices: Sequence[int],
) -> dict[str, Any]:
    suite_means = {}
    for suite in sorted({manifest[index]["suite"] for index in discovery_indices}):
        selected = [index for index in discovery_indices if manifest[index]["suite"] == suite]
        suite_means[suite] = np.mean(np.asarray(delta[selected], dtype=np.float32), axis=0)
    prediction = _predict_mean(mean, z, actions, evaluation_indices)
    target = np.asarray(delta[list(evaluation_indices)], dtype=np.float32)
    full_losses = _cosine_loss_rows(prediction, target)
    baseline = np.stack([suite_means[str(manifest[index]["suite"])] for index in evaluation_indices])
    task_mean_losses = _cosine_loss_rows(baseline, target)
    per_task = {}
    for task in sorted({manifest[index]["task_identity"] for index in evaluation_indices}):
        positions = [position for position, index in enumerate(evaluation_indices) if manifest[index]["task_identity"] == task]
        per_task[task] = {
            "full_cosine_loss": float(np.mean(full_losses[positions])),
            "zero_change_cosine_loss": 1.0,
            "suite_mean_cosine_loss": float(np.mean(task_mean_losses[positions])),
        }
    aggregate = {
        name: float(np.mean([row[name] for row in per_task.values()]))
        for name in ("full_cosine_loss", "zero_change_cosine_loss", "suite_mean_cosine_loss")
    }
    aggregate["passed"] = bool(
        aggregate["full_cosine_loss"] < aggregate["zero_change_cosine_loss"]
        and aggregate["full_cosine_loss"] < aggregate["suite_mean_cosine_loss"]
    )
    return {"aggregate": aggregate, "per_task": per_task}


def _episode_means(
    mean: Any,
    manifest: Sequence[Mapping[str, Any]],
    z: np.ndarray,
    delta: np.ndarray,
    actions: np.ndarray,
    discovery_indices: Sequence[int],
) -> Any:
    import torch

    grouped: dict[tuple[str, int], list[int]] = defaultdict(list)
    for index in discovery_indices:
        grouped[(str(manifest[index]["task_identity"]), int(manifest[index]["demo_id"]))].append(index)
    means = []
    device = torch.device("cuda")
    with torch.no_grad():
        for indices in grouped.values():
            prediction = mean(_batch(z, np.asarray(indices), device), _batch(actions, np.asarray(indices), device))
            target = _batch(delta, np.asarray(indices), device)
            means.append((target - prediction).mean(dim=0).cpu())
    return torch.stack(means).to(device)


def _calibration_and_diagnostics(
    mean: Any,
    covariance: Any,
    manifest: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
    z: np.ndarray,
    delta: np.ndarray,
    actions: np.ndarray,
    previous: np.ndarray,
    calibration_indices: Sequence[int],
    evaluation_indices: Sequence[int],
    deadband: float,
) -> dict[str, Any]:
    cal_pred, cal_var, cal_nice, _, _ = _natural_scores(
        mean, covariance, z, delta, actions, previous, calibration_indices, deadband
    )
    del cal_pred, cal_var
    eval_pred, eval_var, eval_nice, eval_prior, scale_summary = _natural_scores(
        mean, covariance, z, delta, actions, previous, evaluation_indices, deadband
    )
    cal_groups: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for position, index in enumerate(calibration_indices):
        row = manifest[index]
        cal_groups[str(row["task_identity"])][int(row["demo_id"])].append(float(cal_nice[position]))
    episode_scores_by_task = {
        task: [episode_cluster_score(values) for _, values in sorted(episodes.items())]
        for task, episodes in cal_groups.items()
    }
    calibration = conformal_threshold(episode_scores_by_task, 0.95)
    threshold = float(calibration["threshold"])
    eval_groups: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for position, index in enumerate(evaluation_indices):
        row = manifest[index]
        eval_groups[str(row["task_identity"])][int(row["demo_id"])].append(float(eval_nice[position]))
    task_coverages = {
        task: float(np.mean([episode_cluster_score(values) <= threshold for values in episodes.values()]))
        for task, episodes in eval_groups.items()
    }
    empirical_coverage = float(np.mean(list(task_coverages.values())))

    eval_position = {index: position for position, index in enumerate(evaluation_indices)}
    row_by_key = {str(row["pair_key"]): row for row in rows}
    next_demo = {35: 36, 36: 37, 37: 38, 38: 39, 39: 35}
    cross_lookup = {
        (str(manifest[index]["task_identity"]), int(manifest[index]["demo_id"]), int(manifest[index]["sample_slot"])): index
        for index in evaluation_indices
    }
    transitions = {}
    for index in evaluation_indices:
        transitions[index] = int(
            abs(float(actions[index, 6]) - float(previous[index, 6])) >= deadband
        )
    task_indices: dict[str, list[int]] = defaultdict(list)
    for index in evaluation_indices:
        task_indices[str(manifest[index]["task_identity"])].append(index)

    diagnostics: dict[str, dict[str, dict[str, list[float]]]] = defaultdict(
        lambda: defaultdict(lambda: {"natural_nice": [], "diagnostic_nice": [], "natural_prior": [], "diagnostic_prior": []})
    )
    swapped_requests = []
    for index in evaluation_indices:
        row = manifest[index]
        position = eval_position[index]
        task = str(row["task_identity"])
        prediction = eval_pred[position].astype(np.float32)
        variance = eval_var[position].astype(np.float32)
        natural_nice = float(eval_nice[position])
        natural_prior = float(eval_prior[position])
        feature = _load_feature(Path(row_by_key[str(row["pair_key"])]["feature_path"]))
        if bool(row["temporal_t_plus_20_eligible"]):
            target = feature["delta_z_t_plus_20"].astype(np.float32)
            residual = target - prediction
            diagnostics["temporal_offset"][task]["natural_nice"].append(natural_nice)
            diagnostics["temporal_offset"][task]["diagnostic_nice"].append(float(np.mean(residual * residual / variance)))
            diagnostics["temporal_offset"][task]["natural_prior"].append(natural_prior)
            diagnostics["temporal_offset"][task]["diagnostic_prior"].append(float(_cosine_loss_rows(prediction[None], target[None])[0]))
        cross_index = cross_lookup[(task, next_demo[int(row["demo_id"])], int(row["sample_slot"]))]
        cross_target = np.asarray(delta[cross_index], dtype=np.float32)
        cross_residual = cross_target - prediction
        diagnostics["cross_episode"][task]["natural_nice"].append(natural_nice)
        diagnostics["cross_episode"][task]["diagnostic_nice"].append(float(np.mean(cross_residual * cross_residual / variance)))
        diagnostics["cross_episode"][task]["natural_prior"].append(natural_prior)
        diagnostics["cross_episode"][task]["diagnostic_prior"].append(float(_cosine_loss_rows(prediction[None], cross_target[None])[0]))
        candidates = task_indices[task]
        start = candidates.index(index)
        swap_index = next((candidates[(start + offset) % len(candidates)] for offset in range(1, len(candidates))
                           if transitions[candidates[(start + offset) % len(candidates)]] != transitions[index]), None)
        if swap_index is not None:
            swapped_requests.append((index, swap_index, task, natural_nice, natural_prior))

    import torch

    device = torch.device("cuda")
    with torch.no_grad():
        for start in range(0, len(swapped_requests), BATCH_SIZE):
            request = swapped_requests[start : start + BATCH_SIZE]
            source_indices = np.asarray([item[0] for item in request])
            swap_indices = np.asarray([item[1] for item in request])
            visual = _batch(z, source_indices, device)
            target = _batch(delta, source_indices, device)
            action = _batch(actions, swap_indices, device)
            prior_action = _batch(previous, swap_indices, device)
            prediction = mean(visual, action)
            diagonal, _ = covariance(visual, condition_vector(action, prior_action, deadband))
            score, _, _ = innovation_terms(target - prediction, diagonal)
            prior_score = mean_cosine_per_row(prediction, target)
            for offset, item in enumerate(request):
                task = item[2]
                diagnostics["action_regime"][task]["natural_nice"].append(float(item[3]))
                diagnostics["action_regime"][task]["diagnostic_nice"].append(float(score[offset].item()))
                diagnostics["action_regime"][task]["natural_prior"].append(float(item[4]))
                diagnostics["action_regime"][task]["diagnostic_prior"].append(float(prior_score[offset].item()))

    family_reports = {}
    nice_family = []
    prior_family = []
    for family in ("temporal_offset", "cross_episode", "action_regime"):
        task_reports = {}
        for task in sorted(eval_groups):
            values = diagnostics[family][task]
            if not values["diagnostic_nice"]:
                raise RuntimeError(f"diagnostic family {family} collapsed for {task}")
            nice_auc = auroc_average_ranks(values["natural_nice"], values["diagnostic_nice"])
            prior_auc = auroc_average_ranks(values["natural_prior"], values["diagnostic_prior"])
            task_reports[task] = {"count": len(values["diagnostic_nice"]), "nice_auroc": nice_auc, "prior_auroc": prior_auc}
        family_nice = float(np.mean([value["nice_auroc"] for value in task_reports.values()]))
        family_prior = float(np.mean([value["prior_auroc"] for value in task_reports.values()]))
        nice_family.append(family_nice)
        prior_family.append(family_prior)
        family_reports[family] = {"nice_auroc": family_nice, "prior_auroc": family_prior, "tasks": task_reports}
    aggregate_nice = float(np.mean(nice_family))
    aggregate_prior = float(np.mean(prior_family))
    score_std = {
        task: float(np.std([eval_nice[eval_position[index]] for index in evaluation_indices if manifest[index]["task_identity"] == task]))
        for task in sorted(eval_groups)
    }
    return {
        "calibration": calibration,
        "task_empirical_coverage": task_coverages,
        "empirical_coverage": empirical_coverage,
        "coverage_absolute_error": abs(empirical_coverage - 0.95),
        "scale_summary": scale_summary,
        "natural_score_std_by_task": score_std,
        "diagnostic_families": family_reports,
        "aggregate_nice_auroc": aggregate_nice,
        "aggregate_prior_auroc": aggregate_prior,
        "aggregate_auroc_margin": aggregate_nice - aggregate_prior,
        "passed": bool(
            abs(empirical_coverage - 0.95) <= 0.03
            and aggregate_nice > 0.60
            and aggregate_nice - aggregate_prior >= 0.03
            and all(value > 1e-6 for value in score_std.values())
            and scale_summary["clamped_fraction"] < 0.05
        ),
    }


def _preflight(paths: Mapping[str, Path]) -> dict[str, Any]:
    import torch

    required = (
        paths["checkpoint"], paths["data_root"], paths["source_root"], VLM_PATH,
        PROPOSAL_FILE, STAGE0A_RESULT, STAGE0B_PROTOCOL, STAGE0B_CLARIFICATION,
    )
    missing = [str(path) for path in required if not path.exists()]
    partial_error = None
    partial_summary = None
    if paths["partial"].is_file():
        try:
            partial = json.loads(paths["partial"].read_text(encoding="utf-8-sig"))
            partial_summary = {key: partial.get(key) for key in ("planned_pair_count", "completed_pair_count", "exception_count")}
        except Exception as exc:
            partial_error = f"{type(exc).__name__}: {exc}"
    stage0a = json.loads(STAGE0A_RESULT.read_text(encoding="utf-8-sig")) if STAGE0A_RESULT.is_file() else {}
    registry = json.loads(RESOURCE_REGISTRY.read_text(encoding="utf-8-sig")) if RESOURCE_REGISTRY.is_file() else {"intervals": []}
    workers = _active_linux_workers()
    return {
        "passed": bool(
            not missing and torch.cuda.is_available() and not workers and partial_error is None
            and not paths["result_json"].exists() and _proposal_hash_text() == PROPOSAL_HASH
            and _sha256(PROPOSAL_FILE) == PROPOSAL_HASH
            and stage0a.get("final_decision") == "NICE_STAGE_0A_PASS_STAGE_0B_ALLOWED"
        ),
        "missing_paths": missing,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "active_linux_workers": workers,
        "partial_parse_error": partial_error,
        "partial_summary": partial_summary,
        "result_absent": not paths["result_json"].exists(),
        "proposal_hash_ok": _proposal_hash_text() == PROPOSAL_HASH and _sha256(PROPOSAL_FILE) == PROPOSAL_HASH,
        "stage0a_passed": stage0a.get("final_decision") == "NICE_STAGE_0A_PASS_STAGE_0B_ALLOWED",
        "resource_evidence": _resource_evidence(registry, time.time()),
    }


def _write_md(path: Path, result: Mapping[str, Any]) -> None:
    mechanism = result["mechanism_audit"]
    mean = result["mean_headroom"]["aggregate"]
    lines = [
        "# NICE-VLA Stage 0B1 Result", "",
        f"Decision: `{result['final_decision']}`.", "",
        f"Pairs: `{result['completed_pair_count']} / {result['planned_pair_count']}`; exceptions: `{result['exception_count']}`.",
        f"Mean cosine loss: `{mean['full_cosine_loss']}`; suite-mean: `{mean['suite_mean_cosine_loss']}`; zero: `1.0`.",
        f"Empirical episode coverage: `{mechanism['empirical_coverage']}`.",
        f"NICE diagnostic AUROC: `{mechanism['aggregate_nice_auroc']}`.",
        f"Prior diagnostic AUROC: `{mechanism['aggregate_prior_auroc']}`.",
        f"AUROC margin: `{mechanism['aggregate_auroc_margin']}`.", "",
        "No simulator rollout, task outcome, reward, done, reset identity, or confirmatory task was read.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    paths = _paths(args)
    paths["report"].mkdir(parents=True, exist_ok=True)
    paths["run"].mkdir(parents=True, exist_ok=True)
    preflight = _preflight(paths)
    _write_json(paths["preflight"], preflight)
    if not preflight["passed"]:
        raise RuntimeError(f"Stage 0B1 preflight failed: {preflight}")
    _write_text(paths["pid"], f"{os.getpid()}\n")
    heartbeat = {"pid": os.getpid(), "status": "running", "planned_pair_count": PLANNED_PAIRS,
                 "completed_pair_count": 0, "exception_count": 0, "phase": "manifest"}
    _write_json(paths["status"], {**heartbeat, "started_at": _utc_now()})
    _write_json(paths["heartbeat"], {**heartbeat, "updated_at": _utc_now()})
    stop = threading.Event()
    thread = threading.Thread(target=_heartbeat_loop, args=(paths["heartbeat"], heartbeat, stop), daemon=True)
    thread.start()
    started = time.time()
    rows: list[dict[str, Any]] = []
    try:
        _set_offline_environment()
        source = _source_map(paths["source_root"])
        manifest, manifest_summary = _build_manifest(paths["data_root"])
        manifest_payload = {"method": "NICE-VLA", "stage": "0B1", "proposal_hash": PROPOSAL_HASH,
                            "rows": manifest, **manifest_summary}
        manifest_hash = canonical_json_sha256(manifest_payload)
        manifest_payload["manifest_hash"] = manifest_hash
        _write_json(paths["manifest"], manifest_payload)
        prior_exceptions = 0
        if paths["partial"].is_file():
            prior_exceptions = int(json.loads(paths["partial"].read_text(encoding="utf-8-sig")).get("exception_count", 0))
        rows = _load_resume(paths, manifest, manifest_hash)
        resumed_count = len(rows)
        completed = {str(row["pair_key"]) for row in rows}
        heartbeat["completed_pair_count"] = len(rows)
        heartbeat["phase"] = "extraction"
        policy, _, preprocessor, postprocessor = _load_policy_and_processors(paths["checkpoint"])
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        policy.eval()
        for manifest_row in manifest:
            key = str(manifest_row["pair_key"])
            if key in completed:
                continue
            feature_path = _feature_path(paths["feature_dir"], key)
            values = _extract_pair(policy, manifest_row)
            _save_feature(feature_path, **values)
            rows.append(_row_summary(manifest_row, feature_path, values))
            completed.add(key)
            heartbeat["completed_pair_count"] = len(rows)
            _write_json(paths["partial"], _partial_payload(manifest_hash, rows, prior_exceptions))
            if len(rows) % 16 == 0:
                print(f"[nice-stage0b1] extracted {len(rows)}/{PLANNED_PAIRS}", flush=True)
        passthrough_row = {**manifest[0], "episode": manifest[0]["demo_id"], "frame": manifest[0]["frame_t"]}
        passthrough = _base_passthrough(policy, preprocessor, postprocessor, passthrough_row)
        del policy, preprocessor, postprocessor
        gc.collect()
        torch.cuda.empty_cache()

        heartbeat["phase"] = "materialize"
        _materialize_memmaps(paths, manifest, rows)
        z = np.load(paths["z_memmap"], mmap_mode="r")
        delta = np.load(paths["delta_memmap"], mmap_mode="r")
        actions = np.load(paths["action_memmap"], mmap_mode="r")
        previous = np.load(paths["previous_memmap"], mmap_mode="r")
        discovery_indices = np.asarray([i for i, row in enumerate(manifest) if row["role"] == "discovery"])
        calibration_indices = np.asarray([i for i, row in enumerate(manifest) if row["role"] == "validation_calibration"])
        evaluation_indices = np.asarray([i for i, row in enumerate(manifest) if row["role"] == "validation_evaluation"])
        episode_actions: dict[tuple[str, int], list[tuple[int, np.ndarray]]] = defaultdict(list)
        for index in discovery_indices:
            row = manifest[index]
            episode_actions[(str(row["task_identity"]), int(row["demo_id"]))].append((int(row["frame_t"]), actions[index]))
        deadband = discovery_gripper_deadband([
            np.stack([value for _, value in sorted(items)]) for items in episode_actions.values()
        ])

        heartbeat["phase"] = "mean_training"
        mean, mean_training = _fit_mean(paths, z, delta, actions, discovery_indices)
        mean_headroom = _mean_headroom(mean, manifest, z, delta, actions, discovery_indices, evaluation_indices)
        heartbeat["phase"] = "covariance_training"
        covariance, covariance_training = _fit_covariance(
            paths, mean, z, delta, actions, previous, discovery_indices, deadband
        )
        heartbeat["phase"] = "rank8_audit"
        episode_residual_means = _episode_means(mean, manifest, z, delta, actions, discovery_indices)
        basis = deterministic_pca_basis(episode_residual_means, LOW_RANK)
        np.save(paths["basis"], basis.float().cpu().numpy())
        basis_audit = {
            "episode_count": int(episode_residual_means.shape[0]),
            "shape": list(basis.shape),
            "orthonormal_max_abs_error": float(
                torch.max(torch.abs(basis.T @ basis - torch.eye(LOW_RANK, device=basis.device))).item()
            ),
            "sha256": _sha256(paths["basis"]),
        }
        del episode_residual_means, basis
        torch.cuda.empty_cache()
        heartbeat["phase"] = "calibration_diagnostics"
        mechanism = _calibration_and_diagnostics(
            mean, covariance, manifest, rows, z, delta, actions, previous,
            calibration_indices, evaluation_indices, deadband,
        )
        manifest_audit = validate_manifest(
            manifest,
            rows,
            allowed_partitions=("discovery", "validation_calibration", "validation_evaluation"),
        )
        role_task_counts = defaultdict(int)
        for row in manifest:
            role_task_counts[f"{row['role']}|{row['task_identity']}"] += 1
        quota_passed = bool(
            all(value == 192 for key, value in role_task_counts.items() if key.startswith("discovery|"))
            and all(value == 80 for key, value in role_task_counts.items() if key.startswith("validation_"))
        )
        source_passed = bool(source["commit_matches"] and source["license_matches"] and not source["missing"])
        implementation_passed = bool(
            manifest_audit["passed"] and len(rows) == PLANNED_PAIRS and quota_passed and source_passed
            and mean_training["first_gradient_norm"] > 0 and covariance_training["first_gradient_norm"] > 0
            and covariance_training["frozen_mean_gradient_norm"] == 0.0
            and mean_training["reload_max_abs_error"] <= 1e-6
            and covariance_training["reload_max_abs_error"] <= 1e-6
            and basis_audit["orthonormal_max_abs_error"] <= 1e-5
            and passthrough["action_identity_max_abs_error"] == 0.0
            and float(torch.cuda.max_memory_allocated() / 1024**3) <= 15.5
        )
        scientific_gates_passed = bool(mean_headroom["aggregate"]["passed"] and mechanism["passed"])
        if not source_passed or not quota_passed:
            decision = "NICE_STAGE_0B1_DATA_FAILURE"
        elif not implementation_passed:
            decision = "NICE_STAGE_0B1_IMPLEMENTATION_FAILURE"
        elif not scientific_gates_passed:
            decision = "NICE_STAGE_0B1_DESIGN_FAILURE_NONOBSERVABLE"
        else:
            decision = "NICE_STAGE_0B1_PASS_STAGE_0B2_HEADROOM_ALLOWED"
        result = {
            "method": "NICE-VLA", "stage": "0B1", "proposal_hash": PROPOSAL_HASH,
            "source_commit": SOURCE_COMMIT, "implementation_commit": None,
            "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(), "completed_at": _utc_now(),
            "pid": os.getpid(), "planned_pair_count": PLANNED_PAIRS, "completed_pair_count": len(rows),
            "resumed_pair_count": resumed_count, "new_pair_count": len(rows) - resumed_count,
            "exception_count": 0, "prior_attempt_exception_count": prior_exceptions,
            "manifest_hash": manifest_hash, "manifest_audit": manifest_audit,
            "role_counts": manifest_summary["role_counts"], "role_task_counts": dict(role_task_counts),
            "quota_passed": quota_passed, "source_audit": source, "gripper_deadband": deadband,
            "mean_training": mean_training, "mean_headroom": mean_headroom,
            "covariance_training": covariance_training, "rank8_basis_audit": basis_audit,
            "mechanism_audit": mechanism, "base_passthrough": passthrough,
            "implementation_passed": implementation_passed, "scientific_gates_passed": scientific_gates_passed,
            "peak_cuda_gib": float(torch.cuda.max_memory_allocated() / 1024**3),
            "validation_pair_records_read": CALIBRATION_PAIRS + EVALUATION_PAIRS,
            "confirmatory_records_read": 0, "task_outcome_read_count": 0, "reward_read_count": 0,
            "done_read_count": 0, "reset_identity_read_count": 0, "simulator_rollout_count": 0,
            "validation_search_happened": False, "closed_loop_experiment_happened": False,
            "confirmatory_test_tuning_happened": False,
            "resource_evidence": preflight["resource_evidence"],
            "elapsed_seconds_not_paper_evidence": time.time() - started,
            "final_decision": decision,
            "stage_0b2_allowed": decision == "NICE_STAGE_0B1_PASS_STAGE_0B2_HEADROOM_ALLOWED",
        }
        _write_json(paths["result_json"], result)
        _write_md(paths["result_md"], result)
        validation = {
            "result_json_parsed": True,
            "proposal_hash_recomputed": _sha256(PROPOSAL_FILE) == PROPOSAL_HASH,
            "manifest_hash_recomputed": canonical_json_sha256(
                {key: value for key, value in manifest_payload.items() if key != "manifest_hash"}
            ) == manifest_hash,
            "manifest_audit": manifest_audit,
            "partial_json_parsed": bool(json.loads(paths["partial"].read_text(encoding="utf-8-sig"))),
            "worker_completed": True, "final_decision": decision,
        }
        _write_json(paths["validation"], validation)
        heartbeat.update({"status": "completed", "phase": "completed"})
        _write_json(paths["status"], {**heartbeat, "completed_at": _utc_now(), "final_decision": decision})
        _write_json(paths["heartbeat"], {**heartbeat, "updated_at": _utc_now(), "final_decision": decision})
        return result
    except Exception:
        heartbeat.update({"status": "failed", "exception_count": 1, "completed_pair_count": len(rows)})
        if paths["partial"].is_file():
            try:
                partial = json.loads(paths["partial"].read_text(encoding="utf-8-sig"))
                partial["exception_count"] = int(partial.get("exception_count", 0)) + 1
                partial["last_exception"] = traceback.format_exc()
                _write_json(paths["partial"], partial)
            except Exception:
                pass
        _write_json(paths["status"], {**heartbeat, "failed_at": _utc_now()})
        _write_json(paths["heartbeat"], {**heartbeat, "updated_at": _utc_now()})
        raise
    finally:
        stop.set()
        thread.join(timeout=2.0)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stage0b1",), default="stage0b1")
    parser.add_argument("--checkpoint", default=str(_asset_path("checkpoints", "smolvla_libero")))
    parser.add_argument("--libero-data-root", default=str(_asset_path("data", "libero")))
    parser.add_argument("--vla-corrector-source", default=str(_default_source_root()))
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "nice_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "nice_vla" / "stage0b1"))
    return parser.parse_args()


def main() -> int:
    args = _arguments()
    os.chdir(REPO_ROOT)
    import torch

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
        torch.cuda.reset_peak_memory_stats()
    paths = _paths(args)
    try:
        result = run(args)
        print(json.dumps({"final_decision": result["final_decision"]}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        blocker = {
            "method": "NICE-VLA", "stage": "0B1", "proposal_hash": PROPOSAL_HASH,
            "final_decision": "NICE_STAGE_0B1_IMPLEMENTATION_FAILURE",
            "exception_type": type(exc).__name__, "exception": str(exc), "traceback": traceback.format_exc(),
            "scientific_kill": False, "protocol_change_allowed": False,
        }
        _write_json(paths["blocker"], blocker)
        print(json.dumps(blocker, sort_keys=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
