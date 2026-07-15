"""Run the frozen NICE-VLA Stage 0A source, latent, and math audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
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
    _clone_batch,
    _load_policy_and_processors,
    _preprocess,
    _raw_sample,
    _resource_evidence,
    _set_offline_environment,
)
from scripts.run_pcav_vla_stage0 import _postprocess_chunk  # noqa: E402
from tca_map.smolvla.nice_vla import (  # noqa: E402
    ACTION_DIM,
    K_STEP,
    LOW_RANK,
    PROPOSAL_HASH,
    VARIANCE_CEILING,
    VARIANCE_FLOOR,
    Stage0DecisionInputs,
    TinyCovariance,
    TinyResidualMean,
    action_validity,
    canonical_json_sha256,
    classify_stage0a,
    condition_vector,
    conformal_threshold,
    covariance_nll,
    dense_innovation_reference,
    deterministic_pca_basis,
    discovery_gripper_deadband,
    episode_cluster_score,
    innovation_terms,
    mean_cosine_loss,
    pair_key,
    passthrough_queue_action,
    validate_manifest,
)


SOURCE_COMMIT = "9d23a0ba6fad562d3ed1a68fc52c8a12459abb41"
SEED = 20262011
EXPECTED_PAIRS = 128
FRAMES_PER_DEMO = 32
MEAN_STEPS = 20
COVARIANCE_STEPS = 20
PROPOSAL_FILE = REPO_ROOT / "reports" / "nice_vla" / "researcher_proposal.md"
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "nice_vla" / "proposal_hash.txt"
RESOURCE_REGISTRY = REPO_ROOT / "reports" / "resource_contention_intervals.json"


def _asset_path(*parts: str) -> Path:
    root = Path("C:/assets") if os.name == "nt" else Path("/mnt/c/assets")
    return root.joinpath(*parts)


def _default_source_root() -> Path:
    if os.name == "nt":
        return Path(r"C:\Users\jiheo\AppData\Local\Temp\vla-corrector-cycle20")
    return Path("/mnt/c/Users/jiheo/AppData/Local/Temp/vla-corrector-cycle20")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _edge_hash(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if size > 1024 * 1024:
            handle.seek(max(0, size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest().upper()


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
        "proposal": PROPOSAL_FILE,
        "proposal_hash": PROPOSAL_HASH_FILE,
        "pid": report / "stage_0a_pid.txt",
        "heartbeat": report / "stage_0a_heartbeat.json",
        "status": report / "stage_0a_status.json",
        "partial": report / "stage_0a_partial.json",
        "manifest": report / "stage_0a_pair_manifest.json",
        "preflight": report / "stage_0a_preflight.json",
        "result_json": report / "stage_0a_result.json",
        "result_md": report / "stage_0a_result.md",
        "validation": report / "stage_0a_validation.json",
        "blocker": report / "stage_0a_implementation_blocker.json",
        "smoke_checkpoint": run / "nice_stage0a_smoke.pt",
    }


def _heartbeat_loop(path: Path, state: dict[str, Any], stop: threading.Event) -> None:
    while not stop.wait(10.0):
        _write_json(path, {**state, "updated_at": _utc_now()})


def _proposal_hash_text() -> str:
    tokens = PROPOSAL_HASH_FILE.read_text(encoding="utf-8").strip().split()
    return tokens[1] if len(tokens) >= 2 and tokens[0].upper() == "SHA256" else ""


def _source_map(source_root: Path) -> dict[str, Any]:
    relative = {
        "pair_construction": "src/siglip_dynamics/dataset.py",
        "mean_model": "src/siglip_dynamics/MLP.py",
        "training_objective": "src/siglip_dynamics/train.py",
        "circuit_breaker": "src/siglip_dynamics/inference/circuit_breaker.py",
    }
    files = {name: source_root / path for name, path in relative.items()}
    files["license"] = source_root / "LICENSE"
    missing = [str(path) for path in files.values() if not path.is_file()]
    commit = None
    if not missing:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=source_root, check=True, capture_output=True, text=True
        )
        commit = completed.stdout.strip()
    license_text = files["license"].read_text(encoding="utf-8", errors="replace") if files["license"].is_file() else ""
    return {
        "repository": "https://github.com/ZJU-OmniAI/vla-corrector",
        "expected_commit": SOURCE_COMMIT,
        "observed_commit": commit,
        "commit_matches": commit == SOURCE_COMMIT,
        "license": "Apache-2.0",
        "license_matches": "Apache License" in license_text and "Version 2.0" in license_text,
        "missing": missing,
        "files": {
            name: {"path": str(path), "sha256": _sha256(path)}
            for name, path in files.items()
            if path.is_file()
        },
        "symbols": {
            "pair_construction": "SiglipDynamicsDataset._load_quantized_frame_cache",
            "mean_model": "SiglipResidualMLP",
            "training_objective": "_cosine_loss",
            "circuit_breaker": "CircuitBreaker.compute_error/get_dynamic_threshold/check",
        },
    }


def _resolve_task(data_root: Path, suite: str, task_index: int) -> Path:
    files = sorted((data_root / suite).glob("*.hdf5"), key=lambda path: path.name)
    if task_index < 0 or task_index >= len(files):
        raise RuntimeError(f"cannot resolve {suite}/task_{task_index}: found {len(files)} files")
    return files[task_index]


def _sample_frames(length: int) -> list[int]:
    valid_count = length - K_STEP
    if valid_count < FRAMES_PER_DEMO:
        return list(range(max(0, valid_count)))
    values = np.floor(np.linspace(0, valid_count - 1, FRAMES_PER_DEMO)).astype(int).tolist()
    if len(values) != len(set(values)):
        raise RuntimeError("fixed frame sampler produced duplicate indices")
    return values


def _problem_language(data: Any) -> str:
    raw = data.attrs.get("problem_info", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        parsed = json.loads(str(raw))
        return str(parsed.get("language_instruction") or parsed.get("language") or "")
    except json.JSONDecodeError:
        return str(raw)


def _build_manifest(paths: Mapping[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any], list[np.ndarray]]:
    import h5py

    specifications = (("libero_10", 1), ("libero_goal", 1))
    manifest: list[dict[str, Any]] = []
    tasks = []
    action_episodes: list[np.ndarray] = []
    for task_number, (suite, task_index) in enumerate(specifications):
        source = _resolve_task(paths["data_root"], suite, task_index)
        with h5py.File(source, "r") as handle:
            data = handle["data"]
            language = _problem_language(data)
            demo_reports = []
            for demo_id in (0, 1):
                demo = data[f"demo_{demo_id}"]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
                    raise RuntimeError(f"unexpected action shape {actions.shape} in {source} demo_{demo_id}")
                observations = demo["obs"]
                required = ("agentview_rgb", "eye_in_hand_rgb")
                if any(name not in observations for name in required):
                    raise RuntimeError(f"missing image stream in {source} demo_{demo_id}")
                length = len(actions)
                if len(observations["agentview_rgb"]) != length or len(observations["eye_in_hand_rgb"]) != length:
                    raise RuntimeError("image/action length mismatch")
                frames = _sample_frames(length)
                action_episodes.append(actions)
                demo_reports.append({"demo_id": demo_id, "length": length, "sampled_frames": frames})
                for frame in frames:
                    row = {
                        "partition": "discovery",
                        "suite": suite,
                        "task_id": task_index,
                        "task_identity": f"{suite}/task_{task_index}",
                        "task_index": task_number,
                        "task_language": language,
                        "source_path": str(source),
                        "demo_id": demo_id,
                        "episode": demo_id,
                        "frame_t": frame,
                        "frame_t_plus_10": frame + K_STEP,
                        "frame": frame,
                    }
                    row["pair_key"] = pair_key(row)
                    manifest.append(row)
        stat = source.stat()
        tasks.append(
            {
                "suite": suite,
                "task_id": task_index,
                "task_identity": f"{suite}/task_{task_index}",
                "resolved_zero_indexed_sorted_filename": source.name,
                "source_path": str(source),
                "source_size_bytes": stat.st_size,
                "source_mtime_ns": stat.st_mtime_ns,
                "source_edge_sha256": _edge_hash(source),
                "task_language": language,
                "demonstrations": demo_reports,
            }
        )
    return manifest, {"tasks": tasks, "planned_pair_count": len(manifest)}, action_episodes


def _feature_path(feature_dir: Path, key: str) -> Path:
    return feature_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest().upper()}.npz"


def _load_feature(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as loaded:
        return {name: np.asarray(loaded[name]) for name in loaded.files}


def _save_feature(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def _prepare_source_images(policy: Any, images: Sequence[Any]) -> Any:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

    tensors = []
    for image in images:
        value = image if torch.is_tensor(image) else torch.as_tensor(np.asarray(image).copy())
        if value.ndim != 3:
            raise RuntimeError(f"expected CHW/HWC image, got {tuple(value.shape)}")
        if value.shape[-1] in (1, 3):
            value = value.permute(2, 0, 1)
        value = value.to(dtype=torch.float32)
        if float(value.max()) > 1.0:
            value = value / 255.0
        tensors.append(value)
    batch = torch.stack(tensors).to("cuda")
    resize_cfg = getattr(policy.config, "resize_imgs_with_padding", None)
    if resize_cfg is not None:
        batch = resize_with_pad(batch, *resize_cfg, pad_value=0)
    dtype = next(policy.model.parameters()).dtype
    return (batch * 2.0 - 1.0).to(dtype=dtype)


def _extract_pair(policy: Any, row: Mapping[str, Any]) -> dict[str, np.ndarray]:
    import h5py
    import torch

    with h5py.File(str(row["source_path"]), "r") as handle:
        demo = handle["data"][f"demo_{int(row['demo_id'])}"]
        observations = demo["obs"]
        current = int(row["frame_t"])
        future = int(row["frame_t_plus_10"])
        current_agent, current_wrist = _apply_official_env_image_processor(
            observations["agentview_rgb"][current], observations["eye_in_hand_rgb"][current]
        )
        future_agent, future_wrist = _apply_official_env_image_processor(
            observations["agentview_rgb"][future], observations["eye_in_hand_rgb"][future]
        )
        actions = np.asarray(demo["actions"], dtype=np.float32)
        action = actions[current]
        previous = actions[current - 1] if current > 0 else action
    prepared = _prepare_source_images(policy, (current_agent, current_wrist, future_agent, future_wrist))
    with torch.no_grad():
        tokens = policy.model.vlm_with_expert.embed_image(prepared).float().cpu()
    if tokens.ndim != 3 or tokens.shape[0] != 4:
        raise RuntimeError(f"unexpected source visual token shape {tuple(tokens.shape)}")
    current_tokens = torch.cat((tokens[0], tokens[1]), dim=0).numpy()
    future_tokens = torch.cat((tokens[2], tokens[3]), dim=0).numpy()
    return {
        "z_t": current_tokens.astype(np.float16),
        "delta_z": (future_tokens - current_tokens).astype(np.float16),
        "action": action.astype(np.float32),
        "previous_action": previous.astype(np.float32),
    }


def _row_summary(row: Mapping[str, Any], feature_path: Path, values: Mapping[str, np.ndarray]) -> dict[str, Any]:
    z_t = np.asarray(values["z_t"], dtype=np.float32)
    delta = np.asarray(values["delta_z"], dtype=np.float32)
    action = np.asarray(values["action"], dtype=np.float32)
    return {
        **dict(row),
        "feature_path": str(feature_path),
        "latent_shape": list(z_t.shape),
        "latent_finite_fraction": float(np.mean(np.isfinite(z_t))),
        "delta_finite_fraction": float(np.mean(np.isfinite(delta))),
        "delta_variance": float(np.var(delta)),
        "action_finite_fraction": float(np.mean(np.isfinite(action))),
        "action_min": float(np.min(action)),
        "action_max": float(np.max(action)),
        "feature_sha256": _sha256(feature_path),
    }


def _partial_payload(manifest_hash: str, rows: Sequence[Mapping[str, Any]], exception_count: int = 0) -> dict[str, Any]:
    return {
        "method": "NICE-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "planned_pair_count": EXPECTED_PAIRS,
        "completed_pair_count": len(rows),
        "completed_pair_keys": [str(row["pair_key"]) for row in rows],
        "rows": list(rows),
        "exception_count": int(exception_count),
        "updated_at": _utc_now(),
    }


def _load_resume(paths: Mapping[str, Path], manifest: Sequence[Mapping[str, Any]], manifest_hash: str) -> list[dict[str, Any]]:
    if not paths["partial"].is_file():
        return []
    partial = _read_json(paths["partial"])
    if partial.get("proposal_hash") != PROPOSAL_HASH or partial.get("manifest_hash") != manifest_hash:
        raise RuntimeError("partial result identity does not match frozen proposal/manifest")
    rows = list(partial.get("rows") or [])
    audit = validate_manifest(manifest, rows)
    if audit["duplicate_result_key_count"] or audit["extra_result_key_count"]:
        raise RuntimeError(f"partial result key audit failed: {audit}")
    for row in rows:
        path = Path(str(row.get("feature_path", "")))
        if not path.is_file() or _sha256(path) != row.get("feature_sha256"):
            raise RuntimeError(f"missing or changed feature cache for {row.get('pair_key')}")
    return rows


def _gradient_norm(parameters: Sequence[Any]) -> float:
    squares = []
    for parameter in parameters:
        if parameter.grad is not None:
            squares.append(float(torch_sum_square(parameter.grad)))
    return float(np.sqrt(sum(squares)))


def torch_sum_square(value: Any) -> float:
    return float((value.detach().float() ** 2).sum().item())


def _tiny_smoke(features: Sequence[Mapping[str, np.ndarray]], deadband: float, checkpoint_path: Path) -> dict[str, Any]:
    import torch

    device = torch.device("cuda")
    z_all = torch.from_numpy(np.stack([row["z_t"] for row in features]).astype(np.float32)).to(device)
    y_all = torch.from_numpy(np.stack([row["delta_z"] for row in features]).astype(np.float32)).to(device)
    a_all = torch.from_numpy(np.stack([row["action"] for row in features]).astype(np.float32)).to(device)
    p_all = torch.from_numpy(np.stack([row["previous_action"] for row in features]).astype(np.float32)).to(device)
    batch = slice(0, 8)
    mean = TinyResidualMean(z_all.shape[-1]).to(device)
    optimizer = torch.optim.AdamW(mean.parameters(), lr=1e-3, weight_decay=0.0)
    before_prediction = mean(z_all[batch], a_all[batch])
    mean_before = float(mean_cosine_loss(before_prediction, y_all[batch]).item())
    mse_before = float(torch.mean((before_prediction - y_all[batch]) ** 2).item())
    optimizer.zero_grad(set_to_none=True)
    mean_loss = mean_cosine_loss(mean(z_all[batch], a_all[batch]), y_all[batch])
    mean_loss.backward()
    mean_gradient = _gradient_norm(list(mean.parameters()))
    optimizer.step()
    for _ in range(MEAN_STEPS - 1):
        optimizer.zero_grad(set_to_none=True)
        loss = mean_cosine_loss(mean(z_all[batch], a_all[batch]), y_all[batch])
        loss.backward()
        optimizer.step()
    mean_after = float(mean_cosine_loss(mean(z_all[batch], a_all[batch]), y_all[batch]).item())
    for parameter in mean.parameters():
        parameter.grad = None
        parameter.requires_grad_(False)
    with torch.no_grad():
        residual_all = y_all - mean(z_all, a_all)
    condition = condition_vector(a_all, p_all, deadband)
    diagonal_model = TinyCovariance(z_all.shape[-1], rank=0).to(device)
    diagonal_optimizer = torch.optim.AdamW(diagonal_model.parameters(), lr=1e-3, weight_decay=0.0)
    diagonal, _ = diagonal_model(z_all[batch].detach(), condition[batch].detach())
    covariance_before_tensor = covariance_nll(residual_all[batch].detach(), diagonal)
    covariance_before = float(covariance_before_tensor.item())
    diagonal_optimizer.zero_grad(set_to_none=True)
    covariance_before_tensor.backward()
    covariance_gradient = _gradient_norm(list(diagonal_model.parameters()))
    frozen_mean_gradient = _gradient_norm(list(mean.parameters()))
    diagonal_optimizer.step()
    for _ in range(COVARIANCE_STEPS - 1):
        diagonal_optimizer.zero_grad(set_to_none=True)
        diagonal, _ = diagonal_model(z_all[batch].detach(), condition[batch].detach())
        loss = covariance_nll(residual_all[batch].detach(), diagonal)
        loss.backward()
        diagonal_optimizer.step()
    diagonal_after, _ = diagonal_model(z_all[batch].detach(), condition[batch].detach())
    covariance_after = float(covariance_nll(residual_all[batch].detach(), diagonal_after).item())
    basis = deterministic_pca_basis(residual_all[:16].detach(), LOW_RANK)
    rank_model = TinyCovariance(z_all.shape[-1], rank=LOW_RANK).to(device)
    rank_model.zero_grad(set_to_none=True)
    rank_diagonal, rank_values = rank_model(z_all[batch].detach(), condition[batch].detach())
    if rank_values is None:
        raise AssertionError("rank model omitted rank scales")
    rank_loss = covariance_nll(
        residual_all[batch].detach(), rank_diagonal, basis=basis, rank_variance=rank_values
    )
    rank_loss.backward()
    rank_gradient = _gradient_norm(list(rank_model.parameters()))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "proposal_hash": PROPOSAL_HASH,
            "token_dim": int(z_all.shape[-1]),
            "mean": mean.state_dict(),
            "diagonal": diagonal_model.state_dict(),
            "rank": rank_model.state_dict(),
            "basis": basis.detach().cpu(),
        },
        checkpoint_path,
    )
    loaded = torch.load(checkpoint_path, map_location=device, weights_only=True)
    reloaded_mean = TinyResidualMean(int(loaded["token_dim"])).to(device)
    reloaded_mean.load_state_dict(loaded["mean"])
    reloaded_diagonal = TinyCovariance(int(loaded["token_dim"]), rank=0).to(device)
    reloaded_diagonal.load_state_dict(loaded["diagonal"])
    reloaded_rank = TinyCovariance(int(loaded["token_dim"]), rank=LOW_RANK).to(device)
    reloaded_rank.load_state_dict(loaded["rank"])
    with torch.no_grad():
        reference_mean = mean(z_all[batch], a_all[batch])
        loaded_mean = reloaded_mean(z_all[batch], a_all[batch])
        reference_diag, _ = diagonal_model(z_all[batch], condition[batch])
        loaded_diag, _ = reloaded_diagonal(z_all[batch], condition[batch])
        reference_rank = rank_model(z_all[batch], condition[batch])
        loaded_rank = reloaded_rank(z_all[batch], condition[batch])
    reload_error = max(
        float(torch.max(torch.abs(reference_mean - loaded_mean)).item()),
        float(torch.max(torch.abs(reference_diag - loaded_diag)).item()),
        float(torch.max(torch.abs(reference_rank[0] - loaded_rank[0])).item()),
        float(torch.max(torch.abs(reference_rank[1] - loaded_rank[1])).item()),
        float(torch.max(torch.abs(basis.cpu() - loaded["basis"].cpu())).item()),
    )
    all_scales = torch.cat((diagonal_after.flatten(), rank_diagonal.flatten(), rank_values.flatten()))
    clamped = torch.mean(((all_scales >= VARIANCE_CEILING) | (all_scales <= VARIANCE_FLOOR)).float())
    finite_gradients = all(
        math_is_finite_positive(value) for value in (mean_gradient, covariance_gradient, rank_gradient)
    )
    return {
        "latent_shape": [int(z_all.shape[1]), int(z_all.shape[2])],
        "action_width": int(a_all.shape[1]),
        "mean_loss_before": mean_before,
        "mean_mse_before": mse_before,
        "mean_loss_after": mean_after,
        "covariance_nll_before": covariance_before,
        "covariance_nll_after": covariance_after,
        "rank8_nll": float(rank_loss.item()),
        "mean_gradient_norm": mean_gradient,
        "covariance_gradient_norm": covariance_gradient,
        "rank8_gradient_norm": rank_gradient,
        "frozen_mean_gradient_norm_during_covariance": frozen_mean_gradient,
        "intended_gradients_finite_nonzero": finite_gradients,
        "frozen_gradients_zero": frozen_mean_gradient == 0.0,
        "variance_min": float(all_scales.min().item()),
        "variance_median": float(all_scales.median().item()),
        "variance_max": float(all_scales.max().item()),
        "variance_clamped_fraction": float(clamped.item()),
        "basis_shape": list(basis.shape),
        "basis_orthonormal_max_abs_error": float(
            torch.max(torch.abs(basis.T @ basis - torch.eye(LOW_RANK, device=device))).item()
        ),
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _sha256(checkpoint_path),
        "checkpoint_reload_max_abs_error": reload_error,
    }


def math_is_finite_positive(value: float) -> bool:
    return bool(np.isfinite(value) and value > 0.0)


def _algebra_smoke() -> dict[str, Any]:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    residual = torch.randn((3, 4, 4), generator=generator, dtype=torch.float64)
    diagonal = torch.rand((3, 16), generator=generator, dtype=torch.float64) + 0.2
    raw_basis = torch.randn((16, LOW_RANK), generator=generator, dtype=torch.float64)
    basis, _ = torch.linalg.qr(raw_basis)
    rank_values = torch.rand((3, LOW_RANK), generator=generator, dtype=torch.float64) + 0.1
    _, fast_mahal, fast_logdet = innovation_terms(
        residual, diagonal, basis=basis, rank_variance=rank_values
    )
    dense_mahal, dense_logdet = dense_innovation_reference(
        residual, diagonal, basis=basis, rank_variance=rank_values
    )
    eigenvalues = []
    for index in range(3):
        covariance = torch.diag(diagonal[index]) + basis @ torch.diag(rank_values[index]) @ basis.T
        eigenvalues.append(float(torch.linalg.eigvalsh(covariance).min().item()))
    return {
        "residual_width": 16,
        "rank": LOW_RANK,
        "mahalanobis_max_abs_error": float(torch.max(torch.abs(fast_mahal - dense_mahal)).item()),
        "logdet_max_abs_error": float(torch.max(torch.abs(fast_logdet - dense_logdet)).item()),
        "minimum_dense_eigenvalue": min(eigenvalues),
    }


def _calibration_smoke() -> dict[str, Any]:
    cluster = episode_cluster_score(list(range(1, 21)))
    fixtures = {
        str(coverage): conformal_threshold(
            {"task_a": [1.0, 2.0, 2.0, 3.0], "task_b": [1.5, 2.0, 4.0, 5.0]}, coverage
        )
        for coverage in (0.90, 0.95, 0.975)
    }
    return {
        "episode_nearest_rank_90_fixture": cluster,
        "episode_nearest_rank_90_expected": 18.0,
        "coverage_fixtures": fixtures,
        "ties_present": True,
        "passed": cluster == 18.0 and all(value["one_indexed_rank"] == 8 for value in fixtures.values()),
    }


def _noise() -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(SEED)
    return torch.randn((1, 50, 32), generator=generator, dtype=torch.float32).to("cuda")


def _base_passthrough(policy: Any, preprocessor: Any, postprocessor: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    raw = _raw_sample(row)
    raw.pop("action", None)
    raw.pop("action_is_pad", None)
    batch = _preprocess(preprocessor, raw)
    if hasattr(policy, "reset"):
        policy.reset()
    with torch.no_grad():
        native = policy.predict_action_chunk(_clone_batch(batch), noise=_noise())
    queue = _postprocess_chunk(native, postprocessor)
    passed_queue, passed_action = passthrough_queue_action(queue, monitor_enabled=False)
    return {
        "queue_shape": list(queue.shape),
        "base_action": queue[0].tolist(),
        "passthrough_action": passed_action.tolist(),
        "queue_identity_max_abs_error": float(np.max(np.abs(queue - passed_queue))),
        "action_identity_max_abs_error": float(np.max(np.abs(queue[0] - passed_action))),
    }


def _write_markdown(path: Path, result: Mapping[str, Any]) -> None:
    manifest = result["manifest_audit"]
    tiny = result["tiny_model_smoke"]
    lines = [
        "# NICE-VLA Stage 0A Result",
        "",
        f"Decision: `{result['final_decision']}`.",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`.",
        "",
        f"Pairs: `{result['completed_pair_count']} / {result['planned_pair_count']}`.",
        f"Exceptions: `{result['exception_count']}`.",
        f"Duplicate result keys: `{manifest['duplicate_result_key_count']}`.",
        f"Missing manifest keys: `{manifest['missing_manifest_key_count']}`.",
        f"Extra result keys: `{manifest['extra_result_key_count']}`.",
        "",
        f"Measured visual latent shape: `{tiny['latent_shape']}`.",
        f"Action width: `{tiny['action_width']}`; `k={K_STEP}`.",
        f"Mean loss: `{tiny['mean_loss_before']} -> {tiny['mean_loss_after']}`.",
        f"Covariance NLL: `{tiny['covariance_nll_before']} -> {tiny['covariance_nll_after']}`.",
        f"Reload max error: `{tiny['checkpoint_reload_max_abs_error']}`.",
        f"Base action identity error: `{result['base_passthrough']['action_identity_max_abs_error']}`.",
        "",
        "No validation or confirmatory record, simulator rollout, or task outcome was read.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _preflight(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, Any]:
    import torch

    required = (
        paths["checkpoint"],
        VLM_PATH,
        paths["data_root"],
        paths["source_root"],
        paths["proposal"],
        paths["proposal_hash"],
    )
    missing = [str(path) for path in required if not path.exists()]
    partial_parse_error = None
    partial_summary = None
    if paths["partial"].is_file():
        try:
            partial = _read_json(paths["partial"])
            partial_summary = {
                "planned_pair_count": partial.get("planned_pair_count"),
                "completed_pair_count": partial.get("completed_pair_count"),
                "exception_count": partial.get("exception_count"),
            }
        except Exception as exc:
            partial_parse_error = f"{type(exc).__name__}: {exc}"
    registry = _read_json(RESOURCE_REGISTRY) if RESOURCE_REGISTRY.is_file() else {"intervals": []}
    return {
        "passed": bool(
            not missing
            and torch.cuda.is_available()
            and _proposal_hash_text() == PROPOSAL_HASH
            and _sha256(PROPOSAL_FILE) == PROPOSAL_HASH
            and partial_parse_error is None
            and not paths["result_json"].exists()
            and not _active_linux_workers()
        ),
        "missing_paths": missing,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "proposal_hash_file": _proposal_hash_text(),
        "proposal_hash_observed": _sha256(PROPOSAL_FILE),
        "partial_summary": partial_summary,
        "partial_parse_error": partial_parse_error,
        "result_absent": not paths["result_json"].exists(),
        "active_linux_workers": _active_linux_workers(),
        "resource_evidence": _resource_evidence(registry, time.time()),
    }


def run_stage0a(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    paths = _paths(args)
    paths["report"].mkdir(parents=True, exist_ok=True)
    paths["run"].mkdir(parents=True, exist_ok=True)
    preflight = _preflight(args, paths)
    _write_json(paths["preflight"], preflight)
    if not preflight["passed"]:
        raise RuntimeError(f"preflight failed: {preflight}")

    _write_text(paths["pid"], f"{os.getpid()}\n")
    heartbeat_state: dict[str, Any] = {
        "pid": os.getpid(),
        "status": "running",
        "planned_pair_count": EXPECTED_PAIRS,
        "completed_pair_count": 0,
        "exception_count": 0,
    }
    _write_json(paths["status"], {**heartbeat_state, "started_at": _utc_now()})
    _write_json(paths["heartbeat"], {**heartbeat_state, "updated_at": _utc_now()})
    stop = threading.Event()
    thread = threading.Thread(target=_heartbeat_loop, args=(paths["heartbeat"], heartbeat_state, stop), daemon=True)
    thread.start()
    started = time.time()
    rows: list[dict[str, Any]] = []
    try:
        _set_offline_environment()
        source = _source_map(paths["source_root"])
        manifest, source_data, action_episodes = _build_manifest(paths)
        if len(manifest) != EXPECTED_PAIRS:
            raise RuntimeError(f"fixed manifest has {len(manifest)} rows, expected {EXPECTED_PAIRS}")
        manifest_payload = {
            "method": "NICE-VLA",
            "proposal_hash": PROPOSAL_HASH,
            "k_step": K_STEP,
            "rows": manifest,
            **source_data,
        }
        manifest_hash = canonical_json_sha256(manifest_payload)
        manifest_payload["manifest_hash"] = manifest_hash
        _write_json(paths["manifest"], manifest_payload)
        prior_partial_exception_count = 0
        if paths["partial"].is_file():
            prior_partial_exception_count = int(_read_json(paths["partial"]).get("exception_count", 0))
        rows = _load_resume(paths, manifest, manifest_hash)
        resumed_pair_count = len(rows)
        completed = {str(row["pair_key"]) for row in rows}
        heartbeat_state["completed_pair_count"] = len(rows)
        deadband = discovery_gripper_deadband(action_episodes)

        policy, _, preprocessor, postprocessor = _load_policy_and_processors(paths["checkpoint"])
        for parameter in policy.parameters():
            parameter.requires_grad_(False)
        policy.eval()
        for manifest_row in manifest:
            key = str(manifest_row["pair_key"])
            if key in completed:
                continue
            feature_path = _feature_path(paths["feature_dir"], key)
            feature = _extract_pair(policy, manifest_row)
            _save_feature(feature_path, **feature)
            summary = _row_summary(manifest_row, feature_path, feature)
            rows.append(summary)
            completed.add(key)
            heartbeat_state["completed_pair_count"] = len(rows)
            _write_json(
                paths["partial"],
                _partial_payload(manifest_hash, rows, exception_count=prior_partial_exception_count),
            )
            print(f"[nice-stage0a] pair {len(rows)}/{EXPECTED_PAIRS}", flush=True)

        manifest_audit = validate_manifest(manifest, rows)
        ordered = {str(row["pair_key"]): row for row in rows}
        features = [_load_feature(Path(ordered[str(row["pair_key"])]["feature_path"])) for row in manifest]
        all_actions = np.stack([feature["action"] for feature in features])
        validity = action_validity(all_actions)
        latent_shapes = {tuple(feature["z_t"].shape) for feature in features}
        latent_passed = bool(
            len(latent_shapes) == 1
            and all(np.all(np.isfinite(feature["z_t"])) for feature in features)
            and all(np.all(np.isfinite(feature["delta_z"])) for feature in features)
            and all(float(np.var(feature["delta_z"].astype(np.float32))) > 0.0 for feature in features)
        )
        tiny = _tiny_smoke(features, deadband, paths["smoke_checkpoint"])
        algebra = _algebra_smoke()
        calibration = _calibration_smoke()
        passthrough_row = {**manifest[0], "episode": manifest[0]["demo_id"], "frame": manifest[0]["frame_t"]}
        passthrough = _base_passthrough(policy, preprocessor, postprocessor, passthrough_row)
        source_passed = bool(
            source["commit_matches"]
            and source["license_matches"]
            and not source["missing"]
            and len(manifest) == EXPECTED_PAIRS
            and len(source_data["tasks"]) == 2
        )
        gradient_passed = bool(
            tiny["intended_gradients_finite_nonzero"]
            and tiny["frozen_gradients_zero"]
            and tiny["variance_clamped_fraction"] < 0.05
        )
        algebra_passed = bool(
            algebra["mahalanobis_max_abs_error"] <= 1e-5
            and algebra["logdet_max_abs_error"] <= 1e-5
            and algebra["minimum_dense_eigenvalue"] >= VARIANCE_FLOOR - 1e-8
        )
        passthrough_passed = bool(
            passthrough["queue_shape"] == [50, 7]
            and passthrough["queue_identity_max_abs_error"] == 0.0
            and passthrough["action_identity_max_abs_error"] == 0.0
        )
        reload_passed = tiny["checkpoint_reload_max_abs_error"] <= 1e-6
        action_passed = validity["finite_fraction"] == 1.0 and validity["inside_fraction"] == 1.0
        decision_inputs = Stage0DecisionInputs(
            completed_pairs=len(rows),
            planned_pairs=len(manifest),
            exception_count=0,
            manifest_passed=bool(manifest_audit["passed"]),
            source_passed=source_passed,
            latent_passed=latent_passed,
            gradient_passed=gradient_passed,
            algebra_passed=algebra_passed,
            calibration_passed=bool(calibration["passed"]),
            passthrough_passed=passthrough_passed,
            reload_passed=reload_passed,
            action_validity_passed=action_passed,
            forbidden_reads_zero=True,
        )
        decision = classify_stage0a(decision_inputs)
        result = {
            "method": "NICE-VLA",
            "proposal_hash": PROPOSAL_HASH,
            "source_commit": SOURCE_COMMIT,
            "attempt_number": int(args.attempt_number),
            "repair_consumed": bool(args.repair_consumed),
            "started_at": datetime.fromtimestamp(started, timezone.utc).isoformat(),
            "completed_at": _utc_now(),
            "pid": os.getpid(),
            "planned_pair_count": len(manifest),
            "completed_pair_count": len(rows),
            "resumed_pair_count": resumed_pair_count,
            "new_pair_count": len(rows) - resumed_pair_count,
            "exception_count": 0,
            "prior_attempt_exception_count": prior_partial_exception_count,
            "manifest_hash": manifest_hash,
            "manifest_audit": manifest_audit,
            "source_audit": source,
            "source_data_audit": source_data,
            "gripper_deadband": deadband,
            "latent_shape_set": [list(shape) for shape in sorted(latent_shapes)],
            "latent_passed": latent_passed,
            "action_validity": validity,
            "tiny_model_smoke": tiny,
            "algebra_smoke": algebra,
            "calibration_smoke": calibration,
            "base_passthrough": passthrough,
            "gates": {**decision_inputs.__dict__},
            "privileged_inference_input_count": 0,
            "validation_records_read": 0,
            "confirmatory_records_read": 0,
            "simulator_rollout_count": 0,
            "task_outcome_read_count": 0,
            "training_happened": True,
            "scientific_model_training_happened": False,
            "tiny_interface_smoke_training_happened": True,
            "validation_search_happened": False,
            "closed_loop_experiment_happened": False,
            "confirmatory_test_tuning_happened": False,
            "resource_evidence": preflight["resource_evidence"],
            "elapsed_seconds_quarantined_from_paper_evidence": time.time() - started,
            "final_decision": decision,
            "stage_0b_allowed": decision == "NICE_STAGE_0A_PASS_STAGE_0B_ALLOWED",
        }
        _write_json(paths["result_json"], result)
        _write_markdown(paths["result_md"], result)
        validation = {
            "proposal_hash_recomputed": _sha256(PROPOSAL_FILE) == PROPOSAL_HASH,
            "result_json_parsed": True,
            "manifest_hash_recomputed": canonical_json_sha256(
                {key: value for key, value in manifest_payload.items() if key != "manifest_hash"}
            )
            == manifest_hash,
            "manifest_audit": manifest_audit,
            "partial_json_parsed": bool(_read_json(paths["partial"])),
            "worker_completed": True,
            "accepted_without_duplicate_rows": manifest_audit["duplicate_result_key_count"] == 0,
            "final_decision": decision,
        }
        _write_json(paths["validation"], validation)
        heartbeat_state["status"] = "completed"
        _write_json(paths["status"], {**heartbeat_state, "completed_at": _utc_now(), "final_decision": decision})
        _write_json(paths["heartbeat"], {**heartbeat_state, "updated_at": _utc_now(), "final_decision": decision})
        return result
    except Exception:
        heartbeat_state["status"] = "failed"
        heartbeat_state["exception_count"] = 1
        heartbeat_state["completed_pair_count"] = len(rows)
        if paths["partial"].is_file():
            try:
                partial = _read_json(paths["partial"])
                partial["exception_count"] = int(partial.get("exception_count", 0)) + 1
                partial["last_exception"] = traceback.format_exc()
                _write_json(paths["partial"], partial)
            except Exception:
                pass
        _write_json(paths["status"], {**heartbeat_state, "failed_at": _utc_now()})
        _write_json(paths["heartbeat"], {**heartbeat_state, "updated_at": _utc_now()})
        raise
    finally:
        stop.set()
        thread.join(timeout=2.0)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("stage0a",), default="stage0a")
    parser.add_argument("--checkpoint", default=str(_asset_path("checkpoints", "smolvla_libero")))
    parser.add_argument("--libero-data-root", default=str(_asset_path("data", "libero")))
    parser.add_argument("--vla-corrector-source", default=str(_default_source_root()))
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "nice_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "nice_vla" / "stage0a"))
    parser.add_argument("--attempt-number", type=int, choices=(1, 2), default=1)
    parser.add_argument("--repair-consumed", action="store_true")
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
        result = run_stage0a(args)
        print(json.dumps({"final_decision": result["final_decision"]}, sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        blocker = {
            "method": "NICE-VLA",
            "proposal_hash": PROPOSAL_HASH,
            "final_decision": "NICE_STAGE_0A_IMPLEMENTATION_FAILURE",
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": traceback.format_exc(),
            "bounded_implementation_repair_possible": args.attempt_number == 1 and not args.repair_consumed,
            "scientific_kill": False,
        }
        _write_json(paths["blocker"], blocker)
        print(json.dumps(blocker, sort_keys=True), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
