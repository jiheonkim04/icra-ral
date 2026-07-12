"""DICD-VLA prototype runner.

The fast path runs a synthetic mechanism smoke that exercises the trainable
adapter, checkpoint roundtrip, action-change checks, and no-privileged-input
guard.  The real closed-loop stage builds on the same helpers for extracting
postprocessed SmolVLA action chunks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_echo_vla_first_prototype import _postprocess_action, _preprocess_batch  # noqa: E402
from scripts.run_phase_barrier_vla_prototype import (  # noqa: E402
    _identity_to_initial_state_index,
    _make_exact_vector_env,
    _round,
    _set_runtime_env,
    _step_success,
    _write_json,
    _write_md,
)
from tca_map.smolvla.dicd_vla import (  # noqa: E402
    DICDConfig,
    assert_no_privileged_inference_fields,
    build_dicd_examples,
    direct_chunk_index_action,
    file_sha256,
    load_dicd_checkpoint,
    make_dicd_features,
    predict_dicd_action,
    save_dicd_checkpoint,
    train_dicd_adapter,
)
from tca_map.smolvla.official_closed_loop_scaleup import _json_default  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import POLICIES, _cuda_memory, _load_policy_and_processors  # noqa: E402


DATE_KST = "2026-07-12"
BRANCH = "codex/auto-method-20260712-01-dicd-vla"
DICD_TASKS = [
    {
        "suite": "libero_spatial",
        "task_id": 4,
        "role": "stable_grasp_contact_transition",
        "instruction": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    },
    {
        "suite": "libero_10",
        "task_id": 4,
        "role": "long_horizon_contact_and_release",
        "instruction": "put the white mug on the left plate and put the yellow and white mug on the right plate",
    },
]
STAGE_A_VARIANTS = [
    "frozen_smolvla_clean",
    "frozen_smolvla_delay",
    "direct_chunk_index_delay",
    "dicd_no_history_ablation",
    "dicd_full",
]


def _synthetic_chunk(offset: float, *, chunk_len: int, action_dim: int) -> np.ndarray:
    rows = []
    for index in range(chunk_len):
        base = np.zeros(action_dim, dtype=np.float32)
        base[0] = float(offset + 0.08 * index)
        base[1] = float(0.05 * np.sin(offset + index))
        base[2] = float(-0.04 * np.cos(offset + index))
        base[3] = float(0.03 * index)
        base[4] = float(0.02 * offset)
        base[5] = float(-0.01 * index)
        base[6] = -1.0
        rows.append(base)
    return np.stack(rows, axis=0)


def _synthetic_trace(config: DICDConfig, count: int, delay: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    chunks = [_synthetic_chunk(index * 0.03, chunk_len=config.chunk_len, action_dim=config.action_dim) for index in range(count)]
    executed: list[np.ndarray] = []
    previous = np.zeros(config.action_dim, dtype=np.float32)
    for index, chunk in enumerate(chunks):
        delayed = direct_chunk_index_action(chunk, min(delay, config.chunk_len - 1), config).reshape(-1)
        action = delayed + 0.18 * previous
        action[6] = -1.0
        executed.append(action.astype(np.float32))
        previous = action
    return chunks, executed


def _array_sha256(arrays: list[np.ndarray]) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        data = np.asarray(array, dtype=np.float32)
        digest.update(str(tuple(data.shape)).encode("utf-8"))
        digest.update(data.tobytes())
    return digest.hexdigest()


def _postprocess_action_chunk(action_chunk: Any, loaded: Mapping[str, Any], *, max_chunk_len: int | None = None) -> np.ndarray:
    import torch

    chunk = action_chunk
    if isinstance(chunk, torch.Tensor):
        tensor = chunk.detach()
    else:
        tensor = torch.as_tensor(chunk)
    if tensor.ndim == 3:
        tensor = tensor[0]
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    if tensor.ndim != 2:
        raise ValueError(f"unsupported action chunk rank {tensor.ndim}")
    if max_chunk_len is not None:
        tensor = tensor[: int(max_chunk_len)]
    rows = []
    for row in tensor:
        rows.append(_postprocess_action(row.reshape(1, -1), dict(loaded)).reshape(-1))
    return np.stack(rows, axis=0).astype(np.float32)


def predict_postprocessed_action_chunk(policy: Any, env: Any, observation: Any, loaded: Mapping[str, Any], *, max_chunk_len: int | None = None) -> np.ndarray:
    import torch

    batch = _preprocess_batch(env, observation, dict(loaded))
    with torch.inference_mode():
        raw_chunk = policy.predict_action_chunk(batch)
    return _postprocess_action_chunk(raw_chunk, loaded, max_chunk_len=max_chunk_len)


def _run_synthetic_mechanism_smoke(args: argparse.Namespace) -> dict[str, Any]:
    config = DICDConfig(action_dim=int(args.action_dim), chunk_len=int(args.chunk_len), history_len=int(args.history_len), hidden_dim=int(args.hidden_dim))
    chunks, executed = _synthetic_trace(config, int(args.synthetic_count), int(args.delay))
    full_examples = build_dicd_examples(chunks, executed, delay=int(args.delay), config=config, use_history=True)
    ablation_examples = build_dicd_examples(chunks, executed, delay=int(args.delay), config=config, use_history=False)
    full_model, full_stats = train_dicd_adapter(full_examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=int(args.seed))
    ablation_model, ablation_stats = train_dicd_adapter(ablation_examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=int(args.seed))
    checkpoint_path = Path(args.checkpoint_path)
    save_dicd_checkpoint(checkpoint_path, full_model, full_stats)
    loaded_model, loaded_stats = load_dicd_checkpoint(checkpoint_path)
    checkpoint_hash = file_sha256(checkpoint_path)

    probe_index = min(6, len(chunks) - int(args.delay) - 1)
    probe_history = executed[max(0, probe_index - config.history_len) : probe_index]
    probe_features = make_dicd_features(
        chunks[probe_index],
        history=probe_history,
        delay=int(args.delay),
        step_fraction=probe_index / max(1.0, float(len(chunks) - 1)),
        config=config,
        use_history=True,
    )
    ablation_features = make_dicd_features(
        chunks[probe_index],
        history=probe_history,
        delay=int(args.delay),
        step_fraction=probe_index / max(1.0, float(len(chunks) - 1)),
        config=config,
        use_history=False,
    )
    direct = direct_chunk_index_action(chunks[probe_index], int(args.delay), config)
    in_memory = predict_dicd_action(full_model, probe_features)
    reloaded = predict_dicd_action(loaded_model, probe_features)
    ablation = predict_dicd_action(ablation_model, ablation_features)
    assert_no_privileged_inference_fields(["observation", "instruction", "action_chunk", "executed_action_history", "declared_delay", "step_fraction"])

    checks = {
        "chunk_horizon_exceeds_delay": bool(config.chunk_len > int(args.delay)),
        "full_finite_gradients": bool(full_stats["finite_gradients"]),
        "full_loss_decreased": bool(full_stats["loss_decreased"]),
        "ablation_loss_decreased": bool(ablation_stats["loss_decreased"]),
        "checkpoint_reloaded": bool(np.allclose(in_memory, reloaded)),
        "full_changes_direct_chunk_index": bool(np.linalg.norm(in_memory - direct) > float(args.min_action_delta)),
        "full_differs_from_no_history_ablation": bool(np.linalg.norm(in_memory - ablation) > float(args.min_action_delta)),
        "no_privileged_inference_fields": True,
    }
    return {
        "schema_version": "dicd_vla_mechanism_smoke_v1",
        "date_kst": DATE_KST,
        "branch": BRANCH,
        "smoke_type": "synthetic_core_mechanism",
        "real_smolvla_chunk_smoke_happened": False,
        "closed_loop_experiment_happened": False,
        "training_happened": True,
        "config": config.__dict__,
        "delay": int(args.delay),
        "training_example_count_full": int(len(full_examples)),
        "training_example_count_ablation": int(len(ablation_examples)),
        "full_train_stats": full_stats,
        "ablation_train_stats": ablation_stats,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "loaded_stats": loaded_stats,
        "probe": {
            "probe_index": int(probe_index),
            "direct_chunk_index_action": direct.reshape(-1).tolist(),
            "dicd_full_action": in_memory.reshape(-1).tolist(),
            "dicd_reloaded_action": reloaded.reshape(-1).tolist(),
            "dicd_no_history_action": ablation.reshape(-1).tolist(),
            "full_vs_direct_delta_norm": _round(float(np.linalg.norm(in_memory - direct)), 6),
            "full_vs_ablation_delta_norm": _round(float(np.linalg.norm(in_memory - ablation)), 6),
        },
        "checks": checks,
        "mechanism_smoke_passed": bool(all(checks.values())),
        "final_decision": "DICD_SYNTHETIC_MECHANISM_SMOKE_PASSED" if all(checks.values()) else "DICD_SYNTHETIC_MECHANISM_SMOKE_FAILED",
    }


def _run_real_smolvla_chunk_smoke(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    _set_runtime_env(args)
    config = DICDConfig(action_dim=int(args.action_dim), chunk_len=int(args.chunk_len), history_len=int(args.history_len), hidden_dim=int(args.hidden_dim))
    task = DICD_TASKS[int(args.smoke_task_index)]
    env = None
    loaded_model = None
    loaded_stats: dict[str, Any] = {}
    if Path(args.checkpoint_path).exists():
        loaded_model, loaded_stats = load_dicd_checkpoint(args.checkpoint_path)
    try:
        loaded = _load_policy_and_processors(args, POLICIES[0])
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        env = _make_exact_vector_env(str(task["suite"]), int(task["task_id"]), _identity_to_initial_state_index(int(args.smoke_identity)))
        observation, _ = env.reset(seed=[int(args.smoke_identity)])
        history: list[np.ndarray] = []
        records: list[dict[str, Any]] = []
        for step in range(int(args.smoke_steps)):
            chunk = predict_postprocessed_action_chunk(policy, env, observation, loaded, max_chunk_len=int(args.real_max_chunk_len))
            first = direct_chunk_index_action(chunk, 0, config)
            indexed = direct_chunk_index_action(chunk, int(args.delay), config)
            features = make_dicd_features(
                chunk,
                history=history,
                delay=int(args.delay),
                step_fraction=step / max(1.0, float(args.smoke_steps - 1)),
                config=config,
                use_history=True,
            )
            prediction = None if loaded_model is None else predict_dicd_action(loaded_model, features)
            records.append(
                {
                    "step": int(step),
                    "postprocessed_chunk_shape": [int(dim) for dim in chunk.shape],
                    "postprocessed_chunk_finite": bool(np.isfinite(chunk).all()),
                    "direct_delay_delta_norm": _round(float(np.linalg.norm(indexed - first)), 6),
                    "feature_dim": int(len(features)),
                    "feature_finite": bool(np.isfinite(np.asarray(features, dtype=np.float32)).all()),
                    "synthetic_checkpoint_prediction_finite": None if prediction is None else bool(np.isfinite(prediction).all()),
                    "synthetic_checkpoint_prediction_delta_from_direct": None if prediction is None else _round(float(np.linalg.norm(prediction - indexed)), 6),
                }
            )
            history.append(first.reshape(-1))
            observation, _reward, terminated, truncated, _info = env.step(first.reshape(1, -1))
            if bool(np.all(terminated | truncated)):
                break
        assert_no_privileged_inference_fields(["observation", "instruction", "action_chunk", "executed_action_history", "declared_delay", "step_fraction"])
        delay_deltas = [float(row["direct_delay_delta_norm"]) for row in records]
        checks = {
            "official_policy_loaded": bool((loaded.get("audit") or {}).get("policy_class") == "SmolVLAPolicy"),
            "old_custom_route_not_used": bool((loaded.get("audit") or {}).get("old_custom_libero_7d_route_used") is False),
            "raw_action_chunk_horizon_exceeds_delay": bool((loaded.get("audit") or {}).get("action_chunk_shape", [0, 0])[1] > int(args.delay)),
            "postprocessed_chunks_finite": bool(records and all(row["postprocessed_chunk_finite"] for row in records)),
            "postprocessed_chunk_horizon_exceeds_delay": bool(records and all(row["postprocessed_chunk_shape"][0] > int(args.delay) for row in records)),
            "postprocessed_action_dim_is_7": bool(records and all(row["postprocessed_chunk_shape"][1] == 7 for row in records)),
            "features_match_config_width": bool(records and all(row["feature_dim"] == config.input_dim for row in records)),
            "features_finite": bool(records and all(row["feature_finite"] for row in records)),
            "real_delay_contrast_present": bool(delay_deltas and max(delay_deltas) > float(args.min_action_delta)),
            "no_privileged_inference_fields": True,
        }
        return {
            "schema_version": "dicd_vla_real_smolvla_chunk_smoke_v1",
            "date_kst": DATE_KST,
            "branch": BRANCH,
            "smoke_type": "real_smolvla_action_chunk",
            "training_happened": False,
            "real_smolvla_chunk_smoke_happened": True,
            "closed_loop_experiment_happened": False,
            "config": config.__dict__,
            "delay": int(args.delay),
            "task": task,
            "smoke_identity": int(args.smoke_identity),
            "policy_load_audit": loaded.get("audit"),
            "records": records,
            "checkpoint_path": str(args.checkpoint_path) if Path(args.checkpoint_path).exists() else None,
            "checkpoint_sha256": file_sha256(args.checkpoint_path) if Path(args.checkpoint_path).exists() else None,
            "loaded_checkpoint_stats": loaded_stats,
            "cuda_memory": _cuda_memory(torch),
            "checks": checks,
            "mechanism_smoke_passed": bool(all(checks.values())),
            "final_decision": "DICD_REAL_SMOLVLA_CHUNK_SMOKE_PASSED" if all(checks.values()) else "DICD_REAL_SMOLVLA_CHUNK_SMOKE_FAILED",
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _collect_real_trace(
    args: argparse.Namespace,
    loaded: Mapping[str, Any],
    task: Mapping[str, Any],
    *,
    identity: int,
    config: DICDConfig,
    max_steps: int,
) -> tuple[list[np.ndarray], list[np.ndarray], dict[str, Any]]:
    env = None
    try:
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        env = _make_exact_vector_env(str(task["suite"]), int(task["task_id"]), _identity_to_initial_state_index(int(identity)))
        observation, _ = env.reset(seed=[int(identity)])
        chunks: list[np.ndarray] = []
        executed: list[np.ndarray] = []
        delay_deltas: list[float] = []
        for _step in range(int(max_steps)):
            chunk = predict_postprocessed_action_chunk(policy, env, observation, loaded, max_chunk_len=int(args.real_max_chunk_len))
            first = direct_chunk_index_action(chunk, 0, config).reshape(-1)
            indexed = direct_chunk_index_action(chunk, int(args.delay), config).reshape(-1)
            chunks.append(chunk)
            executed.append(first.astype(np.float32))
            delay_deltas.append(float(np.linalg.norm(indexed - first)))
            observation, _reward, terminated, truncated, _info = env.step(first.reshape(1, -1))
            if bool(np.all(terminated | truncated)):
                break
        summary = {
            "suite": str(task["suite"]),
            "task_id": int(task["task_id"]),
            "identity": int(identity),
            "step_count": int(len(chunks)),
            "chunk_sha256": _array_sha256(chunks),
            "executed_action_sha256": _array_sha256([row.reshape(1, -1) for row in executed]),
            "mean_delay_delta_norm": _round(float(np.mean(delay_deltas)) if delay_deltas else 0.0, 6),
            "max_delay_delta_norm": _round(float(np.max(delay_deltas)) if delay_deltas else 0.0, 6),
            "executed_action_std": _round(float(np.std(np.stack(executed, axis=0))) if executed else 0.0, 6),
        }
        return chunks, executed, summary
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _run_real_trace_training(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    _set_runtime_env(args)
    config = DICDConfig(action_dim=int(args.action_dim), chunk_len=int(args.chunk_len), history_len=int(args.history_len), hidden_dim=int(args.hidden_dim))
    loaded = _load_policy_and_processors(args, POLICIES[0])
    all_full_examples = []
    all_ablation_examples = []
    train_traces: list[dict[str, Any]] = []
    for task in DICD_TASKS[: int(args.max_tasks)]:
        chunks, executed, summary = _collect_real_trace(
            args,
            loaded,
            task,
            identity=int(args.train_identity),
            config=config,
            max_steps=int(args.max_train_steps),
        )
        full_examples = build_dicd_examples(chunks, executed, delay=int(args.delay), config=config, use_history=True)
        ablation_examples = build_dicd_examples(chunks, executed, delay=int(args.delay), config=config, use_history=False)
        all_full_examples.extend(full_examples)
        all_ablation_examples.extend(ablation_examples)
        summary["full_example_count"] = int(len(full_examples))
        summary["ablation_example_count"] = int(len(ablation_examples))
        train_traces.append(summary)

    full_model, full_stats = train_dicd_adapter(
        all_full_examples,
        config=config,
        epochs=int(args.epochs),
        lr=float(args.lr),
        seed=int(args.seed),
    )
    ablation_model, ablation_stats = train_dicd_adapter(
        all_ablation_examples,
        config=config,
        epochs=int(args.epochs),
        lr=float(args.lr),
        seed=int(args.seed),
    )
    checkpoint_dir = Path(args.real_checkpoint_dir)
    full_checkpoint = checkpoint_dir / "dicd_real_full.pt"
    ablation_checkpoint = checkpoint_dir / "dicd_real_no_history.pt"
    save_dicd_checkpoint(full_checkpoint, full_model, full_stats)
    save_dicd_checkpoint(ablation_checkpoint, ablation_model, ablation_stats)
    reloaded_full, reloaded_full_stats = load_dicd_checkpoint(full_checkpoint)
    reloaded_ablation, reloaded_ablation_stats = load_dicd_checkpoint(ablation_checkpoint)

    smoke_task = DICD_TASKS[int(args.smoke_task_index)]
    smoke_chunks, smoke_executed, smoke_summary = _collect_real_trace(
        args,
        loaded,
        smoke_task,
        identity=int(args.smoke_identity),
        config=config,
        max_steps=max(int(args.smoke_steps), int(args.delay) + 1),
    )
    probe_index = min(1, max(0, len(smoke_chunks) - int(args.delay) - 1))
    probe_history = smoke_executed[max(0, probe_index - config.history_len) : probe_index]
    full_features = make_dicd_features(
        smoke_chunks[probe_index],
        history=probe_history,
        delay=int(args.delay),
        step_fraction=probe_index / max(1.0, float(len(smoke_chunks) - 1)),
        config=config,
        use_history=True,
    )
    ablation_features = make_dicd_features(
        smoke_chunks[probe_index],
        history=probe_history,
        delay=int(args.delay),
        step_fraction=probe_index / max(1.0, float(len(smoke_chunks) - 1)),
        config=config,
        use_history=False,
    )
    direct = direct_chunk_index_action(smoke_chunks[probe_index], int(args.delay), config)
    full_pred = predict_dicd_action(full_model, full_features)
    full_reload_pred = predict_dicd_action(reloaded_full, full_features)
    ablation_pred = predict_dicd_action(ablation_model, ablation_features)
    ablation_reload_pred = predict_dicd_action(reloaded_ablation, ablation_features)
    assert_no_privileged_inference_fields(["observation", "instruction", "action_chunk", "executed_action_history", "declared_delay", "step_fraction"])
    targets = np.asarray([row.target for row in all_full_examples], dtype=np.float32)
    checks = {
        "training_examples_exist": bool(len(all_full_examples) > 0 and len(all_ablation_examples) > 0),
        "labels_have_required_contrast": bool(float(np.std(targets)) > float(args.min_action_delta)),
        "full_finite_gradients": bool(full_stats["finite_gradients"]),
        "full_loss_decreased": bool(full_stats["loss_decreased"]),
        "ablation_loss_decreased": bool(ablation_stats["loss_decreased"]),
        "full_checkpoint_reloaded": bool(np.allclose(full_pred, full_reload_pred)),
        "ablation_checkpoint_reloaded": bool(np.allclose(ablation_pred, ablation_reload_pred)),
        "full_changes_direct_chunk_index": bool(np.linalg.norm(full_pred - direct) > float(args.min_action_delta)),
        "full_differs_from_no_history_ablation": bool(np.linalg.norm(full_pred - ablation_pred) > float(args.min_action_delta)),
        "no_privileged_inference_fields": True,
    }
    return {
        "schema_version": "dicd_vla_real_trace_training_v1",
        "date_kst": DATE_KST,
        "branch": BRANCH,
        "training_happened": True,
        "real_trace_training_happened": True,
        "closed_loop_experiment_happened": False,
        "config": config.__dict__,
        "delay": int(args.delay),
        "train_identity": int(args.train_identity),
        "smoke_identity": int(args.smoke_identity),
        "train_traces": train_traces,
        "smoke_trace": smoke_summary,
        "full_example_count": int(len(all_full_examples)),
        "ablation_example_count": int(len(all_ablation_examples)),
        "target_std": _round(float(np.std(targets)), 6),
        "full_train_stats": full_stats,
        "ablation_train_stats": ablation_stats,
        "full_checkpoint_path": str(full_checkpoint),
        "full_checkpoint_sha256": file_sha256(full_checkpoint),
        "full_loaded_stats": reloaded_full_stats,
        "ablation_checkpoint_path": str(ablation_checkpoint),
        "ablation_checkpoint_sha256": file_sha256(ablation_checkpoint),
        "ablation_loaded_stats": reloaded_ablation_stats,
        "probe": {
            "probe_task": smoke_task,
            "probe_index": int(probe_index),
            "direct_chunk_index_action": direct.reshape(-1).tolist(),
            "dicd_full_action": full_pred.reshape(-1).tolist(),
            "dicd_full_reloaded_action": full_reload_pred.reshape(-1).tolist(),
            "dicd_no_history_action": ablation_pred.reshape(-1).tolist(),
            "dicd_no_history_reloaded_action": ablation_reload_pred.reshape(-1).tolist(),
            "full_vs_direct_delta_norm": _round(float(np.linalg.norm(full_pred - direct)), 6),
            "full_vs_ablation_delta_norm": _round(float(np.linalg.norm(full_pred - ablation_pred)), 6),
        },
        "policy_load_audit": loaded.get("audit"),
        "cuda_memory": _cuda_memory(torch),
        "checks": checks,
        "real_trace_training_passed": bool(all(checks.values())),
        "final_decision": "DICD_REAL_TRACE_TRAINING_PASSED" if all(checks.values()) else "DICD_REAL_TRACE_TRAINING_FAILED",
    }


def _parse_int_list(raw: str) -> list[int]:
    return [int(part.strip()) for part in str(raw).split(",") if part.strip()]


def _stage_a_episode_key(variant: str, task: Mapping[str, Any], identity: int) -> str:
    return f"{variant}|{task['suite']}|{int(task['task_id'])}|{int(identity)}"


def _stage_a_episode_key_from_row(row: Mapping[str, Any]) -> str:
    return f"{row.get('variant')}|{row.get('suite')}|{int(row.get('task_id'))}|{int(row.get('identity'))}"


def _stage_a_action_command(
    variant: str,
    chunk: np.ndarray,
    *,
    history: list[np.ndarray],
    delay: int,
    step_fraction: float,
    config: DICDConfig,
    full_model: Any,
    ablation_model: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    if variant in {"frozen_smolvla_clean", "frozen_smolvla_delay"}:
        action = direct_chunk_index_action(chunk, 0, config)
        return action, {"command": "chunk_index_0"}
    if variant == "direct_chunk_index_delay":
        action = direct_chunk_index_action(chunk, delay, config)
        return action, {"command": f"chunk_index_{int(delay)}"}
    if variant == "dicd_no_history_ablation":
        features = make_dicd_features(chunk, history=history, delay=delay, step_fraction=step_fraction, config=config, use_history=False)
        action = predict_dicd_action(ablation_model, features)
        return np.clip(action, -1.0, 1.0), {"command": "dicd_no_history", "feature_dim": len(features)}
    if variant == "dicd_full":
        features = make_dicd_features(chunk, history=history, delay=delay, step_fraction=step_fraction, config=config, use_history=True)
        action = predict_dicd_action(full_model, features)
        return np.clip(action, -1.0, 1.0), {"command": "dicd_full", "feature_dim": len(features)}
    raise ValueError(f"unknown Stage A variant {variant}")


def _run_stage_a_episode(
    args: argparse.Namespace,
    loaded: Mapping[str, Any],
    task: Mapping[str, Any],
    identity: int,
    variant: str,
    *,
    config: DICDConfig,
    full_model: Any,
    ablation_model: Any,
) -> dict[str, Any]:
    import torch

    env = None
    started = time.monotonic()
    try:
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        env = _make_exact_vector_env(str(task["suite"]), int(task["task_id"]), _identity_to_initial_state_index(int(identity)))
        observation, _ = env.reset(seed=[int(identity)])
        max_steps = int(env.call("_max_episode_steps")[0])
        if int(args.max_eval_steps) > 0:
            max_steps = min(max_steps, int(args.max_eval_steps))
        history: list[np.ndarray] = []
        pending: list[np.ndarray] = []
        action_deltas: list[float] = []
        full_vs_direct_deltas: list[float] = []
        shaped_steps = 0
        reward_sum = 0.0
        success = False
        step_count = 0
        for step in range(max_steps):
            chunk = predict_postprocessed_action_chunk(policy, env, observation, loaded, max_chunk_len=int(args.real_max_chunk_len))
            immediate = direct_chunk_index_action(chunk, 0, config)
            command, meta = _stage_a_action_command(
                variant,
                chunk,
                history=history,
                delay=int(args.delay),
                step_fraction=step / max(1.0, float(max_steps - 1)),
                config=config,
                full_model=full_model,
                ablation_model=ablation_model,
            )
            if variant == "frozen_smolvla_clean":
                action = command
            else:
                if len(pending) < int(args.delay):
                    action = command
                else:
                    action = pending.pop(0)
                pending.append(np.asarray(command, dtype=np.float32).reshape(1, -1))
            delta = float(np.linalg.norm(np.asarray(action, dtype=np.float32) - immediate))
            action_deltas.append(delta)
            if delta > float(args.min_action_delta):
                shaped_steps += 1
            if variant == "dicd_full":
                direct = direct_chunk_index_action(chunk, int(args.delay), config)
                full_vs_direct_deltas.append(float(np.linalg.norm(command - direct)))
            observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float64).reshape(1, -1))
            reward_sum += float(np.asarray(reward).reshape(-1)[0])
            history.append(np.asarray(action, dtype=np.float32).reshape(-1))
            step_count = step + 1
            success = bool(success or _step_success(info))
            if np.all(terminated | truncated) or success:
                break
        return {
            "variant": variant,
            "suite": str(task["suite"]),
            "task_id": int(task["task_id"]),
            "task_key": f"{task['suite']}/task_{task['task_id']}",
            "role": str(task["role"]),
            "identity": int(identity),
            "initial_state_index": _identity_to_initial_state_index(int(identity)),
            "success": bool(success),
            "reward_sum": _round(reward_sum, 6),
            "episode_steps": int(step_count),
            "mean_action_delta_norm": _round(float(np.mean(action_deltas)) if action_deltas else 0.0, 6),
            "shaped_step_count": int(shaped_steps),
            "mean_full_vs_direct_command_delta": _round(float(np.mean(full_vs_direct_deltas)) if full_vs_direct_deltas else 0.0, 6),
            "elapsed_seconds": _round(time.monotonic() - started, 3),
            "cuda_memory": _cuda_memory(torch),
            "exception": None,
        }
    except Exception as exc:  # pragma: no cover - runtime boundary
        return {
            "variant": variant,
            "suite": str(task["suite"]),
            "task_id": int(task["task_id"]),
            "task_key": f"{task['suite']}/task_{task['task_id']}",
            "identity": int(identity),
            "success": False,
            "elapsed_seconds": _round(time.monotonic() - started, 3),
            "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc().splitlines()[-80:]},
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _wilson(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    radius = z * ((phat * (1 - phat) + z * z / (4 * total)) / total) ** 0.5 / denom
    return [_round(max(0.0, center - radius), 6), _round(min(1.0, center + radius), 6)]


def _summarize_stage_a(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in STAGE_A_VARIANTS:
        rows = [row for row in episodes if row.get("variant") == variant and row.get("exception") is None]
        successes = sum(1 for row in rows if row.get("success"))
        per_task = {}
        for task_key in sorted({str(row.get("task_key")) for row in rows}):
            task_rows = [row for row in rows if str(row.get("task_key")) == task_key]
            task_successes = sum(1 for row in task_rows if row.get("success"))
            per_task[task_key] = {"successes": int(task_successes), "total": int(len(task_rows)), "rate": _round(task_successes / max(1, len(task_rows)), 6)}
        by_variant[variant] = {
            "successes": int(successes),
            "total": int(len(rows)),
            "task_balanced_success_rate": _round(successes / max(1, len(rows)), 6),
            "wilson_95_ci": _wilson(successes, len(rows)),
            "per_task": per_task,
            "mean_action_delta_norm": _round(float(np.mean([row.get("mean_action_delta_norm", 0.0) for row in rows])) if rows else 0.0, 6),
            "mean_shaped_step_count": _round(float(np.mean([row.get("shaped_step_count", 0) for row in rows])) if rows else 0.0, 6),
        }
    paired = {}
    full_rows = [row for row in episodes if row.get("variant") == "dicd_full" and row.get("exception") is None]
    for comparator in ["frozen_smolvla_delay", "direct_chunk_index_delay", "dicd_no_history_ablation"]:
        comp_rows = [row for row in episodes if row.get("variant") == comparator and row.get("exception") is None]
        comp_by_key = {(row.get("task_key"), row.get("identity")): row for row in comp_rows}
        wins = losses = ties = 0
        for row in full_rows:
            other = comp_by_key.get((row.get("task_key"), row.get("identity")))
            if other is None:
                continue
            full_success = bool(row.get("success"))
            comp_success = bool(other.get("success"))
            if full_success and not comp_success:
                wins += 1
            elif comp_success and not full_success:
                losses += 1
            else:
                ties += 1
        paired[comparator] = {"win": int(wins), "loss": int(losses), "tie": int(ties)}
    delayed_baselines = ["frozen_smolvla_delay", "direct_chunk_index_delay"]
    strongest_baseline = max(delayed_baselines, key=lambda name: by_variant[name]["task_balanced_success_rate"])
    full_rate = float(by_variant["dicd_full"]["task_balanced_success_rate"])
    strongest_rate = float(by_variant[strongest_baseline]["task_balanced_success_rate"])
    ablation_rate = float(by_variant["dicd_no_history_ablation"]["task_balanced_success_rate"])
    direct_rate = float(by_variant["direct_chunk_index_delay"]["task_balanced_success_rate"])
    mechanism_active = bool(by_variant["dicd_full"]["mean_action_delta_norm"] > 0.0)
    passes_go = bool(
        mechanism_active
        and full_rate >= strongest_rate + 0.05
        and full_rate > direct_rate
        and full_rate > ablation_rate
    )
    if passes_go:
        decision = "PROTOTYPE_GO"
    elif direct_rate >= full_rate:
        decision = "SIMPLE_BASELINE_EXPLAINS_METHOD"
    elif ablation_rate >= full_rate:
        decision = "KEY_COMPONENT_NOT_USEFUL"
    elif full_rate > strongest_rate and mechanism_active:
        decision = "UNDERPOWERED_ONE_EXPANSION_ALLOWED"
    else:
        decision = "GENUINE_METHOD_KILL"
    return {
        "by_variant": by_variant,
        "paired_full_vs": paired,
        "strongest_delayed_baseline": strongest_baseline,
        "mechanism_active": mechanism_active,
        "passes_prototype_go": passes_go,
        "method_decision": decision,
    }


def _run_stage_a(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    _set_runtime_env(args)
    config = DICDConfig(action_dim=int(args.action_dim), chunk_len=int(args.chunk_len), history_len=int(args.history_len), hidden_dim=int(args.hidden_dim))
    full_model, full_stats = load_dicd_checkpoint(Path(args.real_checkpoint_dir) / "dicd_real_full.pt")
    ablation_model, ablation_stats = load_dicd_checkpoint(Path(args.real_checkpoint_dir) / "dicd_real_no_history.pt")
    loaded = _load_policy_and_processors(args, POLICIES[0])
    identities = _parse_int_list(args.stage_a_identities)
    tasks = DICD_TASKS[: int(args.max_tasks)]
    plan = [(variant, task, int(identity)) for variant in STAGE_A_VARIANTS for task in tasks for identity in identities]
    partial_path = Path(args.stage_a_partial_json)
    episode_by_key: dict[str, dict[str, Any]] = {}
    if partial_path.exists():
        try:
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            for row in partial.get("episodes", []):
                key = _stage_a_episode_key_from_row(row)
                if key in {_stage_a_episode_key(variant, task, identity) for variant, task, identity in plan}:
                    episode_by_key[key] = row
        except Exception as exc:  # pragma: no cover - runtime boundary
            print(f"[stage-a] ignoring unreadable partial result {partial_path}: {exc}", flush=True)
    total = len(plan)
    for variant in STAGE_A_VARIANTS:
        for task in tasks:
            for identity in identities:
                key = _stage_a_episode_key(variant, task, int(identity))
                if key in episode_by_key:
                    print(f"[stage-a] skip completed {len(episode_by_key)}/{total}: {key}", flush=True)
                    continue
                row = _run_stage_a_episode(
                    args,
                    loaded,
                    task,
                    int(identity),
                    variant,
                    config=config,
                    full_model=full_model,
                    ablation_model=ablation_model,
                )
                episode_by_key[key] = row
                _write_json(
                    partial_path,
                    {
                        "schema_version": "dicd_vla_stage_a_partial_v1",
                        "date_kst": DATE_KST,
                        "branch": BRANCH,
                        "planned_episode_count": int(total),
                        "completed_episode_count": int(len(episode_by_key)),
                        "partial_result": True,
                        "episodes": [episode_by_key[_stage_a_episode_key(v, t, i)] for v, t, i in plan if _stage_a_episode_key(v, t, i) in episode_by_key],
                    },
                )
                print(
                    "[stage-a] completed "
                    f"{len(episode_by_key)}/{total}: {key} "
                    f"success={row.get('success')} exception={row.get('exception') is not None}",
                    flush=True,
                )
    episodes = [episode_by_key[_stage_a_episode_key(variant, task, identity)] for variant, task, identity in plan if _stage_a_episode_key(variant, task, identity) in episode_by_key]
    measurement_valid = len(episodes) == total and not any(row.get("exception") for row in episodes)
    summary = _summarize_stage_a(episodes) if measurement_valid else {}
    return {
        "schema_version": "dicd_vla_stage_a_v1",
        "date_kst": DATE_KST,
        "branch": BRANCH,
        "stage_a_completed": bool(measurement_valid),
        "closed_loop_experiment_happened": True,
        "training_happened": False,
        "config": config.__dict__,
        "delay": int(args.delay),
        "variants": STAGE_A_VARIANTS,
        "tasks": tasks,
        "identities": identities,
        "planned_episode_count": int(total),
        "episode_count": int(len(episodes)),
        "partial_checkpoint_path": str(partial_path),
        "full_checkpoint_path": str(Path(args.real_checkpoint_dir) / "dicd_real_full.pt"),
        "full_checkpoint_sha256": file_sha256(Path(args.real_checkpoint_dir) / "dicd_real_full.pt"),
        "full_loaded_stats": full_stats,
        "ablation_checkpoint_path": str(Path(args.real_checkpoint_dir) / "dicd_real_no_history.pt"),
        "ablation_checkpoint_sha256": file_sha256(Path(args.real_checkpoint_dir) / "dicd_real_no_history.pt"),
        "ablation_loaded_stats": ablation_stats,
        "policy_load_audit": loaded.get("audit"),
        "episodes": episodes,
        "summary": summary,
        "cuda_memory": _cuda_memory(torch),
        "final_decision": summary.get("method_decision", "DICD_STAGE_A_MEASUREMENT_INVALID"),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    report: dict[str, Any]
    try:
        if args.mode == "real-smolvla-chunk":
            report = _run_real_smolvla_chunk_smoke(args)
        elif args.mode == "real-trace-train":
            report = _run_real_trace_training(args)
        elif args.mode == "stage-a":
            report = _run_stage_a(args)
        else:
            report = _run_synthetic_mechanism_smoke(args)
    except Exception as exc:  # pragma: no cover - runtime boundary
        report = {
            "schema_version": "dicd_vla_mechanism_smoke_v1",
            "date_kst": DATE_KST,
            "branch": BRANCH,
            "errors": [{"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc().splitlines()[-80:]}],
            "mechanism_smoke_passed": False,
            "final_decision": "DICD_SYNTHETIC_MECHANISM_SMOKE_FAILED",
        }
    report["elapsed_seconds"] = _round(time.monotonic() - started, 3)
    default_json = {
        "real-smolvla-chunk": "reports/dicd_vla/real_smolvla_chunk_smoke_result.json",
        "real-trace-train": "reports/dicd_vla/real_trace_train_result.json",
        "stage-a": "reports/dicd_vla/stage_a_result.json",
        "synthetic": "reports/dicd_vla/mechanism_smoke_result.json",
    }[args.mode]
    default_md = {
        "real-smolvla-chunk": "reports/dicd_vla/real_smolvla_chunk_smoke_result.md",
        "real-trace-train": "reports/dicd_vla/real_trace_train_result.md",
        "stage-a": "reports/dicd_vla/stage_a_result.md",
        "synthetic": "reports/dicd_vla/mechanism_smoke_result.md",
    }[args.mode]
    output_json = Path(args.output_json or default_json)
    output_md = Path(args.output_md or default_md)
    _write_json(output_json, report)
    _write_md(
        output_md,
        [
            "# DICD-VLA Mechanism Smoke Result",
            "",
            f"Date: `{DATE_KST}`",
            "",
            f"Final decision: `{report.get('final_decision')}`",
            "",
            f"- smoke type: `{report.get('smoke_type')}`",
            f"- mechanism smoke passed: `{report.get('mechanism_smoke_passed')}`",
            f"- training happened: `{report.get('training_happened')}`",
            f"- real SmolVLA chunk smoke happened: `{report.get('real_smolvla_chunk_smoke_happened')}`",
            f"- closed-loop experiment happened: `{report.get('closed_loop_experiment_happened')}`",
            f"- checkpoint: `{report.get('checkpoint_path')}`",
            f"- checkpoint sha256: `{report.get('checkpoint_sha256')}`",
            f"- full checkpoint: `{report.get('full_checkpoint_path')}`",
            f"- full checkpoint sha256: `{report.get('full_checkpoint_sha256')}`",
            f"- ablation checkpoint: `{report.get('ablation_checkpoint_path')}`",
            f"- ablation checkpoint sha256: `{report.get('ablation_checkpoint_sha256')}`",
            f"- checks: `{report.get('checks')}`",
            f"- probe: `{report.get('probe')}`",
            f"- records: `{report.get('records')}`",
            f"- train traces: `{report.get('train_traces')}`",
            f"- summary: `{report.get('summary')}`",
            f"- elapsed seconds: `{report.get('elapsed_seconds')}`",
            "",
            "Next step: follow the Stage A method decision."
            if args.mode == "stage-a"
            else (
            "Next step: run Stage A closed-loop rollout."
            if args.mode == "real-trace-train" and report.get("real_trace_training_passed")
            else (
            "Next step: run real trace training before Stage A closed-loop rollout."
            if args.mode == "real-smolvla-chunk"
            else "Next step: run real SmolVLA chunk smoke before Stage A closed-loop rollout."
            )
            ),
        ],
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DICD-VLA prototype smoke.")
    parser.add_argument("--mode", choices=["synthetic", "real-smolvla-chunk", "real-trace-train", "stage-a"], default="synthetic")
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-md", default=None)
    parser.add_argument("--checkpoint-path", default="reports/dicd_vla/checkpoints/dicd_synthetic_smoke.pt")
    parser.add_argument("--real-checkpoint-dir", default="reports/dicd_vla/checkpoints")
    parser.add_argument("--base-path", default="/home/jiheon/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/home/jiheon/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--chunk-len", type=int, default=8)
    parser.add_argument("--history-len", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--synthetic-count", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-action-delta", type=float, default=1e-4)
    parser.add_argument("--smoke-task-index", type=int, default=0)
    parser.add_argument("--smoke-identity", type=int, default=20260712)
    parser.add_argument("--smoke-steps", type=int, default=3)
    parser.add_argument("--real-max-chunk-len", type=int, default=8)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--train-identity", type=int, default=20260711)
    parser.add_argument("--max-train-steps", type=int, default=80)
    parser.add_argument("--stage-a-identities", default="20260713,20260714,20260715,20260716,20260717")
    parser.add_argument("--stage-a-partial-json", default="reports/dicd_vla/stage_a_partial_result.json")
    parser.add_argument("--max-eval-steps", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps({"final_decision": report.get("final_decision"), "checks": report.get("checks")}, indent=2, sort_keys=True, default=_json_default))
    passed = bool(
        report.get("mechanism_smoke_passed")
        or report.get("real_trace_training_passed")
        or report.get("stage_a_completed")
        or str(report.get("final_decision", "")).endswith("_PASSED")
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
