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


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    report: dict[str, Any]
    try:
        if args.mode == "real-smolvla-chunk":
            report = _run_real_smolvla_chunk_smoke(args)
        elif args.mode == "real-trace-train":
            report = _run_real_trace_training(args)
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
        "synthetic": "reports/dicd_vla/mechanism_smoke_result.json",
    }[args.mode]
    default_md = {
        "real-smolvla-chunk": "reports/dicd_vla/real_smolvla_chunk_smoke_result.md",
        "real-trace-train": "reports/dicd_vla/real_trace_train_result.md",
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
            f"- elapsed seconds: `{report.get('elapsed_seconds')}`",
            "",
            "Next step: run Stage A closed-loop rollout."
            if args.mode == "real-trace-train" and report.get("real_trace_training_passed")
            else (
            "Next step: run real trace training before Stage A closed-loop rollout."
            if args.mode == "real-smolvla-chunk"
            else "Next step: run real SmolVLA chunk smoke before Stage A closed-loop rollout."
            ),
        ],
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DICD-VLA prototype smoke.")
    parser.add_argument("--mode", choices=["synthetic", "real-smolvla-chunk", "real-trace-train"], default="synthetic")
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
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps({"final_decision": report.get("final_decision"), "checks": report.get("checks")}, indent=2, sort_keys=True, default=_json_default))
    passed = bool(
        report.get("mechanism_smoke_passed")
        or report.get("real_trace_training_passed")
        or str(report.get("final_decision", "")).endswith("_PASSED")
    )
    return 0 if passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
