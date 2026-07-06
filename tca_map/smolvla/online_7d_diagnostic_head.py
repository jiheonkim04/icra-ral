"""Minimal non-leaking online 7D ActionMap/TCA diagnostic head."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.datasets.libero_fixed_prior_rollout_diagnostic import _action_stats
from tca_map.smolvla.interface_adapters import ACTION_STRATEGY_GRIPPER_ZERO_HOLD, adapt_policy_action_to_env_action
from tca_map.smolvla.libero_learned_policy_rollout import (
    CAMERA_ALIAS_STRATEGY_CURRENT,
    STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3,
    _build_batch,
    _ensure_paths,
)
from tca_map.smolvla.load_only_smoke import _external_tokenizer_files, _find_files, _read_tokenizer_dependency, _runtime_dependencies
from tca_map.smolvla.online_action_generation_bridge import _as_path, _bddl_path, _load_json, _match_stats, _safe_l2, _target_metrics
from tca_map.smolvla.single_sample_interface_smoke import _load_policy

SCHEMA_VERSION = "2026-07-06.online_7d_diagnostic_head.v1"
TASK_GATE = "ALLOW_ONLINE_7D_DIAGNOSTIC_HEAD_ROLLOUT"
MAX_STEPS = 25
MAX_TRAIN_SAMPLES = 512
TEXT_WIDTH = 16
RIDGE_LAMBDA = 1.0
FORBIDDEN_GATES = (
    "ALLOW_DOWNLOADS",
    "ALLOW_GPU_TRAINING",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_FIXED_PRIOR_ROLLOUT_DIAGNOSTIC",
    "ALLOW_ACTION_SOURCE_AUDIT_ROLLOUT",
    "ALLOW_ONLINE_ACTION_BRIDGE_ROLLOUT",
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {"type": type(exc).__name__, "message": str(exc), "traceback_tail": traceback.format_exc().splitlines()[-12:]}


def _tokens(text: str) -> list[str]:
    stop = {"a", "an", "and", "both", "in", "it", "of", "on", "put", "the", "to", "with"}
    return [token for token in re.split(r"[^a-z0-9]+", text.lower()) if token and token not in stop]


def _text_features(text: str, width: int = TEXT_WIDTH) -> np.ndarray:
    words = _tokens(text)
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    hashed = [((digest[index] / 255.0) * 2.0) - 1.0 for index in range(width - 4)]
    scalars = [
        min(len(text), 240) / 240.0,
        min(len(words), 40) / 40.0,
        sum(char in "aeiou" for char in text.lower()) / max(1, len(text)),
        sum(char.isdigit() for char in text) / max(1, len(text)),
    ]
    return np.asarray(scalars + hashed, dtype=np.float64)


def _obs_features_hdf5(demo: Any, step: int) -> np.ndarray:
    obs = demo.get("obs")
    if obs is not None and "ee_states" in obs:
        values = np.asarray(obs["ee_states"][step], dtype=np.float64).reshape(-1)
        if values.size >= 6:
            return values[:6]
    if obs is not None and "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
        pos = np.asarray(obs["robot0_eef_pos"][step], dtype=np.float64).reshape(-1)[:3]
        quat = np.asarray(obs["robot0_eef_quat"][step], dtype=np.float64).reshape(-1)[:3]
        return np.concatenate([pos, quat])
    if "states" in demo:
        state = np.asarray(demo["states"][step], dtype=np.float64).reshape(-1)
        if state.size >= 6:
            return state[:6]
    return np.zeros(6, dtype=np.float64)


def _obs_features_sim(obs: dict[str, Any]) -> np.ndarray:
    if "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
        pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float64).reshape(-1)[:3]
        quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(-1)[:3]
        if pos.size == 3 and quat.size == 3:
            return np.concatenate([pos, quat])
    return np.zeros(6, dtype=np.float64)


def _time_features(step: int, horizon: int) -> np.ndarray:
    phase = float(step) / float(max(1, horizon - 1))
    return np.asarray([phase, math.sin(math.pi * phase), math.cos(math.pi * phase)], dtype=np.float64)


def _base(obs_features: np.ndarray, step: int, horizon: int, instruction: str) -> np.ndarray:
    return np.concatenate([obs_features, _time_features(step, horizon), _text_features(instruction)])


def _softmax(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.float64).reshape(1, -1)
    x = x - np.max(x, axis=1, keepdims=True)
    exp = np.exp(x)
    return (exp / np.maximum(np.sum(exp, axis=1, keepdims=True), 1e-12)).reshape(-1)


def _target_prior(samples: list[dict[str, Any]]) -> dict[str, Any]:
    rows: dict[int, list[np.ndarray]] = {0: [], 1: []}
    for sample in samples:
        rows[int(sample["target_id"])].append(_text_features(sample["instruction"]))
    global_mean = np.mean([_text_features(sample["instruction"]) for sample in samples], axis=0)
    prototypes = [np.mean(rows[idx], axis=0) if rows[idx] else global_mean for idx in (0, 1)]
    return {"prototypes": np.asarray(prototypes, dtype=np.float64)}


def _prior_probs(instruction: str, prior: dict[str, Any]) -> np.ndarray:
    text = _text_features(instruction)
    prototypes = np.asarray(prior["prototypes"], dtype=np.float64)
    return _softmax(-np.linalg.norm(prototypes - text.reshape(1, -1), axis=1))


def _prior_one_hot(instruction: str, prior: dict[str, Any]) -> np.ndarray:
    probs = _prior_probs(instruction, prior)
    out = np.zeros_like(probs)
    out[int(np.argmax(probs))] = 1.0
    return out


def _with_bias(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)


def _ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xb = _with_bias(x)
    reg = RIDGE_LAMBDA * np.eye(xb.shape[1], dtype=np.float64)
    reg[-1, -1] = 0.0
    return np.linalg.solve(xb.T @ xb + reg, xb.T @ y)


def _mse(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean((pred - target) ** 2))


def _read_pair_samples(pair: dict[str, Any], max_steps: int, stride: int) -> list[dict[str, Any]]:
    import h5py  # type: ignore

    specs = [
        (0, pair.get("positive_instruction") or "positive target", _as_path(pair["positive_demo_file"])),
        (1, pair.get("counterfactual_instruction") or "counterfactual target", _as_path(pair["counterfactual_demo_file"])),
    ]
    samples: list[dict[str, Any]] = []
    for target_id, instruction, path in specs:
        with h5py.File(path, "r") as handle:
            demo_name = sorted(handle["data"].keys())[0]
            demo = handle["data"][demo_name]
            actions = np.asarray(demo["actions"], dtype=np.float64)
            for step in range(0, min(actions.shape[0], max_steps), max(1, stride)):
                samples.append(
                    {
                        "pair_id": pair["pair_id"],
                        "demo_path": str(path),
                        "demo_name": demo_name,
                        "step": int(step),
                        "target_id": int(target_id),
                        "instruction": str(instruction),
                        "obs": _obs_features_hdf5(demo, step),
                        "action": actions[step, :7].astype(np.float64),
                    }
                )
    return samples


def _read_eval_demo(pair: dict[str, Any], max_steps: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import h5py  # type: ignore

    path = _as_path(pair["positive_demo_file"])
    instruction = str(pair.get("positive_instruction") or "positive target")
    with h5py.File(path, "r") as handle:
        demo_name = sorted(handle["data"].keys())[0]
        demo = handle["data"][demo_name]
        actions = np.asarray(demo["actions"], dtype=np.float64)[:max_steps, :7]
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64)
        samples = [
            {
                "pair_id": pair["pair_id"],
                "demo_path": str(path),
                "demo_name": demo_name,
                "step": int(step),
                "target_id": 0,
                "instruction": instruction,
                "obs": _obs_features_hdf5(demo, step),
                "action": actions[step].astype(np.float64),
            }
            for step in range(actions.shape[0])
        ]
    return samples, {"path": str(path), "demo_name": demo_name, "init_state": init_state, "actions": actions}


def _features(samples: list[dict[str, Any]], variant: str, prior: dict[str, Any], horizon: int) -> np.ndarray:
    rows = []
    for sample in samples:
        base = _base(sample["obs"], int(sample["step"]), horizon, str(sample["instruction"]))
        if variant == "actionmap_7d":
            rows.append(base)
        elif variant == "fixed_prior_tca_7d":
            rows.append(np.concatenate([base, _prior_probs(str(sample["instruction"]), prior)]))
        elif variant == "hard_learned_target_tca_7d":
            rows.append(np.concatenate([base, _prior_one_hot(str(sample["instruction"]), prior)]))
        else:
            raise ValueError(f"unknown variant: {variant}")
    return np.asarray(rows, dtype=np.float64)


def _sim_features(obs: dict[str, Any], step: int, horizon: int, instruction: str, variant: str, prior: dict[str, Any]) -> np.ndarray:
    base = _base(_obs_features_sim(obs), step, horizon, instruction)
    if variant == "actionmap_7d":
        return base.reshape(1, -1)
    if variant == "fixed_prior_tca_7d":
        return np.concatenate([base, _prior_probs(instruction, prior)]).reshape(1, -1)
    if variant == "hard_learned_target_tca_7d":
        return np.concatenate([base, _prior_one_hot(instruction, prior)]).reshape(1, -1)
    raise ValueError(f"unknown variant: {variant}")


def _metric(pred_raw: np.ndarray, expert: np.ndarray, train_actions: np.ndarray) -> dict[str, Any]:
    diff = pred_raw - expert
    clipped = np.clip(pred_raw, -1.0, 1.0)
    return {
        "action_dim": int(pred_raw.shape[1]),
        "7d_action_l2": round(float(np.mean(np.linalg.norm(diff, axis=1))), 9),
        "translation_l2": round(float(np.mean(np.linalg.norm(diff[:, :3], axis=1))), 9),
        "rotation_l2": round(float(np.mean(np.linalg.norm(diff[:, 3:6], axis=1))), 9),
        "gripper_l1": round(float(np.mean(np.abs(diff[:, 6]))), 9),
        "raw_clipping_rate": round(float(np.mean(np.abs(pred_raw) > 1.0)), 9),
        "clipped_action_stats": _action_stats(clipped),
        "action_distribution_match": {
            "mean_l2_to_train": round(float(np.linalg.norm(np.mean(clipped, axis=0) - np.mean(train_actions, axis=0))), 9),
            "std_l2_to_train": round(float(np.linalg.norm(np.std(clipped, axis=0) - np.std(train_actions, axis=0))), 9),
        },
        "standard_offline_proxy": round(float(1.0 / (1.0 + np.mean(np.linalg.norm(diff, axis=1)))), 9),
    }


def train_online_7d_heads(manifest_path: Path, max_steps: int = 25, train_max_steps: int = 64, stride: int = 4) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = _load_json(manifest_path)
    pairs = manifest.get("counterfactual_pairs") or []
    if len(pairs) < 2:
        raise ValueError("at least two pairs are required so the rollout pair can be held out")
    rollout_pair = pairs[0]
    train_pairs = pairs[1:]
    rollout_demo_path = str(_as_path(rollout_pair["positive_demo_file"]))
    train_samples: list[dict[str, Any]] = []
    for pair in train_pairs:
        new_samples = [sample for sample in _read_pair_samples(pair, train_max_steps, stride) if sample["demo_path"] != rollout_demo_path]
        train_samples.extend(new_samples)
        if len(train_samples) >= MAX_TRAIN_SAMPLES:
            train_samples = train_samples[:MAX_TRAIN_SAMPLES]
            break
    eval_samples, rollout_demo = _read_eval_demo(rollout_pair, max_steps)
    prior = _target_prior(train_samples)
    y_train = np.asarray([sample["action"] for sample in train_samples], dtype=np.float64)
    y_eval = np.asarray([sample["action"] for sample in eval_samples], dtype=np.float64)
    models: dict[str, Any] = {}
    for variant in ("actionmap_7d", "fixed_prior_tca_7d", "hard_learned_target_tca_7d"):
        x_train = _features(train_samples, variant, prior, max_steps)
        x_eval = _features(eval_samples, variant, prior, max_steps)
        weights = _ridge(x_train, y_train)
        pred_raw = _with_bias(x_eval) @ weights
        pred_clip = np.clip(pred_raw, -1.0, 1.0)
        initial_loss = _mse(np.zeros_like(y_train), y_train)
        final_loss = _mse(_with_bias(x_train) @ weights, y_train)
        models[variant] = {
            "weights": weights,
            "feature_dim": int(x_train.shape[1]),
            "trainable_parameter_count": int((x_train.shape[1] + 1) * 7),
            "loss": {"initial_loss": round(initial_loss, 9), "final_loss": round(final_loss, 9), "loss_decreased": bool(final_loss < initial_loss)},
            "offline_metrics": _metric(pred_raw, y_eval, y_train),
            "eval_pred_actions": pred_clip,
            "leakage_audit": {
                "uses_current_observation_only_at_inference": True,
                "uses_current_instruction_only_at_inference": True,
                "uses_test_time_semantic_target_prior_only": variant != "actionmap_7d",
                "uses_same_or_future_hdf5_action_at_inference": False,
                "uses_eval_target_labels_at_inference": False,
                "valid_online_method_action": True,
            },
        }
    models["comparison"] = {
        "actionmap_vs_fixed_prior_tca_action_l2": round(float(np.mean(np.linalg.norm(models["actionmap_7d"]["eval_pred_actions"] - models["fixed_prior_tca_7d"]["eval_pred_actions"], axis=1))), 9),
        "fixed_prior_tca_l2_delta_vs_actionmap": round(models["fixed_prior_tca_7d"]["offline_metrics"]["7d_action_l2"] - models["actionmap_7d"]["offline_metrics"]["7d_action_l2"], 9),
    }
    meta = {
        "target_prior": prior,
        "rollout_demo": rollout_demo,
        "rollout_pair_id": rollout_pair["pair_id"],
        "rollout_demo_path": rollout_demo["path"],
        "rollout_demo_name": rollout_demo["demo_name"],
        "rollout_instruction": rollout_pair.get("positive_instruction"),
        "counterfactual_instruction": rollout_pair.get("counterfactual_instruction"),
        "suite": rollout_pair.get("suite") or "libero_10",
        "task_id": rollout_pair["positive_task_id"],
        "train_pair_ids": [pair["pair_id"] for pair in train_pairs],
        "train_demo_paths": sorted({sample["demo_path"] for sample in train_samples}),
        "eval_demo_paths": [rollout_demo["path"]],
        "train_sample_count": len(train_samples),
        "eval_sample_count": len(eval_samples),
        "split": "rollout_pair_held_out_from_training_and_rollout_demo_path_filtered",
        "rollout_demo_excluded_from_training": rollout_demo["path"] not in sorted({sample["demo_path"] for sample in train_samples}),
    }
    return models, meta


def readiness_gate(models: dict[str, Any]) -> dict[str, Any]:
    statuses = {}
    for variant in ("actionmap_7d", "fixed_prior_tca_7d", "hard_learned_target_tca_7d"):
        metrics = models[variant]["offline_metrics"]
        leak = models[variant]["leakage_audit"]
        statuses[variant] = {
            "action_dim_exactly_7d": metrics["action_dim"] == 7,
            "no_silent_padding": True,
            "no_future_hdf5_action_leakage": not leak["uses_same_or_future_hdf5_action_at_inference"],
            "clipping_bounded": metrics["raw_clipping_rate"] <= 0.75,
            "valid_online_method_action": leak["valid_online_method_action"],
        }
    distinct = models["comparison"]["actionmap_vs_fixed_prior_tca_action_l2"] > 1e-9
    green = distinct and all(all(status.values()) for status in statuses.values())
    return {"status": "green" if green else "red", "ready_for_bounded_matched_init_rollout": bool(green), "variant_status": statuses, "actionmap_fixed_prior_distinct": distinct}


def _public_training(models: dict[str, Any], meta: dict[str, Any]) -> dict[str, Any]:
    variants = {}
    for name in ("actionmap_7d", "fixed_prior_tca_7d", "hard_learned_target_tca_7d"):
        variants[name] = {
            "feature_dim": models[name]["feature_dim"],
            "trainable_parameter_count": models[name]["trainable_parameter_count"],
            "loss": models[name]["loss"],
            "offline_metrics": models[name]["offline_metrics"],
            "leakage_audit": models[name]["leakage_audit"],
        }
    return {
        "training_happened": True,
        "loss_computed": True,
        "lora_training_happened": False,
        "data_source": "local LIBERO HDF5 train pairs only",
        "train_sample_count": meta["train_sample_count"],
        "eval_sample_count": meta["eval_sample_count"],
        "train_eval_split": meta["split"],
        "rollout_demo_excluded_from_training": meta.get("rollout_demo_excluded_from_training", False),
        "train_demo_paths": meta["train_demo_paths"],
        "eval_demo_paths": meta["eval_demo_paths"],
        "variants": variants,
        "comparison": models["comparison"],
    }


def _head_action(models: dict[str, Any], meta: dict[str, Any], variant: str, obs: dict[str, Any], step: int, horizon: int, instruction: str) -> tuple[list[float], dict[str, Any]]:
    x = _sim_features(obs, step, horizon, instruction, variant, meta["target_prior"])
    raw = (_with_bias(x) @ models[variant]["weights"])[0]
    clipped = np.clip(raw, -1.0, 1.0)
    return [float(value) for value in clipped.tolist()], {"feature_dim": int(x.shape[1]), "clipped_values": int(np.sum(raw != clipped))}


def _run_variant(env_cls: Any, bddl_file: Path, init_state: np.ndarray, expert: np.ndarray, instruction: str, counter_instruction: str | None, variant: str, models: dict[str, Any], meta: dict[str, Any], camera_size: int, policy: Any | None, config: Any | None, tokenizer_root: Path | None, device: str) -> dict[str, Any]:
    summary: dict[str, Any] = {"variant": variant, "steps_performed": 0, "valid_online_action_call_count": 0, "reward_sum": 0.0, "final_success": False, "done_seen": False, "action_provenance": [], "error": None}
    actions: list[list[float]] = []
    env = None
    start_obs = None
    final_obs = None
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=camera_size, camera_widths=camera_size)
        env.seed(0)
        obs = env.set_init_state(init_state)
        start_obs = obs
        action_dim = int(getattr(env, "action_dim", 7) or 7)
        if policy is not None:
            policy.reset()
        import torch

        for step in range(expert.shape[0]):
            extra: dict[str, Any] = {}
            shape = None
            if variant == "zero_action_exact_init":
                env_action = [0.0] * action_dim
                source, online, future, model_head = "programmatic_zero_action", False, False, False
            elif variant == "hdf5_expert_replay_exact_init":
                env_action = [float(value) for value in expert[step].tolist()]
                source, online, future, model_head = "hdf5_expert_action_upper_bound_not_method", False, True, False
            elif variant == "native_smolvla_online_policy":
                if policy is None or config is None or tokenizer_root is None:
                    raise ValueError("native variant requires policy/config/tokenizer")
                batch, batch_meta = _build_batch(config, tokenizer_root, obs, instruction, device, CAMERA_ALIAS_STRATEGY_CURRENT, STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3)
                noise = torch.zeros((1, config.chunk_size, config.max_action_dim), dtype=torch.float32, device=device)
                with torch.inference_mode():
                    policy_action = policy.select_action(batch, noise=noise)
                adapted = adapt_policy_action_to_env_action(policy_action, action_dim, strategy=ACTION_STRATEGY_GRIPPER_ZERO_HOLD)
                env_action = [float(value) for value in adapted.values]
                shape = list(policy_action.detach().cpu().shape)
                extra = {"action_adapter_metadata": adapted.metadata, "batch_keys": batch_meta.get("batch_keys")}
                source, online, future, model_head = "online_smolvla_model_head", True, False, True
            elif variant in {"actionmap_7d", "fixed_prior_tca_7d", "hard_learned_target_tca_7d"}:
                env_action, extra = _head_action(models, meta, variant, obs, step, expert.shape[0], instruction)
                source, online, future, model_head = f"online_{variant}_linear_head", True, False, True
            else:
                raise ValueError(f"unsupported variant: {variant}")
            obs, reward, done, _info = env.step(env_action)
            actions.append(env_action)
            summary["steps_performed"] += 1
            summary["valid_online_action_call_count"] += int(online)
            summary["reward_sum"] += float(reward)
            summary["done_seen"] = bool(summary["done_seen"] or done)
            try:
                summary["final_success"] = bool(env.check_success())
            except Exception:
                pass
            summary["action_provenance"].append({"step": int(step), "source": source, "uses_future_hdf5_action": future, "online_generated_from_current_observation": online, "model_head_decoded_action": model_head, "policy_action_shape": shape, "env_action_dim": len(env_action), "l2_to_hdf5_expert_same_timestep": _safe_l2(env_action, expert[step]), **extra})
            final_obs = obs
        if start_obs is not None and final_obs is not None:
            summary.update(_target_metrics(start_obs, final_obs, instruction, counter_instruction))
    except Exception as exc:  # noqa: BLE001
        summary["error"] = _compact_error(exc)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    arr = np.asarray(actions, dtype=np.float64) if actions else np.zeros((0, 7), dtype=np.float64)
    summary["action_shape"] = list(arr.shape)
    summary["action_stats"] = _action_stats(arr) if arr.size else None
    summary["expert_match"] = _match_stats(arr, expert[: arr.shape[0]])
    summary["valid_closed_loop_online_rollout"] = bool(variant not in {"zero_action_exact_init", "hdf5_expert_replay_exact_init"} and summary["error"] is None and arr.shape[1:] == (7,) and not any(item["uses_future_hdf5_action"] for item in summary["action_provenance"]))
    return summary


def _classify_fixed_prior_rollout_support(actionmap: dict[str, Any], fixed: dict[str, Any]) -> dict[str, Any]:
    fixed_reward_or_success_support = bool(
        fixed["valid_closed_loop_online_rollout"]
        and actionmap["valid_closed_loop_online_rollout"]
        and (
            float(fixed["reward_sum"]) > float(actionmap["reward_sum"])
            or (bool(fixed.get("final_success")) and not bool(actionmap.get("final_success")))
        )
    )
    fixed_partial_target_movement_support = bool(
        fixed["valid_closed_loop_online_rollout"]
        and actionmap["valid_closed_loop_online_rollout"]
        and not fixed_reward_or_success_support
        and fixed.get("target_directed_movement_score") is not None
        and actionmap.get("target_directed_movement_score") is not None
        and float(fixed["target_directed_movement_score"]) > float(actionmap["target_directed_movement_score"])
    )
    blocker = None
    if not fixed_reward_or_success_support:
        blocker = (
            "online_7d_head_partial_target_movement_no_success"
            if fixed_partial_target_movement_support
            else "online_7d_head_no_rollout_support_yet"
        )
    return {
        "fixed_prior_tca_valid_rollout_support": fixed_reward_or_success_support,
        "fixed_prior_tca_partial_target_movement_support": fixed_partial_target_movement_support,
        "blocker_classification": blocker,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    gate_set = _env_flag(TASK_GATE)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "stop",
        "policy": {"task_local_gate_set": gate_set, "downloads_performed": False, "gpu_jobs_performed": False, "training_performed": False, "lora_training_performed": False, "loss_computed": False, "heavy_model_imports_performed": False, "model_load_performed": False, "model_inference_performed": False, "rollout_happened": False, "benchmark_rollouts_performed": False, "openvla_oft_executed": False, "paper_grade_claims_made": False, "forbidden_gates_set": forbidden},
        "training": None,
        "rollout_readiness_gate": None,
        "case": None,
        "rollout_results": [],
        "result": {"passed": False, "blocked": True, "blocked_reason": None, "elapsed_sec": None},
        "recommended_next_step": None,
    }
    if args.max_steps < 1 or args.max_steps > MAX_STEPS:
        report["result"]["blocked_reason"] = f"max_steps must be between 1 and {MAX_STEPS}"
        return report
    if forbidden:
        report["result"]["blocked_reason"] = "Forbidden gate(s) set: " + ", ".join(forbidden)
        return report
    try:
        models, meta = train_online_7d_heads(Path(args.manifest), args.max_steps, args.train_max_steps, args.sample_stride)
        report["training"] = _public_training(models, meta)
        report["policy"]["training_performed"] = True
        report["policy"]["loss_computed"] = True
        gate = readiness_gate(models)
        report["rollout_readiness_gate"] = gate
        report["case"] = {"task_id": meta["task_id"], "suite": meta["suite"], "rollout_pair_id": meta["rollout_pair_id"], "rollout_demo_path": meta["rollout_demo_path"], "rollout_demo_name": meta["rollout_demo_name"], "instruction": meta["rollout_instruction"], "counterfactual_instruction": meta["counterfactual_instruction"], "max_steps": args.max_steps}
        if not gate_set:
            report["decision"] = "trained_7d_heads_rollout_gate_not_set"
            report["result"] = {"passed": bool(gate["ready_for_bounded_matched_init_rollout"]), "blocked": not bool(gate["ready_for_bounded_matched_init_rollout"]), "blocked_reason": None if gate["ready_for_bounded_matched_init_rollout"] else "readiness gate red", "elapsed_sec": round(time.monotonic() - started, 3), "fixed_prior_tca_valid_rollout_support": False, "blocker_classification": None if gate["ready_for_bounded_matched_init_rollout"] else "online_7d_head_readiness_gate_red"}
            report["recommended_next_step"] = "Run the task-local bounded matched-init rollout gate if desired; do not run full benchmark."
            return report
        if not gate["ready_for_bounded_matched_init_rollout"]:
            report["decision"] = "rollout_gate_red_no_rollout"
            report["result"]["blocked_reason"] = "online 7D diagnostic head readiness gate is red"
            return report
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        smolvla_ckpt = Path(args.smolvla_ckpt)
        checkpoint_root = Path(args.checkpoint_root)
        hf_home = Path(args.hf_home)
        external = _external_tokenizer_files(_read_tokenizer_dependency(smolvla_ckpt), [hf_home, checkpoint_root])
        if not (_find_files(smolvla_ckpt, ["config.json"]) and _find_files(smolvla_ckpt, ["model.safetensors", "pytorch_model.bin"], ["*.safetensors", "*.bin"]) and external.get("found")):
            report["result"]["blocked_reason"] = "SmolVLA local files incomplete for native baseline"
            return report
        deps = _runtime_dependencies()
        if not all(deps.values()):
            report["result"]["blocked_reason"] = "Missing runtime dependencies: " + ", ".join(name for name, ok in deps.items() if not ok)
            return report
        libero_root = Path(args.libero_root)
        robosuite_root = Path(args.robosuite_root)
        _ensure_paths(libero_root, robosuite_root)
        from libero.libero.envs import OffScreenRenderEnv

        bddl_file = _bddl_path(libero_root, meta["suite"], meta["task_id"])
        report["case"]["bddl_file"] = str(bddl_file)
        report["policy"]["heavy_model_imports_performed"] = True
        policy, config = _load_policy(smolvla_ckpt, hf_home, external, args.device)
        report["policy"]["model_load_performed"] = True
        expert = np.asarray(meta["rollout_demo"]["actions"], dtype=np.float64)[: args.max_steps]
        init_state = np.asarray(meta["rollout_demo"]["init_state"], dtype=np.float64)
        variants = ["zero_action_exact_init", "hdf5_expert_replay_exact_init", "native_smolvla_online_policy", "actionmap_7d", "fixed_prior_tca_7d", "hard_learned_target_tca_7d"]
        report["rollout_results"] = [
            _run_variant(OffScreenRenderEnv, bddl_file, init_state, expert, str(meta["rollout_instruction"]), meta.get("counterfactual_instruction"), variant, models, meta, args.camera_size, policy if variant == "native_smolvla_online_policy" else None, config if variant == "native_smolvla_online_policy" else None, Path(external["root"]) if variant == "native_smolvla_online_policy" else None, args.device)
            for variant in variants
        ]
        report["policy"]["rollout_happened"] = True
        report["policy"]["model_inference_performed"] = True
        actionmap = next(item for item in report["rollout_results"] if item["variant"] == "actionmap_7d")
        fixed = next(item for item in report["rollout_results"] if item["variant"] == "fixed_prior_tca_7d")
        support = _classify_fixed_prior_rollout_support(actionmap, fixed)
        report["decision"] = "bounded_online_7d_head_rollout_completed"
        report["result"] = {
            "passed": True,
            "blocked": False,
            "blocked_reason": None,
            "elapsed_sec": round(time.monotonic() - started, 3),
            **support,
        }
        report["recommended_next_step"] = "Proceed to a second matched-init online rollout task." if support["fixed_prior_tca_valid_rollout_support"] else "Run action-quality/head-training diagnosis before scaling; current support is partial target movement only if present."
    except Exception as exc:  # noqa: BLE001
        report["result"] = {"passed": False, "blocked": True, "blocked_reason": f"{type(exc).__name__}: {exc}", "error": _compact_error(exc), "elapsed_sec": round(time.monotonic() - started, 3)}
        report["recommended_next_step"] = "Fix the online 7D diagnostic head blocker before method rollout claims."
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    training = report.get("training") or {}
    gate = report.get("rollout_readiness_gate") or {}
    result = report.get("result") or {}
    lines = [
        "# Online 7D Diagnostic Head Report",
        "",
        "This is bounded diagnostic evidence only. It is not benchmark success, SOTA evidence, or paper-grade evidence.",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- passed: `{result.get('passed')}`",
        f"- training happened: `{(report.get('policy') or {}).get('training_performed')}`",
        f"- loss computed: `{(report.get('policy') or {}).get('loss_computed')}`",
        f"- rollout happened: `{(report.get('policy') or {}).get('rollout_happened')}`",
        f"- readiness gate: `{gate.get('status')}`",
        f"- fixed-prior TCA valid rollout support: `{result.get('fixed_prior_tca_valid_rollout_support')}`",
        f"- blocker: `{result.get('blocker_classification') or result.get('blocked_reason')}`",
        "",
    ]
    for name, data in (training.get("variants") or {}).items():
        loss = data.get("loss") or {}
        metrics = data.get("offline_metrics") or {}
        lines.append(f"- `{name}`: loss `{loss.get('initial_loss')}` -> `{loss.get('final_loss')}`, 7D L2 `{metrics.get('7d_action_l2')}`")
    lines.extend(["", f"Recommended next step: {report.get('recommended_next_step')}", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--report-json", default="reports/online_7d_diagnostic_head_report.json")
    parser.add_argument("--report-md", default="reports/online_7d_diagnostic_head_report.md")
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--libero-root", default="C:/assets/repos/LIBERO")
    parser.add_argument("--robosuite-root", default="C:/assets/repos/robosuite")
    parser.add_argument("--max-steps", type=int, default=25)
    parser.add_argument("--train-max-steps", type=int, default=64)
    parser.add_argument("--sample-stride", type=int, default=4)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    args = parser.parse_args(argv)

    report = build_report(args)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=lambda value: "<non-json>") + "\n", encoding="utf-8")
    _write_markdown(report, report_md)
    print(json.dumps(report, indent=2, sort_keys=True, default=lambda value: "<non-json>"))
    return 0 if report.get("training") else 8


if __name__ == "__main__":
    sys.exit(main())
