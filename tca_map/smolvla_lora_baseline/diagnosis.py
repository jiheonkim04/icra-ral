"""SmolVLA LoRA baseline diagnosis.

This runner diagnoses why the standard SmolVLA LoRA baseline loses to a
mean-action baseline. It is not a new method route.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla_lora_baseline import diagnostic as base


HEAVY_IMPORT_GATE = "ALLOW_HEAVY_IMPORT"
DIAGNOSIS_GATE = "ALLOW_SMOLVLA_LORA_BASELINE_DIAGNOSIS"
TRAINING_GATE = "ALLOW_SMOLVLA_LORA_BASELINE_DIAGNOSIS_TRAINING"
DEFAULT_HDF5_PATH = base.DEFAULT_HDF5_PATH
FINAL_DECISIONS = {
    "READY_FOR_REAL_METHOD_AFTER_BASELINE",
    "ACTION_INTERFACE_BUG",
    "DATA_TOO_SMALL_OR_LOW_VARIANCE",
    "LORA_CAPACITY_OR_TARGET_MODULE_BLOCKED",
    "KILL_SMOLVLA_LORA_BASELINE",
}
FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_PATCHGUARD_VLA_STATE1B",
    "ALLOW_PATCHGUARD_TINY_LORA_TRAINING",
    "ALLOW_TARGET_GROUNDED_ACTIONMAP",
    "ALLOW_SAFELORA",
]
TARGET_MODULE_VARIANTS = {
    "current_projection_lora": ["state_proj", "action_in_proj", "action_out_proj"],
    "action_head_only_lora": ["action_out_proj"],
    "projection_action_head_lora": ["action_in_proj", "action_out_proj"],
}


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _round(value: float | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _load_actions(path: Path) -> dict[str, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as handle:
        return {
            demo_name: np.asarray(handle["data"][demo_name]["actions"], dtype=np.float32)
            for demo_name in sorted(handle["data"].keys(), key=base._demo_sort_key)
        }


def _records_for_demo_times(path: Path, demo_times: dict[str, list[int]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    task_text = base._safe_task_text(path)
    for demo_name, timesteps in demo_times.items():
        action_length = max(timesteps) + 1 if timesteps else 0
        for timestep in timesteps:
            records.append(
                {
                    "hdf5_path": str(path),
                    "task_name": path.stem,
                    "task_text": task_text,
                    "demo_name": demo_name,
                    "timestep": int(timestep),
                    "action_length": int(action_length),
                    "action_dim": 7,
                }
            )
    return records


def _sample_timesteps(length: int, count: int, start: int = 0, stop: int | None = None) -> list[int]:
    stop = length if stop is None else min(stop, length)
    if stop <= start:
        return [max(0, min(length - 1, start))]
    if count <= 1:
        return [start]
    values = np.linspace(start, stop - 1, num=count, dtype=np.int64).tolist()
    return list(dict.fromkeys(int(x) for x in values))


def _action_stats(actions: np.ndarray) -> dict[str, Any]:
    first6 = actions[:, :6]
    gripper = actions[:, 6] if actions.shape[1] > 6 else np.asarray([], dtype=np.float32)
    return {
        "count": int(actions.shape[0]),
        "action_dim": int(actions.shape[1]),
        "mean": [_round(x) for x in np.mean(actions, axis=0)],
        "std": [_round(x) for x in np.std(actions, axis=0)],
        "variance": [_round(x) for x in np.var(actions, axis=0)],
        "min": [_round(x) for x in np.min(actions, axis=0)],
        "max": [_round(x) for x in np.max(actions, axis=0)],
        "translation_variance_mean": _round(np.mean(np.var(first6[:, :3], axis=0))),
        "rotation_variance_mean": _round(np.mean(np.var(first6[:, 3:6], axis=0))),
        "gripper_variance": _round(np.var(gripper)) if gripper.size else None,
        "gripper_unique_values": [_round(x) for x in np.unique(gripper)[:10]] if gripper.size else [],
    }


def _feature_matrix(records: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    features = []
    labels = []
    cache: dict[Path, Any] = {}
    try:
        for record in records:
            path = Path(record["hdf5_path"])
            if path not in cache:
                cache[path] = h5py.File(path, "r")
            handle = cache[path]
            demo = handle["data"][record["demo_name"]]
            timestep = int(record["timestep"])
            actions = np.asarray(demo["actions"], dtype=np.float32)
            obs = demo["obs"]
            ee = np.asarray(obs["ee_states"][timestep], dtype=np.float32).reshape(-1)[:6]
            frac = np.asarray([timestep / max(1, actions.shape[0] - 1)], dtype=np.float32)
            features.append(np.concatenate([ee, frac], axis=0))
            labels.append(actions[timestep, :7])
    finally:
        for handle in cache.values():
            handle.close()
    return np.stack(features, axis=0).astype(np.float32), np.stack(labels, axis=0).astype(np.float32)


def _ridge_predict(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]], alpha: float = 1e-3) -> dict[str, Any]:
    x_train, y_train = _feature_matrix(train_records)
    x_eval, y_eval = _feature_matrix(eval_records)
    x_mean = x_train.mean(axis=0, keepdims=True)
    x_std = x_train.std(axis=0, keepdims=True) + 1e-6
    xt = (x_train - x_mean) / x_std
    xe = (x_eval - x_mean) / x_std
    xt = np.concatenate([xt, np.ones((xt.shape[0], 1), dtype=np.float32)], axis=1)
    xe = np.concatenate([xe, np.ones((xe.shape[0], 1), dtype=np.float32)], axis=1)
    reg = alpha * np.eye(xt.shape[1], dtype=np.float32)
    weights = np.linalg.solve(xt.T @ xt + reg, xt.T @ y_train)
    pred_train = xt @ weights
    pred_eval = xe @ weights
    return {
        "feature_name": "eef_state6_plus_time_fraction",
        "train": base._metrics_from_predictions([row for row in pred_train], [row for row in y_train]),
        "eval": base._metrics_from_predictions([row for row in pred_eval], [row for row in y_eval]),
    }


def _mlp_predict(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]], steps: int = 400) -> dict[str, Any]:
    import torch

    torch.manual_seed(7)
    x_train_np, y_train_np = _feature_matrix(train_records)
    x_eval_np, y_eval_np = _feature_matrix(eval_records)
    x_mean = x_train_np.mean(axis=0, keepdims=True)
    x_std = x_train_np.std(axis=0, keepdims=True) + 1e-6
    x_train = torch.tensor((x_train_np - x_mean) / x_std, dtype=torch.float32)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    x_eval = torch.tensor((x_eval_np - x_mean) / x_std, dtype=torch.float32)
    model = torch.nn.Sequential(torch.nn.Linear(x_train.shape[1], 32), torch.nn.Tanh(), torch.nn.Linear(32, 7))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2, weight_decay=1e-4)
    losses: list[float] = []
    for _step in range(steps):
        opt.zero_grad(set_to_none=True)
        pred = model(x_train)
        loss = torch.nn.functional.mse_loss(pred, y_train)
        loss.backward()
        opt.step()
        losses.append(_round(loss.detach().cpu()))
    with torch.no_grad():
        pred_train = model(x_train).detach().cpu().numpy()
        pred_eval = model(x_eval).detach().cpu().numpy()
    return {
        "feature_name": "eef_state6_plus_time_fraction",
        "steps": int(steps),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "train": base._metrics_from_predictions([row for row in pred_train], [row for row in y_train_np]),
        "eval": base._metrics_from_predictions([row for row in pred_eval], [row for row in y_eval_np]),
    }


def _previous_action_baseline(records: list[dict[str, Any]], fallback: np.ndarray) -> dict[str, Any]:
    predictions = []
    experts = []
    for record in records:
        timestep = int(record["timestep"])
        if timestep > 0:
            pred = base._expert_action(Path(record["hdf5_path"]), record["demo_name"], timestep - 1)
        else:
            pred = fallback
        predictions.append(np.asarray(pred, dtype=np.float32)[:7])
        experts.append(base._expert_action(Path(record["hdf5_path"]), record["demo_name"], timestep))
    return base._metrics_from_predictions(predictions, experts)


def _per_demo_mean_oracle(records: list[dict[str, Any]]) -> dict[str, Any]:
    actions_by_demo: dict[tuple[str, str], np.ndarray] = {}
    predictions = []
    experts = []
    for record in records:
        key = (record["hdf5_path"], record["demo_name"])
        if key not in actions_by_demo:
            import h5py

            with h5py.File(key[0], "r") as handle:
                actions_by_demo[key] = np.asarray(handle["data"][key[1]]["actions"], dtype=np.float32)
        pred = actions_by_demo[key].mean(axis=0)
        predictions.append(pred[:7])
        experts.append(base._expert_action(Path(record["hdf5_path"]), record["demo_name"], int(record["timestep"])))
    return base._metrics_from_predictions(predictions, experts)


def _mean_baselines(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    mean_action = base._mean_train_action(train_records)
    return {
        "global_mean_action": base.evaluate_constant_action(eval_records, mean_action),
        "previous_action": _previous_action_baseline(eval_records, mean_action),
        "per_demo_mean_oracle_leaky": _per_demo_mean_oracle(eval_records),
    }


def _data_and_split_audit(hdf5_path: Path) -> dict[str, Any]:
    actions_by_demo = _load_actions(hdf5_path)
    demo_names = list(actions_by_demo.keys())
    all_actions = np.concatenate(list(actions_by_demo.values()), axis=0)
    previous_split = base.select_records(hdf5_path, max_train_demos=3, max_eval_demos=2, records_per_demo=3)
    larger_demo_split = base.select_records(hdf5_path, max_train_demos=30, max_eval_demos=10, records_per_demo=10)

    same_demo_train_times: dict[str, list[int]] = {}
    same_demo_eval_times: dict[str, list[int]] = {}
    for demo_name in demo_names[:10]:
        length = actions_by_demo[demo_name].shape[0]
        boundary = int(length * 0.7)
        same_demo_train_times[demo_name] = _sample_timesteps(length, 8, 0, boundary)
        same_demo_eval_times[demo_name] = _sample_timesteps(length, 4, boundary, length)
    same_demo_train = _records_for_demo_times(hdf5_path, same_demo_train_times)
    same_demo_eval = _records_for_demo_times(hdf5_path, same_demo_eval_times)

    task_files = sorted(hdf5_path.parent.glob("*.hdf5"))
    task_holdout = {
        "available_task_count_in_suite": len(task_files),
        "example_train_tasks": [path.stem for path in task_files[:3]],
        "example_holdout_tasks": [path.stem for path in task_files[3:6]],
        "feasible_without_download": len(task_files) >= 2,
        "executed_training": False,
    }

    train_actions = np.stack(
        [base._expert_action(Path(record["hdf5_path"]), record["demo_name"], int(record["timestep"])) for record in larger_demo_split["train_records"]],
        axis=0,
    )
    eval_actions = np.stack(
        [base._expert_action(Path(record["hdf5_path"]), record["demo_name"], int(record["timestep"])) for record in larger_demo_split["eval_records"]],
        axis=0,
    )
    return {
        "previous_split_why_9_train_6_eval": (
            "The prior runner sampled 3 train demos and 2 eval demos with 3 timesteps per demo, "
            "so records were sampled timestep/action-window records: 3*3 train and 2*3 eval."
        ),
        "records_are": "sampled observation/action-chunk windows; each record is one demo/timestep plus a 50-step action chunk target",
        "raw_demo_count": len(demo_names),
        "raw_timestep_count": int(all_actions.shape[0]),
        "raw_timestep_count_by_demo_first10": {name: int(actions_by_demo[name].shape[0]) for name in demo_names[:10]},
        "raw_action_stats": _action_stats(all_actions),
        "previous_sampled_split": {
            "train_count": previous_split["train_count"],
            "eval_count": previous_split["eval_count"],
            "train_demos": previous_split["train_demos"],
            "eval_demos": previous_split["eval_demos"],
            "mean_baselines": _mean_baselines(previous_split["train_records"], previous_split["eval_records"]),
        },
        "larger_demo_holdout_split": {
            "train_count": larger_demo_split["train_count"],
            "eval_count": larger_demo_split["eval_count"],
            "train_demos": larger_demo_split["train_demos"],
            "eval_demos": larger_demo_split["eval_demos"],
            "train_action_stats": _action_stats(train_actions),
            "eval_action_stats": _action_stats(eval_actions),
            "mean_baselines": _mean_baselines(larger_demo_split["train_records"], larger_demo_split["eval_records"]),
        },
        "same_demo_time_holdout_split": {
            "train_count": len(same_demo_train),
            "eval_count": len(same_demo_eval),
            "demos": demo_names[:10],
            "mean_baselines": _mean_baselines(same_demo_train, same_demo_eval),
        },
        "task_holdout_audit": task_holdout,
        "low_variance_indicators": {
            "global_action_l2_std_first6": _round(float(np.linalg.norm(np.std(all_actions[:, :6], axis=0)))),
            "gripper_variance": _round(np.var(all_actions[:, 6])),
            "mean_action_is_strong_on_previous_split": bool(
                _mean_baselines(previous_split["train_records"], previous_split["eval_records"])["global_mean_action"]["action_l2"]
                < 0.6
            ),
        },
    }


def _normalizer_stats(checkpoint: Path) -> dict[str, Any]:
    try:
        from safetensors.torch import load_file

        pre = load_file(str(checkpoint / "policy_preprocessor_step_5_normalizer_processor.safetensors"))
        post = load_file(str(checkpoint / "policy_postprocessor_step_0_unnormalizer_processor.safetensors"))
        return {
            "preprocessor_keys": sorted(pre.keys()),
            "postprocessor_keys": sorted(post.keys()),
            "action_means": {key: [_round(x) for x in value.detach().cpu().numpy()] for key, value in pre.items() if key.endswith(".action.mean")},
            "action_stds": {key: [_round(x) for x in value.detach().cpu().numpy()] for key, value in pre.items() if key.endswith(".action.std")},
        }
    except Exception as exc:  # noqa: BLE001
        return {"error": _compact_error(exc)}


def _interface_audit(hdf5_path: Path, checkpoint: Path, raw_stats: dict[str, Any]) -> dict[str, Any]:
    import h5py

    config = _read_json(checkpoint / "config.json")
    pre_json = _read_json(checkpoint / "policy_preprocessor.json")
    post_json = _read_json(checkpoint / "policy_postprocessor.json")
    norm = _normalizer_stats(checkpoint)
    local_mean = np.asarray(raw_stats["mean"][:6], dtype=np.float32)
    local_std = np.asarray(raw_stats["std"][:6], dtype=np.float32)
    ckpt_action_means = norm.get("action_means") or {}
    ckpt_action_stds = norm.get("action_stds") or {}
    z_deltas: dict[str, list[float]] = {}
    for mean_key, mean_values in ckpt_action_means.items():
        std_key = mean_key.replace(".mean", ".std")
        if std_key in ckpt_action_stds:
            mean = np.asarray(mean_values, dtype=np.float32)
            std = np.asarray(ckpt_action_stds[std_key], dtype=np.float32)
            z_deltas[mean_key] = [_round(x) for x in np.abs(local_mean - mean) / np.maximum(std, 1e-6)]
    action_dim_mismatch = (config.get("output_features") or {}).get("action", {}).get("shape") != [7]
    normalization_mismatch = bool(z_deltas and max(max(values) for values in z_deltas.values()) > 2.0)
    gripper_synthesized = action_dim_mismatch and raw_stats["action_dim"] == 7
    with h5py.File(hdf5_path, "r") as handle:
        demo_name = sorted(handle["data"].keys(), key=base._demo_sort_key)[0]
        actions = np.asarray(handle["data"][demo_name]["actions"], dtype=np.float32)
        timestep = min(3, actions.shape[0] - 2)
    chunk = base._action_chunk(
        hdf5_path,
        demo_name,
        timestep,
        int(config.get("chunk_size") or 50),
        int(((config.get("output_features") or {}).get("action") or {}).get("shape", [6])[0]),
    )
    label_reconstruction = {
        "demo_name": demo_name,
        "timestep": int(timestep),
        "chunk_first_matches_action_t_first6": bool(np.allclose(chunk[0, :6], actions[timestep, :6])),
        "chunk_second_matches_action_t_plus_1_first6": bool(np.allclose(chunk[1, :6], actions[timestep + 1, :6])),
        "chunk_shape": list(chunk.shape),
        "expert_action_t_shape": list(actions[timestep].shape),
    }
    return {
        "hdf5_action_dim": raw_stats["action_dim"],
        "model_action_shape": (config.get("output_features") or {}).get("action", {}).get("shape"),
        "model_state_shape": (config.get("input_features") or {}).get("observation.state", {}).get("shape"),
        "chunk_size": config.get("chunk_size"),
        "normalization_mapping": config.get("normalization_mapping"),
        "policy_preprocessor_action_shape": (((pre_json.get("steps") or [])[-1].get("config") or {}).get("features") or {}).get("action", {}).get("shape"),
        "policy_postprocessor_action_shape": (((post_json.get("steps") or [])[0].get("config") or {}).get("features") or {}).get("action", {}).get("shape"),
        "checkpoint_action_normalizer": norm,
        "local_action_first6_mean": [_round(x) for x in local_mean],
        "local_action_first6_std": [_round(x) for x in local_std],
        "local_action_min": raw_stats.get("min"),
        "local_action_max": raw_stats.get("max"),
        "translation_scale": {
            "local_translation_std": raw_stats.get("std", [])[:3],
            "local_translation_min": raw_stats.get("min", [])[:3],
            "local_translation_max": raw_stats.get("max", [])[:3],
        },
        "rotation_scale": {
            "local_rotation_std": raw_stats.get("std", [])[3:6],
            "local_rotation_min": raw_stats.get("min", [])[3:6],
            "local_rotation_max": raw_stats.get("max", [])[3:6],
        },
        "local_vs_checkpoint_action_mean_abs_z": z_deltas,
        "label_reconstruction_sanity": label_reconstruction,
        "action_chunk_horizon_alignment": {
            "chunk_size": config.get("chunk_size"),
            "chunk_starts_at_observation_timestep": label_reconstruction["chunk_first_matches_action_t_first6"],
            "chunk_second_step_is_next_hdf5_action": label_reconstruction["chunk_second_matches_action_t_plus_1_first6"],
            "off_by_one_detected_in_chunk_builder": not (
                label_reconstruction["chunk_first_matches_action_t_first6"]
                and label_reconstruction["chunk_second_matches_action_t_plus_1_first6"]
            ),
        },
        "gripper_convention": {
            "hdf5_gripper_unique_values": raw_stats.get("gripper_unique_values"),
            "current_adapter": "ACTION_STRATEGY_GRIPPER_CLOSE fills the 7th action dimension outside the 6D model head",
        },
        "interface_bug_indicators": {
            "action_dimension_mismatch_6d_model_7d_hdf5": bool(action_dim_mismatch),
            "checkpoint_action_normalization_mismatch": normalization_mismatch,
            "gripper_dimension_synthesized_not_learned": bool(gripper_synthesized),
        },
        "audit_result": "ACTION_INTERFACE_BUG" if action_dim_mismatch or normalization_mismatch else "NO_INTERFACE_BUG_FOUND",
    }


def _run_lora_variant(
    *,
    name: str,
    target_modules: list[str],
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    smolvla_ckpt: Path,
    hf_home: Path,
    checkpoint_root: Path,
    device: str,
    steps: int,
    rank: int,
    learning_rate: float,
) -> dict[str, Any]:
    import torch

    torch.cuda.reset_peak_memory_stats()
    policy, config, tokenizer_root, external_dependency = base._load_policy(
        smolvla_ckpt=smolvla_ckpt,
        hf_home=hf_home,
        checkpoint_root=checkpoint_root,
        device=device,
        lora_rank=rank,
        target_modules=target_modules,
    )
    trainable = base._trainable_params(policy)
    training = base._train_lora(
        policy=policy,
        config=config,
        tokenizer_root=tokenizer_root,
        train_records=train_records,
        device=device,
        max_steps=steps,
        learning_rate=learning_rate,
    )
    train_metrics = base._evaluate_policy(
        policy=policy,
        config=config,
        tokenizer_root=tokenizer_root,
        records=train_records,
        device=device,
    )
    eval_metrics = base._evaluate_policy(
        policy=policy,
        config=config,
        tokenizer_root=tokenizer_root,
        records=eval_records,
        device=device,
    )
    vram = _round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
    del policy
    gc.collect()
    torch.cuda.empty_cache()
    return {
        "name": name,
        "target_modules": target_modules,
        "external_dependency_found": bool(external_dependency.get("found")),
        "lora_rank": int(rank),
        **trainable,
        "training": training,
        "train_metrics": train_metrics,
        "eval_metrics": eval_metrics,
        "vram_peak_mb": vram,
    }


def _capacity_audit(
    *,
    hdf5_path: Path,
    smolvla_ckpt: Path,
    hf_home: Path,
    checkpoint_root: Path,
    device: str,
    steps: int,
    rank: int,
    learning_rate: float,
) -> dict[str, Any]:
    split = base.select_records(hdf5_path, max_train_demos=3, max_eval_demos=2, records_per_demo=3)
    mean_action = base._mean_train_action(split["train_records"])
    mean_metric = base.evaluate_constant_action(split["eval_records"], mean_action)
    ridge = _ridge_predict(split["train_records"], split["eval_records"])
    mlp = _mlp_predict(split["train_records"], split["eval_records"])
    lora_variants = {}
    started = time.monotonic()
    for name, target_modules in TARGET_MODULE_VARIANTS.items():
        lora_variants[name] = _run_lora_variant(
            name=name,
            target_modules=target_modules,
            train_records=split["train_records"],
            eval_records=split["eval_records"],
            smolvla_ckpt=smolvla_ckpt,
            hf_home=hf_home,
            checkpoint_root=checkpoint_root,
            device=device,
            steps=steps,
            rank=rank,
            learning_rate=learning_rate,
        )
    best_lora_name, best_lora = min(
        lora_variants.items(),
        key=lambda item: item[1]["eval_metrics"]["action_l2"],
    )
    best_small_name, best_small = min(
        {"state_time_ridge": ridge, "state_time_mlp": mlp}.items(),
        key=lambda item: item[1]["eval"]["action_l2"],
    )
    return {
        "split": {
            "train_count": split["train_count"],
            "eval_count": split["eval_count"],
            "train_demos": split["train_demos"],
            "eval_demos": split["eval_demos"],
        },
        "mean_action": mean_metric,
        "state_time_ridge": ridge,
        "state_time_mlp": mlp,
        "lora_variants": lora_variants,
        "best_lora": {"name": best_lora_name, "eval_metrics": best_lora["eval_metrics"]},
        "best_small_mlp_or_ridge": {"name": best_small_name, "eval_metrics": best_small["eval"]},
        "lora_beats_mean_action": bool(best_lora["eval_metrics"]["action_l2"] < mean_metric["action_l2"]),
        "lora_beats_small_mlp_or_ridge": bool(best_lora["eval_metrics"]["action_l2"] < best_small["eval"]["action_l2"]),
        "runtime_sec": _round(time.monotonic() - started, 3),
    }


def _overfit_audit(
    *,
    hdf5_path: Path,
    smolvla_ckpt: Path,
    hf_home: Path,
    checkpoint_root: Path,
    device: str,
    steps: int,
    rank: int,
    learning_rate: float,
) -> dict[str, Any]:
    split = base.select_records(hdf5_path, max_train_demos=1, max_eval_demos=1, records_per_demo=3)
    one_sample = [split["train_records"][1 if len(split["train_records"]) > 1 else 0]]
    one_demo = split["train_records"]

    one_sample_result = _run_lora_variant(
        name="one_sample_overfit_current_lora",
        target_modules=TARGET_MODULE_VARIANTS["current_projection_lora"],
        train_records=one_sample,
        eval_records=one_sample,
        smolvla_ckpt=smolvla_ckpt,
        hf_home=hf_home,
        checkpoint_root=checkpoint_root,
        device=device,
        steps=steps,
        rank=rank,
        learning_rate=learning_rate,
    )
    one_demo_result = _run_lora_variant(
        name="one_demo_overfit_current_lora",
        target_modules=TARGET_MODULE_VARIANTS["current_projection_lora"],
        train_records=one_demo,
        eval_records=one_demo,
        smolvla_ckpt=smolvla_ckpt,
        hf_home=hf_home,
        checkpoint_root=checkpoint_root,
        device=device,
        steps=steps,
        rank=rank,
        learning_rate=learning_rate,
    )
    sample_pass = bool(
        one_sample_result["training"]["loss_decreased_meaningfully"]
        and one_sample_result["eval_metrics"]["action_l2"] < 0.25
    )
    demo_mean = base.evaluate_constant_action(one_demo, base._mean_train_action(one_demo))
    demo_pass = bool(
        one_demo_result["training"]["loss_decreased_meaningfully"]
        and one_demo_result["eval_metrics"]["action_l2"] < demo_mean["action_l2"]
    )
    return {
        "one_sample_overfit": one_sample_result,
        "one_sample_overfit_passed": sample_pass,
        "one_sample_pass_rule": "loss decreases and same-record select_action action L2 < 0.25",
        "one_demo_overfit": one_demo_result,
        "one_demo_mean_action": demo_mean,
        "one_demo_overfit_passed": demo_pass,
        "one_demo_pass_rule": "loss decreases and same-demo select_action action L2 beats same-demo mean action",
    }


def _write_report_bundle(report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    data = report.get("data_split_audit") or {}
    interface = report.get("action_interface_audit") or {}
    overfit = report.get("overfit_audit") or {}
    capacity = report.get("capacity_audit") or {}

    Path("reports/smolvla_lora_baseline_diagnosis.md").write_text(
        "\n".join(
            [
                "# SmolVLA LoRA Baseline Diagnosis",
                "",
                f"Final decision: `{summary.get('final_decision')}`",
                "",
                "This is a baseline diagnosis, not a new method or paper claim.",
                "",
                "## Key Findings",
                "",
                f"- raw HDF5 timesteps: `{summary.get('raw_timestep_count')}`",
                f"- previous split records: `{summary.get('previous_train_eval_count')}`",
                f"- larger split possible: `{summary.get('larger_split_train_eval_count')}`",
                f"- interface audit result: `{summary.get('interface_audit_result')}`",
                f"- one-sample overfit passed: `{summary.get('one_sample_overfit_passed')}`",
                f"- one-demo overfit passed: `{summary.get('one_demo_overfit_passed')}`",
                f"- mean-action metric: `{summary.get('mean_action_metric')}`",
                f"- frozen/base metric: `{summary.get('frozen_base_metric')}`",
                f"- best LoRA metric: `{summary.get('best_lora_metric')}`",
                f"- best small MLP/ridge metric: `{summary.get('best_small_mlp_or_ridge_metric')}`",
                f"- LoRA beats mean-action: `{summary.get('lora_beats_mean_action')}`",
                f"- exact next step: {summary.get('exact_next_step')}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    Path("reports/smolvla_split_and_variance_audit.md").write_text(
        "\n".join(
            [
                "# SmolVLA Split And Variance Audit",
                "",
                f"- previous 9/6 explanation: {data.get('previous_split_why_9_train_6_eval')}",
                f"- records are: {data.get('records_are')}",
                f"- raw demo count: `{data.get('raw_demo_count')}`",
                f"- raw timestep count: `{data.get('raw_timestep_count')}`",
                f"- global action variance: `{((data.get('raw_action_stats') or {}).get('variance'))}`",
                f"- translation variance mean: `{((data.get('raw_action_stats') or {}).get('translation_variance_mean'))}`",
                f"- rotation variance mean: `{((data.get('raw_action_stats') or {}).get('rotation_variance_mean'))}`",
                f"- gripper variance: `{((data.get('raw_action_stats') or {}).get('gripper_variance'))}`",
                f"- previous split mean-action L2: `{(((data.get('previous_sampled_split') or {}).get('mean_baselines') or {}).get('global_mean_action') or {}).get('action_l2')}`",
                f"- previous split previous-action L2: `{(((data.get('previous_sampled_split') or {}).get('mean_baselines') or {}).get('previous_action') or {}).get('action_l2')}`",
                f"- larger demo split train/eval: `{((data.get('larger_demo_holdout_split') or {}).get('train_count'))} / {((data.get('larger_demo_holdout_split') or {}).get('eval_count'))}`",
                f"- same-demo time split train/eval: `{((data.get('same_demo_time_holdout_split') or {}).get('train_count'))} / {((data.get('same_demo_time_holdout_split') or {}).get('eval_count'))}`",
                f"- task holdout feasible without download: `{((data.get('task_holdout_audit') or {}).get('feasible_without_download'))}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    Path("reports/smolvla_action_interface_audit.md").write_text(
        "\n".join(
            [
                "# SmolVLA Action Interface Audit",
                "",
                f"- audit result: `{interface.get('audit_result')}`",
                f"- HDF5 action dim: `{interface.get('hdf5_action_dim')}`",
                f"- model action shape: `{interface.get('model_action_shape')}`",
                f"- policy preprocessor action shape: `{interface.get('policy_preprocessor_action_shape')}`",
                f"- policy postprocessor action shape: `{interface.get('policy_postprocessor_action_shape')}`",
                f"- normalization mapping: `{interface.get('normalization_mapping')}`",
                f"- local action first6 mean: `{interface.get('local_action_first6_mean')}`",
                f"- local action first6 std: `{interface.get('local_action_first6_std')}`",
                f"- local action min: `{interface.get('local_action_min')}`",
                f"- local action max: `{interface.get('local_action_max')}`",
                f"- translation scale: `{interface.get('translation_scale')}`",
                f"- rotation scale: `{interface.get('rotation_scale')}`",
                f"- local/checkpoint action mean abs z: `{interface.get('local_vs_checkpoint_action_mean_abs_z')}`",
                f"- label reconstruction sanity: `{interface.get('label_reconstruction_sanity')}`",
                f"- action chunk horizon alignment: `{interface.get('action_chunk_horizon_alignment')}`",
                f"- bug indicators: `{interface.get('interface_bug_indicators')}`",
                f"- gripper convention: `{interface.get('gripper_convention')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    Path("reports/smolvla_lora_capacity_audit.md").write_text(
        "\n".join(
            [
                "# SmolVLA LoRA Capacity Audit",
                "",
                f"- split train/eval: `{((capacity.get('split') or {}).get('train_count'))} / {((capacity.get('split') or {}).get('eval_count'))}`",
                f"- mean-action eval action L2: `{((capacity.get('mean_action') or {}).get('action_l2'))}`",
                f"- best LoRA: `{((capacity.get('best_lora') or {}).get('name'))}`",
                f"- best LoRA eval action L2: `{(((capacity.get('best_lora') or {}).get('eval_metrics') or {}).get('action_l2'))}`",
                f"- best LoRA eval per-dim MAE: `{(((capacity.get('best_lora') or {}).get('eval_metrics') or {}).get('per_dim_mae'))}`",
                f"- current LoRA train action L2: `{(((capacity.get('lora_variants') or {}).get('current_projection_lora') or {}).get('train_metrics') or {}).get('action_l2')}`",
                f"- current LoRA eval action L2: `{(((capacity.get('lora_variants') or {}).get('current_projection_lora') or {}).get('eval_metrics') or {}).get('action_l2')}`",
                f"- best small MLP/ridge: `{((capacity.get('best_small_mlp_or_ridge') or {}).get('name'))}`",
                f"- best small MLP/ridge eval action L2: `{(((capacity.get('best_small_mlp_or_ridge') or {}).get('eval_metrics') or {}).get('action_l2'))}`",
                f"- best small MLP/ridge eval per-dim MAE: `{(((capacity.get('best_small_mlp_or_ridge') or {}).get('eval_metrics') or {}).get('per_dim_mae'))}`",
                f"- LoRA beats mean-action: `{capacity.get('lora_beats_mean_action')}`",
                f"- LoRA beats small MLP/ridge: `{capacity.get('lora_beats_small_mlp_or_ridge')}`",
                f"- one-sample overfit passed: `{overfit.get('one_sample_overfit_passed')}`",
                f"- one-demo overfit passed: `{overfit.get('one_demo_overfit_passed')}`",
                f"- runtime sec: `{capacity.get('runtime_sec')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    Path("reports/smolvla_lora_next_decision.md").write_text(
        "\n".join(
            [
                "# SmolVLA LoRA Next Decision",
                "",
                f"Final decision: `{summary.get('final_decision')}`",
                "",
                f"Exact next step: {summary.get('exact_next_step')}",
                "",
                "Do not propose a new paper method unless the decision is `READY_FOR_REAL_METHOD_AFTER_BASELINE`.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    hdf5_path = Path(args.hdf5_path)
    smolvla_ckpt = Path(args.smolvla_ckpt)
    checkpoint_root = Path(args.checkpoint_root)
    hf_home = Path(args.hf_home)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]

    report: dict[str, Any] = {
        "schema_version": "smolvla-lora-baseline-diagnosis-v1",
        "evidence_label": "smolvla_lora_baseline_diagnosis",
        "decision": "ACTION_INTERFACE_BUG",
        "policy": {
            "bounded_baseline_diagnosis": True,
            "new_method_created": False,
            "patchguard_continued": False,
            "downloads_performed": False,
            "large_model_or_dataset_downloads_performed": False,
            "rollouts_performed": False,
            "benchmark_rollouts_performed": False,
            "openvla_oft_executed": False,
            "paper_grade_claims_made": False,
            "heavy_import_gate_set": _env_flag(HEAVY_IMPORT_GATE),
            "diagnosis_gate_set": _env_flag(DIAGNOSIS_GATE),
            "training_gate_set": _env_flag(TRAINING_GATE),
            "forbidden_gates_set": forbidden,
            "training_performed": False,
            "loss_computed": False,
        },
        "paths": {
            "hdf5_path": str(hdf5_path),
            "smolvla_ckpt": str(smolvla_ckpt),
            "checkpoint_root": str(checkpoint_root),
            "hf_home": str(hf_home),
        },
        "data_split_audit": {},
        "action_interface_audit": {},
        "overfit_audit": {},
        "capacity_audit": {},
        "summary": {},
        "error": None,
    }

    def finish(decision: str, next_step: str, code: int) -> tuple[dict[str, Any], int]:
        if decision not in FINAL_DECISIONS:
            raise ValueError(f"invalid final decision: {decision}")
        report["decision"] = decision
        report["summary"]["final_decision"] = decision
        report["summary"]["exact_next_step"] = next_step
        report["summary"]["runtime_sec"] = _round(time.monotonic() - started, 3)
        try:
            import torch

            report["summary"]["vram_peak_mb"] = (
                _round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3) if torch.cuda.is_available() else 0.0
            )
        except Exception:
            report["summary"]["vram_peak_mb"] = None
        return report, code

    if not report["policy"]["heavy_import_gate_set"]:
        return finish("ACTION_INTERFACE_BUG", f"Set {HEAVY_IMPORT_GATE}=1 for this bounded diagnosis.", 2)
    if not report["policy"]["diagnosis_gate_set"]:
        return finish("ACTION_INTERFACE_BUG", f"Set {DIAGNOSIS_GATE}=1 for this bounded diagnosis.", 3)
    if forbidden:
        report["error"] = {"message": "Forbidden gate(s) set: " + ", ".join(forbidden)}
        return finish("ACTION_INTERFACE_BUG", "Clear forbidden rollout/download/OpenVLA-OFT/method gates and rerun.", 4)
    if not hdf5_path.exists():
        return finish("ACTION_INTERFACE_BUG", f"Missing local HDF5 path: {hdf5_path}", 5)

    try:
        report["data_split_audit"] = _data_and_split_audit(hdf5_path)
        report["action_interface_audit"] = _interface_audit(
            hdf5_path,
            smolvla_ckpt,
            report["data_split_audit"]["raw_action_stats"],
        )
        if not report["policy"]["training_gate_set"]:
            report["summary"].update(
                {
                    "raw_timestep_count": report["data_split_audit"].get("raw_timestep_count"),
                    "interface_audit_result": report["action_interface_audit"].get("audit_result"),
                }
            )
            return finish("ACTION_INTERFACE_BUG", f"Set {TRAINING_GATE}=1 to run bounded overfit/capacity checks.", 6)

        import torch

        torch.cuda.reset_peak_memory_stats()
        report["overfit_audit"] = _overfit_audit(
            hdf5_path=hdf5_path,
            smolvla_ckpt=smolvla_ckpt,
            hf_home=hf_home,
            checkpoint_root=checkpoint_root,
            device=args.device,
            steps=int(args.overfit_steps),
            rank=int(args.lora_rank),
            learning_rate=float(args.learning_rate),
        )
        report["capacity_audit"] = _capacity_audit(
            hdf5_path=hdf5_path,
            smolvla_ckpt=smolvla_ckpt,
            hf_home=hf_home,
            checkpoint_root=checkpoint_root,
            device=args.device,
            steps=int(args.capacity_steps),
            rank=int(args.lora_rank),
            learning_rate=float(args.learning_rate),
        )
        report["policy"]["training_performed"] = True
        report["policy"]["loss_computed"] = True

        previous = report["data_split_audit"]["previous_sampled_split"]
        larger = report["data_split_audit"]["larger_demo_holdout_split"]
        capacity = report["capacity_audit"]
        overfit = report["overfit_audit"]
        interface = report["action_interface_audit"]
        best_lora = capacity["best_lora"]["eval_metrics"]
        best_small = capacity["best_small_mlp_or_ridge"]["eval_metrics"]
        frozen_metric = _read_json(Path("reports/smolvla_lora_baseline_state1_result.json"))["summary"]["frozen_base_metric"]
        interface_bug = interface.get("audit_result") == "ACTION_INTERFACE_BUG"
        overfit_failed = not overfit["one_sample_overfit_passed"] or not overfit["one_demo_overfit_passed"]
        low_variance = bool(report["data_split_audit"]["low_variance_indicators"]["mean_action_is_strong_on_previous_split"])

        report["summary"].update(
            {
                "raw_timestep_count": report["data_split_audit"]["raw_timestep_count"],
                "previous_train_eval_count": f"{previous['train_count']} / {previous['eval_count']}",
                "larger_split_train_eval_count": f"{larger['train_count']} / {larger['eval_count']}",
                "interface_audit_result": interface.get("audit_result"),
                "one_sample_overfit_passed": overfit["one_sample_overfit_passed"],
                "one_demo_overfit_passed": overfit["one_demo_overfit_passed"],
                "mean_action_metric": capacity["mean_action"]["action_l2"],
                "frozen_base_metric": frozen_metric,
                "best_lora_metric": best_lora["action_l2"],
                "best_lora_name": capacity["best_lora"]["name"],
                "best_small_mlp_or_ridge_metric": best_small["action_l2"],
                "best_small_mlp_or_ridge_name": capacity["best_small_mlp_or_ridge"]["name"],
                "lora_beats_mean_action": capacity["lora_beats_mean_action"],
                "lora_beats_small_mlp_or_ridge": capacity["lora_beats_small_mlp_or_ridge"],
                "action_interface_bug": bool(interface_bug),
                "overfit_failed": bool(overfit_failed),
                "low_variance_indicator": bool(low_variance),
                "training_happened": True,
                "loss_computed": True,
                "model_used": str(smolvla_ckpt),
                "dataset_used": str(hdf5_path),
                "lora_rank": int(args.lora_rank),
            }
        )

        if interface_bug or overfit_failed:
            return finish(
                "ACTION_INTERFACE_BUG",
                "Fix the SmolVLA/LIBERO action interface before any method work: the local data is 7D LIBERO action space while the checkpoint action head and normalizer are 6D SO100-style, and overfit sanity did not clear the action metric gate.",
                0,
            )
        if low_variance:
            return finish(
                "DATA_TOO_SMALL_OR_LOW_VARIANCE",
                "Build a larger or more diverse standard split before rerunning standard LoRA.",
                0,
            )
        if not capacity["lora_beats_small_mlp_or_ridge"]:
            return finish(
                "LORA_CAPACITY_OR_TARGET_MODULE_BLOCKED",
                "LoRA target modules did not beat the small ridge/MLP baseline; audit action-head targets before method work.",
                0,
            )
        if not capacity["lora_beats_mean_action"]:
            return finish(
                "KILL_SMOLVLA_LORA_BASELINE",
                "Interface and data passed, but LoRA still did not beat mean-action.",
                0,
            )
        return finish(
            "READY_FOR_REAL_METHOD_AFTER_BASELINE",
            "Only now plan a future method, with standard LoRA, mean-action, ridge/MLP, and frozen/base baselines predeclared.",
            0,
        )
    except Exception as exc:  # noqa: BLE001
        report["error"] = _compact_error(exc)
        return finish("ACTION_INTERFACE_BUG", "Diagnosis failed before proving the interface correct; fix runner/environment and rerun.", 10)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-path", default=DEFAULT_HDF5_PATH)
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--report-path", default="reports/smolvla_lora_baseline_diagnosis.json")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--overfit-steps", type=int, default=80)
    parser.add_argument("--capacity-steps", type=int, default=50)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_report_bundle(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
