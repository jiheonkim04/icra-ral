"""Bounded SmolVLA/LIBERO 7D standard replay baseline reproduction.

This runner checks whether the fixed LIBERO_7D SmolVLA adapter baseline
transfers from offline action L2 to exact-init replay progress on a small
standard LIBERO split. It is a baseline reproduction, not a method.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tca_map.datasets.libero_full_demo_expert_replay_sanity import _run_replay_variant
from tca_map.smolvla_lora_baseline import diagnostic as base
from tca_map.smolvla_lora_baseline import libero_7d_baseline_reproduction as repro
from tca_map.smolvla_lora_baseline import libero_7d_interface_fix as fix
from tca_map.smolvla_lora_baseline import replay_bridge


RUN_GATE = "ALLOW_SMOLVLA_7D_STANDARD_REPLAY_BASELINE"
TRAINING_GATE = "ALLOW_SMOLVLA_7D_STANDARD_REPLAY_BASELINE_TRAINING"
REPLAY_GATE = "ALLOW_SMOLVLA_7D_STANDARD_REPLAY_BASELINE_REPLAY"
SCHEMA_VERSION = "smolvla-7d-standard-replay-baseline-v1"
FINAL_DECISIONS = {
    "READY_FOR_RA_L_METHOD_AFTER_STANDARD_BASELINE",
    "READY_BUT_NEEDS_ACTION_RANGE_FIX",
    "OFFLINE_TO_CONTROL_GAP",
    "MEAN_OR_MLP_REPLAY_DOMINATED",
    "EXPERT_REPLAY_UNSTABLE",
    "TOO_HEAVY_LOCAL",
}
FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_PATCHGUARD_VLA_STATE1B",
    "ALLOW_PATCHGUARD_TINY_LORA_TRAINING",
    "ALLOW_TARGET_GROUNDED_ACTIONMAP",
    "ALLOW_SAFELORA",
    "ALLOW_PRISM",
    "ALLOW_TG7D_ADAPTER",
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _round(value: float | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)


def _compact_error(exc: BaseException | str) -> dict[str, Any]:
    if isinstance(exc, str):
        return {"type": "Error", "message": exc, "traceback_tail": []}
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _first_index(values: np.ndarray, threshold: float = 0.0) -> int | None:
    for index, value in enumerate(np.asarray(values).reshape(-1)):
        if float(value) > threshold:
            return int(index)
    return None


def _demo_sort_key(name: str) -> tuple[str, int | str]:
    return base._demo_sort_key(name)


def _task_id_from_demo_path(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_demo")] if stem.endswith("_demo") else stem


def _instruction_from_path(path: Path) -> str:
    return _task_id_from_demo_path(path).replace("_", " ")


def _load_demo_metadata(path: Path) -> dict[str, dict[str, Any]]:
    import h5py

    result: dict[str, dict[str, Any]] = {}
    with h5py.File(path, "r") as handle:
        for demo_name in sorted(handle["data"].keys(), key=_demo_sort_key):
            demo = handle["data"][demo_name]
            actions = np.asarray(demo["actions"], dtype=np.float32)
            rewards = np.asarray(demo["rewards"], dtype=np.float32).reshape(-1) if "rewards" in demo else np.zeros((actions.shape[0],), dtype=np.float32)
            dones = np.asarray(demo["dones"], dtype=np.float32).reshape(-1) if "dones" in demo else np.zeros((actions.shape[0],), dtype=np.float32)
            first_reward = _first_index(rewards, 0.0)
            first_done = _first_index(dones, 0.5)
            signals = [value for value in [first_reward, first_done] if value is not None]
            result[demo_name] = {
                "length": int(actions.shape[0]),
                "first_reward_index": first_reward,
                "first_done_index": first_done,
                "first_signal_index": min(signals) if signals else None,
            }
    return result


def _selected_eval_demos(metadata: dict[str, dict[str, Any]], train_demos: list[str], *, count: int, max_replay_steps: int) -> list[str]:
    train_set = set(train_demos)
    candidates = [name for name in sorted(metadata, key=_demo_sort_key) if name not in train_set]
    bounded = [
        name
        for name in candidates
        if metadata[name].get("first_signal_index") is not None
        and int(metadata[name]["first_signal_index"]) < int(max_replay_steps)
    ]
    selected = bounded[:count]
    if len(selected) < count:
        selected.extend([name for name in candidates if name not in selected][: count - len(selected)])
    return selected


def _records_for_demos(path: Path, demos: list[str], metadata: dict[str, dict[str, Any]], records_per_demo: int) -> list[dict[str, Any]]:
    demo_times = {
        demo_name: repro._sample_timesteps(int(metadata[demo_name]["length"]), int(records_per_demo))
        for demo_name in demos
    }
    return repro._records_for_demo_times(path, demo_times)


def _gripper_distribution(actions: np.ndarray) -> dict[str, Any]:
    grip = np.asarray(actions[:, 6], dtype=np.float32).reshape(-1)
    return {
        "negative": int(np.sum(grip < 0.0)),
        "zero": int(np.sum(grip == 0.0)),
        "positive": int(np.sum(grip > 0.0)),
        "min": _round(float(grip.min())),
        "max": _round(float(grip.max())),
        "mean": _round(float(grip.mean())),
    }


def _action_validity(actions: np.ndarray, *, low: float = -1.0, high: float = 1.0) -> dict[str, Any]:
    arr = np.asarray(actions, dtype=np.float32)
    base_validity = replay_bridge._action_validity(arr, low=low, high=high)
    if arr.ndim == 2 and arr.shape[1] == 7 and arr.size:
        clipped = (arr < low) | (arr > high)
        base_validity["per_dim_clip_rate"] = [_round(x) for x in np.mean(clipped, axis=0)]
        base_validity["dominant_clip_dim"] = int(np.argmax(np.mean(clipped, axis=0)))
        base_validity["gripper_clip_rate"] = _round(float(np.mean(clipped[:, 6])))
    else:
        base_validity["per_dim_clip_rate"] = None
        base_validity["dominant_clip_dim"] = None
        base_validity["gripper_clip_rate"] = None
    return base_validity


def _metrics(pred: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    return fix._metrics_from_arrays(np.asarray(pred, dtype=np.float32), np.asarray(expert, dtype=np.float32))


def _build_standard_split(
    *,
    data_root: Path,
    max_tasks: int,
    train_demos_per_task: int,
    eval_demos_per_task: int,
    records_per_demo: int,
    replay_demos_per_task: int,
    max_replay_steps: int,
) -> dict[str, Any]:
    task_paths = sorted(data_root.glob("*.hdf5"))[: int(max_tasks)]
    if not task_paths:
        raise FileNotFoundError(f"no HDF5 tasks found under {data_root}")
    train_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    replay_cases: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    raw_timesteps = 0
    for path in task_paths:
        metadata = _load_demo_metadata(path)
        demo_names = sorted(metadata, key=_demo_sort_key)
        train_demos = demo_names[: int(train_demos_per_task)]
        eval_demos = _selected_eval_demos(metadata, train_demos, count=int(eval_demos_per_task), max_replay_steps=max_replay_steps)
        replay_demos = eval_demos[: int(replay_demos_per_task)]
        task_train = _records_for_demos(path, train_demos, metadata, records_per_demo)
        task_eval = _records_for_demos(path, eval_demos, metadata, records_per_demo)
        train_records.extend(task_train)
        eval_records.extend(task_eval)
        raw_timesteps += int(sum(item["length"] for item in metadata.values()))
        for demo_name in replay_demos:
            replay_cases.append({"hdf5_path": str(path), "demo_name": demo_name, "task_name": path.stem})
        tasks.append(
            {
                "path": str(path),
                "task_name": path.stem,
                "demo_count": len(demo_names),
                "train_demos": train_demos,
                "eval_demos": eval_demos,
                "replay_demos": replay_demos,
                "raw_timesteps": int(sum(item["length"] for item in metadata.values())),
                "first_signal_by_replay_demo": {
                    demo: metadata[demo].get("first_signal_index") for demo in replay_demos
                },
            }
        )
    train_actions = repro._concat_record_actions(train_records)
    eval_actions = repro._concat_record_actions(eval_records)
    mean_metrics = repro._mean_action_metrics(train_records, eval_records)
    leakage = repro._split_leakage(train_records, eval_records)
    report = {
        "name": "bounded_libero10_standard_demo_holdout",
        "tasks": tasks,
        "task_names": [item["task_name"] for item in tasks],
        "task_count": len(tasks),
        "train_demo_count": sum(len(item["train_demos"]) for item in tasks),
        "eval_demo_count": sum(len(item["eval_demos"]) for item in tasks),
        "replay_case_count": len(replay_cases),
        "raw_timestep_count": raw_timesteps,
        "sampled_train_records": len(train_records),
        "sampled_eval_records": len(eval_records),
        "records_per_demo": int(records_per_demo),
        "train_action_stats": fix._action_stats(train_actions),
        "eval_action_stats": fix._action_stats(eval_actions),
        "train_gripper_distribution": _gripper_distribution(train_actions),
        "eval_gripper_distribution": _gripper_distribution(eval_actions),
        "mean_action_metric": mean_metrics,
        "task_variance_enough_to_make_mean_nontrivial": bool(mean_metrics["action_l2"] > 0.4),
        "leakage": leakage,
    }
    return {"train_records": train_records, "eval_records": eval_records, "replay_cases": replay_cases, "report": report}


@dataclass
class Predictor:
    name: str
    predict: Callable[[np.ndarray], np.ndarray]


def _fit_ridge(train_records: list[dict[str, Any]]) -> Predictor:
    x_train, y_train = fix._feature_matrix(train_records)
    x_mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = (x_train.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    xt = ((x_train - x_mean) / x_std).astype(np.float32)
    xt_aug = np.concatenate([xt, np.ones((xt.shape[0], 1), dtype=np.float32)], axis=1)
    weights = np.linalg.solve(xt_aug.T @ xt_aug + 1e-3 * np.eye(xt_aug.shape[1], dtype=np.float32), xt_aug.T @ y_train).astype(np.float32)

    def predict(features: np.ndarray) -> np.ndarray:
        x = ((np.asarray(features, dtype=np.float32) - x_mean) / x_std).astype(np.float32)
        x_aug = np.concatenate([x, np.ones((x.shape[0], 1), dtype=np.float32)], axis=1)
        return (x_aug @ weights).astype(np.float32)

    return Predictor("ridge", predict)


def _torch_train_loop(
    *,
    pred_fn: Callable[[Any], Any],
    params: list[Any],
    x_train: Any,
    y_train: Any,
    steps: int,
    learning_rate: float,
) -> list[dict[str, float]]:
    import torch

    optimizer = torch.optim.AdamW(params, lr=float(learning_rate), weight_decay=1e-5)
    losses: list[dict[str, float]] = []
    row_count = int(y_train.shape[0])
    for step in range(int(steps)):
        index = step % row_count
        optimizer.zero_grad(set_to_none=True)
        pred = pred_fn(index)
        target = y_train[index : index + 1]
        pose_loss = torch.nn.functional.mse_loss(pred[:, :6], target[:, :6])
        gripper_loss = torch.nn.functional.mse_loss(pred[:, 6:], target[:, 6:])
        loss = pose_loss + gripper_loss
        loss.backward()
        optimizer.step()
        losses.append(
            {
                "loss": _round(loss.detach().cpu()),
                "pose_loss": _round(pose_loss.detach().cpu()),
                "gripper_mse_loss": _round(gripper_loss.detach().cpu()),
            }
        )
    return losses


def _train_feature_mlp(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    *,
    steps: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
) -> tuple[dict[str, Any], Predictor]:
    import torch

    started = time.monotonic()
    x_train, y_train_np = fix._feature_matrix(train_records)
    x_eval, y_eval_np = fix._feature_matrix(eval_records)
    x_mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = (x_train.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    normalizer = fix.Libero7DNormalizer.fit(y_train_np)
    torch.manual_seed(int(seed))
    model = torch.nn.Sequential(
        torch.nn.Linear(7, int(hidden_dim)),
        torch.nn.SiLU(),
        torch.nn.Linear(int(hidden_dim), 7),
    )
    x_train_t = torch.tensor(((x_train - x_mean) / x_std).astype(np.float32), dtype=torch.float32)
    y_train_t = torch.tensor(normalizer.normalize(y_train_np), dtype=torch.float32)
    losses = _torch_train_loop(
        pred_fn=lambda index: model(x_train_t[index : index + 1]),
        params=list(model.parameters()),
        x_train=x_train_t,
        y_train=y_train_t,
        steps=steps,
        learning_rate=learning_rate,
    )

    def predict(features: np.ndarray) -> np.ndarray:
        x = torch.tensor(((np.asarray(features, dtype=np.float32) - x_mean) / x_std).astype(np.float32), dtype=torch.float32)
        with torch.no_grad():
            pred = model(x).detach().cpu().numpy()
        return normalizer.unnormalize(pred)

    train_pred = predict(x_train)
    eval_pred = predict(x_eval)
    report = {
        "name": "small_mlp",
        "feature_schema": "SmolVLA observation.state 6D plus timestep fraction",
        "trainable_params": int(sum(p.numel() for p in model.parameters())),
        "batch_size": 1,
        "training": _training_report(losses, steps, hidden_dim, learning_rate),
        "train_metrics": _metrics(train_pred, y_train_np),
        "eval_metrics": _metrics(eval_pred, y_eval_np),
        "train_eval_gap": _round(_metrics(eval_pred, y_eval_np)["action_l2"] - _metrics(train_pred, y_train_np)["action_l2"]),
        "action_validity": _action_validity(eval_pred),
        "runtime_sec": _round(time.monotonic() - started, 3),
        "uses_eval_labels_for_training": False,
    }
    return report, Predictor("small_mlp", predict)


def _state_time_tensors_from_features(features: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray):
    import torch

    x_norm = ((np.asarray(features, dtype=np.float32) - x_mean) / x_std).astype(np.float32)
    state = np.concatenate([x_norm[:, :6], np.zeros((x_norm.shape[0], 26), dtype=np.float32)], axis=1)
    time_feature = x_norm[:, 6:7]
    return torch.tensor(state, dtype=torch.float32), torch.tensor(time_feature, dtype=torch.float32)


def _training_report(losses: list[dict[str, float]], steps: int, hidden_dim: int | None, learning_rate: float) -> dict[str, Any]:
    return {
        "steps": int(steps),
        "batch_size": 1,
        "hidden_dim": hidden_dim,
        "learning_rate": float(learning_rate),
        "loss_start": losses[0]["loss"] if losses else None,
        "loss_end": losses[-1]["loss"] if losses else None,
        "pose_loss_end": losses[-1]["pose_loss"] if losses else None,
        "gripper_mse_loss_end": losses[-1]["gripper_mse_loss"] if losses else None,
        "loss_decreased": bool(losses and losses[-1]["loss"] < losses[0]["loss"]),
        "loss_curve_sample": {"first5": losses[:5], "last5": losses[-5:]},
        "loss_curve": losses,
    }


def _train_state_proj_adapter(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    *,
    checkpoint: Path,
    name: str,
    steps: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
    lora_rank: int | None,
    linear_head: bool = False,
    artifact_path: Path | None = None,
) -> tuple[dict[str, Any], Predictor]:
    import torch

    started = time.monotonic()
    state_weight, state_bias, weight_file = repro._state_proj_weights(checkpoint)
    train_state, train_time, y_train_np, x_mean, x_std = repro._state_time_tensors(train_records)
    eval_state, eval_time, y_eval_np, _, _ = repro._state_time_tensors(eval_records, x_mean, x_std)
    normalizer = fix.Libero7DNormalizer.fit(y_train_np)
    y_train = torch.tensor(normalizer.normalize(y_train_np), dtype=torch.float32)
    torch.manual_seed(int(seed))
    if linear_head:
        head = torch.nn.Linear(961, 7)
        head_params = 961 * 7 + 7
        hidden = None
    else:
        head = torch.nn.Sequential(
            torch.nn.Linear(961, int(hidden_dim)),
            torch.nn.SiLU(),
            torch.nn.Linear(int(hidden_dim), 7),
        )
        head_params = repro._mlp_param_count(961, int(hidden_dim), 7)
        hidden = int(hidden_dim)
    params: list[Any] = list(head.parameters())
    lora_a = None
    lora_b = None
    lora_alpha = None
    if lora_rank is not None:
        lora_alpha = int(lora_rank) * 2
        lora_a = torch.nn.Parameter(torch.randn(int(lora_rank), 32) * 0.01)
        lora_b = torch.nn.Parameter(torch.zeros(960, int(lora_rank)))
        params.extend([lora_a, lora_b])

    def projected(state_tensor: Any) -> Any:
        if lora_rank is None:
            return state_tensor @ state_weight.T + state_bias
        assert lora_a is not None and lora_b is not None
        scale = float(lora_alpha) / float(lora_rank)
        delta = (lora_b @ lora_a) * scale
        return state_tensor @ (state_weight + delta).T + state_bias

    losses = _torch_train_loop(
        pred_fn=lambda index: head(torch.cat([projected(train_state[index : index + 1]), train_time[index : index + 1]], dim=1)),
        params=params,
        x_train=train_state,
        y_train=y_train,
        steps=steps,
        learning_rate=learning_rate,
    )

    def predict(features: np.ndarray) -> np.ndarray:
        state, time_feature = _state_time_tensors_from_features(features, x_mean, x_std)
        with torch.no_grad():
            pred_norm = head(torch.cat([projected(state), time_feature], dim=1)).detach().cpu().numpy()
        return normalizer.unnormalize(pred_norm)

    with torch.no_grad():
        train_pred = predict(fix._feature_matrix(train_records)[0])
        eval_features, _ = fix._feature_matrix(eval_records)
        eval_pred = predict(eval_features)
    lora_params = 0 if lora_rank is None else int(lora_rank) * (32 + 960)
    report = {
        "name": name,
        "adapter_schema": "LIBERO_7D",
        "feature_schema": "train-normalized SmolVLA observation.state padded to 32, checkpoint state_proj, timestep fraction",
        "state_proj_weight_file": str(weight_file),
        "target_modules": ["state_proj", "libero_7d_adapter"] if lora_rank is not None else ["libero_7d_adapter"],
        "excluded_native_6d_modules": ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"],
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "trainable_params": head_params + lora_params,
        "batch_size": 1,
        "normalization": normalizer.report(),
        "feature_normalization": {
            "source": "train_split_only",
            "shape": [7],
            "mean": [_round(x) for x in x_mean.reshape(-1)],
            "std": [_round(x) for x in x_std.reshape(-1)],
            "uses_eval_labels": False,
        },
        "gripper_handling": {"learned": True, "hard_coded": False},
        "training": _training_report(losses, steps, hidden, learning_rate),
        "train_metrics": _metrics(train_pred, y_train_np),
        "eval_metrics": _metrics(eval_pred, y_eval_np),
        "train_eval_gap": _round(_metrics(eval_pred, y_eval_np)["action_l2"] - _metrics(train_pred, y_train_np)["action_l2"]),
        "action_validity": _action_validity(eval_pred),
        "runtime_sec": _round(time.monotonic() - started, 3),
        "uses_eval_labels_for_training": False,
        "uses_so100_action_normalizer": False,
        "uses_hard_coded_gripper_fill": False,
    }
    if artifact_path is not None and lora_rank is not None:
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "smolvla-7d-standard-replay-adapter-artifact-v1",
                "name": name,
                "checkpoint_path": str(checkpoint),
                "hidden_dim": int(hidden_dim),
                "lora_rank": int(lora_rank),
                "lora_alpha": int(lora_alpha),
                "normalization": normalizer.report(),
                "feature_normalization": report["feature_normalization"],
                "head_state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
                "lora_a": lora_a.detach().cpu(),
                "lora_b": lora_b.detach().cpu(),
            },
            str(artifact_path),
        )
        report["artifact_path"] = str(artifact_path)
    return report, Predictor(name, predict)


def _evaluate_predictor(name: str, predictor: Predictor, eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    features, expert = fix._feature_matrix(eval_records)
    pred = predictor.predict(features)
    return {
        "name": name,
        "eval_metrics": _metrics(pred, expert),
        "action_validity": _action_validity(pred),
    }


def _run_offline_baselines(args: argparse.Namespace, split: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Predictor]]:
    import torch

    started = time.monotonic()
    checkpoint = Path(args.smolvla_ckpt)
    train_records = split["train_records"]
    eval_records = split["eval_records"]
    features_eval, y_eval = fix._feature_matrix(eval_records)
    train_actions = repro._concat_record_actions(train_records)
    mean_action = train_actions.mean(axis=0).astype(np.float32)
    mean_pred = np.repeat(mean_action.reshape(1, 7), y_eval.shape[0], axis=0).astype(np.float32)
    baselines: dict[str, Any] = {
        "expert": {"name": "expert", "eval_metrics": _metrics(y_eval, y_eval), "action_validity": _action_validity(y_eval)},
        "mean_action": {"name": "mean_action", "eval_metrics": _metrics(mean_pred, y_eval), "action_validity": _action_validity(mean_pred)},
    }
    predictors: dict[str, Predictor] = {
        "mean_action": Predictor("mean_action", lambda features: np.repeat(mean_action.reshape(1, 7), np.asarray(features).shape[0], axis=0).astype(np.float32))
    }
    ridge = _fit_ridge(train_records)
    predictors["ridge"] = ridge
    baselines["ridge"] = _evaluate_predictor("ridge", ridge, eval_records)
    baselines["ridge"]["training"] = {"loss_curve": [], "batch_size": None, "trainable_params": 56}
    mlp_report, mlp_predictor = _train_feature_mlp(
        train_records,
        eval_records,
        steps=int(args.mlp_steps),
        hidden_dim=int(args.mlp_hidden_dim),
        learning_rate=float(args.learning_rate),
        seed=31,
    )
    baselines["small_mlp"] = mlp_report
    predictors["small_mlp"] = mlp_predictor
    for key, rank, linear, seed in [
        ("frozen_base_smolvla_7d_adapter", None, True, 41),
        ("smolvla_7d_adapter_no_lora", None, False, 43),
        ("smolvla_state_proj_lora_rank4_7d_adapter", 4, False, 47),
        ("smolvla_state_proj_lora_rank8_7d_adapter", 8, False, 53),
    ]:
        artifact = None
        if rank is not None:
            artifact = Path(args.output_dir) / f"{key}.pt"
        report, predictor = _train_state_proj_adapter(
            train_records,
            eval_records,
            checkpoint=checkpoint,
            name=key,
            steps=int(args.adapter_steps),
            hidden_dim=int(args.adapter_hidden_dim),
            learning_rate=float(args.lora_learning_rate if rank is not None else args.learning_rate),
            seed=seed,
            lora_rank=rank,
            linear_head=linear,
            artifact_path=artifact,
        )
        baselines[key] = report
        predictors[key] = predictor
    lora_names = ["smolvla_state_proj_lora_rank4_7d_adapter", "smolvla_state_proj_lora_rank8_7d_adapter"]
    simple_names = ["mean_action", "ridge", "small_mlp"]
    best_lora_name = min(lora_names, key=lambda name: baselines[name]["eval_metrics"]["action_l2"])
    best_simple_name = min(simple_names, key=lambda name: baselines[name]["eval_metrics"]["action_l2"])
    vram_peak_mb = 0.0
    if torch.cuda.is_available():
        vram_peak_mb = _round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
    return (
        {
            "baselines": baselines,
            "best_lora_name": best_lora_name,
            "best_simple_name": best_simple_name,
            "lora_ranks_tested": [4, 8],
            "rank16_tested": False,
            "rank16_skip_reason": "Skipped to keep this standard replay reproduction bounded; rank 8 was the largest required rank.",
            "vram_peak_mb": vram_peak_mb,
            "runtime_sec": _round(time.monotonic() - started, 3),
        },
        predictors,
    )


def _demo_window(path: Path, demo_name: str, max_steps_cap: int) -> dict[str, Any]:
    return replay_bridge._demo_window(path, demo_name, int(max_steps_cap), 0)


def _policy_actions_for_case(
    *,
    demo_window: dict[str, Any],
    predictors: dict[str, Predictor],
    best_lora_name: str,
) -> dict[str, np.ndarray]:
    features = np.asarray(demo_window["features"], dtype=np.float32)
    return {
        "expert": np.asarray(demo_window["actions"], dtype=np.float32),
        "mean_action": predictors["mean_action"].predict(features),
        "ridge": predictors["ridge"].predict(features),
        "small_mlp": predictors["small_mlp"].predict(features),
        "smolvla_7d_adapter": predictors[best_lora_name].predict(features),
    }


def _run_replay(args: argparse.Namespace, split: dict[str, Any], suite: dict[str, Any], predictors: dict[str, Predictor]) -> dict[str, Any]:
    started = time.monotonic()
    if not _env_flag(REPLAY_GATE):
        return {"executed": False, "reason": f"{REPLAY_GATE}=1 is required.", "cases": [], "aggregate": {}, "runtime_sec": _round(time.monotonic() - started, 3)}
    env_cls, env_meta = replay_bridge._load_env_class_noninteractive(
        libero_root=Path(args.libero_root),
        robosuite_root=Path(args.robosuite_root),
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
    )
    cases: list[dict[str, Any]] = []
    for replay_case in split["replay_cases"]:
        path = Path(replay_case["hdf5_path"])
        demo_name = replay_case["demo_name"]
        demo_window = _demo_window(path, demo_name, int(args.max_replay_steps))
        bddl_file = Path(args.libero_root) / "libero" / "libero" / "bddl_files" / path.parent.name / f"{_task_id_from_demo_path(path)}.bddl"
        instruction = _instruction_from_path(path)
        actions_by_policy = _policy_actions_for_case(
            demo_window=demo_window,
            predictors=predictors,
            best_lora_name=suite["best_lora_name"],
        )
        offline_case = {
            name: {"action_metrics": _metrics(actions, demo_window["actions"]), "action_validity": _action_validity(actions)}
            for name, actions in actions_by_policy.items()
        }
        results: dict[str, Any] = {}
        expert_variant = {
            "name": "expert",
            "claim_role": "expert_replay_upper_bound",
            "actions": actions_by_policy["expert"],
            "use_exact_init_state": True,
        }
        results["expert"] = _run_replay_variant(
            env_cls=env_cls,
            bddl_file=bddl_file,
            camera_size=int(args.camera_size),
            init_state=np.asarray(demo_window["init_state"], dtype=np.float64),
            variant=expert_variant,
            instruction=instruction,
        )
        expert_ok = replay_bridge._success(results["expert"])
        if expert_ok:
            for name in ["mean_action", "ridge", "small_mlp", "smolvla_7d_adapter"]:
                variant = {
                    "name": name,
                    "claim_role": {
                        "mean_action": "mean_action_baseline",
                        "ridge": "ridge_baseline",
                        "small_mlp": "small_mlp_baseline",
                        "smolvla_7d_adapter": "best_standard_smolvla_7d_lora_adapter",
                    }[name],
                    "actions": actions_by_policy[name],
                    "use_exact_init_state": True,
                }
                results[name] = _run_replay_variant(
                    env_cls=env_cls,
                    bddl_file=bddl_file,
                    camera_size=int(args.camera_size),
                    init_state=np.asarray(demo_window["init_state"], dtype=np.float64),
                    variant=variant,
                    instruction=instruction,
                )
        cases.append(
            {
                "task_name": replay_case["task_name"],
                "hdf5_path": str(path),
                "demo_name": demo_name,
                "bddl_file": str(bddl_file),
                "instruction": instruction,
                "target_horizon": demo_window["target_horizon"],
                "hdf5_first_signal_index": demo_window["first_signal_index"],
                "expert_ok_for_judging": expert_ok,
                "offline_case_metrics": offline_case,
                "results": results,
            }
        )
    aggregate = _aggregate_replay(cases)
    return {
        "executed": True,
        "reason": "bounded exact-init standard replay attempted",
        "env": env_meta,
        "best_lora_policy": suite["best_lora_name"],
        "cases": cases,
        "aggregate": aggregate,
        "runtime_sec": _round(time.monotonic() - started, 3),
    }


def _progress(result: dict[str, Any] | None) -> float | None:
    return replay_bridge._progress_metric(result)


def _aggregate_replay(cases: list[dict[str, Any]]) -> dict[str, Any]:
    policy_names = ["expert", "mean_action", "ridge", "small_mlp", "smolvla_7d_adapter"]
    aggregate: dict[str, Any] = {}
    judgeable_cases = [case for case in cases if case.get("expert_ok_for_judging")]
    for policy in policy_names:
        values = []
        for case in judgeable_cases if policy != "expert" else cases:
            result = (case.get("results") or {}).get(policy)
            if result:
                values.append(result)
        progress_values = [_progress(item) for item in values]
        progress_values = [float(value) for value in progress_values if value is not None]
        aggregate[policy] = {
            "case_count": len(values),
            "success_count": int(sum(1 for item in values if replay_bridge._success(item))),
            "success_rate": _round(float(np.mean([replay_bridge._success(item) for item in values]))) if values else None,
            "reward_sum_mean": _round(float(np.mean([float(item.get("reward_sum") or 0.0) for item in values]))) if values else None,
            "first_done_indices": [item.get("first_done_index") for item in values],
            "progress_proxy_mean": _round(float(np.mean(progress_values))) if progress_values else None,
            "object_movement_mean": _round(float(np.mean([float((item.get("object_movement") or {}).get("target_object_displacement_l2") or 0.0) for item in values]))) if values else None,
            "runtime_case_steps": [item.get("steps_performed") for item in values],
        }
    aggregate["judgeable_case_count"] = len(judgeable_cases)
    aggregate["expert_all_succeeded"] = bool(cases and len(judgeable_cases) == len(cases))
    return aggregate


def _clip_replay_audit(suite: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    best = suite["best_lora_name"]
    best_offline = suite["baselines"][best]["action_validity"]
    case_rows = []
    for case in replay.get("cases") or []:
        validity = ((case.get("offline_case_metrics") or {}).get("smolvla_7d_adapter") or {}).get("action_validity") or {}
        result = ((case.get("results") or {}).get("smolvla_7d_adapter") or {})
        case_rows.append(
            {
                "task_name": case.get("task_name"),
                "demo_name": case.get("demo_name"),
                "expert_ok_for_judging": case.get("expert_ok_for_judging"),
                "clip_rate_element": validity.get("clip_rate_element"),
                "clip_rate_step": validity.get("clip_rate_step"),
                "controller_valid_rate_proxy": validity.get("controller_valid_rate_proxy"),
                "per_dim_clip_rate": validity.get("per_dim_clip_rate"),
                "dominant_clip_dim": validity.get("dominant_clip_dim"),
                "gripper_clip_rate": validity.get("gripper_clip_rate"),
                "adapter_success": replay_bridge._success(result) if result else None,
                "adapter_progress_proxy": _progress(result) if result else None,
            }
        )
    offline_per_dim = best_offline.get("per_dim_clip_rate") or []
    offline_gripper_dominates = bool(
        len(offline_per_dim) == 7
        and float(offline_per_dim[6]) > 0.0
        and float(offline_per_dim[6]) >= max(float(value) for value in offline_per_dim)
    )
    replay_gripper_dominates = bool(
        case_rows
        and all(
            row.get("dominant_clip_dim") in (None, 6)
            for row in case_rows
            if row.get("clip_rate_element") is not None and float(row.get("clip_rate_element") or 0.0) > 0.0
        )
    )
    return {
        "best_lora_policy": best,
        "offline_eval_action_validity": best_offline,
        "case_rows": case_rows,
        "dimensions_clip_most": best_offline.get("dominant_clip_dim"),
        "gripper_dominates_clipping": bool(offline_gripper_dominates and replay_gripper_dominates),
        "unnormalized_action_range_correct": bool(best_offline.get("shape_exactly_7d") and best_offline.get("finite")),
        "adapter_progress_improves_despite_clipping": _adapter_beats_replay_baselines(replay),
        "action_range_or_normalization_fix_needed": bool(
            float(best_offline.get("clip_rate_step") or 0.0) > 0.3
            or float(best_offline.get("controller_valid_rate_proxy") or 1.0) < 0.7
        ),
    }


def _adapter_beats_offline_baselines(suite: dict[str, Any]) -> bool:
    baselines = suite.get("baselines") or {}
    best = baselines[suite["best_lora_name"]]["eval_metrics"]["action_l2"]
    return all(best < baselines[name]["eval_metrics"]["action_l2"] for name in ["mean_action", "ridge", "small_mlp"])


def _adapter_beats_replay_baselines(replay: dict[str, Any]) -> bool:
    aggregate = replay.get("aggregate") or {}
    adapter = ((aggregate.get("smolvla_7d_adapter") or {}).get("progress_proxy_mean"))
    if adapter is None:
        return False
    return all(
        ((aggregate.get(name) or {}).get("progress_proxy_mean") is None)
        or float(adapter) > float((aggregate.get(name) or {}).get("progress_proxy_mean"))
        for name in ["mean_action", "ridge", "small_mlp"]
    )


def _simple_replay_beats_adapter(replay: dict[str, Any]) -> bool:
    aggregate = replay.get("aggregate") or {}
    adapter = ((aggregate.get("smolvla_7d_adapter") or {}).get("progress_proxy_mean"))
    if adapter is None:
        return True
    return any(
        ((aggregate.get(name) or {}).get("progress_proxy_mean") is not None)
        and float((aggregate.get(name) or {}).get("progress_proxy_mean")) >= float(adapter)
        for name in ["mean_action", "ridge", "small_mlp"]
    )


def _decide(report: dict[str, Any]) -> tuple[str, str]:
    replay = report.get("state3_replay") or {}
    suite = report.get("state2_baselines") or {}
    audit = report.get("state4_action_validity") or {}
    aggregate = replay.get("aggregate") or {}
    if not replay.get("executed") or not aggregate.get("expert_all_succeeded"):
        return "EXPERT_REPLAY_UNSTABLE", "Fix or narrow exact-init replay until expert succeeds on every evaluated replay case."
    offline_ok = _adapter_beats_offline_baselines(suite)
    replay_ok = _adapter_beats_replay_baselines(replay)
    if not offline_ok and _simple_replay_beats_adapter(replay):
        return "MEAN_OR_MLP_REPLAY_DOMINATED", "Simple baselines dominate the standard replay baseline; do not start method work."
    if offline_ok and not replay_ok:
        return "OFFLINE_TO_CONTROL_GAP", "Offline action improvement did not transfer to replay progress."
    if replay_ok and bool(audit.get("action_range_or_normalization_fix_needed")):
        return "READY_BUT_NEEDS_ACTION_RANGE_FIX", "Fix action range/normalization before method work; do not invent a new method yet."
    if offline_ok and replay_ok and int(aggregate.get("judgeable_case_count") or 0) > 1:
        return "READY_FOR_RA_L_METHOD_AFTER_STANDARD_BASELINE", "Predeclare this baseline and only then consider a new RA-L method."
    return "OFFLINE_TO_CONTROL_GAP", "Standard replay baseline evidence is not consistent enough for method work."


def _write_reports(report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    split = (report.get("state1_split") or {}).get("report") or {}
    suite = report.get("state2_baselines") or {}
    replay = report.get("state3_replay") or {}
    audit = report.get("state4_action_validity") or {}
    aggregate = replay.get("aggregate") or {}
    baselines = suite.get("baselines") or {}

    Path("reports/smolvla_7d_standard_replay_baseline_task_definition.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Standard Replay Baseline Task Definition",
                "",
                "Objective: reproduce fixed LIBERO_7D SmolVLA LoRA/adapter baselines on a bounded standard LIBERO split and test offline-to-control transfer.",
                "",
                "This is not a new method, not OpenVLA-OFT, not a full benchmark, and not a paper claim.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_standard_replay_baseline_plan.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Standard Replay Baseline Plan",
                "",
                "- Use local LIBERO HDF5 tasks only.",
                "- Use train/eval demo holdout with no train/eval demo leakage.",
                "- Train fixed LIBERO_7D mean, ridge, MLP, frozen/base adapter, rank-4 LoRA, and rank-8 LoRA baselines.",
                "- Replay expert, mean, ridge, MLP, and best LoRA on held-out exact-init demos.",
                "- Diagnose clipping and action range; do not change the method.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    lines = [
        "# SmolVLA 7D Standard Replay Baseline Result",
        "",
        f"Final decision: `{summary.get('final_decision')}`",
        "",
        "## Split",
        "",
        f"- tasks: `{split.get('task_names')}`",
        f"- train/eval demos: `{split.get('train_demo_count')} / {split.get('eval_demo_count')}`",
        f"- sampled train/eval records: `{split.get('sampled_train_records')} / {split.get('sampled_eval_records')}`",
        f"- raw timesteps: `{split.get('raw_timestep_count')}`",
        f"- leakage: `{split.get('leakage')}`",
        f"- mean-action nontrivial: `{split.get('task_variance_enough_to_make_mean_nontrivial')}`",
        "",
        "## Offline Metrics",
        "",
    ]
    for name in ["mean_action", "ridge", "small_mlp", "frozen_base_smolvla_7d_adapter", "smolvla_7d_adapter_no_lora", "smolvla_state_proj_lora_rank4_7d_adapter", "smolvla_state_proj_lora_rank8_7d_adapter"]:
        item = baselines.get(name) or {}
        metrics = item.get("eval_metrics") or {}
        lines.append(
            f"- {name}: action_l2 `{metrics.get('action_l2')}`, translation `{metrics.get('translation_l2')}`, rotation `{metrics.get('rotation_l2')}`, gripper_error `{metrics.get('gripper_error')}`"
        )
    lines.extend(
        [
            "",
            "## Replay Aggregate",
            "",
        ]
    )
    for name in ["expert", "mean_action", "ridge", "small_mlp", "smolvla_7d_adapter"]:
        lines.append(f"- {name}: `{aggregate.get(name)}`")
    lines.extend(
        [
            "",
            "## Action Validity",
            "",
            f"- best LoRA policy: `{audit.get('best_lora_policy')}`",
            f"- offline eval validity: `{audit.get('offline_eval_action_validity')}`",
            f"- dimensions clip most: `{audit.get('dimensions_clip_most')}`",
            f"- gripper dominates clipping: `{audit.get('gripper_dominates_clipping')}`",
            f"- action range/normalization fix needed: `{audit.get('action_range_or_normalization_fix_needed')}`",
            "",
            f"Exact next step: {summary.get('exact_next_step')}",
            "",
        ]
    )
    Path("reports/smolvla_7d_standard_replay_baseline_result.md").write_text("\n".join(lines), encoding="utf-8")
    Path("reports/smolvla_7d_action_validity_audit.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Action Validity Audit",
                "",
                f"- best LoRA policy: `{audit.get('best_lora_policy')}`",
                f"- offline eval action validity: `{audit.get('offline_eval_action_validity')}`",
                f"- replay case rows: `{audit.get('case_rows')}`",
                f"- dimensions clip most: `{audit.get('dimensions_clip_most')}`",
                f"- gripper dominates clipping: `{audit.get('gripper_dominates_clipping')}`",
                f"- unnormalized action range correct: `{audit.get('unnormalized_action_range_correct')}`",
                f"- adapter progress improves despite clipping: `{audit.get('adapter_progress_improves_despite_clipping')}`",
                f"- action range / normalization fix needed: `{audit.get('action_range_or_normalization_fix_needed')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_replay_baseline_decision.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Replay Baseline Decision",
                "",
                f"Final decision: `{summary.get('final_decision')}`",
                "",
                f"Exact next step: {summary.get('exact_next_step')}",
                "",
                "Hard rule: do not propose a new research method unless the decision is `READY_FOR_RA_L_METHOD_AFTER_STANDARD_BASELINE`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    project_lines = [
        "# Project State",
        "",
        "Date: 2026-07-09 KST",
        "",
        f"Branch: `{summary.get('branch')}`",
        "",
        f"Current decision: `{summary.get('final_decision')}`",
        "",
        "## Current Route",
        "",
        "SmolVLA 7D standard replay baseline reproduction is the active gate before any new method work.",
        "",
        "## Standard Replay Baseline",
        "",
        f"- tasks: `{split.get('task_names')}`",
        f"- expert aggregate: `{aggregate.get('expert')}`",
        f"- mean aggregate: `{aggregate.get('mean_action')}`",
        f"- ridge aggregate: `{aggregate.get('ridge')}`",
        f"- MLP aggregate: `{aggregate.get('small_mlp')}`",
        f"- SmolVLA adapter aggregate: `{aggregate.get('smolvla_7d_adapter')}`",
        f"- action range fix needed: `{audit.get('action_range_or_normalization_fix_needed')}`",
        "",
        "## Conclusion",
        "",
        f"`{summary.get('final_decision')}`",
        "",
        summary.get("exact_next_step") or "",
        "",
    ]
    Path("reports/project_state.md").write_text("\n".join(project_lines), encoding="utf-8")
    Path("reports/next_actions.md").write_text(
        "\n".join(
            [
                "# Next Actions",
                "",
                "Date: 2026-07-09 KST",
                "",
                f"Current decision: `{summary.get('final_decision')}`",
                "",
                "## Immediate Next Action",
                "",
                summary.get("exact_next_step") or "",
                "",
                "Do not start a new method unless the standard replay baseline decision permits it.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    decision_path = Path("reports/decision_log.md")
    existing = decision_path.read_text(encoding="utf-8") if decision_path.exists() else "# Decision Log\n"
    marker = "## 2026-07-09: SmolVLA 7D Standard Replay Baseline"
    entry = "\n".join(
        [
            "",
            marker,
            "",
            f"Decision: `{summary.get('final_decision')}`",
            "",
            f"- experiments happened: `{summary.get('experiments_happened')}`",
            f"- training happened: `{summary.get('training_happened')}`",
            f"- loss computed: `{summary.get('loss_computed')}`",
            f"- replay/control happened: `{summary.get('replay_control_happened')}`",
            f"- model/adapter used: `{summary.get('model_adapter_used')}`",
            f"- tasks/demos used: `{summary.get('tasks_demos_used')}`",
            f"- expert aggregate: `{aggregate.get('expert')}`",
            f"- mean/ridge/MLP/adapter progress: `{summary.get('replay_progress_summary')}`",
            f"- action validity: `{audit.get('offline_eval_action_validity')}`",
            f"- exact next step: {summary.get('exact_next_step')}",
            "",
        ]
    )
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + entry
    else:
        existing = existing.rstrip() + entry
    decision_path.write_text(existing + "\n", encoding="utf-8")


def _strip_large(payload: Any) -> Any:
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "reward_trajectory":
                result["reward_trajectory_summary_only"] = True
            elif key == "loss_curve":
                arr = value if isinstance(value, list) else []
                result["loss_curve_sample"] = {"first5": arr[:5], "last5": arr[-5:], "length": len(arr)}
            else:
                result[key] = _strip_large(value)
        return result
    if isinstance(payload, list):
        return [_strip_large(value) for value in payload]
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "TOO_HEAVY_LOCAL",
        "policy": {
            "new_method_created": False,
            "fixed_libero_7d_path_only": True,
            "downloads_performed": False,
            "openvla_oft_executed": False,
            "full_benchmark_executed": False,
            "paper_claims_made": False,
            "run_gate_set": _env_flag(RUN_GATE),
            "training_gate_set": _env_flag(TRAINING_GATE),
            "replay_gate_set": _env_flag(REPLAY_GATE),
            "forbidden_gates_set": forbidden,
            "training_performed": False,
            "loss_computed": False,
            "replay_control_performed": False,
        },
        "paths": {
            "data_root": str(Path(args.data_root)),
            "smolvla_ckpt": str(Path(args.smolvla_ckpt)),
            "output_dir": str(Path(args.output_dir)),
        },
        "state1_split": {},
        "state2_baselines": {},
        "state3_replay": {},
        "state4_action_validity": {},
        "summary": {},
        "error": None,
    }

    def finish(decision: str, next_step: str, code: int) -> tuple[dict[str, Any], int]:
        if decision not in FINAL_DECISIONS:
            raise ValueError(f"invalid final decision: {decision}")
        report["decision"] = decision
        report["summary"].update(
            {
                "final_decision": decision,
                "exact_next_step": next_step,
                "runtime_sec": _round(time.monotonic() - started, 3),
            }
        )
        return report, code

    if not report["policy"]["run_gate_set"]:
        return finish("TOO_HEAVY_LOCAL", f"Set {RUN_GATE}=1 for this bounded standard replay baseline.", 2)
    if forbidden:
        report["error"] = {"message": "Forbidden gate(s) set: " + ", ".join(forbidden)}
        return finish("TOO_HEAVY_LOCAL", "Clear forbidden download/rollout/OpenVLA-OFT/method gates and rerun.", 3)
    if not Path(args.data_root).exists() or not Path(args.smolvla_ckpt).exists():
        return finish("TOO_HEAVY_LOCAL", "Missing local LIBERO data root or SmolVLA checkpoint.", 4)
    if not report["policy"]["training_gate_set"]:
        return finish("TOO_HEAVY_LOCAL", f"Set {TRAINING_GATE}=1 to train standard fixed-7D baselines.", 5)
    if not report["policy"]["replay_gate_set"]:
        return finish("TOO_HEAVY_LOCAL", f"Set {REPLAY_GATE}=1 to run bounded exact-init replay.", 6)

    try:
        split = _build_standard_split(
            data_root=Path(args.data_root),
            max_tasks=int(args.max_tasks),
            train_demos_per_task=int(args.train_demos_per_task),
            eval_demos_per_task=int(args.eval_demos_per_task),
            records_per_demo=int(args.records_per_demo),
            replay_demos_per_task=int(args.replay_demos_per_task),
            max_replay_steps=int(args.max_replay_steps),
        )
        report["state1_split"] = {"report": split["report"]}
        suite, predictors = _run_offline_baselines(args, split)
        report["state2_baselines"] = suite
        report["policy"]["training_performed"] = True
        report["policy"]["loss_computed"] = True
        replay = _run_replay(args, split, suite, predictors)
        report["state3_replay"] = replay
        report["policy"]["replay_control_performed"] = bool(replay.get("executed"))
        report["state4_action_validity"] = _clip_replay_audit(suite, replay)
        decision, next_step = _decide(report)
        split_report = split["report"]
        aggregate = replay.get("aggregate") or {}
        report["summary"].update(
            {
                "branch": _current_branch(),
                "experiments_happened": True,
                "training_happened": True,
                "loss_computed": True,
                "replay_control_happened": bool(replay.get("executed")),
                "downloads_happened": False,
                "openvla_oft_happened": False,
                "model_adapter_used": suite.get("best_lora_name"),
                "tasks_demos_used": {
                    "tasks": split_report.get("task_names"),
                    "replay_cases": split.get("replay_cases"),
                },
                "train_eval_split": {
                    "train_demo_count": split_report.get("train_demo_count"),
                    "eval_demo_count": split_report.get("eval_demo_count"),
                    "sampled_train_records": split_report.get("sampled_train_records"),
                    "sampled_eval_records": split_report.get("sampled_eval_records"),
                    "leakage": split_report.get("leakage"),
                },
                "best_lora_name": suite.get("best_lora_name"),
                "best_simple_name": suite.get("best_simple_name"),
                "offline_to_control_transfer_summary": {
                    "adapter_beats_offline_baselines": _adapter_beats_offline_baselines(suite),
                    "adapter_beats_replay_baselines": _adapter_beats_replay_baselines(replay),
                },
                "expert_replay_aggregate": aggregate.get("expert"),
                "mean_replay_aggregate": aggregate.get("mean_action"),
                "ridge_replay_aggregate": aggregate.get("ridge"),
                "mlp_replay_aggregate": aggregate.get("small_mlp"),
                "adapter_replay_aggregate": aggregate.get("smolvla_7d_adapter"),
                "replay_progress_summary": {
                    name: (aggregate.get(name) or {}).get("progress_proxy_mean")
                    for name in ["mean_action", "ridge", "small_mlp", "smolvla_7d_adapter"]
                },
                "clip_action_validity_audit": report["state4_action_validity"],
                "vram_peak_mb": suite.get("vram_peak_mb"),
                "offline_runtime_sec": suite.get("runtime_sec"),
                "replay_runtime_sec": replay.get("runtime_sec"),
            }
        )
        return finish(decision, next_step, 0)
    except Exception as exc:  # noqa: BLE001
        report["error"] = _compact_error(exc)
        if "out of memory" in str(exc).lower():
            return finish("TOO_HEAVY_LOCAL", "Stop: local standard baseline exceeded memory.", 20)
        return finish("TOO_HEAVY_LOCAL", "Fix the reported standard replay baseline runner error and rerun.", 11)


def _current_branch() -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="C:/assets/data/libero/libero_10")
    parser.add_argument("--libero-root", default="C:/assets/repos/LIBERO")
    parser.add_argument("--robosuite-root", default="C:/assets/repos/robosuite")
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--output-dir", default="runs/smolvla_7d_standard_replay_baseline")
    parser.add_argument("--report-path", default="reports/smolvla_7d_standard_replay_baseline_result.json")
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--train-demos-per-task", type=int, default=5)
    parser.add_argument("--eval-demos-per-task", type=int, default=2)
    parser.add_argument("--replay-demos-per-task", type=int, default=1)
    parser.add_argument("--records-per-demo", type=int, default=8)
    parser.add_argument("--adapter-steps", type=int, default=800)
    parser.add_argument("--adapter-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-steps", type=int, default=800)
    parser.add_argument("--mlp-hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-replay-steps", type=int, default=280)
    parser.add_argument("--camera-size", type=int, default=64)
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    json_report = _strip_large(report)
    _write_json(Path(args.report_path), json_report)
    _write_reports(json_report)
    print(json.dumps(json_report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
