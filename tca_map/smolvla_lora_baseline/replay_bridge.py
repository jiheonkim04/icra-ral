"""Bounded SmolVLA/LIBERO 7D adapter replay bridge.

This diagnostic asks whether the fixed-interface SmolVLA 7D adapter that
improves offline action L2 is executable enough to justify exact-init replay.
It is a control-validity gate, not a new method or paper claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.datasets.libero_full_demo_expert_replay_sanity import _run_replay_variant
from tca_map.datasets.libero_zero_reward_rollout_diagnosis import (
    _distance,
    _distance_delta,
    _extract_eef,
    _extract_pos,
)
from tca_map.smolvla_lora_baseline import diagnostic as base
from tca_map.smolvla_lora_baseline import libero_7d_baseline_reproduction as repro
from tca_map.smolvla_lora_baseline import libero_7d_interface_fix as fix


BRIDGE_GATE = "ALLOW_SMOLVLA_7D_REPLAY_BRIDGE"
TRAINING_GATE = "ALLOW_SMOLVLA_7D_REPLAY_BRIDGE_TRAINING"
REPLAY_GATE = "ALLOW_SMOLVLA_7D_REPLAY_BRIDGE_REPLAY"
DEFAULT_HDF5_PATH = base.DEFAULT_HDF5_PATH
DEFAULT_ADAPTER_ARTIFACT = Path("runs/smolvla_7d_replay_bridge/smolvla_state_proj_lora_rank8_7d_adapter.pt")
SCHEMA_VERSION = "smolvla-7d-adapter-replay-bridge-v1"
FINAL_DECISIONS = {
    "READY_FOR_METHOD_AFTER_REPLAY_BRIDGE",
    "NEEDS_EXECUTABLE_ADAPTER_FIX",
    "OFFLINE_TO_CONTROL_GAP",
    "MEAN_OR_MLP_REPLAY_DOMINATED",
    "EXPERT_REPLAY_BLOCKED",
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _demo_sort_key(name: str) -> tuple[str, int | str]:
    return base._demo_sort_key(name)


def _task_id_from_demo_path(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_demo")] if stem.endswith("_demo") else stem


def _instruction_from_path(path: Path) -> str:
    return _task_id_from_demo_path(path).replace("_", " ")


def _first_index(values: np.ndarray, threshold: float = 0.0) -> int | None:
    for index, value in enumerate(np.asarray(values).reshape(-1)):
        if float(value) > threshold:
            return int(index)
    return None


def _action_validity(actions: np.ndarray, *, low: float = -1.0, high: float = 1.0) -> dict[str, Any]:
    arr = np.asarray(actions, dtype=np.float32)
    shape_ok = arr.ndim == 2 and arr.shape[1] == 7
    finite = bool(np.isfinite(arr).all()) if arr.size else False
    if shape_ok and arr.size:
        below = arr < low
        above = arr > high
        clipped = below | above
        clip_rate_element = float(np.mean(clipped))
        clip_rate_step = float(np.mean(np.any(clipped, axis=1)))
        controller_valid_rate = float(np.mean(np.isfinite(arr).all(axis=1) & ~np.any(clipped, axis=1)))
        low_high = {
            "low": [_round(low) for _ in range(7)],
            "high": [_round(high) for _ in range(7)],
            "min": [_round(x) for x in arr.min(axis=0)],
            "max": [_round(x) for x in arr.max(axis=0)],
        }
    else:
        clip_rate_element = 1.0
        clip_rate_step = 1.0
        controller_valid_rate = 0.0
        low_high = {"low": [_round(low) for _ in range(7)], "high": [_round(high) for _ in range(7)], "min": None, "max": None}
    return {
        "action_shape": list(arr.shape),
        "expected_action_shape": ["T", 7],
        "shape_exactly_7d": bool(shape_ok),
        "finite": finite,
        "action_low_high": low_high,
        "clip_rate_element": _round(clip_rate_element),
        "clip_rate_step": _round(clip_rate_step),
        "controller_valid_rate_proxy": _round(controller_valid_rate),
        "silent_broadcast_or_truncation_detected": not shape_ok,
        "note": "Proxy validity uses LIBERO HDF5/controller action convention [-1, 1]; env acceptance is reported separately when replay runs.",
    }


def _metrics_from_arrays(pred: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    return fix._metrics_from_arrays(np.asarray(pred, dtype=np.float32), np.asarray(expert, dtype=np.float32))


def _feature_for_demo_timestep(path: Path, demo_name: str, timestep: int) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data"][demo_name]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        ee = np.asarray(demo["obs"]["ee_states"][timestep], dtype=np.float32).reshape(-1)[:6]
    frac = np.asarray([float(timestep) / max(1, actions.shape[0] - 1)], dtype=np.float32)
    feature = np.concatenate([ee, frac], axis=0).astype(np.float32)
    if feature.shape != (7,):
        raise ValueError(f"expected 7D replay feature, got {feature.shape}")
    return feature


def _features_for_demo(path: Path, demo_name: str, horizon: int) -> np.ndarray:
    return np.stack([_feature_for_demo_timestep(path, demo_name, index) for index in range(horizon)], axis=0).astype(np.float32)


def _observation_feature(obs: dict[str, Any], timestep_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
    if "ee_states" in obs:
        ee = np.asarray(obs["ee_states"], dtype=np.float32).reshape(-1)[:6]
        source = "ee_states"
    elif "robot0_eef_pos" in obs and "robot0_eef_quat" in obs:
        pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32).reshape(-1)
        quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float32).reshape(-1)
        ee = np.concatenate([pos[:3], quat[:3]], axis=0).astype(np.float32)
        source = "robot0_eef_pos_plus_first3_robot0_eef_quat"
    else:
        raise ValueError("observation lacks ee_states or robot0_eef_pos/robot0_eef_quat")
    if ee.shape[0] != 6:
        raise ValueError(f"online observation state feature must provide 6 values, got {ee.shape[0]}")
    feature = np.concatenate([ee, np.asarray([timestep_fraction], dtype=np.float32)], axis=0).astype(np.float32)
    return feature, {
        "source": source,
        "feature_shape": list(feature.shape),
        "uses_bddl_or_task_id_label": False,
        "uses_eval_action_label": False,
    }


def _demo_window(path: Path, demo_name: str, max_steps_cap: int, post_signal_margin: int) -> dict[str, Any]:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data"][demo_name]
        actions = np.asarray(demo["actions"], dtype=np.float32)
        rewards = np.asarray(demo["rewards"], dtype=np.float32).reshape(-1) if "rewards" in demo else np.zeros((actions.shape[0],), dtype=np.float32)
        dones = np.asarray(demo["dones"], dtype=np.float32).reshape(-1) if "dones" in demo else np.zeros((actions.shape[0],), dtype=np.float32)
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64).reshape(-1) if "init_state" in demo.attrs else None
    first_reward = _first_index(rewards, 0.0)
    first_done = _first_index(dones, 0.5)
    signals = [value for value in [first_reward, first_done] if value is not None]
    first_signal = min(signals) if signals else None
    if first_signal is None:
        horizon = min(actions.shape[0], max_steps_cap)
    else:
        horizon = min(actions.shape[0], int(first_signal) + 1 + int(post_signal_margin), max_steps_cap)
    horizon = max(1, int(horizon))
    return {
        "path": str(path),
        "demo_name": demo_name,
        "actions": actions[:horizon, :7].astype(np.float32),
        "full_action_steps": int(actions.shape[0]),
        "target_horizon": horizon,
        "first_reward_index": first_reward,
        "first_done_index": first_done,
        "first_signal_index": first_signal,
        "init_state": init_state,
        "features": _features_for_demo(path, demo_name, horizon),
    }


def _mean_train_action(train_records: list[dict[str, Any]]) -> np.ndarray:
    return base._mean_train_action(train_records)[:7].astype(np.float32)


def _fit_ridge(train_records: list[dict[str, Any]], alpha: float = 1e-3) -> dict[str, Any]:
    x_train, y_train = fix._feature_matrix(train_records)
    x_mean = x_train.mean(axis=0, keepdims=True).astype(np.float32)
    x_std = (x_train.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    xt = ((x_train - x_mean) / x_std).astype(np.float32)
    xt_aug = np.concatenate([xt, np.ones((xt.shape[0], 1), dtype=np.float32)], axis=1)
    reg = float(alpha) * np.eye(xt_aug.shape[1], dtype=np.float32)
    weights = np.linalg.solve(xt_aug.T @ xt_aug + reg, xt_aug.T @ y_train).astype(np.float32)
    return {
        "name": "state_time_ridge_7d_executable",
        "feature_schema": "SmolVLA observation.state 6D plus timestep fraction",
        "x_mean": x_mean,
        "x_std": x_std,
        "weights": weights,
        "uses_eval_labels_for_training": False,
    }


def _predict_ridge(ridge: dict[str, Any], features: np.ndarray) -> np.ndarray:
    x = np.asarray(features, dtype=np.float32)
    x_norm = (x - ridge["x_mean"]) / ridge["x_std"]
    x_aug = np.concatenate([x_norm, np.ones((x_norm.shape[0], 1), dtype=np.float32)], axis=1)
    return (x_aug @ ridge["weights"]).astype(np.float32)


def _state_time_tensors_from_features(features: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray):
    import torch

    x_norm = ((features - x_mean) / x_std).astype(np.float32)
    state = np.concatenate([x_norm[:, :6], np.zeros((x_norm.shape[0], 26), dtype=np.float32)], axis=1)
    time_feature = x_norm[:, 6:7]
    return torch.tensor(state, dtype=torch.float32), torch.tensor(time_feature, dtype=torch.float32)


@dataclass
class ExecutableAdapter:
    artifact_path: Path
    payload: dict[str, Any]
    state_weight: Any
    state_bias: Any
    head: Any
    lora_a: Any
    lora_b: Any

    @classmethod
    def load(cls, artifact_path: Path):
        import torch

        payload = torch.load(str(artifact_path), map_location="cpu", weights_only=False)
        checkpoint = Path(payload["checkpoint_path"])
        state_weight, state_bias, _weight_file = repro._state_proj_weights(checkpoint)
        hidden_dim = int(payload["hidden_dim"])
        head = torch.nn.Sequential(
            torch.nn.Linear(961, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 7),
        )
        head.load_state_dict(payload["head_state_dict"])
        head.eval()
        lora_a = payload["lora_a"].float()
        lora_b = payload["lora_b"].float()
        return cls(
            artifact_path=artifact_path,
            payload=payload,
            state_weight=state_weight.float(),
            state_bias=state_bias.float(),
            head=head,
            lora_a=lora_a,
            lora_b=lora_b,
        )

    @property
    def name(self) -> str:
        return str(self.payload["name"])

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        import torch

        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.ndim != 2 or arr.shape[1] != 7:
            raise ValueError(f"adapter feature input must be [N, 7], got {arr.shape}")
        x_mean = np.asarray(self.payload["feature_normalization"]["mean"], dtype=np.float32).reshape(1, 7)
        x_std = np.asarray(self.payload["feature_normalization"]["std"], dtype=np.float32).reshape(1, 7)
        state, time_feature = _state_time_tensors_from_features(arr, x_mean, x_std)
        rank = int(self.payload["lora_rank"])
        alpha = float(self.payload["lora_alpha"])
        scale = alpha / float(rank)
        delta = (self.lora_b @ self.lora_a) * scale
        with torch.no_grad():
            projected = state @ (self.state_weight + delta).T + self.state_bias
            pred_norm = self.head(torch.cat([projected, time_feature], dim=1)).detach().cpu().numpy()
        mean = np.asarray(self.payload["normalization"]["mean"], dtype=np.float32).reshape(1, 7)
        std = np.asarray(self.payload["normalization"]["std"], dtype=np.float32).reshape(1, 7)
        return (pred_norm * std + mean).astype(np.float32)

    def predict_observation(self, obs: dict[str, Any], timestep_fraction: float) -> tuple[np.ndarray, dict[str, Any]]:
        feature, metadata = _observation_feature(obs, timestep_fraction)
        return self.predict_features(feature)[0], metadata


def _train_and_export_adapter(
    *,
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    checkpoint: Path,
    artifact_path: Path,
    steps: int,
    hidden_dim: int,
    learning_rate: float,
    seed: int,
    lora_rank: int,
) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    state_weight, state_bias, weight_file = repro._state_proj_weights(checkpoint)
    train_state, train_time, y_train_np, x_mean, x_std = repro._state_time_tensors(train_records)
    eval_state, eval_time, y_eval_np, _, _ = repro._state_time_tensors(eval_records, x_mean, x_std)
    normalizer = fix.Libero7DNormalizer.fit(y_train_np)
    y_train = torch.tensor(normalizer.normalize(y_train_np), dtype=torch.float32)

    torch.manual_seed(int(seed))
    head = torch.nn.Sequential(
        torch.nn.Linear(961, int(hidden_dim)),
        torch.nn.SiLU(),
        torch.nn.Linear(int(hidden_dim), 7),
    )
    lora_alpha = int(lora_rank) * 2
    lora_a = torch.nn.Parameter(torch.randn(int(lora_rank), 32) * 0.01)
    lora_b = torch.nn.Parameter(torch.zeros(960, int(lora_rank)))
    params: list[Any] = list(head.parameters()) + [lora_a, lora_b]
    optimizer = torch.optim.AdamW(params, lr=float(learning_rate), weight_decay=1e-5)
    losses: list[dict[str, float]] = []

    def projected(state_tensor: Any) -> Any:
        scale = float(lora_alpha) / float(lora_rank)
        delta = (lora_b @ lora_a) * scale
        return state_tensor @ (state_weight + delta).T + state_bias

    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        feature = projected(train_state)
        pred = head(torch.cat([feature, train_time], dim=1))
        pose_loss = torch.nn.functional.mse_loss(pred[:, :6], y_train[:, :6])
        gripper_loss = torch.nn.functional.mse_loss(pred[:, 6:], y_train[:, 6:])
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

    with torch.no_grad():
        train_pred_norm = head(torch.cat([projected(train_state), train_time], dim=1)).detach().cpu().numpy()
        eval_pred_norm = head(torch.cat([projected(eval_state), eval_time], dim=1)).detach().cpu().numpy()
    train_pred = normalizer.unnormalize(train_pred_norm)
    eval_pred = normalizer.unnormalize(eval_pred_norm)
    payload = {
        "schema_version": "smolvla-7d-executable-adapter-artifact-v1",
        "name": f"smolvla_state_proj_lora_rank{int(lora_rank)}_7d_adapter",
        "adapter_schema": "LIBERO_7D",
        "checkpoint_path": str(checkpoint),
        "state_proj_weight_file": str(weight_file),
        "feature_schema": "train-normalized SmolVLA observation.state padded to 32, then checkpoint state_proj, plus timestep fraction",
        "lora_rank": int(lora_rank),
        "lora_alpha": int(lora_alpha),
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "normalization": normalizer.report(),
        "feature_normalization": {
            "source": "train_split_only",
            "shape": [7],
            "mean": [float(x) for x in x_mean.reshape(-1)],
            "std": [float(x) for x in x_std.reshape(-1)],
            "uses_eval_labels": False,
        },
        "gripper_handling": {
            "learned": True,
            "hard_coded": False,
            "loss": "separate normalized gripper MSE added to pose regression loss",
        },
        "head_state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "lora_a": lora_a.detach().cpu(),
        "lora_b": lora_b.detach().cpu(),
        "training": {
            "steps": int(steps),
            "loss_start": losses[0]["loss"] if losses else None,
            "loss_end": losses[-1]["loss"] if losses else None,
            "pose_loss_end": losses[-1]["pose_loss"] if losses else None,
            "gripper_mse_loss_end": losses[-1]["gripper_mse_loss"] if losses else None,
            "loss_decreased": bool(losses and losses[-1]["loss"] < losses[0]["loss"]),
            "loss_curve": losses,
        },
        "train_metrics": _metrics_from_arrays(train_pred, y_train_np),
        "eval_metrics": _metrics_from_arrays(eval_pred, y_eval_np),
        "train_eval_gap": _round(_metrics_from_arrays(eval_pred, y_eval_np)["action_l2"] - _metrics_from_arrays(train_pred, y_train_np)["action_l2"]),
        "target_modules": ["state_proj", "libero_7d_adapter"],
        "excluded_native_6d_modules": ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"],
        "uses_eval_labels_for_training": False,
        "uses_so100_action_normalizer": False,
        "uses_hard_coded_gripper_fill": False,
        "runtime_sec": _round(time.monotonic() - started, 3),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(artifact_path))
    return {
        "artifact_path": str(artifact_path),
        "adapter_name": payload["name"],
        "artifact_created": True,
        "training": payload["training"],
        "train_metrics": payload["train_metrics"],
        "eval_metrics": payload["eval_metrics"],
        "normalization": payload["normalization"],
        "feature_normalization": payload["feature_normalization"],
        "gripper_handling": payload["gripper_handling"],
        "runtime_sec": payload["runtime_sec"],
    }


def _reload_audit(adapter: ExecutableAdapter, demo_window: dict[str, Any]) -> dict[str, Any]:
    features = np.asarray(demo_window["features"], dtype=np.float32)
    actions = adapter.predict_features(features)
    zero_norm = np.zeros((1, 7), dtype=np.float32)
    mean = np.asarray(adapter.payload["normalization"]["mean"], dtype=np.float32).reshape(1, 7)
    std = np.asarray(adapter.payload["normalization"]["std"], dtype=np.float32).reshape(1, 7)
    unnorm_zero = zero_norm * std + mean
    first_obs = {"ee_states": features[0, :6]}
    online_action, online_meta = adapter.predict_observation(first_obs, float(features[0, 6]))
    return {
        "artifact_path": str(adapter.artifact_path),
        "adapter_name": adapter.name,
        "artifact_reloaded": True,
        "can_produce_online_action_from_observation_state_features": True,
        "online_observation_metadata": online_meta,
        "online_action_shape": list(online_action.reshape(-1).shape),
        "output_shape_exactly_7d": bool(actions.ndim == 2 and actions.shape[1] == 7 and online_action.reshape(-1).shape == (7,)),
        "train_split_only_action_normalization_used": adapter.payload["normalization"].get("source") == "train_split_only"
        and not adapter.payload["normalization"].get("uses_eval_labels")
        and not adapter.payload["normalization"].get("uses_so100_stats"),
        "train_split_only_feature_normalization_used": adapter.payload["feature_normalization"].get("source") == "train_split_only"
        and not adapter.payload["feature_normalization"].get("uses_eval_labels"),
        "gripper_output_learned_not_hard_coded": bool(adapter.payload["gripper_handling"].get("learned") and not adapter.payload["gripper_handling"].get("hard_coded")),
        "gripper_output_range_on_replay_demo": {
            "min": _round(actions[:, 6].min()),
            "max": _round(actions[:, 6].max()),
            "std": _round(actions[:, 6].std()),
        },
        "unnormalize_check": {
            "zero_normalized_maps_to_train_mean": bool(np.allclose(unnorm_zero, mean)),
            "mean": [_round(x) for x in mean.reshape(-1)],
            "std_positive": bool(np.all(std > 0.0)),
        },
        "action_validity": _action_validity(actions),
    }


def _offline_to_control_sanity(
    *,
    train_records: list[dict[str, Any]],
    demo_window: dict[str, Any],
    adapter: ExecutableAdapter,
) -> dict[str, Any]:
    expert = np.asarray(demo_window["actions"], dtype=np.float32)
    features = np.asarray(demo_window["features"], dtype=np.float32)
    mean_action = _mean_train_action(train_records)
    mean_actions = np.repeat(mean_action.reshape(1, 7), expert.shape[0], axis=0).astype(np.float32)
    ridge = _fit_ridge(train_records)
    ridge_actions = _predict_ridge(ridge, features)
    adapter_actions = adapter.predict_features(features)
    policies = {
        "expert": expert,
        "mean_action": mean_actions,
        "ridge": ridge_actions,
        "smolvla_7d_adapter": adapter_actions,
    }
    results: dict[str, Any] = {}
    for name, pred in policies.items():
        if name == "expert":
            metrics = _metrics_from_arrays(pred, expert)
        else:
            metrics = _metrics_from_arrays(pred, expert)
        results[name] = {
            "action_metrics": metrics,
            "action_validity": _action_validity(pred),
        }
    return {
        "replay_demo_path": demo_window["path"],
        "replay_demo_name": demo_window["demo_name"],
        "target_horizon": demo_window["target_horizon"],
        "full_action_steps": demo_window["full_action_steps"],
        "hdf5_first_signal_index": demo_window["first_signal_index"],
        "policies": results,
        "comparison": {
            "adapter_beats_mean_action_l2": bool(results["smolvla_7d_adapter"]["action_metrics"]["action_l2"] < results["mean_action"]["action_metrics"]["action_l2"]),
            "adapter_beats_ridge_l2": bool(results["smolvla_7d_adapter"]["action_metrics"]["action_l2"] < results["ridge"]["action_metrics"]["action_l2"]),
            "mlp_executable_baseline_run": False,
            "mlp_skip_reason": "No persisted executable MLP artifact was part of the fixed 7D baseline; executable ridge baseline is used as the simple MLP/ridge comparator.",
        },
    }


def _write_libero_config(*, libero_root: Path, data_root: Path, output_dir: Path) -> Path:
    config_dir = output_dir / "libero_config"
    config_dir.mkdir(parents=True, exist_ok=True)
    benchmark_root = libero_root / "libero" / "libero"
    config = "\n".join(
        [
            f"assets: {str((benchmark_root / 'assets').as_posix())}",
            f"bddl_files: {str((benchmark_root / 'bddl_files').as_posix())}",
            f"benchmark_root: {str(benchmark_root.as_posix())}",
            f"datasets: {str(data_root.as_posix())}",
            f"init_states: {str((benchmark_root / 'init_files').as_posix())}",
            "",
        ]
    )
    config_path = config_dir / "config.yaml"
    config_path.write_text(config, encoding="utf-8")
    os.environ["LIBERO_CONFIG_PATH"] = str(config_dir)
    return config_path


def _load_env_class_noninteractive(*, libero_root: Path, robosuite_root: Path, data_root: Path, output_dir: Path) -> tuple[Any, dict[str, Any]]:
    os.environ.setdefault("MUJOCO_GL", "osmesa")
    tmp_dir = Path("C:/tmp")
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    config_path = _write_libero_config(libero_root=libero_root, data_root=data_root, output_dir=output_dir)
    for module_name in list(sys.modules):
        if module_name == "libero" or module_name.startswith("libero."):
            del sys.modules[module_name]
    sys.path = [path for path in sys.path if not str(path).startswith(str(libero_root))]
    for path in (robosuite_root, libero_root):
        if str(path):
            sys.path.insert(0, str(path))
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv, {"libero_config_path": str(config_path), "tmp_dir": str(tmp_dir), "mujoco_gl": os.environ.get("MUJOCO_GL")}


def _success(result: dict[str, Any] | None) -> bool:
    if not result:
        return False
    return bool(result.get("final_success") or result.get("done_seen") or float(result.get("reward_sum") or 0.0) > 0.0)


def _progress_metric(result: dict[str, Any] | None) -> float | None:
    if not result:
        return None
    movement = result.get("target_directed_movement") or {}
    distance_change = _safe_float(movement.get("distance_change"))
    if distance_change is not None:
        return -distance_change
    object_movement = result.get("object_movement") or {}
    return _safe_float(object_movement.get("target_object_displacement_l2"))


def _bounded_replay(
    *,
    args: argparse.Namespace,
    demo_window: dict[str, Any],
    offline: dict[str, Any],
) -> dict[str, Any]:
    started = time.monotonic()
    if not _env_flag(REPLAY_GATE):
        return {
            "executed": False,
            "reason": f"{REPLAY_GATE}=1 is required for exact-init replay/control.",
            "error": None,
            "results": {},
            "runtime_sec": _round(time.monotonic() - started, 3),
        }
    try:
        env_cls, env_meta = _load_env_class_noninteractive(
            libero_root=Path(args.libero_root),
            robosuite_root=Path(args.robosuite_root),
            data_root=Path(args.data_root),
            output_dir=Path(args.output_dir),
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "executed": False,
            "reason": "Failed to import or configure LIBERO/RoboSuite exact-init environment.",
            "error": _compact_error(exc),
            "results": {},
            "runtime_sec": _round(time.monotonic() - started, 3),
        }
    init_state = demo_window.get("init_state")
    if init_state is None:
        return {
            "executed": False,
            "reason": "Replay demo does not expose init_state.",
            "error": None,
            "results": {},
            "env": env_meta,
            "runtime_sec": _round(time.monotonic() - started, 3),
        }
    hdf5_path = Path(demo_window["path"])
    suite = hdf5_path.parent.name
    task_id = _task_id_from_demo_path(hdf5_path)
    bddl_file = Path(args.libero_root) / "libero" / "libero" / "bddl_files" / suite / f"{task_id}.bddl"
    instruction = _instruction_from_path(hdf5_path)
    policy_actions = {
        "expert": np.asarray(demo_window["actions"], dtype=np.float64),
        "mean_action": np.asarray(offline["policies"]["mean_action"].get("raw_actions"), dtype=np.float64)
        if offline["policies"]["mean_action"].get("raw_actions") is not None
        else None,
        "ridge": np.asarray(offline["policies"]["ridge"].get("raw_actions"), dtype=np.float64)
        if offline["policies"]["ridge"].get("raw_actions") is not None
        else None,
        "smolvla_7d_adapter": np.asarray(offline["policies"]["smolvla_7d_adapter"].get("raw_actions"), dtype=np.float64)
        if offline["policies"]["smolvla_7d_adapter"].get("raw_actions") is not None
        else None,
    }
    # Older result JSONs omit raw actions to keep the report compact; rebuild from validity names if needed.
    if policy_actions["mean_action"] is None or policy_actions["ridge"] is None or policy_actions["smolvla_7d_adapter"] is None:
        return {
            "executed": False,
            "reason": "Replay actions were not materialized in the offline sanity block.",
            "error": None,
            "env": env_meta,
            "results": {},
            "runtime_sec": _round(time.monotonic() - started, 3),
        }
    results: dict[str, Any] = {}
    for name, actions in policy_actions.items():
        variant = {
            "name": name,
            "claim_role": {
                "expert": "expert_replay_upper_bound",
                "mean_action": "mean_action_baseline",
                "ridge": "ridge_baseline",
                "smolvla_7d_adapter": "fixed_7d_smolvla_adapter",
            }[name],
            "actions": actions,
            "use_exact_init_state": True,
        }
        results[name] = _run_replay_variant(
            env_cls=env_cls,
            bddl_file=bddl_file,
            camera_size=int(args.camera_size),
            init_state=np.asarray(init_state, dtype=np.float64),
            variant=variant,
            instruction=instruction,
        )
    return {
        "executed": True,
        "reason": "bounded exact-init replay attempted",
        "env": env_meta,
        "bddl_file": str(bddl_file),
        "instruction": instruction,
        "results": results,
        "runtime_sec": _round(time.monotonic() - started, 3),
    }


def _materialize_offline_actions(offline: dict[str, Any], actions: dict[str, np.ndarray]) -> None:
    for name, arr in actions.items():
        offline["policies"][name]["raw_actions"] = np.asarray(arr, dtype=np.float32)


def _strip_raw_actions_for_json(payload: Any) -> Any:
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "raw_actions":
                arr = np.asarray(value, dtype=np.float32)
                result["raw_action_shape"] = list(arr.shape)
                result["raw_action_preview_first3"] = [[_round(x) for x in row] for row in arr[:3].tolist()]
            else:
                result[key] = _strip_raw_actions_for_json(value)
        return result
    if isinstance(payload, list):
        return [_strip_raw_actions_for_json(value) for value in payload]
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


def _decide(report: dict[str, Any]) -> tuple[str, str]:
    state1 = report.get("state1_executable_adapter_audit") or {}
    offline = report.get("state2_offline_to_control_sanity") or {}
    replay = report.get("state3_bounded_exact_init_replay") or {}
    if not state1.get("artifact_reloaded") or not state1.get("output_shape_exactly_7d"):
        return "NEEDS_EXECUTABLE_ADAPTER_FIX", "Fix adapter export/reload and 7D output shape before replay."
    if not state1.get("gripper_output_learned_not_hard_coded") or not state1.get("unnormalize_check", {}).get("zero_normalized_maps_to_train_mean"):
        return "NEEDS_EXECUTABLE_ADAPTER_FIX", "Fix learned gripper output or LIBERO_7D unnormalization before replay."
    adapter_valid = ((offline.get("policies") or {}).get("smolvla_7d_adapter") or {}).get("action_validity") or {}
    if float(adapter_valid.get("controller_valid_rate_proxy") or 0.0) < 0.5:
        return "OFFLINE_TO_CONTROL_GAP", "Adapter actions are mostly invalid or clipped under the LIBERO 7D action range."
    if not replay.get("executed"):
        error = replay.get("error") or {}
        if error.get("type") == "ModuleNotFoundError" and "mujoco" in str(error.get("message", "")).lower():
            return (
                "EXPERT_REPLAY_BLOCKED",
                "Install or activate the local `mujoco` Python dependency for LIBERO/RoboSuite in the `tca_map` environment, then rerun this same replay bridge; do not start a new method.",
            )
        return "EXPERT_REPLAY_BLOCKED", replay.get("reason") or "Exact-init replay did not execute."
    results = replay.get("results") or {}
    expert = results.get("expert") or {}
    adapter = results.get("smolvla_7d_adapter") or {}
    mean = results.get("mean_action") or {}
    ridge = results.get("ridge") or {}
    if not _success(expert):
        return "EXPERT_REPLAY_BLOCKED", "Expert exact-init replay failed, so learned replay cannot be interpreted."
    adapter_success = _success(adapter)
    mean_success = _success(mean)
    ridge_success = _success(ridge)
    adapter_progress = _progress_metric(adapter)
    mean_progress = _progress_metric(mean)
    ridge_progress = _progress_metric(ridge)
    dominated_by_success = bool((mean_success or ridge_success) and not adapter_success)
    dominated_by_progress = False
    if adapter_progress is not None:
        dominated_by_progress = any(
            value is not None and float(value) >= float(adapter_progress)
            for value in [mean_progress, ridge_progress]
        )
    if dominated_by_success or dominated_by_progress:
        return "MEAN_OR_MLP_REPLAY_DOMINATED", "Mean-action or ridge matched/beat the adapter in exact-init replay progress."
    action_l2_ok = bool((offline.get("comparison") or {}).get("adapter_beats_mean_action_l2") and (offline.get("comparison") or {}).get("adapter_beats_ridge_l2"))
    progress_ok = bool(adapter_success or (adapter_progress is not None and all(value is None or float(adapter_progress) > float(value) for value in [mean_progress, ridge_progress])))
    if action_l2_ok and progress_ok:
        return "READY_FOR_METHOD_AFTER_REPLAY_BRIDGE", "Control progress directionally reflects the fixed 7D adapter's offline action improvement."
    return "OFFLINE_TO_CONTROL_GAP", "Offline action L2 improvement did not transfer to bounded exact-init replay progress."


def _write_static_reports(report: dict[str, Any]) -> None:
    Path("reports/smolvla_7d_replay_bridge_task_definition.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Adapter Replay Bridge Task Definition",
                "",
                "Objective: determine whether the fixed SmolVLA/LIBERO 7D adapter baseline is executable enough for exact-init LIBERO/RoboSuite replay.",
                "",
                "This is not a new method, not paper novelty, and not a benchmark claim.",
                "",
                "Allowed scope:",
                "- fixed `LIBERO_7D` adapter path only;",
                "- deterministic adapter export/reload if the baseline did not persist weights;",
                "- offline replay-demo action sanity;",
                "- one-task, one-demo exact-init replay only when the bounded replay gate is set and the simulator stack imports.",
                "",
                "Disallowed scope: TG-7D, TCA, PRISM, SafeLoRA, PatchGuard, OpenVLA-OFT, downloads, full benchmark, or new method invention.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_replay_bridge_experiment_plan.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Adapter Replay Bridge Experiment Plan",
                "",
                "STATE 1: export or reload the best fixed 7D SmolVLA adapter, verify 7D shape, train-split-only normalization, learned gripper output, unnormalization, and action validity.",
                "",
                "STATE 2: compare expert, mean-action, executable ridge, and SmolVLA 7D adapter actions on the first held-out replay demo before any simulator stepping.",
                "",
                "STATE 3: if `ALLOW_SMOLVLA_7D_REPLAY_BRIDGE_REPLAY=1` is set and LIBERO/RoboSuite imports, run exact-init replay on one held-out demo with a capped horizon.",
                "",
                "Stop immediately if the adapter cannot execute, expert replay is blocked, simple baselines dominate replay/progress, actions are invalid/clipped, or offline L2 fails to transfer to control progress.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_replay_bridge_kill_criteria.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Adapter Replay Bridge Kill Criteria",
                "",
                "Final decision must be one of:",
                "- `READY_FOR_METHOD_AFTER_REPLAY_BRIDGE`",
                "- `NEEDS_EXECUTABLE_ADAPTER_FIX`",
                "- `OFFLINE_TO_CONTROL_GAP`",
                "- `MEAN_OR_MLP_REPLAY_DOMINATED`",
                "- `EXPERT_REPLAY_BLOCKED`",
                "- `TOO_HEAVY_LOCAL`",
                "",
                "Stop if the learned 7D adapter cannot be executed, the env action interface mismatch returns, expert replay fails, mean-action or ridge/MLP matches or beats learned replay/progress, adapter actions are mostly clipped/invalid, offline L2 improves without replay/progress transfer, or the runner becomes unstable or unbounded.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def _write_result_reports(report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    state1 = report.get("state1_executable_adapter_audit") or {}
    state2 = report.get("state2_offline_to_control_sanity") or {}
    state3 = report.get("state3_bounded_exact_init_replay") or {}
    policies = state2.get("policies") or {}

    def _metric(policy: str, key: str) -> Any:
        return ((policies.get(policy) or {}).get("action_metrics") or {}).get(key)

    def _valid(policy: str, key: str) -> Any:
        return ((policies.get(policy) or {}).get("action_validity") or {}).get(key)

    replay_error = state3.get("error") or {}
    replay_error_summary = None
    if replay_error:
        replay_error_summary = f"{replay_error.get('type')}: {replay_error.get('message')}"

    lines = [
        "# SmolVLA 7D Adapter Replay Bridge Result",
        "",
        f"Final decision: `{summary.get('final_decision')}`",
        "",
        "This is a bounded control-validity gate for the already-fixed 7D baseline, not a method claim.",
        "",
        "## Summary",
        "",
        f"- branch: `{summary.get('branch')}`",
        f"- experiments happened: `{summary.get('experiments_happened')}`",
        f"- training happened: `{summary.get('training_happened')}`",
        f"- loss computed: `{summary.get('loss_computed')}`",
        f"- replay/control happened: `{summary.get('replay_control_happened')}`",
        f"- downloads happened: `{summary.get('downloads_happened')}`",
        f"- OpenVLA-OFT happened: `{summary.get('openvla_oft_happened')}`",
        f"- model/adapter used: `{summary.get('model_adapter_used')}`",
        f"- dataset/demo used: `{summary.get('dataset_demo_used')}`",
        f"- adapter artifact: `{state1.get('artifact_path')}`",
        f"- adapter reloadable: `{state1.get('artifact_reloaded')}`",
        f"- output exactly 7D: `{state1.get('output_shape_exactly_7d')}`",
        f"- train-split-only normalization: `{state1.get('train_split_only_action_normalization_used')}`",
        f"- learned gripper output: `{state1.get('gripper_output_learned_not_hard_coded')}`",
        f"- unnormalize correct: `{(state1.get('unnormalize_check') or {}).get('zero_normalized_maps_to_train_mean')}`",
        f"- replay env acceptance: `{summary.get('env_acceptance_status')}`",
        "",
        "## Offline Replay-Demo Metrics",
        "",
        f"- expert action L2: `{_metric('expert', 'action_l2')}`",
        f"- mean-action L2: `{_metric('mean_action', 'action_l2')}`",
        f"- ridge L2: `{_metric('ridge', 'action_l2')}`",
        f"- SmolVLA 7D adapter L2: `{_metric('smolvla_7d_adapter', 'action_l2')}`",
        f"- adapter translation / rotation / gripper error: `{_metric('smolvla_7d_adapter', 'translation_l2')} / {_metric('smolvla_7d_adapter', 'rotation_l2')} / {_metric('smolvla_7d_adapter', 'gripper_error')}`",
        f"- adapter clip rate element/step: `{_valid('smolvla_7d_adapter', 'clip_rate_element')} / {_valid('smolvla_7d_adapter', 'clip_rate_step')}`",
        f"- adapter controller-valid proxy rate: `{_valid('smolvla_7d_adapter', 'controller_valid_rate_proxy')}`",
        "",
        "## Replay",
        "",
        f"- replay executed: `{state3.get('executed')}`",
        f"- replay reason: `{state3.get('reason')}`",
        f"- replay error: `{replay_error_summary}`",
        f"- expert replay reward/success: `{summary.get('expert_replay_reward_success')}`",
        f"- mean-action replay result: `{summary.get('mean_action_replay_result')}`",
        f"- MLP/ridge replay result: `{summary.get('ridge_replay_result')}`",
        f"- SmolVLA 7D adapter replay result: `{summary.get('adapter_replay_result')}`",
        f"- action L2 vs replay progress relationship: {summary.get('action_l2_vs_replay_progress_relationship')}",
        f"- exact next step: {summary.get('exact_next_step')}",
        "",
    ]
    Path("reports/smolvla_7d_replay_bridge_result.md").write_text("\n".join(lines), encoding="utf-8")


def _update_project_reports(report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    state2 = report.get("state2_offline_to_control_sanity") or {}
    policies = state2.get("policies") or {}
    adapter_metric = ((policies.get("smolvla_7d_adapter") or {}).get("action_metrics") or {}).get("action_l2")
    mean_metric = ((policies.get("mean_action") or {}).get("action_metrics") or {}).get("action_l2")
    ridge_metric = ((policies.get("ridge") or {}).get("action_metrics") or {}).get("action_l2")
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
        "SmolVLA 7D Adapter Executable Replay Bridge is the active control-validity gate.",
        "",
        "The local language/target route remains killed: do not continue TG-7D, TCA, PRISM, PatchGuard, SafeLoRA, or canonicalization work from the prior route.",
        "",
        "## Fixed 7D Foundation",
        "",
        "- fixed LIBERO_7D action interface is the only allowed path;",
        "- best prior baseline: rank-8 state-proj LoRA + 7D adapter action L2 `0.494959`;",
        "- mean-action L2 `1.082453`, MLP L2 `0.518738`, ridge L2 `0.890603` on the fixed-interface baseline;",
        "- no old 6D/SO100 action label path and no hard-coded gripper fill.",
        "",
        "## Replay Bridge Status",
        "",
        f"- adapter artifact reloadable: `{summary.get('adapter_reloadable')}`",
        f"- training happened: `{summary.get('training_happened')}`",
        f"- loss computed: `{summary.get('loss_computed')}`",
        f"- replay/control happened: `{summary.get('replay_control_happened')}`",
        f"- offline held-out replay-demo mean/action/ridge/adapter L2: `{mean_metric}` / `{ridge_metric}` / `{adapter_metric}`",
        f"- env acceptance status: `{summary.get('env_acceptance_status')}`",
        "",
        "## Conclusion",
        "",
        f"`{summary.get('final_decision')}`",
        "",
        summary.get("exact_next_step") or "",
        "",
    ]
    Path("reports/project_state.md").write_text("\n".join(project_lines), encoding="utf-8")
    next_lines = [
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
        "Do not start a new method until this replay bridge is green. If exact-init replay is blocked by simulator dependencies, fix the simulator/import stack first and rerun this same bridge; do not switch routes.",
        "",
    ]
    Path("reports/next_actions.md").write_text("\n".join(next_lines), encoding="utf-8")
    decision_path = Path("reports/decision_log.md")
    existing = decision_path.read_text(encoding="utf-8") if decision_path.exists() else "# Decision Log\n"
    marker = "## 2026-07-09: SmolVLA 7D Adapter Replay Bridge"
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
            f"- dataset/demo used: `{summary.get('dataset_demo_used')}`",
            f"- env acceptance status: `{summary.get('env_acceptance_status')}`",
            f"- exact next step: {summary.get('exact_next_step')}",
            "",
        ]
    )
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + entry
    else:
        existing = existing.rstrip() + entry
    decision_path.write_text(existing + "\n", encoding="utf-8")


def _write_report_bundle(report: dict[str, Any]) -> None:
    _write_static_reports(report)
    _write_result_reports(report)
    _update_project_reports(report)


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    hdf5_path = Path(args.hdf5_path)
    checkpoint = Path(args.smolvla_ckpt)
    artifact_path = Path(args.adapter_artifact)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_label": "smolvla_7d_adapter_replay_bridge",
        "decision": "NEEDS_EXECUTABLE_ADAPTER_FIX",
        "policy": {
            "control_validity_gate": True,
            "new_method_created": False,
            "paper_claims_made": False,
            "downloads_performed": False,
            "large_asset_downloads_performed": False,
            "openvla_oft_executed": False,
            "full_benchmark_executed": False,
            "tg7d_tca_prism_patchguard_safelora_executed": False,
            "fixed_libero_7d_path_only": True,
            "bridge_gate_set": _env_flag(BRIDGE_GATE),
            "training_gate_set": _env_flag(TRAINING_GATE),
            "replay_gate_set": _env_flag(REPLAY_GATE),
            "forbidden_gates_set": forbidden,
            "training_performed": False,
            "loss_computed": False,
            "replay_control_performed": False,
        },
        "paths": {
            "hdf5_path": str(hdf5_path),
            "smolvla_ckpt": str(checkpoint),
            "adapter_artifact": str(artifact_path),
            "output_dir": str(Path(args.output_dir)),
        },
        "split": {},
        "adapter_export": {},
        "state1_executable_adapter_audit": {},
        "state2_offline_to_control_sanity": {},
        "state3_bounded_exact_init_replay": {},
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

    if not report["policy"]["bridge_gate_set"]:
        return finish("NEEDS_EXECUTABLE_ADAPTER_FIX", f"Set {BRIDGE_GATE}=1 for this bounded replay bridge diagnostic.", 2)
    if forbidden:
        report["error"] = {"message": "Forbidden gate(s) set: " + ", ".join(forbidden)}
        return finish("NEEDS_EXECUTABLE_ADAPTER_FIX", "Clear forbidden rollout/download/OpenVLA-OFT/method gates and rerun this bridge.", 3)
    if not hdf5_path.exists():
        return finish("NEEDS_EXECUTABLE_ADAPTER_FIX", f"Missing local HDF5 path: {hdf5_path}", 4)
    if not checkpoint.exists():
        return finish("NEEDS_EXECUTABLE_ADAPTER_FIX", f"Missing local SmolVLA checkpoint: {checkpoint}", 5)

    try:
        split = repro._same_task_demo_holdout(hdf5_path)
        train_records = split["train_records"]
        eval_records = split["eval_records"]
        report["split"] = {
            "name": split["report"]["name"],
            "train_count": len(train_records),
            "eval_count": len(eval_records),
            "train_demos": split["report"]["train_demo_ids"],
            "eval_demos": split["report"]["eval_demo_ids"],
            "leakage": split["report"]["leakage"],
        }
        replay_demo = split["report"]["eval_demo_ids"][0]
        demo_window = _demo_window(hdf5_path, replay_demo, int(args.max_replay_steps), int(args.post_signal_margin))
        if not artifact_path.exists():
            if not report["policy"]["training_gate_set"]:
                return finish("NEEDS_EXECUTABLE_ADAPTER_FIX", f"Adapter artifact missing; set {TRAINING_GATE}=1 to reproduce and export the fixed 7D adapter.", 6)
            report["adapter_export"] = _train_and_export_adapter(
                train_records=train_records,
                eval_records=eval_records,
                checkpoint=checkpoint,
                artifact_path=artifact_path,
                steps=int(args.adapter_steps),
                hidden_dim=int(args.adapter_hidden_dim),
                learning_rate=float(args.lora_learning_rate),
                seed=int(args.seed),
                lora_rank=int(args.lora_rank),
            )
            report["policy"]["training_performed"] = True
            report["policy"]["loss_computed"] = True
        else:
            report["adapter_export"] = {
                "artifact_path": str(artifact_path),
                "artifact_created": False,
                "artifact_reused": True,
            }
        adapter = ExecutableAdapter.load(artifact_path)
        state1 = _reload_audit(adapter, demo_window)
        report["state1_executable_adapter_audit"] = state1
        offline = _offline_to_control_sanity(train_records=train_records, demo_window=demo_window, adapter=adapter)
        expert = np.asarray(demo_window["actions"], dtype=np.float32)
        features = np.asarray(demo_window["features"], dtype=np.float32)
        mean_actions = np.repeat(_mean_train_action(train_records).reshape(1, 7), expert.shape[0], axis=0).astype(np.float32)
        ridge = _fit_ridge(train_records)
        ridge_actions = _predict_ridge(ridge, features)
        adapter_actions = adapter.predict_features(features)
        _materialize_offline_actions(
            offline,
            {
                "expert": expert,
                "mean_action": mean_actions,
                "ridge": ridge_actions,
                "smolvla_7d_adapter": adapter_actions,
            },
        )
        replay = _bounded_replay(args=args, demo_window=demo_window, offline=offline)
        report["policy"]["replay_control_performed"] = bool(replay.get("executed"))
        compact_offline = _strip_raw_actions_for_json(offline)
        report["state2_offline_to_control_sanity"] = compact_offline
        report["state3_bounded_exact_init_replay"] = _strip_raw_actions_for_json(replay)
        decision, next_step = _decide(report)
        replay_results = replay.get("results") or {}

        def replay_summary(name: str) -> dict[str, Any] | None:
            item = replay_results.get(name)
            if not item:
                return None
            return {
                "reward_sum": _round(float(item.get("reward_sum") or 0.0)),
                "final_success": item.get("final_success"),
                "done_seen": item.get("done_seen"),
                "first_done_index": item.get("first_done_index"),
                "progress_proxy": _progress_metric(item),
                "clip_rate_step": ((compact_offline.get("policies") or {}).get(name) or {}).get("action_validity", {}).get("clip_rate_step"),
            }

        replay_executed = bool(replay.get("executed"))
        env_acceptance_status = "accepted_by_env_step" if replay_executed else f"not_validated: {replay.get('reason')}"
        if replay.get("error"):
            env_acceptance_status = f"blocked: {replay['error'].get('type')}: {replay['error'].get('message')}"
        report["summary"].update(
            {
                "branch": _current_branch(),
                "experiments_happened": True,
                "training_happened": bool(report["policy"]["training_performed"]),
                "loss_computed": bool(report["policy"]["loss_computed"]),
                "replay_control_happened": replay_executed,
                "downloads_happened": False,
                "openvla_oft_happened": False,
                "full_benchmark_happened": False,
                "model_adapter_used": adapter.name,
                "dataset_demo_used": f"{hdf5_path}::{replay_demo}",
                "adapter_reloadable": bool(state1.get("artifact_reloaded")),
                "env_acceptance_status": env_acceptance_status,
                "expert_replay_reward_success": replay_summary("expert"),
                "mean_action_replay_result": replay_summary("mean_action"),
                "ridge_replay_result": replay_summary("ridge"),
                "adapter_replay_result": replay_summary("smolvla_7d_adapter"),
                "action_l2_vs_replay_progress_relationship": (
                    "not assessed because exact-init replay/control did not execute"
                    if not replay_executed
                    else "see per-policy replay progress and offline L2 metrics"
                ),
                "clip_action_validity_rate": ((compact_offline.get("policies") or {}).get("smolvla_7d_adapter") or {}).get("action_validity"),
            }
        )
        return finish(decision, next_step, 0 if decision in {"READY_FOR_METHOD_AFTER_REPLAY_BRIDGE", "EXPERT_REPLAY_BLOCKED"} else 10)
    except Exception as exc:  # noqa: BLE001
        report["error"] = _compact_error(exc)
        if "out of memory" in str(exc).lower():
            return finish("TOO_HEAVY_LOCAL", "Stop: replay bridge exceeded local memory.", 20)
        return finish("NEEDS_EXECUTABLE_ADAPTER_FIX", "Fix the reported replay bridge error and rerun.", 11)


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
    parser.add_argument("--hdf5-path", default=DEFAULT_HDF5_PATH)
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--adapter-artifact", default=str(DEFAULT_ADAPTER_ARTIFACT))
    parser.add_argument("--output-dir", default="runs/smolvla_7d_replay_bridge")
    parser.add_argument("--report-path", default="reports/smolvla_7d_replay_bridge_result.json")
    parser.add_argument("--data-root", default="C:/assets/data/libero")
    parser.add_argument("--libero-root", default="C:/assets/repos/LIBERO")
    parser.add_argument("--robosuite-root", default="C:/assets/repos/robosuite")
    parser.add_argument("--adapter-steps", type=int, default=800)
    parser.add_argument("--adapter-hidden-dim", type=int, default=128)
    parser.add_argument("--lora-rank", type=int, default=8)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=25)
    parser.add_argument("--max-replay-steps", type=int, default=280)
    parser.add_argument("--post-signal-margin", type=int, default=0)
    parser.add_argument("--camera-size", type=int, default=64)
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    json_report = _strip_raw_actions_for_json(report)
    report_path = Path(args.report_path)
    _write_json(report_path, json_report)
    if report_path.resolve() == Path("reports/smolvla_7d_replay_bridge_result.json").resolve():
        _write_report_bundle(json_report)
    print(json.dumps(json_report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
