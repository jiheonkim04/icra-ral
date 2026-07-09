"""SmolVLA LIBERO 7D action range and controller-validity fix.

This runner is infrastructure for the fixed LIBERO_7D path. It audits the
previous executable adapter, trains a bounded range-aware adapter using only
the train split, and reruns exact-init replay on the expert-success eligible
set. It does not introduce a new method or tune on replay reward.
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
from tca_map.datasets.libero_zero_reward_rollout_diagnosis import (
    _best_object_key,
    _distance,
    _distance_delta,
    _extract_eef,
    _extract_pos,
    _object_position_keys,
)
from tca_map.smolvla_lora_baseline import diagnostic as base
from tca_map.smolvla_lora_baseline import exact_init_replay_stabilization as exact_init
from tca_map.smolvla_lora_baseline import libero_7d_baseline_reproduction as repro
from tca_map.smolvla_lora_baseline import libero_7d_interface_fix as fix
from tca_map.smolvla_lora_baseline import replay_bridge
from tca_map.smolvla_lora_baseline import standard_replay_baseline as standard


RUN_GATE = "ALLOW_SMOLVLA_7D_ACTION_RANGE_FIX"
SCHEMA_VERSION = "smolvla-7d-action-range-fix-v1"
FINAL_DECISIONS = {
    "READY_FOR_METHOD_AFTER_RANGE_FIX",
    "RANGE_FIXED_BUT_CONTROL_GAP_REMAINS",
    "GRIPPER_CONVENTION_FAILURE",
    "NORMALIZATION_STILL_INVALID",
    "CLIP_ONLY_BASELINE_DOMINATES",
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
POLICIES = [
    "expert",
    "mean_action",
    "ridge",
    "small_mlp",
    "previous_unfixed_adapter",
    "previous_unfixed_adapter_clip_only",
    "range_fixed_smolvla_7d_adapter",
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def _task_id_from_demo_path(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_demo")] if stem.endswith("_demo") else stem


def _instruction_from_path(path: Path) -> str:
    return _task_id_from_demo_path(path).replace("_", " ")


def _bddl_file(libero_root: Path, hdf5_path: Path) -> Path:
    return (
        libero_root
        / "libero"
        / "libero"
        / "bddl_files"
        / hdf5_path.parent.name
        / f"{_task_id_from_demo_path(hdf5_path)}.bddl"
    )


def _load_eligible_cases(path: Path) -> list[dict[str, Any]]:
    report = _read_json(path)
    return list((report.get("state3_fixed_eligibility_set") or {}).get("eligible_cases") or [])


def _best_adapter_name(path: Path) -> str:
    report = _read_json(path)
    return str((report.get("summary") or {}).get("best_lora_name") or "smolvla_state_proj_lora_rank4_7d_adapter")


def _stats(actions: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.size == 0:
        return {"shape": list(arr.shape), "dtype": str(arr.dtype), "min": None, "max": None, "mean": None, "std": None}
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": [_round(x) for x in arr.min(axis=0)],
        "max": [_round(x) for x in arr.max(axis=0)],
        "mean": [_round(x) for x in arr.mean(axis=0)],
        "std": [_round(x) for x in arr.std(axis=0)],
    }


def _gripper_distribution(actions: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 7 or arr.shape[0] == 0:
        return {"available": False}
    grip = arr[:, 6]
    unique = np.unique(np.round(grip, 6))
    sign_changes = np.where(np.diff(np.sign(grip)) != 0)[0] + 1
    binary_signed = bool(np.all(np.isin(unique, [-1.0, 1.0])))
    sign_based = bool(np.all(np.isin(np.sign(unique), [-1.0, 0.0, 1.0])) and np.any(unique < 0.0) and np.any(unique > 0.0))
    return {
        "available": True,
        "count": int(grip.shape[0]),
        "negative": int(np.sum(grip < 0.0)),
        "zero": int(np.sum(grip == 0.0)),
        "positive": int(np.sum(grip > 0.0)),
        "min": _round(float(grip.min())),
        "max": _round(float(grip.max())),
        "mean": _round(float(grip.mean())),
        "std": _round(float(grip.std())),
        "unique_values_first20": [_round(x) for x in unique[:20]],
        "binary_signed": binary_signed,
        "continuous": bool(unique.shape[0] > 8),
        "sign_based": sign_based,
        "first_nonnegative_index": next((int(index) for index, value in enumerate(grip) if float(value) >= 0.0), None),
        "first_positive_index": next((int(index) for index, value in enumerate(grip) if float(value) > 0.0), None),
        "sign_change_indices_first10": [int(x) for x in sign_changes[:10]],
    }


def _gripper_accuracy(pred: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    pred_arr = np.asarray(pred, dtype=np.float32)
    expert_arr = np.asarray(expert, dtype=np.float32)
    if pred_arr.ndim != 2 or expert_arr.ndim != 2 or pred_arr.shape[1] < 7 or expert_arr.shape[1] < 7:
        return {"available": False}
    n = min(pred_arr.shape[0], expert_arr.shape[0])
    pred_sign = np.sign(pred_arr[:n, 6])
    expert_sign = np.sign(expert_arr[:n, 6])
    nonzero = expert_sign != 0
    return {
        "available": True,
        "sample_count": int(n),
        "sign_accuracy": _round(float(np.mean(pred_sign[nonzero] == expert_sign[nonzero]))) if np.any(nonzero) else None,
        "first_sign_mismatch_index": next(
            (int(index) for index in range(n) if expert_sign[index] != 0 and pred_sign[index] != expert_sign[index]),
            None,
        ),
        "pred_distribution": _gripper_distribution(pred_arr[:n]),
        "label_distribution": _gripper_distribution(expert_arr[:n]),
    }


def _validity(actions: np.ndarray, low: np.ndarray | None = None, high: np.ndarray | None = None) -> dict[str, Any]:
    arr = np.asarray(actions, dtype=np.float32)
    if low is None:
        return standard._action_validity(arr)
    low_arr = np.asarray(low, dtype=np.float32).reshape(1, 7)
    high_arr = np.asarray(high, dtype=np.float32).reshape(1, 7)
    shape_ok = arr.ndim == 2 and arr.shape[1] == 7
    finite = bool(np.isfinite(arr).all()) if arr.size else False
    if shape_ok and arr.size:
        clipped = (arr < low_arr) | (arr > high_arr)
        return {
            "action_shape": list(arr.shape),
            "expected_action_shape": ["T", 7],
            "shape_exactly_7d": True,
            "finite": finite,
            "action_low_high": {
                "low": [_round(x) for x in low_arr.reshape(-1)],
                "high": [_round(x) for x in high_arr.reshape(-1)],
                "min": [_round(x) for x in arr.min(axis=0)],
                "max": [_round(x) for x in arr.max(axis=0)],
            },
            "clip_rate_element": _round(float(np.mean(clipped))),
            "clip_rate_step": _round(float(np.mean(np.any(clipped, axis=1)))),
            "controller_valid_rate_proxy": _round(float(np.mean(np.isfinite(arr).all(axis=1) & ~np.any(clipped, axis=1)))),
            "per_dim_clip_rate": [_round(x) for x in np.mean(clipped, axis=0)],
            "dominant_clip_dim": int(np.argmax(np.mean(clipped, axis=0))),
            "gripper_clip_rate": _round(float(np.mean(clipped[:, 6]))),
            "silent_broadcast_or_truncation_detected": False,
            "note": "Proxy validity uses env action low/high when available.",
        }
    return {
        "action_shape": list(arr.shape),
        "expected_action_shape": ["T", 7],
        "shape_exactly_7d": False,
        "finite": finite,
        "clip_rate_element": 1.0,
        "clip_rate_step": 1.0,
        "controller_valid_rate_proxy": 0.0,
        "per_dim_clip_rate": None,
        "dominant_clip_dim": None,
        "gripper_clip_rate": None,
        "silent_broadcast_or_truncation_detected": True,
    }


def _action_range_rows(validity: dict[str, Any]) -> list[dict[str, Any]]:
    low_high = validity.get("action_low_high") or {}
    per_dim = validity.get("per_dim_clip_rate") or [None] * 7
    rows = []
    for dim in range(7):
        rows.append(
            {
                "dim": dim,
                "role": "gripper" if dim == 6 else ("translation" if dim < 3 else "rotation"),
                "low": (low_high.get("low") or [None] * 7)[dim],
                "high": (low_high.get("high") or [None] * 7)[dim],
                "min": (low_high.get("min") or [None] * 7)[dim],
                "max": (low_high.get("max") or [None] * 7)[dim],
                "clip_rate": per_dim[dim],
            }
        )
    return rows


def _concat_hdf5_actions(data_root: Path) -> np.ndarray:
    import h5py

    arrays: list[np.ndarray] = []
    for path in sorted(data_root.glob("*.hdf5")):
        with h5py.File(path, "r") as handle:
            for demo_name in sorted(handle["data"].keys(), key=base._demo_sort_key):
                arrays.append(np.asarray(handle["data"][demo_name]["actions"], dtype=np.float32)[:, :7])
    if not arrays:
        raise FileNotFoundError(f"no HDF5 action arrays found under {data_root}")
    return np.concatenate(arrays, axis=0).astype(np.float32)


def _case_actions(eligible_cases: list[dict[str, Any]], max_replay_steps: int, post_signal_margin: int) -> np.ndarray:
    arrays = []
    for case in eligible_cases:
        window = replay_bridge._demo_window(Path(case["hdf5_path"]), case["demo_name"], int(max_replay_steps), int(post_signal_margin))
        arrays.append(np.asarray(window["actions"], dtype=np.float32))
    return np.concatenate(arrays, axis=0).astype(np.float32)


def _env_action_range(args: argparse.Namespace, eligible_cases: list[dict[str, Any]]) -> dict[str, Any]:
    started = time.monotonic()
    if not eligible_cases:
        return {"available": False, "reason": "no eligible cases", "runtime_sec": _round(time.monotonic() - started)}
    env_cls, env_meta = replay_bridge._load_env_class_noninteractive(
        libero_root=Path(args.libero_root),
        robosuite_root=Path(args.robosuite_root),
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
    )
    case = eligible_cases[0]
    path = Path(case["hdf5_path"])
    bddl_file = _bddl_file(Path(args.libero_root), path)
    env = None
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=int(args.camera_size), camera_widths=int(args.camera_size))
        spec = getattr(env, "action_spec", None)
        if callable(spec):
            spec = spec()
        low = high = None
        if isinstance(spec, tuple) and len(spec) == 2:
            low = np.asarray(spec[0], dtype=np.float32).reshape(-1)[:7]
            high = np.asarray(spec[1], dtype=np.float32).reshape(-1)[:7]
        if low is None or high is None or low.shape[0] != 7 or high.shape[0] != 7:
            low = np.full((7,), -1.0, dtype=np.float32)
            high = np.full((7,), 1.0, dtype=np.float32)
            source = "fallback_libero_controller_convention"
        else:
            source = "env.action_spec"
        return {
            "available": True,
            "source": source,
            "env": env_meta,
            "low": [_round(x) for x in low],
            "high": [_round(x) for x in high],
            "range_rows": [
                {"dim": dim, "role": "gripper" if dim == 6 else ("translation" if dim < 3 else "rotation"), "low": _round(low[dim]), "high": _round(high[dim])}
                for dim in range(7)
            ],
            "runtime_sec": _round(time.monotonic() - started, 3),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "error": _compact_error(exc),
            "low": [-1.0] * 7,
            "high": [1.0] * 7,
            "source": "fallback_after_env_probe_error",
            "runtime_sec": _round(time.monotonic() - started, 3),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _adapter_norm_and_actions(adapter: replay_bridge.ExecutableAdapter, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import torch

    arr = np.asarray(features, dtype=np.float32)
    x_mean = np.asarray(adapter.payload["feature_normalization"]["mean"], dtype=np.float32).reshape(1, 7)
    x_std = np.asarray(adapter.payload["feature_normalization"]["std"], dtype=np.float32).reshape(1, 7)
    state, time_feature = replay_bridge._state_time_tensors_from_features(arr, x_mean, x_std)
    rank = int(adapter.payload["lora_rank"])
    alpha = float(adapter.payload["lora_alpha"])
    scale = alpha / float(rank)
    delta = (adapter.lora_b @ adapter.lora_a) * scale
    with torch.no_grad():
        projected = state @ (adapter.state_weight + delta).T + adapter.state_bias
        pred_norm = adapter.head(torch.cat([projected, time_feature], dim=1)).detach().cpu().numpy()
    mean = np.asarray(adapter.payload["normalization"]["mean"], dtype=np.float32).reshape(1, 7)
    std = np.asarray(adapter.payload["normalization"]["std"], dtype=np.float32).reshape(1, 7)
    return pred_norm.astype(np.float32), (pred_norm * std + mean).astype(np.float32)


@dataclass
class Predictor:
    name: str
    predict: Callable[[np.ndarray], np.ndarray]
    report: dict[str, Any]


@dataclass
class RangeFixedAdapter:
    name: str
    checkpoint_path: str
    state_weight: Any
    state_bias: Any
    head: Any
    lora_a: Any
    lora_b: Any
    x_mean: np.ndarray
    x_std: np.ndarray
    lora_alpha: int
    lora_rank: int
    metadata: dict[str, Any]

    def predict(self, features: np.ndarray) -> np.ndarray:
        import torch

        arr = np.asarray(features, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        state, time_feature = replay_bridge._state_time_tensors_from_features(arr, self.x_mean, self.x_std)
        scale = float(self.lora_alpha) / float(self.lora_rank)
        delta = (self.lora_b @ self.lora_a) * scale
        with torch.no_grad():
            projected = state @ (self.state_weight + delta).T + self.state_bias
            raw = self.head(torch.cat([projected, time_feature], dim=1)).detach().cpu()
            continuous = torch.tanh(raw[:, :6])
            gripper = torch.sigmoid(raw[:, 6:7]) * 2.0 - 1.0
            pred = torch.cat([continuous, gripper], dim=1).numpy()
        return pred.astype(np.float32)


def _train_range_fixed_adapter(
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
) -> tuple[dict[str, Any], RangeFixedAdapter]:
    import torch

    started = time.monotonic()
    state_weight, state_bias, weight_file = repro._state_proj_weights(checkpoint)
    train_state, train_time, y_train_np, x_mean, x_std = repro._state_time_tensors(train_records)
    eval_state, eval_time, y_eval_np, _, _ = repro._state_time_tensors(eval_records, x_mean, x_std)
    y_train = torch.tensor(y_train_np, dtype=torch.float32)
    y_eval = torch.tensor(y_eval_np, dtype=torch.float32)
    gripper_target = (y_train[:, 6:7] > 0.0).float()
    torch.manual_seed(int(seed))
    head = torch.nn.Sequential(
        torch.nn.Linear(961, int(hidden_dim)),
        torch.nn.SiLU(),
        torch.nn.Linear(int(hidden_dim), 7),
    )
    lora_alpha = int(lora_rank) * 2
    lora_a = torch.nn.Parameter(torch.randn(int(lora_rank), 32) * 0.01)
    lora_b = torch.nn.Parameter(torch.zeros(960, int(lora_rank)))
    optimizer = torch.optim.AdamW(list(head.parameters()) + [lora_a, lora_b], lr=float(learning_rate), weight_decay=1e-5)
    losses: list[dict[str, float]] = []

    def projected(state_tensor: Any) -> Any:
        scale = float(lora_alpha) / float(lora_rank)
        delta = (lora_b @ lora_a) * scale
        return state_tensor @ (state_weight + delta).T + state_bias

    def actions_from_raw(raw: Any) -> Any:
        continuous = torch.tanh(raw[:, :6])
        gripper = torch.sigmoid(raw[:, 6:7]) * 2.0 - 1.0
        return torch.cat([continuous, gripper], dim=1)

    for _step in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        raw = head(torch.cat([projected(train_state), train_time], dim=1))
        pred_action = actions_from_raw(raw)
        pose_loss = torch.nn.functional.mse_loss(pred_action[:, :6], y_train[:, :6])
        gripper_bce = torch.nn.functional.binary_cross_entropy_with_logits(raw[:, 6:7], gripper_target)
        gripper_mse = torch.nn.functional.mse_loss(pred_action[:, 6:7], y_train[:, 6:7])
        range_penalty = torch.relu(torch.abs(pred_action) - 1.0).pow(2).mean()
        loss = pose_loss + gripper_bce + 0.5 * gripper_mse + 0.1 * range_penalty
        loss.backward()
        optimizer.step()
        losses.append(
            {
                "loss": _round(loss.detach().cpu()),
                "pose_loss": _round(pose_loss.detach().cpu()),
                "gripper_bce_loss": _round(gripper_bce.detach().cpu()),
                "gripper_mse_loss": _round(gripper_mse.detach().cpu()),
                "range_penalty": _round(range_penalty.detach().cpu()),
            }
        )

    adapter = RangeFixedAdapter(
        name=f"smolvla_state_proj_lora_rank{int(lora_rank)}_7d_range_fixed_adapter",
        checkpoint_path=str(checkpoint),
        state_weight=state_weight.float(),
        state_bias=state_bias.float(),
        head=head.eval(),
        lora_a=lora_a.detach().cpu().float(),
        lora_b=lora_b.detach().cpu().float(),
        x_mean=x_mean,
        x_std=x_std,
        lora_alpha=int(lora_alpha),
        lora_rank=int(lora_rank),
        metadata={},
    )
    train_features, _ = fix._feature_matrix(train_records)
    eval_features, _ = fix._feature_matrix(eval_records)
    train_pred = adapter.predict(train_features)
    eval_pred = adapter.predict(eval_features)
    vram_peak_mb = 0.0
    if torch.cuda.is_available():
        vram_peak_mb = _round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
    payload = {
        "schema_version": "smolvla-7d-range-fixed-adapter-artifact-v1",
        "name": adapter.name,
        "checkpoint_path": str(checkpoint),
        "state_proj_weight_file": str(weight_file),
        "feature_schema": "fixed LIBERO ee_states 6D plus timestep fraction, train-split feature normalization",
        "adapter_schema": "LIBERO_7D_RANGE_FIXED",
        "lora_rank": int(lora_rank),
        "lora_alpha": int(lora_alpha),
        "hidden_dim": int(hidden_dim),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "output_mapping": {
            "continuous_dims_0_5": "tanh(raw) maps to env action range [-1, 1]",
            "gripper_dim_6": "2 * sigmoid(raw_logit) - 1, trained with BCE on signed binary label plus MSE",
            "post_unnormalization_clipping_is_main_fix": False,
        },
        "normalization": {
            "source": "train_split_only_feature_normalization_only",
            "action_normalization": "not used for output unnormalization; adapter predicts bounded env actions directly",
            "uses_eval_labels": False,
            "uses_so100_stats": False,
        },
        "feature_normalization": {
            "source": "train_split_only",
            "shape": [7],
            "mean": [_round(x) for x in x_mean.reshape(-1)],
            "std": [_round(x) for x in x_std.reshape(-1)],
            "uses_eval_labels": False,
        },
        "gripper_handling": {
            "learned": True,
            "hard_coded": False,
            "label_convention": "signed binary LIBERO gripper command",
            "loss": "BCEWithLogits on positive-vs-negative sign plus signed-sigmoid MSE",
        },
        "target_modules": ["state_proj", "libero_7d_adapter"],
        "excluded_native_6d_modules": ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"],
        "head_state_dict": {key: value.detach().cpu() for key, value in head.state_dict().items()},
        "lora_a": lora_a.detach().cpu(),
        "lora_b": lora_b.detach().cpu(),
        "training": {
            "steps": int(steps),
            "batch_size": "full_train_split",
            "loss_start": losses[0]["loss"] if losses else None,
            "loss_end": losses[-1]["loss"] if losses else None,
            "loss_decreased": bool(losses and losses[-1]["loss"] < losses[0]["loss"]),
            "loss_curve_sample": {"first5": losses[:5], "last5": losses[-5:]},
        },
        "train_metrics": fix._metrics_from_arrays(train_pred, y_train_np),
        "eval_metrics": fix._metrics_from_arrays(eval_pred, y_eval_np),
        "train_eval_gap": _round(fix._metrics_from_arrays(eval_pred, y_eval_np)["action_l2"] - fix._metrics_from_arrays(train_pred, y_train_np)["action_l2"]),
        "action_validity": standard._action_validity(eval_pred),
        "gripper_accuracy": _gripper_accuracy(eval_pred, y_eval_np),
        "uses_eval_labels_for_training": False,
        "uses_hard_coded_gripper_fill": False,
        "uses_replay_reward_for_training": False,
        "vram_peak_mb": vram_peak_mb,
        "runtime_sec": _round(time.monotonic() - started, 3),
    }
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, str(artifact_path))
    adapter.metadata = payload
    return (
        {
            "artifact_path": str(artifact_path),
            "adapter_name": adapter.name,
            "artifact_created": True,
            "training": payload["training"],
            "train_metrics": payload["train_metrics"],
            "eval_metrics": payload["eval_metrics"],
            "train_eval_gap": payload["train_eval_gap"],
            "action_validity": payload["action_validity"],
            "normalization": payload["normalization"],
            "feature_normalization": payload["feature_normalization"],
            "gripper_handling": payload["gripper_handling"],
            "gripper_accuracy": payload["gripper_accuracy"],
            "vram_peak_mb": payload["vram_peak_mb"],
            "runtime_sec": payload["runtime_sec"],
        },
        adapter,
    )


def _fit_affine_calibrator(train_features: np.ndarray, train_labels: np.ndarray, base_predict: Callable[[np.ndarray], np.ndarray]) -> Predictor:
    pred = np.asarray(base_predict(train_features), dtype=np.float32)
    labels = np.asarray(train_labels, dtype=np.float32)
    coeff = []
    for dim in range(7):
        design = np.stack([pred[:, dim], np.ones((pred.shape[0],), dtype=np.float32)], axis=1)
        sol, *_ = np.linalg.lstsq(design, labels[:, dim], rcond=None)
        coeff.append(sol.astype(np.float32))
    coeff_arr = np.stack(coeff, axis=0).astype(np.float32)

    def predict(features: np.ndarray) -> np.ndarray:
        values = np.asarray(base_predict(features), dtype=np.float32)
        out = np.empty_like(values)
        for dim in range(7):
            out[:, dim] = values[:, dim] * coeff_arr[dim, 0] + coeff_arr[dim, 1]
        return np.clip(out, -1.0, 1.0).astype(np.float32)

    return Predictor(
        "train_split_affine_range_calibrated_adapter_diagnostic",
        predict,
        {
            "calibration": "per-dim affine least squares from previous adapter train predictions to train labels, then env-range clip",
            "uses_train_labels": True,
            "uses_eval_labels": False,
            "uses_replay_reward": False,
            "not_method_success": True,
            "coefficients": [[_round(x) for x in row] for row in coeff_arr],
        },
    )


def _offline_eval_entry(name: str, pred: np.ndarray, expert: np.ndarray, train_pred: np.ndarray | None = None, train_labels: np.ndarray | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    entry = {
        "name": name,
        "eval_metrics": fix._metrics_from_arrays(pred, expert),
        "action_validity": standard._action_validity(pred),
        "gripper_accuracy": _gripper_accuracy(pred, expert),
        "gripper_distribution": _gripper_distribution(pred),
    }
    if train_pred is not None and train_labels is not None:
        entry["train_metrics"] = fix._metrics_from_arrays(train_pred, train_labels)
        entry["train_eval_gap"] = _round(entry["eval_metrics"]["action_l2"] - entry["train_metrics"]["action_l2"])
    if extra:
        entry.update(extra)
    return entry


def _build_predictors_and_offline(args: argparse.Namespace, split: dict[str, Any], eligible_cases: list[dict[str, Any]], best_adapter_name: str) -> tuple[dict[str, Any], dict[str, Predictor]]:
    started = time.monotonic()
    checkpoint = Path(args.smolvla_ckpt)
    adapter_path = Path(args.adapter_dir) / f"{best_adapter_name}.pt"
    previous_adapter = replay_bridge.ExecutableAdapter.load(adapter_path)
    train_records = split["train_records"]
    eval_records = split["eval_records"]
    x_train, y_train = fix._feature_matrix(train_records)
    x_eval, y_eval = fix._feature_matrix(eval_records)
    train_actions = repro._concat_record_actions(train_records)
    mean_action = train_actions.mean(axis=0).astype(np.float32)
    mean_predict = lambda features: np.repeat(mean_action.reshape(1, 7), np.asarray(features).shape[0], axis=0).astype(np.float32)
    ridge = standard._fit_ridge(train_records)
    mlp_report, mlp_predictor = standard._train_feature_mlp(
        train_records,
        eval_records,
        steps=int(args.mlp_steps),
        hidden_dim=int(args.mlp_hidden_dim),
        learning_rate=float(args.learning_rate),
        seed=31,
    )
    range_report, range_adapter = _train_range_fixed_adapter(
        train_records=train_records,
        eval_records=eval_records,
        checkpoint=checkpoint,
        artifact_path=Path(args.output_dir) / "smolvla_state_proj_lora_rank4_7d_range_fixed_adapter.pt",
        steps=int(args.adapter_steps),
        hidden_dim=int(args.adapter_hidden_dim),
        learning_rate=float(args.lora_learning_rate),
        seed=59,
        lora_rank=4,
    )
    previous_predict = previous_adapter.predict_features
    clip_predict = lambda features: np.clip(previous_predict(features), -1.0, 1.0).astype(np.float32)
    affine = _fit_affine_calibrator(x_train, y_train, previous_predict)

    previous_train_norm, previous_train = _adapter_norm_and_actions(previous_adapter, x_train)
    previous_eval_norm, previous_eval = _adapter_norm_and_actions(previous_adapter, x_eval)
    previous_clip_eval = clip_predict(x_eval)
    affine_eval = affine.predict(x_eval)
    range_train = range_adapter.predict(x_train)
    range_eval = range_adapter.predict(x_eval)
    baselines = {
        "expert": _offline_eval_entry("expert", y_eval, y_eval),
        "mean_action": _offline_eval_entry("mean_action", mean_predict(x_eval), y_eval),
        "ridge": _offline_eval_entry("ridge", ridge.predict(x_eval), y_eval),
        "small_mlp": {
            **mlp_report,
            "gripper_accuracy": _gripper_accuracy(mlp_predictor.predict(x_eval), y_eval),
            "gripper_distribution": _gripper_distribution(mlp_predictor.predict(x_eval)),
        },
        "previous_unfixed_adapter": _offline_eval_entry(
            "previous_unfixed_adapter",
            previous_eval,
            y_eval,
            previous_train,
            y_train,
            {
                "artifact_path": str(adapter_path),
                "raw_normalized_output_stats": _stats(previous_eval_norm),
                "train_raw_normalized_output_stats": _stats(previous_train_norm),
                "normalization": previous_adapter.payload.get("normalization"),
                "feature_normalization": previous_adapter.payload.get("feature_normalization"),
            },
        ),
        "previous_unfixed_adapter_clip_only": _offline_eval_entry(
            "previous_unfixed_adapter_clip_only",
            previous_clip_eval,
            y_eval,
            np.clip(previous_train, -1.0, 1.0).astype(np.float32),
            y_train,
            {"postprocess": "np.clip(previous_unfixed_adapter_action, -1, 1); baseline only, not main fix"},
        ),
        "train_split_affine_range_calibrated_adapter_diagnostic": _offline_eval_entry(
            "train_split_affine_range_calibrated_adapter_diagnostic",
            affine_eval,
            y_eval,
            affine.predict(x_train),
            y_train,
            affine.report,
        ),
        "range_fixed_smolvla_7d_adapter": _offline_eval_entry(
            "range_fixed_smolvla_7d_adapter",
            range_eval,
            y_eval,
            range_train,
            y_train,
            range_report,
        ),
    }
    predictors = {
        "mean_action": Predictor("mean_action", mean_predict, {"kind": "constant_mean_train_action"}),
        "ridge": Predictor("ridge", ridge.predict, {"kind": "closed_form_ridge_train_split"}),
        "small_mlp": Predictor("small_mlp", mlp_predictor.predict, {"kind": "bounded_train_split_mlp"}),
        "previous_unfixed_adapter": Predictor("previous_unfixed_adapter", previous_predict, {"kind": "persisted_previous_executable_adapter", "artifact_path": str(adapter_path)}),
        "previous_unfixed_adapter_clip_only": Predictor("previous_unfixed_adapter_clip_only", clip_predict, {"kind": "clip_only_baseline", "not_method_success": True}),
        "train_split_affine_range_calibrated_adapter_diagnostic": affine,
        "range_fixed_smolvla_7d_adapter": Predictor("range_fixed_smolvla_7d_adapter", range_adapter.predict, {"kind": "range_fixed_bounded_output_adapter", "artifact_path": range_report["artifact_path"]}),
    }
    return (
        {
            "executed": True,
            "reason": "bounded train/eval on fixed LIBERO_7D feature path",
            "best_previous_adapter_name": best_adapter_name,
            "best_previous_adapter_artifact": str(adapter_path),
            "baselines": baselines,
            "range_fixed_training": range_report,
            "mlp_training": mlp_report.get("training"),
            "vram_peak_mb": range_report.get("vram_peak_mb"),
            "runtime_sec": _round(time.monotonic() - started, 3),
            "eligible_case_count": len(eligible_cases),
        },
        predictors,
    )


def _first_divergence(actions: np.ndarray, expert: np.ndarray, threshold: float = 1.0) -> dict[str, Any]:
    arr = np.asarray(actions, dtype=np.float32)
    ref = np.asarray(expert, dtype=np.float32)
    n = min(arr.shape[0], ref.shape[0])
    if n == 0:
        return {"available": False}
    error = np.linalg.norm(arr[:n] - ref[:n], axis=1)
    pred_sign = np.sign(arr[:n, 6])
    expert_sign = np.sign(ref[:n, 6])
    return {
        "available": True,
        "threshold": float(threshold),
        "first_action_l2_gt_threshold": next((int(index) for index, value in enumerate(error) if float(value) > float(threshold)), None),
        "first_gripper_sign_mismatch": next((int(index) for index in range(n) if expert_sign[index] != 0 and pred_sign[index] != expert_sign[index]), None),
        "mean_action_l2": _round(float(np.mean(error))),
        "max_action_l2": _round(float(np.max(error))),
    }


def _gripper_trace(actions: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(actions, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] < 7 or arr.shape[0] == 0:
        return {"available": False}
    grip = arr[:, 6]
    return {
        "available": True,
        "first20": [_round(x) for x in grip[:20]],
        "last20": [_round(x) for x in grip[-20:]],
        "summary": _gripper_distribution(arr),
    }


def _attach_action_diagnostics(result: dict[str, Any], actions: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    result = dict(result)
    result["action_validity"] = standard._action_validity(actions)
    result["gripper_command_trace"] = _gripper_trace(actions)
    result["first_divergence_timestep"] = _first_divergence(actions, expert)
    return result


def _run_online_variant(
    *,
    env_cls: Any,
    bddl_file: Path,
    camera_size: int,
    init_state: np.ndarray,
    name: str,
    claim_role: str,
    instruction: str,
    horizon: int,
    full_action_steps: int,
    expert_actions: np.ndarray,
    predict: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "variant": name,
        "claim_role": claim_role,
        "use_exact_init_state": True,
        "feature_builder": "fixed_live_libero_ee_states",
        "env_created": False,
        "reset_ok": False,
        "set_init_state_ok": False,
        "steps_requested": int(horizon),
        "steps_performed": 0,
        "reward_sum": 0.0,
        "final_reward": 0.0,
        "final_success": None,
        "done_seen": False,
        "first_positive_reward_index": None,
        "first_done_index": None,
        "first_success_index": None,
        "done_indices": [],
        "success_indices": [],
        "reward_trajectory": [],
        "action_feature_metadata_first": None,
        "available_obs_keys": [],
        "target_key_audit": None,
        "target_directed_movement": None,
        "object_movement": None,
        "eef_start": None,
        "eef_final": None,
        "eef_displacement_l2": None,
        "error": None,
    }
    env = None
    actions = []
    obs = None
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=int(camera_size), camera_widths=int(camera_size))
        summary["env_created"] = True
        env.seed(0)
        obs = env.reset()
        summary["reset_ok"] = True
        obs = env.set_init_state(init_state)
        summary["set_init_state_ok"] = True
        summary["eef_start"] = _extract_eef(obs)
        if isinstance(obs, dict):
            summary["available_obs_keys"] = sorted(str(key) for key in obs.keys())[:80]
        target_audit = _best_object_key(obs, instruction)
        target_key = target_audit["best_key"]
        target_start = _extract_pos(obs, target_key)
        summary["target_key_audit"] = target_audit
        for index in range(int(horizon)):
            timestep_fraction = float(index) / max(1, int(full_action_steps) - 1)
            feature, metadata = replay_bridge._observation_feature(obs, timestep_fraction)
            if summary["action_feature_metadata_first"] is None:
                summary["action_feature_metadata_first"] = metadata
            action = np.asarray(predict(feature.reshape(1, 7)), dtype=np.float32).reshape(-1)[:7]
            actions.append(action)
            obs, reward, done, _info = env.step(action.astype(np.float64))
            reward_value = float(reward)
            try:
                success_value = bool(env.check_success())
            except Exception:
                success_value = None
            summary["steps_performed"] += 1
            summary["reward_sum"] += reward_value
            summary["final_reward"] = reward_value
            summary["reward_trajectory"].append(round(reward_value, 6))
            if reward_value > 0.0 and summary["first_positive_reward_index"] is None:
                summary["first_positive_reward_index"] = int(index)
            if bool(done):
                summary["done_seen"] = True
                summary["done_indices"].append(int(index))
                if summary["first_done_index"] is None:
                    summary["first_done_index"] = int(index)
            if success_value:
                summary["success_indices"].append(int(index))
                if summary["first_success_index"] is None:
                    summary["first_success_index"] = int(index)
            if bool(done) or reward_value > 0.0 or success_value:
                break
        summary["eef_final"] = _extract_eef(obs)
        if summary["eef_start"] is not None and summary["eef_final"] is not None:
            summary["eef_displacement_l2"] = _round(float(np.linalg.norm(np.asarray(summary["eef_final"]) - np.asarray(summary["eef_start"]))))
        target_final = _extract_pos(obs, target_key)
        summary["target_directed_movement"] = _distance_delta(summary["eef_start"], summary["eef_final"], target_start, target_final)
        target_object_distance = _distance(target_start, target_final)
        summary["object_movement"] = {
            "available": target_object_distance is not None,
            "target_object_key": target_key,
            "target_object_displacement_l2": target_object_distance,
            "object_position_keys_missing": not bool(_object_position_keys(obs)),
        }
        try:
            summary["final_success"] = bool(env.check_success())
        except Exception:
            summary["final_success"] = None
    except Exception as exc:  # noqa: BLE001
        summary["error"] = _compact_error(exc)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    action_arr = np.asarray(actions, dtype=np.float32).reshape((-1, 7)) if actions else np.zeros((0, 7), dtype=np.float32)
    summary["action_shape"] = list(action_arr.shape)
    summary["action_validity"] = standard._action_validity(action_arr) if action_arr.size else {}
    summary["gripper_command_trace"] = _gripper_trace(action_arr)
    summary["first_divergence_timestep"] = _first_divergence(action_arr, expert_actions)
    summary["passed"] = bool(summary["env_created"] and summary["reset_ok"] and summary["set_init_state_ok"] and summary["error"] is None)
    return summary


def _aggregate_replay(cases: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for policy in POLICIES:
        values = [(case.get("results") or {}).get(policy) for case in cases]
        values = [item for item in values if item]
        progress = [replay_bridge._progress_metric(item) for item in values]
        progress = [float(value) for value in progress if value is not None]
        aggregate[policy] = {
            "case_count": len(values),
            "success_count": int(sum(1 for item in values if replay_bridge._success(item))),
            "success_rate": _round(float(np.mean([replay_bridge._success(item) for item in values]))) if values else None,
            "reward_sum_mean": _round(float(np.mean([float(item.get("reward_sum") or 0.0) for item in values]))) if values else None,
            "first_done_indices": [item.get("first_done_index") for item in values],
            "progress_proxy_mean": _round(float(np.mean(progress))) if progress else None,
            "object_movement_mean": _round(float(np.mean([float((item.get("object_movement") or {}).get("target_object_displacement_l2") or 0.0) for item in values]))) if values else None,
            "runtime_case_steps": [item.get("steps_performed") for item in values],
            "clip_rate_step_mean": _round(float(np.mean([float((item.get("action_validity") or {}).get("clip_rate_step") or 0.0) for item in values]))) if values else None,
            "controller_valid_rate_proxy_mean": _round(float(np.mean([float((item.get("action_validity") or {}).get("controller_valid_rate_proxy") or 0.0) for item in values]))) if values else None,
        }
    return aggregate


def _replay_after_range_fix(args: argparse.Namespace, eligible_cases: list[dict[str, Any]], predictors: dict[str, Predictor]) -> dict[str, Any]:
    started = time.monotonic()
    env_cls, env_meta = replay_bridge._load_env_class_noninteractive(
        libero_root=Path(args.libero_root),
        robosuite_root=Path(args.robosuite_root),
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
    )
    cases: list[dict[str, Any]] = []
    for case in eligible_cases:
        path = Path(case["hdf5_path"])
        demo_window = replay_bridge._demo_window(path, case["demo_name"], int(args.max_replay_steps), int(args.post_signal_margin))
        bddl_file = _bddl_file(Path(args.libero_root), path)
        instruction = _instruction_from_path(path)
        expert_actions = np.asarray(demo_window["actions"], dtype=np.float32)
        features = np.asarray(demo_window["features"], dtype=np.float32)
        mean_actions = predictors["mean_action"].predict(features)
        results = {
            "expert": _attach_action_diagnostics(
                _run_replay_variant(
                    env_cls=env_cls,
                    bddl_file=bddl_file,
                    camera_size=int(args.camera_size),
                    init_state=np.asarray(demo_window["init_state"], dtype=np.float64),
                    variant={"name": "expert", "claim_role": "expert_replay_upper_bound", "actions": expert_actions, "use_exact_init_state": True},
                    instruction=instruction,
                ),
                expert_actions,
                expert_actions,
            ),
            "mean_action": _attach_action_diagnostics(
                _run_replay_variant(
                    env_cls=env_cls,
                    bddl_file=bddl_file,
                    camera_size=int(args.camera_size),
                    init_state=np.asarray(demo_window["init_state"], dtype=np.float64),
                    variant={"name": "mean_action", "claim_role": "mean_action_baseline", "actions": mean_actions, "use_exact_init_state": True},
                    instruction=instruction,
                ),
                mean_actions,
                expert_actions,
            ),
        }
        for name, role in [
            ("ridge", "ridge_baseline_fixed_live_features"),
            ("small_mlp", "small_mlp_fixed_live_features"),
            ("previous_unfixed_adapter", "previous_unfixed_adapter_fixed_live_features"),
            ("previous_unfixed_adapter_clip_only", "clip_only_postprocessed_previous_adapter_baseline"),
            ("range_fixed_smolvla_7d_adapter", "range_fixed_smolvla_7d_adapter"),
        ]:
            results[name] = _run_online_variant(
                env_cls=env_cls,
                bddl_file=bddl_file,
                camera_size=int(args.camera_size),
                init_state=np.asarray(demo_window["init_state"], dtype=np.float64),
                name=name,
                claim_role=role,
                instruction=instruction,
                horizon=int(demo_window["target_horizon"]),
                full_action_steps=int(demo_window["full_action_steps"]),
                expert_actions=expert_actions,
                predict=predictors[name].predict,
            )
        cases.append(
            {
                "task_name": case["task_name"],
                "demo_name": case["demo_name"],
                "hdf5_path": str(path),
                "target_horizon": int(demo_window["target_horizon"]),
                "hdf5_first_signal_index": demo_window.get("first_signal_index"),
                "results": results,
            }
        )
    return {
        "executed": True,
        "reason": "bounded exact-init replay on expert-success eligible cases after range fix",
        "env": env_meta,
        "cases": cases,
        "aggregate": _aggregate_replay(cases),
        "runtime_sec": _round(time.monotonic() - started, 3),
    }


def _state1_audit(args: argparse.Namespace, split: dict[str, Any], eligible_cases: list[dict[str, Any]], best_adapter_name: str) -> dict[str, Any]:
    started = time.monotonic()
    all_actions = _concat_hdf5_actions(Path(args.data_root))
    train_actions = repro._concat_record_actions(split["train_records"])
    env_range = _env_action_range(args, eligible_cases)
    low = np.asarray(env_range.get("low") or [-1.0] * 7, dtype=np.float32)
    high = np.asarray(env_range.get("high") or [1.0] * 7, dtype=np.float32)
    expert_actions = _case_actions(eligible_cases, int(args.max_replay_steps), int(args.post_signal_margin))
    adapter = replay_bridge.ExecutableAdapter.load(Path(args.adapter_dir) / f"{best_adapter_name}.pt")
    eligible_features = []
    for case in eligible_cases:
        window = replay_bridge._demo_window(Path(case["hdf5_path"]), case["demo_name"], int(args.max_replay_steps), int(args.post_signal_margin))
        eligible_features.append(np.asarray(window["features"], dtype=np.float32))
    features = np.concatenate(eligible_features, axis=0).astype(np.float32)
    raw_norm, unnormalized = _adapter_norm_and_actions(adapter, features)
    clipped = np.clip(unnormalized, low.reshape(1, 7), high.reshape(1, 7)).astype(np.float32)
    validity_before = _validity(unnormalized, low, high)
    return {
        "executed": True,
        "hdf5_action_stats_all_local": _stats(all_actions),
        "train_split_action_stats": _stats(train_actions),
        "env_action_range": env_range,
        "expert_replay_action_stats_eligible": _stats(expert_actions),
        "adapter_raw_normalized_output_distribution": _stats(raw_norm),
        "adapter_unnormalized_output_distribution": _stats(unnormalized),
        "adapter_clipped_output_distribution": _stats(clipped),
        "before_fix_action_validity": validity_before,
        "before_fix_action_validity_table": _action_range_rows(validity_before),
        "gripper_output_distribution": _gripper_distribution(unnormalized),
        "gripper_label_distribution_train": _gripper_distribution(train_actions),
        "gripper_label_distribution_eligible_expert": _gripper_distribution(expert_actions),
        "gripper_convention": {
            "target_type": "binary_signed" if _gripper_distribution(train_actions).get("binary_signed") else "other",
            "env_expected_range": {"low": _round(low[6]), "high": _round(high[6])},
            "adapter_matches_env_range_before_fix": bool(float(validity_before.get("gripper_clip_rate") or 0.0) == 0.0),
            "dominant_clip_dimension": validity_before.get("dominant_clip_dim"),
            "dominant_clip_dimension_is_gripper": bool(validity_before.get("dominant_clip_dim") == 6),
        },
        "unnormalize_maps_beyond_env_bounds": bool(float(validity_before.get("clip_rate_element") or 0.0) > 0.0),
        "normalization_mapping": adapter.payload.get("normalization"),
        "runtime_sec": _round(time.monotonic() - started, 3),
    }


def _decide(report: dict[str, Any]) -> tuple[str, str]:
    offline = report.get("state3_offline_after_range_fix") or {}
    replay = report.get("state4_replay_after_range_fix") or {}
    state1 = report.get("state1_action_range_and_clipping_audit") or {}
    baselines = offline.get("baselines") or {}
    aggregate = replay.get("aggregate") or {}
    previous_valid = ((baselines.get("previous_unfixed_adapter") or {}).get("action_validity") or {})
    fixed_valid = ((baselines.get("range_fixed_smolvla_7d_adapter") or {}).get("action_validity") or {})

    def value_or(mapping: dict[str, Any], key: str, default: float) -> float:
        value = mapping.get(key)
        return float(default) if value is None else float(value)

    if value_or(fixed_valid, "clip_rate_step", 1.0) > 0.05 or value_or(fixed_valid, "controller_valid_rate_proxy", 0.0) < 0.95:
        return "NORMALIZATION_STILL_INVALID", "Range-fixed adapter still emits out-of-range actions; inspect output mapping and env bounds."
    previous_clip = value_or(previous_valid, "clip_rate_step", 0.0)
    fixed_clip = value_or(fixed_valid, "clip_rate_step", 0.0)
    fixed_quality = (baselines.get("range_fixed_smolvla_7d_adapter") or {}).get("eval_metrics") or {}
    previous_quality = (baselines.get("previous_unfixed_adapter") or {}).get("eval_metrics") or {}
    if fixed_quality and previous_quality and float(fixed_quality.get("action_l2") or 999.0) > float(previous_quality.get("action_l2") or 0.0) * 1.5:
        return "GRIPPER_CONVENTION_FAILURE", "Range fix preserved bounds but severely damaged offline action quality; gripper convention remains suspect."
    adapter = aggregate.get("range_fixed_smolvla_7d_adapter") or {}
    clip_only = aggregate.get("previous_unfixed_adapter_clip_only") or {}
    mean = aggregate.get("mean_action") or {}
    ridge = aggregate.get("ridge") or {}
    mlp = aggregate.get("small_mlp") or {}
    previous = aggregate.get("previous_unfixed_adapter") or {}
    if int((aggregate.get("expert") or {}).get("success_count") or 0) < int((aggregate.get("expert") or {}).get("case_count") or 0):
        return "TOO_HEAVY_LOCAL", "Expert replay did not remain stable on all evaluated cases; rerun eligibility before judging learned policies."
    fixed_progress = adapter.get("progress_proxy_mean")
    clip_progress = clip_only.get("progress_proxy_mean")
    if fixed_progress is not None and clip_progress is not None and float(clip_progress) >= float(fixed_progress):
        return "CLIP_ONLY_BASELINE_DOMINATES", "Clip-only postprocessing matched or beat the range-fixed adapter; do not count the range fix as method success."
    simple_progress = [mean.get("progress_proxy_mean"), ridge.get("progress_proxy_mean"), mlp.get("progress_proxy_mean")]
    beats_progress = fixed_progress is not None and all(value is None or float(fixed_progress) > float(value) for value in simple_progress)
    beats_success = int(adapter.get("success_count") or 0) > max(int(mean.get("success_count") or 0), int(ridge.get("success_count") or 0), int(mlp.get("success_count") or 0))
    validity_improved = fixed_clip < previous_clip and value_or(fixed_valid, "controller_valid_rate_proxy", 0.0) > value_or(previous_valid, "controller_valid_rate_proxy", 0.0)
    if validity_improved and (beats_progress or beats_success):
        return "READY_FOR_METHOD_AFTER_RANGE_FIX", "Preserve fixed-feature and range-fixed baseline; only now consider method work."
    state1_grip = ((state1.get("gripper_convention") or {}).get("dominant_clip_dimension_is_gripper"))
    fixed_grip = ((baselines.get("range_fixed_smolvla_7d_adapter") or {}).get("gripper_accuracy") or {})
    if state1_grip and fixed_grip.get("sign_accuracy") is not None and float(fixed_grip.get("sign_accuracy") or 0.0) < 0.75:
        return "GRIPPER_CONVENTION_FAILURE", "Gripper remains the likely failure point after range repair; learned sign convention is too weak."
    return "RANGE_FIXED_BUT_CONTROL_GAP_REMAINS", "Action validity improved, but range-fixed adapter still fails to beat simple replay baselines."


def _write_reports(report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    state1 = report.get("state1_action_range_and_clipping_audit") or {}
    state3 = report.get("state3_offline_after_range_fix") or {}
    state4 = report.get("state4_replay_after_range_fix") or {}
    baselines = state3.get("baselines") or {}
    aggregate = state4.get("aggregate") or {}
    main_lines = [
        "# SmolVLA 7D Action Range Fix",
        "",
        f"Final decision: `{summary.get('final_decision')}`",
        "",
        "This is executable LIBERO_7D infrastructure, not a new RA-L method.",
        "",
        f"- action range before/after: `{summary.get('action_range_before')}` / `{summary.get('action_range_after')}`",
        f"- clip rate before/after: `{summary.get('clip_rate_before')}` / `{summary.get('clip_rate_after')}`",
        f"- controller-valid proxy before/after: `{summary.get('controller_valid_before')}` / `{summary.get('controller_valid_after')}`",
        f"- gripper handling before/after: `{summary.get('gripper_handling_before')}` / `{summary.get('gripper_handling_after')}`",
        f"- offline metrics before/after: `{summary.get('offline_metrics_before')}` / `{summary.get('offline_metrics_after')}`",
        f"- offline baseline comparison: `{summary.get('offline_baseline_comparison')}`",
        f"- replay metrics before/after: `{summary.get('replay_metrics_before')}` / `{summary.get('replay_metrics_after')}`",
        f"- simple baseline comparison: `{summary.get('simple_baseline_comparison')}`",
        "",
        f"Exact next step: {summary.get('exact_next_step')}",
        "",
    ]
    Path("reports/smolvla_7d_action_range_fix.md").write_text("\n".join(main_lines), encoding="utf-8")
    Path("reports/smolvla_7d_action_validity_audit.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Action Validity Audit",
                "",
                f"- before-fix action validity: `{state1.get('before_fix_action_validity')}`",
                f"- per-dim clipping table: `{state1.get('before_fix_action_validity_table')}`",
                f"- env action range table: `{((state1.get('env_action_range') or {}).get('range_rows'))}`",
                f"- unnormalize maps beyond env bounds: `{state1.get('unnormalize_maps_beyond_env_bounds')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_gripper_range_audit.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Gripper Range Audit",
                "",
                f"- gripper convention: `{state1.get('gripper_convention')}`",
                f"- label distribution train: `{state1.get('gripper_label_distribution_train')}`",
                f"- label distribution eligible expert: `{state1.get('gripper_label_distribution_eligible_expert')}`",
                f"- previous output distribution: `{state1.get('gripper_output_distribution')}`",
                f"- range-fixed gripper handling: `{((baselines.get('range_fixed_smolvla_7d_adapter') or {}).get('gripper_handling'))}`",
                f"- range-fixed gripper accuracy: `{((baselines.get('range_fixed_smolvla_7d_adapter') or {}).get('gripper_accuracy'))}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_normalization_range_audit.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Normalization Range Audit",
                "",
                f"- previous normalization mapping: `{state1.get('normalization_mapping')}`",
                f"- raw normalized output distribution: `{state1.get('adapter_raw_normalized_output_distribution')}`",
                f"- unnormalized output distribution: `{state1.get('adapter_unnormalized_output_distribution')}`",
                f"- clipped output distribution: `{state1.get('adapter_clipped_output_distribution')}`",
                f"- range-fixed normalization: `{((baselines.get('range_fixed_smolvla_7d_adapter') or {}).get('normalization'))}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_range_fix_replay_result.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Range-Fix Replay Result",
                "",
                f"- aggregate: `{aggregate}`",
                f"- runtime sec: `{state4.get('runtime_sec')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_range_fix_decision.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Range-Fix Decision",
                "",
                f"Final decision: `{summary.get('final_decision')}`",
                "",
                f"Exact next step: {summary.get('exact_next_step')}",
                "",
                "Do not propose a new RA-L method unless the decision is `READY_FOR_METHOD_AFTER_RANGE_FIX`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    project_state = [
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
        "SmolVLA 7D action range and controller-validity fix is the active infrastructure gate.",
        "",
        "## Range Fix",
        "",
        f"- clip rate before/after: `{summary.get('clip_rate_before')}` / `{summary.get('clip_rate_after')}`",
        f"- controller-valid proxy before/after: `{summary.get('controller_valid_before')}` / `{summary.get('controller_valid_after')}`",
        f"- offline metrics before/after: `{summary.get('offline_metrics_before')}` / `{summary.get('offline_metrics_after')}`",
        f"- offline baseline comparison: `{summary.get('offline_baseline_comparison')}`",
        f"- replay metrics before/after: `{summary.get('replay_metrics_before')}` / `{summary.get('replay_metrics_after')}`",
        "",
        "## Conclusion",
        "",
        f"`{summary.get('final_decision')}`",
        "",
        str(summary.get("exact_next_step")),
        "",
    ]
    Path("reports/project_state.md").write_text("\n".join(project_state), encoding="utf-8")
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
                str(summary.get("exact_next_step")),
                "",
                "Do not start a new method unless the range-fix decision is `READY_FOR_METHOD_AFTER_RANGE_FIX`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    decision_append = [
        "## 2026-07-09: SmolVLA 7D Action Range Fix",
        "",
        f"Decision: `{summary.get('final_decision')}`",
        "",
        f"- experiments happened: `{summary.get('experiments_happened')}`",
        f"- training happened: `{summary.get('training_happened')}`",
        f"- replay/control happened: `{summary.get('replay_control_happened')}`",
        f"- action range before/after: `{summary.get('action_range_before')}` / `{summary.get('action_range_after')}`",
        f"- clip rate before/after: `{summary.get('clip_rate_before')}` / `{summary.get('clip_rate_after')}`",
        f"- controller-valid proxy before/after: `{summary.get('controller_valid_before')}` / `{summary.get('controller_valid_after')}`",
        f"- offline metrics before/after: `{summary.get('offline_metrics_before')}` / `{summary.get('offline_metrics_after')}`",
        f"- offline baseline comparison: `{summary.get('offline_baseline_comparison')}`",
        f"- replay metrics before/after: `{summary.get('replay_metrics_before')}` / `{summary.get('replay_metrics_after')}`",
        f"- simple baseline comparison: `{summary.get('simple_baseline_comparison')}`",
        f"- exact next step: {summary.get('exact_next_step')}",
        "",
    ]
    log_path = Path("reports/decision_log.md")
    prior = log_path.read_text(encoding="utf-8") if log_path.exists() else "# Decision Log\n\n"
    marker = "## 2026-07-09: SmolVLA 7D Action Range Fix"
    if marker in prior:
        prior = prior[: prior.index(marker)].rstrip() + "\n"
    log_path.write_text(prior.rstrip() + "\n" + "\n".join(decision_append), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "TOO_HEAVY_LOCAL",
        "policy": {
            "run_gate_set": _env_flag(RUN_GATE),
            "forbidden_gates_set": forbidden,
            "new_method_created": False,
            "downloads_performed": False,
            "openvla_oft_executed": False,
            "full_benchmark_executed": False,
            "paper_claims_made": False,
            "uses_old_6d_so100_path": False,
            "uses_hard_coded_gripper_fill": False,
            "uses_eval_labels_for_calibration": False,
            "uses_replay_reward_for_training": False,
        },
        "state0_split_and_eligible_set": {},
        "state1_action_range_and_clipping_audit": {},
        "state2_bounded_range_gripper_fix": {},
        "state3_offline_after_range_fix": {},
        "state4_replay_after_range_fix": {},
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
        return finish("TOO_HEAVY_LOCAL", f"Set {RUN_GATE}=1 for this bounded action range fix.", 2)
    if forbidden:
        return finish("TOO_HEAVY_LOCAL", f"Forbidden gates are set for this infrastructure run: {forbidden}", 3)

    try:
        split = standard._build_standard_split(
            data_root=Path(args.data_root),
            max_tasks=int(args.max_tasks),
            train_demos_per_task=int(args.train_demos_per_task),
            eval_demos_per_task=int(args.eval_demos_per_task),
            records_per_demo=int(args.records_per_demo),
            replay_demos_per_task=1,
            max_replay_steps=int(args.max_replay_steps),
        )
        eligible_cases = _load_eligible_cases(Path(args.exact_init_report_path))
        if not eligible_cases:
            return finish("TOO_HEAVY_LOCAL", "No expert-success eligible set found; rerun exact-init stabilization first.", 4)
        best_name = _best_adapter_name(Path(args.exact_init_report_path))
        report["state0_split_and_eligible_set"] = {
            "standard_split": split["report"],
            "eligible_cases": eligible_cases,
            "eligible_case_count": len(eligible_cases),
            "best_previous_adapter_name": best_name,
            "adapter_dir": str(args.adapter_dir),
        }
        state1 = _state1_audit(args, split, eligible_cases, best_name)
        report["state1_action_range_and_clipping_audit"] = state1
        offline, predictors = _build_predictors_and_offline(args, split, eligible_cases, best_name)
        report["state2_bounded_range_gripper_fix"] = {
            "implemented": True,
            "allowed_fix_types_used": [
                "documented tanh output squashing for bounded continuous dims",
                "signed sigmoid learned gripper output with BCE plus MSE based on signed binary labels",
                "train-split-only affine/range calibration diagnostic baseline",
                "clip-only postprocessed baseline for comparison only",
                "shape and convention guards",
            ],
            "not_used": [
                "eval-set calibration",
                "hard-coded expert gripper fill",
                "oracle gripper timing",
                "task-specific hand tuning",
                "BDDL/eval metadata leakage",
                "replay reward optimization",
            ],
        }
        report["state3_offline_after_range_fix"] = offline
        replay = _replay_after_range_fix(args, eligible_cases, predictors)
        report["state4_replay_after_range_fix"] = replay
        decision, next_step = _decide(report)

        baselines = offline.get("baselines") or {}
        aggregate = replay.get("aggregate") or {}
        prev_valid = ((baselines.get("previous_unfixed_adapter") or {}).get("action_validity") or {})
        fixed_valid = ((baselines.get("range_fixed_smolvla_7d_adapter") or {}).get("action_validity") or {})
        report["summary"].update(
            {
                "branch": _current_branch(),
                "experiments_happened": True,
                "training_happened": True,
                "replay_control_happened": True,
                "downloads_happened": False,
                "openvla_oft_happened": False,
                "eligible_demos_used": [f"{case.get('task_name')}::{case.get('demo_name')}" for case in eligible_cases],
                "action_range_before": prev_valid.get("action_low_high"),
                "action_range_after": fixed_valid.get("action_low_high"),
                "clip_rate_before": prev_valid.get("clip_rate_step"),
                "clip_rate_after": fixed_valid.get("clip_rate_step"),
                "controller_valid_before": prev_valid.get("controller_valid_rate_proxy"),
                "controller_valid_after": fixed_valid.get("controller_valid_rate_proxy"),
                "gripper_handling_before": ((baselines.get("previous_unfixed_adapter") or {}).get("normalization") or {}),
                "gripper_handling_after": ((baselines.get("range_fixed_smolvla_7d_adapter") or {}).get("gripper_handling") or {}),
                "offline_metrics_before": ((baselines.get("previous_unfixed_adapter") or {}).get("eval_metrics") or {}),
                "offline_metrics_after": ((baselines.get("range_fixed_smolvla_7d_adapter") or {}).get("eval_metrics") or {}),
                "offline_baseline_comparison": {
                    name: {
                        "action_l2": (((baselines.get(name) or {}).get("eval_metrics") or {}).get("action_l2")),
                        "translation_l2": (((baselines.get(name) or {}).get("eval_metrics") or {}).get("translation_l2")),
                        "rotation_l2": (((baselines.get(name) or {}).get("eval_metrics") or {}).get("rotation_l2")),
                        "gripper_error": (((baselines.get(name) or {}).get("eval_metrics") or {}).get("gripper_error")),
                        "gripper_accuracy": (((baselines.get(name) or {}).get("eval_metrics") or {}).get("gripper_accuracy")),
                        "clip_rate_step": (((baselines.get(name) or {}).get("action_validity") or {}).get("clip_rate_step")),
                        "controller_valid_rate_proxy": (((baselines.get(name) or {}).get("action_validity") or {}).get("controller_valid_rate_proxy")),
                        "train_eval_gap": (baselines.get(name) or {}).get("train_eval_gap"),
                    }
                    for name in [
                        "mean_action",
                        "ridge",
                        "small_mlp",
                        "previous_unfixed_adapter",
                        "previous_unfixed_adapter_clip_only",
                        "train_split_affine_range_calibrated_adapter_diagnostic",
                        "range_fixed_smolvla_7d_adapter",
                    ]
                },
                "replay_metrics_before": aggregate.get("previous_unfixed_adapter"),
                "replay_metrics_after": aggregate.get("range_fixed_smolvla_7d_adapter"),
                "simple_baseline_comparison": {
                    "mean_action": aggregate.get("mean_action"),
                    "ridge": aggregate.get("ridge"),
                    "small_mlp": aggregate.get("small_mlp"),
                    "clip_only": aggregate.get("previous_unfixed_adapter_clip_only"),
                    "affine_diagnostic_offline": ((baselines.get("train_split_affine_range_calibrated_adapter_diagnostic") or {}).get("eval_metrics") or {}),
                },
                "range_fixed_offline_action_l2": ((baselines.get("range_fixed_smolvla_7d_adapter") or {}).get("eval_metrics") or {}).get("action_l2"),
                "previous_offline_action_l2": ((baselines.get("previous_unfixed_adapter") or {}).get("eval_metrics") or {}).get("action_l2"),
                "range_fixed_replay_progress": (aggregate.get("range_fixed_smolvla_7d_adapter") or {}).get("progress_proxy_mean"),
                "previous_replay_progress": (aggregate.get("previous_unfixed_adapter") or {}).get("progress_proxy_mean"),
                "clip_only_replay_progress": (aggregate.get("previous_unfixed_adapter_clip_only") or {}).get("progress_proxy_mean"),
                "vram_peak_mb": offline.get("vram_peak_mb"),
            }
        )
        return finish(decision, next_step, 0)
    except Exception as exc:  # noqa: BLE001
        report["error"] = _compact_error(exc)
        return finish("TOO_HEAVY_LOCAL", "Fix the reported action range runner error and rerun.", 11)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="C:/assets/data/libero/libero_10")
    parser.add_argument("--libero-root", default="C:/assets/repos/LIBERO")
    parser.add_argument("--robosuite-root", default="C:/assets/repos/robosuite")
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--adapter-dir", default="runs/smolvla_7d_standard_replay_baseline")
    parser.add_argument("--exact-init-report-path", default="reports/exact_init_expert_replay_stabilization.json")
    parser.add_argument("--output-dir", default="runs/smolvla_7d_action_range_fix")
    parser.add_argument("--report-path", default="reports/smolvla_7d_action_range_fix.json")
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--train-demos-per-task", type=int, default=5)
    parser.add_argument("--eval-demos-per-task", type=int, default=2)
    parser.add_argument("--records-per-demo", type=int, default=8)
    parser.add_argument("--adapter-steps", type=int, default=800)
    parser.add_argument("--adapter-hidden-dim", type=int, default=128)
    parser.add_argument("--mlp-steps", type=int, default=800)
    parser.add_argument("--mlp-hidden-dim", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-replay-steps", type=int, default=320)
    parser.add_argument("--post-signal-margin", type=int, default=16)
    parser.add_argument("--camera-size", type=int, default=64)
    args = parser.parse_args(argv)
    report, code = build_report(args)
    _write_json(Path(args.report_path), report)
    if code == 0:
        _write_reports(report)
    print(json.dumps({"decision": report.get("decision"), "summary": report.get("summary"), "error": report.get("error")}, indent=2, sort_keys=True))
    return int(code)


if __name__ == "__main__":
    raise SystemExit(main())
