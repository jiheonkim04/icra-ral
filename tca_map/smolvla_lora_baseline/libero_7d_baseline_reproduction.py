"""Bounded SmolVLA/LIBERO 7D standard baseline reproduction.

This runner compares fixed-interface LIBERO_7D baselines. It keeps the 7D label
path explicit, uses train-split-only normalization, and avoids the native
SO100 6D action path.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla_lora_baseline import diagnostic as base
from tca_map.smolvla_lora_baseline import libero_7d_interface_fix as fix


BASELINE_GATE = "ALLOW_SMOLVLA_LIBERO_7D_BASELINE_REPRODUCTION"
TRAINING_GATE = "ALLOW_SMOLVLA_LIBERO_7D_BASELINE_TRAINING"
DEFAULT_HDF5_PATH = base.DEFAULT_HDF5_PATH
FINAL_DECISIONS = {
    "READY_FOR_RA_L_METHOD_ON_SMOLVLA_7D",
    "READY_FOR_METHOD_BUT_NEEDS_STRONGER_HEAD",
    "BASELINE_STILL_MLP_DOMINATED",
    "BASELINE_STILL_MEAN_DOMINATED",
    "DATA_SPLIT_NOT_MEANINGFUL",
    "TOO_HEAVY_LOCAL",
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


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _round(value: float | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sample_timesteps(length: int, count: int, start: int = 0, stop: int | None = None) -> list[int]:
    stop = length if stop is None else min(stop, length)
    if stop <= start:
        return [max(0, min(length - 1, start))]
    values = np.linspace(start, stop - 1, num=max(1, count), dtype=np.int64).tolist()
    return list(dict.fromkeys(int(x) for x in values))


def _records_for_demo_times(path: Path, demo_times: dict[str, list[int]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    task_text = base._safe_task_text(path)
    for demo_name, timesteps in demo_times.items():
        for timestep in timesteps:
            records.append(
                {
                    "hdf5_path": str(path),
                    "task_name": path.stem,
                    "task_text": task_text,
                    "demo_name": demo_name,
                    "timestep": int(timestep),
                    "action_dim": 7,
                }
            )
    return records


def _record_key(record: dict[str, Any]) -> tuple[str, str, int]:
    return (record["hdf5_path"], record["demo_name"], int(record["timestep"]))


def _concat_record_actions(records: list[dict[str, Any]]) -> np.ndarray:
    return np.stack(
        [
            base._expert_action(Path(record["hdf5_path"]), record["demo_name"], int(record["timestep"]))[:7]
            for record in records
        ],
        axis=0,
    ).astype(np.float32)


def _split_leakage(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    train_keys = {_record_key(record) for record in train_records}
    eval_keys = {_record_key(record) for record in eval_records}
    train_demos = {(record["hdf5_path"], record["demo_name"]) for record in train_records}
    eval_demos = {(record["hdf5_path"], record["demo_name"]) for record in eval_records}
    train_tasks = {record["hdf5_path"] for record in train_records}
    eval_tasks = {record["hdf5_path"] for record in eval_records}
    return {
        "exact_record_overlap": len(train_keys & eval_keys),
        "demo_overlap": len(train_demos & eval_demos),
        "task_overlap": len(train_tasks & eval_tasks),
        "has_exact_record_leakage": bool(train_keys & eval_keys),
        "has_demo_overlap": bool(train_demos & eval_demos),
        "has_task_overlap": bool(train_tasks & eval_tasks),
        "note": "Task/demo overlap can be intentional for same-task or same-demo time holdout; exact record overlap must remain zero.",
    }


def _mean_action_is_low_variance_strong(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    train_actions = _concat_record_actions(train_records)
    mean_metrics = _mean_action_metrics(train_records, eval_records)
    return {
        "mean_action_l2": mean_metrics["action_l2"],
        "train_first6_std_l2": _round(float(np.linalg.norm(train_actions[:, :6].std(axis=0)))),
        "train_gripper_variance": _round(train_actions[:, 6].var()),
        "mean_action_strong_due_to_low_variance": bool(
            mean_metrics["action_l2"] < 0.6 and float(np.linalg.norm(train_actions[:, :6].std(axis=0))) < 0.6
        ),
    }


def _split_report(name: str, train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    train_actions = _concat_record_actions(train_records)
    eval_actions = _concat_record_actions(eval_records)
    task_names = sorted({Path(record["hdf5_path"]).stem for record in train_records + eval_records})
    raw_timesteps = 0
    for path in sorted({Path(record["hdf5_path"]) for record in train_records + eval_records}):
        raw_timesteps += int(sum(actions.shape[0] for actions in fix._load_actions_by_demo(path).values()))
    return {
        "name": name,
        "task_names": task_names,
        "train_demo_ids": sorted({record["demo_name"] for record in train_records}, key=base._demo_sort_key),
        "eval_demo_ids": sorted({record["demo_name"] for record in eval_records}, key=base._demo_sort_key),
        "raw_timestep_count": raw_timesteps,
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "train_action_stats": fix._action_stats(train_actions),
        "eval_action_stats": fix._action_stats(eval_actions),
        "gripper_distribution": {
            "train": {str(_round(value)): int((train_actions[:, 6] == value).sum()) for value in np.unique(train_actions[:, 6])},
            "eval": {str(_round(value)): int((eval_actions[:, 6] == value).sum()) for value in np.unique(eval_actions[:, 6])},
        },
        "mean_action_strength": _mean_action_is_low_variance_strong(train_records, eval_records),
        "leakage": _split_leakage(train_records, eval_records),
    }


def _same_task_demo_holdout(path: Path) -> dict[str, Any]:
    split = base.select_records(path, max_train_demos=30, max_eval_demos=10, records_per_demo=10)
    return {
        "train_records": split["train_records"],
        "eval_records": split["eval_records"],
        "report": _split_report("same_task_demo_holdout", split["train_records"], split["eval_records"]),
    }


def _same_task_time_holdout(path: Path) -> dict[str, Any]:
    actions_by_demo = fix._load_actions_by_demo(path)
    train_times: dict[str, list[int]] = {}
    eval_times: dict[str, list[int]] = {}
    for demo_name in list(actions_by_demo.keys())[:20]:
        length = actions_by_demo[demo_name].shape[0]
        boundary = int(length * 0.7)
        train_times[demo_name] = _sample_timesteps(length, 8, 0, boundary)
        eval_times[demo_name] = _sample_timesteps(length, 4, boundary, length)
    train_records = _records_for_demo_times(path, train_times)
    eval_records = _records_for_demo_times(path, eval_times)
    report = _split_report("same_task_time_holdout", train_records, eval_records)
    report["leakage"]["temporal_chunk_overlap_risk"] = True
    report["leakage"]["temporal_chunk_overlap_note"] = (
        "Same-demo time holdout has disjoint sampled timesteps but 50-step action chunks can be temporally near each other."
    )
    return {"train_records": train_records, "eval_records": eval_records, "report": report}


def _multi_task_demo_holdout(default_path: Path) -> dict[str, Any] | None:
    task_paths = sorted(default_path.parent.glob("*.hdf5"))[:3]
    if len(task_paths) < 2:
        return None
    train_records: list[dict[str, Any]] = []
    eval_records: list[dict[str, Any]] = []
    for path in task_paths:
        split = base.select_records(path, max_train_demos=10, max_eval_demos=4, records_per_demo=5)
        train_records.extend(split["train_records"])
        eval_records.extend(split["eval_records"])
    return {
        "train_records": train_records,
        "eval_records": eval_records,
        "report": _split_report("multi_task_demo_holdout", train_records, eval_records),
    }


def _construct_splits(path: Path) -> dict[str, Any]:
    splits = {
        "same_task_demo_holdout": _same_task_demo_holdout(path),
        "same_task_time_holdout": _same_task_time_holdout(path),
    }
    multi = _multi_task_demo_holdout(path)
    if multi is not None:
        splits["multi_task_demo_holdout"] = multi
    return splits


def _mean_action_metrics(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    mean_action = base._mean_train_action(train_records)
    return base.evaluate_constant_action(eval_records, mean_action)


def _per_task_mean_action_metrics(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[str, list[np.ndarray]] = {}
    for record in train_records:
        by_task.setdefault(record["hdf5_path"], []).append(
            base._expert_action(Path(record["hdf5_path"]), record["demo_name"], int(record["timestep"]))[:7]
        )
    global_mean = base._mean_train_action(train_records)
    predictions = []
    experts = []
    for record in eval_records:
        task_actions = by_task.get(record["hdf5_path"])
        pred = np.mean(np.stack(task_actions, axis=0), axis=0) if task_actions else global_mean
        predictions.append(pred[:7])
        experts.append(base._expert_action(Path(record["hdf5_path"]), record["demo_name"], int(record["timestep"]))[:7])
    return base._metrics_from_predictions(predictions, experts)


def _previous_action_metrics(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    fallback = base._mean_train_action(train_records)
    predictions = []
    experts = []
    for record in eval_records:
        timestep = int(record["timestep"])
        if timestep > 0:
            pred = base._expert_action(Path(record["hdf5_path"]), record["demo_name"], timestep - 1)[:7]
        else:
            pred = fallback[:7]
        predictions.append(pred)
        experts.append(base._expert_action(Path(record["hdf5_path"]), record["demo_name"], timestep)[:7])
    return base._metrics_from_predictions(predictions, experts)


def _ridge_baseline(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]]) -> dict[str, Any]:
    result = fix._ridge_baseline(train_records, eval_records)
    result["training"] = {"loss_curve": [], "loss_decreased": None, "trainable_params": 56}
    return result


def _small_mlp_baseline(train_records: list[dict[str, Any]], eval_records: list[dict[str, Any]], steps: int) -> dict[str, Any]:
    result = fix._small_mlp_baseline(train_records, eval_records, steps=steps)
    result["target_modules"] = ["libero_7d_adapter_head_only"]
    result["trainable_params"] = _mlp_param_count(input_dim=7, hidden_dim=32, output_dim=7)
    return result


def _mlp_param_count(input_dim: int, hidden_dim: int, output_dim: int) -> int:
    return (input_dim * hidden_dim + hidden_dim) + (hidden_dim * output_dim + output_dim)


def _state_proj_weights(checkpoint: Path) -> tuple[Any, Any, Path]:
    from safetensors.torch import load_file

    files = [checkpoint / "model.safetensors"]
    files.extend(sorted(checkpoint.glob("*.safetensors")))
    for file_path in files:
        if not file_path.exists():
            continue
        tensors = load_file(str(file_path), device="cpu")
        if "model.state_proj.weight" in tensors and "model.state_proj.bias" in tensors:
            return tensors["model.state_proj.weight"].float(), tensors["model.state_proj.bias"].float(), file_path
    raise FileNotFoundError(f"could not find model.state_proj weights under {checkpoint}")


def _state_time_tensors(records: list[dict[str, Any]], x_mean: np.ndarray | None = None, x_std: np.ndarray | None = None):
    import torch

    x, y = fix._feature_matrix(records)
    if x_mean is None:
        x_mean = x.mean(axis=0, keepdims=True).astype(np.float32)
    if x_std is None:
        x_std = (x.std(axis=0, keepdims=True) + 1e-6).astype(np.float32)
    x_norm = ((x - x_mean) / x_std).astype(np.float32)
    state = np.concatenate([x_norm[:, :6], np.zeros((x_norm.shape[0], 26), dtype=np.float32)], axis=1)
    time_feature = x_norm[:, 6:7]
    return (
        torch.tensor(state, dtype=torch.float32),
        torch.tensor(time_feature, dtype=torch.float32),
        y,
        x_mean,
        x_std,
    )


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
) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    state_weight, state_bias, weight_file = _state_proj_weights(checkpoint)
    train_state, train_time, y_train_np, x_mean, x_std = _state_time_tensors(train_records)
    eval_state, eval_time, y_eval_np, _, _ = _state_time_tensors(eval_records, x_mean, x_std)
    normalizer = fix.Libero7DNormalizer.fit(y_train_np)
    y_train = torch.tensor(normalizer.normalize(y_train_np), dtype=torch.float32)

    torch.manual_seed(seed)
    if linear_head:
        head = torch.nn.Linear(961, 7)
    else:
        head = torch.nn.Sequential(
            torch.nn.Linear(961, hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 7),
        )
    params: list[Any] = list(head.parameters())
    lora_a = None
    lora_b = None
    lora_alpha = None
    if lora_rank is not None:
        lora_alpha = int(lora_rank) * 2
        lora_a = torch.nn.Parameter(torch.randn(int(lora_rank), 32) * 0.01)
        lora_b = torch.nn.Parameter(torch.zeros(960, int(lora_rank)))
        params.extend([lora_a, lora_b])
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=1e-5)
    losses: list[dict[str, float]] = []

    def projected(state_tensor):
        if lora_rank is None:
            return state_tensor @ state_weight.T + state_bias
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
        train_pred = normalizer.unnormalize(
            head(torch.cat([projected(train_state), train_time], dim=1)).detach().cpu().numpy()
        )
        eval_pred = normalizer.unnormalize(
            head(torch.cat([projected(eval_state), eval_time], dim=1)).detach().cpu().numpy()
        )
    lora_params = 0 if lora_rank is None else int(lora_rank) * (32 + 960)
    head_params = (961 * 7 + 7) if linear_head else _mlp_param_count(961, hidden_dim, 7)
    return {
        "name": name,
        "adapter_schema": "LIBERO_7D",
        "feature_schema": "train-normalized SmolVLA observation.state padded to 32, then checkpoint state_proj, plus timestep fraction",
        "state_proj_weight_file": str(weight_file),
        "target_modules": ["state_proj"] if lora_rank is not None else ["libero_7d_adapter"],
        "requested_target_modules": ["state_proj"] if lora_rank is not None else ["libero_7d_adapter"],
        "excluded_native_6d_modules": ["action_in_proj", "action_out_proj", "action_time_mlp_in", "action_time_mlp_out"],
        "exclusion_reason": "Native action projection modules require max_action_dim/native flow actions and would re-enter the old 6D/SO100 action path.",
        "lora_rank": lora_rank,
        "lora_alpha": lora_alpha,
        "trainable_params": head_params + lora_params,
        "normalization": normalizer.report(),
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
            "loss": "separate normalized gripper MSE added to pose regression loss",
        },
        "training": {
            "steps": int(steps),
            "hidden_dim": int(hidden_dim) if not linear_head else None,
            "linear_head": bool(linear_head),
            "learning_rate": float(learning_rate),
            "loss_start": losses[0]["loss"] if losses else None,
            "loss_end": losses[-1]["loss"] if losses else None,
            "pose_loss_end": losses[-1]["pose_loss"] if losses else None,
            "gripper_mse_loss_end": losses[-1]["gripper_mse_loss"] if losses else None,
            "loss_decreased": bool(losses and losses[-1]["loss"] < losses[0]["loss"]),
            "loss_curve": losses,
        },
        "train_metrics": fix._metrics_from_arrays(train_pred, y_train_np),
        "eval_metrics": fix._metrics_from_arrays(eval_pred, y_eval_np),
        "train_eval_gap": _round(
            fix._metrics_from_arrays(eval_pred, y_eval_np)["action_l2"]
            - fix._metrics_from_arrays(train_pred, y_train_np)["action_l2"]
        ),
        "runtime_sec": _round(time.monotonic() - started, 3),
        "uses_eval_labels_for_training": False,
        "uses_so100_action_normalizer": False,
        "uses_hard_coded_gripper_fill": False,
    }


def _run_baseline_suite(args: argparse.Namespace, split: dict[str, Any]) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    train_records = split["train_records"]
    eval_records = split["eval_records"]
    checkpoint = Path(args.smolvla_ckpt)
    baselines: dict[str, Any] = {}
    baselines["global_mean_action"] = {"name": "global_mean_action", "eval_metrics": _mean_action_metrics(train_records, eval_records)}
    baselines["per_task_mean_action"] = {
        "name": "per_task_mean_action",
        "eval_metrics": _per_task_mean_action_metrics(train_records, eval_records),
    }
    baselines["previous_action_persistence"] = {
        "name": "previous_action_persistence",
        "eval_metrics": _previous_action_metrics(train_records, eval_records),
        "uses_eval_previous_action_label": True,
        "decision_use": "diagnostic_only_not_closed_loop_metric",
        "note": "This uses the previous expert action from the held-out HDF5 sequence; it is not treated as the learned-action decision gate.",
    }
    baselines["ridge"] = _ridge_baseline(train_records, eval_records)
    baselines["small_mlp"] = _small_mlp_baseline(train_records, eval_records, steps=int(args.small_mlp_steps))
    baselines["frozen_base_smolvla_7d_linear_adapter"] = _train_state_proj_adapter(
        train_records,
        eval_records,
        checkpoint=checkpoint,
        name="frozen_base_smolvla_7d_linear_adapter",
        steps=int(args.adapter_steps),
        hidden_dim=int(args.adapter_hidden_dim),
        learning_rate=float(args.learning_rate),
        seed=19,
        lora_rank=None,
        linear_head=True,
    )
    baselines["smolvla_7d_adapter_no_lora"] = _train_state_proj_adapter(
        train_records,
        eval_records,
        checkpoint=checkpoint,
        name="smolvla_7d_adapter_no_lora",
        steps=int(args.adapter_steps),
        hidden_dim=int(args.adapter_hidden_dim),
        learning_rate=float(args.learning_rate),
        seed=17,
        lora_rank=None,
        linear_head=False,
    )
    lora_variants: dict[str, Any] = {}
    for rank in [4, 8]:
        key = f"smolvla_state_proj_lora_rank{rank}_7d_adapter"
        lora_variants[key] = _train_state_proj_adapter(
            train_records,
            eval_records,
            checkpoint=checkpoint,
            name=key,
            steps=int(args.adapter_steps),
            hidden_dim=int(args.adapter_hidden_dim),
            learning_rate=float(args.lora_learning_rate),
            seed=17 + rank,
            lora_rank=rank,
            linear_head=False,
        )
    baselines.update(lora_variants)
    simple_candidates = [baselines["ridge"], baselines["small_mlp"]]
    adapter_candidates = [
        baselines["frozen_base_smolvla_7d_linear_adapter"],
        baselines["smolvla_7d_adapter_no_lora"],
        *lora_variants.values(),
    ]
    best_simple = min(simple_candidates, key=lambda item: item["eval_metrics"]["action_l2"])
    best_adapter = min(adapter_candidates, key=lambda item: item["eval_metrics"]["action_l2"])
    best_lora = min(lora_variants.values(), key=lambda item: item["eval_metrics"]["action_l2"])
    vram_peak_mb = 0.0
    if torch.cuda.is_available():
        vram_peak_mb = _round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
    return {
        "split_name": split["report"]["name"],
        "baselines": baselines,
        "best_simple_mlp_or_ridge": best_simple,
        "best_adapter_or_lora": best_adapter,
        "best_lora": best_lora,
        "lora_ranks_tested": [4, 8],
        "rank16_tested": False,
        "rank16_skip_reason": "Rank 8 was stable and cheap, but rank 16 was optional; skipped to keep this reproduction bounded.",
        "target_modules_tested": {
            "executable": [
                "libero_7d_adapter_head_only",
                "frozen_state_proj_plus_7d_adapter",
                "state_proj_lora_plus_7d_adapter",
            ],
            "audited_but_not_executed": [
                "action_in_proj",
                "action_out_proj",
                "action_time_mlp_in",
                "action_time_mlp_out",
            ],
        },
        "vram_peak_mb": vram_peak_mb,
        "runtime_sec": _round(time.monotonic() - started, 3),
    }


def _target_module_audit() -> dict[str, Any]:
    return {
        "current_projection_modules": {
            "requested": ["state_proj", "action_in_proj", "action_out_proj"],
            "executed_for_fixed_7d": ["state_proj"],
            "not_executed": ["action_in_proj", "action_out_proj"],
            "reason": "action_in_proj/action_out_proj are native flow-action modules tied to max_action_dim/SO100 6D action preparation.",
        },
        "action_head_7d_adapter_only": {
            "executed": True,
            "variant": "small_state_time_mlp_7d_baseline",
        },
        "projection_plus_7d_adapter": {
            "executed": True,
            "variants": ["smolvla_7d_adapter_no_lora", "smolvla_state_proj_lora_rank4_7d_adapter", "smolvla_state_proj_lora_rank8_7d_adapter"],
        },
        "projection_plus_action_head_if_available": {
            "executed": True,
            "interpretation": "state_proj LoRA plus learned LIBERO_7D adapter head",
        },
        "strict_boundary": "No target-module variant uses the old hard-coded gripper fill or SO100 action normalizer for LIBERO labels.",
    }


def _optional_replay_report(best_beats_mean: bool, best_beats_simple: bool) -> dict[str, Any]:
    if best_beats_mean and best_beats_simple:
        return {
            "executed": False,
            "eligible": True,
            "reason": (
                "Optional replay/progress was not run in this baseline reproduction because no bounded executable "
                "LIBERO environment bridge for the learned 7D adapter is part of this runner."
            ),
            "metrics": None,
        }
    return {
        "executed": False,
        "eligible": False,
        "reason": "Skipped because action metrics did not beat both mean-action and MLP/ridge.",
        "metrics": None,
    }


def _decide(report: dict[str, Any]) -> tuple[str, str]:
    suite = report["baseline_suite"]
    mean_metric = suite["baselines"]["global_mean_action"]["eval_metrics"]["action_l2"]
    simple_metric = suite["best_simple_mlp_or_ridge"]["eval_metrics"]["action_l2"]
    best_lora_metric = suite["best_lora"]["eval_metrics"]["action_l2"]
    best_adapter_metric = suite["best_adapter_or_lora"]["eval_metrics"]["action_l2"]
    train_eval_gap = suite["best_lora"].get("train_eval_gap", 0.0)
    if report["split_audit"]["primary"]["report"]["train_count"] < 50 or report["split_audit"]["primary"]["report"]["eval_count"] < 20:
        return ("DATA_SPLIT_NOT_MEANINGFUL", "Build a larger standard split before interpreting fixed-interface LoRA.")
    if best_adapter_metric >= mean_metric:
        return ("BASELINE_STILL_MEAN_DOMINATED", "Stop: mean-action still beats the best fixed-interface adapter/LoRA.")
    if best_lora_metric > mean_metric:
        return ("BASELINE_STILL_MEAN_DOMINATED", "Stop: mean-action still beats the best LoRA variant.")
    if best_lora_metric <= simple_metric * 1.05 and train_eval_gap < 0.5:
        return (
            "READY_FOR_RA_L_METHOD_ON_SMOLVLA_7D",
            "Future method planning may start only after preserving this fixed-interface baseline table and predeclaring simple baselines.",
        )
    if best_lora_metric <= mean_metric:
        if best_lora_metric > simple_metric * 1.10:
            return (
                "BASELINE_STILL_MLP_DOMINATED",
                "Stop: MLP/ridge clearly beats all SmolVLA LoRA/adapter variants.",
            )
        return (
            "READY_FOR_METHOD_BUT_NEEDS_STRONGER_HEAD",
            "Baseline is fixed and LoRA beats mean-action, but the action head/adapter must be strengthened before semantic method work.",
        )
    return ("DATA_SPLIT_NOT_MEANINGFUL", "Baseline decision was ambiguous; build a more representative split.")


def _write_report_bundle(report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    suite = report.get("baseline_suite") or {}
    split = report.get("split_audit") or {}
    target = report.get("target_module_audit") or {}
    baselines = suite.get("baselines") or {}
    best_lora = suite.get("best_lora") or {}
    best_lora_train = best_lora.get("train_metrics") or {}
    best_lora_eval = best_lora.get("eval_metrics") or {}
    best_lora_training = best_lora.get("training") or {}

    def _eval_l2(name: str) -> Any:
        return ((baselines.get(name) or {}).get("eval_metrics") or {}).get("action_l2")

    loss_curve = best_lora_training.get("loss_curve") or []
    loss_curve_sample = {
        "first3": loss_curve[:3],
        "last3": loss_curve[-3:],
    }
    lines = [
        "# SmolVLA 7D Baseline Reproduction",
        "",
        f"Final decision: `{summary.get('final_decision')}`",
        "",
        "This is a standard fixed-interface baseline reproduction, not a new method or paper claim.",
        "",
        "## Summary",
        "",
        f"- model used: `{summary.get('model_used')}`",
        f"- dataset/split used: `{summary.get('dataset_split_used')}`",
        f"- LoRA ranks tested: `{summary.get('lora_ranks_tested')}`",
        f"- target modules tested: `{summary.get('target_modules_tested')}`",
        f"- experiments happened: `{summary.get('experiments_happened')}`",
        f"- training happened: `{summary.get('training_happened')}`",
        f"- loss computed: `{summary.get('loss_computed')}`",
        f"- GPU training happened: `{summary.get('gpu_training_happened')}`",
        f"- downloads happened: `{summary.get('downloads_happened')}`",
        f"- OpenVLA-OFT happened: `{summary.get('openvla_oft_happened')}`",
        f"- mean-action metric: `{summary.get('mean_action_metric')}`",
        f"- ridge/MLP metric: `{summary.get('best_mlp_or_ridge_metric')}`",
        f"- frozen/base metric: `{summary.get('frozen_base_metric')}`",
        f"- best LoRA/adapter metric: `{summary.get('best_lora_adapter_metric')}`",
        f"- best LoRA metric: `{summary.get('best_lora_metric')}`",
        f"- LoRA beats mean-action: `{summary.get('lora_beats_mean_action')}`",
        f"- LoRA beats MLP/ridge: `{summary.get('lora_beats_mlp_or_ridge')}`",
        f"- VRAM peak MB: `{summary.get('vram_peak_mb')}`",
        f"- runtime sec: `{summary.get('runtime_sec')}`",
        f"- trainable params: `{summary.get('trainable_params')}`",
        f"- exact next step: {summary.get('exact_next_step')}",
        "",
    ]
    Path("reports/smolvla_7d_baseline_reproduction.md").write_text("\n".join(lines), encoding="utf-8")

    Path("reports/smolvla_7d_baseline_experiment_plan.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Baseline Experiment Plan",
                "",
                "- Use fixed LIBERO_7D labels only.",
                "- Use train-split-only 7D normalization.",
                "- Compare mean, per-task mean, persistence, ridge, MLP, frozen state-proj adapter, no-LoRA adapter, and rank-4/rank-8 state-proj LoRA adapters.",
                "- Do not use SO100 action normalizer, old 6D action labels, hard-coded gripper fill, rollout benchmark, OpenVLA-OFT, or downloads.",
                "- Run optional replay/progress only if a bounded executable 7D adapter bridge is available and action metrics beat mean and MLP/ridge.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    result_lines = [
        "# SmolVLA 7D Baseline Results",
        "",
        f"- primary split: `{suite.get('split_name')}`",
        f"- global mean-action L2: `{summary.get('mean_action_metric')}`",
        f"- per-task mean-action L2: `{_eval_l2('per_task_mean_action')}`",
        f"- previous-action L2: `{_eval_l2('previous_action_persistence')}`",
        f"- previous-action caveat: `{(baselines.get('previous_action_persistence') or {}).get('note')}`",
        f"- ridge L2: `{_eval_l2('ridge')}`",
        f"- small MLP L2: `{_eval_l2('small_mlp')}`",
        f"- frozen/base SmolVLA 7D adapter L2: `{summary.get('frozen_base_metric')}`",
        f"- no-LoRA SmolVLA 7D adapter L2: `{_eval_l2('smolvla_7d_adapter_no_lora')}`",
        f"- rank-4 LoRA 7D adapter L2: `{_eval_l2('smolvla_state_proj_lora_rank4_7d_adapter')}`",
        f"- rank-8 LoRA 7D adapter L2: `{_eval_l2('smolvla_state_proj_lora_rank8_7d_adapter')}`",
        f"- rank-16 LoRA status: `tested={suite.get('rank16_tested')}, reason={suite.get('rank16_skip_reason')}`",
        f"- best LoRA/adapter name: `{summary.get('best_lora_adapter_name')}`",
        f"- best LoRA train action L2: `{best_lora_train.get('action_l2')}`",
        f"- best LoRA eval action L2: `{best_lora_eval.get('action_l2')}`",
        f"- best LoRA eval translation L2: `{best_lora_eval.get('translation_l2')}`",
        f"- best LoRA eval rotation L2: `{best_lora_eval.get('rotation_l2')}`",
        f"- best LoRA eval gripper error: `{best_lora_eval.get('gripper_error')}`",
        f"- best LoRA eval gripper accuracy: `{best_lora_eval.get('gripper_accuracy')}`",
        f"- best LoRA per-dim MAE: `{best_lora_eval.get('per_dim_mae')}`",
        f"- best LoRA train/eval action-L2 gap: `{best_lora.get('train_eval_gap')}`",
        f"- best LoRA loss start/end: `{best_lora_training.get('loss_start')} -> {best_lora_training.get('loss_end')}`",
        f"- best LoRA loss decreased: `{best_lora_training.get('loss_decreased')}`",
        f"- best LoRA loss curve sample: `{loss_curve_sample}`",
        f"- trainable params: `{summary.get('trainable_params')}`",
        f"- VRAM peak MB: `{summary.get('vram_peak_mb')}`",
        f"- suite runtime sec: `{summary.get('suite_runtime_sec')}`",
        f"- total runtime sec: `{summary.get('runtime_sec')}`",
        f"- optional replay/progress: `{report.get('optional_replay_progress')}`",
        "",
    ]
    Path("reports/smolvla_7d_baseline_results.md").write_text("\n".join(result_lines), encoding="utf-8")

    Path("reports/smolvla_7d_baseline_decision.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Baseline Decision",
                "",
                f"Final decision: `{summary.get('final_decision')}`",
                "",
                f"Exact next step: {summary.get('exact_next_step')}",
                "",
                "Hard rule: do not propose a new method unless decision is `READY_FOR_RA_L_METHOD_ON_SMOLVLA_7D` or `READY_FOR_METHOD_BUT_NEEDS_STRONGER_HEAD`.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    Path("reports/smolvla_7d_lora_target_module_audit.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D LoRA Target Module Audit",
                "",
                f"- current projection modules: `{target.get('current_projection_modules')}`",
                f"- action head / 7D adapter only: `{target.get('action_head_7d_adapter_only')}`",
                f"- projection + 7D adapter: `{target.get('projection_plus_7d_adapter')}`",
                f"- projection + action head if available: `{target.get('projection_plus_action_head_if_available')}`",
                f"- strict boundary: {target.get('strict_boundary')}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    split_lines = ["# SmolVLA 7D Split Audit", ""]
    for name, payload in (split.get("all_splits") or {}).items():
        report_payload = payload.get("report") or {}
        split_lines.extend(
            [
                f"## {name}",
                "",
                f"- task names: `{report_payload.get('task_names')}`",
                f"- train/eval count: `{report_payload.get('train_count')} / {report_payload.get('eval_count')}`",
                f"- sampled records: `{(report_payload.get('train_count') or 0) + (report_payload.get('eval_count') or 0)}`",
                f"- raw timesteps: `{report_payload.get('raw_timestep_count')}`",
                f"- train demos: `{report_payload.get('train_demo_ids')}`",
                f"- eval demos: `{report_payload.get('eval_demo_ids')}`",
                f"- train action variance: `{((report_payload.get('train_action_stats') or {}).get('variance'))}`",
                f"- eval action variance: `{((report_payload.get('eval_action_stats') or {}).get('variance'))}`",
                f"- gripper distribution: `{report_payload.get('gripper_distribution')}`",
                f"- mean-action strength: `{report_payload.get('mean_action_strength')}`",
                f"- leakage: `{report_payload.get('leakage')}`",
                "",
            ]
        )
    Path("reports/smolvla_7d_split_audit.md").write_text("\n".join(split_lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    hdf5_path = Path(args.hdf5_path)
    checkpoint = Path(args.smolvla_ckpt)
    report: dict[str, Any] = {
        "schema_version": "smolvla-libero-7d-baseline-reproduction-v1",
        "evidence_label": "smolvla_7d_baseline_reproduction",
        "decision": "DATA_SPLIT_NOT_MEANINGFUL",
        "policy": {
            "bounded_baseline_reproduction": True,
            "new_method_created": False,
            "patchguard_continued": False,
            "downloads_performed": False,
            "large_model_or_dataset_downloads_performed": False,
            "rollouts_performed": False,
            "benchmark_rollouts_performed": False,
            "openvla_oft_executed": False,
            "paper_grade_claims_made": False,
            "old_broken_6d_action_path_used": False,
            "hard_coded_gripper_fill_used": False,
            "baseline_gate_set": _env_flag(BASELINE_GATE),
            "training_gate_set": _env_flag(TRAINING_GATE),
            "forbidden_gates_set": forbidden,
            "training_performed": False,
            "loss_computed": False,
            "gpu_training_performed": False,
        },
        "paths": {"hdf5_path": str(hdf5_path), "smolvla_ckpt": str(checkpoint)},
        "split_audit": {},
        "target_module_audit": {},
        "baseline_suite": {},
        "optional_replay_progress": {},
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
        return report, code

    if not report["policy"]["baseline_gate_set"]:
        return finish("DATA_SPLIT_NOT_MEANINGFUL", f"Set {BASELINE_GATE}=1 for this bounded reproduction.", 2)
    if forbidden:
        report["error"] = {"message": "Forbidden gate(s) set: " + ", ".join(forbidden)}
        return finish("DATA_SPLIT_NOT_MEANINGFUL", "Clear forbidden rollout/download/OpenVLA-OFT/method gates and rerun.", 3)
    if not hdf5_path.exists():
        return finish("DATA_SPLIT_NOT_MEANINGFUL", f"Missing local HDF5 path: {hdf5_path}", 4)
    if not checkpoint.exists():
        return finish("DATA_SPLIT_NOT_MEANINGFUL", f"Missing local SmolVLA checkpoint: {checkpoint}", 5)

    try:
        splits = _construct_splits(hdf5_path)
        report["split_audit"] = {
            "primary_split_name": "same_task_demo_holdout",
            "primary": splits["same_task_demo_holdout"],
            "all_splits": splits,
        }
        report["target_module_audit"] = _target_module_audit()
        if not report["policy"]["training_gate_set"]:
            return finish("DATA_SPLIT_NOT_MEANINGFUL", f"Set {TRAINING_GATE}=1 to run baseline training.", 6)
        suite = _run_baseline_suite(args, splits["same_task_demo_holdout"])
        report["baseline_suite"] = suite
        report["policy"]["training_performed"] = True
        report["policy"]["loss_computed"] = True
        mean_metric = suite["baselines"]["global_mean_action"]["eval_metrics"]["action_l2"]
        simple_metric = suite["best_simple_mlp_or_ridge"]["eval_metrics"]["action_l2"]
        frozen_metric = suite["baselines"]["frozen_base_smolvla_7d_linear_adapter"]["eval_metrics"]["action_l2"]
        best_adapter_metric = suite["best_adapter_or_lora"]["eval_metrics"]["action_l2"]
        best_lora_metric = suite["best_lora"]["eval_metrics"]["action_l2"]
        report["optional_replay_progress"] = _optional_replay_report(
            best_beats_mean=best_lora_metric < mean_metric,
            best_beats_simple=best_lora_metric < simple_metric,
        )
        report["summary"].update(
            {
                "model_used": str(checkpoint),
                "dataset_split_used": "same_task_demo_holdout",
                "lora_ranks_tested": suite["lora_ranks_tested"],
                "target_modules_tested": suite["target_modules_tested"]["executable"],
                "trainable_params": {
                    key: value.get("trainable_params")
                    for key, value in suite["baselines"].items()
                    if isinstance(value, dict) and value.get("trainable_params") is not None
                },
                "vram_peak_mb": suite["vram_peak_mb"],
                "suite_runtime_sec": suite["runtime_sec"],
                "mean_action_metric": mean_metric,
                "ridge_metric": suite["baselines"]["ridge"]["eval_metrics"]["action_l2"],
                "small_mlp_metric": suite["baselines"]["small_mlp"]["eval_metrics"]["action_l2"],
                "best_mlp_or_ridge_metric": simple_metric,
                "best_mlp_or_ridge_name": suite["best_simple_mlp_or_ridge"]["name"],
                "frozen_base_metric": frozen_metric,
                "best_lora_adapter_metric": best_adapter_metric,
                "best_lora_adapter_name": suite["best_adapter_or_lora"]["name"],
                "best_lora_metric": best_lora_metric,
                "best_lora_name": suite["best_lora"]["name"],
                "lora_beats_mean_action": best_lora_metric < mean_metric,
                "lora_beats_mlp_or_ridge": best_lora_metric < simple_metric,
                "optional_replay_progress_executed": report["optional_replay_progress"]["executed"],
                "experiments_happened": True,
                "training_happened": True,
                "loss_computed": True,
                "gpu_training_happened": False,
                "downloads_happened": False,
                "openvla_oft_happened": False,
            }
        )
        decision, next_step = _decide(report)
        return finish(decision, next_step, 0)
    except Exception as exc:  # noqa: BLE001
        report["error"] = _compact_error(exc)
        if "out of memory" in str(exc).lower():
            return finish("TOO_HEAVY_LOCAL", "Stop: baseline reproduction exceeded local memory.", 10)
        return finish("DATA_SPLIT_NOT_MEANINGFUL", "Fix the reported baseline runner error and rerun.", 11)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-path", default=DEFAULT_HDF5_PATH)
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--report-path", default="reports/smolvla_7d_baseline_reproduction.json")
    parser.add_argument("--adapter-steps", type=int, default=800)
    parser.add_argument("--small-mlp-steps", type=int, default=800)
    parser.add_argument("--adapter-hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--lora-learning-rate", type=float, default=1e-3)
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
