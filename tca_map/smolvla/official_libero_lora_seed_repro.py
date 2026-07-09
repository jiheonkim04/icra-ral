"""Official SmolVLA-LIBERO rank-4 LoRA seed reproduction.

This runner audits standard rank-4 LoRA seed robustness under the fixed stable
SmolVLA-LIBERO split and metric protocol. It reuses the existing stable artifact
for frozen/base and mean-action predictions, trains only independent standard
rank-4 LoRA baselines, and evaluates each seed with the frozen metric protocol.

It does not implement a new method, revive FCAR, tune FCAR, train any routing
model, run rollouts, run a full benchmark, run OpenVLA-OFT, download assets, or
use the archived custom LIBERO_7D route.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import sys
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.official_libero_baseline_scaleup import (
    _add_training_batch_dims,
    _cuda_memory,
    _gradient_summary,
    _json_default,
    _loss_from_output,
    _parameter_summary,
    _rss_mb,
    _safe_autocast_status,
    _stat_vector,
    _tensor_devices,
    _tensor_shapes,
    _to_float,
)
from tca_map.smolvla.official_libero_stable_artifact_eval import (
    _evaluate_baselines,
    _evaluate_policy_rows,
    _manifest_samples,
    _read_json,
    _record_key,
    _round,
    _round_vector,
    _row_key,
)


DATE = "2026-07-10 KST"
SEED_ARTIFACT_VERSION = 3
DEFAULT_SEEDS = [11, 22, 33]
MAX_RUNTIME_SECONDS = 3 * 60 * 60
MAX_STEPS = 100
STATIC_GRID = [0.0, 0.25, 0.5, 0.75, 1.0]
REPORT_BASELINES = [
    "frozen_base",
    "rank4_lora",
    "mean_action_prior",
    "frame_oracle",
    "task_oracle",
    "moira_style_instruction_task_router",
    "static_mix_val_selected",
]
REALISTIC_BASELINES = [
    "frozen_base",
    "rank4_lora",
    "mean_action_prior",
    "moira_style_instruction_task_router",
    "static_mix_val_selected",
]
FINAL_DECISIONS = {
    "STANDARD_LORA_ROBUST_BASELINE_READY",
    "STATIC_MERGE_ROBUST_BASELINE_READY",
    "LORA_SEED_INSTABILITY_CONFIRMED",
    "FRAME_ORACLE_HEADROOM_REMAINS_AFTER_SEEDS",
    "SIMPLE_BASELINES_EXPLAIN_GAP",
    "METHOD_DESIGN_STILL_BLOCKED",
    "TOO_HEAVY_LOCAL",
    "TRAINING_FAILURE",
    "CPU_FALLBACK_BUG",
}
FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SIMULATOR_IMPORT_SMOKE",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_TINY_ROLLOUT",
    "ALLOW_CLOUD_HANDOFF",
]


class SeedReproError(RuntimeError):
    """Reportable bounded failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _stat(values: list[float]) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None}
    return {
        "mean": _round(float(array.mean())),
        "std": _round(float(array.std(ddof=0))),
        "min": _round(float(array.min())),
        "max": _round(float(array.max())),
    }


def _parse_seeds(text: str) -> list[int]:
    seeds = [int(part.strip()) for part in text.split(",") if part.strip()]
    if not seeds:
        raise SeedReproError("TRAINING_FAILURE", "At least one seed is required.")
    return seeds


def _preflight(args: argparse.Namespace, seeds: list[int], stable_artifact: dict[str, Any]) -> dict[str, Any]:
    checkpoint = Path(args.checkpoint_path)
    dataset = Path(args.dataset_root)
    manifest = Path(args.split_manifest)
    metric = Path(args.metric_protocol)
    artifact = Path(args.stable_artifact)
    records = stable_artifact.get("records") or []
    if not records:
        raise SeedReproError("TRAINING_FAILURE", "Existing stable artifact has no records.")
    stable_eval = _evaluate_baselines(stable_artifact, seed=int(args.seed))
    static_metric = stable_eval["metrics"]["static_mix_val_selected"]["action_l2_mean"]
    selected_alpha = stable_eval["static_selection"]["selected_weight"]
    return {
        "git_branch_expected": "codex/official-smolvla-lora-seed-repro",
        "checkpoint_path": str(checkpoint),
        "checkpoint_exists": checkpoint.exists(),
        "dataset_path": str(dataset),
        "dataset_exists": dataset.exists(),
        "manifest_path": str(manifest),
        "manifest_exists": manifest.exists(),
        "metric_protocol_path": str(metric),
        "metric_protocol_exists": metric.exists(),
        "stable_artifact_path": str(artifact),
        "stable_artifact_exists": artifact.exists(),
        "stable_artifact_record_count": len(records),
        "static_merge_reproduced_from_existing_artifact": {
            "selected_alpha": selected_alpha,
            "test_action_l2": static_metric,
        },
        "selected_seeds": seeds,
        "estimated_runtime": "about one LoRA-only 2800-record evaluation per seed; bounded by a three-hour cap",
        "device_plan": "CUDA rank-4 LoRA seed training; stop with CPU_FALLBACK_BUG if params or tensors remain on CPU",
        "old_custom_libero_7d_route_used": False,
    }


def _artifact_path(pattern: str, seed: int) -> Path:
    return Path(pattern.format(seed=seed))


def _seed_record_from_base(base_record: dict[str, Any], lora_row: dict[str, Any]) -> dict[str, Any]:
    record = copy.deepcopy(base_record)
    base_l2 = float(record["base_action_l2"])
    lora_l2 = float(lora_row["action_l2"])
    record["lora_action"] = _round_vector(lora_row["pred_preview"], 9)
    record["lora_eval_loss"] = lora_row.get("eval_loss")
    record["lora_action_l2"] = lora_row.get("action_l2")
    record["lora_translation_l2"] = lora_row.get("translation_l2")
    record["lora_rotation_l2"] = lora_row.get("rotation_l2")
    record["lora_gripper_abs"] = lora_row.get("gripper_abs")
    record["oracle_help_label"] = int(lora_l2 < base_l2)
    record["base_minus_lora_action_l2"] = _round(base_l2 - lora_l2)
    return record


def _train_one_seed(
    *,
    args: argparse.Namespace,
    seed: int,
    manifest: dict[str, Any],
    stable_artifact: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    existing_path = _artifact_path(args.seed_artifact_pattern, seed)
    if existing_path.exists() and not bool(args.force):
        artifact = _read_json(existing_path)
        if int(artifact.get("artifact_version", 0)) == SEED_ARTIFACT_VERSION and artifact.get("records"):
            artifact["artifact_status"] = "loaded_existing"
            return artifact

    import torch
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not torch.cuda.is_available():
        raise SeedReproError("CPU_FALLBACK_BUG", "CUDA unavailable; refusing LoRA seed training on CPU.")
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))
    torch.cuda.reset_peak_memory_stats()

    checkpoint_path = Path(args.checkpoint_path)
    dataset_root = Path(args.dataset_root)
    hf_home = Path(args.hf_home)
    vlm_root = Path(args.vlm_root)
    info = _read_json(dataset_root / "meta" / "info.json")
    stats = _read_json(dataset_root / "meta" / "stats.json")
    fps = float(info.get("fps", 10.0))
    selected_episodes, split_samples, all_samples = _manifest_samples(manifest)
    delta_timestamps = {"action": [i / fps for i in range(int(args.chunk_size))]}
    action_min = np.asarray(_stat_vector(stats, "action", "min"), dtype=np.float32)
    action_max = np.asarray(_stat_vector(stats, "action", "max"), dtype=np.float32)

    dataset = LeRobotDataset(
        "lerobot/libero",
        root=dataset_root,
        episodes=selected_episodes,
        delta_timestamps=delta_timestamps,
        video_backend=args.video_backend,
    )
    cfg = PreTrainedConfig.from_pretrained(checkpoint_path, local_files_only=True, cache_dir=hf_home)
    cfg.device = "cuda"
    cfg.load_vlm_weights = True
    cfg.compile_model = False
    cfg.push_to_hub = False
    cfg.vlm_model_name = str(vlm_root)
    if hasattr(cfg, "chunk_size"):
        cfg.chunk_size = int(args.chunk_size)

    policy = SmolVLAPolicy.from_pretrained(
        checkpoint_path,
        config=cfg,
        local_files_only=True,
        cache_dir=hf_home,
        token=False,
        strict=False,
    )
    policy.to("cuda")
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(vlm_root)},
            "device_processor": {"device": "cuda"},
        },
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    probe = _add_training_batch_dims(preprocessor(dataset[int(split_samples["train"][0]["dataset_local_index"])]))
    input_devices = _tensor_devices(probe)
    param_summary = _parameter_summary(policy)
    if not str(param_summary["first_parameter_device"]).startswith("cuda") or not all(
        value.startswith("cuda") for value in input_devices.values()
    ):
        raise SeedReproError("CPU_FALLBACK_BUG", f"CUDA available but params/inputs are not CUDA: params={param_summary}, inputs={input_devices}")

    policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": 4})
    policy.to("cuda")
    policy.train()
    lora_param_summary = _parameter_summary(policy)
    optimizer = torch.optim.AdamW([param for param in policy.parameters() if param.requires_grad], lr=float(args.lr))
    rng = np.random.default_rng(int(seed))
    train_order = rng.permutation(len(split_samples["train"])).tolist()
    loss_curve = []
    grad_curve = []
    first_batch_devices: dict[str, str] | None = None
    first_batch_shapes: dict[str, list[int]] | None = None
    training_started = time.monotonic()
    for step in range(int(args.steps)):
        if time.monotonic() - started > MAX_RUNTIME_SECONDS:
            raise SeedReproError("TOO_HEAVY_LOCAL", "Seed reproduction exceeded three-hour runtime cap during training.")
        train_sample = split_samples["train"][train_order[step % len(train_order)]]
        batch = _add_training_batch_dims(preprocessor(dataset[int(train_sample["dataset_local_index"])]))
        devices = _tensor_devices(batch)
        if first_batch_devices is None:
            first_batch_devices = devices
            first_batch_shapes = _tensor_shapes(batch)
        if not all(value.startswith("cuda") for value in devices.values()):
            raise SeedReproError("CPU_FALLBACK_BUG", f"CUDA available but seed {seed} training tensors are on CPU: {devices}")
        optimizer.zero_grad(set_to_none=True)
        loss = _loss_from_output(policy.forward(batch))
        loss_value = _to_float(loss)
        if not math.isfinite(loss_value):
            raise SeedReproError("TRAINING_FAILURE", f"Non-finite rank-4 LoRA loss for seed {seed} at step {step}: {loss_value}")
        loss.backward()
        grad_summary = _gradient_summary(policy)
        if int(grad_summary["nonzero_grad_tensors"]) == 0:
            raise SeedReproError("TRAINING_FAILURE", f"No nonzero rank-4 LoRA gradients for seed {seed} at step {step}.")
        optimizer.step()
        cuda_now = _cuda_memory(torch)
        loss_curve.append(
            {
                "step": int(step),
                "loss": _round(loss_value),
                "allocated_mb": cuda_now["allocated_mb"],
                "max_allocated_mb": cuda_now["max_allocated_mb"],
            }
        )
        grad_curve.append({"step": int(step), **grad_summary})
    training_elapsed = time.monotonic() - training_started

    print(f"[seed-repro] seed {seed}: evaluating rank-4 LoRA on {len(all_samples)} records", flush=True)
    lora_rows = _evaluate_policy_rows(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        samples=all_samples,
        action_min=action_min,
        action_max=action_max,
        include_eval_loss=bool(args.include_eval_loss),
        label=f"rank4_lora_seed_{seed}",
        started=started,
        progress_every=int(args.progress_every),
    )

    base_records = stable_artifact.get("records") or []
    base_by_key = {_record_key(record): record for record in base_records}
    lora_by_key = {_row_key(row): row for row in lora_rows}
    records = []
    for sample in all_samples:
        key = _record_key(sample)
        records.append(_seed_record_from_base(base_by_key[key], lora_by_key[key]))

    artifact = {
        "artifact_version": SEED_ARTIFACT_VERSION,
        "date": DATE,
        "artifact_status": "generated",
        "source": "official_smolvla_rank4_lora_seed_repro",
        "seed": int(seed),
        "policy": {
            "downloads_performed": False,
            "openvla_oft_executed": False,
            "rollouts_performed": False,
            "full_benchmark_performed": False,
            "old_custom_route_used": False,
            "official_model_dataset_used": True,
            "rank4_lora_regenerated": True,
            "smolvla_backbone_trained": False,
            "new_method_implemented": False,
            "fcar_tuned": False,
        },
        "paths": {
            "checkpoint": str(checkpoint_path),
            "dataset": str(dataset_root),
            "hf_home": str(hf_home),
            "vlm_root": str(vlm_root),
            "split_manifest": str(Path(args.split_manifest)),
            "metric_protocol": str(Path(args.metric_protocol)),
            "stable_artifact": str(Path(args.stable_artifact)),
        },
        "dataset": {
            "selected_episode_count": len(selected_episodes),
            "selected_episodes": selected_episodes,
            "split_frame_counts": {split: len(samples) for split, samples in split_samples.items()},
            "prediction_record_count": len(records),
        },
        "action_range": (stable_artifact.get("action_range") or {}),
        "device_audit": {
            "cuda_available": True,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "model_parameter_device": param_summary["first_parameter_device"],
            "model_parameter_dtype": param_summary["first_parameter_dtype"],
            "input_tensor_devices": input_devices,
            "input_tensor_shapes": _tensor_shapes(probe),
            "autocast_status_initial_final": _safe_autocast_status(torch),
            "cuda_memory": _cuda_memory(torch),
        },
        "rank4_lora_regeneration": {
            "seed": int(seed),
            "steps": int(args.steps),
            "batch_size": 1,
            "train_split": "train",
            "train_frame_count": len(split_samples["train"]),
            "trainable_params": lora_param_summary["trainable_params"],
            "total_params": lora_param_summary["total_params"],
            "loss_before": loss_curve[0]["loss"] if loss_curve else None,
            "loss_after": loss_curve[-1]["loss"] if loss_curve else None,
            "loss_curve": loss_curve,
            "last_grad_norm": grad_curve[-1]["grad_norm"] if grad_curve else None,
            "last_nonzero_grad_tensors": grad_curve[-1]["nonzero_grad_tensors"] if grad_curve else None,
            "training_elapsed_sec": _round(training_elapsed, 3),
            "steps_per_sec": _round(len(loss_curve) / max(training_elapsed, 1e-12), 6),
            "input_tensor_devices_first_batch": first_batch_devices,
            "input_tensor_shapes_first_batch": first_batch_shapes,
            "autocast_status": _safe_autocast_status(torch),
        },
        "records": records,
        "runtime": {
            "seed_elapsed_sec": _round(time.monotonic() - training_started, 3),
            "rss_final_mb": _rss_mb(),
            "cuda": _cuda_memory(torch),
        },
    }
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    existing_path.write_text(json.dumps(artifact, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    return artifact


def _seed_summary(seed: int, artifact: dict[str, Any], *, seed_offset: int, artifact_path: Path) -> dict[str, Any]:
    evaluation = _evaluate_baselines(artifact, seed=seed_offset)
    metrics = evaluation["metrics"]
    action_l2 = {name: metrics[name]["action_l2_mean"] for name in REPORT_BASELINES}
    static_l2 = float(action_l2["static_mix_val_selected"])
    frame_l2 = float(action_l2["frame_oracle"])
    base_l2 = float(action_l2["frozen_base"])
    lora_l2 = float(action_l2["rank4_lora"])
    task_l2 = float(action_l2["task_oracle"])
    return {
        "seed": int(seed),
        "artifact_path": str(artifact_path),
        "artifact_status": artifact.get("artifact_status"),
        "artifact_size_bytes": artifact_path.stat().st_size if artifact_path.exists() else None,
        "record_count": len(artifact.get("records") or []),
        "rank4_lora_regeneration": artifact.get("rank4_lora_regeneration"),
        "device_audit": artifact.get("device_audit"),
        "metrics": {name: metrics[name] for name in REPORT_BASELINES},
        "static_selection": evaluation["static_selection"],
        "rank_order_realistic": evaluation["rank_order_realistic"],
        "rank_order_with_oracles": evaluation["rank_order_with_oracles"],
        "win_counts_by_task": evaluation["win_counts_by_task"],
        "win_counts_by_phase": evaluation["win_counts_by_phase"],
        "analysis": {
            "lora_minus_frozen_base": _round(lora_l2 - base_l2),
            "static_minus_lora": _round(static_l2 - lora_l2),
            "static_minus_frozen_base": _round(static_l2 - base_l2),
            "frame_oracle_headroom_after_static": _round(static_l2 - frame_l2),
            "frame_oracle_headroom_over_base": _round(base_l2 - frame_l2),
            "task_oracle_headroom_over_base": _round(base_l2 - task_l2),
            "static_is_best_realistic": evaluation["rank_order_realistic"][0]["baseline"] == "static_mix_val_selected",
            "lora_beats_base": lora_l2 < base_l2,
            "lora_beats_static": lora_l2 < static_l2,
        },
    }


def _aggregate_seed_summaries(seed_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    baseline_summary = {}
    for baseline in REPORT_BASELINES:
        baseline_summary[baseline] = {
            "action_l2": _stat([float(seed["metrics"][baseline]["action_l2_mean"]) for seed in seed_summaries]),
            "task_balanced_action_l2": _stat([float(seed["metrics"][baseline]["task_balanced_action_l2_mean"]) for seed in seed_summaries]),
            "translation_l2": _stat([float(seed["metrics"][baseline]["translation_l2_mean"]) for seed in seed_summaries]),
            "rotation_l2": _stat([float(seed["metrics"][baseline]["rotation_l2_mean"]) for seed in seed_summaries]),
            "gripper_abs": _stat([float(seed["metrics"][baseline]["gripper_abs_mean"]) for seed in seed_summaries]),
        }
    seed_wins = Counter(seed["rank_order_realistic"][0]["baseline"] for seed in seed_summaries)
    task_wins = Counter()
    for seed in seed_summaries:
        task_wins.update(seed["win_counts_by_task"]["counts"])
    lora_values = [float(seed["metrics"]["rank4_lora"]["action_l2_mean"]) for seed in seed_summaries]
    static_values = [float(seed["metrics"]["static_mix_val_selected"]["action_l2_mean"]) for seed in seed_summaries]
    frame_values = [float(seed["metrics"]["frame_oracle"]["action_l2_mean"]) for seed in seed_summaries]
    task_values = [float(seed["metrics"]["task_oracle"]["action_l2_mean"]) for seed in seed_summaries]
    base_values = [float(seed["metrics"]["frozen_base"]["action_l2_mean"]) for seed in seed_summaries]
    moira_values = [float(seed["metrics"]["moira_style_instruction_task_router"]["action_l2_mean"]) for seed in seed_summaries]
    lora_beats_base = [bool(seed["analysis"]["lora_beats_base"]) for seed in seed_summaries]
    lora_beats_static = [bool(seed["analysis"]["lora_beats_static"]) for seed in seed_summaries]
    static_best = [bool(seed["analysis"]["static_is_best_realistic"]) for seed in seed_summaries]
    frame_after_static = [float(seed["analysis"]["frame_oracle_headroom_after_static"]) for seed in seed_summaries]
    task_headroom = [float(seed["analysis"]["task_oracle_headroom_over_base"]) for seed in seed_summaries]
    moira_weak = [m >= min(b, l, s) for m, b, l, s in zip(moira_values, base_values, lora_values, static_values)]
    return {
        "baseline_summary": baseline_summary,
        "seed_win_counts_realistic": dict(sorted(seed_wins.items())),
        "task_win_counts_realistic_sum": dict(sorted(task_wins.items())),
        "lora_seed_variance": {
            "action_l2": _stat(lora_values),
            "range": _round(max(lora_values) - min(lora_values)),
            "relative_std": _round(float(np.std(lora_values, ddof=0)) / max(float(np.mean(lora_values)), 1e-12)),
        },
        "static_seed_variance": {"action_l2": _stat(static_values), "range": _round(max(static_values) - min(static_values))},
        "frame_oracle_after_static": {"values": [_round(value) for value in frame_after_static], **_stat(frame_after_static)},
        "task_oracle_headroom": {"values": [_round(value) for value in task_headroom], **_stat(task_headroom)},
        "answers": {
            "rank4_lora_robustly_beats_frozen_base": all(lora_beats_base),
            "rank4_lora_robustly_beats_static_merge": all(lora_beats_static),
            "static_merge_remains_strongest_realistic_baseline": all(static_best),
            "lora_seed_variance_action_l2_std": _stat(lora_values)["std"],
            "frame_oracle_headroom_remains_after_static": all(value >= 0.005 for value in frame_after_static),
            "task_oracle_remains_meaningful": all(value >= 0.005 for value in task_headroom),
            "moira_style_task_router_remains_weak": all(moira_weak),
            "method_worthy_gap_left_after_static": all(value >= 0.005 for value in frame_after_static),
            "lora_instability_confirmed": (any(lora_beats_base) and not all(lora_beats_base))
            or (_stat(lora_values)["std"] is not None and float(_stat(lora_values)["std"]) >= 0.005),
        },
    }


def _choose_decision(aggregate: dict[str, Any]) -> str:
    answers = aggregate["answers"]
    if answers["rank4_lora_robustly_beats_frozen_base"] and answers["rank4_lora_robustly_beats_static_merge"]:
        return "STANDARD_LORA_ROBUST_BASELINE_READY"
    if answers["static_merge_remains_strongest_realistic_baseline"]:
        return "STATIC_MERGE_ROBUST_BASELINE_READY"
    if answers["lora_instability_confirmed"]:
        return "LORA_SEED_INSTABILITY_CONFIRMED"
    if answers["frame_oracle_headroom_remains_after_static"] and answers["moira_style_task_router_remains_weak"]:
        return "FRAME_ORACLE_HEADROOM_REMAINS_AFTER_SEEDS"
    if not answers["method_worthy_gap_left_after_static"]:
        return "SIMPLE_BASELINES_EXPLAIN_GAP"
    return "METHOD_DESIGN_STILL_BLOCKED"


def _decision_reason(decision: str) -> str:
    return {
        "STANDARD_LORA_ROBUST_BASELINE_READY": "Rank-4 LoRA consistently beats frozen/base and static merge across reproduced seeds.",
        "STATIC_MERGE_ROBUST_BASELINE_READY": "Validation-selected static merge consistently beats frozen/base and rank-4 LoRA across reproduced seeds.",
        "LORA_SEED_INSTABILITY_CONFIRMED": "Rank-4 LoRA ranking varies strongly across seeds, making seed robustness the main blocker.",
        "FRAME_ORACLE_HEADROOM_REMAINS_AFTER_SEEDS": "Frame oracle remains meaningfully better than static merge while task/instruction routing remains insufficient.",
        "SIMPLE_BASELINES_EXPLAIN_GAP": "Frozen/base, standard LoRA, or static merge explain most of the available gain.",
        "METHOD_DESIGN_STILL_BLOCKED": "Metric or seed instability remains too high for method design.",
        "TOO_HEAVY_LOCAL": "The bounded local RTX/RAM/runtime budget could not support seed reproduction.",
        "TRAINING_FAILURE": "Rank-4 LoRA training failed, OOMed, produced NaNs, or could not produce fair artifacts.",
        "CPU_FALLBACK_BUG": "CUDA was available but the intended CUDA training path fell back to CPU.",
    }[decision]


def _next_step(decision: str) -> str:
    return {
        "STANDARD_LORA_ROBUST_BASELINE_READY": "Plan longer official rank-4 LoRA baseline scaleup under the fixed manifest.",
        "STATIC_MERGE_ROBUST_BASELINE_READY": "Treat validation-selected static merge as the main realistic baseline for any later planning gate.",
        "LORA_SEED_INSTABILITY_CONFIRMED": "Diagnose rank-4 LoRA seed instability under the fixed manifest before method design.",
        "FRAME_ORACLE_HEADROOM_REMAINS_AFTER_SEEDS": "A later planning-only frame-level method gate is allowed; do not implement a method in this run.",
        "SIMPLE_BASELINES_EXPLAIN_GAP": "Stop method design under this evidence and preserve the stable seed table.",
        "METHOD_DESIGN_STILL_BLOCKED": "Continue diagnosis of seed/metric instability before method design.",
        "TOO_HEAVY_LOCAL": "Reduce seed count through a predeclared smaller audit or move the same command to a larger GPU host.",
        "TRAINING_FAILURE": "Fix the exact training blocker and rerun the same fixed seed audit.",
        "CPU_FALLBACK_BUG": "Fix CUDA device placement before rerunning seed reproduction.",
    }[decision]


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ["HF_HOME"] = str(Path(args.hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    seeds = _parse_seeds(args.seeds)
    stable_artifact = _read_json(Path(args.stable_artifact))
    manifest = _read_json(Path(args.split_manifest))
    preflight = _preflight(args, seeds, stable_artifact)
    report: dict[str, Any] = {
        "date": DATE,
        "status": "started",
        "final_decision": None,
        "preflight": preflight,
        "policy": {
            "experiments_performed": True,
            "training_performed": True,
            "trained_components": ["standard rank-4 LoRA baseline seeds"],
            "gpu_used": True,
            "downloads_performed": False,
            "openvla_oft_executed": False,
            "rollouts_performed": False,
            "full_benchmark_performed": False,
            "new_method_implemented": False,
            "fcar_tuned": False,
            "official_model_dataset_used": True,
            "old_custom_route_used": False,
            "paper_claims_made": False,
        },
        "paths": {
            "checkpoint": str(Path(args.checkpoint_path)),
            "dataset": str(Path(args.dataset_root)),
            "split_manifest": str(Path(args.split_manifest)),
            "metric_protocol": str(Path(args.metric_protocol)),
            "stable_artifact": str(Path(args.stable_artifact)),
        },
        "errors": [],
    }
    try:
        forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
        if forbidden:
            raise SeedReproError("TRAINING_FAILURE", "Forbidden gate(s) set: " + ", ".join(forbidden))
        if int(args.steps) < 1 or int(args.steps) > MAX_STEPS:
            raise SeedReproError("TOO_HEAVY_LOCAL", f"Steps must be in [1, {MAX_STEPS}], got {args.steps}.")
        seed_summaries = []
        for seed in seeds:
            if time.monotonic() - started > MAX_RUNTIME_SECONDS:
                raise SeedReproError("TOO_HEAVY_LOCAL", "Seed reproduction exceeded three-hour runtime cap.")
            print(f"[seed-repro] seed {seed}: training standard rank-4 LoRA", flush=True)
            artifact = _train_one_seed(
                args=args,
                seed=seed,
                manifest=manifest,
                stable_artifact=stable_artifact,
                started=started,
            )
            seed_summaries.append(
                _seed_summary(
                    seed,
                    artifact,
                    seed_offset=int(seed),
                    artifact_path=_artifact_path(args.seed_artifact_pattern, seed),
                )
            )
        aggregate = _aggregate_seed_summaries(seed_summaries)
        final_decision = _choose_decision(aggregate)
        report.update(
            {
                "status": "completed",
                "final_decision": final_decision,
                "decision_reason": _decision_reason(final_decision),
                "exact_next_step": _next_step(final_decision),
                "seeds": seeds,
                "seed_count": len(seeds),
                "seed_summaries": seed_summaries,
                "aggregate": aggregate,
                "manifest_summary": (manifest.get("summary") or {}),
                "runtime": {
                    "total_elapsed_sec": _round(time.monotonic() - started, 3),
                    "rss_final_mb": _rss_mb(),
                },
            }
        )
    except SeedReproError as exc:
        decision = exc.code if exc.code in FINAL_DECISIONS else "TRAINING_FAILURE"
        report["status"] = "blocked"
        report["final_decision"] = decision
        report["decision_reason"] = _decision_reason(decision)
        report["exact_next_step"] = _next_step(decision)
        report["errors"].append({"code": exc.code, "message": str(exc)})
        report["runtime"] = {"total_elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        if decision == "CPU_FALLBACK_BUG":
            report["policy"]["gpu_used"] = False
        return report, 41
    except Exception as exc:  # pragma: no cover - runtime reporting boundary
        report["status"] = "blocked"
        report["final_decision"] = "TRAINING_FAILURE"
        report["decision_reason"] = _decision_reason("TRAINING_FAILURE")
        report["exact_next_step"] = _next_step("TRAINING_FAILURE")
        report["errors"].append({"code": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        report["runtime"] = {"total_elapsed_sec": _round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, 42
    return report, 0


def _write_lines(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plan(report: dict[str, Any], path: Path) -> None:
    preflight = report["preflight"]
    lines = [
        "# Official SmolVLA Rank-4 LoRA Seed Reproduction Plan",
        "",
        f"Date: {report['date']}",
        "",
        "Purpose: audit standard rank-4 LoRA seed robustness under the fixed official SmolVLA-LIBERO manifest and metric protocol.",
        "",
        "Boundary:",
        "",
        "- no new method",
        "- no FCAR revival or tuning",
        "- no routing model",
        "- no simulator rollout or full benchmark",
        "- no OpenVLA-OFT",
        "- no downloads",
        "- no old custom LIBERO_7D route",
        "- no static-alpha tuning on test",
        "",
        "Preflight:",
        "",
        f"- model path: `{preflight['checkpoint_path']}`",
        f"- dataset path: `{preflight['dataset_path']}`",
        f"- manifest path: `{preflight['manifest_path']}`",
        f"- existing artifact path: `{preflight['stable_artifact_path']}`",
        f"- selected seeds: `{preflight['selected_seeds']}`",
        f"- device plan: `{preflight['device_plan']}`",
        f"- static merge reproduced: `{preflight['static_merge_reproduced_from_existing_artifact']}`",
        f"- estimated runtime: `{preflight['estimated_runtime']}`",
    ]
    _write_lines(path, lines)


def _write_result(report: dict[str, Any], path: Path) -> None:
    aggregate = report.get("aggregate") or {}
    summary = aggregate.get("baseline_summary") or {}
    lines = [
        "# Official SmolVLA Rank-4 LoRA Seed Reproduction Result",
        "",
        f"Date: {report['date']}",
        "",
        f"- final decision: `{report.get('final_decision')}`",
        f"- experiments happened: `{report['policy']['experiments_performed']}`",
        f"- training happened: `{report['policy']['training_performed']}`",
        f"- trained components: `{report['policy']['trained_components']}`",
        f"- GPU/download/OpenVLA-OFT happened: `{report['policy']['gpu_used']}` / `{report['policy']['downloads_performed']}` / `{report['policy']['openvla_oft_executed']}`",
        f"- official model/dataset used: `{report['policy']['official_model_dataset_used']}`",
        f"- old custom route used: `{report['policy']['old_custom_route_used']}`",
        f"- seeds: `{report.get('seeds')}`",
        "",
        "## Mean/Std Across Seeds",
        "",
        "| baseline | action L2 mean | action L2 std | task-balanced mean | translation mean | rotation mean | gripper abs mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for baseline in REPORT_BASELINES:
        item = summary.get(baseline) or {}
        lines.append(
            f"| {baseline} | {(item.get('action_l2') or {}).get('mean')} | {(item.get('action_l2') or {}).get('std')} | "
            f"{(item.get('task_balanced_action_l2') or {}).get('mean')} | {(item.get('translation_l2') or {}).get('mean')} | "
            f"{(item.get('rotation_l2') or {}).get('mean')} | {(item.get('gripper_abs') or {}).get('mean')} |"
        )
    lines.extend(
        [
            "",
            "## Seed Robustness Answers",
            "",
            *[f"- {key}: `{value}`" for key, value in (aggregate.get("answers") or {}).items()],
            "",
            f"- seed win counts realistic: `{aggregate.get('seed_win_counts_realistic')}`",
            f"- task win counts realistic sum: `{aggregate.get('task_win_counts_realistic_sum')}`",
            f"- LoRA seed variance: `{aggregate.get('lora_seed_variance')}`",
            f"- frame oracle after static: `{aggregate.get('frame_oracle_after_static')}`",
            f"- task oracle headroom: `{aggregate.get('task_oracle_headroom')}`",
            "",
            "## Exact Next Step",
            "",
            str(report.get("exact_next_step")),
        ]
    )
    _write_lines(path, lines)


def _write_table(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Official SmolVLA Rank-4 LoRA Seed Reproduction Table",
        "",
        f"Date: {report['date']}",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        "| seed | frozen/base | rank-4 LoRA | static mix | frame oracle | task oracle | MoIRA router | realistic winner | LoRA-base | frame-static |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: |",
    ]
    for seed in report.get("seed_summaries") or []:
        metrics = seed["metrics"]
        lines.append(
            f"| {seed['seed']} | {metrics['frozen_base']['action_l2_mean']} | {metrics['rank4_lora']['action_l2_mean']} | "
            f"{metrics['static_mix_val_selected']['action_l2_mean']} | {metrics['frame_oracle']['action_l2_mean']} | "
            f"{metrics['task_oracle']['action_l2_mean']} | {metrics['moira_style_instruction_task_router']['action_l2_mean']} | "
            f"{seed['rank_order_realistic'][0]['baseline']} | {seed['analysis']['lora_minus_frozen_base']} | "
            f"{seed['analysis']['frame_oracle_headroom_after_static']} |"
        )
    _write_lines(path, lines)


def _write_decision(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Official SmolVLA Rank-4 LoRA Seed Reproduction Decision",
        "",
        f"Date: {report['date']}",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        f"Reason: {report.get('decision_reason')}",
        "",
        f"Exact next step: {report.get('exact_next_step')}",
    ]
    _write_lines(path, lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default="C:/assets/checkpoints/smolvla_libero")
    parser.add_argument("--dataset-root", default="C:/assets/datasets/lerobot_libero")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--vlm-root", default="C:/assets/hf_home/HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--split-manifest", default="reports/official_smolvla_split_manifest.json")
    parser.add_argument("--metric-protocol", default="reports/official_smolvla_metric_protocol.md")
    parser.add_argument("--stable-artifact", default="reports/official_smolvla_stable_prediction_artifact.json")
    parser.add_argument("--seed-artifact-pattern", default="reports/official_smolvla_lora_seed_{seed}_prediction_artifact.json")
    parser.add_argument("--report-json", default="reports/official_smolvla_lora_seed_repro_result.json")
    parser.add_argument("--result-md", default="reports/official_smolvla_lora_seed_repro_result.md")
    parser.add_argument("--plan-md", default="reports/official_smolvla_lora_seed_repro_plan.md")
    parser.add_argument("--table-md", default="reports/official_smolvla_lora_seed_repro_table.md")
    parser.add_argument("--decision-md", default="reports/official_smolvla_lora_seed_repro_decision.md")
    parser.add_argument("--seeds", default=",".join(str(seed) for seed in DEFAULT_SEEDS))
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--progress-every", type=int, default=100)
    parser.add_argument("--include-eval-loss", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    _write_plan(report, Path(args.plan_md))
    _write_result(report, Path(args.result_md))
    _write_table(report, Path(args.table_md))
    _write_decision(report, Path(args.decision_md))
    summary = {
        "status": report.get("status"),
        "final_decision": report.get("final_decision"),
        "seeds": report.get("seeds"),
        "aggregate": report.get("aggregate", {}).get("baseline_summary"),
        "runtime": report.get("runtime"),
        "errors": report.get("errors"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True, default=_json_default))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
