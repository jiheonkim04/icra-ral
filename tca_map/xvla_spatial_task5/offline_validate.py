"""Offline validation gate for frozen R2P-XVLA task-5 training arms.

The gate reloads saved PEFT adapters only after the frozen training gate has
produced them, compares the primary R2P-XVLA arm against the uniform adaptation
ablation and cached X-VLA-Libero prior on fixed validation chunks, and records
offline loss/action-delta metrics only.

Importing or testing this module performs no model loading, training, optimizer
step, checkpoint write, simulator rollout, closed-loop Ours evaluation, or
download.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.xvla_spatial_task5.data_adapter_smoke import DEFAULT_XVLA_ROOT
from tca_map.xvla_spatial_task5.gradient_smoke import (
    PHASE_LOSS_WEIGHTS,
    cuda_memory,
    install_optional_server_import_shims,
    install_xvla_transformers_compat_patches,
    nvidia_smi,
    package_version,
    prepare_inputs,
)
from tca_map.xvla_spatial_task5.train_lora import (
    DEFAULT_CLIP_STEPS,
    DEFAULT_OUTPUT_ROOT,
    XVLA_CACHE_DIR,
    _arm_by_id,
    _first_xvla_reader_sample,
    _git_commit,
    _json_default,
    _load_spec,
    _phase_cycle_from_sampler,
    _prepare_xvla_imports,
    _write_json,
    build_phase_clip_index,
    materialize_xvla_clip,
    select_clip_for_step,
)
from tca_map.xvla_spatial_task5.training_spec import MODEL_ID, MODEL_REVISION, SPEC_ARTIFACT, TASK5_HDF5_WSL


DEFAULT_OUTPUT = Path("runs/xvla_prior/epoch5_r2p_xvla_task5_offline_validation_step0064.json")
DEFAULT_NUM_CHUNKS = 24
PRIMARY_LABEL = "r2p_xvla_primary"
UNIFORM_LABEL = "uniform_task5_xvla_ablation"
PRIOR_LABEL = "xvla_prior_base"


@dataclass(frozen=True)
class OfflineValidationConfig:
    spec_path: Path = SPEC_ARTIFACT
    output_path: Path = DEFAULT_OUTPUT
    training_output_root: Path = DEFAULT_OUTPUT_ROOT
    xvla_root: Path = DEFAULT_XVLA_ROOT
    hdf5_path: Path = Path(TASK5_HDF5_WSL)
    primary_adapter_dir: Path | None = None
    ablation_adapter_dir: Path | None = None
    num_chunks: int = DEFAULT_NUM_CHUNKS
    device_index: int = 0
    denoise_steps: int = 10
    local_files_only: bool = True
    clip_steps: int = DEFAULT_CLIP_STEPS


def _assert_output_path_allowed(output_path: Path) -> None:
    allowed = (Path.cwd() / DEFAULT_OUTPUT).resolve()
    candidate = output_path.resolve()
    if candidate != allowed:
        raise ValueError(f"output_path must be exactly {DEFAULT_OUTPUT.as_posix()}, got {output_path}")


def select_fixed_validation_clips(
    *,
    hdf5_path: Path,
    spec: dict[str, Any],
    num_chunks: int,
    clip_steps: int = DEFAULT_CLIP_STEPS,
) -> list[dict[str, Any]]:
    """Select deterministic held-out validation clips using the frozen phase cycle."""

    if int(num_chunks) < 1:
        raise ValueError("num_chunks must be at least 1")
    if int(num_chunks) > DEFAULT_NUM_CHUNKS:
        raise ValueError(f"num_chunks must not exceed frozen limit {DEFAULT_NUM_CHUNKS}")
    grouped = build_phase_clip_index(
        hdf5_path,
        demo_indices=list(spec["data"]["validation_demo_indices"]),
        clip_steps=int(clip_steps),
    )
    rng = np.random.default_rng(int(spec["shared_training"]["seed"]) + 1)
    cycle = _phase_cycle_from_sampler(spec["arms"][0]["sampler"])
    clips = [
        select_clip_for_step(grouped, cycle=cycle, step_index_zero_based=index, rng=rng)
        for index in range(int(num_chunks))
    ]
    for index, clip in enumerate(clips):
        clip["validation_index"] = int(index)
    return clips


def _phase_key(phase: str, metric: str) -> str:
    return f"phase_{phase}_{metric}"


def _summarize_policy_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    losses = np.asarray([row["loss_total"] for row in rows], dtype=np.float64)
    weighted = np.asarray([row["weighted_loss"] for row in rows], dtype=np.float64)
    summary: dict[str, Any] = {
        "count": int(len(rows)),
        "mean_loss_total": float(np.mean(losses)) if losses.size else None,
        "median_loss_total": float(np.median(losses)) if losses.size else None,
        "mean_weighted_loss": float(np.mean(weighted)) if weighted.size else None,
        "all_losses_finite": bool(losses.size and np.isfinite(losses).all() and np.isfinite(weighted).all()),
    }
    for phase in PHASE_LOSS_WEIGHTS:
        phase_rows = [row for row in rows if str(row["phase_label"]) == str(phase)]
        phase_losses = np.asarray([row["loss_total"] for row in phase_rows], dtype=np.float64)
        phase_weighted = np.asarray([row["weighted_loss"] for row in phase_rows], dtype=np.float64)
        summary[_phase_key(str(phase), "count")] = int(len(phase_rows))
        summary[_phase_key(str(phase), "mean_loss_total")] = float(np.mean(phase_losses)) if phase_losses.size else None
        summary[_phase_key(str(phase), "mean_weighted_loss")] = (
            float(np.mean(phase_weighted)) if phase_weighted.size else None
        )
    return summary


def _summarize_action_delta(rows: list[dict[str, Any]], prior_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prior_by_index = {int(row["validation_index"]): np.asarray(row["generated_action"], dtype=np.float64) for row in prior_rows}
    mean_abs: list[float] = []
    max_abs: list[float] = []
    for row in rows:
        ours = np.asarray(row["generated_action"], dtype=np.float64)
        prior = prior_by_index[int(row["validation_index"])]
        delta = np.abs(ours - prior)
        mean_abs.append(float(np.mean(delta)))
        max_abs.append(float(np.max(delta)))
    return {
        "fixed_chunk_mean_abs_action_delta": float(np.mean(mean_abs)),
        "fixed_chunk_max_abs_action_delta": float(np.max(max_abs)),
    }


def _required_float(value: Any, name: str) -> float:
    if value is None:
        raise ValueError(f"missing required offline metric: {name}")
    return float(value)


def compute_offline_selection_decision(
    *,
    spec: dict[str, Any],
    summaries: dict[str, dict[str, Any]],
    runtimes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen R2P-XVLA offline selection criteria to summaries."""

    criteria = spec["validation_selection"]["offline_pass_criteria"]
    primary = summaries[PRIMARY_LABEL]
    uniform = summaries[UNIFORM_LABEL]
    primary_delta = primary.get("delta_vs_prior", {})
    reasons: list[str] = []

    finite_losses = all(bool(summaries[label].get("all_losses_finite", False)) for label in (PRIOR_LABEL, PRIMARY_LABEL, UNIFORM_LABEL))
    if not finite_losses:
        reasons.append("nonfinite_or_missing_losses")

    primary_weighted = _required_float(primary.get("mean_weighted_loss"), "primary mean_weighted_loss")
    uniform_weighted = _required_float(uniform.get("mean_weighted_loss"), "uniform mean_weighted_loss")
    primary_beats_uniform_weighted = bool(primary_weighted < uniform_weighted)
    if not primary_beats_uniform_weighted:
        reasons.append("primary_does_not_beat_uniform_on_phase_weighted_validation_loss")

    source_key = _phase_key("source_on_ramekin", "mean_loss_total")
    primary_source = _required_float(primary.get(source_key), f"primary {source_key}")
    uniform_source = _required_float(uniform.get(source_key), f"uniform {source_key}")
    source_relative_degradation = float((primary_source - uniform_source) / max(uniform_source, 1e-8))
    source_bound = float(criteria["primary_must_not_worsen_source_phase_loss_vs_uniform_by_more_than"])
    source_within_bound = bool(source_relative_degradation <= source_bound)
    if not source_within_bound:
        reasons.append("source_phase_degradation_vs_uniform_exceeds_bound")

    mean_delta = _required_float(primary_delta.get("fixed_chunk_mean_abs_action_delta"), "primary mean action delta")
    max_delta = _required_float(primary_delta.get("fixed_chunk_max_abs_action_delta"), "primary max action delta")
    mean_delta_within_bound = bool(mean_delta <= float(criteria["fixed_chunk_mean_abs_action_delta_max"]))
    max_delta_within_bound = bool(max_delta <= float(criteria["fixed_chunk_max_abs_action_delta_max"]))
    if not mean_delta_within_bound:
        reasons.append("fixed_chunk_mean_abs_action_delta_exceeds_bound")
    if not max_delta_within_bound:
        reasons.append("fixed_chunk_max_abs_action_delta_exceeds_bound")

    cuda_peak = max(
        float(runtime.get("cuda_memory_after_eval", {}).get("max_allocated_mib", 0.0)) for runtime in runtimes.values()
    )
    cuda_within_bound = bool(cuda_peak <= float(criteria["cuda_peak_mib_max"]))
    if not cuda_within_bound:
        reasons.append("cuda_peak_exceeds_bound")

    passed = bool(not reasons)
    return {
        "decision": "R2P_XVLA_OFFLINE_PASS_BEATS_UNIFORM_ABLATION"
        if passed
        else "R2P_XVLA_OFFLINE_SELECTION_NOT_PASSED",
        "passed": passed,
        "reasons": reasons,
        "primary_beats_uniform_on_phase_weighted_validation_loss": primary_beats_uniform_weighted,
        "primary_mean_weighted_loss": primary_weighted,
        "uniform_mean_weighted_loss": uniform_weighted,
        "source_phase_relative_degradation_vs_uniform": source_relative_degradation,
        "source_phase_degradation_bound": source_bound,
        "fixed_chunk_mean_abs_action_delta": mean_delta,
        "fixed_chunk_max_abs_action_delta": max_delta,
        "cuda_peak_mib": cuda_peak,
    }


def _load_policy(config: OfflineValidationConfig, adapter_dir: Path | None) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from peft import PeftModel

    import_report = _prepare_xvla_imports(config.xvla_root)
    import_report.setdefault("optional_server_import_shims_used", install_optional_server_import_shims())
    import_report.setdefault("transformers_compat_patches", install_xvla_transformers_compat_patches())
    import_report.setdefault(
        "runtime_dependency_versions",
        {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "timm": package_version("timm"),
        },
    )
    from models.modeling_xvla import XVLA  # type: ignore
    from models.processing_xvla import XVLAProcessor  # type: ignore

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable for R2P-XVLA offline validation")
    torch.cuda.set_device(int(config.device_index))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=int(config.device_index))
    device = torch.device(f"cuda:{int(config.device_index)}")
    from tca_map.xvla_spatial_task5.gradient_smoke import LOCAL_MODEL_SNAPSHOT

    load_source = str(LOCAL_MODEL_SNAPSHOT) if LOCAL_MODEL_SNAPSHOT.exists() else MODEL_ID
    load_kwargs: dict[str, Any] = {
        "trust_remote_code": True,
        "torch_dtype": torch.float32,
        "local_files_only": bool(config.local_files_only),
    }
    if load_source == MODEL_ID:
        load_kwargs["revision"] = MODEL_REVISION
        load_kwargs["cache_dir"] = XVLA_CACHE_DIR
    import_report["pretrained_load_source"] = load_source
    import_report["pretrained_load_source_is_local_snapshot"] = load_source != MODEL_ID
    model = XVLA.from_pretrained(load_source, **load_kwargs)
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    processor = XVLAProcessor.from_pretrained(load_source, **load_kwargs)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    import_report["cuda_memory_after_load"] = cuda_memory()
    return model, processor, import_report


def _generate_actions(model: Any, inputs: dict[str, Any], steps: int) -> Any:
    if hasattr(model, "generate_actions"):
        return model.generate_actions(**inputs, steps=int(steps))
    if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        return model.base_model.model.generate_actions(**inputs, steps=int(steps))
    raise AttributeError("policy does not expose generate_actions")


def _evaluate_policy_rows(
    *,
    label: str,
    adapter_dir: Path | None,
    evaluation_arm: dict[str, Any],
    clips: list[dict[str, Any]],
    config: OfflineValidationConfig,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.monotonic()
    model = None
    work_clip_dir = config.output_path.parent / f"{config.output_path.stem}_{label}_working_clip"
    try:
        import torch

        model, processor, load_report = _load_policy(config, adapter_dir)
        device = torch.device(f"cuda:{int(config.device_index)}")
        rows: list[dict[str, Any]] = []
        with torch.no_grad():
            for clip in clips:
                materialized = materialize_xvla_clip(config.hdf5_path, work_clip_dir, clip, spec, evaluation_arm)
                sample = _first_xvla_reader_sample(config.xvla_root, Path(materialized["meta_path"]))
                inputs = prepare_inputs(sample, processor, device, torch.float32)
                shutil.rmtree(work_clip_dir, ignore_errors=True)
                seed = int(20260718 + int(clip["validation_index"]))
                torch.manual_seed(seed)
                loss_dict = model(**inputs)
                loss_total = sum(loss_dict.values())
                phase_weight = float(materialized["phase_weight_mean"])
                weighted_loss = loss_total * phase_weight
                torch.manual_seed(seed)
                generated = _generate_actions(
                    model,
                    {
                        key: value
                        for key, value in inputs.items()
                        if key in {"input_ids", "image_input", "image_mask", "domain_id", "proprio"}
                    },
                    steps=int(config.denoise_steps),
                )
                generated_np = generated.detach().float().cpu().numpy()
                rows.append(
                    {
                        "policy": label,
                        "validation_index": int(clip["validation_index"]),
                        "phase_label": str(clip["phase_label"]),
                        "clip": {
                            key: clip[key]
                            for key in ("demo_index", "demo_name", "source_start_index", "source_end_index", "phase_label")
                        },
                        "materialized": {
                            key: materialized[key]
                            for key in ("clip_steps", "phase_counts", "phase_weight_mean", "abs_action_6d_shape")
                        },
                        "losses": {key: float(value.detach().float().item()) for key, value in loss_dict.items()},
                        "loss_total": float(loss_total.detach().float().item()),
                        "evaluation_phase_weight_mean": float(phase_weight),
                        "weighted_loss": float(weighted_loss.detach().float().item()),
                        "generated_action_shape": [int(x) for x in generated_np.shape],
                        "generated_action": generated_np.tolist(),
                    }
                )
        runtime = {
            **load_report,
            "elapsed_seconds": float(time.monotonic() - started),
            "cuda_memory_after_eval": cuda_memory(),
        }
        return rows, runtime
    finally:
        shutil.rmtree(work_clip_dir, ignore_errors=True)
        try:
            del model
            gc.collect()
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass


def _default_adapter_dirs(config: OfflineValidationConfig, spec: dict[str, Any]) -> tuple[Path, Path]:
    arms = {arm["role"]: arm["arm_id"] for arm in spec["arms"]}
    step = int(spec["shared_training"]["max_optimizer_steps"])
    primary = config.primary_adapter_dir or (
        config.training_output_root / arms["primary_selected_method"] / "checkpoints" / f"step_{step:04d}" / "adapter"
    )
    ablation = config.ablation_adapter_dir or (
        config.training_output_root / arms["uniform_weight_ablation"] / "checkpoints" / f"step_{step:04d}" / "adapter"
    )
    return primary, ablation


def run_offline_validation(config: OfflineValidationConfig) -> dict[str, Any]:
    _assert_output_path_allowed(config.output_path)
    if not bool(config.local_files_only):
        raise ValueError("R2P-XVLA frozen offline validation requires local_files_only=True; downloads are not allowed")
    if int(config.num_chunks) < 1:
        raise ValueError("num_chunks must be at least 1")
    if int(config.num_chunks) > DEFAULT_NUM_CHUNKS:
        raise ValueError(f"num_chunks must not exceed frozen limit {DEFAULT_NUM_CHUNKS}")

    started = time.monotonic()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path = config.output_path.with_name(config.output_path.stem + "_heartbeat.json")
    status_path = config.output_path.with_name(config.output_path.stem + "_status.json")
    stdout_path = config.output_path.with_name(config.output_path.stem + "_stdout.log")
    stderr_path = config.output_path.with_name(config.output_path.stem + "_stderr.log")
    exit_code_path = config.output_path.with_name(config.output_path.stem + "_exit_code.txt")
    worker_pid_path = config.output_path.with_name(config.output_path.stem + "_worker.pid")
    stdout_path.write_text(
        json.dumps({"event": "r2p_xvla_offline_validation_start", "pid": os.getpid(), "time_unix": time.time()}, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    stderr_path.write_text("", encoding="utf-8")
    exit_code_path.write_text("RUNNING\n", encoding="utf-8")
    worker_pid_path.write_text(str(os.getpid()) + "\n", encoding="utf-8")
    result: dict[str, Any] = {
        "schema_version": "2026-07-18.epoch5_R2P_XVLA_task5_offline_validation.v1",
        "method": "R2P-XVLA",
        "status": "RUNNING",
        "success": False,
        "decision": "R2P_XVLA_OFFLINE_VALIDATION_RUNNING",
        "git_commit": _git_commit(),
        "spec_path": str(config.spec_path),
        "output_path": str(config.output_path),
        "stdout_path": str(stdout_path),
        "stderr_path": str(stderr_path),
        "exit_code_path": str(exit_code_path),
        "worker_pid_path": str(worker_pid_path),
        "worker_pid": os.getpid(),
        "training_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "closed_loop_ours_evaluation_happened": False,
        "simulator_rollout_happened": False,
        "local_files_only": bool(config.local_files_only),
        "denoise_steps": int(config.denoise_steps),
        "nvidia_smi_before": nvidia_smi(),
    }
    _write_json(status_path, result)
    _write_json(heartbeat_path, {"status": "select_validation_clips", "pid": os.getpid(), "time_unix": time.time()})
    try:
        spec = _load_spec(config.spec_path)
        primary_arm = _arm_by_id(spec, next(arm["arm_id"] for arm in spec["arms"] if arm["role"] == "primary_selected_method"))
        ablation_arm = _arm_by_id(spec, next(arm["arm_id"] for arm in spec["arms"] if arm["role"] == "uniform_weight_ablation"))
        primary_adapter, ablation_adapter = _default_adapter_dirs(config, spec)
        if not primary_adapter.exists():
            raise FileNotFoundError(f"missing primary adapter: {primary_adapter}")
        if not ablation_adapter.exists():
            raise FileNotFoundError(f"missing ablation adapter: {ablation_adapter}")
        clips = select_fixed_validation_clips(
            hdf5_path=config.hdf5_path,
            spec=spec,
            num_chunks=int(config.num_chunks),
            clip_steps=int(config.clip_steps),
        )
        policies = [
            (PRIOR_LABEL, None),
            (PRIMARY_LABEL, primary_adapter),
            (UNIFORM_LABEL, ablation_adapter),
        ]
        all_rows: dict[str, list[dict[str, Any]]] = {}
        runtimes: dict[str, dict[str, Any]] = {}
        for label, adapter in policies:
            _write_json(
                heartbeat_path,
                {
                    "status": f"evaluating_{label}",
                    "pid": os.getpid(),
                    "time_unix": time.time(),
                    "completed_policies": sorted(all_rows),
                },
            )
            rows, runtime = _evaluate_policy_rows(
                label=label,
                adapter_dir=adapter,
                evaluation_arm=primary_arm,
                clips=clips,
                config=config,
                spec=spec,
            )
            all_rows[label] = rows
            runtimes[label] = runtime
            _write_json(status_path, {**result, "status": "RUNNING", "completed_policies": sorted(all_rows)})

        summaries = {label: _summarize_policy_rows(rows) for label, rows in all_rows.items()}
        for label in (PRIMARY_LABEL, UNIFORM_LABEL):
            summaries[label]["delta_vs_prior"] = _summarize_action_delta(all_rows[label], all_rows[PRIOR_LABEL])
        selection = compute_offline_selection_decision(spec=spec, summaries=summaries, runtimes=runtimes)
        result.update(
            {
                "status": "COMPLETE",
                "success": bool(selection["passed"]),
                "decision": selection["decision"],
                "num_chunks": int(config.num_chunks),
                "validation_phase_counts": {
                    str(phase): int(sum(1 for clip in clips if str(clip["phase_label"]) == str(phase)))
                    for phase in PHASE_LOSS_WEIGHTS
                },
                "clips": clips,
                "adapter_dirs": {"primary": str(primary_adapter), "ablation": str(ablation_adapter)},
                "evaluation_phase_loss_weights": dict(primary_arm["phase_loss_weights"]),
                "primary_training_phase_loss_weights": dict(primary_arm["phase_loss_weights"]),
                "uniform_training_phase_loss_weights": dict(ablation_arm["phase_loss_weights"]),
                "summaries": summaries,
                "runtimes": runtimes,
                "offline_selection": selection,
                "cuda_peak_mib": float(selection["cuda_peak_mib"]),
                "elapsed_seconds": float(time.monotonic() - started),
                "nvidia_smi_after": nvidia_smi(),
            }
        )
    except Exception as exc:  # pragma: no cover - runtime boundary
        exit_code_path.write_text("1\n", encoding="utf-8")
        stderr_path.write_text(traceback.format_exc() + "\n", encoding="utf-8")
        result.update(
            {
                "status": "FAILED",
                "success": False,
                "decision": "R2P_XVLA_OFFLINE_VALIDATION_FAILED",
                "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                "elapsed_seconds": float(time.monotonic() - started),
                "nvidia_smi_after": nvidia_smi(),
            }
        )
    finally:
        if result.get("status") == "COMPLETE":
            exit_code_path.write_text("0\n", encoding="utf-8")
        _write_json(config.output_path, result)
        _write_json(status_path, result)
        _write_json(
            heartbeat_path,
            {
                "status": str(result["status"]).lower(),
                "pid": os.getpid(),
                "success": bool(result.get("success", False)),
                "decision": result.get("decision"),
                "result_path": str(config.output_path),
                "time_unix": time.time(),
            },
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_ARTIFACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--training-output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--xvla-root", type=Path, default=DEFAULT_XVLA_ROOT)
    parser.add_argument("--hdf5-path", type=Path, default=Path(TASK5_HDF5_WSL))
    parser.add_argument("--primary-adapter-dir", type=Path, default=None)
    parser.add_argument("--ablation-adapter-dir", type=Path, default=None)
    parser.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--clip-steps", type=int, default=DEFAULT_CLIP_STEPS)
    args = parser.parse_args(argv)
    report = run_offline_validation(
        OfflineValidationConfig(
            spec_path=args.spec,
            output_path=args.output,
            training_output_root=args.training_output_root,
            xvla_root=args.xvla_root,
            hdf5_path=args.hdf5_path,
            primary_adapter_dir=args.primary_adapter_dir,
            ablation_adapter_dir=args.ablation_adapter_dir,
            num_chunks=int(args.num_chunks),
            device_index=int(args.device_index),
            denoise_steps=int(args.denoise_steps),
            local_files_only=not bool(args.allow_download),
            clip_steps=int(args.clip_steps),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return 0 if report.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
