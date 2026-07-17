"""Offline validation gate for frozen BR-XVLA training arms.

The gate reloads the saved PEFT adapters from disk, compares them against the
cached X-VLA-Libero prior on fixed validation chunks, and records only offline
loss/action-delta metrics.  It performs no closed-loop Ours evaluation.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.xvla_task1.data_adapter_smoke import DEFAULT_XVLA_ROOT
from tca_map.xvla_task1.gradient_smoke import (
    cuda_memory,
    install_optional_server_import_shims,
    install_xvla_transformers_compat_patches,
    nvidia_smi,
    package_version,
    prepare_inputs,
)
from tca_map.xvla_task1.train_lora import (
    DEFAULT_CLIP_STEPS,
    DEFAULT_OUTPUT_ROOT,
    XVLA_CACHE_DIR,
    _arm_by_id,
    _first_xvla_reader_sample,
    _git_commit,
    _json_default,
    _load_spec,
    _phase_for_step,
    _prepare_xvla_imports,
    _write_json,
    build_phase_clip_index,
    materialize_xvla_clip,
    select_clip_for_step,
)
from tca_map.xvla_task1.training_spec import MODEL_ID, MODEL_REVISION, SPEC_ARTIFACT, TASK1_HDF5_WSL

DEFAULT_OUTPUT = Path("runs/xvla_prior/epoch5_br_xvla_offline_validation_step0064.json")


@dataclass(frozen=True)
class OfflineValidationConfig:
    spec_path: Path = SPEC_ARTIFACT
    output_path: Path = DEFAULT_OUTPUT
    training_output_root: Path = DEFAULT_OUTPUT_ROOT
    xvla_root: Path = DEFAULT_XVLA_ROOT
    hdf5_path: Path = Path(TASK1_HDF5_WSL)
    primary_adapter_dir: Path | None = None
    ablation_adapter_dir: Path | None = None
    num_chunks: int = 24
    device_index: int = 0
    denoise_steps: int = 10
    local_files_only: bool = True
    clip_steps: int = DEFAULT_CLIP_STEPS


def select_fixed_validation_clips(
    *,
    hdf5_path: Path,
    spec: dict[str, Any],
    num_chunks: int,
    clip_steps: int = DEFAULT_CLIP_STEPS,
) -> list[dict[str, Any]]:
    """Select deterministic validation clips using the frozen phase cycle."""

    threshold = float(spec["data"]["phase_state_layout"]["basket_xy_threshold"])
    grouped = build_phase_clip_index(
        hdf5_path,
        demo_indices=list(spec["data"]["validation_demo_indices"]),
        clip_steps=int(clip_steps),
        basket_xy_threshold=threshold,
    )
    rng = np.random.default_rng(int(spec["shared_training"]["seed"]) + 1)
    cycle = list(spec["arms"][0]["sampler"]["cycle_phase_counts"])
    clips = [
        select_clip_for_step(grouped, cycle=cycle, step_index_zero_based=index, rng=rng)
        for index in range(int(num_chunks))
    ]
    for index, clip in enumerate(clips):
        clip["validation_index"] = int(index)
    return clips


def _summarize_policy_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    losses = np.asarray([row["loss_total"] for row in rows], dtype=np.float64)
    weighted = np.asarray([row["weighted_loss"] for row in rows], dtype=np.float64)
    summary: dict[str, Any] = {
        "count": int(len(rows)),
        "mean_loss_total": float(np.mean(losses)),
        "median_loss_total": float(np.median(losses)),
        "mean_weighted_loss": float(np.mean(weighted)),
        "all_losses_finite": bool(np.isfinite(losses).all() and np.isfinite(weighted).all()),
    }
    for phase in (0, 1, 2):
        phase_rows = [row for row in rows if int(row["phase_count_in_basket"]) == phase]
        phase_losses = np.asarray([row["loss_total"] for row in phase_rows], dtype=np.float64)
        summary[f"phase_{phase}_count"] = int(len(phase_rows))
        summary[f"phase_{phase}_mean_loss_total"] = float(np.mean(phase_losses)) if phase_losses.size else None
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


def _load_policy(config: OfflineValidationConfig, adapter_dir: Path | None) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from peft import PeftModel

    import_report = _prepare_xvla_imports(config.xvla_root)
    # Keep these explicit for provenance when _prepare_xvla_imports becomes a no-op
    # because modules were already imported in a previous policy pass.
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
        raise RuntimeError("CUDA unavailable for BR-XVLA offline validation")
    torch.cuda.set_device(int(config.device_index))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=int(config.device_index))
    device = torch.device(f"cuda:{int(config.device_index)}")
    model = XVLA.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        local_files_only=bool(config.local_files_only),
        cache_dir=XVLA_CACHE_DIR,
    )
    if adapter_dir is not None:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=False)
    processor = XVLAProcessor.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        trust_remote_code=True,
        local_files_only=bool(config.local_files_only),
        cache_dir=XVLA_CACHE_DIR,
    )
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
    phase_weight_lambda: float,
    clips: list[dict[str, Any]],
    config: OfflineValidationConfig,
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
                materialized = materialize_xvla_clip(config.hdf5_path, work_clip_dir, clip)
                sample = _first_xvla_reader_sample(config.xvla_root, Path(materialized["meta_path"]))
                inputs = prepare_inputs(sample, processor, device, torch.float32)
                shutil.rmtree(work_clip_dir, ignore_errors=True)
                seed = int(20260717 + int(clip["validation_index"]))
                torch.manual_seed(seed)
                loss_dict = model(**inputs)
                loss_total = sum(loss_dict.values())
                phase_weight = 1.0 + float(phase_weight_lambda) * float(int(clip["phase_count_in_basket"]) == 1)
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
                        "clip": {
                            key: clip[key]
                            for key in (
                                "demo_index",
                                "demo_name",
                                "source_start_index",
                                "source_end_index",
                                "phase_count_in_basket",
                            )
                        },
                        "phase_count_in_basket": int(clip["phase_count_in_basket"]),
                        "losses": {key: float(value.detach().float().item()) for key, value in loss_dict.items()},
                        "loss_total": float(loss_total.detach().float().item()),
                        "phase_weight": float(phase_weight),
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
    started = time.monotonic()
    config.output_path.parent.mkdir(parents=True, exist_ok=True)
    heartbeat_path = config.output_path.with_name(config.output_path.stem + "_heartbeat.json")
    status_path = config.output_path.with_name(config.output_path.stem + "_status.json")
    result: dict[str, Any] = {
        "schema_version": "2026-07-17.epoch5_br_xvla_offline_validation.v1",
        "method": "BR-XVLA",
        "status": "RUNNING",
        "success": False,
        "decision": "BR_XVLA_OFFLINE_VALIDATION_RUNNING",
        "git_commit": _git_commit(),
        "spec_path": str(config.spec_path),
        "output_path": str(config.output_path),
        "worker_pid": os.getpid(),
        "closed_loop_ours_evaluation_happened": False,
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
            ("xvla_prior_base", None, 0.0),
            ("br_xvla_primary", primary_adapter, float(primary_arm["phase_weight_lambda"])),
            ("uniform_xvla_ablation", ablation_adapter, float(ablation_arm["phase_weight_lambda"])),
        ]
        all_rows: dict[str, list[dict[str, Any]]] = {}
        runtimes: dict[str, dict[str, Any]] = {}
        for label, adapter, phase_lambda in policies:
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
                phase_weight_lambda=float(phase_lambda),
                clips=clips,
                config=config,
            )
            all_rows[label] = rows
            runtimes[label] = runtime
            _write_json(status_path, {**result, "status": "RUNNING", "completed_policies": sorted(all_rows)})

        summaries = {label: _summarize_policy_rows(rows) for label, rows in all_rows.items()}
        for label in ("br_xvla_primary", "uniform_xvla_ablation"):
            summaries[label]["delta_vs_prior"] = _summarize_action_delta(all_rows[label], all_rows["xvla_prior_base"])

        prior = summaries["xvla_prior_base"]
        primary = summaries["br_xvla_primary"]
        ablation = summaries["uniform_xvla_ablation"]
        criteria = spec["validation_selection"]["offline_pass_criteria"]
        primary_clean_values = [
            value
            for value in (primary["phase_0_mean_loss_total"], primary["phase_2_mean_loss_total"])
            if value is not None
        ]
        prior_clean_values = [
            value for value in (prior["phase_0_mean_loss_total"], prior["phase_2_mean_loss_total"]) if value is not None
        ]
        primary_clean = float(np.mean(primary_clean_values))
        prior_clean = float(np.mean(prior_clean_values))
        clean_degradation = float((primary_clean - prior_clean) / max(prior_clean, 1e-8))
        primary_delta = primary["delta_vs_prior"]
        cuda_peak = max(
            float(runtime.get("cuda_memory_after_eval", {}).get("max_allocated_mib", 0.0)) for runtime in runtimes.values()
        )
        primary_pass = bool(
            primary["all_losses_finite"]
            and float(primary["phase_1_mean_loss_total"]) <= float(prior["phase_1_mean_loss_total"])
            and clean_degradation <= float(criteria["clean_phase_validation_loss_relative_degradation_max"])
            and float(primary_delta["fixed_chunk_mean_abs_action_delta"])
            <= float(criteria["fixed_chunk_mean_abs_action_delta_max"])
            and float(primary_delta["fixed_chunk_max_abs_action_delta"]) <= float(criteria["fixed_chunk_max_abs_action_delta_max"])
            and cuda_peak <= float(criteria["cuda_peak_mib_max"])
        )
        primary_beats_ablation = bool(float(primary["phase_1_mean_loss_total"]) < float(ablation["phase_1_mean_loss_total"]))
        decision = (
            "BR_XVLA_OFFLINE_PASS_BEATS_ABLATION"
            if primary_pass and primary_beats_ablation
            else "BR_XVLA_OFFLINE_SELECTION_NOT_PASSED"
        )
        result.update(
            {
                "status": "COMPLETE",
                "success": bool(primary_pass and primary_beats_ablation),
                "decision": decision,
                "num_chunks": int(config.num_chunks),
                "validation_phase_counts": {
                    str(phase): int(sum(1 for clip in clips if int(clip["phase_count_in_basket"]) == phase))
                    for phase in (0, 1, 2)
                },
                "clips": clips,
                "adapter_dirs": {"primary": str(primary_adapter), "ablation": str(ablation_adapter)},
                "summaries": summaries,
                "runtimes": runtimes,
                "primary_clean_phase_relative_degradation_vs_prior": clean_degradation,
                "primary_passes_offline_gate": bool(primary_pass),
                "primary_beats_uniform_ablation_on_phase1_loss": bool(primary_beats_ablation),
                "cuda_peak_mib": cuda_peak,
                "elapsed_seconds": float(time.monotonic() - started),
                "nvidia_smi_after": nvidia_smi(),
            }
        )
    except Exception as exc:  # pragma: no cover - runtime boundary
        result.update(
            {
                "status": "FAILED",
                "success": False,
                "decision": "BR_XVLA_OFFLINE_VALIDATION_FAILED",
                "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                "elapsed_seconds": float(time.monotonic() - started),
                "nvidia_smi_after": nvidia_smi(),
            }
        )
    finally:
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
    parser.add_argument("--hdf5-path", type=Path, default=Path(TASK1_HDF5_WSL))
    parser.add_argument("--primary-adapter-dir", type=Path, default=None)
    parser.add_argument("--ablation-adapter-dir", type=Path, default=None)
    parser.add_argument("--num-chunks", type=int, default=24)
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
