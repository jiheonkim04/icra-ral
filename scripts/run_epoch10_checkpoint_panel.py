"""Train the frozen Epoch 10 prospective SmolVLA checkpoint panel.

This is panel construction, not a scientific method.  Each seed is trained once
with the verified standard rank-4 LoRA recipe and saved at the preregistered
steps 10, 30, and 100.  Development/holdout membership is assigned by whole
seed.  The only post-save evaluation is a single common outcome-blind adapter
validity query; no simulator success is read.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
import traceback
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.official_libero_baseline_scaleup import (
    _add_training_batch_dims,
    _cuda_memory,
    _gradient_summary,
    _loss_from_output,
    _parameter_summary,
    _rss_mb,
    _tensor_devices,
    _tensor_shapes,
    _to_float,
)
from tca_map.smolvla.official_libero_stable_artifact_eval import (
    _evaluate_policy_rows,
    _manifest_samples,
    _read_json,
)


SCHEMA_VERSION = 1
DEFAULT_SEEDS = (101, 202, 303, 404)
DEFAULT_STAGES = (10, 30, 100)
DEVELOPMENT_SEEDS = {101, 202}
HOLDOUT_SEEDS = {303, 404}
REQUIRED_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "training_manifest.json",
    "eval_preprocessor_postprocessor_refs.json",
    "source_repro_lock.yaml",
    "sha256_manifest.json",
)
MAX_RUNTIME_SECONDS = 6 * 60 * 60


class PanelError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _package_versions() -> dict[str, str | None]:
    names = ("torch", "lerobot", "transformers", "peft", "datasets", "numpy", "safetensors")
    result: dict[str, str | None] = {}
    for name in names:
        try:
            result[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            result[name] = None
    return result


def _file_inventory(root: Path) -> dict[str, dict[str, Any]]:
    return {
        path.relative_to(root).as_posix(): {"bytes": path.stat().st_size, "sha256": _sha256(path)}
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "sha256_manifest.json"
    }


def _is_complete(root: Path) -> bool:
    return all((root / name).is_file() and (root / name).stat().st_size > 0 for name in REQUIRED_FILES)


def _parse_ints(text: str) -> tuple[int, ...]:
    values = tuple(int(part.strip()) for part in text.split(",") if part.strip())
    if not values or len(set(values)) != len(values):
        raise PanelError("INVALID_PANEL_SPEC", f"Expected unique nonempty integers, got {text!r}")
    return values


def _partition(seed: int) -> str:
    if seed in DEVELOPMENT_SEEDS:
        return "checkpoint_development_panel"
    if seed in HOLDOUT_SEEDS:
        return "checkpoint_holdout_panel"
    raise PanelError("INVALID_PANEL_SPEC", f"Seed {seed} has no frozen panel partition")


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    seeds = _parse_ints(args.seeds)
    stages = tuple(sorted(_parse_ints(args.stages)))
    if seeds != DEFAULT_SEEDS or stages != DEFAULT_STAGES:
        raise PanelError(
            "FROZEN_PANEL_MISMATCH",
            f"Epoch 10 panel is frozen to seeds={DEFAULT_SEEDS}, stages={DEFAULT_STAGES}; got {seeds}, {stages}",
        )
    if stages[-1] > 100 or stages[0] < 1:
        raise PanelError("INVALID_PANEL_SPEC", f"Stages outside [1, 100]: {stages}")
    required = {
        "checkpoint_path": Path(args.checkpoint_path),
        "dataset_root": Path(args.dataset_root),
        "hf_home": Path(args.hf_home),
        "vlm_root": Path(args.vlm_root),
        "split_manifest": Path(args.split_manifest),
        "source_repro_lock": Path(args.source_repro_lock),
        "metric_protocol": Path(args.metric_protocol),
    }
    missing = [f"{name}={path}" for name, path in required.items() if not path.exists()]
    if missing:
        raise PanelError("MISSING_REQUIRED_ASSET", "; ".join(missing))
    output_root = Path(args.output_root).resolve()
    if output_root == Path(output_root.anchor) or len(output_root.parts) < 4:
        raise PanelError("UNSAFE_OUTPUT_ROOT", f"Refusing broad output path: {output_root}")
    return {
        "seeds": list(seeds),
        "stages": list(stages),
        "partitions": {str(seed): _partition(seed) for seed in seeds},
        "required_paths": {name: str(path.resolve()) for name, path in required.items()},
        "required_hashes": {
            name: _sha256(path) for name, path in required.items() if path.is_file()
        },
        "output_root": str(output_root),
        "checkpoint_count": len(seeds) * len(stages),
        "historical_outcomes_read": False,
        "simulator_outcomes_read": False,
    }


def _save_snapshot(
    *,
    args: argparse.Namespace,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    optimizer: Any,
    seed: int,
    stage: int,
    train_order: list[int],
    loss_curve: list[dict[str, Any]],
    grad_curve: list[dict[str, Any]],
    training_started: float,
    parameter_summary: dict[str, Any],
    device_audit: dict[str, Any],
) -> dict[str, Any]:
    import torch

    destination = Path(args.output_root) / f"seed_{seed}" / f"step_{stage:04d}"
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{destination.name}.tmp_{os.getpid()}_{time.time_ns()}")
    if temporary.exists():
        raise PanelError("CHECKPOINT_BUNDLE_INCOMPLETE", f"Unexpected temporary path: {temporary}")
    temporary.mkdir(parents=True)
    try:
        if hasattr(policy, "peft_config"):
            for peft_config in policy.peft_config.values():
                peft_config.base_model_name_or_path = str(Path(args.checkpoint_path).resolve())
        policy.save_pretrained(temporary)
        if hasattr(policy, "config") and hasattr(policy.config, "save_pretrained"):
            policy.config.save_pretrained(temporary)
        if hasattr(preprocessor, "save_pretrained"):
            preprocessor.save_pretrained(temporary)
        if hasattr(postprocessor, "save_pretrained"):
            postprocessor.save_pretrained(temporary)
        shutil.copy2(Path(args.source_repro_lock), temporary / "source_repro_lock.yaml")
        torch.save(
            {
                "seed": seed,
                "optimizer_step": stage,
                "python_random_state_repr": repr(random.getstate()),
                "numpy_random_state": np.random.get_state(),
                "torch_cpu_rng_state": torch.get_rng_state(),
                "torch_cuda_rng_state_all": torch.cuda.get_rng_state_all(),
                "train_order_first_20": train_order[:20],
            },
            temporary / "rng_state.pt",
        )
        torch.save(optimizer.state_dict(), temporary / "optimizer_state.pt")
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "campaign": "epoch10_icae_vla",
            "status": "CHECKPOINT_TRAINED_SAVED_PENDING_RELOAD",
            "policy_identity": f"epoch10_rank4_seed_{seed}_step_{stage:04d}",
            "lineage_cluster": f"epoch10_rank4_seed_{seed}",
            "panel_partition": _partition(seed),
            "seed": seed,
            "optimizer_step": stage,
            "predeclared_stage": {10: "early", 30: "intermediate", 100: "converged"}[stage],
            "base_model_path": str(Path(args.checkpoint_path).resolve()),
            "dataset_path": str(Path(args.dataset_root).resolve()),
            "lora_rank": 4,
            "target_modules": [
                r"model\.vlm_with_expert\.lm_expert\..*\.(q|v)_proj",
                r"model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out)",
            ],
            "optimizer": "torch.optim.AdamW",
            "learning_rate": args.learning_rate,
            "batch_size": 1,
            "scheduler": None,
            "precision_autocast": "none",
            "trainable_parameter_count": parameter_summary["trainable_params"],
            "total_parameter_count": parameter_summary["total_params"],
            "loss_curve_through_stage": loss_curve[:stage],
            "last_gradient_summary": grad_curve[stage - 1],
            "elapsed_seconds_at_save": round(time.monotonic() - training_started, 3),
            "source_commit": _git_head(),
            "source_files": {
                "verified_recipe_sha256": "343e6465dbfa3ff52606d3e414e8334f61db18e407705ba84d57ca9a5ba9b076",
                "split_manifest_sha256": _sha256(Path(args.split_manifest)),
                "repro_lock_sha256": _sha256(Path(args.source_repro_lock)),
                "metric_protocol_sha256": _sha256(Path(args.metric_protocol)),
            },
            "package_versions": _package_versions(),
            "device_audit": device_audit,
            "official_outcomes_opened": False,
            "exact_command": " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]]),
        }
        _write_json(temporary / "training_manifest.json", manifest)
        _write_json(
            temporary / "eval_preprocessor_postprocessor_refs.json",
            {
                "preprocessor_config": "policy_preprocessor.json",
                "postprocessor_config": "policy_postprocessor.json",
                "official_device_processor": "cuda",
                "custom_normalizer_involved": False,
                "custom_action_adapter_involved": False,
            },
        )
        inventory = _file_inventory(temporary)
        _write_json(
            temporary / "sha256_manifest.json",
            {
                "schema_version": SCHEMA_VERSION,
                "policy_identity": manifest["policy_identity"],
                "files": inventory,
            },
        )
        missing = [name for name in REQUIRED_FILES if not (temporary / name).is_file()]
        if missing:
            raise PanelError("CHECKPOINT_BUNDLE_INCOMPLETE", f"Missing files at {temporary}: {missing}")
        new_hash = _sha256(temporary / "adapter_model.safetensors")
        if destination.exists():
            if not _is_complete(destination):
                raise PanelError("CHECKPOINT_BUNDLE_INCOMPLETE", f"Existing destination is incomplete: {destination}")
            old_hash = _sha256(destination / "adapter_model.safetensors")
            if old_hash != new_hash:
                raise PanelError(
                    "NONDETERMINISTIC_RESUME_MISMATCH",
                    f"Existing {destination} hash {old_hash} != deterministic rerun {new_hash}",
                )
            shutil.rmtree(temporary)
            status = "EXISTING_COMPLETE_BUNDLE_DETERMINISTICALLY_VERIFIED"
        else:
            temporary.rename(destination)
            status = "CHECKPOINT_SAVED_PENDING_RELOAD"
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "policy_identity": f"epoch10_rank4_seed_{seed}_step_{stage:04d}",
        "lineage_cluster": f"epoch10_rank4_seed_{seed}",
        "panel_partition": _partition(seed),
        "path": str(destination.resolve()),
        "status": status,
        "adapter_sha256": _sha256(destination / "adapter_model.safetensors"),
        "adapter_bytes": (destination / "adapter_model.safetensors").stat().st_size,
        "bundle_file_count": sum(1 for path in destination.rglob("*") if path.is_file()),
    }


def _reload_smoke(
    *,
    args: argparse.Namespace,
    bundle: dict[str, Any],
    dataset: Any,
    smoke_sample: dict[str, Any],
    action_min: np.ndarray,
    action_max: np.ndarray,
    started: float,
) -> dict[str, Any]:
    import torch
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import PeftConfig, PeftModel

    checkpoint_path = Path(args.checkpoint_path)
    cfg = PreTrainedConfig.from_pretrained(checkpoint_path, local_files_only=True, cache_dir=args.hf_home)
    cfg.device = "cuda"
    cfg.load_vlm_weights = True
    cfg.compile_model = False
    cfg.push_to_hub = False
    cfg.vlm_model_name = str(Path(args.vlm_root))
    cfg.chunk_size = args.chunk_size
    base = SmolVLAPolicy.from_pretrained(
        checkpoint_path,
        config=cfg,
        local_files_only=True,
        cache_dir=args.hf_home,
        token=False,
        strict=False,
    )
    peft_config = PeftConfig.from_pretrained(bundle["path"])
    policy = PeftModel.from_pretrained(
        base,
        bundle["path"],
        config=peft_config,
        is_trainable=False,
        local_files_only=True,
    )
    policy.to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(Path(args.vlm_root))},
            "device_processor": {"device": "cuda"},
        },
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    rows = _evaluate_policy_rows(
        policy=policy,
        preprocessor=preprocessor,
        postprocessor=postprocessor,
        dataset=dataset,
        samples=[smoke_sample],
        action_min=action_min,
        action_max=action_max,
        include_eval_loss=False,
        label=bundle["policy_identity"],
        started=started,
        progress_every=0,
    )
    preview = [float(value) for value in rows[0]["pred_preview"]]
    parameter_summary = _parameter_summary(policy)
    result = {
        "loaded_from_disk": True,
        "smoke_sample_id": smoke_sample["sample_id"],
        "action_shape": [len(preview)],
        "action_finite": all(math.isfinite(value) for value in preview),
        "action_max_abs": max(abs(value) for value in preview),
        "model_parameter_device": parameter_summary["first_parameter_device"],
        "model_parameter_dtype": parameter_summary["first_parameter_dtype"],
        "scientific_performance_screen": False,
        "simulator_outcome_read": False,
    }
    del policy, base, preprocessor, postprocessor
    torch.cuda.empty_cache()
    if not result["action_finite"] or result["action_shape"] != [7]:
        raise PanelError("CHECKPOINT_LOAD_FAILED", f"Invalid smoke action for {bundle['policy_identity']}: {result}")
    return result


def _train_seed(
    *,
    args: argparse.Namespace,
    seed: int,
    stages: tuple[int, ...],
    manifest: dict[str, Any],
    started: float,
) -> dict[str, Any]:
    import psutil
    import torch
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not torch.cuda.is_available():
        raise PanelError("CPU_FALLBACK_BUG", "CUDA unavailable; refusing prospective panel training on CPU")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device_index = torch.cuda.current_device()
    torch.cuda.reset_peak_memory_stats(device_index)
    checkpoint_path = Path(args.checkpoint_path)
    dataset_root = Path(args.dataset_root)
    info = _read_json(dataset_root / "meta" / "info.json")
    stats = _read_json(dataset_root / "meta" / "stats.json")
    selected_episodes, split_samples, _ = _manifest_samples(manifest)
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=dataset_root,
        episodes=selected_episodes,
        delta_timestamps={"action": [index / float(info["fps"]) for index in range(args.chunk_size)]},
        video_backend=args.video_backend,
    )
    cfg = PreTrainedConfig.from_pretrained(checkpoint_path, local_files_only=True, cache_dir=args.hf_home)
    cfg.device = "cuda"
    cfg.load_vlm_weights = True
    cfg.compile_model = False
    cfg.push_to_hub = False
    cfg.vlm_model_name = str(Path(args.vlm_root))
    cfg.chunk_size = args.chunk_size
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint_path,
        config=cfg,
        local_files_only=True,
        cache_dir=args.hf_home,
        token=False,
        strict=False,
    )
    policy.to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(checkpoint_path),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(Path(args.vlm_root))},
            "device_processor": {"device": "cuda"},
        },
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    probe = _add_training_batch_dims(preprocessor(dataset[int(split_samples["train"][0]["dataset_local_index"])]))
    if not all(device.startswith("cuda") for device in _tensor_devices(probe).values()):
        raise PanelError("CPU_FALLBACK_BUG", "Training probe tensors are not all on CUDA")
    policy = policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": 4})
    policy.to("cuda").train()
    parameter_summary = _parameter_summary(policy)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in policy.parameters() if parameter.requires_grad],
        lr=args.learning_rate,
    )
    rng = np.random.default_rng(seed)
    train_order = rng.permutation(len(split_samples["train"])).tolist()
    loss_curve: list[dict[str, Any]] = []
    grad_curve: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []
    training_started = time.monotonic()
    peak_host_percent = psutil.virtual_memory().percent
    for step in range(1, stages[-1] + 1):
        if time.monotonic() - started > MAX_RUNTIME_SECONDS:
            raise PanelError("LOCAL_RUNTIME_EXCEEDED", "Panel construction exceeded six hours")
        host_percent = psutil.virtual_memory().percent
        peak_host_percent = max(peak_host_percent, host_percent)
        if host_percent >= 90.0:
            raise PanelError("HOST_RAM_HARD_STOP", f"Host RAM reached {host_percent:.2f}%")
        sample = split_samples["train"][train_order[(step - 1) % len(train_order)]]
        batch = _add_training_batch_dims(preprocessor(dataset[int(sample["dataset_local_index"])]))
        optimizer.zero_grad(set_to_none=True)
        loss = _loss_from_output(policy.forward(batch))
        loss_value = _to_float(loss)
        if not math.isfinite(loss_value):
            raise PanelError("TRAINING_FAILURE", f"Non-finite loss for seed {seed} at step {step}")
        loss.backward()
        gradient = _gradient_summary(policy)
        if int(gradient["nonzero_grad_tensors"]) == 0:
            raise PanelError("TRAINING_FAILURE", f"No nonzero gradients for seed {seed} at step {step}")
        optimizer.step()
        loss_curve.append({"step": step, "loss": loss_value, **_cuda_memory(torch)})
        grad_curve.append({"step": step, **gradient})
        if step in stages:
            device_audit = {
                "cuda_device_name": torch.cuda.get_device_name(device_index),
                "model": _parameter_summary(policy),
                "input_tensor_devices": _tensor_devices(batch),
                "input_tensor_shapes": _tensor_shapes(batch),
                "cuda": _cuda_memory(torch),
                "rss_mb": _rss_mb(),
                "host_ram_percent": host_percent,
            }
            bundles.append(
                _save_snapshot(
                    args=args,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    optimizer=optimizer,
                    seed=seed,
                    stage=step,
                    train_order=train_order,
                    loss_curve=loss_curve,
                    grad_curve=grad_curve,
                    training_started=training_started,
                    parameter_summary=parameter_summary,
                    device_audit=device_audit,
                )
            )
    del policy, optimizer, preprocessor, postprocessor, probe
    torch.cuda.empty_cache()
    action_min = np.asarray(stats["action"]["min"], dtype=np.float32)
    action_max = np.asarray(stats["action"]["max"], dtype=np.float32)
    for bundle in bundles:
        bundle["reload_smoke"] = _reload_smoke(
            args=args,
            bundle=bundle,
            dataset=dataset,
            smoke_sample=split_samples["train"][0],
            action_min=action_min,
            action_max=action_max,
            started=started,
        )
        bundle["status"] = "CHECKPOINT_COMPLETE_VERIFIED"
    return {
        "seed": seed,
        "lineage_cluster": f"epoch10_rank4_seed_{seed}",
        "panel_partition": _partition(seed),
        "training_steps": stages[-1],
        "loss_first": loss_curve[0]["loss"],
        "loss_last": loss_curve[-1]["loss"],
        "last_gradient": grad_curve[-1],
        "training_elapsed_seconds": round(time.monotonic() - training_started, 3),
        "peak_host_ram_percent": peak_host_percent,
        "cuda": _cuda_memory(torch),
        "checkpoints": bundles,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default="C:/assets/checkpoints/smolvla_libero")
    parser.add_argument("--dataset-root", default="C:/assets/datasets/lerobot_libero")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--vlm-root", default="C:/assets/hf_home/HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--split-manifest", default="reports/official_smolvla_split_manifest.json")
    parser.add_argument("--source-repro-lock", default="configs/official_smolvla_repro_lock.yaml")
    parser.add_argument("--metric-protocol", default="reports/official_smolvla_metric_protocol.md")
    parser.add_argument("--output-root", default="C:/assets/checkpoints/epoch10_icae_panel/rank4")
    parser.add_argument("--result-json", default="reports/epoch10_checkpoint_generation_result.json")
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--stages", default=",".join(map(str, DEFAULT_STAGES)))
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args(argv)
    os.environ["HF_HOME"] = str(Path(args.hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    started = time.monotonic()
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "campaign": "epoch10_icae_vla",
        "status": "STARTED",
        "source_commit": _git_head(),
        "official_outcomes_opened": False,
        "simulator_outcomes_read": False,
        "seeds": [],
    }
    exit_code = 0
    try:
        report["preflight"] = _preflight(args)
        if args.preflight_only:
            report["status"] = "PREFLIGHT_PASS"
        else:
            manifest = _read_json(Path(args.split_manifest))
            for seed in DEFAULT_SEEDS:
                print(f"[epoch10-panel] training seed {seed}", flush=True)
                report["seeds"].append(
                    _train_seed(
                        args=args,
                        seed=seed,
                        stages=DEFAULT_STAGES,
                        manifest=manifest,
                        started=started,
                    )
                )
                _write_json(Path(args.result_json), report)
            checkpoints = [
                checkpoint
                for seed_result in report["seeds"]
                for checkpoint in seed_result["checkpoints"]
            ]
            report["checkpoint_count"] = len(checkpoints)
            report["all_disk_reloads_passed"] = all(
                checkpoint["status"] == "CHECKPOINT_COMPLETE_VERIFIED" for checkpoint in checkpoints
            )
            report["status"] = (
                "PROSPECTIVE_CHECKPOINT_PANEL_COMPLETE"
                if report["checkpoint_count"] == 12 and report["all_disk_reloads_passed"]
                else "CHECKPOINT_PANEL_INCOMPLETE"
            )
    except PanelError as exc:
        report["status"] = exc.code
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        exit_code = 2
    except Exception as exc:  # pragma: no cover - integration failure path
        report["status"] = "UNEXPECTED_IMPLEMENTATION_FAILURE"
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        exit_code = 3
    report["elapsed_seconds"] = round(time.monotonic() - started, 3)
    report["package_versions"] = _package_versions()
    _write_json(Path(args.result_json), report)
    print(json.dumps({"status": report["status"], "elapsed_seconds": report["elapsed_seconds"]}, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
