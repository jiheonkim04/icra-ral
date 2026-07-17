"""One-batch no-optimizer BR-XVLA gradient smoke.

This gate verifies that the selected BR-XVLA path can load the cached X-VLA
prior, attach the official PEFT LoRA configuration, consume the local
X-VLA-format task-1 adapter, compute a weighted supervised loss on a
one-target-remaining clip, and backpropagate finite nonzero adapter gradients.

It intentionally does not create an optimizer, call optimizer.step, write a
checkpoint, or run closed-loop evaluation.
"""

from __future__ import annotations

import argparse
import gc
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from tca_map.xvla_task1.data_adapter_smoke import (
    DEFAULT_HDF5_PATH,
    DEFAULT_XVLA_ROOT,
    TASK_DESCRIPTION,
    _write_encoded_rgb_frames,
    build_abs_action_6d,
    smoke_xvla_reader,
)
from tca_map.xvla_task1.training_spec import MODEL_ID, MODEL_REVISION

CREAM_CHEESE_POS_SLICE = slice(17, 20)
BUTTER_POS_SLICE = slice(52, 55)
BASKET_POS_SLICE = slice(59, 62)
DEFAULT_CLIP_STEPS = 96


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=10,
        ).strip()
    except Exception as exc:  # pragma: no cover - runtime boundary
        return f"nvidia_smi_failed: {type(exc).__name__}: {exc}"


def cuda_memory() -> dict[str, Any]:
    if not torch.cuda.is_available():
        return {"available": False}
    return {
        "available": True,
        "allocated_mib": round(float(torch.cuda.memory_allocated()) / (1024**2), 3),
        "max_allocated_mib": round(float(torch.cuda.max_memory_allocated()) / (1024**2), 3),
        "reserved_mib": round(float(torch.cuda.memory_reserved()) / (1024**2), 3),
        "max_reserved_mib": round(float(torch.cuda.max_memory_reserved()) / (1024**2), 3),
    }


def target_count_in_basket(states: np.ndarray, threshold: float) -> np.ndarray:
    states = np.asarray(states, dtype=np.float64)
    cream = states[:, CREAM_CHEESE_POS_SLICE]
    butter = states[:, BUTTER_POS_SLICE]
    basket = states[:, BASKET_POS_SLICE]
    cream_xy = np.linalg.norm(cream[:, :2] - basket[:, :2], axis=1)
    butter_xy = np.linalg.norm(butter[:, :2] - basket[:, :2], axis=1)
    return (cream_xy <= threshold).astype(np.int64) + (butter_xy <= threshold).astype(np.int64)


def select_one_target_clip_start(states: np.ndarray, threshold: float, clip_steps: int) -> int:
    counts = target_count_in_basket(states, threshold)
    candidates = np.flatnonzero(counts == 1)
    if candidates.size == 0:
        raise ValueError("no one-target-remaining phase found")
    max_start = max(0, int(states.shape[0]) - int(clip_steps))
    viable = [int(index) for index in candidates if int(index) <= max_start]
    return viable[0] if viable else int(min(int(candidates[0]), max_start))


def materialize_one_target_clip(
    source_hdf5: Path,
    output_dir: Path,
    *,
    demo_name: str,
    threshold: float,
    clip_steps: int,
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    converted = output_dir / "converted_hdf5"
    converted.mkdir(parents=True, exist_ok=True)
    out_hdf5 = converted / f"{demo_name}_one_target_clip.hdf5"

    with h5py.File(source_hdf5, "r") as source:
        demo = source["data"][demo_name]
        states = np.asarray(demo["states"], dtype=np.float64)
        start = select_one_target_clip_start(states, threshold, clip_steps)
        end = min(int(states.shape[0]), int(start) + int(clip_steps))
        actions = np.asarray(demo["actions"][start:end], dtype=np.float64)
        robot_states = np.asarray(demo["robot_states"][start:end], dtype=np.float64)
        clip_states = np.asarray(demo["states"][start:end], dtype=np.float64)
        agentview = np.asarray(demo["obs"]["agentview_rgb"][start:end], dtype=np.uint8)
        wrist = np.asarray(demo["obs"]["eye_in_hand_rgb"][start:end], dtype=np.uint8)

    counts = target_count_in_basket(clip_states, threshold)
    abs_action_6d = build_abs_action_6d(robot_states, actions)
    with h5py.File(out_hdf5, "w") as target:
        target.create_dataset("abs_action_6d", data=abs_action_6d, compression="gzip")
        _write_encoded_rgb_frames(target, "agentview_rgb", agentview)
        _write_encoded_rgb_frames(target, "eye_in_hand_rgb", wrist)
        target.create_dataset("language_instruction", data=np.bytes_(TASK_DESCRIPTION))

    meta = {
        "dataset_name": "libero",
        "datalist": [str(out_hdf5)],
        "observation_key": ["agentview_rgb", "eye_in_hand_rgb"],
        "language_instruction_key": "language_instruction",
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "meta_path": str(meta_path),
        "clip_hdf5": str(out_hdf5),
        "source_demo": demo_name,
        "source_start_index": int(start),
        "source_end_index": int(end),
        "clip_steps": int(end - start),
        "target_count_first_40": [int(x) for x in counts[:40]],
        "first_count": int(counts[0]),
        "one_target_fraction": float(np.mean(counts == 1)),
        "abs_action_6d_shape": [int(x) for x in abs_action_6d.shape],
    }


def _to_model(value: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if value.is_floating_point():
        return value.to(device=device, dtype=dtype)
    return value.to(device=device)


def prepare_inputs(sample: dict[str, Any], processor: Any, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    lang = processor.encode_language(str(sample["language_instruction"]))
    inputs = {
        "input_ids": lang["input_ids"],
        "image_input": sample["image_input"].unsqueeze(0),
        "image_mask": sample["image_mask"].unsqueeze(0),
        "domain_id": sample["domain_id"].view(1),
        "proprio": sample["proprio"].unsqueeze(0),
        "action": sample["action"].unsqueeze(0),
    }
    return {key: _to_model(value, device, dtype) for key, value in inputs.items()}


def gradient_summary(model: torch.nn.Module) -> dict[str, Any]:
    trainable = 0
    grad_tensors = 0
    finite_grad_tensors = 0
    nonzero_grad_tensors = 0
    grad_sq = 0.0
    trainable_names_first_20: list[str] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable += int(param.numel())
        if len(trainable_names_first_20) < 20:
            trainable_names_first_20.append(str(name))
        if param.grad is None:
            continue
        grad_tensors += 1
        grad = param.grad.detach().float()
        finite = bool(torch.isfinite(grad).all().item())
        finite_grad_tensors += int(finite)
        nonzero = bool(torch.count_nonzero(grad).item() > 0)
        nonzero_grad_tensors += int(nonzero)
        if finite:
            grad_sq += float(torch.sum(grad * grad).item())
    return {
        "trainable_parameter_count": int(trainable),
        "grad_tensor_count": int(grad_tensors),
        "finite_grad_tensor_count": int(finite_grad_tensors),
        "nonzero_grad_tensor_count": int(nonzero_grad_tensors),
        "gradient_global_norm": float(grad_sq**0.5),
        "trainable_names_first_20": trainable_names_first_20,
    }


def _module_is_available(name: str) -> bool:
    try:
        return importlib.util.find_spec(name) is not None
    except ValueError as exc:
        # importlib raises here when a previous local shim left
        # sys.modules[name].__spec__ unset. Treat that as unavailable so we
        # replace it with an importlib-compliant shim below.
        module = sys.modules.get(name)
        if module is not None and getattr(module, "__spec__", None) is None:
            return False
        raise exc


def _make_import_shim(name: str, *, is_package: bool = False) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None, is_package=is_package)
    if is_package:
        module.__path__ = []  # type: ignore[attr-defined]
    return module


def install_optional_server_import_shims() -> list[str]:
    """Shim X-VLA's unused serving dependencies when absent.

    `models.modeling_xvla` imports FastAPI/uvicorn/json_numpy at module import
    time for its `.run()` serving method. The gradient smoke never calls that
    path, so a tiny shim is enough to let the training graph import without
    installing optional server packages.
    """

    used: list[str] = []
    if not _module_is_available("fastapi"):
        fastapi = _make_import_shim("fastapi", is_package=True)

        class FastAPI:  # noqa: D401 - tiny optional dependency shim
            def post(self, *_args: Any, **_kwargs: Any) -> Any:
                def decorator(fn: Any) -> Any:
                    return fn

                return decorator

        fastapi.FastAPI = FastAPI  # type: ignore[attr-defined]
        responses = _make_import_shim("fastapi.responses")

        class JSONResponse(dict):
            def __init__(self, content: Any = None, status_code: int = 200, **kwargs: Any) -> None:
                super().__init__(content=content, status_code=status_code, **kwargs)

        responses.JSONResponse = JSONResponse  # type: ignore[attr-defined]
        fastapi.responses = responses  # type: ignore[attr-defined]
        sys.modules["fastapi"] = fastapi
        sys.modules["fastapi.responses"] = responses
        used.append("fastapi")
    if not _module_is_available("uvicorn"):
        uvicorn = _make_import_shim("uvicorn")

        def run(*_args: Any, **_kwargs: Any) -> None:
            raise RuntimeError("uvicorn shim is import-only for gradient smoke")

        uvicorn.run = run  # type: ignore[attr-defined]
        sys.modules["uvicorn"] = uvicorn
        used.append("uvicorn")
    if not _module_is_available("json_numpy"):
        json_numpy = _make_import_shim("json_numpy")
        json_numpy.loads = json.loads  # type: ignore[attr-defined]
        json_numpy.dumps = json.dumps  # type: ignore[attr-defined]
        sys.modules["json_numpy"] = json_numpy
        used.append("json_numpy")
    return used


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def install_xvla_transformers_compat_patches() -> list[str]:
    """Install import-time compatibility patches for local X-VLA + Transformers.

    The checked-out X-VLA Florence2 class targets an older Transformers API that
    did not require `_supports_sdpa` on every `PreTrainedModel` subclass.
    Current local Transformers probes that attribute during model
    construction. Setting it to False preserves the conservative eager-attn
    path and does not change BR-XVLA's scientific objective or data path.
    """

    patches: list[str] = []
    from models import modeling_florence2  # type: ignore

    florence_cls = modeling_florence2.Florence2ForConditionalGeneration
    if "_supports_sdpa" not in getattr(florence_cls, "__dict__", {}):
        florence_cls._supports_sdpa = False
        patches.append("Florence2ForConditionalGeneration._supports_sdpa=False")

    language_cls = modeling_florence2.Florence2LanguageForConditionalGeneration
    if not getattr(language_cls.get_output_embeddings, "_br_xvla_safe_missing_lm_head", False):
        original_get_output_embeddings = language_cls.get_output_embeddings

        def get_output_embeddings_or_none(self: Any) -> Any:
            if not hasattr(self, "lm_head"):
                return None
            return original_get_output_embeddings(self)

        get_output_embeddings_or_none._br_xvla_safe_missing_lm_head = True  # type: ignore[attr-defined]
        language_cls.get_output_embeddings = get_output_embeddings_or_none
        patches.append("Florence2LanguageForConditionalGeneration.get_output_embeddings_missing_lm_head_safe")
    return patches


def run_gradient_smoke(
    *,
    output_dir: Path,
    source_hdf5: Path,
    xvla_root: Path,
    demo_name: str,
    basket_xy_threshold: float,
    clip_steps: int,
    local_files_only: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = output_dir / "heartbeat.txt"
    result_path = output_dir / "result.json"
    report: dict[str, Any] = {
        "schema_version": "2026-07-17.epoch5_br_xvla_gradient_smoke.v1",
        "stage": "epoch_5_br_xvla_gradient_smoke",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "local_files_only": bool(local_files_only),
        "source_hdf5": str(source_hdf5),
        "xvla_root": str(xvla_root),
        "policy": {
            "model_loaded": False,
            "peft_lora_attached": False,
            "forward_happened": False,
            "backward_happened": False,
            "optimizer_created": False,
            "optimizer_step_happened": False,
            "checkpoint_written": False,
            "training_run_happened": False,
            "closed_loop_ours_evaluation_happened": False,
        },
        "nvidia_smi_before": nvidia_smi(),
    }
    model = None
    try:
        heartbeat.write_text("materialize_adapter\n", encoding="utf-8")
        materialized = materialize_one_target_clip(
            source_hdf5=source_hdf5,
            output_dir=output_dir / "adapter",
            demo_name=demo_name,
            threshold=float(basket_xy_threshold),
            clip_steps=int(clip_steps),
        )
        report["materialized_adapter"] = materialized
        sample_summary = smoke_xvla_reader(xvla_root, Path(materialized["meta_path"]))
        report["reader_smoke"] = sample_summary

        heartbeat.write_text("import_model\n", encoding="utf-8")
        os.environ.setdefault("HF_HOME", "/home/jiheon/assets/checkpoints/xvla_hf_cache")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/home/jiheon/assets/checkpoints/xvla_hf_cache/transformers")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        root = str(xvla_root)
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
        report["optional_server_import_shims_used"] = install_optional_server_import_shims()
        report["runtime_dependency_versions"] = {
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "timm": package_version("timm"),
            "torch": package_version("torch"),
        }
        report["transformers_compat_patches"] = install_xvla_transformers_compat_patches()
        from models.modeling_xvla import XVLA  # type: ignore
        from models.processing_xvla import XVLAProcessor  # type: ignore
        from peft import LoraConfig, get_peft_model

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for X-VLA gradient smoke")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda:0")

        heartbeat.write_text("load_model\n", encoding="utf-8")
        model = XVLA.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=bool(local_files_only),
        )
        processor = XVLAProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            local_files_only=bool(local_files_only),
        )
        report["policy"]["model_loaded"] = True
        report["model_type"] = type(model).__name__
        report["processor_type"] = type(processor).__name__

        lora_config = LoraConfig(
            lora_alpha=16,
            r=8,
            bias="none",
            target_modules="all-linear",
            modules_to_save=[
                "transformer.soft_prompt_hub",
                "transformer.action_encoder",
                "transformer.action_decoder",
            ],
        )
        model = get_peft_model(model, lora_config)
        report["policy"]["peft_lora_attached"] = True
        model.to(device=device, dtype=torch.float32)
        model.train()

        heartbeat.write_text("prepare_batch\n", encoding="utf-8")
        # Pull the exact first sample from the official reader after LoRA attach;
        # this keeps the materialized adapter path authoritative for the batch.
        root = str(xvla_root)
        if root in sys.path:
            sys.path.remove(root)
        sys.path.insert(0, root)
        from datasets.dataset import InfiniteDataReader  # type: ignore

        reader = InfiniteDataReader(str(materialized["meta_path"]), num_actions=30, num_views=3, training=False, action_mode="ee6d")
        sample = next(iter(reader))
        inputs = prepare_inputs(sample, processor, device, torch.float32)
        report["batch_shapes"] = {key: [int(x) for x in value.shape] for key, value in inputs.items()}
        report["cuda_memory_after_load"] = cuda_memory()

        heartbeat.write_text("forward_backward\n", encoding="utf-8")
        loss_dict = model(**inputs)
        report["policy"]["forward_happened"] = True
        base_loss = sum(loss_dict.values())
        phase_weight_lambda = 2.0
        phase_weight = 1.0 + phase_weight_lambda * float(materialized["first_count"] == 1)
        weighted_loss = base_loss * phase_weight
        report["losses"] = {
            key: float(value.detach().float().item()) for key, value in loss_dict.items()
        }
        report["losses"]["loss_total"] = float(base_loss.detach().float().item())
        report["losses"]["phase_weight"] = float(phase_weight)
        report["losses"]["weighted_loss"] = float(weighted_loss.detach().float().item())
        weighted_loss.backward()
        report["policy"]["backward_happened"] = True
        report["gradient"] = gradient_summary(model)
        report["cuda_memory_after_backward"] = cuda_memory()

        grad = report["gradient"]
        report["passed"] = bool(
            report["policy"]["model_loaded"]
            and report["policy"]["peft_lora_attached"]
            and report["policy"]["forward_happened"]
            and report["policy"]["backward_happened"]
            and not report["policy"]["optimizer_created"]
            and not report["policy"]["optimizer_step_happened"]
            and not report["policy"]["checkpoint_written"]
            and np.isfinite(report["losses"]["weighted_loss"])
            and int(grad["trainable_parameter_count"]) > 0
            and int(grad["grad_tensor_count"]) > 0
            and int(grad["finite_grad_tensor_count"]) == int(grad["grad_tensor_count"])
            and int(grad["nonzero_grad_tensor_count"]) > 0
            and float(grad["gradient_global_norm"]) > 0.0
        )
        report["decision"] = "BR_XVLA_GRADIENT_SMOKE_PASS" if report["passed"] else "BR_XVLA_GRADIENT_SMOKE_FAIL"
    except Exception as exc:  # pragma: no cover - runtime boundary
        report["passed"] = False
        report["decision"] = "BR_XVLA_GRADIENT_SMOKE_BLOCKED_OR_FAIL"
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-80:],
        }
    finally:
        try:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        report["elapsed_seconds"] = round(float(time.monotonic() - started), 3)
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        report["nvidia_smi_after"] = nvidia_smi()
        heartbeat.write_text("finished\n", encoding="utf-8")
        _write_json(result_path, report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-hdf5", type=Path, default=DEFAULT_HDF5_PATH)
    parser.add_argument("--xvla-root", type=Path, default=DEFAULT_XVLA_ROOT)
    parser.add_argument("--demo-name", default="demo_0")
    parser.add_argument("--basket-xy-threshold", type=float, default=0.08)
    parser.add_argument("--clip-steps", type=int, default=DEFAULT_CLIP_STEPS)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    report = run_gradient_smoke(
        output_dir=Path(args.output_dir),
        source_hdf5=Path(args.source_hdf5),
        xvla_root=Path(args.xvla_root),
        demo_name=str(args.demo_name),
        basket_xy_threshold=float(args.basket_xy_threshold),
        clip_steps=int(args.clip_steps),
        local_files_only=not bool(args.allow_download),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
