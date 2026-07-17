"""One-batch no-optimizer R2P-XVLA gradient smoke.

This gate verifies that the selected R2P-XVLA path can load the cached X-VLA
prior, attach the official PEFT LoRA configuration, consume a local X-VLA-format
task-5 adapter clip, compute a phase-weighted supervised loss, and backpropagate
finite nonzero adapter gradients.

It intentionally does not create an optimizer, call optimizer.step, write a
checkpoint, run a training loop, launch a simulator, or run closed-loop Ours
evaluation.
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
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from tca_map.xvla_spatial_task5.data_adapter_smoke import (
    TASK_DESCRIPTION,
    _write_encoded_rgb_frames,
    build_abs_action_6d,
    smoke_xvla_reader,
)
from tca_map.xvla_spatial_task5.data_audit import SpatialTask5DataAuditConfig, _phase_labels
from tca_map.xvla_spatial_task5.training_spec import MODEL_ID, MODEL_REVISION
from tca_map.xvla_task1.gradient_smoke import (
    _write_json,
    cuda_memory,
    gradient_summary,
    install_optional_server_import_shims,
    install_xvla_transformers_compat_patches,
    nvidia_smi,
    package_version,
    prepare_inputs,
)


DEFAULT_HDF5_PATH = Path(
    "/mnt/c/assets/data/libero/libero_spatial/"
    "pick_up_the_black_bowl_on_the_ramekin_and_place_it_on_the_plate_demo.hdf5"
)
DEFAULT_XVLA_ROOT = Path("/mnt/c/assets/repos/X-VLA")
DEFAULT_CLIP_STEPS = 96
LOCAL_MODEL_SNAPSHOT = Path(
    "/home/jiheon/assets/checkpoints/xvla_hf_cache/transformers/"
    "models--2toINF--X-VLA-Libero/snapshots/129e71460678b7236cee6fc9707f09d9fa0c3590"
)
PHASE_LOSS_WEIGHTS = {
    "source_on_ramekin": 1.0,
    "transit": 2.0,
    "target_on_plate": 1.5,
}


def task5_phase_labels(states: np.ndarray) -> dict[str, np.ndarray]:
    return _phase_labels(np.asarray(states, dtype=np.float64), SpatialTask5DataAuditConfig(hdf5_path=DEFAULT_HDF5_PATH))


def select_phase_rich_clip_start(states: np.ndarray, *, clip_steps: int, preferred_phase: str = "transit") -> int:
    labels = task5_phase_labels(states)
    phases = np.asarray(labels["phase"]).astype(str)
    candidates = np.flatnonzero(phases == preferred_phase)
    if candidates.size == 0:
        raise ValueError(f"no {preferred_phase} phase found")
    max_start = max(0, int(states.shape[0]) - int(clip_steps))
    viable = [int(index) for index in candidates if int(index) <= max_start]
    return viable[0] if viable else int(min(int(candidates[0]), max_start))


def _phase_counts_and_weight(phases: np.ndarray) -> tuple[dict[str, int], float]:
    names = np.asarray(phases).astype(str)
    counts = {name: int(np.sum(names == name)) for name in PHASE_LOSS_WEIGHTS}
    weights = np.asarray([PHASE_LOSS_WEIGHTS.get(str(name), 1.0) for name in names], dtype=np.float64)
    return counts, float(np.mean(weights)) if weights.size else 1.0


def materialize_phase_rich_clip(
    source_hdf5: Path,
    output_dir: Path,
    *,
    demo_name: str,
    clip_steps: int,
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    converted = output_dir / "converted_hdf5"
    converted.mkdir(parents=True, exist_ok=True)
    out_hdf5 = converted / f"{demo_name}_r2p_phase_clip.hdf5"

    with h5py.File(source_hdf5, "r") as source:
        demo = source["data"][demo_name]
        states = np.asarray(demo["states"], dtype=np.float64)
        start = select_phase_rich_clip_start(states, clip_steps=clip_steps)
        end = min(int(states.shape[0]), int(start) + int(clip_steps))
        actions = np.asarray(demo["actions"][start:end], dtype=np.float64)
        robot_states = np.asarray(demo["robot_states"][start:end], dtype=np.float64)
        clip_states = np.asarray(demo["states"][start:end], dtype=np.float64)
        agentview = np.asarray(demo["obs"]["agentview_rgb"][start:end], dtype=np.uint8)
        wrist = np.asarray(demo["obs"]["eye_in_hand_rgb"][start:end], dtype=np.uint8)

    labels = task5_phase_labels(clip_states)
    phases = np.asarray(labels["phase"]).astype(str)
    phase_counts, phase_weight_mean = _phase_counts_and_weight(phases)
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
        "phase_counts": phase_counts,
        "phase_sequence_first_40": [str(x) for x in phases[:40]],
        "phase_weight_mean": float(phase_weight_mean),
        "abs_action_6d_shape": [int(x) for x in abs_action_6d.shape],
    }


def run_gradient_smoke(
    *,
    output_dir: Path,
    source_hdf5: Path,
    xvla_root: Path,
    demo_name: str,
    clip_steps: int,
    local_files_only: bool,
) -> dict[str, Any]:
    started = time.monotonic()
    output_dir.mkdir(parents=True, exist_ok=True)
    heartbeat = output_dir / "heartbeat.txt"
    result_path = output_dir / "result.json"
    report: dict[str, Any] = {
        "schema_version": "2026-07-18.epoch5_r2p_xvla_gradient_smoke.v1",
        "stage": "epoch_5_r2p_xvla_gradient_smoke",
        "method": "R2P-XVLA",
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
            "simulator_rollout_happened": False,
        },
        "nvidia_smi_before": nvidia_smi(),
    }
    model = None
    try:
        heartbeat.write_text("materialize_adapter\n", encoding="utf-8")
        materialized = materialize_phase_rich_clip(
            source_hdf5=source_hdf5,
            output_dir=output_dir / "adapter",
            demo_name=demo_name,
            clip_steps=int(clip_steps),
        )
        report["materialized_adapter"] = materialized
        sample_summary = smoke_xvla_reader(xvla_root, Path(materialized["meta_path"]))
        report["reader_smoke"] = sample_summary

        heartbeat.write_text("import_model\n", encoding="utf-8")
        os.environ.setdefault("HF_HOME", "/home/jiheon/assets/checkpoints/xvla_hf_cache")
        os.environ.setdefault("TRANSFORMERS_CACHE", "/home/jiheon/assets/checkpoints/xvla_hf_cache/transformers")
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
            report["offline_env_enforced"] = {
                "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE"),
                "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE"),
            }
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
        load_source = str(LOCAL_MODEL_SNAPSHOT) if LOCAL_MODEL_SNAPSHOT.exists() else MODEL_ID
        load_kwargs: dict[str, Any] = {
            "trust_remote_code": True,
            "torch_dtype": torch.float32,
            "local_files_only": bool(local_files_only),
        }
        if load_source == MODEL_ID:
            load_kwargs["revision"] = MODEL_REVISION
        report["pretrained_load_source"] = load_source
        report["pretrained_load_source_is_local_snapshot"] = load_source != MODEL_ID

        heartbeat.write_text("load_model\n", encoding="utf-8")
        model = XVLA.from_pretrained(load_source, **load_kwargs)
        processor = XVLAProcessor.from_pretrained(load_source, **load_kwargs)
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
        phase_weight = float(materialized["phase_weight_mean"])
        weighted_loss = base_loss * phase_weight
        report["losses"] = {key: float(value.detach().float().item()) for key, value in loss_dict.items()}
        report["losses"]["loss_total"] = float(base_loss.detach().float().item())
        report["losses"]["phase_weight_mean"] = float(phase_weight)
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
            and not report["policy"]["simulator_rollout_happened"]
            and np.isfinite(report["losses"]["weighted_loss"])
            and int(grad["trainable_parameter_count"]) > 0
            and int(grad["grad_tensor_count"]) > 0
            and int(grad["finite_grad_tensor_count"]) == int(grad["grad_tensor_count"])
            and int(grad["nonzero_grad_tensor_count"]) > 0
            and float(grad["gradient_global_norm"]) > 0.0
        )
        report["decision"] = "R2P_XVLA_GRADIENT_SMOKE_PASS" if report["passed"] else "R2P_XVLA_GRADIENT_SMOKE_FAIL"
    except Exception as exc:  # pragma: no cover - runtime boundary
        report["passed"] = False
        report["decision"] = "R2P_XVLA_GRADIENT_SMOKE_BLOCKED_OR_FAIL"
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
    parser.add_argument("--clip-steps", type=int, default=DEFAULT_CLIP_STEPS)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    report = run_gradient_smoke(
        output_dir=Path(args.output_dir),
        source_hdf5=Path(args.source_hdf5),
        xvla_root=Path(args.xvla_root),
        demo_name=str(args.demo_name),
        clip_steps=int(args.clip_steps),
        local_files_only=not bool(args.allow_download),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
