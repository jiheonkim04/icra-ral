"""Pre-training numerical-noise and actual-path microbatch preflight.

The ``noise`` mode performs frozen X-VLA forwards only.  The ``microbatch``
mode is run only after a separate Stage 0 preregistration and performs one
throwaway optimizer step per candidate size.  Neither mode accesses simulator
outcomes or confirmatory identities.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pathlib
import random
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .adapter import ActionConsistentMissingViewAdapter, adapter_parameter_count
from .spec import DEFAULT_SPEC, load_frozen_method_spec


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_NOISE_REPORT = REPO_ROOT / "reports" / "action_consistent_missing_view_distillation_noise_calibration_result.json"
DEFAULT_NOISE_REPORT_MD = REPO_ROOT / "reports" / "action_consistent_missing_view_distillation_noise_calibration_result.md"
DEFAULT_MICROBATCH_REPORT = REPO_ROOT / "reports" / "action_consistent_missing_view_distillation_microbatch_preflight_result.json"
DEFAULT_MICROBATCH_REPORT_MD = REPO_ROOT / "reports" / "action_consistent_missing_view_distillation_microbatch_preflight_result.md"
IMPLEMENTATION_LABEL = "ACTION_CONSISTENT_MISSING_VIEW_DISTILLATION_XVLA_PREFLIGHT"

TRANSLATION_INDICES = (0, 1, 2, 10, 11, 12)
ROTATION_INDICES = (3, 4, 5, 6, 7, 8, 13, 14, 15, 16, 17, 18)
GRIPPER_INDICES = (9, 19)


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def atomic_write_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in pathlib.Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, rest = line.split(":", 1)
        values[key] = int(rest.strip().split()[0]) * 1024
    total = values["MemTotal"]
    available = values["MemAvailable"]
    swap_total = values.get("SwapTotal", 0)
    swap_free = values.get("SwapFree", 0)
    return {
        "mem_total_bytes": total,
        "mem_available_bytes": available,
        "mem_used_bytes": total - available,
        "mem_used_fraction": (total - available) / total,
        "swap_total_bytes": swap_total,
        "swap_used_bytes": swap_total - swap_free,
    }


def disk_report(path: pathlib.Path) -> dict[str, int]:
    stat = os.statvfs(path)
    return {
        "total_bytes": stat.f_blocks * stat.f_frsize,
        "available_bytes": stat.f_bavail * stat.f_frsize,
    }


def nvidia_smi() -> str:
    try:
        return subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu",
                "--format=csv,noheader",
            ],
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except Exception as exc:  # pragma: no cover - environment diagnostic
        return f"unavailable: {type(exc).__name__}: {exc}"


def stable_seed(*parts: object) -> int:
    payload = "|".join(str(part) for part in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**31 - 1)


def t_schedule(batch_size: int, *, effective_offset: int = 0, effective_batch: int = 8) -> torch.Tensor:
    if batch_size <= 0 or effective_batch <= 0:
        raise ValueError("batch sizes must be positive")
    values = [((effective_offset + index) % effective_batch + 0.5) / effective_batch for index in range(batch_size)]
    return torch.tensor(values, dtype=torch.float32)


def parameter_vector(module: nn.Module) -> torch.Tensor:
    values = [parameter.detach().float().reshape(-1).cpu() for parameter in module.parameters()]
    return torch.cat(values) if values else torch.empty(0)


def gradient_global_norm(module: nn.Module) -> tuple[float, int, int]:
    total = 0.0
    tensors = 0
    nonzero = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        tensors += 1
        if not bool(torch.isfinite(grad).all().item()):
            return float("nan"), tensors, nonzero
        if bool(torch.count_nonzero(grad).item()):
            nonzero += 1
        total += float(torch.sum(grad * grad).item())
    return math.sqrt(total), tensors, nonzero


def task_key(task: dict[str, Any]) -> str:
    return f"{task['suite']}_task{int(task['task_id'])}"


@dataclass
class FrozenRuntime:
    model: nn.Module
    processor: Any
    optional_shims: list[str]
    compatibility_patches: list[str]


def load_frozen_xvla(spec: dict[str, Any], device: torch.device) -> FrozenRuntime:
    from tca_map.rifa_xvla.stage0 import (
        freeze_module,
        install_optional_xvla_shims,
        install_xvla_transformers_compat_patches,
    )

    xvla = spec["xvla"]
    source_root = str(xvla["source_root"])
    if source_root in sys.path:
        sys.path.remove(source_root)
    sys.path.insert(0, source_root)
    shims = install_optional_xvla_shims()
    patches = install_xvla_transformers_compat_patches()
    from models.modeling_xvla import XVLA  # type: ignore
    from models.processing_xvla import XVLAProcessor  # type: ignore

    model = XVLA.from_pretrained(
        xvla["model_id"],
        revision=xvla["model_revision"],
        trust_remote_code=True,
        torch_dtype=torch.float32,
        local_files_only=True,
        cache_dir=xvla["cache_dir"],
    )
    processor = XVLAProcessor.from_pretrained(
        xvla["model_id"],
        revision=xvla["model_revision"],
        trust_remote_code=True,
        local_files_only=True,
        cache_dir=xvla["cache_dir"],
    )
    freeze_module(model)
    model.to(device=device, dtype=torch.float32)
    model.eval()
    return FrozenRuntime(model=model, processor=processor, optional_shims=shims, compatibility_patches=patches)


def materialize_calibration_records(
    spec: dict[str, Any],
    run_dir: pathlib.Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    # InfiniteDataReader is part of the pinned official X-VLA source tree.
    # Register that local root before importing/calling the reader; the first
    # frozen attempt preserved in the repair report failed at this exact path.
    source_root = str(spec["xvla"]["source_root"])
    if source_root in sys.path:
        sys.path.remove(source_root)
    sys.path.insert(0, source_root)
    from tca_map.cvlr_xvla.stage0 import read_fixed_official_samples
    from tca_map.rifa_xvla.stage0 import materialize_xvla_clip

    positions = list(spec["data_splits"]["discovery"]["official_reader_positions"])
    root = run_dir / "materialized_calibration_rows"
    root.mkdir(parents=True, exist_ok=False)
    records: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    for task in spec["training_panel"]:
        output = root / f"{task_key(task)}_demo0"
        manifest = materialize_xvla_clip(
            pathlib.Path(task["hdf5"]),
            output,
            demo_index=0,
            instruction=str(task["instruction"]),
            clip_steps=48,
        )
        for position, sample in zip(
            positions,
            read_fixed_official_samples(pathlib.Path(manifest["meta_path"]), positions),
        ):
            records.append(
                {
                    "task_key": task_key(task),
                    "demo_index": 0,
                    "reader_position": int(position),
                    "sample": sample,
                }
            )
        manifest.pop("agent_frame")
        manifest.pop("wrist_frame")
        manifests.append(manifest)
    if len(records) != 12:
        raise RuntimeError(f"calibration row count drift: {len(records)}")
    return records, manifests


def prepared_batch(
    records: list[dict[str, Any]],
    processor: Any,
    device: torch.device,
    *,
    condition: str,
) -> dict[str, torch.Tensor]:
    from tca_map.rifa_xvla.stage0 import prepare_offline_inputs

    prepared = [prepare_offline_inputs(record["sample"], processor, device, condition=condition) for record in records]
    shapes: dict[str, set[tuple[int, ...]]] = {}
    for key in prepared[0]:
        shapes[key] = {tuple(item[key].shape[1:]) for item in prepared}
        if len(shapes[key]) != 1:
            raise ValueError(f"variable nonbatch shape for {key}: {shapes[key]}")
    return {key: torch.cat([item[key] for item in prepared], dim=0) for key in prepared[0]}


class ActionHiddenHook:
    """Capture teacher hidden or apply the frozen adapter on the student path."""

    def __init__(self) -> None:
        self.mode = "inactive"
        self.adapter: ActionConsistentMissingViewAdapter | None = None
        self.missing_indicator: torch.Tensor | None = None
        self.compute_reconstruction = False
        self.last_hidden_before: torch.Tensor | None = None
        self.last_hidden_after: torch.Tensor | None = None
        self.last_reconstruction: torch.Tensor | None = None
        self.forward_counts = {"teacher_capture": 0, "student_adapter": 0, "inactive": 0}

    def activate_teacher(self) -> None:
        self.mode = "teacher"
        self.adapter = None
        self.missing_indicator = None
        self.compute_reconstruction = False

    def activate_student(
        self,
        adapter: ActionConsistentMissingViewAdapter,
        missing_indicator: torch.Tensor,
        *,
        compute_reconstruction: bool,
    ) -> None:
        self.mode = "student"
        self.adapter = adapter
        self.missing_indicator = missing_indicator
        self.compute_reconstruction = bool(compute_reconstruction)

    def deactivate(self) -> None:
        self.mode = "inactive"
        self.adapter = None
        self.missing_indicator = None
        self.compute_reconstruction = False

    def __call__(self, _module: nn.Module, _inputs: tuple[Any, ...], output: torch.Tensor) -> torch.Tensor:
        if self.mode == "teacher":
            self.last_hidden_before = output
            self.last_hidden_after = output
            self.last_reconstruction = None
            self.forward_counts["teacher_capture"] += 1
            return output
        if self.mode == "student":
            if self.adapter is None or self.missing_indicator is None:
                raise RuntimeError("student hook is missing its adapter context")
            adapted, reconstruction, _ = self.adapter(
                output,
                self.missing_indicator,
                compute_reconstruction=self.compute_reconstruction,
            )
            self.last_hidden_before = output
            self.last_hidden_after = adapted
            self.last_reconstruction = reconstruction
            self.forward_counts["student_adapter"] += 1
            return adapted
        self.forward_counts["inactive"] += 1
        return output


def _transformer_raw(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
    enc: dict[str, torch.Tensor],
    action_with_noise: torch.Tensor,
    t: torch.Tensor,
) -> torch.Tensor:
    proprio_m, action_m = model.action_space.preprocess(inputs["proprio"], action_with_noise)
    return model.transformer(
        domain_id=inputs["domain_id"],
        action_with_noise=action_m,
        t=t,
        proprio=proprio_m,
        **enc,
    )


def paired_forward(
    model: nn.Module,
    hook: ActionHiddenHook,
    adapter: ActionConsistentMissingViewAdapter,
    clean: dict[str, torch.Tensor],
    dropout: dict[str, torch.Tensor],
    *,
    epsilon_seed: int,
    effective_offset: int = 0,
) -> dict[str, torch.Tensor]:
    action = clean["action"]
    if not torch.equal(clean["action"], dropout["action"]):
        raise ValueError("teacher/student demonstration action mismatch")
    if not torch.equal(clean["image_mask"], dropout["image_mask"]):
        raise ValueError("frozen black-pixel dropout must preserve image_mask")
    batch_size = action.shape[0]
    t = t_schedule(batch_size, effective_offset=effective_offset).to(device=action.device, dtype=action.dtype)
    generator = torch.Generator(device=action.device)
    generator.manual_seed(int(epsilon_seed))
    epsilon = torch.randn(action.shape, generator=generator, device=action.device, dtype=action.dtype)
    x_t = epsilon * t.view(-1, 1, 1) + action * (1.0 - t).view(-1, 1, 1)

    hook.activate_teacher()
    with torch.no_grad():
        clean_enc = model.forward_vlm(clean["input_ids"], clean["image_input"], clean["image_mask"])
        clean_wrist = clean_enc["aux_visual_inputs"][:, :50, :].detach()
        teacher_raw = _transformer_raw(model, clean, clean_enc, x_t, t).detach()
        if hook.last_hidden_after is None:
            raise RuntimeError("teacher hidden capture did not execute")
        teacher_hidden = hook.last_hidden_after.detach()

    hook.activate_student(
        adapter,
        torch.ones((batch_size, 1), device=action.device, dtype=action.dtype),
        compute_reconstruction=True,
    )
    with torch.no_grad():
        dropout_enc = model.forward_vlm(
            dropout["input_ids"], dropout["image_input"], dropout["image_mask"]
        )
    student_raw = _transformer_raw(model, dropout, dropout_enc, x_t, t)
    if hook.last_hidden_before is None or hook.last_hidden_after is None or hook.last_reconstruction is None:
        raise RuntimeError("student adapter hook did not expose required tensors")
    output = {
        "clean_wrist": clean_wrist,
        "teacher_hidden": teacher_hidden,
        "teacher_raw": teacher_raw,
        "dropout_hidden": hook.last_hidden_before,
        "student_hidden": hook.last_hidden_after,
        "student_raw": student_raw,
        "reconstruction": hook.last_reconstruction,
        "t": t,
    }
    hook.deactivate()
    return output


def component_mse(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "hidden_mse": F.mse_loss(outputs["student_hidden"], outputs["teacher_hidden"]),
        "translation_mse": F.mse_loss(
            outputs["student_raw"][..., TRANSLATION_INDICES],
            outputs["teacher_raw"][..., TRANSLATION_INDICES],
        ),
        "rotation_mse": F.mse_loss(
            outputs["student_raw"][..., ROTATION_INDICES],
            outputs["teacher_raw"][..., ROTATION_INDICES],
        ),
        "raw_gripper_margin_mse": F.mse_loss(
            outputs["student_raw"][..., GRIPPER_INDICES],
            outputs["teacher_raw"][..., GRIPPER_INDICES],
        ),
        "wrist_reconstruction_mse": F.mse_loss(outputs["reconstruction"], outputs["clean_wrist"]),
    }


def normalized_full_loss(
    components: dict[str, torch.Tensor],
    denominators: dict[str, float],
) -> torch.Tensor:
    return (
        0.25 * components["hidden_mse"] / denominators["hidden_mse"]
        + components["translation_mse"] / denominators["translation_mse"]
        + components["rotation_mse"] / denominators["rotation_mse"]
        + components["raw_gripper_margin_mse"] / denominators["raw_gripper_margin_mse"]
        + 0.25 * components["wrist_reconstruction_mse"] / denominators["wrist_reconstruction_mse"]
    )


def generate_raw_and_postprocessed(
    model: nn.Module,
    hook: ActionHiddenHook,
    inputs: dict[str, torch.Tensor],
    *,
    seed: int,
    steps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    hook.deactivate()
    generator = torch.Generator(device=inputs["proprio"].device)
    generator.manual_seed(int(seed))
    with torch.no_grad():
        enc = model.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
        batch_size = inputs["input_ids"].shape[0]
        x1 = torch.randn(
            (batch_size, model.num_actions, model.action_space.dim_action),
            generator=generator,
            device=inputs["proprio"].device,
            dtype=inputs["proprio"].dtype,
        )
        action = torch.zeros_like(x1)
        for index in range(steps, 0, -1):
            t = torch.full(
                (batch_size,), index / steps, device=action.device, dtype=action.dtype
            )
            x_t = x1 * t.view(-1, 1, 1) + action * (1.0 - t).view(-1, 1, 1)
            action = _transformer_raw(model, inputs, enc, x_t, t)
        raw = action.detach().clone()
        post = model.action_space.postprocess(action.detach().clone())
    return raw, post


def semantic_repeat_delta(left_post: torch.Tensor, right_post: torch.Tensor) -> dict[str, Any]:
    from tca_map.cvlr_xvla.stage0 import semantic_action_delta

    return semantic_action_delta(
        left_post.detach().float().cpu().numpy()[0, :, :10],
        right_post.detach().float().cpu().numpy()[0, :, :10],
    )


def _new_adapter(spec: dict[str, Any], device: torch.device) -> ActionConsistentMissingViewAdapter:
    module = spec["trainable_module"]
    adapter = ActionConsistentMissingViewAdapter(
        hidden_size=int(module["hidden_size"]),
        bottleneck_dim=int(module["bottleneck_dim"]),
        wrist_token_count=int(module["wrist_token_count"]),
        wrist_token_dim=int(module["wrist_token_dim"]),
        residual_scale=float(module["residual_scale"]),
    ).to(device=device, dtype=torch.float32)
    if adapter_parameter_count(adapter) != int(module["trainable_parameter_count_exact"]):
        raise RuntimeError("runtime adapter parameter count drift")
    return adapter


def _write_markdown(path: pathlib.Path, result: dict[str, Any]) -> None:
    if result["mode"] == "noise":
        values = result.get("derived") or {}
        lines = [
            "# Action-Consistent Missing-View Distillation: Numerical-Noise Calibration",
            "",
            f"Decision: `{result.get('decision')}`",
            "",
            f"- Frozen forward repetitions per row: `{result.get('repeat_count')}`",
            f"- Fixed discovery calibration rows: `{result.get('calibration_row_count')}`",
            f"- Optimizer steps: `{result.get('optimizer_steps')}`",
            f"- Confirmatory outcomes accessed: `{result.get('confirmatory_outcomes_accessed')}`",
            f"- Condition image mask preserved: `{(result.get('condition_wiring') or {}).get('image_mask_preserved')}`",
            "",
            "## Frozen normalization denominators",
            "",
        ]
        for key, value in (values.get("normalization_denominators") or {}).items():
            lines.append(f"- `{key}`: `{value}`")
        lines.extend(["", "## Repeated-forward numerical noise", ""])
        for key, value in (values.get("numerical_noise") or {}).items():
            lines.append(f"- `{key}`: `{value}`")
        lines.extend(["", "## Practical absolute thresholds", ""])
        for key, value in (values.get("practical_absolute_thresholds") or {}).items():
            lines.append(f"- `{key}`: `{value}`")
    else:
        lines = [
            "# Action-Consistent Missing-View Distillation: Actual-Path Microbatch Preflight",
            "",
            f"Decision: `{result.get('decision')}`",
            "",
            f"- Selected microbatch: `{result.get('selected_microbatch')}`",
            f"- Gradient accumulation: `{result.get('gradient_accumulation')}`",
            f"- Effective batch: `{result.get('effective_batch')}`",
            f"- Candidate rows: `{len(result.get('candidates') or [])}`",
            f"- Stage 0 optimizer budget consumed: `0`",
            "",
            "Each candidate used a fresh zero-initialized adapter and exactly one throwaway optimizer step.",
        ]
    lines.extend(
        [
            "",
            "## Execution",
            "",
            f"- CUDA PID: `{result.get('cuda_pid')}`",
            f"- Elapsed seconds: `{result.get('elapsed_seconds')}`",
            f"- Exceptions: `{len(result.get('exceptions') or [])}`",
            "",
        ]
    )
    atomic_write_text(path, "\n".join(lines))


def run_noise_calibration(
    spec: dict[str, Any],
    runtime: FrozenRuntime,
    records: list[dict[str, Any]],
    device: torch.device,
    heartbeat: Any,
) -> dict[str, Any]:
    model = runtime.model
    hook = ActionHiddenHook()
    handle = model.transformer.norm.register_forward_hook(hook)
    adapter = _new_adapter(spec, device)
    adapter.eval()
    repeat_count = 3
    component_values: dict[str, list[float]] = {
        "hidden_mse": [],
        "translation_mse": [],
        "rotation_mse": [],
        "raw_gripper_margin_mse": [],
        "wrist_reconstruction_mse": [],
    }
    noise = {
        "translation_RMSE": 0.0,
        "rotation_RMSE": 0.0,
        "raw_gripper_margin_MAE": 0.0,
        "action_hidden_MSE": 0.0,
    }
    discrete_repeat_flips = 0
    clean_actions: list[np.ndarray] = []
    condition_masks: list[dict[str, Any]] = []
    paired_forward_count = 0
    generation_count = 0
    try:
        for row_index, record in enumerate(records):
            heartbeat(f"noise_row_{row_index + 1}_of_{len(records)}")
            clean = prepared_batch([record], runtime.processor, device, condition="clean")
            dropout = prepared_batch(
                [record], runtime.processor, device, condition="mask_1_in_hand_dropout"
            )
            condition_masks.append(
                {
                    "task_key": record["task_key"],
                    "reader_position": record["reader_position"],
                    "clean_image_mask": clean["image_mask"].detach().cpu().tolist(),
                    "dropout_image_mask": dropout["image_mask"].detach().cpu().tolist(),
                    "image_mask_equal": bool(torch.equal(clean["image_mask"], dropout["image_mask"])),
                    "wrist_pixel_tensor_changed": bool(
                        not torch.equal(clean["image_input"][:, 1], dropout["image_input"][:, 1])
                    ),
                }
            )
            step_outputs: list[dict[str, torch.Tensor]] = []
            with torch.no_grad():
                for repeat in range(repeat_count):
                    output = paired_forward(
                        model,
                        hook,
                        adapter,
                        clean,
                        dropout,
                        epsilon_seed=stable_seed(spec["training_budget"]["seed"], row_index, "noise"),
                    )
                    step_outputs.append({key: value.detach().clone() for key, value in output.items()})
                    paired_forward_count += 1
            components = component_mse(step_outputs[0])
            for key, value in components.items():
                component_values[key].append(float(value.detach().cpu()))
            for repeat in step_outputs[1:]:
                noise["action_hidden_MSE"] = max(
                    noise["action_hidden_MSE"],
                    float(F.mse_loss(repeat["student_hidden"], step_outputs[0]["student_hidden"]).cpu()),
                )
                noise["translation_RMSE"] = max(
                    noise["translation_RMSE"],
                    float(
                        torch.sqrt(
                            F.mse_loss(
                                repeat["student_raw"][..., TRANSLATION_INDICES],
                                step_outputs[0]["student_raw"][..., TRANSLATION_INDICES],
                            )
                        ).cpu()
                    ),
                )
                noise["rotation_RMSE"] = max(
                    noise["rotation_RMSE"],
                    float(
                        torch.sqrt(
                            F.mse_loss(
                                repeat["student_raw"][..., ROTATION_INDICES],
                                step_outputs[0]["student_raw"][..., ROTATION_INDICES],
                            )
                        ).cpu()
                    ),
                )
                noise["raw_gripper_margin_MAE"] = max(
                    noise["raw_gripper_margin_MAE"],
                    float(
                        torch.mean(
                            torch.abs(
                                repeat["student_raw"][..., GRIPPER_INDICES]
                                - step_outputs[0]["student_raw"][..., GRIPPER_INDICES]
                            )
                        ).cpu()
                    ),
                )

            clean_plans: list[tuple[torch.Tensor, torch.Tensor]] = []
            dropout_plans: list[tuple[torch.Tensor, torch.Tensor]] = []
            for _repeat in range(repeat_count):
                clean_plans.append(
                    generate_raw_and_postprocessed(
                        model,
                        hook,
                        {key: value for key, value in clean.items() if key != "action"},
                        seed=stable_seed(spec["training_budget"]["seed"], row_index, "plan"),
                        steps=int(spec["xvla"]["denoise_steps"]),
                    )
                )
                dropout_plans.append(
                    generate_raw_and_postprocessed(
                        model,
                        hook,
                        {key: value for key, value in dropout.items() if key != "action"},
                        seed=stable_seed(spec["training_budget"]["seed"], row_index, "plan"),
                        steps=int(spec["xvla"]["denoise_steps"]),
                    )
                )
                generation_count += 2
            clean_actions.append(clean_plans[0][1].detach().float().cpu().numpy()[0, :, :10])
            for plan_set in (clean_plans, dropout_plans):
                base_raw, base_post = plan_set[0]
                for raw, post in plan_set[1:]:
                    delta = semantic_repeat_delta(post, base_post)
                    noise["translation_RMSE"] = max(noise["translation_RMSE"], float(delta["translation_rms"]))
                    noise["rotation_RMSE"] = max(noise["rotation_RMSE"], float(delta["rotation_rms"]))
                    noise["raw_gripper_margin_MAE"] = max(
                        noise["raw_gripper_margin_MAE"],
                        float(torch.mean(torch.abs(raw[..., GRIPPER_INDICES] - base_raw[..., GRIPPER_INDICES])).cpu()),
                    )
                    discrete_repeat_flips += int(delta["gripper_flip_count"])

        floors = spec["objectives"]["normalization_calibration"]["floors"]
        denominators = {
            key: max(float(np.mean(values)), float(floors[key]))
            for key, values in component_values.items()
        }
        effect = spec["practical_effect_rule"]
        practical = {
            key: max(float(effect["absolute_improvement_floors"][key]), int(effect["noise_multiplier"]) * float(value))
            for key, value in noise.items()
        }

        from tca_map.rifa_xvla.stage0 import plan_to_libero_actions

        translation_adjacent: list[float] = []
        rotation_adjacent: list[float] = []
        for plan in clean_actions:
            actions = plan_to_libero_actions(plan)
            translation_adjacent.extend(np.abs(np.diff(actions[:, :3], axis=0)).reshape(-1).tolist())
            rotation_adjacent.extend(np.abs(np.diff(actions[:, 3:6], axis=0)).reshape(-1).tolist())
        smoothness = {
            "translation_abs_adjacent_p99": float(np.percentile(translation_adjacent, 99)),
            "rotation_abs_adjacent_p99": float(np.percentile(rotation_adjacent, 99)),
            "translation_envelope": float(np.percentile(translation_adjacent, 99)) + 10 * noise["translation_RMSE"],
            "rotation_envelope": float(np.percentile(rotation_adjacent, 99)) + 10 * noise["rotation_RMSE"],
        }
        return {
            "repeat_count": repeat_count,
            "calibration_row_count": len(records),
            "optimizer_steps": 0,
            "paired_forward_count": paired_forward_count,
            "generation_count": generation_count,
            "hook_forward_counts": hook.forward_counts,
            "condition_wiring": {
                "rows": condition_masks,
                "image_mask_preserved": all(row["image_mask_equal"] for row in condition_masks),
                "wrist_pixel_tensor_changed": all(row["wrist_pixel_tensor_changed"] for row in condition_masks),
                "expected_official_image_mask": [[True, True, False]],
            },
            "derived": {
                "normalization_component_means_before_floor": {
                    key: float(np.mean(values)) for key, values in component_values.items()
                },
                "normalization_denominators": denominators,
                "numerical_noise": noise,
                "discrete_repeat_gripper_flips": discrete_repeat_flips,
                "practical_absolute_thresholds": practical,
                "clean_teacher_smoothness": smoothness,
            },
        }
    finally:
        handle.remove()


def run_microbatch_preflight(
    spec: dict[str, Any],
    runtime: FrozenRuntime,
    records: list[dict[str, Any]],
    device: torch.device,
    noise_report: dict[str, Any],
    run_dir: pathlib.Path,
    heartbeat: Any,
) -> dict[str, Any]:
    denominators = (noise_report.get("derived") or {}).get("normalization_denominators")
    if not isinstance(denominators, dict):
        raise ValueError("noise report does not contain frozen normalization denominators")
    model = runtime.model
    hook = ActionHiddenHook()
    handle = model.transformer.norm.register_forward_hook(hook)
    candidates: list[dict[str, Any]] = []
    selected: int | None = None
    effective_batch = int(spec["training_budget"]["effective_batch"])
    gpu_total = int(torch.cuda.get_device_properties(device).total_memory)
    baseline_swap = meminfo()["swap_used_bytes"]
    try:
        for size in spec["training_budget"]["microbatch_candidates"]:
            size = int(size)
            heartbeat(f"microbatch_candidate_{size}")
            candidate_dir = run_dir / f"candidate_{size}"
            candidate_dir.mkdir(parents=True, exist_ok=False)
            before_mem = meminfo()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            adapter: ActionConsistentMissingViewAdapter | None = None
            record: dict[str, Any] = {
                "microbatch": size,
                "safe": False,
                "exception": None,
                "optimizer_steps": 0,
            }
            try:
                chosen = [records[index % len(records)] for index in range(size)]
                clean = prepared_batch(chosen, runtime.processor, device, condition="clean")
                dropout = prepared_batch(
                    chosen, runtime.processor, device, condition="mask_1_in_hand_dropout"
                )
                adapter = _new_adapter(spec, device)
                adapter.train()
                before = parameter_vector(adapter)
                budget = spec["training_budget"]
                optimizer = torch.optim.AdamW(
                    adapter.parameters(),
                    lr=float(budget["learning_rate_peak"]),
                    betas=tuple(float(value) for value in budget["betas"]),
                    eps=float(budget["epsilon"]),
                    weight_decay=float(budget["weight_decay"]),
                )
                outputs = paired_forward(
                    model,
                    hook,
                    adapter,
                    clean,
                    dropout,
                    epsilon_seed=stable_seed(budget["seed"], "microbatch", size),
                )
                components = component_mse(outputs)
                loss = normalized_full_loss(components, {key: float(value) for key, value in denominators.items()})
                loss.backward()
                grad_norm, grad_tensors, nonzero_grad_tensors = gradient_global_norm(adapter)
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(budget["max_gradient_norm"]))
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                record["optimizer_steps"] = 1
                after = parameter_vector(adapter)
                weight_change = float(torch.linalg.vector_norm(after - before))
                checkpoint = candidate_dir / "throwaway_adapter.pt"
                torch.save(
                    {
                        "adapter_state_dict": adapter.state_dict(),
                        "microbatch": size,
                        "spec_sha256": sha256_file(DEFAULT_SPEC),
                        "throwaway_preflight": True,
                    },
                    checkpoint,
                )
                reloaded = _new_adapter(spec, torch.device("cpu"))
                payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
                reloaded.load_state_dict(payload["adapter_state_dict"], strict=True)
                reload_delta = float(torch.linalg.vector_norm(parameter_vector(reloaded) - after))
                torch.cuda.synchronize(device)
                after_mem = meminfo()
                peak_allocated = int(torch.cuda.max_memory_allocated(device))
                peak_reserved = int(torch.cuda.max_memory_reserved(device))
                swap_growth = int(after_mem["swap_used_bytes"] - baseline_swap)
                record.update(
                    {
                        "loss": float(loss.detach().cpu()),
                        "loss_finite": bool(torch.isfinite(loss.detach()).item()),
                        "gradient_global_norm": grad_norm,
                        "gradient_tensor_count": grad_tensors,
                        "nonzero_gradient_tensor_count": nonzero_grad_tensors,
                        "weight_change_l2": weight_change,
                        "checkpoint": {
                            "path": str(checkpoint),
                            "bytes": checkpoint.stat().st_size,
                            "sha256": sha256_file(checkpoint),
                            "reload_parameter_delta_l2": reload_delta,
                        },
                        "peak_cuda_allocated_bytes": peak_allocated,
                        "peak_cuda_reserved_bytes": peak_reserved,
                        "peak_cuda_reserved_fraction": peak_reserved / gpu_total,
                        "system_ram_after": after_mem,
                        "system_ram_used_fraction": after_mem["mem_used_fraction"],
                        "research_induced_swap_growth_bytes": swap_growth,
                    }
                )
                record["safe"] = bool(
                    record["loss_finite"]
                    and math.isfinite(grad_norm)
                    and grad_norm > 0
                    and nonzero_grad_tensors > 0
                    and weight_change > 0
                    and reload_delta == 0
                    and record["peak_cuda_reserved_fraction"] <= 0.88
                    and record["system_ram_used_fraction"] <= 0.82
                    and swap_growth <= 0
                )
            except Exception as exc:
                record["exception"] = {
                    "type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                }
                if isinstance(exc, torch.cuda.OutOfMemoryError):
                    record["unsafe_reason"] = "CUDA_OUT_OF_MEMORY"
            finally:
                hook.deactivate()
                candidates.append(record)
                del adapter
                gc.collect()
                torch.cuda.empty_cache()
            if record["safe"]:
                selected = size
            else:
                break
        return {
            "candidates": candidates,
            "selected_microbatch": selected,
            "effective_batch": effective_batch,
            "gradient_accumulation": (effective_batch // selected) if selected else None,
            "candidate_sequence_stopped_after_first_unsafe": bool(
                candidates and not candidates[-1]["safe"]
            ),
            "stage0_optimizer_budget_consumed": 0,
            "throwaway_preflight_optimizer_steps": int(
                sum(int(candidate["optimizer_steps"]) for candidate in candidates)
            ),
        }
    finally:
        handle.remove()


def run_preflight(
    *,
    mode: str,
    run_dir: pathlib.Path,
    spec_path: pathlib.Path,
    report_json: pathlib.Path,
    report_md: pathlib.Path,
    noise_report_path: pathlib.Path | None,
) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    # The durable shell worker writes its PID/timestamp into this directory
    # before Python starts, so accept that pre-created wrapper directory.
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    heartbeat_path = run_dir / "heartbeat.json"
    partial_path = run_dir / "partial_result.json"
    exit_path = run_dir / "exit_code.txt"
    spec = load_frozen_method_spec(spec_path)
    result: dict[str, Any] = {
        "schema_version": f"2026-07-19.epoch5_action_consistent_missing_view_distillation_{mode}_preflight.v1",
        "implementation_label": IMPLEMENTATION_LABEL,
        "method": spec["method"],
        "mode": mode,
        "decision": "PREFLIGHT_IMPLEMENTATION_OR_RESOURCE_FAILURE",
        "source_head": git_head(),
        "spec": {"path": str(spec_path), "sha256": sha256_file(spec_path)},
        "run_dir": str(run_dir),
        "pid": os.getpid(),
        "cuda_pid": os.getpid() if torch.cuda.is_available() else None,
        "started_at": timestamp(),
        "exceptions": [],
        "confirmatory_outcomes_accessed": False,
        "closed_loop_rollout_executed": False,
        "downloads_used": False,
        "model_offload_used": False,
        "physical_robot_manipulation": False,
        "nvidia_smi_before": nvidia_smi(),
        "system_memory_before": meminfo(),
        "disk_before": disk_report(REPO_ROOT),
    }

    def heartbeat(stage: str) -> None:
        payload = {
            "timestamp": timestamp(),
            "stage": stage,
            "mode": mode,
            "pid": os.getpid(),
            "elapsed_seconds": round(time.monotonic() - started, 3),
        }
        atomic_write_json(heartbeat_path, payload)
        atomic_write_json(status_path, {**payload, "state": "running"})
        atomic_write_json(partial_path, payload)

    atomic_write_text(run_dir / "worker_pid.txt", f"{os.getpid()}\n")
    atomic_write_json(
        run_dir / "launch_manifest.json",
        {
            "source_head": result["source_head"],
            "spec": result["spec"],
            "mode": mode,
            "python": sys.executable,
            "argv": sys.argv,
            "confirmatory_outcomes_authorized": False,
            "closed_loop_authorized": False,
        },
    )

    runtime: FrozenRuntime | None = None
    try:
        heartbeat("risk_validation")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for actual X-VLA preflight")
        if any(not pathlib.Path(task["hdf5"]).is_file() for task in spec["training_panel"]):
            raise FileNotFoundError("one or more frozen training-panel HDF5 files are missing")
        if disk_report(REPO_ROOT)["available_bytes"] < 10 * 1024**3:
            raise RuntimeError("less than 10 GiB available on the run filesystem")
        before_memory = meminfo()
        if before_memory["mem_used_fraction"] > 0.82:
            raise RuntimeError("system RAM already exceeds the frozen 82% ceiling")
        result["risk_assessment"] = {
            "source": "tracked local code and already-present X-VLA/LIBERO assets",
            "downloads": "disabled",
            "token_or_secret_access": False,
            "license_clickthrough": False,
            "environment_install_or_mutation": False,
            "dataset_files_present": True,
            "disk_minimum_free_bytes": 10 * 1024**3,
            "gpu_reserved_ceiling_fraction": 0.88,
            "system_ram_ceiling_fraction": 0.82,
            "research_induced_swap_growth_allowed_bytes": 0,
            "cpu_or_disk_model_offload": False,
        }

        heartbeat("materialize_fixed_discovery_calibration_rows")
        records, materialized = materialize_calibration_records(spec, run_dir)
        result["materialized"] = materialized
        result["calibration_row_count"] = len(records)

        heartbeat("load_frozen_xvla")
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        seed = int(spec["training_budget"]["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        runtime = load_frozen_xvla(spec, device)
        result["xvla"] = {
            "model_id": spec["xvla"]["model_id"],
            "model_revision": spec["xvla"]["model_revision"],
            "source_revision": spec["xvla"]["source_revision"],
            "device": str(next(runtime.model.parameters()).device),
            "dtype": str(next(runtime.model.parameters()).dtype),
            "base_trainable_parameter_count": int(
                sum(parameter.numel() for parameter in runtime.model.parameters() if parameter.requires_grad)
            ),
            "optional_shims": runtime.optional_shims,
            "compatibility_patches": runtime.compatibility_patches,
        }

        if mode == "noise":
            heartbeat("frozen_repeated_forward_noise_calibration")
            payload = run_noise_calibration(spec, runtime, records, device, heartbeat)
            result.update(payload)
            valid = bool(
                result["optimizer_steps"] == 0
                and result["condition_wiring"]["image_mask_preserved"]
                and result["condition_wiring"]["wrist_pixel_tensor_changed"]
                and result["derived"]["discrete_repeat_gripper_flips"] == 0
                and all(math.isfinite(float(value)) for value in result["derived"]["normalization_denominators"].values())
                and all(float(value) > 0 for value in result["derived"]["normalization_denominators"].values())
            )
            result["decision"] = (
                "NUMERICAL_NOISE_CALIBRATION_VALID" if valid else "NUMERICAL_NOISE_CALIBRATION_INVALID"
            )
        elif mode == "microbatch":
            if noise_report_path is None:
                raise ValueError("microbatch mode requires --noise-report")
            noise_report = json.loads(noise_report_path.read_text(encoding="utf-8"))
            if noise_report.get("decision") != "NUMERICAL_NOISE_CALIBRATION_VALID":
                raise ValueError("microbatch preflight requires a valid frozen noise report")
            result["noise_report"] = {
                "path": str(noise_report_path),
                "sha256": sha256_file(noise_report_path),
            }
            heartbeat("actual_path_microbatch_preflight")
            payload = run_microbatch_preflight(
                spec, runtime, records, device, noise_report, run_dir, heartbeat
            )
            result.update(payload)
            result["decision"] = (
                "ACTUAL_PATH_MICROBATCH_PREFLIGHT_VALID"
                if result["selected_microbatch"] is not None
                else "ACTUAL_PATH_MICROBATCH_PREFLIGHT_INVALID"
            )
        else:
            raise ValueError(f"unknown preflight mode {mode!r}")

        result["success"] = result["decision"] in {
            "NUMERICAL_NOISE_CALIBRATION_VALID",
            "ACTUAL_PATH_MICROBATCH_PREFLIGHT_VALID",
        }
    except Exception as exc:
        result["exceptions"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        result["success"] = False
    finally:
        runtime = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            result["peak_cuda_allocated_bytes"] = int(torch.cuda.max_memory_allocated())
            result["peak_cuda_reserved_bytes"] = int(torch.cuda.max_memory_reserved())
        result["system_memory_after"] = meminfo()
        result["research_induced_swap_growth_bytes"] = int(
            result["system_memory_after"]["swap_used_bytes"]
            - result["system_memory_before"]["swap_used_bytes"]
        )
        result["disk_after"] = disk_report(REPO_ROOT)
        result["nvidia_smi_after"] = nvidia_smi()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["finished_at"] = timestamp()
        atomic_write_json(run_dir / "result.json", result)
        atomic_write_json(report_json, result)
        _write_markdown(report_md, result)
        _write_markdown(run_dir / "result.md", result)
        exit_code = 0 if result.get("success") else 1
        atomic_write_text(exit_path, f"{exit_code}\n")
        atomic_write_json(
            status_path,
            {
                "timestamp": timestamp(),
                "state": "complete" if exit_code == 0 else "failed",
                "decision": result["decision"],
                "pid": os.getpid(),
            },
        )
    return (0 if result.get("success") else 1), result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("noise", "microbatch"), required=True)
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--noise-report", type=pathlib.Path)
    parser.add_argument("--report-json", type=pathlib.Path)
    parser.add_argument("--report-md", type=pathlib.Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report_json = args.report_json or (
        DEFAULT_NOISE_REPORT if args.mode == "noise" else DEFAULT_MICROBATCH_REPORT
    )
    report_md = args.report_md or (
        DEFAULT_NOISE_REPORT_MD if args.mode == "noise" else DEFAULT_MICROBATCH_REPORT_MD
    )
    code, _ = run_preflight(
        mode=args.mode,
        run_dir=args.run_dir,
        spec_path=args.spec,
        report_json=report_json,
        report_md=report_md,
        noise_report_path=args.noise_report,
    )
    return code


if __name__ == "__main__":
    raise SystemExit(main())
