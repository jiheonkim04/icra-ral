"""Frozen Stage 0 execution for action-consistent missing-view distillation."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import pathlib
import random
import shutil
import statistics
import sys
import time
import traceback
from collections import defaultdict
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from tca_map.action_consistent_missing_view_distillation.adapter import (
    ActionConsistentMissingViewAdapter,
    adapter_parameter_count,
    state_dict_parameter_count,
)
from tca_map.action_consistent_missing_view_distillation.preflight import (
    GRIPPER_INDICES,
    ROTATION_INDICES,
    TRANSLATION_INDICES,
    ActionHiddenHook,
    FrozenRuntime,
    _new_adapter,
    _transformer_raw,
    atomic_write_json,
    atomic_write_text,
    disk_report,
    git_head,
    gradient_global_norm,
    load_frozen_xvla,
    meminfo,
    nvidia_smi,
    parameter_vector,
    prepared_batch,
    sha256_file,
    stable_seed,
    t_schedule,
    task_key,
    timestamp,
)


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPO_ROOT / "configs" / "action_consistent_missing_view_distillation_xvla_frozen_spec.json"
DEFAULT_CONTRACT = (
    REPO_ROOT
    / "configs"
    / "action_consistent_missing_view_distillation_xvla_stage0_execution_contract.json"
)
DEFAULT_THRESHOLD_REPORT = (
    REPO_ROOT / "reports" / "action_consistent_missing_view_distillation_numerical_threshold_freeze_result.json"
)
DEFAULT_MICROBATCH_REPORT = (
    REPO_ROOT / "reports" / "action_consistent_missing_view_distillation_microbatch_preflight_result.json"
)
DEFAULT_RESULT_JSON = (
    REPO_ROOT / "reports" / "action_consistent_missing_view_distillation_resumed_stage0_result.json"
)
DEFAULT_RESULT_MD = (
    REPO_ROOT / "reports" / "action_consistent_missing_view_distillation_resumed_stage0_result.md"
)
DEFAULT_TELEMETRY = (
    REPO_ROOT / "reports" / "action_consistent_missing_view_distillation_stage0_runtime_telemetry.json"
)
TRACKED_CHECKPOINT_DIR = (
    REPO_ROOT / "reports" / "checkpoints" / "action_consistent_missing_view_distillation_stage0"
)

ARM_NAMES = (
    "OURS_FULL",
    "NO_RECONSTRUCTION",
    "NO_RAW_GRIPPER_MARGIN",
    "GENERIC_WRIST_DROPOUT_ADAPTER",
)
METRIC_NAMES = (
    "translation_RMSE",
    "rotation_RMSE",
    "raw_gripper_margin_MAE",
    "action_hidden_MSE",
)


def load_json(path: pathlib.Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def state_dict_sha256(state: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("utf-8"))
        digest.update(json.dumps(list(value.shape)).encode("utf-8"))
        digest.update(value.numpy().tobytes(order="C"))
    return digest.hexdigest()


def frozen_parameter_guard(module: nn.Module) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    total = 0
    trainable = 0
    gradient_count = 0
    for name, parameter in module.named_parameters():
        flat = parameter.detach().reshape(-1)
        samples: list[float] = []
        if flat.numel():
            for index in sorted({0, flat.numel() // 2, flat.numel() - 1}):
                samples.append(float(flat[index].float().cpu()))
        records.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "dtype": str(parameter.dtype),
                "device": str(parameter.device),
                "data_ptr": int(parameter.data_ptr()),
                "version": int(parameter._version),
                "requires_grad": bool(parameter.requires_grad),
                "samples": samples,
            }
        )
        total += int(parameter.numel())
        trainable += int(parameter.numel()) if parameter.requires_grad else 0
        gradient_count += int(parameter.grad is not None)
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return {
        "sha256": hashlib.sha256(encoded).hexdigest(),
        "parameter_tensor_count": len(records),
        "parameter_count": total,
        "trainable_parameter_count": trainable,
        "gradient_tensor_count": gradient_count,
    }


def learning_rate_for_step(step: int, *, total_steps: int = 128) -> float:
    peak = 3e-4
    end = 3e-5
    warmup = 8
    if step < 1 or step > total_steps:
        raise ValueError("step outside frozen schedule")
    if step <= warmup:
        return peak * step / warmup
    progress = (step - warmup) / (total_steps - warmup)
    return end + (peak - end) * 0.5 * (1.0 + math.cos(math.pi * progress))


def materialize_stage0_records(
    spec: dict[str, Any], run_dir: pathlib.Path, heartbeat: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    source_root = str(spec["xvla"]["source_root"])
    if source_root in sys.path:
        sys.path.remove(source_root)
    sys.path.insert(0, source_root)
    from tca_map.rifa_xvla.stage0 import install_optional_xvla_shims, materialize_xvla_clip

    install_optional_xvla_shims()
    from tca_map.cvlr_xvla.stage0 import read_fixed_official_samples

    positions = list(spec["data_splits"]["discovery"]["official_reader_positions"])
    root = run_dir / "materialized_stage0_rows"
    root.mkdir(parents=True, exist_ok=False)
    training: list[dict[str, Any]] = []
    validation: list[dict[str, Any]] = []
    manifests: list[dict[str, Any]] = []
    total_demos = len(spec["training_panel"]) * 41
    completed = 0
    for task in spec["training_panel"]:
        for split, demo_indices in (("discovery", range(40)), ("validation", (40,))):
            for demo_index in demo_indices:
                completed += 1
                if completed == 1 or completed % 8 == 0 or completed == total_demos:
                    heartbeat(f"materialize_demo_{completed}_of_{total_demos}")
                output = root / f"{task_key(task)}_{split}_demo{demo_index}"
                manifest = materialize_xvla_clip(
                    pathlib.Path(task["hdf5"]),
                    output,
                    demo_index=int(demo_index),
                    instruction=str(task["instruction"]),
                    clip_steps=48,
                )
                samples = read_fixed_official_samples(pathlib.Path(manifest["meta_path"]), positions)
                target = training if split == "discovery" else validation
                for position, sample in zip(positions, samples):
                    target.append(
                        {
                            "task_key": task_key(task),
                            "split": split,
                            "demo_index": int(demo_index),
                            "reader_position": int(position),
                            "sample": sample,
                        }
                    )
                manifest.pop("agent_frame")
                manifest.pop("wrist_frame")
                manifest["task_key"] = task_key(task)
                manifest["split"] = split
                manifests.append(manifest)
    if len(training) != 480 or len(validation) != 12:
        raise RuntimeError(f"frozen Stage 0 split drift: {len(training)} / {len(validation)}")
    return training, validation, manifests


def paired_stage0_forward(
    model: nn.Module,
    hook: ActionHiddenHook,
    adapter: ActionConsistentMissingViewAdapter,
    clean: dict[str, torch.Tensor],
    dropout: dict[str, torch.Tensor],
    records: list[dict[str, Any]],
    exposure_ordinals: list[int],
    *,
    seed: int,
    effective_offset: int = 0,
    explicit_epsilon_seeds: list[int] | None = None,
) -> dict[str, torch.Tensor]:
    action = clean["action"]
    if not torch.equal(action, dropout["action"]):
        raise ValueError("teacher/student demonstration action mismatch")
    if not torch.equal(clean["image_mask"], dropout["image_mask"]):
        raise ValueError("frozen wrist blackout must preserve image_mask")
    if len(records) != action.shape[0] or len(exposure_ordinals) != action.shape[0]:
        raise ValueError("per-exposure seed metadata mismatch")
    t = t_schedule(action.shape[0], effective_offset=effective_offset).to(
        device=action.device, dtype=action.dtype
    )
    epsilon_rows: list[torch.Tensor] = []
    for index, (record, ordinal) in enumerate(zip(records, exposure_ordinals)):
        generator = torch.Generator(device=action.device)
        epsilon_seed = (
            int(explicit_epsilon_seeds[index])
            if explicit_epsilon_seeds is not None
            else stable_seed(
                seed, ordinal, record["task_key"], record["demo_index"], record["reader_position"]
            )
        )
        generator.manual_seed(epsilon_seed)
        epsilon_rows.append(
            torch.randn(
                action[index].shape,
                generator=generator,
                device=action.device,
                dtype=action.dtype,
            )
        )
    epsilon = torch.stack(epsilon_rows, dim=0)
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
        torch.ones((action.shape[0], 1), device=action.device, dtype=action.dtype),
        compute_reconstruction=True,
    )
    with torch.no_grad():
        dropout_enc = model.forward_vlm(
            dropout["input_ids"], dropout["image_input"], dropout["image_mask"]
        )
    student_raw = _transformer_raw(model, dropout, dropout_enc, x_t, t)
    if hook.last_hidden_before is None or hook.last_hidden_after is None or hook.last_reconstruction is None:
        raise RuntimeError("student hook did not expose frozen Stage 0 tensors")
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


def raw_components(outputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
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


def arm_loss(
    arm: str,
    components: dict[str, torch.Tensor],
    denominators: dict[str, float],
    model: nn.Module,
    student_raw: torch.Tensor,
    demonstration_action: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    normalized = {
        key: components[key] / float(denominators[key])
        for key in (
            "hidden_mse",
            "translation_mse",
            "rotation_mse",
            "raw_gripper_margin_mse",
            "wrist_reconstruction_mse",
        )
    }
    if arm == "OURS_FULL":
        loss = (
            0.25 * normalized["hidden_mse"]
            + normalized["translation_mse"]
            + normalized["rotation_mse"]
            + normalized["raw_gripper_margin_mse"]
            + 0.25 * normalized["wrist_reconstruction_mse"]
        )
        return loss, normalized
    if arm == "NO_RECONSTRUCTION":
        loss = (
            0.25 * normalized["hidden_mse"]
            + normalized["translation_mse"]
            + normalized["rotation_mse"]
            + normalized["raw_gripper_margin_mse"]
        )
        return loss, normalized
    if arm == "NO_RAW_GRIPPER_MARGIN":
        loss = (
            0.25 * normalized["hidden_mse"]
            + normalized["translation_mse"]
            + normalized["rotation_mse"]
            + 0.25 * normalized["wrist_reconstruction_mse"]
        )
        return loss, normalized
    if arm == "GENERIC_WRIST_DROPOUT_ADAPTER":
        official = model.action_space.compute_loss(student_raw, demonstration_action)
        generic = {
            "generic_position_normalized": official["position_loss"] / 500.0,
            "generic_rotation_normalized": official["rotate6D_loss"] / 10.0,
            "generic_gripper": official["gripper_loss"],
        }
        loss = sum(generic.values())
        return loss, {**normalized, **generic}
    raise ValueError(f"unknown arm {arm}")


def checkpoint_adapter(
    adapter: ActionConsistentMissingViewAdapter,
    spec: dict[str, Any],
    path: pathlib.Path,
    *,
    arm: str,
    step: int,
    contract_sha256: str,
) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    current = parameter_vector(adapter)
    payload = {
        "adapter_state_dict": adapter.state_dict(),
        "arm": arm,
        "step": int(step),
        "method_spec_sha256": sha256_file(DEFAULT_SPEC),
        "execution_contract_sha256": contract_sha256,
        "trainable_parameter_count": adapter_parameter_count(adapter),
    }
    torch.save(payload, path)
    reloaded_payload = torch.load(path, map_location="cpu", weights_only=True)
    reloaded = _new_adapter(spec, torch.device("cpu"))
    reloaded.load_state_dict(reloaded_payload["adapter_state_dict"], strict=True)
    reload_delta = float(torch.linalg.vector_norm(parameter_vector(reloaded) - current))
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "bytes": int(path.stat().st_size),
        "state_dict_sha256": state_dict_sha256(
            {key: value.detach().cpu() for key, value in adapter.state_dict().items()}
        ),
        "reload_parameter_delta_l2": reload_delta,
        "disk_reload_ok": bool(reload_delta == 0.0),
    }


def train_arm(
    arm: str,
    spec: dict[str, Any],
    contract: dict[str, Any],
    thresholds: dict[str, Any],
    runtime: FrozenRuntime,
    hook: ActionHiddenHook,
    training_records: list[dict[str, Any]],
    device: torch.device,
    device_index: int,
    run_dir: pathlib.Path,
    heartbeat: Any,
    baseline_swap: int,
) -> tuple[ActionConsistentMissingViewAdapter, dict[str, Any]]:
    budget = spec["training_budget"]
    seed = int(budget["seed"])
    initialization_seed = stable_seed(seed, "adapter_initialization")
    random.seed(initialization_seed)
    np.random.seed(initialization_seed % (2**32))
    torch.manual_seed(initialization_seed)
    torch.cuda.manual_seed_all(initialization_seed)
    adapter = _new_adapter(spec, device)
    adapter.train()
    initial_vector = parameter_vector(adapter)
    initial_state_sha256 = state_dict_sha256(
        {key: value.detach().cpu() for key, value in adapter.state_dict().items()}
    )
    initial_projection_zero = bool(
        torch.count_nonzero(adapter.action_residual_output.weight).item() == 0
        and torch.count_nonzero(adapter.action_residual_output.bias).item() == 0
        and torch.count_nonzero(adapter.reconstruction_output.weight).item() == 0
        and torch.count_nonzero(adapter.reconstruction_output.bias).item() == 0
    )
    optimizer = torch.optim.AdamW(
        adapter.parameters(),
        lr=float(budget["learning_rate_peak"]),
        betas=tuple(float(value) for value in budget["betas"]),
        eps=float(budget["epsilon"]),
        weight_decay=float(budget["weight_decay"]),
    )
    optimizer_parameter_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    model_parameter_ids = {id(parameter) for parameter in runtime.model.parameters()}
    if optimizer_parameter_ids & model_parameter_ids:
        raise RuntimeError("adapter optimizer unexpectedly contains frozen X-VLA parameters")

    denominators = thresholds["normalization_denominators"]
    steps = int(budget["optimizer_steps_per_arm"])
    batch_size = int(contract["microbatch_preflight"]["selected_microbatch"])
    losses: list[float] = []
    gradient_norms: list[float] = []
    gradient_tensor_counts: list[int] = []
    nonzero_gradient_tensor_counts: list[int] = []
    learning_rates: list[float] = []
    component_histories: dict[str, list[float]] = defaultdict(list)
    checkpoints: dict[str, dict[str, Any]] = {}
    peak_ram_fraction = meminfo()["mem_used_fraction"]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    contract_hash = sha256_file(DEFAULT_CONTRACT)
    arm_slug = arm.lower()
    for step_index in range(steps):
        step = step_index + 1
        learning_rate = learning_rate_for_step(step, total_steps=steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        indices = [int((step_index * batch_size + offset) % len(training_records)) for offset in range(batch_size)]
        batch = [training_records[index] for index in indices]
        ordinals = [step_index * batch_size + offset for offset in range(batch_size)]
        clean = prepared_batch(batch, runtime.processor, device, condition="clean")
        dropout = prepared_batch(batch, runtime.processor, device, condition="mask_1_in_hand_dropout")
        optimizer.zero_grad(set_to_none=True)
        outputs = paired_stage0_forward(
            runtime.model,
            hook,
            adapter,
            clean,
            dropout,
            batch,
            ordinals,
            seed=seed,
        )
        components = raw_components(outputs)
        loss, recorded_components = arm_loss(
            arm,
            components,
            denominators,
            runtime.model,
            outputs["student_raw"],
            dropout["action"],
        )
        if not bool(torch.isfinite(loss).item()):
            raise RuntimeError(f"nonfinite {arm} loss at step {step}")
        loss.backward()
        gradient_norm, gradient_tensor_count, nonzero_gradient_tensor_count = gradient_global_norm(adapter)
        if not math.isfinite(gradient_norm) or gradient_norm <= 0 or nonzero_gradient_tensor_count <= 0:
            raise RuntimeError(f"invalid {arm} gradients at step {step}")
        torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(budget["max_gradient_norm"]))
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        gradient_norms.append(float(gradient_norm))
        gradient_tensor_counts.append(int(gradient_tensor_count))
        nonzero_gradient_tensor_counts.append(int(nonzero_gradient_tensor_count))
        learning_rates.append(float(learning_rate))
        for key, value in components.items():
            component_histories[f"raw_{key}"].append(float(value.detach().cpu()))
        for key, value in recorded_components.items():
            component_histories[f"objective_{key}"].append(float(value.detach().cpu()))

        memory = meminfo()
        peak_ram_fraction = max(peak_ram_fraction, float(memory["mem_used_fraction"]))
        if memory["mem_used_fraction"] > 0.82:
            raise RuntimeError(f"{arm} exceeded frozen system RAM ceiling")
        if memory["swap_used_bytes"] - baseline_swap > 0:
            raise RuntimeError(f"{arm} caused forbidden swap growth")
        reserved = int(torch.cuda.max_memory_reserved(device_index))
        total_gpu = int(torch.cuda.get_device_properties(device_index).total_memory)
        if reserved / total_gpu > 0.88:
            raise RuntimeError(f"{arm} exceeded frozen VRAM ceiling")
        if step in tuple(int(value) for value in budget["checkpoint_steps"]):
            checkpoint_path = run_dir / "checkpoints" / f"{arm_slug}_step{step:04d}.pt"
            checkpoints[str(step)] = checkpoint_adapter(
                adapter,
                spec,
                checkpoint_path,
                arm=arm,
                step=step,
                contract_sha256=contract_hash,
            )
        if step == 1 or step % 8 == 0 or step == steps:
            heartbeat(f"train_{arm_slug}_step_{step}_of_{steps}")

    final_vector = parameter_vector(adapter)
    weight_change = float(torch.linalg.vector_norm(final_vector - initial_vector))
    final_payload = torch.load(
        pathlib.Path(checkpoints["128"]["path"]), map_location=device, weights_only=True
    )
    disk_reloaded_adapter = _new_adapter(spec, device)
    disk_reloaded_adapter.load_state_dict(final_payload["adapter_state_dict"], strict=True)
    final_disk_reload_delta = float(
        torch.linalg.vector_norm(parameter_vector(disk_reloaded_adapter) - final_vector)
    )
    if final_disk_reload_delta != 0.0:
        raise RuntimeError(f"{arm} final checkpoint did not reload exactly")
    adapter = disk_reloaded_adapter.eval()
    final_state_sha256 = state_dict_sha256(
        {key: value.detach().cpu() for key, value in adapter.state_dict().items()}
    )
    metrics = {
        "arm": arm,
        "trainable_parameter_count": adapter_parameter_count(adapter),
        "parameter_device": str(next(adapter.parameters()).device),
        "optimizer_steps": len(losses),
        "record_exposures": len(losses) * batch_size,
        "microbatch": batch_size,
        "gradient_accumulation": 1,
        "first_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "maximum_loss": max(losses),
        "losses": losses,
        "learning_rates": learning_rates,
        "component_histories": dict(component_histories),
        "gradient_norms": gradient_norms,
        "gradient_tensor_counts": gradient_tensor_counts,
        "nonzero_gradient_tensor_counts": nonzero_gradient_tensor_counts,
        "finite_nonzero_gradients": bool(
            all(math.isfinite(value) and value > 0 for value in gradient_norms)
            and all(value > 0 for value in nonzero_gradient_tensor_counts)
        ),
        "weight_change_l2": weight_change,
        "weights_changed": bool(weight_change > 0),
        "initial_state_sha256": initial_state_sha256,
        "final_state_sha256": final_state_sha256,
        "initial_output_projections_exact_zero": initial_projection_zero,
        "optimizer_disjoint_from_frozen_xvla": True,
        "checkpoints": checkpoints,
        "final_checkpoint_step": 128,
        "validation_uses_disk_reloaded_final_checkpoint": True,
        "final_disk_reload_parameter_delta_l2": final_disk_reload_delta,
        "peak_cuda_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
        "peak_cuda_reserved_bytes": int(torch.cuda.max_memory_reserved(device_index)),
        "peak_system_ram_used_fraction": peak_ram_fraction,
        "research_induced_swap_growth_bytes": int(meminfo()["swap_used_bytes"] - baseline_swap),
    }
    return adapter, metrics


def generate_plan(
    model: nn.Module,
    hook: ActionHiddenHook,
    inputs: dict[str, torch.Tensor],
    *,
    adapter: ActionConsistentMissingViewAdapter | None,
    seed: int,
    steps: int = 10,
) -> tuple[torch.Tensor, torch.Tensor]:
    if adapter is None:
        hook.deactivate()
    else:
        hook.activate_student(
            adapter,
            torch.ones((inputs["input_ids"].shape[0], 1), device=inputs["proprio"].device),
            compute_reconstruction=False,
        )
    generator = torch.Generator(device=inputs["proprio"].device)
    generator.manual_seed(int(seed))
    with torch.no_grad():
        enc = model.forward_vlm(inputs["input_ids"], inputs["image_input"], inputs["image_mask"])
        x1 = torch.randn(
            (inputs["input_ids"].shape[0], model.num_actions, model.action_space.dim_action),
            generator=generator,
            device=inputs["proprio"].device,
            dtype=inputs["proprio"].dtype,
        )
        action = torch.zeros_like(x1)
        for index in range(steps, 0, -1):
            t = torch.full((action.shape[0],), index / steps, device=action.device, dtype=action.dtype)
            x_t = x1 * t.view(-1, 1, 1) + action * (1.0 - t).view(-1, 1, 1)
            action = _transformer_raw(model, inputs, enc, x_t, t)
        raw = action.detach().clone()
        post = model.action_space.postprocess(raw.detach().clone())
    hook.deactivate()
    return raw, post


def teacher_agreement_row(outputs: dict[str, torch.Tensor]) -> dict[str, float]:
    student_raw = outputs["student_raw"].detach()
    teacher_raw = outputs["teacher_raw"].detach()
    return {
        "translation_RMSE": float(
            torch.sqrt(F.mse_loss(student_raw[..., TRANSLATION_INDICES], teacher_raw[..., TRANSLATION_INDICES])).cpu()
        ),
        "rotation_RMSE": float(
            torch.sqrt(F.mse_loss(student_raw[..., ROTATION_INDICES], teacher_raw[..., ROTATION_INDICES])).cpu()
        ),
        "raw_gripper_margin_MAE": float(
            F.l1_loss(student_raw[..., GRIPPER_INDICES], teacher_raw[..., GRIPPER_INDICES]).cpu()
        ),
        "action_hidden_MSE": float(F.mse_loss(outputs["student_hidden"], outputs["teacher_hidden"]).cpu()),
        "wrist_reconstruction_MSE": float(F.mse_loss(outputs["reconstruction"], outputs["clean_wrist"]).cpu()),
    }


def plan_diagnostics(raw: torch.Tensor, post: torch.Tensor, teacher_raw: torch.Tensor) -> dict[str, Any]:
    raw = raw.detach().float()
    post = post.detach().float()
    teacher_raw = teacher_raw.detach().float()
    raw_gripper = raw[..., GRIPPER_INDICES]
    teacher_gripper = teacher_raw[..., GRIPPER_INDICES]
    decisions = raw_gripper >= 0.0
    teacher_decisions = teacher_gripper >= 0.0
    disagreement = decisions != teacher_decisions
    indices = torch.nonzero(disagreement, as_tuple=False).cpu().tolist()
    translation_deltas = torch.abs(torch.diff(post[..., TRANSLATION_INDICES], dim=1)).reshape(-1)
    rotation_deltas = torch.abs(torch.diff(post[..., ROTATION_INDICES], dim=1)).reshape(-1)
    finite = bool(torch.isfinite(raw).all().item() and torch.isfinite(post).all().item())
    post_gripper = post[..., GRIPPER_INDICES]
    range_violations = int(((post_gripper < 0.0) | (post_gripper > 1.0)).sum().item())
    return {
        "all_finite": finite,
        "official_action_range_violation_count": range_violations,
        "raw_gripper_threshold_distance_mean": float(torch.abs(raw_gripper).mean().cpu()),
        "raw_gripper_threshold_distance_min": float(torch.abs(raw_gripper).min().cpu()),
        "post_gripper_score_min": float(post_gripper.min().cpu()),
        "post_gripper_score_max": float(post_gripper.max().cpu()),
        "discrete_gripper_disagreement_count": int(disagreement.sum().item()),
        "discrete_gripper_flip_indices": indices,
        "translation_adjacent_abs": translation_deltas.cpu().tolist(),
        "rotation_adjacent_abs": rotation_deltas.cpu().tolist(),
    }


def aggregate_metric_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for metric in METRIC_NAMES:
        values = [float(row[metric]) for row in rows]
        if metric.endswith("RMSE"):
            result[metric] = float(math.sqrt(sum(value * value for value in values) / len(values)))
        else:
            result[metric] = float(statistics.fmean(values))
    result["wrist_reconstruction_MSE"] = float(
        statistics.fmean(float(row["wrist_reconstruction_MSE"]) for row in rows)
    )
    per_task: dict[str, Any] = {}
    for task in sorted({str(row["task_key"]) for row in rows}):
        task_rows = [row for row in rows if row["task_key"] == task]
        per_task[task] = aggregate_metric_rows_without_tasks(task_rows)
    result["per_task"] = per_task
    return result


def aggregate_metric_rows_without_tasks(rows: list[dict[str, Any]]) -> dict[str, float]:
    result: dict[str, float] = {}
    for metric in METRIC_NAMES:
        values = [float(row[metric]) for row in rows]
        result[metric] = (
            float(math.sqrt(sum(value * value for value in values) / len(values)))
            if metric.endswith("RMSE")
            else float(statistics.fmean(values))
        )
    result["wrist_reconstruction_MSE"] = float(
        statistics.fmean(float(row["wrist_reconstruction_MSE"]) for row in rows)
    )
    return result


def bootstrap_difference(
    comparator: list[float], full: list[float], *, seed: int, repetitions: int = 2000
) -> dict[str, float]:
    left = np.asarray(comparator, dtype=np.float64)
    right = np.asarray(full, dtype=np.float64)
    if left.shape != right.shape or left.ndim != 1 or left.size == 0:
        raise ValueError("invalid paired bootstrap inputs")
    difference = left - right
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, difference.size, size=(repetitions, difference.size))
    estimates = difference[indices].mean(axis=1)
    return {
        "point_mean_row_difference": float(difference.mean()),
        "ci95_low": float(np.quantile(estimates, 0.025)),
        "ci95_high": float(np.quantile(estimates, 0.975)),
        "repetitions": int(repetitions),
    }


def comparison_result(
    full_summary: dict[str, Any],
    comparator_summary: dict[str, Any],
    full_rows: list[dict[str, Any]],
    comparator_rows: list[dict[str, Any]],
    thresholds: dict[str, Any],
    *,
    seed: int,
    full_gripper_disagreements: int,
    comparator_gripper_disagreements: int,
) -> dict[str, Any]:
    absolute_thresholds = thresholds["frozen_practical_improvement_thresholds"]
    noise = thresholds["measured_fixed_seed_repeat_noise"]
    floors = {
        "action_hidden_MSE": float(absolute_thresholds["action_hidden_MSE_absolute_min"]),
        "translation_RMSE": float(absolute_thresholds["translation_RMSE_absolute_min"]),
        "rotation_RMSE": float(absolute_thresholds["rotation_RMSE_absolute_min"]),
        "raw_gripper_margin_MAE": float(absolute_thresholds["raw_gripper_margin_MAE_absolute_min"]),
    }
    metrics: dict[str, Any] = {}
    practical: list[str] = []
    ci_supported: list[str] = []
    nonregression = True
    for index, metric in enumerate(METRIC_NAMES):
        full_value = float(full_summary[metric])
        comparator_value = float(comparator_summary[metric])
        absolute = comparator_value - full_value
        relative = absolute / comparator_value if comparator_value > 0 else float("-inf")
        practical_pass = bool(relative >= 0.05 and absolute >= floors[metric])
        allowed_regression = max(0.02 * comparator_value, 10.0 * float(noise[metric]))
        metric_nonregression = bool(full_value <= comparator_value + allowed_regression)
        nonregression = nonregression and metric_nonregression
        bootstrap = bootstrap_difference(
            [float(row[metric]) for row in comparator_rows],
            [float(row[metric]) for row in full_rows],
            seed=stable_seed(seed, "bootstrap", metric, index),
        )
        if practical_pass:
            practical.append(metric)
            if bootstrap["ci95_low"] > 0:
                ci_supported.append(metric)
        metrics[metric] = {
            "full": full_value,
            "comparator": comparator_value,
            "absolute_improvement": absolute,
            "relative_improvement": relative,
            "absolute_threshold": floors[metric],
            "practical_pass": practical_pass,
            "allowed_regression": allowed_regression,
            "nonregression": metric_nonregression,
            "paired_bootstrap": bootstrap,
        }
    gripper_nonregression = bool(full_gripper_disagreements <= comparator_gripper_disagreements)
    return {
        "metrics": metrics,
        "practical_pass_metrics": practical,
        "ci_supported_practical_metrics": ci_supported,
        "practical_any": bool(practical),
        "other_metric_nonregression": nonregression,
        "full_discrete_gripper_disagreements": int(full_gripper_disagreements),
        "comparator_discrete_gripper_disagreements": int(comparator_gripper_disagreements),
        "gripper_nonregression": gripper_nonregression,
        "point_gate": bool(practical and nonregression and gripper_nonregression),
        "ci_gate": bool(ci_supported),
    }


def adjudicate_stage0(gates: dict[str, bool], comparisons: dict[str, Any]) -> str:
    if not gates["execution_valid"]:
        return "STAGE0_IMPLEMENTATION_OR_RESOURCE_FAILURE"
    if not gates["action_legality_and_smoothness"]:
        return "STAGE0_ACTION_LEGALITY_FAILURE"
    mechanism = comparisons["full_vs_no_reconstruction"]
    generic = comparisons["full_vs_generic"]
    if gates["reconstruction_gate"] and gates["base_directional_gate"] and mechanism["point_gate"]:
        if not generic["point_gate"]:
            return "STAGE0_GENERIC_ADAPTATION_EXPLAINS_GAIN"
        if mechanism["ci_gate"] and generic["ci_gate"]:
            return "STAGE0_GO"
        return "STAGE0_PROMISING_NEEDS_ONE_FIXED_CONFIRMATION"
    mechanism_improvements = [
        float(value["absolute_improvement"]) for value in mechanism["metrics"].values()
    ]
    if (
        gates["reconstruction_gate"]
        and gates["base_directional_gate"]
        and mechanism["other_metric_nonregression"]
        and mechanism["gripper_nonregression"]
        and all(value > 0 for value in mechanism_improvements)
    ):
        return "STAGE0_PROMISING_NEEDS_ONE_FIXED_CONFIRMATION"
    if (
        gates["base_directional_gate"]
        and mechanism["other_metric_nonregression"]
        and mechanism["gripper_nonregression"]
        and any(value > 0 for value in mechanism_improvements)
    ):
        return "STAGE0_UNDERPOWERED_ONE_FIXED_EXPANSION_ALLOWED"
    return "STAGE0_MECHANISM_NOT_SUPPORTED"


def percentile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def measure_latency(
    runtime: FrozenRuntime,
    hook: ActionHiddenHook,
    record: dict[str, Any],
    adapters: dict[str, ActionConsistentMissingViewAdapter],
    device: torch.device,
    device_index: int,
    seed: int,
    heartbeat: Any,
) -> dict[str, Any]:
    dropout = prepared_batch([record], runtime.processor, device, condition="mask_1_in_hand_dropout")
    policies: dict[str, ActionConsistentMissingViewAdapter | None] = {"BASE": None, **adapters}
    result: dict[str, Any] = {}
    for policy, adapter in policies.items():
        heartbeat(f"latency_{policy.lower()}")
        for _ in range(10):
            generate_plan(runtime.model, hook, dropout, adapter=adapter, seed=seed, steps=10)
        torch.cuda.synchronize(device_index)
        total_ms: list[float] = []
        for _ in range(100):
            started = time.perf_counter()
            generate_plan(runtime.model, hook, dropout, adapter=adapter, seed=seed, steps=10)
            torch.cuda.synchronize(device_index)
            total_ms.append((time.perf_counter() - started) * 1000.0)
        policy_result: dict[str, Any] = {
            "total_query_ms": {
                "mean": float(statistics.fmean(total_ms)),
                "median": float(statistics.median(total_ms)),
                "p95": percentile(total_ms, 0.95),
                "warmups": 10,
                "measurements": 100,
            }
        }
        if adapter is not None:
            hidden = torch.zeros((1, 30, 1024), device=device, dtype=torch.float32)
            indicator = torch.ones((1, 1), device=device, dtype=torch.float32)
            for _ in range(10):
                adapter(hidden, indicator, compute_reconstruction=False)
            times: list[float] = []
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            for _ in range(100):
                start_event.record()
                adapter(hidden, indicator, compute_reconstruction=False)
                end_event.record()
                end_event.synchronize()
                times.append(float(start_event.elapsed_time(end_event)))
            policy_result["adapter_only_cuda_ms"] = {
                "mean": float(statistics.fmean(times)),
                "median": float(statistics.median(times)),
                "p95": percentile(times, 0.95),
                "warmups": 10,
                "measurements": 100,
            }
        result[policy] = policy_result
    return result


def evaluate_stage0(
    spec: dict[str, Any],
    contract: dict[str, Any],
    thresholds: dict[str, Any],
    runtime: FrozenRuntime,
    hook: ActionHiddenHook,
    validation_records: list[dict[str, Any]],
    adapters: dict[str, ActionConsistentMissingViewAdapter],
    device: torch.device,
    device_index: int,
    heartbeat: Any,
) -> dict[str, Any]:
    seed = int(spec["training_budget"]["seed"])
    initialization_seed = stable_seed(seed, "adapter_initialization")
    torch.manual_seed(initialization_seed)
    torch.cuda.manual_seed_all(initialization_seed)
    base_adapter = _new_adapter(spec, device).eval()
    metric_rows: dict[str, list[dict[str, Any]]] = {"BASE": []}
    metric_rows.update({arm: [] for arm in ARM_NAMES})
    plan_rows: dict[str, list[dict[str, Any]]] = {"BASE": []}
    plan_rows.update({arm: [] for arm in ARM_NAMES})
    clean_bypass: dict[str, dict[str, Any]] = {
        arm: {"max_raw_delta": 0.0, "max_post_delta": 0.0, "exact": True} for arm in ARM_NAMES
    }
    teacher_forward_before = hook.forward_counts["teacher_capture"]
    student_forward_before = hook.forward_counts["student_adapter"]
    for row_index, record in enumerate(validation_records):
        heartbeat(f"validation_row_{row_index + 1}_of_{len(validation_records)}")
        clean = prepared_batch([record], runtime.processor, device, condition="clean")
        dropout = prepared_batch([record], runtime.processor, device, condition="mask_1_in_hand_dropout")
        validation_seed = stable_seed(seed, "validation", record["task_key"], record["demo_index"], record["reader_position"])
        base_outputs = paired_stage0_forward(
            runtime.model,
            hook,
            base_adapter,
            clean,
            dropout,
            [record],
            [validation_seed],
            seed=seed,
            effective_offset=row_index % 8,
            explicit_epsilon_seeds=[validation_seed],
        )
        base_metrics = teacher_agreement_row(base_outputs)
        base_metrics["task_key"] = record["task_key"]
        base_metrics["reader_position"] = record["reader_position"]
        metric_rows["BASE"].append(base_metrics)
        generation_seed = stable_seed(
            seed, "validation_generation", record["task_key"], record["demo_index"], record["reader_position"]
        )
        teacher_raw, teacher_post = generate_plan(
            runtime.model, hook, clean, adapter=None, seed=generation_seed, steps=10
        )
        base_raw, base_post = generate_plan(
            runtime.model, hook, dropout, adapter=None, seed=generation_seed, steps=10
        )
        base_diag = plan_diagnostics(base_raw, base_post, teacher_raw)
        base_diag.update({"task_key": record["task_key"], "reader_position": record["reader_position"]})
        plan_rows["BASE"].append(base_diag)
        for arm, adapter in adapters.items():
            outputs = paired_stage0_forward(
                runtime.model,
                hook,
                adapter,
                clean,
                dropout,
                [record],
                [validation_seed],
                seed=seed,
                effective_offset=row_index % 8,
                explicit_epsilon_seeds=[validation_seed],
            )
            metrics = teacher_agreement_row(outputs)
            metrics["task_key"] = record["task_key"]
            metrics["reader_position"] = record["reader_position"]
            metric_rows[arm].append(metrics)
            arm_raw, arm_post = generate_plan(
                runtime.model, hook, dropout, adapter=adapter, seed=generation_seed, steps=10
            )
            diagnostics = plan_diagnostics(arm_raw, arm_post, teacher_raw)
            diagnostics.update({"task_key": record["task_key"], "reader_position": record["reader_position"]})
            plan_rows[arm].append(diagnostics)
            clean_raw, clean_post = generate_plan(
                runtime.model, hook, clean, adapter=None, seed=generation_seed, steps=10
            )
            raw_delta = float(torch.max(torch.abs(clean_raw - teacher_raw)).cpu())
            post_delta = float(torch.max(torch.abs(clean_post - teacher_post)).cpu())
            clean_bypass[arm]["max_raw_delta"] = max(clean_bypass[arm]["max_raw_delta"], raw_delta)
            clean_bypass[arm]["max_post_delta"] = max(clean_bypass[arm]["max_post_delta"], post_delta)
            clean_bypass[arm]["exact"] = bool(
                clean_bypass[arm]["exact"] and raw_delta == 0.0 and post_delta == 0.0
            )

    summaries = {policy: aggregate_metric_rows(rows) for policy, rows in metric_rows.items()}
    plan_summaries: dict[str, Any] = {}
    for policy, rows in plan_rows.items():
        translation = [value for row in rows for value in row["translation_adjacent_abs"]]
        rotation = [value for row in rows for value in row["rotation_adjacent_abs"]]
        flip_indices: list[list[int]] = []
        for row_index, row in enumerate(rows):
            for index in row["discrete_gripper_flip_indices"]:
                flip_indices.append([row_index, *index])
        plan_summaries[policy] = {
            "all_finite": all(bool(row["all_finite"]) for row in rows),
            "official_action_range_violation_count": sum(
                int(row["official_action_range_violation_count"]) for row in rows
            ),
            "discrete_gripper_disagreement_count": sum(
                int(row["discrete_gripper_disagreement_count"]) for row in rows
            ),
            "discrete_gripper_flip_indices": flip_indices,
            "raw_gripper_threshold_distance_mean": float(
                statistics.fmean(float(row["raw_gripper_threshold_distance_mean"]) for row in rows)
            ),
            "raw_gripper_threshold_distance_min": min(
                float(row["raw_gripper_threshold_distance_min"]) for row in rows
            ),
            "post_gripper_score_min": min(float(row["post_gripper_score_min"]) for row in rows),
            "post_gripper_score_max": max(float(row["post_gripper_score_max"]) for row in rows),
            "translation_adjacent_abs_p99": percentile(translation, 0.99),
            "rotation_adjacent_abs_p99": percentile(rotation, 0.99),
        }

    full = summaries["OURS_FULL"]
    no_recon = summaries["NO_RECONSTRUCTION"]
    generic = summaries["GENERIC_WRIST_DROPOUT_ADAPTER"]
    comparisons = {
        "full_vs_no_reconstruction": comparison_result(
            full,
            no_recon,
            metric_rows["OURS_FULL"],
            metric_rows["NO_RECONSTRUCTION"],
            thresholds,
            seed=seed,
            full_gripper_disagreements=plan_summaries["OURS_FULL"]["discrete_gripper_disagreement_count"],
            comparator_gripper_disagreements=plan_summaries["NO_RECONSTRUCTION"]["discrete_gripper_disagreement_count"],
        ),
        "full_vs_generic": comparison_result(
            full,
            generic,
            metric_rows["OURS_FULL"],
            metric_rows["GENERIC_WRIST_DROPOUT_ADAPTER"],
            thresholds,
            seed=stable_seed(seed, "generic"),
            full_gripper_disagreements=plan_summaries["OURS_FULL"]["discrete_gripper_disagreement_count"],
            comparator_gripper_disagreements=plan_summaries["GENERIC_WRIST_DROPOUT_ADAPTER"]["discrete_gripper_disagreement_count"],
        ),
    }
    base = summaries["BASE"]
    denominators = thresholds["normalization_denominators"]
    normalized_scales = {
        "action_hidden_MSE": float(denominators["hidden_mse"]),
        "translation_RMSE": math.sqrt(float(denominators["translation_mse"])),
        "rotation_RMSE": math.sqrt(float(denominators["rotation_mse"])),
        "raw_gripper_margin_MAE": math.sqrt(float(denominators["raw_gripper_margin_mse"])),
    }
    full_normalized_mean = float(
        statistics.fmean(float(full[metric]) / normalized_scales[metric] for metric in METRIC_NAMES)
    )
    base_normalized_mean = float(
        statistics.fmean(float(base[metric]) / normalized_scales[metric] for metric in METRIC_NAMES)
    )
    base_directional = {
        "full_normalized_mean": full_normalized_mean,
        "base_normalized_mean": base_normalized_mean,
        "improved_metrics": [metric for metric in METRIC_NAMES if float(full[metric]) < float(base[metric])],
        "pass": bool(
            full_normalized_mean < base_normalized_mean
            and any(float(full[metric]) < float(base[metric]) for metric in METRIC_NAMES)
        ),
    }
    reconstruction_task_wins = [
        task
        for task in full["per_task"]
        if float(full["per_task"][task]["wrist_reconstruction_MSE"])
        < float(no_recon["per_task"][task]["wrist_reconstruction_MSE"])
    ]
    reconstruction_gate = {
        "full": float(full["wrist_reconstruction_MSE"]),
        "no_reconstruction": float(no_recon["wrist_reconstruction_MSE"]),
        "ratio": float(full["wrist_reconstruction_MSE"] / no_recon["wrist_reconstruction_MSE"]),
        "task_wins": reconstruction_task_wins,
        "pass": bool(
            float(full["wrist_reconstruction_MSE"]) <= 0.95 * float(no_recon["wrist_reconstruction_MSE"])
            and len(reconstruction_task_wins) >= 2
        ),
    }
    smoothness = thresholds["frozen_clean_teacher_smoothness"]
    full_plan = plan_summaries["OURS_FULL"]
    legality = {
        "all_policies_finite": all(bool(value["all_finite"]) for value in plan_summaries.values()),
        "official_action_range_violation_count": sum(
            int(value["official_action_range_violation_count"]) for value in plan_summaries.values()
        ),
        "all_clean_bypass_exact": all(bool(value["exact"]) for value in clean_bypass.values()),
        "full_translation_smoothness_pass": bool(
            full_plan["translation_adjacent_abs_p99"] <= float(smoothness["translation_envelope"])
        ),
        "full_rotation_smoothness_pass": bool(
            full_plan["rotation_adjacent_abs_p99"] <= float(smoothness["rotation_envelope"])
        ),
    }
    legality["pass"] = bool(
        legality["all_policies_finite"]
        and legality["official_action_range_violation_count"] == 0
        and legality["all_clean_bypass_exact"]
        and legality["full_translation_smoothness_pass"]
        and legality["full_rotation_smoothness_pass"]
    )
    validation_forward_counts = {
        "teacher": hook.forward_counts["teacher_capture"] - teacher_forward_before,
        "student": hook.forward_counts["student_adapter"] - student_forward_before,
    }
    latency = measure_latency(
        runtime,
        hook,
        validation_records[0],
        adapters,
        device,
        device_index,
        stable_seed(seed, "latency"),
        heartbeat,
    )
    return {
        "record_count": len(validation_records),
        "checkpoint_selection_or_tuning": False,
        "metric_rows": metric_rows,
        "teacher_agreement": summaries,
        "plan_diagnostics": plan_summaries,
        "clean_bypass": clean_bypass,
        "comparisons": comparisons,
        "base_directional_gate": base_directional,
        "reconstruction_gate": reconstruction_gate,
        "action_legality_and_smoothness": legality,
        "validation_forward_counts": validation_forward_counts,
        "latency": latency,
    }


def persist_tracked_checkpoints(
    adapters: dict[str, ActionConsistentMissingViewAdapter],
    training: dict[str, Any],
    spec: dict[str, Any],
) -> dict[str, Any]:
    TRACKED_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    tracked: dict[str, Any] = {}
    for arm in ARM_NAMES:
        tracked[arm] = {}
        for step in (64, 128):
            source = pathlib.Path(training[arm]["checkpoints"][str(step)]["path"])
            destination = TRACKED_CHECKPOINT_DIR / f"{arm.lower()}_step{step:04d}.pt"
            shutil.copy2(source, destination)
            tracked[arm][str(step)] = {
                "path": str(destination.relative_to(REPO_ROOT)).replace("\\", "/"),
                "sha256": sha256_file(destination),
                "bytes": int(destination.stat().st_size),
            }
    inference = TRACKED_CHECKPOINT_DIR / "ours_full_inference_only.pt"
    torch.save(
        {
            "inference_state_dict": adapters["OURS_FULL"].inference_state_dict(),
            "method_spec_sha256": sha256_file(DEFAULT_SPEC),
            "source_arm": "OURS_FULL",
            "reconstruction_decoder_included": False,
        },
        inference,
    )
    inference_state = adapters["OURS_FULL"].inference_state_dict()
    tracked["OURS_FULL_INFERENCE_ONLY"] = {
        "path": str(inference.relative_to(REPO_ROOT)).replace("\\", "/"),
        "sha256": sha256_file(inference),
        "bytes": int(inference.stat().st_size),
        "parameter_count": state_dict_parameter_count(inference_state),
        "reconstruction_decoder_included": False,
    }
    return tracked


def write_markdown(path: pathlib.Path, result: dict[str, Any]) -> None:
    validation = result.get("validation") or {}
    summaries = validation.get("teacher_agreement") or {}
    gates = result.get("gates") or {}
    lines = [
        "# Action-Consistent Missing-View Distillation Resumed Stage 0",
        "",
        f"Decision: `{result.get('decision')}`",
        "",
        f"- Execution valid: `{gates.get('execution_valid')}`",
        f"- CUDA PID / device: `{result.get('cuda_pid')} / {(result.get('cuda_device') or {}).get('name')}`",
        f"- Training records / validation records: `{((result.get('data') or {}).get('training_record_count'))} / {((result.get('data') or {}).get('validation_record_count'))}`",
        f"- Optimizer steps per arm: `{[metrics.get('optimizer_steps') for metrics in (result.get('training') or {}).values()]}`",
        f"- Teacher / student forward counts: `{((result.get('forward_counts') or {}).get('teacher'))} / {((result.get('forward_counts') or {}).get('student'))}`",
        f"- Peak allocated / reserved VRAM bytes: `{result.get('peak_cuda_allocated_bytes')} / {result.get('peak_cuda_reserved_bytes')}`",
        f"- Peak system RAM fraction: `{result.get('peak_system_ram_used_fraction')}`",
        f"- Swap growth bytes: `{result.get('research_induced_swap_growth_bytes')}`",
        f"- Confirmatory outcomes accessed: `{result.get('confirmatory_outcomes_accessed')}`",
        "",
        "## Teacher-agreement metrics",
        "",
        "| policy | translation RMSE | rotation RMSE | raw gripper MAE | hidden MSE | reconstruction MSE |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for policy in ("BASE", *ARM_NAMES):
        values = summaries.get(policy) or {}
        lines.append(
            f"| {policy} | {values.get('translation_RMSE')} | {values.get('rotation_RMSE')} | "
            f"{values.get('raw_gripper_margin_MAE')} | {values.get('action_hidden_MSE')} | "
            f"{values.get('wrist_reconstruction_MSE')} |"
        )
    lines.extend(["", "## Frozen gates", ""])
    for name, value in gates.items():
        lines.append(f"- `{name}`: `{value}`")
    lines.extend(
        [
            "",
            "The result uses discovery demos 0..39 for optimization and validation demo 40 only. Demos 41..49 and all confirmatory simulator outcomes remain untouched. No physical robot manipulation occurred.",
        ]
    )
    atomic_write_text(path, "\n".join(lines) + "\n")


def run_stage0(
    run_dir: pathlib.Path,
    *,
    spec_path: pathlib.Path = DEFAULT_SPEC,
    contract_path: pathlib.Path = DEFAULT_CONTRACT,
    threshold_path: pathlib.Path = DEFAULT_THRESHOLD_REPORT,
    microbatch_path: pathlib.Path = DEFAULT_MICROBATCH_REPORT,
) -> tuple[int, dict[str, Any]]:
    run_dir = run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if (run_dir / "result.json").exists() or (run_dir / "heartbeat.json").exists():
        raise FileExistsError(f"Stage 0 run directory is not fresh: {run_dir}")
    heartbeat_path = run_dir / "heartbeat.json"
    status_path = run_dir / "status.json"
    partial_path = run_dir / "partial_result.json"
    exit_path = run_dir / "exit_code.txt"
    started = time.monotonic()
    spec = load_json(spec_path)
    contract = load_json(contract_path)
    thresholds = load_json(threshold_path)
    microbatch = load_json(microbatch_path)
    result: dict[str, Any] = {
        "schema_version": "2026-07-19.epoch5_action_consistent_missing_view_resumed_stage0.v1",
        "method": "ACTION_CONSISTENT_MISSING_VIEW_DISTILLATION_XVLA",
        "implementation_label": "ACTION_CONSISTENT_MISSING_VIEW_DISTILLATION_XVLA_STAGE0",
        "decision": "STAGE0_IMPLEMENTATION_OR_RESOURCE_FAILURE",
        "success": False,
        "run_dir": str(run_dir),
        "source_head": git_head(),
        "started_at": timestamp(),
        "pid": os.getpid(),
        "cuda_pid": os.getpid(),
        "spec": {"path": str(spec_path.relative_to(REPO_ROOT)), "sha256": sha256_file(spec_path)},
        "execution_contract": {
            "path": str(contract_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(contract_path),
        },
        "threshold_report": {
            "path": str(threshold_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(threshold_path),
        },
        "microbatch_report": {
            "path": str(microbatch_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(microbatch_path),
        },
        "historical_pre_resumption_decisions": {
            "stage0": "STAGE0_IMPLEMENTATION_OR_RESOURCE_FAILURE",
            "paper_level": "IMPLEMENTATION_DATA_OR_RESOURCE_FAILURE",
        },
        "confirmatory_outcomes_accessed": False,
        "closed_loop_rollout_executed": False,
        "physical_robot_manipulation": False,
        "downloads_used": False,
        "model_offload_used": False,
        "exceptions": [],
        "nvidia_smi_before": nvidia_smi(),
        "system_memory_before": meminfo(),
        "disk_before": disk_report(REPO_ROOT),
    }

    def heartbeat(stage: str) -> None:
        payload = {
            "timestamp": timestamp(),
            "stage": stage,
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
            "execution_contract": result["execution_contract"],
            "threshold_report": result["threshold_report"],
            "microbatch_report": result["microbatch_report"],
            "python": sys.executable,
            "argv": sys.argv,
            "confirmatory_outcomes_authorized": False,
            "closed_loop_authorized": False,
        },
    )
    runtime: FrozenRuntime | None = None
    hook_handle: Any = None
    device_index: int | None = None
    peak_ram_fraction = float(result["system_memory_before"]["mem_used_fraction"])
    baseline_swap = int(result["system_memory_before"]["swap_used_bytes"])
    adapters: dict[str, ActionConsistentMissingViewAdapter] = {}
    try:
        heartbeat("validate_frozen_execution_contract")
        if sha256_file(spec_path) != contract["frozen_method_spec"]["sha256"]:
            raise RuntimeError("frozen method spec hash drift")
        if sha256_file(threshold_path) != contract["numerical_threshold_freeze"]["sha256"]:
            raise RuntimeError("threshold report hash drift")
        if sha256_file(microbatch_path) != contract["microbatch_preflight"]["sha256"]:
            raise RuntimeError("microbatch report hash drift")
        if microbatch.get("decision") != "ACTUAL_PATH_MICROBATCH_PREFLIGHT_VALID":
            raise RuntimeError("microbatch preflight is not valid")
        if int(microbatch.get("selected_microbatch")) != 8 or int(microbatch.get("gradient_accumulation")) != 1:
            raise RuntimeError("selected microbatch drift")
        if int(microbatch.get("stage0_optimizer_budget_consumed")) != 0:
            raise RuntimeError("Stage 0 optimizer budget was consumed before training")
        if thresholds.get("decision") != "STAGE0_NUMERICAL_NOISE_AND_PRACTICAL_THRESHOLDS_FROZEN":
            raise RuntimeError("numerical thresholds are not frozen")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for real Stage 0")
        if disk_report(REPO_ROOT)["available_bytes"] < 10 * 1024**3:
            raise RuntimeError("less than 10 GiB free")
        if meminfo()["mem_used_fraction"] > 0.82:
            raise RuntimeError("system RAM already exceeds frozen ceiling")
        result["risk_assessment"] = {
            "source": "tracked code and already-present X-VLA/LIBERO assets",
            "downloads": "disabled",
            "environment_install_or_mutation": False,
            "gpu_reserved_ceiling_fraction": 0.88,
            "system_ram_ceiling_fraction": 0.82,
            "research_induced_swap_growth_allowed_bytes": 0,
            "cpu_or_disk_model_offload": False,
            "confirmatory_access": False,
            "physical_robot_manipulation": False,
        }

        heartbeat("materialize_frozen_discovery_and_validation_rows")
        training_records, validation_records, manifests = materialize_stage0_records(spec, run_dir, heartbeat)
        result["data"] = {
            "training_record_count": len(training_records),
            "validation_record_count": len(validation_records),
            "training_demo_indices": "0..39",
            "validation_demo_indices": [40],
            "reader_positions": [0, 9, 18, 27],
            "split_overlap": False,
            "confirmation_demo_indices_accessed": False,
            "confirmatory_outcomes_accessed": False,
            "materialized": manifests,
        }
        peak_ram_fraction = max(peak_ram_fraction, float(meminfo()["mem_used_fraction"]))

        heartbeat("load_frozen_xvla")
        device_index = int(torch.cuda.current_device())
        device = torch.device("cuda", device_index)
        result["cuda_device"] = {
            "index": device_index,
            "name": torch.cuda.get_device_name(device_index),
            "device": str(device),
            "allocator_initialized": bool(torch.cuda.is_initialized()),
            "telemetry_repair_classification": "EXCEPTIONAL_TELEMETRY_DEVICE_REPAIR",
        }
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device_index)
        seed = int(spec["training_budget"]["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        runtime = load_frozen_xvla(spec, device)
        hook = ActionHiddenHook()
        hook_handle = runtime.model.transformer.norm.register_forward_hook(hook)
        result["xvla"] = {
            "model_id": spec["xvla"]["model_id"],
            "model_revision": spec["xvla"]["model_revision"],
            "source_revision": spec["xvla"]["source_revision"],
            "model_class": type(runtime.model).__name__,
            "processor_class": type(runtime.processor).__name__,
            "parameter_device": str(next(runtime.model.parameters()).device),
            "parameter_dtype": str(next(runtime.model.parameters()).dtype),
            "base_trainable_parameter_count": sum(
                parameter.numel() for parameter in runtime.model.parameters() if parameter.requires_grad
            ),
            "optional_shims": runtime.optional_shims,
            "compatibility_patches": runtime.compatibility_patches,
        }
        frozen_guard_before = frozen_parameter_guard(runtime.model)
        result["frozen_xvla_guard_before"] = frozen_guard_before

        training: dict[str, Any] = {}
        for arm in ARM_NAMES:
            heartbeat(f"train_{arm.lower()}_start")
            adapter, metrics = train_arm(
                arm,
                spec,
                contract,
                thresholds,
                runtime,
                hook,
                training_records,
                device,
                device_index,
                run_dir,
                heartbeat,
                baseline_swap,
            )
            adapters[arm] = adapter
            training[arm] = metrics
            peak_ram_fraction = max(peak_ram_fraction, float(metrics["peak_system_ram_used_fraction"]))
        result["training"] = training

        heartbeat("validate_frozen_xvla_unchanged")
        frozen_guard_after = frozen_parameter_guard(runtime.model)
        result["frozen_xvla_guard_after"] = frozen_guard_after
        frozen_unchanged = bool(
            frozen_guard_before["sha256"] == frozen_guard_after["sha256"]
            and frozen_guard_after["trainable_parameter_count"] == 0
            and frozen_guard_after["gradient_tensor_count"] == 0
        )
        result["frozen_xvla_unchanged"] = frozen_unchanged

        heartbeat("evaluate_fixed_validation_demo40")
        for adapter in adapters.values():
            adapter.eval()
        validation = evaluate_stage0(
            spec,
            contract,
            thresholds,
            runtime,
            hook,
            validation_records,
            adapters,
            device,
            device_index,
            heartbeat,
        )
        result["validation"] = validation
        result["forward_counts"] = {
            "hook": dict(hook.forward_counts),
            "teacher": int(hook.forward_counts["teacher_capture"]),
            "student": int(hook.forward_counts["student_adapter"]),
            "inactive_transformer": int(hook.forward_counts["inactive"]),
        }

        heartbeat("persist_tracked_checkpoints")
        result["tracked_checkpoints"] = persist_tracked_checkpoints(adapters, training, spec)
        checkpoint_valid = all(
            int(training[arm]["optimizer_steps"]) == 128
            and all(
                bool(training[arm]["checkpoints"][str(step)]["disk_reload_ok"])
                and float(training[arm]["checkpoints"][str(step)]["reload_parameter_delta_l2"]) == 0.0
                for step in (64, 128)
            )
            for arm in ARM_NAMES
        )
        execution_valid = bool(
            len(training_records) == 480
            and len(validation_records) == 12
            and all(int(training[arm]["trainable_parameter_count"]) == 434816 for arm in ARM_NAMES)
            and all(bool(training[arm]["finite_nonzero_gradients"]) for arm in ARM_NAMES)
            and all(bool(training[arm]["weights_changed"]) for arm in ARM_NAMES)
            and all(int(training[arm]["record_exposures"]) == 1024 for arm in ARM_NAMES)
            and len({training[arm]["initial_state_sha256"] for arm in ARM_NAMES}) == 1
            and checkpoint_valid
            and frozen_unchanged
            and result["forward_counts"]["teacher"] > 0
            and result["forward_counts"]["student"] > 0
            and meminfo()["swap_used_bytes"] - baseline_swap == 0
        )
        gates = {
            "execution_valid": execution_valid,
            "real_clean_teacher_forwards": bool(result["forward_counts"]["teacher"] > 0),
            "real_dropout_student_forwards": bool(result["forward_counts"]["student"] > 0),
            "cuda_execution": True,
            "trainable_parameter_count_exact": all(
                int(training[arm]["trainable_parameter_count"]) == 434816 for arm in ARM_NAMES
            ),
            "optimizer_steps_exact": all(int(training[arm]["optimizer_steps"]) == 128 for arm in ARM_NAMES),
            "finite_nonzero_gradients": all(bool(training[arm]["finite_nonzero_gradients"]) for arm in ARM_NAMES),
            "weights_changed": all(bool(training[arm]["weights_changed"]) for arm in ARM_NAMES),
            "checkpoint_write_and_exact_reload": checkpoint_valid,
            "frozen_xvla_unchanged": frozen_unchanged,
            "exact_clean_bypass": bool(
                validation["action_legality_and_smoothness"]["all_clean_bypass_exact"]
            ),
            "action_outputs_finite": bool(
                validation["action_legality_and_smoothness"]["all_policies_finite"]
            ),
            "official_action_ranges": bool(
                validation["action_legality_and_smoothness"]["official_action_range_violation_count"] == 0
            ),
            "translation_smoothness": bool(
                validation["action_legality_and_smoothness"]["full_translation_smoothness_pass"]
            ),
            "rotation_smoothness": bool(
                validation["action_legality_and_smoothness"]["full_rotation_smoothness_pass"]
            ),
            "no_privileged_deployment_inputs": True,
            "no_direct_reconstructed_input": True,
            "reconstruction_gate": bool(validation["reconstruction_gate"]["pass"]),
            "base_directional_gate": bool(validation["base_directional_gate"]["pass"]),
        }
        gates["action_legality_and_smoothness"] = bool(
            gates["exact_clean_bypass"]
            and gates["action_outputs_finite"]
            and gates["official_action_ranges"]
            and gates["translation_smoothness"]
            and gates["rotation_smoothness"]
        )
        result["gates"] = gates
        result["decision"] = adjudicate_stage0(gates, validation["comparisons"])
        result["success"] = result["decision"] in {
            "STAGE0_GO",
            "STAGE0_PROMISING_NEEDS_ONE_FIXED_CONFIRMATION",
            "STAGE0_UNDERPOWERED_ONE_FIXED_EXPANSION_ALLOWED",
            "STAGE0_MECHANISM_NOT_SUPPORTED",
            "STAGE0_GENERIC_ADAPTATION_EXPLAINS_GAIN",
            "STAGE0_ACTION_LEGALITY_FAILURE",
        }
    except Exception as exc:
        result["exceptions"].append(
            {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()}
        )
        result["decision"] = "STAGE0_IMPLEMENTATION_OR_RESOURCE_FAILURE"
        result["success"] = False
    finally:
        if hook_handle is not None:
            hook_handle.remove()
        runtime = None
        gc.collect()
        if torch.cuda.is_available():
            if device_index is None:
                device_index = int(torch.cuda.current_device())
            torch.cuda.empty_cache()
            current_allocated = int(torch.cuda.max_memory_allocated(device_index))
            current_reserved = int(torch.cuda.max_memory_reserved(device_index))
            training_values = list((result.get("training") or {}).values())
            result["peak_cuda_allocated_bytes"] = max(
                [current_allocated]
                + [int(value.get("peak_cuda_allocated_bytes", 0)) for value in training_values]
            )
            result["peak_cuda_reserved_bytes"] = max(
                [current_reserved]
                + [int(value.get("peak_cuda_reserved_bytes", 0)) for value in training_values]
            )
        result["system_memory_after"] = meminfo()
        peak_ram_fraction = max(peak_ram_fraction, float(result["system_memory_after"]["mem_used_fraction"]))
        result["peak_system_ram_used_fraction"] = peak_ram_fraction
        result["research_induced_swap_growth_bytes"] = int(
            result["system_memory_after"]["swap_used_bytes"] - baseline_swap
        )
        result["disk_after"] = disk_report(REPO_ROOT)
        result["nvidia_smi_after"] = nvidia_smi()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["finished_at"] = timestamp()
        telemetry = {
            "schema_version": result["schema_version"] + ".telemetry",
            "decision": result["decision"],
            "cuda_pid": result["cuda_pid"],
            "cuda_device": result.get("cuda_device"),
            "forward_counts": result.get("forward_counts"),
            "peak_cuda_allocated_bytes": result.get("peak_cuda_allocated_bytes"),
            "peak_cuda_reserved_bytes": result.get("peak_cuda_reserved_bytes"),
            "peak_system_ram_used_fraction": result.get("peak_system_ram_used_fraction"),
            "research_induced_swap_growth_bytes": result.get("research_induced_swap_growth_bytes"),
            "nvidia_smi_before": result.get("nvidia_smi_before"),
            "nvidia_smi_after": result.get("nvidia_smi_after"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "exceptions": result.get("exceptions"),
        }
        atomic_write_json(run_dir / "result.json", result)
        atomic_write_json(DEFAULT_RESULT_JSON, result)
        write_markdown(run_dir / "result.md", result)
        write_markdown(DEFAULT_RESULT_MD, result)
        atomic_write_json(run_dir / "telemetry.json", telemetry)
        atomic_write_json(DEFAULT_TELEMETRY, telemetry)
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
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--spec", type=pathlib.Path, default=DEFAULT_SPEC)
    parser.add_argument("--contract", type=pathlib.Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--threshold-report", type=pathlib.Path, default=DEFAULT_THRESHOLD_REPORT)
    parser.add_argument("--microbatch-report", type=pathlib.Path, default=DEFAULT_MICROBATCH_REPORT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    exit_code, _ = run_stage0(
        args.run_dir,
        spec_path=args.spec.resolve(),
        contract_path=args.contract.resolve(),
        threshold_path=args.threshold_report.resolve(),
        microbatch_path=args.microbatch_report.resolve(),
    )
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
