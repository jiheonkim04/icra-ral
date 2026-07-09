"""Bounded PatchGuard-VLA STATE 1 patch-sensitivity diagnostic.

This runner uses local LIBERO HDF5 observations and the local SmolVLA checkpoint
only. It performs clean/patched offline action decoding on CPU, records
proprioceptive signal availability, and refuses downloads, rollouts, training,
OpenVLA-OFT execution, and GPU jobs.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tca_map.smolvla.interface_adapters import ACTION_STRATEGY_GRIPPER_CLOSE, adapt_policy_action_to_env_action
from tca_map.smolvla.libero_learned_policy_rollout import (
    CAMERA_ALIAS_STRATEGY_CURRENT,
    STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3,
    _build_batch,
)
from tca_map.smolvla.load_only_smoke import (
    _external_tokenizer_files,
    _find_files,
    _nvidia_smi,
    _read_tokenizer_dependency,
    _rss_mb,
    _runtime_dependencies,
)
from tca_map.smolvla.offline_demo_action_decoding import _load_first_hdf5_sample
from tca_map.smolvla.vlm_enabled_repeated_offline_decoding import _load_policy_with_vlm


HEAVY_IMPORT_GATE = "ALLOW_HEAVY_IMPORT"
PATCHGUARD_GATE = "ALLOW_PATCHGUARD_VLA_STATE1"
MAX_TIMESTEPS = 3
MAX_RUNTIME_SECONDS = 1200
MAX_VRAM_MB = 14336
PATCH_EFFECT_L1_THRESHOLD = 0.01
PATCH_EFFECT_TRANSLATION_L2_THRESHOLD = 0.01

VARIANT_CLEAN = "clean"
VARIANT_RANDOM_PATCH = "random_patch"
VARIANT_FIXED_VISIBLE_PATCH = "fixed_visible_patch"
VARIANT_CUTOUT_DEFENSE = "fixed_patch_cutout_defense"
VARIANT_VISUAL_AUG_PROXY = "fixed_patch_visual_aug_proxy"
VARIANTS = (
    VARIANT_CLEAN,
    VARIANT_RANDOM_PATCH,
    VARIANT_FIXED_VISIBLE_PATCH,
    VARIANT_CUTOUT_DEFENSE,
    VARIANT_VISUAL_AUG_PROXY,
)

STATE1_DECISIONS = {
    "READY_FOR_PATCHGUARD_LORA_SMOKE",
    "KILL_ATTACK_NOT_REPRODUCIBLE",
    "KILL_NO_KINEMATIC_SIGNAL",
    "KILL_BASELINE_DOMINATED",
    "TOO_HEAVY_LOCAL",
    "SOURCE_BLOCKED",
}

FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_GPU_TRAINING",
    "ALLOW_TINY_TRAINING",
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
    "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    attack: str | None
    defense: str | None = None


VARIANT_SPECS: tuple[VariantSpec, ...] = (
    VariantSpec(VARIANT_CLEAN, None),
    VariantSpec(VARIANT_RANDOM_PATCH, "random"),
    VariantSpec(VARIANT_FIXED_VISIBLE_PATCH, "fixed"),
    VariantSpec(VARIANT_CUTOUT_DEFENSE, "fixed", "cutout"),
    VariantSpec(VARIANT_VISUAL_AUG_PROXY, "fixed", "visual_aug_proxy"),
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _finite_list(values: Any, limit: int | None = None) -> list[float]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    return [round(float(x), 6) for x in flat]


def _copy_obs(obs: dict[str, Any]) -> dict[str, Any]:
    copied: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            copied[key] = value.copy()
        else:
            copied[key] = np.asarray(value).copy()
    return copied


def _patch_bounds(image: np.ndarray, fraction: float = 0.25) -> tuple[int, int, int, int]:
    height, width = image.shape[:2]
    size = max(4, int(round(min(height, width) * fraction)))
    y0 = max(0, height - size - max(1, height // 12))
    x0 = max(0, width - size - max(1, width // 12))
    return y0, x0, min(height, y0 + size), min(width, x0 + size)


def _fixed_pattern(height: int, width: int) -> np.ndarray:
    yy, xx = np.indices((height, width))
    checker = ((yy // 4 + xx // 4) % 2).astype(np.uint8)
    pattern = np.zeros((height, width, 3), dtype=np.uint8)
    pattern[checker == 0] = [255, 0, 255]
    pattern[checker == 1] = [0, 255, 0]
    return pattern


def _blocky_aug(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float32)
    h_even = arr.shape[0] - (arr.shape[0] % 2)
    w_even = arr.shape[1] - (arr.shape[1] % 2)
    core = arr[:h_even, :w_even]
    pooled = core.reshape(h_even // 2, 2, w_even // 2, 2, arr.shape[2]).mean(axis=(1, 3))
    up = np.repeat(np.repeat(pooled, 2, axis=0), 2, axis=1)
    out = arr.copy()
    out[:h_even, :w_even] = up
    out = np.clip(out * 0.95 + 4.0, 0, 255)
    return out.astype(image.dtype)


def apply_patch_variant(
    obs: dict[str, Any],
    variant: str,
    *,
    seed: int = 0,
    patch_fraction: float = 0.25,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return an observation copy with the selected bounded patch variant."""
    if variant not in VARIANTS:
        raise ValueError(f"unknown PatchGuard variant: {variant}")

    patched = _copy_obs(obs)
    metadata: dict[str, Any] = {
        "variant": variant,
        "agentview_modified": False,
        "eye_in_hand_modified": False,
        "state_modified": False,
        "patch_fraction": patch_fraction,
        "patch_bounds": None,
        "attack": None,
        "defense": None,
    }
    if variant == VARIANT_CLEAN:
        return patched, metadata

    image = np.asarray(patched["agentview_image"])
    y0, x0, y1, x1 = _patch_bounds(image, patch_fraction)
    metadata["patch_bounds"] = [int(y0), int(x0), int(y1), int(x1)]
    patch_h, patch_w = y1 - y0, x1 - x0
    rng = np.random.default_rng(seed)

    if variant == VARIANT_RANDOM_PATCH:
        patch = rng.integers(0, 256, size=(patch_h, patch_w, image.shape[2]), dtype=np.uint8)
        metadata["attack"] = "random_visible_square"
    else:
        patch = _fixed_pattern(patch_h, patch_w)
        metadata["attack"] = "fixed_magenta_green_checkerboard"

    for key in ("agentview_image", "agentview_rgb"):
        arr = np.asarray(patched[key]).copy()
        arr[y0:y1, x0:x1, :] = patch.astype(arr.dtype)
        patched[key] = arr
    metadata["agentview_modified"] = True

    if variant == VARIANT_CUTOUT_DEFENSE:
        for key in ("agentview_image", "agentview_rgb"):
            original = np.asarray(obs[key])
            arr = np.asarray(patched[key]).copy()
            fill = np.asarray(original, dtype=np.float32).mean(axis=(0, 1), keepdims=True)
            arr[y0:y1, x0:x1, :] = np.clip(fill, 0, 255).astype(arr.dtype)
            patched[key] = arr
        metadata["defense"] = "mean_color_cutout_over_patch_region"
    elif variant == VARIANT_VISUAL_AUG_PROXY:
        for key in ("agentview_image", "agentview_rgb"):
            patched[key] = _blocky_aug(np.asarray(patched[key]))
        metadata["defense"] = "cheap_blocky_smoothing_and_brightness_proxy"

    return patched, metadata


PolicyLoader = Callable[[Path, Path, dict[str, Any], str], Any]


def _decode_variant(
    *,
    policy: Any,
    config: Any,
    tokenizer_root: Path,
    hdf5_path: Path,
    demo_name: str | None,
    timestep: int,
    task_text: str,
    device: str,
    variant: str,
    seed: int,
) -> dict[str, Any]:
    import torch

    sample = _load_first_hdf5_sample(hdf5_path, demo_name, timestep)
    expert_action = np.asarray(sample["expert_action"], dtype=np.float32).reshape(-1)
    obs, patch_metadata = apply_patch_variant(sample["obs"], variant, seed=seed)
    batch, batch_metadata = _build_batch(
        config,
        tokenizer_root,
        obs,
        task_text,
        device,
        CAMERA_ALIAS_STRATEGY_CURRENT,
        STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3,
    )
    policy.reset()
    noise = torch.zeros((1, config.chunk_size, config.max_action_dim), dtype=torch.float32, device=device)
    inference_started = time.monotonic()
    with torch.inference_mode():
        policy_action = policy.select_action(batch, noise=noise)
    inference_elapsed = time.monotonic() - inference_started

    action_adapter = adapt_policy_action_to_env_action(
        policy_action,
        int(expert_action.shape[0]),
        strategy=ACTION_STRATEGY_GRIPPER_CLOSE,
        action_scale=1.0,
    )
    adapted = np.asarray(action_adapter.values, dtype=np.float32)
    policy_np = policy_action.detach().cpu().numpy().reshape(-1).astype(np.float32)
    proprio = np.asarray(sample["obs"]["robot0_eef_pos"], dtype=np.float32).reshape(-1)
    quat = np.asarray(sample["obs"]["robot0_eef_quat"], dtype=np.float32).reshape(-1)
    prefix = min(6, expert_action.shape[0], policy_np.shape[0])

    return {
        "variant": variant,
        "demo_name": sample["metadata"]["demo_name"],
        "timestep": int(timestep),
        "task": task_text,
        "inference_elapsed_sec": round(inference_elapsed, 6),
        "action_l1_to_expert": round(float(np.mean(np.abs(adapted - expert_action))), 6),
        "action_mse_to_expert": round(float(np.mean((adapted - expert_action) ** 2)), 6),
        "policy6_l1_to_expert_first6": round(float(np.mean(np.abs(policy_np[:prefix] - expert_action[:prefix]))), 6),
        "action_finite": bool(np.isfinite(adapted).all() and np.isfinite(policy_np).all()),
        "policy_action_shape": list(policy_np.shape),
        "expert_action_shape": list(expert_action.shape),
        "policy_action_preview": _finite_list(policy_np, 6),
        "adapted_action_preview": _finite_list(adapted, int(expert_action.shape[0])),
        "expert_action_preview": _finite_list(expert_action, int(expert_action.shape[0])),
        "translation_action_preview": _finite_list(adapted[:3], 3),
        "eef_pos_preview": _finite_list(proprio, 3),
        "eef_quat_proxy_preview": _finite_list(quat, 3),
        "patch_metadata": patch_metadata,
        "action_adapter_metadata": action_adapter.metadata,
        "batch_metadata": batch_metadata,
        "sample_metadata": sample["metadata"],
    }


def _default_task_text() -> str:
    return "perform the task"


def _task_from_hdf5_name(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    parts = stem.split("_")
    while parts and (parts[0].isupper() or any(char.isdigit() for char in parts[0])):
        parts = parts[1:]
    return " ".join(parts) if parts else _default_task_text()


def _task_from_previous_report(plan: dict[str, Any]) -> str | None:
    previous_path = (plan.get("inputs") or {}).get("previous_repeated_report")
    if not previous_path:
        return None
    try:
        previous = _read_json(Path(previous_path))
    except Exception:
        return None
    candidates = [
        (((previous.get("plan") or {}).get("planned_sample") or {}).get("selected_task_text")),
        (((previous.get("plan") or {}).get("planned_sample") or {}).get("selected_language")),
    ]
    samples = previous.get("samples") or []
    if samples:
        candidates.append((samples[0] or {}).get("task"))
    for value in candidates:
        if value and value != _default_task_text():
            return str(value)
    return None


def _plan_inputs(plan: dict[str, Any], args: argparse.Namespace) -> tuple[Path, list[int], str, str | None]:
    hdf5_path = Path((plan.get("inputs") or {}).get("hdf5_path") or args.hdf5_path)
    timesteps = list((plan.get("inputs") or {}).get("selected_timesteps") or [])
    if not timesteps:
        timesteps = list((((plan.get("planned_sample") or {}).get("hdf5") or {}).get("selected_timesteps") or []))
    cleaned: list[int] = []
    for value in timesteps:
        item = int(value)
        if item >= 0 and item not in cleaned:
            cleaned.append(item)
    task_text = (
        ((plan.get("planned_sample") or {}).get("selected_task_text"))
        or ((plan.get("planned_sample") or {}).get("selected_language"))
        or _task_from_previous_report(plan)
        or (args.task if args.task != _default_task_text() else _task_from_hdf5_name(hdf5_path))
    )
    demo_name = (((plan.get("planned_sample") or {}).get("hdf5") or {}).get("demo_name"))
    return hdf5_path, cleaned[:MAX_TIMESTEPS], task_text, demo_name


def _variant_metrics(samples: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {variant: [] for variant in VARIANTS}
    clean_by_step: dict[int, dict[str, Any]] = {}
    for item in samples:
        grouped[item["variant"]].append(item)
        if item["variant"] == VARIANT_CLEAN:
            clean_by_step[item["timestep"]] = item

    metrics: dict[str, Any] = {}
    for variant, items in grouped.items():
        if not items:
            continue
        action_l1 = [float(item["action_l1_to_expert"]) for item in items]
        action_mse = [float(item["action_mse_to_expert"]) for item in items]
        finite = all(bool(item["action_finite"]) for item in items)
        vs_clean_l1: list[float] = []
        vs_clean_mse: list[float] = []
        translation_l2: list[float] = []
        state_delta_l2: list[float] = []
        for item in items:
            clean = clean_by_step.get(int(item["timestep"]))
            if clean is None:
                continue
            policy = np.asarray(item["policy_action_preview"], dtype=np.float32)
            clean_policy = np.asarray(clean["policy_action_preview"], dtype=np.float32)
            adapted = np.asarray(item["adapted_action_preview"], dtype=np.float32)
            clean_adapted = np.asarray(clean["adapted_action_preview"], dtype=np.float32)
            eef = np.asarray(item["eef_pos_preview"], dtype=np.float32)
            clean_eef = np.asarray(clean["eef_pos_preview"], dtype=np.float32)
            vs_clean_l1.append(float(np.mean(np.abs(policy - clean_policy))))
            vs_clean_mse.append(float(np.mean((policy - clean_policy) ** 2)))
            translation_l2.append(float(np.linalg.norm(adapted[:3] - clean_adapted[:3])))
            state_delta_l2.append(float(np.linalg.norm(eef - clean_eef)))
        metrics[variant] = {
            "sample_count": len(items),
            "mean_action_l1_to_expert": round(float(np.mean(action_l1)), 6),
            "mean_action_mse_to_expert": round(float(np.mean(action_mse)), 6),
            "all_actions_finite": finite,
            "mean_policy6_l1_vs_clean": round(float(np.mean(vs_clean_l1)), 6) if vs_clean_l1 else 0.0,
            "max_policy6_l1_vs_clean": round(float(np.max(vs_clean_l1)), 6) if vs_clean_l1 else 0.0,
            "mean_policy6_mse_vs_clean": round(float(np.mean(vs_clean_mse)), 6) if vs_clean_mse else 0.0,
            "mean_translation_l2_vs_clean": round(float(np.mean(translation_l2)), 6) if translation_l2 else 0.0,
            "max_translation_l2_vs_clean": round(float(np.max(translation_l2)), 6) if translation_l2 else 0.0,
            "mean_proprio_state_delta_l2_vs_clean": round(float(np.mean(state_delta_l2)), 6) if state_delta_l2 else 0.0,
            "mean_expert_alignment_delta_vs_clean": 0.0,
        }
    clean_l1 = (metrics.get(VARIANT_CLEAN) or {}).get("mean_action_l1_to_expert")
    if clean_l1 is not None:
        for variant, payload in metrics.items():
            payload["mean_expert_alignment_delta_vs_clean"] = round(
                float(payload["mean_action_l1_to_expert"] - clean_l1), 6
            )
    return metrics


def _state1_decision(
    *,
    patch_effect_nontrivial: bool,
    kinematic_signal_available: bool,
    baseline_dominated: bool,
    real_vla_used: bool,
    local_adapter_path_feasible_now: bool,
) -> str:
    if not real_vla_used:
        return "SOURCE_BLOCKED"
    if not kinematic_signal_available:
        return "KILL_NO_KINEMATIC_SIGNAL"
    if not patch_effect_nontrivial:
        return "KILL_ATTACK_NOT_REPRODUCIBLE"
    if baseline_dominated:
        return "KILL_BASELINE_DOMINATED"
    if not local_adapter_path_feasible_now:
        return "TOO_HEAVY_LOCAL"
    return "READY_FOR_PATCHGUARD_LORA_SMOKE"


def _summarize_state1(
    variant_metrics: dict[str, Any],
    *,
    kinematic_signal_available: bool,
    local_adapter_path_feasible_now: bool,
    real_vla_used: bool,
) -> dict[str, Any]:
    attacked_variants = [VARIANT_RANDOM_PATCH, VARIANT_FIXED_VISIBLE_PATCH]
    attacked = [variant_metrics.get(name, {}) for name in attacked_variants]
    max_policy_l1 = max([float(item.get("max_policy6_l1_vs_clean") or 0.0) for item in attacked] or [0.0])
    max_translation_l2 = max([float(item.get("max_translation_l2_vs_clean") or 0.0) for item in attacked] or [0.0])
    fixed_l1 = float((variant_metrics.get(VARIANT_FIXED_VISIBLE_PATCH) or {}).get("mean_policy6_l1_vs_clean") or 0.0)
    cutout_l1 = float((variant_metrics.get(VARIANT_CUTOUT_DEFENSE) or {}).get("mean_policy6_l1_vs_clean") or 0.0)
    patch_effect_nontrivial = bool(
        max_policy_l1 >= PATCH_EFFECT_L1_THRESHOLD or max_translation_l2 >= PATCH_EFFECT_TRANSLATION_L2_THRESHOLD
    )
    baseline_dominated = bool(
        patch_effect_nontrivial
        and fixed_l1 >= PATCH_EFFECT_L1_THRESHOLD
        and cutout_l1 <= max(PATCH_EFFECT_L1_THRESHOLD * 0.5, fixed_l1 * 0.25)
    )
    decision = _state1_decision(
        patch_effect_nontrivial=patch_effect_nontrivial,
        kinematic_signal_available=kinematic_signal_available,
        baseline_dominated=baseline_dominated,
        real_vla_used=real_vla_used,
        local_adapter_path_feasible_now=local_adapter_path_feasible_now,
    )
    return {
        "decision": decision,
        "patch_effect_nontrivial": patch_effect_nontrivial,
        "max_attacked_policy6_l1_vs_clean": round(max_policy_l1, 6),
        "max_attacked_translation_l2_vs_clean": round(max_translation_l2, 6),
        "cutout_defense_dominated_fixed_patch": baseline_dominated,
        "fixed_patch_mean_policy6_l1_vs_clean": round(fixed_l1, 6),
        "cutout_mean_policy6_l1_vs_clean": round(cutout_l1, 6),
        "kinematic_signal_available": kinematic_signal_available,
        "real_vla_used": real_vla_used,
        "local_adapter_path_feasible_now": local_adapter_path_feasible_now,
    }


def _module_available(name: str) -> bool:
    import importlib.util

    return importlib.util.find_spec(name) is not None


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    decision = report.get("state1_summary", {})
    metrics = report.get("variant_metrics", {})
    policy = report.get("policy", {})
    lines = [
        "# PatchGuard-VLA STATE 1 Result",
        "",
        "Bounded offline VLA patch-sensitivity diagnostic only. This is not a benchmark, rollout, training result, or paper claim.",
        "",
        f"- STATE 1 decision: `{decision.get('decision')}`",
        f"- real VLA model used: `{decision.get('real_vla_used')}`",
        f"- patch effect measured: `{decision.get('patch_effect_nontrivial')}`",
        f"- max attacked policy L1 vs clean: `{decision.get('max_attacked_policy6_l1_vs_clean')}`",
        f"- max attacked translation L2 vs clean: `{decision.get('max_attacked_translation_l2_vs_clean')}`",
        f"- kinematic signal available: `{decision.get('kinematic_signal_available')}`",
        f"- cutout baseline dominated fixed patch: `{decision.get('cutout_defense_dominated_fixed_patch')}`",
        f"- local adapter path feasible now: `{decision.get('local_adapter_path_feasible_now')}`",
        f"- training happened: `{policy.get('training_performed')}`",
        f"- rollouts happened: `{policy.get('rollouts_performed')}`",
        f"- downloads happened: `{policy.get('downloads_performed')}`",
        f"- GPU jobs happened: `{policy.get('gpu_jobs_performed')}`",
        "",
        "## Variant Metrics",
        "",
        "| variant | mean action L1 to expert | mean policy L1 vs clean | max policy L1 vs clean | mean translation L2 vs clean | expert alignment delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        payload = metrics.get(variant) or {}
        lines.append(
            "| {variant} | {action_l1} | {mean_l1} | {max_l1} | {translation} | {align} |".format(
                variant=variant,
                action_l1=payload.get("mean_action_l1_to_expert"),
                mean_l1=payload.get("mean_policy6_l1_vs_clean"),
                max_l1=payload.get("max_policy6_l1_vs_clean"),
                translation=payload.get("mean_translation_l2_vs_clean"),
                align=payload.get("mean_expert_alignment_delta_vs_clean"),
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            str(report.get("recommended_next_step") or ""),
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace, loader: PolicyLoader = _load_policy_with_vlm) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.device == "cpu":
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

    plan_path = Path(args.plan_report)
    smolvla_ckpt = Path(os.environ.get("SMOLVLA_CKPT") or args.smolvla_ckpt)
    checkpoint_root = Path(os.environ.get("CHECKPOINT_ROOT") or args.checkpoint_root)
    hf_home = Path(os.environ.get("HF_HOME") or args.hf_home)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    deps = _runtime_dependencies()
    dependency_name = _read_tokenizer_dependency(smolvla_ckpt)
    external_dependency = _external_tokenizer_files(dependency_name, [hf_home, checkpoint_root])
    peft_available = _module_available("peft")
    bitsandbytes_available = _module_available("bitsandbytes")

    report: dict[str, Any] = {
        "schema_version": "patchguard-vla-state1-diagnostic-v1",
        "evidence_label": "patchguard_vla_state1_diagnostic",
        "patchguard_vla_state1_diagnostic_passed": False,
        "decision": "stop",
        "ready_for_paper_claim": False,
        "ready_for_benchmark_claim": False,
        "ready_for_rollout_scaling": False,
        "policy": {
            "bounded_patchguard_vla_state1": True,
            "task_local_gates_required": [f"{HEAVY_IMPORT_GATE}=1", f"{PATCHGUARD_GATE}=1"],
            "downloads_performed": False,
            "installs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "simulator_environment_created": False,
            "rollouts_performed": False,
            "benchmark_rollouts_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
            "heavy_import_gate_set": _env_flag(HEAVY_IMPORT_GATE),
            "patchguard_gate_set": _env_flag(PATCHGUARD_GATE),
            "forbidden_gates_set": forbidden,
        },
        "claims": {
            "standard_success_claimed": False,
            "benchmark_success_claimed": False,
            "counterfactual_robustness_claimed": False,
            "sota_claimed": False,
            "paper_grade_claim_made": False,
        },
        "risk_limits": {
            "max_timesteps": MAX_TIMESTEPS,
            "max_policy_calls": MAX_TIMESTEPS * len(VARIANTS),
            "max_runtime_seconds": MAX_RUNTIME_SECONDS,
            "max_vram_mb": MAX_VRAM_MB,
            "device": args.device,
            "simulator_allowed": False,
            "rollout_allowed": False,
            "training_allowed": False,
        },
        "paths": {
            "plan_report": str(plan_path),
            "smolvla_ckpt": str(smolvla_ckpt),
            "checkpoint_root": str(checkpoint_root),
            "hf_home": str(hf_home),
        },
        "files": {
            "config_found": _find_files(smolvla_ckpt, ["config.json"]),
            "weights_found": _find_files(smolvla_ckpt, ["model.safetensors"], ["*.safetensors"]),
            "external_tokenizer_dependency": external_dependency,
        },
        "runtime_dependencies": deps,
        "module_availability": {
            "peft": peft_available,
            "bitsandbytes": bitsandbytes_available,
            "torch": _module_available("torch"),
            "lerobot": _module_available("lerobot"),
            "transformers": _module_available("transformers"),
            "h5py": _module_available("h5py"),
        },
        "gpu": _nvidia_smi(),
        "runtime": {"rss_before_mb": _rss_mb(), "rss_after_mb": None, "elapsed_sec": None},
        "samples": [],
        "variant_metrics": {},
        "signal_inventory": {},
        "state1_summary": {},
        "error": None,
        "recommended_next_step": None,
    }

    def block(reason: str, code: int, decision: str = "stop") -> tuple[dict[str, Any], int]:
        report["decision"] = decision
        report["recommended_next_step"] = reason
        report["error"] = {"message": reason}
        report["runtime"]["rss_after_mb"] = _rss_mb()
        report["runtime"]["elapsed_sec"] = round(time.monotonic() - started, 3)
        if decision in STATE1_DECISIONS:
            report["state1_summary"] = {"decision": decision}
        return report, code

    if not report["policy"]["heavy_import_gate_set"]:
        return block(f"{HEAVY_IMPORT_GATE}=1 is required only inside this bounded PatchGuard-VLA diagnostic.", 2)
    if not report["policy"]["patchguard_gate_set"]:
        return block(f"{PATCHGUARD_GATE}=1 is required only inside this bounded PatchGuard-VLA diagnostic.", 3)
    if forbidden:
        return block("Forbidden gate(s) set: " + ", ".join(forbidden), 4)
    if args.device != "cpu":
        return block("The PatchGuard-VLA STATE 1 diagnostic is CPU-only.", 5)
    if not plan_path.exists():
        return block(f"VLM-enabled repeated offline plan report is missing: {plan_path}", 6, "SOURCE_BLOCKED")
    if not all(deps.values()):
        missing = [name for name, present in deps.items() if not present]
        return block("Missing runtime dependencies: " + ", ".join(missing), 7, "SOURCE_BLOCKED")
    if not external_dependency.get("found"):
        return block("External tokenizer/VLM dependency root is missing.", 8, "SOURCE_BLOCKED")

    try:
        plan = _read_json(plan_path)
        hdf5_path, timesteps, task_text, demo_name = _plan_inputs(plan, args)
        if not hdf5_path.exists():
            return block(f"Selected HDF5 file is missing: {hdf5_path}", 9, "SOURCE_BLOCKED")
        if not timesteps:
            return block("Plan did not provide selected HDF5 timesteps.", 10, "SOURCE_BLOCKED")

        report["paths"]["hdf5_path"] = str(hdf5_path)
        report["inputs"] = {
            "timesteps": timesteps,
            "task_text": task_text,
            "demo_name": demo_name,
            "variant_count": len(VARIANTS),
            "policy_call_count": len(timesteps) * len(VARIANTS),
        }

        report["policy"]["heavy_model_imports_performed"] = True
        policy, config = loader(smolvla_ckpt, hf_home, external_dependency, args.device)
        report["policy"]["model_load_performed"] = True
        tokenizer_root = Path(external_dependency["root"])

        samples: list[dict[str, Any]] = []
        for timestep in timesteps:
            for spec in VARIANT_SPECS:
                samples.append(
                    _decode_variant(
                        policy=policy,
                        config=config,
                        tokenizer_root=tokenizer_root,
                        hdf5_path=hdf5_path,
                        demo_name=demo_name,
                        timestep=timestep,
                        task_text=task_text,
                        device=args.device,
                        variant=spec.name,
                        seed=args.seed + int(timestep),
                    )
                )
        report["policy"]["model_inference_performed"] = True
        report["samples"] = samples

        import torch

        cuda_max = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3) if torch.cuda.is_available() else 0.0
        del policy
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        first_sample = samples[0] if samples else {}
        first_metadata = first_sample.get("sample_metadata") or {}
        first_batch = first_sample.get("batch_metadata") or {}
        state_adapter = first_batch.get("state_adapter") or {}
        kinematic_signal_available = bool(
            first_metadata.get("ee_states_shape")
            and state_adapter
            and state_adapter.get("uses_privileged_state") is False
            and state_adapter.get("silent_truncation_performed") is False
        )
        local_adapter_path_feasible_now = bool(peft_available and (args.require_bitsandbytes_for_lora is False or bitsandbytes_available))
        real_vla_used = bool(report["policy"]["model_load_performed"] and report["policy"]["model_inference_performed"])
        variant_metrics = _variant_metrics(samples)
        summary = _summarize_state1(
            variant_metrics,
            kinematic_signal_available=kinematic_signal_available,
            local_adapter_path_feasible_now=local_adapter_path_feasible_now,
            real_vla_used=real_vla_used,
        )

        report["variant_metrics"] = variant_metrics
        report["signal_inventory"] = {
            "local_smolvla_runtime_available": real_vla_used,
            "local_libero_visual_observations_available": bool(first_metadata.get("agentview_shape") and first_metadata.get("eye_in_hand_shape")),
            "eef_or_joint_or_proprio_available": kinematic_signal_available,
            "state_adapter": state_adapter,
            "simulator_segmentation_or_mask_available_in_this_diagnostic": False,
            "approximate_arm_mask_available_in_this_diagnostic": False,
            "patched_observations_generated_without_downloads": True,
            "clean_vs_patched_action_divergence_measurable": bool(summary["max_attacked_policy6_l1_vs_clean"] is not None),
            "lora_adapter_training_path_feasible_later": bool(report["module_availability"]["torch"] and report["module_availability"]["lerobot"]),
            "lora_adapter_training_path_feasible_now_without_installs": local_adapter_path_feasible_now,
            "peft_missing_blocks_standard_local_lora_now": not peft_available,
            "bitsandbytes_missing_blocks_qlora_now": not bitsandbytes_available,
            "non_leaking_patchguard_signal": (
                "Compare image-induced action divergence against unchanged observation.state/EEF proxy; "
                "no success labels, BDDL labels, or simulator privileged state are used."
            ),
        }
        report["state1_summary"] = summary
        report["runtime"]["cuda_max_allocated_mb"] = cuda_max

        elapsed = time.monotonic() - started
        if any(not bool(item["action_finite"]) for item in samples):
            return block("At least one decoded action contained non-finite values.", 11, "SOURCE_BLOCKED")
        if elapsed > MAX_RUNTIME_SECONDS:
            return block("PatchGuard-VLA STATE 1 diagnostic exceeded the runtime budget.", 12, "TOO_HEAVY_LOCAL")
        if cuda_max > MAX_VRAM_MB:
            return block("PatchGuard-VLA STATE 1 diagnostic exceeded the 14GB VRAM budget.", 13, "TOO_HEAVY_LOCAL")
        if cuda_max > 0:
            return block("CPU-only PatchGuard-VLA diagnostic unexpectedly allocated CUDA memory.", 14, "TOO_HEAVY_LOCAL")

        report["patchguard_vla_state1_diagnostic_passed"] = True
        report["decision"] = summary["decision"]
        report["runtime"]["rss_after_mb"] = _rss_mb()
        report["runtime"]["elapsed_sec"] = round(elapsed, 3)
        if summary["decision"] == "READY_FOR_PATCHGUARD_LORA_SMOKE":
            report["recommended_next_step"] = (
                "Run one separately approved PatchGuard adapter smoke with real SmolVLA/LeRobot adapter wiring, "
                "comparing fixed/random patch, cutout, and generic visual augmentation baselines."
            )
        elif summary["decision"] == "TOO_HEAVY_LOCAL":
            report["recommended_next_step"] = (
                "Do not train in this run. Resolve real local LoRA/adapter tooling without unapproved installs "
                "or move the adapter smoke to approved WSL/Linux/cloud before STATE 2."
            )
        elif summary["decision"] == "KILL_BASELINE_DOMINATED":
            report["recommended_next_step"] = (
                "Do not train PatchGuard-VLA; the cheap cutout/random-erasing proxy already removes the measured fixed-patch effect."
            )
        elif summary["decision"] == "KILL_ATTACK_NOT_REPRODUCIBLE":
            report["recommended_next_step"] = (
                "Do not train PatchGuard-VLA; first find a stronger local physical-patch failure using a bounded, non-training attack setup."
            )
        else:
            report["recommended_next_step"] = "Do not continue until the blocking STATE 1 signal is resolved."
        return report, 0
    except Exception as exc:  # noqa: BLE001
        report["error"] = _compact_error(exc)
        report["runtime"]["rss_after_mb"] = _rss_mb()
        report["runtime"]["elapsed_sec"] = round(time.monotonic() - started, 3)
        report["state1_summary"] = {"decision": "SOURCE_BLOCKED"}
        report["decision"] = "SOURCE_BLOCKED"
        report["recommended_next_step"] = "Fix the PatchGuard-VLA STATE 1 diagnostic blocker before any training or paper claim."
        return report, 15


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-report", default="reports/vlm_enabled_repeated_offline_decoding_plan_report.json")
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--hdf5-path", default="")
    parser.add_argument("--task", default="perform the task")
    parser.add_argument("--report-path", default="reports/patchguard_vla_state1_result.json")
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--require-bitsandbytes-for-lora", action="store_true")
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report_path.suffix == ".json":
        _write_markdown_report(report, report_path.with_suffix(".md"))
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
