"""Minimal rollout-first diagnostic for CSS-Shield.

The diagnostic runs one small LIBERO/RoboSuite exact-init rollout task and
compares simple runtime shield variants. It is intentionally bounded and
interpretable: no training, no downloads, no OpenVLA-OFT, and no paper-grade
claim.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.datasets.libero_fixed_prior_rollout_diagnostic import _action_stats
from tca_map.datasets.libero_zero_reward_rollout_diagnosis import _best_object_key, _distance, _extract_eef, _extract_pos
from tca_map.smolvla.interface_adapters import ACTION_STRATEGY_GRIPPER_ZERO_HOLD, adapt_policy_action_to_env_action
from tca_map.smolvla.libero_learned_policy_rollout import (
    CAMERA_ALIAS_STRATEGY_CURRENT,
    STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3,
    _build_batch,
    _ensure_paths,
)
from tca_map.smolvla.load_only_smoke import _external_tokenizer_files, _find_files, _read_tokenizer_dependency, _runtime_dependencies
from tca_map.smolvla.online_action_generation_bridge import _as_path, _bddl_path, _load_first_case, _safe_l2, _target_metrics
from tca_map.smolvla.single_sample_interface_smoke import _load_policy

SCHEMA_VERSION = "2026-07-06.css_shield_minimal_rollout_diagnostic.v1"
TASK_GATE = "ALLOW_CSS_SHIELD_ROLLOUT"
MAX_STEPS = 25
SHIELD_VARIANTS = ("no_shield", "clipping_only", "safety_only", "semantic_target_only", "full_css_shield")
FORBIDDEN_GATES = (
    "ALLOW_DOWNLOADS",
    "ALLOW_GPU_TRAINING",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_BENCHMARK_ROLLOUT",
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {"type": type(exc).__name__, "message": str(exc), "traceback_tail": traceback.format_exc().splitlines()[-12:]}


def _round(value: float | int | np.floating | None, digits: int = 9) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _norm(values: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64).reshape(-1)))


def _unit(vec: np.ndarray | None) -> np.ndarray | None:
    if vec is None:
        return None
    norm = _norm(vec)
    if norm < 1e-9:
        return None
    return np.asarray(vec, dtype=np.float64) / norm


def _safe_rate(count: int, total: int) -> float:
    return round(float(count) / float(total), 9) if total else 0.0


def _path_text(path: str | Path) -> str:
    return str(path).replace("\\", "/").rstrip("/")


def _prepare_libero_import_path(libero_root: Path, robosuite_root: Path) -> dict[str, Any]:
    """Prefer the LIBERO repo root over stale inner-package path injections.

    Some local WSL environments have a .pth entry for LIBERO/libero. That makes
    Python import the inner package as top-level `libero`, while the simulator
    code expects the repository-root namespace `libero.libero`. Remove only that
    conflicting path at process runtime; do not mutate the venv.
    """

    inner_libero = libero_root / "libero"
    before = [_path_text(item) for item in sys.path]
    sys.path[:] = [item for item in sys.path if _path_text(item) != _path_text(inner_libero)]
    for candidate in (robosuite_root, libero_root):
        text = str(candidate)
        if text and _path_text(text) not in {_path_text(item) for item in sys.path}:
            sys.path.insert(0, text)
    for name in list(sys.modules):
        if name == "libero" or name.startswith("libero."):
            del sys.modules[name]
    after = [_path_text(item) for item in sys.path]
    return {
        "removed_libero_inner_path": _path_text(inner_libero) in before and _path_text(inner_libero) not in after,
        "libero_root": str(libero_root),
        "robosuite_root": str(robosuite_root),
        "sys_path_prefix": after[:6],
    }


def _scene_proxy(obs: dict[str, Any], instruction: str, counterfactual_instruction: str | None) -> dict[str, Any]:
    eef = _extract_eef(obs)
    target_audit = _best_object_key(obs, instruction)
    wrong_audit = _best_object_key(obs, counterfactual_instruction) if counterfactual_instruction else {"best_key": None}
    target_key = target_audit.get("best_key")
    wrong_key = wrong_audit.get("best_key")
    target_pos = _extract_pos(obs, target_key) if target_key else None
    wrong_pos = _extract_pos(obs, wrong_key) if wrong_key else None
    return {
        "eef": eef,
        "target_audit": target_audit,
        "wrong_audit": wrong_audit,
        "target_key": target_key,
        "wrong_key": wrong_key,
        "target_pos": target_pos,
        "wrong_pos": wrong_pos,
    }


def assess_action(
    action: list[float] | np.ndarray,
    obs: dict[str, Any],
    instruction: str,
    counterfactual_instruction: str | None,
    *,
    max_translation_norm: float = 0.20,
    wrong_margin: float = 0.01,
    near_object_margin: float = 0.04,
) -> dict[str, Any]:
    arr = np.asarray(action, dtype=np.float64).reshape(-1)
    if arr.size != 7:
        raise ValueError(f"CSS-Shield expects 7D action, got {arr.size}D")
    translation = arr[:3]
    proxy = _scene_proxy(obs, instruction, counterfactual_instruction)
    eef = proxy["eef"]
    target_pos = proxy["target_pos"]
    wrong_pos = proxy["wrong_pos"]
    target_dir = _unit(np.asarray(target_pos) - np.asarray(eef)) if eef is not None and target_pos is not None else None
    wrong_dir = _unit(np.asarray(wrong_pos) - np.asarray(eef)) if eef is not None and wrong_pos is not None else None
    target_projection = float(np.dot(translation, target_dir)) if target_dir is not None else None
    wrong_projection = float(np.dot(translation, wrong_dir)) if wrong_dir is not None else None
    wrong_target = False
    if target_projection is not None and wrong_projection is not None:
        wrong_target = bool(wrong_projection > target_projection + wrong_margin and wrong_projection > 0.0)
    translation_norm = _norm(translation)
    excessive_translation = bool(translation_norm > max_translation_norm)
    unsafe_down = bool(translation[2] < -max_translation_norm)
    near_collision = False
    distance_to_wrong = None
    if eef is not None and wrong_pos is not None:
        distance_to_wrong = _distance(eef, wrong_pos)
        near_collision = bool(distance_to_wrong is not None and distance_to_wrong < near_object_margin and wrong_projection is not None and wrong_projection > 0.0)
    unsafe = bool(excessive_translation or unsafe_down or near_collision)
    return {
        "wrong_target_action": wrong_target,
        "unsafe_action": unsafe,
        "semantic_proxy_available": bool(target_projection is not None and wrong_projection is not None),
        "safety_proxy_available": True,
        "target_projection": _round(target_projection),
        "wrong_target_projection": _round(wrong_projection),
        "translation_norm": _round(translation_norm),
        "excessive_translation": excessive_translation,
        "unsafe_downward_translation": unsafe_down,
        "near_collision_proxy": near_collision,
        "distance_to_wrong_target": _round(distance_to_wrong),
        "target_key_audit": proxy["target_audit"],
        "wrong_target_key_audit": proxy["wrong_audit"],
        "object_position_keys": {"target_key": proxy["target_key"], "wrong_key": proxy["wrong_key"]},
    }


def _limit_translation(action: np.ndarray, max_translation_norm: float) -> np.ndarray:
    out = action.copy()
    norm = _norm(out[:3])
    if norm > max_translation_norm and norm > 1e-9:
        out[:3] = out[:3] * (max_translation_norm / norm)
    return out


def apply_shield(
    action: list[float] | np.ndarray,
    obs: dict[str, Any],
    instruction: str,
    counterfactual_instruction: str | None,
    variant: str,
    *,
    max_translation_norm: float = 0.20,
) -> tuple[np.ndarray, dict[str, Any]]:
    raw = np.asarray(action, dtype=np.float64).reshape(-1)
    if raw.size != 7:
        raise ValueError(f"CSS-Shield expects 7D action, got {raw.size}D")
    clipped = np.clip(raw, -1.0, 1.0)
    before = assess_action(clipped, obs, instruction, counterfactual_instruction, max_translation_norm=max_translation_norm)
    out = clipped.copy()
    intervention = "accept"
    if variant == "no_shield":
        pass
    elif variant == "clipping_only":
        intervention = "clip" if not np.allclose(raw, clipped) else "accept"
    elif variant == "safety_only":
        if before["unsafe_action"]:
            out = _limit_translation(out, max_translation_norm)
            intervention = "damp"
    elif variant == "semantic_target_only":
        if before["wrong_target_action"]:
            proxy = _scene_proxy(obs, instruction, counterfactual_instruction)
            direction = _unit(np.asarray(proxy["target_pos"]) - np.asarray(proxy["eef"])) if proxy["eef"] is not None and proxy["target_pos"] is not None else None
            if direction is not None:
                out[:3] = direction * min(_norm(out[:3]), max_translation_norm)
                intervention = "redirect"
            else:
                out[:3] = 0.0
                intervention = "safe_stop"
    elif variant == "full_css_shield":
        if before["wrong_target_action"]:
            proxy = _scene_proxy(obs, instruction, counterfactual_instruction)
            direction = _unit(np.asarray(proxy["target_pos"]) - np.asarray(proxy["eef"])) if proxy["eef"] is not None and proxy["target_pos"] is not None else None
            out[:3] = direction * min(_norm(out[:3]), max_translation_norm) if direction is not None else 0.0
            intervention = "redirect" if direction is not None else "safe_stop"
        if before["unsafe_action"]:
            out = _limit_translation(out, max_translation_norm)
            intervention = "damp" if intervention == "accept" else intervention
    else:
        raise ValueError(f"unknown shield variant: {variant}")
    after = assess_action(out, obs, instruction, counterfactual_instruction, max_translation_norm=max_translation_norm)
    return out, {
        "variant": variant,
        "intervention": intervention,
        "intervened": bool(intervention != "accept"),
        "action_modification_l2": _round(_norm(out - clipped)),
        "before": before,
        "after": after,
    }


def _synthetic_proposal(obs: dict[str, Any], instruction: str, counterfactual_instruction: str | None, step: int) -> list[float]:
    proxy = _scene_proxy(obs, instruction, counterfactual_instruction)
    eef = proxy["eef"]
    wrong_pos = proxy["wrong_pos"]
    target_pos = proxy["target_pos"]
    direction = None
    if step % 3 == 0 and eef is not None and wrong_pos is not None:
        direction = _unit(np.asarray(wrong_pos) - np.asarray(eef))
    elif eef is not None and target_pos is not None:
        direction = _unit(np.asarray(target_pos) - np.asarray(eef))
    if direction is None:
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    scale = 0.28 if step % 4 == 0 else 0.12
    action = np.zeros(7, dtype=np.float64)
    action[:3] = direction * scale
    action[6] = -1.0
    return [float(v) for v in action.tolist()]


def _native_action(policy: Any, config: Any, tokenizer_root: Path, obs: dict[str, Any], instruction: str, action_dim: int, device: str) -> tuple[list[float], dict[str, Any]]:
    import torch

    batch, batch_meta = _build_batch(config, tokenizer_root, obs, instruction, device, CAMERA_ALIAS_STRATEGY_CURRENT, STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3)
    noise = torch.zeros((1, config.chunk_size, config.max_action_dim), dtype=torch.float32, device=device)
    with torch.inference_mode():
        policy_action = policy.select_action(batch, noise=noise)
    adapted = adapt_policy_action_to_env_action(policy_action, action_dim, strategy=ACTION_STRATEGY_GRIPPER_ZERO_HOLD)
    return [float(value) for value in adapted.values], {
        "policy_action_shape": list(policy_action.detach().cpu().shape),
        "adapter_metadata": adapted.metadata,
        "batch_keys": batch_meta.get("batch_keys"),
    }


def _load_native_policy(args: argparse.Namespace) -> tuple[Any | None, Any | None, Path | None, dict[str, Any]]:
    smolvla_ckpt = _as_path(args.smolvla_ckpt)
    checkpoint_root = _as_path(args.checkpoint_root)
    hf_home = _as_path(args.hf_home)
    info = {"attempted": False, "available": False, "error": None, "external_tokenizer": None}
    external = _external_tokenizer_files(_read_tokenizer_dependency(smolvla_ckpt), [hf_home, checkpoint_root])
    info["external_tokenizer"] = external
    if not (_find_files(smolvla_ckpt, ["config.json"]) and _find_files(smolvla_ckpt, ["model.safetensors", "pytorch_model.bin"], ["*.safetensors", "*.bin"]) and external.get("found")):
        info["error"] = "local SmolVLA files incomplete"
        return None, None, None, info
    deps = _runtime_dependencies()
    if not all(deps.values()):
        info["error"] = "missing runtime dependencies: " + ", ".join(name for name, ok in deps.items() if not ok)
        return None, None, None, info
    try:
        info["attempted"] = True
        policy, config = _load_policy(smolvla_ckpt, hf_home, external, args.device)
        policy.reset()
        info["available"] = True
        return policy, config, Path(external["root"]), info
    except Exception as exc:  # noqa: BLE001
        info["error"] = f"{type(exc).__name__}: {exc}"
        return None, None, None, info


def _run_rollout_variant(env_cls: Any, bddl_file: Path, case: dict[str, Any], shield_variant: str, proposal_source: str, policy: Any | None, config: Any | None, tokenizer_root: Path | None, args: argparse.Namespace) -> dict[str, Any]:
    init_state = np.asarray(case["init_state"], dtype=np.float64)
    expert = np.asarray(case["expert_actions"], dtype=np.float64)
    instruction = str(case["instruction"])
    counter_instruction = case.get("counterfactual_instruction")
    env = None
    actions: list[list[float]] = []
    proposed_actions: list[list[float]] = []
    step_records: list[dict[str, Any]] = []
    start_obs = None
    final_obs = None
    reward_sum = 0.0
    final_success = False
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=args.camera_size, camera_widths=args.camera_size)
        env.seed(0)
        obs = env.set_init_state(init_state)
        start_obs = obs
        action_dim = int(getattr(env, "action_dim", 7) or 7)
        for step in range(min(args.max_steps, expert.shape[0])):
            proposal_meta: dict[str, Any] = {}
            source_used = proposal_source
            if proposal_source == "native_smolvla" and policy is not None and config is not None and tokenizer_root is not None:
                proposed, proposal_meta = _native_action(policy, config, tokenizer_root, obs, instruction, action_dim, args.device)
            else:
                proposed = _synthetic_proposal(obs, instruction, counter_instruction, step)
                source_used = "synthetic_counterfactual_probe"
            shielded, shield_meta = apply_shield(proposed, obs, instruction, counter_instruction, shield_variant, max_translation_norm=args.max_translation_norm)
            obs, reward, done, _info = env.step([float(v) for v in shielded.tolist()])
            actions.append([float(v) for v in shielded.tolist()])
            proposed_actions.append([float(v) for v in proposed])
            reward_sum += float(reward)
            try:
                final_success = bool(env.check_success())
            except Exception:
                pass
            final_obs = obs
            step_records.append(
                {
                    "step": step,
                    "proposal_source": source_used,
                    "shield_variant": shield_variant,
                    "reward": float(reward),
                    "done": bool(done),
                    "final_success_so_far": final_success,
                    "l2_to_hdf5_expert_same_timestep": _safe_l2(shielded, expert[step]),
                    "proposal_l2_to_hdf5_expert_same_timestep": _safe_l2(proposed, expert[step]),
                    "shield": shield_meta,
                    "proposal_meta": proposal_meta,
                }
            )
        movement = _target_metrics(start_obs, final_obs, instruction, counter_instruction) if start_obs is not None and final_obs is not None else {}
    except Exception as exc:  # noqa: BLE001
        return {"shield_variant": shield_variant, "error": _compact_error(exc), "steps_performed": len(actions)}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    arr = np.asarray(actions, dtype=np.float64) if actions else np.zeros((0, 7), dtype=np.float64)
    prop = np.asarray(proposed_actions, dtype=np.float64) if proposed_actions else np.zeros((0, 7), dtype=np.float64)
    total = len(step_records)
    before_wrong = sum(1 for item in step_records if item["shield"]["before"]["wrong_target_action"])
    before_unsafe = sum(1 for item in step_records if item["shield"]["before"]["unsafe_action"])
    after_wrong = sum(1 for item in step_records if item["shield"]["after"]["wrong_target_action"])
    after_unsafe = sum(1 for item in step_records if item["shield"]["after"]["unsafe_action"])
    intervened = sum(1 for item in step_records if item["shield"]["intervened"])
    false_positive = sum(1 for item in step_records if item["shield"]["intervened"] and not item["shield"]["before"]["wrong_target_action"] and not item["shield"]["before"]["unsafe_action"])
    false_negative = sum(1 for item in step_records if item["shield"]["after"]["wrong_target_action"] or item["shield"]["after"]["unsafe_action"])
    mod = [float(item["shield"]["action_modification_l2"] or 0.0) for item in step_records]
    return {
        "shield_variant": shield_variant,
        "proposal_source_requested": proposal_source,
        "steps_performed": total,
        "reward_sum": round(float(reward_sum), 9),
        "final_success": final_success,
        "wrong_target_action_rate_before": _safe_rate(before_wrong, total),
        "wrong_target_action_rate_after": _safe_rate(after_wrong, total),
        "unsafe_action_rate_before": _safe_rate(before_unsafe, total),
        "unsafe_action_rate_after": _safe_rate(after_unsafe, total),
        "intervention_rate": _safe_rate(intervened, total),
        "false_positive_intervention_rate": _safe_rate(false_positive, total),
        "false_negative_unsafe_or_wrong_rate": _safe_rate(false_negative, total),
        "action_modification_l2_mean": _round(float(np.mean(mod)) if mod else 0.0),
        "action_modification_l2_max": _round(float(np.max(mod)) if mod else 0.0),
        "utility_preservation_proxy": movement.get("target_directed_movement_score"),
        "target_directed_movement_score": movement.get("target_directed_movement_score"),
        "wrong_target_movement": movement.get("wrong_target_movement"),
        "target_key_audit": movement.get("target_key_audit"),
        "eef_displacement_l2": movement.get("eef_displacement_l2"),
        "target_object_displacement_l2": movement.get("target_object_displacement_l2"),
        "action_shape": list(arr.shape),
        "proposal_action_stats": _action_stats(prop) if prop.size else None,
        "shielded_action_stats": _action_stats(arr) if arr.size else None,
        "expert_match": {
            "mean_l2": _round(float(np.mean(np.linalg.norm(arr - expert[: arr.shape[0]], axis=1))) if arr.size else None),
            "proposal_mean_l2": _round(float(np.mean(np.linalg.norm(prop - expert[: prop.shape[0]], axis=1))) if prop.size else None),
        },
        "step_records": step_records,
    }


def _compare_variants(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_name = {item["shield_variant"]: item for item in results if not item.get("error")}
    no = by_name.get("no_shield", {})
    clipping = by_name.get("clipping_only", {})
    safety = by_name.get("safety_only", {})
    full = by_name.get("full_css_shield", {})
    wrong_delta = None
    unsafe_delta = None
    clipping_wrong_delta = None
    clipping_unsafe_delta = None
    safety_wrong_delta = None
    safety_unsafe_delta = None
    if full and no:
        wrong_delta = _round(float(no.get("wrong_target_action_rate_after", 0.0)) - float(full.get("wrong_target_action_rate_after", 0.0)))
        unsafe_delta = _round(float(no.get("unsafe_action_rate_after", 0.0)) - float(full.get("unsafe_action_rate_after", 0.0)))
    if full and clipping:
        clipping_wrong_delta = _round(float(clipping.get("wrong_target_action_rate_after", 0.0)) - float(full.get("wrong_target_action_rate_after", 0.0)))
        clipping_unsafe_delta = _round(float(clipping.get("unsafe_action_rate_after", 0.0)) - float(full.get("unsafe_action_rate_after", 0.0)))
    if full and safety:
        safety_wrong_delta = _round(float(safety.get("wrong_target_action_rate_after", 0.0)) - float(full.get("wrong_target_action_rate_after", 0.0)))
        safety_unsafe_delta = _round(float(safety.get("unsafe_action_rate_after", 0.0)) - float(full.get("unsafe_action_rate_after", 0.0)))
    utility_drop = None
    if full and no and full.get("target_directed_movement_score") is not None and no.get("target_directed_movement_score") is not None:
        utility_drop = _round(float(no["target_directed_movement_score"]) - float(full["target_directed_movement_score"]))
    full_beats_clipping = bool((clipping_wrong_delta is not None and clipping_wrong_delta > 0.0) or (clipping_unsafe_delta is not None and clipping_unsafe_delta > 0.0))
    full_beats_safety = bool((safety_wrong_delta is not None and safety_wrong_delta > 0.0) or (safety_unsafe_delta is not None and safety_unsafe_delta > 0.0))
    return {
        "full_vs_no_shield_wrong_target_rate_reduction": wrong_delta,
        "full_vs_no_shield_unsafe_rate_reduction": unsafe_delta,
        "full_vs_clipping_wrong_target_rate_reduction": clipping_wrong_delta,
        "full_vs_clipping_unsafe_rate_reduction": clipping_unsafe_delta,
        "full_vs_safety_only_wrong_target_rate_reduction": safety_wrong_delta,
        "full_vs_safety_only_unsafe_rate_reduction": safety_unsafe_delta,
        "utility_drop_vs_no_shield": utility_drop,
        "full_shield_beats_clipping_only": full_beats_clipping,
        "full_shield_beats_safety_only": full_beats_safety,
        "full_shield_reduces_wrong_or_unsafe": bool((wrong_delta is not None and wrong_delta > 0.0) or (unsafe_delta is not None and unsafe_delta > 0.0)),
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    gate_set = _env_flag(TASK_GATE)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "stop",
        "policy": {
            "task_local_gate_set": gate_set,
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "lora_training_performed": False,
            "loss_computed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "rollout_happened": False,
            "benchmark_rollouts_performed": False,
            "openvla_oft_executed": False,
            "paper_grade_claims_made": False,
            "forbidden_gates_set": forbidden,
        },
        "result": {"passed": False, "blocked_reason": None},
    }
    if forbidden:
        report["result"]["blocked_reason"] = "Forbidden gate(s) set: " + ", ".join(forbidden)
        return report
    if args.max_steps < 1 or args.max_steps > MAX_STEPS:
        report["result"]["blocked_reason"] = f"max_steps must be between 1 and {MAX_STEPS}"
        return report
    if not gate_set:
        report["result"]["blocked_reason"] = f"{TASK_GATE}=1 is required for simulator rollout"
        return report
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        libero_root = _as_path(args.libero_root)
        robosuite_root = _as_path(args.robosuite_root)
        report["import_path_audit"] = _prepare_libero_import_path(libero_root, robosuite_root)
        from libero.libero.envs import OffScreenRenderEnv

        case = _load_first_case(_as_path(args.manifest), args.max_steps)
        _ensure_paths(libero_root, robosuite_root)
        bddl_file = _bddl_path(libero_root, case["suite"], case["task_id"])
        native_policy = native_config = None
        tokenizer_root = None
        native_info = {"attempted": False, "available": False}
        proposal_source = args.proposal_source
        if proposal_source in {"native_or_synthetic", "native_smolvla"}:
            native_policy, native_config, tokenizer_root, native_info = _load_native_policy(args)
            report["policy"]["heavy_model_imports_performed"] = bool(native_info.get("attempted"))
            report["policy"]["model_load_performed"] = bool(native_info.get("available"))
            if native_info.get("available"):
                proposal_source = "native_smolvla"
            elif args.proposal_source == "native_smolvla":
                report["result"]["blocked_reason"] = native_info.get("error") or "native SmolVLA unavailable"
                return report
            else:
                proposal_source = "synthetic_counterfactual_probe"
        results = [
            _run_rollout_variant(OffScreenRenderEnv, bddl_file, case, variant, proposal_source, native_policy, native_config, tokenizer_root, args)
            for variant in SHIELD_VARIANTS
        ]
        report["policy"]["rollout_happened"] = True
        report["policy"]["model_inference_performed"] = bool(proposal_source == "native_smolvla")
        comparison = _compare_variants(results)
        all_stop = all((item.get("intervention_rate") == 1.0) for item in results if not item.get("error"))
        continue_signal = bool(comparison["full_shield_reduces_wrong_or_unsafe"] and (comparison["full_shield_beats_clipping_only"] or comparison["full_shield_beats_safety_only"]) and not all_stop)
        report.update(
            {
                "decision": "css_shield_minimal_rollout_diagnostic_completed",
                "case": {
                    "manifest": str(_as_path(args.manifest)),
                    "suite": case["suite"],
                    "task_id": case["task_id"],
                    "pair_id": case.get("pair_id"),
                    "instruction": case["instruction"],
                    "counterfactual_instruction": case.get("counterfactual_instruction"),
                    "bddl_file": str(bddl_file),
                    "max_steps": args.max_steps,
                },
                "proposal_source": {"requested": args.proposal_source, "used": proposal_source, "native": native_info},
                "variants": results,
                "comparison": comparison,
                "state1_decision": {
                    "continue_to_state2": continue_signal,
                    "kill_now": not continue_signal,
                    "reason": "full shield reduced wrong-target or unsafe rate and beat at least one simple shield baseline" if continue_signal else "first diagnostic did not show a convincing full-shield advantage over simple shield baselines",
                },
                "result": {"passed": True, "blocked_reason": None, "elapsed_sec": _round(time.monotonic() - started, 3)},
            }
        )
    except Exception as exc:  # noqa: BLE001
        report["result"] = {"passed": False, "blocked_reason": f"{type(exc).__name__}: {exc}", "error": _compact_error(exc), "elapsed_sec": _round(time.monotonic() - started, 3)}
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    comparison = report.get("comparison") or {}
    state = report.get("state1_decision") or {}
    lines = [
        "# CSS-Shield Minimal Rollout Diagnostic",
        "",
        "This is bounded diagnostic rollout evidence only. It is not paper-grade benchmark evidence.",
        "",
        f"- decision: `{report.get('decision')}`",
        f"- rollout happened: `{(report.get('policy') or {}).get('rollout_happened')}`",
        f"- proposal source used: `{(report.get('proposal_source') or {}).get('used')}`",
        f"- full vs no-shield wrong-target reduction: `{comparison.get('full_vs_no_shield_wrong_target_rate_reduction')}`",
        f"- full vs no-shield unsafe reduction: `{comparison.get('full_vs_no_shield_unsafe_rate_reduction')}`",
        f"- full shield beats clipping-only: `{comparison.get('full_shield_beats_clipping_only')}`",
        f"- continue to State 2: `{state.get('continue_to_state2')}`",
        f"- kill now: `{state.get('kill_now')}`",
        f"- reason: {state.get('reason')}",
        "",
        "## Variants",
        "",
    ]
    for item in report.get("variants") or []:
        if item.get("error"):
            lines.append(f"- `{item.get('shield_variant')}`: error `{item['error'].get('message')}`")
            continue
        lines.append(f"- `{item.get('shield_variant')}`: wrong `{item.get('wrong_target_action_rate_after')}`, unsafe `{item.get('unsafe_action_rate_after')}`, intervention `{item.get('intervention_rate')}`, reward `{item.get('reward_sum')}`, success `{item.get('final_success')}`")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": report.get("schema_version"),
        "decision": report.get("decision"),
        "result": report.get("result"),
        "policy": report.get("policy"),
        "case": report.get("case"),
        "proposal_source": report.get("proposal_source"),
        "comparison": report.get("comparison"),
        "state1_decision": report.get("state1_decision"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--report-json", default="reports/css_shield_minimal_rollout_diagnostic_report.json")
    parser.add_argument("--report-md", default="reports/css_shield_minimal_rollout_diagnostic_report.md")
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--libero-root", default="C:/assets/repos/LIBERO")
    parser.add_argument("--robosuite-root", default="C:/assets/repos/robosuite")
    parser.add_argument("--max-steps", type=int, default=10)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--max-translation-norm", type=float, default=0.20)
    parser.add_argument("--proposal-source", choices=["native_or_synthetic", "native_smolvla", "synthetic_counterfactual_probe"], default="native_or_synthetic")
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    args = parser.parse_args(argv)

    report = build_report(args)
    json_path = Path(args.report_json)
    md_path = Path(args.report_md)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=lambda value: "<non-json>") + "\n", encoding="utf-8")
    _write_markdown(report, md_path)
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True, default=lambda value: "<non-json>"))
    return 0 if report.get("result", {}).get("passed") else 8


if __name__ == "__main__":
    sys.exit(main())
