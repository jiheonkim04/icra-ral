"""CSS-Shield State 1.5 semantic observability diagnostics.

This module runs a bounded executable diagnostic in the already validated
LIBERO/RoboSuite environment. It inventories observable objects, resolves an
intended target and distractor from instruction text plus scene names, and
tests whether semantic shielding adds anything beyond clipping/safety-only.

It is diagnostic-only: no training, no downloads, no GPU jobs, no OpenVLA-OFT,
and no paper-grade claim.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.datasets.libero_zero_reward_rollout_diagnosis import _distance, _extract_eef, _extract_pos
from tca_map.smolvla.online_action_generation_bridge import _as_path, _bddl_path, _load_first_case

from .minimal_rollout_diagnostic import (
    FORBIDDEN_GATES,
    SHIELD_VARIANTS,
    _compact_error,
    _load_native_policy,
    _native_action,
    _norm,
    _prepare_libero_import_path,
    _round,
    _safe_rate,
    _unit,
    apply_shield,
)

SCHEMA_VERSION = "2026-07-07.css_shield_state1_5_semantic_observability.v1"
TASK_GATE = "ALLOW_CSS_SHIELD_STATE15"
STATE2_TRIALS_DEFAULT = 20


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _tokenize(text: str) -> set[str]:
    stop = {
        "a",
        "an",
        "and",
        "both",
        "close",
        "in",
        "into",
        "it",
        "of",
        "on",
        "open",
        "pick",
        "place",
        "put",
        "the",
        "to",
        "turn",
        "up",
    }
    return {tok for tok in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")) if tok not in stop and not tok.isdigit()}


def _base_object_name(name: str) -> str:
    text = re.sub(r"(_to_robot0_eef)?_pos$", "", name)
    text = re.sub(r"_[0-9]+$", "", text)
    return text


def _human_name(name: str) -> str:
    return _base_object_name(name).replace("_", " ")


def _object_position_keys(obs: Any) -> list[str]:
    if not isinstance(obs, dict):
        return []
    keys = []
    for key in obs:
        if not isinstance(key, str) or not key.endswith("_pos"):
            continue
        if key.startswith("robot") or key in {"ee_pos", "eef_pos"}:
            continue
        if "_to_robot0_eef_pos" in key:
            continue
        keys.append(key)
    return sorted(keys)


def _eef_keys(obs: Any) -> list[str]:
    if not isinstance(obs, dict):
        return []
    out = []
    for key in obs:
        if isinstance(key, str) and ("eef" in key or key in {"ee_pos", "eef_pos"}):
            arr = np.asarray(obs[key]).reshape(-1)
            if arr.size >= 3:
                out.append(key)
    return sorted(out)


def resolve_semantic_targets(instruction: str, scene_object_names: list[str], *, counterfactual_instruction: str | None = None) -> dict[str, Any]:
    """Resolve target/distractors from instruction text and scene names only."""

    instruction_tokens = _tokenize(instruction)
    scored = []
    for name in scene_object_names:
        tokens = _tokenize(_human_name(name))
        overlap = sorted(instruction_tokens & tokens)
        scored.append({"name": name, "human_name": _human_name(name), "score": len(overlap), "overlap": overlap})
    scored.sort(key=lambda item: (-int(item["score"]), str(item["name"])))
    intended = scored[0] if scored and int(scored[0]["score"]) > 0 else None

    cf_tokens = _tokenize(counterfactual_instruction or "")
    distractors = []
    for item in scored:
        if intended and item["name"] == intended["name"]:
            continue
        cf_overlap = sorted(cf_tokens & _tokenize(item["human_name"]))
        distractors.append({**item, "counterfactual_overlap": cf_overlap, "counterfactual_score": len(cf_overlap)})
    distractors.sort(key=lambda item: (-int(item["counterfactual_score"]), -int(item["score"]), str(item["name"])))

    reason = None
    if not intended:
        reason = "intended target unresolved from instruction text and scene object names"
    elif not distractors:
        reason = "no distractor object available in scene object names"

    return {
        "uses_instruction_text": True,
        "uses_visible_scene_names": True,
        "uses_bddl_metadata": False,
        "uses_eval_labels": False,
        "uses_dataset_target_labels": False,
        "uses_task_id_or_filename": False,
        "intended_target": intended,
        "distractor_candidates": distractors,
        "selected_distractor": distractors[0] if distractors else None,
        "confidence": 0.0 if not intended else round(float(intended["score"]) / max(1, len(instruction_tokens)), 6),
        "failure_reason": reason,
        "all_scores": scored,
    }


def _names_from_model(model: Any, kind: str) -> list[str]:
    attr = f"{kind}_names"
    values = getattr(model, attr, None)
    if values:
        return [str(value) for value in values if str(value)]
    mapping = getattr(model, f"_{kind}_name2id", None)
    if isinstance(mapping, dict):
        return sorted(str(key) for key in mapping)
    count = int(getattr(model, f"n{kind}", 0) or 0)
    out = []
    accessor = getattr(model, kind, None)
    if callable(accessor):
        for idx in range(count):
            try:
                name = accessor(idx).name
            except Exception:
                name = None
            if name:
                out.append(str(name))
    return sorted(set(out))


def _find_sim_model(env: Any) -> tuple[Any | None, Any | None]:
    stack = [env]
    seen: set[int] = set()
    attrs = ("sim", "env", "_env", "base_env", "wrapped_env", "unwrapped")
    while stack:
        current = stack.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        sim = getattr(current, "sim", None)
        model = getattr(sim, "model", None) if sim is not None else None
        if sim is not None and model is not None:
            return sim, model
        for attr in attrs:
            child = getattr(current, attr, None)
            if child is not None and id(child) not in seen:
                stack.append(child)
    return None, None


def _positions_from_obs(obs: dict[str, Any], keys: list[str]) -> dict[str, list[float]]:
    out = {}
    for key in keys:
        pos = _extract_pos(obs, key)
        if pos is not None:
            out[key] = pos
    return out


def _positions_from_sim(sim: Any, model: Any, names: list[str], kind: str, limit: int = 80) -> dict[str, list[float]]:
    data = getattr(sim, "data", None)
    if data is None:
        return {}
    array = getattr(data, f"{kind}_xpos", None)
    if array is None:
        return {}
    id_getter = getattr(model, f"{kind}_name2id", None)
    out = {}
    for name in names[:limit]:
        try:
            idx = id_getter(name) if callable(id_getter) else None
            if idx is None and isinstance(getattr(model, f"_{kind}_name2id", None), dict):
                idx = getattr(model, f"_{kind}_name2id")[name]
            if idx is None:
                continue
            arr = np.asarray(array[int(idx)], dtype=np.float64).reshape(-1)
            if arr.size >= 3:
                out[name] = [float(value) for value in arr[:3]]
        except Exception:
            continue
    return out


def build_object_inventory(env: Any, obs: dict[str, Any], instruction: str, counterfactual_instruction: str | None) -> dict[str, Any]:
    sim, model = _find_sim_model(env)
    object_keys = _object_position_keys(obs)
    body_names = _names_from_model(model, "body") if model is not None else []
    site_names = _names_from_model(model, "site") if model is not None else []
    geom_names = _names_from_model(model, "geom") if model is not None else []
    resolver = resolve_semantic_targets(instruction, object_keys, counterfactual_instruction=counterfactual_instruction)
    target_key = (resolver.get("intended_target") or {}).get("name")
    wrong_key = (resolver.get("selected_distractor") or {}).get("name")
    eef = _extract_eef(obs)
    wrong_target_metric_computable = bool(eef is not None and target_key and wrong_key and _extract_pos(obs, target_key) is not None and _extract_pos(obs, wrong_key) is not None)
    return {
        "observation_keys": sorted(str(key) for key in obs.keys()),
        "object_related_observation_keys": object_keys,
        "eef_position_keys": _eef_keys(obs),
        "mujoco_body_names": body_names,
        "mujoco_site_names": site_names,
        "mujoco_geom_names": geom_names,
        "object_positions_from_obs": _positions_from_obs(obs, object_keys),
        "object_positions_from_mujoco_body": _positions_from_sim(sim, model, body_names, "body") if sim is not None and model is not None else {},
        "object_positions_from_mujoco_site": _positions_from_sim(sim, model, site_names, "site") if sim is not None and model is not None else {},
        "object_positions_from_mujoco_geom": _positions_from_sim(sim, model, geom_names, "geom") if sim is not None and model is not None else {},
        "semantic_resolver": resolver,
        "intended_target_resolvable": resolver.get("intended_target") is not None,
        "distractor_resolvable": resolver.get("selected_distractor") is not None,
        "wrong_target_metric_computable": wrong_target_metric_computable,
    }


def _direction_action(obs: dict[str, Any], target_key: str | None, *, scale: float, away: bool = False) -> list[float]:
    eef = _extract_eef(obs)
    pos = _extract_pos(obs, target_key) if target_key else None
    direction = _unit(np.asarray(pos) - np.asarray(eef)) if eef is not None and pos is not None else None
    if direction is None:
        direction = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    if away:
        direction = -direction
    action = np.zeros(7, dtype=np.float64)
    action[:3] = direction * scale
    action[6] = -1.0
    return [float(value) for value in action]


def _movement_scores(start_obs: dict[str, Any], final_obs: dict[str, Any], target_key: str | None, wrong_key: str | None) -> dict[str, Any]:
    start_eef = _extract_eef(start_obs)
    final_eef = _extract_eef(final_obs)
    target_start = _extract_pos(start_obs, target_key) if target_key else None
    target_final = _extract_pos(final_obs, target_key) if target_key else None
    wrong_start = _extract_pos(start_obs, wrong_key) if wrong_key else None
    wrong_final = _extract_pos(final_obs, wrong_key) if wrong_key else None

    def score(start_obj: list[float] | None, final_obj: list[float] | None) -> float | None:
        start_dist = _distance(start_eef, start_obj)
        final_dist = _distance(final_eef, final_obj)
        if start_dist is None or final_dist is None:
            return None
        return _round(float(start_dist - final_dist), 9)

    return {
        "intended_target_movement_score": score(target_start, target_final),
        "wrong_target_movement_score": score(wrong_start, wrong_final),
        "eef_displacement_l2": _round(_norm(np.asarray(final_eef) - np.asarray(start_eef)) if start_eef is not None and final_eef is not None else None),
        "target_position_available": target_start is not None and target_final is not None,
        "wrong_target_position_available": wrong_start is not None and wrong_final is not None,
    }


def _proposal_set(obs: dict[str, Any], target_key: str | None, wrong_key: str | None, native_action: list[float] | None = None) -> list[dict[str, Any]]:
    proposals = [
        {"name": "toward_intended_target", "action": _direction_action(obs, target_key, scale=0.12), "expected_wrong": False},
        {"name": "toward_distractor_target", "action": _direction_action(obs, wrong_key, scale=0.12), "expected_wrong": True},
        {"name": "away_from_targets", "action": _direction_action(obs, target_key, scale=0.12, away=True), "expected_wrong": False},
        {"name": "high_magnitude_unsafe", "action": _direction_action(obs, wrong_key or target_key, scale=0.85), "expected_wrong": bool(wrong_key)},
    ]
    if native_action is not None:
        proposals.append({"name": "native_smolvla_action", "action": native_action, "expected_wrong": None})
    return proposals


def _run_single_step(env: Any, init_state: np.ndarray, action: list[float], instruction: str, distractor_instruction: str, variant: str, target_key: str | None, wrong_key: str | None, max_translation_norm: float) -> dict[str, Any]:
    start_obs = env.set_init_state(init_state)
    shielded, shield = apply_shield(action, start_obs, instruction, distractor_instruction, variant, max_translation_norm=max_translation_norm)
    final_obs, reward, done, _info = env.step([float(value) for value in shielded.tolist()])
    try:
        success = bool(env.check_success())
    except Exception:
        success = False
    return {
        "shield_variant": variant,
        "reward": float(reward),
        "done": bool(done),
        "success": success,
        "shield": shield,
        "movement": _movement_scores(start_obs, final_obs, target_key, wrong_key),
    }


def _summarize_trials(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, list[dict[str, Any]]] = {variant: [] for variant in SHIELD_VARIANTS}
    for record in records:
        by_variant.setdefault(record["shield_variant"], []).append(record)

    summary: dict[str, Any] = {}
    for variant, items in by_variant.items():
        total = len(items)
        wrong_before = sum(1 for item in items if item["shield"]["before"]["wrong_target_action"])
        wrong_after = sum(1 for item in items if item["shield"]["after"]["wrong_target_action"])
        unsafe_before = sum(1 for item in items if item["shield"]["before"]["unsafe_action"])
        unsafe_after = sum(1 for item in items if item["shield"]["after"]["unsafe_action"])
        interventions = sum(1 for item in items if item["shield"]["intervened"])
        false_positive = sum(1 for item in items if item["shield"]["intervened"] and not item["shield"]["before"]["wrong_target_action"] and not item["shield"]["before"]["unsafe_action"])
        false_negative = sum(1 for item in items if item["shield"]["after"]["wrong_target_action"] or item["shield"]["after"]["unsafe_action"])
        mods = [float(item["shield"]["action_modification_l2"] or 0.0) for item in items]
        target_scores = [item["movement"]["intended_target_movement_score"] for item in items if item["movement"]["intended_target_movement_score"] is not None]
        wrong_scores = [item["movement"]["wrong_target_movement_score"] for item in items if item["movement"]["wrong_target_movement_score"] is not None]
        summary[variant] = {
            "trial_count": total,
            "semantic_wrong_target_rate_before": _safe_rate(wrong_before, total),
            "semantic_wrong_target_rate_after": _safe_rate(wrong_after, total),
            "unsafe_rate_before": _safe_rate(unsafe_before, total),
            "unsafe_rate_after": _safe_rate(unsafe_after, total),
            "intervention_rate": _safe_rate(interventions, total),
            "false_positive_intervention_rate": _safe_rate(false_positive, total),
            "false_negative_unsafe_or_wrong_rate": _safe_rate(false_negative, total),
            "action_modification_l2_mean": _round(float(np.mean(mods)) if mods else 0.0),
            "target_directed_movement_mean": _round(float(np.mean(target_scores)) if target_scores else None),
            "wrong_target_movement_mean": _round(float(np.mean(wrong_scores)) if wrong_scores else None),
        }
    full = summary.get("full_css_shield", {})
    safety = summary.get("safety_only", {})
    clipping = summary.get("clipping_only", {})
    return {
        "by_variant": summary,
        "comparison": {
            "full_vs_safety_wrong_target_delta": _round(float(safety.get("semantic_wrong_target_rate_after", 0.0)) - float(full.get("semantic_wrong_target_rate_after", 0.0))),
            "full_vs_clipping_wrong_target_delta": _round(float(clipping.get("semantic_wrong_target_rate_after", 0.0)) - float(full.get("semantic_wrong_target_rate_after", 0.0))),
            "full_vs_safety_unsafe_delta": _round(float(safety.get("unsafe_rate_after", 0.0)) - float(full.get("unsafe_rate_after", 0.0))),
            "full_vs_clipping_unsafe_delta": _round(float(clipping.get("unsafe_rate_after", 0.0)) - float(full.get("unsafe_rate_after", 0.0))),
            "full_intervention_rate": full.get("intervention_rate"),
        },
    }


def _run_proposals(env_cls: Any, bddl_file: Path, init_state: np.ndarray, proposals: list[dict[str, Any]], instruction: str, distractor_instruction: str, target_key: str | None, wrong_key: str | None, args: argparse.Namespace) -> list[dict[str, Any]]:
    records = []
    for proposal in proposals:
        env = None
        try:
            env = env_cls(bddl_file_name=str(bddl_file), camera_heights=args.camera_size, camera_widths=args.camera_size)
            env.seed(0)
            for variant in SHIELD_VARIANTS:
                result = _run_single_step(env, init_state, proposal["action"], instruction, distractor_instruction, variant, target_key, wrong_key, args.max_translation_norm)
                records.append({**result, "proposal_name": proposal["name"], "proposal_expected_wrong": proposal.get("expected_wrong"), "proposed_action": proposal["action"]})
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    return records


def _randomized_proposals(obs: dict[str, Any], target_key: str | None, wrong_key: str | None, count: int, seed: int) -> list[dict[str, Any]]:
    rng = np.random.default_rng(seed)
    proposals = []
    for idx in range(count):
        kind = idx % 4
        if kind == 0:
            action = _direction_action(obs, target_key, scale=float(rng.uniform(0.06, 0.16)))
            expected_wrong = False
            name = "random_toward_intended"
        elif kind == 1:
            action = _direction_action(obs, wrong_key, scale=float(rng.uniform(0.06, 0.16)))
            expected_wrong = True
            name = "random_toward_distractor"
        elif kind == 2:
            action = _direction_action(obs, wrong_key or target_key, scale=float(rng.uniform(0.24, 0.42)))
            expected_wrong = bool(wrong_key)
            name = "random_high_magnitude"
        else:
            vec = rng.normal(size=3)
            vec = vec / max(1e-9, float(np.linalg.norm(vec)))
            action = np.zeros(7, dtype=np.float64)
            action[:3] = vec * float(rng.uniform(0.04, 0.14))
            action[6] = -1.0
            expected_wrong = None
            name = "random_free_direction"
        proposals.append({"name": f"{name}_{idx:03d}", "action": [float(v) for v in action], "expected_wrong": expected_wrong})
    return proposals


def _decision_from_summary(summary: dict[str, Any], inventory: dict[str, Any], *, state2: bool = False) -> dict[str, Any]:
    comparison = summary["comparison"]
    by_variant = summary["by_variant"]
    full = by_variant.get("full_css_shield", {})
    semantic_only = by_variant.get("semantic_target_only", {})
    all_stop = bool(full.get("intervention_rate") == 1.0 and (full.get("target_directed_movement_mean") is None or float(full.get("target_directed_movement_mean") or 0.0) <= 0.0))
    semantic_catches = bool(semantic_only.get("semantic_wrong_target_rate_before", 0.0) > semantic_only.get("semantic_wrong_target_rate_after", 0.0))
    full_beats_safety_semantic = bool((comparison.get("full_vs_safety_wrong_target_delta") or 0.0) > 0.0)
    full_beats_clipping = bool((comparison.get("full_vs_clipping_wrong_target_delta") or 0.0) > 0.0 or (comparison.get("full_vs_clipping_unsafe_delta") or 0.0) > 0.0)
    green = bool(
        inventory.get("intended_target_resolvable")
        and inventory.get("distractor_resolvable")
        and inventory.get("wrong_target_metric_computable")
        and full_beats_safety_semantic
        and full_beats_clipping
        and not all_stop
    )
    return {
        "state": "STATE 2" if state2 else "STATE 1.5",
        "continue": green,
        "kill_now": not green,
        "reframe": not green,
        "semantic_only_catches_wrong_target": semantic_catches,
        "full_beats_safety_on_semantic_metric": full_beats_safety_semantic,
        "full_beats_clipping_on_semantic_or_safety_metric": full_beats_clipping,
        "full_stop_all": all_stop,
        "reason": "semantic wrong-target shielding beat safety-only and clipping-only without stop-all behavior"
        if green
        else "semantic wrong-target novelty gate did not pass",
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "task_local_gate_set": _env_flag(TASK_GATE),
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
    if not _env_flag(TASK_GATE):
        report["result"]["blocked_reason"] = f"{TASK_GATE}=1 is required"
        return report

    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
        libero_root = _as_path(args.libero_root)
        robosuite_root = _as_path(args.robosuite_root)
        report["import_path_audit"] = _prepare_libero_import_path(libero_root, robosuite_root)
        from libero.libero.envs import OffScreenRenderEnv

        case = _load_first_case(_as_path(args.manifest), max(5, args.max_steps))
        bddl_file = _bddl_path(libero_root, case["suite"], case["task_id"])
        env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=args.camera_size, camera_widths=args.camera_size)
        env.seed(0)
        obs = env.set_init_state(np.asarray(case["init_state"], dtype=np.float64))
        inventory = build_object_inventory(env, obs, str(case["instruction"]), case.get("counterfactual_instruction"))
        target_key = (inventory["semantic_resolver"].get("intended_target") or {}).get("name")
        wrong_key = (inventory["semantic_resolver"].get("selected_distractor") or {}).get("name")
        distractor_name = _human_name(str(wrong_key)) if wrong_key else "distractor object"
        distractor_instruction = f"move toward the {distractor_name}"

        native_action = None
        native_info = {"attempted": False, "available": False}
        if args.include_native:
            policy, config, tokenizer_root, native_info = _load_native_policy(args)
            report["policy"]["heavy_model_imports_performed"] = bool(native_info.get("attempted"))
            report["policy"]["model_load_performed"] = bool(native_info.get("available"))
            if policy is not None and config is not None and tokenizer_root is not None:
                native_action, _meta = _native_action(policy, config, tokenizer_root, obs, str(case["instruction"]), int(getattr(env, "action_dim", 7) or 7), args.device)
                report["policy"]["model_inference_performed"] = True
        try:
            env.close()
        except Exception:
            pass

        controlled = _proposal_set(obs, target_key, wrong_key, native_action)
        controlled_records = _run_proposals(OffScreenRenderEnv, bddl_file, np.asarray(case["init_state"], dtype=np.float64), controlled, str(case["instruction"]), distractor_instruction, target_key, wrong_key, args)
        controlled_summary = _summarize_trials(controlled_records)
        state15_decision = _decision_from_summary(controlled_summary, inventory)

        state2_report = None
        if args.run_state2_if_green and state15_decision["continue"]:
            randomized = _randomized_proposals(obs, target_key, wrong_key, args.state2_trials, args.seed)
            state2_records = _run_proposals(OffScreenRenderEnv, bddl_file, np.asarray(case["init_state"], dtype=np.float64), randomized, str(case["instruction"]), distractor_instruction, target_key, wrong_key, args)
            state2_summary = _summarize_trials(state2_records)
            state2_decision = _decision_from_summary(state2_summary, inventory, state2=True)
            state2_report = {
                "trial_count": args.state2_trials,
                "records": state2_records,
                "summary": state2_summary,
                "decision": state2_decision,
            }
        report["policy"]["rollout_happened"] = True
        report.update(
            {
                "decision": "css_shield_state1_5_completed",
                "case": {
                    "manifest": str(_as_path(args.manifest)),
                    "suite": case["suite"],
                    "task_id": case["task_id"],
                    "instruction": case["instruction"],
                    "counterfactual_instruction": case.get("counterfactual_instruction"),
                    "bddl_file": str(bddl_file),
                },
                "inventory": inventory,
                "distractor_instruction_used": distractor_instruction,
                "native_policy": native_info,
                "stage_c_controlled_diagnostic": {
                    "proposal_count": len(controlled),
                    "records": controlled_records,
                    "summary": controlled_summary,
                    "decision": state15_decision,
                },
                "state2_randomized_diagnostic": state2_report,
                "result": {"passed": True, "blocked_reason": None, "elapsed_sec": _round(time.monotonic() - started, 3)},
            }
        )
    except Exception as exc:  # noqa: BLE001
        report["result"] = {"passed": False, "blocked_reason": f"{type(exc).__name__}: {exc}", "error": _compact_error(exc), "elapsed_sec": _round(time.monotonic() - started, 3)}
    return report


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    stage = report.get("stage_c_controlled_diagnostic") or {}
    summary = stage.get("summary") or {}
    decision = stage.get("decision") or {}
    inventory = report.get("inventory") or {}
    resolver = inventory.get("semantic_resolver") or {}
    lines = [
        "# CSS-Shield State 1.5 Semantic Observability",
        "",
        "Diagnostic-only evidence. This is not paper-grade benchmark evidence.",
        "",
        f"- result passed: `{(report.get('result') or {}).get('passed')}`",
        f"- rollout happened: `{(report.get('policy') or {}).get('rollout_happened')}`",
        f"- model inference happened: `{(report.get('policy') or {}).get('model_inference_performed')}`",
        f"- intended target: `{(resolver.get('intended_target') or {}).get('name')}`",
        f"- selected distractor: `{(resolver.get('selected_distractor') or {}).get('name')}`",
        f"- wrong-target metric computable: `{inventory.get('wrong_target_metric_computable')}`",
        f"- State 1.5 continue: `{decision.get('continue')}`",
        f"- reason: {decision.get('reason')}",
        "",
        "## Controlled Summary",
        "",
    ]
    for variant, item in (summary.get("by_variant") or {}).items():
        lines.append(f"- `{variant}`: wrong `{item.get('semantic_wrong_target_rate_after')}`, unsafe `{item.get('unsafe_rate_after')}`, intervention `{item.get('intervention_rate')}`")
    if report.get("state2_randomized_diagnostic"):
        state2 = report["state2_randomized_diagnostic"]
        lines += [
            "",
            "## State 2 Randomized Batch",
            "",
            f"- trial count: `{state2.get('trial_count')}`",
            f"- continue: `{(state2.get('decision') or {}).get('continue')}`",
            f"- reason: {(state2.get('decision') or {}).get('reason')}",
        ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_inventory(report: dict[str, Any], json_path: Path, md_path: Path) -> None:
    inventory = report.get("inventory") or {}
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(inventory, indent=2, sort_keys=True, default=lambda value: "<non-json>") + "\n", encoding="utf-8")
    resolver = inventory.get("semantic_resolver") or {}
    lines = [
        "# CSS-Shield State 1.5 Object Inventory",
        "",
        f"- observation keys: `{len(inventory.get('observation_keys') or [])}`",
        f"- object-related observation keys: `{inventory.get('object_related_observation_keys')}`",
        f"- EEF keys: `{inventory.get('eef_position_keys')}`",
        f"- MuJoCo body names: `{len(inventory.get('mujoco_body_names') or [])}`",
        f"- MuJoCo site names: `{len(inventory.get('mujoco_site_names') or [])}`",
        f"- MuJoCo geom names: `{len(inventory.get('mujoco_geom_names') or [])}`",
        f"- intended target: `{(resolver.get('intended_target') or {}).get('name')}`",
        f"- selected distractor: `{(resolver.get('selected_distractor') or {}).get('name')}`",
        f"- wrong-target metric computable: `{inventory.get('wrong_target_metric_computable')}`",
        "",
    ]
    md_path.write_text("\n".join(lines), encoding="utf-8")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    stage = report.get("stage_c_controlled_diagnostic") or {}
    state2 = report.get("state2_randomized_diagnostic") or {}
    return {
        "schema_version": report.get("schema_version"),
        "result": report.get("result"),
        "policy": report.get("policy"),
        "case": report.get("case"),
        "intended_target": (((report.get("inventory") or {}).get("semantic_resolver") or {}).get("intended_target") or {}).get("name"),
        "selected_distractor": (((report.get("inventory") or {}).get("semantic_resolver") or {}).get("selected_distractor") or {}).get("name"),
        "wrong_target_metric_computable": (report.get("inventory") or {}).get("wrong_target_metric_computable"),
        "state1_5_decision": stage.get("decision"),
        "state2_decision": state2.get("decision") if state2 else None,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--report-json", default="reports/css_shield_state1_5_semantic_diagnostic_report.json")
    parser.add_argument("--report-md", default="reports/css_shield_state1_5_semantic_diagnostic_report.md")
    parser.add_argument("--inventory-json", default="reports/css_shield_state1_5_object_inventory.json")
    parser.add_argument("--inventory-md", default="reports/css_shield_state1_5_object_inventory.md")
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--libero-root", default="C:/assets/repos/LIBERO")
    parser.add_argument("--robosuite-root", default="C:/assets/repos/robosuite")
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--max-translation-norm", type=float, default=0.20)
    parser.add_argument("--include-native", action="store_true")
    parser.add_argument("--run-state2-if-green", action="store_true")
    parser.add_argument("--state2-trials", type=int, default=STATE2_TRIALS_DEFAULT)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", default="cpu", choices=["cpu"])
    args = parser.parse_args(argv)

    report = build_report(args)
    report_json = Path(args.report_json)
    report_md = Path(args.report_md)
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True, default=lambda value: "<non-json>") + "\n", encoding="utf-8")
    _write_markdown(report, report_md)
    _write_inventory(report, Path(args.inventory_json), Path(args.inventory_md))
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True, default=lambda value: "<non-json>"))
    return 0 if report.get("result", {}).get("passed") else 8


if __name__ == "__main__":
    sys.exit(main())
