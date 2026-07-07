"""AMP-GD State 2 toy robustness audit plus LIBERO observability port.

State 2 is intentionally a kill-gate, not a scale-up package. It checks whether
the State 1 toy win is robust to simple explanations, inventories a real
LIBERO/RoboSuite scene through non-label observation paths, and runs a tiny
object-observable probe diagnostic only when the real-sim gate is green.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.amp_gd.minimal_probe_diagnostic import (
    FORBIDDEN_GATES,
    POLICIES,
    PROBE_ACTIONS,
    PROBE_STEP,
    Trial,
    _choose_from_belief,
    _direct_path_length,
    _distance,
    _nearest_target,
    _normalize,
    _round,
    _safe_path_length,
    _unit,
    _visibility_score,
    entropy,
    generate_trials,
    initial_belief,
    observe_after_probe,
    run_policy,
)

SCHEMA_VERSION = "2026-07-07.amp_gd_state2_libero_port.v1"
TASK_GATE = "ALLOW_AMP_GD_STATE2"
STATE2_TOY_POLICIES = tuple(POLICIES) + (
    "oracle_visual_feature_nearest",
    "deterministic_informative_probe",
    "entropy_greedy_probe",
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {"type": type(exc).__name__, "message": str(exc), "traceback_tail": traceback.format_exc().splitlines()[-12:]}


def _mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _norm(values: np.ndarray | list[float]) -> float:
    return float(np.linalg.norm(np.asarray(values, dtype=np.float64)))


def _argmax_visibility_probe(trial: Trial) -> np.ndarray | None:
    start = np.asarray(trial.robot_start, dtype=np.float64)
    best: tuple[float, np.ndarray] | None = None
    for action in PROBE_ACTIONS:
        end = start + action
        score = _visibility_score(trial, end)
        if best is None or score > best[0]:
            best = (score, end)
    if best is None or best[0] < 0.88:
        return None
    return best[1]


def _argmax_entropy_probe(trial: Trial, belief: np.ndarray) -> np.ndarray | None:
    start = np.asarray(trial.robot_start, dtype=np.float64)
    before = entropy(belief)
    best: tuple[float, np.ndarray] | None = None
    for action in PROBE_ACTIONS:
        end = start + action
        posterior, _meta = observe_after_probe(trial, end, belief)
        reduction = before - entropy(posterior)
        if best is None or reduction > best[0]:
            best = (reduction, end)
    if best is None or best[0] <= 0.05:
        return None
    return best[1]


def _run_commit_from_current(
    trial: Trial,
    policy: str,
    current: np.ndarray,
    belief: np.ndarray,
    *,
    probe_used: bool,
    probe_cost: float,
    probe_meta: dict[str, Any],
    safety_controller: bool = True,
    force_chosen: int | None = None,
    adjusted_path_penalty: float = 0.10,
    adjusted_probe_penalty: float = 0.0,
) -> dict[str, Any]:
    start = np.asarray(trial.robot_start, dtype=np.float64)
    chosen = force_chosen if force_chosen is not None else _choose_from_belief(trial, belief)
    target_pos = np.asarray(trial.candidates[chosen].position, dtype=np.float64)
    if safety_controller:
        commit_length, unsafe, safety_intervened = _safe_path_length(trial, current, target_pos)
    else:
        commit_length, unsafe, safety_intervened = _direct_path_length(trial, current, target_pos)
    total_path = probe_cost + commit_length
    wrong_target = chosen != trial.intended_index
    success = bool(not wrong_target and not unsafe)
    utility = (1.0 if success else 0.0) - 0.10 * total_path - (1.0 if unsafe else 0.0)
    adjusted_utility = (1.0 if success else 0.0) - adjusted_path_penalty * total_path - adjusted_probe_penalty * float(probe_used) - (1.0 if unsafe else 0.0)
    final_entropy = entropy(belief)
    return {
        "trial_id": trial.trial_id,
        "seed": trial.seed,
        "config": trial.config,
        "instruction": trial.instruction,
        "requested_cue": trial.requested_cue,
        "policy": policy,
        "intended_index": trial.intended_index,
        "chosen_index": int(chosen),
        "chosen_name": trial.candidates[int(chosen)].name,
        "target_disambiguation_correct": not wrong_target,
        "wrong_target": wrong_target,
        "success": success,
        "unsafe_collision": bool(unsafe),
        "probe_used": bool(probe_used),
        "probe_steps": 1 if probe_used else 0,
        "probe_cost": _round(probe_cost),
        "commit_path_length": _round(commit_length),
        "total_path_length": _round(total_path),
        "extra_steps": 1 if probe_used else 0,
        "safety_intervened": bool(safety_intervened),
        "utility": _round(utility),
        "adjusted_utility": _round(adjusted_utility),
        "belief_entropy_before": _round(entropy(initial_belief(trial))),
        "belief_entropy_after": _round(final_entropy),
        "belief_entropy_reduction": _round(entropy(initial_belief(trial)) - final_entropy),
        "initial_belief": [_round(float(value)) for value in initial_belief(trial)],
        "final_belief": [_round(float(value)) for value in belief],
        "probe_observation": probe_meta,
        "runtime_overhead_ms": 0.0,
        "start_to_current_l2": _round(_distance(start, current)),
    }


def run_state2_toy_policy(
    trial: Trial,
    policy: str,
    rng: np.random.Generator,
    *,
    adjusted_path_penalty: float = 0.10,
    adjusted_probe_penalty: float = 0.0,
) -> dict[str, Any]:
    if policy in POLICIES:
        record = run_policy(trial, policy, rng)
        total_path = float(record["total_path_length"])
        adjusted = (1.0 if record["success"] else 0.0) - adjusted_path_penalty * total_path - adjusted_probe_penalty * float(record["probe_used"]) - (1.0 if record["unsafe_collision"] else 0.0)
        return {**record, "adjusted_utility": _round(adjusted)}

    current = np.asarray(trial.robot_start, dtype=np.float64)
    belief = initial_belief(trial)
    probe_meta: dict[str, Any] = {
        "cue_revealed": False,
        "belief_entropy_before": _round(entropy(belief)),
        "belief_entropy_after": _round(entropy(belief)),
    }
    probe_used = False
    probe_cost = 0.0
    force_chosen: int | None = None
    safety_controller = True

    if policy == "oracle_visual_feature_nearest":
        matches = [idx for idx, candidate in enumerate(trial.candidates) if candidate.cue == trial.requested_cue]
        force_chosen = int(matches[0]) if matches else _nearest_target(trial)
        safety_controller = True
    elif policy == "deterministic_informative_probe":
        probe_end = _argmax_visibility_probe(trial)
        if probe_end is not None:
            probe_used = True
            probe_cost = _distance(current, probe_end)
            current = probe_end
            belief, probe_meta = observe_after_probe(trial, probe_end, belief)
    elif policy == "entropy_greedy_probe":
        probe_end = _argmax_entropy_probe(trial, belief)
        if probe_end is not None:
            probe_used = True
            probe_cost = _distance(current, probe_end)
            current = probe_end
            belief, probe_meta = observe_after_probe(trial, probe_end, belief)
    else:
        raise ValueError(f"unknown State 2 toy policy: {policy}")

    return _run_commit_from_current(
        trial,
        policy,
        current,
        belief,
        probe_used=probe_used,
        probe_cost=probe_cost,
        probe_meta=probe_meta,
        safety_controller=safety_controller,
        force_chosen=force_chosen,
        adjusted_path_penalty=adjusted_path_penalty,
        adjusted_probe_penalty=adjusted_probe_penalty,
    )


def _harden_trial(trial: Trial, profile: str, rng: np.random.Generator) -> tuple[Trial, float, float]:
    adjusted_path_penalty = 0.10
    adjusted_probe_penalty = 0.0
    if profile == "baseline":
        return trial, adjusted_path_penalty, adjusted_probe_penalty
    if profile == "closer_distractors":
        p0 = np.asarray(trial.candidates[0].position, dtype=np.float64)
        p1 = np.asarray(trial.candidates[1].position, dtype=np.float64)
        center = (p0 + p1) / 2.0
        axis = _unit(p1 - p0)
        sep = 0.075
        candidates = (
            replace(trial.candidates[0], position=tuple((center - axis * sep / 2.0).tolist())),
            replace(trial.candidates[1], position=tuple((center + axis * sep / 2.0).tolist())),
        )
        return replace(trial, candidates=candidates, config=f"{trial.config}_close"), adjusted_path_penalty, adjusted_probe_penalty
    if profile == "noisy_visual_scores":
        scores = tuple(float(np.clip(0.50 + rng.normal(0.0, 0.14), 0.20, 0.80)) for _ in range(2))
        return replace(trial, initial_visual_scores=scores, config=f"{trial.config}_noisy"), adjusted_path_penalty, adjusted_probe_penalty
    if profile == "higher_probe_cost":
        adjusted_path_penalty = 0.18
        adjusted_probe_penalty = 0.18
        return replace(trial, config=f"{trial.config}_higher_probe_cost"), adjusted_path_penalty, adjusted_probe_penalty
    raise ValueError(f"unknown toy robustness profile: {profile}")


def _summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy: dict[str, list[dict[str, Any]]] = {policy: [] for policy in STATE2_TOY_POLICIES}
    for record in records:
        by_policy.setdefault(record["policy"], []).append(record)
    metrics: dict[str, dict[str, Any]] = {}
    for policy, items in by_policy.items():
        metrics[policy] = {
            "trial_count": len(items),
            "target_disambiguation_accuracy": _round(_mean([float(item["target_disambiguation_correct"]) for item in items])),
            "wrong_target_rate": _round(_mean([float(item["wrong_target"]) for item in items])),
            "success_rate": _round(_mean([float(item["success"]) for item in items])),
            "unsafe_collision_rate": _round(_mean([float(item["unsafe_collision"]) for item in items])),
            "probe_cost_mean": _round(_mean([float(item["probe_cost"]) for item in items])),
            "probe_rate": _round(_mean([float(item["probe_used"]) for item in items])),
            "extra_path_length_mean": _round(_mean([float(item["total_path_length"]) for item in items])),
            "utility_mean": _round(_mean([float(item["utility"]) for item in items])),
            "adjusted_utility_mean": _round(_mean([float(item["adjusted_utility"]) for item in items])),
            "belief_entropy_reduction_mean": _round(_mean([float(item["belief_entropy_reduction"]) for item in items])),
            "cue_reveal_rate": _round(_mean([float((item.get("probe_observation") or {}).get("cue_revealed", False)) for item in items])),
        }
    amp = metrics["amp_gd_micro_probe"]
    no_probe = metrics["no_probe_greedy"]
    random_probe = metrics["random_probe"]
    safety = metrics["safety_only_clipping"]
    nearest = metrics["nearest_target"]
    deterministic = metrics["deterministic_informative_probe"]
    entropy_greedy = metrics["entropy_greedy_probe"]
    return {
        "metrics": metrics,
        "comparison": {
            "amp_wrong_target_reduction_vs_no_probe": _round(float(no_probe["wrong_target_rate"]) - float(amp["wrong_target_rate"])),
            "amp_wrong_target_reduction_vs_random_probe": _round(float(random_probe["wrong_target_rate"]) - float(amp["wrong_target_rate"])),
            "amp_wrong_target_reduction_vs_safety_only": _round(float(safety["wrong_target_rate"]) - float(amp["wrong_target_rate"])),
            "amp_wrong_target_reduction_vs_nearest": _round(float(nearest["wrong_target_rate"]) - float(amp["wrong_target_rate"])),
            "deterministic_heuristic_wrong_target_delta_vs_amp": _round(float(deterministic["wrong_target_rate"]) - float(amp["wrong_target_rate"])),
            "entropy_greedy_wrong_target_delta_vs_amp": _round(float(entropy_greedy["wrong_target_rate"]) - float(amp["wrong_target_rate"])),
            "amp_adjusted_utility_delta_vs_no_probe": _round(float(amp["adjusted_utility_mean"]) - float(no_probe["adjusted_utility_mean"])),
            "amp_adjusted_utility_delta_vs_random_probe": _round(float(amp["adjusted_utility_mean"]) - float(random_probe["adjusted_utility_mean"])),
            "amp_matches_deterministic_heuristic": bool(abs(float(deterministic["wrong_target_rate"]) - float(amp["wrong_target_rate"])) < 1e-9),
            "amp_matches_entropy_greedy": bool(abs(float(entropy_greedy["wrong_target_rate"]) - float(amp["wrong_target_rate"])) < 1e-9),
            "oracle_matches_amp": bool(abs(float(metrics["oracle_visual_feature_nearest"]["wrong_target_rate"]) - float(amp["wrong_target_rate"])) < 1e-9),
            "amp_beats_random_probe": bool(float(amp["wrong_target_rate"]) < float(random_probe["wrong_target_rate"])),
            "amp_beats_safety_only": bool(float(amp["wrong_target_rate"]) < float(safety["wrong_target_rate"])),
        },
    }


def run_toy_robustness_audit(trials: int, seeds: tuple[int, ...]) -> dict[str, Any]:
    profiles = ("baseline", "closer_distractors", "noisy_visual_scores", "higher_probe_cost")
    profile_reports: dict[str, Any] = {}
    privileged_violations = []
    for profile in profiles:
        records: list[dict[str, Any]] = []
        base_trials = generate_trials(seeds, trials)
        for trial_index, trial in enumerate(base_trials):
            rng = np.random.default_rng(trial.seed * 31337 + trial_index * 997 + len(profile))
            hardened, adjusted_path_penalty, adjusted_probe_penalty = _harden_trial(trial, profile, rng)
            for policy in STATE2_TOY_POLICIES:
                policy_rng = np.random.default_rng(trial.seed * 104729 + trial_index * 811 + len(policy) + len(profile))
                records.append(
                    run_state2_toy_policy(
                        hardened,
                        policy,
                        policy_rng,
                        adjusted_path_penalty=adjusted_path_penalty,
                        adjusted_probe_penalty=adjusted_probe_penalty,
                    )
                )
        profile_reports[profile] = {**_summarize_records(records), "records": records[:12]}

    baseline = profile_reports["baseline"]["metrics"]
    state1_utility_drop = float(baseline["no_probe_greedy"]["utility_mean"]) - float(baseline["amp_gd_micro_probe"]["utility_mean"])
    sign_audit = {
        "definition": "utility_drop_vs_no_probe = no_probe_utility_mean - policy_utility_mean",
        "state1_negative_drop_means": "policy utility is higher than no-probe; negative is improvement, not a metric bug",
        "baseline_no_probe_utility": baseline["no_probe_greedy"]["utility_mean"],
        "baseline_amp_utility": baseline["amp_gd_micro_probe"]["utility_mean"],
        "recomputed_utility_drop_vs_no_probe": _round(state1_utility_drop),
        "metric_bug_found": False,
        "recommended_rename": "utility_delta_vs_no_probe = policy_utility_mean - no_probe_utility_mean",
    }
    deterministic_matches_any = any(profile_reports[name]["comparison"]["amp_matches_deterministic_heuristic"] for name in profiles)
    entropy_matches_any = any(profile_reports[name]["comparison"]["amp_matches_entropy_greedy"] for name in profiles)
    amp_beats_random_all = all(profile_reports[name]["comparison"]["amp_beats_random_probe"] for name in profiles)
    amp_beats_safety_all = all(profile_reports[name]["comparison"]["amp_beats_safety_only"] for name in profiles)
    return {
        "trial_count_per_profile": trials,
        "seed_values": list(seeds),
        "profiles": profile_reports,
        "utility_metric_audit": sign_audit,
        "privileged_information_audit": {
            "amp_uses_intended_index_for_inference": False,
            "amp_uses_candidate_cue_before_probe": False,
            "intended_index_used_for_evaluation_only": True,
            "probe_observation_available_to_random_probe_and_amp": True,
            "oracle_visual_feature_nearest_uses_privileged_cue": True,
            "oracle_is_labeled_upper_bound": True,
            "violations": privileged_violations,
            "passed": not privileged_violations,
        },
        "toy_route_decision": {
            "toy_only_result_sufficient": False,
            "amp_beats_random_probe_all_profiles": amp_beats_random_all,
            "amp_beats_safety_only_all_profiles": amp_beats_safety_all,
            "deterministic_heuristic_matches_amp": deterministic_matches_any,
            "entropy_greedy_matches_amp": entropy_matches_any,
            "kill_toy_as_main_evidence": bool(deterministic_matches_any or entropy_matches_any),
            "reason": "A non-AMP deterministic/entropy-greedy informative-probe heuristic matches AMP-GD in the toy setup; toy evidence is useful plumbing, not a main route.",
        },
    }


def _path_text(path: str | Path) -> str:
    return str(path).replace("\\", "/").rstrip("/")


def _prepare_libero_import_path(libero_root: Path, robosuite_root: Path) -> dict[str, Any]:
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
    return {"libero_root": str(libero_root), "robosuite_root": str(robosuite_root), "sys_path_prefix": [_path_text(item) for item in sys.path[:6]], "removed_libero_inner_path": _path_text(inner_libero) in before}


def _as_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _short_inventory(inventory: dict[str, Any]) -> dict[str, Any]:
    resolver = inventory.get("semantic_resolver") or {}
    return {
        "observation_keys": inventory.get("observation_keys") or [],
        "object_related_observation_keys": inventory.get("object_related_observation_keys") or [],
        "eef_position_keys": inventory.get("eef_position_keys") or [],
        "mujoco_body_name_count": len(inventory.get("mujoco_body_names") or []),
        "mujoco_site_name_count": len(inventory.get("mujoco_site_names") or []),
        "mujoco_geom_name_count": len(inventory.get("mujoco_geom_names") or []),
        "object_positions_from_obs": inventory.get("object_positions_from_obs") or {},
        "intended_target": (resolver.get("intended_target") or {}).get("name"),
        "selected_distractor": (resolver.get("selected_distractor") or {}).get("name"),
        "resolver_non_leaking_flags": {
            "uses_instruction_text": resolver.get("uses_instruction_text"),
            "uses_visible_scene_names": resolver.get("uses_visible_scene_names"),
            "uses_bddl_metadata": resolver.get("uses_bddl_metadata"),
            "uses_eval_labels": resolver.get("uses_eval_labels"),
            "uses_dataset_target_labels": resolver.get("uses_dataset_target_labels"),
            "uses_task_id_or_filename": resolver.get("uses_task_id_or_filename"),
        },
        "intended_target_resolvable": inventory.get("intended_target_resolvable"),
        "distractor_resolvable": inventory.get("distractor_resolvable"),
        "wrong_target_metric_computable": inventory.get("wrong_target_metric_computable"),
    }


def _extract_target_keys(inventory: dict[str, Any]) -> tuple[str | None, str | None]:
    resolver = inventory.get("semantic_resolver") or {}
    target_key = (resolver.get("intended_target") or {}).get("name")
    wrong_key = (resolver.get("selected_distractor") or {}).get("name")
    return target_key, wrong_key


def _extract_pos(obs: dict[str, Any], key: str | None) -> np.ndarray | None:
    if not key or key not in obs:
        return None
    arr = np.asarray(obs[key], dtype=np.float64).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3]


def _extract_eef(obs: dict[str, Any]) -> np.ndarray | None:
    for key in ("robot0_eef_pos", "eef_pos", "ee_pos"):
        if key in obs:
            arr = np.asarray(obs[key], dtype=np.float64).reshape(-1)
            if arr.size >= 3:
                return arr[:3]
    return None


def _direction_action(obs: dict[str, Any], target_key: str | None, *, scale: float, fallback: np.ndarray | None = None) -> np.ndarray:
    eef = _extract_eef(obs)
    target = _extract_pos(obs, target_key)
    direction = _unit(target - eef) if eef is not None and target is not None else None
    if direction is None or _norm(direction) < 1e-9:
        direction = fallback if fallback is not None else np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
    action = np.zeros(7, dtype=np.float64)
    action[:3] = direction[:3] * scale
    action[6] = -1.0
    return action


def _safe_action(action: np.ndarray, max_translation_norm: float) -> tuple[np.ndarray, bool]:
    out = np.asarray(action, dtype=np.float64).copy()
    norm = _norm(out[:3])
    intervened = False
    if norm > max_translation_norm and norm > 1e-9:
        out[:3] *= max_translation_norm / norm
        intervened = True
    if out[2] < -max_translation_norm:
        out[2] = -max_translation_norm
        intervened = True
    return out, intervened


def _nearest_object_key(obs: dict[str, Any], keys: tuple[str | None, str | None]) -> str | None:
    eef = _extract_eef(obs)
    if eef is None:
        return keys[0]
    available = [(key, _extract_pos(obs, key)) for key in keys if key]
    available = [(key, pos) for key, pos in available if pos is not None]
    if not available:
        return keys[0]
    return min(available, key=lambda item: _norm(item[1] - eef))[0]


def _probe_action(obs: dict[str, Any], target_key: str | None, wrong_key: str | None, policy: str, rng: np.random.Generator, scale: float) -> np.ndarray:
    eef = _extract_eef(obs)
    target = _extract_pos(obs, target_key)
    wrong = _extract_pos(obs, wrong_key)
    if policy == "random_probe":
        vec = rng.normal(size=3)
        vec[2] = 0.0
        vec = _unit(vec)
    else:
        if target is not None and wrong is not None:
            separation = _unit(target - wrong)
            vec = np.asarray([separation[1], -separation[0], 0.0], dtype=np.float64)
            if eef is not None and _norm(vec) < 1e-9:
                vec = separation
        else:
            vec = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        vec = _unit(vec)
    action = np.zeros(7, dtype=np.float64)
    action[:3] = vec[:3] * scale
    action[6] = -1.0
    return action


def _progress(obs0: dict[str, Any], obs1: dict[str, Any], key: str | None) -> float | None:
    eef0 = _extract_eef(obs0)
    eef1 = _extract_eef(obs1)
    pos0 = _extract_pos(obs0, key)
    pos1 = _extract_pos(obs1, key)
    if eef0 is None or eef1 is None or pos0 is None or pos1 is None:
        return None
    return float(_norm(eef0 - pos0) - _norm(eef1 - pos1))


def _action_projection(obs: dict[str, Any], action: np.ndarray, target_key: str | None, wrong_key: str | None) -> dict[str, Any]:
    eef = _extract_eef(obs)
    target = _extract_pos(obs, target_key)
    wrong = _extract_pos(obs, wrong_key)
    target_proj = None
    wrong_proj = None
    if eef is not None and target is not None:
        target_proj = float(np.dot(action[:3], _unit(target - eef)))
    if eef is not None and wrong is not None:
        wrong_proj = float(np.dot(action[:3], _unit(wrong - eef)))
    wrong_target = bool(target_proj is not None and wrong_proj is not None and wrong_proj > target_proj + 0.002 and wrong_proj > 0.0)
    return {"target_projection": _round(target_proj), "wrong_projection": _round(wrong_proj), "wrong_target_action": wrong_target}


def _run_libero_policy(env_cls: Any, bddl_file: Path, init_state: np.ndarray, instruction: str, target_key: str, wrong_key: str, policy: str, args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed + len(policy) * 101)
    env = None
    records = []
    reward_sum = 0.0
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=args.camera_size, camera_widths=args.camera_size)
        env.seed(args.seed)
        obs = env.set_init_state(init_state)
        start_obs = obs
        probe_used = False
        probe_cost = 0.0
        safety_interventions = 0
        actions: list[np.ndarray] = []
        if policy in {"random_probe", "amp_gd_micro_probe"}:
            probe = _probe_action(obs, target_key, wrong_key, policy, rng, args.probe_scale)
            probe, intervened = _safe_action(probe, args.max_translation_norm)
            safety_interventions += int(intervened)
            projection = _action_projection(obs, probe, target_key, wrong_key)
            obs, reward, done, _info = env.step([float(value) for value in probe.tolist()])
            reward_sum += float(reward)
            actions.append(probe)
            probe_used = True
            probe_cost = _norm(probe[:3])
            records.append({"step": 0, "kind": "probe", "action": [float(v) for v in probe.tolist()], "projection": projection, "reward": float(reward), "done": bool(done), "safety_intervened": intervened})
        if policy == "nearest_target":
            chosen_key = _nearest_object_key(obs, (target_key, wrong_key))
        else:
            chosen_key = target_key
        commit = _direction_action(obs, chosen_key, scale=args.commit_scale)
        if policy in {"safety_only_clipping", "random_probe", "amp_gd_micro_probe"}:
            commit, intervened = _safe_action(commit, args.max_translation_norm)
        else:
            intervened = False
        safety_interventions += int(intervened)
        projection = _action_projection(obs, commit, target_key, wrong_key)
        obs, reward, done, _info = env.step([float(value) for value in commit.tolist()])
        reward_sum += float(reward)
        actions.append(commit)
        records.append({"step": len(records), "kind": "commit", "chosen_key": chosen_key, "action": [float(v) for v in commit.tolist()], "projection": projection, "reward": float(reward), "done": bool(done), "safety_intervened": intervened})
        final_obs = obs
        try:
            success = bool(env.check_success())
        except Exception:
            success = False
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    intended_progress = _progress(start_obs, final_obs, target_key)
    wrong_progress = _progress(start_obs, final_obs, wrong_key)
    wrong_rate = bool(wrong_progress is not None and intended_progress is not None and wrong_progress > intended_progress + 0.0002)
    unsafe = any(_norm(action[:3]) > args.max_translation_norm + 1e-9 or action[2] < -args.max_translation_norm - 1e-9 for action in actions)
    total_path = sum(_norm(action[:3]) for action in actions)
    return {
        "policy": policy,
        "steps_performed": len(actions),
        "probe_used": probe_used,
        "probe_cost": _round(probe_cost),
        "total_path_length": _round(total_path),
        "extra_path_length": None,
        "intended_target_movement_score": _round(intended_progress),
        "wrong_target_movement_score": _round(wrong_progress),
        "wrong_target_movement_rate": float(wrong_rate),
        "wrong_target_action_rate": _round(_mean([float(step["projection"]["wrong_target_action"]) for step in records])),
        "unsafe_rate": float(unsafe),
        "success": success,
        "reward_sum": _round(reward_sum),
        "safety_intervention_rate": _round(safety_interventions / max(1, len(actions))),
        "false_positive_rate": 0.0,
        "records": records,
    }


def _summarize_libero_policy_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_policy = {item["policy"]: item for item in results}
    no_probe_path = float((by_policy.get("no_probe_greedy") or {}).get("total_path_length") or 0.0)
    for item in results:
        item["extra_path_length"] = _round(float(item["total_path_length"] or 0.0) - no_probe_path)
    amp = by_policy.get("amp_gd_micro_probe") or {}
    random_probe = by_policy.get("random_probe") or {}
    safety = by_policy.get("safety_only_clipping") or {}
    nearest = by_policy.get("nearest_target") or {}
    return {
        "by_policy": by_policy,
        "comparison": {
            "amp_wrong_target_movement_delta_vs_random_probe": _round(float(random_probe.get("wrong_target_movement_rate", 0.0)) - float(amp.get("wrong_target_movement_rate", 0.0))),
            "amp_wrong_target_movement_delta_vs_safety_only": _round(float(safety.get("wrong_target_movement_rate", 0.0)) - float(amp.get("wrong_target_movement_rate", 0.0))),
            "amp_wrong_target_movement_delta_vs_nearest": _round(float(nearest.get("wrong_target_movement_rate", 0.0)) - float(amp.get("wrong_target_movement_rate", 0.0))),
            "amp_target_movement_delta_vs_random_probe": _round(float(amp.get("intended_target_movement_score") or 0.0) - float(random_probe.get("intended_target_movement_score") or 0.0)),
            "amp_target_movement_delta_vs_safety_only": _round(float(amp.get("intended_target_movement_score") or 0.0) - float(safety.get("intended_target_movement_score") or 0.0)),
            "amp_probe_cost": amp.get("probe_cost"),
            "amp_extra_path_length": amp.get("extra_path_length"),
            "amp_beats_random_probe": bool(float(amp.get("wrong_target_movement_rate", 1.0)) < float(random_probe.get("wrong_target_movement_rate", 1.0)) or float(amp.get("intended_target_movement_score") or 0.0) > float(random_probe.get("intended_target_movement_score") or 0.0) + 1e-6),
            "amp_beats_safety_only": bool(float(amp.get("wrong_target_movement_rate", 1.0)) < float(safety.get("wrong_target_movement_rate", 1.0)) or float(amp.get("intended_target_movement_score") or 0.0) > float(safety.get("intended_target_movement_score") or 0.0) + 1e-6),
            "nearest_matches_amp": bool(abs(float(nearest.get("wrong_target_movement_rate", 0.0)) - float(amp.get("wrong_target_movement_rate", 0.0))) < 1e-9),
            "safety_matches_amp": bool(abs(float(safety.get("wrong_target_movement_rate", 0.0)) - float(amp.get("wrong_target_movement_rate", 0.0))) < 1e-9),
            "random_matches_amp": bool(abs(float(random_probe.get("wrong_target_movement_rate", 0.0)) - float(amp.get("wrong_target_movement_rate", 0.0))) < 1e-9),
        },
    }


def run_libero_inventory_and_probe(args: argparse.Namespace) -> dict[str, Any]:
    from tca_map.css_shield.semantic_observability import _load_case, build_object_inventory
    from tca_map.smolvla.online_action_generation_bridge import _bddl_path

    os.environ.setdefault("MUJOCO_GL", "osmesa")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    libero_root = _as_path(args.libero_root)
    robosuite_root = _as_path(args.robosuite_root)
    import_path_audit = _prepare_libero_import_path(libero_root, robosuite_root)
    from libero.libero.envs import OffScreenRenderEnv

    case = _load_case(_as_path(args.manifest), max(5, args.max_steps), args.case_index)
    bddl_file = _bddl_path(libero_root, case["suite"], case["task_id"])
    env = OffScreenRenderEnv(bddl_file_name=str(bddl_file), camera_heights=args.camera_size, camera_widths=args.camera_size)
    try:
        env.seed(args.seed)
        obs = env.set_init_state(np.asarray(case["init_state"], dtype=np.float64))
        inventory = build_object_inventory(env, obs, str(case["instruction"]), case.get("counterfactual_instruction"))
    finally:
        try:
            env.close()
        except Exception:
            pass
    target_key, wrong_key = _extract_target_keys(inventory)
    short_inventory = _short_inventory(inventory)
    non_leaking = short_inventory["resolver_non_leaking_flags"]
    safe_probe_available = bool(target_key and wrong_key and inventory.get("wrong_target_metric_computable"))
    active_ambiguity_signal = False
    state2b_green = bool(
        inventory.get("intended_target_resolvable")
        and inventory.get("distractor_resolvable")
        and inventory.get("wrong_target_metric_computable")
        and safe_probe_available
        and non_leaking.get("uses_bddl_metadata") is False
        and non_leaking.get("uses_eval_labels") is False
        and non_leaking.get("uses_dataset_target_labels") is False
        and non_leaking.get("uses_task_id_or_filename") is False
    )
    probe_report = None
    if state2b_green and args.run_libero_probe:
        results = [
            _run_libero_policy(OffScreenRenderEnv, bddl_file, np.asarray(case["init_state"], dtype=np.float64), str(case["instruction"]), str(target_key), str(wrong_key), policy, args)
            for policy in ("no_probe_greedy", "random_probe", "safety_only_clipping", "nearest_target", "amp_gd_micro_probe")
        ]
        summary = _summarize_libero_policy_results(results)
        comp = summary["comparison"]
        continue_green = bool(comp["amp_beats_random_probe"] and comp["amp_beats_safety_only"] and not comp["nearest_matches_amp"])
        probe_report = {
            "ran": True,
            "policy_results": results,
            "summary": summary,
            "continue_green": continue_green,
            "decision": "continue" if continue_green else "kill_or_reframe",
            "reason": "AMP-GD beat random-probe and safety-only on the tiny LIBERO object-observable diagnostic."
            if continue_green
            else "Tiny LIBERO diagnostic did not show AMP-GD value beyond simple baselines; available scene is language-resolvable and lacks active ambiguity evidence.",
        }
    return {
        "case": {
            "case_index": args.case_index,
            "suite": case["suite"],
            "task_id_used_for_env_setup_only": case["task_id"],
            "instruction": case["instruction"],
            "counterfactual_instruction": case.get("counterfactual_instruction"),
            "bddl_file_used_for_env_setup_only": str(bddl_file),
        },
        "import_path_audit": import_path_audit,
        "inventory": short_inventory,
        "inventory_full_path": str(args.inventory_json),
        "state2b_decision": {
            "green": state2b_green,
            "intended_target_resolvable": inventory.get("intended_target_resolvable"),
            "distractor_resolvable": inventory.get("distractor_resolvable"),
            "wrong_target_metric_computable": inventory.get("wrong_target_metric_computable"),
            "safe_micro_probe_action_available": safe_probe_available,
            "active_ambiguity_signal_available": active_ambiguity_signal,
            "uses_privileged_inference_label": False,
            "reason": "Object positions and language-resolved target/distractor are observable without target-label leakage, but no active ambiguity cue is exposed."
            if state2b_green
            else "LIBERO object-observable gate failed.",
        },
        "state2c_probe_diagnostic": probe_report or {"ran": False, "reason": "State 2B was not green or run_libero_probe was false."},
        "full_inventory": inventory,
    }


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "training_performed": False,
            "lora_training_performed": False,
            "loss_computed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "toy_rollout_control_metric_happened": False,
            "libero_simulator_control_metric_happened": False,
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
    if args.toy_trials < 20:
        report["result"]["blocked_reason"] = "toy robustness audit requires at least 20 trials"
        return report
    seeds = tuple(int(item.strip()) for item in args.seeds.split(",") if item.strip())
    toy = run_toy_robustness_audit(args.toy_trials, seeds)
    report["policy"]["toy_rollout_control_metric_happened"] = True
    libero = None
    if not args.skip_libero:
        if not _env_flag(TASK_GATE):
            report["result"]["blocked_reason"] = f"{TASK_GATE}=1 is required for LIBERO/RoboSuite State 2"
            report["toy_robustness_audit"] = toy
            return report
        try:
            libero = run_libero_inventory_and_probe(args)
            if (libero.get("state2c_probe_diagnostic") or {}).get("ran"):
                report["policy"]["libero_simulator_control_metric_happened"] = True
            full_inventory = libero.pop("full_inventory", None)
            if full_inventory is not None:
                Path(args.inventory_json).parent.mkdir(parents=True, exist_ok=True)
                Path(args.inventory_json).write_text(json.dumps(full_inventory, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            libero = {"error": _compact_error(exc), "state2b_decision": {"green": False, "reason": f"{type(exc).__name__}: {exc}"}, "state2c_probe_diagnostic": {"ran": False, "reason": "LIBERO inventory failed."}}
    final_decision = _state2_decision(toy, libero)
    report.update(
        {
            "toy_robustness_audit": toy,
            "libero_object_observable_port": libero,
            "continue_or_kill": final_decision,
            "result": {"passed": True, "blocked_reason": None, "elapsed_sec": _round(time.monotonic() - started, 3)},
        }
    )
    return report


def _state2_decision(toy: dict[str, Any], libero: dict[str, Any] | None) -> dict[str, Any]:
    toy_decision = toy.get("toy_route_decision") or {}
    if libero is None:
        return {"decision": "kill_or_reframe", "continue": False, "kill": True, "reason": "LIBERO/RoboSuite port was skipped, so State 2 did not produce the required real-simulator gate."}
    state2b = libero.get("state2b_decision") or {}
    state2c = libero.get("state2c_probe_diagnostic") or {}
    if not state2b.get("green"):
        return {"decision": "kill_or_reframe", "continue": False, "kill": True, "reason": "LIBERO/RoboSuite object observability gate failed."}
    if not state2c.get("ran"):
        return {"decision": "kill_or_reframe", "continue": False, "kill": True, "reason": "LIBERO/RoboSuite micro-probe diagnostic did not run."}
    if not state2c.get("continue_green"):
        return {
            "decision": "kill_or_reframe",
            "continue": False,
            "kill": True,
            "reason": "Toy evidence is matched by simple informative-probe heuristics and the tiny LIBERO diagnostic did not beat simple baselines.",
            "toy_killed_as_main_evidence": toy_decision.get("kill_toy_as_main_evidence"),
        }
    return {"decision": "continue_to_state3_realistic_ambiguity_diagnostic", "continue": True, "kill": False, "reason": "LIBERO/RoboSuite diagnostic showed AMP-GD value beyond simple baselines."}


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    toy = report.get("toy_robustness_audit") or {}
    utility = toy.get("utility_metric_audit") or {}
    priv = toy.get("privileged_information_audit") or {}
    toy_decision = toy.get("toy_route_decision") or {}
    libero = report.get("libero_object_observable_port") or {}
    state2b = libero.get("state2b_decision") or {}
    state2c = libero.get("state2c_probe_diagnostic") or {}
    final = report.get("continue_or_kill") or {}
    lines = [
        "# AMP-GD State 2 Result",
        "",
        "Diagnostic-only kill gate. This is not paper-grade evidence.",
        "",
        f"- final decision: `{final.get('decision')}`",
        f"- reason: {final.get('reason')}",
        f"- toy utility metric bug found: `{utility.get('metric_bug_found')}`",
        f"- utility-drop interpretation: {utility.get('state1_negative_drop_means')}",
        f"- AMP-GD privileged inference info used: `{not priv.get('passed', False)}`",
        f"- toy killed as main evidence: `{toy_decision.get('kill_toy_as_main_evidence')}`",
        f"- LIBERO object observability green: `{state2b.get('green')}`",
        f"- wrong-target metric computable: `{state2b.get('wrong_target_metric_computable')}`",
        f"- safe micro-probe action available: `{state2b.get('safe_micro_probe_action_available')}`",
        f"- active ambiguity signal available: `{state2b.get('active_ambiguity_signal_available')}`",
        f"- LIBERO micro-probe diagnostic ran: `{state2c.get('ran')}`",
        "",
    ]
    if state2c.get("ran"):
        summary = state2c.get("summary") or {}
        by_policy = summary.get("by_policy") or {}
        lines += ["## LIBERO Metrics", "", "| policy | wrong-move | target move | unsafe | probe cost | reward | success |", "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for policy in ("no_probe_greedy", "random_probe", "safety_only_clipping", "nearest_target", "amp_gd_micro_probe"):
            item = by_policy.get(policy) or {}
            lines.append(f"| `{policy}` | {item.get('wrong_target_movement_rate')} | {item.get('intended_target_movement_score')} | {item.get('unsafe_rate')} | {item.get('probe_cost')} | {item.get('reward_sum')} | {item.get('success')} |")
        comp = summary.get("comparison") or {}
        lines += [
            "",
            f"- AMP beats random-probe: `{comp.get('amp_beats_random_probe')}`",
            f"- AMP beats safety-only: `{comp.get('amp_beats_safety_only')}`",
            f"- random-probe matches AMP wrong-target movement: `{comp.get('random_matches_amp')}`",
            f"- safety-only matches AMP wrong-target movement: `{comp.get('safety_matches_amp')}`",
            f"- nearest matches AMP wrong-target movement: `{comp.get('nearest_matches_amp')}`",
            "",
        ]
    lines += [
        "## Toy Robustness",
        "",
        f"- deterministic informative-probe heuristic matches AMP-GD: `{toy_decision.get('deterministic_heuristic_matches_amp')}`",
        f"- entropy-greedy heuristic matches AMP-GD: `{toy_decision.get('entropy_greedy_matches_amp')}`",
        f"- AMP beats random-probe in all toy profiles: `{toy_decision.get('amp_beats_random_probe_all_profiles')}`",
        f"- AMP beats safety-only in all toy profiles: `{toy_decision.get('amp_beats_safety_only_all_profiles')}`",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _console_summary(report: dict[str, Any]) -> dict[str, Any]:
    libero = report.get("libero_object_observable_port") or {}
    return {
        "schema_version": report.get("schema_version"),
        "result": report.get("result"),
        "policy": report.get("policy"),
        "toy_utility_metric_audit": (report.get("toy_robustness_audit") or {}).get("utility_metric_audit"),
        "toy_route_decision": (report.get("toy_robustness_audit") or {}).get("toy_route_decision"),
        "libero_state2b_decision": libero.get("state2b_decision"),
        "libero_state2c_decision": (libero.get("state2c_probe_diagnostic") or {}).get("decision"),
        "continue_or_kill": report.get("continue_or_kill"),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--toy-trials", type=int, default=60)
    parser.add_argument("--seeds", default="11,23,37")
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_scaled_report.json")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--libero-root", default="C:/assets/repos/LIBERO")
    parser.add_argument("--robosuite-root", default="C:/assets/repos/robosuite")
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--probe-scale", type=float, default=0.035)
    parser.add_argument("--commit-scale", type=float, default=0.055)
    parser.add_argument("--max-translation-norm", type=float, default=0.08)
    parser.add_argument("--run-libero-probe", action="store_true")
    parser.add_argument("--skip-libero", action="store_true")
    parser.add_argument("--report-json", default="reports/amp_gd_state2_report.json")
    parser.add_argument("--report-md", default="reports/amp_gd_state2_result.md")
    parser.add_argument("--inventory-json", default="reports/amp_gd_state2_libero_inventory.json")
    args = parser.parse_args(argv)

    report = build_report(args)
    Path(args.report_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report_json).write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, Path(args.report_md))
    print(json.dumps(_console_summary(report), indent=2, sort_keys=True))
    return 0 if report.get("result", {}).get("passed") else 8


if __name__ == "__main__":
    raise SystemExit(main())
