"""PhaseBarrier-VLA first implemented prototype.

This is the implementation-heavy continuation after the literature-only
campaign stop was rejected.  It trains a tiny phase-conditioned feasibility
field from short exact-state simulator interventions, then evaluates a
postprocessed-action shaping wrapper around frozen official SmolVLA-LIBERO.

The method is not candidate ranking, confidence scoring, or replanning.  Every
variant emits exactly one continuous action per policy step.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_echo_vla_first_prototype import (  # noqa: E402
    _compact_for_hash,
    _effect_from_observations,
    _extract_nested_eef,
    _postprocess_action,
    _preprocess_batch,
    _sim_body_positions,
    _sim_state_hash,
)
from tca_map.smolvla.echo_vla import compatibility_score, stable_hash  # noqa: E402
from tca_map.smolvla.exact_hard_slice_rollout import _make_exact_vector_env  # noqa: E402
from tca_map.smolvla.official_closed_loop_scaleup import _json_default, _set_runtime_env  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import (  # noqa: E402
    POLICIES,
    _cuda_memory,
    _load_policy_and_processors,
)
from tca_map.smolvla.phase_barrier_vla import (  # noqa: E402
    BarrierRecord,
    PhaseBarrierModel,
    action_feature_dict,
    fit_phase_barrier,
    infer_phase_from_step,
    pre_vla_style_halt_proxy,
    project_action_with_barrier,
    simple_global_damping,
)


DATE_KST = "2026-07-11"
BRANCH = "codex/autonomous-ral-research-implementation-v2"
TASKS = [
    {
        "suite": "libero_spatial",
        "task_id": 4,
        "role": "stable_grasp_contact_transition",
        "instruction": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    },
    {
        "suite": "libero_10",
        "task_id": 4,
        "role": "long_horizon_contact_and_release",
        "instruction": "put the white mug on the left plate and put the yellow and white mug on the right plate",
    },
]
RESET_IDENTITIES = [20260711 + index for index in range(20)]
VARIANTS = [
    "frozen_smolvla",
    "pre_vla_style_halt_proxy",
    "simple_global_damping",
    "phase_barrier_no_phase_ablation",
    "phase_barrier_full",
]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _json_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _round(value: float | int | np.floating[Any] | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _identity_to_initial_state_index(identity: int) -> int:
    if int(identity) not in RESET_IDENTITIES:
        raise ValueError(f"unknown reset identity {identity}")
    return RESET_IDENTITIES.index(int(identity))


def _state_flat(env: Any) -> np.ndarray:
    raw_env = env.envs[0] if hasattr(env, "envs") else env
    sim = _find_attr_chain(raw_env, "sim")
    if sim is None or not hasattr(sim, "get_state"):
        raise RuntimeError("could not locate simulator get_state")
    return np.asarray(sim.get_state().flatten(), dtype=np.float64).copy()


def _find_attr_chain(root: Any, attr: str, max_depth: int = 5) -> Any | None:
    seen: set[int] = set()
    queue: list[tuple[Any, int]] = [(root, 0)]
    while queue:
        item, depth = queue.pop(0)
        if id(item) in seen:
            continue
        seen.add(id(item))
        if hasattr(item, attr):
            return getattr(item, attr)
        if depth >= max_depth:
            continue
        for child_name in ("env", "unwrapped", "base_env", "_env", "envs"):
            try:
                child = getattr(item, child_name)
            except Exception:
                continue
            if isinstance(child, list):
                queue.extend((entry, depth + 1) for entry in child)
            elif child is not None:
                queue.append((child, depth + 1))
    return None


def _vectorize_observation(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _vectorize_observation(item) for key, item in value.items()}
    if isinstance(value, np.ndarray):
        return value[None, ...]
    return value


def _restore_observation_from_flat_state(env: Any, state_flat: np.ndarray) -> Any:
    raw_env = env.envs[0] if hasattr(env, "envs") else env
    inner = getattr(raw_env, "_env", None)
    if inner is not None and hasattr(inner, "regenerate_obs_from_state"):
        raw_obs = inner.regenerate_obs_from_state(np.asarray(state_flat, dtype=np.float64))
        return _vectorize_observation(raw_env._format_raw_obs(raw_obs))
    sim = _find_attr_chain(raw_env, "sim")
    if sim is None or not hasattr(sim, "set_state_from_flattened"):
        raise RuntimeError("could not locate simulator state restoration API")
    sim.set_state_from_flattened(np.asarray(state_flat, dtype=np.float64))
    sim.forward()
    raise RuntimeError("state restored but observation regeneration API was unavailable")


def _policy_action(policy: Any, env: Any, observation: Any, loaded: Mapping[str, Any]) -> np.ndarray:
    import torch

    batch = _preprocess_batch(env, observation, dict(loaded))
    with torch.inference_mode():
        action = policy.select_action(batch)
    return _postprocess_action(action, dict(loaded)).reshape(1, -1)


def _step_success(info: Mapping[str, Any]) -> bool:
    for key in ("final_info", "is_success"):
        if key not in info:
            continue
        value = info[key]
        try:
            if isinstance(value, Mapping) and "is_success" in value:
                return bool(np.asarray(value["is_success"]).reshape(-1)[0])
            return bool(np.asarray(value).reshape(-1)[0])
        except Exception:
            continue
    return False


def _phase_score_label(effect: Mapping[str, float], phase: str, task_success: bool) -> float:
    score_phase = "grasp_contact" if str(phase) == "contact" else str(phase)
    score = compatibility_score(effect, score_phase)
    return 1.0 if bool(task_success) or score > 0.03 else -1.0


def _candidate_training_actions(default_action: np.ndarray) -> list[dict[str, Any]]:
    base = np.asarray(default_action, dtype=np.float64).reshape(1, -1)
    candidates = [
        {"name": "default", "action": base},
        {"name": "global_damping_0p70", "action": simple_global_damping(base, scale=0.70)},
        {"name": "translation_scale_1p35", "action": np.clip(base * np.asarray([[1.35, 1.35, 1.0, 1.0, 1.0, 1.0, 1.0]]), -1.0, 1.0)},
        {"name": "contact_z_boost", "action": np.clip(base + np.asarray([[0.0, 0.0, 0.025, 0.0, 0.0, 0.0, 0.0]]), -1.0, 1.0)},
    ]
    return candidates


def _capture_training_states(
    *,
    task: Mapping[str, Any],
    identity: int,
    loaded: Mapping[str, Any],
    fractions: list[float],
) -> list[dict[str, Any]]:
    env = None
    try:
        env = _make_exact_vector_env(str(task["suite"]), int(task["task_id"]), _identity_to_initial_state_index(identity))
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        observation, _ = env.reset(seed=[int(identity)])
        max_steps = int(env.call("_max_episode_steps")[0])
        wanted_steps = sorted({int(round(float(frac) * max(1, max_steps - 1))) for frac in fractions})
        snapshots = []
        for step in range(max_steps):
            if step in wanted_steps:
                flat = _state_flat(env)
                snapshots.append(
                    {
                        "suite": str(task["suite"]),
                        "task_id": int(task["task_id"]),
                        "instruction": str(task["instruction"]),
                        "identity": int(identity),
                        "initial_state_index": _identity_to_initial_state_index(identity),
                        "step": int(step),
                        "max_steps": int(max_steps),
                        "phase": infer_phase_from_step(step, max_steps),
                        "state_flat": flat,
                        "state_hash": stable_hash(flat),
                    }
                )
            action = _policy_action(policy, env, observation, loaded)
            observation, _reward, terminated, truncated, info = env.step(action)
            if np.all(terminated | truncated) or _step_success(info):
                break
        return snapshots
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _evaluate_short_candidate(
    *,
    state: Mapping[str, Any],
    action: np.ndarray,
    candidate_name: str,
    horizon: int,
) -> dict[str, Any]:
    env = None
    try:
        env = _make_exact_vector_env(str(state["suite"]), int(state["task_id"]), int(state["initial_state_index"]))
        env.reset(seed=[int(state["identity"])])
        observation = _restore_observation_from_flat_state(env, np.asarray(state["state_flat"], dtype=np.float64))
        start_hash = _sim_state_hash(env)
        start_body_positions = _sim_body_positions(env)
        start_observation = observation
        success = False
        terminated_last = False
        truncated_last = False
        repeated = np.repeat(np.asarray(action, dtype=np.float64).reshape(1, -1), int(horizon), axis=0)
        for row in repeated:
            observation, _reward, terminated, truncated, info = env.step(row.reshape(1, -1))
            success = bool(success or _step_success(info))
            terminated_last = bool(np.asarray(terminated).reshape(-1)[0])
            truncated_last = bool(np.asarray(truncated).reshape(-1)[0])
            if np.all(terminated | truncated):
                break
        final_body_positions = _sim_body_positions(env)
        effect = _effect_from_observations(
            start_observation,
            observation,
            repeated,
            str(state["instruction"]),
            success,
            start_body_positions,
            final_body_positions,
        )
        phase = str(state["phase"])
        eef = _extract_nested_eef(start_observation)
        features = action_feature_dict(action, eef=eef, step_fraction=float(state["step"]) / max(1.0, float(state["max_steps"])))
        label = _phase_score_label(effect, phase, success)
        return {
            "state_id": f"{state['suite']}|task_{state['task_id']}|identity_{state['identity']}|step_{state['step']}",
            "candidate_name": candidate_name,
            "phase": phase,
            "features": features,
            "label": label,
            "effect": effect,
            "effect_compatibility": _round(
                compatibility_score(effect, "grasp_contact" if phase == "contact" else phase),
                6,
            ),
            "task_success_within_short_horizon": bool(success),
            "terminated": terminated_last,
            "truncated": truncated_last,
            "start_state_hash": start_hash,
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _train_barriers(args: argparse.Namespace, loaded: Mapping[str, Any]) -> dict[str, Any]:
    states = []
    train_tasks = TASKS[: int(args.max_tasks)]
    train_identities = RESET_IDENTITIES[: int(args.train_identities)]
    fractions = [float(value) for value in str(args.train_fractions).split(",") if value.strip()]
    for task in train_tasks:
        for identity in train_identities:
            states.extend(_capture_training_states(task=task, identity=identity, loaded=loaded, fractions=fractions))
    rows = []
    for state in states:
        env = None
        try:
            env = _make_exact_vector_env(str(state["suite"]), int(state["task_id"]), int(state["initial_state_index"]))
            env.reset(seed=[int(state["identity"])])
            observation = _restore_observation_from_flat_state(env, np.asarray(state["state_flat"], dtype=np.float64))
            default_action = _policy_action(loaded["policy"], env, observation, loaded)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        for candidate in _candidate_training_actions(default_action):
            rows.append(
                _evaluate_short_candidate(
                    state=state,
                    action=np.asarray(candidate["action"], dtype=np.float64),
                    candidate_name=str(candidate["name"]),
                    horizon=int(args.short_horizon),
                )
            )
    records = [
        BarrierRecord(
            phase=str(row["phase"]),
            features={key: float(value) for key, value in row["features"].items()},
            label=float(row["label"]),
        )
        for row in rows
    ]
    phase_model = fit_phase_barrier(records, use_phase=True, l2=float(args.l2))
    no_phase_model = fit_phase_barrier(records, use_phase=False, l2=float(args.l2))
    return {
        "train_tasks": train_tasks,
        "train_identities": train_identities,
        "training_state_count": len(states),
        "training_record_count": len(rows),
        "positive_label_count": int(sum(1 for row in rows if float(row["label"]) > 0.0)),
        "negative_label_count": int(sum(1 for row in rows if float(row["label"]) <= 0.0)),
        "rows": rows,
        "phase_model": phase_model.to_json(),
        "no_phase_model": no_phase_model.to_json(),
    }


def _transform_action(
    *,
    variant: str,
    action: np.ndarray,
    phase: str,
    features: Mapping[str, float],
    phase_model: Any,
    no_phase_model: Any,
    args: argparse.Namespace,
) -> tuple[np.ndarray, dict[str, Any]]:
    if variant == "frozen_smolvla":
        return np.asarray(action, dtype=np.float64), {"margin": None, "transform": "none"}
    if variant == "simple_global_damping":
        return simple_global_damping(action, scale=float(args.damping_scale)), {"margin": None, "transform": "global_damping"}
    if variant == "pre_vla_style_halt_proxy":
        margin = phase_model.score(features, phase)
        return pre_vla_style_halt_proxy(action, margin=margin, threshold=float(args.margin_threshold)), {
            "margin": _round(margin, 6),
            "transform": "halt_if_negative_margin",
        }
    if variant == "phase_barrier_no_phase_ablation":
        margin = no_phase_model.score(features, phase)
        return project_action_with_barrier(
            action,
            margin=margin,
            phase=phase,
            threshold=float(args.margin_threshold),
            strength=float(args.projection_strength),
        ), {"margin": _round(margin, 6), "transform": "no_phase_barrier_projection"}
    if variant == "phase_barrier_full":
        margin = phase_model.score(features, phase)
        return project_action_with_barrier(
            action,
            margin=margin,
            phase=phase,
            threshold=float(args.margin_threshold),
            strength=float(args.projection_strength),
        ), {"margin": _round(margin, 6), "transform": "phase_conditioned_barrier_projection"}
    raise ValueError(f"unknown variant {variant}")


def _run_episode(
    *,
    task: Mapping[str, Any],
    identity: int,
    variant: str,
    loaded: Mapping[str, Any],
    phase_model: Any,
    no_phase_model: Any,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    env = None
    started = time.monotonic()
    try:
        env = _make_exact_vector_env(str(task["suite"]), int(task["task_id"]), _identity_to_initial_state_index(identity))
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        observation, _ = env.reset(seed=[int(identity)])
        max_steps = int(env.call("_max_episode_steps")[0])
        if int(args.max_eval_steps) > 0:
            max_steps = min(max_steps, int(args.max_eval_steps))
        success = False
        rewards = []
        margin_values = []
        shaped_steps = 0
        action_delta_norms = []
        for step in range(max_steps):
            base_action = _policy_action(policy, env, observation, loaded)
            phase = infer_phase_from_step(step, max_steps)
            eef = _extract_nested_eef(observation)
            features = action_feature_dict(base_action, eef=eef, step_fraction=float(step) / max(1.0, float(max_steps)))
            action, transform = _transform_action(
                variant=variant,
                action=base_action,
                phase=phase,
                features=features,
                phase_model=phase_model,
                no_phase_model=no_phase_model,
                args=args,
            )
            if transform.get("margin") is not None:
                margin_values.append(float(transform["margin"]))
            delta = float(np.linalg.norm(np.asarray(action) - np.asarray(base_action)))
            if delta > 1e-9:
                shaped_steps += 1
            action_delta_norms.append(delta)
            observation, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
            rewards.append(float(np.asarray(reward).reshape(-1)[0]))
            success = bool(success or _step_success(info))
            if success or np.all(terminated | truncated):
                break
        return {
            "variant": variant,
            "suite": str(task["suite"]),
            "task_id": int(task["task_id"]),
            "task_key": f"{task['suite']}/task_{task['task_id']}",
            "role": str(task["role"]),
            "reset_identity": int(identity),
            "initial_state_index": _identity_to_initial_state_index(identity),
            "success": bool(success),
            "reward_sum": _round(float(np.sum(rewards)) if rewards else 0.0, 6),
            "episode_steps": int(step + 1 if "step" in locals() else 0),
            "shaped_step_count": int(shaped_steps),
            "mean_action_delta_norm": _round(float(np.mean(action_delta_norms)) if action_delta_norms else 0.0, 6),
            "mean_margin": _round(float(np.mean(margin_values)) if margin_values else None, 6),
            "min_margin": _round(float(np.min(margin_values)) if margin_values else None, 6),
            "max_margin": _round(float(np.max(margin_values)) if margin_values else None, 6),
            "elapsed_seconds": _round(time.monotonic() - started, 3),
            "cuda_memory": _cuda_memory(torch),
            "exception": None,
        }
    except Exception as exc:  # pragma: no cover - simulator boundary
        return {
            "variant": variant,
            "suite": str(task["suite"]),
            "task_id": int(task["task_id"]),
            "task_key": f"{task['suite']}/task_{task['task_id']}",
            "role": str(task["role"]),
            "reset_identity": int(identity),
            "success": False,
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc().splitlines()[-40:],
            },
            "elapsed_seconds": _round(time.monotonic() - started, 3),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _summarize(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    by_variant = defaultdict(list)
    by_variant_task = defaultdict(list)
    for row in episodes:
        by_variant[row["variant"]].append(row)
        by_variant_task[(row["variant"], row["task_key"])].append(row)
    summary = {}
    for variant, rows in sorted(by_variant.items()):
        successes = sum(1 for row in rows if row.get("success"))
        task_rates = []
        for (v, _task), task_rows in by_variant_task.items():
            if v != variant:
                continue
            task_rates.append(sum(1 for row in task_rows if row.get("success")) / max(1, len(task_rows)))
        summary[variant] = {
            "successes": int(successes),
            "total": int(len(rows)),
            "success_rate": _round(successes / max(1, len(rows)), 6),
            "task_balanced_success_rate": _round(float(np.mean(task_rates)) if task_rates else 0.0, 6),
            "mean_shaped_steps": _round(float(np.mean([row.get("shaped_step_count", 0) for row in rows])), 3),
            "mean_action_delta_norm": _round(float(np.mean([row.get("mean_action_delta_norm", 0.0) or 0.0 for row in rows])), 6),
            "exceptions": int(sum(1 for row in rows if row.get("exception"))),
        }
    baselines = ["frozen_smolvla", "pre_vla_style_halt_proxy", "simple_global_damping"]
    strongest_baseline = max(baselines, key=lambda name: summary.get(name, {}).get("task_balanced_success_rate", -1.0))
    full = summary.get("phase_barrier_full", {}).get("task_balanced_success_rate", 0.0)
    strongest = summary.get(strongest_baseline, {}).get("task_balanced_success_rate", 0.0)
    ablation = summary.get("phase_barrier_no_phase_ablation", {}).get("task_balanced_success_rate", 0.0)
    absolute_gain_pp = 100.0 * (float(full) - float(strongest))
    failure_rate_baseline = 1.0 - float(strongest)
    failure_rate_full = 1.0 - float(full)
    relative_failure_reduction = 0.0 if failure_rate_baseline <= 0 else (failure_rate_baseline - failure_rate_full) / failure_rate_baseline
    go = absolute_gain_pp >= 5.0 and float(full) > float(ablation)
    route_b = (
        float(full) > float(strongest)
        and float(full) > float(ablation)
        and relative_failure_reduction >= 0.10
    )
    return {
        "by_variant": summary,
        "strongest_non_ablation_baseline": strongest_baseline,
        "full_task_balanced_success_rate": _round(float(full), 6),
        "strongest_baseline_task_balanced_success_rate": _round(float(strongest), 6),
        "ablation_task_balanced_success_rate": _round(float(ablation), 6),
        "absolute_gain_over_strongest_baseline_pp": _round(absolute_gain_pp, 3),
        "relative_failure_rate_reduction": _round(relative_failure_reduction, 6),
        "route_a_go": bool(go),
        "route_b_go": bool(route_b),
        "passes_prototype_go": bool(go or route_b),
    }


def _write_protocol(report_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# PhaseBarrier-VLA Prototype Protocol",
        "",
        f"Date: {args.run_date_kst} KST",
        "",
        "## Method",
        "",
        "PhaseBarrier-VLA trains a phase-conditioned linear feasibility margin from short exact-state simulator interventions. At deployment it receives only current observation-derived proprio features, phase inferred from episode fraction, and the current postprocessed SmolVLA action. It reshapes the action continuously; it does not rank candidates, query future success, or replan.",
        "",
        "## Fixed Split",
        "",
        f"- tasks: `{[(item['suite'], item['task_id']) for item in TASKS[: int(args.max_tasks)]]}`",
        f"- training identities: `{RESET_IDENTITIES[: int(args.train_identities)]}`",
        f"- eval identities: `{RESET_IDENTITIES[int(args.train_identities): int(args.train_identities) + int(args.eval_identities)]}`",
        f"- loaded training/result JSON: `{args.load_train_json or None}`",
        f"- training state fractions: `{args.train_fractions}`",
        f"- short intervention horizon: `{args.short_horizon}`",
        f"- max eval steps override: `{args.max_eval_steps}` (`0` means official max)",
        "",
        "## Variants",
        "",
        "1. `frozen_smolvla`",
        "2. `pre_vla_style_halt_proxy`",
        "3. `simple_global_damping`",
        "4. `phase_barrier_no_phase_ablation`",
        "5. `phase_barrier_full`",
        "",
        "## GO/KILL",
        "",
        "- Route A: full method improves task-balanced success by at least 5 absolute percentage points over the strongest non-ablation baseline and beats the no-phase ablation.",
        "- Route B: full method beats strongest baseline and ablation, and relative failure rate decreases by at least 10%.",
        "- Kill: full method fails both routes, simple baseline matches/beats it, ablation matches/beats it, or infrastructure/runtime invalidates measurement.",
    ]
    _write_md(report_dir / f"{args.output_stem}_protocol.md", lines)


def _write_overlap_and_reclassification(report_dir: Path) -> None:
    _write_md(
        report_dir / "implementation_v2_reclassification.md",
        [
            "# Implementation V2 Reclassification",
            "",
            f"Date: {DATE_KST} KST",
            "",
            "Previous decision `NO_METHOD_AFTER_3_VALID_CYCLES` is reclassified as `PREMATURE_LITERATURE_ONLY_TERMINATION`.",
            "",
            "The prior reports are preserved as literature triage and hostile-review evidence. They do not count as valid method cycles because no method was implemented or tested in a decisive prototype.",
        ],
    )
    _write_md(
        report_dir / "phase_barrier_vla_exact_overlap_matrix.md",
        [
            "# Exact Overlap Matrix For Prior Literature-Only Kills",
            "",
            f"Date: {DATE_KST} KST",
            "",
            "Legend: `same`, `partial`, `different`, `unavailable`.",
            "",
            "## Action Conditioning Route",
            "",
            "| Closest paper | Problem | Inputs | Representation | Supervision | Objective | Modified component | Inference intervention | Claim | Exact duplicate? |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "| CAC-VLA | partial | partial | partial | partial | different | partial | different | partial | no |",
            "| ActionMap | partial | partial | partial | different | different | partial | different | partial | no |",
            "| AEM/LAWM/LARA | partial | partial | partial | partial | partial | different | different | partial | no |",
            "",
            "Result: prior kill was too broad, but local ECHO and local ActionMap evidence still make this route lower priority for immediate implementation.",
            "",
            "## Censored Correction Route",
            "",
            "| Closest paper | Problem | Inputs | Representation | Supervision | Objective | Modified component | Inference intervention | Claim | Exact duplicate? |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "| TORL-VLA | partial | different | partial | different | partial | different | different | partial | no |",
            "| SDP | partial | partial | different | partial | partial | different | partial | partial | no |",
            "| VLA-Corrector | partial | partial | different | different | different | different | partial | partial | no |",
            "",
            "Result: not an exact duplicate, but the strongest version requires intervention or correction data not available locally. Kept as second-cycle candidate if PhaseBarrier fails.",
            "",
            "## Contact Barrier Route",
            "",
            "| Closest paper | Problem | Inputs | Representation | Supervision | Objective | Modified component | Inference intervention | Claim | Exact duplicate? |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            "| VeriSpace | partial | partial | different | partial | different | different | different | partial | no |",
            "| Pre-VLA | partial | partial | different | partial | different | different | partial | partial | no |",
            "| VLA-Corrector | partial | partial | different | different | different | different | different | partial | no |",
            "| SEAM/AAC/Legato | partial | partial | different | different | different | partial | partial | partial | no |",
            "",
            "Result: a technically distinct survivor exists: phase-conditioned feasibility-field action projection. It changes the physical/control representation and the action-generation distribution without candidate ranking or generic correction.",
        ],
    )


def run_prototype(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    report_dir = Path(args.report_dir)
    _write_overlap_and_reclassification(report_dir)
    _write_protocol(report_dir, args)
    report: dict[str, Any] = {
        "schema_version": "phase_barrier_vla_prototype_v1",
        "date_kst": str(args.run_date_kst),
        "branch": str(args.run_branch),
        "previous_decision_reclassified_as": "PREMATURE_LITERATURE_ONLY_TERMINATION",
        "method": "PhaseBarrier-VLA",
        "training_happened": False,
        "closed_loop_experiment_happened": False,
        "tasks": TASKS[: int(args.max_tasks)],
        "variants": list(VARIANTS),
        "config": {
            "max_tasks": int(args.max_tasks),
            "train_identities": int(args.train_identities),
            "eval_identities": int(args.eval_identities),
            "train_fractions": str(args.train_fractions),
            "short_horizon": int(args.short_horizon),
            "max_eval_steps": int(args.max_eval_steps),
            "l2": float(args.l2),
            "damping_scale": float(args.damping_scale),
            "margin_threshold": float(args.margin_threshold),
            "projection_strength": float(args.projection_strength),
            "load_train_json": str(args.load_train_json) if args.load_train_json else None,
            "output_stem": str(args.output_stem),
        },
        "train": {},
        "episodes": [],
        "summary": {},
        "errors": [],
        "final_decision": "PHASE_BARRIER_MEASUREMENT_INVALID",
        "exact_next_step": None,
    }
    loaded = None
    try:
        _set_runtime_env(args)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for official SmolVLA PhaseBarrier prototype")
        spec = next(item for item in POLICIES if item.name == "frozen_base")
        loaded = _load_policy_and_processors(args, spec)
        report["policy_load_audit"] = loaded["audit"]
        if args.load_train_json:
            source_path = Path(args.load_train_json)
            source_report = json.loads(source_path.read_text(encoding="utf-8"))
            train = source_report["train"]
            report["training_happened"] = False
            report["training_reused_from"] = str(source_path)
            report["training_checkpoint_identity"] = {
                "source_final_decision": source_report.get("final_decision"),
                "source_training_record_count": train.get("training_record_count"),
                "phase_model_sha256": _json_sha256(train.get("phase_model")),
                "no_phase_model_sha256": _json_sha256(train.get("no_phase_model")),
            }
        else:
            train = _train_barriers(args, loaded)
            report["training_happened"] = True
        report["train"] = train
        if args.load_train_json and train.get("phase_model") and train.get("no_phase_model"):
            phase_model = PhaseBarrierModel.from_json(train["phase_model"])
            no_phase_model = PhaseBarrierModel.from_json(train["no_phase_model"])
        else:
            phase_model = fit_phase_barrier(
                [
                    BarrierRecord(str(row["phase"]), {key: float(value) for key, value in row["features"].items()}, float(row["label"]))
                    for row in train["rows"]
                ],
                use_phase=True,
                l2=float(args.l2),
            )
            no_phase_model = fit_phase_barrier(
                [
                    BarrierRecord(str(row["phase"]), {key: float(value) for key, value in row["features"].items()}, float(row["label"]))
                    for row in train["rows"]
                ],
                use_phase=False,
                l2=float(args.l2),
            )
        eval_tasks = TASKS[: int(args.max_tasks)]
        eval_start = int(args.train_identities)
        eval_identities = RESET_IDENTITIES[eval_start : eval_start + int(args.eval_identities)]
        report["eval_manifest"] = {
            "tasks": eval_tasks,
            "eval_identities": eval_identities,
            "planned_episodes": len(eval_tasks) * len(eval_identities) * len(VARIANTS),
        }
        episodes = []
        for variant in VARIANTS:
            for task in eval_tasks:
                for identity in eval_identities:
                    episodes.append(
                        _run_episode(
                            task=task,
                            identity=identity,
                            variant=variant,
                            loaded=loaded,
                            phase_model=phase_model,
                            no_phase_model=no_phase_model,
                            args=args,
                        )
                    )
        report["episodes"] = episodes
        report["closed_loop_experiment_happened"] = True
        report["summary"] = _summarize(episodes)
        measurement_valid = bool(episodes) and not any(row.get("exception") for row in episodes)
        if not measurement_valid:
            report["final_decision"] = "PHASE_BARRIER_MEASUREMENT_INVALID"
            report["exact_next_step"] = "Fix measurement/runtime errors and rerun the frozen PhaseBarrier protocol without changing thresholds."
        elif report["summary"]["passes_prototype_go"]:
            report["final_decision"] = "PHASE_BARRIER_PROTOTYPE_GO"
            report["exact_next_step"] = "Scale PhaseBarrier-VLA with larger rollout count, confidence intervals, second backbone, and second condition."
        else:
            report["final_decision"] = "PHASE_BARRIER_VALID_KILL"
            report["exact_next_step"] = "Archive PhaseBarrier-VLA as first implemented kill and select a genuinely different second implemented method cycle."
    except Exception as exc:  # pragma: no cover - runtime boundary
        report["errors"].append({"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc().splitlines()[-80:]})
        report["final_decision"] = "PHASE_BARRIER_MEASUREMENT_INVALID"
        report["exact_next_step"] = "Resolve the runtime blocker, then rerun the same frozen protocol."
    finally:
        try:
            del loaded
            if "torch" in sys.modules:
                torch.cuda.empty_cache()
        except Exception:
            pass
        report["elapsed_seconds"] = _round(time.monotonic() - started, 3)
        try:
            report["latency_vram_summary"] = {"elapsed_seconds": report["elapsed_seconds"], "cuda_memory": _cuda_memory(torch)}
        except Exception:
            report["latency_vram_summary"] = {"elapsed_seconds": report["elapsed_seconds"]}
        _write_json(report_dir / f"{args.output_stem}_result.json", report)
        summary = report.get("summary") or {}
        _write_md(
            report_dir / f"{args.output_stem}_result.md",
            [
                "# PhaseBarrier-VLA Prototype Result",
                "",
                f"Final decision: `{report.get('final_decision')}`",
                "",
                f"- training happened: `{report.get('training_happened')}`",
                f"- closed-loop experiment happened: `{report.get('closed_loop_experiment_happened')}`",
                f"- variants: `{report.get('variants')}`",
                f"- training records: `{(report.get('train') or {}).get('training_record_count')}`",
                f"- eval manifest: `{report.get('eval_manifest')}`",
                f"- summary: `{summary}`",
                f"- latency/VRAM: `{report.get('latency_vram_summary')}`",
                "",
                "## Exact Next Step",
                "",
                str(report.get("exact_next_step")),
            ],
        )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default="/home/jiheon/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/home/jiheon/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--output-stem", default="phase_barrier_vla_prototype")
    parser.add_argument("--run-branch", default=BRANCH)
    parser.add_argument("--run-date-kst", default=DATE_KST)
    parser.add_argument("--load-train-json", default=None)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--train-identities", type=int, default=1)
    parser.add_argument("--eval-identities", type=int, default=1)
    parser.add_argument("--train-fractions", default="0.0,0.35,0.65")
    parser.add_argument("--short-horizon", type=int, default=4)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--damping-scale", type=float, default=0.80)
    parser.add_argument("--margin-threshold", type=float, default=0.0)
    parser.add_argument("--projection-strength", type=float, default=0.35)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.max_tasks) < 1 or int(args.max_tasks) > len(TASKS):
        raise SystemExit("--max-tasks must be between 1 and 2")
    if int(args.train_identities) < 1 or int(args.train_identities) >= len(RESET_IDENTITIES):
        raise SystemExit(f"--train-identities must be between 1 and {len(RESET_IDENTITIES) - 1}")
    if int(args.eval_identities) < 1 or int(args.train_identities) + int(args.eval_identities) > len(RESET_IDENTITIES):
        raise SystemExit("--eval-identities exceeds available held-out reset identities")
    report = run_prototype(args)
    print(
        json.dumps(
            {
                "final_decision": report.get("final_decision"),
                "summary": report.get("summary"),
                "errors": report.get("errors"),
            },
            indent=2,
            sort_keys=True,
            default=_json_default,
        )
    )
    return 0 if report.get("final_decision") != "PHASE_BARRIER_MEASUREMENT_INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
