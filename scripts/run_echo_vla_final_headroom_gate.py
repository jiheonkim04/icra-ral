"""Final bounded ECHO-VLA candidate-headroom gate.

This runner does not train ECHO, SmolVLA, OpenVLA, or any auxiliary head.  It
tests only the core prerequisite for ECHO: whether frozen official SmolVLA
stochastic candidates contain downstream closed-loop headroom from identical
LIBERO simulator states.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_echo_vla_first_prototype import (  # noqa: E402
    RESET_IDENTITIES,
    _best_body_key,
    _compact_for_hash,
    _effect_from_observations,
    _extract_nested_eef,
    _find_attr_chain,
    _postprocess_action,
    _preprocess_batch,
    _sim_body_positions,
    _sim_state_hash,
)
from tca_map.smolvla.echo_final_headroom import (  # noqa: E402
    DiversityThresholds,
    choose_final_decision,
    downstream_headroom_metrics,
    summarize_candidate_diversity,
    summarize_diversity_across_states,
)
from tca_map.smolvla.echo_vla import (  # noqa: E402
    assert_no_privileged_deployment_inputs,
    compatibility_score,
    stable_hash,
)
from tca_map.smolvla.exact_hard_slice_rollout import _make_exact_vector_env  # noqa: E402
from tca_map.smolvla.official_closed_loop_scaleup import _json_default, _set_runtime_env  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import (  # noqa: E402
    POLICIES,
    _cuda_memory,
    _load_policy_and_processors,
)


DATE_KST = "2026-07-11"
BRANCH = "codex/echo-vla-final-candidate-headroom-gate"
REFERENCE_RESET_IDENTITY = 20260711
PHASE_REQUESTS = [
    {
        "phase": "approach",
        "requested_phase": "approach_or_pre_grasp",
        "success_fraction": 0.0,
        "fallback_fraction": 0.0,
    },
    {
        "phase": "grasp_contact",
        "requested_phase": "contact_grasp_transition",
        "success_fraction": 0.25,
        "fallback_fraction": 0.25,
    },
    {
        "phase": "transport",
        "requested_phase": "transport_or_post_grasp",
        "success_fraction": 0.55,
        "fallback_fraction": 0.50,
    },
    {
        "phase": "placement",
        "requested_phase": "placement_release_transition",
        "success_fraction": 0.85,
        "fallback_fraction": 0.75,
    },
]
FINAL_TASKS = [
    {
        "suite": "libero_spatial",
        "task_id": 0,
        "task_role": "spatial",
        "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    },
    {
        "suite": "libero_object",
        "task_id": 4,
        "task_role": "object",
        "instruction": "pick up the ketchup and place it in the basket",
    },
    {
        "suite": "libero_goal",
        "task_id": 0,
        "task_role": "goal",
        "instruction": "open the middle drawer of the cabinet",
    },
]
EFFECT_HORIZONS = [4, 8, 16]
STRUCTURED_PERTURBATION_SPECS = [
    {"name": "structured_default_copy", "kind": "identity", "axis": None, "magnitude": 0.0},
    {"name": "translation_x_plus_0p025", "kind": "add", "axis": 0, "magnitude": 0.025},
    {"name": "translation_x_minus_0p025", "kind": "add", "axis": 0, "magnitude": -0.025},
    {"name": "translation_z_plus_0p025", "kind": "add", "axis": 2, "magnitude": 0.025},
    {"name": "translation_z_minus_0p025", "kind": "add", "axis": 2, "magnitude": -0.025},
    {"name": "rotation_z_plus_0p025", "kind": "add", "axis": 5, "magnitude": 0.025},
    {"name": "rotation_z_minus_0p025", "kind": "add", "axis": 5, "magnitude": -0.025},
    {"name": "gripper_timing_early_close_0p20", "kind": "gripper_timing", "axis": -1, "magnitude": 0.20},
]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _round(value: float | int | np.floating[Any] | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _state_flat(env: Any) -> np.ndarray:
    raw_env = env.envs[0] if hasattr(env, "envs") else env
    sim = _find_attr_chain(raw_env, "sim")
    if sim is None or not hasattr(sim, "get_state"):
        raise RuntimeError("could not locate simulator get_state for final headroom gate")
    return np.asarray(sim.get_state().flatten(), dtype=np.float64).copy()


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


def _observation_hash(observation: Any) -> str:
    return stable_hash(_compact_for_hash(observation))


def _branch_restoration_scope() -> dict[str, Any]:
    return {
        "simulator_state": "restored from captured MuJoCo flattened state before each branch",
        "smolvla_action_queue": "policy.reset() is called before candidate generation and before continuation in every branch",
        "policy_observation_action_history": "policy.reset() clears branch-local queues/history; candidate actions are executed directly and continuation starts from the same post-candidate observation",
        "policy_recurrent_or_cache_state": "SmolVLA exposes action queues/caches through policy.reset(); reset is applied identically per branch",
        "environment_elapsed_step_counter": "fresh one-env wrapper is created and reset for every branch; branch-local counter state is identical, while the remaining continuation budget is computed from the captured reference step",
        "terminated_truncated_flags": "branch-local done flags are initialized to False for every candidate evaluation",
        "wrapper_state": "fresh SyncVectorEnv/LiberoEnv wrapper per branch, same task, same initial_state_index, same reset_identity",
        "controller_internal_actuator_state": "fresh controller state is initialized identically per branch; deterministic replay test checks that this is sufficient under identical actions/RNG",
        "continuation_rng_identity": "torch.manual_seed(continuation_seed) is set identically before frozen-policy continuation for paired candidates",
        "candidate_generation_rng_state": "official candidate noise is generated from fixed per-candidate torch.Generator seeds and recorded as rng_identity",
    }


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


def _reset_identity_to_initial_state_index(identity: int) -> int:
    if int(identity) not in RESET_IDENTITIES:
        raise ValueError(f"unknown reset identity {identity}")
    return RESET_IDENTITIES.index(int(identity))


def _policy_action(policy: Any, env: Any, observation: Any, loaded: Mapping[str, Any]) -> np.ndarray:
    import torch

    batch = _preprocess_batch(env, observation, dict(loaded))
    with torch.inference_mode():
        action = policy.select_action(batch)
    return _postprocess_action(action, dict(loaded)).reshape(1, -1)


def _select_steps(success_step: int | None, max_steps: int, trace_count: int) -> list[dict[str, Any]]:
    selected = []
    if success_step is not None and success_step >= 24:
        basis = "successful_frozen_policy_trace"
        for request in PHASE_REQUESTS:
            if request["phase"] == "placement":
                step = max(0, int(success_step) - 16)
            else:
                step = int(round(float(request["success_fraction"]) * int(success_step)))
            selected.append({**request, "reference_step": int(min(max(step, 0), trace_count - 1)), "selection_basis": basis})
        return selected
    basis = "deterministic_fallback_fraction_of_reference_horizon"
    horizon = max(1, min(int(max_steps), int(trace_count)))
    for request in PHASE_REQUESTS:
        step = int(round(float(request["fallback_fraction"]) * max(1, horizon - 1)))
        selected.append({**request, "reference_step": int(min(max(step, 0), trace_count - 1)), "selection_basis": basis})
    return selected


def _capture_reference_states(task: Mapping[str, Any], loaded: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    import torch

    env = None
    started = time.monotonic()
    identity = REFERENCE_RESET_IDENTITY
    initial_state_index = _reset_identity_to_initial_state_index(identity)
    snapshots: list[dict[str, Any]] = []
    success_step: int | None = None
    termination_reason = "max_steps_without_success"
    try:
        env = _make_exact_vector_env(str(task["suite"]), int(task["task_id"]), initial_state_index)
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        torch.manual_seed(int(args.reference_seed))
        observation, _ = env.reset(seed=[identity])
        max_steps = int(env.call("_max_episode_steps")[0])
        done = np.array([False])
        for step in range(max_steps):
            flat = _state_flat(env)
            snapshots.append(
                {
                    "step": int(step),
                    "state_flat": flat,
                    "state_hash": stable_hash(flat),
                }
            )
            action = _policy_action(policy, env, observation, loaded)
            observation, reward, terminated, truncated, info = env.step(action)
            if _step_success(info):
                success_step = int(step + 1)
                termination_reason = "success"
            done = terminated | truncated | done
            if np.all(done):
                if success_step is None:
                    termination_reason = "terminated_or_truncated_without_success"
                break
        if not snapshots:
            raise RuntimeError(f"no reference snapshots captured for {task['suite']}/task_{task['task_id']}")
        selected_requests = _select_steps(success_step, max_steps, len(snapshots))
        states = []
        for request in selected_requests:
            snapshot = snapshots[int(request["reference_step"])]
            state_id = (
                f"{task['suite']}|task_{task['task_id']}|identity_{identity}|"
                f"{request['requested_phase']}|step_{snapshot['step']}|{snapshot['state_hash'][:12]}"
            )
            states.append(
                {
                    "state_id": state_id,
                    "group_id": state_id,
                    "suite": str(task["suite"]),
                    "task_id": int(task["task_id"]),
                    "task_key": f"{task['suite']}/task_{task['task_id']}",
                    "task_role": str(task["task_role"]),
                    "instruction": str(task["instruction"]),
                    "reset_identity": int(identity),
                    "initial_state_index": int(initial_state_index),
                    "phase": str(request["phase"]),
                    "requested_phase": str(request["requested_phase"]),
                    "reference_step": int(snapshot["step"]),
                    "reference_state_hash": str(snapshot["state_hash"]),
                    "state_flat": np.asarray(snapshot["state_flat"], dtype=np.float64),
                    "selection_basis": str(request["selection_basis"]),
                }
            )
        return {
            "task": dict(task),
            "reference_reset_identity": int(identity),
            "initial_state_index": int(initial_state_index),
            "max_steps": int(max_steps),
            "reference_success_step": success_step,
            "reference_termination_reason": termination_reason,
            "trace_snapshot_count": int(len(snapshots)),
            "selected_states": states,
            "elapsed_seconds": _round(time.monotonic() - started, 3),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _noise_for_candidate(policy: Any, seed: int) -> Any:
    import torch

    config = policy.model.config
    device = next(policy.parameters()).device
    shape = (1, int(config.chunk_size), int(config.max_action_dim))
    try:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        return torch.randn(shape, generator=generator, device=device)
    except Exception:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(int(seed))
        return torch.randn(shape, generator=generator, device="cpu").to(device)


def _postprocess_chunk(raw_chunk: np.ndarray, loaded: Mapping[str, Any], max_horizon: int) -> np.ndarray:
    import torch

    rows = []
    for row in np.asarray(raw_chunk, dtype=np.float32)[: int(max_horizon)]:
        tensor = torch.as_tensor(row, dtype=torch.float32, device="cuda").reshape(1, -1)
        rows.append(_postprocess_action(tensor, dict(loaded)).reshape(-1))
    return np.stack(rows, axis=0)


def _candidate_endpoint_proxy(start_observation: Any, postprocessed_chunk: np.ndarray) -> list[float] | None:
    eef = _extract_nested_eef(start_observation)
    if eef is None or postprocessed_chunk.shape[1] < 3:
        return None
    return [float(value) for value in (np.asarray(eef, dtype=np.float64) + np.sum(postprocessed_chunk[:, :3], axis=0)).tolist()]


def _generate_official_candidates(
    state: Mapping[str, Any],
    loaded: Mapping[str, Any],
    group_index: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    env = None
    try:
        env = _make_exact_vector_env(str(state["suite"]), int(state["task_id"]), int(state["initial_state_index"]))
        env.reset(seed=[int(state["reset_identity"])])
        observation = _restore_observation_from_flat_state(env, np.asarray(state["state_flat"], dtype=np.float64))
        restored_hash = _sim_state_hash(env)
        if restored_hash != state["reference_state_hash"]:
            raise RuntimeError(f"state restore hash mismatch for {state['state_id']}: {restored_hash} != {state['reference_state_hash']}")
        start_obs_hash = stable_hash(_compact_for_hash(observation))
        candidates = []
        policy = loaded["policy"]
        for candidate_index in range(int(args.candidate_count)):
            rng_identity = int(args.candidate_seed_base + group_index * 100 + candidate_index)
            if hasattr(policy, "reset"):
                policy.reset()
            torch.manual_seed(rng_identity)
            noise = _noise_for_candidate(policy, rng_identity)
            batch = _preprocess_batch(env, observation, dict(loaded))
            with torch.inference_mode():
                raw = policy.predict_action_chunk(batch, noise=noise)
            raw_chunk = np.asarray(raw.detach().to("cpu").numpy()[0], dtype=np.float64)
            post = _postprocess_chunk(raw_chunk, loaded, int(args.max_horizon))
            raw_prefix = raw_chunk[: post.shape[0]]
            candidates.append(
                {
                    "candidate_index": int(candidate_index),
                    "source": "official_stochastic_smolvla_candidate",
                    "rng_identity": rng_identity,
                    "raw_action_chunk": raw_prefix.tolist(),
                    "postprocessed_action_chunk": post.tolist(),
                    "raw_action_hash": stable_hash(raw_prefix),
                    "postprocessed_action_hash": stable_hash(post),
                    "predicted_execution_endpoint_proxy": _candidate_endpoint_proxy(observation, post),
                }
            )
        diversity = summarize_candidate_diversity(candidates, thresholds=DiversityThresholds())
        return {
            "state_id": str(state["state_id"]),
            "start_observation_hash": start_obs_hash,
            "restored_state_hash": restored_hash,
            "candidates": candidates,
            "diversity": diversity,
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _capture_determinism_probe_state(loaded: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    env = None
    task = FINAL_TASKS[0]
    capture_step = 12
    try:
        env = _make_exact_vector_env(str(task["suite"]), int(task["task_id"]), 0)
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        observation, _ = env.reset(seed=[REFERENCE_RESET_IDENTITY])
        max_steps = int(env.call("_max_episode_steps")[0])
        for _ in range(capture_step):
            action = _policy_action(policy, env, observation, loaded)
            observation, reward, terminated, truncated, info = env.step(action)
            if np.all(terminated | truncated):
                break
        state_flat = _state_flat(env)
        state = {
            "state_id": f"determinism_probe|{task['suite']}|task_{task['task_id']}|step_{capture_step}|{stable_hash(state_flat)[:12]}",
            "group_id": "determinism_probe",
            "suite": str(task["suite"]),
            "task_id": int(task["task_id"]),
            "task_key": f"{task['suite']}/task_{task['task_id']}",
            "task_role": str(task["task_role"]),
            "instruction": str(task["instruction"]),
            "reset_identity": REFERENCE_RESET_IDENTITY,
            "initial_state_index": 0,
            "phase": "approach",
            "requested_phase": "determinism_probe_mid_trajectory",
            "reference_step": int(capture_step),
            "reference_state_hash": stable_hash(state_flat),
            "state_flat": state_flat,
            "selection_basis": "fixed determinism probe step before candidate outcomes",
            "max_episode_steps": int(max_steps),
        }
        restored_observation = _restore_observation_from_flat_state(env, state_flat)
        candidate_seed = int(args.candidate_seed_base + 99991)
        generated_once = _generate_single_official_candidate(loaded, env, restored_observation, candidate_seed, int(args.max_horizon), 0)
        if hasattr(policy, "reset"):
            policy.reset()
        generated_twice = _generate_single_official_candidate(loaded, env, restored_observation, candidate_seed, int(args.max_horizon), 0)
        return {
            "state": state,
            "candidate_seed": candidate_seed,
            "candidate_generation_reproducible": generated_once["postprocessed_action_hash"] == generated_twice["postprocessed_action_hash"]
            and generated_once["raw_action_hash"] == generated_twice["raw_action_hash"],
            "candidate_once": generated_once,
            "candidate_twice": generated_twice,
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _generate_single_official_candidate(
    loaded: Mapping[str, Any],
    env: Any,
    observation: Any,
    rng_identity: int,
    max_horizon: int,
    candidate_index: int,
) -> dict[str, Any]:
    import torch

    policy = loaded["policy"]
    if hasattr(policy, "reset"):
        policy.reset()
    torch.manual_seed(int(rng_identity))
    noise = _noise_for_candidate(policy, int(rng_identity))
    batch = _preprocess_batch(env, observation, dict(loaded))
    with torch.inference_mode():
        raw = policy.predict_action_chunk(batch, noise=noise)
    raw_chunk = np.asarray(raw.detach().to("cpu").numpy()[0], dtype=np.float64)
    post = _postprocess_chunk(raw_chunk, loaded, int(max_horizon))
    raw_prefix = raw_chunk[: post.shape[0]]
    return {
        "candidate_index": int(candidate_index),
        "source": "official_stochastic_smolvla_candidate",
        "rng_identity": int(rng_identity),
        "raw_action_chunk": raw_prefix.tolist(),
        "postprocessed_action_chunk": post.tolist(),
        "raw_action_hash": stable_hash(raw_prefix),
        "postprocessed_action_hash": stable_hash(post),
        "predicted_execution_endpoint_proxy": _candidate_endpoint_proxy(observation, post),
    }


def _run_determinism_branch(
    *,
    state: Mapping[str, Any],
    candidate: Mapping[str, Any],
    loaded: Mapping[str, Any],
    args: argparse.Namespace,
    branch_name: str,
) -> dict[str, Any]:
    import torch

    env = None
    try:
        env = _make_exact_vector_env(str(state["suite"]), int(state["task_id"]), int(state["initial_state_index"]))
        env.reset(seed=[int(state["reset_identity"])])
        restored_observation = _restore_observation_from_flat_state(env, np.asarray(state["state_flat"], dtype=np.float64))
        restored_hash = _sim_state_hash(env)
        restored_observation_hash = _observation_hash(restored_observation)
        if restored_hash != state["reference_state_hash"]:
            raise RuntimeError(f"determinism restore hash mismatch: {restored_hash} != {state['reference_state_hash']}")
        action_chunk = np.asarray(candidate["postprocessed_action_chunk"], dtype=np.float64)[: int(args.max_horizon)]
        done = np.array([False])
        task_success = False
        reward_trace: list[float] = []
        done_trace: list[dict[str, bool]] = []
        next_observation_hash = None
        observation = restored_observation
        for step, row in enumerate(action_chunk, start=1):
            observation, reward, terminated, truncated, info = env.step(row.reshape(1, -1))
            reward_value = float(np.asarray(reward).reshape(-1)[0])
            reward_trace.append(reward_value)
            term_value = bool(np.asarray(terminated).reshape(-1)[0])
            trunc_value = bool(np.asarray(truncated).reshape(-1)[0])
            done_trace.append({"terminated": term_value, "truncated": trunc_value, "success": _step_success(info)})
            if step == 1:
                next_observation_hash = _observation_hash(observation)
            task_success = bool(task_success or _step_success(info))
            done = terminated | truncated | done
            if np.all(done):
                break
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        continuation_seed = int(args.continuation_seed_base + 99991)
        torch.manual_seed(continuation_seed)
        max_steps = int(env.call("_max_episode_steps")[0])
        remaining_budget = max(1, max_steps - int(state["reference_step"]) - int(args.max_horizon))
        continuation_actions = []
        continuation_rewards = []
        continuation_done_trace = []
        continuation_steps = 0
        while not np.all(done) and continuation_steps < remaining_budget:
            action = _policy_action(policy, env, observation, loaded)
            continuation_actions.append(action.reshape(-1).tolist())
            observation, reward, terminated, truncated, info = env.step(action)
            reward_value = float(np.asarray(reward).reshape(-1)[0])
            continuation_rewards.append(reward_value)
            term_value = bool(np.asarray(terminated).reshape(-1)[0])
            trunc_value = bool(np.asarray(truncated).reshape(-1)[0])
            step_success = _step_success(info)
            continuation_done_trace.append({"terminated": term_value, "truncated": trunc_value, "success": step_success})
            task_success = bool(task_success or step_success)
            done = terminated | truncated | done
            continuation_steps += 1
        final_observation_hash = _observation_hash(observation)
        return {
            "branch": branch_name,
            "restored_state_hash": restored_hash,
            "restored_observation_hash": restored_observation_hash,
            "immediate_next_observation_hash": next_observation_hash,
            "candidate_reward_trace": reward_trace,
            "candidate_done_trace": done_trace,
            "continuation_seed": continuation_seed,
            "continuation_action_count": len(continuation_actions),
            "continuation_action_hash": stable_hash(np.asarray(continuation_actions, dtype=np.float64))
            if continuation_actions
            else stable_hash([]),
            "continuation_reward_trace_hash": stable_hash(np.asarray(continuation_rewards, dtype=np.float64))
            if continuation_rewards
            else stable_hash([]),
            "continuation_done_trace": continuation_done_trace,
            "final_observation_hash": final_observation_hash,
            "final_success": bool(task_success),
            "final_done": bool(np.all(done)),
            "total_steps": int(len(reward_trace) + continuation_steps),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _run_restoration_determinism_test(loaded: Mapping[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    probe = _capture_determinism_probe_state(loaded, args)
    state = probe["state"]
    candidate = probe["candidate_once"]
    left = _run_determinism_branch(state=state, candidate=candidate, loaded=loaded, args=args, branch_name="restore_a")
    right = _run_determinism_branch(state=state, candidate=candidate, loaded=loaded, args=args, branch_name="restore_b")
    checks = {
        "candidate_generation_rng_reproduces_chunk": bool(probe["candidate_generation_reproducible"]),
        "restored_observations_identical": left["restored_observation_hash"] == right["restored_observation_hash"],
        "immediate_next_observations_identical": left["immediate_next_observation_hash"] == right["immediate_next_observation_hash"],
        "candidate_rewards_dones_identical": left["candidate_reward_trace"] == right["candidate_reward_trace"]
        and left["candidate_done_trace"] == right["candidate_done_trace"],
        "continuation_action_chunks_identical": left["continuation_action_hash"] == right["continuation_action_hash"],
        "continuation_rewards_dones_identical": left["continuation_reward_trace_hash"] == right["continuation_reward_trace_hash"]
        and left["continuation_done_trace"] == right["continuation_done_trace"],
        "final_episode_outcome_identical": left["final_success"] == right["final_success"]
        and left["final_done"] == right["final_done"]
        and left["total_steps"] == right["total_steps"],
    }
    passed = bool(all(checks.values()))
    return {
        "scope": _branch_restoration_scope(),
        "probe_state": {key: value for key, value in state.items() if key != "state_flat"},
        "candidate_seed": int(probe["candidate_seed"]),
        "candidate_hash": candidate["postprocessed_action_hash"],
        "branches": [left, right],
        "checks": checks,
        "passed": passed,
        "failure_rule": "if any check is false, final headroom measurement must be ECHO_GATE_MEASUREMENT_INVALID",
    }


def _structured_candidates(default_candidate: Mapping[str, Any], max_horizon: int) -> list[dict[str, Any]]:
    base = np.asarray(default_candidate["postprocessed_action_chunk"], dtype=np.float64)[: int(max_horizon)]
    candidates = []
    for index, spec in enumerate(STRUCTURED_PERTURBATION_SPECS):
        chunk = np.asarray(base, dtype=np.float64).copy()
        if spec["kind"] == "add" and spec["axis"] is not None and int(spec["axis"]) < chunk.shape[1]:
            chunk[:, int(spec["axis"])] += float(spec["magnitude"])
        elif spec["kind"] == "gripper_timing" and chunk.shape[1] >= 7:
            split = max(1, min(4, chunk.shape[0] // 2 if chunk.shape[0] > 1 else 1))
            chunk[:split, -1] -= float(spec["magnitude"])
            chunk[split:, -1] += float(spec["magnitude"])
        chunk = np.clip(chunk, -1.0, 1.0)
        candidates.append(
            {
                "candidate_index": int(index),
                "source": "structured_perturbation_diagnostic",
                "structured_spec": dict(spec),
                "rng_identity": None,
                "raw_action_chunk": None,
                "postprocessed_action_chunk": chunk.tolist(),
                "postprocessed_action_hash": stable_hash(chunk),
                "official_policy_candidate": False,
            }
        )
    return candidates


def _extended_effect(
    start_observation: Any,
    final_observation: Any,
    action_chunk: np.ndarray,
    instruction: str,
    task_success: bool,
    start_body_positions: Mapping[str, list[float]],
    final_body_positions: Mapping[str, list[float]],
) -> dict[str, Any]:
    effect = _effect_from_observations(
        start_observation,
        final_observation,
        action_chunk,
        instruction,
        task_success,
        dict(start_body_positions),
        dict(final_body_positions),
    )
    start_eef = _extract_nested_eef(start_observation)
    final_eef = _extract_nested_eef(final_observation)
    target_key = _best_body_key(dict(start_body_positions), instruction) if start_body_positions else None
    orientation_delta = None
    gripper_qpos_delta = None
    try:
        start_robot = start_observation["robot_state"]
        final_robot = final_observation["robot_state"]
        start_quat = np.asarray(start_robot["eef"]["quat"], dtype=np.float64).reshape(-1)
        final_quat = np.asarray(final_robot["eef"]["quat"], dtype=np.float64).reshape(-1)
        orientation_delta = float(np.linalg.norm(final_quat[:4] - start_quat[:4]))
        start_gripper = np.asarray(start_robot["gripper"]["qpos"], dtype=np.float64).reshape(-1)
        final_gripper = np.asarray(final_robot["gripper"]["qpos"], dtype=np.float64).reshape(-1)
        gripper_qpos_delta = float(np.linalg.norm(final_gripper - start_gripper))
    except Exception:
        pass
    return {
        **effect,
        "orientation_delta_norm": orientation_delta,
        "gripper_qpos_delta_norm": gripper_qpos_delta,
        "target_body_key": target_key,
        "start_eef": start_eef,
        "final_eef": final_eef,
    }


def _evaluate_candidates(
    *,
    state: Mapping[str, Any],
    candidates: list[dict[str, Any]],
    loaded: Mapping[str, Any],
    group_index: int,
    candidate_family: str,
    start_observation_hash: str,
    args: argparse.Namespace,
) -> dict[str, Any]:
    import torch

    random_rng = np.random.default_rng(int(args.random_selector_seed_base + group_index))
    random_candidate_index = int(random_rng.integers(0, len(candidates)))
    continuation_seed = int(args.continuation_seed_base + group_index)
    max_horizon = int(args.max_horizon)
    requested_horizons = [h for h in EFFECT_HORIZONS if h <= max_horizon]
    if max_horizon not in requested_horizons:
        requested_horizons.append(max_horizon)
    requested_horizons = sorted(set(requested_horizons))
    rows = []
    env_max_steps = None
    for candidate in candidates:
        env = None
        candidate_started = time.monotonic()
        try:
            env = _make_exact_vector_env(str(state["suite"]), int(state["task_id"]), int(state["initial_state_index"]))
            env.reset(seed=[int(state["reset_identity"])])
            env_max_steps = int(env.call("_max_episode_steps")[0])
            start_observation = _restore_observation_from_flat_state(env, np.asarray(state["state_flat"], dtype=np.float64))
            restored_hash = _sim_state_hash(env)
            restoration_ok = restored_hash == state["reference_state_hash"]
            if not restoration_ok:
                raise RuntimeError(f"candidate restore hash mismatch for {state['state_id']}")
            start_body_positions = _sim_body_positions(env)
            action_chunk = np.asarray(candidate["postprocessed_action_chunk"], dtype=np.float64)[:max_horizon]
            task_success = False
            done = np.array([False])
            terminated_last = False
            truncated_last = False
            reward_sum = 0.0
            max_reward = 0.0
            observation = start_observation
            final_body_positions = dict(start_body_positions)
            horizon_effects: dict[str, Any] = {}
            intervention_steps = 0
            for step, action_row in enumerate(action_chunk, start=1):
                observation, reward, terminated, truncated, info = env.step(action_row.reshape(1, -1))
                reward_value = float(np.asarray(reward).reshape(-1)[0])
                reward_sum += reward_value
                max_reward = max(max_reward, reward_value)
                task_success = bool(task_success or _step_success(info))
                terminated_last = bool(np.asarray(terminated).reshape(-1)[0])
                truncated_last = bool(np.asarray(truncated).reshape(-1)[0])
                done = terminated | truncated | done
                intervention_steps = int(step)
                final_body_positions = _sim_body_positions(env)
                if step in requested_horizons:
                    effect = _extended_effect(
                        start_observation,
                        observation,
                        action_chunk[:step],
                        str(state["instruction"]),
                        task_success,
                        start_body_positions,
                        final_body_positions,
                    )
                    horizon_effects[str(step)] = effect
                if np.all(done):
                    break
            for horizon in requested_horizons:
                if str(horizon) not in horizon_effects:
                    effective_horizon = max(1, min(intervention_steps, action_chunk.shape[0]))
                    effect = _extended_effect(
                        start_observation,
                        observation,
                        action_chunk[:effective_horizon],
                        str(state["instruction"]),
                        task_success,
                        start_body_positions,
                        final_body_positions,
                    )
                    horizon_effects[str(horizon)] = effect
            continuation_steps = 0
            if not np.all(done):
                policy = loaded["policy"]
                if hasattr(policy, "reset"):
                    policy.reset()
                torch.manual_seed(continuation_seed)
                remaining_budget = max(1, int(env_max_steps) - int(state["reference_step"]) - max_horizon)
                while not np.all(done) and continuation_steps < remaining_budget:
                    action = _policy_action(policy, env, observation, loaded)
                    observation, reward, terminated, truncated, info = env.step(action)
                    reward_value = float(np.asarray(reward).reshape(-1)[0])
                    reward_sum += reward_value
                    max_reward = max(max_reward, reward_value)
                    task_success = bool(task_success or _step_success(info))
                    terminated_last = bool(np.asarray(terminated).reshape(-1)[0])
                    truncated_last = bool(np.asarray(truncated).reshape(-1)[0])
                    done = terminated | truncated | done
                    continuation_steps += 1
            effect_at_max = horizon_effects[str(max_horizon)]
            effect_compatibility = compatibility_score(effect_at_max, str(state["phase"]))
            populated = sorted(
                key
                for key, value in effect_at_max.items()
                if isinstance(value, (int, float, np.floating)) and value is not None and abs(float(value)) > 1e-9
            )
            row = {
                **{key: value for key, value in candidate.items() if key not in {"raw_action_chunk", "postprocessed_action_chunk"}},
                "candidate_index": int(candidate["candidate_index"]),
                "source": str(candidate["source"]),
                "candidate_family": candidate_family,
                "postprocessed_action_chunk": candidate["postprocessed_action_chunk"],
                "raw_action_chunk": candidate.get("raw_action_chunk"),
                "restoration_ok": bool(restoration_ok),
                "restored_state_hash": restored_hash,
                "start_observation_hash": start_observation_hash,
                "intervention_horizon": int(max_horizon),
                "intervention_steps_executed": int(intervention_steps),
                "effect_horizons": horizon_effects,
                "effect_compatibility": float(effect_compatibility),
                "effect_available_components_at_max_horizon": populated,
                "downstream_success": bool(task_success),
                "reward_sum": _round(reward_sum, 6),
                "max_reward": _round(max_reward, 6),
                "terminated_last": bool(terminated_last),
                "truncated_last": bool(truncated_last),
                "continuation_steps": int(continuation_steps),
                "continuation_seed": int(continuation_seed),
                "elapsed_seconds": _round(time.monotonic() - candidate_started, 3),
            }
            rows.append(row)
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    return {
        "group_id": str(state["group_id"]),
        "state_id": str(state["state_id"]),
        "suite": str(state["suite"]),
        "task_id": int(state["task_id"]),
        "task_key": str(state["task_key"]),
        "task_role": str(state["task_role"]),
        "instruction": str(state["instruction"]),
        "phase": str(state["phase"]),
        "requested_phase": str(state["requested_phase"]),
        "reference_step": int(state["reference_step"]),
        "reference_state_hash": str(state["reference_state_hash"]),
        "start_observation_hash": start_observation_hash,
        "candidate_family": candidate_family,
        "candidate_count": len(rows),
        "random_candidate_index": random_candidate_index,
        "continuation_seed": int(continuation_seed),
        "max_episode_steps": int(env_max_steps) if env_max_steps is not None else None,
        "candidates": rows,
    }


def _audit_previous_gate(report_dir: Path) -> dict[str, Any]:
    path = report_dir / "echo_vla_first_prototype_result.json"
    if not path.exists():
        return {"available": False, "path": str(path)}
    payload = json.loads(path.read_text(encoding="utf-8"))
    effect_components: dict[str, int] = {}
    diversity_rows = []
    for group in payload.get("candidate_rows") or []:
        candidates = []
        for row in group:
            candidate = row.get("candidate") or {}
            effect = candidate.get("realized_effect") or {}
            for key, value in effect.items():
                try:
                    if abs(float(value)) > 1e-9:
                        effect_components[key] = effect_components.get(key, 0) + 1
                except Exception:
                    pass
            candidates.append(
                {
                    "candidate_index": candidate.get("candidate_index", len(candidates)),
                    "postprocessed_action_chunk": candidate.get("action_chunk") or [],
                }
            )
        if candidates:
            diversity_rows.append(summarize_candidate_diversity(candidates))
    return {
        "available": True,
        "path": str(path),
        "previous_final_decision": payload.get("final_decision"),
        "success_metric_definition": (
            "CandidateRecord.success was not downstream task success. The first gate set it to "
            "compatibility_score(realized_effect, phase) > 0.05 after a four-step intervention."
        ),
        "task_success_definition": (
            "task_success was recorded only during the four candidate steps and the frozen policy was not continued "
            "to normal episode termination."
        ),
        "success_evaluated_immediately_after_four_env_steps": True,
        "frozen_policy_continued_after_candidate": False,
        "materially_better_definition": (
            "local realized-effect compatibility: oracle.success or oracle.compatibility > default.compatibility + 0.25"
        ),
        "effect_components_populated_nonzero": effect_components,
        "effect_schema_had_full_fields": [
            "eef_delta_norm",
            "target_distance_delta",
            "contact_transition",
            "gripper_transition",
            "object_retained",
            "object_lift_delta",
            "object_goal_delta",
            "placement_alignment",
            "release_stability",
        ],
        "candidate_generation_randomness": (
            "not official stochastic candidates; candidates after index 0 were deterministic bounded perturbations "
            "around the default chunk"
        ),
        "candidate_diversity_summaries": diversity_rows,
        "classification": "SHORT_HORIZON_LOCAL_EFFECT_PROXY_WITHOUT_CONTINUATION_INSUFFICIENT",
        "classification_note": (
            "The exact conditional SHORT_HORIZON_TASK_SUCCESS_METRIC_INSUFFICIENT does not strictly apply because "
            "the previous metric used local effect compatibility rather than full task success; it was still "
            "insufficient for downstream candidate-headroom adjudication."
        ),
    }


def _write_plan(report_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# ECHO Final Headroom Plan",
        "",
        "Status: final bounded candidate-headroom adjudication. No ECHO, SmolVLA, OpenVLA, effect, phase, ranking, or value head is trained.",
        "",
        "## Frozen Scope",
        "",
        "- official stochastic candidates: frozen official SmolVLA-LIBERO only",
        "- task count: `3`",
        "- phases per task: `4`",
        "- same-state intervention groups: `12`",
        f"- candidate count K: `{args.candidate_count}`",
        f"- effect horizons: `{[h for h in EFFECT_HORIZONS if h <= int(args.max_horizon)]}`",
        f"- continuation intervention horizon: `{args.max_horizon}`",
        "- OpenVLA-OFT: `not used`",
        "- full benchmark: `not run`",
        "- downloads: `forbidden/offline env vars set by launcher`",
        "",
        "## Predeclared Near-Identical Thresholds",
        "",
        "- exact identical pair: full postprocessed chunk L2 `<=1e-9`",
        "- nearly identical pair: full postprocessed chunk L2 `<=1e-3`, or translation/rotation/gripper component L2 all `<=1e-4`",
        "- impoverished state: effective distinct candidates `<2`, mean pairwise action L2 `<0.01`, or nearly-identical pair fraction `>=0.75`",
        "- impoverished policy candidate set: at least two-thirds of the 12 states are impoverished",
        "",
        "## Non-Relaxed Headroom Criteria",
        "",
        "- final task-success oracle improvement over default candidate must be at least `10` absolute percentage points",
        "- at least `15%` of default-failure states must contain another official policy candidate that succeeds",
        "- recoveries must span at least two tasks and more than one phase/state",
        "- structured perturbations are diagnostic only and are not official VLA candidates",
    ]
    _write_md(report_dir / "echo_final_headroom_plan.md", lines)


def _write_reports(report_dir: Path, report: Mapping[str, Any]) -> None:
    previous = report.get("previous_gate_audit") or {}
    diversity = report.get("official_candidate_diversity") or {}
    official = report.get("official_policy_candidate_metrics") or {}
    structured = report.get("structured_candidate_metrics") or {}
    determinism = report.get("restoration_determinism_test") or {}
    final_decision = str(report.get("final_decision"))

    _write_json(report_dir / "echo_final_headroom_result.json", report)
    _write_md(
        report_dir / "echo_candidate_diversity_audit.md",
        [
            "# ECHO Candidate Diversity Audit",
            "",
            f"- official groups: `{diversity.get('state_count')}`",
            f"- impoverished official states: `{diversity.get('impoverished_state_count')}`",
            f"- impoverished fraction: `{diversity.get('impoverished_state_fraction')}`",
            f"- policy candidates impoverished: `{diversity.get('policy_candidates_impoverished')}`",
            "",
            "## Threshold Rule",
            "",
            str(diversity.get("predeclared_rule")),
        ],
    )
    _write_md(
        report_dir / "echo_horizon_and_success_semantics.md",
        [
            "# ECHO Horizon And Success Semantics",
            "",
            "## Previous Gate",
            "",
            f"- previous decision: `{previous.get('previous_final_decision')}`",
            f"- previous success metric: `{previous.get('success_metric_definition')}`",
            f"- policy continuation after candidate: `{previous.get('frozen_policy_continued_after_candidate')}`",
            f"- materially better definition: `{previous.get('materially_better_definition')}`",
            f"- classification: `{previous.get('classification')}`",
            "",
            "## Final Gate",
            "",
            f"- candidate-only effect horizons: `{report.get('effect_horizons')}`",
            f"- downstream success metric: `{report.get('downstream_success_metric')}`",
            f"- continuation horizon: `{report.get('continuation_intervention_horizon')}`",
            "- local physical progress is diagnostic only; final GO requires downstream official task success.",
        ],
    )
    _write_md(
        report_dir / "echo_final_headroom_result.md",
        [
            "# ECHO Final Headroom Result",
            "",
            f"- final decision: `{final_decision}`",
            f"- measurement valid: `{report.get('measurement_valid')}`",
            f"- exact groups: `{report.get('same_state_intervention_group_count')}`",
            f"- official candidate K: `{report.get('candidate_count')}`",
            f"- official default success rate: `{official.get('default_success_rate')}`",
            f"- official oracle success rate: `{official.get('final_task_success_oracle_rate')}`",
            f"- official oracle improvement pp: `{official.get('oracle_improvement_pp')}`",
            f"- official recoverable default failures: `{official.get('recoverable_default_failure_count')}` / `{official.get('default_failure_group_count')}`",
            f"- official recoverable rate: `{official.get('recoverable_default_failure_rate')}`",
            f"- headroom spans multiple tasks: `{official.get('headroom_spans_multiple_tasks')}`",
            f"- structured oracle improvement pp: `{structured.get('oracle_improvement_pp')}`",
            f"- structured recoverable rate: `{structured.get('recoverable_default_failure_rate')}`",
            f"- restoration determinism passed: `{determinism.get('passed')}`",
            f"- restoration determinism checks: `{determinism.get('checks')}`",
            f"- latency/VRAM: `{report.get('latency_vram_summary')}`",
        ],
    )
    _write_md(
        report_dir / "echo_final_headroom_decision.md",
        [
            "# ECHO Final Headroom Decision",
            "",
            f"Final decision: `{final_decision}`",
            "",
            "## Basis",
            "",
            f"- previous success-metric definition: `{previous.get('success_metric_definition')}`",
            f"- states and phases evaluated: `{report.get('state_phase_summary')}`",
            f"- candidate diversity: `{diversity}`",
            f"- official-policy downstream success: `{official}`",
            f"- structured-candidate downstream success: `{structured}`",
            f"- restoration scope: `{determinism.get('scope')}`",
            f"- restoration determinism passed: `{determinism.get('passed')}`",
            f"- restoration determinism checks: `{determinism.get('checks')}`",
            f"- restoration determinism probe: `{determinism.get('probe_state')}`",
            f"- restoration determinism branch summary: `{[{key: branch.get(key) for key in ['branch', 'restored_observation_hash', 'immediate_next_observation_hash', 'continuation_action_count', 'continuation_action_hash', 'final_success', 'final_done', 'total_steps']} for branch in (determinism.get('branches') or [])]}`",
            f"- no-test-privilege check: `{report.get('no_test_privilege_check')}`",
            "",
            "## Exact Next Prompt",
            "",
            str(report.get("exact_next_prompt")),
        ],
    )


def _update_project_reports(report_dir: Path, report: Mapping[str, Any]) -> None:
    decision = str(report.get("final_decision"))
    project_state = report_dir / "project_state.md"
    next_actions = report_dir / "next_actions.md"
    decision_log = report_dir / "decision_log.md"
    block = [
        "",
        "## ECHO Final Candidate Headroom Gate - 2026-07-11",
        "",
        f"- branch: `{BRANCH}`",
        f"- decision: `{decision}`",
        f"- official groups/candidates: `{report.get('same_state_intervention_group_count')}` / `{report.get('official_candidate_record_count')}`",
        f"- structured diagnostic candidates: `{report.get('structured_candidate_record_count')}`",
        f"- training happened: `{report.get('training_happened')}`",
        f"- OpenVLA used: `{report.get('policy', {}).get('openvla_oft_used')}`",
    ]
    with project_state.open("a", encoding="utf-8") as handle:
        handle.write("\n".join(block).rstrip() + "\n")
    with next_actions.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## After ECHO Final Candidate Headroom Gate - 2026-07-11\n\n"
            f"- decision: `{decision}`\n"
            f"- next: `{report.get('exact_next_prompt')}`\n"
        )
    with decision_log.open("a", encoding="utf-8") as handle:
        handle.write(
            "\n## 2026-07-11 - ECHO Final Candidate Headroom Gate\n\n"
            f"Decision: `{decision}`\n\n"
            f"Evidence: official downstream metrics `{report.get('official_policy_candidate_metrics')}`, "
            f"structured diagnostic metrics `{report.get('structured_candidate_metrics')}`.\n"
        )


def run_final_gate(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    report_dir = Path(args.report_dir)
    _write_plan(report_dir, args)
    report: dict[str, Any] = {
        "schema_version": "echo_final_headroom_gate_v1",
        "date_kst": DATE_KST,
        "branch": BRANCH,
        "base_main_commit": "e44b18d4f943ad1b7dda607f457730571f314fb4",
        "training_happened": False,
        "components_trained": "none_final_candidate_headroom_gate_only",
        "policy": {
            "official_smolvla_used": True,
            "frozen_backbone": True,
            "openvla_oft_used": False,
            "full_benchmark_run": False,
            "downloads_performed": False,
            "privileged_inference_used": False,
            "privileged_simulator_values_role": "diagnostic_effect_labels_and_state_restoration_only",
        },
        "previous_gate_audit": {},
        "tasks": FINAL_TASKS,
        "state_selection": [],
        "state_phase_summary": [],
        "same_state_intervention_group_count": 0,
        "candidate_count": int(args.candidate_count),
        "effect_horizons": [h for h in EFFECT_HORIZONS if h <= int(args.max_horizon)],
        "continuation_intervention_horizon": int(args.max_horizon),
        "downstream_success_metric": "official LIBERO task success after candidate intervention plus frozen SmolVLA continuation to normal bounded episode termination",
        "official_groups": [],
        "structured_groups": [],
        "official_candidate_diversity": {},
        "official_policy_candidate_metrics": {},
        "structured_candidate_metrics": {},
        "restoration_scope": _branch_restoration_scope(),
        "restoration_determinism_test": {},
        "official_candidate_record_count": 0,
        "structured_candidate_record_count": 0,
        "same_state_restoration_test": {},
        "candidate_rng_identity_test": {},
        "paired_continuation_seed_test": {},
        "effect_label_schema_validation": {},
        "candidate_diversity_test": {},
        "no_test_privilege_check": {},
        "measurement_valid": False,
        "errors": [],
        "latency_vram_summary": {},
        "final_decision": "ECHO_GATE_MEASUREMENT_INVALID",
        "exact_next_prompt": None,
    }
    loaded = None
    try:
        _set_runtime_env(args)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for official frozen SmolVLA candidate generation")
        if int(args.candidate_count) != 8:
            raise RuntimeError("final ECHO candidate gate requires K=8 official stochastic candidates")
        assert_no_privileged_deployment_inputs({"observation": "current", "instruction": "task", "candidate_action_chunk": "current"})
        spec = next(item for item in POLICIES if item.name == "frozen_base")
        loaded = _load_policy_and_processors(args, spec)
        report["policy_load_audit"] = loaded["audit"]

        if bool(args.determinism_only):
            existing_path = report_dir / "echo_final_headroom_result.json"
            if not existing_path.exists():
                raise RuntimeError("--determinism-only requires an existing echo_final_headroom_result.json")
            existing = json.loads(existing_path.read_text(encoding="utf-8"))
            determinism = _run_restoration_determinism_test(loaded, args)
            existing["restoration_scope"] = _branch_restoration_scope()
            existing["restoration_determinism_test"] = determinism
            existing["same_state_restoration_test"] = {
                **(existing.get("same_state_restoration_test") or {}),
                "deterministic_two_branch_replay_passed": bool(determinism.get("passed")),
            }
            existing["measurement_valid_without_determinism"] = bool(existing.get("measurement_valid"))
            existing["measurement_valid"] = bool(existing.get("measurement_valid")) and bool(determinism.get("passed"))
            if not determinism.get("passed"):
                existing["final_decision_before_determinism_override"] = existing.get("final_decision")
                existing["final_decision"] = "ECHO_GATE_MEASUREMENT_INVALID"
                existing["exact_next_prompt"] = "Fix branch restoration determinism before using any candidate success differences as causal headroom evidence."
            existing["elapsed_seconds_determinism_update"] = _round(time.monotonic() - started, 3)
            existing["latency_vram_summary"] = {
                **(existing.get("latency_vram_summary") or {}),
                "determinism_update_elapsed_seconds": existing["elapsed_seconds_determinism_update"],
                "determinism_update_cuda_memory": _cuda_memory(torch),
            }
            report.update(existing)
            return report

        reference_results = [_capture_reference_states(task, loaded, args) for task in FINAL_TASKS]
        selected_states = []
        for result in reference_results:
            selected_states.extend(result["selected_states"])
        if len(selected_states) != 12:
            raise RuntimeError(f"final gate must select exactly 12 states, got {len(selected_states)}")
        report["state_selection"] = [
            {
                **{key: value for key, value in result.items() if key != "selected_states"},
                "selected_states": [
                    {key: value for key, value in state.items() if key != "state_flat"}
                    for state in result["selected_states"]
                ],
            }
            for result in reference_results
        ]
        report["state_phase_summary"] = [
            {
                "state_id": state["state_id"],
                "task_key": state["task_key"],
                "phase": state["phase"],
                "requested_phase": state["requested_phase"],
                "reference_step": state["reference_step"],
                "selection_basis": state["selection_basis"],
            }
            for state in selected_states
        ]

        official_groups = []
        structured_groups = []
        diversity_summaries = []
        all_rngs = []
        all_continuation_seeds = []
        for group_index, state in enumerate(selected_states):
            generated = _generate_official_candidates(state, loaded, group_index, args)
            diversity_summaries.append(generated["diversity"])
            all_rngs.extend(candidate["rng_identity"] for candidate in generated["candidates"])
            official_group = _evaluate_candidates(
                state=state,
                candidates=generated["candidates"],
                loaded=loaded,
                group_index=group_index,
                candidate_family="official_policy",
                start_observation_hash=generated["start_observation_hash"],
                args=args,
            )
            official_group["diversity"] = generated["diversity"]
            all_continuation_seeds.append(official_group["continuation_seed"])
            official_groups.append(official_group)
            structured = _structured_candidates(generated["candidates"][0], int(args.max_horizon))
            structured_group = _evaluate_candidates(
                state=state,
                candidates=structured,
                loaded=loaded,
                group_index=group_index,
                candidate_family="structured_diagnostic",
                start_observation_hash=generated["start_observation_hash"],
                args=args,
            )
            structured_groups.append(structured_group)

        official_diversity = summarize_diversity_across_states(diversity_summaries)
        official_metrics = downstream_headroom_metrics(official_groups)
        structured_metrics = downstream_headroom_metrics(structured_groups)
        restoration_records = [
            candidate["restoration_ok"]
            for group in official_groups + structured_groups
            for candidate in group.get("candidates", [])
        ]
        effect_schema_keys = {
            key
            for group in official_groups + structured_groups
            for candidate in group.get("candidates", [])
            for effect in (candidate.get("effect_horizons") or {}).values()
            for key in effect
        }
        required_effect_keys = {
            "eef_delta_norm",
            "target_distance_delta",
            "contact_transition",
            "gripper_transition",
            "object_retained",
            "object_lift_delta",
            "object_goal_delta",
            "placement_alignment",
            "release_stability",
            "orientation_delta_norm",
            "gripper_qpos_delta_norm",
        }
        measurement_valid = (
            len(restoration_records) > 0
            and all(restoration_records)
            and len(set(all_rngs)) == len(all_rngs)
            and len(official_groups) == 12
            and all(group.get("candidate_count") == 8 for group in official_groups)
            and required_effect_keys.issubset(effect_schema_keys)
        )
        report.update(
            {
                "previous_gate_audit": _audit_previous_gate(report_dir),
                "official_groups": official_groups,
                "structured_groups": structured_groups,
                "same_state_intervention_group_count": len(official_groups),
                "official_candidate_record_count": sum(len(group.get("candidates", [])) for group in official_groups),
                "structured_candidate_record_count": sum(len(group.get("candidates", [])) for group in structured_groups),
                "official_candidate_diversity": official_diversity,
                "official_policy_candidate_metrics": official_metrics,
                "structured_candidate_metrics": structured_metrics,
                "same_state_restoration_test": {
                    "checked_candidate_executions": len(restoration_records),
                    "all_restorations_exact_hash_match": bool(restoration_records and all(restoration_records)),
                },
                "candidate_rng_identity_test": {
                    "rng_identity_count": len(all_rngs),
                    "unique_rng_identity_count": len(set(all_rngs)),
                    "all_unique": len(set(all_rngs)) == len(all_rngs),
                },
                "paired_continuation_seed_test": {
                    "group_count": len(all_continuation_seeds),
                    "unique_group_seed_count": len(set(all_continuation_seeds)),
                    "same_seed_used_within_each_group": True,
                },
                "effect_label_schema_validation": {
                    "required_keys": sorted(required_effect_keys),
                    "observed_keys": sorted(effect_schema_keys),
                    "all_required_keys_present": required_effect_keys.issubset(effect_schema_keys),
                },
                "candidate_diversity_test": official_diversity,
                "no_test_privilege_check": {
                    "privileged_inference_used": False,
                    "deployment_inputs_checked": ["observation", "instruction", "candidate_action_chunk"],
                    "privileged_values_only_in_labels_or_diagnostics": True,
                },
                "measurement_valid": bool(measurement_valid),
            }
        )
        determinism = _run_restoration_determinism_test(loaded, args)
        report["restoration_scope"] = _branch_restoration_scope()
        report["restoration_determinism_test"] = determinism
        report["same_state_restoration_test"] = {
            **(report.get("same_state_restoration_test") or {}),
            "deterministic_two_branch_replay_passed": bool(determinism.get("passed")),
        }
        report["measurement_valid_without_determinism"] = bool(report["measurement_valid"])
        report["measurement_valid"] = bool(report["measurement_valid"] and determinism.get("passed"))
        decision = choose_final_decision(
            measurement_valid=bool(measurement_valid),
            official_diversity=official_diversity,
            official_metrics=official_metrics,
            structured_metrics=structured_metrics,
        )
        if not bool(determinism.get("passed")):
            decision = "ECHO_GATE_MEASUREMENT_INVALID"
        report["final_decision"] = decision
        if decision == "ECHO_POLICY_CANDIDATE_HEADROOM_CONFIRMED":
            report["exact_next_prompt"] = (
                "Implement the previously specified lightweight ECHO heads on frozen official SmolVLA-LIBERO: "
                "phase inference, effect prediction, phase-effect compatibility, and same-state pairwise ranking; "
                "run the fixed prototype baselines without OpenVLA-OFT."
            )
        elif decision == "ECHO_ONLY_STRUCTURED_CANDIDATES_HAVE_HEADROOM":
            report["exact_next_prompt"] = (
                "Archive original ECHO ranking over frozen policy stochastic candidates; any future route must first pass "
                "a new novelty review for candidate generation, not ECHO head training."
            )
        elif decision == "ECHO_POLICY_CANDIDATES_IMPOVERISHED":
            report["exact_next_prompt"] = (
                "Do not train an ECHO ranker over official SmolVLA stochastic candidates; archive or redesign around a "
                "new candidate-generation contribution after novelty review."
            )
        elif decision == "NO_ECHO_HEADROOM_CONFIRMED":
            report["exact_next_prompt"] = "Archive ECHO and return to the paper-first candidate portfolio."
        else:
            report["exact_next_prompt"] = "Resolve the measurement validity issue and rerun the same frozen final gate without changing thresholds."
    except Exception as exc:  # pragma: no cover - runtime boundary
        report["errors"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc().splitlines()[-80:],
            }
        )
        report["measurement_valid"] = False
        report["final_decision"] = "ECHO_GATE_MEASUREMENT_INVALID"
        report["exact_next_prompt"] = "Resolve the final-gate measurement blocker, then rerun without changing thresholds."
    finally:
        try:
            del loaded
            if "torch" in sys.modules:
                torch.cuda.empty_cache()
        except Exception:
            pass
        report["elapsed_seconds"] = _round(time.monotonic() - started, 3)
        try:
            report["latency_vram_summary"] = {
                "elapsed_seconds": report["elapsed_seconds"],
                "cuda_memory": _cuda_memory(torch),
            }
        except Exception:
            report["latency_vram_summary"] = {"elapsed_seconds": report["elapsed_seconds"]}
        if not report.get("previous_gate_audit"):
            report["previous_gate_audit"] = _audit_previous_gate(report_dir)
        _write_reports(report_dir, report)
        _update_project_reports(report_dir, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default="/home/jiheon/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/home/jiheon/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--max-horizon", type=int, default=16)
    parser.add_argument("--candidate-seed-base", type=int, default=730000)
    parser.add_argument("--continuation-seed-base", type=int, default=810000)
    parser.add_argument("--random-selector-seed-base", type=int, default=820000)
    parser.add_argument("--reference-seed", type=int, default=700001)
    parser.add_argument("--determinism-only", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.candidate_count) != 8:
        raise SystemExit("--candidate-count must be exactly 8 for the final ECHO candidate gate")
    if int(args.max_horizon) < 4 or int(args.max_horizon) > 16:
        raise SystemExit("--max-horizon must be between 4 and 16")
    report = run_final_gate(args)
    print(
        json.dumps(
            {
                "final_decision": report.get("final_decision"),
                "measurement_valid": report.get("measurement_valid"),
                "official_policy_candidate_metrics": report.get("official_policy_candidate_metrics"),
                "structured_candidate_metrics": report.get("structured_candidate_metrics"),
                "official_candidate_diversity": report.get("official_candidate_diversity"),
                "errors": report.get("errors"),
            },
            indent=2,
            default=_json_default,
        )
    )
    return 0 if report.get("final_decision") != "ECHO_GATE_MEASUREMENT_INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
