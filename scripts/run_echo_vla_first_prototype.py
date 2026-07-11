"""Bounded ECHO-VLA first-prototype runner.

The first command implemented here is the oracle candidate-headroom gate.  It
uses the frozen official SmolVLA policy only as a candidate proposer, restores
the same LIBERO state for each candidate chunk, and records realized effect
labels for the fixed horizon.  If this gate fails, no ECHO heads are trained.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.datasets.libero_zero_reward_rollout_diagnosis import (  # noqa: E402
    _best_object_key,
    _distance,
    _extract_eef,
    _extract_pos,
)
from tca_map.smolvla.echo_vla import (  # noqa: E402
    build_candidate_record,
    candidate_headroom_metrics,
    compatibility_score,
    serialize_groups,
    stable_hash,
    validate_counterfactual_group,
)
from tca_map.smolvla.exact_hard_slice_rollout import _make_exact_vector_env  # noqa: E402
from tca_map.smolvla.official_closed_loop_scaleup import (  # noqa: E402
    _json_default,
    _set_runtime_env,
)
from tca_map.smolvla.official_wsl_libero_rollout import (  # noqa: E402
    POLICIES,
    _cuda_memory,
    _load_policy_and_processors,
)


TASKS = [
    {
        "suite": "libero_spatial",
        "task_id": 0,
        "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    },
    {
        "suite": "libero_object",
        "task_id": 4,
        "instruction": "pick up the ketchup and place it in the basket",
    },
    {
        "suite": "libero_goal",
        "task_id": 0,
        "instruction": "open the middle drawer of the cabinet",
    },
    {
        "suite": "libero_10",
        "task_id": 0,
        "instruction": "put both the alphabet soup and the tomato sauce in the basket",
    },
]

RESET_IDENTITIES = [20260711, 20260712, 20260713, 20260714, 20260715]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _round(value: float | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _identity_to_initial_state_index(identity: int) -> int:
    if int(identity) not in RESET_IDENTITIES:
        raise ValueError(f"unknown reset identity {identity}")
    return RESET_IDENTITIES.index(int(identity))


def _compact_for_hash(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.size > 512:
            return {
                "array_hash": stable_hash(array),
                "shape": [int(dim) for dim in array.shape],
                "dtype": str(array.dtype),
            }
        return array.tolist()
    if isinstance(value, dict):
        return {str(key): _compact_for_hash(item) for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))}
    if isinstance(value, (list, tuple)):
        return [_compact_for_hash(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


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


def _tokens(text: str) -> set[str]:
    import re

    stop = {
        "the",
        "a",
        "an",
        "and",
        "in",
        "on",
        "of",
        "to",
        "it",
        "put",
        "pick",
        "up",
        "turn",
        "close",
        "open",
        "both",
        "between",
        "from",
        "with",
    }
    return {token for token in re.findall(r"[a-z0-9]+", text.lower().replace("_", " ")) if token not in stop and not token.isdigit()}


def _sim_state_hash(env: Any) -> str:
    raw_env = env.envs[0] if hasattr(env, "envs") else env
    sim = _find_attr_chain(raw_env, "sim")
    if sim is not None and hasattr(sim, "get_state"):
        state = sim.get_state()
        if hasattr(state, "flatten"):
            return stable_hash(np.asarray(state.flatten(), dtype=np.float64))
        if hasattr(state, "__dict__"):
            return stable_hash({key: np.asarray(value).tolist() for key, value in state.__dict__.items() if not key.startswith("_")})
        return stable_hash(str(state))
    state = _find_attr_chain(raw_env, "get_state")
    if callable(state):
        return stable_hash(state())
    raise RuntimeError("could not locate simulator state for exact-state hash")


def _sim_body_positions(env: Any) -> dict[str, list[float]]:
    raw_env = env.envs[0] if hasattr(env, "envs") else env
    sim = _find_attr_chain(raw_env, "sim")
    if sim is None:
        return {}
    positions: dict[str, list[float]] = {}
    try:
        for index in range(int(sim.model.nbody)):
            name = sim.model.body_id2name(index)
            if not name:
                continue
            if any(prefix in name for prefix in ("robot0_", "gripper0_", "mount0_", "world", "table")):
                continue
            arr = np.asarray(sim.data.body_xpos[index], dtype=np.float64).reshape(-1)
            if arr.size >= 3:
                positions[str(name)] = [float(value) for value in arr[:3]]
    except Exception:
        return {}
    return positions


def _best_body_key(body_positions: dict[str, list[float]], instruction: str) -> str | None:
    instruction_tokens = _tokens(instruction)
    scored = []
    for name in body_positions:
        name_tokens = _tokens(name.replace("_main", ""))
        overlap = instruction_tokens & name_tokens
        scored.append((len(overlap), name, sorted(overlap)))
    scored.sort(key=lambda item: (-item[0], item[1]))
    if not scored or scored[0][0] <= 0:
        return None
    return scored[0][1]


def _unwrap_object_array(value: Any) -> Any:
    if isinstance(value, np.ndarray) and value.dtype == object and value.size == 1:
        return value.reshape(-1)[0]
    if isinstance(value, list) and len(value) == 1:
        return value[0]
    return value


def _extract_nested_eef(obs: Any) -> list[float] | None:
    obs = _unwrap_object_array(obs)
    if not isinstance(obs, dict):
        return None
    robot_state = _unwrap_object_array(obs.get("robot_state"))
    if isinstance(robot_state, dict):
        eef = _unwrap_object_array(robot_state.get("eef"))
        if isinstance(eef, dict) and "pos" in eef:
            arr = np.asarray(eef["pos"], dtype=np.float64).reshape(-1)
            if arr.size >= 3:
                return [float(value) for value in arr[:3]]
    fallback = _extract_eef(obs)
    if fallback is not None:
        return fallback
    for value in obs.values():
        nested = _extract_nested_eef(value)
        if nested is not None:
            return nested
    return None


def _successes_from_info(info: dict[str, Any]) -> bool:
    try:
        final_info = info.get("final_info")
        if final_info is not None:
            return bool(np.asarray(final_info["is_success"]).reshape(-1)[0])
    except Exception:
        pass
    return False


def _preprocess_batch(env: Any, observation: Any, loaded: dict[str, Any]) -> Any:
    from lerobot.scripts.lerobot_eval import add_envs_task, preprocess_observation

    lerobot_observation = preprocess_observation(observation)
    lerobot_observation = add_envs_task(env, lerobot_observation)
    lerobot_observation = loaded["env_preprocessor"](lerobot_observation)
    return loaded["preprocessor"](lerobot_observation)


def _postprocess_action(action: Any, loaded: dict[str, Any]) -> np.ndarray:
    from lerobot.scripts.lerobot_eval import ACTION

    action = loaded["postprocessor"](action)
    transition = {ACTION: action}
    transition = loaded["env_postprocessor"](transition)
    action = transition[ACTION]
    return np.asarray(action.to("cpu").numpy(), dtype=np.float64).reshape(1, -1)


def _default_chunk(policy: Any, env: Any, observation: Any, loaded: dict[str, Any], horizon: int) -> np.ndarray:
    import torch

    if hasattr(policy, "reset"):
        policy.reset()
    batch = _preprocess_batch(env, observation, loaded)
    rows = []
    with torch.inference_mode():
        for _ in range(horizon):
            action = policy.select_action(batch)
            rows.append(_postprocess_action(action, loaded).reshape(-1))
    return np.stack(rows, axis=0)


def _candidate_chunks(default_chunk: np.ndarray, candidate_count: int, perturb_scale: float) -> list[dict[str, Any]]:
    chunks = [{"index": 0, "source": "frozen_smolvla_default_candidate", "chunk": np.asarray(default_chunk, dtype=np.float64)}]
    if candidate_count <= 1:
        return chunks
    directions = []
    base = np.zeros_like(default_chunk, dtype=np.float64)
    base[:, 0] = 1.0
    directions.append(base)
    base = np.zeros_like(default_chunk, dtype=np.float64)
    base[:, 1] = 1.0
    directions.append(base)
    base = np.zeros_like(default_chunk, dtype=np.float64)
    base[:, 2] = 1.0
    directions.append(base)
    base = np.zeros_like(default_chunk, dtype=np.float64)
    base[:, :3] = -1.0
    directions.append(base)
    for index in range(1, candidate_count):
        direction = directions[(index - 1) % len(directions)]
        sign = -1.0 if index % 2 == 0 else 1.0
        chunk = np.asarray(default_chunk, dtype=np.float64) + sign * perturb_scale * direction
        chunks.append({"index": index, "source": "bounded_default_chunk_perturbation", "chunk": np.clip(chunk, -1.0, 1.0)})
    return chunks


def _effect_from_observations(
    start_obs: Any,
    final_obs: Any,
    action_chunk: np.ndarray,
    instruction: str,
    task_success: bool,
    start_body_positions: dict[str, list[float]] | None = None,
    final_body_positions: dict[str, list[float]] | None = None,
) -> dict[str, float]:
    start_eef = _extract_nested_eef(start_obs)
    final_eef = _extract_nested_eef(final_obs)
    target_key = None
    start_target = None
    final_target = None
    if start_body_positions and final_body_positions:
        target_key = _best_body_key(start_body_positions, instruction)
        if target_key is not None:
            start_target = start_body_positions.get(target_key)
            final_target = final_body_positions.get(target_key)
    if start_target is None or final_target is None:
        target_audit = _best_object_key(start_obs, instruction)
        target_key = target_audit.get("best_key")
        start_target = _extract_pos(start_obs, target_key)
        final_target = _extract_pos(final_obs, target_key)

    eef_delta = 0.0
    if start_eef is not None and final_eef is not None:
        eef_delta = float(np.linalg.norm(np.asarray(final_eef) - np.asarray(start_eef)))

    start_dist = _distance(start_eef, start_target)
    final_dist = _distance(final_eef, final_target)
    distance_delta = 0.0 if start_dist is None or final_dist is None else float(start_dist - final_dist)

    object_displacement = 0.0
    object_lift = 0.0
    object_retained = 0.0
    placement_alignment = 0.0
    if start_target is not None and final_target is not None:
        start_arr = np.asarray(start_target, dtype=np.float64)
        final_arr = np.asarray(final_target, dtype=np.float64)
        object_displacement = float(np.linalg.norm(final_arr - start_arr))
        object_lift = float(final_arr[2] - start_arr[2])
        if final_eef is not None and _distance(final_eef, final_target) is not None:
            object_retained = 1.0 if float(_distance(final_eef, final_target)) < 0.10 else 0.0
        placement_alignment = max(0.0, min(1.0, object_displacement / 0.08))

    final_contact_like = final_dist is not None and float(final_dist) < 0.08
    start_contact_like = start_dist is not None and float(start_dist) < 0.08
    contact_transition = float(final_contact_like) - float(start_contact_like)
    gripper_transition = float(np.mean(action_chunk[:, -1])) if action_chunk.shape[1] >= 7 else 0.0
    release_stability = 1.0 if bool(task_success) else 0.0

    return {
        "eef_delta_norm": eef_delta,
        "target_distance_delta": distance_delta,
        "contact_transition": contact_transition,
        "gripper_transition": gripper_transition,
        "object_retained": object_retained,
        "object_lift_delta": object_lift,
        "object_goal_delta": object_displacement,
        "placement_alignment": placement_alignment,
        "release_stability": release_stability,
    }


def _phase_for_initial_chunk(task: dict[str, Any]) -> str:
    instruction = str(task.get("instruction", "")).lower()
    if instruction.startswith("open "):
        return "approach"
    return "approach"


def _run_one_candidate(
    *,
    task: dict[str, Any],
    reset_identity: int,
    initial_state_index: int,
    candidate: dict[str, Any],
    horizon: int,
    phase: str,
) -> tuple[Any, dict[str, Any]]:
    env = None
    try:
        env = _make_exact_vector_env(str(task["suite"]), int(task["task_id"]), int(initial_state_index))
        observation, _ = env.reset(seed=[int(reset_identity)])
        start_hash = _sim_state_hash(env)
        start_body_positions = _sim_body_positions(env)
        start_obs_hash_payload = _compact_for_hash(observation)
        start_observation_copy = observation
        done = np.array([False])
        task_success = False
        terminated_last = False
        truncated_last = False
        final_observation = observation
        for row in np.asarray(candidate["chunk"], dtype=np.float64)[:horizon]:
            action = row.reshape(1, -1)
            final_observation, reward, terminated, truncated, info = env.step(action)
            task_success = bool(task_success or _successes_from_info(info))
            terminated_last = bool(np.asarray(terminated).reshape(-1)[0])
            truncated_last = bool(np.asarray(truncated).reshape(-1)[0])
            done = terminated | truncated | done
            if np.all(done):
                break
        final_body_positions = _sim_body_positions(env)
        effect = _effect_from_observations(
            start_observation_copy,
            final_observation,
            np.asarray(candidate["chunk"])[:horizon],
            str(task["instruction"]),
            task_success,
            start_body_positions,
            final_body_positions,
        )
        record = build_candidate_record(
            group_id=f"{task['suite']}|task_{task['task_id']}|identity_{reset_identity}",
            candidate_index=int(candidate["index"]),
            start_state=start_hash,
            start_observation=start_obs_hash_payload,
            action_chunk=np.asarray(candidate["chunk"], dtype=np.float64)[:horizon],
            horizon=horizon,
            phase=phase,
            realized_effect=effect,
            success=compatibility_score(effect, phase) > 0.05,
            terminated=terminated_last,
            truncated=truncated_last,
            source=str(candidate["source"]),
        )
        row = {
            "candidate": record.to_json(),
            "task_success": bool(task_success),
            "effect_compatibility": record.compatibility(),
            "start_state_hash_raw": start_hash,
            "target_body_key": _best_body_key(start_body_positions, str(task["instruction"])) if start_body_positions else None,
            "start_body_position_count": len(start_body_positions),
            "effect_available_components": [
                key
                for key, value in effect.items()
                if key != "gripper_transition" and abs(float(value)) > 1e-9
            ],
        }
        return record, row
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _initial_observation_for_policy(task: dict[str, Any], reset_identity: int, initial_state_index: int) -> tuple[Any, Any, str]:
    env = _make_exact_vector_env(str(task["suite"]), int(task["task_id"]), int(initial_state_index))
    try:
        observation, _ = env.reset(seed=[int(reset_identity)])
        return env, observation, _sim_state_hash(env)
    except Exception:
        try:
            env.close()
        except Exception:
            pass
        raise


def write_plan(report_dir: Path, args: argparse.Namespace) -> None:
    lines = [
        "# ECHO-VLA First Prototype Plan",
        "",
        "Status: candidate-headroom gate first. No ECHO heads are trained unless this gate passes.",
        "",
        "## Scope",
        "",
        "- backbone: frozen official SmolVLA-LIBERO",
        f"- max tasks in first gate: `{args.max_tasks}`",
        f"- reset identities per task: `{args.max_states_per_task}`",
        f"- candidate count: `{args.candidate_count}`",
        f"- horizon: `{args.horizon}`",
        "- OpenVLA-OFT: not used",
        "- full benchmark: not run",
        "- full SmolVLA backbone training: not allowed",
        "",
        "## Baselines To Use After Headroom",
        "",
        "1. frozen_smolvla_default_candidate",
        "2. random_candidate_selector",
        "3. simple_phase_predicate_heuristic",
        "4. direct_success_or_value_head",
        "5. pre_vla_style_validity_advantage_proxy",
        "6. echo_no_counterfactual",
        "7. echo_no_phase",
        "8. echo_full",
    ]
    _write_md(report_dir / "echo_vla_first_prototype_plan.md", lines)


def write_decision_reports(report_dir: Path, report: dict[str, Any]) -> None:
    headroom = report.get("candidate_headroom") or {}
    decision = str(report.get("final_decision"))
    result_lines = [
        "# ECHO-VLA First Prototype Result",
        "",
        f"- final decision: `{decision}`",
        f"- novelty gate: `{report.get('novelty_gate')}`",
        f"- candidate headroom ran: `{report.get('candidate_headroom_ran')}`",
        f"- candidate headroom passed: `{headroom.get('passes_headroom_gate')}`",
        f"- oracle improvement pp: `{headroom.get('oracle_improvement_pp')}`",
        f"- default-failure recoverable rate: `{headroom.get('default_failure_recoverable_rate')}`",
        f"- data generated groups: `{headroom.get('group_count')}`",
        f"- components trained: `{report.get('components_trained')}`",
        f"- closed-loop evaluation run: `{report.get('closed_loop_evaluation_run')}`",
        f"- latency/VRAM: `{report.get('latency_vram_summary')}`",
        "",
        "## Blocker Or Kill Reason",
        "",
        str(report.get("blocker_or_kill_reason")),
    ]
    _write_json(report_dir / "echo_vla_first_prototype_result.json", report)
    _write_md(report_dir / "echo_vla_first_prototype_result.md", result_lines)
    decision_lines = [
        "# ECHO-VLA First Prototype Decision",
        "",
        f"Final decision: `{decision}`",
        "",
        "## Basis",
        "",
        f"- novelty adjudication: `{report.get('novelty_gate')}`",
        f"- candidate oracle headroom: `{headroom}`",
        f"- data generated: `{report.get('data_generated')}`",
        f"- components trained: `{report.get('components_trained')}`",
        f"- prototype baselines: `{report.get('prototype_baselines')}`",
        f"- closed-loop results: `{report.get('closed_loop_results')}`",
        f"- effect/ranking results: `{report.get('effect_ranking_results')}`",
        f"- latency/VRAM: `{report.get('latency_vram_summary')}`",
        "",
        "## Exact Next Step",
        "",
        str(report.get("exact_next_step")),
    ]
    _write_md(report_dir / "echo_vla_first_prototype_decision.md", decision_lines)


def write_candidate_headroom_report(report_dir: Path, report: dict[str, Any]) -> None:
    headroom = report.get("candidate_headroom") or {}
    lines = [
        "# ECHO-VLA Candidate Headroom Result",
        "",
        f"- ran: `{report.get('candidate_headroom_ran')}`",
        f"- passed: `{headroom.get('passes_headroom_gate')}`",
        f"- group count: `{headroom.get('group_count')}`",
        f"- default success rate: `{headroom.get('default_success_rate')}`",
        f"- oracle success rate: `{headroom.get('oracle_success_rate')}`",
        f"- oracle improvement pp: `{headroom.get('oracle_improvement_pp')}`",
        f"- default failure recoverable rate: `{headroom.get('default_failure_recoverable_rate')}`",
        f"- hard kill reason: `{headroom.get('hard_kill_reason')}`",
        "",
        "The oracle is diagnostic only. It uses realized effects after executing all same-state candidates and is not available at deployment.",
    ]
    _write_md(report_dir / "echo_vla_candidate_headroom_result.md", lines)


def run_headroom(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    report_dir = Path(args.report_dir)
    write_plan(report_dir, args)
    report: dict[str, Any] = {
        "schema_version": "echo_vla_first_prototype_result_v1",
        "date_kst": "2026-07-11",
        "branch": "codex/implement-echo-vla-first-prototype",
        "novelty_gate": "ECHO_NOVELTY_SURVIVES_TARGETED_GATE",
        "policy": {
            "official_smolvla_used": True,
            "frozen_backbone": True,
            "openvla_oft_used": False,
            "full_benchmark_run": False,
            "full_smolvla_backbone_training": False,
            "downloads_performed": False,
            "same_state_interventions_required": True,
            "privileged_inference_used": False,
        },
        "candidate_headroom_ran": False,
        "candidate_headroom": {},
        "groups": [],
        "group_proofs": [],
        "errors": [],
        "data_generated": None,
        "components_trained": "none_headroom_gate_first",
        "prototype_baselines": [
            "frozen_smolvla_default_candidate",
            "random_candidate_selector",
            "simple_phase_predicate_heuristic",
            "direct_success_or_value_head",
            "pre_vla_style_validity_advantage_proxy",
            "echo_no_counterfactual",
            "echo_no_phase",
            "echo_full",
        ],
        "closed_loop_evaluation_run": False,
        "closed_loop_results": "not_run_headroom_gate_first",
        "effect_ranking_results": "not_trained",
        "latency_vram_summary": {},
        "final_decision": "ECHO_IMPLEMENTATION_BLOCKED",
        "blocker_or_kill_reason": None,
        "exact_next_step": None,
    }
    loaded = None
    policy_env = None
    try:
        _set_runtime_env(args)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for official SmolVLA candidate proposal")
        spec = next(item for item in POLICIES if item.name == "frozen_base")
        loaded = _load_policy_and_processors(args, spec)
        tasks = TASKS[: int(args.max_tasks)]
        reset_identities = RESET_IDENTITIES[: int(args.max_states_per_task)]
        groups = []
        row_groups = []
        for task in tasks:
            phase = _phase_for_initial_chunk(task)
            for reset_identity in reset_identities:
                initial_state_index = _identity_to_initial_state_index(reset_identity)
                policy_env, observation, policy_start_hash = _initial_observation_for_policy(task, reset_identity, initial_state_index)
                default = _default_chunk(loaded["policy"], policy_env, observation, loaded, int(args.horizon))
                try:
                    policy_env.close()
                except Exception:
                    pass
                policy_env = None
                candidates = _candidate_chunks(default, int(args.candidate_count), float(args.perturb_scale))
                records = []
                rows = []
                for candidate in candidates:
                    record, row = _run_one_candidate(
                        task=task,
                        reset_identity=reset_identity,
                        initial_state_index=initial_state_index,
                        candidate=candidate,
                        horizon=int(args.horizon),
                        phase=phase,
                    )
                    row["policy_start_state_hash"] = policy_start_hash
                    records.append(record)
                    rows.append(row)
                proof = validate_counterfactual_group(records)
                report["group_proofs"].append(proof)
                groups.append(records)
                row_groups.append(rows)

        metrics = candidate_headroom_metrics(groups)
        report["candidate_headroom_ran"] = True
        report["candidate_headroom"] = metrics
        report["groups"] = serialize_groups(groups)
        report["candidate_rows"] = row_groups
        report["data_generated"] = {
            "same_state_intervention_groups": len(groups),
            "candidate_records": sum(len(group) for group in groups),
            "tasks": tasks,
            "reset_identities": reset_identities,
            "candidate_count": int(args.candidate_count),
            "horizon": int(args.horizon),
        }
        report["latency_vram_summary"] = {
            "elapsed_seconds": _round(time.monotonic() - started, 3),
            "cuda_memory": _cuda_memory(torch),
        }
        if metrics["passes_headroom_gate"]:
            report["final_decision"] = "WEAK_ECHO_SIGNAL_NEEDS_ONE_REPEAT"
            report["blocker_or_kill_reason"] = "candidate headroom passed; lightweight ECHO training/evaluation still pending"
            report["exact_next_step"] = "Train lightweight ECHO heads and run the fixed prototype baselines; do not use OpenVLA-OFT."
        else:
            report["final_decision"] = "NO_ECHO_CANDIDATE_HEADROOM"
            report["blocker_or_kill_reason"] = metrics["hard_kill_reason"]
            report["exact_next_step"] = "Stop ECHO implementation or redesign candidate generation/effect representation before training."
    except Exception as exc:  # pragma: no cover - runtime boundary
        report["errors"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc().splitlines()[-40:],
            }
        )
        report["final_decision"] = "ECHO_IMPLEMENTATION_BLOCKED"
        report["blocker_or_kill_reason"] = f"{type(exc).__name__}: {exc}"
        report["exact_next_step"] = "Resolve the runtime/data/schema blocker, then rerun the same candidate-headroom gate without changing thresholds."
    finally:
        try:
            if policy_env is not None:
                policy_env.close()
        except Exception:
            pass
        try:
            del loaded
            if "torch" in sys.modules:
                torch.cuda.empty_cache()
        except Exception:
            pass
        report["elapsed_seconds"] = _round(time.monotonic() - started, 3)
        if not report.get("latency_vram_summary"):
            try:
                report["latency_vram_summary"] = {"elapsed_seconds": report["elapsed_seconds"], "cuda_memory": _cuda_memory(torch)}
            except Exception:
                report["latency_vram_summary"] = {"elapsed_seconds": report["elapsed_seconds"]}
        write_candidate_headroom_report(report_dir, report)
        write_decision_reports(report_dir, report)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["headroom"], nargs="?", default="headroom")
    parser.add_argument("--base-path", default="/home/jiheon/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/home/jiheon/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--report-dir", default="reports")
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--max-states-per-task", type=int, default=2)
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--perturb-scale", type=float, default=0.025)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if int(args.max_tasks) < 1 or int(args.max_tasks) > len(TASKS):
        raise SystemExit("--max-tasks must be between 1 and 4")
    if int(args.max_states_per_task) < 1 or int(args.max_states_per_task) > len(RESET_IDENTITIES):
        raise SystemExit("--max-states-per-task must be between 1 and 5")
    if int(args.candidate_count) < 2 or int(args.candidate_count) > 8:
        raise SystemExit("--candidate-count must be between 2 and 8")
    if int(args.horizon) < 1 or int(args.horizon) > 16:
        raise SystemExit("--horizon must be between 1 and 16")
    report = run_headroom(args)
    print(json.dumps({"final_decision": report.get("final_decision"), "candidate_headroom": report.get("candidate_headroom")}, indent=2, default=_json_default))
    return 0 if report.get("final_decision") != "ECHO_IMPLEMENTATION_BLOCKED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
