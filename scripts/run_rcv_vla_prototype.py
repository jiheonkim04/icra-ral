"""RCV-VLA prototype runner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_echo_vla_first_prototype import _postprocess_action, _preprocess_batch  # noqa: E402
from scripts.run_phase_barrier_vla_prototype import _make_exact_vector_env, _round, _set_runtime_env, _step_success  # noqa: E402
from tca_map.smolvla.official_closed_loop_scaleup import _json_default  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import POLICIES, _cuda_memory, _load_policy_and_processors  # noqa: E402
from tca_map.smolvla.rcv_vla import (  # noqa: E402
    RCVConfig,
    TASK_KEYS,
    VARIANTS,
    action_disagreement,
    build_rcv_features,
    load_verifier,
    predict_replan_probability,
    save_verifier,
    train_verifier,
    verifier_replans,
)


DATE_KST = "2026-07-13"
BRANCH = "codex/autonomous-until-paper-governance-v2"
PROPOSAL_HASH = "86044E841D178DB5AA485B7D12B01FF8E4274CBDFDCDAC7D427477BF0646F26F"
RESET_IDENTITY_BASE = 20260801
MAX_OFFICIAL_INITIAL_STATE_COUNT = 50
TASKS = [
    {"suite": "libero_spatial", "task_id": 4, "role": "stable_grasp_contact_transition"},
    {"suite": "libero_10", "task_id": 4, "role": "long_horizon_contact_and_release"},
]
STAGE_0_IDENTITIES = list(range(20260801, 20260806))
TRAIN_IDENTITIES = list(range(20260806, 20260813))
CALIBRATION_IDENTITIES = list(range(20260813, 20260816))
STAGE_2A_IDENTITIES = list(range(20260816, 20260821))
STAGE_2B_IDENTITIES = list(range(20260821, 20260841))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: list[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, sort_keys=True, default=_json_default) for row in rows) + "\n", encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_md(path: Path, title: str, report: Mapping[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        f"- mode: `{report.get('mode')}`",
        f"- closed-loop experiment happened: `{report.get('closed_loop_experiment_happened')}`",
        f"- summary: `{report.get('summary')}`",
        f"- elapsed seconds: `{report.get('elapsed_seconds')}`",
        "",
        f"Next step: {report.get('next_step')}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _cuda_memory_report() -> dict[str, Any]:
    import torch

    return _cuda_memory(torch)


def _identity_to_initial_state_index(identity: int) -> int:
    index = int(identity) - RESET_IDENTITY_BASE
    if index < 0 or index >= MAX_OFFICIAL_INITIAL_STATE_COUNT:
        raise ValueError(f"identity {identity} maps to invalid official initial state index {index}")
    return index


def _task_key(row: Mapping[str, Any]) -> str:
    return f"{row['suite']}/task_{int(row['task_id'])}"


def _batch_state(batch: Mapping[str, Any]) -> np.ndarray:
    value = batch.get("observation.state")
    if value is None:
        for key, item in batch.items():
            if str(key).endswith(".state"):
                value = item
                break
    if value is None:
        raise KeyError("could not locate observation state in SmolVLA batch")
    if hasattr(value, "detach"):
        value = value.detach().to("cpu").numpy()
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    if array.size != 8:
        raise ValueError(f"RCV expected 8D state, got {array.size}")
    return array


def _policy_action_stateless(policy: Any, batch: Mapping[str, Any], loaded: Mapping[str, Any]) -> np.ndarray:
    import torch

    with torch.inference_mode():
        if hasattr(policy, "predict_action_chunk"):
            action_chunk = policy.predict_action_chunk(dict(batch))
            action = action_chunk[:, 0] if getattr(action_chunk, "ndim", 0) == 3 else action_chunk
        else:
            action = policy.select_action(dict(batch))
    return _postprocess_action(action, dict(loaded)).reshape(-1)


def _policy_action_chunk_stateless(policy: Any, batch: Mapping[str, Any], loaded: Mapping[str, Any]) -> np.ndarray:
    import torch

    with torch.inference_mode():
        if hasattr(policy, "predict_action_chunk"):
            action_chunk = policy.predict_action_chunk(dict(batch))
        else:
            action_chunk = policy.select_action(dict(batch))
    if getattr(action_chunk, "ndim", 0) == 2:
        return _postprocess_action(action_chunk, dict(loaded)).reshape(1, -1)
    rows = []
    for index in range(int(action_chunk.shape[1])):
        rows.append(_postprocess_action(action_chunk[:, index], dict(loaded)).reshape(-1))
    return np.stack(rows, axis=0).astype(np.float64)


def _planned_rows(tasks: list[Mapping[str, Any]], identities: list[int], variants: tuple[str, ...]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for task in tasks:
            for identity in identities:
                rows.append(
                    {
                        "variant": variant,
                        "suite": str(task["suite"]),
                        "task_id": int(task["task_id"]),
                        "task_key": _task_key(task),
                        "role": str(task["role"]),
                        "identity": int(identity),
                    }
                )
    return rows


def _episode_context(row: Mapping[str, Any]) -> tuple[Any, Any]:
    env = _make_exact_vector_env(str(row["suite"]), int(row["task_id"]), _identity_to_initial_state_index(int(row["identity"])))
    observation, _ = env.reset(seed=[int(row["identity"])])
    return env, observation


def _max_steps(env: Any, max_eval_steps: int) -> int:
    steps = int(env.call("_max_episode_steps")[0])
    if int(max_eval_steps) > 0:
        steps = min(steps, int(max_eval_steps))
    return steps


def _step_env(env: Any, action: np.ndarray) -> tuple[Any, bool, bool, float, Any]:
    observation, reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float64).reshape(1, -1))
    success = bool(_step_success(info))
    done = bool(success or np.all(terminated | truncated))
    reward_value = float(np.asarray(reward).reshape(-1)[0])
    return observation, success, done, reward_value, info


def _run_queued_episode(
    *,
    row: Mapping[str, Any],
    loaded: Mapping[str, Any],
    max_eval_steps: int,
) -> dict[str, Any]:
    env = None
    started = time.time()
    disagreements: list[float] = []
    policy_latencies: list[float] = []
    rewards: list[float] = []
    success = False
    try:
        env, observation = _episode_context(row)
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        horizon = int((loaded.get("audit") or {}).get("action_chunk_shape", [1, 50, 7])[1])
        for step in range(_max_steps(env, max_eval_steps)):
            batch = _preprocess_batch(env, observation, dict(loaded))
            start_policy = time.perf_counter()
            action = policy.select_action(dict(batch))
            queued = _postprocess_action(action, dict(loaded)).reshape(-1)
            fresh = _policy_action_stateless(policy, batch, loaded)
            policy_latencies.append(time.perf_counter() - start_policy)
            disagreements.append(action_disagreement(queued, fresh))
            observation, step_success, done, reward_value, _info = _step_env(env, queued)
            rewards.append(reward_value)
            success = bool(success or step_success)
            if done:
                break
        steps = int(step + 1 if "step" in locals() else 0)
        heavy_calls = int(np.ceil(steps / max(1, horizon)))
        return {
            **dict(row),
            "success": bool(success),
            "exception": None,
            "episode_steps": steps,
            "reward_sum": _round(float(np.sum(rewards)) if rewards else 0.0, 6),
            "mean_disagreement": _round(float(np.mean(disagreements)) if disagreements else 0.0, 6),
            "max_disagreement": _round(float(np.max(disagreements)) if disagreements else 0.0, 6),
            "replan_count": 0,
            "replan_rate": 0.0,
            "heavy_policy_call_count": heavy_calls,
            "heavy_policy_calls_per_step": _round(heavy_calls / max(1, steps), 6),
            "policy_latency_mean_s": _round(float(np.mean(policy_latencies)) if policy_latencies else 0.0, 6),
            "elapsed_seconds": _round(time.time() - started, 3),
            "cuda_memory": _cuda_memory_report(),
        }
    except Exception as exc:  # pragma: no cover
        return {**dict(row), "success": False, "exception": "".join(traceback.format_exception_only(type(exc), exc)).strip(), "episode_steps": 0, "elapsed_seconds": _round(time.time() - started, 3), "cuda_memory": _cuda_memory_report()}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _run_external_queue_episode(
    *,
    row: Mapping[str, Any],
    loaded: Mapping[str, Any],
    max_eval_steps: int,
    tau_train: float | None,
    verifier: Mapping[str, Any] | None,
) -> dict[str, Any]:
    env = None
    started = time.time()
    policy_latencies: list[float] = []
    disagreements: list[float] = []
    probabilities: list[float] = []
    rewards: list[float] = []
    replan_count = 0
    heavy_policy_calls = 0
    success = False
    previous_action = np.zeros(7, dtype=np.float64)
    chunk: np.ndarray | None = None
    chunk_index = 0
    try:
        env, observation = _episode_context(row)
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        for step in range(_max_steps(env, max_eval_steps)):
            batch = _preprocess_batch(env, observation, dict(loaded))
            variant = str(row["variant"])
            if variant == "stateless_first_action":
                start_policy = time.perf_counter()
                chunk = _policy_action_chunk_stateless(policy, batch, loaded)
                policy_latencies.append(time.perf_counter() - start_policy)
                heavy_policy_calls += 1
                chunk_index = 0
                action = chunk[0]
                replan_count += 1
            else:
                if chunk is None or chunk_index >= int(chunk.shape[0]):
                    start_policy = time.perf_counter()
                    chunk = _policy_action_chunk_stateless(policy, batch, loaded)
                    policy_latencies.append(time.perf_counter() - start_policy)
                    heavy_policy_calls += 1
                    chunk_index = 0
                queued = np.asarray(chunk[chunk_index], dtype=np.float64).reshape(-1)
                rho = float(chunk_index) / max(1.0, float(chunk.shape[0]))
                if variant == "sv_deviation_proxy":
                    start_policy = time.perf_counter()
                    fresh_chunk = _policy_action_chunk_stateless(policy, batch, loaded)
                    policy_latencies.append(time.perf_counter() - start_policy)
                    heavy_policy_calls += 1
                    disagreement = action_disagreement(queued, fresh_chunk[0])
                    disagreements.append(disagreement)
                    if disagreement > float(tau_train):
                        chunk = fresh_chunk
                        chunk_index = 0
                        queued = np.asarray(chunk[0], dtype=np.float64).reshape(-1)
                        replan_count += 1
                    action = queued
                elif variant in {"rcv_full", "rcv_no_context_ablation"}:
                    if verifier is None:
                        raise ValueError(f"missing verifier for {variant}")
                    probability = predict_replan_probability(
                        verifier,
                        state=_batch_state(batch),
                        queued_action=queued,
                        previous_action=previous_action,
                        chunk_index_fraction=rho,
                        task_key=str(row["task_key"]),
                    )
                    probabilities.append(probability)
                    if verifier_replans(verifier, probability):
                        start_policy = time.perf_counter()
                        chunk = _policy_action_chunk_stateless(policy, batch, loaded)
                        policy_latencies.append(time.perf_counter() - start_policy)
                        heavy_policy_calls += 1
                        chunk_index = 0
                        queued = np.asarray(chunk[0], dtype=np.float64).reshape(-1)
                        replan_count += 1
                    action = queued
                else:
                    raise ValueError(f"unknown external-queue RCV variant: {variant}")
            observation, step_success, done, reward_value, _info = _step_env(env, action)
            rewards.append(reward_value)
            success = bool(success or step_success)
            previous_action = np.asarray(action, dtype=np.float64).reshape(-1)
            chunk_index += 1
            if done:
                break
        steps = int(step + 1 if "step" in locals() else 0)
        return {
            **dict(row),
            "success": bool(success),
            "exception": None,
            "episode_steps": steps,
            "reward_sum": _round(float(np.sum(rewards)) if rewards else 0.0, 6),
            "mean_disagreement": _round(float(np.mean(disagreements)) if disagreements else 0.0, 6),
            "max_disagreement": _round(float(np.max(disagreements)) if disagreements else 0.0, 6),
            "mean_verifier_probability": _round(float(np.mean(probabilities)) if probabilities else 0.0, 6),
            "replan_count": int(replan_count),
            "replan_rate": _round(replan_count / max(1, steps), 6),
            "heavy_policy_call_count": int(heavy_policy_calls),
            "heavy_policy_calls_per_step": _round(heavy_policy_calls / max(1, steps), 6),
            "policy_latency_mean_s": _round(float(np.mean(policy_latencies)) if policy_latencies else 0.0, 6),
            "elapsed_seconds": _round(time.time() - started, 3),
            "cuda_memory": _cuda_memory_report(),
        }
    except Exception as exc:  # pragma: no cover
        return {**dict(row), "success": False, "exception": "".join(traceback.format_exception_only(type(exc), exc)).strip(), "episode_steps": 0, "elapsed_seconds": _round(time.time() - started, 3), "cuda_memory": _cuda_memory_report()}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _run_episode(
    *,
    row: Mapping[str, Any],
    loaded: Mapping[str, Any],
    max_eval_steps: int,
    tau_train: float | None,
    full_verifier: Mapping[str, Any] | None,
    no_context_verifier: Mapping[str, Any] | None,
) -> dict[str, Any]:
    variant = str(row["variant"])
    if variant == "queued_frozen_smolvla":
        return _run_queued_episode(row=row, loaded=loaded, max_eval_steps=max_eval_steps)
    verifier = None
    if variant == "rcv_full":
        verifier = full_verifier
    elif variant == "rcv_no_context_ablation":
        verifier = no_context_verifier
    return _run_external_queue_episode(row=row, loaded=loaded, max_eval_steps=max_eval_steps, tau_train=tau_train, verifier=verifier)


def _acquire_episode_records(
    *,
    row: Mapping[str, Any],
    split: str,
    loaded: Mapping[str, Any],
    max_eval_steps: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    env = None
    started = time.time()
    records: list[dict[str, Any]] = []
    rewards: list[float] = []
    success = False
    previous_action = np.zeros(7, dtype=np.float64)
    chunk: np.ndarray | None = None
    chunk_index = 0
    heavy_calls = 0
    disagreements: list[float] = []
    try:
        env, observation = _episode_context(row)
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        for step in range(_max_steps(env, max_eval_steps)):
            batch = _preprocess_batch(env, observation, dict(loaded))
            if chunk is None or chunk_index >= int(chunk.shape[0]):
                chunk = _policy_action_chunk_stateless(policy, batch, loaded)
                chunk_index = 0
                heavy_calls += 1
            fresh_chunk = _policy_action_chunk_stateless(policy, batch, loaded)
            heavy_calls += 1
            queued = np.asarray(chunk[chunk_index], dtype=np.float64).reshape(-1)
            fresh = np.asarray(fresh_chunk[0], dtype=np.float64).reshape(-1)
            disagreement = action_disagreement(queued, fresh)
            disagreements.append(disagreement)
            state = _batch_state(batch)
            rho = float(chunk_index) / max(1.0, float(chunk.shape[0]))
            _ = build_rcv_features(
                state=state,
                queued_action=queued,
                previous_action=previous_action,
                chunk_index_fraction=rho,
                task_key=str(row["task_key"]),
                include_context=True,
            )
            records.append(
                {
                    "split": split,
                    "suite": str(row["suite"]),
                    "task_id": int(row["task_id"]),
                    "task_key": str(row["task_key"]),
                    "identity": int(row["identity"]),
                    "step": int(step),
                    "state": state.tolist(),
                    "queued_action": queued.tolist(),
                    "fresh_action": fresh.tolist(),
                    "previous_action": previous_action.tolist(),
                    "chunk_index": int(chunk_index),
                    "chunk_horizon": int(chunk.shape[0]),
                    "chunk_index_fraction": float(rho),
                    "disagreement": float(disagreement),
                }
            )
            observation, step_success, done, reward_value, _info = _step_env(env, queued)
            rewards.append(reward_value)
            success = bool(success or step_success)
            previous_action = queued
            chunk_index += 1
            if done:
                break
        summary = {
            **dict(row),
            "split": split,
            "success": bool(success),
            "exception": None,
            "episode_steps": int(step + 1 if "step" in locals() else 0),
            "record_count": len(records),
            "reward_sum": _round(float(np.sum(rewards)) if rewards else 0.0, 6),
            "mean_disagreement": _round(float(np.mean(disagreements)) if disagreements else 0.0, 6),
            "max_disagreement": _round(float(np.max(disagreements)) if disagreements else 0.0, 6),
            "heavy_policy_call_count": int(heavy_calls),
            "elapsed_seconds": _round(time.time() - started, 3),
            "cuda_memory": _cuda_memory_report(),
        }
        return records, summary
    except Exception as exc:  # pragma: no cover
        summary = {**dict(row), "split": split, "success": False, "exception": "".join(traceback.format_exception_only(type(exc), exc)).strip(), "episode_steps": 0, "record_count": len(records), "elapsed_seconds": _round(time.time() - started, 3), "cuda_memory": _cuda_memory_report()}
        return records, summary
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _summarize(rows: list[Mapping[str, Any]], variants: tuple[str, ...]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in variants:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        successes = int(sum(1 for row in variant_rows if row.get("success")))
        total = len(variant_rows)
        per_task: dict[str, Any] = {}
        for key in sorted({str(row.get("task_key")) for row in variant_rows}):
            task_rows = [row for row in variant_rows if row.get("task_key") == key]
            task_success = int(sum(1 for row in task_rows if row.get("success")))
            per_task[key] = {"successes": task_success, "total": len(task_rows), "rate": _round(task_success / max(1, len(task_rows)), 6)}
        task_balanced = float(np.mean([item["rate"] for item in per_task.values()])) if per_task else 0.0
        by_variant[variant] = {
            "successes": successes,
            "total": total,
            "success_rate": _round(successes / max(1, total), 6),
            "task_balanced_success_rate": _round(task_balanced, 6),
            "per_task": per_task,
            "exceptions": int(sum(1 for row in variant_rows if row.get("exception"))),
            "mean_disagreement": _round(float(np.mean([float(row.get("mean_disagreement", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "max_disagreement": _round(float(np.max([float(row.get("max_disagreement", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "mean_replan_rate": _round(float(np.mean([float(row.get("replan_rate", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "mean_heavy_policy_calls_per_step": _round(float(np.mean([float(row.get("heavy_policy_calls_per_step", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "mean_policy_latency_s": _round(float(np.mean([float(row.get("policy_latency_mean_s", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "peak_cuda_allocated_mb": _round(max([float((row.get("cuda_memory") or {}).get("max_allocated_mb") or 0.0) for row in variant_rows] or [0.0]), 3),
        }
    strongest = max((name for name in variants if name != "rcv_full"), key=lambda name: by_variant[name]["task_balanced_success_rate"]) if "rcv_full" in variants else None
    return {
        "by_variant": by_variant,
        "strongest_baseline": strongest,
        "exception_count": int(sum(1 for row in rows if row.get("exception"))),
    }


def _paired_bootstrap_ci(deltas: list[float], *, seed: int = 2026071301, samples: int = 5000) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        means[index] = float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
    return [_round(float(np.quantile(means, 0.025)), 6), _round(float(np.quantile(means, 0.975)), 6)]


def _paired_vs_rcv(rows: list[Mapping[str, Any]], variants: tuple[str, ...]) -> dict[str, Any]:
    by_key = {
        (str(row.get("variant")), str(row.get("task_key")), int(row.get("identity"))): bool(row.get("success"))
        for row in rows
        if not row.get("exception")
    }
    out: dict[str, Any] = {}
    for variant in variants:
        if variant == "rcv_full":
            continue
        deltas: list[float] = []
        wins = losses = ties = 0
        for row in rows:
            if row.get("variant") != "rcv_full" or row.get("exception"):
                continue
            key = (variant, str(row.get("task_key")), int(row.get("identity")))
            if key not in by_key:
                continue
            full_success = bool(row.get("success"))
            base_success = bool(by_key[key])
            delta = float(full_success) - float(base_success)
            deltas.append(delta)
            if delta > 0:
                wins += 1
            elif delta < 0:
                losses += 1
            else:
                ties += 1
        out[variant] = {
            "paired_count": len(deltas),
            "paired_win_count": wins,
            "paired_loss_count": losses,
            "paired_tie_count": ties,
            "paired_success_delta": _round(float(np.mean(deltas)) if deltas else 0.0, 6),
            "paired_bootstrap_ci": _paired_bootstrap_ci(deltas),
        }
    return out


def _stage_0_decision(summary: Mapping[str, Any]) -> str:
    if int(summary.get("exception_count") or 0) > 0:
        return "STAGE_0_MEASUREMENT_INVALID_REPAIR_REQUIRED"
    queued = (summary.get("by_variant") or {}).get("queued_frozen_smolvla") or {}
    if float(queued.get("mean_disagreement", 0.0) or 0.0) < 1e-4 and float(queued.get("max_disagreement", 0.0) or 0.0) < 1e-3:
        return "STAGE_0_HARD_KILL_NO_QUEUE_FRESH_DISAGREEMENT"
    return "STAGE_0_PROBLEM_DIAGNOSTIC_PASS_TO_STAGE_1"


def _stage_1_decision(full: Mapping[str, Any], no_context: Mapping[str, Any]) -> str:
    full_metrics = full.get("calibration_metrics") or {}
    baseline = full.get("majority_baseline_metrics") or {}
    ablation_metrics = no_context.get("calibration_metrics") or {}
    if (
        float(full_metrics.get("balanced_accuracy", 0.0) or 0.0)
        <= float(baseline.get("balanced_accuracy", 0.0) or 0.0) + 0.02
        and float(full_metrics.get("f1", 0.0) or 0.0) <= float(ablation_metrics.get("f1", 0.0) or 0.0)
    ):
        return "STAGE_1_PERMANENT_KILL_VERIFIER_UNPREDICTIVE"
    return "STAGE_1_PROCEED_TO_STAGE_2A"


def _stage_2a_decision(summary: Mapping[str, Any]) -> str:
    if int(summary.get("exception_count") or 0) > 0:
        return "STAGE_2A_MEASUREMENT_INVALID_REPAIR_REQUIRED"
    by = summary["by_variant"]
    full = by["rcv_full"]
    strongest = by[summary["strongest_baseline"]]
    ablation = by["rcv_no_context_ablation"]
    if full["successes"] == 0 and strongest["successes"] >= 4:
        return "STAGE_2A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE"
    if float(strongest["task_balanced_success_rate"]) - float(full["task_balanced_success_rate"]) >= 0.30:
        return "STAGE_2A_CATASTROPHIC_KILL_CLEARLY_WORSE_THAN_BASELINE"
    if float(ablation["task_balanced_success_rate"]) - float(full["task_balanced_success_rate"]) >= 0.30:
        return "STAGE_2A_CATASTROPHIC_KILL_CLEARLY_WORSE_THAN_ABLATION"
    if full["task_balanced_success_rate"] > strongest["task_balanced_success_rate"]:
        return "STAGE_2A_POSITIVE_TO_STAGE_2B_REQUIRED"
    return "STAGE_2A_NONCATASTROPHIC_TO_STAGE_2B_REQUIRED"


def _stage_2b_decision(summary: Mapping[str, Any], paired: Mapping[str, Any]) -> str:
    if int(summary.get("exception_count") or 0) > 0:
        return "STAGE_2B_MEASUREMENT_INVALID_REPAIR_REQUIRED"
    by = summary["by_variant"]
    full_rate = float(by["rcv_full"]["task_balanced_success_rate"])
    strongest_name = str(summary["strongest_baseline"])
    strongest_rate = float(by[strongest_name]["task_balanced_success_rate"])
    ablation_rate = float(by["rcv_no_context_ablation"]["task_balanced_success_rate"])
    full_calls = float(by["rcv_full"].get("mean_heavy_policy_calls_per_step", 0.0) or 0.0)
    stateless_calls = float(by["stateless_first_action"].get("mean_heavy_policy_calls_per_step", 0.0) or 0.0)
    proxy_calls = float(by["sv_deviation_proxy"].get("mean_heavy_policy_calls_per_step", 0.0) or 0.0)
    if full_rate > strongest_rate and full_rate > ablation_rate and full_rate - strongest_rate >= 0.10 and full_calls < min(stateless_calls, proxy_calls):
        return "STAGE_2B_PROTOTYPE_GO"
    pair = paired.get(strongest_name) or {}
    ci = pair.get("paired_bootstrap_ci") or [0.0, 0.0]
    if full_rate <= strongest_rate and float(ci[1]) <= 0.10:
        return "STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    if ablation_rate >= full_rate:
        return "STAGE_2B_PERMANENT_KILL_ABLATION_EXPLAINS_RESULT"
    if by["stateless_first_action"]["task_balanced_success_rate"] >= full_rate and full_calls >= stateless_calls * 0.9:
        return "STAGE_2B_PERMANENT_KILL_STATELESS_EXPLAINS_WITHOUT_SAVINGS"
    return "STAGE_2B_UNRESOLVED_EXPANSION_OPTIONAL"


def _load_policy(args: argparse.Namespace) -> dict[str, Any]:
    _set_runtime_env(args)
    return _load_policy_and_processors(args, POLICIES[0])


def _stage_0_mode(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    loaded = _load_policy(args)
    variants = ("queued_frozen_smolvla", "stateless_first_action")
    rows = _planned_rows(TASKS[: int(args.max_tasks)], STAGE_0_IDENTITIES[: int(args.stage_0_identities)], variants)
    episodes: list[dict[str, Any]] = []
    for row in rows:
        episodes.append(
            _run_episode(
                row=row,
                loaded=loaded,
                max_eval_steps=int(args.max_eval_steps),
                tau_train=None,
                full_verifier=None,
                no_context_verifier=None,
            )
        )
    summary = _summarize(episodes, variants)
    final = _stage_0_decision(summary)
    return {
        "mode": "stage-0",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": True,
        "training_happened": False,
        "reset_identity_base": RESET_IDENTITY_BASE,
        "identities": STAGE_0_IDENTITIES[: int(args.stage_0_identities)],
        "episodes": episodes,
        "summary": summary,
        "final_decision": final,
        "next_step": "Run acquire-train." if final.endswith("PASS_TO_STAGE_1") else "Archive kill or repair according to governance.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }


def _acquire_train_mode(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    record_path = Path(args.acquisition_records)
    summary_path = Path(args.acquisition_summary)
    records: list[dict[str, Any]]
    summaries: list[dict[str, Any]]
    if record_path.exists() and summary_path.exists() and not bool(args.rerun_acquisition):
        records = _read_jsonl(record_path)
        summaries = json.loads(summary_path.read_text(encoding="utf-8-sig")).get("episodes") or []
    else:
        loaded = _load_policy(args)
        records = []
        summaries = []
        for split, identities in [("train", TRAIN_IDENTITIES), ("calibration", CALIBRATION_IDENTITIES)]:
            rows = _planned_rows(TASKS[: int(args.max_tasks)], identities, ("queued_frozen_smolvla",))
            for row in rows:
                episode_records, summary = _acquire_episode_records(row=row, split=split, loaded=loaded, max_eval_steps=int(args.max_eval_steps))
                records.extend(episode_records)
                summaries.append(summary)
                _write_jsonl(record_path, records)
                _write_json(summary_path, {"episodes": summaries, "record_count": len(records)})
    config = RCVConfig(
        disagreement_quantile=float(args.disagreement_quantile),
        learning_rate=float(args.learning_rate),
        l2=float(args.l2),
        max_epochs=int(args.max_epochs),
        seed=int(args.seed),
    )
    full = train_verifier(records, include_context=True, config=config)
    no_context = train_verifier(records, include_context=False, config=config, tau_train=float(full["tau_train"]))
    save_verifier(args.full_verifier, full)
    save_verifier(args.no_context_verifier, no_context)
    final = _stage_1_decision(full, no_context)
    report = {
        "mode": "acquire-train",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": True,
        "training_happened": True,
        "reset_identity_base": RESET_IDENTITY_BASE,
        "train_identities": TRAIN_IDENTITIES,
        "calibration_identities": CALIBRATION_IDENTITIES,
        "record_count": len(records),
        "episode_summaries": summaries,
        "full_verifier_path": str(args.full_verifier),
        "full_verifier_sha256": _sha256_file(Path(args.full_verifier)),
        "no_context_verifier_path": str(args.no_context_verifier),
        "no_context_verifier_sha256": _sha256_file(Path(args.no_context_verifier)),
        "tau_train": full["tau_train"],
        "theta_train_full": full["theta_train"],
        "theta_train_no_context": no_context["theta_train"],
        "full_verifier": {key: value for key, value in full.items() if key not in {"weights", "mean", "scale"}},
        "no_context_verifier": {key: value for key, value in no_context.items() if key not in {"weights", "mean", "scale"}},
        "summary": {
            "full_calibration_balanced_accuracy": full["calibration_metrics"]["balanced_accuracy"],
            "no_context_calibration_balanced_accuracy": no_context["calibration_metrics"]["balanced_accuracy"],
            "majority_baseline_balanced_accuracy": full["majority_baseline_metrics"]["balanced_accuracy"],
            "exception_count": int(sum(1 for row in summaries if row.get("exception"))),
        },
        "final_decision": final,
        "next_step": "Run Stage 2A." if final.endswith("STAGE_2A") else "Archive kill or repair according to governance.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }
    return report


def _stage_2_mode(args: argparse.Namespace, *, stage: str) -> dict[str, Any]:
    started = time.time()
    loaded = _load_policy(args)
    full = load_verifier(args.full_verifier)
    no_context = load_verifier(args.no_context_verifier)
    identities = STAGE_2A_IDENTITIES[: int(args.stage_2a_identities)] if stage == "stage-2a" else STAGE_2B_IDENTITIES[: int(args.stage_2b_identities)]
    rows = _planned_rows(TASKS[: int(args.max_tasks)], identities, VARIANTS)
    partial_path = Path(args.stage_2a_partial_output if stage == "stage-2a" else args.stage_2b_partial_output)
    episodes: list[dict[str, Any]] = []
    if partial_path.exists() and not bool(args.rerun_stage_2):
        episodes = list(json.loads(partial_path.read_text(encoding="utf-8-sig")).get("episodes") or [])
    completed = {(row.get("variant"), row.get("task_key"), int(row.get("identity", -1))) for row in episodes}
    for row in rows:
        key = (row["variant"], row["task_key"], int(row["identity"]))
        if key in completed:
            continue
        result = _run_episode(
            row=row,
            loaded=loaded,
            max_eval_steps=int(args.max_eval_steps),
            tau_train=float(full["tau_train"]),
            full_verifier=full,
            no_context_verifier=no_context,
        )
        episodes.append(result)
        _write_json(partial_path, {"episodes": episodes, "planned_episode_count": len(rows)})
    summary = _summarize(episodes, VARIANTS)
    paired = _paired_vs_rcv(episodes, VARIANTS)
    final = _stage_2a_decision(summary) if stage == "stage-2a" else _stage_2b_decision(summary, paired)
    return {
        "mode": stage,
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": True,
        "training_happened": False,
        "reset_identity_base": RESET_IDENTITY_BASE,
        "identities": identities,
        "tau_train": full["tau_train"],
        "theta_train_full": full["theta_train"],
        "theta_train_no_context": no_context["theta_train"],
        "full_verifier_path": str(args.full_verifier),
        "full_verifier_sha256": _sha256_file(Path(args.full_verifier)),
        "no_context_verifier_path": str(args.no_context_verifier),
        "no_context_verifier_sha256": _sha256_file(Path(args.no_context_verifier)),
        "planned_episode_count": len(rows),
        "completed_episode_count": len(episodes),
        "episodes": episodes,
        "summary": summary,
        "paired_vs_rcv_full": paired,
        "final_decision": final,
        "next_step": "Run Stage 2B." if stage == "stage-2a" and final.endswith("STAGE_2B_REQUIRED") else "Archive, expand, or scale according to governance.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "stage-0":
        report = _stage_0_mode(args)
        _write_json(Path(args.stage_0_output), report)
        _write_md(Path(args.stage_0_md), "RCV-VLA Stage 0 Result", report)
        return report
    if args.mode == "acquire-train":
        report = _acquire_train_mode(args)
        _write_json(Path(args.stage_1_output), report)
        _write_md(Path(args.stage_1_md), "RCV-VLA Stage 1 Train Result", report)
        return report
    if args.mode == "stage-2a":
        report = _stage_2_mode(args, stage="stage-2a")
        _write_json(Path(args.stage_2a_output), report)
        _write_md(Path(args.stage_2a_md), "RCV-VLA Stage 2A Result", report)
        return report
    if args.mode == "stage-2b":
        report = _stage_2_mode(args, stage="stage-2b")
        _write_json(Path(args.stage_2b_output), report)
        _write_md(Path(args.stage_2b_md), "RCV-VLA Stage 2B Result", report)
        return report
    raise ValueError(args.mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["stage-0", "acquire-train", "stage-2a", "stage-2b"], required=True)
    parser.add_argument("--base-path", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--stage-0-output", default="reports/rcv_vla/stage_0_result.json")
    parser.add_argument("--stage-0-md", default="reports/rcv_vla/stage_0_result.md")
    parser.add_argument("--acquisition-records", default="reports/rcv_vla/acquisition_records.jsonl")
    parser.add_argument("--acquisition-summary", default="reports/rcv_vla/acquisition_summary.json")
    parser.add_argument("--stage-1-output", default="reports/rcv_vla/stage_1_train_result.json")
    parser.add_argument("--stage-1-md", default="reports/rcv_vla/stage_1_train_result.md")
    parser.add_argument("--full-verifier", default="reports/rcv_vla/verifier_full.json")
    parser.add_argument("--no-context-verifier", default="reports/rcv_vla/verifier_no_context.json")
    parser.add_argument("--stage-2a-output", default="reports/rcv_vla/stage_2a_result.json")
    parser.add_argument("--stage-2a-md", default="reports/rcv_vla/stage_2a_result.md")
    parser.add_argument("--stage-2a-partial-output", default="reports/rcv_vla/stage_2a_partial_result.json")
    parser.add_argument("--stage-2b-output", default="reports/rcv_vla/stage_2b_result.json")
    parser.add_argument("--stage-2b-md", default="reports/rcv_vla/stage_2b_result.md")
    parser.add_argument("--stage-2b-partial-output", default="reports/rcv_vla/stage_2b_partial_result.json")
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--stage-0-identities", type=int, default=5)
    parser.add_argument("--stage-2a-identities", type=int, default=5)
    parser.add_argument("--stage-2b-identities", type=int, default=20)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--disagreement-quantile", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=260713)
    parser.add_argument("--rerun-acquisition", action="store_true")
    parser.add_argument("--rerun-stage-2", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.max_tasks) < 1 or int(args.max_tasks) > len(TASKS):
        raise SystemExit("--max-tasks must be between 1 and 2")
    if int(args.stage_0_identities) < 1 or int(args.stage_0_identities) > len(STAGE_0_IDENTITIES):
        raise SystemExit(f"--stage-0-identities must be between 1 and {len(STAGE_0_IDENTITIES)}")
    if int(args.stage_2a_identities) < 1 or int(args.stage_2a_identities) > len(STAGE_2A_IDENTITIES):
        raise SystemExit(f"--stage-2a-identities must be between 1 and {len(STAGE_2A_IDENTITIES)}")
    if int(args.stage_2b_identities) < 1 or int(args.stage_2b_identities) > len(STAGE_2B_IDENTITIES):
        raise SystemExit(f"--stage-2b-identities must be between 1 and {len(STAGE_2B_IDENTITIES)}")
    if set(TASK_KEYS) != {f"{task['suite']}/task_{task['task_id']}" for task in TASKS}:
        raise SystemExit("RCV task constants disagree with runner task manifest")
    report = run(args)
    print(json.dumps({"mode": args.mode, "final_decision": report.get("final_decision"), "elapsed_seconds": report.get("elapsed_seconds")}, indent=2, sort_keys=True))
    invalid = "INVALID" in str(report.get("final_decision"))
    return 2 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
