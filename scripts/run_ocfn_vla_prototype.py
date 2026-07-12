"""OCFN-VLA prototype runner."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_echo_vla_first_prototype import _postprocess_action, _preprocess_batch  # noqa: E402
from scripts.run_phase_barrier_vla_prototype import (  # noqa: E402
    _make_exact_vector_env,
    _round,
    _set_runtime_env,
    _step_success,
)
from tca_map.smolvla.ocfn_vla import (  # noqa: E402
    OCFNConfig,
    assert_no_privileged_inference_fields,
    build_all_selections,
    file_sha256,
    full_equals_baseline,
    make_noise_bank,
    noise_sha256,
    selections_to_json,
    stage_a_decision,
    stage_b_decision,
    task_key,
    zero_noise,
)
from tca_map.smolvla.official_closed_loop_scaleup import _json_default  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import POLICIES, _cuda_memory, _load_policy_and_processors  # noqa: E402


DATE_KST = "2026-07-12"
BRANCH = "codex/autonomous-until-paper-governance-v2"
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
TRAIN_IDENTITIES = [20260711, 20260712]
EVAL_IDENTITIES = [20260713, 20260714, 20260715, 20260716, 20260717]
STAGE_B_IDENTITIES = list(range(20260718, 20260758))
RESET_IDENTITY_BASE = 20260711
MAX_OFFICIAL_INITIAL_STATE_COUNT = 50
VARIANTS = [
    "frozen_smolvla",
    "zero_noise_smolvla",
    "global_success_noise_prior",
    "task_shuffled_noise_prior",
    "ocfn_full",
]


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, title: str, report: Mapping[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        f"- mode: `{report.get('mode')}`",
        f"- training happened: `{report.get('training_happened')}`",
        f"- closed-loop experiment happened: `{report.get('closed_loop_experiment_happened')}`",
        f"- summary: `{report.get('summary')}`",
        f"- elapsed seconds: `{report.get('elapsed_seconds')}`",
        "",
        f"Next step: {report.get('next_step')}",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _identity_to_initial_state_index(identity: int) -> int:
    index = int(identity) - RESET_IDENTITY_BASE
    if index < 0 or index >= MAX_OFFICIAL_INITIAL_STATE_COUNT:
        raise ValueError(f"unknown reset identity {identity}")
    return int(index)


def _episode_key(row: Mapping[str, Any]) -> str:
    parts = [str(row.get("variant")), str(row.get("suite")), str(row.get("task_id")), str(row.get("identity"))]
    if row.get("noise_id") is not None:
        parts.append(str(row.get("noise_id")))
    return "|".join(parts)


def _cuda_memory_report() -> dict[str, Any]:
    import torch

    return _cuda_memory(torch)


def _config_from_loaded(args: argparse.Namespace, loaded: Mapping[str, Any]) -> OCFNConfig:
    cfg = loaded.get("cfg")
    audit = loaded.get("audit") or {}
    action_shape = audit.get("action_chunk_shape") or [1, int(args.chunk_size), int(args.max_action_dim)]
    chunk_size = getattr(cfg, "chunk_size", None) or (int(action_shape[1]) if len(action_shape) > 1 else int(args.chunk_size))
    max_action_dim = getattr(cfg, "max_action_dim", None) or (int(action_shape[2]) if len(action_shape) > 2 else int(args.max_action_dim))
    return OCFNConfig(
        noise_count=int(args.noise_count),
        chunk_size=int(chunk_size),
        max_action_dim=int(max_action_dim),
        seed_base=int(args.noise_seed_base),
        task_shuffle_seed=int(args.task_shuffle_seed),
    )


def _noise_tensor(noise: np.ndarray | None, batch: Mapping[str, Any]) -> Any | None:
    if noise is None:
        return None
    import torch

    device = "cuda"
    for value in batch.values():
        if hasattr(value, "device"):
            device = str(value.device)
            break
    tensor = torch.as_tensor(np.asarray(noise, dtype=np.float32), dtype=torch.float32, device=device)
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def _policy_action_with_noise(policy: Any, env: Any, observation: Any, loaded: Mapping[str, Any], noise: np.ndarray | None) -> np.ndarray:
    import torch

    batch = _preprocess_batch(env, observation, dict(loaded))
    tensor_noise = _noise_tensor(noise, batch)
    with torch.inference_mode():
        if tensor_noise is None:
            action = policy.select_action(batch)
        else:
            action = policy.select_action(batch, noise=tensor_noise)
    return _postprocess_action(action, dict(loaded)).reshape(1, -1)


def _run_episode(
    *,
    row: Mapping[str, Any],
    loaded: Mapping[str, Any],
    noise: np.ndarray | None,
    max_eval_steps: int,
) -> dict[str, Any]:
    env = None
    started = time.time()
    try:
        env = _make_exact_vector_env(str(row["suite"]), int(row["task_id"]), _identity_to_initial_state_index(int(row["identity"])))
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        observation, _ = env.reset(seed=[int(row["identity"])])
        max_steps = int(env.call("_max_episode_steps")[0])
        if int(max_eval_steps) > 0:
            max_steps = min(max_steps, int(max_eval_steps))
        success = False
        rewards: list[float] = []
        policy_latencies: list[float] = []
        steps = 0
        for step in range(max_steps):
            start_policy = time.perf_counter()
            action = _policy_action_with_noise(policy, env, observation, loaded, noise).reshape(-1).astype(np.float32)
            policy_latencies.append(time.perf_counter() - start_policy)
            observation, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
            rewards.append(float(np.asarray(reward).reshape(-1)[0]))
            steps = int(step + 1)
            success = bool(success or _step_success(info))
            if success or np.all(terminated | truncated):
                break
        return {
            **dict(row),
            "success": bool(success),
            "exception": None,
            "episode_steps": steps,
            "reward_sum": _round(float(np.sum(rewards)) if rewards else 0.0, 6),
            "policy_latency_mean_s": _round(float(np.mean(policy_latencies)) if policy_latencies else 0.0, 6),
            "policy_latency_max_s": _round(float(np.max(policy_latencies)) if policy_latencies else 0.0, 6),
            "elapsed_seconds": _round(time.time() - started, 3),
            "cuda_memory": _cuda_memory_report(),
        }
    except Exception as exc:  # pragma: no cover - real rollout boundary
        return {
            **dict(row),
            "success": False,
            "exception": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            "episode_steps": 0,
            "elapsed_seconds": _round(time.time() - started, 3),
            "cuda_memory": _cuda_memory_report(),
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _initial_action_diagnostics(
    *,
    task: Mapping[str, Any],
    identity: int,
    loaded: Mapping[str, Any],
    noises: Mapping[str, np.ndarray | None],
) -> dict[str, Any]:
    env = None
    actions: dict[str, np.ndarray] = {}
    try:
        env = _make_exact_vector_env(str(task["suite"]), int(task["task_id"]), _identity_to_initial_state_index(int(identity)))
        observation, _ = env.reset(seed=[int(identity)])
        policy = loaded["policy"]
        for name, noise in noises.items():
            if hasattr(policy, "reset"):
                policy.reset()
            actions[name] = _policy_action_with_noise(policy, env, observation, loaded, noise).reshape(-1).astype(np.float32)
        full = actions.get("ocfn_full")
        diag: dict[str, Any] = {"computed": full is not None, "action_norms": {name: _round(float(np.linalg.norm(action)), 6) for name, action in actions.items()}}
        if full is not None:
            for name, action in actions.items():
                if name == "ocfn_full":
                    continue
                diag[f"ocfn_full_delta_vs_{name}"] = _round(float(np.linalg.norm(full - action)), 6)
        return diag
    except Exception as exc:  # pragma: no cover
        return {"computed": False, "exception": "".join(traceback.format_exception_only(type(exc), exc)).strip()}
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _planned_train_rows(noise_ids: list[int], tasks: list[Mapping[str, Any]], identities: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in tasks:
        for identity in identities:
            for noise_id in noise_ids:
                rows.append(
                    {
                        "variant": "train_noise_identity",
                        "suite": str(task["suite"]),
                        "task_id": int(task["task_id"]),
                        "task_key": task_key(str(task["suite"]), int(task["task_id"])),
                        "role": str(task["role"]),
                        "instruction": str(task["instruction"]),
                        "identity": int(identity),
                        "noise_id": int(noise_id),
                    }
                )
    return rows


def _planned_stage_rows(tasks: list[Mapping[str, Any]], identities: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for task in tasks:
            for identity in identities:
                rows.append(
                    {
                        "variant": variant,
                        "suite": str(task["suite"]),
                        "task_id": int(task["task_id"]),
                        "task_key": task_key(str(task["suite"]), int(task["task_id"])),
                        "role": str(task["role"]),
                        "instruction": str(task["instruction"]),
                        "identity": int(identity),
                    }
                )
    return rows


def _noise_for_selection(
    *,
    variant: str,
    task: str,
    selections: Mapping[str, Mapping[str, Any]],
    noise_bank: Mapping[int, np.ndarray],
    config: OCFNConfig,
) -> tuple[np.ndarray | None, int | None, str]:
    if variant == "frozen_smolvla":
        return None, None, "default_policy_noise"
    if variant == "zero_noise_smolvla":
        return zero_noise(config), None, "zero_noise"
    selected = (selections.get(variant) or {}).get(task)
    if selected is None:
        raise RuntimeError(f"missing selection for {variant} {task}")
    noise_id = selected.get("noise_id")
    if noise_id is None:
        return None, None, selected.get("source", "none")
    return np.asarray(noise_bank[int(noise_id)], dtype=np.float32), int(noise_id), str(selected.get("source"))


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    phat = successes / total
    denom = 1.0 + z * z / total
    centre = phat + z * z / (2 * total)
    margin = z * np.sqrt((phat * (1.0 - phat) + z * z / (4 * total)) / total)
    return [_round((centre - margin) / denom, 6), _round((centre + margin) / denom, 6)]


def _summarize_episodes(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        variant_rows = [row for row in rows if row.get("variant") == variant]
        successes = int(sum(1 for row in variant_rows if bool(row.get("success"))))
        total = int(len(variant_rows))
        per_task: dict[str, Any] = {}
        for key in sorted({str(row.get("task_key")) for row in variant_rows}):
            task_rows = [row for row in variant_rows if str(row.get("task_key")) == key]
            task_successes = int(sum(1 for row in task_rows if bool(row.get("success"))))
            per_task[key] = {"successes": task_successes, "total": len(task_rows), "rate": _round(task_successes / max(1, len(task_rows)), 6)}
        task_balanced = float(np.mean([item["rate"] for item in per_task.values()])) if per_task else 0.0
        initial_delta_global = [float(row.get("initial_action_diagnostics", {}).get("ocfn_full_delta_vs_global_success_noise_prior", 0.0) or 0.0) for row in variant_rows]
        initial_delta_shuffled = [float(row.get("initial_action_diagnostics", {}).get("ocfn_full_delta_vs_task_shuffled_noise_prior", 0.0) or 0.0) for row in variant_rows]
        by_variant[variant] = {
            "successes": successes,
            "total": total,
            "success_rate": _round(successes / max(1, total), 6),
            "task_balanced_success_rate": _round(task_balanced, 6),
            "wilson_95_ci": _wilson_ci(successes, total),
            "per_task": per_task,
            "exceptions": int(sum(1 for row in variant_rows if row.get("exception"))),
            "policy_latency_mean_s": _round(float(np.mean([float(row.get("policy_latency_mean_s", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "policy_latency_max_s": _round(float(np.max([float(row.get("policy_latency_max_s", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "peak_cuda_allocated_mb": _round(
                float(np.max([float((row.get("cuda_memory") or {}).get("max_allocated_mb", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0,
                3,
            ),
            "mean_initial_delta_full_vs_global": _round(float(np.mean(initial_delta_global)) if initial_delta_global else 0.0, 6),
            "mean_initial_delta_full_vs_shuffled": _round(float(np.mean(initial_delta_shuffled)) if initial_delta_shuffled else 0.0, 6),
        }
    return by_variant


def _paired_bootstrap_ci(deltas: list[float], *, seed: int = 2026071205, samples: int = 5000) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        means[index] = float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
    return [_round(float(np.quantile(means, 0.025)), 6), _round(float(np.quantile(means, 0.975)), 6)]


def _mcnemar_exact_p(win_count: int, loss_count: int) -> float:
    discordant = int(win_count) + int(loss_count)
    if discordant <= 0:
        return 1.0
    tail = min(int(win_count), int(loss_count))
    probability = sum(math.comb(discordant, k) for k in range(tail + 1)) / float(2**discordant)
    return _round(min(1.0, 2.0 * probability), 6)


def _paired_vs_full(rows: list[Mapping[str, Any]], baseline_names: list[str]) -> dict[str, Any]:
    by_pair: dict[tuple[str, int], dict[str, Mapping[str, Any]]] = {}
    for row in rows:
        by_pair.setdefault((str(row.get("task_key")), int(row.get("identity"))), {})[str(row.get("variant"))] = row
    out: dict[str, Any] = {}
    for baseline in baseline_names:
        wins = 0
        losses = 0
        ties = 0
        deltas: list[float] = []
        complete_pairs = 0
        full_successes = 0
        baseline_successes = 0
        for pair_rows in by_pair.values():
            full_row = pair_rows.get("ocfn_full")
            baseline_row = pair_rows.get(baseline)
            if full_row is None or baseline_row is None:
                continue
            complete_pairs += 1
            full_success = bool(full_row.get("success"))
            baseline_success = bool(baseline_row.get("success"))
            full_successes += int(full_success)
            baseline_successes += int(baseline_success)
            delta = float(int(full_success) - int(baseline_success))
            deltas.append(delta)
            if full_success and not baseline_success:
                wins += 1
            elif baseline_success and not full_success:
                losses += 1
            else:
                ties += 1
        baseline_failure = max(0, complete_pairs - baseline_successes)
        full_failure = max(0, complete_pairs - full_successes)
        failure_reduction = None if baseline_failure <= 0 else (baseline_failure - full_failure) / baseline_failure
        out[baseline] = {
            "paired_count": int(complete_pairs),
            "paired_win_count": int(wins),
            "paired_loss_count": int(losses),
            "paired_tie_count": int(ties),
            "full_successes": int(full_successes),
            "baseline_successes": int(baseline_successes),
            "paired_success_delta": _round(float(np.mean(deltas)) if deltas else 0.0, 6),
            "paired_bootstrap_ci": _paired_bootstrap_ci(deltas, seed=2026071205 + len(out)),
            "mcnemar_exact_p": _mcnemar_exact_p(wins, losses),
            "failure_rate_reduction": _round(float(failure_reduction), 6) if failure_reduction is not None else None,
        }
    return out


def _synthetic_mode(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    config = OCFNConfig(noise_count=4, chunk_size=3, max_action_dim=2, seed_base=int(args.noise_seed_base), task_shuffle_seed=int(args.task_shuffle_seed))
    rows = [
        {"task_key": "task_a", "noise_id": 0, "success": False, "episode_steps": 10, "reward_sum": 0.0},
        {"task_key": "task_a", "noise_id": 1, "success": True, "episode_steps": 4, "reward_sum": 1.0},
        {"task_key": "task_a", "noise_id": 2, "success": False, "episode_steps": 9, "reward_sum": 0.0},
        {"task_key": "task_a", "noise_id": 3, "success": False, "episode_steps": 8, "reward_sum": 0.0},
        {"task_key": "task_b", "noise_id": 0, "success": False, "episode_steps": 10, "reward_sum": 0.0},
        {"task_key": "task_b", "noise_id": 1, "success": False, "episode_steps": 8, "reward_sum": 0.0},
        {"task_key": "task_b", "noise_id": 2, "success": True, "episode_steps": 3, "reward_sum": 1.0},
        {"task_key": "task_b", "noise_id": 3, "success": False, "episode_steps": 9, "reward_sum": 0.0},
    ]
    selections = build_all_selections(rows, ["task_a", "task_b"], config)
    bank = make_noise_bank(config)
    full_a = bank[int(selections["ocfn_full"]["task_a"].noise_id)]
    full_b = bank[int(selections["ocfn_full"]["task_b"].noise_id)]
    action_delta = float(np.linalg.norm(full_a - full_b))
    passed = bool(
        selections["ocfn_full"]["task_a"].noise_id == 1
        and selections["ocfn_full"]["task_b"].noise_id == 2
        and action_delta > 1e-6
        and not full_equals_baseline(selections, "global_success_noise_prior", ["task_a", "task_b"])
    )
    return {
        "mode": "synthetic",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": False,
        "config": config.to_json(),
        "selection_summary": selections_to_json(selections),
        "summary": {"synthetic_passed": passed, "task_noise_delta": _round(action_delta, 6)},
        "final_decision": "SYNTHETIC_MECHANISM_PASS" if passed else "SYNTHETIC_MECHANISM_FAIL",
        "next_step": "Run train acquisition." if passed else "Repair synthetic OCFN selection logic.",
        "elapsed_seconds": _round(time.time() - start, 3),
    }


def _train_acquisition_mode(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    _set_runtime_env(args)
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    loaded = _load_policy_and_processors(args, POLICIES[0])
    config = _config_from_loaded(args, loaded)
    assert_no_privileged_inference_fields(["task_key", "suite", "task_id", "instruction"])
    noise_bank = make_noise_bank(config)
    tasks = TASKS[: int(args.max_tasks)]
    identities = TRAIN_IDENTITIES[: int(args.train_identities)]
    planned = _planned_train_rows(list(noise_bank.keys()), tasks, identities)
    partial_path = Path(args.train_partial_output)
    completed: dict[str, Any] = {}
    if partial_path.exists() and not bool(args.rerun_train_acquisition):
        try:
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            completed = {_episode_key(row): row for row in partial.get("episodes", [])}
        except Exception:
            completed = {}
    episodes = list(completed.values())
    for row in planned:
        key = _episode_key(row)
        if key in completed:
            continue
        noise = np.asarray(noise_bank[int(row["noise_id"])], dtype=np.float32)
        result = _run_episode(row=row, loaded=loaded, noise=noise, max_eval_steps=int(args.max_eval_steps))
        episodes.append(result)
        partial_report = {
            "mode": "train-acquisition-partial",
            "branch": BRANCH,
            "date_kst": DATE_KST,
            "config": config.to_json(),
            "noise_bank_sha256": {str(noise_id): noise_sha256(noise) for noise_id, noise in noise_bank.items()},
            "planned_episode_count": len(planned),
            "completed_episode_count": len(episodes),
            "episodes": episodes,
        }
        _write_json(partial_path, partial_report)
        print(f"[ocfn-train] completed {len(episodes)}/{len(planned)}: {key} success={result.get('success')} exception={bool(result.get('exception'))}", flush=True)
    task_keys = [task_key(str(task["suite"]), int(task["task_id"])) for task in tasks]
    selections = build_all_selections(episodes, task_keys, config)
    report = {
        "mode": "train-acquisition",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": True,
        "closed_loop_experiment_happened": True,
        "config": config.to_json(),
        "tasks": tasks,
        "train_identities": identities,
        "noise_bank_sha256": {str(noise_id): noise_sha256(noise) for noise_id, noise in noise_bank.items()},
        "episode_count": len(episodes),
        "planned_episode_count": len(planned),
        "exceptions": [row for row in episodes if row.get("exception")],
        "episodes": episodes,
        "selection_summary": selections_to_json(selections),
        "summary": {
            "train_acquisition_complete": len(episodes) == len(planned),
            "full_equals_global": full_equals_baseline(selections, "global_success_noise_prior", task_keys),
            "full_equals_shuffled": full_equals_baseline(selections, "task_shuffled_noise_prior", task_keys),
        },
        "final_decision": "TRAIN_ACQUISITION_PASS" if len(episodes) == len(planned) and not any(row.get("exception") for row in episodes) else "TRAIN_ACQUISITION_MEASUREMENT_INVALID",
        "next_step": "Run Stage A." if len(episodes) == len(planned) else "Repair acquisition/runtime once before Stage A.",
        "elapsed_seconds": _round(time.time() - start, 3),
        "cuda_memory": _cuda_memory_report(),
    }
    return report


def _stage_a_mode(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    train_path = Path(args.train_acquisition_output)
    train_report = json.loads(train_path.read_text(encoding="utf-8"))
    _set_runtime_env(args)
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    loaded = _load_policy_and_processors(args, POLICIES[0])
    config = OCFNConfig(**train_report["config"])
    noise_bank = make_noise_bank(config)
    task_keys = [task_key(str(task["suite"]), int(task["task_id"])) for task in TASKS[: int(args.max_tasks)]]
    selections = build_all_selections(train_report["episodes"], task_keys, config)
    selections_json = selections_to_json(selections)
    tasks = TASKS[: int(args.max_tasks)]
    identities = EVAL_IDENTITIES[: int(args.eval_identities)]
    planned = _planned_stage_rows(tasks, identities)
    partial_path = Path(args.stage_a_partial_output)
    completed: dict[str, Any] = {}
    if partial_path.exists() and not bool(args.rerun_stage_a):
        try:
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            completed = {_episode_key(row): row for row in partial.get("episodes", [])}
        except Exception:
            completed = {}
    episodes = list(completed.values())
    for row in planned:
        key = _episode_key(row)
        if key in completed:
            continue
        noise, selected_noise_id, selection_source = _noise_for_selection(
            variant=str(row["variant"]),
            task=str(row["task_key"]),
            selections=selections_json,
            noise_bank=noise_bank,
            config=config,
        )
        stage_row = {**row, "selected_noise_id": selected_noise_id, "selection_source": selection_source}
        if row["variant"] == "ocfn_full":
            task = next(item for item in tasks if task_key(str(item["suite"]), int(item["task_id"])) == str(row["task_key"]))
            diag_noises: dict[str, np.ndarray | None] = {}
            for variant in VARIANTS:
                diag_noise, _diag_noise_id, _source = _noise_for_selection(
                    variant=variant,
                    task=str(row["task_key"]),
                    selections=selections_json,
                    noise_bank=noise_bank,
                    config=config,
                )
                diag_noises[variant] = diag_noise
            stage_row["initial_action_diagnostics"] = _initial_action_diagnostics(
                task=task,
                identity=int(row["identity"]),
                loaded=loaded,
                noises=diag_noises,
            )
        result = _run_episode(row=stage_row, loaded=loaded, noise=noise, max_eval_steps=int(args.max_eval_steps))
        episodes.append(result)
        partial_report = {
            "mode": "stage-a-partial",
            "branch": BRANCH,
            "date_kst": DATE_KST,
            "config": config.to_json(),
            "selection_summary": selections_json,
            "planned_episode_count": len(planned),
            "completed_episode_count": len(episodes),
            "episodes": episodes,
        }
        _write_json(partial_path, partial_report)
        print(f"[ocfn-stage-a] completed {len(episodes)}/{len(planned)}: {key} success={result.get('success')} exception={bool(result.get('exception'))}", flush=True)
    by_variant = _summarize_episodes(episodes)
    full_global_equiv = full_equals_baseline(selections, "global_success_noise_prior", task_keys)
    full_shuffled_equiv = full_equals_baseline(selections, "task_shuffled_noise_prior", task_keys)
    full_rows = [row for row in episodes if row.get("variant") == "ocfn_full"]
    mean_delta_global = float(np.mean([float(row.get("initial_action_diagnostics", {}).get("ocfn_full_delta_vs_global_success_noise_prior", 0.0) or 0.0) for row in full_rows])) if full_rows else 0.0
    mean_delta_shuffled = float(np.mean([float(row.get("initial_action_diagnostics", {}).get("ocfn_full_delta_vs_task_shuffled_noise_prior", 0.0) or 0.0) for row in full_rows])) if full_rows else 0.0
    final = stage_a_decision(
        by_variant,
        full_global_equivalent=full_global_equiv,
        full_shuffled_equivalent=full_shuffled_equiv,
        full_action_delta_vs_global=mean_delta_global,
        full_action_delta_vs_shuffled=mean_delta_shuffled,
    )
    strongest = max(["frozen_smolvla", "zero_noise_smolvla", "global_success_noise_prior", "task_shuffled_noise_prior"], key=lambda name: float(by_variant.get(name, {}).get("task_balanced_success_rate", 0.0)))
    return {
        "mode": "stage-a",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": True,
        "config": config.to_json(),
        "tasks": tasks,
        "eval_identities": identities,
        "variants": VARIANTS,
        "train_acquisition_output": str(train_path),
        "train_acquisition_sha256": file_sha256(train_path),
        "selection_summary": selections_json,
        "noise_bank_sha256": {str(noise_id): noise_sha256(noise) for noise_id, noise in noise_bank.items()},
        "episode_count": len(episodes),
        "planned_episode_count": len(planned),
        "exceptions": [row for row in episodes if row.get("exception")],
        "episodes": episodes,
        "summary": {
            "by_variant": by_variant,
            "strongest_baseline": strongest,
            "full_equals_global": full_global_equiv,
            "full_equals_shuffled": full_shuffled_equiv,
            "mean_initial_delta_full_vs_global": _round(mean_delta_global, 6),
            "mean_initial_delta_full_vs_shuffled": _round(mean_delta_shuffled, 6),
            "final_decision": final,
        },
        "final_decision": final,
        "next_step": "Archive kill and pivot if permanent kill; otherwise run Stage B under governance.",
        "elapsed_seconds": _round(time.time() - start, 3),
        "cuda_memory": _cuda_memory_report(),
    }


def _stage_b_mode(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    train_path = Path(args.train_acquisition_output)
    stage_a_path = Path(args.stage_a_output)
    train_report = json.loads(train_path.read_text(encoding="utf-8"))
    stage_a_report = json.loads(stage_a_path.read_text(encoding="utf-8"))
    _set_runtime_env(args)
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    loaded = _load_policy_and_processors(args, POLICIES[0])
    config = OCFNConfig(**train_report["config"])
    noise_bank = make_noise_bank(config)
    tasks = TASKS[: int(args.max_tasks)]
    task_keys = [task_key(str(task["suite"]), int(task["task_id"])) for task in tasks]
    selections = build_all_selections(train_report["episodes"], task_keys, config)
    selections_json = selections_to_json(selections)
    identities = STAGE_B_IDENTITIES[: int(args.stage_b_identities)]
    planned = _planned_stage_rows(tasks, identities)
    partial_path = Path(args.stage_b_partial_output)
    completed: dict[str, Any] = {}
    if partial_path.exists() and not bool(args.rerun_stage_b):
        try:
            partial = json.loads(partial_path.read_text(encoding="utf-8"))
            completed = {_episode_key(row): row for row in partial.get("episodes", [])}
        except Exception:
            completed = {}
    episodes = list(completed.values())
    for row in planned:
        key = _episode_key(row)
        if key in completed:
            continue
        noise, selected_noise_id, selection_source = _noise_for_selection(
            variant=str(row["variant"]),
            task=str(row["task_key"]),
            selections=selections_json,
            noise_bank=noise_bank,
            config=config,
        )
        stage_row = {**row, "selected_noise_id": selected_noise_id, "selection_source": selection_source}
        if row["variant"] == "ocfn_full":
            task = next(item for item in tasks if task_key(str(item["suite"]), int(item["task_id"])) == str(row["task_key"]))
            diag_noises: dict[str, np.ndarray | None] = {}
            for variant in VARIANTS:
                diag_noise, _diag_noise_id, _source = _noise_for_selection(
                    variant=variant,
                    task=str(row["task_key"]),
                    selections=selections_json,
                    noise_bank=noise_bank,
                    config=config,
                )
                diag_noises[variant] = diag_noise
            stage_row["initial_action_diagnostics"] = _initial_action_diagnostics(
                task=task,
                identity=int(row["identity"]),
                loaded=loaded,
                noises=diag_noises,
            )
        result = _run_episode(row=stage_row, loaded=loaded, noise=noise, max_eval_steps=int(args.max_eval_steps))
        episodes.append(result)
        partial_report = {
            "mode": "stage-b-partial",
            "branch": BRANCH,
            "date_kst": DATE_KST,
            "config": config.to_json(),
            "selection_summary": selections_json,
            "stage_a_output": str(stage_a_path),
            "planned_pair_count_per_policy": len(tasks) * len(identities),
            "planned_episode_count": len(planned),
            "completed_episode_count": len(episodes),
            "episodes": episodes,
        }
        _write_json(partial_path, partial_report)
        print(f"[ocfn-stage-b] completed {len(episodes)}/{len(planned)}: {key} success={result.get('success')} exception={bool(result.get('exception'))}", flush=True)
    by_variant = _summarize_episodes(episodes)
    baseline_names = ["frozen_smolvla", "zero_noise_smolvla", "global_success_noise_prior", "task_shuffled_noise_prior"]
    paired = _paired_vs_full(episodes, baseline_names)
    full_global_equiv = full_equals_baseline(selections, "global_success_noise_prior", task_keys)
    full_shuffled_equiv = full_equals_baseline(selections, "task_shuffled_noise_prior", task_keys)
    full_rows = [row for row in episodes if row.get("variant") == "ocfn_full"]
    mean_delta_global = float(np.mean([float(row.get("initial_action_diagnostics", {}).get("ocfn_full_delta_vs_global_success_noise_prior", 0.0) or 0.0) for row in full_rows])) if full_rows else 0.0
    mean_delta_shuffled = float(np.mean([float(row.get("initial_action_diagnostics", {}).get("ocfn_full_delta_vs_task_shuffled_noise_prior", 0.0) or 0.0) for row in full_rows])) if full_rows else 0.0
    mechanism_active = bool(
        (not full_global_equiv or abs(mean_delta_global) > 1e-6)
        and (not full_shuffled_equiv or abs(mean_delta_shuffled) > 1e-6)
    )
    exceptions = [row for row in episodes if row.get("exception")]
    complete = len(episodes) == len(planned)
    pairs_per_policy = len(tasks) * len(identities)
    final = stage_b_decision(
        by_variant,
        paired,
        mechanism_active=mechanism_active,
        complete=complete,
        exception_count=len(exceptions),
        pairs_per_policy=pairs_per_policy,
    )
    strongest = max(baseline_names, key=lambda name: float(by_variant.get(name, {}).get("task_balanced_success_rate", 0.0)))
    if final == "STAGE_B_UNRESOLVED_EXPAND_TO_80_REQUIRED":
        next_step = "Expand once to 80 paired episodes per key policy using the same fixed manifest prefix."
    elif final == "STAGE_B_PROTOTYPE_GO":
        next_step = "Proceed to prototype-GO confirmation and second-backbone feasibility work."
    else:
        next_step = "Archive OCFN current formulation and pivot under current governance."
    return {
        "mode": "stage-b",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": True,
        "config": config.to_json(),
        "tasks": tasks,
        "stage_b_identities": identities,
        "variants": VARIANTS,
        "train_acquisition_output": str(train_path),
        "train_acquisition_sha256": file_sha256(train_path),
        "stage_a_output": str(stage_a_path),
        "stage_a_sha256": file_sha256(stage_a_path),
        "stage_a_final_decision": stage_a_report.get("final_decision"),
        "selection_summary": selections_json,
        "noise_bank_sha256": {str(noise_id): noise_sha256(noise) for noise_id, noise in noise_bank.items()},
        "episode_count": len(episodes),
        "planned_episode_count": len(planned),
        "planned_pair_count_per_policy": pairs_per_policy,
        "exceptions": exceptions,
        "episodes": episodes,
        "summary": {
            "by_variant": by_variant,
            "paired_vs_full": paired,
            "strongest_baseline": strongest,
            "full_equals_global": full_global_equiv,
            "full_equals_shuffled": full_shuffled_equiv,
            "mechanism_active": mechanism_active,
            "mean_initial_delta_full_vs_global": _round(mean_delta_global, 6),
            "mean_initial_delta_full_vs_shuffled": _round(mean_delta_shuffled, 6),
            "final_decision": final,
        },
        "final_decision": final,
        "next_step": next_step,
        "elapsed_seconds": _round(time.time() - start, 3),
        "cuda_memory": _cuda_memory_report(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "synthetic":
        report = _synthetic_mode(args)
        _write_json(Path(args.synthetic_output), report)
        _write_md(Path(args.synthetic_md), "OCFN-VLA Synthetic Result", report)
        return report
    if args.mode == "train-acquisition":
        report = _train_acquisition_mode(args)
        _write_json(Path(args.train_acquisition_output), report)
        _write_md(Path(args.train_acquisition_md), "OCFN-VLA Train Acquisition Result", report)
        return report
    if args.mode == "stage-a":
        report = _stage_a_mode(args)
        _write_json(Path(args.stage_a_output), report)
        _write_md(Path(args.stage_a_md), "OCFN-VLA Stage A Result", report)
        return report
    if args.mode == "stage-b":
        report = _stage_b_mode(args)
        _write_json(Path(args.stage_b_output), report)
        _write_md(Path(args.stage_b_md), "OCFN-VLA Stage B Result", report)
        return report
    raise ValueError(f"unknown mode: {args.mode}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "train-acquisition", "stage-a", "stage-b"], required=True)
    parser.add_argument("--checkpoint", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--synthetic-output", default="reports/ocfn_vla/synthetic_result.json")
    parser.add_argument("--synthetic-md", default="reports/ocfn_vla/synthetic_result.md")
    parser.add_argument("--train-acquisition-output", default="reports/ocfn_vla/train_acquisition_result.json")
    parser.add_argument("--train-acquisition-md", default="reports/ocfn_vla/train_acquisition_result.md")
    parser.add_argument("--train-partial-output", default="reports/ocfn_vla/train_acquisition_partial_result.json")
    parser.add_argument("--stage-a-output", default="reports/ocfn_vla/stage_a_result.json")
    parser.add_argument("--stage-a-md", default="reports/ocfn_vla/stage_a_result.md")
    parser.add_argument("--stage-a-partial-output", default="reports/ocfn_vla/stage_a_partial_result.json")
    parser.add_argument("--stage-b-output", default="reports/ocfn_vla/stage_b_result.json")
    parser.add_argument("--stage-b-md", default="reports/ocfn_vla/stage_b_result.md")
    parser.add_argument("--stage-b-partial-output", default="reports/ocfn_vla/stage_b_partial_result.json")
    parser.add_argument("--noise-count", type=int, default=4)
    parser.add_argument("--noise-seed-base", type=int, default=2026071203)
    parser.add_argument("--task-shuffle-seed", type=int, default=2026071204)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--max-action-dim", type=int, default=32)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--train-identities", type=int, default=2)
    parser.add_argument("--eval-identities", type=int, default=5)
    parser.add_argument("--stage-b-identities", type=int, default=20)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--rerun-train-acquisition", action="store_true")
    parser.add_argument("--rerun-stage-a", action="store_true")
    parser.add_argument("--rerun-stage-b", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if int(args.noise_count) != 4:
        raise SystemExit("--noise-count is preregistered as 4")
    if int(args.max_tasks) < 1 or int(args.max_tasks) > len(TASKS):
        raise SystemExit("--max-tasks must be between 1 and 2")
    if int(args.train_identities) < 1 or int(args.train_identities) > len(TRAIN_IDENTITIES):
        raise SystemExit("--train-identities must be between 1 and 2")
    if int(args.eval_identities) < 1 or int(args.eval_identities) > len(EVAL_IDENTITIES):
        raise SystemExit("--eval-identities must be between 1 and 5")
    if int(args.stage_b_identities) < 1 or int(args.stage_b_identities) > len(STAGE_B_IDENTITIES):
        raise SystemExit(f"--stage-b-identities must be between 1 and {len(STAGE_B_IDENTITIES)}")
    report = run(args)
    print(json.dumps({"mode": args.mode, "final_decision": report.get("final_decision"), "elapsed_seconds": report.get("elapsed_seconds")}, indent=2, sort_keys=True))
    return 0 if "MEASUREMENT_INVALID" not in str(report.get("final_decision")) and "FAIL" not in str(report.get("final_decision")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
