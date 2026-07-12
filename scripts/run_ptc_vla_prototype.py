"""PTC-VLA prototype runner."""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_echo_vla_first_prototype import _preprocess_batch  # noqa: E402
from scripts.run_phase_barrier_vla_prototype import (  # noqa: E402
    _identity_to_initial_state_index,
    _make_exact_vector_env,
    _policy_action,
    _round,
    _set_runtime_env,
    _step_success,
)
from tca_map.smolvla.official_closed_loop_scaleup import _json_default  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import POLICIES, _cuda_memory, _load_policy_and_processors  # noqa: E402
from tca_map.smolvla.ptc_vla import (  # noqa: E402
    PTCConfig,
    PTCExample,
    assert_no_privileged_inference_fields,
    file_sha256,
    load_ptc_checkpoint,
    mean_action_from_stats,
    phase_from_fraction,
    predict_ptc_action,
    save_ptc_checkpoint,
    train_ptc_policy,
    transition_context,
    transition_prior_from_stats,
)


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
TRAIN_IDENTITIES = [20260711, 20260712, 20260713]
EVAL_IDENTITIES = [20260713, 20260714, 20260715, 20260716, 20260717]
VARIANTS = [
    "frozen_smolvla",
    "global_mean_action",
    "phase_mean_action",
    "ptc_no_transition_ablation",
    "ptc_full",
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


def _task_key(task: Mapping[str, Any]) -> str:
    return f"{task['suite']}/task_{int(task['task_id'])}"


def _cuda_memory_report() -> dict[str, Any]:
    import torch

    return _cuda_memory(torch)


def _state_from_observation(env: Any, observation: Any, loaded: Mapping[str, Any], config: PTCConfig) -> np.ndarray:
    batch = _preprocess_batch(env, observation, dict(loaded))
    state = batch.get("observation.state")
    if state is None:
        raise RuntimeError("preprocessed observation has no observation.state")
    try:
        state_np = state.detach().to("cpu").numpy()
    except AttributeError:
        state_np = np.asarray(state)
    return np.asarray(state_np, dtype=np.float32).reshape(-1)[: int(config.state_dim)]


def _planned_rows(tasks: list[Mapping[str, Any]], identities: list[int]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
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


def _episode_key(row: Mapping[str, Any]) -> str:
    return "|".join([str(row.get("variant")), str(row.get("suite")), str(row.get("task_id")), str(row.get("identity"))])


def _synthetic_examples(count: int, config: PTCConfig) -> list[PTCExample]:
    rows: list[PTCExample] = []
    for index in range(int(count)):
        group_count = max(1, int(count) // 3)
        group = int(index) // 3
        mode = int(index) % 3
        frac = group / max(1, group_count - 1)
        state = np.asarray(
            [np.sin(np.pi * frac), np.cos(np.pi * frac), frac, frac * frac, 1.0 - frac, (-1.0) ** index * 0.1],
            dtype=np.float32,
        )
        transition_modes = [
            np.asarray([0.35, -0.10, 0.02, 0.00, 0.04, -0.02], dtype=np.float32),
            np.asarray([-0.20, 0.30, 0.08, -0.04, -0.03, 0.01], dtype=np.float32),
            np.asarray([0.05, -0.25, -0.06, 0.07, 0.02, 0.03], dtype=np.float32),
        ]
        transition = transition_modes[mode]
        action = np.asarray(
            [
                0.12 * state[0] + 0.70 * transition[0],
                -0.12 * state[1] + 0.70 * transition[1],
                0.08 * state[2] + 0.55 * transition[2],
                0.55 * transition[3],
                0.45 * transition[4],
                0.45 * transition[5],
                -0.8 if frac < 0.55 else 0.8,
            ],
            dtype=np.float32,
        )
        rows.append(
            PTCExample(
                state=[float(x) for x in state],
                transition=[float(x) for x in transition],
                action=[float(x) for x in np.clip(action, -1.0, 1.0)],
                task_key="libero_spatial/task_4" if index % 2 == 0 else "libero_10/task_4",
                step_fraction=frac,
                phase=phase_from_fraction(frac),
            )
        )
    return rows


def _synthetic_mode(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    config = PTCConfig(hidden_dim=int(args.hidden_dim))
    examples = _synthetic_examples(int(args.synthetic_count), config)
    full_model, full_stats = train_ptc_policy(examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=11, use_transition=True)
    ablation_model, ablation_stats = train_ptc_policy(examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=12, use_transition=False)
    probes = _synthetic_examples(24, config)
    full_errors = []
    ablation_errors = []
    for row in probes:
        full_action, _ = predict_ptc_action(
            full_model,
            state=row.state,
            transition=row.transition,
            step_fraction=row.step_fraction,
            task_key=row.task_key,
            use_transition=True,
        )
        ablation_action, _ = predict_ptc_action(
            ablation_model,
            state=row.state,
            transition=row.transition,
            step_fraction=row.step_fraction,
            task_key=row.task_key,
            use_transition=False,
        )
        target = np.asarray(row.action, dtype=np.float32)
        full_errors.append(float(np.linalg.norm(full_action - target)))
        ablation_errors.append(float(np.linalg.norm(ablation_action - target)))
    full_path = Path(args.full_checkpoint)
    ablation_path = Path(args.no_transition_checkpoint)
    save_ptc_checkpoint(full_path, full_model, full_stats)
    save_ptc_checkpoint(ablation_path, ablation_model, ablation_stats)
    summary = {
        "full_mean_action_l2": _round(float(np.mean(full_errors)), 6),
        "ablation_mean_action_l2": _round(float(np.mean(ablation_errors)), 6),
        "full_loss_decreased": bool(full_stats["loss_decreased"]),
        "ablation_loss_decreased": bool(ablation_stats["loss_decreased"]),
        "synthetic_passed": bool(np.mean(full_errors) < np.mean(ablation_errors) and full_stats["loss_decreased"]),
    }
    return {
        "mode": "synthetic",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": True,
        "closed_loop_experiment_happened": False,
        "config": config.to_json(),
        "full_checkpoint_path": str(full_path),
        "full_checkpoint_sha256": file_sha256(full_path),
        "no_transition_checkpoint_path": str(ablation_path),
        "no_transition_checkpoint_sha256": file_sha256(ablation_path),
        "summary": summary,
        "final_decision": "SYNTHETIC_MECHANISM_PASS" if summary["synthetic_passed"] else "SYNTHETIC_MECHANISM_FAIL",
        "next_step": "Run real trace training." if summary["synthetic_passed"] else "Repair or kill synthetic PTC implementation.",
        "elapsed_seconds": _round(time.time() - start, 3),
    }


def _collect_trace_rows(args: argparse.Namespace, loaded: Mapping[str, Any], config: PTCConfig) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for task in TASKS[: int(args.max_tasks)]:
        for identity in TRAIN_IDENTITIES[: int(args.train_identities)]:
            env = None
            try:
                env = _make_exact_vector_env(str(task["suite"]), int(task["task_id"]), _identity_to_initial_state_index(int(identity)))
                policy = loaded["policy"]
                if hasattr(policy, "reset"):
                    policy.reset()
                observation, _ = env.reset(seed=[int(identity)])
                max_steps = int(env.call("_max_episode_steps")[0])
                for step in range(max_steps):
                    state = _state_from_observation(env, observation, loaded, config)
                    action = _policy_action(policy, env, observation, loaded).reshape(-1).astype(np.float32)
                    next_observation, _reward, terminated, truncated, info = env.step(action.reshape(1, -1))
                    next_state = _state_from_observation(env, next_observation, loaded, config)
                    if step % int(args.train_stride) == 0:
                        rows.append(
                            {
                                "state": [float(x) for x in state],
                                "transition": [float(x) for x in (next_state - state)],
                                "action": [float(x) for x in action],
                                "suite": str(task["suite"]),
                                "task_id": int(task["task_id"]),
                                "task_key": _task_key(task),
                                "identity": int(identity),
                                "step": int(step),
                                "step_fraction": float(step) / max(1.0, float(max_steps - 1)),
                            }
                        )
                    observation = next_observation
                    if len(rows) >= int(args.train_action_limit):
                        return rows
                    if np.all(terminated | truncated) or _step_success(info):
                        break
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
    return rows


def _examples_from_trace_rows(rows: list[Mapping[str, Any]]) -> list[PTCExample]:
    examples: list[PTCExample] = []
    for row in rows:
        step_fraction = float(row["step_fraction"])
        examples.append(
            PTCExample(
                state=[float(x) for x in row["state"]],
                transition=[float(x) for x in row["transition"]],
                action=[float(x) for x in row["action"]],
                task_key=str(row["task_key"]),
                step_fraction=step_fraction,
                phase=phase_from_fraction(step_fraction),
            )
        )
    return examples


def _real_trace_train_mode(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    _set_runtime_env(args)
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    config = PTCConfig(hidden_dim=int(args.hidden_dim))
    loaded = _load_policy_and_processors(args, POLICIES[0])
    rows = _collect_trace_rows(args, loaded, config)
    if len(rows) < 12:
        raise RuntimeError(f"not enough trace rows for PTC training: {len(rows)}")
    examples = _examples_from_trace_rows(rows)
    full_model, full_stats = train_ptc_policy(examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=21, use_transition=True)
    ablation_model, ablation_stats = train_ptc_policy(examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=22, use_transition=False)
    full_path = Path(args.full_checkpoint)
    ablation_path = Path(args.no_transition_checkpoint)
    save_ptc_checkpoint(full_path, full_model, full_stats)
    save_ptc_checkpoint(ablation_path, ablation_model, ablation_stats)
    return {
        "mode": "real-trace-train",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": True,
        "closed_loop_experiment_happened": False,
        "trace_row_count": len(rows),
        "trace_rows": rows,
        "config": config.to_json(),
        "full_loaded_stats": full_stats,
        "ablation_loaded_stats": ablation_stats,
        "full_checkpoint_path": str(full_path),
        "full_checkpoint_sha256": file_sha256(full_path),
        "no_transition_checkpoint_path": str(ablation_path),
        "no_transition_checkpoint_sha256": file_sha256(ablation_path),
        "cuda_memory": _cuda_memory_report(),
        "summary": {
            "full_loss_decreased": bool(full_stats["loss_decreased"]),
            "ablation_loss_decreased": bool(ablation_stats["loss_decreased"]),
            "full_examples": len(examples),
        },
        "final_decision": "REAL_TRACE_TRAIN_PASS"
        if full_stats["loss_decreased"] and ablation_stats["loss_decreased"]
        else "REAL_TRACE_TRAIN_FAIL",
        "next_step": "Run Stage A." if full_stats["loss_decreased"] and ablation_stats["loss_decreased"] else "Repair or kill training.",
        "elapsed_seconds": _round(time.time() - start, 3),
    }


def _variant_action(
    *,
    variant: str,
    state: np.ndarray,
    previous_state: np.ndarray | None,
    step_fraction: float,
    task_key: str,
    frozen_action: np.ndarray | None,
    full_model: Any,
    full_stats: Mapping[str, Any],
    ablation_model: Any,
    config: PTCConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    phase = phase_from_fraction(step_fraction)
    if variant == "frozen_smolvla":
        if frozen_action is None:
            raise RuntimeError("frozen action missing")
        return frozen_action.astype(np.float32), {"transition_norm": 0.0, "scale_mean": 0.0, "action_delta_vs_ablation": 0.0}
    if variant == "global_mean_action":
        return np.asarray(full_stats["global_mean_action"], dtype=np.float32), {"transition_norm": 0.0, "scale_mean": 0.0, "action_delta_vs_ablation": 0.0}
    if variant == "phase_mean_action":
        return mean_action_from_stats(full_stats, phase=phase, task_key=task_key, config=config), {"transition_norm": 0.0, "scale_mean": 0.0, "action_delta_vs_ablation": 0.0}
    if variant == "ptc_no_transition_ablation":
        action, scale = predict_ptc_action(
            ablation_model,
            state=state,
            transition=None,
            step_fraction=step_fraction,
            task_key=task_key,
            use_transition=False,
        )
        return action, {"transition_norm": 0.0, "scale_mean": float(np.mean(scale)), "action_delta_vs_ablation": 0.0}
    if variant == "ptc_full":
        prior = transition_prior_from_stats(full_stats, phase=phase, task_key=task_key, config=config)
        context = transition_context(current_state=state, previous_state=previous_state, prior_transition=prior, config=config)
        action, scale = predict_ptc_action(
            full_model,
            state=state,
            transition=context,
            step_fraction=step_fraction,
            task_key=task_key,
            use_transition=True,
        )
        ablation_action, _ = predict_ptc_action(
            ablation_model,
            state=state,
            transition=None,
            step_fraction=step_fraction,
            task_key=task_key,
            use_transition=False,
        )
        return action, {
            "transition_norm": float(np.linalg.norm(context)),
            "scale_mean": float(np.mean(scale)),
            "action_delta_vs_ablation": float(np.linalg.norm(action - ablation_action)),
        }
    raise ValueError(f"unknown PTC variant: {variant}")


def _run_episode(*, row: Mapping[str, Any], loaded: Mapping[str, Any], full_model: Any, full_stats: Mapping[str, Any], ablation_model: Any, config: PTCConfig, max_eval_steps: int) -> dict[str, Any]:
    env = None
    started = time.time()
    transition_norms: list[float] = []
    scale_means: list[float] = []
    deltas: list[float] = []
    try:
        env = _make_exact_vector_env(str(row["suite"]), int(row["task_id"]), _identity_to_initial_state_index(int(row["identity"])))
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        observation, _ = env.reset(seed=[int(row["identity"])])
        max_steps = int(env.call("_max_episode_steps")[0])
        if int(max_eval_steps) > 0:
            max_steps = min(max_steps, int(max_eval_steps))
        previous_state: np.ndarray | None = None
        success = False
        rewards: list[float] = []
        steps = 0
        for step in range(max_steps):
            step_fraction = float(step) / max(1.0, float(max_steps - 1))
            state = _state_from_observation(env, observation, loaded, config)
            frozen_action = _policy_action(policy, env, observation, loaded).reshape(-1).astype(np.float32) if row["variant"] == "frozen_smolvla" else None
            action, diagnostics = _variant_action(
                variant=str(row["variant"]),
                state=state,
                previous_state=previous_state,
                step_fraction=step_fraction,
                task_key=str(row["task_key"]),
                frozen_action=frozen_action,
                full_model=full_model,
                full_stats=full_stats,
                ablation_model=ablation_model,
                config=config,
            )
            observation, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
            previous_state = state
            rewards.append(float(np.asarray(reward).reshape(-1)[0]))
            transition_norms.append(float(diagnostics["transition_norm"]))
            scale_means.append(float(diagnostics["scale_mean"]))
            deltas.append(float(diagnostics["action_delta_vs_ablation"]))
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
            "mean_transition_context_norm": _round(float(np.mean(transition_norms)) if transition_norms else 0.0, 6),
            "mean_scale": _round(float(np.mean(scale_means)) if scale_means else 0.0, 6),
            "mean_action_delta_vs_ablation": _round(float(np.mean(deltas)) if deltas else 0.0, 6),
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


def _wilson_ci(successes: int, total: int, z: float = 1.96) -> list[float]:
    if total <= 0:
        return [0.0, 0.0]
    phat = successes / total
    denom = 1 + z * z / total
    center = (phat + z * z / (2 * total)) / denom
    margin = z * ((phat * (1 - phat) + z * z / (4 * total)) / total) ** 0.5 / denom
    return [_round(max(0.0, center - margin), 6), _round(min(1.0, center + margin), 6)]


def _summarize_stage_a(episodes: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
        rows = [row for row in episodes if str(row.get("variant")) == variant]
        if not rows:
            continue
        successes = int(sum(1 for row in rows if bool(row.get("success"))))
        total = len(rows)
        per_task = {}
        for task_key in sorted({str(row["task_key"]) for row in rows}):
            task_rows = [row for row in rows if str(row["task_key"]) == task_key]
            task_successes = int(sum(1 for row in task_rows if bool(row.get("success"))))
            per_task[task_key] = {"successes": task_successes, "total": len(task_rows), "rate": _round(task_successes / max(1, len(task_rows)), 6)}
        task_balanced = float(np.mean([entry["rate"] for entry in per_task.values()])) if per_task else 0.0
        by_variant[variant] = {
            "successes": successes,
            "total": total,
            "task_balanced_success_rate": _round(task_balanced, 6),
            "wilson_95_ci": _wilson_ci(successes, total),
            "per_task": per_task,
            "mean_transition_context_norm": _round(float(np.mean([row.get("mean_transition_context_norm") or 0.0 for row in rows])), 6),
            "mean_action_delta_vs_ablation": _round(float(np.mean([row.get("mean_action_delta_vs_ablation") or 0.0 for row in rows])), 6),
        }
    baseline_names = ["frozen_smolvla", "global_mean_action", "phase_mean_action", "ptc_no_transition_ablation"]
    baseline_rates = {name: float(by_variant.get(name, {}).get("task_balanced_success_rate", 0.0)) for name in baseline_names}
    strongest = max(baseline_rates, key=lambda name: baseline_rates[name]) if baseline_rates else None
    strongest_rate = baseline_rates.get(strongest, 0.0) if strongest else 0.0
    full_rate = float(by_variant.get("ptc_full", {}).get("task_balanced_success_rate", 0.0))
    full_successes = int(by_variant.get("ptc_full", {}).get("successes", 0))
    exception_count = int(sum(1 for row in episodes if row.get("exception")))
    mechanism_active = bool(by_variant.get("ptc_full", {}).get("mean_transition_context_norm", 0.0) > 1e-6 and by_variant.get("ptc_full", {}).get("mean_action_delta_vs_ablation", 0.0) > 1e-6)
    if exception_count:
        decision = "MEASUREMENT_INVALID_REPAIR_OR_KILL"
    elif full_successes == 0 and any(by_variant.get(name, {}).get("successes", 0) >= 4 for name in baseline_names):
        decision = "STAGE_A_PERMANENT_KILL_ZERO_VS_BASELINE"
    elif strongest_rate - full_rate >= 0.30:
        decision = "STAGE_A_PERMANENT_KILL_CLEARLY_WORSE"
    elif full_rate > strongest_rate and mechanism_active:
        decision = "STAGE_B_REQUIRED_POSITIVE"
    else:
        decision = "STAGE_B_REQUIRED_MIXED_OR_UNDERPOWERED"
    return {
        "by_variant": by_variant,
        "strongest_baseline": strongest,
        "strongest_baseline_rate": _round(strongest_rate, 6),
        "ptc_full_rate": _round(full_rate, 6),
        "exception_count": exception_count,
        "mechanism_active": mechanism_active,
        "method_decision": decision,
    }


def _load_partial(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[ptc-stage-a] ignoring unreadable partial result {path}: {exc}", flush=True)
        return None


def _write_partial(path: Path, planned: list[Mapping[str, Any]], episodes: list[Mapping[str, Any]]) -> None:
    _write_json(
        path,
        {
            "schema_version": "ptc_vla_stage_a_partial_v1",
            "date_kst": DATE_KST,
            "branch": BRANCH,
            "planned_episode_count": len(planned),
            "completed_episode_count": len(episodes),
            "episodes": episodes,
        },
    )


def _stage_a_mode(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    _set_runtime_env(args)
    assert_no_privileged_inference_fields(["current_policy_state", "previous_policy_state", "phase", "task_code", "transition_prior"])
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    loaded = _load_policy_and_processors(args, POLICIES[0])
    full_model, full_stats = load_ptc_checkpoint(args.full_checkpoint)
    ablation_model, ablation_stats = load_ptc_checkpoint(args.no_transition_checkpoint)
    config = full_model.config
    planned = _planned_rows(TASKS[: int(args.max_tasks)], [int(x) for x in str(args.stage_a_identities).split(",") if x])
    partial_path = Path(args.stage_a_partial_json)
    partial = _load_partial(partial_path)
    completed = {_episode_key(row): dict(row) for row in (partial or {}).get("episodes", [])}
    for row in planned:
        key = _episode_key(row)
        if key in completed:
            print(f"[ptc-stage-a] skip completed {len(completed)}/{len(planned)}: {key}", flush=True)
            continue
        result = _run_episode(
            row=row,
            loaded=loaded,
            full_model=full_model,
            full_stats=full_stats,
            ablation_model=ablation_model,
            config=config,
            max_eval_steps=int(args.max_eval_steps),
        )
        completed[key] = result
        episodes = [completed[_episode_key(item)] for item in planned if _episode_key(item) in completed]
        _write_partial(partial_path, planned, episodes)
        print(f"[ptc-stage-a] completed {len(episodes)}/{len(planned)}: {key} success={result.get('success')} exception={bool(result.get('exception'))}", flush=True)
    episodes = [completed[_episode_key(item)] for item in planned if _episode_key(item) in completed]
    summary = _summarize_stage_a(list(episodes))
    return {
        "mode": "stage-a",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": True,
        "stage_a_completed": bool(len(episodes) == len(planned) and summary.get("exception_count") == 0),
        "episode_count": len(episodes),
        "planned_episode_count": len(planned),
        "variants": VARIANTS,
        "tasks": TASKS[: int(args.max_tasks)],
        "identities": [int(x) for x in str(args.stage_a_identities).split(",") if x],
        "episodes": episodes,
        "summary": summary,
        "full_loaded_stats": full_stats,
        "ablation_loaded_stats": ablation_stats,
        "full_checkpoint_path": str(args.full_checkpoint),
        "full_checkpoint_sha256": file_sha256(args.full_checkpoint),
        "no_transition_checkpoint_path": str(args.no_transition_checkpoint),
        "no_transition_checkpoint_sha256": file_sha256(args.no_transition_checkpoint),
        "cuda_memory": _cuda_memory_report(),
        "final_decision": str(summary["method_decision"]),
        "next_step": "Follow the Stage A method decision.",
        "elapsed_seconds": _round(time.time() - start, 3),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "synthetic":
        return _synthetic_mode(args)
    if args.mode == "real-trace-train":
        return _real_trace_train_mode(args)
    if args.mode == "stage-a":
        return _stage_a_mode(args)
    raise ValueError(f"unknown mode {args.mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "real-trace-train", "stage-a"], default="synthetic")
    parser.add_argument("--checkpoint", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--synthetic-count", type=int, default=96)
    parser.add_argument("--train-identities", type=int, default=3)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--train-action-limit", type=int, default=360)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--stage-a-identities", default="20260713,20260714,20260715,20260716,20260717")
    parser.add_argument("--full-checkpoint", default="reports/ptc_vla/checkpoints/ptc_full.pt")
    parser.add_argument("--no-transition-checkpoint", default="reports/ptc_vla/checkpoints/ptc_no_transition.pt")
    parser.add_argument("--stage-a-partial-json", default="reports/ptc_vla/stage_a_partial_result.json")
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--result-md", default=None)
    args = parser.parse_args()

    default_json = {
        "synthetic": "reports/ptc_vla/synthetic_result.json",
        "real-trace-train": "reports/ptc_vla/real_trace_train_result.json",
        "stage-a": "reports/ptc_vla/stage_a_result.json",
    }
    default_md = {
        "synthetic": "reports/ptc_vla/synthetic_result.md",
        "real-trace-train": "reports/ptc_vla/real_trace_train_result.md",
        "stage-a": "reports/ptc_vla/stage_a_result.md",
    }
    result_json = Path(args.result_json or default_json[args.mode])
    result_md = Path(args.result_md or default_md[args.mode])
    report = run(args)
    _write_json(result_json, report)
    _write_md(result_md, "PTC-VLA Prototype Result", report)
    print(json.dumps({"final_decision": report.get("final_decision"), "summary": report.get("summary")}, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
