"""FEDO-VLA prototype runner.

This script trains a small feedback execution-disturbance observer and evaluates
it under a controlled action-realization fault in official SmolVLA/LIBERO.
"""

from __future__ import annotations

import argparse
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

from scripts.run_phase_barrier_vla_prototype import (  # noqa: E402
    _identity_to_initial_state_index,
    _make_exact_vector_env,
    _policy_action,
    _round,
    _set_runtime_env,
    _step_success,
)
from tca_map.smolvla.fedo_vla import (  # noqa: E402
    FEDOConfig,
    FEDOExample,
    apex_feedback_proxy_action,
    apply_control_fault,
    assert_no_privileged_inference_fields,
    build_fedo_examples,
    file_sha256,
    inverse_fault_command,
    load_fedo_checkpoint,
    make_fedo_features,
    phase_from_fraction,
    predict_fedo_command,
    save_fedo_checkpoint,
    static_inverse_gain_action,
    train_fedo_compensator,
)
from tca_map.smolvla.official_closed_loop_scaleup import _json_default  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import POLICIES, _cuda_memory, _load_policy_and_processors  # noqa: E402


DATE_KST = "2026-07-12"
BRANCH = "codex/ral-cycle-02-fedo-vla"
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
FAULTED_VARIANTS = [
    "faulted_frozen_smolvla",
    "static_inverse_gain",
    "apex_feedback_proxy",
    "fedo_no_feedback_ablation",
    "fedo_full",
]
CLEAN_VARIANTS = [
    "clean_frozen_smolvla",
    "clean_fedo_full",
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
        f"- full checkpoint: `{report.get('full_checkpoint_path')}`",
        f"- full checkpoint sha256: `{report.get('full_checkpoint_sha256')}`",
        f"- no-feedback checkpoint: `{report.get('no_feedback_checkpoint_path')}`",
        f"- no-feedback checkpoint sha256: `{report.get('no_feedback_checkpoint_sha256')}`",
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


def _episode_key(row: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(row.get("variant")),
            str(row.get("suite")),
            str(row.get("task_id")),
            str(row.get("identity")),
            str(row.get("condition")),
        ]
    )


def _planned_rows(tasks: list[Mapping[str, Any]], identities: list[int], include_clean: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in FAULTED_VARIANTS:
        for task in tasks:
            for identity in identities:
                rows.append(
                    {
                        "variant": variant,
                        "condition": "faulted",
                        "suite": str(task["suite"]),
                        "task_id": int(task["task_id"]),
                        "task_key": _task_key(task),
                        "role": str(task["role"]),
                        "identity": int(identity),
                    }
                )
    if include_clean:
        for variant in CLEAN_VARIANTS:
            for task in tasks:
                for identity in identities:
                    rows.append(
                        {
                            "variant": variant,
                            "condition": "clean",
                            "suite": str(task["suite"]),
                            "task_id": int(task["task_id"]),
                            "task_key": _task_key(task),
                            "role": str(task["role"]),
                            "identity": int(identity),
                        }
                    )
    return rows


def _make_synthetic_actions(count: int = 96) -> list[np.ndarray]:
    actions = []
    for index in range(int(count)):
        frac = index / max(1, count - 1)
        actions.append(
            np.asarray(
                [
                    0.28 * np.sin(2.0 * np.pi * frac),
                    -0.22 * np.cos(np.pi * frac),
                    0.16 * (frac - 0.5),
                    0.04 * np.sin(np.pi * frac),
                    -0.03,
                    0.02,
                    0.85 if frac < 0.45 else -0.85,
                ],
                dtype=np.float32,
            )
        )
    return actions


def _synthetic_mode(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    config = FEDOConfig(hidden_dim=int(args.hidden_dim), max_residual_norm=float(args.max_residual_norm))
    actions = _make_synthetic_actions(int(args.synthetic_count))
    full_examples = build_fedo_examples(
        actions,
        identities=TRAIN_IDENTITIES,
        task_keys=[_task_key(task) for task in TASKS],
        config=config,
        use_feedback=True,
        use_phase=True,
    )
    ablation_examples = build_fedo_examples(
        actions,
        identities=TRAIN_IDENTITIES,
        task_keys=[_task_key(task) for task in TASKS],
        config=config,
        use_feedback=False,
        use_phase=False,
    )
    full_model, full_stats = train_fedo_compensator(full_examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=11)
    ablation_model, ablation_stats = train_fedo_compensator(ablation_examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=22)
    full_path = Path(args.full_checkpoint)
    ablation_path = Path(args.no_feedback_checkpoint)
    save_fedo_checkpoint(full_path, full_model, full_stats)
    save_fedo_checkpoint(ablation_path, ablation_model, ablation_stats)

    heldout = _make_synthetic_actions(24)
    realized_errors = defaultdict(list)
    prev_command: np.ndarray | None = None
    prev_realized: np.ndarray | None = None
    for index, action in enumerate(heldout):
        step_fraction = index / max(1, len(heldout) - 1)
        identity = EVAL_IDENTITIES[index % len(EVAL_IDENTITIES)]
        static_cmd = static_inverse_gain_action(action, step_fraction=step_fraction)
        apex_cmd = apex_feedback_proxy_action(action, previous_command=prev_command, previous_realized=prev_realized)
        features = make_fedo_features(
            action,
            previous_command=prev_command,
            previous_realized=prev_realized,
            step_fraction=step_fraction,
            task_key="libero_spatial/task_4",
            config=config,
        )
        full_cmd = predict_fedo_command(full_model, features, action)
        for name, command in [
            ("faulted_frozen_smolvla", action),
            ("static_inverse_gain", static_cmd),
            ("apex_feedback_proxy", apex_cmd),
            ("fedo_full", full_cmd),
        ]:
            realized = apply_control_fault(command, identity=identity, step_fraction=step_fraction, action_dim=config.action_dim)
            realized_errors[name].append(float(np.linalg.norm(realized - action)))
        prev_command = full_cmd
        prev_realized = apply_control_fault(full_cmd, identity=identity, step_fraction=step_fraction, action_dim=config.action_dim)
    summary = {name: _round(float(np.mean(values)), 6) for name, values in sorted(realized_errors.items())}
    passed = bool(summary["fedo_full"] < summary["faulted_frozen_smolvla"] and full_stats["loss_decreased"])
    return {
        "mode": "synthetic",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": True,
        "closed_loop_experiment_happened": False,
        "config": config.__dict__,
        "full_loaded_stats": full_stats,
        "ablation_loaded_stats": ablation_stats,
        "full_checkpoint_path": str(full_path),
        "full_checkpoint_sha256": file_sha256(full_path),
        "no_feedback_checkpoint_path": str(ablation_path),
        "no_feedback_checkpoint_sha256": file_sha256(ablation_path),
        "summary": {"mean_realized_error_norm": summary, "synthetic_passed": passed},
        "final_decision": "SYNTHETIC_MECHANISM_PASS" if passed else "SYNTHETIC_MECHANISM_FAIL",
        "next_step": "Run real trace training." if passed else "Kill or repair synthetic FEDO implementation.",
        "elapsed_seconds": _round(time.time() - start, 3),
    }


def _collect_trace_actions(args: argparse.Namespace, loaded: Mapping[str, Any]) -> list[dict[str, Any]]:
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
                    action = _policy_action(policy, env, observation, loaded).reshape(-1)
                    if step % int(args.train_stride) == 0:
                        rows.append(
                            {
                                "action": [float(value) for value in action],
                                "suite": str(task["suite"]),
                                "task_id": int(task["task_id"]),
                                "task_key": _task_key(task),
                                "identity": int(identity),
                                "step": int(step),
                                "step_fraction": float(step) / max(1.0, float(max_steps - 1)),
                            }
                        )
                    observation, _reward, terminated, truncated, info = env.step(action.reshape(1, -1))
                    if np.all(terminated | truncated) or _step_success(info):
                        break
                    if len(rows) >= int(args.train_action_limit):
                        return rows
            finally:
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
    return rows


def _examples_from_trace_rows(
    rows: list[Mapping[str, Any]],
    *,
    config: FEDOConfig,
    use_feedback: bool,
    use_phase: bool,
) -> list[FEDOExample]:
    examples: list[FEDOExample] = []
    previous_command: np.ndarray | None = None
    previous_realized: np.ndarray | None = None
    for row in rows:
        action = np.asarray(row["action"], dtype=np.float32)
        step_fraction = float(row["step_fraction"])
        identity = int(row["identity"])
        target_command = inverse_fault_command(action, identity=identity, step_fraction=step_fraction, action_dim=config.action_dim)
        target_residual = np.clip(target_command - action, -float(config.max_residual_norm), float(config.max_residual_norm))
        features = make_fedo_features(
            action,
            previous_command=previous_command,
            previous_realized=previous_realized,
            step_fraction=step_fraction,
            task_key=str(row["task_key"]),
            config=config,
            use_feedback=use_feedback,
            use_phase=use_phase,
        )
        examples.append(
            FEDOExample(
                features=features,
                target_residual=[float(value) for value in target_residual],
                step_fraction=step_fraction,
                phase=phase_from_fraction(step_fraction),
                uses_feedback=bool(use_feedback),
                uses_phase=bool(use_phase),
            )
        )
        previous_command = target_command
        previous_realized = apply_control_fault(target_command, identity=identity, step_fraction=step_fraction, action_dim=config.action_dim)
    return examples


def _real_trace_train_mode(args: argparse.Namespace) -> dict[str, Any]:
    start = time.time()
    _set_runtime_env(args)
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    loaded = _load_policy_and_processors(args, POLICIES[0])
    config = FEDOConfig(hidden_dim=int(args.hidden_dim), max_residual_norm=float(args.max_residual_norm))
    rows = _collect_trace_actions(args, loaded)
    if len(rows) < 8:
        raise RuntimeError(f"not enough trace rows for FEDO training: {len(rows)}")
    full_examples = _examples_from_trace_rows(rows, config=config, use_feedback=True, use_phase=True)
    ablation_examples = _examples_from_trace_rows(rows, config=config, use_feedback=False, use_phase=False)
    full_model, full_stats = train_fedo_compensator(full_examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=31)
    ablation_model, ablation_stats = train_fedo_compensator(ablation_examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=41)
    full_path = Path(args.full_checkpoint)
    ablation_path = Path(args.no_feedback_checkpoint)
    save_fedo_checkpoint(full_path, full_model, full_stats)
    save_fedo_checkpoint(ablation_path, ablation_model, ablation_stats)
    return {
        "mode": "real-trace-train",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": True,
        "closed_loop_experiment_happened": False,
        "trace_row_count": len(rows),
        "trace_rows": rows,
        "config": config.__dict__,
        "full_loaded_stats": full_stats,
        "ablation_loaded_stats": ablation_stats,
        "full_checkpoint_path": str(full_path),
        "full_checkpoint_sha256": file_sha256(full_path),
        "no_feedback_checkpoint_path": str(ablation_path),
        "no_feedback_checkpoint_sha256": file_sha256(ablation_path),
        "cuda_memory": _cuda_memory_report(),
        "summary": {
            "full_loss_decreased": bool(full_stats["loss_decreased"]),
            "ablation_loss_decreased": bool(ablation_stats["loss_decreased"]),
        },
        "final_decision": "REAL_TRACE_TRAIN_PASS"
        if full_stats["loss_decreased"] and ablation_stats["loss_decreased"]
        else "REAL_TRACE_TRAIN_FAIL",
        "next_step": "Run Stage A." if full_stats["loss_decreased"] and ablation_stats["loss_decreased"] else "Kill or repair training.",
        "elapsed_seconds": _round(time.time() - start, 3),
    }


def _variant_command(
    *,
    variant: str,
    action: np.ndarray,
    previous_command: np.ndarray | None,
    previous_realized: np.ndarray | None,
    step_fraction: float,
    task_key: str,
    full_model: Any,
    ablation_model: Any,
    config: FEDOConfig,
) -> np.ndarray:
    if variant in {"faulted_frozen_smolvla", "clean_frozen_smolvla"}:
        return np.asarray(action, dtype=np.float32).reshape(-1)
    if variant == "static_inverse_gain":
        return static_inverse_gain_action(action, step_fraction=step_fraction, action_dim=config.action_dim)
    if variant == "apex_feedback_proxy":
        return apex_feedback_proxy_action(action, previous_command=previous_command, previous_realized=previous_realized, action_dim=config.action_dim)
    if variant == "fedo_no_feedback_ablation":
        features = make_fedo_features(
            action,
            previous_command=previous_command,
            previous_realized=previous_realized,
            step_fraction=step_fraction,
            task_key=task_key,
            config=config,
            use_feedback=False,
            use_phase=False,
        )
        return predict_fedo_command(ablation_model, features, action)
    if variant in {"fedo_full", "clean_fedo_full"}:
        features = make_fedo_features(
            action,
            previous_command=previous_command,
            previous_realized=previous_realized,
            step_fraction=step_fraction,
            task_key=task_key,
            config=config,
            use_feedback=True,
            use_phase=True,
        )
        return predict_fedo_command(full_model, features, action)
    raise ValueError(f"unknown FEDO variant: {variant}")


def _run_episode(
    *,
    row: Mapping[str, Any],
    loaded: Mapping[str, Any],
    full_model: Any,
    ablation_model: Any,
    config: FEDOConfig,
) -> dict[str, Any]:
    env = None
    started = time.time()
    command_norms: list[float] = []
    residual_norms: list[float] = []
    realized_error_norms: list[float] = []
    try:
        env = _make_exact_vector_env(str(row["suite"]), int(row["task_id"]), _identity_to_initial_state_index(int(row["identity"])))
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        observation, _ = env.reset(seed=[int(row["identity"])])
        max_steps = int(env.call("_max_episode_steps")[0])
        previous_command: np.ndarray | None = None
        previous_realized: np.ndarray | None = None
        success = False
        steps = 0
        for step in range(max_steps):
            step_fraction = float(step) / max(1.0, float(max_steps - 1))
            intended = _policy_action(policy, env, observation, loaded).reshape(-1).astype(np.float32)
            command = _variant_command(
                variant=str(row["variant"]),
                action=intended,
                previous_command=previous_command,
                previous_realized=previous_realized,
                step_fraction=step_fraction,
                task_key=str(row["task_key"]),
                full_model=full_model,
                ablation_model=ablation_model,
                config=config,
            )
            if row.get("condition") == "faulted":
                realized = apply_control_fault(
                    command,
                    identity=int(row["identity"]),
                    step_fraction=step_fraction,
                    action_dim=config.action_dim,
                )
            else:
                realized = command
            command_norms.append(float(np.linalg.norm(command)))
            residual_norms.append(float(np.linalg.norm(command - intended)))
            realized_error_norms.append(float(np.linalg.norm(realized - intended)))
            observation, _reward, terminated, truncated, info = env.step(realized.reshape(1, -1))
            steps = int(step + 1)
            previous_command = command
            previous_realized = realized
            if _step_success(info):
                success = True
                break
            if np.all(terminated | truncated):
                break
        return {
            **dict(row),
            "success": bool(success),
            "exception": None,
            "episode_steps": int(steps),
            "elapsed_seconds": _round(time.time() - started, 3),
            "mean_command_norm": _round(float(np.mean(command_norms)) if command_norms else 0.0, 6),
            "mean_residual_norm": _round(float(np.mean(residual_norms)) if residual_norms else 0.0, 6),
            "mean_realized_error_norm": _round(float(np.mean(realized_error_norms)) if realized_error_norms else 0.0, 6),
            "cuda_memory": _cuda_memory_report(),
        }
    except Exception as exc:  # pragma: no cover - exercised by real runner only
        return {
            **dict(row),
            "success": False,
            "exception": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
            "episode_steps": 0,
            "elapsed_seconds": _round(time.time() - started, 3),
            "mean_command_norm": None,
            "mean_residual_norm": None,
            "mean_realized_error_norm": None,
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
    faulted = [row for row in episodes if row.get("condition") == "faulted"]
    clean = [row for row in episodes if row.get("condition") == "clean"]
    for condition_rows in (faulted, clean):
        per_variant = defaultdict(list)
        for row in condition_rows:
            per_variant[str(row["variant"])].append(row)
        for variant, rows in per_variant.items():
            successes = int(sum(1 for row in rows if bool(row.get("success"))))
            total = int(len(rows))
            per_task: dict[str, Any] = {}
            for task_key in sorted({str(row["task_key"]) for row in rows}):
                task_rows = [row for row in rows if str(row["task_key"]) == task_key]
                task_successes = int(sum(1 for row in task_rows if bool(row.get("success"))))
                per_task[task_key] = {
                    "successes": task_successes,
                    "total": len(task_rows),
                    "rate": _round(task_successes / max(1, len(task_rows)), 6),
                }
            task_balanced = float(np.mean([entry["rate"] for entry in per_task.values()])) if per_task else 0.0
            by_variant[variant] = {
                "condition": str(rows[0].get("condition")),
                "successes": successes,
                "total": total,
                "task_balanced_success_rate": _round(task_balanced, 6),
                "wilson_95_ci": _wilson_ci(successes, total),
                "per_task": per_task,
                "mean_residual_norm": _round(float(np.mean([row.get("mean_residual_norm") or 0.0 for row in rows])), 6),
                "mean_realized_error_norm": _round(float(np.mean([row.get("mean_realized_error_norm") or 0.0 for row in rows])), 6),
            }
    faulted_rates = {
        key: value["task_balanced_success_rate"]
        for key, value in by_variant.items()
        if value.get("condition") == "faulted"
    }
    full_rate = float(faulted_rates.get("fedo_full", 0.0))
    frozen_rate = float(faulted_rates.get("faulted_frozen_smolvla", 0.0))
    static_rate = float(faulted_rates.get("static_inverse_gain", 0.0))
    apex_rate = float(faulted_rates.get("apex_feedback_proxy", 0.0))
    ablation_rate = float(faulted_rates.get("fedo_no_feedback_ablation", 0.0))
    baseline_rates = {key: faulted_rates.get(key, 0.0) for key in ["faulted_frozen_smolvla", "static_inverse_gain", "apex_feedback_proxy"]}
    strongest_baseline = max(baseline_rates, key=lambda key: float(baseline_rates[key])) if baseline_rates else None
    strongest_rate = float(baseline_rates.get(strongest_baseline, 0.0)) if strongest_baseline else 0.0
    clean_frozen = float((by_variant.get("clean_frozen_smolvla") or {}).get("task_balanced_success_rate", 0.0))
    clean_full = float((by_variant.get("clean_fedo_full") or {}).get("task_balanced_success_rate", 0.0))
    clean_drop = clean_frozen - clean_full if "clean_frozen_smolvla" in by_variant and "clean_fedo_full" in by_variant else 0.0
    exception_count = int(sum(1 for row in episodes if row.get("exception")))
    passes_go = bool(
        exception_count == 0
        and full_rate >= strongest_rate + 0.05
        and full_rate > static_rate
        and full_rate > apex_rate
        and full_rate > ablation_rate
        and clean_drop <= 0.02
    )
    if exception_count:
        method_decision = "MEASUREMENT_INVALID_REPAIR_OR_KILL"
    elif clean_drop > 0.02:
        method_decision = "CLEAN_RETENTION_FAILURE"
    elif full_rate <= frozen_rate:
        method_decision = "NO_FAULT_ROBUSTNESS_GAIN"
    elif static_rate >= full_rate:
        method_decision = "SIMPLE_BASELINE_EXPLAINS_METHOD"
    elif apex_rate >= full_rate:
        method_decision = "DIRECT_PRIOR_EXPLAINS_METHOD"
    elif ablation_rate >= full_rate:
        method_decision = "KEY_COMPONENT_NOT_USEFUL"
    elif passes_go:
        method_decision = "PROTOTYPE_GO"
    elif full_rate > strongest_rate and clean_drop <= 0.02:
        method_decision = "UNDERPOWERED_ONE_EXPANSION_ALLOWED"
    else:
        method_decision = "GENUINE_METHOD_KILL"
    return {
        "by_variant": by_variant,
        "strongest_faulted_baseline": strongest_baseline,
        "strongest_faulted_baseline_rate": _round(strongest_rate, 6),
        "clean_retention_drop": _round(clean_drop, 6),
        "exception_count": exception_count,
        "passes_prototype_go": passes_go,
        "method_decision": method_decision,
    }


def _load_partial(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[fedo-stage-a] ignoring unreadable partial result {path}: {exc}", flush=True)
        return None


def _write_partial(path: Path, planned: list[Mapping[str, Any]], episodes: list[Mapping[str, Any]]) -> None:
    _write_json(
        path,
        {
            "schema_version": "fedo_vla_stage_a_partial_v1",
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
    assert_no_privileged_inference_fields(
        [
            "intended_action",
            "previous_command",
            "previous_realized_action",
            "previous_execution_error",
            "step_fraction",
            "task_key",
        ]
    )
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    loaded = _load_policy_and_processors(args, POLICIES[0])
    full_model, full_stats = load_fedo_checkpoint(args.full_checkpoint)
    ablation_model, ablation_stats = load_fedo_checkpoint(args.no_feedback_checkpoint)
    config = full_model.config
    planned = _planned_rows(TASKS[: int(args.max_tasks)], [int(x) for x in str(args.stage_a_identities).split(",") if x], bool(args.include_clean_retention))
    partial_path = Path(args.stage_a_partial_json)
    partial = _load_partial(partial_path)
    episodes: list[Mapping[str, Any]] = []
    if partial:
        episodes = list(partial.get("episodes") or [])
    completed = {_episode_key(row): dict(row) for row in episodes}
    for index, row in enumerate(planned, start=1):
        key = _episode_key(row)
        if key in completed:
            print(f"[fedo-stage-a] skip completed {len(completed)}/{len(planned)}: {key}", flush=True)
            continue
        result = _run_episode(
            row=row,
            loaded=loaded,
            full_model=full_model,
            ablation_model=ablation_model,
            config=config,
        )
        completed[key] = result
        episodes = [completed[_episode_key(item)] for item in planned if _episode_key(item) in completed]
        _write_partial(partial_path, planned, episodes)
        print(
            "[fedo-stage-a] completed "
            f"{len(episodes)}/{len(planned)}: {key} success={result.get('success')} exception={bool(result.get('exception'))}",
            flush=True,
        )
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
        "episodes": episodes,
        "summary": summary,
        "full_loaded_stats": full_stats,
        "ablation_loaded_stats": ablation_stats,
        "full_checkpoint_path": str(args.full_checkpoint),
        "full_checkpoint_sha256": file_sha256(args.full_checkpoint),
        "no_feedback_checkpoint_path": str(args.no_feedback_checkpoint),
        "no_feedback_checkpoint_sha256": file_sha256(args.no_feedback_checkpoint),
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
    parser.add_argument("--max-residual-norm", type=float, default=0.75)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--synthetic-count", type=int, default=96)
    parser.add_argument("--train-identities", type=int, default=3)
    parser.add_argument("--train-stride", type=int, default=8)
    parser.add_argument("--train-action-limit", type=int, default=360)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--stage-a-identities", default="20260713,20260714,20260715,20260716,20260717")
    parser.add_argument("--include-clean-retention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--full-checkpoint", default="reports/fedo_vla/checkpoints/fedo_full.pt")
    parser.add_argument("--no-feedback-checkpoint", default="reports/fedo_vla/checkpoints/fedo_no_feedback.pt")
    parser.add_argument("--stage-a-partial-json", default="reports/fedo_vla/stage_a_partial_result.json")
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--result-md", default=None)
    args = parser.parse_args()

    default_json = {
        "synthetic": "reports/fedo_vla/synthetic_result.json",
        "real-trace-train": "reports/fedo_vla/real_trace_train_result.json",
        "stage-a": "reports/fedo_vla/stage_a_result.json",
    }
    default_md = {
        "synthetic": "reports/fedo_vla/synthetic_result.md",
        "real-trace-train": "reports/fedo_vla/real_trace_train_result.md",
        "stage-a": "reports/fedo_vla/stage_a_result.md",
    }
    result_json = Path(args.result_json or default_json[args.mode])
    result_md = Path(args.result_md or default_md[args.mode])

    report = run(args)
    _write_json(result_json, report)
    _write_md(result_md, "FEDO-VLA Prototype Result", report)
    print(json.dumps({"final_decision": report.get("final_decision"), "summary": report.get("summary")}, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
