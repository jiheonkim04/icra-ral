"""PSE-VLA prototype runner."""

from __future__ import annotations

import argparse
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
from tca_map.smolvla.pse_vla import (  # noqa: E402
    PSEConfig,
    VARIANTS,
    action_l2_delta,
    average_action_arrays,
    mechanism_active,
    transform_batch_images,
    transforms_for_variant,
)


DATE_KST = "2026-07-12"
BRANCH = "codex/autonomous-until-paper-governance-v2"
RESET_IDENTITY_BASE = 20260711
MAX_OFFICIAL_INITIAL_STATE_COUNT = 50
TASKS = [
    {"suite": "libero_spatial", "task_id": 4, "role": "stable_grasp_contact_transition"},
    {"suite": "libero_10", "task_id": 4, "role": "long_horizon_contact_and_release"},
]
EVAL_IDENTITIES = [20260741, 20260742, 20260743, 20260744, 20260745]
STAGE_B_IDENTITIES = list(range(20260721, 20260761))


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


def _policy_action_stateless(policy: Any, batch: Mapping[str, Any], loaded: Mapping[str, Any]) -> np.ndarray:
    import torch

    with torch.inference_mode():
        if hasattr(policy, "predict_action_chunk"):
            action_chunk = policy.predict_action_chunk(dict(batch))
            action = action_chunk[:, 0] if getattr(action_chunk, "ndim", 0) == 3 else action_chunk
        else:
            action = policy.select_action(dict(batch))
    return _postprocess_action(action, dict(loaded)).reshape(1, -1)


def _synthetic_mode(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.time()
    config = PSEConfig(
        bright_gain=float(args.bright_gain),
        bright_bias=float(args.bright_bias),
        dark_gain=float(args.dark_gain),
        dark_bias=float(args.dark_bias),
    )
    image = torch.linspace(0.05, 0.95, 3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16)
    bright, bright_diag = transform_batch_images({"observation.images.camera1": image}, transform="bright_low_contrast", config=config)
    dark, dark_diag = transform_batch_images({"observation.images.camera1": image}, transform="dark_high_contrast", config=config)
    clean_action = np.asarray([[0.1, 0.2, 0.3, 0.4, 0.5, -0.6, 0.7]], dtype=np.float64)
    duplicate = average_action_arrays([clean_action, clean_action.copy(), clean_action.copy()])
    full = average_action_arrays([clean_action, clean_action + 0.1, clean_action - 0.05])
    passed = bool(
        bright_diag["image_mean_abs_delta"]["observation.images.camera1"] > 0.01
        and dark_diag["image_mean_abs_delta"]["observation.images.camera1"] > 0.01
        and np.allclose(duplicate, clean_action)
        and action_l2_delta(full, clean_action) > 0.0
        and float(bright["observation.images.camera1"].min()) >= 0.0
        and float(dark["observation.images.camera1"].max()) <= 1.0
    )
    return {
        "mode": "synthetic",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": False,
        "config": config.to_json(),
        "summary": {
            "synthetic_passed": passed,
            "bright_mean_abs_delta": _round(bright_diag["image_mean_abs_delta"]["observation.images.camera1"], 6),
            "dark_mean_abs_delta": _round(dark_diag["image_mean_abs_delta"]["observation.images.camera1"], 6),
            "duplicate_delta_vs_clean": _round(action_l2_delta(duplicate, clean_action), 6),
            "full_delta_vs_clean": _round(action_l2_delta(full, clean_action), 6),
        },
        "final_decision": "SYNTHETIC_MECHANISM_PASS" if passed else "SYNTHETIC_MECHANISM_FAIL",
        "next_step": "Run Stage A." if passed else "Repair or kill PSE implementation.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }


def _planned_rows(tasks: list[Mapping[str, Any]], identities: list[int], *, variants: tuple[str, ...] = VARIANTS) -> list[dict[str, Any]]:
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


def _action_for_transform(
    *,
    policy: Any,
    batch: Mapping[str, Any],
    loaded: Mapping[str, Any],
    transform: str,
    config: PSEConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    transformed, diagnostics = transform_batch_images(batch, transform=str(transform), config=config)
    return _policy_action_stateless(policy, transformed, loaded), diagnostics


def _policy_action_for_variant(
    *,
    policy: Any,
    env: Any,
    observation: Any,
    loaded: Mapping[str, Any],
    row: Mapping[str, Any],
    config: PSEConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    batch = _preprocess_batch(env, observation, dict(loaded))
    transforms = transforms_for_variant(str(row["variant"]))
    actions: list[np.ndarray] = []
    image_abs_deltas: list[float] = []
    image_mses: list[float] = []
    transform_actions: dict[str, list[np.ndarray]] = {}
    transform_order: list[str] = []

    for transform in transforms:
        action, diagnostics = _action_for_transform(policy=policy, batch=batch, loaded=loaded, transform=transform, config=config)
        actions.append(action)
        transform_actions.setdefault(transform, []).append(action)
        transform_order.append(transform)
        image_abs_deltas.extend(float(v) for v in (diagnostics.get("image_mean_abs_delta") or {}).values())
        image_mses.extend(float(v) for v in (diagnostics.get("image_mse_vs_identity") or {}).values())

    combined = average_action_arrays(actions)
    clean_reference = transform_actions.get("identity", [None])[0]
    bright_reference = transform_actions.get("bright_low_contrast", [None])[0]
    dark_reference = transform_actions.get("dark_high_contrast", [None])[0]
    component_deltas = [action_l2_delta(combined, action) for action in actions]
    diagnostics = {
        "transform_order": transform_order,
        "transform_count": len(transforms),
        "mean_image_abs_delta": _round(float(np.mean(image_abs_deltas)) if image_abs_deltas else 0.0, 6),
        "mean_image_mse_vs_identity": _round(float(np.mean(image_mses)) if image_mses else 0.0, 6),
        "mean_component_delta": _round(float(np.mean(component_deltas)) if component_deltas else 0.0, 6),
        "delta_vs_clean": _round(action_l2_delta(combined, clean_reference), 6) if clean_reference is not None else None,
        "delta_vs_bright": _round(action_l2_delta(combined, bright_reference), 6) if bright_reference is not None else None,
        "delta_vs_dark": _round(action_l2_delta(combined, dark_reference), 6) if dark_reference is not None else None,
    }
    return combined, diagnostics


def _run_episode(*, row: Mapping[str, Any], loaded: Mapping[str, Any], config: PSEConfig, max_eval_steps: int) -> dict[str, Any]:
    env = None
    started = time.time()
    delta_vs_clean: list[float] = []
    delta_vs_bright: list[float] = []
    delta_vs_dark: list[float] = []
    component_delta: list[float] = []
    image_abs_delta: list[float] = []
    image_mse: list[float] = []
    transform_counts: list[int] = []
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
        for step in range(max_steps):
            action, diagnostics = _policy_action_for_variant(policy=policy, env=env, observation=observation, loaded=loaded, row=row, config=config)
            if diagnostics.get("delta_vs_clean") is not None:
                delta_vs_clean.append(float(diagnostics["delta_vs_clean"]))
            if diagnostics.get("delta_vs_bright") is not None:
                delta_vs_bright.append(float(diagnostics["delta_vs_bright"]))
            if diagnostics.get("delta_vs_dark") is not None:
                delta_vs_dark.append(float(diagnostics["delta_vs_dark"]))
            component_delta.append(float(diagnostics.get("mean_component_delta", 0.0) or 0.0))
            image_abs_delta.append(float(diagnostics.get("mean_image_abs_delta", 0.0) or 0.0))
            image_mse.append(float(diagnostics.get("mean_image_mse_vs_identity", 0.0) or 0.0))
            transform_counts.append(int(diagnostics.get("transform_count", 0) or 0))
            observation, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
            rewards.append(float(np.asarray(reward).reshape(-1)[0]))
            success = bool(success or _step_success(info))
            if success or np.all(terminated | truncated):
                break
        return {
            **dict(row),
            "success": bool(success),
            "exception": None,
            "episode_steps": int(step + 1 if "step" in locals() else 0),
            "reward_sum": _round(float(np.sum(rewards)) if rewards else 0.0, 6),
            "mean_delta_vs_clean": _round(float(np.mean(delta_vs_clean)) if delta_vs_clean else 0.0, 6),
            "mean_delta_vs_bright": _round(float(np.mean(delta_vs_bright)) if delta_vs_bright else 0.0, 6),
            "mean_delta_vs_dark": _round(float(np.mean(delta_vs_dark)) if delta_vs_dark else 0.0, 6),
            "mean_component_delta": _round(float(np.mean(component_delta)) if component_delta else 0.0, 6),
            "mean_image_abs_delta": _round(float(np.mean(image_abs_delta)) if image_abs_delta else 0.0, 6),
            "mean_image_mse_vs_identity": _round(float(np.mean(image_mse)) if image_mse else 0.0, 6),
            "mean_transform_count": _round(float(np.mean(transform_counts)) if transform_counts else 0.0, 6),
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


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
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
            "mean_delta_vs_clean": _round(float(np.mean([float(row.get("mean_delta_vs_clean", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "mean_delta_vs_bright": _round(float(np.mean([float(row.get("mean_delta_vs_bright", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "mean_delta_vs_dark": _round(float(np.mean([float(row.get("mean_delta_vs_dark", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "mean_component_delta": _round(float(np.mean([float(row.get("mean_component_delta", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "mean_image_abs_delta": _round(float(np.mean([float(row.get("mean_image_abs_delta", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "mean_transform_count": _round(float(np.mean([float(row.get("mean_transform_count", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
        }
    strongest = max((name for name in VARIANTS if name != "pse_full"), key=lambda name: by_variant[name]["task_balanced_success_rate"])
    return {
        "by_variant": by_variant,
        "strongest_baseline": strongest,
        "mechanism_active": mechanism_active({"by_variant": by_variant}),
        "exception_count": int(sum(1 for row in rows if row.get("exception"))),
    }


def _paired_bootstrap_ci(deltas: list[float], *, seed: int = 2026071231, samples: int = 5000) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        means[index] = float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
    return [_round(float(np.quantile(means, 0.025)), 6), _round(float(np.quantile(means, 0.975)), 6)]


def _paired_vs_full(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_key = {
        (str(row.get("variant")), str(row.get("task_key")), int(row.get("identity"))): bool(row.get("success"))
        for row in rows
        if not row.get("exception")
    }
    out: dict[str, Any] = {}
    for variant in VARIANTS:
        if variant == "pse_full":
            continue
        deltas: list[float] = []
        wins = losses = ties = 0
        for row in rows:
            if row.get("variant") != "pse_full" or row.get("exception"):
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


def _stage_a_decision(summary: Mapping[str, Any]) -> str:
    by = summary["by_variant"]
    full = by["pse_full"]
    strongest = by[summary["strongest_baseline"]]
    duplicate = by["pse_duplicate_clean"]
    if int(summary.get("exception_count") or 0) > 0:
        return "STAGE_A_MEASUREMENT_INVALID_REPAIR_REQUIRED"
    if not summary.get("mechanism_active"):
        return "STAGE_A_MECHANISM_INVALID_KILL"
    if full["successes"] == 0 and strongest["successes"] >= 4:
        return "STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE"
    if float(strongest["task_balanced_success_rate"]) - float(full["task_balanced_success_rate"]) >= 0.30:
        return "STAGE_A_PERMANENT_KILL_CLEARLY_WORSE"
    if float(duplicate["mean_delta_vs_clean"]) == 0.0 and float(full["mean_delta_vs_clean"]) == 0.0:
        return "STAGE_A_PERMANENT_KILL_EXACT_TRIVIAL_EQUIVALENCE"
    if full["task_balanced_success_rate"] > strongest["task_balanced_success_rate"]:
        return "STAGE_A_POSITIVE_TO_STAGE_B_REQUIRED"
    return "STAGE_A_NON_GO_TO_STAGE_B_REQUIRED"


def _stage_b_decision(summary: Mapping[str, Any], paired: Mapping[str, Any], *, expanded: bool) -> str:
    by = summary["by_variant"]
    full_rate = float(by["pse_full"]["task_balanced_success_rate"])
    baselines = {name: data for name, data in by.items() if name != "pse_full"}
    strongest_name, strongest = max(baselines.items(), key=lambda item: float(item[1]["task_balanced_success_rate"]))
    strongest_rate = float(strongest["task_balanced_success_rate"])
    if int(summary.get("exception_count") or 0) > 0:
        return "STAGE_B_MEASUREMENT_INVALID_REPAIR_REQUIRED"
    if not summary.get("mechanism_active"):
        return "STAGE_B_PERMANENT_KILL_MECHANISM_INACTIVE"
    if full_rate > strongest_rate and full_rate - strongest_rate >= 0.10:
        return "STAGE_B_PROTOTYPE_GO"
    pair = paired.get(strongest_name) or {}
    ci = pair.get("paired_bootstrap_ci") or [0.0, 0.0]
    upper = float(ci[1])
    if full_rate <= strongest_rate and upper <= 0.10:
        return "STAGE_B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    if full_rate <= strongest_rate - 0.10:
        return "STAGE_B_PERMANENT_KILL_CLEARLY_WORSE"
    if not expanded:
        return "STAGE_B_UNRESOLVED_EXPAND_TO_80_REQUIRED"
    return "STAGE_B_PERMANENT_KILL_UNRESOLVED_AFTER_EXPANSION"


def _rollout_mode(args: argparse.Namespace, *, stage: str) -> dict[str, Any]:
    started = time.time()
    _set_runtime_env(args)
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    config = PSEConfig(
        bright_gain=float(args.bright_gain),
        bright_bias=float(args.bright_bias),
        dark_gain=float(args.dark_gain),
        dark_bias=float(args.dark_bias),
    )
    loaded = _load_policy_and_processors(args, POLICIES[0])
    identities = EVAL_IDENTITIES[: int(args.eval_identities)] if stage == "stage-a" else STAGE_B_IDENTITIES[: int(args.stage_b_identities)]
    planned = _planned_rows(TASKS[: int(args.max_tasks)], identities)
    partial_path = Path(args.stage_a_partial_output if stage == "stage-a" else args.stage_b_partial_output)
    episodes: list[dict[str, Any]] = []
    seeded_from_stage_a_count = 0
    rerun = bool(args.rerun_stage_a if stage == "stage-a" else args.rerun_stage_b)
    if partial_path.exists() and not rerun:
        episodes = list(json.loads(partial_path.read_text(encoding="utf-8-sig")).get("episodes") or [])
    elif stage == "stage-b" and bool(args.stage_b_reuse_stage_a) and Path(args.stage_a_output).exists() and not rerun:
        stage_a_payload = json.loads(Path(args.stage_a_output).read_text(encoding="utf-8-sig"))
        planned_keys = {(row["variant"], row["task_key"], int(row["identity"])) for row in planned}
        episodes = [
            dict(row)
            for row in (stage_a_payload.get("episodes") or [])
            if (row.get("variant"), row.get("task_key"), int(row.get("identity", -1))) in planned_keys
        ]
        seeded_from_stage_a_count = len(episodes)
        if episodes:
            _write_json(partial_path, {"episodes": episodes, "planned_episode_count": len(planned), "seeded_from_stage_a_count": seeded_from_stage_a_count})
    completed = {(row.get("variant"), row.get("task_key"), int(row.get("identity", -1))) for row in episodes}
    for row in planned:
        key = (row["variant"], row["task_key"], int(row["identity"]))
        if key in completed:
            continue
        result = _run_episode(row=row, loaded=loaded, config=config, max_eval_steps=int(args.max_eval_steps))
        episodes.append(result)
        _write_json(partial_path, {"episodes": episodes, "planned_episode_count": len(planned)})
    summary = _summarize(episodes)
    paired = _paired_vs_full(episodes) if stage == "stage-b" else {}
    final = _stage_a_decision(summary) if stage == "stage-a" else _stage_b_decision(summary, paired, expanded=int(args.stage_b_identities) >= 40)
    report: dict[str, Any] = {
        "mode": stage,
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": True,
        "config": config.to_json(),
        "identities": identities,
        "planned_episode_count": len(planned),
        "completed_episode_count": len(episodes),
        "episodes": episodes,
        "summary": summary,
        "final_decision": final,
        "next_step": "Run Stage B." if stage == "stage-a" and "STAGE_B_REQUIRED" in final else "Archive or scale according to governance.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }
    if stage == "stage-b":
        report["seeded_from_stage_a_count"] = seeded_from_stage_a_count
    if paired:
        report["paired_vs_full"] = paired
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "synthetic":
        report = _synthetic_mode(args)
        _write_json(Path(args.synthetic_output), report)
        _write_md(Path(args.synthetic_md), "PSE-VLA Synthetic Result", report)
        return report
    if args.mode == "stage-a":
        report = _rollout_mode(args, stage="stage-a")
        _write_json(Path(args.stage_a_output), report)
        _write_md(Path(args.stage_a_md), "PSE-VLA Stage A Result", report)
        return report
    if args.mode == "stage-b":
        report = _rollout_mode(args, stage="stage-b")
        _write_json(Path(args.stage_b_output), report)
        _write_md(Path(args.stage_b_md), "PSE-VLA Stage B Result", report)
        return report
    raise ValueError(args.mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "stage-a", "stage-b"], required=True)
    parser.add_argument("--checkpoint", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--synthetic-output", default="reports/pse_vla/synthetic_result.json")
    parser.add_argument("--synthetic-md", default="reports/pse_vla/synthetic_result.md")
    parser.add_argument("--stage-a-output", default="reports/pse_vla/stage_a_result.json")
    parser.add_argument("--stage-a-md", default="reports/pse_vla/stage_a_result.md")
    parser.add_argument("--stage-a-partial-output", default="reports/pse_vla/stage_a_partial_result.json")
    parser.add_argument("--stage-b-output", default="reports/pse_vla/stage_b_result.json")
    parser.add_argument("--stage-b-md", default="reports/pse_vla/stage_b_result.md")
    parser.add_argument("--stage-b-partial-output", default="reports/pse_vla/stage_b_partial_result.json")
    parser.add_argument("--bright-gain", type=float, default=0.42)
    parser.add_argument("--bright-bias", type=float, default=0.28)
    parser.add_argument("--dark-gain", type=float, default=1.25)
    parser.add_argument("--dark-bias", type=float, default=-0.10)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--eval-identities", type=int, default=5)
    parser.add_argument("--stage-b-identities", type=int, default=20)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--rerun-stage-a", action="store_true")
    parser.add_argument("--rerun-stage-b", action="store_true")
    parser.add_argument("--no-stage-b-reuse-stage-a", dest="stage_b_reuse_stage_a", action="store_false")
    parser.set_defaults(stage_b_reuse_stage_a=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.max_tasks) < 1 or int(args.max_tasks) > len(TASKS):
        raise SystemExit("--max-tasks must be between 1 and 2")
    if int(args.eval_identities) < 1 or int(args.eval_identities) > len(EVAL_IDENTITIES):
        raise SystemExit(f"--eval-identities must be between 1 and {len(EVAL_IDENTITIES)}")
    if int(args.stage_b_identities) < 1 or int(args.stage_b_identities) > len(STAGE_B_IDENTITIES):
        raise SystemExit(f"--stage-b-identities must be between 1 and {len(STAGE_B_IDENTITIES)}")
    report = run(args)
    print(json.dumps({"mode": args.mode, "final_decision": report.get("final_decision"), "elapsed_seconds": report.get("elapsed_seconds")}, indent=2, sort_keys=True))
    return 0 if "FAIL" not in str(report.get("final_decision")) and "INVALID" not in str(report.get("final_decision")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
