"""SCVC-VLA prototype runner."""

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
from tca_map.smolvla.scvc_vla import (  # noqa: E402
    CAMERA_KEYS,
    SCVCConfig,
    apply_sensor_shift,
    known_inverse_affine,
    mechanism_active,
    merge_camera_stats,
    stats_to_json,
    tensor_stats,
    transform_batch_images,
)


DATE_KST = "2026-07-12"
BRANCH = "codex/autonomous-until-paper-governance-v2"
RESET_IDENTITY_BASE = 20260711
MAX_OFFICIAL_INITIAL_STATE_COUNT = 50
TASKS = [
    {"suite": "libero_spatial", "task_id": 4, "role": "stable_grasp_contact_transition"},
    {"suite": "libero_10", "task_id": 4, "role": "long_horizon_contact_and_release"},
]
CALIBRATION_IDENTITIES = [20260711, 20260712, 20260713, 20260714, 20260715]
EVAL_IDENTITIES = [20260716, 20260717, 20260718, 20260719, 20260720]
STAGE_B_IDENTITIES = list(range(20260721, 20260761))
VARIANTS = [
    "clean_frozen_smolvla",
    "shifted_frozen_smolvla",
    "known_inverse_affine",
    "scvc_no_temporal",
    "scvc_full",
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


def _policy_action(policy: Any, batch: Mapping[str, Any], loaded: Mapping[str, Any]) -> np.ndarray:
    import torch

    with torch.inference_mode():
        action = policy.select_action(dict(batch))
    return _postprocess_action(action, dict(loaded)).reshape(1, -1)


def _synthetic_mode(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.time()
    config = SCVCConfig(gain=float(args.gain), bias=float(args.bias), temporal_blend=float(args.temporal_blend))
    clean = torch.linspace(0.05, 0.85, 3 * 32 * 32, dtype=torch.float32).reshape(1, 3, 32, 32)
    shifted = apply_sensor_shift(clean, config)
    inverse = known_inverse_affine(shifted, config)
    target_mean, target_std = tensor_stats(clean)
    calibration = {"camera_stats": {"observation.images.camera1": stats_to_json(target_mean, target_std)}}
    batch = {"observation.images.camera1": clean}
    _, diag_full = transform_batch_images(batch, variant="scvc_full", calibration=calibration, memory={}, config=config)
    mse_shifted = float(torch.mean((shifted - clean) ** 2).item())
    mse_inverse = float(torch.mean((inverse - clean) ** 2).item())
    mse_full = float(diag_full["image_mse_output_vs_clean"]["observation.images.camera1"])
    passed = bool(mse_shifted > 0.01 and mse_inverse < mse_shifted and mse_full < mse_shifted)
    return {
        "mode": "synthetic",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": False,
        "config": config.to_json(),
        "summary": {"synthetic_passed": passed, "mse_shifted": _round(mse_shifted), "mse_inverse": _round(mse_inverse), "mse_full": _round(mse_full)},
        "final_decision": "SYNTHETIC_MECHANISM_PASS" if passed else "SYNTHETIC_MECHANISM_FAIL",
        "next_step": "Run calibration." if passed else "Repair or kill SCVC implementation.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }


def _planned_rows(tasks: list[Mapping[str, Any]], identities: list[int], *, variants: list[str] | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in (variants or ["calibration"]):
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


def _calibration_mode(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    _set_runtime_env(args)
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    config = SCVCConfig(gain=float(args.gain), bias=float(args.bias), temporal_blend=float(args.temporal_blend))
    loaded = _load_policy_and_processors(args, POLICIES[0])
    policy = loaded["policy"]
    rows: list[dict[str, Any]] = []
    for planned in _planned_rows(TASKS[: int(args.max_tasks)], CALIBRATION_IDENTITIES[: int(args.calibration_identities)]):
        env = None
        row = {**planned, "camera_stats": {}, "exception": None}
        try:
            env = _make_exact_vector_env(str(planned["suite"]), int(planned["task_id"]), _identity_to_initial_state_index(int(planned["identity"])))
            if hasattr(policy, "reset"):
                policy.reset()
            observation, _ = env.reset(seed=[int(planned["identity"])])
            max_steps = min(int(env.call("_max_episode_steps")[0]), int(args.calibration_steps))
            per_camera: dict[str, list[dict[str, list[float]]]] = {key: [] for key in CAMERA_KEYS}
            for _step in range(max_steps):
                batch = _preprocess_batch(env, observation, dict(loaded))
                for key in CAMERA_KEYS:
                    if key in batch:
                        mean, std = tensor_stats(batch[key])
                        per_camera[key].append(stats_to_json(mean, std))
                action = _policy_action(policy, batch, loaded)
                observation, _reward, terminated, truncated, info = env.step(action.reshape(1, -1))
                if _step_success(info) or np.all(terminated | truncated):
                    break
            row["camera_stats"] = merge_camera_stats([{"camera_stats": {key: merge_camera_stats([{"camera_stats": {key: item}} for item in values]).get(key) for key, values in per_camera.items() if values}}])
        except Exception as exc:  # pragma: no cover
            row["exception"] = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
        rows.append(row)
    camera_stats = merge_camera_stats(rows)
    passed = bool(camera_stats and not any(row.get("exception") for row in rows))
    return {
        "mode": "calibration",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": False,
        "config": config.to_json(),
        "calibration_rows": rows,
        "camera_stats": camera_stats,
        "summary": {"calibration_passed": passed, "row_count": len(rows), "camera_keys": sorted(camera_stats.keys())},
        "final_decision": "CALIBRATION_PASS" if passed else "CALIBRATION_FAIL",
        "next_step": "Run Stage A." if passed else "Repair or kill calibration.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }


def _policy_action_for_variant(
    *,
    policy: Any,
    env: Any,
    observation: Any,
    loaded: Mapping[str, Any],
    row: Mapping[str, Any],
    calibration: Mapping[str, Any],
    memory: dict[str, Any],
    config: SCVCConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    batch = _preprocess_batch(env, observation, dict(loaded))
    transformed, diagnostics = transform_batch_images(batch, variant=str(row["variant"]), calibration=calibration, memory=memory, config=config)
    return _policy_action(policy, transformed, loaded), diagnostics


def _run_episode(*, row: Mapping[str, Any], loaded: Mapping[str, Any], calibration: Mapping[str, Any], config: SCVCConfig, max_eval_steps: int) -> dict[str, Any]:
    env = None
    started = time.time()
    mse_shifted: list[float] = []
    mse_output: list[float] = []
    delta_shifted: list[float] = []
    try:
        env = _make_exact_vector_env(str(row["suite"]), int(row["task_id"]), _identity_to_initial_state_index(int(row["identity"])))
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        observation, _ = env.reset(seed=[int(row["identity"])])
        max_steps = int(env.call("_max_episode_steps")[0])
        if int(max_eval_steps) > 0:
            max_steps = min(max_steps, int(max_eval_steps))
        memory: dict[str, Any] = {}
        success = False
        rewards: list[float] = []
        for step in range(max_steps):
            action, diagnostics = _policy_action_for_variant(
                policy=policy,
                env=env,
                observation=observation,
                loaded=loaded,
                row=row,
                calibration=calibration,
                memory=memory,
                config=config,
            )
            mse_shifted.extend(float(v) for v in (diagnostics.get("image_mse_shifted_vs_clean") or {}).values())
            mse_output.extend(float(v) for v in (diagnostics.get("image_mse_output_vs_clean") or {}).values())
            delta_shifted.extend(float(v) for v in (diagnostics.get("image_delta_vs_shifted") or {}).values())
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
            "mean_mse_shifted_vs_clean": _round(float(np.mean(mse_shifted)) if mse_shifted else 0.0, 6),
            "mean_mse_output_vs_clean": _round(float(np.mean(mse_output)) if mse_output else 0.0, 6),
            "mean_image_delta_vs_shifted": _round(float(np.mean(delta_shifted)) if delta_shifted else 0.0, 6),
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
            "mean_mse_shifted_vs_clean": _round(float(np.mean([float(row.get("mean_mse_shifted_vs_clean", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "mean_mse_output_vs_clean": _round(float(np.mean([float(row.get("mean_mse_output_vs_clean", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
            "mean_image_delta_vs_shifted": _round(float(np.mean([float(row.get("mean_image_delta_vs_shifted", 0.0) or 0.0) for row in variant_rows])) if variant_rows else 0.0, 6),
        }
    strongest = max((name for name in VARIANTS if name != "scvc_full"), key=lambda name: by_variant[name]["task_balanced_success_rate"])
    return {
        "by_variant": by_variant,
        "strongest_baseline": strongest,
        "mechanism_active": mechanism_active({"by_variant": by_variant}),
        "exception_count": int(sum(1 for row in rows if row.get("exception"))),
    }


def _paired_bootstrap_ci(deltas: list[float], *, seed: int = 2026071217, samples: int = 5000) -> list[float]:
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
        if variant == "scvc_full":
            continue
        deltas: list[float] = []
        wins = losses = ties = 0
        for row in rows:
            if row.get("variant") != "scvc_full" or row.get("exception"):
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
    full = by["scvc_full"]
    strongest = by[summary["strongest_baseline"]]
    clean = by["clean_frozen_smolvla"]
    shifted = by["shifted_frozen_smolvla"]
    if not summary.get("mechanism_active"):
        return "STAGE_A_MECHANISM_INVALID_KILL"
    if shifted["task_balanced_success_rate"] >= clean["task_balanced_success_rate"] and full["task_balanced_success_rate"] <= shifted["task_balanced_success_rate"]:
        return "STAGE_A_NO_SHIFT_HEADROOM_KILL"
    if full["successes"] == 0 and strongest["successes"] >= 4:
        return "STAGE_A_PERMANENT_KILL_ZERO_VS_STRONG_BASELINE"
    if float(strongest["task_balanced_success_rate"]) - float(full["task_balanced_success_rate"]) >= 0.30:
        return "STAGE_A_PERMANENT_KILL_CLEARLY_WORSE"
    if full["task_balanced_success_rate"] > strongest["task_balanced_success_rate"]:
        return "STAGE_A_POSITIVE_TO_STAGE_B_REQUIRED"
    return "STAGE_A_NON_GO_TO_STAGE_B_REQUIRED"


def _stage_b_decision(summary: Mapping[str, Any], paired: Mapping[str, Any], *, expanded: bool) -> str:
    by = summary["by_variant"]
    full_rate = float(by["scvc_full"]["task_balanced_success_rate"])
    baselines = {name: data for name, data in by.items() if name != "scvc_full"}
    strongest_name, strongest = max(baselines.items(), key=lambda item: float(item[1]["task_balanced_success_rate"]))
    strongest_rate = float(strongest["task_balanced_success_rate"])
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


def _stage_a_mode(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    _set_runtime_env(args)
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    config = SCVCConfig(gain=float(args.gain), bias=float(args.bias), temporal_blend=float(args.temporal_blend))
    calibration = json.loads(Path(args.calibration_output).read_text(encoding="utf-8-sig"))
    loaded = _load_policy_and_processors(args, POLICIES[0])
    planned = _planned_rows(TASKS[: int(args.max_tasks)], EVAL_IDENTITIES[: int(args.eval_identities)], variants=VARIANTS)
    partial_path = Path(args.stage_a_partial_output)
    episodes: list[dict[str, Any]] = []
    if partial_path.exists() and not args.rerun_stage_a:
        episodes = list(json.loads(partial_path.read_text(encoding="utf-8-sig")).get("episodes") or [])
    completed = {(row.get("variant"), row.get("task_key"), int(row.get("identity", -1))) for row in episodes}
    for row in planned:
        key = (row["variant"], row["task_key"], int(row["identity"]))
        if key in completed:
            continue
        result = _run_episode(row=row, loaded=loaded, calibration=calibration, config=config, max_eval_steps=int(args.max_eval_steps))
        episodes.append(result)
        _write_json(partial_path, {"episodes": episodes, "planned_episode_count": len(planned)})
    summary = _summarize(episodes)
    final = _stage_a_decision(summary)
    return {
        "mode": "stage-a",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": True,
        "config": config.to_json(),
        "planned_episode_count": len(planned),
        "completed_episode_count": len(episodes),
        "episodes": episodes,
        "summary": summary,
        "final_decision": final,
        "next_step": "Run Stage B." if "STAGE_B_REQUIRED" in final else "Archive or repair according to governance.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }


def _stage_b_mode(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    _set_runtime_env(args)
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    config = SCVCConfig(gain=float(args.gain), bias=float(args.bias), temporal_blend=float(args.temporal_blend))
    calibration = json.loads(Path(args.calibration_output).read_text(encoding="utf-8-sig"))
    loaded = _load_policy_and_processors(args, POLICIES[0])
    identities = STAGE_B_IDENTITIES[: int(args.stage_b_identities)]
    planned = _planned_rows(TASKS[: int(args.max_tasks)], identities, variants=VARIANTS)
    partial_path = Path(args.stage_b_partial_output)
    episodes: list[dict[str, Any]] = []
    if partial_path.exists() and not args.rerun_stage_b:
        episodes = list(json.loads(partial_path.read_text(encoding="utf-8-sig")).get("episodes") or [])
    completed = {(row.get("variant"), row.get("task_key"), int(row.get("identity", -1))) for row in episodes}
    for row in planned:
        key = (row["variant"], row["task_key"], int(row["identity"]))
        if key in completed:
            continue
        result = _run_episode(row=row, loaded=loaded, calibration=calibration, config=config, max_eval_steps=int(args.max_eval_steps))
        episodes.append(result)
        _write_json(partial_path, {"episodes": episodes, "planned_episode_count": len(planned)})
    summary = _summarize(episodes)
    paired = _paired_vs_full(episodes)
    expanded = int(args.stage_b_identities) >= 40
    final = _stage_b_decision(summary, paired, expanded=expanded)
    return {
        "mode": "stage-b",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": True,
        "config": config.to_json(),
        "stage_b_identities": identities,
        "planned_episode_count": len(planned),
        "completed_episode_count": len(episodes),
        "episodes": episodes,
        "summary": summary,
        "paired_vs_full": paired,
        "final_decision": final,
        "next_step": "Expand Stage B to 80 paired episodes." if "EXPAND" in final else "Archive or scale according to governance.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "synthetic":
        report = _synthetic_mode(args)
        _write_json(Path(args.synthetic_output), report)
        _write_md(Path(args.synthetic_md), "SCVC-VLA Synthetic Result", report)
        return report
    if args.mode == "calibration":
        report = _calibration_mode(args)
        _write_json(Path(args.calibration_output), report)
        _write_md(Path(args.calibration_md), "SCVC-VLA Calibration Result", report)
        return report
    if args.mode == "stage-a":
        report = _stage_a_mode(args)
        _write_json(Path(args.stage_a_output), report)
        _write_md(Path(args.stage_a_md), "SCVC-VLA Stage A Result", report)
        return report
    if args.mode == "stage-b":
        report = _stage_b_mode(args)
        _write_json(Path(args.stage_b_output), report)
        _write_md(Path(args.stage_b_md), "SCVC-VLA Stage B Result", report)
        return report
    raise ValueError(args.mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "calibration", "stage-a", "stage-b"], required=True)
    parser.add_argument("--checkpoint", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--synthetic-output", default="reports/scvc_vla/synthetic_result.json")
    parser.add_argument("--synthetic-md", default="reports/scvc_vla/synthetic_result.md")
    parser.add_argument("--calibration-output", default="reports/scvc_vla/calibration_result.json")
    parser.add_argument("--calibration-md", default="reports/scvc_vla/calibration_result.md")
    parser.add_argument("--stage-a-output", default="reports/scvc_vla/stage_a_result.json")
    parser.add_argument("--stage-a-md", default="reports/scvc_vla/stage_a_result.md")
    parser.add_argument("--stage-a-partial-output", default="reports/scvc_vla/stage_a_partial_result.json")
    parser.add_argument("--stage-b-output", default="reports/scvc_vla/stage_b_result.json")
    parser.add_argument("--stage-b-md", default="reports/scvc_vla/stage_b_result.md")
    parser.add_argument("--stage-b-partial-output", default="reports/scvc_vla/stage_b_partial_result.json")
    parser.add_argument("--gain", type=float, default=0.42)
    parser.add_argument("--bias", type=float, default=0.28)
    parser.add_argument("--temporal-blend", type=float, default=0.80)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--calibration-identities", type=int, default=5)
    parser.add_argument("--eval-identities", type=int, default=5)
    parser.add_argument("--stage-b-identities", type=int, default=20)
    parser.add_argument("--calibration-steps", type=int, default=24)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--rerun-stage-a", action="store_true")
    parser.add_argument("--rerun-stage-b", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.max_tasks) < 1 or int(args.max_tasks) > len(TASKS):
        raise SystemExit("--max-tasks must be between 1 and 2")
    if int(args.stage_b_identities) < 1 or int(args.stage_b_identities) > len(STAGE_B_IDENTITIES):
        raise SystemExit(f"--stage-b-identities must be between 1 and {len(STAGE_B_IDENTITIES)}")
    report = run(args)
    print(json.dumps({"mode": args.mode, "final_decision": report.get("final_decision"), "elapsed_seconds": report.get("elapsed_seconds")}, indent=2, sort_keys=True))
    return 0 if "FAIL" not in str(report.get("final_decision")) and "INVALID" not in str(report.get("final_decision")) else 2


if __name__ == "__main__":
    raise SystemExit(main())
