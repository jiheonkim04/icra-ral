"""GCAP-VLA prototype runner.

GCAP-VLA tests whether preserving interaction-region geometric continuity in
camera tensors can improve frozen SmolVLA under controlled visual occlusion.
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

from scripts.run_echo_vla_first_prototype import _postprocess_action, _preprocess_batch  # noqa: E402
from scripts.run_phase_barrier_vla_prototype import (  # noqa: E402
    _identity_to_initial_state_index,
    _make_exact_vector_env,
    _round,
    _set_runtime_env,
    _step_success,
)
from tca_map.smolvla.gcap_vla import (  # noqa: E402
    CAMERA_KEYS,
    GCAPConfig,
    apply_rect_occlusion,
    image_mse,
    occlusion_box,
    repair_camera_tensor,
    transform_batch_images,
)
from tca_map.smolvla.official_closed_loop_scaleup import _json_default  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import POLICIES, _cuda_memory, _load_policy_and_processors  # noqa: E402


DATE_KST = "2026-07-12"
BRANCH = "codex/ral-cycle-03-gcap-vla"
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
EVAL_IDENTITIES = [20260713, 20260714, 20260715, 20260716, 20260717]
VARIANTS = [
    "occluded_frozen_smolvla",
    "full_frame_hold_last",
    "sobel_edge_boost",
    "gcap_no_temporal_ablation",
    "gcap_full",
]
CLEAN_VARIANTS = ["clean_frozen_smolvla", "clean_gcap_full"]


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


def _planned_rows(tasks: list[Mapping[str, Any]], identities: list[int], include_clean: bool) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        for task in tasks:
            for identity in identities:
                rows.append(
                    {
                        "variant": variant,
                        "condition": "occluded",
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


def _episode_key(row: Mapping[str, Any]) -> str:
    return "|".join([str(row.get("variant")), str(row.get("suite")), str(row.get("task_id")), str(row.get("identity")), str(row.get("condition"))])


def _synthetic_image(height: int = 64, width: int = 64, shift: int = 0) -> Any:
    import torch

    y = torch.linspace(0, 1, height).reshape(1, 1, height, 1)
    x = torch.linspace(0, 1, width).reshape(1, 1, 1, width)
    diagonal = ((x + y + float(shift) / max(1, width)) % 1.0)
    edge_bar = ((torch.abs(x - 0.52) < 0.035) | (torch.abs(y - 0.56) < 0.035)).float()
    image = torch.cat([diagonal, 1.0 - diagonal, 0.35 + 0.65 * edge_bar], dim=1)
    return image.float()


def _synthetic_mode(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    start = time.time()
    config = GCAPConfig(edge_gain=float(args.edge_gain), temporal_blend=float(args.temporal_blend), mask_dilate=int(args.mask_dilate))
    clean = _synthetic_image()
    previous = _synthetic_image(shift=-3)
    box = occlusion_box(height=clean.shape[2], width=clean.shape[3], identity=20260714, step_fraction=0.35, camera_key=CAMERA_KEYS[0], config=config)
    occluded, mask = apply_rect_occlusion(clean, box=box, fill_value=float(config.fill_value))
    hold = repair_camera_tensor(occluded, previous=previous, mask=mask, variant="full_frame_hold_last", config=config)
    edge_only = repair_camera_tensor(occluded, previous=None, mask=mask, variant="gcap_no_temporal_ablation", config=config)
    full = repair_camera_tensor(occluded, previous=previous, mask=mask, variant="gcap_full", config=config)
    checks = {
        "mask_nonempty": bool(mask.sum().item() > 0),
        "occlusion_degrades_image": bool(image_mse(occluded, clean) > 0.01),
        "full_improves_over_occluded": bool(image_mse(full, clean) < image_mse(occluded, clean)),
        "full_improves_over_edge_only": bool(image_mse(full, clean) < image_mse(edge_only, clean)),
        "full_not_worse_than_hold_last": bool(image_mse(full, clean) <= image_mse(hold, clean) + 1e-6),
        "finite_outputs": bool(torch.isfinite(full).all().item()),
    }
    return {
        "mode": "synthetic",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "training_happened": False,
        "closed_loop_experiment_happened": False,
        "config": config.to_json(),
        "box": None if box is None else [int(value) for value in box],
        "mse": {
            "occluded": _round(image_mse(occluded, clean), 6),
            "hold_last": _round(image_mse(hold, clean), 6),
            "edge_only": _round(image_mse(edge_only, clean), 6),
            "gcap_full": _round(image_mse(full, clean), 6),
        },
        "checks": checks,
        "summary": {"synthetic_passed": bool(all(checks.values()))},
        "final_decision": "SYNTHETIC_MECHANISM_PASS" if all(checks.values()) else "SYNTHETIC_MECHANISM_FAIL",
        "next_step": "Run Stage A." if all(checks.values()) else "Repair or kill synthetic GCAP implementation.",
        "elapsed_seconds": _round(time.time() - start, 3),
    }


def _policy_action_for_variant(
    *,
    policy: Any,
    env: Any,
    observation: Any,
    loaded: Mapping[str, Any],
    row: Mapping[str, Any],
    step_fraction: float,
    memory: dict[str, Any],
    config: GCAPConfig,
) -> tuple[np.ndarray, dict[str, Any]]:
    import torch

    batch = _preprocess_batch(env, observation, dict(loaded))
    transformed, diagnostics = transform_batch_images(
        batch,
        variant=str(row["variant"]),
        condition=str(row["condition"]),
        identity=int(row["identity"]),
        step_fraction=float(step_fraction),
        memory=memory,
        config=config,
    )
    with torch.inference_mode():
        action = policy.select_action(transformed)
    return _postprocess_action(action, dict(loaded)).reshape(1, -1), diagnostics


def _run_episode(*, row: Mapping[str, Any], loaded: Mapping[str, Any], config: GCAPConfig, max_eval_steps: int) -> dict[str, Any]:
    env = None
    started = time.time()
    mask_fracs: list[float] = []
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
                step_fraction=float(step) / max(1.0, float(max_steps - 1)),
                memory=memory,
                config=config,
            )
            mask_values = list((diagnostics.get("mean_mask_fraction") or {}).values())
            if mask_values:
                mask_fracs.append(float(np.mean(mask_values)))
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
            "mean_mask_fraction": _round(float(np.mean(mask_fracs)) if mask_fracs else 0.0, 6),
            "active_mask_step_fraction": _round(float(np.mean([value > 0.0 for value in mask_fracs])) if mask_fracs else 0.0, 6),
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
    for variant in VARIANTS + CLEAN_VARIANTS:
        rows = [row for row in episodes if str(row.get("variant")) == variant]
        if not rows:
            continue
        successes = int(sum(1 for row in rows if bool(row.get("success"))))
        total = int(len(rows))
        per_task = {}
        for task_key in sorted({str(row["task_key"]) for row in rows}):
            task_rows = [row for row in rows if str(row["task_key"]) == task_key]
            task_successes = int(sum(1 for row in task_rows if bool(row.get("success"))))
            per_task[task_key] = {"successes": task_successes, "total": len(task_rows), "rate": _round(task_successes / max(1, len(task_rows)), 6)}
        task_balanced = float(np.mean([entry["rate"] for entry in per_task.values()])) if per_task else 0.0
        by_variant[variant] = {
            "condition": str(rows[0].get("condition")),
            "successes": successes,
            "total": total,
            "task_balanced_success_rate": _round(task_balanced, 6),
            "wilson_95_ci": _wilson_ci(successes, total),
            "per_task": per_task,
            "mean_mask_fraction": _round(float(np.mean([row.get("mean_mask_fraction") or 0.0 for row in rows])), 6),
        }
    occluded_baselines = ["occluded_frozen_smolvla", "full_frame_hold_last", "sobel_edge_boost"]
    baseline_rates = {name: by_variant.get(name, {}).get("task_balanced_success_rate", 0.0) for name in occluded_baselines}
    strongest = max(baseline_rates, key=lambda name: float(baseline_rates[name])) if baseline_rates else None
    strongest_rate = float(baseline_rates.get(strongest, 0.0)) if strongest else 0.0
    full_rate = float(by_variant.get("gcap_full", {}).get("task_balanced_success_rate", 0.0))
    hold_rate = float(by_variant.get("full_frame_hold_last", {}).get("task_balanced_success_rate", 0.0))
    no_temporal_rate = float(by_variant.get("gcap_no_temporal_ablation", {}).get("task_balanced_success_rate", 0.0))
    frozen_occluded_rate = float(by_variant.get("occluded_frozen_smolvla", {}).get("task_balanced_success_rate", 0.0))
    clean_frozen = float(by_variant.get("clean_frozen_smolvla", {}).get("task_balanced_success_rate", 0.0))
    clean_full = float(by_variant.get("clean_gcap_full", {}).get("task_balanced_success_rate", 0.0))
    clean_drop = clean_frozen - clean_full if "clean_frozen_smolvla" in by_variant and "clean_gcap_full" in by_variant else 0.0
    exception_count = int(sum(1 for row in episodes if row.get("exception")))
    passes_go = bool(exception_count == 0 and full_rate >= strongest_rate + 0.05 and full_rate > hold_rate and full_rate > no_temporal_rate and clean_drop <= 0.02)
    if exception_count:
        decision = "MEASUREMENT_INVALID_REPAIR_OR_KILL"
    elif clean_drop > 0.02:
        decision = "CLEAN_RETENTION_FAILURE"
    elif full_rate <= frozen_occluded_rate:
        decision = "NO_OCCLUSION_ROBUSTNESS_GAIN"
    elif hold_rate >= full_rate:
        decision = "SIMPLE_TEMPORAL_BASELINE_EXPLAINS_METHOD"
    elif no_temporal_rate >= full_rate:
        decision = "TEMPORAL_COMPONENT_NOT_USEFUL"
    elif passes_go:
        decision = "PROTOTYPE_GO"
    elif full_rate > strongest_rate and clean_drop <= 0.02:
        decision = "UNDERPOWERED_ONE_EXPANSION_ALLOWED"
    else:
        decision = "GENUINE_METHOD_KILL"
    return {
        "by_variant": by_variant,
        "strongest_occluded_baseline": strongest,
        "strongest_occluded_baseline_rate": _round(strongest_rate, 6),
        "clean_retention_drop": _round(clean_drop, 6),
        "exception_count": exception_count,
        "passes_prototype_go": passes_go,
        "method_decision": decision,
    }


def _load_partial(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        print(f"[gcap-stage-a] ignoring unreadable partial result {path}: {exc}", flush=True)
        return None


def _write_partial(path: Path, planned: list[Mapping[str, Any]], episodes: list[Mapping[str, Any]]) -> None:
    _write_json(
        path,
        {
            "schema_version": "gcap_vla_stage_a_partial_v1",
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
    args.base_path = str(Path(args.checkpoint))
    args.lora_root = getattr(args, "lora_root", "/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    loaded = _load_policy_and_processors(args, POLICIES[0])
    config = GCAPConfig(edge_gain=float(args.edge_gain), temporal_blend=float(args.temporal_blend), mask_dilate=int(args.mask_dilate))
    planned = _planned_rows(TASKS[: int(args.max_tasks)], [int(x) for x in str(args.stage_a_identities).split(",") if x], bool(args.include_clean_retention))
    partial_path = Path(args.stage_a_partial_json)
    partial = _load_partial(partial_path)
    completed = {_episode_key(row): dict(row) for row in (partial or {}).get("episodes", [])}
    episodes: list[Mapping[str, Any]] = []
    for row in planned:
        key = _episode_key(row)
        if key in completed:
            print(f"[gcap-stage-a] skip completed {len(completed)}/{len(planned)}: {key}", flush=True)
            continue
        result = _run_episode(row=row, loaded=loaded, config=config, max_eval_steps=int(args.max_eval_steps))
        completed[key] = result
        episodes = [completed[_episode_key(item)] for item in planned if _episode_key(item) in completed]
        _write_partial(partial_path, planned, episodes)
        print(f"[gcap-stage-a] completed {len(episodes)}/{len(planned)}: {key} success={result.get('success')} exception={bool(result.get('exception'))}", flush=True)
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
        "config": config.to_json(),
        "variants": VARIANTS + CLEAN_VARIANTS,
        "tasks": TASKS[: int(args.max_tasks)],
        "identities": [int(x) for x in str(args.stage_a_identities).split(",") if x],
        "episodes": episodes,
        "summary": summary,
        "cuda_memory": _cuda_memory_report(),
        "final_decision": str(summary["method_decision"]),
        "next_step": "Follow the Stage A method decision.",
        "elapsed_seconds": _round(time.time() - start, 3),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "synthetic":
        return _synthetic_mode(args)
    if args.mode == "stage-a":
        return _stage_a_mode(args)
    raise ValueError(f"unknown mode {args.mode}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["synthetic", "stage-a"], default="synthetic")
    parser.add_argument("--checkpoint", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--edge-gain", type=float, default=0.08)
    parser.add_argument("--temporal-blend", type=float, default=1.0)
    parser.add_argument("--mask-dilate", type=int, default=5)
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--stage-a-identities", default="20260713,20260714,20260715,20260716,20260717")
    parser.add_argument("--include-clean-retention", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--stage-a-partial-json", default="reports/gcap_vla/stage_a_partial_result.json")
    parser.add_argument("--result-json", default=None)
    parser.add_argument("--result-md", default=None)
    args = parser.parse_args()

    default_json = {
        "synthetic": "reports/gcap_vla/synthetic_result.json",
        "stage-a": "reports/gcap_vla/stage_a_result.json",
    }
    default_md = {
        "synthetic": "reports/gcap_vla/synthetic_result.md",
        "stage-a": "reports/gcap_vla/stage_a_result.md",
    }
    result_json = Path(args.result_json or default_json[args.mode])
    result_md = Path(args.result_md or default_md[args.mode])

    report = run(args)
    _write_json(result_json, report)
    _write_md(result_md, "GCAP-VLA Prototype Result", report)
    print(json.dumps({"final_decision": report.get("final_decision"), "summary": report.get("summary")}, indent=2, sort_keys=True, default=_json_default))


if __name__ == "__main__":
    main()
