"""CensorCredit-VLA second implemented prototype."""

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
    DATE_KST,
    RESET_IDENTITIES,
    TASKS,
    _effect_from_observations,
    _identity_to_initial_state_index,
    _make_exact_vector_env,
    _policy_action,
    _restore_observation_from_flat_state,
    _round,
    _set_runtime_env,
    _sim_body_positions,
    _state_flat,
    _step_success,
    _write_json,
    _write_md,
)
from tca_map.smolvla.censored_credit_vla import (  # noqa: E402
    CensorRecord,
    fit_censor_credit,
    simple_temporal_ema,
    temporal_feature_dict,
    temporal_hold_blend,
    vla_corrector_jump_proxy,
)
from tca_map.smolvla.echo_vla import compatibility_score, stable_hash  # noqa: E402
from tca_map.smolvla.official_closed_loop_scaleup import _json_default  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import POLICIES, _cuda_memory, _load_policy_and_processors  # noqa: E402


BRANCH = "codex/autonomous-ral-research-implementation-v2"
VARIANTS = [
    "frozen_smolvla",
    "vla_corrector_jump_proxy",
    "simple_temporal_ema",
    "uncensored_recovery_ablation",
    "censor_credit_full",
]


def _score_phase_for_fraction(frac: float) -> str:
    if frac < 0.25:
        return "approach"
    if frac < 0.45:
        return "grasp_contact"
    if frac < 0.78:
        return "transport"
    return "placement"


def _capture_states(task: Mapping[str, Any], identity: int, loaded: Mapping[str, Any], fractions: list[float]) -> list[dict[str, Any]]:
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
        previous_action = None
        for step in range(max_steps):
            action = _policy_action(policy, env, observation, loaded)
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
                        "state_flat": flat,
                        "state_hash": stable_hash(flat),
                        "base_action": action.reshape(-1).tolist(),
                        "previous_action": None if previous_action is None else previous_action.reshape(-1).tolist(),
                    }
                )
            previous_action = action
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


def _training_actions(base_action: np.ndarray, previous_action: np.ndarray | None) -> list[dict[str, Any]]:
    base = np.asarray(base_action, dtype=np.float64).reshape(1, -1)
    prev = None if previous_action is None else np.asarray(previous_action, dtype=np.float64).reshape(1, -1)
    rows = [
        {"name": "default", "action": base},
        {"name": "ema", "action": simple_temporal_ema(base, previous_action=prev, ema_strength=0.35)},
        {"name": "jump_plus", "action": np.clip(base + np.asarray([[0.35, -0.35, 0.0, 0.35, 0.0, -0.35, 0.0]]), -1.0, 1.0)},
        {"name": "hold_previous", "action": base if prev is None else prev},
    ]
    return rows


def _evaluate_prefix_with_recovery(state: Mapping[str, Any], action: np.ndarray, recovery_action: np.ndarray, horizon: int) -> dict[str, Any]:
    env = None
    try:
        env = _make_exact_vector_env(str(state["suite"]), int(state["task_id"]), int(state["initial_state_index"]))
        env.reset(seed=[int(state["identity"])])
        observation = _restore_observation_from_flat_state(env, np.asarray(state["state_flat"], dtype=np.float64))
        start_observation = observation
        start_body_positions = _sim_body_positions(env)
        prefix_success = False
        observation, _reward, terminated, truncated, info = env.step(np.asarray(action, dtype=np.float64).reshape(1, -1))
        prefix_success = bool(prefix_success or _step_success(info))
        prefix_body_positions = _sim_body_positions(env)
        prefix_effect = _effect_from_observations(
            start_observation,
            observation,
            np.asarray(action, dtype=np.float64).reshape(1, -1),
            str(state["instruction"]),
            prefix_success,
            start_body_positions,
            prefix_body_positions,
        )
        recovered_success = prefix_success
        final_observation = observation
        for _ in range(max(0, int(horizon) - 1)):
            if np.all(terminated | truncated):
                break
            final_observation, _reward, terminated, truncated, info = env.step(np.asarray(recovery_action, dtype=np.float64).reshape(1, -1))
            recovered_success = bool(recovered_success or _step_success(info))
        final_body_positions = _sim_body_positions(env)
        recovered_effect = _effect_from_observations(
            start_observation,
            final_observation,
            np.repeat(np.asarray(recovery_action, dtype=np.float64).reshape(1, -1), int(horizon), axis=0),
            str(state["instruction"]),
            recovered_success,
            start_body_positions,
            final_body_positions,
        )
        frac = float(state["step"]) / max(1.0, float(state["max_steps"]))
        phase = _score_phase_for_fraction(frac)
        prefix_score = compatibility_score(prefix_effect, phase)
        recovered_score = compatibility_score(recovered_effect, phase)
        return {
            "prefix_effect": prefix_effect,
            "recovered_effect": recovered_effect,
            "prefix_score": _round(prefix_score, 6),
            "recovered_score": _round(recovered_score, 6),
            "prefix_success": bool(prefix_success),
            "recovered_success": bool(recovered_success),
            "censored_label": 1.0 if prefix_success or prefix_score > 0.03 else -1.0,
            "uncensored_label": 1.0 if recovered_success or recovered_score > 0.03 else -1.0,
        }
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass


def _train_models(args: argparse.Namespace, loaded: Mapping[str, Any]) -> dict[str, Any]:
    tasks = TASKS[: int(args.max_tasks)]
    identities = RESET_IDENTITIES[: int(args.train_identities)]
    fractions = [float(value) for value in str(args.train_fractions).split(",") if value.strip()]
    states = []
    for task in tasks:
        for identity in identities:
            states.extend(_capture_states(task, identity, loaded, fractions))
    rows = []
    for state in states:
        base = np.asarray(state["base_action"], dtype=np.float64)
        previous = None if state.get("previous_action") is None else np.asarray(state["previous_action"], dtype=np.float64)
        recovery = simple_temporal_ema(base.reshape(1, -1), previous_action=previous, ema_strength=float(args.ema_strength)).reshape(-1)
        for candidate in _training_actions(base, previous):
            action = np.asarray(candidate["action"], dtype=np.float64).reshape(-1)
            labels = _evaluate_prefix_with_recovery(state, action, recovery, int(args.short_horizon))
            features = temporal_feature_dict(
                action,
                previous_action=previous,
                step_fraction=float(state["step"]) / max(1.0, float(state["max_steps"])),
            )
            rows.append({"state": {k: v for k, v in state.items() if k != "state_flat"}, "candidate_name": candidate["name"], "features": features, **labels})
    censored_model = fit_censor_credit([CensorRecord(row["features"], row["censored_label"]) for row in rows], l2=float(args.l2))
    uncensored_model = fit_censor_credit([CensorRecord(row["features"], row["uncensored_label"]) for row in rows], l2=float(args.l2))
    return {
        "train_tasks": tasks,
        "train_identities": identities,
        "training_state_count": len(states),
        "training_record_count": len(rows),
        "censored_positive_count": int(sum(1 for row in rows if float(row["censored_label"]) > 0.0)),
        "uncensored_positive_count": int(sum(1 for row in rows if float(row["uncensored_label"]) > 0.0)),
        "rows": rows,
        "censored_model": censored_model.to_json(),
        "uncensored_model": uncensored_model.to_json(),
    }


def _transform_action(variant: str, action: np.ndarray, previous: np.ndarray | None, features: Mapping[str, float], censored_model: Any, uncensored_model: Any, args: argparse.Namespace) -> tuple[np.ndarray, dict[str, Any]]:
    if variant == "frozen_smolvla":
        return np.asarray(action, dtype=np.float64).reshape(1, -1), {"margin": None, "transform": "none"}
    if variant == "simple_temporal_ema":
        return simple_temporal_ema(action, previous_action=previous, ema_strength=float(args.ema_strength)), {"margin": None, "transform": "ema"}
    if variant == "vla_corrector_jump_proxy":
        return vla_corrector_jump_proxy(action, previous_action=previous, jump_threshold=float(args.jump_threshold)), {"margin": None, "transform": "jump_hold_proxy"}
    if variant == "uncensored_recovery_ablation":
        margin = uncensored_model.score(features)
        return temporal_hold_blend(action, previous_action=previous, margin=margin, hold_strength=float(args.hold_strength)), {"margin": _round(margin, 6), "transform": "uncensored_hold"}
    if variant == "censor_credit_full":
        margin = censored_model.score(features)
        return temporal_hold_blend(action, previous_action=previous, margin=margin, hold_strength=float(args.hold_strength)), {"margin": _round(margin, 6), "transform": "censored_hold"}
    raise ValueError(f"unknown variant {variant}")


def _run_episode(task: Mapping[str, Any], identity: int, variant: str, loaded: Mapping[str, Any], censored_model: Any, uncensored_model: Any, args: argparse.Namespace) -> dict[str, Any]:
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
        previous_action = None
        success = False
        shaped_steps = 0
        margins = []
        deltas = []
        rewards = []
        for step in range(max_steps):
            base_action = _policy_action(policy, env, observation, loaded)
            features = temporal_feature_dict(base_action, previous_action=previous_action, step_fraction=float(step) / max(1.0, float(max_steps)))
            action, transform = _transform_action(variant, base_action, previous_action, features, censored_model, uncensored_model, args)
            if transform.get("margin") is not None:
                margins.append(float(transform["margin"]))
            delta = float(np.linalg.norm(action - base_action))
            shaped_steps += int(delta > 1e-9)
            deltas.append(delta)
            observation, reward, terminated, truncated, info = env.step(action.reshape(1, -1))
            rewards.append(float(np.asarray(reward).reshape(-1)[0]))
            previous_action = action.reshape(-1)
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
            "mean_action_delta_norm": _round(float(np.mean(deltas)) if deltas else 0.0, 6),
            "mean_margin": _round(float(np.mean(margins)) if margins else None, 6),
            "elapsed_seconds": _round(time.monotonic() - started, 3),
            "cuda_memory": _cuda_memory(torch),
            "exception": None,
        }
    except Exception as exc:  # pragma: no cover
        return {
            "variant": variant,
            "suite": str(task["suite"]),
            "task_id": int(task["task_id"]),
            "task_key": f"{task['suite']}/task_{task['task_id']}",
            "role": str(task["role"]),
            "reset_identity": int(identity),
            "success": False,
            "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc().splitlines()[-40:]},
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
        for (name, _task), task_rows in by_variant_task.items():
            if name == variant:
                task_rates.append(sum(1 for row in task_rows if row.get("success")) / max(1, len(task_rows)))
        summary[variant] = {
            "successes": successes,
            "total": len(rows),
            "success_rate": _round(successes / max(1, len(rows)), 6),
            "task_balanced_success_rate": _round(float(np.mean(task_rates)) if task_rates else 0.0, 6),
            "mean_shaped_steps": _round(float(np.mean([row.get("shaped_step_count", 0) for row in rows])), 3),
            "mean_action_delta_norm": _round(float(np.mean([row.get("mean_action_delta_norm", 0.0) or 0.0 for row in rows])), 6),
            "exceptions": int(sum(1 for row in rows if row.get("exception"))),
        }
    baselines = ["frozen_smolvla", "vla_corrector_jump_proxy", "simple_temporal_ema"]
    strongest_baseline = max(baselines, key=lambda name: summary.get(name, {}).get("task_balanced_success_rate", -1.0))
    full = float(summary.get("censor_credit_full", {}).get("task_balanced_success_rate", 0.0))
    strongest = float(summary.get(strongest_baseline, {}).get("task_balanced_success_rate", 0.0))
    ablation = float(summary.get("uncensored_recovery_ablation", {}).get("task_balanced_success_rate", 0.0))
    failure_rate_baseline = 1.0 - strongest
    failure_rate_full = 1.0 - full
    relative_failure_reduction = 0.0 if failure_rate_baseline <= 0.0 else (failure_rate_baseline - failure_rate_full) / failure_rate_baseline
    return {
        "by_variant": summary,
        "strongest_non_ablation_baseline": strongest_baseline,
        "full_task_balanced_success_rate": _round(full, 6),
        "strongest_baseline_task_balanced_success_rate": _round(strongest, 6),
        "ablation_task_balanced_success_rate": _round(ablation, 6),
        "absolute_gain_over_strongest_baseline_pp": _round(100.0 * (full - strongest), 3),
        "relative_failure_rate_reduction": _round(relative_failure_reduction, 6),
        "route_a_go": bool(100.0 * (full - strongest) >= 5.0 and full > ablation),
        "route_b_go": bool(full > strongest and full > ablation and relative_failure_reduction >= 0.10),
        "passes_prototype_go": bool((100.0 * (full - strongest) >= 5.0 and full > ablation) or (full > strongest and full > ablation and relative_failure_reduction >= 0.10)),
    }


def _write_protocol(report_dir: Path, args: argparse.Namespace) -> None:
    _write_md(
        report_dir / "censor_credit_vla_prototype_protocol.md",
        [
            "# CensorCredit-VLA Prototype Protocol",
            "",
            f"Date: {DATE_KST} KST",
            "",
            "CensorCredit-VLA trains two temporal trust models from short exact-state interventions: an uncensored recovered-outcome ablation and a censored prefix-credit model. At deployment it emits one action per policy step by blending with the previous action when the learned margin says the current prefix should not receive future recovery credit.",
            "",
            f"- tasks: `{[(item['suite'], item['task_id']) for item in TASKS[: int(args.max_tasks)]]}`",
            f"- training identities: `{RESET_IDENTITIES[: int(args.train_identities)]}`",
            f"- eval identities: `{RESET_IDENTITIES[int(args.train_identities): int(args.train_identities) + int(args.eval_identities)]}`",
            f"- variants: `{VARIANTS}`",
            "",
            "GO/KILL follows the same Route A/Route B thresholds as PhaseBarrier, with `uncensored_recovery_ablation` as the key ablation.",
        ],
    )


def run_prototype(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    started = time.monotonic()
    report_dir = Path(args.report_dir)
    _write_protocol(report_dir, args)
    report: dict[str, Any] = {
        "schema_version": "censor_credit_vla_prototype_v1",
        "date_kst": DATE_KST,
        "branch": BRANCH,
        "method": "CensorCredit-VLA",
        "training_happened": False,
        "closed_loop_experiment_happened": False,
        "tasks": TASKS[: int(args.max_tasks)],
        "variants": list(VARIANTS),
        "train": {},
        "episodes": [],
        "summary": {},
        "errors": [],
        "final_decision": "CENSOR_CREDIT_MEASUREMENT_INVALID",
        "exact_next_step": None,
    }
    loaded = None
    try:
        _set_runtime_env(args)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for official SmolVLA CensorCredit prototype")
        spec = next(item for item in POLICIES if item.name == "frozen_base")
        loaded = _load_policy_and_processors(args, spec)
        report["policy_load_audit"] = loaded["audit"]
        train = _train_models(args, loaded)
        report["training_happened"] = True
        report["train"] = train
        censored_model = fit_censor_credit([CensorRecord(row["features"], row["censored_label"]) for row in train["rows"]], l2=float(args.l2))
        uncensored_model = fit_censor_credit([CensorRecord(row["features"], row["uncensored_label"]) for row in train["rows"]], l2=float(args.l2))
        eval_start = int(args.train_identities)
        eval_identities = RESET_IDENTITIES[eval_start : eval_start + int(args.eval_identities)]
        episodes = []
        for variant in VARIANTS:
            for task in TASKS[: int(args.max_tasks)]:
                for identity in eval_identities:
                    episodes.append(_run_episode(task, identity, variant, loaded, censored_model, uncensored_model, args))
        report["episodes"] = episodes
        report["closed_loop_experiment_happened"] = True
        report["eval_manifest"] = {
            "tasks": TASKS[: int(args.max_tasks)],
            "eval_identities": eval_identities,
            "planned_episodes": len(TASKS[: int(args.max_tasks)]) * len(eval_identities) * len(VARIANTS),
        }
        report["summary"] = _summarize(episodes)
        measurement_valid = bool(episodes) and not any(row.get("exception") for row in episodes)
        if not measurement_valid:
            report["final_decision"] = "CENSOR_CREDIT_MEASUREMENT_INVALID"
            report["exact_next_step"] = "Fix runtime/measurement errors and rerun the frozen CensorCredit protocol."
        elif report["summary"]["passes_prototype_go"]:
            report["final_decision"] = "CENSOR_CREDIT_PROTOTYPE_GO"
            report["exact_next_step"] = "Scale CensorCredit-VLA with larger rollouts, second backbone, and second condition."
        else:
            report["final_decision"] = "CENSOR_CREDIT_VALID_KILL"
            report["exact_next_step"] = "Archive as second implemented valid kill; campaign may now conclude TWO_IMPLEMENTED_METHODS_KILLED if no GO exists."
    except Exception as exc:  # pragma: no cover
        report["errors"].append({"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc().splitlines()[-80:]})
        report["final_decision"] = "CENSOR_CREDIT_MEASUREMENT_INVALID"
        report["exact_next_step"] = "Resolve the runtime blocker and rerun the same frozen protocol."
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
        _write_json(report_dir / "censor_credit_vla_prototype_result.json", report)
        _write_md(
            report_dir / "censor_credit_vla_prototype_result.md",
            [
                "# CensorCredit-VLA Prototype Result",
                "",
                f"Final decision: `{report.get('final_decision')}`",
                "",
                f"- training happened: `{report.get('training_happened')}`",
                f"- closed-loop experiment happened: `{report.get('closed_loop_experiment_happened')}`",
                f"- training records: `{(report.get('train') or {}).get('training_record_count')}`",
                f"- eval manifest: `{report.get('eval_manifest')}`",
                f"- summary: `{report.get('summary')}`",
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
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--train-identities", type=int, default=1)
    parser.add_argument("--eval-identities", type=int, default=1)
    parser.add_argument("--train-fractions", default="0.0,0.35,0.65")
    parser.add_argument("--short-horizon", type=int, default=4)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--l2", type=float, default=1e-3)
    parser.add_argument("--ema-strength", type=float, default=0.35)
    parser.add_argument("--hold-strength", type=float, default=0.70)
    parser.add_argument("--jump-threshold", type=float, default=0.45)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_prototype(args)
    print(json.dumps({"final_decision": report.get("final_decision"), "summary": report.get("summary"), "errors": report.get("errors")}, indent=2, sort_keys=True, default=_json_default))
    return 0 if report.get("final_decision") != "CENSOR_CREDIT_MEASUREMENT_INVALID" else 2


if __name__ == "__main__":
    raise SystemExit(main())
