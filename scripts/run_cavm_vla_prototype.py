"""CAVM-VLA prototype runner."""

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
from tca_map.smolvla.cavm_vla import CAVMConfig, TASK_KEYS, VARIANTS, apply_cavm_action, fit_cavm_memory  # noqa: E402
from tca_map.smolvla.official_closed_loop_scaleup import _json_default  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import POLICIES, _cuda_memory, _load_policy_and_processors  # noqa: E402


DATE_KST = "2026-07-13"
BRANCH = "codex/autonomous-until-paper-governance-v2"
PROPOSAL_HASH = "849A98B2F137FC43EAA68C7B7D7DB246FEF58DD2EDBBD1F8869C4BA092DE68F2"
RESET_IDENTITY_BASE = 20260901
MAX_OFFICIAL_INITIAL_STATE_COUNT = 50
TASKS = [
    {"suite": "libero_spatial", "task_id": 4, "role": "stable_grasp_contact_transition"},
    {"suite": "libero_10", "task_id": 4, "role": "long_horizon_contact_and_release"},
]
ACQUISITION_IDENTITIES = list(range(20260901, 20260913))
CALIBRATION_IDENTITIES = list(range(20260913, 20260917))
STAGE_2A_IDENTITIES = list(range(20260917, 20260922))
STAGE_2B_IDENTITIES = list(range(20260922, 20260942))
STAGE_2B_EXPANSION_IDENTITIES = list(range(20260942, 20260951))


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
        raise ValueError(f"CAVM expected 8D state, got {array.size}")
    return array


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


def _run_episode(
    *,
    row: Mapping[str, Any],
    loaded: Mapping[str, Any],
    max_eval_steps: int,
    memory: Mapping[str, Any] | None,
    collect_records: bool = False,
    split: str | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = None
    started = time.time()
    rewards: list[float] = []
    gates: list[float] = []
    action_deltas: list[float] = []
    separations: list[float] = []
    policy_latencies: list[float] = []
    records: list[dict[str, Any]] = []
    success = False
    previous_action = np.zeros(7, dtype=np.float64)
    try:
        env, observation = _episode_context(row)
        policy = loaded["policy"]
        if hasattr(policy, "reset"):
            policy.reset()
        horizon = int((loaded.get("audit") or {}).get("action_chunk_shape", [1, 50, 7])[1])
        for step in range(_max_steps(env, max_eval_steps)):
            batch = _preprocess_batch(env, observation, dict(loaded))
            state = _batch_state(batch)
            start_policy = time.perf_counter()
            raw_action = policy.select_action(dict(batch))
            policy_latencies.append(time.perf_counter() - start_policy)
            base_action = _postprocess_action(raw_action, dict(loaded)).reshape(-1)
            rho = float(step % max(1, horizon)) / max(1.0, float(horizon))
            variant = str(row["variant"])
            diagnostics = {"gate": 0.0, "action_delta_l2": 0.0, "success_failure_separation": 0.0}
            executed = base_action
            if memory is not None and variant != "frozen_smolvla":
                executed, diagnostics = apply_cavm_action(
                    memory,
                    variant=variant,
                    state=state,
                    action=base_action,
                    previous_action=previous_action,
                    chunk_index_fraction=rho,
                    task_key=str(row["task_key"]),
                )
            gates.append(float(diagnostics.get("gate", 0.0) or 0.0))
            action_deltas.append(float(diagnostics.get("action_delta_l2", 0.0) or 0.0))
            separations.append(float(diagnostics.get("success_failure_separation", 0.0) or 0.0))
            if collect_records:
                records.append(
                    {
                        "split": str(split or "acquisition"),
                        "suite": str(row["suite"]),
                        "task_id": int(row["task_id"]),
                        "task_key": str(row["task_key"]),
                        "identity": int(row["identity"]),
                        "step": int(step),
                        "state": state.tolist(),
                        "action": base_action.tolist(),
                        "previous_action": previous_action.tolist(),
                        "chunk_index_fraction": float(rho),
                    }
                )
            observation, step_success, done, reward_value, _info = _step_env(env, executed)
            rewards.append(reward_value)
            success = bool(success or step_success)
            previous_action = np.asarray(executed, dtype=np.float64).reshape(-1)
            if done:
                break
        steps = int(step + 1 if "step" in locals() else 0)
        for record in records:
            record["success"] = bool(success)
        heavy_calls = int(np.ceil(steps / max(1, horizon)))
        summary = {
            **dict(row),
            "success": bool(success),
            "exception": None,
            "episode_steps": steps,
            "reward_sum": _round(float(np.sum(rewards)) if rewards else 0.0, 6),
            "mean_gate": _round(float(np.mean(gates)) if gates else 0.0, 6),
            "gate_activation_rate": _round(float(np.mean(np.asarray(gates) > 1e-6)) if gates else 0.0, 6),
            "mean_action_delta_l2": _round(float(np.mean(action_deltas)) if action_deltas else 0.0, 6),
            "max_action_delta_l2": _round(float(np.max(action_deltas)) if action_deltas else 0.0, 6),
            "mean_success_failure_separation": _round(float(np.mean(separations)) if separations else 0.0, 6),
            "heavy_policy_call_count": heavy_calls,
            "heavy_policy_calls_per_step": _round(heavy_calls / max(1, steps), 6),
            "policy_latency_mean_s": _round(float(np.mean(policy_latencies)) if policy_latencies else 0.0, 6),
            "elapsed_seconds": _round(time.time() - started, 3),
            "cuda_memory": _cuda_memory_report(),
        }
        return summary, records
    except Exception as exc:  # pragma: no cover
        return (
            {
                **dict(row),
                "success": False,
                "exception": "".join(traceback.format_exception_only(type(exc), exc)).strip(),
                "episode_steps": 0,
                "elapsed_seconds": _round(time.time() - started, 3),
                "cuda_memory": _cuda_memory_report(),
            },
            [],
        )
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
        valid_rows = [row for row in variant_rows if not row.get("exception")]
        successes = int(sum(1 for row in valid_rows if bool(row.get("success"))))
        per_task: dict[str, Any] = {}
        for task in TASK_KEYS:
            task_rows = [row for row in valid_rows if row.get("task_key") == task]
            task_successes = int(sum(1 for row in task_rows if bool(row.get("success"))))
            per_task[task] = {
                "successes": task_successes,
                "total": len(task_rows),
                "rate": _round(task_successes / max(1, len(task_rows)), 6),
            }
        task_balanced = float(np.mean([value["rate"] for value in per_task.values()])) if per_task else 0.0
        by_variant[variant] = {
            "successes": successes,
            "total": len(valid_rows),
            "success_rate": _round(successes / max(1, len(valid_rows)), 6),
            "task_balanced_success_rate": _round(task_balanced, 6),
            "per_task": per_task,
            "exceptions": int(sum(1 for row in variant_rows if row.get("exception"))),
            "mean_gate": _round(float(np.mean([float(row.get("mean_gate", 0.0) or 0.0) for row in valid_rows])) if valid_rows else 0.0, 6),
            "mean_gate_activation_rate": _round(float(np.mean([float(row.get("gate_activation_rate", 0.0) or 0.0) for row in valid_rows])) if valid_rows else 0.0, 6),
            "mean_action_delta_l2": _round(float(np.mean([float(row.get("mean_action_delta_l2", 0.0) or 0.0) for row in valid_rows])) if valid_rows else 0.0, 6),
            "mean_success_failure_separation": _round(float(np.mean([float(row.get("mean_success_failure_separation", 0.0) or 0.0) for row in valid_rows])) if valid_rows else 0.0, 6),
            "mean_heavy_policy_calls_per_step": _round(float(np.mean([float(row.get("heavy_policy_calls_per_step", 0.0) or 0.0) for row in valid_rows])) if valid_rows else 0.0, 6),
            "mean_policy_latency_s": _round(float(np.mean([float(row.get("policy_latency_mean_s", 0.0) or 0.0) for row in valid_rows])) if valid_rows else 0.0, 6),
            "peak_cuda_allocated_mb": _round(max([float((row.get("cuda_memory") or {}).get("max_allocated_mb") or 0.0) for row in valid_rows] or [0.0]), 3),
        }
    strongest = max((name for name in variants if name != "cavm_full"), key=lambda name: by_variant[name]["task_balanced_success_rate"]) if "cavm_full" in variants else None
    return {
        "by_variant": by_variant,
        "strongest_baseline": strongest,
        "exception_count": int(sum(1 for row in rows if row.get("exception"))),
    }


def _paired_bootstrap_ci(deltas: list[float], *, seed: int = 2026071302, samples: int = 5000) -> list[float]:
    if not deltas:
        return [0.0, 0.0]
    arr = np.asarray(deltas, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    means = np.empty(int(samples), dtype=np.float64)
    for index in range(int(samples)):
        means[index] = float(np.mean(rng.choice(arr, size=len(arr), replace=True)))
    return [_round(float(np.quantile(means, 0.025)), 6), _round(float(np.quantile(means, 0.975)), 6)]


def _paired_vs_cavm(rows: list[Mapping[str, Any]], variants: tuple[str, ...]) -> dict[str, Any]:
    by_key = {
        (str(row.get("variant")), str(row.get("task_key")), int(row.get("identity"))): bool(row.get("success"))
        for row in rows
        if not row.get("exception")
    }
    out: dict[str, Any] = {}
    for variant in variants:
        if variant == "cavm_full":
            continue
        deltas: list[float] = []
        wins = losses = ties = 0
        for row in rows:
            if row.get("variant") != "cavm_full" or row.get("exception"):
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


def _stage_2a_decision(summary: Mapping[str, Any]) -> str:
    if int(summary.get("exception_count") or 0) > 0:
        return "STAGE_2A_MEASUREMENT_INVALID_REPAIR_REQUIRED"
    by = summary["by_variant"]
    full = by["cavm_full"]
    strongest = by[summary["strongest_baseline"]]
    ablation = by["cavm_no_contrast_ablation"]
    if float(full.get("mean_gate_activation_rate", 0.0) or 0.0) <= 0.0:
        return "STAGE_2A_CATASTROPHIC_KILL_NO_MECHANISM_ACTIVATION"
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
    full_rate = float(by["cavm_full"]["task_balanced_success_rate"])
    strongest_name = str(summary["strongest_baseline"])
    strongest_rate = float(by[strongest_name]["task_balanced_success_rate"])
    ablation_rate = float(by["cavm_no_contrast_ablation"]["task_balanced_success_rate"])
    frozen_calls = float(by["frozen_smolvla"].get("mean_heavy_policy_calls_per_step", 0.0) or 0.0)
    full_calls = float(by["cavm_full"].get("mean_heavy_policy_calls_per_step", 0.0) or 0.0)
    if float(by["cavm_full"].get("mean_gate_activation_rate", 0.0) or 0.0) <= 0.0:
        return "STAGE_2B_PERMANENT_KILL_NO_MECHANISM_ACTIVATION"
    if full_rate > strongest_rate and full_rate > ablation_rate and full_rate - strongest_rate >= 0.10 and full_calls <= frozen_calls + 1e-6:
        return "STAGE_2B_PROTOTYPE_GO"
    for explainer in ("success_only_memory_proxy", "nearest_success_replay", "cavm_no_contrast_ablation"):
        if float(by[explainer]["task_balanced_success_rate"]) >= full_rate:
            return "STAGE_2B_PERMANENT_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT"
    if full_rate < float(by["frozen_smolvla"]["task_balanced_success_rate"]):
        return "STAGE_2B_PERMANENT_KILL_WORSE_THAN_FROZEN"
    pair = paired.get(strongest_name) or {}
    ci = pair.get("paired_bootstrap_ci") or [0.0, 0.0]
    if full_rate <= strongest_rate and float(ci[1]) <= 0.10:
        return "STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    return "STAGE_2B_UNRESOLVED_EXPANSION_OPTIONAL"


def _stage_2b_expanded_decision(summary: Mapping[str, Any], paired: Mapping[str, Any]) -> str:
    decision = _stage_2b_decision(summary, paired)
    if decision == "STAGE_2B_PROTOTYPE_GO":
        return "STAGE_2B_EXPANDED_PROTOTYPE_GO"
    if decision == "STAGE_2B_PERMANENT_KILL_NO_MECHANISM_ACTIVATION":
        return "STAGE_2B_EXPANDED_PERMANENT_KILL_NO_MECHANISM_ACTIVATION"
    if decision == "STAGE_2B_PERMANENT_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT":
        return "STAGE_2B_EXPANDED_PERMANENT_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT"
    if decision == "STAGE_2B_PERMANENT_KILL_WORSE_THAN_FROZEN":
        return "STAGE_2B_EXPANDED_PERMANENT_KILL_WORSE_THAN_FROZEN"
    if decision == "STAGE_2B_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED":
        return "STAGE_2B_EXPANDED_PERMANENT_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    return "STAGE_2B_EXPANDED_NON_GO_NO_THIRD_EXPANSION"


def _load_policy(args: argparse.Namespace) -> dict[str, Any]:
    _set_runtime_env(args)
    return _load_policy_and_processors(args, POLICIES[0])


def _acquire_calibrate_mode(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    record_path = Path(args.acquisition_records)
    summary_path = Path(args.acquisition_summary)
    memory_path = Path(args.memory_config)
    memory: dict[str, Any] | None = None
    if record_path.exists() and summary_path.exists() and not bool(args.rerun_acquisition):
        records = _read_jsonl(record_path)
        episode_summaries = json.loads(summary_path.read_text(encoding="utf-8-sig")).get("episodes") or []
        if memory_path.exists():
            memory = json.loads(memory_path.read_text(encoding="utf-8-sig"))
    else:
        loaded = _load_policy(args)
        records = []
        episode_summaries = []
        for split, identities in [("acquisition", ACQUISITION_IDENTITIES), ("calibration", CALIBRATION_IDENTITIES)]:
            rows = _planned_rows(TASKS[: int(args.max_tasks)], identities, ("frozen_smolvla",))
            for row in rows:
                summary, new_records = _run_episode(
                    row=row,
                    loaded=loaded,
                    max_eval_steps=int(args.max_eval_steps),
                    memory=None,
                    collect_records=True,
                    split=split,
                )
                episode_summaries.append({**summary, "split": split})
                records.extend(new_records)
                _write_jsonl(record_path, records)
                _write_json(summary_path, {"episodes": episode_summaries, "record_count": len(records)})
    if memory is None:
        acquisition_records = [record for record in records if record.get("split") == "acquisition"]
        calibration_records = [record for record in records if record.get("split") == "calibration"]
        memory = fit_cavm_memory(acquisition_records, calibration_records, CAVMConfig())
        _write_json(memory_path, memory)
    final = str(memory.get("final_decision"))
    report = {
        "mode": "acquire-calibrate",
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": True,
        "training_happened": False,
        "reset_identity_base": RESET_IDENTITY_BASE,
        "acquisition_identities": ACQUISITION_IDENTITIES,
        "calibration_identities": CALIBRATION_IDENTITIES,
        "record_count": len(records),
        "episode_summaries": episode_summaries,
        "memory_config_path": str(memory_path),
        "memory_config_sha256": _sha256_file(memory_path),
        "summary": {
            "episode_counts": memory.get("episode_counts"),
            "calibration_metrics": memory.get("calibration_metrics"),
            "sigma": memory.get("sigma"),
            "eta": memory.get("eta"),
            "gamma": memory.get("gamma"),
        },
        "final_decision": final,
        "next_step": "Run Stage 2A." if final == "STAGE_1_PROCEED_TO_STAGE_2A" else "Archive kill according to governance.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }
    return report


def _stage_2_mode(args: argparse.Namespace, *, stage: str) -> dict[str, Any]:
    started = time.time()
    loaded = _load_policy(args)
    memory_path = Path(args.memory_config)
    memory = json.loads(memory_path.read_text(encoding="utf-8-sig"))
    if stage == "stage-2a":
        identities = STAGE_2A_IDENTITIES[: int(args.stage_2a_identities)]
    elif stage == "stage-2b-expansion":
        identities = STAGE_2B_IDENTITIES + STAGE_2B_EXPANSION_IDENTITIES
    else:
        identities = STAGE_2B_IDENTITIES[: int(args.stage_2b_identities)]
    rows = _planned_rows(TASKS[: int(args.max_tasks)], identities, VARIANTS)
    if stage == "stage-2a":
        partial_path = Path(args.stage_2a_partial_output)
    elif stage == "stage-2b-expansion":
        partial_path = Path(args.stage_2b_expansion_partial_output)
    else:
        partial_path = Path(args.stage_2b_partial_output)
    episodes: list[dict[str, Any]] = []
    if stage == "stage-2b-expansion" and Path(args.stage_2b_output).exists():
        episodes = list(json.loads(Path(args.stage_2b_output).read_text(encoding="utf-8-sig")).get("episodes") or [])
    if partial_path.exists() and not bool(args.rerun_stage_2):
        episodes = list(json.loads(partial_path.read_text(encoding="utf-8-sig")).get("episodes") or [])
    completed = {(row.get("variant"), row.get("task_key"), int(row.get("identity", -1))) for row in episodes}
    for row in rows:
        key = (row["variant"], row["task_key"], int(row["identity"]))
        if key in completed:
            continue
        result, _records = _run_episode(row=row, loaded=loaded, max_eval_steps=int(args.max_eval_steps), memory=memory)
        episodes.append(result)
        _write_json(partial_path, {"episodes": episodes, "planned_episode_count": len(rows)})
    summary = _summarize(episodes, VARIANTS)
    paired = _paired_vs_cavm(episodes, VARIANTS)
    if stage == "stage-2a":
        final = _stage_2a_decision(summary)
    elif stage == "stage-2b-expansion":
        final = _stage_2b_expanded_decision(summary, paired)
    else:
        final = _stage_2b_decision(summary, paired)
    return {
        "mode": stage,
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": True,
        "training_happened": False,
        "reset_identity_base": RESET_IDENTITY_BASE,
        "identities": identities,
        "memory_config_path": str(memory_path),
        "memory_config_sha256": _sha256_file(memory_path),
        "planned_episode_count": len(rows),
        "completed_episode_count": len(episodes),
        "episodes": episodes,
        "summary": summary,
        "paired_vs_cavm_full": paired,
        "final_decision": final,
        "next_step": "Run Stage 2B." if stage == "stage-2a" and final.endswith("STAGE_2B_REQUIRED") else "Archive, expand, or scale according to governance.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "acquire-calibrate":
        report = _acquire_calibrate_mode(args)
        _write_json(Path(args.stage_1_output), report)
        _write_md(Path(args.stage_1_md), "CAVM-VLA Stage 0/1 Result", report)
        return report
    if args.mode == "stage-2a":
        report = _stage_2_mode(args, stage="stage-2a")
        _write_json(Path(args.stage_2a_output), report)
        _write_md(Path(args.stage_2a_md), "CAVM-VLA Stage 2A Result", report)
        return report
    if args.mode == "stage-2b":
        report = _stage_2_mode(args, stage="stage-2b")
        _write_json(Path(args.stage_2b_output), report)
        _write_md(Path(args.stage_2b_md), "CAVM-VLA Stage 2B Result", report)
        return report
    if args.mode == "stage-2b-expansion":
        report = _stage_2_mode(args, stage="stage-2b-expansion")
        _write_json(Path(args.stage_2b_expansion_output), report)
        _write_md(Path(args.stage_2b_expansion_md), "CAVM-VLA Stage 2B Expansion Result", report)
        return report
    raise ValueError(args.mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["acquire-calibrate", "stage-2a", "stage-2b", "stage-2b-expansion"], required=True)
    parser.add_argument("--base-path", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--acquisition-records", default="reports/cavm_vla/acquisition_records.jsonl")
    parser.add_argument("--acquisition-summary", default="reports/cavm_vla/acquisition_summary.json")
    parser.add_argument("--memory-config", default="reports/cavm_vla/memory_config.json")
    parser.add_argument("--stage-1-output", default="reports/cavm_vla/stage_1_result.json")
    parser.add_argument("--stage-1-md", default="reports/cavm_vla/stage_1_result.md")
    parser.add_argument("--stage-2a-output", default="reports/cavm_vla/stage_2a_result.json")
    parser.add_argument("--stage-2a-md", default="reports/cavm_vla/stage_2a_result.md")
    parser.add_argument("--stage-2a-partial-output", default="reports/cavm_vla/stage_2a_partial_result.json")
    parser.add_argument("--stage-2b-output", default="reports/cavm_vla/stage_2b_result.json")
    parser.add_argument("--stage-2b-md", default="reports/cavm_vla/stage_2b_result.md")
    parser.add_argument("--stage-2b-partial-output", default="reports/cavm_vla/stage_2b_partial_result.json")
    parser.add_argument("--stage-2b-expansion-output", default="reports/cavm_vla/stage_2b_expansion_result.json")
    parser.add_argument("--stage-2b-expansion-md", default="reports/cavm_vla/stage_2b_expansion_result.md")
    parser.add_argument("--stage-2b-expansion-partial-output", default="reports/cavm_vla/stage_2b_expansion_partial_result.json")
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--stage-2a-identities", type=int, default=5)
    parser.add_argument("--stage-2b-identities", type=int, default=20)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--rerun-acquisition", action="store_true")
    parser.add_argument("--rerun-stage-2", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.max_tasks) < 1 or int(args.max_tasks) > len(TASKS):
        raise SystemExit("--max-tasks must be between 1 and 2")
    if int(args.stage_2a_identities) < 1 or int(args.stage_2a_identities) > len(STAGE_2A_IDENTITIES):
        raise SystemExit(f"--stage-2a-identities must be between 1 and {len(STAGE_2A_IDENTITIES)}")
    if int(args.stage_2b_identities) < 1 or int(args.stage_2b_identities) > len(STAGE_2B_IDENTITIES):
        raise SystemExit(f"--stage-2b-identities must be between 1 and {len(STAGE_2B_IDENTITIES)}")
    if set(TASK_KEYS) != {f"{task['suite']}/task_{task['task_id']}" for task in TASKS}:
        raise SystemExit("CAVM task constants disagree with runner task manifest")
    report = run(args)
    print(json.dumps({"mode": args.mode, "final_decision": report.get("final_decision"), "elapsed_seconds": report.get("elapsed_seconds")}, indent=2, sort_keys=True))
    invalid = "INVALID" in str(report.get("final_decision"))
    return 2 if invalid else 0


if __name__ == "__main__":
    raise SystemExit(main())
