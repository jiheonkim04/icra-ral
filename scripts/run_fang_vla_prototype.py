"""FANG-VLA closed-loop prototype runner."""

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

from scripts.run_cavm_vla_prototype import (  # noqa: E402
    _batch_state,
    _cuda_memory_report,
    _load_policy,
    _max_steps,
    _planned_rows,
    _postprocess_action,
    _preprocess_batch,
    _round,
    _sha256_file,
    _step_env,
)
from scripts.run_phase_barrier_vla_prototype import _make_exact_vector_env  # noqa: E402
from tca_map.smolvla.fang_vla import TASK_KEYS, VARIANTS, apply_fang_action, load_fang_runtime  # noqa: E402


DATE_KST = "2026-07-14"
BRANCH = "codex/autonomous-until-paper-governance-v2"
PROPOSAL_HASH = "6837DBA2A1307F7C9938FA9F5463ED483907AF3C168F1C0514F6E281804E859B"
FANG_RESET_IDENTITY_BASE = 20261001
MAX_OFFICIAL_INITIAL_STATE_COUNT = 50
TASKS = [
    {"suite": "libero_spatial", "task_id": 4, "role": "stable_grasp_contact_transition"},
    {"suite": "libero_10", "task_id": 4, "role": "long_horizon_contact_and_release"},
]
STAGE_A_IDENTITIES = list(range(20261001, 20261006))
STAGE_B_IDENTITIES = list(range(20261006, 20261026))
STAGE_B_EXPANSION_IDENTITIES = list(range(20261026, 20261046))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _write_md(path: Path, title: str, report: Mapping[str, Any]) -> None:
    summary = report.get("summary") or {}
    by_variant = summary.get("by_variant") or {}
    lines = [
        f"# {title}",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Final decision: `{report.get('final_decision')}`",
        "",
        f"- closed-loop experiment happened: `{report.get('closed_loop_experiment_happened')}`",
        f"- completed episodes: `{report.get('completed_episode_count')}` / `{report.get('planned_episode_count')}`",
        f"- exception count: `{summary.get('exception_count')}`",
        f"- strongest baseline: `{summary.get('strongest_baseline')}`",
        "",
        "| Variant | Successes | Total | Task-Balanced Success | Mean Gate | Mean Delta |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for variant in VARIANTS:
        item = by_variant.get(variant) or {}
        lines.append(
            f"| `{variant}` | {item.get('successes')} | {item.get('total')} | {item.get('task_balanced_success_rate')} | {item.get('mean_gate')} | {item.get('mean_action_delta_l2')} |"
        )
    lines.extend(["", f"Next step: {report.get('next_step')}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _identity_to_initial_state_index(identity: int) -> int:
    index = int(identity) - FANG_RESET_IDENTITY_BASE
    if index < 0 or index >= MAX_OFFICIAL_INITIAL_STATE_COUNT:
        raise ValueError(f"identity {identity} maps to invalid official initial state index {index}")
    return index


def _episode_context(row: Mapping[str, Any]) -> tuple[Any, Any]:
    env = _make_exact_vector_env(str(row["suite"]), int(row["task_id"]), _identity_to_initial_state_index(int(row["identity"])))
    observation, _ = env.reset(seed=[int(row["identity"])])
    return env, observation


def _run_episode(
    *,
    row: Mapping[str, Any],
    loaded: Mapping[str, Any],
    runtime: Mapping[str, Any],
    max_eval_steps: int,
) -> dict[str, Any]:
    env = None
    started = time.time()
    rewards: list[float] = []
    gates: list[float] = []
    action_deltas: list[float] = []
    head_separations: list[float] = []
    policy_latencies: list[float] = []
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
            executed, diagnostics = apply_fang_action(
                runtime,
                variant=str(row["variant"]),
                state=state,
                action=base_action,
                previous_action=previous_action,
                chunk_index_fraction=rho,
                task_key=str(row["task_key"]),
            )
            gates.append(float(diagnostics.get("gate", 0.0) or 0.0))
            action_deltas.append(float(diagnostics.get("action_delta_l2", 0.0) or 0.0))
            head_separations.append(float(diagnostics.get("head_separation", 0.0) or 0.0))
            observation, step_success, done, reward_value, _info = _step_env(env, executed)
            rewards.append(reward_value)
            success = bool(success or step_success)
            previous_action = np.asarray(executed, dtype=np.float64).reshape(-1)
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
            "mean_gate": _round(float(np.mean(gates)) if gates else 0.0, 6),
            "gate_activation_rate": _round(float(np.mean(np.asarray(gates) > 0.05)) if gates else 0.0, 6),
            "mean_action_delta_l2": _round(float(np.mean(action_deltas)) if action_deltas else 0.0, 6),
            "max_action_delta_l2": _round(float(np.max(action_deltas)) if action_deltas else 0.0, 6),
            "mean_head_separation": _round(float(np.mean(head_separations)) if head_separations else 0.0, 6),
            "heavy_policy_call_count": heavy_calls,
            "heavy_policy_calls_per_step": _round(heavy_calls / max(1, steps), 6),
            "policy_latency_mean_s": _round(float(np.mean(policy_latencies)) if policy_latencies else 0.0, 6),
            "elapsed_seconds": _round(time.time() - started, 3),
            "cuda_memory": _cuda_memory_report(),
        }
    except Exception as exc:  # pragma: no cover
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


def _summarize(rows: list[Mapping[str, Any]]) -> dict[str, Any]:
    by_variant: dict[str, Any] = {}
    for variant in VARIANTS:
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
            "mean_head_separation": _round(float(np.mean([float(row.get("mean_head_separation", 0.0) or 0.0) for row in valid_rows])) if valid_rows else 0.0, 6),
            "mean_heavy_policy_calls_per_step": _round(float(np.mean([float(row.get("heavy_policy_calls_per_step", 0.0) or 0.0) for row in valid_rows])) if valid_rows else 0.0, 6),
            "mean_policy_latency_s": _round(float(np.mean([float(row.get("policy_latency_mean_s", 0.0) or 0.0) for row in valid_rows])) if valid_rows else 0.0, 6),
            "peak_cuda_allocated_mb": _round(max([float((row.get("cuda_memory") or {}).get("max_allocated_mb") or 0.0) for row in valid_rows] or [0.0]), 3),
        }
    strongest = max((name for name in VARIANTS if name != "fang_full"), key=lambda name: by_variant[name]["task_balanced_success_rate"]) if "fang_full" in VARIANTS else None
    return {
        "by_variant": by_variant,
        "strongest_baseline": strongest,
        "exception_count": int(sum(1 for row in rows if row.get("exception"))),
    }


def _paired_bootstrap_ci(deltas: list[float], *, seed: int = 2026071401, samples: int = 5000) -> list[float]:
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
        if variant == "fang_full":
            continue
        deltas: list[float] = []
        wins = losses = ties = 0
        for row in rows:
            if row.get("variant") != "fang_full" or row.get("exception"):
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
    if int(summary.get("exception_count") or 0) > 0:
        return "STAGE_A_MEASUREMENT_INVALID_REPAIR_REQUIRED"
    by = summary["by_variant"]
    full = by["fang_full"]
    strongest = by[summary["strongest_baseline"]]
    ablation = by["fang_no_failure_ablation"]
    prior = by["afil_local_proxy"]
    if float(full.get("mean_gate_activation_rate", 0.0) or 0.0) <= 0.0:
        return "STAGE_A_KILL_NO_MECHANISM_ACTIVATION"
    if full["successes"] == 0 and max(strongest["successes"], ablation["successes"], prior["successes"]) >= 4:
        return "STAGE_A_KILL_ZERO_VS_STRONG_BASELINE"
    if float(strongest["task_balanced_success_rate"]) - float(full["task_balanced_success_rate"]) >= 0.30:
        return "STAGE_A_KILL_CLEAR_BASELINE_DOMINANCE"
    if float(ablation["task_balanced_success_rate"]) - float(full["task_balanced_success_rate"]) >= 0.30:
        return "STAGE_A_KILL_CLEAR_ABLATION_DOMINANCE"
    if float(prior["task_balanced_success_rate"]) - float(full["task_balanced_success_rate"]) >= 0.30:
        return "STAGE_A_KILL_CLEAR_PRIOR_PROXY_DOMINANCE"
    return "STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"


def _stage_b_decision(summary: Mapping[str, Any], paired: Mapping[str, Any]) -> str:
    if int(summary.get("exception_count") or 0) > 0:
        return "STAGE_B_MEASUREMENT_INVALID_REPAIR_REQUIRED"
    by = summary["by_variant"]
    full_rate = float(by["fang_full"]["task_balanced_success_rate"])
    strongest_name = str(summary["strongest_baseline"])
    strongest_rate = float(by[strongest_name]["task_balanced_success_rate"])
    ablation_rate = float(by["fang_no_failure_ablation"]["task_balanced_success_rate"])
    prior_rate = float(by["afil_local_proxy"]["task_balanced_success_rate"])
    if float(by["fang_full"].get("mean_gate_activation_rate", 0.0) or 0.0) <= 0.0:
        return "STAGE_B_KILL_NO_MECHANISM_ACTIVATION"
    if full_rate > max(strongest_rate, ablation_rate, prior_rate) and full_rate - strongest_rate >= 0.10:
        return "STAGE_B_PROTOTYPE_GO"
    for explainer in ("afil_local_proxy", "fang_no_failure_ablation", "nearest_success_replay"):
        if float(by[explainer]["task_balanced_success_rate"]) >= full_rate:
            return "STAGE_B_KILL_BASELINE_OR_ABLATION_EXPLAINS_RESULT"
    if full_rate < float(by["base_smolvla"]["task_balanced_success_rate"]):
        return "STAGE_B_KILL_WORSE_THAN_BASE"
    pair = paired.get(strongest_name) or {}
    ci = pair.get("paired_bootstrap_ci") or [0.0, 0.0]
    if full_rate <= strongest_rate and float(ci[1]) <= 0.10:
        return "STAGE_B_KILL_USEFUL_IMPROVEMENT_EXCLUDED"
    return "STAGE_B_UNRESOLVED_EXPANSION_OPTIONAL"


def _stage_b_expanded_decision(summary: Mapping[str, Any], paired: Mapping[str, Any]) -> str:
    decision = _stage_b_decision(summary, paired)
    if decision == "STAGE_B_PROTOTYPE_GO":
        return "STAGE_B_EXPANDED_PROTOTYPE_GO"
    if decision == "STAGE_B_UNRESOLVED_EXPANSION_OPTIONAL":
        return "STAGE_B_EXPANDED_NON_GO_NO_THIRD_EXPANSION"
    return decision.replace("STAGE_B_", "STAGE_B_EXPANDED_")


def _load_runtime(args: argparse.Namespace) -> dict[str, Any]:
    selected = json.loads(Path(args.selected_config).read_text(encoding="utf-8-sig"))
    records = _read_jsonl(Path(args.development_records))
    return load_fang_runtime(checkpoint_path=str(args.checkpoint), records=records, selected_config=selected)


def _stage_mode(args: argparse.Namespace, *, stage: str) -> dict[str, Any]:
    started = time.time()
    loaded = _load_policy(args)
    runtime = _load_runtime(args)
    if stage == "stage-a":
        identities = STAGE_A_IDENTITIES[: int(args.stage_a_identities)]
        partial_path = Path(args.stage_a_partial_output)
        output_path = Path(args.stage_a_output)
        md_path = Path(args.stage_a_md)
    elif stage == "stage-b-expansion":
        identities = STAGE_B_IDENTITIES + STAGE_B_EXPANSION_IDENTITIES
        partial_path = Path(args.stage_b_expansion_partial_output)
        output_path = Path(args.stage_b_expansion_output)
        md_path = Path(args.stage_b_expansion_md)
    else:
        identities = STAGE_B_IDENTITIES[: int(args.stage_b_identities)]
        partial_path = Path(args.stage_b_partial_output)
        output_path = Path(args.stage_b_output)
        md_path = Path(args.stage_b_md)
    rows = _planned_rows(TASKS[: int(args.max_tasks)], identities, VARIANTS)
    episodes: list[dict[str, Any]] = []
    if stage == "stage-b-expansion" and Path(args.stage_b_output).exists():
        episodes = list(json.loads(Path(args.stage_b_output).read_text(encoding="utf-8-sig")).get("episodes") or [])
    if partial_path.exists() and not bool(args.rerun_stage):
        episodes = list(json.loads(partial_path.read_text(encoding="utf-8-sig")).get("episodes") or [])
    completed = {(row.get("variant"), row.get("task_key"), int(row.get("identity", -1))) for row in episodes}
    for row in rows:
        key = (row["variant"], row["task_key"], int(row["identity"]))
        if key in completed:
            continue
        result = _run_episode(row=row, loaded=loaded, runtime=runtime, max_eval_steps=int(args.max_eval_steps))
        episodes.append(result)
        _write_json(partial_path, {"episodes": episodes, "planned_episode_count": len(rows)})
    summary = _summarize(episodes)
    paired = _paired_vs_full(episodes)
    if stage == "stage-a":
        final = _stage_a_decision(summary)
    elif stage == "stage-b-expansion":
        final = _stage_b_expanded_decision(summary, paired)
    else:
        final = _stage_b_decision(summary, paired)
    report = {
        "mode": stage,
        "branch": BRANCH,
        "date_kst": DATE_KST,
        "proposal_hash": PROPOSAL_HASH,
        "closed_loop_experiment_happened": True,
        "training_happened": False,
        "reset_identity_base": FANG_RESET_IDENTITY_BASE,
        "identities": identities,
        "checkpoint_path": str(args.checkpoint),
        "checkpoint_sha256": _sha256_file(Path(args.checkpoint)),
        "selected_config_path": str(args.selected_config),
        "selected_config_sha256": _sha256_file(Path(args.selected_config)),
        "planned_episode_count": len(rows),
        "completed_episode_count": len(episodes),
        "episodes": episodes,
        "summary": summary,
        "paired_vs_fang_full": paired,
        "final_decision": final,
        "next_step": "Run Stage B." if stage == "stage-a" and final.endswith("STAGE_B_REQUIRED") else "Archive, expand, or scale according to governance.",
        "elapsed_seconds": _round(time.time() - started, 3),
    }
    _write_json(output_path, report)
    _write_md(md_path, f"FANG-VLA {stage.upper()} Result", report)
    return report


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "stage-a":
        return _stage_mode(args, stage="stage-a")
    if args.mode == "stage-b":
        return _stage_mode(args, stage="stage-b")
    if args.mode == "stage-b-expansion":
        return _stage_mode(args, stage="stage-b-expansion")
    raise ValueError(args.mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["stage-a", "stage-b", "stage-b-expansion"], required=True)
    parser.add_argument("--base-path", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--development-records", default="reports/cavm_vla/acquisition_records.jsonl")
    parser.add_argument("--selected-config", default="reports/fang_vla/selected_config.json")
    parser.add_argument("--checkpoint", default="reports/fang_vla/checkpoints/fang_c01.pt")
    parser.add_argument("--stage-a-output", default="reports/fang_vla/stage_a_result.json")
    parser.add_argument("--stage-a-md", default="reports/fang_vla/stage_a_result.md")
    parser.add_argument("--stage-a-partial-output", default="reports/fang_vla/stage_a_partial_result.json")
    parser.add_argument("--stage-b-output", default="reports/fang_vla/stage_b_result.json")
    parser.add_argument("--stage-b-md", default="reports/fang_vla/stage_b_result.md")
    parser.add_argument("--stage-b-partial-output", default="reports/fang_vla/stage_b_partial_result.json")
    parser.add_argument("--stage-b-expansion-output", default="reports/fang_vla/stage_b_expansion_result.json")
    parser.add_argument("--stage-b-expansion-md", default="reports/fang_vla/stage_b_expansion_result.md")
    parser.add_argument("--stage-b-expansion-partial-output", default="reports/fang_vla/stage_b_expansion_partial_result.json")
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--stage-a-identities", type=int, default=5)
    parser.add_argument("--stage-b-identities", type=int, default=20)
    parser.add_argument("--max-eval-steps", type=int, default=0)
    parser.add_argument("--rerun-stage", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if int(args.max_tasks) < 1 or int(args.max_tasks) > len(TASKS):
        raise SystemExit("--max-tasks must be between 1 and 2")
    if int(args.stage_a_identities) < 1 or int(args.stage_a_identities) > len(STAGE_A_IDENTITIES):
        raise SystemExit(f"--stage-a-identities must be between 1 and {len(STAGE_A_IDENTITIES)}")
    if int(args.stage_b_identities) < 1 or int(args.stage_b_identities) > len(STAGE_B_IDENTITIES):
        raise SystemExit(f"--stage-b-identities must be between 1 and {len(STAGE_B_IDENTITIES)}")
    if set(TASK_KEYS) != {f"{task['suite']}/task_{task['task_id']}" for task in TASKS}:
        raise SystemExit("FANG task constants disagree with runner task manifest")
    report = run(args)
    print(json.dumps({"mode": report["mode"], "final_decision": report["final_decision"], "completed": report["completed_episode_count"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
