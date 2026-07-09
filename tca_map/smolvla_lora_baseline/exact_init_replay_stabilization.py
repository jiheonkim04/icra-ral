"""Exact-init expert replay stabilization and eligible-set construction.

This runner is an evaluation-protocol gate for the fixed SmolVLA/LIBERO 7D
baseline. It first sweeps expert-only exact-init replay candidates, constructs
a fixed eligibility set, and only then optionally replays existing/simple
baselines on expert-success cases.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.datasets.libero_full_demo_expert_replay_sanity import _run_replay_variant
from tca_map.smolvla_lora_baseline import diagnostic as base
from tca_map.smolvla_lora_baseline import replay_bridge
from tca_map.smolvla_lora_baseline import standard_replay_baseline as standard


RUN_GATE = "ALLOW_EXACT_INIT_EXPERT_REPLAY_STABILIZATION"
LEARNED_GATE = "ALLOW_EXACT_INIT_EXPERT_REPLAY_STABILIZATION_LEARNED"
SCHEMA_VERSION = "exact-init-expert-replay-stabilization-v1"
FINAL_DECISIONS = {
    "READY_FOR_METHOD_AFTER_STABLE_REPLAY_BASELINE",
    "READY_BUT_NEEDS_ACTION_VALIDITY_FIX",
    "OFFLINE_TO_CONTROL_GAP",
    "EXPERT_REPLAY_PROTOCOL_BLOCKED",
    "EXPERT_REPLAY_FIX_NEEDED",
    "TOO_HEAVY_LOCAL",
}
FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_ROLLOUT",
    "ALLOW_ROLLOUTS",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_PATCHGUARD_VLA_STATE1B",
    "ALLOW_PATCHGUARD_TINY_LORA_TRAINING",
    "ALLOW_TARGET_GROUNDED_ACTIONMAP",
    "ALLOW_SAFELORA",
    "ALLOW_PRISM",
    "ALLOW_TG7D_ADAPTER",
]
PREFERRED_TASK_STEMS = [
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo",
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo",
]
SEEDED_CANDIDATES = {
    "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo": [
        "demo_7",
        "demo_8",
        "demo_30",
        "demo_31",
    ],
    "KITCHEN_SCENE4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it_demo": [
        "demo_5",
        "demo_6",
        "demo_7",
        "demo_8",
    ],
}


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _round(value: float | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)


def _compact_error(exc: BaseException | str) -> dict[str, Any]:
    if isinstance(exc, str):
        return {"type": "Error", "message": exc, "traceback_tail": []}
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _demo_sort_key(name: str) -> tuple[str, int | str]:
    return base._demo_sort_key(name)


def _task_id_from_demo_path(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_demo")] if stem.endswith("_demo") else stem


def _instruction_from_path(path: Path) -> str:
    return _task_id_from_demo_path(path).replace("_", " ")


def _current_branch() -> str | None:
    try:
        import subprocess

        result = subprocess.run(
            ["git", "branch", "--show-current"],
            cwd=Path(__file__).resolve().parents[2],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            check=False,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def _select_task_paths(data_root: Path, max_tasks: int) -> list[Path]:
    selected = [data_root / f"{stem}.hdf5" for stem in PREFERRED_TASK_STEMS if (data_root / f"{stem}.hdf5").exists()]
    if len(selected) < int(max_tasks):
        existing = set(selected)
        for path in sorted(data_root.glob("*.hdf5")):
            if path not in existing:
                selected.append(path)
            if len(selected) >= int(max_tasks):
                break
    return selected[: int(max_tasks)]


def _select_candidate_demos(
    *,
    task_name: str,
    metadata: dict[str, dict[str, Any]],
    train_demos: list[str],
    candidate_demos_per_task: int,
    max_replay_steps: int,
) -> list[str]:
    train_set = set(train_demos)
    selected: list[str] = []
    for demo_name in SEEDED_CANDIDATES.get(task_name, []):
        if demo_name in metadata and demo_name not in train_set and demo_name not in selected:
            selected.append(demo_name)
        if len(selected) >= int(candidate_demos_per_task):
            return selected
    bounded = [
        name
        for name in sorted(metadata, key=_demo_sort_key)
        if name not in train_set
        and name not in selected
        and metadata[name].get("first_signal_index") is not None
        and int(metadata[name]["first_signal_index"]) < int(max_replay_steps)
    ]
    for demo_name in bounded:
        selected.append(demo_name)
        if len(selected) >= int(candidate_demos_per_task):
            return selected
    for demo_name in sorted(metadata, key=_demo_sort_key):
        if demo_name not in train_set and demo_name not in selected:
            selected.append(demo_name)
        if len(selected) >= int(candidate_demos_per_task):
            return selected
    return selected


def _build_candidate_pool(args: argparse.Namespace) -> dict[str, Any]:
    task_paths = _select_task_paths(Path(args.data_root), int(args.max_tasks))
    tasks: list[dict[str, Any]] = []
    candidates: list[dict[str, Any]] = []
    for path in task_paths:
        metadata = standard._load_demo_metadata(path)
        demo_names = sorted(metadata, key=_demo_sort_key)
        train_demos = demo_names[: int(args.train_demos_per_task)]
        selected = _select_candidate_demos(
            task_name=path.stem,
            metadata=metadata,
            train_demos=train_demos,
            candidate_demos_per_task=int(args.candidate_demos_per_task),
            max_replay_steps=int(args.max_replay_steps),
        )
        for demo_name in selected:
            candidates.append(
                {
                    "task_name": path.stem,
                    "hdf5_path": str(path),
                    "demo_name": demo_name,
                    "metadata": metadata[demo_name],
                    "candidate_source": "seeded_prior_case_or_bounded_nontrain_demo",
                }
            )
        tasks.append(
            {
                "task_name": path.stem,
                "path": str(path),
                "demo_count": len(demo_names),
                "train_demos_excluded_from_candidate_pool": train_demos,
                "candidate_demos": selected,
                "candidate_first_signal": {name: metadata[name].get("first_signal_index") for name in selected},
            }
        )
    return {
        "task_count": len(task_paths),
        "tasks": tasks,
        "candidates": candidates,
        "candidate_count": len(candidates),
        "selection_rule": (
            "Use prior failed/success replay cases first, add non-train demos with bounded HDF5 "
            "reward/done signals, and keep the sweep between 5 and 10 candidates when possible."
        ),
    }


def _first_observed_signal(result: dict[str, Any]) -> int | None:
    values = [
        result.get("first_positive_reward_index"),
        result.get("first_done_index"),
        result.get("first_success_index"),
    ]
    observed = [int(value) for value in values if value is not None]
    return min(observed) if observed else None


def _eligibility(result: dict[str, Any], action_validity: dict[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    reward_or_success = bool(float(result.get("reward_sum") or 0.0) >= 1.0 or result.get("final_success") is True)
    if not replay_bridge._success(result):
        reasons.append("expert replay did not reach reward, done, or final success")
    if not reward_or_success:
        reasons.append("reward did not reach 1.0 and final success was not true")
    if result.get("first_done_index") is None:
        reasons.append("finite done index missing")
    if not bool(result.get("passed")):
        reasons.append("env/reset/exact-init/step execution did not pass")
    if result.get("error"):
        reasons.append("env replay error present")
    if not bool(action_validity.get("shape_exactly_7d")):
        reasons.append("expert action shape is not exactly 7D")
    if not bool(action_validity.get("finite")):
        reasons.append("expert action sequence is not finite")
    return not reasons, reasons


def _diagnose_failure(case: dict[str, Any]) -> dict[str, Any]:
    result = case.get("expert_result") or {}
    metadata = case.get("hdf5_metadata") or {}
    validity = case.get("expert_action_validity") or {}
    hdf5_signal = metadata.get("first_signal_index")
    observed_signal = _first_observed_signal(result)
    state_l2 = result.get("after_set_state_l2_to_hdf5_init")
    checks = {
        "exact_init_state_loading_mismatch": bool(
            not result.get("set_init_state_used")
            or not result.get("set_init_state_ok")
            or (state_l2 is not None and float(state_l2) > 1e-6)
        ),
        "wrong_initial_state_or_reset_problem": bool(not result.get("reset_ok") or not result.get("set_init_state_ok")),
        "hdf5_action_dimension_or_range_issue": bool(
            not validity.get("shape_exactly_7d")
            or not validity.get("finite")
            or float(validity.get("clip_rate_element") or 0.0) > 0.0
        ),
        "env_task_mismatch": bool(case.get("bddl_file_exists") is False or "bddl" in str((result.get("error") or {}).get("message", "")).lower()),
        "off_by_one_action_alignment_suspected": bool(
            hdf5_signal is not None and observed_signal is not None and abs(int(observed_signal) - int(hdf5_signal)) > 1
        ),
        "gripper_convention_mismatch_suspected": bool(
            validity.get("dominant_clip_dim") == 6 or float(validity.get("gripper_clip_rate") or 0.0) > 0.0
        ),
        "max_step_too_short": bool(
            not replay_bridge._success(result)
            and int(result.get("steps_performed") or 0) >= int(case.get("target_horizon") or 0)
            and hdf5_signal is not None
            and int(case.get("target_horizon") or 0) <= int(hdf5_signal)
        ),
        "object_already_displaced_or_unstable_init_suspected": bool(
            result.get("object_movement", {}).get("available") is False
            or result.get("target_key_audit", {}).get("best_key") is None
        ),
        "controller_mismatch_suspected": bool(
            not replay_bridge._success(result)
            and result.get("delta_vs_absolute_action_convention_evidence") != "raw_hdf5_actions_reached_reward_done_or_success"
        ),
        "missing_object_body_or_site_names": bool((result.get("object_movement") or {}).get("object_position_keys_missing")),
        "known_libero_robosuite_nondeterminism_suspected": bool(
            result.get("set_init_state_ok")
            and hdf5_signal is not None
            and observed_signal is None
            and not result.get("error")
            and bool(validity.get("shape_exactly_7d"))
            and bool(validity.get("finite"))
        ),
    }
    primary = "expert_replay_succeeded"
    if not replay_bridge._success(result):
        ordered = [
            ("exact_init_state_loading_mismatch", "exact_init_state_loading_mismatch"),
            ("wrong_initial_state_or_reset_problem", "wrong_initial_state_or_reset_problem"),
            ("env_task_mismatch", "env_task_mismatch"),
            ("hdf5_action_dimension_or_range_issue", "hdf5_action_dimension_or_range_issue"),
            ("max_step_too_short", "max_step_too_short"),
            ("off_by_one_action_alignment_suspected", "off_by_one_action_alignment_suspected"),
            ("controller_mismatch_suspected", "controller_or_action_convention_mismatch_suspected"),
            ("known_libero_robosuite_nondeterminism_suspected", "exact_init_or_controller_nondeterminism_suspected"),
        ]
        primary = next((label for key, label in ordered if checks.get(key)), "expert_replay_failed_without_specific_fix")
    elif result.get("first_done_index") is None:
        primary = "reward_or_success_without_finite_done_index"
    return {
        "primary_failure_reason": primary,
        "checks": checks,
        "hdf5_first_signal_index": hdf5_signal,
        "observed_first_signal_index": observed_signal,
        "after_set_state_l2_to_hdf5_init": state_l2,
        "controller": result.get("controller"),
        "target_key_audit": result.get("target_key_audit"),
    }


def _run_expert_sweep(args: argparse.Namespace, candidate_pool: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    if not _env_flag(RUN_GATE):
        return {
            "executed": False,
            "reason": f"{RUN_GATE}=1 is required.",
            "cases": [],
            "eligible_cases": [],
            "failed_cases": [],
            "runtime_sec": _round(time.monotonic() - started, 3),
        }
    env_cls, env_meta = replay_bridge._load_env_class_noninteractive(
        libero_root=Path(args.libero_root),
        robosuite_root=Path(args.robosuite_root),
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
    )
    cases: list[dict[str, Any]] = []
    for candidate in candidate_pool["candidates"]:
        path = Path(candidate["hdf5_path"])
        demo_name = candidate["demo_name"]
        demo_window = replay_bridge._demo_window(
            path,
            demo_name,
            int(args.max_replay_steps),
            int(args.post_signal_margin),
        )
        bddl_file = (
            Path(args.libero_root)
            / "libero"
            / "libero"
            / "bddl_files"
            / path.parent.name
            / f"{_task_id_from_demo_path(path)}.bddl"
        )
        expert_actions = np.asarray(demo_window["actions"], dtype=np.float32)
        action_validity = standard._action_validity(expert_actions)
        variant = {
            "name": "expert",
            "claim_role": "expert_replay_upper_bound",
            "actions": expert_actions,
            "use_exact_init_state": True,
        }
        result = _run_replay_variant(
            env_cls=env_cls,
            bddl_file=bddl_file,
            camera_size=int(args.camera_size),
            init_state=np.asarray(demo_window["init_state"], dtype=np.float64),
            variant=variant,
            instruction=_instruction_from_path(path),
        )
        eligible, eligibility_reasons = _eligibility(result, action_validity)
        case = {
            "task_name": candidate["task_name"],
            "hdf5_path": str(path),
            "demo_name": demo_name,
            "candidate_source": candidate.get("candidate_source"),
            "bddl_file": str(bddl_file),
            "bddl_file_exists": bddl_file.exists(),
            "instruction": _instruction_from_path(path),
            "max_steps": int(args.max_replay_steps),
            "post_signal_margin": int(args.post_signal_margin),
            "target_horizon": int(demo_window["target_horizon"]),
            "full_action_steps": int(demo_window["full_action_steps"]),
            "hdf5_metadata": {
                "length": candidate["metadata"].get("length"),
                "first_reward_index": candidate["metadata"].get("first_reward_index"),
                "first_done_index": candidate["metadata"].get("first_done_index"),
                "first_signal_index": candidate["metadata"].get("first_signal_index"),
            },
            "expert_action_validity": action_validity,
            "expert_result": result,
            "eligible_for_learned_policy_comparison": eligible,
            "eligibility_reasons": ["expert replay satisfies eligibility criterion"] if eligible else eligibility_reasons,
        }
        case["failure_diagnosis"] = _diagnose_failure(case)
        cases.append(case)
    eligible_cases = [case for case in cases if case["eligible_for_learned_policy_comparison"]]
    failed_cases = [case for case in cases if not case["eligible_for_learned_policy_comparison"]]
    return {
        "executed": True,
        "reason": "bounded expert-only exact-init replay sweep attempted",
        "env": env_meta,
        "cases": cases,
        "eligible_cases": [
            {
                "task_name": case["task_name"],
                "demo_name": case["demo_name"],
                "hdf5_path": case["hdf5_path"],
                "first_done_index": case["expert_result"].get("first_done_index"),
                "reward_sum": case["expert_result"].get("reward_sum"),
                "success": case["expert_result"].get("final_success"),
                "target_horizon": case["target_horizon"],
                "max_steps": case["max_steps"],
                "eligibility_reason": case["eligibility_reasons"],
            }
            for case in eligible_cases
        ],
        "failed_cases": [
            {
                "task_name": case["task_name"],
                "demo_name": case["demo_name"],
                "hdf5_path": case["hdf5_path"],
                "failure_reason": case["failure_diagnosis"].get("primary_failure_reason"),
                "eligibility_reasons": case["eligibility_reasons"],
                "hdf5_first_signal_index": case["hdf5_metadata"].get("first_signal_index"),
                "steps_performed": case["expert_result"].get("steps_performed"),
                "reward_sum": case["expert_result"].get("reward_sum"),
                "first_done_index": case["expert_result"].get("first_done_index"),
            }
            for case in failed_cases
        ],
        "aggregate": _aggregate_expert(cases),
        "runtime_sec": _round(time.monotonic() - started, 3),
    }


def _aggregate_expert(cases: list[dict[str, Any]]) -> dict[str, Any]:
    results = [case.get("expert_result") or {} for case in cases]
    progress_values = [replay_bridge._progress_metric(result) for result in results]
    progress_values = [float(value) for value in progress_values if value is not None]
    return {
        "case_count": len(cases),
        "eligible_case_count": int(sum(1 for case in cases if case.get("eligible_for_learned_policy_comparison"))),
        "failed_case_count": int(sum(1 for case in cases if not case.get("eligible_for_learned_policy_comparison"))),
        "success_count": int(sum(1 for result in results if replay_bridge._success(result))),
        "success_rate": _round(float(np.mean([replay_bridge._success(result) for result in results]))) if results else None,
        "reward_sum_mean": _round(float(np.mean([float(result.get("reward_sum") or 0.0) for result in results]))) if results else None,
        "first_done_indices": [result.get("first_done_index") for result in results],
        "progress_proxy_mean": _round(float(np.mean(progress_values))) if progress_values else None,
        "runtime_case_steps": [result.get("steps_performed") for result in results],
        "eligible_case_ids": [
            f"{case.get('task_name')}::{case.get('demo_name')}"
            for case in cases
            if case.get("eligible_for_learned_policy_comparison")
        ],
    }


def _prior_best_adapter_name(path: Path) -> str:
    result = _read_json_if_exists(path)
    summary = result.get("summary") or {}
    return str(summary.get("best_lora_name") or "smolvla_state_proj_lora_rank4_7d_adapter")


def _build_learned_predictors(args: argparse.Namespace, best_lora_name: str) -> tuple[dict[str, Any], dict[str, Any]]:
    split = standard._build_standard_split(
        data_root=Path(args.data_root),
        max_tasks=int(args.max_tasks),
        train_demos_per_task=int(args.train_demos_per_task),
        eval_demos_per_task=int(args.eval_demos_per_task),
        records_per_demo=int(args.records_per_demo),
        replay_demos_per_task=1,
        max_replay_steps=int(args.max_replay_steps),
    )
    train_records = split["train_records"]
    mean_action = base._mean_train_action(train_records)[:7].astype(np.float32)
    ridge = standard._fit_ridge(train_records)
    adapter_path = Path(args.adapter_dir) / f"{best_lora_name}.pt"
    adapter = replay_bridge.ExecutableAdapter.load(adapter_path)
    predictors = {
        "mean_action": {
            "kind": "constant_mean_action",
            "predict": lambda features: np.repeat(mean_action.reshape(1, 7), features.shape[0], axis=0).astype(np.float32),
        },
        "ridge": {
            "kind": "closed_form_ridge_refit_on_predeclared_standard_train_split",
            "predict": ridge.predict,
        },
        "smolvla_7d_adapter": {
            "kind": "persisted_executable_smolvla_7d_adapter",
            "artifact_path": str(adapter_path),
            "predict": adapter.predict_features,
        },
    }
    meta = {
        "best_lora_name": best_lora_name,
        "adapter_artifact_path": str(adapter_path),
        "adapter_artifact_exists": adapter_path.exists(),
        "standard_train_split": split["report"],
        "small_mlp": {
            "executed": False,
            "skip_reason": "No persisted executable MLP artifact exists; runner does not retrain MLP during this protocol gate.",
        },
        "new_smolvla_training_performed": False,
        "new_mlp_training_performed": False,
        "analytical_ridge_refit_performed": True,
    }
    return predictors, meta


def _run_learned_replay(
    args: argparse.Namespace,
    expert_sweep: dict[str, Any],
    best_lora_name: str,
) -> dict[str, Any]:
    started = time.monotonic()
    if not _env_flag(LEARNED_GATE):
        return {
            "executed": False,
            "reason": f"{LEARNED_GATE}=1 is required for learned-policy replay after eligibility is green.",
            "cases": [],
            "aggregate": {},
            "runtime_sec": _round(time.monotonic() - started, 3),
        }
    eligible_cases = [case for case in expert_sweep.get("cases", []) if case.get("eligible_for_learned_policy_comparison")]
    if len(eligible_cases) < 2:
        return {
            "executed": False,
            "reason": "Fewer than two eligible expert-success cases; hard rule blocks learned-policy replay.",
            "cases": [],
            "aggregate": {},
            "runtime_sec": _round(time.monotonic() - started, 3),
        }
    predictors, meta = _build_learned_predictors(args, best_lora_name)
    env_cls, env_meta = replay_bridge._load_env_class_noninteractive(
        libero_root=Path(args.libero_root),
        robosuite_root=Path(args.robosuite_root),
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
    )
    cases: list[dict[str, Any]] = []
    for expert_case in eligible_cases:
        path = Path(expert_case["hdf5_path"])
        demo_name = expert_case["demo_name"]
        demo_window = replay_bridge._demo_window(path, demo_name, int(args.max_replay_steps), int(args.post_signal_margin))
        bddl_file = Path(expert_case["bddl_file"])
        expert_actions = np.asarray(demo_window["actions"], dtype=np.float32)
        features = np.asarray(demo_window["features"], dtype=np.float32)
        actions_by_policy = {
            "mean_action": predictors["mean_action"]["predict"](features),
            "ridge": predictors["ridge"]["predict"](features),
            "smolvla_7d_adapter": predictors["smolvla_7d_adapter"]["predict"](features),
        }
        offline_case = {
            name: {
                "action_metrics": standard._metrics(actions, expert_actions),
                "action_validity": standard._action_validity(actions),
            }
            for name, actions in actions_by_policy.items()
        }
        results = {"expert": expert_case["expert_result"]}
        for name, actions in actions_by_policy.items():
            variant = {
                "name": name,
                "claim_role": {
                    "mean_action": "mean_action_baseline",
                    "ridge": "ridge_baseline",
                    "smolvla_7d_adapter": "best_prior_smolvla_7d_lora_adapter",
                }[name],
                "actions": actions,
                "use_exact_init_state": True,
            }
            results[name] = _run_replay_variant(
                env_cls=env_cls,
                bddl_file=bddl_file,
                camera_size=int(args.camera_size),
                init_state=np.asarray(demo_window["init_state"], dtype=np.float64),
                variant=variant,
                instruction=_instruction_from_path(path),
            )
        cases.append(
            {
                "task_name": expert_case["task_name"],
                "hdf5_path": str(path),
                "demo_name": demo_name,
                "eligible_case": True,
                "target_horizon": int(demo_window["target_horizon"]),
                "hdf5_first_signal_index": demo_window.get("first_signal_index"),
                "offline_case_metrics": offline_case,
                "results": results,
            }
        )
    return {
        "executed": True,
        "reason": "bounded learned/simple replay run on fixed expert-success eligibility set only",
        "env": env_meta,
        "predictor_meta": meta,
        "cases": cases,
        "aggregate": _aggregate_learned(cases),
        "runtime_sec": _round(time.monotonic() - started, 3),
    }


def _aggregate_learned(cases: list[dict[str, Any]]) -> dict[str, Any]:
    policy_names = ["expert", "mean_action", "ridge", "small_mlp", "smolvla_7d_adapter"]
    aggregate: dict[str, Any] = {}
    for policy in policy_names:
        values = []
        for case in cases:
            result = (case.get("results") or {}).get(policy)
            if result:
                values.append(result)
        progress_values = [replay_bridge._progress_metric(item) for item in values]
        progress_values = [float(value) for value in progress_values if value is not None]
        aggregate[policy] = {
            "case_count": len(values),
            "success_count": int(sum(1 for item in values if replay_bridge._success(item))),
            "success_rate": _round(float(np.mean([replay_bridge._success(item) for item in values]))) if values else None,
            "reward_sum_mean": _round(float(np.mean([float(item.get("reward_sum") or 0.0) for item in values]))) if values else None,
            "first_done_indices": [item.get("first_done_index") for item in values],
            "progress_proxy_mean": _round(float(np.mean(progress_values))) if progress_values else None,
            "object_movement_mean": _round(
                float(
                    np.mean(
                        [
                            float((item.get("object_movement") or {}).get("target_object_displacement_l2") or 0.0)
                            for item in values
                        ]
                    )
                )
            )
            if values
            else None,
            "runtime_case_steps": [item.get("steps_performed") for item in values],
        }
    aggregate["learned_aggregate_uses_only_eligible_cases"] = True
    aggregate["eligible_case_count"] = len(cases)
    return aggregate


def _action_validity_audit(learned_replay: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for case in learned_replay.get("cases") or []:
        for policy in ["mean_action", "ridge", "smolvla_7d_adapter"]:
            validity = ((case.get("offline_case_metrics") or {}).get(policy) or {}).get("action_validity") or {}
            rows.append(
                {
                    "task_name": case.get("task_name"),
                    "demo_name": case.get("demo_name"),
                    "policy": policy,
                    "clip_rate_element": validity.get("clip_rate_element"),
                    "clip_rate_step": validity.get("clip_rate_step"),
                    "controller_valid_rate_proxy": validity.get("controller_valid_rate_proxy"),
                    "dominant_clip_dim": validity.get("dominant_clip_dim"),
                    "gripper_clip_rate": validity.get("gripper_clip_rate"),
                }
            )
    adapter_rows = [row for row in rows if row["policy"] == "smolvla_7d_adapter"]
    return {
        "case_rows": rows,
        "adapter_clip_rate_step_mean": _round(float(np.mean([row["clip_rate_step"] for row in adapter_rows]))) if adapter_rows else None,
        "adapter_controller_valid_rate_proxy_mean": _round(float(np.mean([row["controller_valid_rate_proxy"] for row in adapter_rows]))) if adapter_rows else None,
        "adapter_action_validity_fix_needed": bool(
            adapter_rows
            and (
                float(np.mean([row["clip_rate_step"] for row in adapter_rows])) > 0.3
                or float(np.mean([row["controller_valid_rate_proxy"] for row in adapter_rows])) < 0.7
            )
        ),
        "mlp_replay_executed": False,
        "mlp_skip_reason": "No persisted executable MLP artifact exists; no MLP retraining was performed.",
    }


def _adapter_beats_simple_replay_baselines(learned_replay: dict[str, Any]) -> bool:
    aggregate = learned_replay.get("aggregate") or {}
    adapter = (aggregate.get("smolvla_7d_adapter") or {}).get("progress_proxy_mean")
    if adapter is None:
        return False
    simple_values = [
        (aggregate.get(name) or {}).get("progress_proxy_mean")
        for name in ["mean_action", "ridge", "small_mlp"]
    ]
    return all(value is None or float(adapter) > float(value) for value in simple_values)


def _simple_matches_or_beats_adapter(learned_replay: dict[str, Any]) -> bool:
    aggregate = learned_replay.get("aggregate") or {}
    adapter = (aggregate.get("smolvla_7d_adapter") or {}).get("progress_proxy_mean")
    if adapter is None:
        return True
    for name in ["mean_action", "ridge", "small_mlp"]:
        value = (aggregate.get(name) or {}).get("progress_proxy_mean")
        if value is not None and float(value) >= float(adapter):
            return True
    return False


def _decide(report: dict[str, Any]) -> tuple[str, str]:
    expert_sweep = report.get("state1_expert_replay_sweep") or {}
    eligible_count = int((expert_sweep.get("aggregate") or {}).get("eligible_case_count") or 0)
    failed_cases = expert_sweep.get("failed_cases") or []
    learned = report.get("state4_learned_replay") or {}
    audit = report.get("state4_action_validity") or {}
    if not expert_sweep.get("executed"):
        return "TOO_HEAVY_LOCAL", expert_sweep.get("reason") or "Expert replay sweep did not execute."
    if eligible_count < 2:
        return "EXPERT_REPLAY_PROTOCOL_BLOCKED", "Do not evaluate learned policies; fewer than two stable expert-success cases exist."
    concrete_fix_reasons = {"exact_init_state_loading_mismatch", "wrong_initial_state_or_reset_problem", "env_task_mismatch"}
    if eligible_count < 3 and any((case.get("failure_reason") in concrete_fix_reasons) for case in failed_cases):
        return "EXPERT_REPLAY_FIX_NEEDED", "Fix the concrete exact-init/env mismatch before learned replay evaluation."
    if not learned.get("executed"):
        return "EXPERT_REPLAY_FIX_NEEDED", learned.get("reason") or "Run learned replay only after the eligible set is fixed."
    if _simple_matches_or_beats_adapter(learned):
        return (
            "OFFLINE_TO_CONTROL_GAP",
            "Stop method work; diagnose the offline-to-control gap in the fixed SmolVLA 7D baseline on the eligible set before proposing any new method.",
        )
    if _adapter_beats_simple_replay_baselines(learned) and bool(audit.get("adapter_action_validity_fix_needed")):
        return "READY_BUT_NEEDS_ACTION_VALIDITY_FIX", "Expert replay is stable and adapter improves progress, but action validity remains too weak."
    if _adapter_beats_simple_replay_baselines(learned):
        return "READY_FOR_METHOD_AFTER_STABLE_REPLAY_BASELINE", "Only now consider method work after preserving this stable replay baseline."
    return "OFFLINE_TO_CONTROL_GAP", "Eligible replay did not show reliable SmolVLA 7D adapter control improvement."


def _strip_large(payload: Any) -> Any:
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "reward_trajectory":
                values = value if isinstance(value, list) else []
                result["reward_trajectory_summary"] = {
                    "length": len(values),
                    "first_10": values[:10],
                    "last_10": values[-10:],
                    "nonzero_indices": [index for index, item in enumerate(values) if float(item) > 0.0][:20],
                }
            else:
                result[key] = _strip_large(value)
        return result
    if isinstance(payload, list):
        return [_strip_large(item) for item in payload]
    if isinstance(payload, np.ndarray):
        return payload.tolist()
    if isinstance(payload, np.generic):
        return payload.item()
    return payload


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    best_lora_name = _prior_best_adapter_name(Path(args.prior_result_path))
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "TOO_HEAVY_LOCAL",
        "policy": {
            "new_method_created": False,
            "paper_claims_made": False,
            "downloads_performed": False,
            "openvla_oft_executed": False,
            "full_benchmark_executed": False,
            "tg_patchguard_safelora_executed": False,
            "run_gate_set": _env_flag(RUN_GATE),
            "learned_gate_set": _env_flag(LEARNED_GATE),
            "forbidden_gates_set": forbidden,
            "training_performed": False,
            "new_smolvla_training_performed": False,
            "mlp_training_performed": False,
            "analytical_ridge_refit_performed": False,
            "replay_control_performed": False,
            "learned_evaluated_on_expert_failed_cases": False,
        },
        "paths": {
            "data_root": str(Path(args.data_root)),
            "libero_root": str(Path(args.libero_root)),
            "robosuite_root": str(Path(args.robosuite_root)),
            "adapter_dir": str(Path(args.adapter_dir)),
            "prior_result_path": str(Path(args.prior_result_path)),
            "output_dir": str(Path(args.output_dir)),
        },
        "state0_candidate_pool": {},
        "state1_expert_replay_sweep": {},
        "state2_failure_diagnosis": {},
        "state3_fixed_eligibility_set": {},
        "state4_learned_replay": {},
        "state4_action_validity": {},
        "summary": {},
        "error": None,
    }

    def finish(decision: str, next_step: str, code: int) -> tuple[dict[str, Any], int]:
        if decision not in FINAL_DECISIONS:
            raise ValueError(f"invalid final decision: {decision}")
        report["decision"] = decision
        report["summary"].update(
            {
                "final_decision": decision,
                "exact_next_step": next_step,
                "runtime_sec": _round(time.monotonic() - started, 3),
            }
        )
        return report, code

    if not report["policy"]["run_gate_set"]:
        return finish("TOO_HEAVY_LOCAL", f"Set {RUN_GATE}=1 for this bounded expert replay stabilization gate.", 2)
    if forbidden:
        report["error"] = {"message": "Forbidden gate(s) set: " + ", ".join(forbidden)}
        return finish("TOO_HEAVY_LOCAL", "Clear forbidden method/download/full-rollout gates and rerun.", 3)
    if not Path(args.data_root).exists():
        return finish("TOO_HEAVY_LOCAL", "Missing local LIBERO data root.", 4)
    if not Path(args.adapter_dir, f"{best_lora_name}.pt").exists():
        return finish("EXPERT_REPLAY_FIX_NEEDED", f"Persisted adapter artifact missing for {best_lora_name}; reproduce the prior bounded baseline only if needed.", 5)

    try:
        candidate_pool = _build_candidate_pool(args)
        report["state0_candidate_pool"] = candidate_pool
        expert_sweep = _run_expert_sweep(args, candidate_pool)
        report["state1_expert_replay_sweep"] = expert_sweep
        report["policy"]["replay_control_performed"] = bool(expert_sweep.get("executed"))
        report["state2_failure_diagnosis"] = {
            "failed_cases": expert_sweep.get("failed_cases") or [],
            "diagnostic_scope": [
                "exact-init state loading mismatch",
                "wrong initial state / reset problem",
                "HDF5 action dimension/range issue",
                "env/task mismatch",
                "off-by-one action alignment",
                "gripper convention mismatch",
                "max step too short",
                "object already displaced / unstable init",
                "controller mismatch",
                "missing object/body/site names",
                "known LIBERO/RoboSuite nondeterminism",
            ],
            "bounded_fixes_attempted": [
                "used exact HDF5 demo init_state",
                "verified action sequence length and 7D shape",
                "used bounded max steps and post-signal margin",
                "matched BDDL task file from the HDF5 task stem",
                "kept learned adapter performance out of expert-failure diagnosis",
            ],
        }
        eligible_cases = expert_sweep.get("eligible_cases") or []
        report["state3_fixed_eligibility_set"] = {
            "eligible_case_count": len(eligible_cases),
            "minimum_required_to_continue": 2,
            "preferred_required_to_continue": 3,
            "limitation": "At least two stable cases are accepted only with this limitation noted; three across two tasks is preferred.",
            "eligible_cases": eligible_cases,
            "excluded_cases": expert_sweep.get("failed_cases") or [],
            "learned_comparison_scope": "eligible cases only",
        }
        if len(eligible_cases) >= 2:
            learned = _run_learned_replay(args, expert_sweep, best_lora_name)
        else:
            learned = {
                "executed": False,
                "reason": "Hard rule: fewer than two expert-success eligible cases.",
                "cases": [],
                "aggregate": {},
                "runtime_sec": 0.0,
            }
        report["state4_learned_replay"] = learned
        report["state4_action_validity"] = _action_validity_audit(learned)
        predictor_meta = learned.get("predictor_meta") or {}
        report["policy"]["analytical_ridge_refit_performed"] = bool(predictor_meta.get("analytical_ridge_refit_performed"))
        decision, next_step = _decide(report)
        aggregate = learned.get("aggregate") or {}
        report["summary"].update(
            {
                "branch": _current_branch(),
                "experiments_happened": True,
                "training_happened": False,
                "new_smolvla_training_happened": False,
                "mlp_training_happened": False,
                "analytical_ridge_refit_happened": report["policy"]["analytical_ridge_refit_performed"],
                "replay_control_happened": bool(expert_sweep.get("executed")),
                "learned_policy_replay_happened": bool(learned.get("executed")),
                "downloads_happened": False,
                "openvla_oft_happened": False,
                "best_lora_name": best_lora_name,
                "candidate_demos_tested": [
                    f"{case.get('task_name')}::{case.get('demo_name')}" for case in expert_sweep.get("cases", [])
                ],
                "candidate_demo_count": len(expert_sweep.get("cases", [])),
                "expert_success_eligible_case_count": len(eligible_cases),
                "expert_failed_case_count": len(expert_sweep.get("failed_cases") or []),
                "fixed_eligibility_set": [
                    f"{case.get('task_name')}::{case.get('demo_name')}" for case in eligible_cases
                ],
                "expert_failed_cases_and_reasons": expert_sweep.get("failed_cases") or [],
                "expert_aggregate": expert_sweep.get("aggregate"),
                "learned_replay_aggregate": aggregate,
                "mean_replay_result": aggregate.get("mean_action"),
                "ridge_replay_result": aggregate.get("ridge"),
                "mlp_replay_result": aggregate.get("small_mlp"),
                "adapter_replay_result": aggregate.get("smolvla_7d_adapter"),
                "action_validity_audit": report["state4_action_validity"],
            }
        )
        return finish(decision, next_step, 0)
    except Exception as exc:  # noqa: BLE001
        report["error"] = _compact_error(exc)
        return finish("TOO_HEAVY_LOCAL", "Fix the reported exact-init replay stabilization runner error and rerun.", 11)


def _write_reports(report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    candidate_pool = report.get("state0_candidate_pool") or {}
    expert = report.get("state1_expert_replay_sweep") or {}
    eligibility = report.get("state3_fixed_eligibility_set") or {}
    learned = report.get("state4_learned_replay") or {}
    aggregate = learned.get("aggregate") or {}
    audit = report.get("state4_action_validity") or {}

    Path("reports/exact_init_expert_replay_stabilization.md").write_text(
        "\n".join(
            [
                "# Exact-Init Expert Replay Stabilization",
                "",
                f"Final decision: `{summary.get('final_decision')}`",
                "",
                "This is an evaluation protocol gate, not a new method, not paper novelty, and not OpenVLA-OFT.",
                "",
                "## Candidate Sweep",
                "",
                f"- candidate demos tested: `{summary.get('candidate_demos_tested')}`",
                f"- candidate count: `{summary.get('candidate_demo_count')}`",
                f"- expert aggregate: `{summary.get('expert_aggregate')}`",
                f"- expert-success eligible cases: `{summary.get('expert_success_eligible_case_count')}`",
                f"- expert-failed cases: `{summary.get('expert_failed_case_count')}`",
                "",
                "## Learned Replay Boundary",
                "",
                f"- learned policy replay happened: `{summary.get('learned_policy_replay_happened')}`",
                f"- learned aggregate: `{aggregate}`",
                f"- MLP replay result: `{aggregate.get('small_mlp')}`",
                f"- MLP skip reason: `{audit.get('mlp_skip_reason')}`",
                "",
                "## Action Validity",
                "",
                f"- adapter clip-rate step mean: `{audit.get('adapter_clip_rate_step_mean')}`",
                f"- adapter controller-valid proxy mean: `{audit.get('adapter_controller_valid_rate_proxy_mean')}`",
                f"- action validity fix needed: `{audit.get('adapter_action_validity_fix_needed')}`",
                "",
                f"Exact next step: {summary.get('exact_next_step')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/exact_init_replay_eligibility_set.md").write_text(
        "\n".join(
            [
                "# Exact-Init Replay Eligibility Set",
                "",
                f"- eligible case count: `{eligibility.get('eligible_case_count')}`",
                f"- preferred required to continue: `{eligibility.get('preferred_required_to_continue')}`",
                f"- minimum required to continue: `{eligibility.get('minimum_required_to_continue')}`",
                f"- limitation: {eligibility.get('limitation')}",
                "",
                "## Eligible Cases",
                "",
                f"`{eligibility.get('eligible_cases')}`",
                "",
                "## Excluded Cases",
                "",
                f"`{eligibility.get('excluded_cases')}`",
                "",
                "Hard rule: learned-policy comparison uses eligible cases only.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/exact_init_failure_diagnosis.md").write_text(
        "\n".join(
            [
                "# Exact-Init Failure Diagnosis",
                "",
                "Diagnosed only expert-failed cases; learned adapter performance was not used for failure diagnosis.",
                "",
                f"- diagnostic scope: `{(report.get('state2_failure_diagnosis') or {}).get('diagnostic_scope')}`",
                f"- bounded fixes attempted: `{(report.get('state2_failure_diagnosis') or {}).get('bounded_fixes_attempted')}`",
                "",
                "## Expert-Failed Cases",
                "",
                f"`{(report.get('state2_failure_diagnosis') or {}).get('failed_cases')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_replay_protocol_decision.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Replay Protocol Decision",
                "",
                f"Final decision: `{summary.get('final_decision')}`",
                "",
                f"- experiments happened: `{summary.get('experiments_happened')}`",
                f"- training happened: `{summary.get('training_happened')}`",
                f"- replay/control happened: `{summary.get('replay_control_happened')}`",
                f"- learned policy replay happened: `{summary.get('learned_policy_replay_happened')}`",
                f"- candidate demos tested: `{summary.get('candidate_demos_tested')}`",
                f"- fixed eligibility set: `{summary.get('fixed_eligibility_set')}`",
                f"- mean replay result: `{summary.get('mean_replay_result')}`",
                f"- ridge replay result: `{summary.get('ridge_replay_result')}`",
                f"- MLP replay result: `{summary.get('mlp_replay_result')}`",
                f"- adapter replay result: `{summary.get('adapter_replay_result')}`",
                "",
                f"Exact next step: {summary.get('exact_next_step')}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    project_lines = [
        "# Project State",
        "",
        "Date: 2026-07-09 KST",
        "",
        f"Branch: `{summary.get('branch')}`",
        "",
        f"Current decision: `{summary.get('final_decision')}`",
        "",
        "## Current Route",
        "",
        "Exact-init expert replay stabilization is the active evaluation-protocol gate before any new method work.",
        "",
        "## Replay Stabilization",
        "",
        f"- candidate selection: `{candidate_pool.get('selection_rule')}`",
        f"- candidate demos tested: `{summary.get('candidate_demos_tested')}`",
        f"- fixed eligibility set: `{summary.get('fixed_eligibility_set')}`",
        f"- expert failed cases and reasons: `{summary.get('expert_failed_cases_and_reasons')}`",
        f"- learned replay aggregate: `{summary.get('learned_replay_aggregate')}`",
        f"- action validity audit: `{summary.get('action_validity_audit')}`",
        "",
        "## Conclusion",
        "",
        f"`{summary.get('final_decision')}`",
        "",
        summary.get("exact_next_step") or "",
        "",
    ]
    Path("reports/project_state.md").write_text("\n".join(project_lines), encoding="utf-8")
    Path("reports/next_actions.md").write_text(
        "\n".join(
            [
                "# Next Actions",
                "",
                "Date: 2026-07-09 KST",
                "",
                f"Current decision: `{summary.get('final_decision')}`",
                "",
                "## Immediate Next Action",
                "",
                summary.get("exact_next_step") or "",
                "",
                "Do not start a new method unless the replay protocol decision is `READY_FOR_METHOD_AFTER_STABLE_REPLAY_BASELINE`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    decision_path = Path("reports/decision_log.md")
    existing = decision_path.read_text(encoding="utf-8") if decision_path.exists() else "# Decision Log\n"
    marker = "## 2026-07-09: Exact-Init Expert Replay Stabilization"
    entry = "\n".join(
        [
            "",
            marker,
            "",
            f"Decision: `{summary.get('final_decision')}`",
            "",
            f"- experiments happened: `{summary.get('experiments_happened')}`",
            f"- training happened: `{summary.get('training_happened')}`",
            f"- replay/control happened: `{summary.get('replay_control_happened')}`",
            f"- learned policy replay happened: `{summary.get('learned_policy_replay_happened')}`",
            f"- candidate demos tested: `{summary.get('candidate_demos_tested')}`",
            f"- expert-success eligible cases: `{summary.get('fixed_eligibility_set')}`",
            f"- expert-failed cases and reasons: `{summary.get('expert_failed_cases_and_reasons')}`",
            f"- mean/ridge/MLP/adapter replay: `{summary.get('mean_replay_result')}` / `{summary.get('ridge_replay_result')}` / `{summary.get('mlp_replay_result')}` / `{summary.get('adapter_replay_result')}`",
            f"- action validity audit: `{summary.get('action_validity_audit')}`",
            f"- exact next step: {summary.get('exact_next_step')}",
            "",
        ]
    )
    if marker in existing:
        existing = existing.split(marker)[0].rstrip() + entry
    else:
        existing = existing.rstrip() + entry
    decision_path.write_text(existing.rstrip() + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="C:/assets/data/libero/libero_10")
    parser.add_argument("--libero-root", default="C:/assets/repos/LIBERO")
    parser.add_argument("--robosuite-root", default="C:/assets/repos/robosuite")
    parser.add_argument("--adapter-dir", default="runs/smolvla_7d_standard_replay_baseline")
    parser.add_argument("--prior-result-path", default="reports/smolvla_7d_standard_replay_baseline_result.json")
    parser.add_argument("--output-dir", default="runs/exact_init_expert_replay_stabilization")
    parser.add_argument("--report-path", default="reports/exact_init_expert_replay_stabilization.json")
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--train-demos-per-task", type=int, default=5)
    parser.add_argument("--eval-demos-per-task", type=int, default=2)
    parser.add_argument("--records-per-demo", type=int, default=8)
    parser.add_argument("--candidate-demos-per-task", type=int, default=4)
    parser.add_argument("--max-replay-steps", type=int, default=320)
    parser.add_argument("--post-signal-margin", type=int, default=16)
    parser.add_argument("--camera-size", type=int, default=64)
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    json_report = _strip_large(report)
    _write_json(Path(args.report_path), json_report)
    _write_reports(json_report)
    print(json.dumps(json_report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
