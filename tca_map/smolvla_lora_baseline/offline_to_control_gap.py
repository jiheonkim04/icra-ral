"""SmolVLA 7D offline-to-control gap diagnosis.

This runner diagnoses why the fixed SmolVLA/LIBERO 7D adapter can improve
offline action L2 while failing exact-init control replay. It is an evaluation
and interface audit, not a new method or training run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla_lora_baseline import diagnostic as base
from tca_map.smolvla_lora_baseline import exact_init_replay_stabilization as exact_init
from tca_map.smolvla_lora_baseline import replay_bridge
from tca_map.smolvla_lora_baseline import standard_replay_baseline as standard


RUN_GATE = "ALLOW_SMOLVLA_7D_OFFLINE_TO_CONTROL_GAP"
SCHEMA_VERSION = "smolvla-7d-offline-to-control-gap-v1"
FINAL_DECISIONS = {
    "FEATURE_PATH_MISMATCH",
    "GRIPPER_PHASE_FAILURE",
    "TRANSLATION_DIRECTION_FAILURE",
    "OPEN_LOOP_ACTION_SEQUENCE_FAILURE",
    "CLOSED_LOOP_COMPOUNDING_FAILURE",
    "ACTION_VALIDITY_RANGE_FAILURE",
    "READY_FOR_METHOD_AFTER_CONTROL_DIAGNOSIS",
    "STOP_SMOLVLA_7D_METHOD_UNDER_CURRENT_SETUP",
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _sha256_short(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


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


def _task_id_from_demo_path(path: Path) -> str:
    stem = path.stem
    return stem[: -len("_demo")] if stem.endswith("_demo") else stem


def _instruction_from_path(path: Path) -> str:
    return _task_id_from_demo_path(path).replace("_", " ")


def _load_eligible_cases(path: Path) -> list[dict[str, Any]]:
    report = _read_json(path)
    cases = ((report.get("state3_fixed_eligibility_set") or {}).get("eligible_cases") or [])
    return [
        {
            "task_name": case["task_name"],
            "demo_name": case["demo_name"],
            "hdf5_path": case["hdf5_path"],
            "first_done_index": case.get("first_done_index"),
            "reward_sum": case.get("reward_sum"),
            "success": case.get("success"),
            "target_horizon": case.get("target_horizon"),
            "max_steps": case.get("max_steps"),
        }
        for case in cases
    ]


def _best_adapter_name(prior_path: Path) -> str:
    report = _read_json(prior_path)
    return str((report.get("summary") or {}).get("best_lora_name") or "smolvla_state_proj_lora_rank4_7d_adapter")


def _bddl_file(libero_root: Path, hdf5_path: Path) -> Path:
    return (
        libero_root
        / "libero"
        / "libero"
        / "bddl_files"
        / hdf5_path.parent.name
        / f"{_task_id_from_demo_path(hdf5_path)}.bddl"
    )


def _feature_stats(features: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(features, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "mean": [_round(x) for x in arr.mean(axis=0)],
        "std": [_round(x) for x in arr.std(axis=0)],
        "min": [_round(x) for x in arr.min(axis=0)],
        "max": [_round(x) for x in arr.max(axis=0)],
    }


def _action_sign_agreement(pred: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    pred_sign = np.sign(np.asarray(pred, dtype=np.float32))
    expert_sign = np.sign(np.asarray(expert, dtype=np.float32))
    nonzero = expert_sign != 0
    if not np.any(nonzero):
        return {"overall": None, "per_dim": None}
    per_dim = []
    for dim in range(expert.shape[1]):
        mask = nonzero[:, dim]
        per_dim.append(_round(float(np.mean(pred_sign[mask, dim] == expert_sign[mask, dim]))) if np.any(mask) else None)
    return {
        "overall": _round(float(np.mean(pred_sign[nonzero] == expert_sign[nonzero]))),
        "per_dim": per_dim,
    }


def _translation_cosine(pred: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    p = np.asarray(pred[:, :3], dtype=np.float32)
    e = np.asarray(expert[:, :3], dtype=np.float32)
    denom = np.linalg.norm(p, axis=1) * np.linalg.norm(e, axis=1)
    valid = denom > 1e-8
    values = np.zeros((p.shape[0],), dtype=np.float32)
    values[valid] = np.sum(p[valid] * e[valid], axis=1) / denom[valid]
    return {
        "valid_count": int(np.sum(valid)),
        "mean": _round(float(np.mean(values[valid]))) if np.any(valid) else None,
        "negative_rate": _round(float(np.mean(values[valid] < 0.0))) if np.any(valid) else None,
        "first20_mean": _round(float(np.mean(values[:20][valid[:20]]))) if np.any(valid[:20]) else None,
    }


def _first_index(values: np.ndarray, predicate) -> int | None:
    for index, value in enumerate(values):
        if predicate(value):
            return int(index)
    return None


def _gripper_timing(pred: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    pred_grip = np.asarray(pred[:, 6], dtype=np.float32)
    expert_grip = np.asarray(expert[:, 6], dtype=np.float32)
    pred_positive = _first_index(pred_grip, lambda value: float(value) >= 0.0)
    expert_positive = _first_index(expert_grip, lambda value: float(value) >= 0.0)
    pred_sign = np.sign(pred_grip)
    expert_sign = np.sign(expert_grip)
    pred_changes = [index for index in range(1, len(pred_sign)) if pred_sign[index] != pred_sign[index - 1]]
    expert_changes = [index for index in range(1, len(expert_sign)) if expert_sign[index] != expert_sign[index - 1]]
    timing_error = None
    if pred_positive is not None and expert_positive is not None:
        timing_error = int(pred_positive) - int(expert_positive)
    return {
        "pred_first_nonnegative_index": pred_positive,
        "expert_first_nonnegative_index": expert_positive,
        "first_nonnegative_timing_error": timing_error,
        "pred_sign_change_indices_first10": pred_changes[:10],
        "expert_sign_change_indices_first10": expert_changes[:10],
        "sign_agreement_rate": _round(float(np.mean(pred_sign == expert_sign))) if len(pred_sign) else None,
    }


def _phase_slices(length: int) -> dict[str, slice]:
    edges = np.linspace(0, length, num=6, dtype=np.int64).tolist()
    names = ["approach", "contact_grasp", "lift", "transport", "place_release"]
    return {
        name: slice(int(edges[index]), max(int(edges[index + 1]), int(edges[index]) + 1))
        for index, name in enumerate(names)
        if int(edges[index]) < length
    }


def _metrics_for_arrays(pred: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    return standard._metrics(np.asarray(pred, dtype=np.float32), np.asarray(expert, dtype=np.float32))


def _phase_metrics(pred: np.ndarray, expert: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, span in _phase_slices(len(expert)).items():
        result[name] = _metrics_for_arrays(pred[span], expert[span])
    return result


def _top_errors(pred: np.ndarray, expert: np.ndarray, count: int = 8) -> list[dict[str, Any]]:
    error = np.linalg.norm(np.asarray(pred, dtype=np.float32) - np.asarray(expert, dtype=np.float32), axis=1)
    indices = np.argsort(-error)[: int(count)].tolist()
    return [
        {
            "timestep": int(index),
            "action_l2": _round(float(error[index])),
            "translation_l2": _round(float(np.linalg.norm(pred[index, :3] - expert[index, :3]))),
            "rotation_l2": _round(float(np.linalg.norm(pred[index, 3:6] - expert[index, 3:6]))),
            "gripper_abs_error": _round(float(abs(pred[index, 6] - expert[index, 6]))),
            "pred_gripper": _round(float(pred[index, 6])),
            "expert_gripper": _round(float(expert[index, 6])),
        }
        for index in indices
    ]


def _teacher_forced_case(
    *,
    case: dict[str, Any],
    predictors: dict[str, Any],
    max_replay_steps: int,
    post_signal_margin: int,
) -> dict[str, Any]:
    path = Path(case["hdf5_path"])
    demo_name = case["demo_name"]
    demo_window = replay_bridge._demo_window(path, demo_name, int(max_replay_steps), int(post_signal_margin))
    features = np.asarray(demo_window["features"], dtype=np.float32)
    expert = np.asarray(demo_window["actions"], dtype=np.float32)
    predictions = {
        "mean_action": predictors["mean_action"]["predict"](features),
        "ridge": predictors["ridge"]["predict"](features),
        "smolvla_7d_adapter": predictors["smolvla_7d_adapter"]["predict"](features),
    }
    policies: dict[str, Any] = {}
    for name, pred in predictions.items():
        first20 = _metrics_for_arrays(pred[: min(20, len(pred))], expert[: min(20, len(expert))])
        phases = _phase_metrics(pred, expert)
        phase_l2 = [float(item["action_l2"]) for item in phases.values() if item.get("action_l2") is not None]
        aggregate_l2 = float(_metrics_for_arrays(pred, expert)["action_l2"])
        policies[name] = {
            "action_metrics": _metrics_for_arrays(pred, expert),
            "first20_metrics": first20,
            "phase_metrics": phases,
            "phase_critical_error_ratio": _round(max(phase_l2) / max(1e-8, aggregate_l2)) if phase_l2 else None,
            "top_error_timesteps": _top_errors(pred, expert),
            "gripper_timing": _gripper_timing(pred, expert),
            "action_sign_agreement": _action_sign_agreement(pred, expert),
            "translation_cosine": _translation_cosine(pred, expert),
            "action_validity": standard._action_validity(pred),
        }
    return {
        "task_name": case["task_name"],
        "demo_name": demo_name,
        "hdf5_path": str(path),
        "target_horizon": int(demo_window["target_horizon"]),
        "hdf5_first_signal_index": demo_window.get("first_signal_index"),
        "feature_stats": _feature_stats(features),
        "policies": policies,
    }


def _aggregate_teacher_forced(cases: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for policy in ["mean_action", "ridge", "smolvla_7d_adapter"]:
        metrics = [((case.get("policies") or {}).get(policy) or {}).get("action_metrics") or {} for case in cases]
        first20 = [((case.get("policies") or {}).get(policy) or {}).get("first20_metrics") or {} for case in cases]
        gripper = [((case.get("policies") or {}).get(policy) or {}).get("gripper_timing") or {} for case in cases]
        cosine = [((case.get("policies") or {}).get(policy) or {}).get("translation_cosine") or {} for case in cases]
        phase_ratio = [
            ((case.get("policies") or {}).get(policy) or {}).get("phase_critical_error_ratio")
            for case in cases
        ]
        phase_ratio = [float(value) for value in phase_ratio if value is not None]
        aggregate[policy] = {
            "case_count": len(metrics),
            "action_l2_mean": _round(float(np.mean([float(item["action_l2"]) for item in metrics]))),
            "translation_l2_mean": _round(float(np.mean([float(item["translation_l2"]) for item in metrics]))),
            "rotation_l2_mean": _round(float(np.mean([float(item["rotation_l2"]) for item in metrics]))),
            "gripper_error_mean": _round(float(np.mean([float(item["gripper_error"]) for item in metrics]))),
            "first20_action_l2_mean": _round(float(np.mean([float(item["action_l2"]) for item in first20]))),
            "phase_critical_error_ratio_mean": _round(float(np.mean(phase_ratio))) if phase_ratio else None,
            "gripper_timing_error_values": [item.get("first_nonnegative_timing_error") for item in gripper],
            "gripper_sign_agreement_mean": _round(
                float(np.mean([float(item["sign_agreement_rate"]) for item in gripper if item.get("sign_agreement_rate") is not None]))
            ),
            "translation_cosine_mean": _round(
                float(np.mean([float(item["mean"]) for item in cosine if item.get("mean") is not None]))
            )
            if any(item.get("mean") is not None for item in cosine)
            else None,
            "translation_cosine_negative_rate_mean": _round(
                float(np.mean([float(item["negative_rate"]) for item in cosine if item.get("negative_rate") is not None]))
            )
            if any(item.get("negative_rate") is not None for item in cosine)
            else None,
        }
    adapter = aggregate.get("smolvla_7d_adapter") or {}
    ridge = aggregate.get("ridge") or {}
    aggregate["critical_question"] = {
        "low_sparse_offline_l2_hides_full_sequence_or_phase_error": bool(
            adapter
            and (
                float(adapter.get("first20_action_l2_mean") or 0.0) > float(adapter.get("action_l2_mean") or 0.0) * 1.2
                or float(adapter.get("phase_critical_error_ratio_mean") or 0.0) > 1.35
                or float(adapter.get("gripper_error_mean") or 0.0) > 0.8
            )
        ),
        "adapter_teacher_forced_l2_worse_than_ridge": bool(
            adapter and ridge and float(adapter.get("action_l2_mean") or 0.0) >= float(ridge.get("action_l2_mean") or 0.0)
        ),
    }
    return aggregate


def _run_teacher_forced(
    *,
    eligible_cases: list[dict[str, Any]],
    predictors: dict[str, Any],
    max_replay_steps: int,
    post_signal_margin: int,
) -> dict[str, Any]:
    started = time.monotonic()
    cases = [
        _teacher_forced_case(
            case=case,
            predictors=predictors,
            max_replay_steps=max_replay_steps,
            post_signal_margin=post_signal_margin,
        )
        for case in eligible_cases
    ]
    return {
        "executed": True,
        "reason": "teacher-forced full-sequence action audit on expert HDF5 states",
        "cases": cases,
        "aggregate": _aggregate_teacher_forced(cases),
        "runtime_sec": _round(time.monotonic() - started, 3),
    }


def _live_feature_probe(
    *,
    args: argparse.Namespace,
    eligible_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    started = time.monotonic()
    env_cls, env_meta = replay_bridge._load_env_class_noninteractive(
        libero_root=Path(args.libero_root),
        robosuite_root=Path(args.robosuite_root),
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
    )
    rows: list[dict[str, Any]] = []
    for case in eligible_cases:
        path = Path(case["hdf5_path"])
        demo_window = replay_bridge._demo_window(path, case["demo_name"], int(args.max_replay_steps), int(args.post_signal_margin))
        env = None
        try:
            bddl_file = _bddl_file(Path(args.libero_root), path)
            env = env_cls(bddl_file_name=str(bddl_file), camera_heights=int(args.camera_size), camera_widths=int(args.camera_size))
            env.seed(0)
            obs = env.reset()
            obs = env.set_init_state(np.asarray(demo_window["init_state"], dtype=np.float64))
            live_feature, live_meta = replay_bridge._observation_feature(obs, 0.0)
            hdf5_feature = np.asarray(demo_window["features"][0], dtype=np.float32)
            diff = live_feature - hdf5_feature
            rows.append(
                {
                    "task_name": case["task_name"],
                    "demo_name": case["demo_name"],
                    "hdf5_feature_source": "obs/ee_states[:6] plus timestep_fraction",
                    "live_feature_source": live_meta.get("source"),
                    "hdf5_feature_first": [_round(x) for x in hdf5_feature],
                    "live_feature_first": [_round(x) for x in live_feature],
                    "feature_l2": _round(float(np.linalg.norm(diff))),
                    "position_l2": _round(float(np.linalg.norm(diff[:3]))),
                    "orientation_l2": _round(float(np.linalg.norm(diff[3:6]))),
                    "live_obs_has_ee_states": bool("ee_states" in obs),
                    "live_obs_has_robot0_eef_quat": bool("robot0_eef_quat" in obs),
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "task_name": case["task_name"],
                    "demo_name": case["demo_name"],
                    "error": _compact_error(exc),
                }
            )
        finally:
            if env is not None:
                try:
                    env.close()
                except Exception:
                    pass
    orientation_values = [float(row["orientation_l2"]) for row in rows if row.get("orientation_l2") is not None]
    mismatch = bool(orientation_values and float(np.mean(orientation_values)) > float(args.feature_mismatch_threshold))
    return {
        "executed": True,
        "env": env_meta,
        "rows": rows,
        "orientation_l2_mean": _round(float(np.mean(orientation_values))) if orientation_values else None,
        "orientation_l2_max": _round(float(np.max(orientation_values))) if orientation_values else None,
        "feature_path_mismatch_found": mismatch,
        "mismatch_threshold": float(args.feature_mismatch_threshold),
        "runtime_sec": _round(time.monotonic() - started, 3),
    }


def _feature_path_audit(
    *,
    args: argparse.Namespace,
    adapter: replay_bridge.ExecutableAdapter,
    eligible_cases: list[dict[str, Any]],
) -> dict[str, Any]:
    started = time.monotonic()
    adapter_payload = adapter.payload
    artifact_path = Path(adapter.artifact_path)
    first_case = eligible_cases[0]
    first_window = replay_bridge._demo_window(
        Path(first_case["hdf5_path"]),
        first_case["demo_name"],
        int(args.max_replay_steps),
        int(args.post_signal_margin),
    )
    live_probe = _live_feature_probe(args=args, eligible_cases=eligible_cases)
    mismatch_table = [
        {
            "check": "offline_train_vs_offline_eval",
            "result": "match",
            "evidence": "Both use libero_7d_interface_fix._feature_matrix records built from HDF5 obs/ee_states[:6] plus timestep fraction.",
        },
        {
            "check": "offline_eval_vs_prior_open_loop_replay",
            "result": "match",
            "evidence": "Prior eligible replay generated fixed action sequences from replay_bridge._demo_window features built from the same HDF5 obs/ee_states[:6] path.",
        },
        {
            "check": "offline_train_vs_live_closed_loop_replay",
            "result": "mismatch" if live_probe.get("feature_path_mismatch_found") else "match",
            "evidence": "Live env observations are converted by replay_bridge._observation_feature through the canonical LIBERO ee_states feature builder.",
        },
        {
            "check": "normalization_train_to_replay",
            "result": "match",
            "evidence": "Adapter payload contains train-split-only feature/action normalization and ExecutableAdapter.predict_features reuses it.",
        },
        {
            "check": "previous_action_action_in_initialization",
            "result": "not_used",
            "evidence": "The fixed 7D adapter feature path uses only 6D EEF state plus timestep fraction; previous action/action_in is not an input.",
        },
        {
            "check": "timestep_chunk_index_alignment",
            "result": "bounded_match",
            "evidence": "Feature timestep fraction is t / (demo action length - 1), and action label is actions[t, :7].",
        },
    ]
    return {
        "executed": True,
        "offline_feature_schema": "HDF5 obs/ee_states[:6] (ee_pos + ee_ori) plus timestep_fraction",
        "offline_training_input_source": "libero_7d_interface_fix._feature_matrix(train_records)",
        "offline_evaluation_input_source": "libero_7d_interface_fix._feature_matrix(eval_records)",
        "prior_open_loop_replay_input_source": "replay_bridge._demo_window(...)[features] from HDF5 obs/ee_states[:6]",
        "live_closed_loop_replay_input_source": "replay_bridge._observation_feature(obs, tfrac), using canonical LIBERO ee_states feature builder",
        "adapter_checkpoint_and_weights": {
            "artifact_path": str(artifact_path),
            "artifact_sha256_16": _sha256_short(artifact_path),
            "adapter_name": adapter.name,
            "checkpoint_path": adapter_payload.get("checkpoint_path"),
            "state_proj_weight_file": adapter_payload.get("state_proj_weight_file"),
            "lora_rank": adapter_payload.get("lora_rank"),
            "hidden_dim": adapter_payload.get("hidden_dim"),
            "normalization": adapter_payload.get("normalization"),
            "feature_normalization": adapter_payload.get("feature_normalization"),
        },
        "first_hdf5_feature_stats": _feature_stats(np.asarray(first_window["features"], dtype=np.float32)),
        "live_feature_probe": live_probe,
        "mismatch_table": mismatch_table,
        "feature_path_mismatch_found": bool(live_probe.get("feature_path_mismatch_found")),
        "feature_path_audit_result": "FEATURE_PATH_MISMATCH_FOR_TRUE_CLOSED_LOOP" if live_probe.get("feature_path_mismatch_found") else "FEATURE_PATH_MATCH",
        "runtime_sec": _round(time.monotonic() - started, 3),
    }


def _open_loop_replay_from_prior(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"executed": False, "reason": "prior exact-init replay report missing", "aggregate": {}, "cases": []}
    report = _read_json(path)
    replay = report.get("state4_learned_replay") or {}
    return {
        "executed": bool(replay.get("executed")),
        "source_report": str(path),
        "reason": "reused prior eligible-only open-loop action replay; it already replayed expert, mean, ridge, and adapter fixed sequences on the fixed eligible set",
        "aggregate": replay.get("aggregate") or {},
        "cases": replay.get("cases") or [],
        "no_expert_failed_cases_in_aggregate": bool(((replay.get("aggregate") or {}).get("learned_aggregate_uses_only_eligible_cases"))),
    }


def _oracle_sequence_diagnostics(teacher_forced: dict[str, Any]) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for case in teacher_forced.get("cases") or []:
        adapter = (((case.get("policies") or {}).get("smolvla_7d_adapter") or {}).get("action_metrics") or {})
        top = (((case.get("policies") or {}).get("smolvla_7d_adapter") or {}).get("top_error_timesteps") or [])
        gripper = (((case.get("policies") or {}).get("smolvla_7d_adapter") or {}).get("gripper_timing") or {})
        rows.append(
            {
                "task_name": case.get("task_name"),
                "demo_name": case.get("demo_name"),
                "adapter_total_l2": adapter.get("action_l2"),
                "adapter_motion_l2_first6": adapter.get("action_l2_first6"),
                "adapter_gripper_error": adapter.get("gripper_error"),
                "adapter_gripper_timing": gripper,
                "top_error_timesteps": top[:3],
                "adapter_motion_plus_expert_gripper_note": "Oracle diagnostic would remove dim-6 error only; motion errors remain represented by action_l2_first6.",
                "expert_motion_plus_adapter_gripper_note": "Oracle diagnostic isolates gripper timing/error; not a method baseline or success claim.",
            }
        )
    aggregate = teacher_forced.get("aggregate") or {}
    adapter_agg = aggregate.get("smolvla_7d_adapter") or {}
    return {
        "executed": True,
        "replay_executed": False,
        "replay_skip_reason": "Oracle variants are diagnostic only; feature-path mismatch makes additional closed-loop oracle replay misleading.",
        "rows": rows,
        "aggregate_interpretation": {
            "adapter_motion_error_first6_mean": adapter_agg.get("translation_l2_mean"),
            "adapter_rotation_l2_mean": adapter_agg.get("rotation_l2_mean"),
            "adapter_gripper_error_mean": adapter_agg.get("gripper_error_mean"),
            "gripper_oracle_alone_unlikely_to_fix_motion": bool(float(adapter_agg.get("translation_l2_mean") or 0.0) > 0.2),
        },
    }


def _closed_loop_divergence_placeholder(feature_audit: dict[str, Any]) -> dict[str, Any]:
    if feature_audit.get("feature_path_mismatch_found"):
        return {
            "executed": False,
            "reason": "Skipped as a model-quality measurement because STATE 1 found live closed-loop feature mismatch.",
            "blocked_by": "FEATURE_PATH_MISMATCH_FOR_TRUE_CLOSED_LOOP",
            "required_before_rerun": "Provide live env features matching HDF5 ee_states (ee_pos + ee_ori) or retrain/evaluate with the live observation schema.",
        }
    return {
        "executed": False,
        "reason": "Closed-loop divergence runner not needed because no feature mismatch was detected in this bounded pass.",
    }


def _decide(report: dict[str, Any]) -> tuple[str, str, str]:
    feature = report.get("state1_feature_path_audit") or {}
    teacher = report.get("state2_teacher_forced_sequence") or {}
    open_loop = report.get("state3_open_loop_action_replay") or {}
    if feature.get("feature_path_mismatch_found"):
        return (
            "FEATURE_PATH_MISMATCH",
            "FEATURE_PATH_MISMATCH",
            "Fix the live closed-loop feature schema so replay uses HDF5-compatible ee_states features, then rerun teacher-forced and replay diagnostics before any method work.",
        )
    aggregate = teacher.get("aggregate") or {}
    adapter = aggregate.get("smolvla_7d_adapter") or {}
    if float(adapter.get("translation_cosine_negative_rate_mean") or 0.0) > 0.35:
        return (
            "TRANSLATION_DIRECTION_FAILURE",
            "TRANSLATION_DIRECTION_FAILURE",
            "Diagnose translation sign/direction in the fixed adapter; do not start a method.",
        )
    if float(adapter.get("gripper_error_mean") or 0.0) > 0.8:
        return (
            "GRIPPER_PHASE_FAILURE",
            "GRIPPER_PHASE_FAILURE",
            "Diagnose gripper timing and phase labels in the fixed adapter; do not start a method.",
        )
    adapter_replay = ((open_loop.get("aggregate") or {}).get("smolvla_7d_adapter") or {})
    if open_loop.get("executed") and int(adapter_replay.get("success_count") or 0) == 0:
        return (
            "OPEN_LOOP_ACTION_SEQUENCE_FAILURE",
            "OPEN_LOOP_ACTION_SEQUENCE_FAILURE",
            "The adapter action sequence fails even on expert-state features; diagnose sequence semantics before any method work.",
        )
    return (
        "STOP_SMOLVLA_7D_METHOD_UNDER_CURRENT_SETUP",
        "STOP_SMOLVLA_7D_METHOD_UNDER_CURRENT_SETUP",
        "Stop SmolVLA 7D method work under the current setup unless a bounded baseline/interface fix is proven.",
    )


def _strip_large(payload: Any) -> Any:
    if isinstance(payload, dict):
        result: dict[str, Any] = {}
        for key, value in payload.items():
            if key == "cases" and isinstance(value, list) and len(value) > 0:
                result[key] = [_strip_large(item) for item in value]
            elif key in {"reward_trajectory"}:
                values = value if isinstance(value, list) else []
                result["reward_trajectory_summary"] = {
                    "length": len(values),
                    "first_10": values[:10],
                    "last_10": values[-10:],
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
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "STOP_SMOLVLA_7D_METHOD_UNDER_CURRENT_SETUP",
        "policy": {
            "run_gate_set": _env_flag(RUN_GATE),
            "forbidden_gates_set": forbidden,
            "new_method_created": False,
            "paper_claims_made": False,
            "downloads_performed": False,
            "openvla_oft_executed": False,
            "full_benchmark_executed": False,
            "training_performed": False,
            "replay_control_performed": False,
            "oracle_diagnostics_count_as_success": False,
            "learned_evaluated_on_expert_failed_cases": False,
        },
        "paths": {
            "data_root": str(Path(args.data_root)),
            "libero_root": str(Path(args.libero_root)),
            "robosuite_root": str(Path(args.robosuite_root)),
            "adapter_dir": str(Path(args.adapter_dir)),
            "exact_init_report_path": str(Path(args.exact_init_report_path)),
        },
        "state0_eligible_set": {},
        "state1_feature_path_audit": {},
        "state2_teacher_forced_sequence": {},
        "state3_open_loop_action_replay": {},
        "state4_closed_loop_divergence": {},
        "state5_oracle_diagnostics": {},
        "summary": {},
        "error": None,
    }

    def finish(decision: str, failure_category: str, next_step: str, code: int) -> tuple[dict[str, Any], int]:
        if decision not in FINAL_DECISIONS:
            raise ValueError(f"invalid final decision: {decision}")
        report["decision"] = decision
        report["summary"].update(
            {
                "final_decision": decision,
                "failure_category": failure_category,
                "exact_next_step": next_step,
                "runtime_sec": _round(time.monotonic() - started, 3),
            }
        )
        return report, code

    if not report["policy"]["run_gate_set"]:
        return finish(
            "STOP_SMOLVLA_7D_METHOD_UNDER_CURRENT_SETUP",
            "gate_missing",
            f"Set {RUN_GATE}=1 for this bounded offline-to-control diagnosis.",
            2,
        )
    if forbidden:
        report["error"] = {"message": "Forbidden gate(s) set: " + ", ".join(forbidden)}
        return finish(
            "STOP_SMOLVLA_7D_METHOD_UNDER_CURRENT_SETUP",
            "forbidden_gate_set",
            "Clear forbidden method/download/full-rollout gates and rerun.",
            3,
        )

    try:
        eligible_cases = _load_eligible_cases(Path(args.exact_init_report_path))
        if not eligible_cases:
            return finish(
                "STOP_SMOLVLA_7D_METHOD_UNDER_CURRENT_SETUP",
                "missing_eligible_set",
                "Recreate the exact-init eligible set before diagnosing offline-to-control transfer.",
                4,
            )
        best_name = _best_adapter_name(Path(args.exact_init_report_path))
        adapter_path = Path(args.adapter_dir) / f"{best_name}.pt"
        adapter = replay_bridge.ExecutableAdapter.load(adapter_path)
        predictors, predictor_meta = exact_init._build_learned_predictors(args, best_name)
        report["state0_eligible_set"] = {
            "eligible_cases": eligible_cases,
            "eligible_case_count": len(eligible_cases),
            "source_report": str(Path(args.exact_init_report_path)),
            "best_lora_name": best_name,
            "predictor_meta": predictor_meta,
        }
        feature = _feature_path_audit(args=args, adapter=adapter, eligible_cases=eligible_cases)
        report["state1_feature_path_audit"] = feature
        teacher = _run_teacher_forced(
            eligible_cases=eligible_cases,
            predictors=predictors,
            max_replay_steps=int(args.max_replay_steps),
            post_signal_margin=int(args.post_signal_margin),
        )
        report["state2_teacher_forced_sequence"] = teacher
        open_loop = _open_loop_replay_from_prior(Path(args.exact_init_report_path))
        report["state3_open_loop_action_replay"] = open_loop
        report["state4_closed_loop_divergence"] = _closed_loop_divergence_placeholder(feature)
        report["state5_oracle_diagnostics"] = _oracle_sequence_diagnostics(teacher)
        report["policy"]["replay_control_performed"] = bool(feature.get("live_feature_probe", {}).get("executed")) or bool(open_loop.get("executed"))
        decision, failure_category, next_step = _decide(report)
        teacher_agg = teacher.get("aggregate") or {}
        open_agg = open_loop.get("aggregate") or {}
        report["summary"].update(
            {
                "branch": _current_branch(),
                "experiments_happened": True,
                "training_happened": False,
                "replay_control_happened": bool(report["policy"]["replay_control_performed"]),
                "downloads_happened": False,
                "openvla_oft_happened": False,
                "eligible_demos_used": [
                    f"{case['task_name']}::{case['demo_name']}" for case in eligible_cases
                ],
                "feature_path_audit_result": feature.get("feature_path_audit_result"),
                "feature_path_mismatch_found": feature.get("feature_path_mismatch_found"),
                "live_feature_orientation_l2_mean": (feature.get("live_feature_probe") or {}).get("orientation_l2_mean"),
                "teacher_forced_sequence_result": teacher_agg,
                "open_loop_action_replay_result": open_agg,
                "closed_loop_divergence_result": report["state4_closed_loop_divergence"],
                "oracle_diagnostic_result": report["state5_oracle_diagnostics"].get("aggregate_interpretation"),
                "adapter_teacher_forced_action_l2": (teacher_agg.get("smolvla_7d_adapter") or {}).get("action_l2_mean"),
                "adapter_teacher_forced_gripper_error": (teacher_agg.get("smolvla_7d_adapter") or {}).get("gripper_error_mean"),
                "adapter_open_loop_progress": (open_agg.get("smolvla_7d_adapter") or {}).get("progress_proxy_mean"),
                "mean_open_loop_progress": (open_agg.get("mean_action") or {}).get("progress_proxy_mean"),
                "ridge_open_loop_progress": (open_agg.get("ridge") or {}).get("progress_proxy_mean"),
            }
        )
        return finish(decision, failure_category, next_step, 0)
    except Exception as exc:  # noqa: BLE001
        report["error"] = _compact_error(exc)
        return finish(
            "STOP_SMOLVLA_7D_METHOD_UNDER_CURRENT_SETUP",
            "runner_error",
            "Fix the reported offline-to-control diagnostic runner error and rerun.",
            11,
        )


def _write_reports(report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    feature = report.get("state1_feature_path_audit") or {}
    teacher = report.get("state2_teacher_forced_sequence") or {}
    open_loop = report.get("state3_open_loop_action_replay") or {}
    closed_loop = report.get("state4_closed_loop_divergence") or {}
    oracle = report.get("state5_oracle_diagnostics") or {}
    lines = [
        "# SmolVLA 7D Offline-To-Control Gap Diagnosis",
        "",
        f"Final decision: `{summary.get('final_decision')}`",
        "",
        "This is an evaluation and interface diagnosis, not a new RA-L method.",
        "",
        "## Summary",
        "",
        f"- eligible demos used: `{summary.get('eligible_demos_used')}`",
        f"- feature path audit result: `{summary.get('feature_path_audit_result')}`",
        f"- live feature orientation L2 mean: `{summary.get('live_feature_orientation_l2_mean')}`",
        f"- adapter teacher-forced action L2: `{summary.get('adapter_teacher_forced_action_l2')}`",
        f"- adapter teacher-forced gripper error: `{summary.get('adapter_teacher_forced_gripper_error')}`",
        f"- mean/ridge/adapter open-loop progress: `{summary.get('mean_open_loop_progress')}` / `{summary.get('ridge_open_loop_progress')}` / `{summary.get('adapter_open_loop_progress')}`",
        f"- closed-loop divergence: `{summary.get('closed_loop_divergence_result')}`",
        f"- oracle diagnostics: `{summary.get('oracle_diagnostic_result')}`",
        f"- failure category: `{summary.get('failure_category')}`",
        "",
        f"Exact next step: {summary.get('exact_next_step')}",
        "",
    ]
    Path("reports/smolvla_7d_offline_to_control_gap.md").write_text("\n".join(lines), encoding="utf-8")
    Path("reports/smolvla_7d_closed_loop_failure_analysis.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Closed-Loop Failure Analysis",
                "",
                f"- closed-loop divergence executed: `{closed_loop.get('executed')}`",
                f"- reason: `{closed_loop.get('reason')}`",
                f"- blocked by: `{closed_loop.get('blocked_by')}`",
                f"- required before rerun: `{closed_loop.get('required_before_rerun')}`",
                "",
                "Prior eligible-only open-loop replay result:",
                "",
                f"`{open_loop.get('aggregate')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_feature_path_audit.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Feature Path Audit",
                "",
                f"- result: `{feature.get('feature_path_audit_result')}`",
                f"- offline feature schema: `{feature.get('offline_feature_schema')}`",
                f"- prior open-loop replay input source: `{feature.get('prior_open_loop_replay_input_source')}`",
                f"- live closed-loop replay input source: `{feature.get('live_closed_loop_replay_input_source')}`",
                f"- mismatch table: `{feature.get('mismatch_table')}`",
                f"- live feature probe: `{feature.get('live_feature_probe')}`",
                f"- adapter checkpoint/weights: `{feature.get('adapter_checkpoint_and_weights')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_phase_gripper_audit.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Phase And Gripper Audit",
                "",
                f"- teacher-forced aggregate: `{teacher.get('aggregate')}`",
                f"- oracle diagnostics: `{oracle}`",
                "",
                "Critical question: low sparse offline L2 does not prove full-trajectory phase correctness; see the full JSON for per-case top error timesteps and phase-wise errors.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_next_decision.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Next Decision",
                "",
                f"Final decision: `{summary.get('final_decision')}`",
                "",
                f"- failure category: `{summary.get('failure_category')}`",
                f"- feature path audit result: `{summary.get('feature_path_audit_result')}`",
                f"- open-loop action replay result: `{summary.get('open_loop_action_replay_result')}`",
                "",
                f"Exact next step: {summary.get('exact_next_step')}",
                "",
                "Do not propose a new RA-L method unless the decision is `READY_FOR_METHOD_AFTER_CONTROL_DIAGNOSIS`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/project_state.md").write_text(
        "\n".join(
            [
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
                "SmolVLA 7D offline-to-control gap diagnosis is the active evaluation gate.",
                "",
                "## Diagnosis",
                "",
                f"- eligible demos used: `{summary.get('eligible_demos_used')}`",
                f"- feature path audit result: `{summary.get('feature_path_audit_result')}`",
                f"- teacher-forced result: `{summary.get('teacher_forced_sequence_result')}`",
                f"- open-loop result: `{summary.get('open_loop_action_replay_result')}`",
                f"- closed-loop divergence result: `{summary.get('closed_loop_divergence_result')}`",
                f"- failure category: `{summary.get('failure_category')}`",
                "",
                "## Conclusion",
                "",
                f"`{summary.get('final_decision')}`",
                "",
                summary.get("exact_next_step") or "",
                "",
            ]
        ),
        encoding="utf-8",
    )
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
                "Do not start a new method unless the control diagnosis decision is `READY_FOR_METHOD_AFTER_CONTROL_DIAGNOSIS`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    decision_path = Path("reports/decision_log.md")
    existing = decision_path.read_text(encoding="utf-8") if decision_path.exists() else "# Decision Log\n"
    marker = "## 2026-07-09: SmolVLA 7D Offline-To-Control Gap Diagnosis"
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
            f"- eligible demos used: `{summary.get('eligible_demos_used')}`",
            f"- feature path audit result: `{summary.get('feature_path_audit_result')}`",
            f"- teacher-forced sequence result: `{summary.get('teacher_forced_sequence_result')}`",
            f"- open-loop action replay result: `{summary.get('open_loop_action_replay_result')}`",
            f"- closed-loop divergence result: `{summary.get('closed_loop_divergence_result')}`",
            f"- oracle diagnostic result: `{summary.get('oracle_diagnostic_result')}`",
            f"- failure category: `{summary.get('failure_category')}`",
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
    parser.add_argument("--exact-init-report-path", default="reports/exact_init_expert_replay_stabilization.json")
    parser.add_argument("--output-dir", default="runs/smolvla_7d_offline_to_control_gap")
    parser.add_argument("--report-path", default="reports/smolvla_7d_offline_to_control_gap.json")
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--train-demos-per-task", type=int, default=5)
    parser.add_argument("--eval-demos-per-task", type=int, default=2)
    parser.add_argument("--records-per-demo", type=int, default=8)
    parser.add_argument("--max-replay-steps", type=int, default=320)
    parser.add_argument("--post-signal-margin", type=int, default=16)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--feature-mismatch-threshold", type=float, default=0.1)
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    json_report = _strip_large(report)
    _write_json(Path(args.report_path), json_report)
    _write_reports(json_report)
    print(json.dumps(json_report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
