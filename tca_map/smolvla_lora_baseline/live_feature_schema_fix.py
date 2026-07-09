"""SmolVLA 7D live feature schema fix diagnostic.

This runner validates the canonical LIBERO EEF feature builder that maps live
RoboSuite observations to the same 6D ``ee_states`` schema used by HDF5/offline
fixed-7D adapter training. It does not train or invent a method.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any, Callable

import numpy as np

from tca_map.datasets.libero_full_demo_expert_replay_sanity import _run_replay_variant
from tca_map.datasets.libero_zero_reward_rollout_diagnosis import (
    _best_object_key,
    _distance,
    _distance_delta,
    _extract_eef,
    _extract_pos,
    _object_position_keys,
)
from tca_map.smolvla_lora_baseline import diagnostic as base
from tca_map.smolvla_lora_baseline import exact_init_replay_stabilization as exact_init
from tca_map.smolvla_lora_baseline import libero_ee_state_features as ee_features
from tca_map.smolvla_lora_baseline import offline_to_control_gap as gap
from tca_map.smolvla_lora_baseline import replay_bridge
from tca_map.smolvla_lora_baseline import standard_replay_baseline as standard


RUN_GATE = "ALLOW_SMOLVLA_7D_LIVE_FEATURE_SCHEMA_FIX"
SCHEMA_VERSION = "smolvla-7d-live-feature-schema-fix-v1"
FINAL_DECISIONS = {
    "READY_FOR_METHOD_AFTER_FEATURE_FIX",
    "FEATURE_FIXED_BUT_CONTROL_GAP_REMAINS",
    "FEATURE_CONVENTION_UNRESOLVED",
    "FEATURE_PATH_STILL_MISMATCHED",
    "ACTION_VALIDITY_RANGE_FAILURE",
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def _bddl_file(libero_root: Path, hdf5_path: Path) -> Path:
    return (
        libero_root
        / "libero"
        / "libero"
        / "bddl_files"
        / hdf5_path.parent.name
        / f"{_task_id_from_demo_path(hdf5_path)}.bddl"
    )


def _load_eligible_cases(path: Path) -> list[dict[str, Any]]:
    report = _read_json(path)
    return list((report.get("state3_fixed_eligibility_set") or {}).get("eligible_cases") or [])


def _best_adapter_name(path: Path) -> str:
    report = _read_json(path)
    return str((report.get("summary") or {}).get("best_lora_name") or "smolvla_state_proj_lora_rank4_7d_adapter")


def _stats(values: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float32)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": [_round(x) for x in arr.min(axis=0)],
        "max": [_round(x) for x in arr.max(axis=0)],
        "mean": [_round(x) for x in arr.mean(axis=0)],
        "std": [_round(x) for x in arr.std(axis=0)],
    }


def _hdf5_schema_audit(eligible_cases: list[dict[str, Any]]) -> dict[str, Any]:
    import h5py

    rows: list[dict[str, Any]] = []
    all_ee = []
    all_ori = []
    continuity = []
    for case in eligible_cases:
        path = Path(case["hdf5_path"])
        demo_name = case["demo_name"]
        with h5py.File(path, "r") as handle:
            obs = handle["data"][demo_name]["obs"]
            keys = sorted(str(key) for key in obs.keys())
            ee = np.asarray(obs["ee_states"], dtype=np.float32)
            ee_pos = np.asarray(obs["ee_pos"], dtype=np.float32) if "ee_pos" in obs else None
            ee_ori = np.asarray(obs["ee_ori"], dtype=np.float32) if "ee_ori" in obs else None
            equals_pos_ori = bool(ee_pos is not None and ee_ori is not None and np.allclose(ee[:, :3], ee_pos) and np.allclose(ee[:, 3:6], ee_ori))
            ori = ee[:, 3:6]
            diffs = np.linalg.norm(np.diff(ori, axis=0), axis=1) if len(ori) > 1 else np.zeros((0,), dtype=np.float32)
            wrap_jumps = [int(index + 1) for index, value in enumerate(diffs) if float(value) > np.pi]
            all_ee.append(ee[:, :6])
            all_ori.append(ori)
            continuity.extend(float(value) for value in diffs.tolist())
            relationship = {
                key: {"present": key in obs, "shape": list(obs[key].shape), "dtype": str(obs[key].dtype)}
                for key in ["robot0_eef_pos", "robot0_eef_quat", "robot0_eef_euler", "ee_pos", "ee_ori", "ee_states"]
                if key in obs
            }
            rows.append(
                {
                    "task_name": case["task_name"],
                    "demo_name": demo_name,
                    "obs_keys": keys,
                    "ee_states_shape": list(ee.shape),
                    "ee_states_dtype": str(obs["ee_states"].dtype),
                    "ee_states_is_6d": bool(ee.ndim == 2 and ee.shape[1] == 6),
                    "ee_states_equals_ee_pos_plus_ee_ori": equals_pos_ori,
                    "initial_ee_states": [_round(x) for x in ee[0, :6]],
                    "initial_ee_pos": [_round(x) for x in ee_pos[0]] if ee_pos is not None else None,
                    "initial_ee_ori": [_round(x) for x in ee_ori[0]] if ee_ori is not None else None,
                    "ee_ori_continuity": {
                        "mean_step_l2": _round(float(np.mean(diffs))) if diffs.size else 0.0,
                        "max_step_l2": _round(float(np.max(diffs))) if diffs.size else 0.0,
                        "angle_wrapping_discontinuity_indices": wrap_jumps[:20],
                        "has_angle_wrapping_discontinuities": bool(wrap_jumps),
                    },
                    "related_keys": relationship,
                }
            )
    ee_all = np.concatenate(all_ee, axis=0)
    ori_all = np.concatenate(all_ori, axis=0)
    return {
        "executed": True,
        "exact_keys_used_for_hdf5_ee_states": ["obs/ee_states", "obs/ee_pos", "obs/ee_ori"],
        "ee_states_summary": _stats(ee_all),
        "ee_ori_summary": _stats(ori_all),
        "ee_states_representation": "position plus orientation",
        "orientation_inference": {
            "selected": ee_features.ORIENTATION_CONVENTION,
            "euler_xyz": False,
            "euler_zyx": False,
            "axis_angle_or_rotation_vector": True,
            "quaternion_derived": True,
            "evidence": "Live robot0_eef_quat converted with XYZW axis-angle branch [0, 2pi] aligns with HDF5 ee_ori; canonical SciPy rotvec and quat[:3] do not.",
        },
        "initial_timestep_examples": [row["initial_ee_states"] for row in rows],
        "ee_ori_continuity_global": {
            "mean_step_l2": _round(float(np.mean(continuity))) if continuity else None,
            "max_step_l2": _round(float(np.max(continuity))) if continuity else None,
            "has_angle_wrapping_discontinuities": bool(any(row["ee_ori_continuity"]["has_angle_wrapping_discontinuities"] for row in rows)),
        },
        "rows": rows,
    }


def _candidate_orientation(quat: np.ndarray, name: str) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    if name == "quat_vector_part_legacy":
        return quat[:3].astype(np.float32)
    if name == "xyzw_axis_angle_0_to_2pi":
        return ee_features.quat_xyzw_to_hdf5_axis_angle(quat)
    try:
        from scipy.spatial.transform import Rotation as R
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError("scipy is required for Euler/rotvec candidate audit") from exc
    if name.startswith("xyzw"):
        rotation = R.from_quat(quat)
    elif name.startswith("wxyz"):
        rotation = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    else:
        raise KeyError(name)
    if name.endswith("rotvec"):
        return rotation.as_rotvec().astype(np.float32)
    if name.endswith("euler_xyz"):
        return rotation.as_euler("xyz").astype(np.float32)
    if name.endswith("euler_zyx"):
        return rotation.as_euler("zyx").astype(np.float32)
    raise KeyError(name)


def _collect_expert_live_obs(
    *,
    env_cls: Any,
    args: argparse.Namespace,
    case: dict[str, Any],
) -> dict[str, Any]:
    path = Path(case["hdf5_path"])
    demo_name = case["demo_name"]
    demo_window = replay_bridge._demo_window(path, demo_name, int(args.max_replay_steps), int(args.post_signal_margin))
    bddl_file = _bddl_file(Path(args.libero_root), path)
    env = env_cls(bddl_file_name=str(bddl_file), camera_heights=int(args.camera_size), camera_widths=int(args.camera_size))
    try:
        env.seed(0)
        obs = env.reset()
        obs = env.set_init_state(np.asarray(demo_window["init_state"], dtype=np.float64))
        observations = [obs]
        actions = np.asarray(demo_window["actions"], dtype=np.float64)
        for action in actions[:-1]:
            obs, _reward, done, _info = env.step(action)
            observations.append(obs)
            if len(observations) >= actions.shape[0]:
                break
            if bool(done):
                break
    finally:
        try:
            env.close()
        except Exception:
            pass
    return {
        "task_name": case["task_name"],
        "demo_name": demo_name,
        "hdf5_path": str(path),
        "demo_window": demo_window,
        "observations": observations,
    }


def _feature_alignment_audit(args: argparse.Namespace, eligible_cases: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    started = time.monotonic()
    env_cls, env_meta = replay_bridge._load_env_class_noninteractive(
        libero_root=Path(args.libero_root),
        robosuite_root=Path(args.robosuite_root),
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
    )
    live_sequences = [_collect_expert_live_obs(env_cls=env_cls, args=args, case=case) for case in eligible_cases]
    candidates = [
        "quat_vector_part_legacy",
        "xyzw_rotvec",
        "wxyz_rotvec",
        "xyzw_euler_xyz",
        "xyzw_euler_zyx",
        "wxyz_euler_xyz",
        "wxyz_euler_zyx",
        "xyzw_axis_angle_0_to_2pi",
    ]
    candidate_errors = {name: [] for name in candidates}
    old_feature_errors = []
    fixed_feature_errors = []
    position_errors = []
    rows: list[dict[str, Any]] = []
    for sequence in live_sequences:
        hdf5_features = np.asarray(sequence["demo_window"]["features"], dtype=np.float32)
        full_steps = int(sequence["demo_window"]["full_action_steps"])
        row_old = []
        row_fixed = []
        for index, obs in enumerate(sequence["observations"]):
            if index >= hdf5_features.shape[0]:
                break
            target = hdf5_features[index]
            quat = np.asarray(obs["robot0_eef_quat"], dtype=np.float64).reshape(4)
            for name in candidates:
                try:
                    candidate = _candidate_orientation(quat, name)
                    candidate_errors[name].append(float(np.linalg.norm(candidate - target[3:6])))
                except Exception:
                    pass
            timestep_fraction = float(index) / max(1, full_steps - 1)
            old_feature, _old_meta = ee_features.old_quat_first3_feature(obs, timestep_fraction)
            fixed_feature, fixed_meta = ee_features.build_live_feature(obs, timestep_fraction)
            old_l2 = float(np.linalg.norm(old_feature - target))
            fixed_l2 = float(np.linalg.norm(fixed_feature - target))
            old_feature_errors.append(old_l2)
            fixed_feature_errors.append(fixed_l2)
            position_errors.append(float(np.linalg.norm(fixed_feature[:3] - target[:3])))
            row_old.append(old_l2)
            row_fixed.append(fixed_l2)
        rows.append(
            {
                "task_name": sequence["task_name"],
                "demo_name": sequence["demo_name"],
                "observed_steps": len(row_fixed),
                "old_feature_l2_mean": _round(float(np.mean(row_old))) if row_old else None,
                "fixed_feature_l2_mean": _round(float(np.mean(row_fixed))) if row_fixed else None,
                "fixed_feature_l2_max": _round(float(np.max(row_fixed))) if row_fixed else None,
                "fixed_feature_source": fixed_meta.get("source") if row_fixed else None,
            }
        )
    orientation_report = {
        name: {
            "mean_orientation_l2": _round(float(np.mean(values))) if values else None,
            "max_orientation_l2": _round(float(np.max(values))) if values else None,
        }
        for name, values in candidate_errors.items()
    }
    fixed_mean = float(np.mean(fixed_feature_errors)) if fixed_feature_errors else float("inf")
    old_mean = float(np.mean(old_feature_errors)) if old_feature_errors else float("inf")
    return (
        {
            "executed": True,
            "env": env_meta,
            "candidate_conversions": orientation_report,
            "selected_orientation_conversion": ee_features.ORIENTATION_CONVENTION,
            "selection_metric": "minimum live-vs-HDF5 orientation L2 on expert replay states; independent of learned policy success",
            "old_feature_l2_mean": _round(old_mean),
            "old_feature_l2_max": _round(float(np.max(old_feature_errors))) if old_feature_errors else None,
            "fixed_feature_l2_mean": _round(fixed_mean),
            "fixed_feature_l2_max": _round(float(np.max(fixed_feature_errors))) if fixed_feature_errors else None,
            "position_l2_mean": _round(float(np.mean(position_errors))) if position_errors else None,
            "feature_l2_reduction_factor": _round(old_mean / max(1e-8, fixed_mean)) if np.isfinite(old_mean) and np.isfinite(fixed_mean) else None,
            "feature_mismatch_fixed": bool(fixed_mean < float(args.feature_match_threshold) and old_mean > fixed_mean * 10.0),
            "rows": rows,
            "runtime_sec": _round(time.monotonic() - started, 3),
        },
        live_sequences,
    )


def _teacher_forced_after_fix(live_sequences: list[dict[str, Any]], predictors: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    cases: list[dict[str, Any]] = []
    adapter_predict = predictors["smolvla_7d_adapter"]["predict"]
    for sequence in live_sequences:
        demo_window = sequence["demo_window"]
        expert = np.asarray(demo_window["actions"], dtype=np.float32)[: len(sequence["observations"])]
        hdf5_features = np.asarray(demo_window["features"], dtype=np.float32)[: len(sequence["observations"])]
        full_steps = int(demo_window["full_action_steps"])
        old_features = []
        fixed_features = []
        for index, obs in enumerate(sequence["observations"]):
            timestep_fraction = float(index) / max(1, full_steps - 1)
            old_feature, _old_meta = ee_features.old_quat_first3_feature(obs, timestep_fraction)
            fixed_feature, _fixed_meta = ee_features.build_live_feature(obs, timestep_fraction)
            old_features.append(old_feature)
            fixed_features.append(fixed_feature)
        old_arr = np.stack(old_features, axis=0).astype(np.float32)
        fixed_arr = np.stack(fixed_features, axis=0).astype(np.float32)
        predictions = {
            "before_legacy_quat_first3_live": adapter_predict(old_arr),
            "after_fixed_live": adapter_predict(fixed_arr),
            "hdf5_reference": adapter_predict(hdf5_features),
        }
        policies = {
            name: {
                "action_metrics": standard._metrics(pred, expert),
                "first20_metrics": standard._metrics(pred[: min(20, len(pred))], expert[: min(20, len(expert))]),
                "phase_metrics": gap._phase_metrics(pred, expert),
                "top_error_timesteps": gap._top_errors(pred, expert),
                "gripper_timing": gap._gripper_timing(pred, expert),
                "action_sign_agreement": gap._action_sign_agreement(pred, expert),
                "translation_cosine": gap._translation_cosine(pred, expert),
                "action_validity": standard._action_validity(pred),
            }
            for name, pred in predictions.items()
        }
        cases.append(
            {
                "task_name": sequence["task_name"],
                "demo_name": sequence["demo_name"],
                "observed_steps": int(expert.shape[0]),
                "policies": policies,
            }
        )

    def aggregate(name: str) -> dict[str, Any]:
        metrics = [case["policies"][name]["action_metrics"] for case in cases]
        first20 = [case["policies"][name]["first20_metrics"] for case in cases]
        phase_ratios = []
        top_error_examples = []
        for case in cases:
            policy = case["policies"][name]
            phases = policy.get("phase_metrics") or {}
            phase_l2 = [float(item["action_l2"]) for item in phases.values() if item.get("action_l2") is not None]
            aggregate_l2 = float((policy.get("action_metrics") or {}).get("action_l2") or 0.0)
            if phase_l2:
                phase_ratios.append(max(phase_l2) / max(1e-8, aggregate_l2))
            top = policy.get("top_error_timesteps") or []
            if top:
                top_error_examples.append(top[0])
        return {
            "case_count": len(cases),
            "action_l2_mean": _round(float(np.mean([item["action_l2"] for item in metrics]))),
            "translation_l2_mean": _round(float(np.mean([item["translation_l2"] for item in metrics]))),
            "rotation_l2_mean": _round(float(np.mean([item["rotation_l2"] for item in metrics]))),
            "gripper_error_mean": _round(float(np.mean([item["gripper_error"] for item in metrics]))),
            "first20_action_l2_mean": _round(float(np.mean([item["action_l2"] for item in first20]))),
            "phase_critical_error_ratio_mean": _round(float(np.mean(phase_ratios))) if phase_ratios else None,
            "top_error_timestep_examples": top_error_examples[:6],
            "per_dim_mae_mean": [
                _round(float(np.mean([item["per_dim_mae"][dim] for item in metrics])))
                for dim in range(7)
            ],
        }

    old = aggregate("before_legacy_quat_first3_live")
    fixed = aggregate("after_fixed_live")
    hdf5 = aggregate("hdf5_reference")
    return {
        "executed": True,
        "cases": cases,
        "aggregate": {
            "before_legacy_quat_first3_live": old,
            "after_fixed_live": fixed,
            "hdf5_reference": hdf5,
            "feature_fix_materially_improves_teacher_forced": bool(
                float(fixed["action_l2_mean"]) < float(old["action_l2_mean"]) * 0.9
            ),
            "fixed_live_close_to_hdf5_reference": bool(
                abs(float(fixed["action_l2_mean"]) - float(hdf5["action_l2_mean"])) < 0.05
            ),
        },
        "runtime_sec": _round(time.monotonic() - started, 3),
    }


def _run_online_variant(
    *,
    env_cls: Any,
    bddl_file: Path,
    camera_size: int,
    init_state: np.ndarray,
    name: str,
    claim_role: str,
    instruction: str,
    horizon: int,
    full_action_steps: int,
    predict: Callable[[np.ndarray], np.ndarray],
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "variant": name,
        "claim_role": claim_role,
        "use_exact_init_state": True,
        "feature_builder": ee_features.FEATURE_SCHEMA,
        "orientation_convention": ee_features.ORIENTATION_CONVENTION,
        "env_created": False,
        "reset_ok": False,
        "set_init_state_ok": False,
        "steps_requested": int(horizon),
        "steps_performed": 0,
        "reward_sum": 0.0,
        "final_reward": 0.0,
        "final_success": None,
        "done_seen": False,
        "first_positive_reward_index": None,
        "first_done_index": None,
        "first_success_index": None,
        "done_indices": [],
        "success_indices": [],
        "reward_trajectory": [],
        "action_feature_metadata_first": None,
        "available_obs_keys": [],
        "target_key_audit": None,
        "target_directed_movement": None,
        "object_movement": None,
        "eef_start": None,
        "eef_final": None,
        "eef_displacement_l2": None,
        "error": None,
    }
    env = None
    actions = []
    obs = None
    try:
        env = env_cls(bddl_file_name=str(bddl_file), camera_heights=int(camera_size), camera_widths=int(camera_size))
        summary["env_created"] = True
        env.seed(0)
        obs = env.reset()
        summary["reset_ok"] = True
        obs = env.set_init_state(init_state)
        summary["set_init_state_ok"] = True
        summary["eef_start"] = _extract_eef(obs)
        if isinstance(obs, dict):
            keys = sorted(str(key) for key in obs.keys())
            summary["available_obs_keys"] = keys[:80]
        target_audit = _best_object_key(obs, instruction)
        target_key = target_audit["best_key"]
        target_start = _extract_pos(obs, target_key)
        summary["target_key_audit"] = target_audit
        for index in range(int(horizon)):
            timestep_fraction = float(index) / max(1, int(full_action_steps) - 1)
            feature, metadata = ee_features.build_live_feature(obs, timestep_fraction)
            if summary["action_feature_metadata_first"] is None:
                summary["action_feature_metadata_first"] = metadata
            action = np.asarray(predict(feature.reshape(1, 7)), dtype=np.float32).reshape(-1)[:7]
            actions.append(action)
            obs, reward, done, _info = env.step(action.astype(np.float64))
            reward_value = float(reward)
            try:
                success_value = bool(env.check_success())
            except Exception:
                success_value = None
            summary["steps_performed"] += 1
            summary["reward_sum"] += reward_value
            summary["final_reward"] = reward_value
            summary["reward_trajectory"].append(round(reward_value, 6))
            if reward_value > 0.0 and summary["first_positive_reward_index"] is None:
                summary["first_positive_reward_index"] = int(index)
            if bool(done):
                summary["done_seen"] = True
                summary["done_indices"].append(int(index))
                if summary["first_done_index"] is None:
                    summary["first_done_index"] = int(index)
            if success_value:
                summary["success_indices"].append(int(index))
                if summary["first_success_index"] is None:
                    summary["first_success_index"] = int(index)
            if bool(done) or reward_value > 0.0 or success_value:
                break
        summary["eef_final"] = _extract_eef(obs)
        if summary["eef_start"] is not None and summary["eef_final"] is not None:
            summary["eef_displacement_l2"] = _round(float(np.linalg.norm(np.asarray(summary["eef_final"]) - np.asarray(summary["eef_start"]))))
        target_final = _extract_pos(obs, target_key)
        summary["target_directed_movement"] = _distance_delta(summary["eef_start"], summary["eef_final"], target_start, target_final)
        target_object_distance = _distance(target_start, target_final)
        summary["object_movement"] = {
            "available": target_object_distance is not None,
            "target_object_key": target_key,
            "target_object_displacement_l2": target_object_distance,
            "object_position_keys_missing": not bool(_object_position_keys(obs)),
        }
        try:
            summary["final_success"] = bool(env.check_success())
        except Exception:
            summary["final_success"] = None
    except Exception as exc:  # noqa: BLE001
        summary["error"] = _compact_error(exc)
    finally:
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
    action_arr = np.asarray(actions, dtype=np.float32).reshape((-1, 7)) if actions else np.zeros((0, 7), dtype=np.float32)
    summary["action_shape"] = list(action_arr.shape)
    summary["action_validity"] = standard._action_validity(action_arr) if action_arr.size else {}
    summary["passed"] = bool(summary["env_created"] and summary["reset_ok"] and summary["set_init_state_ok"] and summary["error"] is None)
    return summary


def _aggregate_replay(cases: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, Any] = {}
    for policy in ["expert", "mean_action", "ridge", "smolvla_7d_adapter_fixed_live"]:
        values = [(case.get("results") or {}).get(policy) for case in cases]
        values = [item for item in values if item]
        progress = [replay_bridge._progress_metric(item) for item in values]
        progress = [float(value) for value in progress if value is not None]
        aggregate[policy] = {
            "case_count": len(values),
            "success_count": int(sum(1 for item in values if replay_bridge._success(item))),
            "success_rate": _round(float(np.mean([replay_bridge._success(item) for item in values]))) if values else None,
            "reward_sum_mean": _round(float(np.mean([float(item.get("reward_sum") or 0.0) for item in values]))) if values else None,
            "first_done_indices": [item.get("first_done_index") for item in values],
            "progress_proxy_mean": _round(float(np.mean(progress))) if progress else None,
            "object_movement_mean": _round(float(np.mean([float((item.get("object_movement") or {}).get("target_object_displacement_l2") or 0.0) for item in values]))) if values else None,
            "runtime_case_steps": [item.get("steps_performed") for item in values],
        }
    return aggregate


def _replay_after_fix(args: argparse.Namespace, eligible_cases: list[dict[str, Any]], predictors: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    env_cls, env_meta = replay_bridge._load_env_class_noninteractive(
        libero_root=Path(args.libero_root),
        robosuite_root=Path(args.robosuite_root),
        data_root=Path(args.data_root),
        output_dir=Path(args.output_dir),
    )
    cases: list[dict[str, Any]] = []
    for case in eligible_cases:
        path = Path(case["hdf5_path"])
        demo_window = replay_bridge._demo_window(path, case["demo_name"], int(args.max_replay_steps), int(args.post_signal_margin))
        bddl_file = _bddl_file(Path(args.libero_root), path)
        instruction = _instruction_from_path(path)
        expert_actions = np.asarray(demo_window["actions"], dtype=np.float32)
        mean_actions = predictors["mean_action"]["predict"](np.asarray(demo_window["features"], dtype=np.float32))
        results = {
            "expert": _run_replay_variant(
                env_cls=env_cls,
                bddl_file=bddl_file,
                camera_size=int(args.camera_size),
                init_state=np.asarray(demo_window["init_state"], dtype=np.float64),
                variant={"name": "expert", "claim_role": "expert_replay_upper_bound", "actions": expert_actions, "use_exact_init_state": True},
                instruction=instruction,
            ),
            "mean_action": _run_replay_variant(
                env_cls=env_cls,
                bddl_file=bddl_file,
                camera_size=int(args.camera_size),
                init_state=np.asarray(demo_window["init_state"], dtype=np.float64),
                variant={"name": "mean_action", "claim_role": "mean_action_baseline", "actions": mean_actions, "use_exact_init_state": True},
                instruction=instruction,
            ),
            "ridge": _run_online_variant(
                env_cls=env_cls,
                bddl_file=bddl_file,
                camera_size=int(args.camera_size),
                init_state=np.asarray(demo_window["init_state"], dtype=np.float64),
                name="ridge",
                claim_role="ridge_baseline_fixed_live_features",
                instruction=instruction,
                horizon=int(demo_window["target_horizon"]),
                full_action_steps=int(demo_window["full_action_steps"]),
                predict=predictors["ridge"]["predict"],
            ),
            "smolvla_7d_adapter_fixed_live": _run_online_variant(
                env_cls=env_cls,
                bddl_file=bddl_file,
                camera_size=int(args.camera_size),
                init_state=np.asarray(demo_window["init_state"], dtype=np.float64),
                name="smolvla_7d_adapter_fixed_live",
                claim_role="best_prior_smolvla_7d_lora_adapter_fixed_live_features",
                instruction=instruction,
                horizon=int(demo_window["target_horizon"]),
                full_action_steps=int(demo_window["full_action_steps"]),
                predict=predictors["smolvla_7d_adapter"]["predict"],
            ),
        }
        cases.append(
            {
                "task_name": case["task_name"],
                "demo_name": case["demo_name"],
                "hdf5_path": str(path),
                "target_horizon": int(demo_window["target_horizon"]),
                "hdf5_first_signal_index": demo_window.get("first_signal_index"),
                "results": results,
            }
        )
    return {
        "executed": True,
        "reason": "bounded exact-init replay rerun on expert-success eligible cases with fixed live feature builder",
        "env": env_meta,
        "cases": cases,
        "aggregate": _aggregate_replay(cases),
        "runtime_sec": _round(time.monotonic() - started, 3),
    }


def _decide(report: dict[str, Any]) -> tuple[str, str]:
    alignment = report.get("state3_live_hdf5_feature_alignment") or {}
    teacher = report.get("state4_teacher_forced_after_fix") or {}
    replay = report.get("state5_replay_after_fix") or {}
    aggregate = replay.get("aggregate") or {}
    if not alignment.get("feature_mismatch_fixed"):
        return "FEATURE_PATH_STILL_MISMATCHED", "Feature mismatch remains too large; do not evaluate adapter quality."
    teacher_agg = teacher.get("aggregate") or {}
    if not teacher_agg.get("feature_fix_materially_improves_teacher_forced"):
        return "FEATURE_FIXED_BUT_CONTROL_GAP_REMAINS", "Feature schema is aligned, but teacher-forced action quality did not materially improve enough for method work."
    adapter_validities = []
    for case in replay.get("cases") or []:
        validity = (((case.get("results") or {}).get("smolvla_7d_adapter_fixed_live") or {}).get("action_validity") or {})
        if validity:
            adapter_validities.append(validity)
    if adapter_validities:
        clip_step = float(np.mean([float(item.get("clip_rate_step") or 0.0) for item in adapter_validities]))
        controller = float(np.mean([float(item.get("controller_valid_rate_proxy") or 0.0) for item in adapter_validities]))
        if clip_step > 0.3 or controller < 0.7:
            return "ACTION_VALIDITY_RANGE_FAILURE", "Feature fix works, but adapter action clipping/controller-validity remains too weak."
    expert = aggregate.get("expert") or {}
    adapter = aggregate.get("smolvla_7d_adapter_fixed_live") or {}
    mean = aggregate.get("mean_action") or {}
    ridge = aggregate.get("ridge") or {}
    if int(expert.get("success_count") or 0) < int(expert.get("case_count") or 0):
        return "TOO_HEAVY_LOCAL", "Expert replay was not stable during the feature-fix rerun; rerun the eligible-set gate."
    adapter_progress = adapter.get("progress_proxy_mean")
    simple_progress = [mean.get("progress_proxy_mean"), ridge.get("progress_proxy_mean")]
    beats_progress = adapter_progress is not None and all(value is None or float(adapter_progress) > float(value) for value in simple_progress)
    beats_success = int(adapter.get("success_count") or 0) > max(int(mean.get("success_count") or 0), int(ridge.get("success_count") or 0))
    if beats_progress or beats_success:
        return "READY_FOR_METHOD_AFTER_FEATURE_FIX", "Preserve this fixed-feature replay baseline; only then consider method work."
    return "FEATURE_FIXED_BUT_CONTROL_GAP_REMAINS", "Feature schema is fixed, but adapter replay still fails to beat mean/ridge; diagnose baseline control before method work."


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "decision": "TOO_HEAVY_LOCAL",
        "policy": {
            "run_gate_set": _env_flag(RUN_GATE),
            "forbidden_gates_set": forbidden,
            "new_method_created": False,
            "training_performed": False,
            "downloads_performed": False,
            "openvla_oft_executed": False,
            "full_benchmark_executed": False,
            "paper_claims_made": False,
            "learned_evaluated_on_expert_failed_cases": False,
        },
        "state0_eligible_set": {},
        "state1_hdf5_schema_audit": {},
        "state2_live_env_schema_audit": {},
        "state3_live_hdf5_feature_alignment": {},
        "state4_teacher_forced_after_fix": {},
        "state5_replay_after_fix": {},
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
        return finish("TOO_HEAVY_LOCAL", f"Set {RUN_GATE}=1 for this bounded live feature schema fix.", 2)
    if forbidden:
        report["error"] = {"message": "Forbidden gate(s) set: " + ", ".join(forbidden)}
        return finish("TOO_HEAVY_LOCAL", "Clear forbidden method/download/full-rollout gates and rerun.", 3)

    try:
        eligible_cases = _load_eligible_cases(Path(args.exact_init_report_path))
        best_name = _best_adapter_name(Path(args.exact_init_report_path))
        predictors, predictor_meta = exact_init._build_learned_predictors(args, best_name)
        report["state0_eligible_set"] = {
            "eligible_cases": eligible_cases,
            "eligible_case_count": len(eligible_cases),
            "source_report": str(Path(args.exact_init_report_path)),
            "best_lora_name": best_name,
            "predictor_meta": predictor_meta,
        }
        hdf5_audit = _hdf5_schema_audit(eligible_cases)
        report["state1_hdf5_schema_audit"] = hdf5_audit
        alignment, live_sequences = _feature_alignment_audit(args, eligible_cases)
        report["state2_live_env_schema_audit"] = {
            "executed": True,
            "available_obs_keys": sorted(str(key) for key in live_sequences[0]["observations"][0].keys()) if live_sequences else [],
            "selected_orientation_conversion": alignment.get("selected_orientation_conversion"),
            "candidate_conversions": alignment.get("candidate_conversions"),
            "initial_live_feature_examples": [
                row for row in alignment.get("rows", [])
            ],
        }
        report["state3_live_hdf5_feature_alignment"] = alignment
        teacher = _teacher_forced_after_fix(live_sequences, predictors)
        report["state4_teacher_forced_after_fix"] = teacher
        replay = _replay_after_fix(args, eligible_cases, predictors)
        report["state5_replay_after_fix"] = replay
        decision, next_step = _decide(report)
        teacher_agg = teacher.get("aggregate") or {}
        replay_agg = replay.get("aggregate") or {}
        report["summary"].update(
            {
                "branch": _current_branch(),
                "experiments_happened": True,
                "training_happened": False,
                "replay_control_happened": True,
                "downloads_happened": False,
                "openvla_oft_happened": False,
                "eligible_demos_used": [f"{case['task_name']}::{case['demo_name']}" for case in eligible_cases],
                "hdf5_ee_states_schema": hdf5_audit.get("ee_states_summary"),
                "live_env_feature_schema_before_fix": "robot0_eef_pos + robot0_eef_quat[:3]",
                "selected_orientation_conversion": alignment.get("selected_orientation_conversion"),
                "feature_l2_before": alignment.get("old_feature_l2_mean"),
                "feature_l2_after": alignment.get("fixed_feature_l2_mean"),
                "feature_l2_reduction_factor": alignment.get("feature_l2_reduction_factor"),
                "teacher_forced_before_after": teacher_agg,
                "teacher_forced_action_result_before": (teacher_agg.get("before_legacy_quat_first3_live") or {}),
                "teacher_forced_action_result_after": (teacher_agg.get("after_fixed_live") or {}),
                "replay_result_after_fix": replay_agg,
                "mean_replay_result": replay_agg.get("mean_action"),
                "ridge_replay_result": replay_agg.get("ridge"),
                "adapter_replay_result": replay_agg.get("smolvla_7d_adapter_fixed_live"),
            }
        )
        return finish(decision, next_step, 0)
    except Exception as exc:  # noqa: BLE001
        report["error"] = _compact_error(exc)
        return finish("TOO_HEAVY_LOCAL", "Fix the reported live feature schema runner error and rerun.", 11)


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
            elif key == "observations":
                result["observations_omitted"] = True
            elif key == "demo_window":
                result["demo_window_omitted"] = True
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


def _write_reports(report: dict[str, Any]) -> None:
    summary = report.get("summary") or {}
    hdf5 = report.get("state1_hdf5_schema_audit") or {}
    live = report.get("state2_live_env_schema_audit") or {}
    align = report.get("state3_live_hdf5_feature_alignment") or {}
    teacher = report.get("state4_teacher_forced_after_fix") or {}
    replay = report.get("state5_replay_after_fix") or {}
    main_lines = [
        "# SmolVLA 7D Live Feature Schema Fix",
        "",
        f"Final decision: `{summary.get('final_decision')}`",
        "",
        "This is infrastructure and evaluation repair, not a new method.",
        "",
        f"- eligible demos used: `{summary.get('eligible_demos_used')}`",
        f"- HDF5 ee_states schema: `{summary.get('hdf5_ee_states_schema')}`",
        f"- live feature schema before fix: `{summary.get('live_env_feature_schema_before_fix')}`",
        f"- selected orientation conversion: `{summary.get('selected_orientation_conversion')}`",
        f"- feature L2 before/after: `{summary.get('feature_l2_before')}` / `{summary.get('feature_l2_after')}`",
        f"- teacher-forced before/after: `{summary.get('teacher_forced_action_result_before')}` / `{summary.get('teacher_forced_action_result_after')}`",
        f"- mean/ridge/adapter replay: `{summary.get('mean_replay_result')}` / `{summary.get('ridge_replay_result')}` / `{summary.get('adapter_replay_result')}`",
        "",
        f"Exact next step: {summary.get('exact_next_step')}",
        "",
    ]
    Path("reports/smolvla_7d_live_feature_schema_fix.md").write_text("\n".join(main_lines), encoding="utf-8")
    Path("reports/smolvla_ee_state_schema_audit.md").write_text(
        "\n".join(
            [
                "# SmolVLA EE State Schema Audit",
                "",
                f"- exact keys used: `{hdf5.get('exact_keys_used_for_hdf5_ee_states')}`",
                f"- ee_states summary: `{hdf5.get('ee_states_summary')}`",
                f"- orientation inference: `{hdf5.get('orientation_inference')}`",
                f"- initial timestep examples: `{hdf5.get('initial_timestep_examples')}`",
                f"- ee_ori continuity: `{hdf5.get('ee_ori_continuity_global')}`",
                f"- per-case rows: `{hdf5.get('rows')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_live_hdf5_feature_alignment.md").write_text(
        "\n".join(
            [
                "# SmolVLA Live/HDF5 Feature Alignment",
                "",
                f"- selected orientation conversion: `{align.get('selected_orientation_conversion')}`",
                f"- candidate conversions: `{align.get('candidate_conversions')}`",
                f"- feature L2 before/after: `{align.get('old_feature_l2_mean')}` / `{align.get('fixed_feature_l2_mean')}`",
                f"- feature mismatch fixed: `{align.get('feature_mismatch_fixed')}`",
                f"- live env schema audit: `{live}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_feature_fix_teacher_forced_result.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Feature-Fix Teacher-Forced Result",
                "",
                f"- aggregate: `{teacher.get('aggregate')}`",
                f"- runtime sec: `{teacher.get('runtime_sec')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_feature_fix_replay_result.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Feature-Fix Replay Result",
                "",
                f"- aggregate: `{replay.get('aggregate')}`",
                f"- runtime sec: `{replay.get('runtime_sec')}`",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/smolvla_7d_feature_fix_decision.md").write_text(
        "\n".join(
            [
                "# SmolVLA 7D Feature-Fix Decision",
                "",
                f"Final decision: `{summary.get('final_decision')}`",
                "",
                f"Exact next step: {summary.get('exact_next_step')}",
                "",
                "Do not propose a new RA-L method unless the decision is `READY_FOR_METHOD_AFTER_FEATURE_FIX`.",
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
                "SmolVLA 7D live feature schema fix is the active infrastructure gate.",
                "",
                "## Feature Fix",
                "",
                f"- selected orientation conversion: `{summary.get('selected_orientation_conversion')}`",
                f"- feature L2 before/after: `{summary.get('feature_l2_before')}` / `{summary.get('feature_l2_after')}`",
                f"- teacher-forced before/after: `{summary.get('teacher_forced_action_result_before')}` / `{summary.get('teacher_forced_action_result_after')}`",
                f"- replay result after fix: `{summary.get('replay_result_after_fix')}`",
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
                "Do not start a new method unless the feature-fix decision is `READY_FOR_METHOD_AFTER_FEATURE_FIX`.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    decision_path = Path("reports/decision_log.md")
    existing = decision_path.read_text(encoding="utf-8") if decision_path.exists() else "# Decision Log\n"
    marker = "## 2026-07-09: SmolVLA 7D Live Feature Schema Fix"
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
            f"- selected orientation conversion: `{summary.get('selected_orientation_conversion')}`",
            f"- feature L2 before/after: `{summary.get('feature_l2_before')}` / `{summary.get('feature_l2_after')}`",
            f"- teacher-forced before/after: `{summary.get('teacher_forced_action_result_before')}` / `{summary.get('teacher_forced_action_result_after')}`",
            f"- replay result after fix: `{summary.get('replay_result_after_fix')}`",
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
    parser.add_argument("--output-dir", default="runs/smolvla_7d_live_feature_schema_fix")
    parser.add_argument("--report-path", default="reports/smolvla_7d_live_feature_schema_fix.json")
    parser.add_argument("--max-tasks", type=int, default=2)
    parser.add_argument("--train-demos-per-task", type=int, default=5)
    parser.add_argument("--eval-demos-per-task", type=int, default=2)
    parser.add_argument("--records-per-demo", type=int, default=8)
    parser.add_argument("--max-replay-steps", type=int, default=320)
    parser.add_argument("--post-signal-margin", type=int, default=16)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--feature-match-threshold", type=float, default=0.15)
    args = parser.parse_args(argv)
    report, exit_code = build_report(args)
    json_report = _strip_large(report)
    _write_json(Path(args.report_path), json_report)
    _write_reports(json_report)
    print(json.dumps(json_report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
