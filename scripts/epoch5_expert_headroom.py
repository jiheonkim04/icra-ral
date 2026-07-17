"""Generic Epoch 5 LIBERO expert-headroom diagnostic.

This bounded simulator diagnostic checks whether an official LIBERO task has
recoverable expert-action headroom after a Base/Prior residual is found. It
does not train, download, load a VLA model, run optimizer steps, write
checkpoints, design Ours, or make a benchmark claim. It replays local HDF5
expert actions only.
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

from tca_map.datasets.libero_fixed_prior_rollout_diagnostic import _action_stats
from tca_map.datasets.libero_full_demo_expert_replay_sanity import _run_replay_variant

DEFAULT_TASK_SUITE = "libero_10"
DEFAULT_TASK_ID = 6
DEFAULT_TASK_DESCRIPTION = "put the white mug on the plate and put the chocolate pudding to the right of the plate"
DEFAULT_RESET_IDENTITY = 20260725
DEFAULT_IDENTITY_BASE = 20260711
DEFAULT_HDF5_PATH = Path(
    "/mnt/c/assets/data/libero/libero_10/"
    "LIVING_ROOM_SCENE6_put_the_white_mug_on_the_plate_and_put_the_chocolate_pudding_to_the_right_of_the_plate_demo.hdf5"
)
SCHEMA_VERSION = "2026-07-17.epoch5_generic_expert_headroom.v1"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _sha256_array(value: Any) -> str:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return hashlib.sha256(arr.tobytes()).hexdigest()


def _first_index(values: np.ndarray, threshold: float) -> int | None:
    for index, value in enumerate(np.asarray(values, dtype=np.float64).reshape(-1)):
        if float(value) > threshold:
            return int(index)
    return None


def _demo_sort_key(name: str) -> tuple[int, str]:
    prefix, _, suffix = str(name).rpartition("_")
    if prefix == "demo" and suffix.isdigit():
        return (int(suffix), str(name))
    return (10**9, str(name))


def _safe_l2(left: Any, right: Any) -> float | None:
    left_arr = np.asarray(left, dtype=np.float64).reshape(-1)
    right_arr = np.asarray(right, dtype=np.float64).reshape(-1)
    if left_arr.size != right_arr.size or left_arr.size == 0:
        return None
    return round(float(np.linalg.norm(left_arr - right_arr)), 9)


def _read_demo(path: Path, demo_name: str) -> dict[str, Any]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        demo = handle["data"][demo_name]
        actions = np.asarray(demo["actions"], dtype=np.float64)
        rewards = np.asarray(demo["rewards"], dtype=np.float64).reshape(-1) if "rewards" in demo else np.zeros(actions.shape[0])
        dones = np.asarray(demo["dones"], dtype=np.float64).reshape(-1) if "dones" in demo else np.zeros(actions.shape[0])
        init_state = np.asarray(demo.attrs["init_state"], dtype=np.float64).reshape(-1)
        num_samples = int(demo.attrs.get("num_samples", actions.shape[0]))
    if actions.ndim != 2 or actions.shape[1] != 7:
        raise ValueError(f"{path}::{demo_name} actions must be [T, 7], got {list(actions.shape)}")
    first_reward = _first_index(rewards, 0.0)
    first_done = _first_index(dones, 0.5)
    signals = [index for index in (first_reward, first_done) if index is not None]
    first_signal = min(signals) if signals else None
    return {
        "demo_name": demo_name,
        "init_state": init_state,
        "actions": actions,
        "rewards": rewards,
        "dones": dones,
        "num_samples_attr": num_samples,
        "steps": int(actions.shape[0]),
        "first_reward_index": first_reward,
        "first_done_index": first_done,
        "first_signal_index": first_signal,
        "init_state_sha256": _sha256_array(init_state),
        "action_stats": _action_stats(actions),
    }


def _scan_hdf5(path: Path, benchmark_init_state: np.ndarray) -> list[dict[str, Any]]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        data = handle.get("data")
        if data is None:
            raise ValueError(f"{path} has no data group")
        names = sorted([str(name) for name in data.keys()], key=_demo_sort_key)
    rows: list[dict[str, Any]] = []
    for name in names:
        demo = _read_demo(path, name)
        rows.append(
            {
                "demo_name": name,
                "steps": demo["steps"],
                "num_samples_attr": demo["num_samples_attr"],
                "first_reward_index": demo["first_reward_index"],
                "first_done_index": demo["first_done_index"],
                "first_signal_index": demo["first_signal_index"],
                "init_state_sha256": demo["init_state_sha256"],
                "l2_to_benchmark_residual_init": _safe_l2(demo["init_state"], benchmark_init_state),
            }
        )
    return rows


def _expert_success(row: dict[str, Any]) -> bool:
    return bool(row.get("final_success") or row.get("done_seen") or float(row.get("reward_sum") or 0.0) > 0.0)


def _select_demo(rows: list[dict[str, Any]], target_sha256: str, initial_state_index: int) -> tuple[str, str]:
    matches = [row for row in rows if row.get("init_state_sha256") == target_sha256]
    if matches:
        return str(matches[0]["demo_name"]), "same_reset_init_state_hash_match"
    finite = [row for row in rows if row.get("l2_to_benchmark_residual_init") is not None]
    if finite:
        best = min(finite, key=lambda row: float(row["l2_to_benchmark_residual_init"]))
        return str(best["demo_name"]), "nearest_hdf5_demo_init_state_by_l2_no_hash_match"
    fallback_name = f"demo_{int(initial_state_index)}"
    if any(row.get("demo_name") == fallback_name for row in rows):
        return fallback_name, "index_aligned_demo_fallback_no_hash_match"
    return str(rows[0]["demo_name"]), "first_demo_fallback_no_hash_match"


def run_diagnostic(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    run_dir = Path(args.run_dir)
    hdf5_path = Path(args.hdf5_path)
    initial_state_index = int(args.reset_identity) - int(args.identity_base)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": str(args.stage),
        "task_suite": str(args.task_suite),
        "task_id": int(args.task_id),
        "task_description": str(args.task_description),
        "residual_target": {
            "reset_identity": int(args.reset_identity),
            "identity_base": int(args.identity_base),
            "initial_state_index": int(initial_state_index),
            "expected_initial_state_sha256": str(args.expected_initial_state_sha256 or ""),
            "source": str(args.source),
            "xvla_failed": bool(args.xvla_failed),
            "smolvla_base_failed": bool(args.smolvla_base_failed),
        },
        "inputs": {
            "hdf5_path": str(hdf5_path),
            "camera_size": int(args.camera_size),
        },
        "policy": {
            "bounded_expert_replay_diagnostic": True,
            "training_happened": False,
            "optimizer_step_happened": False,
            "checkpoint_written": False,
            "downloads_performed": False,
            "vla_model_loaded": False,
            "learned_policy_inference_performed": False,
            "ours_design_happened": False,
            "benchmark_claim_made": False,
            "paper_grade_claim_made": False,
        },
        "hdf5_scan": {},
        "selected_demo": {},
        "variants": [],
    }

    try:
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[str(args.task_suite)]()
        task = suite.get_task(int(args.task_id))
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        benchmark_initial_states = suite.get_task_init_states(int(args.task_id))
        benchmark_init_state = np.asarray(benchmark_initial_states[int(initial_state_index)], dtype=np.float64).reshape(-1)
        computed_sha = _sha256_array(benchmark_init_state)
        expected = str(args.expected_initial_state_sha256 or computed_sha)
        rows = _scan_hdf5(hdf5_path, benchmark_init_state)
        matches = [row for row in rows if row.get("init_state_sha256") == computed_sha]
        selected_name, selection_reason = _select_demo(rows, computed_sha, initial_state_index)
        selected = _read_demo(hdf5_path, selected_name)

        report["residual_target"]["computed_initial_state_sha256"] = computed_sha
        report["residual_target"]["computed_hash_matches_expected"] = bool(computed_sha == expected)
        report["inputs"]["bddl_file"] = str(bddl_file)
        finite_rows = [row for row in rows if row.get("l2_to_benchmark_residual_init") is not None]
        report["hdf5_scan"] = {
            "demo_count": len(rows),
            "same_reset_demo_match_count": len(matches),
            "same_reset_demo_matches": matches,
            "all_demo_summaries": rows,
            "nearest_demo_to_benchmark_residual_init": min(
                finite_rows,
                key=lambda row: float(row["l2_to_benchmark_residual_init"]),
                default=None,
            ),
        }
        report["selected_demo"] = {
            "demo_name": selected_name,
            "selection_reason": selection_reason,
            "init_state_sha256": selected["init_state_sha256"],
            "l2_to_benchmark_residual_init": next(
                row.get("l2_to_benchmark_residual_init") for row in rows if row.get("demo_name") == selected_name
            ),
            "steps": selected["steps"],
            "first_reward_index": selected["first_reward_index"],
            "first_done_index": selected["first_done_index"],
            "first_signal_index": selected["first_signal_index"],
            "action_stats": selected["action_stats"],
        }

        actions = np.asarray(selected["actions"], dtype=np.float64)
        variants = [
            {
                "name": "zero_action_exact_selected_demo_init",
                "claim_role": "negative_control",
                "actions": np.zeros_like(actions),
                "use_exact_init_state": True,
            },
            {
                "name": "hdf5_expert_replay_exact_selected_demo_init",
                "claim_role": "task_level_expert_headroom",
                "actions": actions,
                "use_exact_init_state": True,
            },
            {
                "name": "hdf5_expert_replay_default_reset",
                "claim_role": "init_state_control",
                "actions": actions,
                "use_exact_init_state": False,
            },
        ]
        for variant in variants:
            result = _run_replay_variant(
                env_cls=OffScreenRenderEnv,
                bddl_file=bddl_file,
                camera_size=int(args.camera_size),
                init_state=np.asarray(selected["init_state"], dtype=np.float64),
                variant=variant,
                instruction=str(args.task_description),
            )
            report["variants"].append(result)

        exact = next(row for row in report["variants"] if row.get("variant") == "hdf5_expert_replay_exact_selected_demo_init")
        zero = next(row for row in report["variants"] if row.get("variant") == "zero_action_exact_selected_demo_init")
        default = next(row for row in report["variants"] if row.get("variant") == "hdf5_expert_replay_default_reset")
        task_level_ok = _expert_success(exact)
        same_reset_available = bool(matches)
        same_reset_headroom = bool(same_reset_available and task_level_ok)
        prefix = f"TASK{int(args.task_id)}"
        report["decision"] = (
            f"{prefix}_SAME_RESET_EXPERT_HEADROOM_POSITIVE"
            if same_reset_headroom
            else (
                f"{prefix}_TASK_LEVEL_EXPERT_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE"
                if task_level_ok and not same_reset_available
                else f"{prefix}_EXPERT_HEADROOM_NOT_VERIFIED"
            )
        )
        report["result"] = {
            "same_benchmark_reset_as_residual_failure": same_reset_available,
            "same_reset_headroom_available": same_reset_headroom,
            "task_level_expert_headroom_positive": task_level_ok,
            "zero_action_succeeded": _expert_success(zero),
            "default_reset_expert_replay_succeeded": _expert_success(default),
            "exact_replay_success": task_level_ok,
            "exact_replay_first_positive_reward_index": exact.get("first_positive_reward_index"),
            "exact_replay_first_done_index": exact.get("first_done_index"),
            "exact_replay_first_success_index": exact.get("first_success_index"),
            "exact_replay_reward_sum": round(float(exact.get("reward_sum") or 0.0), 6),
            "after_set_state_l2_to_selected_hdf5_init": exact.get("after_set_state_l2_to_hdf5_init"),
            "reason_same_reset_unavailable": None
            if same_reset_available
            else f"No task-{int(args.task_id)} HDF5 demo init_state SHA-256 matched the residual benchmark initial-state SHA-256.",
            "upper_headroom_indicates_recoverability": (
                "SAME_RESET_POSITIVE"
                if same_reset_headroom
                else (
                    "TASK_LEVEL_POSITIVE_SAME_RESET_UNAVAILABLE"
                    if task_level_ok and not same_reset_available
                    else "NOT_VERIFIED"
                )
            ),
        }
        report["completed"] = True
    except Exception as exc:  # pragma: no cover - simulator boundary
        report["completed"] = False
        report["decision"] = f"TASK{int(args.task_id)}_EXPERT_HEADROOM_INFRASTRUCTURE_BLOCKED"
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback_tail": traceback.format_exc().splitlines()[-60:],
        }
    finally:
        report["elapsed_seconds"] = round(float(time.monotonic() - started), 3)
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        _write_json(run_dir / "result.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--task-suite", default=DEFAULT_TASK_SUITE)
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--task-description", default=DEFAULT_TASK_DESCRIPTION)
    parser.add_argument("--reset-identity", type=int, default=DEFAULT_RESET_IDENTITY)
    parser.add_argument("--identity-base", type=int, default=DEFAULT_IDENTITY_BASE)
    parser.add_argument("--expected-initial-state-sha256", default="")
    parser.add_argument("--hdf5-path", type=Path, default=DEFAULT_HDF5_PATH)
    parser.add_argument("--camera-size", type=int, default=64)
    parser.add_argument("--stage", default="epoch_5_xvla_task6_headroom")
    parser.add_argument("--source", default="matched X-VLA/SmolVLA-base shared failure")
    parser.add_argument("--xvla-failed", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--smolvla-base-failed", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    report = run_diagnostic(args)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
