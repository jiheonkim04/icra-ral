"""Epoch 5 task-1 expert headroom diagnostic.

This is a bounded simulator diagnostic for the X-VLA task-1 residual found
after the official-prior-first reset. It does not train, download, load a VLA
model, run optimizer steps, or write checkpoints. Its only simulator use is
local LIBERO HDF5 expert action replay.
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

SCHEMA_VERSION = "2026-07-17.epoch5_task1_expert_headroom.v1"
TASK_SUITE = "libero_10"
TASK_ID = 1
TASK_DESCRIPTION = "put both the cream cheese box and the butter in the basket"
RESIDUAL_RESET_IDENTITY = 20260727
IDENTITY_BASE = 20260711
RESIDUAL_INITIAL_STATE_INDEX = RESIDUAL_RESET_IDENTITY - IDENTITY_BASE
EXPECTED_RESIDUAL_INIT_SHA256 = "bb8073f96294281b7008501d0b6ebdec3668f90448421c5937b58f57c1b8c5e2"
DEFAULT_HDF5_PATH = Path(
    "/mnt/c/assets/data/libero/libero_10/"
    "LIVING_ROOM_SCENE2_put_both_the_cream_cheese_box_and_the_butter_in_the_basket_demo.hdf5"
)


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


def _select_demo(rows: list[dict[str, Any]], target_sha256: str) -> tuple[str, str]:
    matches = [row for row in rows if row.get("init_state_sha256") == target_sha256]
    if matches:
        return str(matches[0]["demo_name"]), "same_reset_init_state_hash_match"
    finite = [row for row in rows if row.get("l2_to_benchmark_residual_init") is not None]
    if finite:
        best = min(finite, key=lambda row: float(row["l2_to_benchmark_residual_init"]))
        return str(best["demo_name"]), "nearest_hdf5_demo_init_state_by_l2_no_hash_match"
    if any(row.get("demo_name") == f"demo_{RESIDUAL_INITIAL_STATE_INDEX}" for row in rows):
        return f"demo_{RESIDUAL_INITIAL_STATE_INDEX}", "index_aligned_demo_fallback_no_hash_match"
    return str(rows[0]["demo_name"]), "first_demo_fallback_no_hash_match"


def run_diagnostic(run_dir: Path, hdf5_path: Path, camera_size: int) -> dict[str, Any]:
    started = time.monotonic()
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "epoch_5_xvla_task1_headroom",
        "task_suite": TASK_SUITE,
        "task_id": TASK_ID,
        "task_description": TASK_DESCRIPTION,
        "residual_target": {
            "reset_identity": RESIDUAL_RESET_IDENTITY,
            "initial_state_index": RESIDUAL_INITIAL_STATE_INDEX,
            "expected_initial_state_sha256": EXPECTED_RESIDUAL_INIT_SHA256,
            "source": "matched X-VLA/SmolVLA-base shared failure",
            "xvla_failed": True,
            "smolvla_base_failed": True,
        },
        "inputs": {
            "hdf5_path": str(hdf5_path),
            "camera_size": int(camera_size),
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

        suite = benchmark.get_benchmark_dict()[TASK_SUITE]()
        task = suite.get_task(TASK_ID)
        bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
        benchmark_initial_states = suite.get_task_init_states(TASK_ID)
        benchmark_init_state = np.asarray(
            benchmark_initial_states[RESIDUAL_INITIAL_STATE_INDEX],
            dtype=np.float64,
        ).reshape(-1)
        computed_sha = _sha256_array(benchmark_init_state)
        rows = _scan_hdf5(hdf5_path, benchmark_init_state)
        matches = [row for row in rows if row.get("init_state_sha256") == computed_sha]
        selected_name, selection_reason = _select_demo(rows, computed_sha)
        selected = _read_demo(hdf5_path, selected_name)

        report["residual_target"]["computed_initial_state_sha256"] = computed_sha
        report["residual_target"]["computed_hash_matches_expected"] = bool(computed_sha == EXPECTED_RESIDUAL_INIT_SHA256)
        report["inputs"]["bddl_file"] = str(bddl_file)
        report["hdf5_scan"] = {
            "demo_count": len(rows),
            "same_reset_demo_match_count": len(matches),
            "same_reset_demo_matches": matches,
            "all_demo_summaries": rows,
            "nearest_demo_to_benchmark_residual_init": min(
                [row for row in rows if row.get("l2_to_benchmark_residual_init") is not None],
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
                camera_size=int(camera_size),
                init_state=np.asarray(selected["init_state"], dtype=np.float64),
                variant=variant,
                instruction=TASK_DESCRIPTION,
            )
            report["variants"].append(result)

        exact = next(
            row for row in report["variants"] if row.get("variant") == "hdf5_expert_replay_exact_selected_demo_init"
        )
        zero = next(row for row in report["variants"] if row.get("variant") == "zero_action_exact_selected_demo_init")
        default = next(row for row in report["variants"] if row.get("variant") == "hdf5_expert_replay_default_reset")
        task_level_ok = _expert_success(exact)
        same_reset_available = bool(matches)
        same_reset_headroom = bool(same_reset_available and task_level_ok)
        report["decision"] = (
            "TASK1_SAME_RESET_EXPERT_HEADROOM_POSITIVE"
            if same_reset_headroom
            else (
                "TASK1_TASK_LEVEL_EXPERT_HEADROOM_POSITIVE_SAME_RESET_UNAVAILABLE"
                if task_level_ok and not same_reset_available
                else "TASK1_EXPERT_HEADROOM_NOT_VERIFIED"
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
            else "No task-1 HDF5 demo init_state SHA-256 matched the residual benchmark initial-state SHA-256.",
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
        report["decision"] = "TASK1_EXPERT_HEADROOM_INFRASTRUCTURE_BLOCKED"
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--hdf5-path", type=Path, default=DEFAULT_HDF5_PATH)
    parser.add_argument("--camera-size", type=int, default=64)
    args = parser.parse_args(argv)
    report = run_diagnostic(Path(args.run_dir), Path(args.hdf5_path), int(args.camera_size))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
