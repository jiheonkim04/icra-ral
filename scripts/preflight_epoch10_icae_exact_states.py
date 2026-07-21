"""Preflight the frozen four-suite exact-state substrate for Epoch 10 ICAE.

This script is deliberately outcome blind with respect to every prospective
checkpoint.  It only verifies that selected raw LIBERO demonstrations map to
the official benchmark tasks and that their physics states can be restored in
fresh simulator instances with the registered action convention.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


TASK_SPECS = (
    {
        "suite": "libero_spatial",
        "hdf5_stem": "pick_up_the_black_bowl_between_the_plate_and_the_ramekin_and_place_it_on_the_plate",
        "interaction_family": "spatially_conditioned_pick_place",
    },
    {
        "suite": "libero_object",
        "hdf5_stem": "pick_up_the_alphabet_soup_and_place_it_in_the_basket",
        "interaction_family": "object_conditioned_pick_place",
    },
    {
        "suite": "libero_goal",
        "hdf5_stem": "open_the_middle_drawer_of_the_cabinet",
        "interaction_family": "articulated_fixture",
    },
    {
        "suite": "libero_10",
        "hdf5_stem": "KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it",
        "interaction_family": "long_horizon_multi_goal",
    },
)

DEMO_PARTITIONS = {
    "mechanics_calibration": (0, 1),
    "metric_development_states": (2, 3, 4),
    "metric_holdout_states": (5, 6, 7),
}
STATE_FRACTIONS = (0.15, 0.35, 0.55, 0.75)
DEVELOPMENT_CLOSED_LOOP_INITIAL_STATE_INDICES = tuple(range(20, 35))
OFFICIAL_CLOSED_LOOP_INITIAL_STATE_INDICES = tuple(range(35, 50))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _array_sha256(value: Any) -> str:
    array = np.ascontiguousarray(np.asarray(value))
    return hashlib.sha256(array.tobytes()).hexdigest()


def _canonical_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")
    ).hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _task_for_stem(suite_name: str, stem: str) -> tuple[int, Any, Any]:
    from libero.libero import benchmark

    suite = benchmark.get_benchmark_dict()[suite_name]()
    matches = [
        (index, task)
        for index, task in enumerate(suite.tasks)
        if Path(str(task.bddl_file)).stem.lower() == stem.lower()
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one benchmark match for {suite_name}/{stem}, found {len(matches)}")
    return int(matches[0][0]), matches[0][1], suite


def _resolve_bddl_file(task: Any) -> Path:
    from libero.libero import get_libero_path

    raw = Path(str(task.bddl_file))
    if raw.is_file():
        return raw
    candidate = Path(get_libero_path("bddl_files")) / str(task.problem_folder) / raw.name
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"could not resolve BDDL file for {task.problem_folder}/{raw.name}")


def _make_env(bddl_file: Path, camera_size: int) -> Any:
    from libero.libero.envs import OffScreenRenderEnv

    return OffScreenRenderEnv(
        bddl_file_name=str(bddl_file),
        camera_heights=int(camera_size),
        camera_widths=int(camera_size),
    )


def _sim_state(env: Any) -> np.ndarray:
    return np.asarray(env.sim.get_state().flatten(), dtype=np.float64).copy()


def _frame_indices(length: int, horizon_guard: int = 17) -> list[int]:
    last = max(0, int(length) - int(horizon_guard) - 1)
    return sorted({min(last, max(0, int(round(float(fraction) * (length - 1))))) for fraction in STATE_FRACTIONS})


def _goal_audit(env: Any) -> dict[str, Any]:
    task_env = getattr(env, "env", env)
    goal_state = list(task_env.parsed_problem.get("goal_state", []))
    rows = []
    for predicate in goal_state:
        names = [str(value) for value in predicate[1:]]
        rows.append(
            {
                "predicate": str(predicate[0]),
                "operands": names,
                "operand_states_available": all(name in task_env.object_states_dict for name in names),
            }
        )
    return {
        "goal_predicate_count": len(goal_state),
        "goal_predicates": rows,
        "native_success_at_restored_frame": bool(env.check_success()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    started = time.monotonic()
    raw_root = Path(args.raw_root)
    rows: list[dict[str, Any]] = []
    tasks: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    env_cache: dict[str, Any] = {}

    for spec in TASK_SPECS:
        suite_name = str(spec["suite"])
        stem = str(spec["hdf5_stem"])
        hdf5_path = raw_root / suite_name / f"{stem}_demo.hdf5"
        task_id, task, suite = _task_for_stem(suite_name, stem)
        bddl_file = _resolve_bddl_file(task)
        if not bddl_file.is_file() or not hdf5_path.is_file():
            raise FileNotFoundError(f"missing task asset: {bddl_file} or {hdf5_path}")
        initial_states = suite.get_task_init_states(task_id)
        task_row = {
            **spec,
            "task_id": task_id,
            "instruction": str(task.language),
            "bddl_file": str(bddl_file.resolve()),
            "bddl_sha256": _sha256_file(bddl_file),
            "hdf5_path": str(hdf5_path.resolve()),
            "hdf5_sha256": _sha256_file(hdf5_path),
            "official_initial_state_count": int(len(initial_states)),
        }
        tasks.append(task_row)

        with h5py.File(hdf5_path, "r") as handle:
            demos = handle["data"]
            for partition, demo_ids in DEMO_PARTITIONS.items():
                for demo_id in demo_ids:
                    demo_name = f"demo_{demo_id}"
                    demo = demos[demo_name]
                    actions = np.asarray(demo["actions"], dtype=np.float64)
                    states = np.asarray(demo["states"], dtype=np.float64)
                    if actions.ndim != 2 or actions.shape[1] != 7 or len(actions) != len(states):
                        raise RuntimeError(f"invalid demo tensors in {hdf5_path}/{demo_name}")
                    for frame in _frame_indices(len(actions)):
                        state = states[frame]
                        env = None
                        row = {
                            "state_id": f"{suite_name}|task_{task_id}|{demo_name}|frame_{frame}",
                            "partition": partition,
                            "suite": suite_name,
                            "task_id": task_id,
                            "demo_name": demo_name,
                            "frame": int(frame),
                            "episode_length": int(len(actions)),
                            "trajectory_fraction": round(float(frame / max(1, len(actions) - 1)), 8),
                            "state_sha256": _array_sha256(state),
                            "expert_action_sha256": _array_sha256(actions[frame]),
                            "expert_action_finite": bool(np.isfinite(actions[frame]).all()),
                            "expert_action_max_abs": float(np.max(np.abs(actions[frame]))),
                            "cache_origin_frame": int((frame // 50) * 50),
                            "cache_offset": int(frame % 50),
                            "error": None,
                        }
                        try:
                            env_key = str(bddl_file.resolve())
                            env = env_cache.get(env_key)
                            if env is None:
                                env = _make_env(bddl_file, int(args.camera_size))
                                env.seed(int(args.seed))
                                env.reset()
                                env_cache[env_key] = env
                            observation = env.set_init_state(state)
                            restored = _sim_state(env)
                            state_l2 = float(np.linalg.norm(restored - state))
                            images = {
                                str(key): list(np.asarray(value).shape)
                                for key, value in observation.items()
                                if "image" in str(key)
                            }
                            row.update(
                                {
                                    "state_dimension": int(state.size),
                                    "restored_state_sha256": _array_sha256(restored),
                                    "restore_l2": state_l2,
                                    "restore_exact_at_1e_9": bool(state_l2 <= 1e-9),
                                    "observation_image_shapes": images,
                                    "goal_audit": _goal_audit(env),
                                }
                            )
                        except Exception as exc:
                            row["error"] = f"{type(exc).__name__}: {exc}"
                            errors.append({"state_id": row["state_id"], "error": row["error"]})
                        rows.append(row)

    for env in env_cache.values():
        env.close()

    state_ids = [str(row["state_id"]) for row in rows]
    partition_sets = {
        partition: sorted(str(row["state_id"]) for row in rows if row["partition"] == partition)
        for partition in DEMO_PARTITIONS
    }
    demo_sets = {
        partition: sorted({f"{row['suite']}|task_{row['task_id']}|{row['demo_name']}" for row in rows if row["partition"] == partition})
        for partition in DEMO_PARTITIONS
    }
    disjoint = all(
        set(demo_sets[left]).isdisjoint(demo_sets[right])
        for i, left in enumerate(demo_sets)
        for right in list(demo_sets)[i + 1 :]
    )
    report = {
        "schema_version": 1,
        "campaign": "epoch10_icae_vla",
        "status": "PASS" if not errors and disjoint and len(set(state_ids)) == len(state_ids) else "FAIL",
        "checkpoint_actions_queried": 0,
        "checkpoint_outcomes_read": False,
        "official_outcomes_opened": False,
        "tasks": tasks,
        "task_count": len(tasks),
        "suite_count": len({str(row["suite"]) for row in tasks}),
        "interaction_families": sorted({str(row["interaction_family"]) for row in tasks}),
        "demo_partitions": {key: list(value) for key, value in DEMO_PARTITIONS.items()},
        "state_fractions": list(STATE_FRACTIONS),
        "state_counts": {key: len(value) for key, value in partition_sets.items()},
        "states_per_task_per_metric_partition": len(STATE_FRACTIONS) * 3,
        "whole_demo_partition_sets": demo_sets,
        "whole_demo_partitions_disjoint": disjoint,
        "development_closed_loop_initial_state_indices": list(DEVELOPMENT_CLOSED_LOOP_INITIAL_STATE_INDICES),
        "official_closed_loop_initial_state_indices": list(OFFICIAL_CLOSED_LOOP_INITIAL_STATE_INDICES),
        "closed_loop_seed_partitions_disjoint": set(DEVELOPMENT_CLOSED_LOOP_INITIAL_STATE_INDICES).isdisjoint(
            OFFICIAL_CLOSED_LOOP_INITIAL_STATE_INDICES
        ),
        "rows": rows,
        "errors": errors,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    report["canonical_payload_sha256"] = _canonical_sha256(report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", default="/mnt/c/assets/data/libero")
    parser.add_argument("--camera-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260721)
    parser.add_argument("--output", default="reports/epoch10_icae_exact_state_preflight_attempt2.json")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = run(args)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "tasks": report["task_count"], "states": len(report["rows"]), "errors": len(report["errors"]), "elapsed_seconds": report["elapsed_seconds"]}))
    return 0 if report["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
