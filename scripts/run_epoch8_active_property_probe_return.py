#!/usr/bin/env python3
"""Run the frozen scripted probe-return feasibility oracle."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.epoch7_latent_dynamics import apply_intervention, atomic_write_json, target_contact_state

PROTOCOL = REPO_ROOT / "reports/epoch8_active_property_probe_return_protocol.json"
OUTPUT = REPO_ROOT / "reports/epoch8_active_property_probe_return_result.json"
EXPECTED_PROTOCOL_SHA256 = "2D9D78FD6C0166613CADBB055FC417B22B23B2E63594DCD485E049AD8E42575A"
LIBERO_ROOT = Path("/mnt/c/assets/repos/LIBERO")
BDDL_ROOT = LIBERO_ROOT / "libero/libero/bddl_files/libero_90"
DATA_ROOT = Path("/mnt/c/assets/data/libero/libero_90")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def read_demo(path: Path, name: str) -> tuple[np.ndarray, np.ndarray]:
    import h5py

    with h5py.File(path, "r") as handle:
        demo = handle["data"][name]
        return np.asarray(demo.attrs["init_state"], dtype=np.float64), np.asarray(demo["actions"], dtype=np.float32)


def execute_forward(env: Any, body: str, actions: np.ndarray) -> tuple[list[np.ndarray], bool]:
    executed: list[np.ndarray] = []
    first_contact = None
    for step, action in enumerate(actions):
        env.step(action)
        executed.append(np.asarray(action, dtype=np.float32).copy())
        if bool(target_contact_state(env.sim, body)["target_contact"]) and first_contact is None:
            first_contact = step
        if first_contact is not None and step >= first_contact + 30:
            break
    return executed, first_contact is not None


def run_episode(env_class: Any, task: dict[str, Any], factor: float, fixed_prefix: list[np.ndarray] | None) -> tuple[dict[str, Any], list[np.ndarray]]:
    env = None
    row: dict[str, Any] = {"target": task["target"], "factor": factor, "exception": None}
    try:
        init_state, actions = read_demo(DATA_ROOT / task["hdf5"], task["demo"])
        env = env_class(bddl_file_name=str(BDDL_ROOT / task["bddl"]), camera_heights=64, camera_widths=64)
        env.seed(0)
        env.reset()
        observation = env.set_init_state(init_state)
        initial_target = np.asarray(env.sim.data.get_body_xpos(task["body"]), dtype=np.float64).copy()
        initial_eef = np.asarray(observation["robot0_eef_pos"], dtype=np.float64).copy()
        if factor != 1.0:
            apply_intervention(env.sim.model, {"axis": "target_mass", "body_name": task["body"], "arrays": ["body_mass", "body_inertia"], "factor": factor})
            env.sim.forward()
        if fixed_prefix is None:
            prefix, contacted = execute_forward(env, task["body"], actions)
        else:
            prefix = [np.asarray(value, dtype=np.float32).copy() for value in fixed_prefix]
            contacted = False
            for action in prefix:
                env.step(action)
                contacted = contacted or bool(target_contact_state(env.sim, task["body"])["target_contact"])
        inverse_actions = []
        for action in reversed(prefix):
            inverse = np.asarray(action, dtype=np.float32).copy()
            inverse[:6] *= -1.0
            inverse_actions.append(inverse)
            env.step(inverse)
        release = np.asarray([0, 0, 0, 0, 0, 0, 1], dtype=np.float32)
        for _ in range(10):
            env.step(release)
        final_target = np.asarray(env.sim.data.get_body_xpos(task["body"]), dtype=np.float64).copy()
        final_eef = np.asarray(env.env.robots[0].controller.ee_pos, dtype=np.float64).copy()
        all_actions = np.asarray(prefix + inverse_actions + [release] * 10, dtype=np.float32)
        row.update(
            {
                "completed": True,
                "target_contact": contacted,
                "prefix_steps": len(prefix),
                "total_steps": len(all_actions),
                "finite_bounded_actions": bool(np.isfinite(all_actions).all() and np.max(np.abs(all_actions)) <= 1.0 + 1e-7),
                "initial_target": initial_target.tolist(),
                "final_target": final_target.tolist(),
                "initial_eef": initial_eef.tolist(),
                "final_eef": final_eef.tolist(),
                "final_target_displacement_m": float(np.linalg.norm(final_target - initial_target)),
                "final_eef_displacement_m": float(np.linalg.norm(final_eef - initial_eef)),
            }
        )
        return row, prefix
    except Exception as exc:  # pragma: no cover
        row.update({"completed": False, "exception": f"{type(exc).__name__}: {exc}"})
        return row, []
    finally:
        if env is not None:
            env.close()


def main() -> int:
    if OUTPUT.exists():
        raise FileExistsError(f"refusing to overwrite {OUTPUT}")
    actual = sha256_file(PROTOCOL)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"protocol not frozen: {actual}")
    protocol = json.loads(PROTOCOL.read_text(encoding="utf-8"))
    package_root = str(LIBERO_ROOT / "libero")
    sys.path = [value for value in sys.path if value.rstrip("/") != package_root.rstrip("/")]
    sys.path.insert(0, str(LIBERO_ROOT))
    from libero.libero.envs import OffScreenRenderEnv

    result = {"schema_version": "epoch8.active_property.probe_return_result.v1", "protocol_sha256": EXPECTED_PROTOCOL_SHA256, "episodes": []}
    for task in protocol["tasks"]:
        fixed = None
        for factor in protocol["mass_factors"]:
            row, prefix = run_episode(OffScreenRenderEnv, task, float(factor), fixed)
            result["episodes"].append(row)
            atomic_write_json(OUTPUT, result)
            if fixed is None:
                fixed = prefix
    gates = protocol["gates"]
    valid = len(result["episodes"]) == int(gates["completed_episodes"]) and all(row.get("completed") and row.get("exception") is None for row in result["episodes"])
    feasible = bool(
        valid
        and all(row["target_contact"] and row["finite_bounded_actions"] for row in result["episodes"])
        and all(row["final_target_displacement_m"] <= float(gates["final_target_displacement_m_max"]) for row in result["episodes"])
        and all(row["final_eef_displacement_m"] <= float(gates["final_eef_displacement_m_max"]) for row in result["episodes"])
    )
    result["summary"] = {
        "episodes": len(result["episodes"]),
        "valid": valid,
        "target_contact_count": sum(bool(row.get("target_contact")) for row in result["episodes"]),
        "target_return_count": sum(row.get("final_target_displacement_m", np.inf) <= float(gates["final_target_displacement_m_max"]) for row in result["episodes"]),
        "eef_return_count": sum(row.get("final_eef_displacement_m", np.inf) <= float(gates["final_eef_displacement_m_max"]) for row in result["episodes"]),
        "max_target_displacement_m": max(row.get("final_target_displacement_m", np.inf) for row in result["episodes"]),
        "max_eef_displacement_m": max(row.get("final_eef_displacement_m", np.inf) for row in result["episodes"]),
    }
    result["decision"] = protocol["decisions"]["invalid"] if not valid else protocol["decisions"]["positive"] if feasible else protocol["decisions"]["negative"]
    atomic_write_json(OUTPUT, result)
    print(json.dumps({"decision": result["decision"], **result["summary"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
