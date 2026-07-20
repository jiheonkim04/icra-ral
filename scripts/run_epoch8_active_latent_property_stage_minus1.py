#!/usr/bin/env python3
"""Preflight and run the frozen active hidden-mass Base problem screen."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import random
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_epoch6_schedule_closed_loop as closed_loop  # noqa: E402
import run_epoch6_schedule_invariance_stage0 as stage0  # noqa: E402
from tca_map.epoch7_latent_dynamics import (  # noqa: E402
    apply_intervention,
    compare_observations,
    target_contact_state,
)

PROTOCOL_PATH = REPO_ROOT / "reports/epoch8_active_latent_property_protocol.json"
OUTPUT_ROOT = REPO_ROOT / "reports/epoch8_active_latent_property"
EXPECTED_PROTOCOL_SHA256 = "B68309214B3E53A0798E97295B3729DBF01AF227CB14B5BBB4D72144BCE3C506"
LIBERO_ROOT = Path("/mnt/c/assets/repos/LIBERO")
BDDL_ROOT = LIBERO_ROOT / "libero/libero/bddl_files/libero_90"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def load_protocol() -> dict[str, Any]:
    actual = sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"protocol hash mismatch or not frozen: {actual}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_STAGE_MINUS1_BASE_OUTCOMES":
        raise RuntimeError("protocol is not outcome-blind frozen")
    if sha256_file(Path(protocol["simulator"]["source_init_file"])) != protocol["simulator"]["source_init_file_sha256"]:
        raise RuntimeError("source init hash mismatch")
    for target in protocol["targets"]:
        if sha256_file(BDDL_ROOT / target["bddl"]) != target["bddl_sha256"]:
            raise RuntimeError(f"BDDL hash mismatch: {target['id']}")
    return protocol


def import_libero() -> tuple[Any, Any]:
    package_root = str(LIBERO_ROOT / "libero")
    sys.path = [value for value in sys.path if value.rstrip("/") != package_root.rstrip("/")]
    sys.path.insert(0, str(LIBERO_ROOT))
    import torch
    from libero.libero.envs import OffScreenRenderEnv

    return torch, OffScreenRenderEnv


def copy_observation(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: value.copy() if isinstance(value, np.ndarray) else value
        for key, value in observation.items()
    }


def task_intervention(target: Mapping[str, Any], protocol: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "axis": "target_mass",
        "body_name": target["body"],
        "arrays": list(protocol["intervention"]["arrays"]),
        "factor": float(protocol["intervention"]["factor"]),
    }


def make_env(
    env_class: Any,
    protocol: Mapping[str, Any],
    target: Mapping[str, Any],
    init_state: np.ndarray,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    env = env_class(
        bddl_file_name=str(BDDL_ROOT / target["bddl"]),
        camera_heights=int(protocol["simulator"]["resolution"][0]),
        camera_widths=int(protocol["simulator"]["resolution"][1]),
    )
    env.seed(int(protocol["simulator"]["env_seed"]))
    env.reset()
    observation = env.set_init_state(init_state)
    dummy = np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
    for _ in range(int(protocol["simulator"]["settle_steps"])):
        observation, _, _, _ = env.step(dummy)
    for robot in env.env.robots:
        robot.controller.use_delta = False
    cached = copy_observation(observation)
    qpos = np.asarray(env.sim.data.qpos, dtype=np.float64).copy()
    qvel = np.asarray(env.sim.data.qvel, dtype=np.float64).copy()
    mutation = apply_intervention(env.sim.model, task_intervention(target, protocol))
    env.sim.forward()
    residual = {
        "qpos_max_abs": float(np.max(np.abs(qpos - np.asarray(env.sim.data.qpos)))),
        "qvel_max_abs": float(np.max(np.abs(qvel - np.asarray(env.sim.data.qvel)))),
    }
    return env, cached, {"mutation": mutation, "state_residual": residual}


def run_preflight(protocol: dict[str, Any], output: Path) -> int:
    torch, env_class = import_libero()
    states = torch.load(Path(protocol["simulator"]["source_init_file"]), map_location="cpu", weights_only=False)
    state_index = int(protocol["identity_partition"]["discovery_state_indices"][0])
    rows = []
    reference = None
    try:
        for target in protocol["targets"]:
            env = None
            try:
                env, observation, metadata = make_env(
                    env_class, protocol, target, np.asarray(states[state_index], dtype=np.float64)
                )
                hashes = {
                    key: stage0.hash_array(np.asarray(value))
                    for key, value in observation.items()
                    if isinstance(value, np.ndarray)
                }
                if reference is None:
                    reference = copy_observation(observation)
                    equivalence = {"all_array_keys_match": True, "max_abs": 0.0}
                else:
                    compared = compare_observations(reference, observation)
                    equivalence = {
                        "all_array_keys_match": bool(compared["eligible"]),
                        "max_abs": float(
                            max(
                                value["max_abs"] or 0.0
                                for value in compared["keys"].values()
                            )
                        ),
                    }
                rows.append(
                    {
                        "target": target["id"],
                        "observation_hashes": hashes,
                        "reference_equivalence": equivalence,
                        **metadata,
                    }
                )
            finally:
                if env is not None:
                    env.close()
        passed = all(
            row["state_residual"]["qpos_max_abs"] == 0.0
            and row["state_residual"]["qvel_max_abs"] == 0.0
            and row["reference_equivalence"]["all_array_keys_match"]
            and row["reference_equivalence"]["max_abs"] == 0.0
            for row in rows
        )
        result = {
            "schema_version": "epoch8.active_latent_property.preflight.v1",
            "completed_at": closed_loop.utc_now(),
            "execution_type": "SETUP_PREFLIGHT_OUTCOME_SUPPRESSED",
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "script_sha256": sha256_file(Path(__file__)),
            "policy_loaded_or_queried": False,
            "simulator_actions_after_intervention": 0,
            "reward_success_done_read": False,
            "validation_or_confirmation_accessed": False,
            "rows": rows,
            "decision": "ACTIVE_LATENT_PROPERTY_PREFLIGHT_PASS" if passed else "ACTIVE_LATENT_PROPERTY_PREFLIGHT_FAIL",
        }
        stage0.write_json(output, result)
        print(json.dumps({"decision": result["decision"], "rows": len(rows)}, sort_keys=True))
        return 0 if passed else 1
    finally:
        gc.collect()


def seed_episode(torch: Any, seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_episode(
    *,
    protocol: dict[str, Any],
    target: dict[str, Any],
    target_ordinal: int,
    state_index: int,
    condition: str,
    init_state: np.ndarray,
    env_class: Any,
    torch: Any,
    model: Any,
    processor: Any,
) -> dict[str, Any]:
    started = time.monotonic()
    env = None
    seed = 880000 + 100 * int(target_ordinal) + int(state_index)
    row: dict[str, Any] = {
        "target": target["id"],
        "target_body": target["body"],
        "state_index": int(state_index),
        "condition": condition,
        "seed": seed,
        "success": False,
        "completed": False,
        "steps": 0,
        "policy_calls": 0,
        "correct_target_contact": False,
        "first_contacted_bowl": None,
        "wrong_bowl_contact_before_target": False,
        "finite_actions": True,
        "exception": None,
    }
    try:
        env, observation, metadata = make_env(env_class, protocol, target, init_state)
        row.update(metadata)
        instruction = (
            target["canonical_instruction"]
            if condition == "canonical_position"
            else protocol["hidden_instruction"]
        )
        row["instruction"] = instruction
        row["first_observation_sha256"] = {
            key: stage0.hash_array(np.asarray(value))
            for key, value in observation.items()
            if isinstance(value, np.ndarray)
        }
        seed_episode(torch, seed)
        last_raw = None
        horizon = int(protocol["simulator"]["horizon"])
        bowl_bodies = [item["body"] for item in protocol["targets"]]
        while row["steps"] < horizon and not row["success"]:
            agentview, wrist, proprio = closed_loop.make_policy_request_arrays(env, observation, last_raw)
            request = {
                "agentview": agentview,
                "wrist": wrist,
                "proprio": proprio,
                "language": instruction,
            }
            model_inputs, _ = closed_loop.prepare_model_inputs(request, processor, model, torch)
            with torch.no_grad():
                action = model.generate_actions(**model_inputs, steps=int(protocol["model"]["denoising_steps"]))
            torch.cuda.synchronize()
            raw = action.float().detach().cpu().numpy().squeeze(0).astype(np.float32)
            processed = stage0.raw_to_processed_7d(raw).astype(np.float32)
            if raw.shape != (30, 20) or processed.shape != (30, 7) or not np.isfinite(processed).all():
                row["finite_actions"] = False
                raise RuntimeError("nonfinite or malformed action chunk")
            row["policy_calls"] += 1
            last_raw = raw
            for action_row in processed:
                observation, reward, done, _ = env.step(action_row)
                row["steps"] += 1
                contacts = {
                    body: bool(target_contact_state(env.sim, body)["target_contact"])
                    for body in bowl_bodies
                }
                contacted = [body for body in bowl_bodies if contacts[body]]
                if contacted and row["first_contacted_bowl"] is None:
                    row["first_contacted_bowl"] = contacted[0]
                if contacts[target["body"]]:
                    row["correct_target_contact"] = True
                elif contacted and not row["correct_target_contact"]:
                    row["wrong_bowl_contact_before_target"] = True
                if bool(done) or float(reward) > 0.0:
                    row["success"] = True
                    break
                if row["steps"] >= horizon:
                    break
        row["completed"] = True
        row["timeout"] = not row["success"]
    except Exception as exc:  # pragma: no cover - runtime boundary
        row["exception"] = f"{type(exc).__name__}: {exc}"
        row["traceback"] = traceback.format_exc()
    finally:
        if env is not None:
            env.close()
    row["wall_seconds"] = round(time.monotonic() - started, 3)
    return row


def run_stage(protocol: dict[str, Any], preflight: Path, output: Path, resume: bool) -> int:
    if not preflight.is_file():
        raise RuntimeError("passing preflight is required")
    preflight_data = json.loads(preflight.read_text(encoding="utf-8"))
    if preflight_data.get("decision") != "ACTIVE_LATENT_PROPERTY_PREFLIGHT_PASS":
        raise RuntimeError("preflight did not pass")
    torch, env_class = import_libero()
    before = stage0.resource_snapshot()
    stage0.require_safe_resources(before)
    states = torch.load(Path(protocol["simulator"]["source_init_file"]), map_location="cpu", weights_only=False)
    torch = stage0.seed_process_once(880000)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    monitor = stage0.ResourceMonitor(torch, OUTPUT_ROOT / "stage_minus1_heartbeat.json", interval_seconds=1.0)
    monitor.start()
    model = processor = None
    if output.exists():
        if not resume:
            raise FileExistsError(f"result exists; use --resume to execute only missing pairs: {output}")
        result = json.loads(output.read_text(encoding="utf-8"))
        if result.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
            raise RuntimeError("resume result protocol mismatch")
        result["resumed_at"] = closed_loop.utc_now()
        result.pop("fatal_exception", None)
        result.pop("fatal_traceback", None)
    else:
        result = {
            "schema_version": "epoch8.active_latent_property.stage_minus1_result.v1",
            "started_at": closed_loop.utc_now(),
            "execution_type": "VLA_CLOSED_LOOP_ROLLOUT_DISCOVERY",
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "script_sha256": sha256_file(Path(__file__)),
            "preflight_sha256": sha256_file(preflight),
            "validation_or_confirmation_accessed": False,
            "ours_designed_or_executed": False,
            "resources_before": before,
            "episodes": [],
        }
    exit_code = 1
    try:
        model, processor, runtime = stage0.load_xvla(torch)
        result["runtime"] = runtime
        completed_keys = {
            (row["target"], int(row["state_index"]), row["condition"])
            for row in result["episodes"]
            if row.get("completed") and row.get("exception") is None
        }
        for target_ordinal, target in enumerate(protocol["targets"]):
            for state_index in protocol["identity_partition"]["discovery_state_indices"]:
                for condition in protocol["pairing"]["conditions"]:
                    key = (target["id"], int(state_index), condition)
                    if key in completed_keys:
                        continue
                    episode = run_episode(
                        protocol=protocol,
                        target=target,
                        target_ordinal=target_ordinal,
                        state_index=int(state_index),
                        condition=condition,
                        init_state=np.asarray(states[int(state_index)], dtype=np.float64),
                        env_class=env_class,
                        torch=torch,
                        model=model,
                        processor=processor,
                    )
                    result["episodes"].append(episode)
                    stage0.write_json(output, result)
        canonical = [row for row in result["episodes"] if row["condition"] == "canonical_position"]
        hidden = [row for row in result["episodes"] if row["condition"] == "hidden_heaviest"]
        by_key = {(row["target"], row["state_index"]): row for row in hidden}
        paired = []
        for left in canonical:
            right = by_key[(left["target"], left["state_index"])]
            paired.append(
                {
                    "target": left["target"],
                    "state_index": left["state_index"],
                    "canonical_success": left["success"],
                    "hidden_success": right["success"],
                    "canonical_contact": left["correct_target_contact"],
                    "hidden_contact": right["correct_target_contact"],
                    "canonical_win_hidden_loss": bool(left["success"] and not right["success"]),
                    "canonical_contact_hidden_miss": bool(left["correct_target_contact"] and not right["correct_target_contact"]),
                    "first_observation_match": left["first_observation_sha256"] == right["first_observation_sha256"],
                }
            )
        exceptions = sum(row["exception"] is not None for row in result["episodes"])
        canonical_successes = sum(row["success"] for row in canonical)
        canonical_contacts = sum(row["correct_target_contact"] for row in canonical)
        wins = sum(row["canonical_win_hidden_loss"] for row in paired)
        contact_misses = sum(row["canonical_contact_hidden_miss"] for row in paired)
        affected_targets = len({row["target"] for row in paired if row["canonical_win_hidden_loss"]})
        integrity = bool(
            len(result["episodes"]) == 12
            and exceptions == 0
            and all(row["finite_actions"] for row in result["episodes"])
            and all(row["first_observation_match"] for row in paired)
            and all(row["state_residual"]["qpos_max_abs"] == 0.0 and row["state_residual"]["qvel_max_abs"] == 0.0 for row in result["episodes"])
        )
        competence = canonical_successes >= 5 and canonical_contacts >= 5
        gap = wins >= 2 and contact_misses >= 2 and affected_targets >= 2
        if not integrity:
            decision = protocol["decisions"]["invalid"]
        elif not competence:
            decision = protocol["decisions"]["incompetent"]
        elif gap:
            decision = protocol["decisions"]["positive"]
        else:
            decision = protocol["decisions"]["negative"]
        result["paired_rows"] = paired
        result["summary"] = {
            "episodes": len(result["episodes"]),
            "exceptions": exceptions,
            "canonical_successes": canonical_successes,
            "hidden_successes": sum(row["success"] for row in hidden),
            "canonical_correct_contacts": canonical_contacts,
            "hidden_correct_contacts": sum(row["correct_target_contact"] for row in hidden),
            "canonical_win_hidden_loss_count": wins,
            "canonical_contact_hidden_miss_count": contact_misses,
            "affected_target_position_count": affected_targets,
            "integrity_pass": integrity,
            "competence_pass": competence,
            "problem_gap_pass": gap,
        }
        result["decision"] = decision
        result["completed_at"] = closed_loop.utc_now()
        exit_code = 0
    except Exception as exc:
        result["fatal_exception"] = f"{type(exc).__name__}: {exc}"
        result["fatal_traceback"] = traceback.format_exc()
        result["decision"] = protocol["decisions"]["invalid"]
    finally:
        model = processor = None
        gc.collect()
        torch.cuda.empty_cache()
        result["resource_monitor"] = monitor.stop()
        result["resources_after"] = stage0.resource_snapshot(torch)
        if result["resource_monitor"]["maximum_swap_used_bytes"] != 0 or result["resource_monitor"]["exceptions"]:
            result["decision"] = protocol["decisions"]["resource"]
            exit_code = 1
        stage0.write_json(output, result)
    print(json.dumps({"decision": result["decision"], **result.get("summary", {})}, sort_keys=True))
    return exit_code


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "run"], required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    protocol = load_protocol()
    args.output_root.mkdir(parents=True, exist_ok=True)
    preflight = args.output_root / "preflight.json"
    result = args.output_root / "stage_minus1_result.json"
    if args.mode == "preflight":
        return run_preflight(protocol, preflight)
    return run_stage(protocol, preflight, result, bool(args.resume))


if __name__ == "__main__":
    raise SystemExit(main())
