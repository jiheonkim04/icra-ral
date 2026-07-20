#!/usr/bin/env python3
"""Execute the independent Epoch 8 two-shard actual-arrival protocol."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_epoch6_schedule_closed_loop as base  # noqa: E402

PROTOCOL_PATH = REPO_ROOT / "reports/epoch8_two_shard_actual_arrival_protocol.json"
EXPECTED_PROTOCOL_SHA256 = "4C5CD89AAE4C36B978FF3DBBD3658D2217A88115E19F3FACBFA60D18DA96904B"
SCHEDULES = ("epoch8_single_lane_canonical_serial", "epoch8_two_shards_actual_arrival")
SERIAL_ASSIGNMENTS = {0: list(range(20))}
TWO_SHARD_ASSIGNMENTS = {
    0: list(range(0, 20, 2)),
    1: list(range(1, 20, 2)),
}
SERIAL_OFFSETS = {0: 0.0}
TWO_SHARD_OFFSETS = {0: 1.0, 1: 0.0}
REQUIRED_FIRST_ARRIVAL = [1, 0]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def configure_base() -> None:
    base.SCHEDULES = SCHEDULES
    base.SERIAL_ASSIGNMENTS = SERIAL_ASSIGNMENTS
    base.SHARDED_ASSIGNMENTS = TWO_SHARD_ASSIGNMENTS
    base.SERIAL_OFFSETS = SERIAL_OFFSETS
    base.SHARDED_OFFSETS = TWO_SHARD_OFFSETS
    base.REQUIRED_SHARDED_FIRST_ARRIVAL = REQUIRED_FIRST_ARRIVAL


def validate_epoch8_protocol() -> dict[str, Any]:
    actual = sha256_file(PROTOCOL_PATH)
    if actual != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"Epoch 8 protocol hash mismatch: {actual}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["status"] != "FROZEN_BEFORE_TWO_SHARD_RESOURCE_OR_SCIENTIFIC_OUTCOMES":
        raise RuntimeError("Epoch 8 protocol is not frozen")
    if protocol["schedules"][SCHEDULES[1]]["required_initial_first_arrival_shard_order"] != [1, 0]:
        raise RuntimeError("Epoch 8 arrival-order mismatch")
    base.validate_manifest()
    return protocol


def require_resource_lock(run_dir: Path) -> None:
    expected = (run_dir.parent / "epoch8_two_shard_resource.global.lock.json").resolve()
    token = os.environ.get("EPOCH8_TWO_SHARD_RESOURCE_LOCK")
    if token is None or Path(token).resolve() != expected or not expected.is_file():
        raise RuntimeError("two-shard resource smoke requires the host-monitor lock")
    payload = json.loads(expected.read_text(encoding="utf-8-sig"))
    if payload.get("status") != "active" or payload.get("run_id") != run_dir.name:
        raise RuntimeError("invalid two-shard resource lock")
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("two-shard resource lock protocol mismatch")


def resource_smoke(run_dir: Path) -> int:
    protocol = validate_epoch8_protocol()
    require_resource_lock(run_dir)
    provenance = base.stage0.validate_execution_provenance(run_dir)
    before = base.stage0.resource_snapshot()
    base.stage0.require_safe_resources(before)
    result: dict[str, Any] = {
        "schema_version": "epoch8.two_shard_actual_arrival.resource_smoke.v1",
        "started_at": base.utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "provenance": provenance,
        "resources_before": before,
        "simultaneous_env_instances": 0,
        "simultaneous_env_processes": 0,
        "model_inference_calls": 0,
        "simulator_actions_executed": 0,
        "reward_success_done_read": False,
        "status": "TWO_SHARD_RESOURCE_SMOKE_FAILED",
    }
    context = mp.get_context("spawn")
    ready_queue = context.Queue()
    release_event = context.Event()
    holders: list[Any] = []
    model = processor = torch_module = None
    monitor = None
    exit_code = 1
    try:
        for shard_id in range(2):
            holder = context.Process(
                target=base.resource_env_holder,
                args=(shard_id, ready_queue, release_event),
                name=f"epoch8-two-shard-smoke-env-{shard_id}",
            )
            holder.start()
            holders.append(holder)
        torch_module = base.stage0.seed_process_once(base.ROOT_SEED)
        torch_module.cuda.empty_cache()
        torch_module.cuda.reset_peak_memory_stats()
        monitor = base.stage0.ResourceMonitor(
            torch_module, run_dir / "two_shard_resource_heartbeat.json", interval_seconds=0.5
        )
        monitor.start()
        ready_rows = []
        while len(ready_rows) < 2:
            message = ready_queue.get(timeout=300)
            if message.get("status") != "ready":
                raise RuntimeError(f"environment holder failed: {message.get('exception')}")
            ready_rows.append(message)
        ready_rows.sort(key=lambda row: int(row["shard_id"]))
        success_check_calls = sum(int(row["success_check_calls"]) for row in ready_rows)
        result["simultaneous_env_instances"] = len(ready_rows)
        result["simultaneous_env_processes"] = len(holders)
        result["env_holder_pids"] = [int(row["pid"]) for row in ready_rows]
        if success_check_calls != 0:
            raise RuntimeError("resource smoke called success logic")
        model, processor, runtime = base.stage0.load_xvla(torch_module)
        result["runtime"] = runtime
        first = ready_rows[0]
        request = {
            "agentview": first["agentview"],
            "wrist": first["wrist"],
            "proprio": first["proprio"],
            "language": first["language"],
        }
        model_inputs, prepared_hash = base.prepare_model_inputs(
            request, processor, model, torch_module
        )
        started = time.monotonic()
        with torch_module.no_grad():
            action = model.generate_actions(**model_inputs, steps=10)
        torch_module.cuda.synchronize()
        raw = action.float().detach().cpu().numpy().squeeze(0).astype(np.float32)
        result.update(
            {
                "forward_seconds": time.monotonic() - started,
                "prepared_input_sha256": prepared_hash,
                "raw_chunk_shape": list(raw.shape),
                "raw_chunk_finite": bool(np.isfinite(raw).all()),
                "raw_chunk_sha256": base.stage0.hash_array(raw),
                "model_inference_calls": 1,
                "success_check_calls": success_check_calls,
                "status": "TWO_SHARD_ACTUAL_PATH_RESOURCE_SMOKE_PASS",
            }
        )
        exit_code = 0
    except Exception as exc:
        result.update(
            {
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        release_event.set()
        for holder in holders:
            holder.join(timeout=60)
            if holder.is_alive():
                holder.terminate()
                holder.join(timeout=10)
            if holder.exitcode not in (0, None):
                result.setdefault("env_holder_exit_errors", []).append(
                    {"name": holder.name, "exit_code": holder.exitcode}
                )
                result["status"] = "TWO_SHARD_RESOURCE_SMOKE_FAILED_ENV_HOLDER"
                exit_code = 1
        model = processor = None
        if torch_module is not None:
            gc.collect()
            torch_module.cuda.empty_cache()
        if monitor is not None:
            result["resource_monitor"] = monitor.stop()
            if (
                result["resource_monitor"]["maximum_swap_used_bytes"]
                > int(protocol["resource_gate"]["wsl_swap_used_bytes_max"])
                or result["resource_monitor"]["exceptions"]
            ):
                result["status"] = "TWO_SHARD_RESOURCE_SMOKE_FAILED_TELEMETRY_OR_SWAP"
                exit_code = 1
        result["resources_after"] = base.stage0.resource_snapshot(torch_module)
        result["completed_at"] = base.utc_now()
        base.stage0.write_json(run_dir / "two_shard_resource_smoke.json", result)
        base.stage0.write_text(run_dir / "two_shard_resource_smoke_exit_code.txt", f"{exit_code}\n")
    return exit_code


def resource_smoke_valid(run_dir: Path) -> bool:
    try:
        internal = json.loads((run_dir / "two_shard_resource_smoke.json").read_text(encoding="utf-8"))
        host = json.loads((run_dir / "two_shard_resource_smoke_host.json").read_text(encoding="utf-8-sig"))
        return bool(
            internal["status"] == "TWO_SHARD_ACTUAL_PATH_RESOURCE_SMOKE_PASS"
            and internal["simultaneous_env_instances"] == 2
            and internal["simultaneous_env_processes"] == 2
            and internal["model_inference_calls"] == 1
            and internal["raw_chunk_shape"] == [30, 20]
            and internal["raw_chunk_finite"]
            and internal["success_check_calls"] == 0
            and internal["resource_monitor"]["maximum_swap_used_bytes"] == 0
            and not internal["resource_monitor"]["exceptions"]
            and host["final_decision"] == "TWO_SHARD_RESOURCE_SMOKE_PASS"
            and host["peak_used_fraction"] <= 0.82
            and host["child_exit_code"] == 0
            and host["internal_report_sha256"] == sha256_file(run_dir / "two_shard_resource_smoke.json")
        )
    except Exception:
        return False


def epoch8_preflight(run_dir: Path) -> None:
    validate_epoch8_protocol()
    base.static_preflight(run_dir)
    archive = json.loads((REPO_ROOT / "reports/epoch7_epoch6_archive_lock.json").read_text(encoding="utf-8"))
    result = {
        "schema_version": "epoch8.two_shard_actual_arrival.preflight.v1",
        "completed_at": base.utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "runner_sha256": sha256_file(Path(__file__)),
        "epoch6_static_preflight_sha256": sha256_file(run_dir / "closed_loop_static_preflight.json"),
        "epoch6_four_shard_outcomes_observed": archive["preserved_facts"][
            "four_shard_closed_loop_outcomes_observed"
        ],
        "schedules": list(SCHEDULES),
        "assignments": {str(key): value for key, value in TWO_SHARD_ASSIGNMENTS.items()},
        "required_first_arrival": REQUIRED_FIRST_ARRIVAL,
        "scientific_outcomes_read": False,
        "status": "EPOCH8_TWO_SHARD_PREFLIGHT_PASS",
    }
    if result["epoch6_four_shard_outcomes_observed"] != 0:
        raise RuntimeError("archived four-shard evidence lock changed")
    base.stage0.write_json(run_dir / "epoch8_two_shard_preflight.json", result)


def launch_schedule(run_dir: Path, schedule: str) -> None:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--mode",
        "schedule",
        "--schedule",
        schedule,
        "--run-dir",
        str(run_dir),
        "--child",
    ]
    launch_path = run_dir / f"launch_{schedule}.json"
    stdout_path = run_dir / f"launch_{schedule}.stdout.log"
    stderr_path = run_dir / f"launch_{schedule}.stderr.log"
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(base.ROOT_SEED)
    environment["EPOCH6_PARENT_RUN_LOCK"] = str((run_dir / "run.lock.json").resolve())
    base.stage0.write_json(launch_path, {"status": "running", "command": command})
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(command, cwd=REPO_ROOT, env=environment, stdout=stdout, stderr=stderr, check=False)
    base.stage0.write_json(
        launch_path,
        {"status": "completed" if completed.returncode == 0 else "failed", "command": command, "exit_code": completed.returncode},
    )
    if completed.returncode != 0:
        raise RuntimeError(f"schedule failed: {schedule}")


def adjudicate(run_dir: Path) -> dict[str, Any]:
    result = base.adjudicate(run_dir)
    mapping = {
        "PROBLEM_VERIFIED_METHOD_DESIGN_AUTHORIZED": "TWO_SHARD_PROBLEM_VERIFIED_METHOD_DESIGN_AUTHORIZED",
        "NO_REPEATABLE_PROBLEM": "TWO_SHARD_NO_REPEATABLE_PROBLEM",
        "EVALUATION_INVALID": "TWO_SHARD_EVALUATION_INVALID",
    }
    result["schema_version"] = "epoch8.two_shard_actual_arrival.result.v1"
    result["protocol_sha256"] = EXPECTED_PROTOCOL_SHA256
    result["epoch6_four_shard_protocol_executed"] = False
    result["epoch8_final_decision"] = mapping[result["final_decision"]]
    result["method_design_authorized"] = result["epoch8_final_decision"].endswith("METHOD_DESIGN_AUTHORIZED")
    base.stage0.write_json(run_dir / "epoch8_two_shard_result.json", result)
    return result


def run_all(run_dir: Path) -> int:
    validate_epoch8_protocol()
    if not resource_smoke_valid(run_dir):
        raise RuntimeError("valid host-qualified two-shard resource smoke is required")
    lock_path = base.stage0.acquire_run_lock(run_dir)
    try:
        for schedule in SCHEDULES:
            launch_schedule(run_dir, schedule)
        result = adjudicate(run_dir)
        return 0 if result["epoch8_final_decision"] else 1
    finally:
        base.stage0.release_run_lock(run_dir, lock_path)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["preflight", "resource-smoke", "schedule", "adjudicate", "run-all"], default="run-all")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--schedule", choices=SCHEDULES)
    parser.add_argument("--child", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    configure_base()
    args = parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        epoch8_preflight(run_dir)
        return 0
    if args.mode == "resource-smoke":
        if not args.child:
            raise RuntimeError("resource-smoke is host-monitor child-only")
        return resource_smoke(run_dir)
    if args.mode == "schedule":
        if not args.child or args.schedule is None:
            raise RuntimeError("schedule mode requires --child and --schedule")
        validate_epoch8_protocol()
        return base.run_schedule(run_dir, args.schedule)
    if args.mode == "adjudicate":
        adjudicate(run_dir)
        return 0
    return run_all(run_dir)


if __name__ == "__main__":
    raise SystemExit(main())
