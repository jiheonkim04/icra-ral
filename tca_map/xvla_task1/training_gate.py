"""Sequential BR-XVLA bounded training plus offline validation gate."""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tca_map.xvla_task1.offline_validate import DEFAULT_OUTPUT, OfflineValidationConfig, run_offline_validation
from tca_map.xvla_task1.train_lora import (
    DEFAULT_OUTPUT_ROOT,
    TrainArmConfig,
    _git_commit,
    _json_default,
    _load_spec,
    _write_json,
    run_training_arm,
)
from tca_map.xvla_task1.training_spec import SPEC_ARTIFACT


@dataclass(frozen=True)
class TrainingGateConfig:
    spec_path: Path = SPEC_ARTIFACT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    offline_output: Path = DEFAULT_OUTPUT
    max_steps_override: int | None = None
    num_validation_chunks: int = 24
    denoise_steps: int = 10
    device_index: int = 0
    local_files_only: bool = True


def run_training_gate(config: TrainingGateConfig) -> dict[str, Any]:
    started = time.monotonic()
    config.output_root.mkdir(parents=True, exist_ok=True)
    heartbeat_path = config.output_root / "gate_heartbeat.json"
    status_path = config.output_root / "gate_status.json"
    result_path = config.output_root / "gate_result.json"
    result: dict[str, Any] = {
        "schema_version": "2026-07-17.epoch5_br_xvla_training_gate.v1",
        "method": "BR-XVLA",
        "status": "RUNNING",
        "success": False,
        "decision": "BR_XVLA_TRAINING_GATE_RUNNING",
        "git_commit": _git_commit(),
        "worker_pid": os.getpid(),
        "spec_path": str(config.spec_path),
        "output_root": str(config.output_root),
        "offline_output": str(config.offline_output),
        "max_steps_override": config.max_steps_override,
        "closed_loop_ours_evaluation_happened": False,
        "started_unix": time.time(),
    }
    _write_json(status_path, result)
    _write_json(heartbeat_path, {"status": "load_spec", "pid": os.getpid(), "time_unix": time.time()})
    try:
        spec = _load_spec(config.spec_path)
        arm_ids = [str(arm["arm_id"]) for arm in spec["arms"]]
        training_results: dict[str, dict[str, Any]] = {}
        for arm_id in arm_ids:
            _write_json(heartbeat_path, {"status": f"training_{arm_id}", "pid": os.getpid(), "time_unix": time.time()})
            arm_result = run_training_arm(
                TrainArmConfig(
                    spec_path=config.spec_path,
                    arm_id=arm_id,
                    output_root=config.output_root,
                    max_steps_override=config.max_steps_override,
                    device_index=int(config.device_index),
                    local_files_only=bool(config.local_files_only),
                )
            )
            training_results[arm_id] = arm_result
            _write_json(status_path, {**result, "status": "RUNNING", "training_results": training_results})
            if not arm_result.get("success"):
                raise RuntimeError(f"training arm failed: {arm_id}")

        offline_result: dict[str, Any] | None = None
        if config.max_steps_override is None or int(config.max_steps_override) >= int(spec["shared_training"]["max_optimizer_steps"]):
            _write_json(heartbeat_path, {"status": "offline_validation", "pid": os.getpid(), "time_unix": time.time()})
            offline_result = run_offline_validation(
                OfflineValidationConfig(
                    spec_path=config.spec_path,
                    output_path=config.offline_output,
                    training_output_root=config.output_root,
                    num_chunks=int(config.num_validation_chunks),
                    denoise_steps=int(config.denoise_steps),
                    device_index=int(config.device_index),
                    local_files_only=bool(config.local_files_only),
                )
            )
            decision = str(offline_result.get("decision"))
            success = bool(offline_result.get("success"))
        else:
            decision = "BR_XVLA_TRAINING_DEBUG_COMPLETE_OFFLINE_SKIPPED"
            success = True
        result.update(
            {
                "status": "COMPLETE",
                "success": bool(success),
                "decision": decision,
                "training_results": training_results,
                "offline_result": offline_result,
                "closed_loop_ours_evaluation_happened": False,
                "elapsed_seconds": float(time.monotonic() - started),
                "result_path": str(result_path),
                "heartbeat_path": str(heartbeat_path),
                "status_path": str(status_path),
            }
        )
    except Exception as exc:  # pragma: no cover - runtime boundary
        result.update(
            {
                "status": "FAILED",
                "success": False,
                "decision": "BR_XVLA_TRAINING_GATE_FAILED",
                "exception": {"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()},
                "elapsed_seconds": float(time.monotonic() - started),
                "result_path": str(result_path),
                "heartbeat_path": str(heartbeat_path),
                "status_path": str(status_path),
            }
        )
    finally:
        _write_json(result_path, result)
        _write_json(status_path, result)
        _write_json(
            heartbeat_path,
            {
                "status": str(result["status"]).lower(),
                "pid": os.getpid(),
                "success": bool(result.get("success", False)),
                "decision": result.get("decision"),
                "result_path": str(result_path),
                "time_unix": time.time(),
            },
        )
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--offline-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-steps-override", type=int, default=None)
    parser.add_argument("--num-validation-chunks", type=int, default=24)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--device-index", type=int, default=0)
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args(argv)
    result = run_training_gate(
        TrainingGateConfig(
            spec_path=args.spec,
            output_root=args.output_root,
            offline_output=args.offline_output,
            max_steps_override=args.max_steps_override,
            num_validation_chunks=int(args.num_validation_chunks),
            denoise_steps=int(args.denoise_steps),
            device_index=int(args.device_index),
            local_files_only=not bool(args.allow_download),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0 if result.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
