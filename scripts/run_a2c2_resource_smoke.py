#!/usr/bin/env python
"""Resource-only A2C2 actual-path smoke around the byte-frozen runner.

The frozen scientific runner remains untouched. This wrapper calls its exact
policy, environment, reset, and episode functions, but suppresses task success
and reward from every persisted artifact.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import threading
import time
import traceback
from typing import Any


import run_a2c2_problem_verification as frozen


class ResourceSampler:
    def __init__(self, torch_mod: Any, interval_seconds: float = 0.25) -> None:
        self.torch = torch_mod
        self.interval_seconds = float(interval_seconds)
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="a2c2-resource-sampler", daemon=True)
        self.samples = 0
        self.peak_rss_mb = 0.0
        self.peak_wsl_used_gib = 0.0
        self.peak_wsl_used_fraction = 0.0
        self.peak_vram_allocated_mib = 0.0
        self.peak_vram_reserved_mib = 0.0
        self.exceptions: list[str] = []

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=max(5.0, self.interval_seconds * 4))
        return {
            "samples": int(self.samples),
            "interval_seconds": self.interval_seconds,
            "peak_rss_mb": frozen._round(self.peak_rss_mb, 3),
            "peak_wsl_used_gib": frozen._round(self.peak_wsl_used_gib, 3),
            "peak_wsl_used_fraction": frozen._round(self.peak_wsl_used_fraction, 6),
            "peak_vram_allocated_mib": frozen._round(self.peak_vram_allocated_mib, 3),
            "peak_vram_reserved_mib": frozen._round(self.peak_vram_reserved_mib, 3),
            "exceptions": list(self.exceptions),
        }

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                snapshot = frozen._resource_snapshot(self.torch)
                self.samples += 1
                self.peak_rss_mb = max(self.peak_rss_mb, float(snapshot["rss_mb"]))
                self.peak_wsl_used_gib = max(self.peak_wsl_used_gib, float(snapshot["system_ram_used_gib"]))
                self.peak_wsl_used_fraction = max(
                    self.peak_wsl_used_fraction,
                    float(snapshot["system_ram_used_fraction"]),
                )
                self.peak_vram_allocated_mib = max(
                    self.peak_vram_allocated_mib,
                    float(snapshot.get("vram_allocated_mib") or 0.0),
                )
                self.peak_vram_reserved_mib = max(
                    self.peak_vram_reserved_mib,
                    float(snapshot.get("vram_reserved_mib") or 0.0),
                )
            except Exception as exc:  # pragma: no cover - defensive telemetry path
                self.exceptions.append(f"{type(exc).__name__}: {exc}")
            self.stop_event.wait(self.interval_seconds)


def meminfo() -> dict[str, int | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    except Exception:
        return {"mem_total_bytes": None, "swap_total_bytes": None, "swap_free_bytes": None}
    return {
        "mem_total_bytes": values.get("MemTotal"),
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
    }


def kernel_oom_evidence() -> dict[str, Any]:
    try:
        completed = subprocess.run(
            ["dmesg", "--color=never"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
        text = completed.stdout + "\n" + completed.stderr
        lines = [
            line
            for line in text.splitlines()
            if any(token in line.lower() for token in ("out of memory", "oom-kill", "killed process"))
        ]
        return {
            "available": completed.returncode == 0,
            "returncode": int(completed.returncode),
            "matching_lines": lines[-30:],
        }
    except Exception as exc:
        return {
            "available": False,
            "returncode": None,
            "matching_lines": [],
            "exception": f"{type(exc).__name__}: {exc}",
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-path", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--dataset-root", default="/mnt/c/assets/datasets/lerobot_libero")
    parser.add_argument("--hf-home", default="/mnt/c/assets/hf_home")
    parser.add_argument("--vlm-root", default="/mnt/c/assets/hf_home/HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--lora-root", default="/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--resource-cap-gib", type=int, choices=[6, 8, 10, 12, 14], required=True)
    parser.add_argument("--wslconfig-sha256", required=True)
    parser.add_argument("--resource-smoke-output", required=True)
    parser.add_argument("--resource-smoke-md", required=True)
    return parser.parse_args()


def main() -> int:
    import torch
    from lerobot.envs.factory import make_env

    args = parse_args()
    frozen._runtime_environment(args)
    frozen._set_runtime_env(args)
    frozen._set_seed(frozen.BASE_SEED)
    started = time.monotonic()
    before = frozen._resource_snapshot(torch)
    meminfo_before = meminfo()
    kernel_before = kernel_oom_evidence()
    sampler = ResourceSampler(torch)
    sampler.start()

    loaded: dict[str, Any] | None = None
    policy: Any | None = None
    env: Any | None = None
    trace: dict[str, Any] | None = None
    load_audit: dict[str, Any] | None = None
    exception: dict[str, Any] | None = None
    teardown_exceptions: list[str] = []
    environment_constructed = False
    episode_completed = False
    environment_closed = False

    task = frozen.EVAL_TASKS[0]
    init_state_id = frozen.INIT_STATE_IDS[0]
    condition_name = "BASE_STANDARD_E10_D0"
    condition = frozen.BASE_CONDITIONS[condition_name]
    try:
        loaded = frozen._load_policy_and_processors(args, frozen.PolicySpec("frozen_base"))
        policy = loaded["policy"]
        load_audit = dict(loaded["audit"])
        env_cfg = frozen._make_env_cfg(frozen.SUITE, [int(task["task_id"])])
        env = frozen._extract_single_env(
            make_env(env_cfg, n_envs=1, use_async_envs=False),
            frozen.SUITE,
            int(task["task_id"]),
        )
        environment_constructed = True
        with frozen.SmolVLAHiddenCapture(policy) as capture:
            trace = frozen._trace_episode(
                env=env,
                policy=policy,
                capture=capture,
                preprocessor=loaded["preprocessor"],
                postprocessor=loaded["postprocessor"],
                env_preprocessor=loaded["env_preprocessor"],
                condition=condition,
                task_id=int(task["task_id"]),
                init_state_id=int(init_state_id),
                prior=None,
            )
        episode_completed = True
    except Exception as exc:
        exception = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines()[-40:],
        }
    finally:
        if env is not None:
            try:
                env.close()
                environment_closed = True
            except Exception as exc:  # pragma: no cover - actual simulator cleanup
                teardown_exceptions.append(f"env.close {type(exc).__name__}: {exc}")
        try:
            del env
            del policy
            del loaded
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            gc.collect()
        except Exception as exc:  # pragma: no cover - actual runtime cleanup
            teardown_exceptions.append(f"model teardown {type(exc).__name__}: {exc}")

    sampled = sampler.stop()
    after = frozen._resource_snapshot(torch)
    meminfo_after = meminfo()
    kernel_after = kernel_oom_evidence()
    before_lines = set(kernel_before.get("matching_lines") or [])
    new_kernel_oom_lines = [
        line for line in (kernel_after.get("matching_lines") or []) if line not in before_lines
    ]
    teardown_success = bool(environment_closed and not teardown_exceptions)
    trace_evidence = None
    if trace is not None:
        trace_evidence = {
            "episode_length": int(trace["episode_length"]),
            "simulator_step_count": int(trace["episode_length"]),
            "action_finite": bool(trace["action_finite"]),
            "base_model_forward_count": int(trace["base_model_forward_count"]),
            "prior_module_forward_count": int(trace["prior_module_forward_count"]),
            "elapsed_seconds": trace["elapsed_seconds"],
            "peak_vram": trace["peak_vram"],
            "rss_mb_at_trace_return": trace["rss_mb"],
            "task_success_persisted": False,
            "reward_persisted": False,
        }
    no_cpu_or_disk_offload = bool(
        load_audit
        and str(load_audit.get("parameter", {}).get("device")) == "cuda:0"
        and all(str(device) == "cuda:0" for device in load_audit.get("input_tensor_devices", {}).values())
    )
    internal_pass = bool(
        environment_constructed
        and episode_completed
        and trace_evidence is not None
        and trace_evidence["simulator_step_count"] > 0
        and trace_evidence["base_model_forward_count"] > 0
        and trace_evidence["action_finite"]
        and teardown_success
        and exception is None
        and not new_kernel_oom_lines
        and int(meminfo_after.get("swap_total_bytes") or 0) == 0
        and float(sampled["peak_wsl_used_fraction"]) <= frozen.RAM_LIMIT_FRACTION
        and no_cpu_or_disk_offload
    )
    report = {
        "schema_version": frozen.SCHEMA_VERSION,
        "date": f"{frozen.DATE_KST} KST",
        "execution_type": "VLA_INFERENCE",
        "purpose": "RESOURCE_ONLY_ACTUAL_PATH_SMOKE",
        "fidelity_label": frozen.FIDELITY_LABEL,
        "official_commit": frozen.OFFICIAL_COMMIT,
        "frozen_runner_path": "scripts/run_a2c2_problem_verification.py",
        "resource_cap_gib_requested": int(args.resource_cap_gib),
        "wslconfig_sha256": str(args.wslconfig_sha256),
        "scientific_protocol_changed": False,
        "scientific_episode_row_persisted": False,
        "task_success_persisted": False,
        "task_success_counted": False,
        "reward_persisted": False,
        "ours_designed_or_executed": False,
        "prior_retrained": False,
        "condition": condition_name,
        "task_id": int(task["task_id"]),
        "global_task_index": int(task["global_task_index"]),
        "official_init_state_id": int(init_state_id),
        "execution_horizon": int(condition["execution_horizon"]),
        "inference_delay": int(condition["inference_delay"]),
        "max_steps": frozen.MAX_STEPS,
        "environment_constructed": environment_constructed,
        "episode_completed": episode_completed,
        "trace_evidence_without_outcome": trace_evidence,
        "base_policy_load_audit": load_audit,
        "no_cpu_or_disk_model_offload": no_cpu_or_disk_offload,
        "before_resources": before,
        "peak_resources": sampled,
        "after_resources": after,
        "meminfo_before": meminfo_before,
        "meminfo_after": meminfo_after,
        "kernel_oom_before": kernel_before,
        "kernel_oom_after": kernel_after,
        "new_kernel_oom_lines": new_kernel_oom_lines,
        "teardown": {
            "environment_closed": environment_closed,
            "success": teardown_success,
            "exceptions": teardown_exceptions,
        },
        "exception": exception,
        "elapsed_seconds": frozen._round(time.monotonic() - started, 3),
        "internal_pass": internal_pass,
        "final_decision": (
            "A2C2_RESOURCE_SMOKE_INTERNAL_PASS"
            if internal_pass
            else "A2C2_RESOURCE_SMOKE_INTERNAL_FAIL"
        ),
    }
    frozen._write_json(Path(args.resource_smoke_output), report)
    frozen._write_md(Path(args.resource_smoke_md), "A2C2 Resource-Only Actual-Path Smoke", report)
    print(json.dumps({"final_decision": report["final_decision"]}, indent=2))
    return 0 if internal_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
