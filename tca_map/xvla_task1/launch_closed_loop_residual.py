"""Detached WSL launcher for the frozen BR-XVLA residual-manifest screen."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tca_map.xvla_task1.closed_loop_residual_eval import (
    DEFAULT_IDENTITIES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_POLICY_LABELS,
    DEFAULT_TASK_ID,
    DEFAULT_TASK_SUITE_NAME,
)
from tca_map.xvla_task1.data_adapter_smoke import TASK_DESCRIPTION
from tca_map.xvla_task1.train_lora import DEFAULT_OUTPUT_ROOT as DEFAULT_TRAINING_OUTPUT_ROOT
from tca_map.xvla_task1.train_lora import _git_commit, _json_default, _write_json
from tca_map.xvla_task1.training_spec import SPEC_ARTIFACT

DEFAULT_WSL_DISTRO = "Ubuntu-22.04"
DEFAULT_WSL_PYTHON = "/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"


@dataclass(frozen=True)
class LaunchClosedLoopConfig:
    repo_root: Path
    spec_path: Path = SPEC_ARTIFACT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    training_output_root: Path = DEFAULT_TRAINING_OUTPUT_ROOT
    wsl_distro: str = DEFAULT_WSL_DISTRO
    wsl_python: str = DEFAULT_WSL_PYTHON
    task_suite: str = DEFAULT_TASK_SUITE_NAME
    task_id: int = DEFAULT_TASK_ID
    task_description: str = TASK_DESCRIPTION
    identities: tuple[int, ...] = DEFAULT_IDENTITIES
    policies: tuple[str, ...] = DEFAULT_POLICY_LABELS
    eval_horizon: int = 900
    settle_steps: int = 10
    denoise_steps: int = 10
    dry_run: bool = False


def _run_capture(command: list[str]) -> str:
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


def _wslpath(config: LaunchClosedLoopConfig, path: Path) -> str:
    windows_path = str(path.resolve()).replace("\\", "/")
    return _run_capture(["wsl", "-d", config.wsl_distro, "wslpath", "-a", windows_path])


def build_launch_command(config: LaunchClosedLoopConfig) -> dict[str, Any]:
    repo_wsl = _wslpath(config, config.repo_root)
    identities = ",".join(str(identity) for identity in config.identities)
    policies = ",".join(str(label) for label in config.policies)
    inner = (
        f"cd {shlex.quote(repo_wsl)} && "
        "export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 && "
        f"{shlex.quote(config.wsl_python)} -m tca_map.xvla_task1.closed_loop_residual_eval "
        f"--spec {shlex.quote(config.spec_path.as_posix())} "
        f"--output-root {shlex.quote(config.output_root.as_posix())} "
        f"--training-output-root {shlex.quote(config.training_output_root.as_posix())} "
        f"--task-suite {shlex.quote(str(config.task_suite))} "
        f"--task-id {int(config.task_id)} "
        f"--task-description {shlex.quote(str(config.task_description))} "
        f"--identities {shlex.quote(identities)} "
        f"--policies {shlex.quote(policies)} "
        f"--eval-horizon {int(config.eval_horizon)} "
        f"--settle-steps {int(config.settle_steps)} "
        f"--denoise-steps {int(config.denoise_steps)}; "
        "status=$?; "
        f"printf '%s\\n' \"$status\" > {shlex.quote((config.output_root / 'closed_loop_exit_code.txt').as_posix())}; "
        "exit \"$status\""
    )
    return {
        "repo_wsl": repo_wsl,
        "inner_command": inner,
        "popen_args": ["wsl", "-d", config.wsl_distro, "bash", "-lc", inner],
        "paths": {
            "launcher_stdout": str(config.output_root / "closed_loop_launcher_stdout.log"),
            "launcher_stderr": str(config.output_root / "closed_loop_launcher_stderr.log"),
            "worker_pid": str(config.output_root / "closed_loop_worker_pid.txt"),
            "launch_heartbeat": str(config.output_root / "closed_loop_launch_heartbeat.json"),
            "eval_heartbeat": str(config.output_root / "closed_loop_heartbeat.json"),
            "eval_status": str(config.output_root / "closed_loop_status.json"),
            "frozen_manifest": str(config.output_root / "closed_loop_manifest.json"),
            "eval_result": str(config.output_root / "closed_loop_result.json"),
            "exit_code": str(config.output_root / "closed_loop_exit_code.txt"),
            "exact_resume_command": str(config.output_root / "closed_loop_exact_resume_command.txt"),
        },
    }


def launch_closed_loop_residual(config: LaunchClosedLoopConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    launch = build_launch_command(config)
    manifest_path = config.output_root / "closed_loop_launch_manifest.json"
    resume_command = " ".join(shlex.quote(part) for part in launch["popen_args"])
    manifest = {
        "schema_version": "2026-07-17.epoch5_br_xvla_closed_loop_launch.v1",
        "status": "DRY_RUN" if config.dry_run else "LAUNCHING",
        "method": "BR-XVLA",
        "stage": "epoch_5_br_xvla_closed_loop_residual_manifest",
        "wsl_distro": config.wsl_distro,
        "wsl_python": config.wsl_python,
        "spec_path": str(config.spec_path),
        "output_root": str(config.output_root),
        "training_output_root": str(config.training_output_root),
        "git_commit": _git_commit(),
        "created_unix": time.time(),
        "task_suite": str(config.task_suite),
        "task_id": int(config.task_id),
        "task_description": str(config.task_description),
        "identities": [int(identity) for identity in config.identities],
        "policies": [str(policy) for policy in config.policies],
        "eval_horizon": int(config.eval_horizon),
        "settle_steps": int(config.settle_steps),
        "denoise_steps": int(config.denoise_steps),
        "training_happened_at_launch_manifest_write": False,
        "optimizer_step_happened_at_launch_manifest_write": False,
        "checkpoint_written_at_launch_manifest_write": False,
        "closed_loop_ours_evaluation_happened_at_launch_manifest_write": False,
        "retuning_from_result_allowed": False,
        "broader_confirmatory_evaluation_allowed": False,
        "launch_strategy": "windows_subprocess_popen_wsl_foreground_closed_loop_eval",
        "inner_command": launch["inner_command"],
        "popen_args": launch["popen_args"],
        "paths": launch["paths"],
        "exact_resume_command": resume_command,
    }
    _write_json(manifest_path, manifest)
    (config.output_root / "closed_loop_exact_resume_command.txt").write_text(resume_command + "\n", encoding="utf-8")
    _write_json(
        config.output_root / "closed_loop_launch_heartbeat.json",
        {
            "status": manifest["status"].lower(),
            "method": "BR-XVLA",
            "stage": manifest["stage"],
            "time_unix": time.time(),
            "git_commit": manifest["git_commit"],
            "training_happened": False,
            "optimizer_step_happened": False,
            "closed_loop_ours_evaluation_happened": False,
            "manifest_path": str(manifest_path),
        },
    )
    if config.dry_run:
        return {**manifest, "manifest_path": str(manifest_path), "worker_pid_value": None}

    stdout_path = config.output_root / "closed_loop_launcher_stdout.log"
    stderr_path = config.output_root / "closed_loop_launcher_stderr.log"
    stdout_handle = stdout_path.open("ab")
    stderr_handle = stderr_path.open("ab")
    try:
        proc = subprocess.Popen(
            launch["popen_args"],
            cwd=str(config.repo_root),
            stdout=stdout_handle,
            stderr=stderr_handle,
            stdin=subprocess.DEVNULL,
            close_fds=True,
        )
    finally:
        stdout_handle.close()
        stderr_handle.close()
    pid = str(proc.pid)
    (config.output_root / "closed_loop_worker_pid.txt").write_text(pid + "\n", encoding="utf-8")
    launched = {**manifest, "status": "LAUNCHED", "worker_pid_value": pid, "manifest_path": str(manifest_path)}
    _write_json(manifest_path, launched)
    _write_json(
        config.output_root / "closed_loop_launch_heartbeat.json",
        {
            "status": "launched",
            "method": "BR-XVLA",
            "stage": manifest["stage"],
            "worker_pid": pid,
            "time_unix": time.time(),
            "git_commit": manifest["git_commit"],
            "training_happened": False,
            "optimizer_step_happened": False,
            "closed_loop_ours_evaluation_happened": False,
            "manifest_path": str(manifest_path),
        },
    )
    return launched


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--spec", type=Path, default=SPEC_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--training-output-root", type=Path, default=DEFAULT_TRAINING_OUTPUT_ROOT)
    parser.add_argument("--wsl-distro", default=DEFAULT_WSL_DISTRO)
    parser.add_argument("--wsl-python", default=DEFAULT_WSL_PYTHON)
    parser.add_argument("--task-suite", default=DEFAULT_TASK_SUITE_NAME)
    parser.add_argument("--task-id", type=int, default=DEFAULT_TASK_ID)
    parser.add_argument("--task-description", default=TASK_DESCRIPTION)
    parser.add_argument("--identities", default=",".join(str(identity) for identity in DEFAULT_IDENTITIES))
    parser.add_argument("--policies", default=",".join(DEFAULT_POLICY_LABELS))
    parser.add_argument("--eval-horizon", type=int, default=900)
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = launch_closed_loop_residual(
        LaunchClosedLoopConfig(
            repo_root=args.repo_root,
            spec_path=args.spec,
            output_root=args.output_root,
            training_output_root=args.training_output_root,
            wsl_distro=str(args.wsl_distro),
            wsl_python=str(args.wsl_python),
            task_suite=str(args.task_suite),
            task_id=int(args.task_id),
            task_description=str(args.task_description),
            identities=tuple(int(item.strip()) for item in str(args.identities).split(",") if item.strip()),
            policies=tuple(str(item.strip()) for item in str(args.policies).split(",") if item.strip()),
            eval_horizon=int(args.eval_horizon),
            settle_steps=int(args.settle_steps),
            denoise_steps=int(args.denoise_steps),
            dry_run=bool(args.dry_run),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
