"""Detached WSL launcher for the BR-XVLA bounded training gate."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tca_map.xvla_task1.train_lora import DEFAULT_OUTPUT_ROOT, _git_commit, _json_default, _write_json
from tca_map.xvla_task1.offline_validate import DEFAULT_OUTPUT
from tca_map.xvla_task1.training_spec import SPEC_ARTIFACT

DEFAULT_WSL_DISTRO = "Ubuntu-22.04"
DEFAULT_WSL_PYTHON = "/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python"


@dataclass(frozen=True)
class LaunchGateConfig:
    repo_root: Path
    spec_path: Path = SPEC_ARTIFACT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    offline_output: Path = DEFAULT_OUTPUT
    wsl_distro: str = DEFAULT_WSL_DISTRO
    wsl_python: str = DEFAULT_WSL_PYTHON
    max_steps_override: int | None = None
    num_validation_chunks: int = 24
    denoise_steps: int = 10
    dry_run: bool = False


def _run_capture(command: list[str]) -> str:
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


def _wslpath(config: LaunchGateConfig, path: Path) -> str:
    windows_path = str(path.resolve()).replace("\\", "/")
    return _run_capture(["wsl", "-d", config.wsl_distro, "wslpath", "-a", windows_path])


def build_launch_command(config: LaunchGateConfig) -> dict[str, Any]:
    repo_wsl = _wslpath(config, config.repo_root)
    max_steps_args = ""
    if config.max_steps_override is not None:
        max_steps_args = f" --max-steps-override {int(config.max_steps_override)}"
    inner = (
        f"cd {shlex.quote(repo_wsl)} && "
        "export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1 && "
        f"{shlex.quote(config.wsl_python)} -m tca_map.xvla_task1.training_gate "
        f"--spec {shlex.quote(config.spec_path.as_posix())} "
        f"--output-root {shlex.quote(config.output_root.as_posix())} "
        f"--offline-output {shlex.quote(config.offline_output.as_posix())} "
        f"--num-validation-chunks {int(config.num_validation_chunks)} "
        f"--denoise-steps {int(config.denoise_steps)}"
        f"{max_steps_args}; "
        "code=$?; "
        f"echo $code > {shlex.quote((config.output_root / 'gate_exit_code.txt').as_posix())}; "
        "exit $code"
    )
    return {
        "repo_wsl": repo_wsl,
        "inner_command": inner,
        "popen_args": ["wsl", "-d", config.wsl_distro, "bash", "-lc", inner],
        "paths": {
            "launcher_stdout": str(config.output_root / "gate_launcher_stdout.log"),
            "launcher_stderr": str(config.output_root / "gate_launcher_stderr.log"),
            "worker_pid": str(config.output_root / "gate_worker_pid.txt"),
            "launch_heartbeat": str(config.output_root / "gate_launch_heartbeat.json"),
            "gate_heartbeat": str(config.output_root / "gate_heartbeat.json"),
            "gate_status": str(config.output_root / "gate_status.json"),
            "gate_result": str(config.output_root / "gate_result.json"),
            "offline_result": str(config.offline_output),
            "exit_code": str(config.output_root / "gate_exit_code.txt"),
            "exact_resume_command": str(config.output_root / "gate_exact_resume_command.txt"),
        },
    }


def launch_training_gate(config: LaunchGateConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    launch = build_launch_command(config)
    manifest_path = config.output_root / "gate_launch_manifest.json"
    resume_command = " ".join(shlex.quote(part) for part in launch["popen_args"])
    manifest = {
        "schema_version": "2026-07-17.epoch5_br_xvla_training_gate_launch.v1",
        "status": "DRY_RUN" if config.dry_run else "LAUNCHING",
        "method": "BR-XVLA",
        "wsl_distro": config.wsl_distro,
        "wsl_python": config.wsl_python,
        "spec_path": str(config.spec_path),
        "output_root": str(config.output_root),
        "offline_output": str(config.offline_output),
        "git_commit": _git_commit(),
        "created_unix": time.time(),
        "training_happened_at_launch_manifest_write": False,
        "optimizer_step_happened_at_launch_manifest_write": False,
        "closed_loop_ours_evaluation_happened_at_launch_manifest_write": False,
        "max_steps_override": config.max_steps_override,
        "num_validation_chunks": int(config.num_validation_chunks),
        "denoise_steps": int(config.denoise_steps),
        "launch_strategy": "windows_subprocess_popen_wsl_foreground_gate",
        "inner_command": launch["inner_command"],
        "popen_args": launch["popen_args"],
        "paths": launch["paths"],
        "exact_resume_command": resume_command,
    }
    _write_json(manifest_path, manifest)
    (config.output_root / "gate_exact_resume_command.txt").write_text(resume_command + "\n", encoding="utf-8")
    _write_json(
        config.output_root / "gate_launch_heartbeat.json",
        {
            "status": manifest["status"].lower(),
            "method": "BR-XVLA",
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

    stdout_path = config.output_root / "gate_launcher_stdout.log"
    stderr_path = config.output_root / "gate_launcher_stderr.log"
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
    (config.output_root / "gate_worker_pid.txt").write_text(pid + "\n", encoding="utf-8")
    launched = {**manifest, "status": "LAUNCHED", "worker_pid_value": pid, "manifest_path": str(manifest_path)}
    _write_json(manifest_path, launched)
    _write_json(
        config.output_root / "gate_launch_heartbeat.json",
        {
            "status": "launched",
            "method": "BR-XVLA",
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
    parser.add_argument("--offline-output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--wsl-distro", default=DEFAULT_WSL_DISTRO)
    parser.add_argument("--wsl-python", default=DEFAULT_WSL_PYTHON)
    parser.add_argument("--max-steps-override", type=int, default=None)
    parser.add_argument("--num-validation-chunks", type=int, default=24)
    parser.add_argument("--denoise-steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    result = launch_training_gate(
        LaunchGateConfig(
            repo_root=args.repo_root,
            spec_path=args.spec,
            output_root=args.output_root,
            offline_output=args.offline_output,
            wsl_distro=str(args.wsl_distro),
            wsl_python=str(args.wsl_python),
            max_steps_override=args.max_steps_override,
            num_validation_chunks=int(args.num_validation_chunks),
            denoise_steps=int(args.denoise_steps),
            dry_run=bool(args.dry_run),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
