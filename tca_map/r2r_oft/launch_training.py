"""Detached WSL launcher for the frozen R2R-OFT training arms."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tca_map.r2r_oft.training_spec import SPEC_ARTIFACT, build_epoch5_training_spec, write_training_spec
from tca_map.r2r_oft.train_qlora import _arm_by_id, _git_commit, _json_default, _write_json


DEFAULT_WSL_DISTRO = "Ubuntu-22.04"
DEFAULT_WSL_PYTHON = "/home/jiheon/venvs/openvla-oft-int4-rtx5080/bin/python"
DEFAULT_OUTPUT_ROOT = Path("runs/openvla_oft_int4/epoch5_r2r_oft_training")


@dataclass(frozen=True)
class LaunchConfig:
    arm_id: str
    repo_root: Path
    spec_path: Path = SPEC_ARTIFACT
    output_root: Path = DEFAULT_OUTPUT_ROOT
    wsl_distro: str = DEFAULT_WSL_DISTRO
    wsl_python: str = DEFAULT_WSL_PYTHON
    max_steps_override: int | None = None
    dry_run: bool = False


def _run_capture(command: list[str]) -> str:
    result = subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return result.stdout.strip()


def _wslpath(config: LaunchConfig, path: Path) -> str:
    windows_path = str(path.resolve()).replace("\\", "/")
    return _run_capture(["wsl", "-d", config.wsl_distro, "wslpath", "-a", windows_path])


def build_launch_command(config: LaunchConfig) -> dict[str, Any]:
    """Build the WSL command and deterministic launch paths."""

    spec = build_epoch5_training_spec()
    arm = _arm_by_id(spec, config.arm_id)
    repo_wsl = _wslpath(config, config.repo_root)
    output_root_wsl = _wslpath(config, config.repo_root / config.output_root)
    spec_rel = config.spec_path.as_posix()
    output_rel = config.output_root.as_posix()
    arm_dir_rel = f"{output_rel}/{config.arm_id}"
    max_steps_args = ""
    if config.max_steps_override is not None:
        max_steps_args = f" --max-steps-override {int(config.max_steps_override)}"

    inner = (
        f"cd {shlex.quote(repo_wsl)} && "
        "export TOKENIZERS_PARALLELISM=false TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONUNBUFFERED=1 && "
        f"{shlex.quote(config.wsl_python)} -m tca_map.r2r_oft.train_qlora "
        f"--spec {shlex.quote(spec_rel)} "
        f"--arm-id {shlex.quote(config.arm_id)} "
        f"--output-root {shlex.quote(output_rel)}"
        f"{max_steps_args}; "
        "code=$?; "
        f"echo $code > {shlex.quote(arm_dir_rel + '/exit_code.txt')}; "
        "exit $code"
    )
    launcher_stdout = f"{arm_dir_rel}/launcher_stdout.log"
    launcher_stderr = f"{arm_dir_rel}/launcher_stderr.log"
    worker_pid = f"{arm_dir_rel}/worker_pid.txt"
    popen_args = ["wsl", "-d", config.wsl_distro, "bash", "-lc", inner]
    return {
        "arm": arm,
        "repo_wsl": repo_wsl,
        "output_root_wsl": output_root_wsl,
        "inner_command": inner,
        "popen_args": popen_args,
        "launcher_stdout": str(config.output_root / config.arm_id / "launcher_stdout.log"),
        "launcher_stderr": str(config.output_root / config.arm_id / "launcher_stderr.log"),
        "worker_pid": str(config.output_root / config.arm_id / "worker_pid.txt"),
        "trainer_heartbeat": str(config.output_root / config.arm_id / "heartbeat.json"),
        "trainer_status": str(config.output_root / config.arm_id / "status.json"),
        "trainer_result": str(config.output_root / config.arm_id / "result.json"),
    }


def launch_training_arm(config: LaunchConfig) -> dict[str, Any]:
    """Generate spec artifact and launch one arm detached in WSL."""

    config.output_root.mkdir(parents=True, exist_ok=True)
    write_training_spec(config.spec_path)
    launch = build_launch_command(config)
    arm_dir = config.output_root / config.arm_id
    arm_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 1,
        "status": "DRY_RUN" if config.dry_run else "LAUNCHING",
        "arm_id": config.arm_id,
        "role": launch["arm"]["role"],
        "wsl_distro": config.wsl_distro,
        "wsl_python": config.wsl_python,
        "spec_path": str(config.spec_path),
        "output_root": str(config.output_root),
        "git_commit": _git_commit(),
        "created_unix": time.time(),
        "training_happened_at_launch_manifest_write": False,
        "optimizer_step_happened_at_launch_manifest_write": False,
        "launch_strategy": "windows_subprocess_popen_wsl_foreground_trainer",
        "inner_command": launch["inner_command"],
        "popen_args": launch["popen_args"],
        "paths": {
            key: launch[key]
            for key in (
                "launcher_stdout",
                "launcher_stderr",
                "worker_pid",
                "trainer_heartbeat",
                "trainer_status",
                "trainer_result",
            )
        },
    }
    manifest_path = arm_dir / "launch_manifest.json"
    _write_json(manifest_path, manifest)
    _write_json(
        arm_dir / "launch_heartbeat.json",
        {
            "status": manifest["status"].lower(),
            "arm_id": config.arm_id,
            "time_unix": time.time(),
            "git_commit": manifest["git_commit"],
            "training_happened": False,
            "optimizer_step_happened": False,
            "manifest_path": str(manifest_path),
        },
    )
    if config.dry_run:
        return {**manifest, "manifest_path": str(manifest_path), "worker_pid_value": None}

    stdout_path = arm_dir / "launcher_stdout.log"
    stderr_path = arm_dir / "launcher_stderr.log"
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
    (arm_dir / "worker_pid.txt").write_text(pid + "\n", encoding="utf-8")
    launched = {**manifest, "status": "LAUNCHED", "worker_pid_value": pid, "manifest_path": str(manifest_path)}
    _write_json(manifest_path, launched)
    _write_json(
        arm_dir / "launch_heartbeat.json",
        {
            "status": "launched",
            "arm_id": config.arm_id,
            "worker_pid": pid,
            "time_unix": time.time(),
            "git_commit": manifest["git_commit"],
            "training_happened": False,
            "optimizer_step_happened": False,
            "manifest_path": str(manifest_path),
        },
    )
    return launched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm-id", required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--spec", type=Path, default=SPEC_ARTIFACT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--wsl-distro", default=DEFAULT_WSL_DISTRO)
    parser.add_argument("--wsl-python", default=DEFAULT_WSL_PYTHON)
    parser.add_argument("--max-steps-override", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = launch_training_arm(
        LaunchConfig(
            arm_id=args.arm_id,
            repo_root=args.repo_root,
            spec_path=args.spec,
            output_root=args.output_root,
            wsl_distro=args.wsl_distro,
            wsl_python=args.wsl_python,
            max_steps_override=args.max_steps_override,
            dry_run=bool(args.dry_run),
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
