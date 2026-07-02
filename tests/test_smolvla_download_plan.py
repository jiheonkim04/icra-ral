import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "14_plan_smolvla_download.ps1"
BASH_SKIP_REASON = "No usable GNU Bash found; skipping Bash planner test on this Windows environment."


def _is_windowsapps_bash(path: str) -> bool:
    normalized = path.replace("/", "\\").lower()
    return "\\windowsapps\\" in normalized and normalized.endswith("\\bash.exe")


def _validate_bash(candidate: str) -> bool:
    if _is_windowsapps_bash(candidate):
        return False
    try:
        result = subprocess.run(
            [candidate, "--version"],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False

    output = f"{result.stdout}\n{result.stderr}".lower()
    return result.returncode == 0 and "bash" in output and "gnu bash" in output


def find_usable_bash() -> str | None:
    candidates: list[str] = []

    env_bash = os.environ.get("BASH_EXE")
    if env_bash:
        candidates.append(env_bash)

    path_bash = shutil.which("bash")
    if path_bash:
        candidates.append(path_bash)

    for common_path in (
        r"C:\Program Files\Git\bin\bash.exe",
        r"C:\Program Files\Git\usr\bin\bash.exe",
    ):
        if Path(common_path).exists():
            candidates.append(common_path)

    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(candidate))
        if key in seen:
            continue
        seen.add(key)
        if _validate_bash(candidate):
            return candidate
    return None


def _run_download_plan(tmp_path: Path, allow_downloads: bool = False) -> dict:
    powershell = shutil.which("powershell")
    if powershell is None:
        pytest.skip("PowerShell is required for the SmolVLA download planner")

    asset_root = tmp_path / "assets"
    paths_file = tmp_path / "paths.local.yaml"
    paths_file.write_text(
        "\n".join(
            [
                "assets:",
                f'  smolvla_ckpt: "{(asset_root / "checkpoints" / "smolvla").as_posix()}"',
                f'  checkpoint_root: "{(asset_root / "checkpoints").as_posix()}"',
                f'  hf_home: "{(asset_root / "hf_home").as_posix()}"',
            ]
        ),
        encoding="utf-8",
    )

    env = os.environ.copy()
    for key in ("SMOLVLA_CKPT", "CHECKPOINT_ROOT", "HF_HOME"):
        env[key] = ""
    if allow_downloads:
        env["ALLOW_DOWNLOADS"] = "1"
    else:
        env.pop("ALLOW_DOWNLOADS", None)

    result = subprocess.run(
        [
            powershell,
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(SCRIPT),
            "-PathsFile",
            str(paths_file),
            "-AssetRoot",
            str(asset_root),
        ],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    start = result.stdout.find("{")
    assert start >= 0, result.stdout
    report = json.loads(result.stdout[start:])
    assert not (asset_root / "checkpoints" / "smolvla").exists()
    assert not (asset_root / "hf_home").exists()
    return report


def test_smolvla_download_plan_is_dry_run_only(tmp_path):
    report = _run_download_plan(tmp_path, allow_downloads=False)

    assert report["policy"]["dry_run_only"] is True
    assert report["policy"]["allow_downloads_gate_set"] is False
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["directories_created"] is False
    assert report["policy"]["gpu_jobs_performed"] is False
    assert report["policy"]["training_performed"] is False
    assert report["policy"]["real_rollouts_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
    assert report["readiness_semantics"]["path_ready_is_not_adapter_smoke_ready"] is True
    assert "config.json" in report["required_files"]["config"]
    assert "model.safetensors" in report["required_files"]["weights_any"]


def test_allow_downloads_gate_does_not_make_planner_download(tmp_path):
    report = _run_download_plan(tmp_path, allow_downloads=True)

    assert report["policy"]["allow_downloads_gate_set"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["directories_created"] is False


def test_bash_smolvla_download_plan_outputs_valid_dry_run_json():
    bash = find_usable_bash()
    if bash is None:
        pytest.skip(BASH_SKIP_REASON)

    env = os.environ.copy()
    env.pop("ALLOW_DOWNLOADS", None)
    result = subprocess.run(
        [bash, "scripts/14_plan_smolvla_download.sh"],
        cwd=REPO_ROOT,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=True,
    )
    start = result.stdout.find("{")
    assert start >= 0, result.stdout
    report = json.loads(result.stdout[start:])

    assert report["policy"]["dry_run_only"] is True
    assert report["policy"]["downloads_performed"] is False
    assert report["policy"]["heavy_model_imports_performed"] is False
    assert report["policy"]["openvla_oft_executed"] is False
