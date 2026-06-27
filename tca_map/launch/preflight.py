"""Safe repository-local preflight.

This module does not download assets, launch GPU training, or run rollouts.
"""

from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from tca_map.datasets import make_counterfactual_pairs, make_dummy_samples
from tca_map.heads import TCAMapHead
from tca_map.models import DummyAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
CONFIG_PATH = REPO_ROOT / "configs" / "paths.local.yaml"

ENV_KEYS = {
    "openvla_oft_ckpt": "OPENVLA_OFT_CKPT",
    "smolvla_ckpt": "SMOLVLA_CKPT",
    "libero_root": "LIBERO_ROOT",
    "libero_data_root": "LIBERO_DATA_ROOT",
    "robosuite_root": "ROBOSUITE_ROOT",
    "data_root": "DATA_ROOT",
    "checkpoint_root": "CHECKPOINT_ROOT",
    "hf_home": "HF_HOME",
    "wandb_api_key": "WANDB_API_KEY",
}


def _read_simple_paths_config() -> dict[str, str]:
    if not CONFIG_PATH.exists():
        return {}
    values: dict[str, str] = {}
    for raw_line in CONFIG_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or line == "assets:":
            continue
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if value and value.lower() != "null":
                values[key] = value
    return values


def _command_output(command: list[str]) -> dict:
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
            check=False,
        )
        stdout = (completed.stdout or "").strip()
        stderr = (completed.stderr or "").strip()
        return {
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": stdout,
            "stderr": stderr,
        }
    except FileNotFoundError:
        return {"available": False, "warning": f"Command not found: {command[0]}"}
    except subprocess.TimeoutExpired as exc:
        stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        stderr = (exc.stderr or "") if isinstance(exc.stderr, str) else ""
        return {
            "available": False,
            "warning": f"Command timed out: {command[0]}",
            "stdout": stdout.strip(),
            "stderr": stderr.strip(),
        }
    except Exception as exc:  # pragma: no cover - defensive environment probe
        return {"available": False, "warning": str(exc)}


def _check_assets() -> tuple[dict, list[str]]:
    configured = _read_simple_paths_config()
    status: dict[str, dict] = {}
    missing: list[str] = []
    for key, env_name in ENV_KEYS.items():
        value = os.environ.get(env_name) or configured.get(key)
        exists = bool(value) and (key == "wandb_api_key" or Path(value).exists())
        status[key] = {
            "env": env_name,
            "configured": bool(value),
            "exists": bool(exists),
            "value_redacted": "set" if value else None,
        }
        if key != "wandb_api_key" and not exists:
            missing.append(key)
    return status, missing


def _write_missing_assets_runtime(missing: list[str], asset_status: dict) -> None:
    runtime_report = {
        "local_paths_only": True,
        "downloads_performed": False,
        "missing_assets": missing,
        "assets": asset_status,
        "setup": {
            "config_file": "configs/paths.local.yaml",
            "template": "configs/paths.local.yaml.example",
            "environment_variables": list(ENV_KEYS.values()),
        },
    }
    (REPORTS_DIR / "missing_assets_runtime.json").write_text(
        json.dumps(runtime_report, indent=2),
        encoding="utf-8",
    )


def run_preflight() -> dict:
    REPORTS_DIR.mkdir(exist_ok=True)
    samples = make_dummy_samples(count=2)
    pairs = make_counterfactual_pairs(samples)
    adapter = DummyAdapter()
    encoded = adapter.encode(samples[0]["observation"], samples[0]["instruction"])
    tca = TCAMapHead(grid_size=8).predict(encoded["tokens"], samples[0]["observation"]["candidate_objects"])

    asset_status, missing = _check_assets()
    nvidia = _command_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    git = _command_output(["git", "rev-parse", "HEAD"])
    bash = _command_output(["bash", "--version"])
    disk = shutil.disk_usage(REPO_ROOT)

    report = {
        "safe_to_run_dummy_smoke": True,
        "safe_to_run_real_adapter_gpu": False,
        "safe_to_run_rollouts": False,
        "policy": {
            "local_paths_only": True,
            "downloads_allowed": False,
            "gpu_training_allowed_in_scaffold": False,
            "real_rollouts_allowed_in_scaffold": False,
            "offline_proxy_is_not_standard_success": True,
        },
        "python": {"executable": sys.executable, "version": sys.version.split()[0], "platform": platform.platform()},
        "git": git,
        "nvidia_smi": nvidia,
        "bash": bash,
        "disk_free_gb": round(disk.free / (1024 ** 3), 3),
        "assets": asset_status,
        "missing_assets": missing,
        "dummy_dataset_count": len(samples),
        "counterfactual_pair_count": len(pairs),
        "dummy_adapter_ok": bool(encoded["tokens"]),
        "tca_map_head_ok": "action" in tca and "target" in tca,
        "heatmap_grid_size": 8,
        "estimated_heatmap_cells": 8 ** 3,
        "uses_privileged_state_at_default_inference": False,
    }

    _write_missing_assets_runtime(missing, asset_status)
    (REPORTS_DIR / "preflight_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    run_preflight()
