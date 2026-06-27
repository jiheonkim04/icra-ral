#!/usr/bin/env bash
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p reports

PYTHON_BIN="${PYTHON:-python}"

"$PYTHON_BIN" - <<'PY'
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

repo_root = Path.cwd()
reports_dir = repo_root / "reports"
reports_dir.mkdir(exist_ok=True)


def run(command):
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=15,
            check=False,
        )
        return {
            "available": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": (completed.stdout or "").strip(),
            "stderr": (completed.stderr or "").strip(),
        }
    except FileNotFoundError:
        return {"available": False, "warning": f"Command not found: {command[0]}"}
    except subprocess.TimeoutExpired:
        return {"available": False, "warning": f"Command timed out: {command[0]}"}
    except Exception as exc:
        return {"available": False, "warning": str(exc)}


def total_ram_gb():
    try:
        if hasattr(os, "sysconf"):
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return round(pages * page_size / (1024 ** 3), 2)
    except Exception:
        pass
    return None


def env_status(name):
    value = os.environ.get(name)
    return {"configured": bool(value), "value_redacted": "set" if value else None}

try:
    import torch
    torch_report = {
        "import_ok": True,
        "version": getattr(torch, "__version__", None),
        "cuda_is_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
    }
except Exception as exc:
    torch_report = {
        "import_ok": False,
        "error": str(exc),
        "cuda_is_available": False,
        "torch_cuda_version": None,
    }

root_disk = shutil.disk_usage(repo_root)
report = {
    "policy": {
        "downloads_performed": False,
        "gpu_training_performed": False,
        "heavy_vla_imports_performed": False,
        "real_rollouts_performed": False,
    },
    "os": {
        "system": platform.system(),
        "release": platform.release(),
        "version": platform.version(),
        "platform": platform.platform(),
    },
    "cpu": platform.processor() or None,
    "total_system_ram_gb": total_ram_gb(),
    "free_disk": [
        {
            "path": str(repo_root),
            "free_gb": round(root_disk.free / (1024 ** 3), 2),
            "total_gb": round(root_disk.total / (1024 ** 3), 2),
        }
    ],
    "gpu": run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]),
    "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
    "python": {
        "executable": sys.executable,
        "version": sys.version.split()[0],
    },
    "torch": torch_report,
    "wsl2": {
        "status": run(["wsl", "--status"]),
        "distros": run(["wsl", "--list", "--verbose"]),
    },
    "paths": {
        "paths_local_yaml_exists": (repo_root / "configs" / "paths.local.yaml").exists(),
        "HF_HOME": env_status("HF_HOME"),
        "CHECKPOINT_ROOT": env_status("CHECKPOINT_ROOT"),
        "DATA_ROOT": env_status("DATA_ROOT"),
    },
}

out_path = reports_dir / "system_readiness.json"
out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
print(json.dumps(report, indent=2))
PY
