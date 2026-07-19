"""Run the frozen Epoch 6 outcome-suppressed schedule-invariance audit.

This runner deliberately has no simulator-step or outcome-reading path.  It
captures one fixed LIBERO observation, then launches every stochastic sequence
in a fresh process so request order is the sole planned intervention.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import traceback
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = REPO_ROOT / "reports" / "epoch6_schedule_invariant_evaluation" / "problem_verification_protocol.json"
EXPECTED_PROTOCOL_SHA256 = "E5BA74354A1947A00045879A4815CCD09856F127E6809CF8BF649F10E2359946"
RESOURCE_AMENDMENT_PATH = (
    REPO_ROOT
    / "reports"
    / "epoch6_schedule_invariant_evaluation"
    / "resource_governance_amendment_v1.json"
)
EXPECTED_RESOURCE_AMENDMENT_SHA256 = "E98AED352765CEDA55607A2182ECE7B6E44B499DCC0253DA66313C72A6F3C601"
XVLA_ROOT = Path("/mnt/c/assets/repos/X-VLA")
HARNESS_ROOT = Path("/mnt/c/assets/repos/vla-evaluation-harness")
LIBERO_ROOT = Path("/mnt/c/assets/repos/LIBERO")
CHECKPOINT_ROOT = Path(
    "/home/jiheon/assets/checkpoints/xvla_hf_cache/transformers/"
    "models--2toINF--X-VLA-Libero/snapshots/129e71460678b7236cee6fc9707f09d9fa0c3590"
)
XVLA_REVISION = "6bc2513f5f1cbec715cc668b414392a6cae5c671"
HARNESS_REVISION = "a7eb023a962456bb0b6be40aa4336c31b7ac4ce6"
LIBERO_REVISION = "8f1084e3132a39270c3a13ebe37270a43ece2a01"
CHECKPOINT_REVISION = "129e71460678b7236cee6fc9707f09d9fa0c3590"
ROOT_SEED = 620260719
REFERENCE_SEED = 620260720
LOGICAL_KEY_COUNT = 20
DOMAIN_ID = 3
DENOISING_STEPS = 10
RAW_CHUNK_SHAPE = (30, 20)
MIN_MEM_AVAILABLE_BYTES = 6 * 1024**3
MIN_VRAM_FREE_MIB = 8 * 1024

SEQUENCES: dict[str, tuple[int, list[int]]] = {
    "A": (ROOT_SEED, list(range(LOGICAL_KEY_COUNT))),
    "A_repeat": (ROOT_SEED, list(range(LOGICAL_KEY_COUNT))),
    "B": (ROOT_SEED, list(reversed(range(LOGICAL_KEY_COUNT)))),
    "C": (REFERENCE_SEED, list(range(LOGICAL_KEY_COUNT))),
}


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serializable")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def hash_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype="<f4"))
    digest = hashlib.sha256()
    digest.update(b"float32\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def hash_uint8_array(value: np.ndarray) -> str:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.uint8))
    digest = hashlib.sha256()
    digest.update(b"uint8\0")
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes(order="C"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest().upper()


def normalized_rms(x: np.ndarray, y: np.ndarray) -> float:
    left = np.asarray(x, dtype=np.float64)
    right = np.asarray(y, dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} versus {right.shape}")
    numerator = math.sqrt(float(np.mean(np.square(left - right))))
    left_scale = math.sqrt(float(np.mean(np.square(left))))
    right_scale = math.sqrt(float(np.mean(np.square(right))))
    return numerator / max(left_scale, right_scale, 1e-12)


def rot6d_to_axis_angle(rot6d: np.ndarray) -> np.ndarray:
    values = np.asarray(rot6d, dtype=np.float64)
    single = values.ndim == 1
    values = values.reshape(-1, 6)
    output = np.zeros((len(values), 3), dtype=np.float32)
    for index, row in enumerate(values):
        first = row[:3]
        second = row[3:6]
        first = first / max(float(np.linalg.norm(first)), 1e-12)
        second = second - float(np.dot(first, second)) * first
        second = second / max(float(np.linalg.norm(second)), 1e-12)
        third = np.cross(first, second)
        rotation = np.stack([first, second, third], axis=1)
        cosine = float(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))
        angle = math.acos(cosine)
        if angle <= 1e-8:
            axis_angle = np.zeros(3, dtype=np.float64)
        elif abs(math.pi - angle) <= 1e-5:
            diagonal = np.maximum((np.diag(rotation) + 1.0) / 2.0, 0.0)
            axis = np.sqrt(diagonal)
            axis[0] = math.copysign(axis[0], rotation[2, 1] - rotation[1, 2])
            axis[1] = math.copysign(axis[1], rotation[0, 2] - rotation[2, 0])
            axis[2] = math.copysign(axis[2], rotation[1, 0] - rotation[0, 1])
            axis = axis / max(float(np.linalg.norm(axis)), 1e-12)
            axis_angle = axis * angle
        else:
            axis = np.array(
                [
                    rotation[2, 1] - rotation[1, 2],
                    rotation[0, 2] - rotation[2, 0],
                    rotation[1, 0] - rotation[0, 1],
                ],
                dtype=np.float64,
            ) / (2.0 * math.sin(angle))
            axis_angle = axis * angle
        output[index] = axis_angle.astype(np.float32)
    return output[0] if single else output


def raw_to_processed_7d(raw_actions: np.ndarray) -> np.ndarray:
    raw = np.asarray(raw_actions, dtype=np.float32)
    if raw.ndim != 2 or raw.shape[1] != 20:
        raise ValueError(f"expected an Nx20 raw chunk, got {raw.shape}")
    output = np.zeros((raw.shape[0], 7), dtype=np.float32)
    output[:, :3] = raw[:, :3]
    output[:, 3:6] = rot6d_to_axis_angle(raw[:, 3:9])
    output[:, 6] = np.where(raw[:, 9] > 0.5, 1.0, -1.0)
    return output


def gripper_disagreement(left: np.ndarray, right: np.ndarray) -> float:
    x = np.asarray(left, dtype=np.float32)
    y = np.asarray(right, dtype=np.float32)
    if x.shape != y.shape or x.ndim != 2 or x.shape[1] != 7:
        raise ValueError("processed chunks must have matching Nx7 shapes")
    return float(np.mean(x[:, 6] != y[:, 6]))


def logical_key(index: int) -> dict[str, Any]:
    if index < 0 or index >= LOGICAL_KEY_COUNT:
        raise ValueError(f"logical index out of range: {index}")
    return {
        "benchmark_revision": HARNESS_REVISION,
        "suite": "libero_spatial",
        "task_id": 0,
        "episode_index": int(index),
        "action_query_index": 0,
        "sample_index": 0,
    }


def canonical_key_string(index: int) -> str:
    return json.dumps(logical_key(index), sort_keys=True, separators=(",", ":"))


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=20,
    ).strip()


def git_status(path: Path) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(path), "status", "--short"],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=20,
    )
    return [line for line in output.splitlines() if line.strip()]


def git_integrity(path: Path, relevant_paths: Sequence[str] | None = None) -> dict[str, Any]:
    pathspec = list(relevant_paths or [])

    def quiet_diff(*extra: str) -> bool:
        command = ["git", "-C", str(path), "diff", *extra, "--quiet"]
        if pathspec:
            command.extend(["--", *pathspec])
        completed = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, check=False)
        if completed.returncode not in (0, 1):
            raise RuntimeError(f"git integrity command failed for {path}: {completed.stdout}")
        return completed.returncode == 0

    untracked_command = ["git", "-C", str(path), "ls-files", "--others", "--exclude-standard"]
    if pathspec:
        untracked_command.extend(["--", *pathspec])
    untracked_output = subprocess.check_output(
        untracked_command,
        text=True,
        stderr=subprocess.STDOUT,
        timeout=30,
    )
    return {
        "scope": pathspec or ["FULL_REPOSITORY"],
        "worktree_clean_ignoring_cr_at_eol": quiet_diff("--ignore-cr-at-eol"),
        "index_clean": quiet_diff("--cached"),
        "untracked_files": [line for line in untracked_output.splitlines() if line.strip()],
    }


def meminfo() -> dict[str, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, raw = line.split(":", 1)
        values[key] = int(raw.strip().split()[0]) * 1024
    return {
        "mem_total_bytes": values["MemTotal"],
        "mem_available_bytes": values["MemAvailable"],
        "swap_total_bytes": values["SwapTotal"],
        "swap_free_bytes": values["SwapFree"],
        "swap_used_bytes": values["SwapTotal"] - values["SwapFree"],
    }


def nvidia_memory() -> dict[str, Any]:
    output = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        stderr=subprocess.STDOUT,
        timeout=20,
    ).strip()
    parts = [part.strip() for part in output.splitlines()[0].split(",")]
    if len(parts) != 4:
        raise RuntimeError(f"unexpected nvidia-smi response: {output!r}")
    return {
        "name": parts[0],
        "total_mib": int(parts[1]),
        "used_mib": int(parts[2]),
        "free_mib": int(parts[3]),
    }


def resource_snapshot(torch_module: Any | None = None) -> dict[str, Any]:
    import psutil

    process = psutil.Process(os.getpid())
    snapshot: dict[str, Any] = {
        "captured_at": utc_now(),
        "pid": os.getpid(),
        "process_rss_bytes": int(process.memory_info().rss),
        "system": meminfo(),
        "nvidia_smi": nvidia_memory(),
    }
    if torch_module is not None and torch_module.cuda.is_available():
        snapshot["torch_cuda"] = {
            "allocated_bytes": int(torch_module.cuda.memory_allocated(0)),
            "reserved_bytes": int(torch_module.cuda.memory_reserved(0)),
            "peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated(0)),
            "peak_reserved_bytes": int(torch_module.cuda.max_memory_reserved(0)),
        }
    return snapshot


def require_safe_resources(snapshot: Mapping[str, Any]) -> None:
    system = snapshot["system"]
    gpu = snapshot["nvidia_smi"]
    if int(system["swap_used_bytes"]) != 0:
        raise RuntimeError(f"swap is already in use: {system['swap_used_bytes']} bytes")
    if int(system["mem_available_bytes"]) < MIN_MEM_AVAILABLE_BYTES:
        raise RuntimeError(f"insufficient available RAM: {system['mem_available_bytes']} bytes")
    if int(gpu["free_mib"]) < MIN_VRAM_FREE_MIB:
        raise RuntimeError(f"insufficient free VRAM: {gpu['free_mib']} MiB")


def require_host_smoke_lock(run_dir: Path) -> None:
    expected = (run_dir.parent / "host_resource_smoke.global.lock.json").resolve()
    token = os.environ.get("EPOCH6_HOST_SMOKE_LOCK_PATH")
    if token is None or Path(token).resolve() != expected or not expected.is_file():
        raise RuntimeError("resource-smoke mode requires the atomic Windows host-monitor lock")
    payload = json.loads(expected.read_text(encoding="utf-8-sig"))
    if (
        payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or payload.get("status") != "active"
        or payload.get("run_id") != run_dir.name
    ):
        raise RuntimeError("host resource-smoke lock payload is invalid")


def require_parent_run_lock(run_dir: Path) -> None:
    expected = (run_dir / "run.lock.json").resolve()
    token = os.environ.get("EPOCH6_PARENT_RUN_LOCK")
    if token is None or Path(token).resolve() != expected or not expected.is_file():
        raise RuntimeError("sequence mode requires the active parent run lock")
    payload = json.loads(expected.read_text(encoding="utf-8"))
    if payload.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("parent run lock protocol mismatch")
    parent_pid = int(payload.get("pid", -1))
    os.kill(parent_pid, 0)


class ResourceMonitor:
    def __init__(self, torch_module: Any | None, heartbeat: Path, interval_seconds: float = 1.0) -> None:
        self.torch = torch_module
        self.heartbeat = heartbeat
        self.interval_seconds = interval_seconds
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="epoch6-stage0-resource", daemon=True)
        self.samples = 0
        self.peak_rss_bytes = 0
        self.peak_vram_mib = 0
        self.minimum_available_ram_bytes: int | None = None
        self.maximum_swap_used_bytes = 0
        self.exceptions: list[str] = []

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                snapshot = resource_snapshot(self.torch)
                self.samples += 1
                self.peak_rss_bytes = max(self.peak_rss_bytes, int(snapshot["process_rss_bytes"]))
                available = int(snapshot["system"]["mem_available_bytes"])
                self.minimum_available_ram_bytes = (
                    available
                    if self.minimum_available_ram_bytes is None
                    else min(self.minimum_available_ram_bytes, available)
                )
                self.maximum_swap_used_bytes = max(
                    self.maximum_swap_used_bytes,
                    int(snapshot["system"]["swap_used_bytes"]),
                )
                self.peak_vram_mib = max(self.peak_vram_mib, int(snapshot["nvidia_smi"]["used_mib"]))
                write_json(
                    self.heartbeat,
                    {
                        "status": "running",
                        "pid": os.getpid(),
                        "updated_at": utc_now(),
                        "resource_snapshot": snapshot,
                    },
                )
            except Exception as exc:  # pragma: no cover - runtime telemetry boundary
                self.exceptions.append(f"{type(exc).__name__}: {exc}")
            self.stop_event.wait(self.interval_seconds)

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=5.0)
        return {
            "samples": self.samples,
            "interval_seconds": self.interval_seconds,
            "peak_process_rss_bytes": self.peak_rss_bytes,
            "peak_total_gpu_used_mib": self.peak_vram_mib,
            "minimum_available_ram_bytes": self.minimum_available_ram_bytes,
            "maximum_swap_used_bytes": self.maximum_swap_used_bytes,
            "exceptions": self.exceptions,
        }


def validate_protocol() -> dict[str, Any]:
    actual_hash = sha256_file(PROTOCOL_PATH)
    if actual_hash != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"protocol hash mismatch: expected {EXPECTED_PROTOCOL_SHA256}, got {actual_hash}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    stage0 = protocol["stage0"]
    semantics = stage0["execution_semantics"]
    if protocol["ours_authorized"] or protocol["paper_generation_authorized"]:
        raise RuntimeError("protocol unexpectedly authorizes Ours or paper generation")
    expected = {
        "root_seed": ROOT_SEED,
        "logical_episode_keys": LOGICAL_KEY_COUNT,
    }
    for key, value in expected.items():
        if stage0[key] != value:
            raise RuntimeError(f"protocol {key} mismatch: {stage0[key]!r} versus {value!r}")
    if semantics["domain_id"] != DOMAIN_ID or semantics["denoising_steps"] != DENOISING_STEPS:
        raise RuntimeError("protocol X-VLA inference settings mismatch")
    if tuple(semantics["raw_chunk_shape"]) != RAW_CHUNK_SHAPE:
        raise RuntimeError("protocol raw chunk shape mismatch")
    smoke = stage0["resource_smoke"]
    if smoke["forward_calls"] != 1 or smoke["root_seed"] != ROOT_SEED:
        raise RuntimeError("protocol resource-smoke settings mismatch")
    if stage0["fixture_source"]["success_check_calls"] != 0:
        raise RuntimeError("protocol fixture success-check prohibition mismatch")
    if stage0["simulator_actions_executed"] or stage0["reward_success_done_read"]:
        raise RuntimeError("protocol outcome suppression is disabled")
    return protocol


def checkpoint_manifest() -> dict[str, Any]:
    if not CHECKPOINT_ROOT.is_dir():
        raise FileNotFoundError(CHECKPOINT_ROOT)
    files: list[dict[str, Any]] = []
    for path in sorted(item for item in CHECKPOINT_ROOT.iterdir() if item.is_file()):
        resolved = path.resolve(strict=True)
        files.append(
            {
                "name": path.name,
                "resolved_path": str(resolved),
                "bytes": int(resolved.stat().st_size),
                "sha256": sha256_file(resolved),
            }
        )
    required = {"config.json", "model.safetensors", "preprocessor_config.json", "tokenizer.json"}
    observed = {entry["name"] for entry in files}
    if not required.issubset(observed):
        raise RuntimeError(f"checkpoint snapshot missing required files: {sorted(required - observed)}")
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    return {
        "checkpoint_revision": CHECKPOINT_REVISION,
        "snapshot_path": str(CHECKPOINT_ROOT),
        "files": files,
        "manifest_sha256": digest,
    }


def source_provenance() -> dict[str, Any]:
    harness_relevant_paths = [
        "src/vla_eval/model_servers/xvla.py",
        "src/vla_eval/model_servers/base.py",
        "src/vla_eval/model_servers/predict.py",
        "src/vla_eval/orchestrator.py",
        "src/vla_eval/recording.py",
    ]
    source = {
        "runner_sha256": sha256_file(Path(__file__).resolve()),
        "xvla_head": git_head(XVLA_ROOT),
        "xvla_expected_head": XVLA_REVISION,
        "xvla_integrity": git_integrity(XVLA_ROOT),
        "harness_head": git_head(HARNESS_ROOT),
        "harness_expected_head": HARNESS_REVISION,
        "harness_integrity": git_integrity(HARNESS_ROOT, harness_relevant_paths),
        "libero_head": git_head(LIBERO_ROOT),
        "libero_expected_head": LIBERO_REVISION,
        "libero_integrity": git_integrity(LIBERO_ROOT),
        "xvla_generate_source_sha256": sha256_file(XVLA_ROOT / "models" / "modeling_xvla.py"),
        "harness_xvla_adapter_sha256": sha256_file(
            HARNESS_ROOT / "src" / "vla_eval" / "model_servers" / "xvla.py"
        ),
        "libero_env_wrapper_sha256": sha256_file(
            LIBERO_ROOT / "libero" / "libero" / "envs" / "env_wrapper.py"
        ),
        "compatibility_helper_sha256": sha256_file(
            REPO_ROOT / "tca_map" / "xvla_task1" / "gradient_smoke.py"
        ),
        "python_executable": sys.executable,
        "python_version": sys.version,
    }
    source["manifest_sha256"] = hashlib.sha256(
        json.dumps(source, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest().upper()
    return source


def validate_execution_provenance(run_dir: Path) -> dict[str, Any]:
    preflight_path = run_dir / "static_preflight.json"
    checkpoint_path = run_dir / "checkpoint_manifest.json"
    preflight = json.loads(preflight_path.read_text(encoding="utf-8"))
    checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    current = source_provenance()
    if preflight.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("static preflight protocol hash mismatch")
    if current != preflight.get("source"):
        raise RuntimeError("source or runtime provenance changed after static preflight")
    if sha256_file(checkpoint_path) != preflight.get("checkpoint_manifest_file_sha256"):
        raise RuntimeError("checkpoint manifest file changed after static preflight")
    if checkpoint.get("manifest_sha256") != preflight.get("checkpoint_manifest_sha256"):
        raise RuntimeError("checkpoint content manifest identity mismatch")
    for entry in checkpoint["files"]:
        resolved = Path(entry["resolved_path"])
        if not resolved.is_file() or int(resolved.stat().st_size) != int(entry["bytes"]):
            raise RuntimeError(f"checkpoint file changed or disappeared: {resolved}")
        if sha256_file(resolved) != entry["sha256"]:
            raise RuntimeError(f"checkpoint file content hash changed: {resolved}")
    return {
        "source_manifest_sha256": current["manifest_sha256"],
        "checkpoint_manifest_sha256": checkpoint["manifest_sha256"],
        "checkpoint_manifest_file_sha256": sha256_file(checkpoint_path),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    }


def static_preflight(run_dir: Path) -> dict[str, Any]:
    protocol = validate_protocol()
    before = resource_snapshot()
    require_safe_resources(before)
    source = source_provenance()
    if (
        source["xvla_head"] != XVLA_REVISION
        or source["harness_head"] != HARNESS_REVISION
        or source["libero_head"] != LIBERO_REVISION
    ):
        raise RuntimeError("pinned source revision mismatch")
    integrity = {
        "xvla": source["xvla_integrity"],
        "harness": source["harness_integrity"],
        "libero": source["libero_integrity"],
    }
    for name, report in integrity.items():
        if (
            not report["worktree_clean_ignoring_cr_at_eol"]
            or not report["index_clean"]
            or report["untracked_files"]
        ):
            raise RuntimeError(f"pinned source integrity failure for {name}: {report}")
    manifest = checkpoint_manifest()
    write_json(run_dir / "checkpoint_manifest.json", manifest)
    result = {
        "schema_version": "epoch6.schedule_stage0.preflight.v1",
        "captured_at": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "outcomes_read": False,
        "simulator_actions_executed": 0,
        "source": source,
        "checkpoint_manifest_sha256": manifest["manifest_sha256"],
        "checkpoint_manifest_file_sha256": sha256_file(run_dir / "checkpoint_manifest.json"),
        "resources_before": before,
        "protocol_stage0": protocol["stage0"],
        "status": "STATIC_PREFLIGHT_PASS",
    }
    write_json(run_dir / "static_preflight.json", result)
    return result


def import_pinned_libero() -> tuple[Any, Any, Path]:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")
    libero_import_root = str(LIBERO_ROOT)
    for module_name in [name for name in sys.modules if name == "libero" or name.startswith("libero.")]:
        del sys.modules[module_name]
    if libero_import_root in sys.path:
        sys.path.remove(libero_import_root)
    sys.path.insert(0, libero_import_root)
    libero_namespace = types.ModuleType("libero")
    libero_namespace.__package__ = "libero"
    libero_namespace.__path__ = [str(LIBERO_ROOT / "libero")]
    sys.modules["libero"] = libero_namespace
    import inspect
    import libero.libero.envs.env_wrapper as libero_env_wrapper
    from libero.libero import benchmark
    from libero.libero.envs import OffScreenRenderEnv

    executed_wrapper = Path(inspect.getsourcefile(libero_env_wrapper) or "").resolve(strict=True)
    expected_wrapper = (LIBERO_ROOT / "libero" / "libero" / "envs" / "env_wrapper.py").resolve(strict=True)
    if executed_wrapper != expected_wrapper:
        raise RuntimeError(f"executed LIBERO wrapper is not pinned source: {executed_wrapper}")
    if sha256_file(executed_wrapper) != source_provenance()["libero_env_wrapper_sha256"]:
        raise RuntimeError("executed LIBERO wrapper hash mismatch")
    return benchmark, OffScreenRenderEnv, executed_wrapper


def capture_fixture(run_dir: Path) -> dict[str, Any]:
    validate_protocol()
    provenance = validate_execution_provenance(run_dir)
    benchmark, OffScreenRenderEnv, executed_wrapper = import_pinned_libero()

    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    pinned_asset_root = LIBERO_ROOT / "libero" / "libero"
    bddl = (pinned_asset_root / "bddl_files" / task.problem_folder / task.bddl_file).resolve(
        strict=True
    )
    init_states_path = (
        pinned_asset_root / "init_files" / task.problem_folder / task.init_states_file
    ).resolve(strict=True)
    import torch

    initial_states = torch.load(init_states_path, map_location="cpu", weights_only=False)
    if len(initial_states) < 1:
        raise RuntimeError("libero_spatial task 0 has no initial state index 0")
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    success_check_calls = 0

    def forbidden_success_check(*_args: Any, **_kwargs: Any) -> None:
        nonlocal success_check_calls
        success_check_calls += 1
        raise RuntimeError("success checking is forbidden during the outcome-suppressed fixture capture")

    original_wrapper_check = env.check_success
    original_inner_check = env.env._check_success
    env.check_success = forbidden_success_check
    env.env._check_success = forbidden_success_check
    try:
        env.seed(ROOT_SEED)
        env.reset()
        env.set_state(initial_states[0])
        env.env.sim.forward()
        env._post_process()
        env._update_observables(force=True)
        observation = env.env._get_observations()
        raw_agentview = np.asarray(observation["agentview_image"], dtype=np.uint8).copy()
        raw_wrist = np.asarray(observation["robot0_eye_in_hand_image"], dtype=np.uint8).copy()
        controller = env.env.robots[0].controller
        ee_position = np.asarray(controller.ee_pos, dtype=np.float32).copy()
        ee_rotation = np.asarray(controller.ee_ori_mat, dtype=np.float32).copy()
    finally:
        env.check_success = original_wrapper_check
        env.env._check_success = original_inner_check
        env.close()
    if success_check_calls != 0:
        raise RuntimeError(f"fixture capture attempted {success_check_calls} success checks")

    policy_agentview = np.flip(raw_agentview, axis=(0, 1)).copy()
    policy_wrist = raw_wrist.copy()
    first_arm = np.concatenate(
        [ee_position, ee_rotation[:3, 0], ee_rotation[:3, 1], np.zeros(1, dtype=np.float32)]
    ).astype(np.float32)
    proprio = np.concatenate([first_arm, np.zeros_like(first_arm)]).astype(np.float32)
    language = str(task.language)
    if raw_agentview.shape != (256, 256, 3) or raw_wrist.shape != (256, 256, 3):
        raise RuntimeError(f"unexpected fixture image shapes: {raw_agentview.shape}, {raw_wrist.shape}")
    if proprio.shape != (20,):
        raise RuntimeError(f"unexpected proprio shape: {proprio.shape}")

    fixture_path = run_dir / "fixture.npz"
    np.savez_compressed(
        fixture_path,
        raw_agentview=raw_agentview,
        raw_wrist=raw_wrist,
        policy_agentview=policy_agentview,
        policy_wrist=policy_wrist,
        proprio=proprio,
        ee_position=ee_position,
        ee_rotation=ee_rotation,
        language_utf8=np.frombuffer(language.encode("utf-8"), dtype=np.uint8),
    )
    components = {
        "raw_agentview_sha256": hash_uint8_array(raw_agentview),
        "raw_wrist_sha256": hash_uint8_array(raw_wrist),
        "policy_agentview_sha256": hash_uint8_array(policy_agentview),
        "policy_wrist_sha256": hash_uint8_array(policy_wrist),
        "proprio_sha256": hash_array(proprio),
        "language_sha256": hashlib.sha256(language.encode("utf-8")).hexdigest().upper(),
    }
    result = {
        "schema_version": "epoch6.schedule_stage0.fixture.v1",
        "captured_at": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "provenance": provenance,
        "executed_libero_env_wrapper": str(executed_wrapper),
        "executed_libero_env_wrapper_sha256": sha256_file(executed_wrapper),
        "suite": "libero_spatial",
        "task_id": 0,
        "initial_state_index": 0,
        "language": language,
        "bddl_file": str(bddl),
        "bddl_file_sha256": sha256_file(bddl),
        "init_states_file": str(init_states_path),
        "init_states_file_sha256": sha256_file(init_states_path),
        "init_states_load_weights_only": False,
        "fixture_npz": str(fixture_path),
        "fixture_npz_sha256": sha256_file(fixture_path),
        "component_hashes": components,
        "simulator_actions_executed": 0,
        "success_check_calls": success_check_calls,
        "reward_success_done_read": False,
        "status": "FIXTURE_CAPTURED_OUTCOME_SUPPRESSED",
    }
    write_json(run_dir / "fixture_manifest.json", result)
    return result


def load_fixture(run_dir: Path, expected_provenance: Mapping[str, Any] | None = None) -> dict[str, Any]:
    manifest = json.loads((run_dir / "fixture_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError("fixture protocol hash mismatch")
    current_provenance = (
        dict(expected_provenance) if expected_provenance is not None else validate_execution_provenance(run_dir)
    )
    if manifest.get("provenance") != current_provenance:
        raise RuntimeError("fixture provenance mismatch")
    if (
        manifest.get("status") != "FIXTURE_CAPTURED_OUTCOME_SUPPRESSED"
        or manifest.get("simulator_actions_executed") != 0
        or manifest.get("success_check_calls") != 0
        or manifest.get("reward_success_done_read")
    ):
        raise RuntimeError("fixture outcome-suppression attestation is invalid")
    executed_wrapper = Path(manifest["executed_libero_env_wrapper"])
    if (
        not executed_wrapper.is_file()
        or sha256_file(executed_wrapper) != manifest.get("executed_libero_env_wrapper_sha256")
    ):
        raise RuntimeError("fixture executed LIBERO source no longer matches")
    fixture_path = run_dir / "fixture.npz"
    if sha256_file(fixture_path) != manifest["fixture_npz_sha256"]:
        raise RuntimeError("fixture NPZ hash mismatch")
    with np.load(fixture_path, allow_pickle=False) as payload:
        fixture = {key: np.asarray(payload[key]).copy() for key in payload.files}
    language = fixture.pop("language_utf8").tobytes().decode("utf-8")
    fixture["language"] = language
    if hash_uint8_array(fixture["policy_agentview"]) != manifest["component_hashes"]["policy_agentview_sha256"]:
        raise RuntimeError("fixture policy agentview hash mismatch")
    if hash_uint8_array(fixture["policy_wrist"]) != manifest["component_hashes"]["policy_wrist_sha256"]:
        raise RuntimeError("fixture policy wrist hash mismatch")
    if hash_array(fixture["proprio"]) != manifest["component_hashes"]["proprio_sha256"]:
        raise RuntimeError("fixture proprio hash mismatch")
    if hash_uint8_array(fixture["raw_agentview"]) != manifest["component_hashes"]["raw_agentview_sha256"]:
        raise RuntimeError("fixture raw agentview hash mismatch")
    if hash_uint8_array(fixture["raw_wrist"]) != manifest["component_hashes"]["raw_wrist_sha256"]:
        raise RuntimeError("fixture raw wrist hash mismatch")
    if hashlib.sha256(language.encode("utf-8")).hexdigest().upper() != manifest["component_hashes"]["language_sha256"]:
        raise RuntimeError("fixture language hash mismatch")
    return fixture


def seed_process_once(seed: int) -> Any:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    import torch

    torch.manual_seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    torch.cuda.manual_seed_all(seed)
    return torch


def load_xvla(torch_module: Any) -> tuple[Any, Any, dict[str, Any]]:
    os.environ["HF_HOME"] = "/home/jiheon/assets/checkpoints/xvla_hf_cache"
    os.environ["HF_HUB_CACHE"] = "/home/jiheon/assets/checkpoints/xvla_hf_cache/transformers"
    os.environ["TRANSFORMERS_CACHE"] = "/home/jiheon/assets/checkpoints/xvla_hf_cache/transformers"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    source_path = str(XVLA_ROOT)
    if source_path in sys.path:
        sys.path.remove(source_path)
    sys.path.insert(0, source_path)
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from tca_map.xvla_task1.gradient_smoke import (
        install_optional_server_import_shims,
        install_xvla_transformers_compat_patches,
        package_version,
    )

    shims = install_optional_server_import_shims()
    patches = install_xvla_transformers_compat_patches()
    from models.modeling_xvla import XVLA
    from models.processing_xvla import XVLAProcessor

    processor = XVLAProcessor.from_pretrained(
        str(CHECKPOINT_ROOT),
        trust_remote_code=True,
        local_files_only=True,
    )
    model = XVLA.from_pretrained(
        str(CHECKPOINT_ROOT),
        trust_remote_code=True,
        torch_dtype=torch_module.float32,
        local_files_only=True,
    )
    model.eval().to(device="cuda:0", dtype=torch_module.float32)
    parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
    if parameter_devices != ["cuda:0"]:
        raise RuntimeError(f"X-VLA parameters are not exclusively CUDA-resident: {parameter_devices}")
    return model, processor, {
        "optional_import_shims": shims,
        "compatibility_patches": patches,
        "torch": torch_module.__version__,
        "transformers": package_version("transformers"),
        "model_type": type(model).__name__,
        "processor_type": type(processor).__name__,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "parameter_devices": parameter_devices,
        "device_map_requested": False,
        "cpu_or_disk_model_offload": False,
    }


def tensor_mapping_hash(mapping: Mapping[str, Any], torch_module: Any) -> str:
    digest = hashlib.sha256()
    for key in sorted(mapping):
        value = mapping[key]
        digest.update(key.encode("utf-8") + b"\0")
        if isinstance(value, torch_module.Tensor):
            array = value.detach().cpu().contiguous().numpy()
            digest.update(str(array.dtype).encode("ascii") + b"\0")
            digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
            digest.update(array.tobytes(order="C"))
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest().upper()


def prepare_query_inputs(
    fixture: Mapping[str, Any], processor: Any, model: Any, torch_module: Any
) -> tuple[dict[str, Any], str]:
    images = [np.asarray(fixture["policy_agentview"]), np.asarray(fixture["policy_wrist"])]
    inputs = processor(images, str(fixture["language"]))
    device = next(model.parameters()).device
    model_inputs: dict[str, Any] = {}
    for key, value in inputs.items():
        if isinstance(value, torch_module.Tensor):
            dtype = torch_module.float32 if value.is_floating_point() else value.dtype
            model_inputs[key] = value.to(device=device, dtype=dtype)
        else:
            model_inputs[key] = value
    model_inputs["proprio"] = torch_module.as_tensor(
        np.asarray(fixture["proprio"], dtype=np.float32),
        device=device,
        dtype=torch_module.float32,
    ).unsqueeze(0)
    model_inputs["domain_id"] = torch_module.tensor([DOMAIN_ID], device=device, dtype=torch_module.long)
    return model_inputs, tensor_mapping_hash(model_inputs, torch_module)


def run_resource_smoke(run_dir: Path) -> int:
    validate_protocol()
    require_host_smoke_lock(run_dir)
    provenance = validate_execution_provenance(run_dir)
    fixture = load_fixture(run_dir, provenance)
    before = resource_snapshot()
    require_safe_resources(before)
    torch_module = seed_process_once(ROOT_SEED)
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats()
    monitor = ResourceMonitor(torch_module, run_dir / "resource_smoke_heartbeat.json")
    monitor.start()
    model = processor = model_inputs = action = None
    result: dict[str, Any] = {
        "schema_version": "epoch6.schedule_stage0.resource_smoke.v1",
        "started_at": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "provenance": provenance,
        "model_inference_calls": 0,
        "simulator_actions_executed": 0,
        "reward_success_done_read": False,
        "resources_before": before,
    }
    exit_code = 1
    try:
        load_started = time.monotonic()
        model, processor, runtime = load_xvla(torch_module)
        result["load_seconds"] = time.monotonic() - load_started
        model_inputs, prepared_hash = prepare_query_inputs(fixture, processor, model, torch_module)
        forward_started = time.monotonic()
        with torch_module.no_grad():
            action = model.generate_actions(**model_inputs, steps=DENOISING_STEPS)
        torch_module.cuda.synchronize()
        raw = action.float().detach().cpu().numpy().squeeze(0).astype(np.float32, copy=False)
        if raw.shape != RAW_CHUNK_SHAPE or not np.isfinite(raw).all():
            raise RuntimeError(f"resource-smoke raw chunk is invalid: shape={raw.shape}")
        result["model_inference_calls"] = 1
        result["forward_seconds"] = time.monotonic() - forward_started
        result["raw_chunk_shape"] = list(raw.shape)
        result["raw_chunk_finite"] = bool(np.isfinite(raw).all())
        result["raw_chunk_sha256"] = hash_array(raw)
        after_load = resource_snapshot(torch_module)
        if int(after_load["system"]["swap_used_bytes"]) != 0:
            raise RuntimeError("resource smoke observed nonzero swap usage")
        result.update(
            {
                "runtime": runtime,
                "prepared_input_sha256": prepared_hash,
                "resources_after_forward": after_load,
                "status": "ACTUAL_PATH_RESOURCE_SMOKE_PASS",
            }
        )
        exit_code = 0
    except Exception as exc:
        result.update(
            {
                "status": "ACTUAL_PATH_RESOURCE_SMOKE_FAIL",
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
            }
        )
    finally:
        model = processor = model_inputs = action = None
        cleanup_exceptions: list[str] = []
        try:
            gc.collect()
        except Exception as exc:
            cleanup_exceptions.append(f"gc.collect: {type(exc).__name__}: {exc}")
        try:
            torch_module.cuda.empty_cache()
            torch_module.cuda.synchronize()
        except Exception as exc:
            cleanup_exceptions.append(f"cuda cleanup: {type(exc).__name__}: {exc}")
        try:
            result["malloc_trim_result"] = int(ctypes.CDLL("libc.so.6").malloc_trim(0))
        except Exception as exc:
            cleanup_exceptions.append(f"malloc_trim: {type(exc).__name__}: {exc}")
        try:
            result["resources_after_teardown"] = resource_snapshot(torch_module)
        except Exception as exc:
            cleanup_exceptions.append(f"teardown snapshot: {type(exc).__name__}: {exc}")
        if cleanup_exceptions:
            result["cleanup_exceptions"] = cleanup_exceptions
            result["status"] = "ACTUAL_PATH_RESOURCE_SMOKE_FAIL_TEARDOWN"
            exit_code = 1
        result["resource_monitor"] = monitor.stop()
        result["completed_at"] = utc_now()
        monitor_invalid = bool(
            result["resource_monitor"]["maximum_swap_used_bytes"] != 0
            or result["resource_monitor"]["samples"] < 1
            or result["resource_monitor"]["exceptions"]
        )
        if monitor_invalid:
            result["status"] = "ACTUAL_PATH_RESOURCE_SMOKE_FAIL_TELEMETRY_OR_SWAP"
            exit_code = 1
        write_json(run_dir / "resource_smoke.json", result)
        write_text(run_dir / "resource_smoke_exit_code.txt", f"{exit_code}\n")
    return exit_code


def nested_lists(value: Any) -> Any:
    if isinstance(value, tuple):
        return [nested_lists(item) for item in value]
    return value


def nested_tuples(value: Any) -> Any:
    if isinstance(value, list):
        return tuple(nested_tuples(item) for item in value)
    return value


def capture_rng_state(torch_module: Any) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    cuda_states = [state.detach().cpu().numpy().astype(np.uint8, copy=True) for state in torch_module.cuda.get_rng_state_all()]
    return {
        "python_random_state": nested_lists(random.getstate()),
        "numpy_algorithm": str(numpy_state[0]),
        "numpy_keys": np.asarray(numpy_state[1], dtype=np.uint32).copy(),
        "numpy_position": int(numpy_state[2]),
        "numpy_has_gauss": int(numpy_state[3]),
        "numpy_cached_gaussian": float(numpy_state[4]),
        "torch_cpu": torch_module.get_rng_state().detach().cpu().numpy().astype(np.uint8, copy=True),
        "torch_cuda": cuda_states,
    }


def restore_rng_state(state: Mapping[str, Any], torch_module: Any) -> None:
    random.setstate(nested_tuples(state["python_random_state"]))
    np.random.set_state(
        (
            str(state["numpy_algorithm"]),
            np.asarray(state["numpy_keys"], dtype=np.uint32),
            int(state["numpy_position"]),
            int(state["numpy_has_gauss"]),
            float(state["numpy_cached_gaussian"]),
        )
    )
    torch_module.set_rng_state(torch_module.as_tensor(state["torch_cpu"], dtype=torch_module.uint8, device="cpu"))
    cuda_states = [
        torch_module.as_tensor(value, dtype=torch_module.uint8, device="cpu") for value in state["torch_cuda"]
    ]
    if len(cuda_states) != torch_module.cuda.device_count():
        raise RuntimeError("saved CUDA RNG device count does not match current runtime")
    torch_module.cuda.set_rng_state_all(cuda_states)


def write_sequence_partial(
    run_dir: Path,
    sequence_name: str,
    seed: int,
    order: list[int],
    provenance: Mapping[str, Any],
    raw_chunks: np.ndarray,
    processed_chunks: np.ndarray,
    latencies: np.ndarray,
    draw_positions: np.ndarray,
    rows: list[dict[str, Any]],
    rng_state: Mapping[str, Any],
) -> None:
    npz_path = run_dir / f"sequence_{sequence_name}_partial.npz"
    temporary_npz = run_dir / f"sequence_{sequence_name}_partial.tmp.npz"
    rng_arrays = {
        "rng_numpy_keys": np.asarray(rng_state["numpy_keys"], dtype=np.uint32),
        "rng_torch_cpu": np.asarray(rng_state["torch_cpu"], dtype=np.uint8),
    }
    for index, value in enumerate(rng_state["torch_cuda"]):
        rng_arrays[f"rng_torch_cuda_{index}"] = np.asarray(value, dtype=np.uint8)
    np.savez_compressed(
        temporary_npz,
        raw_chunks=raw_chunks,
        processed_chunks=processed_chunks,
        latencies_seconds=latencies,
        draw_positions=draw_positions,
        **rng_arrays,
    )
    temporary_npz.replace(npz_path)
    write_json(
        run_dir / f"sequence_{sequence_name}_partial.json",
        {
            "schema_version": "epoch6.schedule_stage0.sequence_partial.v1",
            "sequence": sequence_name,
            "root_seed": seed,
            "execution_order": order,
            "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
            "provenance": dict(provenance),
            "completed_queries": len(rows),
            "completed_logical_indices": sorted(int(row["logical_index"]) for row in rows),
            "planned_queries": LOGICAL_KEY_COUNT,
            "rows": rows,
            "rng_state": {
                "python_random_state": rng_state["python_random_state"],
                "numpy_algorithm": rng_state["numpy_algorithm"],
                "numpy_position": rng_state["numpy_position"],
                "numpy_has_gauss": rng_state["numpy_has_gauss"],
                "numpy_cached_gaussian": rng_state["numpy_cached_gaussian"],
                "cuda_device_count": len(rng_state["torch_cuda"]),
            },
            "partial_npz_sha256": sha256_file(npz_path),
            "updated_at": utc_now(),
            "simulator_actions_executed": 0,
            "reward_success_done_read": False,
        },
    )


def load_sequence_partial(
    run_dir: Path,
    sequence_name: str,
    seed: int,
    order: list[int],
    provenance: Mapping[str, Any],
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any] | None]:
    json_path = run_dir / f"sequence_{sequence_name}_partial.json"
    npz_path = run_dir / f"sequence_{sequence_name}_partial.npz"
    if not json_path.exists() and not npz_path.exists():
        return {}, [], None
    if not json_path.is_file() or not npz_path.is_file():
        raise RuntimeError(f"sequence {sequence_name} partial transaction is incomplete")
    metadata = json.loads(json_path.read_text(encoding="utf-8"))
    if (
        metadata.get("sequence") != sequence_name
        or metadata.get("root_seed") != seed
        or metadata.get("execution_order") != order
        or metadata.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256
        or metadata.get("provenance") != dict(provenance)
        or metadata.get("simulator_actions_executed") != 0
        or metadata.get("reward_success_done_read")
    ):
        raise RuntimeError(f"sequence {sequence_name} partial provenance mismatch")
    if metadata.get("partial_npz_sha256") != sha256_file(npz_path):
        raise RuntimeError(f"sequence {sequence_name} partial NPZ hash mismatch")
    with np.load(npz_path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]).copy() for key in payload.files}
    rows = list(metadata.get("rows", []))
    rows_by_key = {int(row["logical_index"]): row for row in rows}
    if len(rows_by_key) != len(rows) or set(rows_by_key) != set(metadata.get("completed_logical_indices", [])):
        raise RuntimeError(f"sequence {sequence_name} partial logical-key coverage mismatch")
    for key_index, row in rows_by_key.items():
        if row["logical_key"] != logical_key(key_index):
            raise RuntimeError(f"sequence {sequence_name} partial logical key mismatch")
        if row["draw_position"] != order.index(key_index):
            raise RuntimeError(f"sequence {sequence_name} partial draw position mismatch")
        if row["raw_chunk_sha256"] != hash_array(arrays["raw_chunks"][key_index]):
            raise RuntimeError(f"sequence {sequence_name} partial raw hash mismatch")
        if row["processed_7d_chunk_sha256"] != hash_array(arrays["processed_chunks"][key_index]):
            raise RuntimeError(f"sequence {sequence_name} partial processed hash mismatch")
    completed_positions = sorted(int(row["draw_position"]) for row in rows)
    if completed_positions != list(range(len(rows))):
        raise RuntimeError(f"sequence {sequence_name} partial keys are not an execution-order prefix")
    rng_metadata = metadata.get("rng_state", {})
    cuda_count = int(rng_metadata.get("cuda_device_count", -1))
    required_rng_arrays = {"rng_numpy_keys", "rng_torch_cpu"} | {
        f"rng_torch_cuda_{index}" for index in range(cuda_count)
    }
    if cuda_count < 1 or not required_rng_arrays.issubset(arrays):
        raise RuntimeError(f"sequence {sequence_name} partial RNG state is incomplete")
    rng_state = {
        "python_random_state": rng_metadata["python_random_state"],
        "numpy_algorithm": rng_metadata["numpy_algorithm"],
        "numpy_keys": arrays["rng_numpy_keys"],
        "numpy_position": rng_metadata["numpy_position"],
        "numpy_has_gauss": rng_metadata["numpy_has_gauss"],
        "numpy_cached_gaussian": rng_metadata["numpy_cached_gaussian"],
        "torch_cpu": arrays["rng_torch_cpu"],
        "torch_cuda": [arrays[f"rng_torch_cuda_{index}"] for index in range(cuda_count)],
    }
    return arrays, rows, rng_state


def run_sequence(run_dir: Path, sequence_name: str) -> int:
    validate_protocol()
    require_parent_run_lock(run_dir)
    if sequence_name not in SEQUENCES:
        raise ValueError(f"unknown sequence {sequence_name!r}")
    if not valid_resource_smoke(run_dir):
        raise RuntimeError("host-qualified actual-path resource smoke is required before sequence execution")
    provenance = validate_execution_provenance(run_dir)
    fixture = load_fixture(run_dir, provenance)
    seed, order = SEQUENCES[sequence_name]
    final_metadata_path = run_dir / f"sequence_{sequence_name}.json"
    final_npz_path = run_dir / f"sequence_{sequence_name}.npz"
    if final_metadata_path.exists() or final_npz_path.exists():
        raise RuntimeError(f"refusing to overwrite existing final sequence artifact for {sequence_name}")
    prior_attempts: list[int] = []
    for pattern in (
        f"sequence_{sequence_name}_failure_attempt_*.json",
        f"sequence_{sequence_name}_completion_candidate_attempt_*.npz",
    ):
        for path in run_dir.glob(pattern):
            try:
                prior_attempts.append(int(path.stem.rsplit("_", 1)[-1]))
            except ValueError:
                continue
    attempt_index = max(prior_attempts, default=0) + 1
    candidate_npz_path = run_dir / f"sequence_{sequence_name}_completion_candidate_attempt_{attempt_index:03d}.npz"
    if candidate_npz_path.exists():
        raise RuntimeError(f"refusing to overwrite candidate artifact: {candidate_npz_path}")
    before = resource_snapshot()
    require_safe_resources(before)
    torch_module = seed_process_once(seed)
    torch_module.cuda.empty_cache()
    torch_module.cuda.reset_peak_memory_stats()
    heartbeat = run_dir / f"sequence_{sequence_name}_heartbeat.json"
    monitor = ResourceMonitor(torch_module, heartbeat)
    monitor.start()
    model = processor = None
    raw_chunks = np.full((LOGICAL_KEY_COUNT, *RAW_CHUNK_SHAPE), np.nan, dtype=np.float32)
    processed_chunks = np.full((LOGICAL_KEY_COUNT, RAW_CHUNK_SHAPE[0], 7), np.nan, dtype=np.float32)
    latencies = np.full(LOGICAL_KEY_COUNT, np.nan, dtype=np.float64)
    draw_positions = np.full(LOGICAL_KEY_COUNT, -1, dtype=np.int64)
    input_hashes: list[str | None] = [None] * LOGICAL_KEY_COUNT
    rows: list[dict[str, Any]] = []
    partial_arrays, rows, saved_rng_state = load_sequence_partial(
        run_dir, sequence_name, seed, order, provenance
    )
    if partial_arrays:
        raw_chunks = partial_arrays["raw_chunks"]
        processed_chunks = partial_arrays["processed_chunks"]
        latencies = partial_arrays["latencies_seconds"]
        draw_positions = partial_arrays["draw_positions"]
        for row in rows:
            input_hashes[int(row["logical_index"])] = str(row["input_sha256"])
    completed_keys = {int(row["logical_index"]) for row in rows}
    restored_prefix_queries = len(completed_keys)
    result: dict[str, Any] = {
        "schema_version": "epoch6.schedule_stage0.sequence.v1",
        "sequence": sequence_name,
        "root_seed": seed,
        "execution_order": order,
        "started_at": utc_now(),
        "attempt_index": attempt_index,
        "resumed_completed_logical_indices": sorted(completed_keys),
        "pid": os.getpid(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "provenance": provenance,
        "simulator_actions_executed": 0,
        "reward_success_done_read": False,
        "exceptions": [],
        "resources_before": before,
    }
    exit_code = 1
    try:
        model, processor, runtime = load_xvla(torch_module)
        result["runtime"] = runtime
        if saved_rng_state is not None:
            restore_rng_state(saved_rng_state, torch_module)
        for draw_position, key_index in enumerate(order[restored_prefix_queries:], start=restored_prefix_queries):
            if int(meminfo()["swap_used_bytes"]) != 0:
                raise RuntimeError("nonzero swap use observed before inference query")
            model_inputs, input_hash = prepare_query_inputs(fixture, processor, model, torch_module)
            input_hashes[key_index] = input_hash
            started = time.monotonic()
            with torch_module.no_grad():
                action = model.generate_actions(**model_inputs, steps=DENOISING_STEPS)
            torch_module.cuda.synchronize()
            latency = time.monotonic() - started
            raw = action.float().detach().cpu().numpy().squeeze(0).astype(np.float32, copy=False)
            if raw.shape != RAW_CHUNK_SHAPE:
                raise RuntimeError(f"unexpected raw chunk shape {raw.shape}")
            if not np.isfinite(raw).all():
                raise RuntimeError(f"nonfinite action values for logical key {key_index}")
            processed = raw_to_processed_7d(raw)
            raw_chunks[key_index] = raw
            processed_chunks[key_index] = processed
            latencies[key_index] = latency
            draw_positions[key_index] = draw_position
            row = {
                "logical_index": key_index,
                "logical_key": logical_key(key_index),
                "logical_key_string": canonical_key_string(key_index),
                "draw_position": draw_position,
                "input_sha256": input_hash,
                "raw_chunk_sha256": hash_array(raw),
                "processed_7d_chunk_sha256": hash_array(processed),
                "latency_seconds": latency,
                "completed_attempt_index": attempt_index,
            }
            rows.append(row)
            completed_keys.add(key_index)
            rng_state = capture_rng_state(torch_module)
            write_sequence_partial(
                run_dir,
                sequence_name,
                seed,
                order,
                provenance,
                raw_chunks,
                processed_chunks,
                latencies,
                draw_positions,
                rows,
                rng_state,
            )
        if len(set(input_hashes)) != 1:
            raise RuntimeError("processor/proprio tensor hash changed across identical requests")
        if int(meminfo()["swap_used_bytes"]) != 0:
            raise RuntimeError("nonzero swap use observed after sequence")
        np.savez_compressed(
            candidate_npz_path,
            raw_chunks=raw_chunks,
            processed_chunks=processed_chunks,
            latencies_seconds=latencies,
            draw_positions=draw_positions,
        )
        result.update(
            {
                "rows": rows,
                "input_sha256": input_hashes[0],
                "raw_npz_sha256": sha256_file(candidate_npz_path),
                "completed_queries": LOGICAL_KEY_COUNT,
                "resume_restored_prefix_queries": restored_prefix_queries,
                "completed_queries_replayed": 0,
                "status": "SEQUENCE_COMPLETE",
            }
        )
        exit_code = 0
    except Exception as exc:
        result["exceptions"].append(f"{type(exc).__name__}: {exc}")
        result["traceback"] = traceback.format_exc()
        result["completed_queries"] = len(rows)
        result["rows"] = rows
        result["resume_restored_prefix_queries"] = restored_prefix_queries
        result["completed_queries_replayed"] = 0
        result["status"] = "SEQUENCE_FAILED"
    finally:
        monitor_result = monitor.stop()
        result["resource_monitor"] = monitor_result
        result["resources_after"] = resource_snapshot(torch_module)
        result["completed_at"] = utc_now()
        if (
            monitor_result["maximum_swap_used_bytes"] != 0
            or monitor_result["samples"] < 1
            or monitor_result["exceptions"]
        ):
            result["exceptions"].extend(monitor_result["exceptions"])
            if monitor_result["samples"] < 1:
                result["exceptions"].append("resource monitor captured zero samples")
            result["status"] = "SEQUENCE_FAILED_TELEMETRY_OR_SWAP"
            exit_code = 1
        if exit_code == 0:
            candidate_npz_path.replace(final_npz_path)
            write_json(final_metadata_path, result)
            write_text(run_dir / f"sequence_{sequence_name}_exit_code.txt", "0\n")
        else:
            failure_stem = f"sequence_{sequence_name}_failure_attempt_{attempt_index:03d}"
            write_json(run_dir / f"{failure_stem}.json", result)
            write_text(run_dir / f"{failure_stem}_exit_code.txt", "1\n")
        del model, processor
    return exit_code


def load_sequence(run_dir: Path, name: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    metadata = json.loads((run_dir / f"sequence_{name}.json").read_text(encoding="utf-8"))
    if metadata["status"] != "SEQUENCE_COMPLETE" or metadata["exceptions"]:
        raise RuntimeError(f"sequence {name} is not complete and exception-free")
    expected_seed, expected_order = SEQUENCES[name]
    if metadata.get("protocol_sha256") != EXPECTED_PROTOCOL_SHA256:
        raise RuntimeError(f"sequence {name} protocol hash mismatch")
    if metadata.get("root_seed") != expected_seed or metadata.get("execution_order") != expected_order:
        raise RuntimeError(f"sequence {name} seed or order mismatch")
    if metadata.get("provenance") != validate_execution_provenance(run_dir):
        raise RuntimeError(f"sequence {name} provenance mismatch")
    if metadata.get("completed_queries") != LOGICAL_KEY_COUNT or len(metadata.get("rows", [])) != LOGICAL_KEY_COUNT:
        raise RuntimeError(f"sequence {name} query count mismatch")
    if metadata.get("simulator_actions_executed") != 0 or metadata.get("reward_success_done_read"):
        raise RuntimeError(f"sequence {name} outcome-suppression mismatch")
    monitor = metadata.get("resource_monitor", {})
    if monitor.get("samples", 0) < 1 or monitor.get("exceptions") or monitor.get("maximum_swap_used_bytes") != 0:
        raise RuntimeError(f"sequence {name} resource telemetry is invalid")
    path = run_dir / f"sequence_{name}.npz"
    if sha256_file(path) != metadata["raw_npz_sha256"]:
        raise RuntimeError(f"sequence {name} NPZ hash mismatch")
    with np.load(path, allow_pickle=False) as payload:
        arrays = {key: np.asarray(payload[key]).copy() for key in payload.files}
    expected_draw_positions = np.asarray(
        [expected_order.index(index) for index in range(LOGICAL_KEY_COUNT)], dtype=np.int64
    )
    if not np.array_equal(arrays.get("draw_positions"), expected_draw_positions):
        raise RuntimeError(f"sequence {name} draw-position mapping mismatch")
    rows_by_key = {int(row["logical_index"]): row for row in metadata["rows"]}
    if set(rows_by_key) != set(range(LOGICAL_KEY_COUNT)):
        raise RuntimeError(f"sequence {name} logical-key coverage mismatch")
    observed_input_hashes: set[str] = set()
    for key_index in range(LOGICAL_KEY_COUNT):
        row = rows_by_key[key_index]
        if row["logical_key"] != logical_key(key_index):
            raise RuntimeError(f"sequence {name} logical-key payload mismatch for {key_index}")
        if row["draw_position"] != int(expected_draw_positions[key_index]):
            raise RuntimeError(f"sequence {name} row draw position mismatch for {key_index}")
        if row["raw_chunk_sha256"] != hash_array(arrays["raw_chunks"][key_index]):
            raise RuntimeError(f"sequence {name} raw hash mismatch for {key_index}")
        if row["processed_7d_chunk_sha256"] != hash_array(arrays["processed_chunks"][key_index]):
            raise RuntimeError(f"sequence {name} processed hash mismatch for {key_index}")
        observed_input_hashes.add(row["input_sha256"])
    if observed_input_hashes != {metadata.get("input_sha256")}:
        raise RuntimeError(f"sequence {name} per-query input hashes mismatch")
    return metadata, arrays


def adjudicate_arrays(
    sequences: Mapping[str, Mapping[str, np.ndarray]],
    metadata: Mapping[str, Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    for name in SEQUENCES:
        if name not in sequences or name not in metadata:
            raise ValueError(f"missing sequence {name}")
    a = sequences["A"]
    repeat = sequences["A_repeat"]
    reversed_order = sequences["B"]
    reference = sequences["C"]
    raw_a = np.asarray(a["raw_chunks"])
    raw_repeat = np.asarray(repeat["raw_chunks"])
    raw_b = np.asarray(reversed_order["raw_chunks"])
    raw_c = np.asarray(reference["raw_chunks"])
    for array in (raw_a, raw_repeat, raw_b, raw_c):
        if array.shape != (LOGICAL_KEY_COUNT, *RAW_CHUNK_SHAPE) or not np.isfinite(array).all():
            raise ValueError("sequence array is malformed or nonfinite")
    processed_a = np.asarray(a["processed_chunks"])
    processed_b = np.asarray(reversed_order["processed_chunks"])

    hash_a = [hash_array(raw_a[index]) for index in range(LOGICAL_KEY_COUNT)]
    hash_repeat = [hash_array(raw_repeat[index]) for index in range(LOGICAL_KEY_COUNT)]
    hash_b = [hash_array(raw_b[index]) for index in range(LOGICAL_KEY_COUNT)]
    hash_c = [hash_array(raw_c[index]) for index in range(LOGICAL_KEY_COUNT)]
    cold_rms = [normalized_rms(raw_a[index], raw_repeat[index]) for index in range(LOGICAL_KEY_COUNT)]
    order_rms = [normalized_rms(raw_a[index], raw_b[index]) for index in range(LOGICAL_KEY_COUNT)]
    reference_rms = [normalized_rms(raw_a[index], raw_c[index]) for index in range(LOGICAL_KEY_COUNT)]
    gripper = [gripper_disagreement(processed_a[index], processed_b[index]) for index in range(LOGICAL_KEY_COUNT)]
    cold_matches = [
        hash_a[index] == hash_repeat[index]
        or cold_rms[index] <= float(gates["same_order_cold_restart_normalized_rms_max_if_hash_differs"])
        for index in range(LOGICAL_KEY_COUNT)
    ]
    changed = [hash_a[index] != hash_b[index] for index in range(LOGICAL_KEY_COUNT)]
    median_order = float(np.median(order_rms))
    median_reference = float(np.median(reference_rms))
    ratio = (
        median_order / median_reference
        if median_reference > 1e-12
        else (math.inf if median_order > 1e-12 else 0.0)
    )
    exception_count = sum(
        len(metadata[name].get("exceptions", []))
        + len(metadata[name].get("resource_monitor", {}).get("exceptions", []))
        for name in SEQUENCES
    )
    input_hashes = {metadata[name].get("input_sha256") for name in SEQUENCES}
    provenance_hashes = {
        json.dumps(metadata[name].get("provenance"), sort_keys=True, separators=(",", ":"))
        for name in SEQUENCES
    }
    integrity = {
        "zero_exceptions": exception_count <= int(gates["exception_count_max"]),
        "identical_input_hash_across_sequences": len(input_hashes) == 1 and None not in input_hashes,
        "identical_provenance_across_sequences": len(provenance_hashes) == 1 and "null" not in provenance_hashes,
        "draw_positions_match_frozen_orders": all(
            np.asarray(sequences[name]["draw_positions"]).tolist()
            == [SEQUENCES[name][1].index(index) for index in range(LOGICAL_KEY_COUNT)]
            for name in SEQUENCES
        ),
        "no_simulator_actions": all(metadata[name]["simulator_actions_executed"] == 0 for name in SEQUENCES),
        "no_outcomes_read": all(not metadata[name]["reward_success_done_read"] for name in SEQUENCES),
        "zero_swap_use": all(
            metadata[name]["resource_monitor"]["maximum_swap_used_bytes"] == 0 for name in SEQUENCES
        ),
    }
    cold_gate = all(cold_matches)
    changed_fraction = float(np.mean(changed))
    changed_gate = changed_fraction >= float(gates["reversed_order_changed_hash_fraction_min"])
    ratio_gate = ratio >= float(gates["median_order_rms_over_independent_noise_rms_min"])
    if not cold_gate:
        decision = "EVALUATION_INVALID_CANNOT_ISOLATE_SCHEDULE"
    elif not all(integrity.values()):
        decision = "PROBLEM_GATE_IMPLEMENTATION_OR_RESOURCE_FAILURE"
    elif not (changed_gate and ratio_gate):
        decision = "NO_MATERIAL_ACTION_LEVEL_SCHEDULE_DEPENDENCE"
    else:
        decision = "ACTION_LEVEL_SCHEDULE_DEPENDENCE_GO"
    return {
        "same_order_hash_match_fraction": float(np.mean([x == y for x, y in zip(hash_a, hash_repeat)])),
        "same_order_effective_match_fraction": float(np.mean(cold_matches)),
        "same_order_normalized_rms": cold_rms,
        "reversed_order_changed_hash_fraction": changed_fraction,
        "reversed_order_changed_keys": [index for index, value in enumerate(changed) if value],
        "order_normalized_rms": order_rms,
        "independent_reference_normalized_rms": reference_rms,
        "median_order_normalized_rms": median_order,
        "median_independent_reference_normalized_rms": median_reference,
        "median_order_over_reference_ratio": ratio,
        "processed_gripper_disagreement": gripper,
        "median_processed_gripper_disagreement": float(np.median(gripper)),
        "integrity": integrity,
        "gates_passed": {
            "cold_restart": cold_gate,
            "reversed_order_changed_hash_fraction": changed_gate,
            "order_effect_relative_to_independent_reference": ratio_gate,
            "integrity": all(integrity.values()),
        },
        "exception_count": exception_count,
        "final_decision": decision,
    }


def adjudicate(run_dir: Path) -> dict[str, Any]:
    protocol = validate_protocol()
    metadata: dict[str, dict[str, Any]] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for name in SEQUENCES:
        metadata[name], arrays[name] = load_sequence(run_dir, name)
    metrics = adjudicate_arrays(arrays, metadata, protocol["stage0"]["gates"])
    result = {
        "schema_version": "epoch6.schedule_stage0.result.v1",
        "completed_at": utc_now(),
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
        "checkpoint_manifest_sha256": json.loads(
            (run_dir / "checkpoint_manifest.json").read_text(encoding="utf-8")
        )["manifest_sha256"],
        "fixture_manifest_sha256": sha256_file(run_dir / "fixture_manifest.json"),
        "sequence_metadata_sha256": {
            name: sha256_file(run_dir / f"sequence_{name}.json") for name in SEQUENCES
        },
        "execution_type": "DISCOVERY_PROBLEM_VERIFICATION_OUTCOME_SUPPRESSED",
        "ours_executed": False,
        "training_happened": False,
        "simulator_actions_executed": 0,
        "reward_success_done_read": False,
        "metrics": metrics,
        "final_decision": metrics["final_decision"],
        "closed_loop_authorized": metrics["final_decision"] == "ACTION_LEVEL_SCHEDULE_DEPENDENCE_GO",
        "method_design_authorized": False,
        "paper_generation_authorized": False,
    }
    write_json(run_dir / "result.json", result)
    return result


def valid_completed_step(run_dir: Path, step: str) -> bool:
    if step == "capture":
        try:
            load_fixture(run_dir)
            return True
        except Exception:
            return False
    if step == "resource-smoke":
        return valid_resource_smoke(run_dir)
    if step in SEQUENCES:
        try:
            load_sequence(run_dir, step)
            return True
        except Exception:
            return False
    return False


def valid_resource_smoke(run_dir: Path) -> bool:
    try:
        internal_path = run_dir / "resource_smoke.json"
        host_path = run_dir / "resource_smoke_host.json"
        internal = json.loads(internal_path.read_text(encoding="utf-8"))
        host = json.loads(host_path.read_text(encoding="utf-8-sig"))
        return bool(
            sha256_file(RESOURCE_AMENDMENT_PATH) == EXPECTED_RESOURCE_AMENDMENT_SHA256
            and internal["status"] == "ACTUAL_PATH_RESOURCE_SMOKE_PASS"
            and internal["model_inference_calls"] == 1
            and internal["raw_chunk_shape"] == list(RAW_CHUNK_SHAPE)
            and internal["raw_chunk_finite"]
            and internal["resource_monitor"]["samples"] >= 1
            and not internal["resource_monitor"]["exceptions"]
            and internal["resource_monitor"]["maximum_swap_used_bytes"] == 0
            and internal["runtime"]["parameter_devices"] == ["cuda:0"]
            and not internal["runtime"]["device_map_requested"]
            and not internal["runtime"]["cpu_or_disk_model_offload"]
            and internal["provenance"] == validate_execution_provenance(run_dir)
            and host["schema_version"] == "epoch6.schedule_stage0.host_resource_smoke.v2"
            and host["resource_governance_mode"] == "CALIBRATED_OUTCOME_FREE_V1"
            and host["resource_amendment_sha256"] == EXPECTED_RESOURCE_AMENDMENT_SHA256
            and host["final_decision"] == "EPOCH6_STAGE0_RESOURCE_SMOKE_PASS_CALIBRATED"
            and host["child_exit_code"] == 0
            and host["idle_control_valid"]
            and not host["sustained_paging_detected"]
            and not host["oom_or_kill_signature_detected"]
            and host["clean_state_restored"]
            and host["gpu_release_verified"]
            and host["scientific_gate_rows"] == 0
            and host["simulator_actions_executed"] == 0
            and not host["reward_success_done_read"]
            and host["internal_report_sha256"] == sha256_file(internal_path)
            and host["protocol_sha256"] == EXPECTED_PROTOCOL_SHA256
            and host["monitor_script_sha256"]
            == sha256_file(REPO_ROOT / "scripts" / "monitor_epoch6_schedule_stage0_smoke.ps1")
        )
    except Exception:
        return False


def launch_child(run_dir: Path, mode: str, sequence: str | None = None) -> None:
    command = [sys.executable, str(Path(__file__).resolve()), "--mode", mode, "--run-dir", str(run_dir), "--child"]
    label = mode
    if sequence is not None:
        command.extend(["--sequence", sequence])
        label = f"sequence_{sequence}"
    attempt_index = 1
    while (run_dir / f"{label}_launch_attempt_{attempt_index:03d}.json").exists():
        attempt_index += 1
    stdout_path = run_dir / f"{label}_attempt_{attempt_index:03d}_stdout.log"
    stderr_path = run_dir / f"{label}_attempt_{attempt_index:03d}_stderr.log"
    launch_path = run_dir / f"{label}_launch_attempt_{attempt_index:03d}.json"
    started_at = utc_now()
    started = time.monotonic()
    child_environment = os.environ.copy()
    child_seed = SEQUENCES[sequence][0] if sequence is not None else ROOT_SEED
    child_environment["PYTHONHASHSEED"] = str(child_seed)
    child_environment["EPOCH6_PARENT_RUN_LOCK"] = str((run_dir / "run.lock.json").resolve())
    resources_before = resource_snapshot()
    write_json(
        launch_path,
        {
            "status": "running",
            "command": command,
            "python_hash_seed": child_seed,
            "started_at": started_at,
            "parent_pid": os.getpid(),
            "resources_before": resources_before,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    )
    with stdout_path.open("x", encoding="utf-8") as stdout, stderr_path.open("x", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=child_environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    resources_after = resource_snapshot()
    teardown_valid = bool(
        resources_after["system"]["swap_used_bytes"] == 0
        and resources_after["nvidia_smi"]["free_mib"] >= resources_before["nvidia_smi"]["free_mib"] - 512
    )
    write_json(
        launch_path,
        {
            "status": "completed" if completed.returncode == 0 and teardown_valid else "failed",
            "command": command,
            "python_hash_seed": child_seed,
            "started_at": started_at,
            "completed_at": utc_now(),
            "elapsed_seconds": time.monotonic() - started,
            "exit_code": completed.returncode,
            "teardown_valid": teardown_valid,
            "resources_before": resources_before,
            "resources_after": resources_after,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
        },
    )
    if completed.returncode != 0 or not teardown_valid:
        raise RuntimeError(
            f"child {label} failed or did not release resources; see {launch_path}, {stdout_path}, and {stderr_path}"
        )


def acquire_run_lock(run_dir: Path) -> Path:
    lock_path = run_dir / "run.lock.json"
    payload = {
        "pid": os.getpid(),
        "created_at": utc_now(),
        "command": sys.argv,
        "protocol_sha256": EXPECTED_PROTOCOL_SHA256,
    }
    while True:
        try:
            descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, indent=2, sort_keys=True)
                handle.write("\n")
            return lock_path
        except FileExistsError:
            existing = json.loads(lock_path.read_text(encoding="utf-8"))
            existing_pid = int(existing.get("pid", -1))
            try:
                os.kill(existing_pid, 0)
            except (ProcessLookupError, PermissionError, ValueError):
                stale_path = run_dir / f"stale_run_lock_{int(time.time())}.json"
                lock_path.replace(stale_path)
                continue
            raise RuntimeError(f"another Stage 0 parent holds the run lock with PID {existing_pid}")


def release_run_lock(run_dir: Path, lock_path: Path) -> None:
    if not lock_path.exists():
        return
    history_path = run_dir / f"released_run_lock_{int(time.time())}.json"
    lock_path.replace(history_path)


def run_all(run_dir: Path, resume: bool) -> int:
    if run_dir.exists() and not resume:
        raise FileExistsError(f"run directory already exists; use a new path or --resume: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    lock_path = acquire_run_lock(run_dir)
    parent_attempt = 1 + len(list(run_dir.glob("parent_attempt_*_exit_code.txt")))
    parent_heartbeat = run_dir / "heartbeat.json"
    exit_code = 1
    try:
        write_json(parent_heartbeat, {"status": "static_preflight", "pid": os.getpid(), "updated_at": utc_now()})
        if not (resume and (run_dir / "static_preflight.json").is_file()):
            static_preflight(run_dir)
        if not (resume and valid_completed_step(run_dir, "capture")):
            write_json(parent_heartbeat, {"status": "capture_fixture", "pid": os.getpid(), "updated_at": utc_now()})
            capture_fixture(run_dir)
        if not valid_completed_step(run_dir, "resource-smoke"):
            raise RuntimeError(
                "host-qualified resource smoke is absent; run scripts/monitor_epoch6_schedule_stage0_smoke.ps1"
            )
        for name in SEQUENCES:
            if resume and valid_completed_step(run_dir, name):
                continue
            write_json(
                parent_heartbeat,
                {"status": f"sequence_{name}", "pid": os.getpid(), "updated_at": utc_now()},
            )
            launch_child(run_dir, "sequence", name)
        write_json(parent_heartbeat, {"status": "adjudicate", "pid": os.getpid(), "updated_at": utc_now()})
        result = adjudicate(run_dir)
        write_json(
            parent_heartbeat,
            {
                "status": "completed",
                "pid": os.getpid(),
                "updated_at": utc_now(),
                "final_decision": result["final_decision"],
            },
        )
        exit_code = 0
    except Exception as exc:
        write_json(
            run_dir / f"parent_attempt_{parent_attempt:03d}_failure.json",
            {
                "status": "PROBLEM_GATE_IMPLEMENTATION_OR_RESOURCE_FAILURE",
                "failed_at": utc_now(),
                "exception": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(),
                "simulator_actions_executed": 0,
                "reward_success_done_read": False,
            },
        )
        write_json(
            parent_heartbeat,
            {"status": "failed", "pid": os.getpid(), "updated_at": utc_now()},
        )
    finally:
        write_text(run_dir / f"parent_attempt_{parent_attempt:03d}_exit_code.txt", f"{exit_code}\n")
        if exit_code == 0:
            write_text(run_dir / "exit_code.txt", "0\n")
        release_run_lock(run_dir, lock_path)
    return exit_code


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["run-all", "capture", "resource-smoke", "sequence", "adjudicate", "preflight"],
        default="run-all",
    )
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--sequence", choices=sorted(SEQUENCES))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = Path(args.run_dir).resolve()
    if args.mode == "run-all":
        return run_all(run_dir, bool(args.resume))
    run_dir.mkdir(parents=True, exist_ok=True)
    if args.mode == "preflight":
        static_preflight(run_dir)
        return 0
    if args.mode == "capture":
        capture_fixture(run_dir)
        return 0
    if args.mode == "resource-smoke":
        if not args.child:
            raise RuntimeError("resource-smoke mode is host-monitor child-only")
        return run_resource_smoke(run_dir)
    if args.mode == "sequence":
        if not args.child:
            raise RuntimeError("sequence mode is parent-child-only")
        if args.sequence is None:
            raise ValueError("--sequence is required for sequence mode")
        return run_sequence(run_dir, args.sequence)
    if args.mode == "adjudicate":
        adjudicate(run_dir)
        return 0
    raise AssertionError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())
