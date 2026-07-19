#!/usr/bin/env python
"""Run the one hash-frozen A2C2 fidelity-corrected local-port path.

Modes are metadata-only preflight, outcome-suppressed actual-path smoke,
the single corrected matched panel, and report-only adjudication.  This file
contains no training path and no Ours method.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, is_dataclass
from enum import Enum
import gc
import json
import math
import os
from pathlib import Path
import random
import sys
import threading
import time
import traceback
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.a2c2_fidelity_corrected import (  # noqa: E402
    BASE_MODEL_SHA256,
    BASE_REVISION,
    CHUNK_SIZE,
    CHECKPOINT_COMPATIBLE_COMMIT,
    CONDITIONS,
    DEVELOPMENT_SMOKE_IDENTITY,
    EVAL_TASK_IDS,
    FIDELITY_LABEL,
    OFFICIAL_COMMIT,
    PRIOR_MODEL_SHA256,
    PRIOR_REVISION,
    ROOT_SEED,
    VERIFICATION_INIT_STATE_IDS,
    adjudicate_panel,
    episode_key,
    expected_panel_keys,
    noise_seed,
    phase_feature,
    refresh_action_plan,
    rotate_live_rgb_180,
    sha256_file,
    verify_artifact_configs,
)


SCHEMA_VERSION = 1
DATE_KST = "2026-07-19"
SUITE = "libero_spatial"
MAX_STEPS = 220
NUM_STEPS_WAIT = 10
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
TASKS = {
    0: {
        "global_task_index": 34,
        "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    },
    2: {
        "global_task_index": 32,
        "instruction": "pick up the black bowl from table center and place it on the plate",
    },
    4: {
        "global_task_index": 31,
        "instruction": "pick up the black bowl in the top drawer of the wooden cabinet and place it on the plate",
    },
    8: {
        "global_task_index": 36,
        "instruction": "pick up the black bowl next to the plate and place it on the plate",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("metadata_preflight", "smoke", "panel", "adjudicate"), required=True)
    parser.add_argument(
        "--base-root",
        default=(
            "/mnt/c/assets/checkpoints/a2c2_official/smolvla_libero_spatial_scratch/"
            f"{BASE_REVISION}"
        ),
    )
    parser.add_argument(
        "--prior-root",
        default=(
            "/mnt/c/assets/checkpoints/a2c2_official/"
            "residual_transformer_libero_spatial_add_vlm_context/"
            f"{PRIOR_REVISION}"
        ),
    )
    parser.add_argument(
        "--vlm-root",
        default="/mnt/c/assets/hf_home/HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
    )
    parser.add_argument("--hf-home", default="/mnt/c/assets/hf_home")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--output-root", default="/mnt/c/Users/jiheo/tca_map/runs/a2c2_fidelity_corrected")
    parser.add_argument("--run-id")
    parser.add_argument("--result-json")
    parser.add_argument("--result-md")
    return parser.parse_args()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, Enum):
        return value.value
    raise TypeError(type(value).__name__)


def config_snapshot(config: Any) -> dict[str, Any]:
    """Serialize both current and checkpoint-compatible author configs."""

    if hasattr(config, "to_dict"):
        return dict(config.to_dict())
    if is_dataclass(config):
        return asdict(config)
    raise TypeError(f"unsupported policy config type: {type(config).__name__}")


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def write_md(path: Path, title: str, payload: Mapping[str, Any]) -> None:
    lines = [
        f"# {title}",
        "",
        f"Date: `{DATE_KST} KST`",
        "",
        f"Fidelity label: `{FIDELITY_LABEL}`",
        "",
        f"Final decision: `{payload.get('final_decision')}`",
        "",
        "```json",
        json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default),
        "```",
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True, default=_json_default) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if line.strip():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"invalid JSONL at {path}:{line_number}: {exc}") from exc
    return rows


def set_runtime_environment(args: argparse.Namespace) -> None:
    os.environ["HF_HOME"] = str(args.hf_home)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["LIBERO_CONFIG_PATH"] = str(args.libero_config_dir)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def meminfo() -> dict[str, int | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
            key, raw = line.split(":", 1)
            values[key] = int(raw.strip().split()[0]) * 1024
    except Exception:
        pass
    return {
        "mem_total_bytes": values.get("MemTotal"),
        "mem_available_bytes": values.get("MemAvailable"),
        "swap_total_bytes": values.get("SwapTotal"),
        "swap_free_bytes": values.get("SwapFree"),
    }


def resource_snapshot(torch_mod: Any | None = None) -> dict[str, Any]:
    import psutil

    memory = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    result: dict[str, Any] = {
        "pid": os.getpid(),
        "rss_mib": round(process.memory_info().rss / 1024**2, 3),
        "system_total_gib": round(memory.total / 1024**3, 3),
        "system_used_gib": round(memory.used / 1024**3, 3),
        "system_used_fraction": round(memory.percent / 100.0, 6),
        "meminfo": meminfo(),
    }
    if torch_mod is not None and torch_mod.cuda.is_available():
        props = torch_mod.cuda.get_device_properties(0)
        result.update(
            {
                "cuda_pid": os.getpid(),
                "gpu_name": props.name,
                "vram_total_mib": round(props.total_memory / 1024**2, 3),
                "vram_allocated_mib": round(torch_mod.cuda.memory_allocated(0) / 1024**2, 3),
                "vram_reserved_mib": round(torch_mod.cuda.memory_reserved(0) / 1024**2, 3),
                "vram_peak_allocated_mib": round(torch_mod.cuda.max_memory_allocated(0) / 1024**2, 3),
                "vram_peak_reserved_mib": round(torch_mod.cuda.max_memory_reserved(0) / 1024**2, 3),
            }
        )
    return result


class ResourceSampler:
    def __init__(self, torch_mod: Any, interval: float = 0.25) -> None:
        self.torch = torch_mod
        self.interval = interval
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._run, name="a2c2-fidelity-resource", daemon=True)
        self.samples = 0
        self.peak_rss_mib = 0.0
        self.peak_system_used_gib = 0.0
        self.peak_system_used_fraction = 0.0
        self.peak_vram_allocated_mib = 0.0
        self.peak_vram_reserved_mib = 0.0
        self.exceptions: list[str] = []

    def start(self) -> None:
        self.thread.start()

    def _run(self) -> None:
        while not self.stop_event.is_set():
            try:
                snapshot = resource_snapshot(self.torch)
                self.samples += 1
                self.peak_rss_mib = max(self.peak_rss_mib, float(snapshot["rss_mib"]))
                self.peak_system_used_gib = max(self.peak_system_used_gib, float(snapshot["system_used_gib"]))
                self.peak_system_used_fraction = max(
                    self.peak_system_used_fraction,
                    float(snapshot["system_used_fraction"]),
                )
                self.peak_vram_allocated_mib = max(
                    self.peak_vram_allocated_mib,
                    float(snapshot.get("vram_allocated_mib") or 0.0),
                )
                self.peak_vram_reserved_mib = max(
                    self.peak_vram_reserved_mib,
                    float(snapshot.get("vram_reserved_mib") or 0.0),
                )
            except Exception as exc:  # pragma: no cover - live telemetry
                self.exceptions.append(f"{type(exc).__name__}: {exc}")
            self.stop_event.wait(self.interval)

    def stop(self) -> dict[str, Any]:
        self.stop_event.set()
        self.thread.join(timeout=5.0)
        return {
            "samples": self.samples,
            "interval_seconds": self.interval,
            "peak_rss_mib": round(self.peak_rss_mib, 3),
            "peak_system_used_gib": round(self.peak_system_used_gib, 3),
            "peak_system_used_fraction": round(self.peak_system_used_fraction, 6),
            "peak_vram_allocated_mib": round(self.peak_vram_allocated_mib, 3),
            "peak_vram_reserved_mib": round(self.peak_vram_reserved_mib, 3),
            "exceptions": self.exceptions,
        }


def source_audit() -> dict[str, Any]:
    source_root = Path("/home/jiheon/assets/repos/a2c2-libero")
    sources = {
        "evaluation_libero.py": (
            source_root / "eval_libero/evaluation_libero.py",
            "941E5894CC0A607F35E0295F174BA27E88B957F51E24154CBAB26B75D6CCF400",
        ),
        "modeling_residual_transformer.py": (
            source_root / "src/lerobot/policies/residual_transformer/modeling_residual_transformer.py",
            "190093422E71A59633F4C101619392CE53864B8C75079F053BB98154A7E352DF",
        ),
        "configuration_residual_transformer.py": (
            source_root / "src/lerobot/policies/residual_transformer/configuration_residual_transformer.py",
            "76DD959D317F7091A4BFAEABD31CA0E8B028AD539D7BC660256B28ED1C6B2B83",
        ),
    }
    rows = {}
    for name, (path, expected) in sources.items():
        actual = sha256_file(path)
        rows[name] = {"path": str(path), "expected_sha256": expected, "actual_sha256": actual, "valid": actual == expected}
    return {"official_commit": OFFICIAL_COMMIT, "files": rows, "valid": all(row["valid"] for row in rows.values())}


def metadata_preflight(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    artifacts = verify_artifact_configs(Path(args.base_root), Path(args.prior_root))
    sources = source_audit()
    result = {
        "schema_version": SCHEMA_VERSION,
        "date": f"{DATE_KST} KST",
        "job_classification": "PRIOR_METADATA_PREFLIGHT",
        "fidelity_label": FIDELITY_LABEL,
        "official_commit": OFFICIAL_COMMIT,
        "base_revision": BASE_REVISION,
        "prior_revision": PRIOR_REVISION,
        "artifact_audit": artifacts,
        "source_audit": sources,
        "model_loaded": False,
        "cuda_execution": False,
        "training_happened": False,
        "rollout_happened": False,
        "ours_designed_or_executed": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    valid = bool(artifacts["valid"] and sources["valid"])
    result["final_decision"] = "A2C2_CORRECTED_METADATA_PREFLIGHT_PASS" if valid else "A2C2_CORRECTED_METADATA_PREFLIGHT_FAIL"
    return result


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    value = np.asarray(quat, dtype=np.float64).copy()
    value[3] = np.clip(value[3], -1.0, 1.0)
    denominator = np.sqrt(1.0 - value[3] * value[3])
    if math.isclose(float(denominator), 0.0):
        return np.zeros(3)
    return (value[:3] * 2.0 * math.acos(float(value[3]))) / denominator


def prepare_observation(obs: Mapping[str, Any], task_description: str, torch_mod: Any) -> dict[str, Any]:
    device = torch_mod.device("cuda:0")
    top = rotate_live_rgb_180(np.asarray(obs["agentview_image"]))
    wrist = rotate_live_rgb_180(np.asarray(obs["robot0_eye_in_hand_image"]))
    state = np.concatenate(
        (
            np.asarray(obs["robot0_eef_pos"]),
            _quat2axisangle(np.asarray(obs["robot0_eef_quat"])),
            np.asarray(obs["robot0_gripper_qpos"]),
        )
    ).astype(np.float32)
    return {
        "observation.images.image": torch_mod.from_numpy(top / 255.0).permute(2, 0, 1).to(torch_mod.float32).to(device).unsqueeze(0),
        "observation.images.wrist_image": torch_mod.from_numpy(wrist / 255.0).permute(2, 0, 1).to(torch_mod.float32).to(device).unsqueeze(0),
        "observation.state": torch_mod.from_numpy(state).to(torch_mod.float32).to(device).unsqueeze(0),
        "task": task_description,
    }


def make_env(task_id: int, seed: int) -> tuple[Any, Any, str]:
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv

    suite = benchmark.get_benchmark_dict()[SUITE]()
    task = suite.get_task(int(task_id))
    task_bddl_file = Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env = OffScreenRenderEnv(
        bddl_file_name=str(task_bddl_file),
        camera_heights=256,
        camera_widths=256,
    )
    env.seed(int(seed))
    return env, suite, task.language


def reset_env(env: Any, suite: Any, task_id: int, init_state_id: int) -> Any:
    initial_states = suite.get_task_init_states(int(task_id))
    if int(init_state_id) >= len(initial_states):
        raise ValueError(f"init_state_id {init_state_id} unavailable; task has {len(initial_states)} states")
    env.reset()
    obs = env.set_init_state(initial_states[int(init_state_id)])
    for _ in range(NUM_STEPS_WAIT):
        obs, _, _, _ = env.step(LIBERO_DUMMY_ACTION)
    return obs


def make_noise(base_policy: Any, task_id: int, init_state_id: int, chunk_index: int, torch_mod: Any) -> Any:
    generator = torch_mod.Generator(device="cuda")
    generator.manual_seed(noise_seed(task_id, init_state_id, chunk_index))
    shape = (1, int(base_policy.config.chunk_size), int(base_policy.config.max_action_dim))
    return torch_mod.randn(shape, device="cuda", dtype=torch_mod.float32, generator=generator)


def parameter_audit(model: Any) -> dict[str, Any]:
    parameters = list(model.parameters())
    devices = sorted({str(parameter.device) for parameter in parameters})
    dtypes = sorted({str(parameter.dtype) for parameter in parameters})
    return {
        "total_parameters": int(sum(parameter.numel() for parameter in parameters)),
        "trainable_parameters": int(sum(parameter.numel() for parameter in parameters if parameter.requires_grad)),
        "devices": devices,
        "dtypes": dtypes,
    }


def load_models(args: argparse.Namespace, torch_mod: Any) -> tuple[Any, Any, dict[str, Any]]:
    import inspect

    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.residual_transformer.configuration_residual_transformer import ResidualTransformerConfig  # noqa: F401
    from lerobot.policies.residual_transformer.modeling_residual_transformer import ResidualTransformerPolicy
    from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig  # noqa: F401
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not torch_mod.cuda.is_available():
        raise RuntimeError("WSL_CUDA_BLOCKED: torch.cuda.is_available() is false")
    started = time.monotonic()
    torch_mod.cuda.empty_cache()
    torch_mod.cuda.reset_peak_memory_stats()
    base_config = PreTrainedConfig.from_pretrained(str(args.base_root), local_files_only=True)
    base_config.vlm_model_name = str(args.vlm_root)
    base_config.device = "cuda"
    base_policy = SmolVLAPolicy.from_pretrained(
        str(args.base_root),
        config=base_config,
        local_files_only=True,
        strict=True,
    )
    prior_config = PreTrainedConfig.from_pretrained(str(args.prior_root), local_files_only=True)
    prior_config.device = "cuda"
    prior_policy = ResidualTransformerPolicy.from_pretrained(
        str(args.prior_root),
        config=prior_config,
        local_files_only=True,
        strict=True,
    )
    base_policy.eval()
    prior_policy.eval()
    execution_source = Path(inspect.getfile(ResidualTransformerPolicy)).resolve()
    if "a2c2-libero-checkpoint-compat-c197a01" not in str(execution_source):
        raise RuntimeError(f"checkpoint-compatible author source not active: {execution_source}")
    prior_projection_shape = list(prior_policy.model.image_proj.weight.shape)
    if prior_projection_shape != [512, 512]:
        raise RuntimeError(f"unexpected checkpoint-compatible image projection: {prior_projection_shape}")
    audit = {
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "base": parameter_audit(base_policy),
        "prior": parameter_audit(prior_policy),
        "base_config": config_snapshot(base_config),
        "prior_config": config_snapshot(prior_config),
        "base_model_sha256": sha256_file(Path(args.base_root) / "model.safetensors"),
        "prior_model_sha256": sha256_file(Path(args.prior_root) / "model.safetensors"),
        "checkpoint_compatible_author_commit": CHECKPOINT_COMPATIBLE_COMMIT,
        "execution_source_path": str(execution_source),
        "prior_image_projection_shape": prior_projection_shape,
        "serializer_compatibility_repair": (
            "strict-load public prior with the author's immediately preceding "
            "checkpoint-compatible source; no tensor reshape or non-strict load"
        ),
        "strict_safetensor_load": True,
        "no_cpu_or_disk_offload": all(
            device == "cuda:0"
            for device in parameter_audit(base_policy)["devices"] + parameter_audit(prior_policy)["devices"]
        ),
        "resource_after_load": resource_snapshot(torch_mod),
    }
    if audit["base_model_sha256"] != BASE_MODEL_SHA256 or audit["prior_model_sha256"] != PRIOR_MODEL_SHA256:
        raise RuntimeError("loaded model identity changed after metadata preflight")
    return base_policy, prior_policy, audit


def trace_episode(
    *,
    base_policy: Any,
    prior_policy: Any,
    condition_name: str,
    task_id: int,
    init_state_id: int,
    torch_mod: Any,
) -> dict[str, Any]:
    condition = CONDITIONS[condition_name]
    e = int(condition["execution_horizon"])
    d = int(condition["inference_delay"])
    with_prior = bool(condition["with_prior"])
    env, suite, task_description = make_env(task_id, ROOT_SEED)
    started = time.monotonic()
    base_policy.reset()
    prior_policy.reset()
    obs = reset_env(env, suite, task_id, init_state_id)
    action_plan: deque[dict[str, Any]] = deque()
    pending_actions: list[dict[str, Any]] = []
    first_chunk = True
    chunk_index = 0
    base_forward_count = 0
    prior_forward_count = 0
    image_rotation_count = 0
    correction_deltas: list[float] = []
    action_finite = True
    action_legal = True
    max_action_abs = 0.0
    success = False
    step = 0
    exception: dict[str, Any] | None = None
    try:
        while step < MAX_STEPS:
            if not action_plan:
                observation = prepare_observation(obs, task_description, torch_mod)
                image_rotation_count += 2
                noise = make_noise(base_policy, task_id, init_state_id, chunk_index, torch_mod)
                with torch_mod.inference_mode():
                    raw_chunk = base_policy.predict_action_chunk(dict(observation), noise=noise)
                if tuple(raw_chunk.shape) != (1, CHUNK_SIZE, 7):
                    raise RuntimeError(f"unexpected base chunk shape {tuple(raw_chunk.shape)}")
                hidden = getattr(base_policy, "vlm_hidden", None)
                if hidden is None or tuple(hidden.shape) != (1, 960):
                    raise RuntimeError(f"unexpected base vlm_hidden {None if hidden is None else tuple(hidden.shape)}")
                chunk = raw_chunk[0].detach().to(torch_mod.float32).cpu().numpy()
                entries = [
                    {
                        "action": chunk[index].copy(),
                        "time_offset": index,
                        "source_chunk": chunk.copy(),
                        "vlm_hidden": hidden.detach().clone(),
                    }
                    for index in range(CHUNK_SIZE)
                ]
                plan, pending_actions = refresh_action_plan(
                    new_entries=entries,
                    pending_entries=pending_actions,
                    execution_horizon=e,
                    inference_delay=d,
                    first_chunk=first_chunk,
                )
                action_plan = deque(plan)
                first_chunk = False
                chunk_index += 1
                base_forward_count += 1

            entry = action_plan.popleft()
            action = np.asarray(entry["action"], dtype=np.float32)
            if with_prior:
                observation = prepare_observation(obs, task_description, torch_mod)
                image_rotation_count += 2
                prior_batch = dict(observation)
                prior_batch["action"] = torch_mod.from_numpy(action).to("cuda", dtype=torch_mod.float32).unsqueeze(0)
                prior_batch["base_action_chunk"] = (
                    torch_mod.from_numpy(np.asarray(entry["source_chunk"], dtype=np.float32))
                    .to("cuda", dtype=torch_mod.float32)
                    .unsqueeze(0)
                )
                prior_batch["time_feature"] = (
                    torch_mod.from_numpy(phase_feature(int(entry["time_offset"])))
                    .to("cuda", dtype=torch_mod.float32)
                    .unsqueeze(0)
                )
                prior_batch["vlm_hidden"] = entry["vlm_hidden"].to("cuda", dtype=torch_mod.float32)
                with torch_mod.inference_mode():
                    updated = prior_policy.predict_action_chunk(prior_batch)
                if tuple(updated.shape) != (1, 1, 7):
                    raise RuntimeError(f"unexpected prior output shape {tuple(updated.shape)}")
                corrected = updated[0, 0].detach().to(torch_mod.float32).cpu().numpy()
                correction_deltas.append(float(np.mean(np.abs(corrected - action))))
                action = corrected.astype(np.float32)
                prior_forward_count += 1

            finite = bool(np.isfinite(action).all())
            legal = bool(np.max(np.abs(action)) <= 1.000001)
            action_finite = action_finite and finite
            action_legal = action_legal and legal
            max_action_abs = max(max_action_abs, float(np.max(np.abs(action))))
            if not finite:
                raise RuntimeError("nonfinite action")
            obs, _, done, _ = env.step(action)
            step += 1
            success = bool(done or env.check_success())
            if success:
                break
    except Exception as exc:
        exception = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines()[-40:],
        }
    finally:
        env.close()

    return {
        "condition": condition_name,
        "suite": SUITE,
        "task_id": int(task_id),
        "global_task_index": int(TASKS[task_id]["global_task_index"]),
        "instruction": task_description,
        "official_init_state_id": int(init_state_id),
        "execution_horizon": e,
        "inference_delay": d,
        "uses_prior": with_prior,
        "success": success,
        "episode_length": step,
        "max_steps": MAX_STEPS,
        "base_model_forward_count": base_forward_count,
        "prior_module_forward_count": prior_forward_count,
        "prior_mean_abs_correction": round(float(np.mean(correction_deltas)), 9) if correction_deltas else 0.0,
        "prior_max_mean_abs_correction": round(float(np.max(correction_deltas)), 9) if correction_deltas else 0.0,
        "image_rotation_count": image_rotation_count,
        "reset_stabilization_steps": NUM_STEPS_WAIT,
        "action_finite": action_finite,
        "action_legal": action_legal,
        "max_action_abs": round(max_action_abs, 9),
        "uses_expert_or_future_action_at_live_inference": False,
        "base_chunk_noise_seed_rule": "SHA256(root_seed,task_id,init_state_id,chunk_index)",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "resource_at_return": resource_snapshot(torch_mod),
        "exception": exception,
    }


def run_empirical(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    if args.mode not in {"smoke", "panel"}:
        raise ValueError(args.mode)
    set_runtime_environment(args)
    set_seed(ROOT_SEED)
    run_id = args.run_id or time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    run_dir = Path(args.output_root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    status_path = run_dir / "status.json"
    heartbeat_path = run_dir / "heartbeat.json"
    episodes_path = run_dir / "episodes.jsonl"
    started = time.monotonic()
    sampler = ResourceSampler(torch)
    sampler.start()
    before = resource_snapshot(torch)
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "date": f"{DATE_KST} KST",
        "run_id": run_id,
        "run_dir": str(run_dir),
        "mode": args.mode,
        "job_classification": "PRIOR_ACTUAL_PATH_PREFLIGHT" if args.mode == "smoke" else "PRIOR_CLOSED_LOOP_ROLLOUT",
        "fidelity_label": FIDELITY_LABEL,
        "official_commit": OFFICIAL_COMMIT,
        "checkpoint_compatible_author_commit": CHECKPOINT_COMPATIBLE_COMMIT,
        "base_revision": BASE_REVISION,
        "prior_revision": PRIOR_REVISION,
        "training_happened": False,
        "ours_designed_or_executed": False,
        "expert_action_replay_counted_as_success": False,
        "before_resources": before,
        "exception": None,
    }
    write_json(status_path, {**report, "state": "loading_models", "pid": os.getpid()})
    base_policy = None
    prior_policy = None
    try:
        base_policy, prior_policy, load_audit = load_models(args, torch)
        report["model_load_audit"] = load_audit
        if args.mode == "smoke":
            task_id, init_state_id = DEVELOPMENT_SMOKE_IDENTITY
            requested = [
                ("BASE_STANDARD_E10_D0", task_id, init_state_id),
                ("PRIOR_DELAYED_E40_D10", task_id, init_state_id),
            ]
            completed_technical = []
            for condition_name, task_id, init_state_id in requested:
                trace = trace_episode(
                    base_policy=base_policy,
                    prior_policy=prior_policy,
                    condition_name=condition_name,
                    task_id=task_id,
                    init_state_id=init_state_id,
                    torch_mod=torch,
                )
                technical = {key: value for key, value in trace.items() if key != "success"}
                technical.update(
                    {
                        "task_success_persisted": False,
                        "task_success_counted": False,
                        "scientific_episode_row": False,
                    }
                )
                append_jsonl(episodes_path, technical)
                completed_technical.append(technical)
                write_json(
                    heartbeat_path,
                    {
                        "state": "running",
                        "pid": os.getpid(),
                        "completed": len(completed_technical),
                        "planned": len(requested),
                        "last_key": [condition_name, task_id, init_state_id],
                        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
            smoke_valid = bool(
                len(completed_technical) == 2
                and all(row["exception"] is None for row in completed_technical)
                and all(row["action_finite"] and row["action_legal"] for row in completed_technical)
                and all(row["base_model_forward_count"] > 0 for row in completed_technical)
                and completed_technical[1]["prior_module_forward_count"] > 0
                and completed_technical[1]["prior_mean_abs_correction"] > 0.0
                and all(row["image_rotation_count"] > 0 for row in completed_technical)
                and int(meminfo().get("swap_total_bytes") or 0) == 0
                and bool(load_audit["no_cpu_or_disk_offload"])
            )
            report.update(
                {
                    "task_success_persisted": False,
                    "task_success_counted": False,
                    "scientific_episode_rows": 0,
                    "technical_traces": completed_technical,
                    "final_decision": (
                        "A2C2_CORRECTED_ACTUAL_PATH_SMOKE_PASS"
                        if smoke_valid
                        else "A2C2_CORRECTED_ACTUAL_PATH_SMOKE_FAIL"
                    ),
                }
            )
        else:
            existing = read_jsonl(episodes_path)
            existing_keys = [episode_key(row) for row in existing]
            if len(existing_keys) != len(set(existing_keys)):
                raise RuntimeError("duplicate scientific keys in durable episode file")
            if not set(existing_keys).issubset(expected_panel_keys()):
                raise RuntimeError("durable episode file contains keys outside frozen panel")
            rows_by_key = {episode_key(row): row for row in existing}
            planned = [
                (condition_name, task_id, init_state_id)
                for condition_name in CONDITIONS
                for task_id in EVAL_TASK_IDS
                for init_state_id in VERIFICATION_INIT_STATE_IDS
            ]
            for index, key in enumerate(planned, start=1):
                if key in rows_by_key:
                    continue
                condition_name, task_id, init_state_id = key
                row = trace_episode(
                    base_policy=base_policy,
                    prior_policy=prior_policy,
                    condition_name=condition_name,
                    task_id=task_id,
                    init_state_id=init_state_id,
                    torch_mod=torch,
                )
                append_jsonl(episodes_path, row)
                rows_by_key[key] = row
                write_json(
                    heartbeat_path,
                    {
                        "state": "running",
                        "pid": os.getpid(),
                        "completed": len(rows_by_key),
                        "planned": len(planned),
                        "last_key": list(key),
                        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    },
                )
                write_json(status_path, {**report, "state": "running_panel", "completed": len(rows_by_key), "planned": len(planned), "pid": os.getpid()})
                if row["exception"] is not None:
                    raise RuntimeError(f"episode failed for {key}: {row['exception']}")
            rows = [rows_by_key[key] for key in planned if key in rows_by_key]
            adjudication = adjudicate_panel(rows)
            report.update(
                {
                    "episodes_path": str(episodes_path),
                    "planned_scientific_rows": 45,
                    "completed_scientific_rows": len(rows),
                    "duplicate_scientific_keys": len(rows) - len({episode_key(row) for row in rows}),
                    "adjudication": adjudication,
                    "final_decision": adjudication["final_decision"],
                }
            )
    except Exception as exc:
        report["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc().splitlines()[-60:],
        }
        if args.mode == "smoke":
            report["final_decision"] = "A2C2_CORRECTED_ACTUAL_PATH_SMOKE_FAIL"
        else:
            rows = read_jsonl(episodes_path)
            report["completed_scientific_rows"] = len(rows)
            report["adjudication"] = adjudicate_panel(rows, infrastructure_failure=True)
            report["final_decision"] = "CORRECTED_A2C2_IMPLEMENTATION_OR_RESOURCE_FAILURE"
    finally:
        sampled = sampler.stop()
        report["peak_resources"] = sampled
        report["after_resources"] = resource_snapshot(torch)
        report["elapsed_seconds"] = round(time.monotonic() - started, 3)
        report["pid"] = os.getpid()
        report["swap_total_bytes_at_end"] = int(meminfo().get("swap_total_bytes") or 0)
        try:
            del base_policy
            del prior_policy
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception as exc:  # pragma: no cover - cleanup telemetry
            report["cleanup_exception"] = f"{type(exc).__name__}: {exc}"
        write_json(run_dir / "result.json", report)
        write_md(run_dir / "result.md", "A2C2 Fidelity-Corrected Execution", report)
        write_json(status_path, {**report, "state": "completed"})
    return report


def adjudicate_existing(args: argparse.Namespace) -> dict[str, Any]:
    if not args.run_id:
        raise ValueError("--run-id is required for adjudicate")
    run_dir = Path(args.output_root) / args.run_id
    rows = read_jsonl(run_dir / "episodes.jsonl")
    adjudication = adjudicate_panel(rows)
    return {
        "schema_version": SCHEMA_VERSION,
        "date": f"{DATE_KST} KST",
        "job_classification": "REPORT_ONLY",
        "fidelity_label": FIDELITY_LABEL,
        "run_id": args.run_id,
        "episodes_path": str(run_dir / "episodes.jsonl"),
        "completed_scientific_rows": len(rows),
        "adjudication": adjudication,
        "training_happened": False,
        "rollout_happened_in_this_invocation": False,
        "ours_designed_or_executed": False,
        "final_decision": adjudication["final_decision"],
    }


def main() -> int:
    args = parse_args()
    set_runtime_environment(args)
    if args.mode == "metadata_preflight":
        report = metadata_preflight(args)
        default_json = REPO_ROOT / "reports/a2c2_prior/fidelity_corrected_metadata_preflight_result.json"
        default_md = REPO_ROOT / "reports/a2c2_prior/fidelity_corrected_metadata_preflight_result.md"
    elif args.mode in {"smoke", "panel"}:
        report = run_empirical(args)
        print(json.dumps({"run_id": report.get("run_id"), "final_decision": report["final_decision"]}, indent=2))
        return 0 if report["final_decision"] in {
            "A2C2_CORRECTED_ACTUAL_PATH_SMOKE_PASS",
            "CORRECTED_A2C2_PRIOR_IMPROVES_AND_LEAVES_RESIDUAL",
            "CORRECTED_A2C2_PRIOR_SATURATES_DELAY",
            "CORRECTED_A2C2_PRIOR_NO_IMPROVEMENT",
            "CORRECTED_A2C2_BASE_NOT_COMPETENT",
        } else 1
    else:
        report = adjudicate_existing(args)
        default_json = REPO_ROOT / "reports/a2c2_prior/fidelity_corrected_adjudication_result.json"
        default_md = REPO_ROOT / "reports/a2c2_prior/fidelity_corrected_adjudication_result.md"

    result_json = Path(args.result_json) if args.result_json else default_json
    result_md = Path(args.result_md) if args.result_md else default_md
    write_json(result_json, report)
    write_md(result_md, "A2C2 Fidelity-Corrected Result", report)
    print(json.dumps({"final_decision": report["final_decision"]}, indent=2))
    return 0 if str(report["final_decision"]).endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
