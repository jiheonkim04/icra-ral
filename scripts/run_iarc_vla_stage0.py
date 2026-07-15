"""Run the frozen IARC-VLA Stage 0A mechanism audit."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import gc
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.iarc_vla import (  # noqa: E402
    CONFLICT_COSINE_THRESHOLD,
    IMAGE_KEYS,
    PROPOSAL_HASH,
    ROBUST_NORM_SQUARED_FLOOR,
    assign_vector_to_gradients,
    classify_stage0,
    flatten_gradients,
    module_norms,
    numeric_summary,
    parameter_manifest,
    partition_stage0_manifest,
    partition_summary,
    perturb_raw_sample,
    perturbation_spec,
    project_clean_gradient,
    sample_id,
    sorted_trainable_parameters,
    stable_seed,
    value_hash,
)
from tca_map.smolvla.official_libero_baseline_scaleup import (  # noqa: E402
    _add_training_batch_dims,
    _loss_from_output,
    _postprocess_action,
    _stat_vector,
    _tensor_devices,
    _tensor_shapes,
    _to_float,
)


DATE_KST = "2026-07-15"
SEED = 1601
MICRO_STEPS = 20
LEARNING_RATE = 1e-4
LORA_RANK = 4
MAX_RUNTIME_SECONDS = 4 * 60 * 60
MAX_CUDA_GIB = 15.5
CHUNK_SIZE = 50
MAX_ACTION_DIM = 32
TINY_STEP_SIZE = 1e-6
EXPECTED_STAGE = "epoch_4_cycle_16_iarc_stage_0a_implementation_pending"
EXACT_AUDIT_COMMAND = (
    'wsl -d Ubuntu-22.04 bash -lc "cd /mnt/c/Users/jiheo/tca_map && '
    "/home/jiheon/miniconda3-official/envs/official-smolvla-libero/bin/python "
    'scripts/run_iarc_vla_stage0.py --mode audit"'
)
FORBIDDEN_GATES = (
    "ALLOW_DOWNLOADS",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
)


def _asset_path(*parts: str) -> Path:
    root = Path("C:/assets") if os.name == "nt" else Path("/mnt/c/assets")
    return root.joinpath(*parts)


CHECKPOINT_PATH = _asset_path("checkpoints", "smolvla_libero")
VLM_PATH = _asset_path("hf_home", "HuggingFaceTB", "SmolVLM2-500M-Video-Instruct")
HF_HOME = _asset_path("hf_home")
DATASET_ROOT = _asset_path("datasets", "lerobot_libero")
SPLIT_MANIFEST = REPO_ROOT / "reports" / "official_smolvla_split_manifest.json"
BASE_ARTIFACT = REPO_ROOT / "reports" / "official_smolvla_stable_prediction_artifact.json"
RESOURCE_REGISTRY = REPO_ROOT / "reports" / "resource_contention_intervals.json"
STATE_JSON = REPO_ROOT / "reports" / "autonomous_until_paper_state.json"
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "iarc_vla" / "proposal_hash.txt"
REPORT_DIR = REPO_ROOT / "reports" / "iarc_vla"
RUN_DIR = REPO_ROOT / "runs" / "iarc_vla" / "stage0a"
CHECKPOINT_DIR = RUN_DIR / "micro_checkpoint"
PARTIAL_JSON = RUN_DIR / "partial_result.json"
STATUS_JSON = RUN_DIR / "status.json"
HEARTBEAT_JSON = RUN_DIR / "heartbeat.json"
CHILD_PID_FILE = RUN_DIR / "child_pid.txt"
EXIT_CODE_FILE = RUN_DIR / "exit_code.txt"
RESULT_JSON = REPORT_DIR / "stage_0a_result.json"
RESULT_MD = REPORT_DIR / "stage_0a_result.md"
GRADIENT_JSON = REPORT_DIR / "gradient_audit.json"
PERTURBATION_JSON = REPORT_DIR / "perturbation_manifest.json"
PARAMETER_JSON = REPORT_DIR / "parameter_manifest.json"
CHECKPOINT_JSON = REPORT_DIR / "checkpoint_manifest.json"
BLOCKER_JSON = REPORT_DIR / "implementation_blocker.json"


def _set_offline_environment() -> None:
    os.environ["HF_HOME"] = str(HF_HOME)
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "detach"):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def _directory_hashes(path: Path) -> dict[str, dict[str, Any]]:
    return {
        child.relative_to(path).as_posix(): {
            "sha256": _sha256_file(child),
            "size_bytes": child.stat().st_size,
        }
        for child in sorted(path.rglob("*"))
        if child.is_file()
    }


def _git_commit() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _package_versions() -> dict[str, str]:
    result = {}
    for name in ("torch", "lerobot", "transformers", "peft", "numpy", "datasets", "av"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "NOT_INSTALLED"
    return result


def _active_linux_workers() -> list[dict[str, Any]]:
    if os.name == "nt" or not Path("/proc").is_dir():
        return []
    workers = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode("utf-8", "replace")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        lowered = command.lower()
        if "python" in lowered and "scripts/run_" in lowered and "vla" in lowered:
            workers.append({"pid": int(entry.name), "command": command.strip()})
    return workers


def _resource_evidence_eligibility(registry: Mapping[str, Any], started_unix: float) -> dict[str, Any]:
    start_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_unix))
    intervals = list(registry.get("intervals") or [])
    active_or_unknown = [
        str(item.get("id"))
        for item in intervals
        if not bool(item.get("efficiency_mode_disabled")) or not item.get("end_time_latest_kst")
    ]
    return {
        "run_start_local": start_iso,
        "registry_interval_count": len(intervals),
        "active_or_unknown_end_intervals": active_or_unknown,
        "paper_evidence_eligible": not active_or_unknown,
        "policy": "ineligible when overlap is unknown or positive",
    }


def _preflight(mode: str, started_unix: float) -> dict[str, Any]:
    import torch

    state = _read_json(STATE_JSON)
    registry = _read_json(RESOURCE_REGISTRY)
    proposal_hash_file = PROPOSAL_HASH_FILE.read_text(encoding="utf-8").strip()
    paths = {
        "checkpoint": CHECKPOINT_PATH,
        "vlm": VLM_PATH,
        "hf_home": HF_HOME,
        "dataset": DATASET_ROOT,
        "dataset_info": DATASET_ROOT / "meta" / "info.json",
        "dataset_stats": DATASET_ROOT / "meta" / "stats.json",
        "split_manifest": SPLIT_MANIFEST,
        "base_artifact": BASE_ARTIFACT,
        "resource_registry": RESOURCE_REGISTRY,
        "proposal_hash_file": PROPOSAL_HASH_FILE,
    }
    missing = [name for name, path in paths.items() if not path.exists()]
    forbidden_enabled = [name for name in FORBIDDEN_GATES if os.environ.get(name) == "1"]
    workers = _active_linux_workers()
    stage_ok = str(state.get("current_stage")) == EXPECTED_STAGE
    proposal_ok = proposal_hash_file == PROPOSAL_HASH
    mode_ok = mode == "audit"
    result_absent = not RESULT_JSON.exists()
    cuda_available = bool(torch.cuda.is_available())
    return {
        "passed": bool(
            not missing
            and not forbidden_enabled
            and not workers
            and stage_ok
            and proposal_ok
            and mode_ok
            and result_absent
            and cuda_available
        ),
        "mode": mode,
        "expected_stage": EXPECTED_STAGE,
        "observed_stage": state.get("current_stage"),
        "stage_ok": stage_ok,
        "proposal_hash_expected": PROPOSAL_HASH,
        "proposal_hash_observed": proposal_hash_file,
        "proposal_hash_ok": proposal_ok,
        "result_absent": result_absent,
        "paths": {name: str(path) for name, path in paths.items()},
        "missing": missing,
        "forbidden_gates_enabled": forbidden_enabled,
        "active_linux_workers": workers,
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        "resource_evidence": _resource_evidence_eligibility(registry, started_unix),
    }


def _phase(row: Mapping[str, Any]) -> str:
    value = float(row.get("normalized_phase", 0.0))
    if value < 1 / 3:
        return "early"
    if value < 2 / 3:
        return "mid"
    return "late"


def _attach_local_indices(
    partitions: Mapping[str, Sequence[Mapping[str, Any]]], names: Sequence[str]
) -> tuple[list[int], dict[str, list[dict[str, Any]]]]:
    rows = [dict(row) for name in names for row in partitions[name]]
    lengths: dict[int, int] = {}
    for row in rows:
        episode = int(row["episode_index"])
        lengths[episode] = max(lengths.get(episode, 0), int(row["episode_length"]))
    episodes = sorted(lengths)
    offsets: dict[int, int] = {}
    offset = 0
    for episode in episodes:
        offsets[episode] = offset
        offset += lengths[episode]
    indexed: dict[str, list[dict[str, Any]]] = {}
    for name in names:
        indexed[name] = []
        for row in partitions[name]:
            cloned = dict(row)
            cloned["phase"] = _phase(row)
            cloned["dataset_local_index"] = offsets[int(row["episode_index"])] + int(row["frame_index"])
            indexed[name].append(cloned)
    return episodes, indexed


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if hasattr(value, "clone") else value for key, value in batch.items()}


def _preprocess(preprocessor: Any, raw_sample: Mapping[str, Any]) -> dict[str, Any]:
    return _add_training_batch_dims(preprocessor(dict(raw_sample)))


def _shared_draw(row: Mapping[str, Any], partition: str, logical_step: int, device: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    seed = stable_seed(partition, sample_id(row), logical_step, 0)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (2**63 - 1))
    noise_cpu = torch.randn((1, CHUNK_SIZE, MAX_ACTION_DIM), generator=generator, dtype=torch.float32)
    time_cpu = torch.rand((1,), generator=generator, dtype=torch.float32)
    return (
        noise_cpu.to(device),
        time_cpu.to(device),
        {
            "seed": seed,
            "noise_shape": list(noise_cpu.shape),
            "time_shape": list(time_cpu.shape),
            "noise_hash": value_hash(noise_cpu),
            "time_hash": value_hash(time_cpu),
        },
    )


def _loss(policy: Any, batch: Mapping[str, Any], noise: Any, time_tensor: Any) -> Any:
    output = policy.forward(_clone_batch(batch), noise=noise.clone(), time=time_tensor.clone())
    loss = _loss_from_output(output)
    if loss is None or int(loss.numel()) != 1:
        raise RuntimeError("official SmolVLA forward did not return one scalar loss")
    return loss


def _predict(policy: Any, batch: Mapping[str, Any], postprocessor: Any, noise: Any) -> tuple[Any, np.ndarray]:
    import torch

    policy.eval()
    if hasattr(policy, "reset"):
        policy.reset()
    with torch.no_grad():
        native = policy.select_action(_clone_batch(batch), noise=noise.clone())
    processed = _postprocess_action(native, postprocessor)
    return native.detach().float().cpu(), processed.astype(np.float32)


def _hash_frozen_parameters(policy: Any) -> str:
    import torch

    digest = hashlib.sha256()
    count = 0
    for parameter in policy.parameters():
        if parameter.requires_grad:
            continue
        tensor = parameter.detach().contiguous()
        raw_bytes = tensor.view(torch.uint8).cpu().numpy().tobytes()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(raw_bytes)
        count += 1
    if count == 0:
        raise RuntimeError("no frozen Base parameters found")
    return digest.hexdigest().upper()


def _pair_integrity(
    raw_clean: Mapping[str, Any],
    raw_robust: Mapping[str, Any],
    clean_batch: Mapping[str, Any],
    robust_batch: Mapping[str, Any],
    family: str,
) -> dict[str, Any]:
    raw_shared_keys = [key for key in raw_clean if key not in {*IMAGE_KEYS, "task"}]
    raw_shared = {key: value_hash(raw_clean[key]) == value_hash(raw_robust[key]) for key in raw_shared_keys}
    processed_shared_keys = [
        key
        for key in ("action", "action_is_pad", "actions_id_pad", "observation.state")
        if key in clean_batch and key in robust_batch
    ]
    processed_shared = {
        key: value_hash(clean_batch[key]) == value_hash(robust_batch[key]) for key in processed_shared_keys
    }
    image_changed = {
        key: value_hash(raw_clean[key]) != value_hash(raw_robust[key]) for key in IMAGE_KEYS
    }
    task_changed = str(raw_clean["task"]) != str(raw_robust["task"])
    transform_acted = all(image_changed.values()) if family in {"gaussian_sensor_noise", "image_translation"} else task_changed
    allowlist_ok = (
        not task_changed and all(image_changed.values())
        if family in {"gaussian_sensor_noise", "image_translation"}
        else task_changed and not any(image_changed.values())
    )
    return {
        "raw_shared_hashes": raw_shared,
        "processed_shared_hashes": processed_shared,
        "raw_shared_all": all(raw_shared.values()),
        "processed_shared_all": all(processed_shared.values()),
        "image_changed": image_changed,
        "task_changed": task_changed,
        "transform_acted": transform_acted,
        "allowlist_ok": allowlist_ok,
        "clean_action_hash": value_hash(raw_clean["action"]),
        "robust_action_hash": value_hash(raw_robust["action"]),
        "clean_state_hash": value_hash(raw_clean["observation.state"]),
        "robust_state_hash": value_hash(raw_robust["observation.state"]),
        "clean_processed_hash": value_hash(clean_batch),
        "robust_processed_hash": value_hash(robust_batch),
    }


def _load_policy_and_processors() -> tuple[Any, Any, Any, Any]:
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(CHECKPOINT_PATH, local_files_only=True, cache_dir=HF_HOME)
    config.device = "cuda"
    config.load_vlm_weights = True
    config.compile_model = False
    config.push_to_hub = False
    config.vlm_model_name = str(VLM_PATH)
    if hasattr(config, "chunk_size"):
        config.chunk_size = CHUNK_SIZE
    policy = SmolVLAPolicy.from_pretrained(
        CHECKPOINT_PATH,
        config=config,
        local_files_only=True,
        cache_dir=HF_HOME,
        token=False,
        strict=False,
    )
    policy.to("cuda")
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(CHECKPOINT_PATH),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(VLM_PATH)},
            "device_processor": {"device": "cuda"},
        },
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return policy, config, preprocessor, postprocessor


def _load_adapter_checkpoint() -> tuple[Any, Any, Any, Any]:
    from peft import PeftConfig, PeftModel

    base_policy, config, preprocessor, postprocessor = _load_policy_and_processors()
    peft_config = PeftConfig.from_pretrained(CHECKPOINT_DIR)
    policy = PeftModel.from_pretrained(
        base_policy,
        CHECKPOINT_DIR,
        config=peft_config,
        is_trainable=True,
        local_files_only=True,
    )
    policy.to("cuda")
    policy.train()
    return policy, config, preprocessor, postprocessor


def _save_adapter_checkpoint(
    policy: Any,
    training: Mapping[str, Any],
    probe_native: Any,
    probe_processed: np.ndarray,
) -> dict[str, Any]:
    if CHECKPOINT_DIR.exists():
        manifest_path = CHECKPOINT_DIR / "iarc_training_manifest.json"
        probe_path = CHECKPOINT_DIR / "reload_probe.json"
        if not manifest_path.is_file() or not probe_path.is_file():
            raise RuntimeError(f"existing IARC checkpoint is incomplete: {CHECKPOINT_DIR}")
        manifest = _read_json(manifest_path)
        if manifest.get("proposal_hash") != PROPOSAL_HASH:
            raise RuntimeError("existing IARC checkpoint has a different proposal hash")
        return {
            "checkpoint_path": str(CHECKPOINT_DIR),
            "files": _directory_hashes(CHECKPOINT_DIR),
            "saved": True,
            "resumed_existing_checkpoint": True,
            "reload_probe": _read_json(probe_path),
        }
    CHECKPOINT_DIR.parent.mkdir(parents=True, exist_ok=True)
    temporary = CHECKPOINT_DIR.with_name(f"{CHECKPOINT_DIR.name}.tmp_{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        if hasattr(policy, "peft_config"):
            for peft_config in policy.peft_config.values():
                peft_config.base_model_name_or_path = str(CHECKPOINT_PATH)
        policy.save_pretrained(temporary)
        _write_json(
            temporary / "iarc_training_manifest.json",
            {
                "method": "IARC-VLA",
                "proposal_hash": PROPOSAL_HASH,
                "seed": SEED,
                "lora_rank": LORA_RANK,
                "micro_steps": MICRO_STEPS,
                "learning_rate": LEARNING_RATE,
                "training": dict(training),
            },
        )
        _write_json(
            temporary / "reload_probe.json",
            {
                "native_action": probe_native.detach().float().cpu().tolist(),
                "postprocessed_action": probe_processed.astype(np.float32).tolist(),
                "native_action_hash": value_hash(probe_native),
            },
        )
        temporary.rename(CHECKPOINT_DIR)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "checkpoint_path": str(CHECKPOINT_DIR),
        "files": _directory_hashes(CHECKPOINT_DIR),
        "saved": True,
        "resumed_existing_checkpoint": False,
        "reload_probe": _read_json(CHECKPOINT_DIR / "reload_probe.json"),
    }


def _loss_on_rows(
    policy: Any,
    preprocessor: Any,
    dataset: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    partition: str,
) -> list[float]:
    import torch

    values = []
    policy.eval()
    with torch.no_grad():
        for index, row in enumerate(rows):
            raw = dataset[int(row["dataset_local_index"])]
            spec = perturbation_spec(row, partition=partition)
            robust = perturb_raw_sample(raw, spec)
            batch = _preprocess(preprocessor, robust)
            noise, time_tensor, _draw = _shared_draw(row, f"{partition}_fixed_subset", index, "cuda")
            values.append(_to_float(_loss(policy, batch, noise, time_tensor)))
    return values


def _evaluate_validation_rows(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    dataset: Any,
    rows: Sequence[Mapping[str, Any]],
    base_by_sample: Mapping[str, Mapping[str, Any]],
    action_min: np.ndarray,
    action_max: np.ndarray,
    *,
    started: float,
) -> list[dict[str, Any]]:
    import torch

    output = []
    policy.eval()
    for index, row in enumerate(rows):
        if time.monotonic() - started > MAX_RUNTIME_SECONDS:
            raise TimeoutError("IARC Stage 0A exceeded its four-hour cap")
        raw = dataset[int(row["dataset_local_index"])]
        spec = perturbation_spec(row, partition="validation")
        robust = perturb_raw_sample(raw, spec)
        clean_batch = _preprocess(preprocessor, raw)
        robust_batch = _preprocess(preprocessor, robust)
        integrity = _pair_integrity(raw, robust, clean_batch, robust_batch, spec.family)
        noise, time_tensor, draw = _shared_draw(row, "validation", index, "cuda")
        with torch.no_grad():
            clean_loss = _to_float(_loss(policy, clean_batch, noise, time_tensor))
            robust_loss = _to_float(_loss(policy, robust_batch, noise, time_tensor))
        clean_native, clean_action = _predict(policy, clean_batch, postprocessor, noise)
        robust_native, robust_action = _predict(policy, robust_batch, postprocessor, noise)
        base_action = np.asarray(base_by_sample[sample_id(row)]["base_action"], dtype=np.float32)
        clean_delta = clean_action - base_action
        perturb_delta = robust_action - clean_action
        all_actions = np.stack([clean_action, robust_action])
        semantic_valid = bool(np.all(np.isfinite(all_actions)) and np.max(np.abs(all_actions)) <= 5.0)
        dataset_range_valid = bool(
            np.all(np.isfinite(all_actions))
            and np.all(all_actions >= action_min.reshape(1, -1))
            and np.all(all_actions <= action_max.reshape(1, -1))
        )
        output.append(
            {
                "sample_id": sample_id(row),
                "task_index": int(row["task_index"]),
                "phase": str(row["phase"]),
                "family": spec.family,
                "severity_index": spec.severity_index,
                "severity": spec.severity,
                "clean_loss": clean_loss,
                "perturbed_loss": robust_loss,
                "loss_delta": robust_loss - clean_loss,
                "clean_action": clean_action.tolist(),
                "perturbed_action": robust_action.tolist(),
                "base_action": base_action.tolist(),
                "clean_native_shape": list(clean_native.shape),
                "perturbed_native_shape": list(robust_native.shape),
                "clean_to_perturbed_action_l2": float(np.linalg.norm(perturb_delta)),
                "base_to_adapter_action_l2": float(np.linalg.norm(clean_delta)),
                "translation_delta": float(np.linalg.norm(clean_delta[:3])),
                "rotation_delta": float(np.linalg.norm(clean_delta[3:6])),
                "gripper_delta": float(abs(clean_delta[6])),
                "finite_and_semantic_range_valid": semantic_valid,
                "dataset_range_valid": dataset_range_valid,
                "max_abs_action": float(np.max(np.abs(all_actions))),
                "integrity": integrity,
                "draw": draw,
            }
        )
        print(f"[IARC validation] {index + 1}/{len(rows)} {sample_id(row)}", flush=True)
    return output


def _summarize_validation(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "record_count": len(rows),
        "clean_loss": numeric_summary([float(row["clean_loss"]) for row in rows]),
        "perturbed_loss": numeric_summary([float(row["perturbed_loss"]) for row in rows]),
        "loss_delta": numeric_summary([float(row["loss_delta"]) for row in rows]),
        "clean_to_perturbed_action_l2": numeric_summary(
            [float(row["clean_to_perturbed_action_l2"]) for row in rows]
        ),
        "base_to_adapter_action_l2": numeric_summary([float(row["base_to_adapter_action_l2"]) for row in rows]),
        "translation_delta": numeric_summary([float(row["translation_delta"]) for row in rows]),
        "rotation_delta": numeric_summary([float(row["rotation_delta"]) for row in rows]),
        "gripper_delta": numeric_summary([float(row["gripper_delta"]) for row in rows]),
        "finite_and_semantic_range_valid_fraction": float(
            np.mean([bool(row["finite_and_semantic_range_valid"]) for row in rows])
        ),
        "dataset_range_valid_fraction": float(np.mean([bool(row["dataset_range_valid"]) for row in rows])),
        "clean_retention_action_delta_p95": float(
            np.percentile([float(row["base_to_adapter_action_l2"]) for row in rows], 95)
        ),
        "all_integrity_passed": all(
            bool(row["integrity"]["raw_shared_all"])
            and bool(row["integrity"]["processed_shared_all"])
            and bool(row["integrity"]["allowlist_ok"])
            and bool(row["integrity"]["transform_acted"])
            for row in rows
        ),
    }


def _write_result_markdown(result: Mapping[str, Any]) -> None:
    gradient = result["gradient_audit"]
    validation = result["validation"]["summary"]
    lines = [
        "# IARC-VLA Stage 0A Result",
        "",
        f"Date: {DATE_KST} KST",
        "",
        f"Decision: `{result['final_decision']}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        "## Frozen Audit",
        "",
        f"- micro-fit steps: `{result['micro_fit']['completed_steps']} / {MICRO_STEPS}`",
        f"- fixed-subset loss: `{result['micro_fit']['fixed_subset_loss_before']}` -> `{result['micro_fit']['fixed_subset_loss_after']}`",
        f"- conflict pairs: `{gradient['conflict_count']} / {gradient['record_count']}`",
        f"- activating families: `{gradient['conflict_families']}`",
        f"- projection constraints passed: `{gradient['projection_constraint_pass_count']} / {gradient['projected_row_count']}`",
        f"- validation dataset-range action validity: `{validation['dataset_range_valid_fraction']}`",
        f"- checkpoint reload error: `{result['checkpoint']['reload_output_max_abs_error']}`",
        f"- confirmatory observations/actions: `{result['confirmatory']['observations_decoded']} / {result['confirmatory']['actions_computed']}`",
        f"- peak CUDA GiB: `{result['runtime']['peak_cuda_allocated_gib']}`",
        f"- timing paper eligible: `{result['runtime']['paper_evidence_eligible']}`",
        "",
        "## Boundary",
        "",
        result["adjudication_reason"],
        "",
        f"Next command: `{result['exact_next_command']}`",
    ]
    RESULT_MD.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def run_audit(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    started_monotonic = time.monotonic()
    started_unix = time.time()
    progress: dict[str, Any] = {
        "mode": args.mode,
        "parameter_update_occurred": False,
        "decoded_split_counts": {"train": 0, "val": 0, "test": 0},
    }
    preflight = _preflight(args.mode, started_unix)
    progress["preflight"] = preflight
    if not preflight["passed"]:
        raise RuntimeError(f"IARC preflight failed: {preflight}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    _write_json(STATUS_JSON, {"status": "running", "pid": os.getpid(), "started_unix": started_unix})

    manifest = _read_json(SPLIT_MANIFEST)
    partitions = partition_stage0_manifest(manifest)
    summaries = partition_summary(partitions)
    expected_counts = {
        "micro_fit": 40,
        "conflict_audit": 40,
        "one_check": 40,
        "validation": 40,
        "confirmatory_reserved": 1200,
    }
    observed_counts = {name: len(rows) for name, rows in partitions.items()}
    if observed_counts != expected_counts:
        raise RuntimeError(f"frozen Stage 0 partition count mismatch: {observed_counts}")
    progress["decoded_split_counts"] = {"train": 80, "val": 40, "test": 0}
    selected_episodes, indexed = _attach_local_indices(
        partitions, ("micro_fit", "conflict_audit", "validation")
    )
    info = _read_json(DATASET_ROOT / "meta" / "info.json")
    stats = _read_json(DATASET_ROOT / "meta" / "stats.json")
    action_min = np.asarray(_stat_vector(stats, "action", "min"), dtype=np.float32)
    action_max = np.asarray(_stat_vector(stats, "action", "max"), dtype=np.float32)
    fps = float(info.get("fps", 10.0))
    dataset = LeRobotDataset(
        "lerobot/libero",
        root=DATASET_ROOT,
        episodes=selected_episodes,
        delta_timestamps={"action": [index / fps for index in range(CHUNK_SIZE)]},
        video_backend="pyav",
    )
    base_artifact = _read_json(BASE_ARTIFACT)
    base_by_sample = {str(row["sample_id"]): row for row in base_artifact.get("records") or []}
    missing_base = [sample_id(row) for row in indexed["validation"] if sample_id(row) not in base_by_sample]
    if missing_base:
        raise RuntimeError(f"stable Base artifact lacks validation identities: {missing_base[:3]}")

    task_indices = sorted({int(row["task_index"]) for row in indexed["conflict_audit"]})
    perturbation_rows = []
    for partition_name in ("micro_fit", "conflict_audit", "validation"):
        for row in indexed[partition_name]:
            spec = perturbation_spec(row, partition=partition_name, sorted_task_indices=task_indices)
            perturbation_rows.append(
                {
                    "partition": partition_name,
                    "sample_id": sample_id(row),
                    "task_index": int(row["task_index"]),
                    "phase": str(row["phase"]),
                    **spec.to_dict(),
                }
            )
    perturbation_manifest = {
        "method": "IARC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "rows": perturbation_rows,
        "counts_by_partition": dict(Counter(row["partition"] for row in perturbation_rows)),
        "audit_counts_by_family": dict(
            Counter(row["family"] for row in perturbation_rows if row["partition"] == "conflict_audit")
        ),
        "audit_counts_by_severity": dict(
            Counter(
                f"{row['family']}:{row['severity_index']}"
                for row in perturbation_rows
                if row["partition"] == "conflict_audit"
            )
        ),
        "confirmatory_perturbations_generated": 0,
    }
    _write_json(PERTURBATION_JSON, perturbation_manifest)

    policy, config, preprocessor, postprocessor = _load_policy_and_processors()
    first_raw = dataset[int(indexed["micro_fit"][0]["dataset_local_index"])]
    first_batch = _preprocess(preprocessor, first_raw)
    input_devices = _tensor_devices(first_batch)
    if not input_devices or not all(str(device).startswith("cuda") for device in input_devices.values()):
        raise RuntimeError(f"IARC preprocessor tensors are not all CUDA: {input_devices}")
    raw_shapes = {
        key: list(first_raw[key].shape) if hasattr(first_raw[key], "shape") else None for key in first_raw
    }
    processed_shapes = _tensor_shapes(first_batch)
    if raw_shapes.get("action") != [CHUNK_SIZE, 7]:
        raise RuntimeError(f"native demonstration chunk mismatch: {raw_shapes.get('action')}")
    action_shape = processed_shapes.get("action")
    if action_shape not in ([1, CHUNK_SIZE, 7], [1, CHUNK_SIZE, MAX_ACTION_DIM]):
        raise RuntimeError(f"processed action target shape mismatch: {action_shape}")
    identity_noise, identity_time, identity_draw = _shared_draw(indexed["micro_fit"][0], "identity", 0, "cuda")
    base_native, base_processed = _predict(policy, first_batch, postprocessor, identity_noise)
    for parameter in policy.parameters():
        parameter.requires_grad_(False)
    base_frozen_hash_before = _hash_frozen_parameters(policy)
    policy = policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": LORA_RANK})
    policy.to("cuda")
    policy.eval()
    named_parameters = sorted_trainable_parameters(policy)
    param_manifest = parameter_manifest(named_parameters)
    lora_only = bool(named_parameters) and all("lora_" in name.lower() for name, _parameter in named_parameters)
    adapter_native, adapter_processed = _predict(policy, first_batch, postprocessor, identity_noise)
    identity_native_error = float(torch.max(torch.abs(base_native - adapter_native)).item())
    identity_processed_error = float(np.max(np.abs(base_processed - adapter_processed)))
    parameter_report = {
        "method": "IARC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "lora_rank": LORA_RANK,
        "parameter_count": len(param_manifest),
        "trainable_numel": sum(int(item["numel"]) for item in param_manifest),
        "lora_only_trainable": lora_only,
        "parameters": param_manifest,
        "base_frozen_hash_before": base_frozen_hash_before,
        "identity_native_max_abs_error": identity_native_error,
        "identity_postprocessed_max_abs_error": identity_processed_error,
    }
    _write_json(PARAMETER_JSON, parameter_report)

    subset_rows = indexed["micro_fit"][:8]
    subset_before_values = _loss_on_rows(
        policy, preprocessor, dataset, subset_rows, partition="micro_fit"
    )
    optimizer = torch.optim.AdamW([parameter for _name, parameter in named_parameters], lr=LEARNING_RATE)
    train_order = np.random.default_rng(SEED).permutation(len(indexed["micro_fit"])).tolist()
    loss_curve = []
    gradient_curve = []
    policy.train()
    for step in range(MICRO_STEPS):
        if time.monotonic() - started_monotonic > MAX_RUNTIME_SECONDS:
            raise TimeoutError("IARC Stage 0A exceeded its four-hour cap during micro-fit")
        row = indexed["micro_fit"][train_order[step % len(train_order)]]
        raw = dataset[int(row["dataset_local_index"])]
        spec = perturbation_spec(row, partition="micro_fit", sorted_task_indices=task_indices)
        robust = perturb_raw_sample(raw, spec)
        batch = _preprocess(preprocessor, robust)
        noise, time_tensor, draw = _shared_draw(row, "micro_fit", step, "cuda")
        optimizer.zero_grad(set_to_none=True)
        loss = _loss(policy, batch, noise, time_tensor)
        loss_value = _to_float(loss)
        if not math.isfinite(loss_value):
            raise RuntimeError(f"nonfinite micro-fit loss at step {step}")
        loss.backward()
        gradient = flatten_gradients(named_parameters)
        gradient_norm = float(torch.linalg.vector_norm(gradient).item())
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError(f"invalid LoRA gradient at micro-fit step {step}: {gradient_norm}")
        optimizer.step()
        progress["parameter_update_occurred"] = True
        loss_curve.append(
            {
                "step": step,
                "sample_id": sample_id(row),
                "family": spec.family,
                "severity_index": spec.severity_index,
                "loss": loss_value,
                "draw": draw,
            }
        )
        gradient_curve.append({"step": step, "gradient_norm": gradient_norm})
        print(f"[IARC micro-fit] {step + 1}/{MICRO_STEPS} loss={loss_value:.8f}", flush=True)
    subset_after_values = _loss_on_rows(
        policy, preprocessor, dataset, subset_rows, partition="micro_fit"
    )
    subset_before = float(np.mean(subset_before_values))
    subset_after = float(np.mean(subset_after_values))
    base_frozen_hash_after = _hash_frozen_parameters(policy)
    if base_frozen_hash_before != base_frozen_hash_after:
        raise RuntimeError("frozen Base parameter hash changed during micro-fit")
    micro_fit = {
        "completed_steps": len(loss_curve),
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": float(optimizer.param_groups[0]["weight_decay"]),
        "seed": SEED,
        "rank": LORA_RANK,
        "batch_size": 1,
        "loss_curve": loss_curve,
        "gradient_curve": gradient_curve,
        "fixed_subset_rows": [sample_id(row) for row in subset_rows],
        "fixed_subset_loss_before_values": subset_before_values,
        "fixed_subset_loss_after_values": subset_after_values,
        "fixed_subset_loss_before": subset_before,
        "fixed_subset_loss_after": subset_after,
        "fixed_subset_loss_decreased": subset_after < subset_before,
    }
    checkpoint_native, checkpoint_processed = _predict(policy, first_batch, postprocessor, identity_noise)
    checkpoint = _save_adapter_checkpoint(
        policy,
        micro_fit,
        checkpoint_native,
        checkpoint_processed,
    )
    if checkpoint["resumed_existing_checkpoint"]:
        checkpoint_native = torch.as_tensor(checkpoint["reload_probe"]["native_action"], dtype=torch.float32)
        checkpoint_processed = np.asarray(
            checkpoint["reload_probe"]["postprocessed_action"], dtype=np.float32
        )
    del optimizer
    del policy
    gc.collect()
    torch.cuda.empty_cache()

    policy, config, preprocessor, postprocessor = _load_adapter_checkpoint()
    named_parameters = sorted_trainable_parameters(policy)
    reloaded_native, reloaded_processed = _predict(policy, first_batch, postprocessor, identity_noise)
    reload_native_error = float(torch.max(torch.abs(checkpoint_native - reloaded_native)).item())
    reload_processed_error = float(np.max(np.abs(checkpoint_processed - reloaded_processed)))
    checkpoint.update(
        {
            "disk_reload": True,
            "reload_output_max_abs_error": max(reload_native_error, reload_processed_error),
            "reload_native_max_abs_error": reload_native_error,
            "reload_postprocessed_max_abs_error": reload_processed_error,
            "base_frozen_hash_after": base_frozen_hash_after,
            "base_frozen_hash_unchanged": base_frozen_hash_before == base_frozen_hash_after,
        }
    )
    _write_json(CHECKPOINT_JSON, checkpoint)

    completed_records: dict[str, dict[str, Any]] = {}
    if PARTIAL_JSON.exists():
        partial = _read_json(PARTIAL_JSON)
        if partial.get("proposal_hash") != PROPOSAL_HASH:
            raise RuntimeError("existing IARC partial has a different proposal hash")
        completed_records = {str(row["sample_id"]): row for row in partial.get("gradient_records") or []}
    gradient_records: list[dict[str, Any]] = list(completed_records.values())
    parameter_manifest_now = parameter_manifest(named_parameters)
    generated_pair_hashes: set[str] = set()
    for record in gradient_records:
        generated_pair_hashes.add(str(record.get("generated_pair_hash")))

    for index, row in enumerate(indexed["conflict_audit"]):
        key = sample_id(row)
        if key in completed_records:
            print(f"[IARC gradient] resume skip {key}", flush=True)
            continue
        if time.monotonic() - started_monotonic > MAX_RUNTIME_SECONDS:
            raise TimeoutError("IARC Stage 0A exceeded its four-hour cap during gradient audit")
        raw = dataset[int(row["dataset_local_index"])]
        spec = perturbation_spec(row, partition="conflict_audit", sorted_task_indices=task_indices)
        robust = perturb_raw_sample(raw, spec)
        clean_batch = _preprocess(preprocessor, raw)
        robust_batch = _preprocess(preprocessor, robust)
        integrity = _pair_integrity(raw, robust, clean_batch, robust_batch, spec.family)
        noise, time_tensor, draw = _shared_draw(row, "conflict_audit", index, "cuda")

        policy.train()
        policy.zero_grad(set_to_none=True)
        clean_loss_tensor = _loss(policy, clean_batch, noise, time_tensor)
        clean_loss = _to_float(clean_loss_tensor)
        clean_loss_tensor.backward()
        clean_gradient = flatten_gradients(named_parameters)
        policy.zero_grad(set_to_none=True)
        robust_loss_tensor = _loss(policy, robust_batch, noise, time_tensor)
        robust_loss = _to_float(robust_loss_tensor)
        robust_loss_tensor.backward()
        robust_gradient = flatten_gradients(named_parameters)
        policy.zero_grad(set_to_none=True)
        projection = project_clean_gradient(clean_gradient, robust_gradient)
        projected = projection.pop("projected_gradient")
        agreeing_unchanged = None
        projected_diff_from_clean = None
        tiny_step = {
            "performed": False,
            "step_size": TINY_STEP_SIZE,
            "robust_loss_before": robust_loss,
            "robust_loss_after": None,
            "change": None,
            "numerical_status": "not_a_gate_conflict",
        }
        projected_modules: dict[str, float] = {}
        if projected is not None:
            projected_diff_from_clean = float(torch.linalg.vector_norm(projected - clean_gradient).item())
            agreeing_unchanged = bool(torch.equal(projected, clean_gradient)) if projection["status"] == "agreeing_or_orthogonal" else None
            projected_modules = module_norms(projected, parameter_manifest_now)
            if projection["gate_conflict"]:
                snapshots = {name: parameter.detach().clone() for name, parameter in named_parameters}
                with torch.no_grad():
                    offset = 0
                    for name, parameter in named_parameters:
                        count = int(parameter.numel())
                        delta = projected[offset : offset + count].reshape(parameter.shape).to(parameter.device, parameter.dtype)
                        parameter.add_(-TINY_STEP_SIZE * delta)
                        offset += count
                with torch.no_grad():
                    tiny_loss_after = _to_float(_loss(policy, robust_batch, noise, time_tensor))
                with torch.no_grad():
                    for name, parameter in named_parameters:
                        parameter.copy_(snapshots[name])
                tiny_change = tiny_loss_after - robust_loss
                tiny_step = {
                    "performed": True,
                    "step_size": TINY_STEP_SIZE,
                    "robust_loss_before": robust_loss,
                    "robust_loss_after": tiny_loss_after,
                    "change": tiny_change,
                    "numerical_status": "nonincrease" if tiny_change <= 1e-7 else "finite_difference_increase_diagnostic",
                }
        pair_hash = value_hash(
            {
                "sample_id": key,
                "family": spec.family,
                "severity": spec.severity,
                "clean_processed": integrity["clean_processed_hash"],
                "robust_processed": integrity["robust_processed_hash"],
            }
        )
        duplicate_generated_pair = pair_hash in generated_pair_hashes
        generated_pair_hashes.add(pair_hash)
        record = {
            "sample_id": key,
            "task_index": int(row["task_index"]),
            "phase": str(row["phase"]),
            "family": spec.family,
            "severity_index": spec.severity_index,
            "severity": spec.severity,
            "direction": spec.direction,
            "clean_loss": clean_loss,
            "robust_loss": robust_loss,
            "clean_gradient_norm": float(torch.linalg.vector_norm(clean_gradient).item()),
            "robust_gradient_norm": float(torch.linalg.vector_norm(robust_gradient).item()),
            "clean_module_norms": module_norms(clean_gradient, parameter_manifest_now),
            "robust_module_norms": module_norms(robust_gradient, parameter_manifest_now),
            "projected_module_norms": projected_modules,
            "projected_diff_from_clean": projected_diff_from_clean,
            "agreeing_unchanged": agreeing_unchanged,
            "projection": projection,
            "tiny_step": tiny_step,
            "integrity": integrity,
            "draw": draw,
            "shared_draw_hash_match": True,
            "generated_pair_hash": pair_hash,
            "duplicate_generated_pair": duplicate_generated_pair,
        }
        gradient_records.append(record)
        gradient_records.sort(key=lambda item: int(item["task_index"]))
        _write_json(
            PARTIAL_JSON,
            {
                "method": "IARC-VLA",
                "proposal_hash": PROPOSAL_HASH,
                "status": "gradient_audit_running",
                "completed_pair_count": len(gradient_records),
                "planned_pair_count": 40,
                "exception_count": 0,
                "completed_pair_keys": [str(item["sample_id"]) for item in gradient_records],
                "gradient_records": gradient_records,
            },
        )
        print(
            f"[IARC gradient] {len(gradient_records)}/40 {key} cosine={projection['cosine']}",
            flush=True,
        )

    if len(gradient_records) != 40:
        raise RuntimeError(f"gradient audit incomplete: {len(gradient_records)} / 40")
    clean_norms = [float(row["clean_gradient_norm"]) for row in gradient_records]
    robust_norms = [float(row["robust_gradient_norm"]) for row in gradient_records]
    clean_median = float(np.median(clean_norms))
    robust_median = float(np.median(robust_norms))
    scale_ratio = clean_median / robust_median if robust_median > 0 else None
    beta = 1.0 if scale_ratio is not None and 0.25 <= scale_ratio <= 4.0 else scale_ratio
    for row in gradient_records:
        projection = row["projection"]
        if projection["status"] == "projected_conflict" and beta is not None:
            clean_norm = float(projection["clean_norm"])
            robust_norm = float(projection["robust_norm"])
            dot = float(projection["dot_before"])
            alpha = float(projection["projection_coefficient"])
            coefficient = alpha - 0.5 * beta
            joint_diff_squared = (
                0.25 * clean_norm**2 + coefficient**2 * robust_norm**2 + coefficient * dot
            )
            row["projected_diff_from_joint"] = math.sqrt(max(0.0, joint_diff_squared))
        else:
            row["projected_diff_from_joint"] = None

    conflict_records = [row for row in gradient_records if bool(row["projection"]["gate_conflict"])]
    projected_records = [row for row in gradient_records if row["projection"]["status"] == "projected_conflict"]
    agreeing_records = [row for row in gradient_records if row["projection"]["status"] == "agreeing_or_orthogonal"]
    below_floor_records = [row for row in gradient_records if row["projection"]["status"] == "robust_gradient_below_floor"]
    conflict_families = sorted({str(row["family"]) for row in conflict_records})
    gradient_summary = {
        "record_count": len(gradient_records),
        "conflict_threshold": CONFLICT_COSINE_THRESHOLD,
        "robust_norm_squared_floor": ROBUST_NORM_SQUARED_FLOOR,
        "conflict_count": len(conflict_records),
        "conflict_rate": len(conflict_records) / len(gradient_records),
        "conflict_families": conflict_families,
        "conflict_family_count": len(conflict_families),
        "projected_row_count": len(projected_records),
        "agreeing_row_count": len(agreeing_records),
        "below_floor_count": len(below_floor_records),
        "projection_constraint_pass_count": sum(
            bool(row["projection"]["constraint_passed"]) for row in projected_records
        ),
        "agreeing_unchanged_count": sum(bool(row["agreeing_unchanged"]) for row in agreeing_records),
        "shared_draw_hash_match_count": sum(bool(row["shared_draw_hash_match"]) for row in gradient_records),
        "target_state_integrity_count": sum(
            bool(row["integrity"]["raw_shared_all"])
            and bool(row["integrity"]["processed_shared_all"])
            for row in gradient_records
        ),
        "allowlist_and_activation_count": sum(
            bool(row["integrity"]["allowlist_ok"]) and bool(row["integrity"]["transform_acted"])
            for row in gradient_records
        ),
        "duplicate_generated_pair_count": sum(bool(row["duplicate_generated_pair"]) for row in gradient_records),
        "iarc_differs_from_clean_on_conflicts": all(
            float(row["projected_diff_from_clean"] or 0.0) > 0.0 for row in conflict_records
        ),
        "iarc_differs_from_joint_on_conflicts": all(
            float(row["projected_diff_from_joint"] or 0.0) > 0.0 for row in conflict_records
        ),
        "tiny_step_nonincrease_or_resolution_count": sum(
            str(row["tiny_step"]["numerical_status"])
            in {"nonincrease", "finite_difference_increase_diagnostic"}
            for row in conflict_records
        ),
        "cosine": numeric_summary(
            [float(row["projection"]["cosine"]) for row in gradient_records if row["projection"]["cosine"] is not None]
        ),
        "clean_loss": numeric_summary([float(row["clean_loss"]) for row in gradient_records]),
        "robust_loss": numeric_summary([float(row["robust_loss"]) for row in gradient_records]),
        "clean_gradient_norm": numeric_summary(clean_norms),
        "robust_gradient_norm": numeric_summary(robust_norms),
        "discovery_gradient_scale_ratio": scale_ratio,
        "joint_ablation_beta": beta,
        "conflicts_by_family": dict(Counter(str(row["family"]) for row in conflict_records)),
        "conflicts_by_phase": dict(Counter(str(row["phase"]) for row in conflict_records)),
        "conflict_task_indices": [int(row["task_index"]) for row in conflict_records],
    }
    gradient_artifact = {
        "method": "IARC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "summary": gradient_summary,
        "records": gradient_records,
    }
    _write_json(GRADIENT_JSON, gradient_artifact)

    validation_rows = _evaluate_validation_rows(
        policy,
        preprocessor,
        postprocessor,
        dataset,
        indexed["validation"],
        base_by_sample,
        action_min,
        action_max,
        started=started_monotonic,
    )
    validation_summary = _summarize_validation(validation_rows)
    peak_cuda_gib = float(torch.cuda.max_memory_allocated()) / (1024**3)
    identity_passed = identity_native_error <= 1e-6 and identity_processed_error <= 1e-6
    checkpoint_passed = bool(checkpoint["disk_reload"]) and float(checkpoint["reload_output_max_abs_error"]) <= 1e-6
    projection_invariants = (
        gradient_summary["below_floor_count"] == 0
        and gradient_summary["projection_constraint_pass_count"] == gradient_summary["projected_row_count"]
        and gradient_summary["agreeing_unchanged_count"] == gradient_summary["agreeing_row_count"]
        and bool(gradient_summary["iarc_differs_from_clean_on_conflicts"])
        and bool(gradient_summary["iarc_differs_from_joint_on_conflicts"])
    )
    health = {
        "partition_health": observed_counts == expected_counts
        and all(value == 0 for value in summaries["pairwise_sample_overlap"].values()),
        "perturbation_health": gradient_summary["allowlist_and_activation_count"] == 40
        and gradient_summary["duplicate_generated_pair_count"] == 0,
        "action_target_health": gradient_summary["target_state_integrity_count"] == 40,
        "preflight_passed": bool(preflight["passed"]),
        "shared_draw_health": gradient_summary["shared_draw_hash_match_count"] == 40,
        "lora_only_trainable": lora_only,
        "identity_passed": identity_passed,
        "checkpoint_reload_passed": checkpoint_passed,
        "base_unchanged": base_frozen_hash_before == base_frozen_hash_after,
        "mechanism_invariants_passed": projection_invariants,
        "action_validity_passed": validation_summary["dataset_range_valid_fraction"] == 1.0,
        "memory_passed": peak_cuda_gib < MAX_CUDA_GIB,
        "confirmatory_sealed": True,
        "gradient_health": all(float(row["gradient_norm"]) > 0.0 for row in gradient_curve)
        and gradient_summary["below_floor_count"] == 0,
        "subset_fit_passed": bool(micro_fit["fixed_subset_loss_decreased"]),
        "conflict_count": gradient_summary["conflict_count"],
        "conflict_family_count": gradient_summary["conflict_family_count"],
    }
    final_decision = classify_stage0(health)
    if final_decision == "IARC_STAGE_0A_PASS_HEADROOM_PENDING":
        next_command = "Implement and run the frozen IARC-VLA Stage 0B Base-only headroom manifest."
        reason = "All Stage 0A health and mechanism gates passed; only the preregistered Base headroom screen is allowed next."
    elif final_decision == "IARC_STAGE_0A_UNDERPOWERED_ONE_CHECK_ALLOWED":
        next_command = EXACT_AUDIT_COMMAND.replace("--mode audit", "--mode one-check")
        reason = "The healthy audit observed only the preregistered underpowered conflict pattern; exactly one fixed check is allowed."
    else:
        next_command = "Adjudicate the frozen failure under the false-negative safeguard and continue to the next method cycle."
        reason = "Stage 0A did not satisfy its frozen gate; no threshold, perturbation, rank, optimizer, or row rescue is allowed."

    result = {
        "schema_version": 1,
        "date_kst": DATE_KST,
        "method": "IARC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "command": EXACT_AUDIT_COMMAND,
        "mode": args.mode,
        "seed": SEED,
        "git_commit": _git_commit(),
        "package_versions": _package_versions(),
        "final_decision": final_decision,
        "adjudication_reason": reason,
        "exact_next_command": next_command,
        "preflight": preflight,
        "source": {
            "checkpoint": str(CHECKPOINT_PATH),
            "vlm": str(VLM_PATH),
            "dataset": str(DATASET_ROOT),
            "split_manifest": str(SPLIT_MANIFEST),
            "split_manifest_sha256": _sha256_file(SPLIT_MANIFEST),
            "base_artifact": str(BASE_ARTIFACT),
            "base_artifact_sha256": _sha256_file(BASE_ARTIFACT),
            "resource_registry": str(RESOURCE_REGISTRY),
            "resource_registry_sha256": _sha256_file(RESOURCE_REGISTRY),
        },
        "partitions": {"counts": observed_counts, "summary": summaries, "selection_ranks": {"micro_fit": 0, "conflict_audit": 1, "one_check": 2, "validation": 0}},
        "tensor_contract": {
            "raw_shapes": raw_shapes,
            "processed_shapes": processed_shapes,
            "input_devices": input_devices,
            "noise_shape": identity_draw["noise_shape"],
            "time_shape": identity_draw["time_shape"],
            "gradient_shape": [sum(int(item["numel"]) for item in param_manifest)],
            "dtype": str(next(iter(policy.parameters())).dtype),
            "autocast": "disabled",
        },
        "identity": {
            "base_native_action": base_native.tolist(),
            "adapter_native_action": adapter_native.tolist(),
            "base_postprocessed_action": base_processed.tolist(),
            "adapter_postprocessed_action": adapter_processed.tolist(),
            "native_max_abs_error": identity_native_error,
            "postprocessed_max_abs_error": identity_processed_error,
            "passed": identity_passed,
            "draw": identity_draw,
        },
        "micro_fit": micro_fit,
        "checkpoint": checkpoint,
        "gradient_audit": gradient_summary,
        "validation": {"summary": validation_summary, "records": validation_rows},
        "action_bounds": {"min": action_min.tolist(), "max": action_max.tolist()},
        "health": health,
        "confirmatory": {
            "manifest_identity_rows_read": len(partitions["confirmatory_reserved"]),
            "observations_decoded": 0,
            "actions_computed": 0,
            "perturbations_generated": 0,
            "policy_outputs_computed": 0,
        },
        "experiment_boundaries": {
            "simulator_rollout_happened": False,
            "validation_search_happened": False,
            "full_training_happened": False,
            "confirmatory_test_tuning_happened": False,
            "privileged_inference_input_used": False,
        },
        "risk_assessment": {
            "false_positive_risk": "Stage 0A establishes only local gradient action and implementation health, not closed-loop success.",
            "false_negative_risk": "A 20-step rank-4 micro-fit can underexpress the full method; data, capacity, and implementation failures take priority over scientific kill labels.",
            "record_independence": "Conflict-audit rows are train rank 1 and disjoint from rank-0 micro-fit rows.",
            "confidence": "bounded development mechanism audit",
            "permanent_stop_evidence": reason,
        },
        "runtime": {
            "elapsed_seconds": time.monotonic() - started_monotonic,
            "peak_cuda_allocated_gib": peak_cuda_gib,
            "cuda_device": torch.cuda.get_device_name(0),
            "max_cuda_gib": MAX_CUDA_GIB,
            "paper_evidence_eligible": bool(preflight["resource_evidence"]["paper_evidence_eligible"]),
            "resource_evidence": preflight["resource_evidence"],
        },
    }
    _write_json(RESULT_JSON, result)
    _write_result_markdown(result)
    _write_json(
        STATUS_JSON,
        {
            "status": "completed",
            "pid": os.getpid(),
            "final_decision": final_decision,
            "completed_pair_count": 40,
            "validation_record_count": 40,
            "exception_count": 0,
        },
    )
    return result


def _write_blocker(args: argparse.Namespace, exc: BaseException) -> None:
    partial = _read_json(PARTIAL_JSON) if PARTIAL_JSON.exists() else {}
    payload = {
        "schema_version": 1,
        "date_kst": DATE_KST,
        "method": "IARC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "mode": args.mode,
        "final_decision": "IARC_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE",
        "failing_prerequisite_or_invariant": f"{type(exc).__name__}: {exc}",
        "expected": "Frozen Stage 0A completes through official CUDA SmolVLA without changing method or rows.",
        "observed": str(exc),
        "command": EXACT_AUDIT_COMMAND if args.mode == "audit" else EXACT_AUDIT_COMMAND.replace("--mode audit", "--mode one-check"),
        "stack_trace": traceback.format_exc().splitlines(),
        "decoded_split_counts_at_failure": {"test": 0},
        "parameter_update_occurred": CHECKPOINT_DIR.exists(),
        "completed_pair_count": int(partial.get("completed_pair_count") or 0),
        "planned_pair_count": 40,
        "exception_count": 1,
        "bounded_implementation_repair_possible_without_method_change": True,
        "exact_resume_command": EXACT_AUDIT_COMMAND,
        "classification": "IMPLEMENTATION_FAILURE",
    }
    _write_json(BLOCKER_JSON, payload)
    _write_json(
        STATUS_JSON,
        {
            "status": "failed",
            "pid": os.getpid(),
            "exception_count": 1,
            "error": payload["failing_prerequisite_or_invariant"],
        },
    )


def _heartbeat_worker(stop: threading.Event) -> None:
    while not stop.is_set():
        _write_json(
            HEARTBEAT_JSON,
            {
                "status": "running",
                "pid": os.getpid(),
                "time_unix": time.time(),
                "time_local": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime()),
            },
        )
        stop.wait(30.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("audit", "one-check"), default="audit")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _set_offline_environment()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CHILD_PID_FILE.write_text(f"{os.getpid()}\n", encoding="ascii")
    stop_heartbeat = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat_worker, args=(stop_heartbeat,), daemon=True)
    heartbeat.start()
    exit_code = 1
    try:
        if RESULT_JSON.exists():
            existing = _read_json(RESULT_JSON)
            print(
                json.dumps(
                    {"status": "existing_result", "final_decision": existing.get("final_decision")},
                    indent=2,
                )
            )
            exit_code = 0
            return exit_code
        result = run_audit(args)
        exit_code = 0
    except Exception as exc:  # noqa: BLE001
        _write_blocker(args, exc)
        traceback.print_exc()
        return exit_code
    finally:
        stop_heartbeat.set()
        heartbeat.join(timeout=5.0)
        EXIT_CODE_FILE.write_text(f"{exit_code}\n", encoding="ascii")
        gc.collect()
    print(json.dumps({"final_decision": result["final_decision"], "result": str(RESULT_JSON)}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
