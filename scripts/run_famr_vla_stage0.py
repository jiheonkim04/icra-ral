"""Run the frozen FAMR-VLA Stage 0A provenance and capacity audit."""

from __future__ import annotations

import argparse
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

from tca_map.smolvla.famr_vla import (  # noqa: E402
    PROPOSAL_HASH,
    TARGET_TASK_IDENTITIES,
    action_scales,
    assign_parameter_groups,
    classify_stage0,
    episode_partitions,
    scale_lora_b,
    task_identity_audit,
    validate_episode_partitions,
    validate_partial_payload,
)
from tca_map.smolvla.official_libero_baseline_scaleup import (  # noqa: E402
    _add_training_batch_dims,
    _loss_from_output,
    _postprocess_action,
    _tensor_devices,
    _tensor_shapes,
    _to_float,
)


DATE_KST = "2026-07-15"
SEED = 1701
MICRO_STEPS = 20
GRADIENT_ACCUMULATION = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 0.0
LORA_RANK = 4
LORA_ALPHA = 8
LORA_DROPOUT = 0.0
MAX_RUNTIME_SECONDS = 4 * 60 * 60
MAX_CUDA_GIB = 15.5
CHUNK_SIZE = 50
MAX_ACTION_DIM = 32
EXPECTED_STAGE = "epoch_4_cycle_17_famr_stage_0a_implementation_pending"
EXPECTED_TASK_COUNT = 40
FIXED_ROWS_PER_TASK = 8
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


HF_HOME = _asset_path("hf_home")
VLM_PATH = HF_HOME / "HuggingFaceTB" / "SmolVLM2-500M-Video-Instruct"
OFFICIAL_DATASET_ROOT = _asset_path("datasets", "lerobot_libero")
LIBERO_REPO = _asset_path("repos", "LIBERO")
STATE_FILES = (
    REPO_ROOT / "reports" / "autonomous_until_paper_state.json",
    REPO_ROOT / "reports" / "autonomous_ral_campaign_state.json",
)
RESOURCE_REGISTRY = REPO_ROOT / "reports" / "resource_contention_intervals.json"
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "famr_vla" / "proposal_hash.txt"
DEFAULT_REPORT_DIR = REPO_ROOT / "reports" / "famr_vla"
DEFAULT_RUN_DIR = REPO_ROOT / "runs" / "famr_vla" / "stage0a"

TARGET_FILES = {
    TARGET_TASK_IDENTITIES[0]: "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf_demo.hdf5",
    TARGET_TASK_IDENTITIES[1]: "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray_demo.hdf5",
    TARGET_TASK_IDENTITIES[2]: "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo.hdf5",
}


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


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
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
    versions = {}
    for name in ("torch", "lerobot", "transformers", "peft", "numpy", "h5py", "pyarrow"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "NOT_INSTALLED"
    return versions


def _value_hash(value: Any) -> str:
    digest = hashlib.sha256()
    if hasattr(value, "detach"):
        array = value.detach().cpu().contiguous().numpy()
    else:
        array = np.ascontiguousarray(value)
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(str(tuple(array.shape)).encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest().upper()


def _stable_seed(*parts: Any) -> int:
    payload = "|".join([PROPOSAL_HASH, *(str(part) for part in parts)]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


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


def _resource_evidence(registry: Mapping[str, Any], started_unix: float) -> dict[str, Any]:
    from datetime import datetime

    intervals = list(registry.get("intervals") or [])
    overlap_ids = []
    unresolved_ids = []
    for interval in intervals:
        interval_id = str(interval.get("id"))
        if not bool(interval.get("efficiency_mode_disabled")):
            unresolved_ids.append(interval_id)
            continue
        end_text = interval.get("end_time_latest_kst")
        if not end_text:
            unresolved_ids.append(interval_id)
            continue
        try:
            end_unix = datetime.fromisoformat(str(end_text)).timestamp()
        except ValueError:
            unresolved_ids.append(interval_id)
            continue
        if end_unix >= started_unix:
            overlap_ids.append(interval_id)
    eligible = not overlap_ids and not unresolved_ids
    return {
        "registry_interval_count": len(intervals),
        "overlap_interval_ids": overlap_ids,
        "unresolved_interval_ids": unresolved_ids,
        "timing_throughput_resource_evidence_eligible": eligible,
        "closed_loop_success_row_policy": "not_applicable_stage_0a_no_rollout",
        "policy": "timing, throughput, wall-clock efficiency, and resource utilization are excluded on unknown or positive overlap",
    }


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    report_dir = Path(args.report_root)
    run_dir = Path(args.run_root)
    if not report_dir.is_absolute():
        report_dir = REPO_ROOT / report_dir
    if not run_dir.is_absolute():
        run_dir = REPO_ROOT / run_dir
    checkpoint = Path(args.checkpoint)
    data_root = Path(args.libero_data_root)
    stable_artifact = Path(args.stable_artifact)
    if not stable_artifact.is_absolute():
        stable_artifact = REPO_ROOT / stable_artifact
    return {
        "report_dir": report_dir,
        "run_dir": run_dir,
        "checkpoint": checkpoint,
        "data_root": data_root,
        "stable_artifact": stable_artifact,
        "checkpoint_dir": run_dir / "micro_checkpoint",
        "resume_state": run_dir / "micro_resume.pt",
        "partial": run_dir / "partial_result.json",
        "status": run_dir / "status.json",
        "heartbeat": run_dir / "heartbeat.json",
        "child_pid": run_dir / "child_pid.txt",
        "worker_pid": run_dir / "worker_pid.txt",
        "exit_code": run_dir / "exit_code.txt",
        "resume_command": run_dir / "exact_resume_command.txt",
        "result_json": report_dir / "stage_0a_result.json",
        "result_md": report_dir / "stage_0a_result.md",
        "adjudication": report_dir / "stage_0a_adjudication.md",
        "provenance": report_dir / "task_provenance_manifest.json",
        "data_audit": report_dir / "data_semantics_audit.json",
        "parameter_manifest": report_dir / "parameter_group_manifest.json",
        "checkpoint_manifest": report_dir / "checkpoint_manifest.json",
        "blocker": report_dir / "implementation_blocker.json",
    }


def _preflight(args: argparse.Namespace, paths: Mapping[str, Path], started_unix: float) -> dict[str, Any]:
    import torch

    states = {str(path.relative_to(REPO_ROOT)): _read_json(path) for path in STATE_FILES if path.is_file()}
    registry = _read_json(RESOURCE_REGISTRY)
    target_paths = {
        task: paths["data_root"] / "libero_90" / filename for task, filename in TARGET_FILES.items()
    }
    bddl_paths = {
        task: LIBERO_REPO / "libero" / "libero" / "bddl_files" / "libero_90" / filename.replace("_demo.hdf5", ".bddl")
        for task, filename in TARGET_FILES.items()
    }
    required = {
        "checkpoint": paths["checkpoint"],
        "vlm": VLM_PATH,
        "official_dataset": OFFICIAL_DATASET_ROOT,
        "official_dataset_info": OFFICIAL_DATASET_ROOT / "meta" / "info.json",
        "official_dataset_tasks": OFFICIAL_DATASET_ROOT / "meta" / "tasks.parquet",
        "stable_artifact": paths["stable_artifact"],
        "resource_registry": RESOURCE_REGISTRY,
        "proposal_hash": PROPOSAL_HASH_FILE,
        **{f"target_{index}": path for index, path in enumerate(target_paths.values())},
        **{f"bddl_{index}": path for index, path in enumerate(bddl_paths.values())},
    }
    missing = [name for name, path in required.items() if not path.exists()]
    observed_stages = {name: value.get("current_stage") for name, value in states.items()}
    stage_ok = len(states) == len(STATE_FILES) and all(stage == EXPECTED_STAGE for stage in observed_stages.values())
    proposal_observed = PROPOSAL_HASH_FILE.read_text(encoding="utf-8").strip() if PROPOSAL_HASH_FILE.is_file() else None
    workers = _active_linux_workers()
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name) == "1"]
    result_absent = not paths["result_json"].exists()
    partial_audit: dict[str, Any] | None = None
    partial_parse_error: str | None = None
    if paths["partial"].is_file():
        try:
            partial_audit = validate_partial_payload(paths["partial"].read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            partial_parse_error = str(exc)
    status_payload = _read_json(paths["status"]) if paths["status"].is_file() else None
    inconsistent_completed_status = bool(
        status_payload and status_payload.get("status") == "completed" and result_absent
    )
    mode_ok = args.mode == "audit"
    cuda_available = bool(torch.cuda.is_available())
    disk = shutil.disk_usage(paths["run_dir"].parent if paths["run_dir"].parent.exists() else REPO_ROOT)
    disk_free_gib = disk.free / 1024**3
    passed = bool(
        not missing
        and stage_ok
        and proposal_observed == PROPOSAL_HASH
        and not workers
        and not forbidden
        and result_absent
        and partial_parse_error is None
        and not inconsistent_completed_status
        and mode_ok
        and cuda_available
        and disk_free_gib >= 5.0
    )
    return {
        "passed": passed,
        "mode": args.mode,
        "expected_stage": EXPECTED_STAGE,
        "observed_stages": observed_stages,
        "stage_ok": stage_ok,
        "proposal_hash_expected": PROPOSAL_HASH,
        "proposal_hash_observed": proposal_observed,
        "proposal_hash_ok": proposal_observed == PROPOSAL_HASH,
        "active_linux_workers": workers,
        "forbidden_gates_enabled": forbidden,
        "result_absent": result_absent,
        "partial_audit": partial_audit,
        "partial_parse_error": partial_parse_error,
        "status_payload": status_payload,
        "inconsistent_completed_status": inconsistent_completed_status,
        "missing_paths": missing,
        "paths": {name: str(path) for name, path in required.items()},
        "target_paths": {task: str(path) for task, path in target_paths.items()},
        "bddl_paths": {task: str(path) for task, path in bddl_paths.items()},
        "cuda_available": cuda_available,
        "cuda_device": torch.cuda.get_device_name(0) if cuda_available else None,
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if cuda_available else None,
        "disk_free_gib": disk_free_gib,
        "package_versions": _package_versions(),
        "resource_evidence": _resource_evidence(registry, started_unix),
    }


def _load_official_tasks() -> list[str]:
    import pyarrow.parquet as parquet

    table = parquet.read_table(OFFICIAL_DATASET_ROOT / "meta" / "tasks.parquet")
    rows = table.to_pylist()
    tasks = [str(row["__index_level_0__"]) for row in sorted(rows, key=lambda row: int(row["task_index"]))]
    if len(tasks) != EXPECTED_TASK_COUNT:
        raise RuntimeError(f"official checkpoint dataset has {len(tasks)} tasks, expected {EXPECTED_TASK_COUNT}")
    return tasks


def _demo_sort_key(value: str) -> int:
    return int(value.rsplit("_", 1)[1])


def _identity_hash(task: str, episode: int, frame: int | None = None) -> str:
    suffix = f"|{frame}" if frame is not None else ""
    return hashlib.sha256(f"{task}|{episode}{suffix}".encode("utf-8")).hexdigest().upper()


def _audit_target_sources(paths: Mapping[str, Path], preflight: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import h5py

    partitions = episode_partitions()
    task_rows = []
    fixed_rows = []
    all_episode_hashes = []
    all_frame_hashes = []
    for task_index, task in enumerate(TARGET_TASK_IDENTITIES):
        path = Path(preflight["target_paths"][task])
        bddl = Path(preflight["bddl_paths"][task])
        with h5py.File(path, "r") as handle:
            data = handle["data"]
            demo_names = sorted(data.keys(), key=_demo_sort_key)
            available = [_demo_sort_key(name) for name in demo_names]
            split_audit = validate_episode_partitions(partitions, available)
            problem_info = json.loads(str(data.attrs["problem_info"]))
            env_args = json.loads(str(data.attrs["env_args"]))
            language = str(problem_info["language_instruction"])
            source_success = []
            lengths = []
            for name in demo_names:
                demo = data[name]
                lengths.append(int(demo["actions"].shape[0]))
                source_success.append(bool(float(demo["dones"][-1]) > 0.0 and float(demo["rewards"][-1]) > 0.0))
            training_actions = np.concatenate([np.asarray(data[f"demo_{episode}"]["actions"]) for episode in partitions["train"]])
            training_states = np.concatenate(
                [
                    np.concatenate(
                        [
                            np.asarray(data[f"demo_{episode}"]["obs"]["ee_states"]),
                            np.asarray(data[f"demo_{episode}"]["obs"]["gripper_states"]),
                        ],
                        axis=1,
                    )
                    for episode in partitions["train"]
                ]
            )
            state_concat_error = 0.0
            image_min = 255
            image_max = 0
            selected = []
            for episode in range(FIXED_ROWS_PER_TASK):
                demo = data[f"demo_{episode}"]
                observations = demo["obs"]
                state_concat_error = max(
                    state_concat_error,
                    float(
                        np.max(
                            np.abs(
                                np.asarray(observations["ee_states"])
                                - np.concatenate(
                                    [np.asarray(observations["ee_pos"]), np.asarray(observations["ee_ori"])], axis=1
                                )
                            )
                        )
                    ),
                )
                length = int(demo["actions"].shape[0])
                frame = max(0, (length - CHUNK_SIZE) // 2)
                for camera in ("agentview_rgb", "eye_in_hand_rgb"):
                    image = np.asarray(observations[camera][frame])
                    image_min = min(image_min, int(image.min()))
                    image_max = max(image_max, int(image.max()))
                row = {
                    "row_id": f"target{task_index}_episode{episode}_frame{frame}",
                    "task_index": task_index,
                    "task_identity": task,
                    "task_language": language,
                    "source_path": str(path),
                    "episode": episode,
                    "frame": frame,
                    "episode_length": length,
                }
                fixed_rows.append(row)
                selected.append(row["row_id"])

            episode_hashes = {
                split: [_identity_hash(task, episode) for episode in episodes] for split, episodes in partitions.items()
            }
            frame_hashes = {
                split: [
                    _identity_hash(task, episode, frame)
                    for episode in episodes
                    for frame in range(lengths[episode])
                ]
                for split, episodes in partitions.items()
            }
            all_episode_hashes.extend(value for values in episode_hashes.values() for value in values)
            all_frame_hashes.extend(value for values in frame_hashes.values() for value in values)
            action_iqr = np.percentile(training_actions, 75, axis=0) - np.percentile(training_actions, 25, axis=0)
            task_rows.append(
                {
                    "task_index": task_index,
                    "task_identity": task,
                    "task_language": language,
                    "source_path": str(path),
                    "source_sha256": _sha256_file(path),
                    "bddl_path": str(bddl),
                    "bddl_sha256": _sha256_file(bddl),
                    "demonstration_count": len(demo_names),
                    "frame_count": sum(lengths),
                    "length_min": min(lengths),
                    "length_max": max(lengths),
                    "source_terminal_success_count": sum(source_success),
                    "source_terminal_failure_count": len(source_success) - sum(source_success),
                    "split_audit": split_audit,
                    "camera_keys": ["agentview_rgb", "eye_in_hand_rgb"],
                    "camera_shape": [128, 128, 3],
                    "camera_dtype": "uint8",
                    "camera_source_convention": str(data.attrs.get("macros_image_convention")),
                    "camera_policy_transform": "official LiberoProcessorStep 180-degree H/W flip, then bicubic 256 resize",
                    "camera_discovery_value_range": [image_min, image_max],
                    "state_keys": ["ee_pos", "ee_ori", "ee_states", "gripper_states"],
                    "state_mapping": "concat(ee_states[6], gripper_states[2]) -> observation.state[8]",
                    "state_concat_max_abs_error": state_concat_error,
                    "state_finite_fraction": float(np.mean(np.isfinite(training_states))),
                    "action_shape": [7],
                    "action_finite_fraction": float(np.mean(np.isfinite(training_actions))),
                    "action_min": training_actions.min(axis=0).tolist(),
                    "action_max": training_actions.max(axis=0).tolist(),
                    "action_iqr": action_iqr.tolist(),
                    "action_scale": action_scales(training_actions).tolist(),
                    "gripper_values": sorted(np.unique(training_actions[:, 6]).tolist()),
                    "controller": env_args["env_kwargs"]["controller_configs"],
                    "fixed_discovery_rows": selected,
                    "episode_identity_hashes": episode_hashes,
                    "frame_identity_hash_counts": {split: len(values) for split, values in frame_hashes.items()},
                }
            )

    duplicate_episode_hashes = len(all_episode_hashes) - len(set(all_episode_hashes))
    duplicate_frame_hashes = len(all_frame_hashes) - len(set(all_frame_hashes))
    audit = {
        "method": "FAMR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "target_tasks": task_rows,
        "fixed_discovery_row_count": len(fixed_rows),
        "decoded_split_counts": {"train": len(fixed_rows), "validation": 0, "test": 0},
        "duplicate_episode_identity_hash_count": duplicate_episode_hashes,
        "duplicate_frame_identity_hash_count": duplicate_frame_hashes,
        "source_terminal_success_count": sum(row["source_terminal_success_count"] for row in task_rows),
        "source_terminal_failure_count": sum(row["source_terminal_failure_count"] for row in task_rows),
        "confirmatory_observations_decoded": 0,
        "confirmatory_actions_computed": 0,
    }
    return audit, fixed_rows


def _resize_rgb(image: Any) -> Any:
    import torch
    import torch.nn.functional as functional

    value = torch.as_tensor(np.asarray(image).copy(), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    value = functional.interpolate(value, size=(256, 256), mode="bicubic", align_corners=False, antialias=True)
    return value.clamp(0.0, 1.0)


def _apply_official_env_image_processor(agent: Any, wrist: Any) -> tuple[Any, Any]:
    from lerobot.processor.env_processor import LiberoProcessorStep

    processor = LiberoProcessorStep()
    processed = processor.observation(
        {
            "observation.images.image": _resize_rgb(agent),
            "observation.images.image2": _resize_rgb(wrist),
        }
    )
    return processed["observation.images.image"].squeeze(0), processed["observation.images.image2"].squeeze(0)


def _raw_sample(row: Mapping[str, Any]) -> dict[str, Any]:
    import h5py
    import torch

    with h5py.File(str(row["source_path"]), "r") as handle:
        demo = handle["data"][f"demo_{int(row['episode'])}"]
        observations = demo["obs"]
        frame = int(row["frame"])
        image, image2 = _apply_official_env_image_processor(
            observations["agentview_rgb"][frame], observations["eye_in_hand_rgb"][frame]
        )
        state = np.concatenate(
            [np.asarray(observations["ee_states"][frame]), np.asarray(observations["gripper_states"][frame])]
        )
        actions = np.asarray(demo["actions"])
        stop = min(frame + CHUNK_SIZE, len(actions))
        chunk = actions[frame:stop]
        padding = CHUNK_SIZE - len(chunk)
        if padding:
            chunk = np.concatenate([chunk, np.repeat(chunk[-1:], padding, axis=0)], axis=0)
        action_is_pad = np.zeros(CHUNK_SIZE, dtype=bool)
        if padding:
            action_is_pad[-padding:] = True
    return {
        "observation.images.image": image,
        "observation.images.image2": image2,
        "observation.state": torch.as_tensor(state, dtype=torch.float32),
        "action": torch.as_tensor(chunk, dtype=torch.float32),
        "timestamp": torch.tensor(float(row["frame"]) / 10.0, dtype=torch.float32),
        "frame_index": torch.tensor(int(row["frame"]), dtype=torch.int64),
        "episode_index": torch.tensor(int(row["episode"]), dtype=torch.int64),
        "index": torch.tensor(int(row["task_index"]) * 100000 + int(row["episode"]) * 1000 + int(row["frame"])),
        "task_index": torch.tensor(int(row["task_index"]), dtype=torch.int64),
        "action_is_pad": torch.as_tensor(action_is_pad),
        "task": str(row["task_language"]),
    }


def _official_mapping_calibration(paths: Mapping[str, Path]) -> dict[str, Any]:
    import h5py
    import torch
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    source = paths["data_root"] / "libero_10" / (
        "LIVING_ROOM_SCENE5_put_the_white_mug_on_the_left_plate_and_put_the_yellow_and_white_mug_on_the_right_plate_demo.hdf5"
    )
    dataset = LeRobotDataset("lerobot/libero", root=OFFICIAL_DATASET_ROOT, episodes=[0], video_backend="pyav")
    converted = dataset[0]
    converted_action = converted["action"].numpy()
    with h5py.File(source, "r") as handle:
        demos = handle["data"]
        errors = {
            name: float(np.max(np.abs(np.asarray(demo["actions"][0]) - converted_action)))
            for name, demo in demos.items()
        }
        matched = min(errors, key=errors.get)
        demo = demos[matched]
        raw_state = np.concatenate(
            [np.asarray(demo["obs"]["ee_states"][0]), np.asarray(demo["obs"]["gripper_states"][0])]
        )
        processed_agent, processed_wrist = _apply_official_env_image_processor(
            demo["obs"]["agentview_rgb"][0], demo["obs"]["eye_in_hand_rgb"][0]
        )
        plain_agent = _resize_rgb(demo["obs"]["agentview_rgb"][0]).squeeze(0)
        plain_wrist = _resize_rgb(demo["obs"]["eye_in_hand_rgb"][0]).squeeze(0)
    agent_rotated_mae = float(torch.mean(torch.abs(processed_agent - converted["observation.images.image"])).item())
    agent_plain_mae = float(torch.mean(torch.abs(plain_agent - converted["observation.images.image"])).item())
    wrist_rotated_mae = float(torch.mean(torch.abs(processed_wrist - converted["observation.images.image2"])).item())
    wrist_plain_mae = float(torch.mean(torch.abs(plain_wrist - converted["observation.images.image2"])).item())
    state_error = float(np.max(np.abs(raw_state - converted["observation.state"].numpy())))
    return {
        "official_dataset_episode": 0,
        "matched_raw_source": str(source),
        "matched_raw_demo": matched,
        "action_max_abs_error": errors[matched],
        "state_representation_max_abs_error": state_error,
        "agent_rotated_mae": agent_rotated_mae,
        "agent_unrotated_mae": agent_plain_mae,
        "wrist_rotated_mae": wrist_rotated_mae,
        "wrist_unrotated_mae": wrist_plain_mae,
        "action_mapping_passed": errors[matched] <= 1e-6,
        "state_mapping_passed": state_error <= 0.01,
        "image_orientation_passed": agent_rotated_mae < agent_plain_mae and wrist_rotated_mae < wrist_plain_mae,
        "note": "The state tolerance covers deterministic source-versus-official replay drift; both use pos/axis-angle/gripper semantics.",
    }


def _clone_batch(batch: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value.clone() if hasattr(value, "clone") else value for key, value in batch.items()}


def _preprocess(preprocessor: Any, raw_sample: Mapping[str, Any]) -> dict[str, Any]:
    return _add_training_batch_dims(preprocessor(dict(raw_sample)))


def _shared_draw(row_id: str, partition: str, logical_step: int, device: str) -> tuple[Any, Any, dict[str, Any]]:
    import torch

    seed = _stable_seed(partition, row_id, logical_step)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed % (2**63 - 1))
    noise_cpu = torch.randn((1, CHUNK_SIZE, MAX_ACTION_DIM), generator=generator, dtype=torch.float32)
    time_cpu = torch.rand((1,), generator=generator, dtype=torch.float32)
    return (
        noise_cpu.to(device),
        time_cpu.to(device),
        {
            "seed": seed,
            "noise_hash": _value_hash(noise_cpu),
            "time_hash": _value_hash(time_cpu),
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


def _load_policy_and_processors(checkpoint: Path) -> tuple[Any, Any, Any, Any]:
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True, cache_dir=HF_HOME)
    config.device = "cuda"
    config.load_vlm_weights = True
    config.compile_model = False
    config.push_to_hub = False
    config.vlm_model_name = str(VLM_PATH)
    if hasattr(config, "chunk_size"):
        config.chunk_size = CHUNK_SIZE
    policy = SmolVLAPolicy.from_pretrained(
        checkpoint,
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
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(VLM_PATH)},
            "device_processor": {"device": "cuda"},
        },
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return policy, config, preprocessor, postprocessor


def _load_adapter(paths: Mapping[str, Path]) -> tuple[Any, Any, Any, Any]:
    from peft import PeftConfig, PeftModel

    base, config, preprocessor, postprocessor = _load_policy_and_processors(paths["checkpoint"])
    peft_config = PeftConfig.from_pretrained(paths["checkpoint_dir"])
    policy = PeftModel.from_pretrained(
        base,
        paths["checkpoint_dir"],
        config=peft_config,
        is_trainable=True,
        local_files_only=True,
    )
    policy.to("cuda")
    policy.train()
    return policy, config, preprocessor, postprocessor


def _named_trainable(policy: Any) -> list[tuple[str, Any]]:
    return sorted(
        [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad],
        key=lambda item: item[0],
    )


def _hash_base_parameters(policy: Any) -> str:
    import torch

    digest = hashlib.sha256()
    count = 0
    for name, parameter in policy.named_parameters():
        if "lora_" in name.lower():
            continue
        value = parameter.detach().cpu().contiguous()
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(value.view(torch.uint8).numpy().tobytes())
        count += 1
    if count == 0:
        raise RuntimeError("no frozen Base parameters were hashed")
    return digest.hexdigest().upper()


def _parameter_report(named: Sequence[tuple[str, Any]], groups: Mapping[str, Mapping[str, str]]) -> dict[str, Any]:
    rows = []
    for name, parameter in named:
        rows.append(
            {
                "name": name,
                "shape": list(parameter.shape),
                "numel": int(parameter.numel()),
                "dtype": str(parameter.dtype),
                "coarse_group": groups["coarse"][name],
                "fine_group": groups["fine"][name],
            }
        )
    return {
        "method": "FAMR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "lora_config": {
            "rank": LORA_RANK,
            "alpha": LORA_ALPHA,
            "dropout": LORA_DROPOUT,
            "bias": "none",
        },
        "parameter_count": len(rows),
        "trainable_numel": sum(row["numel"] for row in rows),
        "lora_only_trainable": bool(rows) and all("lora_" in row["name"].lower() for row in rows),
        "coarse_group_counts": dict(sorted(__import__("collections").Counter(row["coarse_group"] for row in rows).items())),
        "fine_group_counts": dict(sorted(__import__("collections").Counter(row["fine_group"] for row in rows).items())),
        "parameters": rows,
    }


def _loss_on_rows(policy: Any, preprocessor: Any, samples: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> list[float]:
    import torch

    values = []
    policy.eval()
    with torch.no_grad():
        for sample, row in zip(samples, rows, strict=True):
            batch = _preprocess(preprocessor, sample)
            noise, time_tensor, _ = _shared_draw(str(row["row_id"]), "fixed_subset", 0, "cuda")
            values.append(_to_float(_loss(policy, batch, noise, time_tensor)))
            del batch
    return values


def _resume_training_state(
    path: Path, named: Sequence[tuple[str, Any]], optimizer: Any
) -> tuple[int, list[dict[str, Any]], list[dict[str, Any]], list[float] | None]:
    import torch

    if not path.is_file():
        return 0, [], [], None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("proposal_hash") != PROPOSAL_HASH or int(payload.get("rank")) != LORA_RANK:
        raise RuntimeError("existing FAMR micro-fit resume state has a different protocol identity")
    by_name = dict(named)
    saved = payload.get("trainable_state") or {}
    if set(saved) != set(by_name):
        raise RuntimeError("existing FAMR micro-fit resume parameter manifest differs")
    with torch.no_grad():
        for name, value in saved.items():
            by_name[name].copy_(value.to(by_name[name].device, by_name[name].dtype))
    optimizer.load_state_dict(payload["optimizer_state"])
    for state in optimizer.state.values():
        for key, value in state.items():
            if hasattr(value, "to"):
                state[key] = value.to("cuda")
    before_values = payload.get("fixed_subset_loss_before_values")
    return (
        int(payload["completed_steps"]),
        list(payload["loss_curve"]),
        list(payload["gradient_curve"]),
        [float(value) for value in before_values] if before_values is not None else None,
    )


def _save_training_state(
    path: Path,
    named: Sequence[tuple[str, Any]],
    optimizer: Any,
    completed_steps: int,
    loss_curve: Sequence[Mapping[str, Any]],
    gradient_curve: Sequence[Mapping[str, Any]],
    fixed_subset_loss_before_values: Sequence[float],
) -> None:
    import torch

    payload = {
        "method": "FAMR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "rank": LORA_RANK,
        "completed_steps": completed_steps,
        "trainable_state": {name: parameter.detach().cpu() for name, parameter in named},
        "optimizer_state": optimizer.state_dict(),
        "loss_curve": list(loss_curve),
        "gradient_curve": list(gradient_curve),
        "fixed_subset_loss_before_values": list(fixed_subset_loss_before_values),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _save_adapter_checkpoint(
    policy: Any,
    paths: Mapping[str, Path],
    training: Mapping[str, Any],
    probe_native: Any,
    probe_processed: np.ndarray,
) -> dict[str, Any]:
    checkpoint_dir = paths["checkpoint_dir"]
    if checkpoint_dir.exists():
        manifest = checkpoint_dir / "famr_training_manifest.json"
        probe = checkpoint_dir / "reload_probe.json"
        if not manifest.is_file() or not probe.is_file():
            raise RuntimeError("existing FAMR adapter checkpoint is incomplete")
        saved = _read_json(manifest)
        if saved.get("proposal_hash") != PROPOSAL_HASH:
            raise RuntimeError("existing FAMR adapter checkpoint has a different proposal hash")
        return {
            "checkpoint_path": str(checkpoint_dir),
            "files": _directory_hashes(checkpoint_dir),
            "saved": True,
            "resumed_existing_checkpoint": True,
            "reload_probe": _read_json(probe),
        }
    temporary = checkpoint_dir.with_name(f"{checkpoint_dir.name}.tmp_{os.getpid()}")
    temporary.mkdir(parents=True)
    try:
        if hasattr(policy, "peft_config"):
            for config in policy.peft_config.values():
                config.base_model_name_or_path = str(paths["checkpoint"])
        policy.save_pretrained(temporary)
        _write_json(
            temporary / "famr_training_manifest.json",
            {
                "method": "FAMR-VLA",
                "proposal_hash": PROPOSAL_HASH,
                "seed": SEED,
                "rank": LORA_RANK,
                "alpha": LORA_ALPHA,
                "dropout": LORA_DROPOUT,
                "micro_steps": MICRO_STEPS,
                "gradient_accumulation": GRADIENT_ACCUMULATION,
                "learning_rate": LEARNING_RATE,
                "weight_decay": WEIGHT_DECAY,
                "training": dict(training),
            },
        )
        _write_json(
            temporary / "reload_probe.json",
            {
                "native_action": probe_native,
                "postprocessed_action": probe_processed,
                "native_action_hash": _value_hash(probe_native),
            },
        )
        temporary.rename(checkpoint_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "checkpoint_path": str(checkpoint_dir),
        "files": _directory_hashes(checkpoint_dir),
        "saved": True,
        "resumed_existing_checkpoint": False,
        "reload_probe": _read_json(checkpoint_dir / "reload_probe.json"),
    }


def _scaling_audit(
    policy: Any,
    named: Sequence[tuple[str, Any]],
    first_batch: Mapping[str, Any],
    postprocessor: Any,
    identity_noise: Any,
    base_native: Any,
    base_processed: np.ndarray,
) -> dict[str, Any]:
    import torch

    full_native, full_processed = _predict(policy, first_batch, postprocessor, identity_noise)
    snapshot = {name: parameter.detach().clone() for name, parameter in named if "lora_b" in name.lower()}
    if not snapshot:
        raise RuntimeError("FAMR scaling audit found no LoRA B tensors")
    with torch.no_grad():
        for name, parameter in named:
            if name in snapshot:
                parameter.zero_()
    zero_native, zero_processed = _predict(policy, first_batch, postprocessor, identity_noise)
    with torch.no_grad():
        for name, parameter in named:
            if name in snapshot:
                parameter.copy_(snapshot[name])
    restored_native, restored_processed = _predict(policy, first_batch, postprocessor, identity_noise)

    by_name = dict(named)
    relative_errors = []
    absolute_errors = []
    pair_count = 0
    for b_name, b_parameter in named:
        if "lora_b" not in b_name.lower():
            continue
        a_name = re_sub_lora_side(b_name, "lora_B", "lora_A")
        if a_name not in by_name:
            raise RuntimeError(f"LoRA B tensor lacks paired A tensor: {b_name}")
        a_parameter = by_name[a_name]
        full_update = b_parameter.detach().float() @ a_parameter.detach().float()
        for coefficient in (0.0, 0.37, 1.0):
            scaled_update = scale_lora_b(b_parameter.detach().float(), coefficient) @ a_parameter.detach().float()
            expected = coefficient * full_update
            error = float(torch.max(torch.abs(scaled_update - expected)).item())
            denominator = max(float(torch.max(torch.abs(expected)).item()), 1e-12)
            absolute_errors.append(error)
            relative_errors.append(error / denominator)
        pair_count += 1
    return {
        "lora_pair_count": pair_count,
        "coefficient_zero_base_native_max_abs_error": float(torch.max(torch.abs(zero_native - base_native)).item()),
        "coefficient_zero_base_postprocessed_max_abs_error": float(np.max(np.abs(zero_processed - base_processed))),
        "coefficient_one_full_native_max_abs_error": float(torch.max(torch.abs(restored_native - full_native)).item()),
        "coefficient_one_full_postprocessed_max_abs_error": float(np.max(np.abs(restored_processed - full_processed))),
        "single_group_effective_weight_max_abs_error": max(absolute_errors),
        "single_group_effective_weight_max_relative_error": max(relative_errors),
        "passed": bool(
            float(torch.max(torch.abs(zero_native - base_native)).item()) <= 1e-6
            and float(np.max(np.abs(zero_processed - base_processed))) <= 1e-6
            and float(torch.max(torch.abs(restored_native - full_native)).item()) <= 1e-6
            and float(np.max(np.abs(restored_processed - full_processed))) <= 1e-6
            and max(relative_errors) <= 1e-6
        ),
    }


def re_sub_lora_side(name: str, old: str, new: str) -> str:
    if old in name:
        return name.replace(old, new)
    return name.replace(old.lower(), new.lower())


def _write_result_markdown(result: Mapping[str, Any], path: Path) -> None:
    data = result["data_semantics"]
    micro = result["micro_fit"]
    lines = [
        "# FAMR-VLA Stage 0A Result",
        "",
        f"Date: {DATE_KST} KST",
        "",
        f"Decision: `{result['final_decision']}`",
        "",
        "## Audit Summary",
        "",
        f"- target/pretraining exact intersection: `{result['task_provenance']['identity_audit']['intersection_count']}`",
        f"- target demonstrations: `{sum(row['demonstration_count'] for row in data['target_tasks'])}`",
        f"- source terminal successes/failures: `{data['source_terminal_success_count']} / {data['source_terminal_failure_count']}`",
        f"- discovery/validation/test model decodes: `{data['decoded_split_counts']}`",
        f"- duplicate episode/frame identity hashes: `{data['duplicate_episode_identity_hash_count']} / {data['duplicate_frame_identity_hash_count']}`",
        f"- adapter identity postprocessed max error: `{result['adapter_identity']['postprocessed_max_abs_error']}`",
        f"- micro-fit optimizer steps: `{micro['completed_steps']} / {MICRO_STEPS}`",
        f"- fixed-subset loss before/after: `{micro['fixed_subset_loss_before']} / {micro['fixed_subset_loss_after']}`",
        f"- fixed-subset relative reduction: `{micro['fixed_subset_relative_reduction']}`",
        f"- checkpoint reload max error: `{result['checkpoint']['reload_output_max_abs_error']}`",
        f"- scaling identity passed: `{result['scaling_audit']['passed']}`",
        f"- peak CUDA allocation GiB: `{result['peak_cuda_allocated_gib']}`",
        f"- confirmatory observations/actions: `{result['confirmatory']['observations_decoded']} / {result['confirmatory']['actions_computed']}`",
        f"- exception count: `{result['exception_count']}`",
        "",
        "## Adjudication",
        "",
        result["adjudication_text"],
        "",
        f"Next command: `{result['next_command']}`",
    ]
    _write_text(path, "\n".join(lines).rstrip() + "\n")


def _write_adjudication(result: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# FAMR-VLA Stage 0A Adjudication",
        "",
        f"Date: {DATE_KST} KST",
        "",
        f"Decision: `{result['final_decision']}`",
        "",
        result["adjudication_text"],
        "",
        "This is a provenance, implementation, and low-compute capacity decision. It is not a closed-loop scientific result.",
        "No confirmatory observation or action was decoded, and no rollout was launched.",
    ]
    _write_text(path, "\n".join(lines).rstrip() + "\n")


def run_audit(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, Any]:
    import torch

    started_unix = time.time()
    started_monotonic = time.monotonic()
    preflight = _preflight(args, paths, started_unix)
    if not preflight["passed"]:
        raise RuntimeError(f"FAMR Stage 0A preflight failed: {preflight}")

    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.cuda.reset_peak_memory_stats()
    _write_json(paths["status"], {"status": "running", "pid": os.getpid(), "started_unix": started_unix})
    _write_json(
        paths["partial"],
        {
            "method": "FAMR-VLA",
            "proposal_hash": PROPOSAL_HASH,
            "phase": "source_audit",
            "completed_count": 0,
            "planned_count": MICRO_STEPS,
            "exception_count": 0,
        },
    )

    official_tasks = _load_official_tasks()
    identity = task_identity_audit(official_tasks, TARGET_TASK_IDENTITIES)
    provenance = {
        "method": "FAMR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "checkpoint": str(paths["checkpoint"]),
        "checkpoint_dataset_info": _read_json(OFFICIAL_DATASET_ROOT / "meta" / "info.json"),
        "official_pretraining_tasks": official_tasks,
        "target_task_identities": list(TARGET_TASK_IDENTITIES),
        "identity_audit": identity,
    }
    _write_json(paths["provenance"], provenance)

    data_semantics, fixed_rows = _audit_target_sources(paths, preflight)
    calibration = _official_mapping_calibration(paths)
    data_semantics["official_mapping_calibration"] = calibration
    data_semantics_passed = bool(
        identity["intersection_count"] == 0
        and all(row["demonstration_count"] >= 45 for row in data_semantics["target_tasks"])
        and data_semantics["source_terminal_failure_count"] == 0
        and data_semantics["duplicate_episode_identity_hash_count"] == 0
        and data_semantics["duplicate_frame_identity_hash_count"] == 0
        and all(row["state_concat_max_abs_error"] <= 1e-12 for row in data_semantics["target_tasks"])
        and all(row["state_finite_fraction"] == 1.0 for row in data_semantics["target_tasks"])
        and all(row["action_finite_fraction"] == 1.0 for row in data_semantics["target_tasks"])
        and calibration["action_mapping_passed"]
        and calibration["state_mapping_passed"]
        and calibration["image_orientation_passed"]
    )
    data_semantics["passed"] = data_semantics_passed
    _write_json(paths["data_audit"], data_semantics)
    if not data_semantics_passed:
        raise RuntimeError(f"FAMR source semantics audit failed: {data_semantics}")

    samples = [_raw_sample(row) for row in fixed_rows]
    _write_json(
        paths["partial"],
        {
            "method": "FAMR-VLA",
            "proposal_hash": PROPOSAL_HASH,
            "phase": "model_load",
            "completed_count": 0,
            "planned_count": MICRO_STEPS,
            "exception_count": 0,
            "fixed_discovery_rows_loaded": len(samples),
        },
    )

    policy, config, preprocessor, postprocessor = _load_policy_and_processors(paths["checkpoint"])
    first_batch = _preprocess(preprocessor, samples[0])
    input_devices = _tensor_devices(first_batch)
    processed_shapes = _tensor_shapes(first_batch)
    if not input_devices or not all(str(device).startswith("cuda") for device in input_devices.values()):
        raise RuntimeError(f"FAMR preprocessor tensors are not all CUDA: {input_devices}")
    if processed_shapes.get("action") not in ([1, CHUNK_SIZE, 7], [1, CHUNK_SIZE, MAX_ACTION_DIM]):
        raise RuntimeError(f"FAMR processed action target shape mismatch: {processed_shapes.get('action')}")
    identity_noise, _, identity_draw = _shared_draw(str(fixed_rows[0]["row_id"]), "identity", 0, "cuda")
    base_native, base_processed = _predict(policy, first_batch, postprocessor, identity_noise)
    base_hash_before = _hash_base_parameters(policy)
    policy = policy.wrap_with_peft(
        peft_cli_overrides={
            "method_type": "LORA",
            "r": LORA_RANK,
            "lora_alpha": LORA_ALPHA,
            "lora_dropout": LORA_DROPOUT,
            "bias": "none",
        }
    )
    policy.to("cuda")
    policy.eval()
    named = _named_trainable(policy)
    groups = assign_parameter_groups([name for name, _ in named])
    parameter_report = _parameter_report(named, groups)
    adapter_native, adapter_processed = _predict(policy, first_batch, postprocessor, identity_noise)
    identity_native_error = float(torch.max(torch.abs(adapter_native - base_native)).item())
    identity_processed_error = float(np.max(np.abs(adapter_processed - base_processed)))
    parameter_report.update(
        {
            "base_frozen_hash_before": base_hash_before,
            "identity_native_max_abs_error": identity_native_error,
            "identity_postprocessed_max_abs_error": identity_processed_error,
        }
    )
    _write_json(paths["parameter_manifest"], parameter_report)

    optimizer = torch.optim.AdamW(
        [parameter for _, parameter in named], lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    completed_steps, loss_curve, gradient_curve, before_values = _resume_training_state(
        paths["resume_state"], named, optimizer
    )
    if before_values is None:
        before_values = _loss_on_rows(policy, preprocessor, samples, fixed_rows)
        _save_training_state(
            paths["resume_state"],
            named,
            optimizer,
            completed_steps,
            loss_curve,
            gradient_curve,
            before_values,
        )
    order = np.random.default_rng(SEED).permutation(len(samples)).tolist()
    policy.train()
    for step in range(completed_steps, MICRO_STEPS):
        if time.monotonic() - started_monotonic > MAX_RUNTIME_SECONDS:
            raise TimeoutError("FAMR Stage 0A exceeded its four-hour cap during micro-fit")
        optimizer.zero_grad(set_to_none=True)
        micro_losses = []
        draws = []
        row_ids = []
        for accumulation_index in range(GRADIENT_ACCUMULATION):
            logical_index = step * GRADIENT_ACCUMULATION + accumulation_index
            row_index = order[logical_index % len(order)]
            row = fixed_rows[row_index]
            batch = _preprocess(preprocessor, samples[row_index])
            noise, time_tensor, draw = _shared_draw(str(row["row_id"]), "micro_fit", logical_index, "cuda")
            loss = _loss(policy, batch, noise, time_tensor)
            loss_value = _to_float(loss)
            if not math.isfinite(loss_value):
                raise RuntimeError(f"nonfinite FAMR micro-fit loss at optimizer step {step}")
            (loss / GRADIENT_ACCUMULATION).backward()
            micro_losses.append(loss_value)
            draws.append(draw)
            row_ids.append(str(row["row_id"]))
            del batch, loss
        gradient_squared = 0.0
        nonzero_gradient_tensors = 0
        for _, parameter in named:
            if parameter.grad is None:
                continue
            gradient = parameter.grad.detach().float()
            if not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(f"nonfinite FAMR gradient at optimizer step {step}")
            norm = float(torch.linalg.vector_norm(gradient).item())
            gradient_squared += norm * norm
            nonzero_gradient_tensors += int(norm > 0.0)
        gradient_norm = math.sqrt(gradient_squared)
        if not math.isfinite(gradient_norm) or gradient_norm <= 0.0:
            raise RuntimeError(f"invalid FAMR gradient norm at optimizer step {step}: {gradient_norm}")
        optimizer.step()
        loss_curve.append(
            {
                "optimizer_step": step + 1,
                "mean_micro_loss": float(np.mean(micro_losses)),
                "micro_losses": micro_losses,
                "row_ids": row_ids,
                "draws": draws,
            }
        )
        gradient_curve.append(
            {
                "optimizer_step": step + 1,
                "gradient_norm": gradient_norm,
                "nonzero_gradient_tensor_count": nonzero_gradient_tensors,
            }
        )
        _save_training_state(
            paths["resume_state"],
            named,
            optimizer,
            step + 1,
            loss_curve,
            gradient_curve,
            before_values,
        )
        _write_json(
            paths["partial"],
            {
                "method": "FAMR-VLA",
                "proposal_hash": PROPOSAL_HASH,
                "phase": "micro_fit",
                "completed_count": step + 1,
                "planned_count": MICRO_STEPS,
                "exception_count": 0,
                "last_mean_micro_loss": float(np.mean(micro_losses)),
            },
        )
        print(
            f"[FAMR micro-fit] {step + 1}/{MICRO_STEPS} mean_loss={np.mean(micro_losses):.8f} grad={gradient_norm:.8f}",
            flush=True,
        )

    after_values = _loss_on_rows(policy, preprocessor, samples, fixed_rows)
    before = float(np.mean(before_values))
    after = float(np.mean(after_values))
    reduction = (before - after) / max(abs(before), 1e-12)
    base_hash_after = _hash_base_parameters(policy)
    micro_fit = {
        "completed_steps": MICRO_STEPS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "physical_batch_size": 1,
        "gradient_accumulation": GRADIENT_ACCUMULATION,
        "seed": SEED,
        "rank": LORA_RANK,
        "fixed_subset_row_count": len(fixed_rows),
        "fixed_subset_rows": [row["row_id"] for row in fixed_rows],
        "fixed_subset_loss_before_values": before_values,
        "fixed_subset_loss_after_values": after_values,
        "fixed_subset_loss_before": before,
        "fixed_subset_loss_after": after,
        "fixed_subset_relative_reduction": reduction,
        "fixed_subset_passed": reduction >= 0.01,
        "loss_curve": loss_curve,
        "gradient_curve": gradient_curve,
        "resumed_from_optimizer_step": completed_steps,
    }

    full_native, full_processed = _predict(policy, first_batch, postprocessor, identity_noise)
    scaling = _scaling_audit(
        policy,
        named,
        first_batch,
        postprocessor,
        identity_noise,
        base_native,
        base_processed,
    )
    checkpoint = _save_adapter_checkpoint(policy, paths, micro_fit, full_native, full_processed)
    del optimizer, policy, first_batch
    gc.collect()
    torch.cuda.empty_cache()

    policy, config, preprocessor, postprocessor = _load_adapter(paths)
    first_batch = _preprocess(preprocessor, samples[0])
    reloaded_native, reloaded_processed = _predict(policy, first_batch, postprocessor, identity_noise)
    reload_native_error = float(torch.max(torch.abs(reloaded_native - full_native)).item())
    reload_processed_error = float(np.max(np.abs(reloaded_processed - full_processed)))
    checkpoint.update(
        {
            "disk_reload": True,
            "reload_native_max_abs_error": reload_native_error,
            "reload_postprocessed_max_abs_error": reload_processed_error,
            "reload_output_max_abs_error": max(reload_native_error, reload_processed_error),
            "base_frozen_hash_after": base_hash_after,
            "base_frozen_hash_unchanged": base_hash_before == base_hash_after,
        }
    )
    _write_json(paths["checkpoint_manifest"], checkpoint)

    peak_cuda_gib = float(torch.cuda.max_memory_allocated() / 1024**3)
    gradient_health = bool(gradient_curve) and all(
        math.isfinite(float(row["gradient_norm"])) and float(row["gradient_norm"]) > 0.0
        for row in gradient_curve
    )
    split_integrity = bool(
        data_semantics["duplicate_episode_identity_hash_count"] == 0
        and data_semantics["duplicate_frame_identity_hash_count"] == 0
        and all(
            all(not values for values in row["split_audit"]["pairwise_overlap"].values())
            for row in data_semantics["target_tasks"]
        )
    )
    summary = {
        "essential_source_unavailable": False,
        "target_overlap_count": identity["intersection_count"],
        "preflight_passed": preflight["passed"],
        "data_semantics_passed": data_semantics_passed,
        "split_integrity_passed": split_integrity,
        "identity_passed": max(identity_native_error, identity_processed_error) <= 1e-6,
        "target_modules_passed": parameter_report["lora_only_trainable"],
        "gradient_health_passed": gradient_health,
        "checkpoint_reload_passed": checkpoint["reload_output_max_abs_error"] <= 1e-6,
        "group_assignment_passed": parameter_report["parameter_count"] == len(groups["coarse"]) == len(groups["fine"]),
        "scaling_identity_passed": scaling["passed"],
        "base_unchanged": base_hash_before == base_hash_after,
        "memory_passed": peak_cuda_gib <= MAX_CUDA_GIB,
        "confirmatory_sealed": True,
        "subset_fit_passed": micro_fit["fixed_subset_passed"],
        "capacity_check_used": False,
    }
    decision = classify_stage0(summary)
    if decision == "FAMR_STAGE_0A_PASS_ENDPOINT_TRAINING_ALLOWED":
        next_command = (
            f"{sys.executable} scripts/run_famr_vla_stage0.py --mode train-endpoint "
            f"--checkpoint {paths['checkpoint']} --libero-data-root {paths['data_root']} "
            f"--stable-artifact {paths['stable_artifact']} --run-root {paths['run_dir']} "
            f"--report-root {paths['report_dir']}"
        )
        adjudication_text = (
            "All frozen provenance, semantic, identity, gradient, fit, checkpoint, grouping, scaling, and memory gates passed. "
            "The fixed 300-step endpoint stage is authorized."
        )
    elif decision == "FAMR_UNDERPOWERED_ONE_CHECK_ALLOWED":
        next_command = "Run the one frozen rank-8 capacity check on the identical rows, steps, seed, optimizer, and targets."
        adjudication_text = (
            "Rank-4 gradients and implementation were healthy but the fixed-subset loss reduction missed 1%. "
            "This is underpowered and authorizes exactly one rank-8 capacity check; it is not a scientific kill."
        )
    else:
        next_command = "Preserve this Stage 0A result and continue to the next method cycle under current governance."
        adjudication_text = (
            "A frozen Stage 0A implementation, data, or low-compute gate failed. This is not a closed-loop scientific result."
        )

    finished_unix = time.time()
    result = {
        "method": "FAMR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "git_commit": _git_commit(),
        "command": " ".join(sys.argv),
        "started_unix": started_unix,
        "finished_unix": finished_unix,
        "elapsed_seconds_diagnostic_only": finished_unix - started_unix,
        "preflight": preflight,
        "experiment_boundaries": {
            "stage": "Stage 0A",
            "training_happened": True,
            "validation_search_happened": False,
            "closed_loop_experiment_happened": False,
            "confirmatory_test_tuning_happened": False,
            "test_episode_observation_decode_happened": False,
        },
        "task_provenance": provenance,
        "data_semantics": data_semantics,
        "input_contract": {
            "raw_sample_shapes": {
                key: list(value.shape) if hasattr(value, "shape") else None for key, value in samples[0].items()
            },
            "processed_shapes": processed_shapes,
            "processed_devices": input_devices,
            "identity_draw": identity_draw,
        },
        "adapter_identity": {
            "native_max_abs_error": identity_native_error,
            "postprocessed_max_abs_error": identity_processed_error,
        },
        "parameter_groups": parameter_report,
        "micro_fit": micro_fit,
        "scaling_audit": scaling,
        "checkpoint": checkpoint,
        "peak_cuda_allocated_gib": peak_cuda_gib,
        "confirmatory": {"observations_decoded": 0, "actions_computed": 0},
        "exception_count": 0,
        "decision_summary": summary,
        "final_decision": decision,
        "adjudication_text": adjudication_text,
        "next_command": next_command,
    }
    _write_json(paths["result_json"], result)
    _write_result_markdown(result, paths["result_md"])
    _write_adjudication(result, paths["adjudication"])
    _write_json(
        paths["partial"],
        {
            "method": "FAMR-VLA",
            "proposal_hash": PROPOSAL_HASH,
            "phase": "completed",
            "completed_count": MICRO_STEPS,
            "planned_count": MICRO_STEPS,
            "exception_count": 0,
            "final_decision": decision,
            "result_json": str(paths["result_json"]),
        },
    )
    _write_json(
        paths["status"],
        {
            "status": "completed",
            "pid": os.getpid(),
            "started_unix": started_unix,
            "finished_unix": finished_unix,
            "final_decision": decision,
        },
    )
    return result


def _write_blocker(paths: Mapping[str, Path], args: argparse.Namespace, exc: BaseException) -> None:
    partial = None
    if paths["partial"].is_file():
        try:
            partial = _read_json(paths["partial"])
        except Exception:
            partial = {"parse_failed": True}
    payload = {
        "method": "FAMR-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "mode": args.mode,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
        "partial": partial,
        "scientific_kill": False,
        "classification": "IMPLEMENTATION_OR_DATA_FAILURE",
    }
    _write_json(paths["blocker"], payload)
    completed = 0
    if isinstance(partial, dict):
        completed = int(partial.get("completed_count") or 0)
    _write_json(
        paths["partial"],
        {
            "method": "FAMR-VLA",
            "proposal_hash": PROPOSAL_HASH,
            "phase": "failed",
            "completed_count": completed,
            "planned_count": MICRO_STEPS,
            "exception_count": 1,
            "blocker": str(paths["blocker"]),
        },
    )
    _write_json(
        paths["status"],
        {
            "status": "failed",
            "pid": os.getpid(),
            "finished_unix": time.time(),
            "exception_type": type(exc).__name__,
            "exception": str(exc),
        },
    )


def _heartbeat_worker(stop: threading.Event, path: Path) -> None:
    while not stop.wait(15.0):
        _write_json(path, {"status": "running", "pid": os.getpid(), "updated_unix": time.time()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "train-endpoint", "headroom", "response-search"), default="audit")
    parser.add_argument("--checkpoint", default=str(_asset_path("checkpoints", "smolvla_libero")))
    parser.add_argument("--libero-data-root", default=str(_asset_path("data", "libero")))
    parser.add_argument(
        "--stable-artifact",
        default=str(REPO_ROOT / "reports" / "official_smolvla_stable_prediction_artifact.json"),
    )
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_DIR))
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_DIR))
    return parser.parse_args()


def main() -> int:
    _set_offline_environment()
    args = parse_args()
    paths = _paths(args)
    paths["run_dir"].mkdir(parents=True, exist_ok=True)
    paths["report_dir"].mkdir(parents=True, exist_ok=True)
    command = " ".join(sys.argv)
    _write_text(paths["worker_pid"], f"{os.getpid()}\n")
    _write_text(paths["child_pid"], f"{os.getpid()}\n")
    _write_text(paths["resume_command"], command + "\n")
    _write_json(paths["heartbeat"], {"status": "starting", "pid": os.getpid(), "updated_unix": time.time()})
    stop = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat_worker, args=(stop, paths["heartbeat"]), daemon=True)
    heartbeat.start()
    exit_code = 1
    try:
        if args.mode != "audit":
            raise RuntimeError(f"FAMR mode {args.mode} is out of order until a validated Stage 0A pass")
        result = run_audit(args, paths)
        print(json.dumps({"final_decision": result["final_decision"]}, sort_keys=True), flush=True)
        exit_code = 0
    except BaseException as exc:
        _write_blocker(paths, args, exc)
        traceback.print_exc()
    finally:
        stop.set()
        heartbeat.join(timeout=2.0)
        _write_json(paths["heartbeat"], {"status": "completed" if exit_code == 0 else "failed", "pid": os.getpid(), "updated_unix": time.time()})
        _write_text(paths["exit_code"], f"{exit_code}\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
