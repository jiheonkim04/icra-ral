"""Run the frozen PCAV-VLA Stage 0A candidate and headroom audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_famr_vla_stage0 import (  # noqa: E402
    HF_HOME,
    LIBERO_REPO,
    OFFICIAL_DATASET_ROOT,
    VLM_PATH,
    _active_linux_workers,
    _apply_official_env_image_processor,
    _clone_batch,
    _directory_hashes,
    _load_official_tasks,
    _load_policy_and_processors,
    _official_mapping_calibration,
    _preprocess,
    _raw_sample,
    _resource_evidence,
    _set_offline_environment,
    _sha256_file,
)
from tca_map.smolvla.famr_vla import task_identity_audit  # noqa: E402
from tca_map.smolvla.pcav_vla import (  # noqa: E402
    EXPANDED_PHASE_QUOTAS,
    INITIAL_PHASE_QUOTAS,
    PROPOSAL_HASH,
    TARGET_TASK_IDENTITIES,
    action_validity,
    aggregate_candidate_audit,
    canonical_json_sha256,
    candidate_diversity,
    classify_stage0a,
    discovery_action_scales,
    grouped_action_error,
    oracle_headroom,
    partition_identity_audit,
    row_key,
    select_stage0_rows,
    stable_seed,
    validate_partial_payload,
)


DATE_KST = "2026-07-15"
EXPECTED_STAGE = "epoch_4_cycle_18_pcav_stage_0a_implementation_pending"
CHUNK_SIZE = 50
MAX_ACTION_DIM = 32
FUTURE_OFFSET = 10
CANDIDATE_COUNT = 4
STATE_FILES = (
    REPO_ROOT / "reports" / "autonomous_until_paper_state.json",
    REPO_ROOT / "reports" / "autonomous_ral_campaign_state.json",
)
RESOURCE_REGISTRY = REPO_ROOT / "reports" / "resource_contention_intervals.json"
PROPOSAL_FILE = REPO_ROOT / "reports" / "pcav_vla" / "researcher_proposal.md"
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "pcav_vla" / "proposal_hash.txt"
DEFAULT_REPORT_ROOT = REPO_ROOT / "reports" / "pcav_vla"
DEFAULT_RUN_ROOT = REPO_ROOT / "runs" / "pcav_vla" / "stage0a"
FORBIDDEN_GATES = (
    "ALLOW_DOWNLOADS",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
)
TARGET_FILES = {
    TARGET_TASK_IDENTITIES[0]: "KITCHEN_SCENE9_put_the_frying_pan_under_the_cabinet_shelf_demo.hdf5",
    TARGET_TASK_IDENTITIES[1]: "LIVING_ROOM_SCENE4_pick_up_the_chocolate_pudding_and_put_it_in_the_tray_demo.hdf5",
    TARGET_TASK_IDENTITIES[2]: "STUDY_SCENE1_pick_up_the_book_and_place_it_in_the_left_compartment_of_the_caddy_demo.hdf5",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _file_hash(path: Path) -> str:
    return _sha256_file(path)


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    report_root = Path(args.report_root)
    run_root = Path(args.run_root)
    return {
        "report_root": report_root,
        "run_root": run_root,
        "checkpoint": Path(args.checkpoint),
        "data_root": Path(args.libero_data_root),
        "pid": run_root / "worker.pid",
        "heartbeat": run_root / "heartbeat.json",
        "status": run_root / "status.json",
        "partial": run_root / "partial_result.json",
        "exit_code": run_root / "exit_code.txt",
        "checkpoint_snapshot": run_root / "checkpoint_snapshot.json",
        "preflight": report_root / "stage_0a_preflight.json",
        "row_manifest": report_root / "stage_0a_row_manifest.json",
        "candidate_manifest": report_root / "stage_0a_candidate_manifest.json",
        "result_json": report_root / "stage_0a_result.json",
        "result_md": report_root / "stage_0a_result.md",
        "blocker": report_root / "stage_0a_implementation_blocker.json",
    }


def _pid_alive(pid: int) -> bool:
    return pid > 0 and Path(f"/proc/{pid}").exists()


def _proposal_hash_observed() -> str | None:
    return _file_hash(PROPOSAL_FILE) if PROPOSAL_FILE.is_file() else None


def _preflight(args: argparse.Namespace, paths: Mapping[str, Path]) -> dict[str, Any]:
    import torch

    started = time.time()
    states = {str(path.relative_to(REPO_ROOT)): _read_json(path) for path in STATE_FILES if path.is_file()}
    observed_stages = {name: state.get("current_stage") for name, state in states.items()}
    proposal_text = PROPOSAL_HASH_FILE.read_text(encoding="utf-8").strip() if PROPOSAL_HASH_FILE.is_file() else None
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
        "resource_registry": RESOURCE_REGISTRY,
        "proposal": PROPOSAL_FILE,
        "proposal_hash": PROPOSAL_HASH_FILE,
        **{f"target_{index}": path for index, path in enumerate(target_paths.values())},
        **{f"bddl_{index}": path for index, path in enumerate(bddl_paths.values())},
    }
    missing = [name for name, path in required.items() if not path.exists()]
    partial_audit = None
    partial_parse_error = None
    if paths["partial"].is_file():
        try:
            partial_audit = validate_partial_payload(_read_json(paths["partial"]))
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            partial_parse_error = str(exc)
    existing_pid = None
    existing_pid_alive = False
    if paths["pid"].is_file():
        try:
            existing_pid = int(paths["pid"].read_text(encoding="utf-8").strip())
            existing_pid_alive = _pid_alive(existing_pid) and existing_pid != os.getpid()
        except ValueError:
            existing_pid = None
    other_workers = _active_linux_workers()
    forbidden = [name for name in FORBIDDEN_GATES if os.environ.get(name) == "1"]
    registry = _read_json(RESOURCE_REGISTRY) if RESOURCE_REGISTRY.is_file() else {"intervals": []}
    disk_root = paths["run_root"].parent if paths["run_root"].parent.exists() else REPO_ROOT
    disk_free_gib = shutil.disk_usage(disk_root).free / 1024**3
    stage_ok = len(states) == len(STATE_FILES) and all(stage == EXPECTED_STAGE for stage in observed_stages.values())
    mode_ok = args.mode in {"audit", "stage0a"}
    result_absent = not paths["result_json"].exists()
    passed = bool(
        not missing
        and stage_ok
        and proposal_text == PROPOSAL_HASH
        and _proposal_hash_observed() == PROPOSAL_HASH
        and not existing_pid_alive
        and not other_workers
        and not forbidden
        and partial_parse_error is None
        and result_absent
        and torch.cuda.is_available()
        and disk_free_gib >= 5.0
        and mode_ok
    )
    return {
        "passed": passed,
        "mode": args.mode,
        "expected_stage": EXPECTED_STAGE,
        "observed_stages": observed_stages,
        "stage_ok": stage_ok,
        "proposal_hash_expected": PROPOSAL_HASH,
        "proposal_hash_file": proposal_text,
        "proposal_hash_observed": _proposal_hash_observed(),
        "proposal_hash_ok": proposal_text == PROPOSAL_HASH and _proposal_hash_observed() == PROPOSAL_HASH,
        "missing_paths": missing,
        "required_paths": {name: str(path) for name, path in required.items()},
        "target_paths": {task: str(path) for task, path in target_paths.items()},
        "bddl_paths": {task: str(path) for task, path in bddl_paths.items()},
        "existing_pid": existing_pid,
        "existing_pid_alive": existing_pid_alive,
        "active_linux_workers": other_workers,
        "partial_audit": partial_audit,
        "partial_parse_error": partial_parse_error,
        "result_absent": result_absent,
        "forbidden_gates_enabled": forbidden,
        "cuda_available": bool(torch.cuda.is_available()),
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "disk_free_gib": disk_free_gib,
        "resource_evidence": _resource_evidence(registry, started),
    }


def _source_audit(
    paths: Mapping[str, Path], preflight: Mapping[str, Any]
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    import h5py

    partitions = {"discovery": range(0, 30), "validation": range(30, 40), "confirmatory_offline": range(40, 50)}
    rows: dict[str, list[dict[str, Any]]] = {name: [] for name in partitions}
    task_reports = []
    discovery_actions = []
    terminal_success_count = 0
    terminal_failure_count = 0
    for task_index, task in enumerate(TARGET_TASK_IDENTITIES):
        source_path = Path(preflight["target_paths"][task])
        with h5py.File(source_path, "r") as handle:
            data = handle["data"]
            demo_names = sorted(data.keys(), key=lambda value: int(value.rsplit("_", 1)[1]))
            if [int(name.rsplit("_", 1)[1]) for name in demo_names] != list(range(50)):
                raise RuntimeError(f"{task} does not contain exactly demo_0..demo_49")
            problem_info = json.loads(str(data.attrs["problem_info"]))
            language = str(problem_info["language_instruction"])
            lengths = []
            repeated_actions = 0
            repeated_frames = 0
            adjacent_count = 0
            padding_count = 0
            task_discovery_actions = []
            task_discovery_states = []
            for episode, name in enumerate(demo_names):
                demo = data[name]
                actions = np.asarray(demo["actions"], dtype=np.float32)
                observations = demo["obs"]
                states = np.concatenate(
                    [
                        np.asarray(observations["ee_states"], dtype=np.float32),
                        np.asarray(observations["gripper_states"], dtype=np.float32),
                    ],
                    axis=-1,
                )
                length = int(actions.shape[0])
                if actions.shape != (length, 7) or states.shape != (length, 8):
                    raise RuntimeError(f"unexpected action/state shape for {task} {name}: {actions.shape} / {states.shape}")
                lengths.append(length)
                success = bool(float(demo["dones"][-1]) > 0.0 and float(demo["rewards"][-1]) > 0.0)
                terminal_success_count += int(success)
                terminal_failure_count += int(not success)
                done_indices = np.flatnonzero(np.asarray(demo["dones"]) > 0)
                if done_indices.size:
                    padding_count += max(0, length - int(done_indices[0]) - 1)
                partition = next(name for name, episodes in partitions.items() if episode in episodes)
                for frame in range(max(0, length - FUTURE_OFFSET)):
                    rows[partition].append(
                        {
                            "task_index": task_index,
                            "task_identity": task,
                            "task_language": language,
                            "source_path": str(source_path),
                            "episode": episode,
                            "frame": frame,
                            "episode_length": length,
                        }
                    )
                if episode < 30:
                    task_discovery_actions.append(actions)
                    task_discovery_states.append(states)
                    discovery_actions.append(actions)
                    if length > 1:
                        repeated_actions += int(np.sum(np.all(np.isclose(actions[1:], actions[:-1], atol=0.0), axis=1)))
                        agent = np.asarray(observations["agentview_rgb"])
                        wrist = np.asarray(observations["eye_in_hand_rgb"])
                        repeated_frames += int(
                            np.sum(
                                np.all(agent[1:] == agent[:-1], axis=(1, 2, 3))
                                & np.all(wrist[1:] == wrist[:-1], axis=(1, 2, 3))
                            )
                        )
                        adjacent_count += length - 1
            discovery_array = np.concatenate(task_discovery_actions, axis=0)
            discovery_state_array = np.concatenate(task_discovery_states, axis=0)
            bddl_path = Path(preflight["bddl_paths"][task])
            task_reports.append(
                {
                    "task_index": task_index,
                    "task_identity": task,
                    "task_language": language,
                    "source_path": str(source_path),
                    "source_sha256": _file_hash(source_path),
                    "bddl_path": str(bddl_path),
                    "bddl_sha256": _file_hash(bddl_path),
                    "demonstration_count": len(demo_names),
                    "episode_length_min": min(lengths),
                    "episode_length_max": max(lengths),
                    "episode_length_mean": float(np.mean(lengths)),
                    "episode_length_std": float(np.std(lengths)),
                    "terminal_padding_count": padding_count,
                    "repeated_action_fraction": repeated_actions / adjacent_count if adjacent_count else 0.0,
                    "repeated_frame_fraction": repeated_frames / adjacent_count if adjacent_count else 0.0,
                    "discovery_action_finite_fraction": float(np.mean(np.isfinite(discovery_array))),
                    "discovery_state_finite_fraction": float(np.mean(np.isfinite(discovery_state_array))),
                    "action_dimension": int(discovery_array.shape[-1]),
                    "state_dimension": int(discovery_state_array.shape[-1]),
                    "normalized_time_variance": float(np.var(np.concatenate([np.linspace(0, 1, length) for length in lengths[:30]]))),
                }
            )

    pretraining_tasks = _load_official_tasks()
    identity = task_identity_audit(pretraining_tasks, TARGET_TASK_IDENTITIES)
    partition_audit = partition_identity_audit(rows)
    scales = discovery_action_scales(discovery_actions)
    source_health_passed = bool(
        terminal_success_count == 150
        and terminal_failure_count == 0
        and identity["intersection_count"] == 0
        and partition_audit["passed"]
        and all(
            report["demonstration_count"] == 50
            and report["discovery_action_finite_fraction"] == 1.0
            and report["discovery_state_finite_fraction"] == 1.0
            and report["action_dimension"] == 7
            and report["state_dimension"] == 8
            and report["normalized_time_variance"] > 0.0
            and bool(report["source_sha256"])
            and bool(report["bddl_sha256"])
            for report in task_reports
        )
    )
    return (
        {
            "task_reports": task_reports,
            "source_terminal_success_count": terminal_success_count,
            "source_terminal_failure_count": terminal_failure_count,
            "task_identity_audit": identity,
            "partition_audit": partition_audit,
            "discovery_action_scales": scales,
            "source_health_passed": source_health_passed,
            "confirmatory_observations_decoded": 0,
            "confirmatory_actions_computed": 0,
        },
        rows,
    )


def _manifest_health(rows: Sequence[Mapping[str, Any]], quotas: Mapping[str, int]) -> dict[str, Any]:
    import h5py

    keys = [row_key(row) for row in rows]
    task_phase_counts = {
        f"{task}|{phase}": sum(
            str(row["task_identity"]) == task and str(row["phase"]) == phase for row in rows
        )
        for task in TARGET_TASK_IDENTITIES
        for phase in ("early", "middle", "late")
    }
    expected_task_phase_counts = {
        f"{task}|{phase}": int(quotas[phase])
        for task in TARGET_TASK_IDENTITIES
        for phase in ("early", "middle", "late")
    }
    prefix_identity_hashes = []
    prefix_value_hashes = []
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["source_path"]), []).append(row)
    for source_path, source_rows in grouped.items():
        with h5py.File(source_path, "r") as handle:
            for row in source_rows:
                actions = np.asarray(
                    handle["data"][f"demo_{int(row['episode'])}"]["actions"][
                        int(row["frame"]) : int(row["frame"]) + FUTURE_OFFSET
                    ],
                    dtype=np.float32,
                )
                if actions.shape != (FUTURE_OFFSET, 7):
                    raise RuntimeError(f"manifest row lacks a full action prefix: {row_key(row)}")
                prefix_value_hash = hashlib.sha256(actions.tobytes()).hexdigest().upper()
                prefix_value_hashes.append(prefix_value_hash)
                prefix_identity_hashes.append(
                    hashlib.sha256(f"{row_key(row)}|{prefix_value_hash}".encode("utf-8")).hexdigest().upper()
                )
    duplicate_row_count = len(keys) - len(set(keys))
    duplicate_prefix_identity_count = len(prefix_identity_hashes) - len(set(prefix_identity_hashes))
    return {
        "row_count": len(rows),
        "duplicate_row_count": duplicate_row_count,
        "duplicate_action_prefix_identity_count": duplicate_prefix_identity_count,
        "duplicate_action_prefix_value_count_diagnostic": len(prefix_value_hashes) - len(set(prefix_value_hashes)),
        "task_phase_counts": task_phase_counts,
        "expected_task_phase_counts": expected_task_phase_counts,
        "passed": bool(
            duplicate_row_count == 0
            and duplicate_prefix_identity_count == 0
            and task_phase_counts == expected_task_phase_counts
        ),
    }


def _noise(row_identity: str, candidate_index: int, device: str) -> tuple[Any, dict[str, Any]]:
    import torch

    seed = stable_seed(PROPOSAL_HASH, "stage0a", row_identity, candidate_index)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    value_cpu = torch.randn((1, CHUNK_SIZE, MAX_ACTION_DIM), generator=generator, dtype=torch.float32)
    value = value_cpu.to(device)
    digest = hashlib.sha256(value_cpu.numpy().tobytes()).hexdigest().upper()
    return value, {"seed": seed, "noise_hash": digest, "noise_shape": [1, CHUNK_SIZE, MAX_ACTION_DIM]}


def _postprocess_chunk(native: Any, postprocessor: Any) -> np.ndarray:
    processed = postprocessor(native)
    if hasattr(processed, "detach"):
        processed = processed.detach().cpu().numpy()
    array = np.asarray(processed, dtype=np.float32)
    if array.size != CHUNK_SIZE * 7:
        raise RuntimeError(f"postprocessed action chunk has {array.size} values, expected {CHUNK_SIZE * 7}")
    return array.reshape(CHUNK_SIZE, 7)


def _predict_chunk(policy: Any, batch: Mapping[str, Any], postprocessor: Any, noise: Any) -> tuple[np.ndarray, np.ndarray]:
    import torch

    if hasattr(policy, "reset"):
        policy.reset()
    policy.eval()
    with torch.no_grad():
        native = policy.predict_action_chunk(_clone_batch(batch), noise=noise.clone())
    native_cpu = native.detach().float().cpu().numpy()
    return native_cpu.reshape(CHUNK_SIZE, 7), _postprocess_chunk(native, postprocessor)


def _heartbeat_loop(path: Path, state: dict[str, Any], stop: threading.Event) -> None:
    while not stop.wait(10.0):
        _write_json(path, {**state, "updated_at": _utc_now()})


def _partial_payload(manifest_hash: str, planned: int, rows: Sequence[Mapping[str, Any]], exception_count: int = 0) -> dict[str, Any]:
    return {
        "method": "PCAV-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "planned_row_count": planned,
        "completed_row_count": len(rows),
        "completed_row_keys": [row_key(row) for row in rows],
        "rows": list(rows),
        "exception_count": exception_count,
        "updated_at": _utc_now(),
    }


def _run_row(
    row: Mapping[str, Any],
    *,
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    scales: Mapping[str, float],
) -> dict[str, Any]:
    raw = _raw_sample(row)
    batch = _preprocess(preprocessor, raw)
    expert = raw["action"].detach().cpu().numpy().reshape(CHUNK_SIZE, 7)
    generated: list[dict[str, Any]] = []
    native_chunks: list[np.ndarray] = []
    processed_chunks: list[np.ndarray] = []
    identity = row_key(row)
    for candidate_index in range(CANDIDATE_COUNT):
        noise, draw = _noise(identity, candidate_index, "cuda")
        native, processed = _predict_chunk(policy, batch, postprocessor, noise)
        native_chunks.append(native)
        processed_chunks.append(processed)
        generated.append(
            {
                "candidate_index": candidate_index,
                **draw,
                "native_shape": list(native.shape),
                "native_finite_fraction": float(np.mean(np.isfinite(native))),
                "processed_action_chunk": processed.tolist(),
                "action_error": grouped_action_error(processed, expert, scales),
            }
        )

    direct_noise, _ = _noise(identity, 0, "cuda")
    _, direct_base = _predict_chunk(policy, batch, postprocessor, direct_noise)
    base = processed_chunks[0]
    identity_error = float(np.max(np.abs(direct_base - base)))
    for candidate, chunk in zip(generated, processed_chunks, strict=True):
        candidate["validity"] = action_validity(chunk, base)
        delta = chunk[:10] - base[:10]
        candidate["base_delta"] = {
            "translation_l2_mean": float(np.mean(np.linalg.norm(delta[:, :3], axis=1))),
            "rotation_l2_mean": float(np.mean(np.linalg.norm(delta[:, 3:6], axis=1))),
            "gripper_disagreement_fraction": float(np.mean(np.abs(delta[:, 6]) > 1e-6)),
        }

    valid_chunks = [base] + [
        chunk for candidate, chunk in zip(generated[1:], processed_chunks[1:], strict=True) if candidate["validity"]["passed"]
    ]
    diversity = candidate_diversity(valid_chunks if len(valid_chunks) >= 2 else [base, base.copy()])
    errors = [float(generated[0]["action_error"]["aggregate"])] + [
        float(candidate["action_error"]["aggregate"])
        if candidate["validity"]["passed"]
        else float(generated[0]["action_error"]["aggregate"])
        for candidate in generated[1:]
    ]
    return {
        **dict(row),
        "row_key": identity,
        "expert_action_prefix": expert[:10].tolist(),
        "base_identity_max_abs_error": identity_error,
        "candidates": generated,
        "diversity": diversity,
        "eligible_action_errors": errors,
    }


def _evaluate_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    mapping_passed: bool,
    partition_passed: bool,
    reload_passed: bool,
    source_health_passed: bool,
    manifest_passed: bool,
) -> tuple[dict[str, Any], str]:
    aggregate = aggregate_candidate_audit(rows)
    identity_error = max((float(row["base_identity_max_abs_error"]) for row in rows), default=float("inf"))
    headroom = oracle_headroom([row["eligible_action_errors"] for row in rows])
    audit = {
        **aggregate,
        "completed_row_count": len(rows),
        "exception_count": 0,
        "duplicate_key_count": len(rows) - len({row_key(row) for row in rows}),
        "mapping_passed": mapping_passed,
        "partition_passed": partition_passed,
        "reload_passed": reload_passed,
        "source_health_passed": source_health_passed,
        "manifest_passed": manifest_passed,
        "base_identity_max_abs_error": identity_error,
        "confirmatory_observations_decoded": 0,
        "confirmatory_actions_computed": 0,
        "headroom": headroom,
    }
    return audit, classify_stage0a(audit)


def _write_markdown(result: Mapping[str, Any], path: Path) -> None:
    audit = result["stage_0a_audit"]
    headroom = audit["headroom"]
    lines = [
        "# PCAV-VLA Stage 0A Result",
        "",
        f"Decision: `{result['final_decision']}`",
        "",
        f"- completed rows: `{audit['completed_row_count']} / {result['planned_row_count']}`",
        f"- exception count: `{audit['exception_count']}`",
        f"- duplicate key count: `{audit['duplicate_key_count']}`",
        f"- Base identity max abs error: `{audit['base_identity_max_abs_error']}`",
        f"- rows with two unique valid chunks: `{audit['fraction_rows_with_two_unique_chunks']}`",
        f"- rows with a valid alternative: `{audit['fraction_rows_with_valid_alternative']}`",
        f"- materially better oracle fraction: `{headroom['materially_better_fraction']}`",
        f"- median oracle reduction on improvable rows: `{headroom['median_oracle_relative_reduction_improvable']}`",
        f"- invalid candidate count: `{audit['invalid_candidate_count']}`",
        f"- confirmatory observations/actions: `0 / 0`",
        f"- Stage 0B allowed: `{str(result['stage_0b_allowed']).lower()}`",
        "",
        "Timing and resource metrics are diagnostic and follow the resource-contention registry.",
    ]
    _write_text(path, "\n".join(lines) + "\n")


def _run_stage0a(args: argparse.Namespace, paths: Mapping[str, Path], preflight: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    started_unix = time.time()
    paths["run_root"].mkdir(parents=True, exist_ok=True)
    paths["report_root"].mkdir(parents=True, exist_ok=True)
    _write_text(paths["pid"], f"{os.getpid()}\n")
    heartbeat_state = {"pid": os.getpid(), "status": "running", "mode": "stage0a"}
    _write_json(paths["status"], {**heartbeat_state, "started_at": _utc_now()})
    _write_json(paths["heartbeat"], {**heartbeat_state, "updated_at": _utc_now()})
    stop = threading.Event()
    thread = threading.Thread(target=_heartbeat_loop, args=(paths["heartbeat"], heartbeat_state, stop), daemon=True)
    thread.start()

    try:
        source_audit, partition_rows = _source_audit(paths, preflight)
        mapping = _official_mapping_calibration({"data_root": paths["data_root"]})
        mapping_passed = bool(
            mapping["action_mapping_passed"] and mapping["state_mapping_passed"] and mapping["image_orientation_passed"]
        )
        initial_rows = select_stage0_rows(partition_rows["discovery"], INITIAL_PHASE_QUOTAS)
        expanded_rows = select_stage0_rows(partition_rows["discovery"], EXPANDED_PHASE_QUOTAS)
        if not {row_key(row) for row in initial_rows} <= {row_key(row) for row in expanded_rows}:
            raise RuntimeError("24-row manifest is not a subset of 96-row manifest")
        initial_manifest_health = _manifest_health(initial_rows, INITIAL_PHASE_QUOTAS)
        expanded_manifest_health = _manifest_health(expanded_rows, EXPANDED_PHASE_QUOTAS)
        if not initial_manifest_health["passed"] or not expanded_manifest_health["passed"]:
            raise RuntimeError("frozen Stage 0A row manifest failed quota or duplicate audit")
        row_manifest = {
            "method": "PCAV-VLA",
            "proposal_hash": PROPOSAL_HASH,
            "selection_rule": "SHA256(proposal_hash|task|episode|frame) within task/phase",
            "initial_phase_quotas_per_task": INITIAL_PHASE_QUOTAS,
            "expanded_phase_quotas_per_task": EXPANDED_PHASE_QUOTAS,
            "initial_rows": initial_rows,
            "expanded_rows": expanded_rows,
            "initial_manifest_health": initial_manifest_health,
            "expanded_manifest_health": expanded_manifest_health,
            "partition_audit": source_audit["partition_audit"],
        }
        row_manifest["manifest_hash"] = canonical_json_sha256(row_manifest)
        _write_json(paths["row_manifest"], row_manifest)
        manifest_hash = str(row_manifest["manifest_hash"])

        completed_rows: list[dict[str, Any]] = []
        planned = 24
        if paths["partial"].is_file():
            partial = _read_json(paths["partial"])
            partial_audit = validate_partial_payload(partial)
            if partial.get("manifest_hash") != manifest_hash:
                raise RuntimeError("partial manifest hash mismatch")
            completed_rows = list(partial["rows"])
            planned = int(partial_audit["planned_row_count"])
            allowed_rows = initial_rows if planned == 24 else expanded_rows
            allowed_keys = {row_key(row) for row in allowed_rows}
            unexpected_keys = sorted({row_key(row) for row in completed_rows} - allowed_keys)
            if unexpected_keys:
                raise RuntimeError(f"partial contains off-manifest completed rows: {unexpected_keys}")

        checkpoint_before = _directory_hashes(paths["checkpoint"])
        _write_json(paths["checkpoint_snapshot"], {"before": checkpoint_before})
        _set_offline_environment()
        policy, _, preprocessor, postprocessor = _load_policy_and_processors(paths["checkpoint"])
        torch.cuda.reset_peak_memory_stats()

        def complete_manifest(target_rows: Sequence[Mapping[str, Any]], planned_count: int) -> None:
            existing = {row_key(row) for row in completed_rows}
            for index, row in enumerate(target_rows):
                key = row_key(row)
                if key in existing:
                    continue
                completed = _run_row(
                    row,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    scales=source_audit["discovery_action_scales"],
                )
                completed_rows.append(completed)
                existing.add(key)
                _write_json(paths["partial"], _partial_payload(manifest_hash, planned_count, completed_rows))
                heartbeat_state.update({"planned_row_count": planned_count, "completed_row_count": len(completed_rows)})
                _write_json(paths["heartbeat"], {**heartbeat_state, "updated_at": _utc_now()})
                print(f"[pcav-stage0a] rows {len(completed_rows)}/{planned_count} ({index + 1}/{len(target_rows)})", flush=True)

        target = initial_rows if planned == 24 else expanded_rows
        complete_manifest(target, planned)

        reload_row = completed_rows[0]
        reload_raw = _raw_sample(reload_row)
        reload_batch = _preprocess(preprocessor, reload_raw)
        reload_noise, _ = _noise(row_key(reload_row), 0, "cuda")
        _, before_reload = _predict_chunk(policy, reload_batch, postprocessor, reload_noise)
        del policy
        torch.cuda.empty_cache()
        reloaded_policy, _, reloaded_preprocessor, reloaded_postprocessor = _load_policy_and_processors(paths["checkpoint"])
        reload_batch_2 = _preprocess(reloaded_preprocessor, reload_raw)
        reload_noise_2, _ = _noise(row_key(reload_row), 0, "cuda")
        _, after_reload = _predict_chunk(reloaded_policy, reload_batch_2, reloaded_postprocessor, reload_noise_2)
        reload_error = float(np.max(np.abs(after_reload - before_reload)))
        reload_passed = reload_error == 0.0

        audit, decision = _evaluate_rows(
            completed_rows,
            mapping_passed=mapping_passed,
            partition_passed=bool(source_audit["partition_audit"]["passed"]),
            reload_passed=reload_passed,
            source_health_passed=bool(source_audit["source_health_passed"]),
            manifest_passed=bool(initial_manifest_health["passed"]),
        )
        expansion_used = planned == 96
        if decision == "PCAV_STAGE_0A_UNRESOLVED_EXPANSION_REQUIRED":
            planned = 96
            expansion_used = True
            _write_json(paths["partial"], _partial_payload(manifest_hash, planned, completed_rows))
            complete_manifest(expanded_rows, planned)
            audit, decision = _evaluate_rows(
                completed_rows,
                mapping_passed=mapping_passed,
                partition_passed=bool(source_audit["partition_audit"]["passed"]),
                reload_passed=reload_passed,
                source_health_passed=bool(source_audit["source_health_passed"]),
                manifest_passed=bool(expanded_manifest_health["passed"]),
            )
            if decision == "PCAV_STAGE_0A_UNRESOLVED_EXPANSION_REQUIRED":
                raise RuntimeError("96-row expansion remained in transient unresolved decision")

        checkpoint_after = _directory_hashes(paths["checkpoint"])
        base_hash_unchanged = checkpoint_before == checkpoint_after
        if not base_hash_unchanged:
            audit["mapping_passed"] = False
            decision = "PCAV_STAGE_0A_IMPLEMENTATION_OR_DATA_FAILURE"
        candidate_manifest = {
            "method": "PCAV-VLA",
            "proposal_hash": PROPOSAL_HASH,
            "row_manifest_hash": manifest_hash,
            "candidate_count_per_row": CANDIDATE_COUNT,
            "rows": completed_rows,
        }
        candidate_manifest["manifest_hash"] = canonical_json_sha256(candidate_manifest)
        _write_json(paths["candidate_manifest"], candidate_manifest)
        registry = _read_json(RESOURCE_REGISTRY)
        result = {
            "method": "PCAV-VLA",
            "date_kst": DATE_KST,
            "proposal_hash": PROPOSAL_HASH,
            "started_at": datetime.fromtimestamp(started_unix, timezone.utc).isoformat(),
            "completed_at": _utc_now(),
            "preflight": preflight,
            "resource_evidence": _resource_evidence(registry, started_unix),
            "source_audit": source_audit,
            "raw_mapping_audit": mapping,
            "row_manifest": str(paths["row_manifest"]),
            "row_manifest_hash": manifest_hash,
            "candidate_manifest": str(paths["candidate_manifest"]),
            "candidate_manifest_hash": candidate_manifest["manifest_hash"],
            "planned_row_count": planned,
            "completed_row_count": len(completed_rows),
            "expansion_used": expansion_used,
            "missing_row_count": planned - len(completed_rows),
            "candidate_count_per_row": CANDIDATE_COUNT,
            "exception_count": 0,
            "duplicate_key_count": audit["duplicate_key_count"],
            "base_identity_max_abs_error": audit["base_identity_max_abs_error"],
            "stage_0a_audit": audit,
            "checkpoint_reload_max_abs_error": reload_error,
            "checkpoint_base_hash_unchanged": base_hash_unchanged,
            "checkpoint_file_count": len(checkpoint_after),
            "peak_cuda_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "confirmatory_observations_decoded": 0,
            "confirmatory_actions_computed": 0,
            "final_decision": decision,
            "failure_class": None
            if decision == "PCAV_STAGE_0A_PASS_STAGE_0B_ALLOWED"
            else (
                "NO_HEADROOM"
                if decision == "PCAV_STAGE_0A_NO_USABLE_HEADROOM"
                else "DESIGN_FAILURE"
                if decision == "PCAV_STAGE_0A_DESIGN_FAILURE_CANDIDATES_COLLAPSED"
                else "IMPLEMENTATION_OR_DATA_FAILURE"
            ),
            "valid_scientific_kill": False,
            "stage_0b_allowed": decision == "PCAV_STAGE_0A_PASS_STAGE_0B_ALLOWED",
        }
        _write_json(paths["result_json"], result)
        _write_markdown(result, paths["result_md"])
        _write_json(paths["checkpoint_snapshot"], {"before": checkpoint_before, "after": checkpoint_after})
        _write_json(
            paths["status"],
            {
                "pid": os.getpid(),
                "status": "completed",
                "completed_at": _utc_now(),
                "final_decision": decision,
                "planned_row_count": planned,
                "completed_row_count": len(completed_rows),
                "exception_count": 0,
            },
        )
        heartbeat_state["status"] = "completed"
        _write_json(paths["heartbeat"], {**heartbeat_state, "updated_at": _utc_now()})
        _write_text(paths["exit_code"], "0\n")
        return result
    finally:
        stop.set()
        thread.join(timeout=2.0)


def _write_blocker(paths: Mapping[str, Path], exc: BaseException) -> None:
    blocker = {
        "method": "PCAV-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "failure_class": "IMPLEMENTATION_OR_DATA_FAILURE",
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": traceback.format_exc(),
        "time": _utc_now(),
    }
    _write_json(paths["blocker"], blocker)
    _write_json(
        paths["status"],
        {"pid": os.getpid(), "status": "failed", "failed_at": _utc_now(), "exception_count": 1},
    )
    _write_text(paths["exit_code"], "1\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("audit", "stage0a"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--libero-data-root", required=True)
    parser.add_argument("--run-root", default=str(DEFAULT_RUN_ROOT))
    parser.add_argument("--report-root", default=str(DEFAULT_REPORT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = _paths(args)
    preflight = _preflight(args, paths)
    _write_json(paths["preflight"], preflight)
    print(json.dumps(preflight, indent=2, sort_keys=True), flush=True)
    if not preflight["passed"]:
        return 2
    if args.mode == "audit":
        return 0
    try:
        result = _run_stage0a(args, paths, preflight)
        print(json.dumps({"final_decision": result["final_decision"]}, sort_keys=True), flush=True)
        return 0
    except BaseException as exc:
        _write_blocker(paths, exc)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
