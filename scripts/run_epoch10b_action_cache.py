"""Freeze and materialize the Epoch 10B development action cache.

The frozen action unit follows the original Epoch 10 queue-origin protocol:
generate a 50-step queue from floor(frame/50)*50 with a fixed RNG identity and
pop through frame modulo 50.  Only the final 7-D environment action is cached.
No simulator result or closed-loop success label is read by this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_epoch10b_checkpoint_expansion import (
    CONTINUATION_PROMPT_SHA256,
    DEVELOPMENT_SEEDS,
    HOLDOUT_SEEDS,
    ORIGINAL_PROMPT_SHA256,
    RETAINED_STAGES,
)
from scripts.run_epoch10b_fresh_controller_assay import (
    DEVELOPMENT_RESERVED_DEMOS,
    HELDOUT_RESERVED_DEMOS,
    STATE_PHASE_SPECS,
    _frame_for_fraction,
)
from tca_map.smolvla.official_libero_stable_artifact_eval import _postprocess_action


SCHEMA_VERSION = 1
CAMPAIGN = "epoch10b_icae_fresh_controller"
CHUNK_SIZE = 50
SELECTED_HORIZON = 4
SELECTED_ENDPOINT = "bounded_expert_recovery_cost"
DEVELOPMENT_RESET_SEEDS = tuple(range(20, 35))
EXPECTED_DEVELOPMENT_STATES = 240
EXPECTED_DEVELOPMENT_CHECKPOINTS = 16
EXPECTED_DEVELOPMENT_ROWS = EXPECTED_DEVELOPMENT_STATES * EXPECTED_DEVELOPMENT_CHECKPOINTS
GENERATION_RESULT_SHA256 = "d8fad9caf6a48bdeb5e34ca897ac381c30f49e68304d9633f3f577a6b89705cf"
EXPANSION_FREEZE_SHA256 = "af1f66bc5d01d8bbb83974a9497da08e7d75510202c291777102370bfc979a29"
SUPERSEDING_ADJUDICATION_SHA256 = "e7c48f116db3b7ade11e280ce1222f11d5a999514fcf148e85b43d193051c0e7"
RESPONSIVE_FRACTIONS = {
    "libero_spatial": {"free_motion": 1.0, "approach": 1.0, "contact_grasp_release": 0.875, "transport_goal": 0.875},
    "libero_object": {"free_motion": 1.0, "approach": 1.0, "contact_grasp_release": 0.0, "transport_goal": 1.0},
    "libero_goal": {"free_motion": 1.0, "approach": 1.0, "contact_grasp_release": 0.75, "transport_goal": 1.0},
    "libero_10": {"free_motion": 1.0, "approach": 0.875, "contact_grasp_release": 0.875, "transport_goal": 1.0},
}


class CacheError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    ).hexdigest()


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _windows_path(value: str) -> Path:
    if value.startswith("/mnt/") and len(value) > 6:
        return Path(f"{value[5].upper()}:/{value[7:]}")
    return Path(value)


def cache_origin_and_offset(frame: int) -> tuple[int, int]:
    if frame < 0:
        raise CacheError("INVALID_FRAME", f"Negative frame {frame}")
    return (frame // CHUNK_SIZE) * CHUNK_SIZE, frame % CHUNK_SIZE


def state_is_selected(suite: str, phase: str) -> bool:
    try:
        return RESPONSIVE_FRACTIONS[suite][phase] > 0.0
    except KeyError as exc:
        raise CacheError("UNKNOWN_STATE_STRATUM", f"Unknown suite/phase {suite}/{phase}") from exc


def unique_increasing_phase_frames(length: int) -> tuple[int, ...]:
    frames = [_frame_for_fraction(length, float(fraction)) for fraction, _phase in STATE_PHASE_SPECS]
    for index in range(len(frames) - 2, -1, -1):
        if frames[index] >= frames[index + 1]:
            frames[index] = frames[index + 1] - 1
    if frames[0] < 0 or any(left >= right for left, right in zip(frames, frames[1:])):
        raise CacheError("PHASE_FRAME_COLLISION_UNRESOLVED", f"Cannot construct eight unique frames for length {length}")
    return tuple(frames)


def baseline_metrics(candidate: np.ndarray, expert: np.ndarray, action_range: np.ndarray, criticality: float) -> dict[str, float]:
    candidate = np.asarray(candidate, dtype=np.float64)
    expert = np.asarray(expert, dtype=np.float64)
    if candidate.shape != (7,) or expert.shape != (7,):
        raise CacheError("ACTION_SHAPE_MISMATCH", f"Expected two 7-D actions, got {candidate.shape}, {expert.shape}")
    delta = candidate - expert
    safe_range = np.maximum(np.asarray(action_range, dtype=np.float64), 1e-8)
    arm_mse = float(np.mean(np.square(delta[:6])))
    gripper_mse = float(np.square(delta[6]))
    normalized_mse = float(np.mean(np.square(delta / safe_range)))
    return {
        "raw_mse": float(np.mean(np.square(delta))),
        "raw_mae": float(np.mean(np.abs(delta))),
        "action_dimension_normalized_mse": normalized_mse,
        "arm_gripper_equal_weight_mse": 0.5 * arm_mse + 0.5 * gripper_mse,
        "phase_state_criticality_weighted_normalized_mse": float(criticality) * normalized_mse,
        "candidate_response_magnitude_l2": float(np.linalg.norm(candidate)),
        "expert_noop_response_magnitude_l2": float(np.linalg.norm(expert)),
    }


def _checkpoint_inventory(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    original_result = _read_json(Path(args.original_generation_result))
    new_result_path = Path(args.new_generation_result)
    if _sha256(new_result_path) != GENERATION_RESULT_SHA256:
        raise CacheError("GENERATION_RESULT_DRIFT", f"Unexpected hash for {new_result_path}")
    new_result = _read_json(new_result_path)
    if new_result.get("status") != "EPOCH10B_CHECKPOINT_LINEAGE_EXPANSION_COMPLETE":
        raise CacheError("LINEAGE_EXPANSION_INCOMPLETE", "New checkpoint generation is not complete")
    records: list[dict[str, Any]] = []
    for source, result in (("immutable_epoch10_panel", original_result), ("epoch10b_expansion", new_result)):
        for seed_row in result.get("seeds", []):
            seed = int(seed_row["seed"])
            for checkpoint in seed_row.get("checkpoints", []):
                step = int(Path(checkpoint["path"]).name.rsplit("_", 1)[-1])
                if step not in RETAINED_STAGES:
                    continue
                adapter = Path(checkpoint["path"]) / "adapter_model.safetensors"
                actual = _sha256(adapter)
                if actual != checkpoint["adapter_sha256"]:
                    raise CacheError("ADAPTER_HASH_MISMATCH", f"Adapter drift: {adapter}")
                records.append(
                    {
                        "policy_identity": checkpoint["policy_identity"],
                        "lineage_cluster": checkpoint["lineage_cluster"],
                        "seed": seed,
                        "optimizer_step": step,
                        "partition": "development" if seed in DEVELOPMENT_SEEDS else "heldout",
                        "path": str(Path(checkpoint["path"]).resolve()),
                        "adapter_sha256": actual,
                        "adapter_bytes": adapter.stat().st_size,
                        "source": source,
                    }
                )
    records.sort(key=lambda row: (row["seed"], row["optimizer_step"]))
    development = [row for row in records if row["partition"] == "development"]
    heldout = [row for row in records if row["partition"] == "heldout"]
    if len(development) != EXPECTED_DEVELOPMENT_CHECKPOINTS or len({row["seed"] for row in development}) != 8:
        raise CacheError("DEVELOPMENT_PANEL_INCOMPLETE", f"Unexpected development inventory: {len(development)}")
    if len(heldout) != 8 or len({row["seed"] for row in heldout}) != 4:
        raise CacheError("HELDOUT_PANEL_INCOMPLETE", f"Unexpected heldout inventory: {len(heldout)}")
    if len({row["adapter_sha256"] for row in records}) != len(records):
        raise CacheError("CHECKPOINT_IDENTITY_COLLISION", "Adapter hashes are not unique")
    return development, heldout


def _build_development_states(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import h5py

    prereg = _read_json(Path(args.assay_preregistration))
    tasks: list[dict[str, Any]] = []
    states: list[dict[str, Any]] = []
    for task in prereg["tasks"]:
        instruction = str(task["instruction"])
        hdf5_path = _windows_path(task["hdf5_path"]).resolve()
        if _sha256(hdf5_path) != task["hdf5_sha256"]:
            raise CacheError("RAW_DEMO_HASH_MISMATCH", f"Unexpected hash for {hdf5_path}")
        with h5py.File(hdf5_path, "r") as handle:
            for demo_id in DEVELOPMENT_RESERVED_DEMOS:
                demo_name = f"demo_{demo_id}"
                group = handle["data"][demo_name]
                actions = np.asarray(group["actions"], dtype=np.float64)
                states_raw = np.asarray(group["states"], dtype=np.float64)
                length = len(actions)
                unique_frames = unique_increasing_phase_frames(length)
                for phase_index, (fraction, phase) in enumerate(STATE_PHASE_SPECS):
                    if not state_is_selected(task["suite"], phase):
                        continue
                    frame = unique_frames[phase_index]
                    origin, offset = cache_origin_and_offset(frame)
                    state_id = f"{task['suite']}|task_{task['task_id']}|{demo_name}|frame_{frame}|{phase}"
                    origin_id = f"{task['suite']}|task_{task['task_id']}|{demo_name}|origin_{origin}"
                    expert = actions[frame]
                    states.append(
                        {
                            "state_id": state_id,
                            "suite": task["suite"],
                            "task_id": int(task["task_id"]),
                            "instruction": instruction,
                            "demo_name": demo_name,
                            "demo_cluster": f"{task['suite']}|task_{task['task_id']}|{demo_name}",
                            "episode_length": length,
                            "frame": frame,
                            "fraction": fraction,
                            "phase_index": phase_index,
                            "phase": phase,
                            "mechanics_responsive_fraction": RESPONSIVE_FRACTIONS[task["suite"]][phase],
                            "cache_origin_frame": origin,
                            "cache_offset": offset,
                            "origin_observation_id": origin_id,
                            "raw_state_sha256": hashlib.sha256(states_raw[frame].astype("<f8").tobytes()).hexdigest(),
                            "expert_action": [float(value) for value in expert],
                            "expert_action_sha256": hashlib.sha256(expert.astype("<f8").tobytes()).hexdigest(),
                            "fixed_rng_seed_rule": "uint32(first_8_hex(sha256(policy_identity|suite|task|demo|origin)))",
                        }
                    )
        tasks.append(
            {
                "suite": task["suite"],
                "task_id": int(task["task_id"]),
                "instruction": instruction,
                "windows_hdf5_path": str(hdf5_path),
                "wsl_hdf5_path": task["hdf5_path"],
                "hdf5_sha256": task["hdf5_sha256"],
                "bddl_file": task["bddl_file"],
                "bddl_sha256": task["bddl_sha256"],
            }
        )
    states.sort(key=lambda row: (row["suite"], row["task_id"], row["demo_name"], row["phase_index"]))
    if len(states) != EXPECTED_DEVELOPMENT_STATES or len({row["state_id"] for row in states}) != len(states):
        raise CacheError("DEVELOPMENT_STATE_MANIFEST_INVALID", f"Expected 240 unique states, found {len(states)}")
    return tasks, states


def _lineage_manifest(development: list[dict[str, Any]], heldout: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    all_rows = development + heldout
    lineages = []
    for seed in sorted(DEVELOPMENT_SEEDS | HOLDOUT_SEEDS):
        checkpoints = sorted((row for row in all_rows if row["seed"] == seed), key=lambda row: row["optimizer_step"])
        lineages.append(
            {
                "lineage_cluster": checkpoints[0]["lineage_cluster"],
                "seed": seed,
                "partition": "development" if seed in DEVELOPMENT_SEEDS else "heldout",
                "independent_training_run": True,
                "official_panel_checkpoints": checkpoints,
                "nested_stages_are_not_independent": True,
            }
        )
    return {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "EPOCH10B_CHECKPOINT_LINEAGE_TARGET_COMPLETE",
        "uncertainty_unit": "whole_training_seed",
        "whole_seed_lineage_count": len(lineages),
        "development_lineage_count": len(DEVELOPMENT_SEEDS),
        "heldout_lineage_count": len(HOLDOUT_SEEDS),
        "retained_stages": list(RETAINED_STAGES),
        "official_checkpoint_count": len(all_rows),
        "lineages": lineages,
        "source_generation_result": args.new_generation_result,
        "source_generation_result_sha256": _sha256(Path(args.new_generation_result)),
        "heldout_checkpoint_actions_queried": 0,
        "heldout_outcomes_opened": False,
    }


def _generation_log(lineage: dict[str, Any], generation: dict[str, Any]) -> str:
    max_host = max(float(row["peak_host_ram_percent"]) for row in generation["seeds"])
    max_vram = max(float(row["cuda"]["max_allocated_mb"]) for row in generation["seeds"])
    return f"""# Epoch 10B checkpoint generation log

Status: `EPOCH10B_CHECKPOINT_LINEAGE_TARGET_COMPLETE`

The outcome-blind expansion trained seeds `{', '.join(str(v) for v in generation['preflight']['new_seeds'])}` serially with the frozen standard rank-4 LoRA recipe. Steps 30 and 100 were retained for each seed. Existing Epoch 10 adapters were hash-verified and never rewritten.

- Independent whole-seed lineages: {lineage['whole_seed_lineage_count']} total ({lineage['development_lineage_count']} development, {lineage['heldout_lineage_count']} held out).
- Official nested checkpoints: {lineage['official_checkpoint_count']} (two per lineage; nested stages are repeated measures).
- New adapter bundles: {generation['new_checkpoint_count']}; unique adapter hashes: {len({row['adapter_sha256'] for seed in generation['seeds'] for row in seed['checkpoints']})}.
- All fresh disk reloads passed: `{generation['all_disk_reloads_passed']}`.
- Expansion wall time: {generation['elapsed_seconds']} seconds.
- Peak sampled host RAM: {max_host:.3f}%.
- Peak CUDA allocated memory: {max_vram:.3f} MiB.
- Checkpoint actions queried during training: `{generation['checkpoint_actions_queried']}`.
- Comparative simulator outcomes opened: `{generation['comparative_simulator_outcomes_opened']}`.

The development/holdout assignment was frozen by whole seed before training. Synthetic action noise, renamed copies, interpolation, and outcome-selected snapshots were prohibited. Held-out actions, validation outcomes, and confirmation outcomes remain sealed.
"""


def _observation_origins(tasks: list[dict[str, Any]], states: list[dict[str, Any]]) -> list[dict[str, Any]]:
    task_map = {(row["suite"], int(row["task_id"])): row for row in tasks}
    origins: dict[str, dict[str, Any]] = {}
    for state in states:
        origin_id = state["origin_observation_id"]
        if origin_id in origins:
            continue
        task = task_map[(state["suite"], int(state["task_id"]))]
        filename = f"{hashlib.sha256(origin_id.encode('utf-8')).hexdigest()}.npz"
        origins[origin_id] = {
            "origin_observation_id": origin_id,
            "suite": state["suite"],
            "task_id": int(state["task_id"]),
            "instruction": state["instruction"],
            "demo_name": state["demo_name"],
            "cache_origin_frame": int(state["cache_origin_frame"]),
            "episode_length": int(state["episode_length"]),
            "windows_hdf5_path": task["windows_hdf5_path"],
            "wsl_hdf5_path": task["wsl_hdf5_path"],
            "hdf5_sha256": task["hdf5_sha256"],
            "bddl_file": task["bddl_file"],
            "bddl_sha256": task["bddl_sha256"],
            "cache_filename": filename,
            "camera_size": 256,
        }
    return sorted(origins.values(), key=lambda row: row["origin_observation_id"])


def _freeze_payload(args: argparse.Namespace) -> dict[str, Any]:
    if _sha256(Path(args.expansion_freeze)) != EXPANSION_FREEZE_SHA256:
        raise CacheError("EXPANSION_FREEZE_DRIFT", args.expansion_freeze)
    if _sha256(Path(args.superseding_adjudication)) != SUPERSEDING_ADJUDICATION_SHA256:
        raise CacheError("SUPERSEDING_ADJUDICATION_DRIFT", args.superseding_adjudication)
    development, heldout = _checkpoint_inventory(args)
    tasks, states = _build_development_states(args)
    observation_origins = _observation_origins(tasks, states)
    lineage = _lineage_manifest(development, heldout, args)
    generation = _read_json(Path(args.new_generation_result))
    _write_json(Path(args.lineage_manifest), lineage)
    _write_text(Path(args.generation_log), _generation_log(lineage, generation))
    payload = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "FROZEN_BEFORE_DEVELOPMENT_CHECKPOINT_ACTION_QUERY",
        "source_commit": _git_head(),
        "authority": {
            "original_prompt_sha256": ORIGINAL_PROMPT_SHA256,
            "continuation_prompt_sha256": CONTINUATION_PROMPT_SHA256,
            "superseding_adjudication_sha256": SUPERSEDING_ADJUDICATION_SHA256,
            "checkpoint_expansion_freeze_sha256": EXPANSION_FREEZE_SHA256,
            "checkpoint_generation_result_sha256": GENERATION_RESULT_SHA256,
        },
        "selected_assay": {
            "constructor": "prefix_reconstructed_fresh_controller",
            "horizon": SELECTED_HORIZON,
            "endpoint": SELECTED_ENDPOINT,
            "signed_score": "candidate bounded recovery cost minus nominal bounded recovery cost",
            "aggregation": "state mean within task, equal-task macro-average; lower is better",
        },
        "deployment_equivalent_action_unit": {
            "kind": "one_7d_environment_action",
            "chunk_size": CHUNK_SIZE,
            "cache_origin_rule": "floor(frame / 50) * 50",
            "cache_offset_rule": "frame modulo 50",
            "queue_reconstruction": "reset queue at origin; fixed-RNG generation; pop through registered offset",
            "normalization": "official SmolVLA preprocessor and postprocessor",
            "environment_conversion": "official LeRobot LIBERO environment postprocessor is identity for SmolVLA",
            "execution_clipping": "componentwise [-1, 1], with raw and executed values retained",
        },
        "development_checkpoint_count": len(development),
        "development_lineage_count": len({row["seed"] for row in development}),
        "development_checkpoints": development,
        "heldout_checkpoint_count": len(heldout),
        "heldout_checkpoint_actions_queried": 0,
        "development_tasks": tasks,
        "development_demo_ids": list(DEVELOPMENT_RESERVED_DEMOS),
        "heldout_demo_ids_remain_sealed": list(HELDOUT_RESERVED_DEMOS),
        "development_state_selector": {
            "rule": "retain suite-phase strata with nonzero H=4 mechanics responsiveness across demos 8-11",
            "mechanics_responsive_fractions": RESPONSIVE_FRACTIONS,
            "excluded_strata": ["libero_object/contact_grasp_release"],
            "state_count": len(states),
            "checkpoint_blind": True,
        },
        "development_states": states,
        "development_observation_origins": observation_origins,
        "development_observation_origin_count": len(observation_origins),
        "observation_reconstruction": {
            "constructor": "fresh 256x256 LIBERO environment per cache origin",
            "prefix": "replay expert actions from frame 0 through origin-1, then exact-set registered origin physics state",
            "model_resident_during_render": False,
            "windows_cache_root": "C:\\assets\\cache\\epoch10b_stage0_observations",
            "wsl_cache_root": "/mnt/c/assets/cache/epoch10b_stage0_observations",
        },
        "development_closed_loop": {
            "tasks": [row["suite"] for row in tasks],
            "common_reset_seeds": list(DEVELOPMENT_RESET_SEEDS),
            "episodes_per_task_checkpoint": len(DEVELOPMENT_RESET_SEEDS),
            "success_semantics": "official LeRobot/LIBERO native is_success; every timeout and invalid episode retained",
        },
        "equal_input_baselines": {
            "raw_mse": "mean squared error over seven environment-action dimensions",
            "raw_mae": "mean absolute error over seven environment-action dimensions",
            "action_dimension_normalized_mse": "per-dimension squared error divided by frozen training action range squared",
            "arm_gripper_equal_weight_mse": "0.5 * six-arm-dimension MSE + 0.5 * gripper squared error",
            "phase_state_criticality": "normalized MSE weighted by mechanics-only suite-phase responsiveness fraction",
            "response_magnitude_control": "candidate action L2 magnitude without expert label",
            "unpaired_icae": "deterministic within-task cyclic action-to-state permutation fixed before branch execution",
            "state_shuffled_icae": "deterministic global state-score permutation fixed before branch execution",
            "ci_mse": {
                "status": "NOT_IMPLEMENTED_NO_PROXY",
                "reason": "No official code/configuration was released or locally available at the 2026-07-22 freeze; the frozen protocol forbids an unofficial proxy.",
                "paper": "https://arxiv.org/abs/2606.29898",
                "project": "https://ci-mse.github.io/",
            },
        },
        "stage0_gate": {
            "performance": "distinguishable development performance or at least three bands",
            "minimum_icae_concordance": 0.60,
            "minimum_gain_over_normalized_mse": 0.08,
            "minimum_bootstrap_probability_gain_positive": 0.90,
            "strong_baseline_non_domination_required": True,
            "negative_controls_must_not_reproduce_gain": True,
            "maximum_icae_step_fraction_of_exhaustive_rollout": 0.20,
            "bootstrap_replicates": 10000,
            "bootstrap_seed": 20260722,
            "bootstrap_units": ["task", "whole_seed_lineage", "whole_demo_cluster"],
        },
        "leakage_boundaries": {
            "development_checkpoint_actions_queried": 0,
            "development_simulator_scores_opened": False,
            "development_success_labels_opened": False,
            "heldout_checkpoint_actions_queried": 0,
            "heldout_simulator_scores_opened": False,
            "heldout_success_labels_opened": False,
            "confirmation_results_opened": False,
        },
    }
    payload["frozen_payload_sha256"] = _canonical_hash(payload)
    return payload


def _observation_asset_rows(path: Path, cache_root: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = row["origin_observation_id"]
            if key in rows:
                raise CacheError("DUPLICATE_OBSERVATION_KEY", f"Duplicate {key} at line {line_number}")
            asset = cache_root / row["asset_filename"]
            if not asset.is_file() or _sha256(asset) != row["asset_sha256"]:
                raise CacheError("OBSERVATION_ASSET_DRIFT", str(asset))
            rows[key] = row
    return rows


def _run_observations(args: argparse.Namespace) -> dict[str, Any]:
    import h5py

    from scripts.run_epoch10b_fresh_controller_assay import (
        _make_env,
        _prefix_reconstruct,
        _sim_state,
        closing_environment,
        resource_sample,
        seed_runtime,
    )

    freeze = _read_json(Path(args.freeze_json))
    origins = freeze["development_observation_origins"]
    raw_path = Path(args.observation_jsonl)
    cache_root = Path(args.observation_cache_root).resolve()
    cache_root.mkdir(parents=True, exist_ok=True)
    completed = _observation_asset_rows(raw_path, cache_root)
    allowed = {row["origin_observation_id"] for row in origins}
    extras = set(completed) - allowed
    if extras:
        raise CacheError("EXTRA_OBSERVATION_KEYS", f"Unexpected observation keys: {sorted(extras)[:3]}")
    started = time.monotonic()
    peak_host_ram = 0.0
    peak_wsl_memory = 0
    peak_wsl_swap = 0
    for origin in origins:
        key = origin["origin_observation_id"]
        if key in completed:
            continue
        hdf5_path = Path(origin["wsl_hdf5_path"] if os.name != "nt" else origin["windows_hdf5_path"])
        if _sha256(hdf5_path) != origin["hdf5_sha256"]:
            raise CacheError("RAW_DEMO_HASH_MISMATCH", str(hdf5_path))
        with h5py.File(hdf5_path, "r") as handle:
            group = handle["data"][origin["demo_name"]]
            states = np.asarray(group["states"], dtype=np.float64)
            actions = np.asarray(group["actions"], dtype=np.float64)
        frame = int(origin["cache_origin_frame"])
        registered_seed = int(hashlib.sha256(f"epoch10b-observation|{key}".encode("utf-8")).hexdigest()[:8], 16)
        before = resource_sample()
        with closing_environment(
            lambda: _make_env(Path(origin["bddl_file"]), int(origin["camera_size"]))
        ) as (env, cleanup):
            seed_runtime(registered_seed)
            env.seed(registered_seed)
            env.reset()
            prefix = _prefix_reconstruct(env, states, actions, frame)
            observation = env.set_init_state(states[frame])
            restored = _sim_state(env)
            restore_l2 = float(np.linalg.norm(restored - states[frame]))
            required = (
                "agentview_image",
                "robot0_eye_in_hand_image",
                "robot0_eef_pos",
                "robot0_eef_quat",
                "robot0_gripper_qpos",
            )
            missing = [name for name in required if name not in observation]
            if missing:
                raise CacheError("OBSERVATION_SCHEMA_MISMATCH", f"Missing {missing} for {key}")
            if restore_l2 > 1e-8:
                raise CacheError("OBSERVATION_RESTORE_MISMATCH", f"Restore L2 {restore_l2} for {key}")
            arrays = {name: np.asarray(observation[name]) for name in required}
        if cleanup.get("close_called") is not True:
            raise CacheError("ENVIRONMENT_CLOSE_FAILED", key)
        destination = cache_root / origin["cache_filename"]
        temporary = destination.with_name(f"{destination.name}.tmp_{os.getpid()}")
        with temporary.open("wb") as handle:
            np.savez_compressed(handle, **arrays)
        temporary.replace(destination)
        after = resource_sample()
        for sample in (before, after):
            peak_host_ram = max(peak_host_ram, float(sample.get("host_ram_percent") or 0.0))
            used_memory = max(
                0,
                int(sample.get("wsl_mem_total_bytes") or 0) - int(sample.get("wsl_mem_available_bytes") or 0),
            )
            peak_wsl_memory = max(peak_wsl_memory, used_memory)
            peak_wsl_swap = max(peak_wsl_swap, int(sample.get("wsl_swap_used_bytes") or 0))
        row = {
            "schema_version": SCHEMA_VERSION,
            "campaign": CAMPAIGN,
            "origin_observation_id": key,
            "suite": origin["suite"],
            "task_id": origin["task_id"],
            "demo_name": origin["demo_name"],
            "cache_origin_frame": frame,
            "registered_seed": registered_seed,
            "prefix": prefix,
            "restore_l2": restore_l2,
            "camera_size": origin["camera_size"],
            "asset_filename": origin["cache_filename"],
            "runtime_asset_path": str(destination),
            "asset_sha256": _sha256(destination),
            "asset_bytes": destination.stat().st_size,
            "environment_close_called": True,
            "model_resident": False,
            "checkpoint_action_queried": False,
            "simulator_outcome_used": False,
        }
        _append_jsonl(raw_path, [row])
        completed[key] = row
        print(f"[epoch10b-observation-cache] {len(completed)}/{len(origins)} {key}", flush=True)
    if len(completed) != len(origins):
        raise CacheError("OBSERVATION_CACHE_INCOMPLETE", f"Expected {len(origins)}, found {len(completed)}")
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "EPOCH10B_DEVELOPMENT_OBSERVATION_CACHE_COMPLETE",
        "source_commit": _git_head(),
        "freeze_sha256": _sha256(Path(args.freeze_json)),
        "row_count": len(completed),
        "unique_key_count": len(completed),
        "raw_path": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "asset_root": str(cache_root),
        "asset_bytes": sum(int(row["asset_bytes"]) for row in completed.values()),
        "all_restore_l2_within_1e_8": all(float(row["restore_l2"]) <= 1e-8 for row in completed.values()),
        "all_environments_closed": all(bool(row["environment_close_called"]) for row in completed.values()),
        "peak_host_ram_percent": peak_host_ram,
        "peak_wsl_memory_used_bytes": peak_wsl_memory,
        "peak_wsl_swap_used_bytes": peak_wsl_swap,
        "checkpoint_actions_queried": 0,
        "simulator_outcomes_used": False,
        "elapsed_seconds": round(time.monotonic() - started, 3),
    }
    _write_json(Path(args.observation_manifest), manifest)
    return manifest


def _load_policy(args: argparse.Namespace, checkpoint: dict[str, Any]) -> dict[str, Any]:
    import torch
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from peft import PeftConfig, PeftModel

    if not torch.cuda.is_available():
        raise CacheError("CPU_FALLBACK_BUG", "CUDA unavailable for checkpoint action caching")
    base_path = Path(args.checkpoint_path)
    cfg = PreTrainedConfig.from_pretrained(base_path, local_files_only=True, cache_dir=args.hf_home)
    cfg.device = "cuda"
    cfg.load_vlm_weights = True
    cfg.compile_model = False
    cfg.push_to_hub = False
    cfg.vlm_model_name = str(Path(args.vlm_root))
    cfg.chunk_size = CHUNK_SIZE
    base = SmolVLAPolicy.from_pretrained(
        base_path,
        config=cfg,
        local_files_only=True,
        cache_dir=args.hf_home,
        token=False,
        strict=False,
    )
    peft_config = PeftConfig.from_pretrained(checkpoint["path"])
    policy = PeftModel.from_pretrained(
        base,
        checkpoint["path"],
        config=peft_config,
        is_trainable=False,
        local_files_only=True,
    )
    policy.to("cuda").eval()
    preprocessor, postprocessor = make_pre_post_processors(
        cfg,
        pretrained_path=str(base_path),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(Path(args.vlm_root))},
            "device_processor": {"device": "cuda"},
        },
        postprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    return {"policy": policy, "base": base, "preprocessor": preprocessor, "postprocessor": postprocessor}


def _rng_seed(policy_identity: str, state: dict[str, Any]) -> int:
    key = f"{policy_identity}|{state['suite']}|{state['task_id']}|{state['demo_name']}|{state['cache_origin_frame']}"
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:8], 16)


def _observation_batch(asset: Path, instruction: str, preprocessor: Any) -> tuple[dict[str, Any], str]:
    from lerobot.envs.utils import preprocess_observation
    from lerobot.processor.env_processor import LiberoProcessorStep

    with np.load(asset, allow_pickle=False) as values:
        observation = {
            "pixels": {
                "image": np.asarray(values["agentview_image"], dtype=np.uint8)[None, ...],
                "image2": np.asarray(values["robot0_eye_in_hand_image"], dtype=np.uint8)[None, ...],
            },
            "robot_state": {
                "eef": {
                    "pos": np.asarray(values["robot0_eef_pos"], dtype=np.float32)[None, ...],
                    "quat": np.asarray(values["robot0_eef_quat"], dtype=np.float32)[None, ...],
                },
                "gripper": {
                    "qpos": np.asarray(values["robot0_gripper_qpos"], dtype=np.float32)[None, ...],
                },
            },
        }
    lerobot_observation = preprocess_observation(observation)
    lerobot_observation["task"] = [instruction]
    deployment_observation = LiberoProcessorStep().observation(lerobot_observation)
    return preprocessor(deployment_observation), _sha256(asset)


def _append_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _existing_rows(path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            key = (row["policy_identity"], row["state_id"])
            if key in rows:
                raise CacheError("DUPLICATE_CACHE_KEY", f"Duplicate {key} at line {line_number}")
            rows[key] = row
    return rows


def _cache_checkpoint(
    args: argparse.Namespace,
    observations: dict[str, dict[str, Any]],
    observation_root: Path,
    checkpoint: dict[str, Any],
    states: list[dict[str, Any]],
    completed: dict[tuple[str, str], dict[str, Any]],
    action_range: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    import psutil
    import torch

    missing = [state for state in states if (checkpoint["policy_identity"], state["state_id"]) not in completed]
    if not missing:
        return [], {"status": "COMPLETE_REUSED", "repeatability_max_abs": 0.0}
    loaded = _load_policy(args, checkpoint)
    policy = loaded["policy"]
    preprocessor = loaded["preprocessor"]
    postprocessor = loaded["postprocessor"]
    grouped: dict[tuple[str, int, str, int], list[dict[str, Any]]] = defaultdict(list)
    for state in missing:
        grouped[(state["suite"], state["task_id"], state["demo_name"], state["cache_origin_frame"])].append(state)
    output: list[dict[str, Any]] = []
    repeatability_max_abs = 0.0
    first_group = True
    peak_host_ram = float(psutil.virtual_memory().percent)
    torch.cuda.reset_peak_memory_stats()
    try:
        for group_states in grouped.values():
            host_ram = float(psutil.virtual_memory().percent)
            peak_host_ram = max(peak_host_ram, host_ram)
            if host_ram >= 90.0:
                raise CacheError("HOST_RAM_HARD_STOP", f"Host RAM reached {host_ram:.2f}%")
            exemplar = group_states[0]
            observation_row = observations[exemplar["origin_observation_id"]]
            observation_asset = observation_root / observation_row["asset_filename"]
            batch, observation_asset_hash = _observation_batch(
                observation_asset,
                exemplar["instruction"],
                preprocessor,
            )
            seed = _rng_seed(checkpoint["policy_identity"], exemplar)
            wanted = {int(state["cache_offset"]): state for state in group_states}

            def generate() -> dict[int, np.ndarray]:
                torch.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)
                if hasattr(policy, "reset"):
                    policy.reset()
                actions: dict[int, np.ndarray] = {}
                with torch.inference_mode():
                    for offset in range(max(wanted) + 1):
                        selected = policy.select_action(batch)
                        if offset in wanted:
                            actions[offset] = _postprocess_action(selected, postprocessor).astype(np.float64)
                return actions

            actions = generate()
            if first_group:
                duplicate = generate()
                repeatability_max_abs = max(
                    float(np.max(np.abs(actions[offset] - duplicate[offset]))) for offset in actions
                )
                if repeatability_max_abs > 1e-8:
                    raise CacheError("ACTION_CACHE_NONDETERMINISTIC", f"Repeatability max abs {repeatability_max_abs}")
                first_group = False
            for offset, state in wanted.items():
                raw_action = actions[offset]
                if raw_action.shape != (7,) or not np.isfinite(raw_action).all():
                    raise CacheError(
                        "INVALID_CHECKPOINT_ACTION",
                        f"Invalid action for {checkpoint['policy_identity']} at {state['state_id']}: {raw_action}",
                    )
                executed = np.clip(raw_action, -1.0, 1.0)
                expert = np.asarray(state["expert_action"], dtype=np.float64)
                metrics = baseline_metrics(
                    executed,
                    expert,
                    action_range,
                    float(state["mechanics_responsive_fraction"]),
                )
                output.append(
                    {
                        "schema_version": SCHEMA_VERSION,
                        "campaign": CAMPAIGN,
                        "partition": "development",
                        "policy_identity": checkpoint["policy_identity"],
                        "lineage_cluster": checkpoint["lineage_cluster"],
                        "seed": checkpoint["seed"],
                        "optimizer_step": checkpoint["optimizer_step"],
                        "adapter_sha256": checkpoint["adapter_sha256"],
                        "state_id": state["state_id"],
                        "suite": state["suite"],
                        "task_id": state["task_id"],
                        "demo_cluster": state["demo_cluster"],
                        "demo_name": state["demo_name"],
                        "phase": state["phase"],
                        "frame": state["frame"],
                        "cache_origin_frame": state["cache_origin_frame"],
                        "cache_offset": offset,
                        "fixed_rng_seed": seed,
                        "origin_observation_asset_sha256": observation_asset_hash,
                        "raw_action": [float(value) for value in raw_action],
                        "executed_action": [float(value) for value in executed],
                        "executed_action_sha256": hashlib.sha256(executed.astype("<f8").tobytes()).hexdigest(),
                        "expert_action": state["expert_action"],
                        "expert_action_sha256": state["expert_action_sha256"],
                        "clipped": bool(np.any(raw_action != executed)),
                        "finite": bool(np.isfinite(raw_action).all()),
                        "baseline_metrics": metrics,
                        "simulator_outcome_read": False,
                        "closed_loop_success_label_read": False,
                    }
                )
        return output, {
            "status": "CACHE_COMPLETE",
            "new_rows": len(output),
            "origin_group_count": len(grouped),
            "repeatability_max_abs": repeatability_max_abs,
            "peak_host_ram_percent": peak_host_ram,
            "peak_cuda_allocated_mib": round(float(torch.cuda.max_memory_allocated()) / (1024**2), 3),
        }
    finally:
        del policy, loaded
        torch.cuda.empty_cache()


def _run(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    freeze_path = Path(args.freeze_json)
    freeze = _read_json(freeze_path)
    if freeze.get("status") != "FROZEN_BEFORE_DEVELOPMENT_CHECKPOINT_ACTION_QUERY":
        raise CacheError("ACTION_CACHE_NOT_FROZEN", str(freeze_path))
    development = freeze["development_checkpoints"]
    states = freeze["development_states"]
    raw_path = Path(args.raw_jsonl)
    completed = _existing_rows(raw_path)
    allowed_keys = {(row["policy_identity"], state["state_id"]) for row in development for state in states}
    extras = set(completed) - allowed_keys
    if extras:
        raise CacheError("EXTRA_CACHE_KEYS", f"Unexpected completed keys: {sorted(extras)[:3]}")
    dataset_root = Path(args.dataset_root)
    observation_manifest = _read_json(Path(args.observation_manifest))
    if observation_manifest.get("status") != "EPOCH10B_DEVELOPMENT_OBSERVATION_CACHE_COMPLETE":
        raise CacheError("OBSERVATION_CACHE_INCOMPLETE", args.observation_manifest)
    observation_root = Path(args.observation_cache_root).resolve()
    observations = _observation_asset_rows(Path(args.observation_jsonl), observation_root)
    expected_observations = {row["origin_observation_id"] for row in freeze["development_observation_origins"]}
    if set(observations) != expected_observations:
        raise CacheError(
            "OBSERVATION_CACHE_KEY_MISMATCH",
            f"Expected {len(expected_observations)} observations, found {len(observations)}",
        )
    stats = _read_json(dataset_root / "meta" / "stats.json")
    action_range = np.asarray(stats["action"]["max"], dtype=np.float64) - np.asarray(
        stats["action"]["min"], dtype=np.float64
    )
    started = time.monotonic()
    checkpoint_audits = []
    for checkpoint in development:
        print(f"[epoch10b-action-cache] {checkpoint['policy_identity']}", flush=True)
        rows, audit = _cache_checkpoint(
            args,
            observations,
            observation_root,
            checkpoint,
            states,
            completed,
            action_range,
        )
        if rows:
            _append_jsonl(raw_path, rows)
            for row in rows:
                completed[(row["policy_identity"], row["state_id"])] = row
        checkpoint_audits.append({"policy_identity": checkpoint["policy_identity"], **audit})
    if len(completed) != EXPECTED_DEVELOPMENT_ROWS:
        raise CacheError("ACTION_CACHE_INCOMPLETE", f"Expected {EXPECTED_DEVELOPMENT_ROWS}, found {len(completed)}")
    values = list(completed.values())
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "campaign": CAMPAIGN,
        "status": "EPOCH10B_DEVELOPMENT_ACTION_CACHE_COMPLETE",
        "source_commit": _git_head(),
        "freeze_path": str(freeze_path),
        "freeze_sha256": _sha256(freeze_path),
        "raw_path": str(raw_path),
        "raw_sha256": _sha256(raw_path),
        "row_count": len(values),
        "expected_row_count": EXPECTED_DEVELOPMENT_ROWS,
        "unique_key_count": len(completed),
        "checkpoint_count": len({row["policy_identity"] for row in values}),
        "lineage_count": len({row["lineage_cluster"] for row in values}),
        "state_count": len({row["state_id"] for row in values}),
        "finite_rows": sum(bool(row["finite"]) for row in values),
        "clipped_rows": sum(bool(row["clipped"]) for row in values),
        "checkpoint_audits": checkpoint_audits,
        "maximum_repeatability_abs": max(float(row["repeatability_max_abs"]) for row in checkpoint_audits),
        "peak_host_ram_percent": max(float(row.get("peak_host_ram_percent", 0.0)) for row in checkpoint_audits),
        "peak_cuda_allocated_mib": max(float(row.get("peak_cuda_allocated_mib", 0.0)) for row in checkpoint_audits),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "simulator_outcomes_read": False,
        "development_success_labels_opened": False,
        "heldout_checkpoint_actions_queried": 0,
        "heldout_outcomes_opened": False,
        "wsl_simulator_used": False,
        "model_unloaded_before_simulator": True,
        "torch_cuda_available": bool(torch.cuda.is_available()),
    }
    _write_json(Path(args.manifest_json), manifest)
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "preflight", "observations", "run"), default="preflight")
    parser.add_argument("--checkpoint-path", default="C:/assets/checkpoints/smolvla_libero")
    parser.add_argument("--dataset-root", default="C:/assets/datasets/lerobot_libero")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--vlm-root", default="C:/assets/hf_home/HuggingFaceTB/SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--original-generation-result", default="reports/epoch10_checkpoint_generation_result.json")
    parser.add_argument("--new-generation-result", default="reports/epoch10b_checkpoint_generation_result.json")
    parser.add_argument("--expansion-freeze", default="reports/epoch10b_checkpoint_expansion_freeze.json")
    parser.add_argument("--superseding-adjudication", default="reports/epoch10b_erratum_superseding_adjudication.json")
    parser.add_argument("--assay-preregistration", default="reports/epoch10b_assay_preregistration.json")
    parser.add_argument("--lineage-manifest", default="reports/epoch10b_checkpoint_lineage_manifest.json")
    parser.add_argument("--generation-log", default="reports/epoch10b_checkpoint_generation_log.md")
    parser.add_argument("--freeze-json", default="reports/epoch10b_action_cache_freeze.json")
    parser.add_argument("--raw-jsonl", default="runs/epoch10b_stage0_action_cache/development_actions.jsonl")
    parser.add_argument("--manifest-json", default="reports/epoch10b_action_cache_manifest.json")
    parser.add_argument(
        "--observation-cache-root",
        default=(
            "C:/assets/cache/epoch10b_stage0_observations"
            if os.name == "nt"
            else "/mnt/c/assets/cache/epoch10b_stage0_observations"
        ),
    )
    parser.add_argument(
        "--observation-jsonl",
        default="runs/epoch10b_stage0_action_cache/development_observations.jsonl",
    )
    parser.add_argument(
        "--observation-manifest",
        default="reports/epoch10b_observation_cache_manifest.json",
    )
    parser.add_argument("--video-backend", default="pyav")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    os.environ["HF_HOME"] = str(Path(args.hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    try:
        if args.mode == "freeze":
            payload = _freeze_payload(args)
            _write_json(Path(args.freeze_json), payload)
        elif args.mode == "observations":
            payload = _run_observations(args)
        elif args.mode == "preflight":
            development, heldout = _checkpoint_inventory(args)
            tasks, states = _build_development_states(args)
            payload = {
                "status": "EPOCH10B_ACTION_CACHE_PREFLIGHT_PASS",
                "development_checkpoints": len(development),
                "heldout_checkpoints_sealed": len(heldout),
                "tasks": len(tasks),
                "development_states": len(states),
                "expected_rows": len(development) * len(states),
            }
        else:
            payload = _run(args)
        print(json.dumps({"status": payload["status"]}, sort_keys=True))
        return 0
    except CacheError as exc:
        print(json.dumps({"status": exc.code, "error": str(exc), "traceback": traceback.format_exc()}, sort_keys=True))
        return 2
    except Exception as exc:  # pragma: no cover - integration failure path
        print(json.dumps({"status": "UNEXPECTED_IMPLEMENTATION_FAILURE", "error": str(exc), "traceback": traceback.format_exc()}, sort_keys=True))
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
