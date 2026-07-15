"""Run the frozen KITE-VLA Stage 0A realization and policy-path audit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import threading
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_famr_vla_stage0 import (  # noqa: E402
    _clone_batch,
    _hash_base_parameters,
    _load_policy_and_processors,
    _loss,
    _preprocess,
    _raw_sample,
    _set_offline_environment,
)
from scripts.run_pcav_vla_stage0 import _postprocess_chunk  # noqa: E402
from tca_map.smolvla.kite_vla import (  # noqa: E402
    ACTION_DIM,
    ARM_DIM,
    HORIZONS,
    PROPOSAL_HASH,
    RIDGE_COEFFICIENT,
    STD_FLOOR,
    Stage0ADecisionInputs,
    canonical_json_sha256,
    classify_stage0a,
    cumulative_arm_command,
    differentiable_mean_std_unnormalize,
    fit_realization_operator,
    frame_key,
    huber_loss,
    json_default,
    predict_realization,
    realization_metrics,
    realization_row_key,
    state_displacement,
    torch_realization_normalized,
    validate_manifest,
)


SEED = 20262300
FLOW_TIME = 0.5
MODEL_ROWS_PER_TASK_SPLIT_HORIZON = 8
MAX_MODEL_ROWS = 128
HEADROOM_RELATIVE_GATE = 0.25
HEADROOM_ABSOLUTE_GATE = 0.02
OPERATOR_IMPROVEMENT_GATE = 0.50
GRADIENT_RATIO_MAX = 100.0
PROPOSAL_FILE = REPO_ROOT / "reports" / "kite_vla" / "researcher_proposal.md"
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "kite_vla" / "proposal_hash.txt"

TASK_SOURCES = (
    (
        "libero_spatial",
        "libero_spatial/task_3",
        "libero_spatial/pick_up_the_black_bowl_next_to_the_cookie_box_and_place_it_on_the_plate_demo.hdf5",
    ),
    (
        "libero_object",
        "libero_object/task_3",
        "libero_object/pick_up_the_chocolate_pudding_and_place_it_in_the_basket_demo.hdf5",
    ),
    (
        "libero_goal",
        "libero_goal/task_5",
        "libero_goal/put_the_bowl_on_top_of_the_cabinet_demo.hdf5",
    ),
    (
        "libero_10",
        "libero_10/task_5",
        "libero_10/LIVING_ROOM_SCENE2_put_both_the_alphabet_soup_and_the_tomato_sauce_in_the_basket_demo.hdf5",
    ),
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(
            dict(payload),
            indent=2,
            sort_keys=True,
            allow_nan=False,
            default=json_default,
        )
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    temporary.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest().upper()


def _edge_hash(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as handle:
        digest.update(handle.read(1024 * 1024))
        if size > 1024 * 1024:
            handle.seek(max(0, size - 1024 * 1024))
            digest.update(handle.read(1024 * 1024))
    digest.update(str(size).encode("ascii"))
    return digest.hexdigest().upper()


def _registry_hash() -> str:
    for token in PROPOSAL_HASH_FILE.read_text(encoding="utf-8").split():
        candidate = token.upper()
        if len(candidate) == 64 and all(char in "0123456789ABCDEF" for char in candidate):
            return candidate
    return ""


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    report = Path(args.report_root)
    run = Path(args.run_root)
    return {
        "report": report,
        "run": run,
        "feature_dir": run / "features",
        "adapter_dir": run / "identity_adapter",
        "data_root": Path(args.data_root),
        "checkpoint": Path(args.checkpoint),
        "pid": report / "stage_0a_pid.txt",
        "heartbeat": report / "stage_0a_heartbeat.json",
        "status": report / "stage_0a_status.json",
        "serializer_preflight": report / "stage_0a_serializer_preflight.json",
        "preflight": report / "stage_0a_preflight.json",
        "manifest": report / "stage_0a_manifest.json",
        "partial": report / "stage_0a_partial.json",
        "result_json": report / "stage_0a_result.json",
        "result_md": report / "stage_0a_result.md",
        "validation": report / "stage_0a_validation.json",
        "blocker": report / "stage_0a_implementation_blocker.json",
    }


def _serializer_preflight(path: Path) -> dict[str, Any]:
    fixture = {
        "method": "KITE-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "operator": {
            "command_mean": np.arange(6, dtype=np.float32),
            "command_std": np.ones(6, dtype=np.float64),
            "coefficient": np.eye(6, dtype=np.float64),
            "rank": np.int64(6),
        },
        "normalization": {"mean": np.zeros(7, dtype=np.float32), "std": np.ones(7, dtype=np.float32)},
    }
    fixture_hash = canonical_json_sha256(fixture)
    _write_json(path, {"fixture": fixture, "fixture_hash": fixture_hash, "written_at": _utc_now()})
    parsed = _read_json(path)
    reproduced = canonical_json_sha256(parsed["fixture"])
    passed = reproduced == fixture_hash and parsed["fixture_hash"] == fixture_hash
    result = {
        **parsed,
        "parsed": True,
        "reproduced_hash": reproduced,
        "passed": passed,
    }
    _write_json(path, result)
    if not passed:
        raise RuntimeError("KITE serializer preflight hash did not reproduce")
    return result


def _problem_language(data: Any) -> str:
    raw = data.attrs.get("problem_info", "")
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    try:
        parsed = json.loads(str(raw))
        return str(parsed.get("language_instruction") or parsed.get("language") or "")
    except json.JSONDecodeError:
        return str(raw)


def _evenly_spaced(rows: Sequence[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if len(rows) <= count:
        return list(rows)
    indices = np.floor(np.linspace(0, len(rows) - 1, count) + 0.5).astype(int).tolist()
    if len(indices) != len(set(indices)):
        raise RuntimeError("deterministic KITE sampler produced duplicate indices")
    return [rows[index] for index in indices]


def _build_manifest(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import h5py

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for task_index, (suite, task_identity, relative) in enumerate(TASK_SOURCES):
        source = data_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = _edge_hash(source)
        with h5py.File(source, "r") as handle:
            data = handle["data"]
            language = _problem_language(data)
            demos: list[dict[str, Any]] = []
            for demo_id in range(10):
                demo_key = f"demo_{demo_id}"
                if demo_key not in data:
                    raise KeyError(f"missing {demo_key} in {source}")
                demo = data[demo_key]
                actions = np.asarray(demo["actions"], dtype=np.float64)
                observations = demo["obs"]
                required = ("agentview_rgb", "eye_in_hand_rgb", "ee_states", "gripper_states")
                if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
                    raise ValueError(f"invalid action array {actions.shape} in {source}:{demo_key}")
                if any(name not in observations for name in required):
                    raise KeyError(f"missing required observation in {source}:{demo_key}")
                if any(len(observations[name]) != len(actions) for name in required):
                    raise ValueError(f"observation/action length mismatch in {source}:{demo_key}")
                states = np.asarray(observations["ee_states"], dtype=np.float64)
                if states.shape != (len(actions), ARM_DIM) or not np.isfinite(states).all():
                    raise ValueError(f"invalid ee_states {states.shape} in {source}:{demo_key}")
                partition = "discovery" if demo_id <= 7 else "validation"
                demos.append({"demo_id": demo_id, "partition": partition, "length": len(actions)})
                for frame in range(len(actions) - max(HORIZONS)):
                    for horizon in HORIZONS:
                        row: dict[str, Any] = {
                            "partition": partition,
                            "suite": suite,
                            "task_identity": task_identity,
                            "task_index": task_index,
                            "task_language": language,
                            "source_path": str(source),
                            "source_edge_sha256": source_hash,
                            "demo_id": demo_id,
                            "episode": demo_id,
                            "frame_index": frame,
                            "frame": frame,
                            "horizon": horizon,
                            "command": cumulative_arm_command(actions, frame, horizon),
                            "target_displacement": state_displacement(states, frame, horizon),
                        }
                        row["row_key"] = realization_row_key(row)
                        row["frame_key"] = frame_key(row)
                        rows.append(row)
        sources.append(
            {
                "suite": suite,
                "task_identity": task_identity,
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "edge_sha256": source_hash,
                "language": language,
                "demonstrations": demos,
            }
        )

    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = {}
    for row in rows:
        group = (str(row["partition"]), str(row["task_identity"]), int(row["horizon"]))
        groups.setdefault(group, []).append(row)
    selected: set[str] = set()
    for group in sorted(groups):
        ordered = sorted(groups[group], key=lambda row: (int(row["demo_id"]), int(row["frame_index"]), int(row["horizon"])))
        for row in _evenly_spaced(ordered, MODEL_ROWS_PER_TASK_SPLIT_HORIZON):
            selected.add(str(row["row_key"]))
    if len(selected) > MAX_MODEL_ROWS:
        raise RuntimeError(f"selected {len(selected)} model rows, maximum is {MAX_MODEL_ROWS}")
    for row in rows:
        row["selected_for_model_audit"] = str(row["row_key"]) in selected
    return rows, sources


def _fit_operators(rows: Sequence[Mapping[str, Any]]) -> tuple[dict[str, Any], dict[str, Any]]:
    operators: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    for horizon in HORIZONS:
        discovery = [row for row in rows if row["partition"] == "discovery" and int(row["horizon"]) == horizon]
        validation = [row for row in rows if row["partition"] == "validation" and int(row["horizon"]) == horizon]
        operator = fit_realization_operator(
            [row["command"] for row in discovery],
            [row["target_displacement"] for row in discovery],
            ridge=RIDGE_COEFFICIENT,
            std_floor=STD_FLOOR,
        )
        global_metrics = realization_metrics(
            operator,
            [row["command"] for row in validation],
            [row["target_displacement"] for row in validation],
        )
        by_task: dict[str, Any] = {}
        for task_identity in sorted({str(row["task_identity"]) for row in validation}):
            task_rows = [row for row in validation if row["task_identity"] == task_identity]
            by_task[task_identity] = {
                "row_count": len(task_rows),
                **realization_metrics(
                    operator,
                    [row["command"] for row in task_rows],
                    [row["target_displacement"] for row in task_rows],
                ),
            }
        operators[str(horizon)] = operator
        audit[str(horizon)] = {
            "discovery_row_count": len(discovery),
            "validation_row_count": len(validation),
            "discovery_command_variance": np.var([row["command"] for row in discovery], axis=0),
            "discovery_state_variance": np.var([row["target_displacement"] for row in discovery], axis=0),
            "global_validation": global_metrics,
            "per_task_validation": by_task,
        }
    return operators, audit


def _data_summary(
    rows: Sequence[Mapping[str, Any]],
    model_rows: Sequence[Mapping[str, Any]],
    operators: Mapping[str, Any],
    operator_audit: Mapping[str, Any],
) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    for horizon in HORIZONS:
        counts[str(horizon)] = {
            partition: sum(
                row["partition"] == partition and int(row["horizon"]) == horizon for row in rows
            )
            for partition in ("discovery", "validation")
        }
    task_counts: dict[str, int] = {}
    for row in model_rows:
        task = str(row["task_identity"])
        task_counts[task] = task_counts.get(task, 0) + 1
    task_fractions = {task: count / max(len(model_rows), 1) for task, count in task_counts.items()}
    sampled_coverage: dict[str, Any] = {}
    for partition in ("discovery", "validation"):
        for task_identity in sorted({str(row["task_identity"]) for row in model_rows}):
            selected = [
                row
                for row in model_rows
                if row["partition"] == partition and row["task_identity"] == task_identity
            ]
            sampled_coverage[f"{partition}|{task_identity}"] = {
                "row_count": len(selected),
                "demo_ids": sorted({int(row["demo_id"]) for row in selected}),
                "minimum_frame_index": min((int(row["frame_index"]) for row in selected), default=None),
                "maximum_frame_index": max((int(row["frame_index"]) for row in selected), default=None),
                "horizons": sorted({int(row["horizon"]) for row in selected}),
            }
    return {
        "counts": counts,
        "minimum_discovery_rows_per_horizon": min(value["discovery"] for value in counts.values()),
        "minimum_validation_rows_per_horizon": min(value["validation"] for value in counts.values()),
        "command_variance_all_positive": all(
            bool(np.all(np.asarray(operator_audit[str(h)]["discovery_command_variance"]) > 0)) for h in HORIZONS
        ),
        "state_variance_all_positive": all(
            bool(np.all(np.asarray(operator_audit[str(h)]["discovery_state_variance"]) > 0)) for h in HORIZONS
        ),
        "sampled_model_task_counts": task_counts,
        "sampled_model_task_fractions": task_fractions,
        "sampled_model_coverage": sampled_coverage,
        "maximum_sampled_task_fraction": max(task_fractions.values(), default=1.0),
        "all_operator_ranks_six": all(int(operators[str(h)]["rank"]) == ARM_DIM for h in HORIZONS),
        "minimum_operator_relative_improvement": min(
            float(operator_audit[str(h)]["global_validation"]["normalized_relative_improvement"])
            for h in HORIZONS
        ),
        "all_tasks_reported": all(
            len(operator_audit[str(h)]["per_task_validation"]) == len(TASK_SOURCES) for h in HORIZONS
        ),
        "operator_audit": operator_audit,
    }


def _feature_path(feature_dir: Path, row: Mapping[str, Any]) -> Path:
    digest = hashlib.sha256(str(row["frame_key"]).encode("utf-8")).hexdigest().upper()
    return feature_dir / f"{digest}.npz"


def _save_feature(path: Path, **values: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **values)
    temporary.replace(path)


def _load_feature(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as loaded:
        return {key: np.asarray(loaded[key]) for key in loaded.files}


def _stable_seed(identity: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{purpose}|{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _noise(identity: str, purpose: str, shape: Sequence[int], device: str) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(identity, purpose))
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device)


def _core_policy(policy: Any) -> Any:
    return policy.get_base_model() if hasattr(policy, "get_base_model") else policy


def _native_velocity(policy: Any, batch: Mapping[str, Any], noise: Any, time: Any) -> tuple[Any, Any, Any]:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    core = _core_policy(policy)
    images, image_masks = core.prepare_images(batch)
    state = core.prepare_state(batch)
    actions = core.prepare_action(batch)
    language_tokens = batch["observation.language.tokens"]
    language_masks = batch["observation.language.attention_mask"]
    time_expanded = time[:, None, None]
    x_t = time_expanded * noise + (1.0 - time_expanded) * actions
    model = core.model
    prefix_embeddings, prefix_pad_masks, prefix_attention_masks = model.embed_prefix(
        images, image_masks, language_tokens, language_masks, state=state
    )
    suffix_embeddings, suffix_pad_masks, suffix_attention_masks = model.embed_suffix(x_t, time)
    pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
    attention_masks = torch.cat([prefix_attention_masks, suffix_attention_masks], dim=1)
    attention_2d = make_att_2d_masks(pad_masks, attention_masks)
    position_ids = torch.cumsum(pad_masks, dim=1) - 1
    (_, suffix_output), _ = model.vlm_with_expert.forward(
        attention_mask=attention_2d,
        position_ids=position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embeddings, suffix_embeddings],
        use_cache=False,
        fill_kv_cache=False,
    )
    suffix_output = suffix_output[:, -core.config.chunk_size :].to(dtype=torch.float32)
    velocity = model.action_out_proj(suffix_output)
    return actions, x_t, velocity


def _action_stats(postprocessor: Any) -> dict[str, Any]:
    for step in postprocessor.steps:
        tensor_stats = getattr(step, "_tensor_stats", None)
        if not tensor_stats or "action" not in tensor_stats:
            continue
        stats = tensor_stats["action"]
        if "mean" not in stats or "std" not in stats:
            raise RuntimeError("KITE requires checkpoint MEAN_STD action statistics")
        mean = stats["mean"].detach().float().cpu().numpy().reshape(ACTION_DIM)
        std = stats["std"].detach().float().cpu().numpy().reshape(ACTION_DIM)
        mode = str(getattr(step, "norm_map", {}).get(__import__("lerobot.configs.types", fromlist=["FeatureType"]).FeatureType.ACTION))
        if "MEAN_STD" not in mode:
            raise RuntimeError(f"KITE expected MEAN_STD action normalization, received {mode}")
        return {"mode": "MEAN_STD", "mean": mean, "std": std, "processor_step": type(step).__name__}
    raise RuntimeError("checkpoint postprocessor has no action unnormalizer statistics")


def _evaluate_frame(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    row: Mapping[str, Any],
    action_stats: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    import torch

    raw = _raw_sample(row)
    target_actions = raw["action"].detach().float().cpu().numpy().reshape(50, ACTION_DIM)
    batch = _preprocess(preprocessor, raw)
    core = _core_policy(policy)
    shape = (1, core.config.chunk_size, core.config.max_action_dim)
    noise = _noise(str(row["frame_key"]), "headroom", shape, "cuda")
    time = torch.full((1,), FLOW_TIME, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        normalized_target, x_t, velocity = _native_velocity(policy, batch, noise, time)
        clean = x_t - time[:, None, None] * velocity
        raw_actions = differentiable_mean_std_unnormalize(
            clean[:, :, :ACTION_DIM], action_stats["mean"], action_stats["std"]
        )
        processor_actions = postprocessor(clean[:, :, :ACTION_DIM])
    if hasattr(processor_actions, "detach"):
        processor_actions = processor_actions.detach().cpu().numpy()
    raw_array = raw_actions.detach().float().cpu().numpy().reshape(50, ACTION_DIM)
    processor_array = np.asarray(processor_actions, dtype=np.float32).reshape(50, ACTION_DIM)
    return {
        "normalized_target": normalized_target.detach().float().cpu().numpy(),
        "x_t": x_t.detach().float().cpu().numpy(),
        "native_velocity": velocity.detach().float().cpu().numpy(),
        "base_action_chunk": raw_array.astype(np.float32),
        "target_action_chunk": target_actions.astype(np.float32),
        "processor_max_abs_error": np.asarray([np.max(np.abs(raw_array - processor_array))], dtype=np.float64),
    }


def _partial_row(
    row: Mapping[str, Any], feature_path: Path, feature: Mapping[str, np.ndarray], operator: Mapping[str, Any]
) -> dict[str, Any]:
    horizon = int(row["horizon"])
    base_actions = np.asarray(feature["base_action_chunk"], dtype=np.float64)
    target_actions = np.asarray(feature["target_action_chunk"], dtype=np.float64)
    base_command = base_actions[:horizon, :ARM_DIM].sum(axis=0)
    demo_command = target_actions[:horizon, :ARM_DIM].sum(axis=0)
    base_prediction = predict_realization(operator, base_command.reshape(1, ARM_DIM))[0]
    demo_prediction = predict_realization(operator, demo_command.reshape(1, ARM_DIM))[0]
    target = np.asarray(row["target_displacement"], dtype=np.float64)
    mean = np.asarray(operator["displacement_mean"], dtype=np.float64)
    std = np.asarray(operator["displacement_std"], dtype=np.float64)
    normalized_target = (target - mean) / std
    normalized_base = (base_prediction - mean) / std
    normalized_demo = (demo_prediction - mean) / std
    return {
        "row_key": str(row["row_key"]),
        "frame_key": str(row["frame_key"]),
        "partition": str(row["partition"]),
        "suite": str(row["suite"]),
        "task_identity": str(row["task_identity"]),
        "source_edge_sha256": str(row["source_edge_sha256"]),
        "demo_id": int(row["demo_id"]),
        "frame_index": int(row["frame_index"]),
        "horizon": horizon,
        "feature_cache_path": str(feature_path),
        "feature_cache_sha256": _sha256(feature_path),
        "base_action": base_actions[:horizon],
        "target_action": target_actions[:horizon],
        "base_command": base_command,
        "target_command": demo_command,
        "base_predicted_displacement": base_prediction,
        "demo_predicted_displacement": demo_prediction,
        "target_displacement": target,
        "base_normalized_huber": huber_loss(normalized_base, normalized_target),
        "demo_operator_normalized_huber": huber_loss(normalized_demo, normalized_target),
        "processor_max_abs_error": float(np.asarray(feature["processor_max_abs_error"]).reshape(-1)[0]),
        "native_velocity_finite_fraction": float(np.mean(np.isfinite(feature["native_velocity"]))),
        "base_action_finite": bool(np.isfinite(base_actions).all()),
        "base_action_in_bounds": bool(np.all(np.abs(base_actions) <= 1.0)),
    }


def _partial_payload(
    manifest_hash: str | None,
    operator_hash: str | None,
    planned: int | None,
    rows: Sequence[Mapping[str, Any]],
    *,
    exception_count: int = 0,
    last_exception: str | None = None,
) -> dict[str, Any]:
    return {
        "method": "KITE-VLA",
        "stage": "0A",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "operator_hash": operator_hash,
        "planned_model_row_count": planned,
        "completed_model_row_count": len(rows),
        "completed_row_keys": [str(row["row_key"]) for row in rows],
        "rows": list(rows),
        "exception_count": int(exception_count),
        "last_exception": last_exception,
        "updated_at": _utc_now(),
    }


def _load_resume(
    path: Path,
    model_rows: Sequence[Mapping[str, Any]],
    manifest_hash: str,
    operator_hash: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    partial = _read_json(path)
    if partial.get("proposal_hash") != PROPOSAL_HASH:
        raise RuntimeError("partial proposal hash mismatch")
    if partial.get("manifest_hash") != manifest_hash or partial.get("operator_hash") != operator_hash:
        raise RuntimeError("partial manifest or operator hash mismatch")
    rows = list(partial.get("rows") or [])
    audit = validate_manifest(model_rows, rows)
    if audit["duplicate_partial_key_count"] or audit["extra_partial_key_count"]:
        raise RuntimeError(f"partial contains duplicate or off-manifest keys: {audit}")
    for row in rows:
        cache = Path(str(row["feature_cache_path"]))
        if not cache.is_file() or _sha256(cache) != row["feature_cache_sha256"]:
            raise RuntimeError(f"cached feature hash mismatch for {row['row_key']}")
    return rows


def _headroom(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    validation = [row for row in rows if row["partition"] == "validation"]
    base = np.asarray([row["base_normalized_huber"] for row in validation], dtype=np.float64)
    demo = np.asarray([row["demo_operator_normalized_huber"] for row in validation], dtype=np.float64)
    base_median = float(np.median(base))
    demo_median = float(np.median(demo))
    gap = base_median - demo_median
    relative = gap / max(demo_median, 1e-12)
    by_horizon = {}
    for horizon in HORIZONS:
        selected = [row for row in validation if int(row["horizon"]) == horizon]
        by_horizon[str(horizon)] = {
            "row_count": len(selected),
            "base_median_normalized_huber": float(np.median([row["base_normalized_huber"] for row in selected])),
            "demo_median_normalized_huber": float(
                np.median([row["demo_operator_normalized_huber"] for row in selected])
            ),
        }
    return {
        "validation_row_count": len(validation),
        "base_median_normalized_huber": base_median,
        "demo_operator_median_normalized_huber": demo_median,
        "absolute_gap": gap,
        "relative_deficit": relative,
        "relative_gate": HEADROOM_RELATIVE_GATE,
        "absolute_gate": HEADROOM_ABSOLUTE_GATE,
        "passed": relative >= HEADROOM_RELATIVE_GATE or gap >= HEADROOM_ABSOLUTE_GATE,
        "by_horizon": by_horizon,
    }


def _decoded_chunk(policy: Any, batch: Mapping[str, Any], postprocessor: Any, noise: Any) -> tuple[Any, np.ndarray]:
    import torch

    if hasattr(policy, "reset"):
        policy.reset()
    policy.eval()
    with torch.no_grad():
        native = policy.predict_action_chunk(_clone_batch(batch), noise=noise.clone())
    return native.detach().float().cpu(), _postprocess_chunk(native, postprocessor)


def _kite_loss(
    policy: Any,
    batch: Mapping[str, Any],
    noise: Any,
    time: Any,
    action_stats: Mapping[str, Any],
    operators: Mapping[str, Any],
    targets: Mapping[int, Sequence[float]],
) -> Any:
    import torch
    import torch.nn.functional as functional

    _, x_t, velocity = _native_velocity(policy, batch, noise, time)
    clean = x_t - time[:, None, None] * velocity
    raw = differentiable_mean_std_unnormalize(clean[:, :, :ACTION_DIM], action_stats["mean"], action_stats["std"])
    losses = []
    for horizon in HORIZONS:
        command = raw[:, :horizon, :ARM_DIM].sum(dim=1)
        prediction = torch_realization_normalized(operators[str(horizon)], command)
        operator = operators[str(horizon)]
        target = torch.as_tensor(targets[horizon], dtype=prediction.dtype, device=prediction.device)
        mean = torch.as_tensor(operator["displacement_mean"], dtype=prediction.dtype, device=prediction.device)
        std = torch.as_tensor(operator["displacement_std"], dtype=prediction.dtype, device=prediction.device)
        target_normalized = (target - mean) / std
        losses.append(functional.smooth_l1_loss(prediction, target_normalized.reshape(1, ARM_DIM), beta=1.0))
    return torch.stack(losses).mean()


def _gradient_values(loss: Any, named: Sequence[tuple[str, Any]]) -> tuple[list[Any | None], float, float]:
    import torch

    gradients = torch.autograd.grad(loss, [parameter for _, parameter in named], allow_unused=True)
    squared = 0.0
    finite_count = 0
    value_count = 0
    detached: list[Any | None] = []
    for gradient in gradients:
        if gradient is None:
            detached.append(None)
            continue
        value = gradient.detach().float().cpu()
        detached.append(value)
        squared += float(torch.sum(value.square()).item())
        finite_count += int(torch.isfinite(value).sum().item())
        value_count += int(value.numel())
    return detached, math.sqrt(squared), finite_count / max(value_count, 1)


def _gradient_audit(
    policy: Any,
    preprocessor: Any,
    row_by_horizon: Mapping[int, Mapping[str, Any]],
    action_stats: Mapping[str, Any],
    operators: Mapping[str, Any],
) -> dict[str, Any]:
    import torch

    row = row_by_horizon[20]
    raw = _raw_sample(row)
    batch = _preprocess(preprocessor, raw)
    core = _core_policy(policy)
    shape = (1, core.config.chunk_size, core.config.max_action_dim)
    noise = _noise(str(row["frame_key"]), "gradient", shape, "cuda")
    time = torch.full((1,), FLOW_TIME, dtype=torch.float32, device="cuda")
    named = sorted(
        [(name, parameter) for name, parameter in policy.named_parameters() if parameter.requires_grad],
        key=lambda item: item[0],
    )
    if not named or not all("lora_" in name.lower() for name, _ in named):
        raise RuntimeError("KITE expected only LoRA trainable parameters")

    policy.train()
    policy.zero_grad(set_to_none=True)
    flow_loss = _loss(policy, batch, noise, time)
    flow_gradients, flow_norm, flow_finite = _gradient_values(flow_loss, named)
    flow_value = float(flow_loss.detach().item())
    del flow_loss
    gc.collect()
    torch.cuda.empty_cache()

    policy.zero_grad(set_to_none=True)
    kite_loss = _kite_loss(
        policy,
        batch,
        noise,
        time,
        action_stats,
        operators,
        {horizon: row_by_horizon[horizon]["target_displacement"] for horizon in HORIZONS},
    )
    kite_gradients, kite_norm, kite_finite = _gradient_values(kite_loss, named)
    kite_value = float(kite_loss.detach().item())

    dot = 0.0
    for flow_gradient, kite_gradient in zip(flow_gradients, kite_gradients, strict=True):
        if flow_gradient is not None and kite_gradient is not None:
            dot += float(torch.sum(flow_gradient * kite_gradient).item())
    cosine = dot / max(flow_norm * kite_norm, 1e-12)
    frozen_gradient_names = [
        name for name, parameter in policy.named_parameters() if "lora_" not in name.lower() and parameter.grad is not None
    ]
    policy.zero_grad(set_to_none=True)
    policy.eval()
    return {
        "flow_time": FLOW_TIME,
        "flow_loss": flow_value,
        "kite_loss": kite_value,
        "lambda_reference": 0.3,
        "reference_total_loss": flow_value + 0.3 * kite_value,
        "trainable_parameter_names": [name for name, _ in named],
        "trainable_parameter_count": len(named),
        "trainable_numel": sum(int(parameter.numel()) for _, parameter in named),
        "flow_gradient_norm": flow_norm,
        "kite_gradient_norm": kite_norm,
        "kite_to_flow_gradient_ratio": kite_norm / max(flow_norm, 1e-12),
        "gradient_cosine": cosine,
        "flow_gradient_finite_fraction": flow_finite,
        "kite_gradient_finite_fraction": kite_finite,
        "frozen_parameter_gradient_count": len(frozen_gradient_names),
        "frozen_parameter_gradient_names": frozen_gradient_names,
        "kite_gradient_nonzero": kite_norm > 0.0,
        "finite_objectives_and_gradients": bool(
            np.isfinite([flow_value, kite_value, flow_norm, kite_norm, cosine]).all()
            and flow_finite == 1.0
            and kite_finite == 1.0
        ),
    }


def _identity_and_gradient_audit(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    identity_rows: Mapping[int, Mapping[str, Any]],
    action_stats: Mapping[str, Any],
    operators: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch
    from peft import PeftConfig, PeftModel

    row = identity_rows[20]
    raw = _raw_sample(row)
    batch = _preprocess(preprocessor, raw)
    core = _core_policy(policy)
    shape = (1, core.config.chunk_size, core.config.max_action_dim)
    flow_noise = _noise(str(row["frame_key"]), "identity_flow", shape, "cuda")
    solver_noise = _noise(str(row["frame_key"]), "identity_solver", shape, "cuda")
    time = torch.full((1,), FLOW_TIME, dtype=torch.float32, device="cuda")
    with torch.no_grad():
        _, _, base_flow = _native_velocity(policy, batch, flow_noise, time)
    base_native, base_actions = _decoded_chunk(policy, batch, postprocessor, solver_noise)
    base_hash_before = _hash_base_parameters(policy)

    policy = policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": 4})
    policy.to("cuda")
    policy.eval()
    with torch.no_grad():
        _, _, initialized_flow = _native_velocity(policy, batch, flow_noise, time)
    initialized_native, initialized_actions = _decoded_chunk(policy, batch, postprocessor, solver_noise)
    base_hash_after = _hash_base_parameters(policy)
    initialized_errors = {
        "flow": float(torch.max(torch.abs(initialized_flow - base_flow)).item()),
        "native": float(torch.max(torch.abs(initialized_native - base_native)).item()),
        "actions": float(np.max(np.abs(initialized_actions - base_actions))),
    }

    gradient = _gradient_audit(policy, preprocessor, identity_rows, action_stats, operators)
    paths["adapter_dir"].mkdir(parents=True, exist_ok=True)
    if hasattr(policy, "peft_config"):
        for config in policy.peft_config.values():
            config.base_model_name_or_path = str(paths["checkpoint"])
    policy.save_pretrained(paths["adapter_dir"], safe_serialization=True)
    adapter_files = sorted(path.name for path in paths["adapter_dir"].iterdir() if path.is_file())

    del policy
    gc.collect()
    torch.cuda.empty_cache()
    base, _, reloaded_preprocessor, reloaded_postprocessor = _load_policy_and_processors(paths["checkpoint"])
    peft_config = PeftConfig.from_pretrained(paths["adapter_dir"])
    peft_config.base_model_name_or_path = str(paths["checkpoint"])
    reloaded = PeftModel.from_pretrained(
        base,
        paths["adapter_dir"],
        config=peft_config,
        is_trainable=False,
        local_files_only=True,
    )
    reloaded.to("cuda")
    reloaded.eval()
    reloaded_batch = _preprocess(reloaded_preprocessor, raw)
    with torch.no_grad():
        _, _, reloaded_flow = _native_velocity(reloaded, reloaded_batch, flow_noise, time)
    reloaded_native, reloaded_actions = _decoded_chunk(
        reloaded, reloaded_batch, reloaded_postprocessor, solver_noise
    )
    reload_errors = {
        "flow": float(torch.max(torch.abs(reloaded_flow - base_flow)).item()),
        "native": float(torch.max(torch.abs(reloaded_native - base_native)).item()),
        "actions": float(np.max(np.abs(reloaded_actions - base_actions))),
    }
    identity_max = max(*initialized_errors.values(), *reload_errors.values())
    identity = {
        "rank": 4,
        "base_flow_shape": list(base_flow.shape),
        "base_native_shape": list(base_native.shape),
        "base_action_shape": list(base_actions.shape),
        "initialized_max_abs_errors": initialized_errors,
        "reload_max_abs_errors": reload_errors,
        "identity_max_abs_error": identity_max,
        "base_parameter_hash_before": base_hash_before,
        "base_parameter_hash_after": base_hash_after,
        "base_hash_unchanged": base_hash_before == base_hash_after,
        "checkpoint_reload_ok": identity_max <= 1e-6 and bool(adapter_files),
        "adapter_checkpoint": str(paths["adapter_dir"]),
        "adapter_files": adapter_files,
        "base_action_finite": bool(np.isfinite(base_actions).all()),
        "base_action_in_bounds": bool(np.all(np.abs(base_actions) <= 1.0)),
        "training_only_operator_absent_from_policy_parameters": not any(
            "kite" in name.lower() for name, _ in reloaded.named_parameters()
        ),
    }
    del reloaded
    gc.collect()
    torch.cuda.empty_cache()
    return identity, gradient


def _result_markdown(result: Mapping[str, Any]) -> str:
    data = result["data_audit"]
    headroom = result.get("headroom") or {}
    gradient = result.get("gradient") or {}
    identity = result.get("identity") or {}
    return "\n".join(
        [
            "# KITE-VLA Stage 0A Result",
            "",
            f"Final decision: `{result['final_decision']}`.",
            "",
            f"Label rows / model rows: `{result['planned_label_row_count']} / {result['completed_model_row_count']}`.",
            f"Minimum operator validation improvement: `{data['minimum_operator_relative_improvement']}`.",
            f"Headroom relative deficit / absolute gap: `{headroom.get('relative_deficit')} / {headroom.get('absolute_gap')}`.",
            f"Flow / KITE gradient norm: `{gradient.get('flow_gradient_norm')} / {gradient.get('kite_gradient_norm')}`.",
            f"Identity maximum error: `{identity.get('identity_max_abs_error')}`.",
            f"Exceptions: `{result['exception_count']}`.",
            "",
            "No adapter optimization, simulator load, reward/success/done read, confirmatory identity access, or closed-loop experiment occurred.",
            "",
        ]
    )


def _data_gates_pass(data: Mapping[str, Any], manifest_audit: Mapping[str, Any]) -> bool:
    return bool(
        manifest_audit["duplicate_manifest_key_count"] == 0
        and manifest_audit["split_overlap_key_count"] == 0
        and int(data["minimum_discovery_rows_per_horizon"]) >= 512
        and int(data["minimum_validation_rows_per_horizon"]) >= 96
        and data["command_variance_all_positive"]
        and data["state_variance_all_positive"]
        and float(data["maximum_sampled_task_fraction"]) <= 0.40
        and data["all_operator_ranks_six"]
        and float(data["minimum_operator_relative_improvement"]) >= OPERATOR_IMPROVEMENT_GATE
        and data["all_tasks_reported"]
    )


def run(args: argparse.Namespace, paths: Mapping[str, Path], state: dict[str, Any]) -> dict[str, Any]:
    _set_offline_environment()
    if paths["result_json"].is_file() and _read_json(paths["result_json"]).get("final_decision"):
        raise RuntimeError("completed KITE Stage 0A result exists; refusing duplicate execution")
    serializer = _read_json(paths["serializer_preflight"])
    serializer_ok = bool(
        serializer.get("passed")
        and canonical_json_sha256(serializer["fixture"]) == serializer.get("fixture_hash")
    )
    if not serializer_ok:
        raise RuntimeError("foreground KITE serializer preflight is absent or invalid")
    proposal_recomputed = _sha256(PROPOSAL_FILE)
    proposal_registry = _registry_hash()
    proposal_ok = proposal_recomputed == proposal_registry == PROPOSAL_HASH
    if not proposal_ok:
        raise RuntimeError("frozen KITE proposal hash mismatch")

    state.update({"phase": "manifest", "status": "running"})
    rows, sources = _build_manifest(paths["data_root"])
    model_rows = [row for row in rows if row["selected_for_model_audit"]]
    operators, operator_audit = _fit_operators(rows)
    operator_hash = canonical_json_sha256(operators)
    manifest_payload = {
        "method": "KITE-VLA",
        "stage": "0A",
        "proposal_hash": PROPOSAL_HASH,
        "sources": sources,
        "horizons": list(HORIZONS),
        "ridge_coefficient": RIDGE_COEFFICIENT,
        "std_floor": STD_FLOOR,
        "model_rows_per_task_split_horizon": MODEL_ROWS_PER_TASK_SPLIT_HORIZON,
        "planned_label_row_count": len(rows),
        "planned_model_row_count": len(model_rows),
        "operator_hash": operator_hash,
        "operators": operators,
        "operator_audit": operator_audit,
        "rows": rows,
    }
    manifest_hash = canonical_json_sha256(manifest_payload)
    manifest_payload["manifest_hash"] = manifest_hash
    _write_json(paths["manifest"], manifest_payload)
    parsed_manifest = _read_json(paths["manifest"])
    parsed_hash_payload = dict(parsed_manifest)
    parsed_hash_payload.pop("manifest_hash")
    if canonical_json_sha256(parsed_hash_payload) != manifest_hash:
        raise RuntimeError("persisted KITE manifest hash did not reproduce")

    manifest_audit = validate_manifest(rows, [{"row_key": row["row_key"]} for row in rows])
    data = _data_summary(rows, model_rows, operators, operator_audit)
    partial_rows = _load_resume(paths["partial"], model_rows, manifest_hash, operator_hash)
    resumed_count = len(partial_rows)
    _write_json(
        paths["preflight"],
        {
            "method": "KITE-VLA",
            "proposal_hash_recomputed": proposal_recomputed,
            "proposal_hash_registry": proposal_registry,
            "proposal_hash_ok": proposal_ok,
            "serializer_preflight_path": str(paths["serializer_preflight"]),
            "serializer_preflight_ok": serializer_ok,
            "manifest_hash": manifest_hash,
            "manifest_json_parsed": True,
            "operator_hash": operator_hash,
            "planned_label_row_count": len(rows),
            "planned_model_row_count": len(model_rows),
            "resumed_model_row_count": resumed_count,
            "data_gates_passed": _data_gates_pass(data, manifest_audit),
            "adapter_training_happened": False,
            "simulator_load_count": 0,
            "confirmatory_records_read": 0,
            "written_at": _utc_now(),
        },
    )
    if not _data_gates_pass(data, manifest_audit):
        decision_inputs = Stage0ADecisionInputs(
            proposal_hash_ok=proposal_ok,
            serializer_preflight_ok=serializer_ok,
            manifest_integrity_ok=manifest_audit["duplicate_manifest_key_count"] == 0
            and manifest_audit["split_overlap_key_count"] == 0,
            source_alignment_ok=True,
            minimum_discovery_rows_per_horizon=data["minimum_discovery_rows_per_horizon"],
            minimum_validation_rows_per_horizon=data["minimum_validation_rows_per_horizon"],
            command_variance_all_positive=data["command_variance_all_positive"],
            state_variance_all_positive=data["state_variance_all_positive"],
            maximum_sampled_task_fraction=data["maximum_sampled_task_fraction"],
            all_operator_ranks_six=data["all_operator_ranks_six"],
            minimum_operator_relative_improvement=data["minimum_operator_relative_improvement"],
            all_tasks_reported=data["all_tasks_reported"],
            base_headroom_passed=False,
            finite_objectives_and_gradients=False,
            kite_gradient_nonzero=False,
            gradient_ratio_at_most_100=False,
            frozen_parameter_gradient_count=0,
            identity_max_error=0.0,
            base_hash_unchanged=True,
            checkpoint_reload_ok=True,
            action_validity_ok=True,
            exception_count=0,
        )
        decision = classify_stage0a(decision_inputs)
        result = {
            "method": "KITE-VLA",
            "stage": "0A",
            "proposal_hash": PROPOSAL_HASH,
            "manifest_hash": manifest_hash,
            "operator_hash": operator_hash,
            "worker_pid": os.getpid(),
            "planned_label_row_count": len(rows),
            "planned_model_row_count": len(model_rows),
            "completed_model_row_count": resumed_count,
            "resumed_model_row_count": resumed_count,
            "exception_count": 0,
            "manifest_audit": manifest_audit,
            "data_audit": data,
            "headroom": None,
            "gradient": None,
            "identity": None,
            "final_decision": decision,
            "stage_0b_allowed": False,
            "valid_scientific_result": False,
            "scientific_kill": False,
            "adapter_training_happened": False,
            "simulator_load_count": 0,
            "reward_read_count": 0,
            "success_read_count": 0,
            "done_read_count": 0,
            "confirmatory_records_read": 0,
            "timing_throughput_resource_evidence_eligible_for_paper": False,
            "completed_at": _utc_now(),
        }
        _write_json(paths["result_json"], result)
        _write_text(paths["result_md"], _result_markdown(result))
        _write_json(paths["validation"], {"final_decision": decision, "data_gate_stop": True, **manifest_audit})
        return result

    completed = {str(row["row_key"]) for row in partial_rows}
    state.update(
        {
            "phase": "model_audit",
            "planned_model_row_count": len(model_rows),
            "completed_model_row_count": len(partial_rows),
            "exception_count": 0,
        }
    )
    _write_json(paths["partial"], _partial_payload(manifest_hash, operator_hash, len(model_rows), partial_rows))
    policy, _, preprocessor, postprocessor = _load_policy_and_processors(paths["checkpoint"])
    action_stats = _action_stats(postprocessor)
    for row in model_rows:
        if row["row_key"] in completed:
            continue
        feature_path = _feature_path(paths["feature_dir"], row)
        if feature_path.is_file():
            feature = _load_feature(feature_path)
        else:
            feature = _evaluate_frame(policy, preprocessor, postprocessor, row, action_stats)
            _save_feature(feature_path, **feature)
        summary = _partial_row(row, feature_path, feature, operators[str(row["horizon"])])
        partial_rows.append(summary)
        completed.add(str(row["row_key"]))
        state["completed_model_row_count"] = len(partial_rows)
        _write_json(paths["partial"], _partial_payload(manifest_hash, operator_hash, len(model_rows), partial_rows))
        _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
        print(f"[kite-stage0a] model rows {len(partial_rows)}/{len(model_rows)}", flush=True)

    ordered = {str(row["row_key"]): row for row in partial_rows}
    partial_rows = [ordered[str(row["row_key"])] for row in model_rows]
    partial_audit = validate_manifest(model_rows, partial_rows)
    headroom = _headroom(partial_rows)
    identity_candidates = [row for row in model_rows if row["partition"] == "discovery" and row["demo_id"] == 0]
    first_frame = min(int(row["frame_index"]) for row in identity_candidates)
    identity_rows = {
        int(row["horizon"]): row
        for row in identity_candidates
        if int(row["frame_index"]) == first_frame
    }
    if set(identity_rows) != set(HORIZONS):
        raise RuntimeError("identity audit could not align both frozen horizons")
    identity, gradient = _identity_and_gradient_audit(
        policy, preprocessor, postprocessor, identity_rows, action_stats, operators, paths
    )
    del policy
    gc.collect()

    action_validity = bool(
        all(
            row["native_velocity_finite_fraction"] == 1.0
            and row["base_action_finite"]
            and row["base_action_in_bounds"]
            and row["processor_max_abs_error"] <= 1e-6
            for row in partial_rows
        )
        and identity["base_action_finite"]
        and identity["base_action_in_bounds"]
    )
    manifest_ok = bool(
        manifest_audit["duplicate_manifest_key_count"] == 0
        and manifest_audit["split_overlap_key_count"] == 0
        and partial_audit["duplicate_manifest_key_count"] == 0
        and partial_audit["duplicate_partial_key_count"] == 0
        and partial_audit["missing_manifest_key_count"] == 0
        and partial_audit["extra_partial_key_count"] == 0
        and partial_audit["split_overlap_key_count"] == 0
        and partial_audit["key_sets_equal"]
    )
    decision_inputs = Stage0ADecisionInputs(
        proposal_hash_ok=proposal_ok,
        serializer_preflight_ok=serializer_ok,
        manifest_integrity_ok=manifest_ok,
        source_alignment_ok=True,
        minimum_discovery_rows_per_horizon=data["minimum_discovery_rows_per_horizon"],
        minimum_validation_rows_per_horizon=data["minimum_validation_rows_per_horizon"],
        command_variance_all_positive=data["command_variance_all_positive"],
        state_variance_all_positive=data["state_variance_all_positive"],
        maximum_sampled_task_fraction=data["maximum_sampled_task_fraction"],
        all_operator_ranks_six=data["all_operator_ranks_six"],
        minimum_operator_relative_improvement=data["minimum_operator_relative_improvement"],
        all_tasks_reported=data["all_tasks_reported"],
        base_headroom_passed=headroom["passed"],
        finite_objectives_and_gradients=gradient["finite_objectives_and_gradients"],
        kite_gradient_nonzero=gradient["kite_gradient_nonzero"],
        gradient_ratio_at_most_100=gradient["kite_to_flow_gradient_ratio"] <= GRADIENT_RATIO_MAX,
        frozen_parameter_gradient_count=gradient["frozen_parameter_gradient_count"],
        identity_max_error=identity["identity_max_abs_error"],
        base_hash_unchanged=identity["base_hash_unchanged"],
        checkpoint_reload_ok=identity["checkpoint_reload_ok"],
        action_validity_ok=action_validity,
        exception_count=0,
    )
    decision = classify_stage0a(decision_inputs)
    result = {
        "method": "KITE-VLA",
        "stage": "0A",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "operator_hash": operator_hash,
        "worker_pid": os.getpid(),
        "planned_label_row_count": len(rows),
        "planned_model_row_count": len(model_rows),
        "completed_model_row_count": len(partial_rows),
        "resumed_model_row_count": resumed_count,
        "exception_count": 0,
        "manifest_audit": manifest_audit,
        "partial_audit": partial_audit,
        "data_audit": data,
        "action_normalization": action_stats,
        "headroom": headroom,
        "gradient": gradient,
        "identity": identity,
        "action_validity_ok": action_validity,
        "final_decision": decision,
        "stage_0b_allowed": decision == "KITE_STAGE_0A_PASS_STAGE_0B_ALLOWED",
        "valid_scientific_result": False,
        "scientific_kill": False,
        "adapter_training_happened": False,
        "optimizer_step_count": 0,
        "simulator_load_count": 0,
        "reward_read_count": 0,
        "success_read_count": 0,
        "done_read_count": 0,
        "confirmatory_records_read": 0,
        "closed_loop_experiment_happened": False,
        "timing_throughput_resource_evidence_eligible_for_paper": False,
        "completed_at": _utc_now(),
    }
    validation = {
        "proposal_hash_ok": proposal_ok,
        "serializer_preflight_ok": serializer_ok,
        "manifest_json_parsed": True,
        "partial_json_parsed": True,
        "result_decision_recomputed": classify_stage0a(decision_inputs),
        "action_validity_ok": action_validity,
        "exception_count": 0,
        "final_decision": decision,
        **partial_audit,
    }
    _write_json(paths["result_json"], result)
    _write_text(paths["result_md"], _result_markdown(result))
    _write_json(paths["validation"], validation)
    _write_json(paths["partial"], _partial_payload(manifest_hash, operator_hash, len(model_rows), partial_rows))
    state.update({"status": "completed", "phase": "complete", "completed_model_row_count": len(partial_rows)})
    _write_json(paths["status"], {**state, "completed_at": _utc_now()})
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
    return result


def _write_blocker(paths: Mapping[str, Path], state: Mapping[str, Any], exc: BaseException) -> None:
    detail = traceback.format_exc()
    manifest_hash = None
    operator_hash = None
    planned = None
    rows: list[dict[str, Any]] = []
    previous_exceptions = 0
    if paths["manifest"].is_file():
        try:
            manifest = _read_json(paths["manifest"])
            manifest_hash = manifest.get("manifest_hash")
            operator_hash = manifest.get("operator_hash")
            planned = manifest.get("planned_model_row_count")
        except Exception:
            pass
    if paths["partial"].is_file():
        try:
            partial = _read_json(paths["partial"])
            rows = list(partial.get("rows") or [])
            previous_exceptions = int(partial.get("exception_count") or 0)
            manifest_hash = partial.get("manifest_hash", manifest_hash)
            operator_hash = partial.get("operator_hash", operator_hash)
            planned = partial.get("planned_model_row_count", planned)
        except Exception:
            rows = []
    _write_json(
        paths["partial"],
        _partial_payload(
            manifest_hash,
            operator_hash,
            planned,
            rows,
            exception_count=previous_exceptions + 1,
            last_exception=detail,
        ),
    )
    _write_json(
        paths["blocker"],
        {
            "method": "KITE-VLA",
            "stage": "0A",
            "proposal_hash": PROPOSAL_HASH,
            "exception_type": type(exc).__name__,
            "exception": str(exc),
            "traceback": detail,
            "completed_model_row_count": len(rows),
            "planned_model_row_count": planned,
            "manifest_persisted": paths["manifest"].is_file(),
            "partial_persisted": True,
            "scientific_kill": False,
            "failed_at": _utc_now(),
        },
    )
    failed = {**state, "status": "failed", "phase": "failed", "exception_count": previous_exceptions + 1}
    _write_json(paths["status"], {**failed, "failed_at": _utc_now()})
    _write_json(paths["heartbeat"], {**failed, "updated_at": _utc_now()})


def _heartbeat_loop(path: Path, state: dict[str, Any], stop: threading.Event) -> None:
    while not stop.wait(10.0):
        _write_json(path, {**state, "updated_at": _utc_now()})


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/c/assets/data/libero")
    parser.add_argument("--checkpoint", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "kite_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "kite_vla" / "stage0a"))
    parser.add_argument("--serializer-preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    paths = _paths(args)
    paths["report"].mkdir(parents=True, exist_ok=True)
    paths["run"].mkdir(parents=True, exist_ok=True)
    if args.serializer_preflight:
        result = _serializer_preflight(paths["serializer_preflight"])
        print(json.dumps({"serializer_preflight_passed": result["passed"]}, sort_keys=True), flush=True)
        return 0

    state: dict[str, Any] = {
        "method": "KITE-VLA",
        "stage": "0A",
        "pid": os.getpid(),
        "status": "starting",
        "phase": "startup",
        "planned_model_row_count": None,
        "completed_model_row_count": 0,
        "exception_count": 0,
    }
    _write_text(paths["pid"], f"{os.getpid()}\n")
    _write_json(paths["status"], {**state, "started_at": _utc_now()})
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
    stop = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat_loop, args=(paths["heartbeat"], state, stop), daemon=True)
    heartbeat.start()
    try:
        result = run(args, paths, state)
    except Exception as exc:
        _write_blocker(paths, state, exc)
        traceback.print_exc()
        return 1
    finally:
        stop.set()
        heartbeat.join(timeout=2.0)
        gc.collect()
    print(json.dumps({"final_decision": result["final_decision"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
