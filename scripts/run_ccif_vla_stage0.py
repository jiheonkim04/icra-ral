"""Run the frozen CCIF-VLA Stage 0 continuous coarse-intent development audit."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.ccif_vla import (  # noqa: E402
    ACTION_DIM,
    ACTION_HUBER_DELTA,
    CHUNK_SIZE,
    DEFAULT_TEMPLATE_BETA,
    HEADROOM_ABSOLUTE_HUBER_GATE,
    HEADROOM_RELATIVE_GATE,
    INTENT_DIM,
    PHASE_BINS,
    PROPOSAL_HASH,
    TASK_COUNT,
    VISUAL_FEATURE_DIM,
    WAYPOINT_INDICES,
    Stage0DecisionInputs,
    action_chunk,
    action_delta_summary,
    apply_ccif_residual,
    apply_discovery_zscore,
    canonical_json_sha256,
    ccif_feature_key,
    ccif_row_key,
    classify_stage0,
    coarse_intent,
    denormalize_intent,
    endpoint_only_intent,
    fit_discovery_zscore,
    fit_intent_normalizer,
    fit_intent_probe,
    fit_ridge,
    fit_task_phase_mean_intent,
    flattened_chunks,
    intent_consistency_summary,
    json_default,
    mean_huber,
    normalize_intent,
    phase_bin,
    predict_intent_probe,
    predict_ridge,
    predict_task_phase_mean_intent,
    prediction_metrics,
    raw_ccif_feature,
    residual_cap_from_discovery,
    validate_manifest,
)
from scripts.run_cfr_vla_stage0 import (  # noqa: E402
    _array_sha256,
    _asset_path,
    _base_chunk_path,
    _edge_hash,
    _evenly_spaced,
    _load_base_chunk,
    _load_feature,
    _problem_language,
    _proprio_from_obs,
    _read_json,
    _sha256,
    _utc_now,
    _write_json,
    _write_text,
)
from scripts.run_famr_vla_stage0 import (  # noqa: E402
    HF_HOME,
    VLM_PATH,
    _apply_official_env_image_processor,
    _active_linux_workers,
    _preprocess,
    _raw_sample,
    _resource_evidence,
    _set_offline_environment,
)
from scripts.run_vdr_vla_stage0a import (  # noqa: E402
    _core_policy,
    _decoded_chunk,
)


POLICY_PROBE = "ccif_stage0_continuous_coarse_intent"
SEED = 20262900
DISCOVERY_ROWS_PER_TASK = 128
VALIDATION_ROWS_PER_TASK = 32
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "ccif_vla" / "proposal_hash.txt"
RESOURCE_REGISTRY = REPO_ROOT / "reports" / "resource_contention_intervals.json"

MODEL_OR_PROBE_ROWS = (
    "smolvla_base",
    "coarse_to_control_continuous_proxy",
    "ccif_full",
    "ccif_no_coarse_intent_ablation",
    "standard_lora_proxy",
    "task_phase_mean_intent",
    "endpoint_only_intent",
)

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


def _paths(args: argparse.Namespace) -> dict[str, Path]:
    report = Path(args.report_root)
    run = Path(args.run_root)
    return {
        "report": report,
        "run": run,
        "feature_dir": run / "visual_features",
        "base_chunk_dir": run / "base_chunks",
        "identity_dir": run / "identity_adapter",
        "checkpoint": Path(args.checkpoint),
        "data_root": Path(args.libero_data_root),
        "pid": report / "stage_0_pid.txt",
        "heartbeat": report / "stage_0_heartbeat.json",
        "status": report / "stage_0_status.json",
        "preflight": report / "stage_0_preflight.json",
        "official_prior_asset_check": report / "stage_0_official_prior_asset_check.json",
        "action_semantics": report / "stage_0_action_semantics.json",
        "serializer_preflight": report / "stage_0_serializer_preflight.json",
        "manifest": report / "stage_0_manifest.json",
        "partial": report / "stage_0_partial.json",
        "result_json": report / "stage_0_result.json",
        "result_md": report / "stage_0_result.md",
        "validation": report / "stage_0_validation.json",
        "adjudication": report / "stage_0_adjudication.md",
        "blocker": report / "stage_0_implementation_blocker.json",
        "exit_code": report / "stage_0_exit_code.txt",
    }


def _proposal_hash_text() -> str:
    if not PROPOSAL_HASH_FILE.is_file():
        return ""
    for token in PROPOSAL_HASH_FILE.read_text(encoding="utf-8").split():
        candidate = token.upper()
        if len(candidate) == 64 and all(char in "0123456789ABCDEF" for char in candidate):
            return candidate
    return ""


def _serializer_preflight(path: Path) -> dict[str, Any]:
    manifest_row = {
        "partition": "validation",
        "suite": "libero_spatial",
        "task_identity": "libero_spatial/task_3",
        "source_edge_sha256": "ABC",
        "demo_id": 8,
        "frame_index": 3,
        "model_or_probe": "ccif_full",
        "policy_probe": POLICY_PROBE,
    }
    manifest_row["row_key"] = ccif_row_key(manifest_row)
    fixture: dict[str, Any] = {
        "method": "CCIF-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_row": manifest_row,
        "intent_dimension": np.int64(INTENT_DIM),
        "waypoint_indices": np.asarray(WAYPOINT_INDICES, dtype=np.int64),
        "base_chunk": np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32),
        "nested_metrics": {"intent": {"huber": np.float32(0.0), "passed": np.bool_(True)}},
        "decision_inputs": Stage0DecisionInputs(
            proposal_hash_ok=True,
            serializer_preflight_ok=True,
            official_prior_asset_check_persisted=True,
            manifest_integrity_ok=True,
            source_alignment_ok=True,
            feature_action_proprio_finite_aligned=True,
            split_integrity_ok=True,
            minimum_discovery_windows=512,
            minimum_validation_windows=128,
            all_tasks_reported=True,
            maximum_validation_task_fraction=0.25,
            labels_noncollapsed_discovery=True,
            labels_noncollapsed_validation=True,
            collapsed_intent_component_count=0,
            intent_probe_beats_task_phase_mean=True,
            intent_probe_relative_improvement=0.05,
            intent_probe_absolute_huber=0.0,
            endpoint_only_explains_ccif=False,
            ccif_beats_prior_relative=0.05,
            ccif_beats_prior_absolute_huber=0.0,
            ccif_beats_ablation_relative=0.05,
            ccif_beats_ablation_absolute_huber=0.0,
            action_validity_ok=True,
            identity_max_abs_error=0.0,
            checkpoint_reload_ok=True,
            finite_objectives_and_gradients=True,
            ccif_gradient_nonzero=True,
            frozen_parameter_gradient_count=0,
            weighted_gradient_norm_ratio_max=1.0,
            reward_read_count=0,
            success_read_count=0,
            done_read_count=0,
            confirmatory_records_read=0,
            closed_loop_experiment_happened=False,
            simulator_load_count=0,
            training_happened=False,
            validation_search_happened=False,
            exception_count=0,
        ),
    }
    try:
        import torch

        fixture["torch_tensor"] = torch.zeros(2, dtype=torch.float32)
        tensor_serialization_checked = True
    except Exception as exc:  # pragma: no cover - depends on local torch install
        fixture["torch_tensor_unavailable"] = f"{type(exc).__name__}: {exc}"
        tensor_serialization_checked = False

    fixture["decision"] = classify_stage0(fixture["decision_inputs"])
    fixture_hash = canonical_json_sha256(fixture)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"fixture": fixture, "fixture_hash": fixture_hash}, sort_keys=True, default=json_default), encoding="utf-8")
    parsed = json.loads(path.read_text(encoding="utf-8"))
    reproduced = canonical_json_sha256(parsed["fixture"])
    result = {
        "method": "CCIF-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "path": str(path),
        "parsed": True,
        "fixture_hash": fixture_hash,
        "reproduced_hash": reproduced,
        "tensor_serialization_checked": tensor_serialization_checked,
        "passed": fixture_hash == reproduced,
        "fixture": parsed["fixture"],
        "written_at": _utc_now(),
    }
    _write_json(path, result)
    if not result["passed"]:
        raise RuntimeError("CCIF serializer preflight hash did not reproduce")
    return result


def _official_prior_asset_check(path: Path) -> dict[str, Any]:
    candidates = {
        "official_repository": REPO_ROOT / "third_party" / "Coarse-to-Control",
        "alternate_repository": REPO_ROOT / "third_party" / "coarse-to-control",
        "checkpoint_dir": REPO_ROOT / "third_party" / "Coarse-to-Control" / "checkpoints",
        "inference_code": REPO_ROOT / "third_party" / "Coarse-to-Control" / "inference.py",
    }
    exists = {name: candidate.exists() for name, candidate in candidates.items()}
    official_ready = bool((exists["official_repository"] or exists["alternate_repository"]) and exists["checkpoint_dir"] and exists["inference_code"])
    label = "coarse_to_control_official" if official_ready else "coarse_to_control_continuous_proxy"
    deviations = [] if official_ready else [
        "official Coarse-to-Control repository/checkpoint/inference assets are not all locally verified",
        "Stage 0 fixes policy 2 as a transparent continuous coarse-intent-to-action proxy until official assets are installed",
    ]
    result = {
        "method": "CCIF-VLA",
        "stage": "0",
        "closest_prior": "Coarse-to-Control",
        "policy_2_label": label,
        "official_ready": official_ready,
        "asset_exists": {name: bool(value) for name, value in exists.items()},
        "checked_paths": {name: str(candidate) for name, candidate in candidates.items()},
        "proxy_deviations": deviations,
        "comparison_position": 2,
        "written_at": _utc_now(),
    }
    _write_json(path, result)
    return result


def _action_stats(postprocessor: Any) -> dict[str, Any]:
    for step in postprocessor.steps:
        tensor_stats = getattr(step, "_tensor_stats", None)
        if not tensor_stats or "action" not in tensor_stats:
            continue
        stats = tensor_stats["action"]
        if "mean" not in stats or "std" not in stats:
            raise RuntimeError("CCIF requires checkpoint MEAN_STD action statistics")
        mean = stats["mean"].detach().float().cpu().numpy().reshape(ACTION_DIM)
        std = stats["std"].detach().float().cpu().numpy().reshape(ACTION_DIM)
        return {"mode": "MEAN_STD", "mean": mean, "std": std, "processor_step": type(step).__name__}
    raise RuntimeError("checkpoint postprocessor has no action unnormalizer statistics")


def _write_action_semantics(path: Path, action_stats: Mapping[str, Any]) -> dict[str, Any]:
    semantics = {
        "method": "CCIF-VLA",
        "stage": "0",
        "model_native_action_shape": [CHUNK_SIZE, ACTION_DIM],
        "environment_action_shape": [ACTION_DIM],
        "postprocessor_mode": action_stats.get("mode"),
        "postprocessor_step": action_stats.get("processor_step"),
        "postprocessor_action_mean_shape": list(np.asarray(action_stats.get("mean"), dtype=np.float64).shape),
        "postprocessor_action_std_shape": list(np.asarray(action_stats.get("std"), dtype=np.float64).shape),
        "environment_action_space_low_high_exposed": False,
        "environment_action_space_low": None,
        "environment_action_space_high": None,
        "gripper_convention": "LIBERO/SmolVLA checkpoint action dimension 6 after postprocessor",
        "finite_checks_required": True,
        "action_bound_validity_rule": "not_used_without_official_environment_bounds",
        "final_action_validity_definition": (
            "valid iff postprocessed action chunk has shape [50,7], all entries are finite, "
            "and the same SmolVLA postprocessor statistics are used for every policy"
        ),
        "same_definition_applies_to_policies": list(MODEL_OR_PROBE_ROWS),
        "written_at": _utc_now(),
    }
    _write_json(path, semantics)
    return semantics


def _stable_seed(identity: str, purpose: str) -> int:
    digest = hashlib.sha256(f"{SEED}|{purpose}|{identity}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)


def _noise(identity: str, purpose: str, shape: Sequence[int], device: str) -> Any:
    import torch

    generator = torch.Generator(device="cpu")
    generator.manual_seed(_stable_seed(identity, purpose))
    return torch.randn(tuple(shape), generator=generator, dtype=torch.float32).to(device)


def _policy_device(policy: Any) -> str:
    import torch

    try:
        return str(next(_core_policy(policy).model.parameters()).device)
    except StopIteration:
        return "cuda" if torch.cuda.is_available() else "cpu"


def _load_policy_and_processors_for_ccif(checkpoint: Path) -> tuple[Any, Any, Any, Any]:
    import torch
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = PreTrainedConfig.from_pretrained(checkpoint, local_files_only=True, cache_dir=HF_HOME)
    config.device = device
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
    policy.to(device)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=str(checkpoint),
        preprocessor_overrides={
            "tokenizer_processor": {"tokenizer_name": str(VLM_PATH)},
            "device_processor": {"device": device},
        },
        postprocessor_overrides={"device_processor": {"device": device}},
    )
    return policy, preprocessor, postprocessor, config


def _visual_feature_path(feature_dir: Path, feature_key: str) -> Path:
    digest = hashlib.sha256(feature_key.encode("utf-8")).hexdigest().upper()
    return feature_dir / f"{digest}.npz"


def _save_feature(path: Path, feature: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, feature=np.asarray(feature, dtype=np.float16))
    temporary.replace(path)


def _save_base_chunk(path: Path, chunk: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, base_chunk=np.asarray(chunk, dtype=np.float32))
    temporary.replace(path)


def _prepare_images_for_ccif(policy: Any, images: Sequence[Any]) -> Any:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import resize_with_pad

    device = _policy_device(policy)
    tensors = []
    for image in images:
        value = image if torch.is_tensor(image) else torch.as_tensor(np.asarray(image).copy())
        if value.ndim != 3:
            raise RuntimeError(f"expected CHW/HWC image, got {tuple(value.shape)}")
        if value.shape[-1] in (1, 3):
            value = value.permute(2, 0, 1)
        value = value.float()
        if float(value.max()) > 1.0:
            value = value / 255.0
        tensors.append(value)
    batch = torch.stack(tensors).to(device)
    resize_cfg = getattr(_core_policy(policy).config, "resize_imgs_with_padding", None)
    if resize_cfg is not None:
        batch = resize_with_pad(batch, *resize_cfg, pad_value=0)
    dtype = next(_core_policy(policy).model.parameters()).dtype
    return (batch * 2.0 - 1.0).to(dtype=dtype)


def _extract_visual_feature_for_ccif(policy: Any, row: Mapping[str, Any], frame_index: int) -> np.ndarray:
    import h5py
    import torch

    with h5py.File(str(row["source_path"]), "r") as handle:
        demo = handle["data"][f"demo_{int(row['demo_id'])}"]
        observations = demo["obs"]
        agent, wrist = _apply_official_env_image_processor(
            observations["agentview_rgb"][frame_index], observations["eye_in_hand_rgb"][frame_index]
        )
    prepared = _prepare_images_for_ccif(policy, (agent, wrist))
    with torch.no_grad():
        tokens = _core_policy(policy).model.vlm_with_expert.embed_image(prepared).float().cpu()
    if tokens.shape[:2] != (2, 64) or tokens.shape[2] != VISUAL_FEATURE_DIM:
        raise RuntimeError(f"unexpected CCIF visual token shape {tuple(tokens.shape)}")
    return torch.cat((tokens[0], tokens[1]), dim=0).mean(dim=0).numpy().astype(np.float32)


def _load_or_extract_feature_for_ccif(policy: Any, paths: Mapping[str, Path], row: Mapping[str, Any]) -> tuple[Path, np.ndarray]:
    path = _visual_feature_path(paths["feature_dir"], str(row["feature_key"]))
    if path.is_file():
        return path, _load_feature(path)
    feature = _extract_visual_feature_for_ccif(policy, row, int(row["frame_index"]))
    _save_feature(path, feature)
    return path, _load_feature(path)


def _build_manifest(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    import h5py

    candidates: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for task_index, (suite, task_identity, relative) in enumerate(TASK_SOURCES):
        source = data_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = _edge_hash(source)
        demo_reports = []
        with h5py.File(source, "r") as handle:
            data = handle["data"]
            language = _problem_language(data)
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
                gripper = np.asarray(observations["gripper_states"], dtype=np.float64)
                if states.shape[:2] != (len(actions), 6) or not np.isfinite(states).all():
                    raise ValueError(f"invalid ee_states {states.shape} in {source}:{demo_key}")
                if gripper.shape[0] != len(actions) or not np.isfinite(gripper).all():
                    raise ValueError(f"invalid gripper_states {gripper.shape} in {source}:{demo_key}")
                valid_count = len(actions) - CHUNK_SIZE + 1
                partition = "discovery" if demo_id <= 7 else "validation"
                demo_reports.append({"demo_id": demo_id, "partition": partition, "length": len(actions), "valid_count": valid_count})
                for frame in range(max(0, valid_count)):
                    phase = frame / max(valid_count - 1, 1)
                    base_row: dict[str, Any] = {
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
                        "chunk_size": CHUNK_SIZE,
                        "episode_length": int(len(actions)),
                        "phase": float(phase),
                        "phase_bin": phase_bin(phase),
                    }
                    candidates.append(base_row)
        sources.append(
            {
                "suite": suite,
                "task_identity": task_identity,
                "path": str(source),
                "size_bytes": source.stat().st_size,
                "edge_sha256": source_hash,
                "language": language,
                "demonstrations": demo_reports,
            }
        )

    selected_base_rows: list[dict[str, Any]] = []
    groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in candidates:
        groups.setdefault((str(row["partition"]), str(row["task_identity"])), []).append(row)
    for group in sorted(groups):
        partition, _ = group
        target_count = DISCOVERY_ROWS_PER_TASK if partition == "discovery" else VALIDATION_ROWS_PER_TASK
        ordered = sorted(groups[group], key=lambda item: (int(item["demo_id"]), int(item["frame_index"])))
        selected = _evenly_spaced(ordered, target_count)
        if len(selected) != target_count:
            raise RuntimeError(f"CCIF manifest group {group} has {len(selected)} rows, expected {target_count}")
        selected_base_rows.extend(selected)

    rows: list[dict[str, Any]] = []
    for base_row in selected_base_rows:
        feature_stub = dict(base_row)
        feature_stub["model_or_probe"] = "feature_cache_identity"
        feature_stub["policy_probe"] = POLICY_PROBE
        feature_key = ccif_feature_key(feature_stub)
        for model_or_probe in MODEL_OR_PROBE_ROWS:
            row = dict(base_row)
            row["model_or_probe"] = model_or_probe
            row["proxy_variant"] = model_or_probe
            row["policy_probe"] = POLICY_PROBE
            row["feature_key"] = feature_key
            row["row_key"] = ccif_row_key(row)
            rows.append(row)
    rows.sort(
        key=lambda row: (
            row["partition"],
            row["suite"],
            row["task_identity"],
            int(row["demo_id"]),
            int(row["frame_index"]),
            MODEL_OR_PROBE_ROWS.index(str(row["model_or_probe"])),
        )
    )
    return rows, sources


def _base_chunk_for_feature(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    paths: Mapping[str, Path],
    row: Mapping[str, Any],
) -> tuple[Path, np.ndarray]:
    path = _base_chunk_path(paths["base_chunk_dir"], str(row["feature_key"]))
    if path.is_file():
        return path, _load_base_chunk(path)
    import torch

    decode_row = dict(row)
    decode_row["row_key"] = str(row["feature_key"])
    raw = _raw_sample(decode_row)
    batch = _preprocess(preprocessor, raw)
    core = _core_policy(policy)
    device = _policy_device(policy)
    shape = (1, core.config.chunk_size, core.config.max_action_dim)
    noise = _noise(str(row["feature_key"]), "base_decode", shape, device)
    _, base_actions = _decoded_chunk(policy, batch, postprocessor, noise)
    del batch, noise
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    _save_base_chunk(path, np.asarray(base_actions[:CHUNK_SIZE, :ACTION_DIM], dtype=np.float32))
    return path, _load_base_chunk(path)


def _partial_row(
    row: Mapping[str, Any],
    feature_path: Path,
    feature: np.ndarray,
    base_chunk_path: Path,
    base_chunk: np.ndarray,
) -> dict[str, Any]:
    import h5py

    with h5py.File(str(row["source_path"]), "r") as handle:
        demo = handle["data"][f"demo_{int(row['demo_id'])}"]
        chunk = action_chunk(np.asarray(demo["actions"], dtype=np.float64), int(row["frame_index"]))
        observations = demo["obs"]
        proprio = _proprio_from_obs(observations, int(row["frame_index"]))
    return {
        "row_key": str(row["row_key"]),
        "feature_key": str(row["feature_key"]),
        "partition": str(row["partition"]),
        "suite": str(row["suite"]),
        "task_identity": str(row["task_identity"]),
        "source_edge_sha256": str(row["source_edge_sha256"]),
        "demo_id": int(row["demo_id"]),
        "frame_index": int(row["frame_index"]),
        "model_or_probe": str(row["model_or_probe"]),
        "proxy_variant": str(row["proxy_variant"]),
        "policy_probe": str(row["policy_probe"]),
        "feature_cache_path": str(feature_path),
        "feature_cache_sha256": _sha256(feature_path),
        "base_chunk_cache_path": str(base_chunk_path),
        "base_chunk_cache_sha256": _sha256(base_chunk_path),
        "base_chunk_sha256": _array_sha256(base_chunk),
        "base_action_shape": list(np.asarray(base_chunk).shape),
        "base_action_min": float(np.min(base_chunk)),
        "base_action_max": float(np.max(base_chunk)),
        "base_action_finite": bool(np.isfinite(base_chunk).all()),
        "visual_feature_dim": int(feature.shape[0]),
        "visual_feature_finite_fraction": float(np.mean(np.isfinite(feature))),
        "proprio_dim": int(proprio.shape[0]),
        "proprio_finite_fraction": float(np.mean(np.isfinite(proprio))),
        "action_chunk_sha256": _array_sha256(chunk),
        "action_min": float(np.min(chunk)),
        "action_max": float(np.max(chunk)),
        "action_finite": bool(np.isfinite(chunk).all()),
    }


def _partial_payload(
    manifest_hash: str | None,
    planned: int | None,
    rows: Sequence[Mapping[str, Any]],
    *,
    exception_count: int = 0,
    last_exception: str | None = None,
) -> dict[str, Any]:
    return {
        "method": "CCIF-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "planned_model_row_count": planned,
        "completed_model_row_count": len(rows),
        "completed_row_keys": [str(row["row_key"]) for row in rows],
        "rows": list(rows),
        "exception_count": int(exception_count),
        "last_exception": last_exception,
        "updated_at": _utc_now(),
    }


def _load_resume(
    path: Path, manifest_rows: Sequence[Mapping[str, Any]], manifest_hash: str
) -> tuple[list[dict[str, Any]], int, str | None]:
    if not path.is_file():
        return [], 0, None
    partial = _read_json(path)
    if partial.get("method") != "CCIF-VLA" or partial.get("proposal_hash") != PROPOSAL_HASH:
        raise RuntimeError("partial result identity does not match frozen CCIF proposal/manifest")
    rows = list(partial.get("rows") or [])
    if partial.get("manifest_hash") is None and not rows:
        return [], 0, None
    audit = validate_manifest(manifest_rows, rows)
    if partial.get("manifest_hash") != manifest_hash:
        if not rows or not audit["key_sets_equal"] or audit["duplicate_partial_key_count"] or audit["extra_partial_key_count"]:
            raise RuntimeError("partial result identity does not match frozen CCIF manifest")
    if audit["duplicate_partial_key_count"] or audit["extra_partial_key_count"]:
        raise RuntimeError(f"partial contains duplicate or off-manifest keys: {audit}")
    for row in rows:
        cache = Path(str(row["feature_cache_path"]))
        if not cache.is_file() or _sha256(cache) != row["feature_cache_sha256"]:
            raise RuntimeError(f"feature cache hash mismatch for {row['row_key']}")
        base_cache = Path(str(row["base_chunk_cache_path"]))
        if not base_cache.is_file() or _sha256(base_cache) != row["base_chunk_cache_sha256"]:
            raise RuntimeError(f"base chunk cache hash mismatch for {row['row_key']}")
    return rows, int(partial.get("exception_count") or 0), partial.get("last_exception")


def _unique_manifest_rows(manifest: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    by_feature: dict[str, Mapping[str, Any]] = {}
    for row in manifest:
        by_feature.setdefault(str(row["feature_key"]), row)
    return [by_feature[key] for key in sorted(by_feature)]


def _materialize_arrays(manifest: Sequence[Mapping[str, Any]], partial_rows: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    import h5py

    partial_by_feature: dict[str, Mapping[str, Any]] = {}
    for row in partial_rows:
        partial_by_feature.setdefault(str(row["feature_key"]), row)
    visuals = []
    proprios = []
    raw_features = []
    chunks = []
    base_chunks = []
    task_indices = []
    partitions = []
    phases = []
    task_names = []
    feature_keys = []
    action_finite = []
    for row in _unique_manifest_rows(manifest):
        partial = partial_by_feature[str(row["feature_key"])]
        visual = _load_feature(Path(str(partial["feature_cache_path"])))
        base_chunk = _load_base_chunk(Path(str(partial["base_chunk_cache_path"])))
        with h5py.File(str(row["source_path"]), "r") as handle:
            demo = handle["data"][f"demo_{int(row['demo_id'])}"]
            observations = demo["obs"]
            actions = np.asarray(demo["actions"], dtype=np.float64)
            proprio = _proprio_from_obs(observations, int(row["frame_index"]))
            chunk = action_chunk(actions, int(row["frame_index"]))
        if _array_sha256(chunk) != partial["action_chunk_sha256"]:
            raise RuntimeError(f"action chunk hash mismatch for {row['row_key']}")
        if _array_sha256(base_chunk) != partial["base_chunk_sha256"]:
            raise RuntimeError(f"base chunk hash mismatch for {row['row_key']}")
        raw_feature = raw_ccif_feature(visual, proprio, int(row["task_index"]), float(row["phase"]), base_chunk)
        visuals.append(visual)
        proprios.append(proprio)
        raw_features.append(raw_feature)
        chunks.append(chunk)
        base_chunks.append(base_chunk)
        task_indices.append(int(row["task_index"]))
        partitions.append(str(row["partition"]))
        phases.append(float(row["phase"]))
        task_names.append(str(row["task_identity"]))
        feature_keys.append(str(row["feature_key"]))
        action_finite.append(bool(np.isfinite(chunk).all()))
    return {
        "visual": np.asarray(visuals, dtype=np.float64),
        "proprio": np.asarray(proprios, dtype=np.float64),
        "raw_feature": np.asarray(raw_features, dtype=np.float64),
        "chunk": np.asarray(chunks, dtype=np.float64),
        "base_chunk": np.asarray(base_chunks, dtype=np.float64),
        "residual_chunk": np.asarray(chunks, dtype=np.float64) - np.asarray(base_chunks, dtype=np.float64),
        "task_index": np.asarray(task_indices, dtype=np.int64),
        "partition": np.asarray(partitions, dtype=object),
        "phase": np.asarray(phases, dtype=np.float64),
        "task_identity": np.asarray(task_names, dtype=object),
        "feature_key": np.asarray(feature_keys, dtype=object),
        "action_finite": np.asarray(action_finite, dtype=bool),
    }


def _fit_ccif_models(arrays: dict[str, np.ndarray]) -> tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray]]:
    discovery = arrays["partition"] == "discovery"
    validation = arrays["partition"] == "validation"
    discovery_indices = np.flatnonzero(discovery)
    validation_indices = np.flatnonzero(validation)

    zscore = fit_discovery_zscore(arrays["raw_feature"][discovery])
    zfeatures = apply_discovery_zscore(zscore, arrays["raw_feature"])
    arrays["zfeature"] = zfeatures

    raw_intent = coarse_intent(arrays["chunk"])
    arrays["raw_intent"] = raw_intent
    intent_stats = fit_intent_normalizer(raw_intent[discovery])
    normalized_intent = normalize_intent(intent_stats, raw_intent)
    arrays["normalized_intent"] = normalized_intent
    validation_stats = fit_intent_normalizer(raw_intent[validation])

    task_phase_model = fit_task_phase_mean_intent(
        arrays["task_index"][discovery],
        arrays["phase"][discovery],
        normalized_intent[discovery],
    )
    task_phase_prediction = predict_task_phase_mean_intent(
        task_phase_model,
        arrays["task_index"][validation],
        arrays["phase"][validation],
    )
    intent_probe = fit_intent_probe(zfeatures[discovery], normalized_intent[discovery])
    intent_prediction = predict_intent_probe(intent_probe, zfeatures)
    endpoint_prediction = endpoint_only_intent(intent_prediction, intent_stats)

    intent_vs_task_phase = prediction_metrics(
        intent_prediction[validation],
        task_phase_prediction,
        normalized_intent[validation],
    )
    endpoint_intent_huber = mean_huber(endpoint_prediction[validation], normalized_intent[validation])

    intent_to_action = fit_ridge(normalized_intent[discovery], flattened_chunks(arrays["chunk"][discovery]))
    prior_prediction = predict_ridge(intent_to_action, intent_prediction[validation]).reshape(-1, CHUNK_SIZE, ACTION_DIM)
    endpoint_action_prediction = predict_ridge(intent_to_action, endpoint_prediction[validation]).reshape(-1, CHUNK_SIZE, ACTION_DIM)

    residual_cap = residual_cap_from_discovery(arrays["residual_chunk"][discovery])
    full_features_discovery = np.concatenate([zfeatures[discovery], normalized_intent[discovery]], axis=1)
    full_features_validation = np.concatenate([zfeatures[validation], intent_prediction[validation]], axis=1)
    full_residual_model = fit_ridge(full_features_discovery, flattened_chunks(arrays["residual_chunk"][discovery]))
    full_residual = predict_ridge(full_residual_model, full_features_validation).reshape(-1, CHUNK_SIZE, ACTION_DIM)
    ccif_prediction = apply_ccif_residual(
        arrays["base_chunk"][validation],
        full_residual,
        intent_prediction[validation],
        intent_stats,
        gate=1.0,
        residual_cap=residual_cap,
        beta=DEFAULT_TEMPLATE_BETA,
    )

    no_intent_model = fit_ridge(zfeatures[discovery], flattened_chunks(arrays["residual_chunk"][discovery]))
    no_intent_residual = predict_ridge(no_intent_model, zfeatures[validation]).reshape(-1, CHUNK_SIZE, ACTION_DIM)
    no_intent_prediction = arrays["base_chunk"][validation] + np.clip(no_intent_residual, -abs(residual_cap), abs(residual_cap))

    lora_features_discovery = np.concatenate([zfeatures[discovery], flattened_chunks(arrays["base_chunk"][discovery])], axis=1)
    lora_features_validation = np.concatenate([zfeatures[validation], flattened_chunks(arrays["base_chunk"][validation])], axis=1)
    lora_model = fit_ridge(lora_features_discovery, flattened_chunks(arrays["residual_chunk"][discovery]))
    lora_residual = predict_ridge(lora_model, lora_features_validation).reshape(-1, CHUNK_SIZE, ACTION_DIM)
    lora_prediction = arrays["base_chunk"][validation] + np.clip(lora_residual, -abs(residual_cap), abs(residual_cap))

    target_validation = arrays["chunk"][validation]
    base_validation = arrays["base_chunk"][validation]
    ccif_vs_prior = prediction_metrics(
        flattened_chunks(ccif_prediction),
        flattened_chunks(prior_prediction),
        flattened_chunks(target_validation),
        delta=ACTION_HUBER_DELTA,
    )
    ccif_vs_ablation = prediction_metrics(
        flattened_chunks(ccif_prediction),
        flattened_chunks(no_intent_prediction),
        flattened_chunks(target_validation),
        delta=ACTION_HUBER_DELTA,
    )
    ccif_vs_endpoint = prediction_metrics(
        flattened_chunks(ccif_prediction),
        flattened_chunks(endpoint_action_prediction),
        flattened_chunks(target_validation),
        delta=ACTION_HUBER_DELTA,
    )
    ccif_vs_base = prediction_metrics(
        flattened_chunks(ccif_prediction),
        flattened_chunks(base_validation),
        flattened_chunks(target_validation),
        delta=ACTION_HUBER_DELTA,
    )
    delta_summary = action_delta_summary(base_validation, ccif_prediction)
    action_validity_ok = bool(
        np.isfinite(base_validation).all()
        and np.isfinite(prior_prediction).all()
        and np.isfinite(ccif_prediction).all()
        and np.isfinite(no_intent_prediction).all()
        and np.isfinite(lora_prediction).all()
    )

    ccif_endpoint_relative = ccif_vs_endpoint["relative_huber_improvement"]
    ccif_endpoint_absolute = ccif_vs_endpoint["absolute_huber_improvement"]
    endpoint_only_explains_ccif = not (
        ccif_endpoint_relative >= HEADROOM_RELATIVE_GATE
        or ccif_endpoint_absolute >= HEADROOM_ABSOLUTE_HUBER_GATE
    )

    task_counts = {
        str(task): int(np.sum(arrays["task_identity"][validation] == task))
        for task in sorted(set(str(value) for value in arrays["task_identity"][validation]))
    }
    maximum_validation_task_fraction = max(task_counts.values()) / max(1, int(validation.sum()))

    arrays["ccif_full_chunk"] = np.zeros_like(arrays["chunk"], dtype=np.float64)
    arrays["coarse_to_control_proxy_chunk"] = np.zeros_like(arrays["chunk"], dtype=np.float64)
    arrays["ccif_no_intent_ablation_chunk"] = np.zeros_like(arrays["chunk"], dtype=np.float64)
    arrays["standard_lora_proxy_chunk"] = np.zeros_like(arrays["chunk"], dtype=np.float64)
    arrays["ccif_full_chunk"][validation_indices] = ccif_prediction
    arrays["coarse_to_control_proxy_chunk"][validation_indices] = prior_prediction
    arrays["ccif_no_intent_ablation_chunk"][validation_indices] = no_intent_prediction
    arrays["standard_lora_proxy_chunk"][validation_indices] = lora_prediction

    models = {
        "zscore": zscore,
        "intent_stats": intent_stats,
        "task_phase_mean": task_phase_model,
        "intent_probe": intent_probe,
        "intent_to_action": intent_to_action,
        "full_residual": full_residual_model,
        "no_intent_residual": no_intent_model,
        "standard_lora_proxy": lora_model,
        "residual_cap": residual_cap,
        "template_beta": DEFAULT_TEMPLATE_BETA,
        "discovery_feature_keys": arrays["feature_key"][discovery].tolist(),
        "validation_feature_keys": arrays["feature_key"][validation].tolist(),
    }
    audit = {
        "model_hash": canonical_json_sha256(models),
        "discovery_row_count": int(len(discovery_indices)),
        "validation_row_count": int(len(validation_indices)),
        "intent_dimension": INTENT_DIM,
        "waypoint_indices": list(WAYPOINT_INDICES),
        "collapsed_intent_component_count": int(intent_stats["collapsed_intent_component_count"]),
        "validation_collapsed_intent_component_count": int(validation_stats["collapsed_intent_component_count"]),
        "labels_noncollapsed_discovery": int(intent_stats["collapsed_intent_component_count"]) == 0,
        "labels_noncollapsed_validation": int(validation_stats["collapsed_intent_component_count"]) == 0,
        "intent_consistency": intent_consistency_summary(raw_intent[discovery]),
        "task_phase_mean_intent_huber": mean_huber(task_phase_prediction, normalized_intent[validation]),
        "endpoint_only_intent_huber": endpoint_intent_huber,
        "deployment_intent_probe_huber": mean_huber(intent_prediction[validation], normalized_intent[validation]),
        "intent_probe_relative_improvement": intent_vs_task_phase["relative_huber_improvement"],
        "intent_probe_absolute_huber": intent_vs_task_phase["absolute_huber_improvement"],
        "intent_probe_beats_task_phase_mean": bool(
            intent_vs_task_phase["relative_huber_improvement"] >= HEADROOM_RELATIVE_GATE
            or intent_vs_task_phase["absolute_huber_improvement"] >= HEADROOM_ABSOLUTE_HUBER_GATE
        ),
        "endpoint_only_action_huber": mean_huber(endpoint_action_prediction, target_validation, delta=ACTION_HUBER_DELTA),
        "endpoint_only_explains_ccif": bool(endpoint_only_explains_ccif),
        "base_to_expert_huber": mean_huber(base_validation, target_validation, delta=ACTION_HUBER_DELTA),
        "coarse_to_control_proxy_huber": mean_huber(prior_prediction, target_validation, delta=ACTION_HUBER_DELTA),
        "ccif_full_huber": mean_huber(ccif_prediction, target_validation, delta=ACTION_HUBER_DELTA),
        "ccif_no_intent_ablation_huber": mean_huber(no_intent_prediction, target_validation, delta=ACTION_HUBER_DELTA),
        "standard_lora_proxy_huber": mean_huber(lora_prediction, target_validation, delta=ACTION_HUBER_DELTA),
        "ccif_beats_prior_relative": ccif_vs_prior["relative_huber_improvement"],
        "ccif_beats_prior_absolute_huber": ccif_vs_prior["absolute_huber_improvement"],
        "ccif_beats_ablation_relative": ccif_vs_ablation["relative_huber_improvement"],
        "ccif_beats_ablation_absolute_huber": ccif_vs_ablation["absolute_huber_improvement"],
        "ccif_beats_endpoint_relative": ccif_endpoint_relative,
        "ccif_beats_endpoint_absolute_huber": ccif_endpoint_absolute,
        "ccif_vs_base": ccif_vs_base,
        "ccif_vs_prior": ccif_vs_prior,
        "ccif_vs_ablation": ccif_vs_ablation,
        "ccif_vs_endpoint_only": ccif_vs_endpoint,
        "residual_cap_discovery_quantile": residual_cap,
        "template_beta": DEFAULT_TEMPLATE_BETA,
        "action_delta_summary": delta_summary,
        "residual_activation_fraction": float(np.mean(np.abs(ccif_prediction - base_validation) > 1e-12)),
        "action_validity_ok": action_validity_ok,
        "feature_action_proprio_finite_aligned": bool(
            np.isfinite(arrays["raw_feature"]).all()
            and np.isfinite(arrays["proprio"]).all()
            and np.isfinite(arrays["chunk"]).all()
            and np.isfinite(arrays["base_chunk"]).all()
            and bool(arrays["action_finite"].all())
        ),
        "minimum_discovery_windows": int(discovery.sum()),
        "minimum_validation_windows": int(validation.sum()),
        "all_tasks_reported": len(task_counts) == TASK_COUNT,
        "validation_task_counts": task_counts,
        "maximum_validation_task_fraction": float(maximum_validation_task_fraction),
        "demo_action_validity_ok": action_validity_ok,
    }
    return models, audit, arrays


def _identity_audit(paths: Mapping[str, Path], arrays: Mapping[str, np.ndarray], models: Mapping[str, Any]) -> dict[str, Any]:
    base = np.asarray(arrays["base_chunk"][: min(8, len(arrays["base_chunk"]))], dtype=np.float64)
    zeros = np.zeros_like(base)
    zero_intent = np.zeros((len(base), INTENT_DIM), dtype=np.float64)
    output = apply_ccif_residual(
        base,
        zeros,
        zero_intent,
        models["intent_stats"],
        gate=0.0,
        residual_cap=float(models["residual_cap"]),
        beta=0.0,
    )
    identity_max = float(np.max(np.abs(output - base)))
    checkpoint = paths["identity_dir"] / "ccif_initialized_identity.json"
    payload = {
        "method": "CCIF-VLA",
        "initialized_residual_zero": True,
        "initialized_gate_zero": True,
        "initialized_beta": 0.0,
        "identity_max_abs_error": identity_max,
        "model_hash": canonical_json_sha256(
            {
                "intent_stats": models["intent_stats"],
                "residual_cap": models["residual_cap"],
                "template_beta": 0.0,
            }
        ),
    }
    _write_json(checkpoint, payload)
    reloaded = _read_json(checkpoint)
    return {
        "identity_max_abs_error": identity_max,
        "checkpoint_path": str(checkpoint),
        "checkpoint_reload_ok": reloaded.get("model_hash") == payload["model_hash"] and identity_max <= 1e-6,
        "base_action": base[0].tolist() if len(base) else [],
        "ours_action": output[0].tolist() if len(output) else [],
        "residual_norm": 0.0,
        "gate_value": 0.0,
        "dimensions_changed": [],
        "activation_context": "initialized zero residual, zero gate, beta zero",
    }


def _gradient_smoke(arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    try:
        import torch
        import torch.nn.functional as F
    except Exception as exc:  # pragma: no cover - depends on local torch install
        return {
            "finite_objectives_and_gradients": False,
            "ccif_gradient_nonzero": False,
            "frozen_parameter_gradient_count": 0,
            "weighted_gradient_norm_ratio_max": 1.0e12,
            "error": f"{type(exc).__name__}: {exc}",
        }

    discovery = np.asarray(arrays["partition"] == "discovery")
    indices = np.flatnonzero(discovery)[:16]
    x = torch.tensor(np.asarray(arrays["zfeature"][indices], dtype=np.float32))
    c_target = torch.tensor(np.asarray(arrays["normalized_intent"][indices], dtype=np.float32))
    base = torch.tensor(np.asarray(arrays["base_chunk"][indices], dtype=np.float32))
    expert = torch.tensor(np.asarray(arrays["chunk"][indices], dtype=np.float32))

    torch.manual_seed(SEED)
    intent_head = torch.nn.Linear(x.shape[1], INTENT_DIM)
    residual_head = torch.nn.Linear(x.shape[1] + INTENT_DIM, CHUNK_SIZE * ACTION_DIM)
    gate_head = torch.nn.Linear(x.shape[1] + INTENT_DIM, CHUNK_SIZE)
    frozen_base_probe = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
    params = list(intent_head.parameters()) + list(residual_head.parameters()) + list(gate_head.parameters())

    def forward() -> tuple[Any, Any, Any]:
        c_hat = intent_head(x)
        conditioned = torch.cat([x, c_hat], dim=1)
        residual = residual_head(conditioned).reshape(-1, CHUNK_SIZE, ACTION_DIM)
        gate = torch.sigmoid(gate_head(conditioned)).reshape(-1, CHUNK_SIZE, 1) * 0.1
        ours = base + gate * residual + frozen_base_probe * 0.0
        return c_hat, residual, ours

    def grad_norm(loss: Any) -> float:
        for param in params:
            param.grad = None
        loss.backward(retain_graph=True)
        total = 0.0
        for param in params:
            if param.grad is not None:
                total += float(torch.sum(param.grad.detach() ** 2).cpu())
        return float(total ** 0.5)

    c_hat, _, ours = forward()
    intent_loss = F.smooth_l1_loss(c_hat, c_target, beta=1.0)
    action_loss = F.smooth_l1_loss(ours, expert, beta=ACTION_HUBER_DELTA)
    intent_norm = grad_norm(intent_loss)
    action_norm = grad_norm(action_loss)
    for param in params:
        param.grad = None
    total_loss = intent_loss * 0.3 + action_loss
    total_loss.backward()
    trainable_norms = [
        float(torch.linalg.vector_norm(param.grad.detach()).cpu())
        for param in params
        if param.grad is not None and torch.isfinite(param.grad).all()
    ]
    finite = bool(
        torch.isfinite(intent_loss)
        and torch.isfinite(action_loss)
        and all(torch.isfinite(param.grad).all() for param in params if param.grad is not None)
    )
    positive_term_norms = [norm for norm in (intent_norm * 0.3, action_norm) if norm > 0.0]
    ratio = max(positive_term_norms) / max(min(positive_term_norms), 1e-12) if positive_term_norms else float("inf")
    return {
        "finite_objectives_and_gradients": finite,
        "intent_loss": float(intent_loss.detach().cpu()),
        "action_loss": float(action_loss.detach().cpu()),
        "weighted_intent_gradient_norm": float(intent_norm * 0.3),
        "weighted_action_gradient_norm": float(action_norm),
        "ccif_gradient_nonzero": bool(any(norm > 0.0 for norm in trainable_norms)),
        "frozen_parameter_gradient_count": 1 if frozen_base_probe.grad is not None else 0,
        "weighted_gradient_norm_ratio_max": float(ratio),
    }


def _data_summary(
    manifest: Sequence[Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
    model_audit: Mapping[str, Any],
    partial_audit: Mapping[str, Any],
) -> dict[str, Any]:
    partitions = {
        partition: sum(1 for row in manifest if row["partition"] == partition)
        for partition in sorted(set(str(row["partition"]) for row in manifest))
    }
    model_counts = {
        model: sum(1 for row in manifest if row["model_or_probe"] == model)
        for model in MODEL_OR_PROBE_ROWS
    }
    return {
        "manifest_partitions_model_rows": partitions,
        "manifest_policy_probe_counts": model_counts,
        "unique_observation_row_count": int(len(arrays["feature_key"])),
        "closed_loop_experiment_happened": False,
        "simulator_load_count": 0,
        "confirmatory_records_read": 0,
        "training_happened": False,
        "validation_search_happened": False,
        "reward_read_count": 0,
        "success_read_count": 0,
        "done_read_count": 0,
        "partial_audit": partial_audit,
        **dict(model_audit),
    }


def _decision_inputs(
    proposal_ok: bool,
    serializer_ok: bool,
    prior_check: Mapping[str, Any],
    manifest_ok: bool,
    data: Mapping[str, Any],
    identity: Mapping[str, Any] | None,
    gradient: Mapping[str, Any] | None,
    exception_count: int,
) -> Stage0DecisionInputs:
    identity = identity or {}
    gradient = gradient or {}
    return Stage0DecisionInputs(
        proposal_hash_ok=proposal_ok,
        serializer_preflight_ok=serializer_ok,
        official_prior_asset_check_persisted=bool(prior_check),
        manifest_integrity_ok=manifest_ok,
        source_alignment_ok=True,
        feature_action_proprio_finite_aligned=bool(data["feature_action_proprio_finite_aligned"]),
        split_integrity_ok=manifest_ok,
        minimum_discovery_windows=int(data["minimum_discovery_windows"]),
        minimum_validation_windows=int(data["minimum_validation_windows"]),
        all_tasks_reported=bool(data["all_tasks_reported"]),
        maximum_validation_task_fraction=float(data["maximum_validation_task_fraction"]),
        labels_noncollapsed_discovery=bool(data["labels_noncollapsed_discovery"]),
        labels_noncollapsed_validation=bool(data["labels_noncollapsed_validation"]),
        collapsed_intent_component_count=int(data["collapsed_intent_component_count"]),
        intent_probe_beats_task_phase_mean=bool(data["intent_probe_beats_task_phase_mean"]),
        intent_probe_relative_improvement=float(data["intent_probe_relative_improvement"]),
        intent_probe_absolute_huber=float(data["intent_probe_absolute_huber"]),
        endpoint_only_explains_ccif=bool(data["endpoint_only_explains_ccif"]),
        ccif_beats_prior_relative=float(data["ccif_beats_prior_relative"]),
        ccif_beats_prior_absolute_huber=float(data["ccif_beats_prior_absolute_huber"]),
        ccif_beats_ablation_relative=float(data["ccif_beats_ablation_relative"]),
        ccif_beats_ablation_absolute_huber=float(data["ccif_beats_ablation_absolute_huber"]),
        action_validity_ok=bool(data["action_validity_ok"]),
        identity_max_abs_error=float(identity.get("identity_max_abs_error", 0.0)),
        checkpoint_reload_ok=bool(identity.get("checkpoint_reload_ok", True)),
        finite_objectives_and_gradients=bool(gradient.get("finite_objectives_and_gradients", False)),
        ccif_gradient_nonzero=bool(gradient.get("ccif_gradient_nonzero", False)),
        frozen_parameter_gradient_count=int(gradient.get("frozen_parameter_gradient_count", 0)),
        weighted_gradient_norm_ratio_max=float(gradient.get("weighted_gradient_norm_ratio_max", float("inf"))),
        reward_read_count=0,
        success_read_count=0,
        done_read_count=0,
        confirmatory_records_read=0,
        closed_loop_experiment_happened=False,
        simulator_load_count=0,
        training_happened=False,
        validation_search_happened=False,
        exception_count=int(exception_count),
    )


def _result_markdown(result: Mapping[str, Any]) -> str:
    data = result["data_audit"]
    gradient = result.get("gradient") or {}
    identity = result.get("identity") or {}
    return "\n".join(
        [
            "# CCIF-VLA Stage 0 Result",
            "",
            f"Final decision: `{result['final_decision']}`.",
            "",
            f"Rows: `{result['completed_model_row_count']} / {result['planned_model_row_count']}` model rows.",
            f"Unique observation rows: `{data['unique_observation_row_count']}`.",
            f"Coarse-to-Control prior label: `{result['official_prior_asset_check']['policy_2_label']}`.",
            f"Intent dimension: `{data['intent_dimension']}`; waypoints: `{data['waypoint_indices']}`.",
            f"Intent probe Huber: `{data['deployment_intent_probe_huber']}`.",
            f"Task/phase mean intent Huber: `{data['task_phase_mean_intent_huber']}`.",
            f"Endpoint-only intent Huber: `{data['endpoint_only_intent_huber']}`.",
            f"Base / prior / CCIF / no-intent / LoRA-proxy Huber: `{data['base_to_expert_huber']} / {data['coarse_to_control_proxy_huber']} / {data['ccif_full_huber']} / {data['ccif_no_intent_ablation_huber']} / {data['standard_lora_proxy_huber']}`.",
            f"CCIF minus prior relative / absolute Huber gain: `{data['ccif_beats_prior_relative']} / {data['ccif_beats_prior_absolute_huber']}`.",
            f"CCIF minus ablation relative / absolute Huber gain: `{data['ccif_beats_ablation_relative']} / {data['ccif_beats_ablation_absolute_huber']}`.",
            f"Identity max abs error: `{identity.get('identity_max_abs_error')}`.",
            f"Gradient finite/nonzero/ratio: `{gradient.get('finite_objectives_and_gradients')} / {gradient.get('ccif_gradient_nonzero')} / {gradient.get('weighted_gradient_norm_ratio_max')}`.",
            f"Exceptions: `{result['exception_count']}`.",
            "",
            "No simulator rollout, reward, success, done flag, validation search, or confirmatory identity was used.",
        ]
    ) + "\n"


def _adjudication_markdown(result: Mapping[str, Any]) -> str:
    return "\n".join(
        [
            "# CCIF-VLA Stage 0 Adjudication",
            "",
            f"Decision: `{result['final_decision']}`.",
            "",
            f"Bounded validation allowed: `{result['bounded_validation_allowed']}`.",
            f"Valid scientific result: `{result['valid_scientific_result']}`.",
            "",
            "The frozen Stage 0 gates were applied without changing tasks, demonstration identities, memory construction, closest-prior proxy, ablation, simple LoRA proxy, thresholds, or confirmatory-test access.",
            "Any stop here is a development audit result, not a closed-loop scientific kill.",
        ]
    ) + "\n"


def _preflight(paths: Mapping[str, Path], started_unix: float) -> dict[str, Any]:
    registry = _read_json(RESOURCE_REGISTRY) if RESOURCE_REGISTRY.is_file() else {"intervals": []}
    partial_parse_error = None
    partial_summary = None
    if paths["partial"].is_file():
        try:
            partial = _read_json(paths["partial"])
            partial_summary = {
                "planned_model_row_count": partial.get("planned_model_row_count"),
                "completed_model_row_count": partial.get("completed_model_row_count"),
                "exception_count": partial.get("exception_count"),
            }
        except Exception as exc:
            partial_parse_error = f"{type(exc).__name__}: {exc}"
    preflight = {
        "method": "CCIF-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "active_linux_workers": _active_linux_workers(),
        "partial_summary": partial_summary,
        "partial_parse_error": partial_parse_error,
        "result_absent": not paths["result_json"].exists(),
        "resource_evidence": _resource_evidence(registry, started_unix),
        "timing_throughput_resource_evidence_eligible_for_paper": False,
        "passed": partial_parse_error is None,
        "written_at": _utc_now(),
    }
    _write_json(paths["preflight"], preflight)
    return preflight


def run(args: argparse.Namespace, paths: Mapping[str, Path], state: dict[str, Any]) -> dict[str, Any]:
    started_unix = time.time()
    _set_offline_environment()
    paths["report"].mkdir(parents=True, exist_ok=True)
    paths["run"].mkdir(parents=True, exist_ok=True)
    preflight = _preflight(paths, started_unix)
    if not preflight["passed"]:
        raise RuntimeError(f"CCIF Stage 0 preflight failed: {preflight}")
    if not paths["serializer_preflight"].is_file():
        _serializer_preflight(paths["serializer_preflight"])
    serializer = _read_json(paths["serializer_preflight"])
    serializer_ok = bool(
        serializer.get("passed") and canonical_json_sha256(serializer["fixture"]) == serializer.get("fixture_hash")
    )
    if not serializer_ok:
        raise RuntimeError("foreground CCIF serializer preflight is absent or invalid")
    prior_check = _official_prior_asset_check(paths["official_prior_asset_check"])
    proposal_ok = _proposal_hash_text() == PROPOSAL_HASH
    if not proposal_ok:
        raise RuntimeError("frozen CCIF proposal hash mismatch")

    state.update({"phase": "load_policy", "status": "running"})
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
    policy, preprocessor, postprocessor, _ = _load_policy_and_processors_for_ccif(paths["checkpoint"])
    action_stats = _action_stats(postprocessor)
    action_semantics = _write_action_semantics(paths["action_semantics"], action_stats)

    state.update({"phase": "manifest"})
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
    rows, sources = _build_manifest(paths["data_root"])
    manifest_payload = {
        "method": "CCIF-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "policy_probe": POLICY_PROBE,
        "model_or_probe_rows": list(MODEL_OR_PROBE_ROWS),
        "planned_model_row_count": len(rows),
        "unique_observation_row_count": len(_unique_manifest_rows(rows)),
        "rows": rows,
        "sources": sources,
        "frozen_data": {
            "discovery_demos": "0..7",
            "validation_demos": "8..9",
            "confirmatory_identities_read": 0,
        },
    }
    manifest_hash = canonical_json_sha256(manifest_payload)
    manifest_payload["manifest_hash"] = manifest_hash
    manifest_payload["created_at"] = _utc_now()
    _write_json(paths["manifest"], manifest_payload)
    parsed = _read_json(paths["manifest"])
    parsed_without_hash = dict(parsed)
    parsed_without_hash.pop("manifest_hash")
    parsed_without_hash.pop("created_at", None)
    if canonical_json_sha256(parsed_without_hash) != manifest_hash:
        raise RuntimeError("persisted CCIF manifest hash did not reproduce")
    manifest_audit = validate_manifest(rows, [{"row_key": row["row_key"]} for row in rows])

    partial_rows, prior_exception_count, prior_last_exception = _load_resume(paths["partial"], rows, manifest_hash)
    resumed_count = len(partial_rows)
    completed = {str(row["row_key"]) for row in partial_rows}
    state.update(
        {
            "phase": "partial_rows",
            "planned_model_row_count": len(rows),
            "completed_model_row_count": len(partial_rows),
            "resumed_model_row_count": resumed_count,
        }
    )
    _write_json(
        paths["partial"],
        _partial_payload(manifest_hash, len(rows), partial_rows, exception_count=prior_exception_count, last_exception=prior_last_exception),
    )
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})

    for row in rows:
        if str(row["row_key"]) in completed:
            continue
        feature_path, feature = _load_or_extract_feature_for_ccif(policy, paths, row)
        base_path, base_chunk = _base_chunk_for_feature(policy, preprocessor, postprocessor, paths, row)
        partial_rows.append(_partial_row(row, feature_path, feature, base_path, base_chunk))
        completed.add(str(row["row_key"]))
        state["completed_model_row_count"] = len(partial_rows)
        _write_json(
            paths["partial"],
            _partial_payload(manifest_hash, len(rows), partial_rows, exception_count=prior_exception_count, last_exception=prior_last_exception),
        )
        _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
        if len(partial_rows) % 112 == 0 or len(partial_rows) == len(rows):
            print(f"[ccif-stage0] rows {len(partial_rows)}/{len(rows)}", flush=True)

    ordered = {str(row["row_key"]): row for row in partial_rows}
    partial_rows = [ordered[str(row["row_key"])] for row in rows]
    partial_audit = validate_manifest(rows, partial_rows)
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

    state.update({"phase": "offline_models"})
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
    arrays = _materialize_arrays(rows, partial_rows)
    models, model_audit, arrays = _fit_ccif_models(arrays)
    data = _data_summary(rows, arrays, model_audit, partial_audit)

    state.update({"phase": "identity_and_gradient"})
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
    identity = _identity_audit(paths, arrays, models)
    gradient = _gradient_smoke(arrays)
    current_exception_count = 0
    decision_inputs = _decision_inputs(
        proposal_ok, serializer_ok, prior_check, manifest_ok, data, identity, gradient, current_exception_count
    )
    decision = classify_stage0(decision_inputs)
    decision_input_payload = asdict(decision_inputs)
    validation_payload = {
        "method": "CCIF-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "decision_inputs": decision_input_payload,
        "decision": decision,
        "data_audit": data,
        "identity": identity,
        "gradient": gradient,
        "written_at": _utc_now(),
    }
    _write_json(paths["validation"], validation_payload)

    result = {
        "method": "CCIF-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "final_decision": decision,
        "result_created_at": _utc_now(),
        "completed_model_row_count": len(partial_rows),
        "planned_model_row_count": len(rows),
        "exception_count": int(current_exception_count),
        "resume_exception_count": int(prior_exception_count),
        "resume_last_exception": prior_last_exception,
        "implementation_blocker_repaired_before_final_result": bool(prior_exception_count),
        "manifest_row_count": partial_audit["manifest_row_count"],
        "partial_row_count": partial_audit["partial_row_count"],
        "official_prior_asset_check": prior_check,
        "action_semantics": action_semantics,
        "manifest_audit": manifest_audit,
        "partial_audit": partial_audit,
        "data_audit": data,
        "identity": identity,
        "gradient": gradient,
        "decision_inputs": decision_input_payload,
        "bounded_validation_allowed": decision == "CCIF_STAGE_0_PASS_TO_BOUNDED_VALIDATION",
        "valid_scientific_result": False,
        "closed_loop_experiment_happened": False,
        "simulator_load_count": 0,
        "confirmatory_records_read": 0,
        "training_happened": False,
        "validation_search_happened": False,
        "intent_dimension": INTENT_DIM,
        "waypoint_indices": list(WAYPOINT_INDICES),
        "collapsed_intent_component_count": data["collapsed_intent_component_count"],
        "task_phase_mean_intent_huber": data["task_phase_mean_intent_huber"],
        "endpoint_only_intent_huber": data["endpoint_only_intent_huber"],
        "deployment_intent_probe_huber": data["deployment_intent_probe_huber"],
        "intent_probe_relative_improvement": data["intent_probe_relative_improvement"],
        "base_to_expert_huber": data["base_to_expert_huber"],
        "coarse_to_control_proxy_huber": data["coarse_to_control_proxy_huber"],
        "ccif_full_huber": data["ccif_full_huber"],
        "ccif_no_intent_ablation_huber": data["ccif_no_intent_ablation_huber"],
        "ccif_beats_prior_relative": data["ccif_beats_prior_relative"],
        "ccif_beats_prior_absolute_huber": data["ccif_beats_prior_absolute_huber"],
        "ccif_beats_ablation_relative": data["ccif_beats_ablation_relative"],
        "ccif_beats_ablation_absolute_huber": data["ccif_beats_ablation_absolute_huber"],
        "action_validity_ok": data["action_validity_ok"],
        "identity_max_abs_error": identity["identity_max_abs_error"],
        "checkpoint_reload_ok": identity["checkpoint_reload_ok"],
        "finite_objectives_and_gradients": gradient.get("finite_objectives_and_gradients", False),
        "ccif_gradient_nonzero": gradient.get("ccif_gradient_nonzero", False),
        "frozen_parameter_gradient_count": gradient.get("frozen_parameter_gradient_count", 0),
        "weighted_gradient_norm_ratio_max": gradient.get("weighted_gradient_norm_ratio_max", float("inf")),
        "residual_activation_fraction": data["residual_activation_fraction"],
        "timing_throughput_resource_evidence_eligible_for_paper": False,
        "resource_evidence": preflight["resource_evidence"],
        "proposal_hash_ok": proposal_ok,
        "serializer_preflight_ok": serializer_ok,
        "official_prior_asset_check_persisted": bool(prior_check),
        "manifest_json_parsed": True,
        "partial_json_parsed": True,
        "result_decision_recomputed": classify_stage0(decision_inputs),
        **partial_audit,
    }
    _write_json(paths["result_json"], result)
    _write_text(paths["result_md"], _result_markdown(result))
    _write_text(paths["adjudication"], _adjudication_markdown(result))
    _write_json(
        paths["partial"],
        _partial_payload(manifest_hash, len(rows), partial_rows, exception_count=current_exception_count, last_exception=None),
    )
    state.update({"status": "completed", "phase": "complete", "completed_model_row_count": len(partial_rows)})
    _write_json(paths["status"], {**state, "completed_at": _utc_now(), "final_decision": decision})
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now(), "final_decision": decision})
    del policy, preprocessor, postprocessor
    gc.collect()
    return result


def _write_blocker(paths: Mapping[str, Path], state: Mapping[str, Any], exc: BaseException) -> None:
    detail = traceback.format_exc()
    manifest_hash = None
    planned = None
    rows: list[dict[str, Any]] = []
    previous_exceptions = 0
    if paths["manifest"].is_file():
        try:
            manifest = _read_json(paths["manifest"])
            manifest_hash = manifest.get("manifest_hash")
            planned = manifest.get("planned_model_row_count")
        except Exception:
            pass
    if paths["partial"].is_file():
        try:
            partial = _read_json(paths["partial"])
            rows = list(partial.get("rows") or [])
            previous_exceptions = int(partial.get("exception_count") or 0)
            manifest_hash = partial.get("manifest_hash", manifest_hash)
            planned = partial.get("planned_model_row_count", planned)
        except Exception:
            rows = []
    _write_json(
        paths["partial"],
        _partial_payload(
            manifest_hash,
            planned,
            rows,
            exception_count=previous_exceptions + 1,
            last_exception=f"{type(exc).__name__}: {exc}",
        ),
    )
    blocker = {
        "method": "CCIF-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "status": "implementation_blocker",
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "traceback": detail,
        "state": dict(state),
        "partial_rows_preserved": len(rows),
        "written_at": _utc_now(),
    }
    _write_json(paths["blocker"], blocker)
    _write_json(
        paths["status"],
        {
            **state,
            "status": "failed",
            "phase": "implementation_blocker",
            "exception_count": previous_exceptions + 1,
            "completed_model_row_count": len(rows),
            "updated_at": _utc_now(),
        },
    )
    _write_json(
        paths["heartbeat"],
        {
            **state,
            "status": "failed",
            "phase": "implementation_blocker",
            "exception_count": previous_exceptions + 1,
            "completed_model_row_count": len(rows),
            "updated_at": _utc_now(),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", default="audit", choices=["audit"])
    parser.add_argument("--serializer-preflight", action="store_true")
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "ccif_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "ccif_vla" / "stage0"))
    parser.add_argument("--checkpoint", default=str(_asset_path("checkpoints", "smolvla_libero")))
    parser.add_argument("--libero-data-root", default=str(_asset_path("data", "libero")))
    args = parser.parse_args(argv)
    paths = _paths(args)
    if args.serializer_preflight:
        result = _serializer_preflight(paths["serializer_preflight"])
        print(f"CCIF serializer preflight passed: {paths['serializer_preflight']} {result['fixture_hash']}")
        return 0

    state: dict[str, Any] = {
        "method": "CCIF-VLA",
        "stage": "0",
        "proposal_hash": PROPOSAL_HASH,
        "pid": os.getpid(),
        "status": "starting",
        "phase": "startup",
        "started_at": _utc_now(),
        "local_started_iso": datetime.now(timezone.utc).isoformat(),
    }
    _write_text(paths["pid"], f"{os.getpid()}\n")
    _write_json(paths["status"], state)
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
    exit_code = 0
    try:
        result = run(args, paths, state)
        print(json.dumps({"final_decision": result["final_decision"], "result": str(paths["result_json"])}))
    except BaseException as exc:
        exit_code = 1
        _write_blocker(paths, state, exc)
        print(f"CCIF Stage 0 failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        traceback.print_exc()
    _write_text(paths["exit_code"], f"{exit_code}\n")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
