"""Run the frozen HASTE-VLA Stage 0A data, headroom, probe, and identity audit."""

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
    _preprocess,
    _raw_sample,
    _set_offline_environment,
)
from scripts.run_pcav_vla_stage0 import _postprocess_chunk  # noqa: E402
from tca_map.smolvla.haste_vla import (  # noqa: E402
    ACTION_DIM,
    ARM_DIM,
    EVENT_HORIZONS,
    PROPOSAL_HASH,
    Stage0ADecisionInputs,
    canonical_json_sha256,
    classify_stage0a,
    construct_event_label,
    displacement_statistics,
    event_row_key,
    event_stratum,
    fit_constant_hazard,
    frame_key,
    hazard_nll_from_probabilities,
    huber_loss,
    normalize_displacement,
    offset_quintile,
    validate_manifest,
)


SEED = 20262200
MODEL_ROWS_PER_HORIZON_STRATUM_TASK_SPLIT = 8
PROBE_STEPS = 100
PROBE_LR = 1e-3
PROBE_WEIGHT_DECAY = 1e-4
PROPOSAL_FILE = REPO_ROOT / "reports" / "haste_vla" / "researcher_proposal.md"
PROPOSAL_HASH_FILE = REPO_ROOT / "reports" / "haste_vla" / "proposal_hash.txt"

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


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"cannot serialize {type(value).__name__}")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


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
        "head_checkpoint": run / "identity_heads.pt",
        "data_root": Path(args.data_root),
        "checkpoint": Path(args.checkpoint),
        "pid": report / "stage_0a_pid.txt",
        "heartbeat": report / "stage_0a_heartbeat.json",
        "status": report / "stage_0a_status.json",
        "preflight": report / "stage_0a_preflight.json",
        "manifest": report / "stage_0a_manifest.json",
        "partial": report / "stage_0a_partial.json",
        "result_json": report / "stage_0a_result.json",
        "result_md": report / "stage_0a_result.md",
        "validation": report / "stage_0a_validation.json",
        "blocker": report / "stage_0a_implementation_blocker.json",
    }


def _heartbeat_loop(path: Path, state: dict[str, Any], stop: threading.Event) -> None:
    while not stop.wait(10.0):
        _write_json(path, {**state, "updated_at": _utc_now()})


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
        raise RuntimeError("deterministic model-row sampler produced duplicate indices")
    return [rows[index] for index in indices]


def _select_model_rows(rows: Sequence[dict[str, Any]]) -> set[str]:
    groups: dict[tuple[str, str, int, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["partition"]),
            str(row["task_identity"]),
            int(row["event_horizon"]),
            str(row["event_stratum"]),
        )
        groups.setdefault(key, []).append(row)
    selected: set[str] = set()
    for key in sorted(groups):
        ordered = sorted(groups[key], key=lambda row: (int(row["demo_id"]), int(row["frame_index"])))
        for row in _evenly_spaced(ordered, MODEL_ROWS_PER_HORIZON_STRATUM_TASK_SPLIT):
            selected.add(str(row["event_row_key"]))
    return selected


def _build_manifest(data_root: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    import h5py

    rows: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    uncensored_discovery: list[np.ndarray] = []
    for task_index, (suite, task_identity, relative) in enumerate(TASK_SOURCES):
        source = data_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        source_hash = _edge_hash(source)
        with h5py.File(source, "r") as handle:
            data = handle["data"]
            language = _problem_language(data)
            demo_reports = []
            for demo_id in range(10):
                demo_key = f"demo_{demo_id}"
                if demo_key not in data:
                    raise KeyError(f"missing {demo_key} in {source}")
                demo = data[demo_key]
                actions = np.asarray(demo["actions"], dtype=np.float64)
                observations = demo["obs"]
                required = (
                    "agentview_rgb",
                    "eye_in_hand_rgb",
                    "ee_states",
                    "gripper_states",
                )
                if actions.ndim != 2 or actions.shape[1] != ACTION_DIM or not np.isfinite(actions).all():
                    raise ValueError(f"invalid action array {actions.shape} in {source}:{demo_key}")
                if any(name not in observations for name in required):
                    raise KeyError(f"missing required observation in {source}:{demo_key}")
                if any(len(observations[name]) != len(actions) for name in required):
                    raise ValueError(f"observation/action length mismatch in {source}:{demo_key}")
                observation_finite = {}
                for name in required:
                    dataset = observations[name]
                    if dataset.dtype.kind in "iu":
                        observation_finite[name] = True
                    else:
                        observation_finite[name] = bool(np.isfinite(np.asarray(dataset)).all())
                if not all(observation_finite.values()):
                    raise ValueError(f"nonfinite observation in {source}:{demo_key}")
                partition = "discovery" if demo_id <= 7 else "validation"
                demo_reports.append(
                    {
                        "demo_id": demo_id,
                        "partition": partition,
                        "length": len(actions),
                        "actions_finite": True,
                        "observation_finite": observation_finite,
                    }
                )
                for frame in range(len(actions) - 1):
                    for horizon in EVENT_HORIZONS:
                        label = construct_event_label(actions, frame, horizon)
                        displacement = label["relative_displacement"]
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
                            "event_horizon": horizon,
                            "valid_interval_count": label["valid_interval_count"],
                            "likelihood_term_count": label["likelihood_term_count"],
                            "transition_offset": label["transition_offset"],
                            "censored": label["censored"],
                            "censor_reason": label["censor_reason"],
                            "event_target": label["event_target"].tolist(),
                            "survival_mask": label["survival_mask"].tolist(),
                            "relative_displacement": None if displacement is None else displacement.tolist(),
                        }
                        row["event_stratum"] = event_stratum(row)
                        row["frame_key"] = frame_key(row)
                        row["event_row_key"] = event_row_key(row)
                        if partition == "discovery" and displacement is not None:
                            uncensored_discovery.append(np.asarray(displacement, dtype=np.float64))
                        rows.append(row)
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

    mean, std = displacement_statistics(uncensored_discovery)
    for row in rows:
        value = row["relative_displacement"]
        row["normalized_displacement"] = (
            None if value is None else normalize_displacement(value, mean, std).tolist()
        )
        row["offset_quintile"] = (
            None
            if row["transition_offset"] is None
            else offset_quintile(int(row["transition_offset"]), int(row["event_horizon"]))
        )
    selected = _select_model_rows(rows)
    for row in rows:
        row["selected_for_model_audit"] = row["event_row_key"] in selected
    normalization = {
        "discovery_relative_displacement_mean": mean,
        "discovery_relative_displacement_std": std,
        "std_floor": 1e-6,
    }
    return rows, sources, normalization


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


def _noise(row: Mapping[str, Any]) -> Any:
    import torch

    digest = hashlib.sha256(f"{SEED}|{row['frame_key']}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") % (2**63 - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn((1, 50, 32), generator=generator, dtype=torch.float32).to("cuda")


def _model_batch(preprocessor: Any, row: Mapping[str, Any]) -> dict[str, Any]:
    raw = _raw_sample(row)
    raw.pop("action", None)
    raw.pop("action_is_pad", None)
    return _preprocess(preprocessor, raw)


def _pooled_prefix(policy: Any, batch: Mapping[str, Any]) -> np.ndarray:
    import torch

    with torch.no_grad():
        images, img_masks = policy.prepare_images(batch)
        state = policy.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]
        embeddings, pad_masks, _ = policy.model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        mask = pad_masks.to(dtype=embeddings.dtype).unsqueeze(-1)
        pooled = (embeddings * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
    value = pooled[0].float().cpu().numpy()
    if value.shape != (960,) or not np.isfinite(value).all():
        raise RuntimeError(f"unexpected pooled prefix feature {value.shape}")
    return value.astype(np.float32)


def _batch_input_finite_fraction(batch: Mapping[str, Any]) -> float:
    import torch

    image_keys = sorted(
        key
        for key, value in batch.items()
        if key.startswith("observation.images.")
        and not key.endswith("_padding_mask")
        and isinstance(value, torch.Tensor)
        and value.ndim >= 4
    )
    if len(image_keys) < 2:
        raise RuntimeError(f"expected at least two processed image inputs, received {image_keys}")
    required = (*image_keys, "observation.state")
    missing = [key for key in required if key not in batch]
    if missing:
        raise RuntimeError(f"missing required processed inputs: {missing}")
    fractions = []
    for key in required:
        value = batch[key]
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"processed input {key} is not a tensor")
        fractions.append(float(torch.isfinite(value).float().mean().item()))
    fraction = min(fractions)
    if fraction != 1.0:
        raise RuntimeError(f"nonfinite processed input fraction {fraction}")
    return fraction


def _flow_vector(policy: Any, preprocessor: Any, row: Mapping[str, Any]) -> Any:
    import torch
    from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks

    core = policy.get_base_model() if hasattr(policy, "get_base_model") else policy
    batch = _model_batch(preprocessor, row)
    _batch_input_finite_fraction(batch)
    with torch.no_grad():
        images, img_masks = core.prepare_images(batch)
        state = core.prepare_state(batch)
        lang_tokens = batch["observation.language.tokens"]
        lang_masks = batch["observation.language.attention_mask"]
        model = core.model
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
        _, past_key_values = model.vlm_with_expert.forward(
            attention_mask=prefix_att_2d_masks,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=model.config.use_cache,
            fill_kv_cache=True,
        )
        time = torch.ones(state.shape[0], dtype=torch.float32, device=state.device)
        flow = model.denoise_step(
            x_t=_noise(row),
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            timestep=time,
        )
    return flow.detach().float().cpu()


def _predict_chunk(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    row: Mapping[str, Any],
) -> tuple[Any, np.ndarray, dict[str, Any]]:
    import torch

    batch = _model_batch(preprocessor, row)
    if hasattr(policy, "reset"):
        policy.reset()
    with torch.no_grad():
        native = policy.predict_action_chunk(_clone_batch(batch), noise=_noise(row))
    queue = _postprocess_chunk(native, postprocessor)
    return native.detach().float().cpu(), queue.astype(np.float32), batch


def _extract_feature(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    row: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    import h5py

    native, queue, batch = _predict_chunk(policy, preprocessor, postprocessor, row)
    input_finite_fraction = _batch_input_finite_fraction(batch)
    pooled = _pooled_prefix(policy, batch)
    with h5py.File(str(row["source_path"]), "r") as handle:
        target = np.asarray(
            handle["data"][f"demo_{int(row['demo_id'])}"]["actions"][int(row["frame_index"])],
            dtype=np.float32,
        )
    return {
        "pooled_feature": pooled,
        "base_native_chunk": native.numpy().astype(np.float32),
        "base_action_chunk": queue.astype(np.float32),
        "target_action": target,
        "input_finite_fraction": np.asarray([input_finite_fraction], dtype=np.float32),
    }


def _row_summary(row: Mapping[str, Any], path: Path, feature: Mapping[str, np.ndarray]) -> dict[str, Any]:
    pooled = np.asarray(feature["pooled_feature"], dtype=np.float32)
    predicted = np.asarray(feature["base_action_chunk"], dtype=np.float32)[0]
    target = np.asarray(feature["target_action"], dtype=np.float32)
    delta = predicted - target
    return {
        "event_row_key": row["event_row_key"],
        "frame_key": row["frame_key"],
        "partition": row["partition"],
        "suite": row["suite"],
        "task_identity": row["task_identity"],
        "source_path": row["source_path"],
        "source_edge_sha256": row["source_edge_sha256"],
        "demo_id": row["demo_id"],
        "frame_index": row["frame_index"],
        "event_horizon": row["event_horizon"],
        "transition_offset": row["transition_offset"],
        "censored": row["censored"],
        "event_stratum": row["event_stratum"],
        "event_target": row["event_target"],
        "survival_mask": row["survival_mask"],
        "normalized_displacement": row["normalized_displacement"],
        "feature_path": str(path),
        "feature_sha256": _sha256(path),
        "feature_shape": list(pooled.shape),
        "feature_finite_fraction": float(np.isfinite(pooled).mean()),
        "input_finite_fraction": float(np.asarray(feature["input_finite_fraction"]).reshape(-1)[0]),
        "base_action": predicted,
        "target_action": target,
        "arm_error_l2": float(np.linalg.norm(delta[:ARM_DIM])),
        "arm_error_huber": huber_loss(predicted[:ARM_DIM], target[:ARM_DIM]),
        "gripper_sign_error": bool(np.sign(predicted[6]) != np.sign(target[6])),
        "base_action_finite": bool(np.isfinite(predicted).all()),
    }


def _partial_payload(
    manifest_hash: str,
    planned_count: int,
    rows: Sequence[Mapping[str, Any]],
    *,
    exception_count: int = 0,
    last_exception: str | None = None,
) -> dict[str, Any]:
    return {
        "method": "HASTE-VLA",
        "stage": "0A",
        "proposal_hash": PROPOSAL_HASH,
        "manifest_hash": manifest_hash,
        "planned_model_row_count": planned_count,
        "completed_model_row_count": len(rows),
        "completed_event_row_keys": [str(row["event_row_key"]) for row in rows],
        "exception_count": exception_count,
        "last_exception": last_exception,
        "rows": list(rows),
        "updated_at": _utc_now(),
    }


def _load_resume(
    path: Path,
    model_manifest: Sequence[Mapping[str, Any]],
    manifest_hash: str,
) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    partial = _read_json(path)
    if partial.get("proposal_hash") != PROPOSAL_HASH or partial.get("manifest_hash") != manifest_hash:
        raise RuntimeError("partial proposal or manifest hash mismatch")
    rows = list(partial.get("rows") or [])
    audit = validate_manifest(model_manifest, rows)
    if audit["duplicate_partial_key_count"] or audit["extra_partial_key_count"]:
        raise RuntimeError(f"invalid partial model keys: {audit}")
    for row in rows:
        path_value = Path(str(row["feature_path"]))
        if not path_value.is_file() or _sha256(path_value) != row["feature_sha256"]:
            raise RuntimeError(f"missing or changed feature cache for {row['event_row_key']}")
    return rows


def _torch_hazard_loss(logits: Any, rows: Sequence[Mapping[str, Any]]) -> Any:
    import torch
    import torch.nn.functional as functional

    horizon = logits.shape[1]
    targets = torch.as_tensor(
        np.asarray([row["event_target"][:horizon] for row in rows], dtype=np.float32),
        device=logits.device,
    )
    masks = torch.as_tensor(
        np.asarray([row["survival_mask"][:horizon] for row in rows], dtype=bool),
        device=logits.device,
    )
    terms = functional.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    per_row = (terms * masks).sum(dim=1) / masks.sum(dim=1).clamp_min(1)
    return per_row.mean()


def _run_probes(model_rows: Sequence[Mapping[str, Any]], full_manifest: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    features = {
        str(row["event_row_key"]): _load_feature(Path(str(row["feature_path"])))['pooled_feature']
        for row in model_rows
    }
    hazard_reports = []
    for horizon in EVENT_HORIZONS:
        discovery = [
            row for row in model_rows if row["partition"] == "discovery" and int(row["event_horizon"]) == horizon
        ]
        validation = [
            row for row in model_rows if row["partition"] == "validation" and int(row["event_horizon"]) == horizon
        ]
        x_train = torch.as_tensor(
            np.stack([features[str(row["event_row_key"])] for row in discovery]), device="cuda"
        )
        x_validation = torch.as_tensor(
            np.stack([features[str(row["event_row_key"])] for row in validation]), device="cuda"
        )
        probe = torch.nn.Linear(960, horizon).to("cuda")
        optimizer = torch.optim.AdamW(probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY)
        losses = []
        for _ in range(PROBE_STEPS):
            optimizer.zero_grad(set_to_none=True)
            loss = _torch_hazard_loss(probe(x_train), discovery)
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        with torch.no_grad():
            probabilities = torch.sigmoid(probe(x_validation)).cpu().numpy()
        probe_nll = hazard_nll_from_probabilities(probabilities, validation)
        constant = fit_constant_hazard(
            [row for row in full_manifest if row["partition"] == "discovery"], horizon
        )
        constant_nll = hazard_nll_from_probabilities(constant, validation)
        improvement = (constant_nll - probe_nll) / max(abs(constant_nll), 1e-12)
        hazard_reports.append(
            {
                "event_horizon": horizon,
                "discovery_row_count": len(discovery),
                "validation_row_count": len(validation),
                "step_count": PROBE_STEPS,
                "loss_first": losses[0],
                "loss_last": losses[-1],
                "validation_probe_nll": probe_nll,
                "validation_constant_hazard_nll": constant_nll,
                "relative_improvement": improvement,
                "constant_hazard": constant,
            }
        )

    discovery_disp = [
        row
        for row in model_rows
        if row["partition"] == "discovery" and row["normalized_displacement"] is not None
    ]
    validation_disp = [
        row
        for row in model_rows
        if row["partition"] == "validation" and row["normalized_displacement"] is not None
    ]
    x_train = torch.as_tensor(
        np.stack([features[str(row["event_row_key"])] for row in discovery_disp]), device="cuda"
    )
    y_train = torch.as_tensor(
        np.asarray([row["normalized_displacement"] for row in discovery_disp], dtype=np.float32), device="cuda"
    )
    x_validation = torch.as_tensor(
        np.stack([features[str(row["event_row_key"])] for row in validation_disp]), device="cuda"
    )
    y_validation = np.asarray(
        [row["normalized_displacement"] for row in validation_disp], dtype=np.float32
    )
    displacement_probe = torch.nn.Linear(960, ARM_DIM).to("cuda")
    optimizer = torch.optim.AdamW(
        displacement_probe.parameters(), lr=PROBE_LR, weight_decay=PROBE_WEIGHT_DECAY
    )
    losses = []
    for _ in range(PROBE_STEPS):
        optimizer.zero_grad(set_to_none=True)
        loss = functional.huber_loss(displacement_probe(x_train), y_train, delta=1.0, reduction="mean")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    with torch.no_grad():
        prediction = displacement_probe(x_validation).cpu().numpy()
    probe_huber = huber_loss(prediction, y_validation)
    mean_huber = huber_loss(np.zeros_like(y_validation), y_validation)
    displacement_improvement = (mean_huber - probe_huber) / max(abs(mean_huber), 1e-12)
    return {
        "feature_definition": "valid-mask mean of frozen SmolVLA prefix embeddings",
        "feature_width": 960,
        "hazard": hazard_reports,
        "minimum_hazard_relative_improvement": min(row["relative_improvement"] for row in hazard_reports),
        "displacement": {
            "discovery_row_count": len(discovery_disp),
            "validation_row_count": len(validation_disp),
            "step_count": PROBE_STEPS,
            "loss_first": losses[0],
            "loss_last": losses[-1],
            "validation_probe_huber": probe_huber,
            "validation_discovery_mean_huber": mean_huber,
            "relative_improvement": displacement_improvement,
        },
    }


def _headroom(model_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    def summarize(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "row_count": len(rows),
            "arm_error_l2_mean": (
                float(np.mean([row["arm_error_l2"] for row in rows])) if rows else None
            ),
            "arm_error_huber_mean": (
                float(np.mean([row["arm_error_huber"] for row in rows])) if rows else None
            ),
            "gripper_sign_error_rate": (
                float(np.mean([row["gripper_sign_error"] for row in rows])) if rows else None
            ),
        }

    rows = [row for row in model_rows if int(row["event_horizon"]) == 50]
    validation_rows = [row for row in rows if row["partition"] == "validation"]
    strata = {
        stratum: summarize([row for row in validation_rows if row["event_stratum"] == stratum])
        for stratum in ("event_near", "event_far", "censored")
    }
    by_partition = {
        partition: {
            stratum: summarize(
                [
                    row
                    for row in rows
                    if row["partition"] == partition and row["event_stratum"] == stratum
                ]
            )
            for stratum in ("event_near", "event_far", "censored")
        }
        for partition in ("discovery", "validation")
    }
    by_task = {
        task: {
            partition: {
                stratum: summarize(
                    [
                        row
                        for row in rows
                        if row["task_identity"] == task
                        and row["partition"] == partition
                        and row["event_stratum"] == stratum
                    ]
                )
                for stratum in ("event_near", "event_far", "censored")
            }
            for partition in ("discovery", "validation")
        }
        for _, task, _ in TASK_SOURCES
    }
    near = [row for row in validation_rows if row["event_stratum"] == "event_near"]
    comparison = [
        row for row in validation_rows if row["event_stratum"] in {"event_far", "censored"}
    ]
    if not near or not comparison:
        return {
            "headroom_horizon": 50,
            "strata": strata,
            "by_partition": by_partition,
            "by_task": by_task,
            "event_near_arm_relative_deficit": None,
            "event_near_gripper_sign_error_deficit": None,
            "passed": False,
            "offline_diagnostic_only": True,
        }
    near_arm = float(np.mean([row["arm_error_l2"] for row in near]))
    comparison_arm = float(np.mean([row["arm_error_l2"] for row in comparison]))
    arm_relative_deficit = (near_arm - comparison_arm) / max(abs(comparison_arm), 1e-12)
    near_gripper = float(np.mean([row["gripper_sign_error"] for row in near]))
    comparison_gripper = float(np.mean([row["gripper_sign_error"] for row in comparison]))
    gripper_deficit = near_gripper - comparison_gripper
    return {
        "headroom_horizon": 50,
        "strata": strata,
        "by_partition": by_partition,
        "by_task": by_task,
        "event_near_arm_relative_deficit": arm_relative_deficit,
        "event_near_gripper_sign_error_deficit": gripper_deficit,
        "passed": arm_relative_deficit >= 0.10 or gripper_deficit >= 0.05,
        "offline_diagnostic_only": True,
    }


def _identity_audit(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    row: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> tuple[dict[str, Any], Any, Any, Any]:
    import torch
    from peft import PeftConfig, PeftModel

    base_flow = _flow_vector(policy, preprocessor, row)
    base_native, base_queue, _ = _predict_chunk(policy, preprocessor, postprocessor, row)
    base_hash_before = _hash_base_parameters(policy)
    policy = policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": 4})
    policy.to("cuda")
    policy.eval()
    initialized_flow = _flow_vector(policy, preprocessor, row)
    initialized_native, initialized_queue, _ = _predict_chunk(policy, preprocessor, postprocessor, row)
    initialized_flow_error = float(torch.max(torch.abs(initialized_flow - base_flow)).item())
    initialized_native_error = float(torch.max(torch.abs(initialized_native - base_native)).item())
    initialized_queue_error = float(np.max(np.abs(initialized_queue - base_queue)))
    base_hash_after = _hash_base_parameters(policy)

    paths["adapter_dir"].mkdir(parents=True, exist_ok=True)
    if hasattr(policy, "peft_config"):
        for config in policy.peft_config.values():
            config.base_model_name_or_path = str(paths["checkpoint"])
    policy.save_pretrained(paths["adapter_dir"], safe_serialization=True)

    torch.manual_seed(SEED)
    heads = torch.nn.ModuleDict(
        {
            "hazard_20": torch.nn.Sequential(torch.nn.Linear(960, 128), torch.nn.SiLU(), torch.nn.Linear(128, 20)),
            "hazard_50": torch.nn.Sequential(torch.nn.Linear(960, 128), torch.nn.SiLU(), torch.nn.Linear(128, 50)),
            "displacement": torch.nn.Sequential(torch.nn.Linear(960, 128), torch.nn.SiLU(), torch.nn.Linear(128, 6)),
        }
    )
    fixture = torch.zeros((1, 960), dtype=torch.float32)
    before_heads = {name: module(fixture).detach().clone() for name, module in heads.items()}
    temporary = paths["head_checkpoint"].with_suffix(".tmp.pt")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"proposal_hash": PROPOSAL_HASH, "state_dict": heads.state_dict()}, temporary)
    temporary.replace(paths["head_checkpoint"])
    payload = torch.load(paths["head_checkpoint"], map_location="cpu", weights_only=True)
    reloaded_heads = torch.nn.ModuleDict(
        {
            "hazard_20": torch.nn.Sequential(torch.nn.Linear(960, 128), torch.nn.SiLU(), torch.nn.Linear(128, 20)),
            "hazard_50": torch.nn.Sequential(torch.nn.Linear(960, 128), torch.nn.SiLU(), torch.nn.Linear(128, 50)),
            "displacement": torch.nn.Sequential(torch.nn.Linear(960, 128), torch.nn.SiLU(), torch.nn.Linear(128, 6)),
        }
    )
    reloaded_heads.load_state_dict(payload["state_dict"])
    head_reload_error = max(
        float(torch.max(torch.abs(reloaded_heads[name](fixture) - before_heads[name])).item())
        for name in before_heads
    )

    del policy, heads, reloaded_heads
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
    reload_flow = _flow_vector(reloaded, reloaded_preprocessor, row)
    reload_native, reload_queue, _ = _predict_chunk(
        reloaded, reloaded_preprocessor, reloaded_postprocessor, row
    )
    reload_native_error = float(torch.max(torch.abs(reload_native - base_native)).item())
    reload_queue_error = float(np.max(np.abs(reload_queue - base_queue)))
    reload_flow_error = float(torch.max(torch.abs(reload_flow - base_flow)).item())
    identity_max_error = max(
        initialized_flow_error,
        initialized_native_error,
        initialized_queue_error,
        reload_flow_error,
        reload_native_error,
        reload_queue_error,
    )
    return (
        {
            "rank": 4,
            "base_flow_shape": list(base_flow.shape),
            "base_native_shape": list(base_native.shape),
            "base_queue_shape": list(base_queue.shape),
            "initialized_flow_max_abs_error": initialized_flow_error,
            "initialized_native_max_abs_error": initialized_native_error,
            "initialized_queue_max_abs_error": initialized_queue_error,
            "reload_flow_max_abs_error": reload_flow_error,
            "reload_native_max_abs_error": reload_native_error,
            "reload_queue_max_abs_error": reload_queue_error,
            "identity_max_abs_error": identity_max_error,
            "base_parameter_hash_before": base_hash_before,
            "base_parameter_hash_after": base_hash_after,
            "base_hash_unchanged": base_hash_before == base_hash_after,
            "adapter_checkpoint": str(paths["adapter_dir"]),
            "head_checkpoint": str(paths["head_checkpoint"]),
            "head_reload_max_abs_error": head_reload_error,
            "checkpoint_reload_ok": max(
                reload_flow_error, reload_native_error, reload_queue_error, head_reload_error
            )
            <= 1e-6,
        },
        reloaded,
        reloaded_preprocessor,
        reloaded_postprocessor,
    )


def _label_summary(manifest: Sequence[Mapping[str, Any]], normalization: Mapping[str, Any]) -> dict[str, Any]:
    discovery = [row for row in manifest if row["partition"] == "discovery"]
    validation = [row for row in manifest if row["partition"] == "validation"]
    discovery_uncensored = [row for row in discovery if not row["censored"]]
    discovery_censored = [row for row in discovery if row["censored"]]
    validation_uncensored_by_task = {
        task: sum(row["task_identity"] == task and not row["censored"] for row in validation)
        for _, task, _ in TASK_SOURCES
    }
    task_uncensored = {
        task: sum(row["task_identity"] == task for row in discovery_uncensored)
        for _, task, _ in TASK_SOURCES
    }
    occupied_quintiles = sorted(
        {int(row["offset_quintile"]) for row in manifest if row["offset_quintile"] is not None}
    )
    variance = np.square(np.asarray(normalization["discovery_relative_displacement_std"], dtype=np.float64))
    return {
        "manifest_row_count": len(manifest),
        "discovery_row_count": len(discovery),
        "validation_row_count": len(validation),
        "discovery_uncensored_count": len(discovery_uncensored),
        "discovery_censored_count": len(discovery_censored),
        "validation_uncensored_by_task": validation_uncensored_by_task,
        "minimum_validation_uncensored_per_task": min(validation_uncensored_by_task.values()),
        "discovery_uncensored_by_task": task_uncensored,
        "maximum_uncensored_task_fraction": max(task_uncensored.values()) / max(len(discovery_uncensored), 1),
        "occupied_offset_quintiles": occupied_quintiles,
        "occupied_offset_quintile_count": len(occupied_quintiles),
        "displacement_variance": variance,
        "displacement_variance_all_positive": bool(np.all(variance > 0.0)),
    }


def _result_markdown(result: Mapping[str, Any]) -> str:
    audit = result["manifest_audit"]
    labels = result["labels"]
    return "\n".join(
        [
            "# HASTE-VLA Stage 0A Result",
            "",
            f"Decision: `{result['final_decision']}`.",
            "",
            f"Proposal hash: `{result['proposal_hash']}`.",
            "",
            f"Model rows completed: `{result['completed_model_row_count']} / {result['planned_model_row_count']}`.",
            "",
            f"Exceptions: `{result['exception_count']}`.",
            "",
            f"Duplicate partial / missing / extra / overlap: `{audit['duplicate_partial_key_count']} / {audit['missing_manifest_key_count']} / {audit['extra_partial_key_count']} / {audit['partition_overlap_count']}`.",
            "",
            f"Discovery uncensored / censored: `{labels['discovery_uncensored_count']} / {labels['discovery_censored_count']}`.",
            "",
            f"Base event-near headroom passed: `{result['headroom']['passed']}`.",
            "",
            f"Minimum hazard-probe improvement: `{result['probes']['minimum_hazard_relative_improvement']}`.",
            "",
            f"Displacement-probe improvement: `{result['probes']['displacement']['relative_improvement']}`.",
            "",
            f"Identity maximum error: `{result['identity']['identity_max_abs_error']}`.",
            "",
            "No adapter optimization, simulator rollout, reward, success, done flag, or confirmatory reset identity was used. This is not a closed-loop scientific result.",
            "",
        ]
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _set_offline_environment()
    paths = _paths(args)
    paths["report"].mkdir(parents=True, exist_ok=True)
    paths["run"].mkdir(parents=True, exist_ok=True)
    _write_text(paths["pid"], f"{os.getpid()}\n")
    if paths["result_json"].is_file() and _read_json(paths["result_json"]).get("final_decision"):
        raise RuntimeError("completed HASTE Stage 0A result exists; refusing duplicate execution")

    proposal_hash_recomputed = _sha256(PROPOSAL_FILE)
    proposal_hash_registry = _registry_hash()
    proposal_hash_ok = proposal_hash_recomputed == proposal_hash_registry == PROPOSAL_HASH
    if not proposal_hash_ok:
        raise RuntimeError("frozen HASTE proposal hash mismatch")

    manifest_rows, sources, normalization = _build_manifest(paths["data_root"])
    model_manifest = [row for row in manifest_rows if row["selected_for_model_audit"]]
    manifest_payload = {
        "method": "HASTE-VLA",
        "stage": "0A",
        "proposal_hash": PROPOSAL_HASH,
        "sources": sources,
        "normalization": normalization,
        "planned_label_row_count": len(manifest_rows),
        "planned_model_row_count": len(model_manifest),
        "model_rows_per_horizon_stratum_task_split": MODEL_ROWS_PER_HORIZON_STRATUM_TASK_SPLIT,
        "rows": manifest_rows,
    }
    manifest_hash = canonical_json_sha256(manifest_payload)
    manifest_payload["manifest_hash"] = manifest_hash
    _write_json(paths["manifest"], manifest_payload)

    partial_rows = _load_resume(paths["partial"], model_manifest, manifest_hash)
    completed = {str(row["event_row_key"]) for row in partial_rows}
    state = {
        "method": "HASTE-VLA",
        "stage": "0A",
        "pid": os.getpid(),
        "status": "running",
        "phase": "model_audit",
        "planned_model_row_count": len(model_manifest),
        "completed_model_row_count": len(partial_rows),
        "exception_count": 0,
    }
    _write_json(
        paths["preflight"],
        {
            "proposal_hash_recomputed": proposal_hash_recomputed,
            "proposal_hash_registry": proposal_hash_registry,
            "proposal_hash_ok": proposal_hash_ok,
            "manifest_hash": manifest_hash,
            "planned_label_row_count": len(manifest_rows),
            "planned_model_row_count": len(model_manifest),
            "resumed_model_row_count": len(partial_rows),
            "feature_hook": {
                "module": "SmolVLAModel.embed_prefix",
                "tensor_position": "returned embeddings tensor before suffix concatenation",
                "pooling": "valid-prefix-mask mean over sequence dimension",
                "width": 960,
            },
            "adapter_training_happened": False,
            "simulator_loaded": False,
            "confirmatory_records_read": 0,
            "started_at": _utc_now(),
        },
    )
    _write_json(paths["status"], {**state, "started_at": _utc_now()})
    _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
    stop = threading.Event()
    heartbeat = threading.Thread(target=_heartbeat_loop, args=(paths["heartbeat"], state, stop), daemon=True)
    heartbeat.start()
    policy = None
    try:
        policy, _, preprocessor, postprocessor = _load_policy_and_processors(paths["checkpoint"])
        for row in model_manifest:
            if row["event_row_key"] in completed:
                continue
            feature_path = _feature_path(paths["feature_dir"], row)
            if feature_path.is_file():
                feature = _load_feature(feature_path)
            else:
                feature = _extract_feature(policy, preprocessor, postprocessor, row)
                _save_feature(feature_path, **feature)
            summary = _row_summary(row, feature_path, feature)
            partial_rows.append(summary)
            completed.add(str(row["event_row_key"]))
            state["completed_model_row_count"] = len(partial_rows)
            _write_json(
                paths["partial"],
                _partial_payload(manifest_hash, len(model_manifest), partial_rows),
            )
            _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
            print(f"[haste-stage0a] model rows {len(partial_rows)}/{len(model_manifest)}", flush=True)

        ordered = {str(row["event_row_key"]): row for row in partial_rows}
        partial_rows = [ordered[str(row["event_row_key"])] for row in model_manifest]
        manifest_audit = validate_manifest(model_manifest, partial_rows)
        labels = _label_summary(manifest_rows, normalization)
        headroom = _headroom(partial_rows)
        probes = _run_probes(partial_rows, manifest_rows)
        identity_row = next(row for row in model_manifest if row["partition"] == "discovery")
        identity, policy, preprocessor, postprocessor = _identity_audit(
            policy, preprocessor, postprocessor, identity_row, paths
        )
        finite_source_features = bool(
            all(
                float(row["input_finite_fraction"]) == 1.0
                and float(row["feature_finite_fraction"]) == 1.0
                and row["base_action_finite"]
                for row in partial_rows
            )
            and math.isfinite(float(probes["minimum_hazard_relative_improvement"]))
            and math.isfinite(float(probes["displacement"]["relative_improvement"]))
        )
        manifest_ok = (
            manifest_audit["manifest_row_count"] == len(model_manifest)
            and manifest_audit["partial_row_count"] == len(model_manifest)
            and manifest_audit["duplicate_manifest_key_count"] == 0
            and manifest_audit["duplicate_partial_key_count"] == 0
            and manifest_audit["missing_manifest_key_count"] == 0
            and manifest_audit["extra_partial_key_count"] == 0
            and manifest_audit["partition_overlap_count"] == 0
            and bool(manifest_audit["key_sets_equal"])
        )
        decision_inputs = Stage0ADecisionInputs(
            proposal_hash_ok=proposal_hash_ok,
            manifest_integrity_ok=manifest_ok,
            finite_source_and_features=finite_source_features,
            discovery_uncensored_count=labels["discovery_uncensored_count"],
            discovery_censored_count=labels["discovery_censored_count"],
            minimum_validation_uncensored_per_task=labels["minimum_validation_uncensored_per_task"],
            occupied_offset_quintile_count=labels["occupied_offset_quintile_count"],
            displacement_variance_all_positive=labels["displacement_variance_all_positive"],
            maximum_uncensored_task_fraction=labels["maximum_uncensored_task_fraction"],
            base_event_near_headroom=headroom["passed"],
            hazard_probe_improvement=probes["minimum_hazard_relative_improvement"],
            displacement_probe_improvement=probes["displacement"]["relative_improvement"],
            identity_max_error=identity["identity_max_abs_error"],
            base_hash_unchanged=identity["base_hash_unchanged"],
            checkpoint_reload_ok=identity["checkpoint_reload_ok"],
            exception_count=0,
        )
        decision = classify_stage0a(decision_inputs)
        result = {
            "method": "HASTE-VLA",
            "stage": "0A",
            "proposal_hash": PROPOSAL_HASH,
            "manifest_hash": manifest_hash,
            "worker_pid": os.getpid(),
            "planned_label_row_count": len(manifest_rows),
            "planned_model_row_count": len(model_manifest),
            "completed_model_row_count": len(partial_rows),
            "resumed_model_row_count": int(_read_json(paths["preflight"])["resumed_model_row_count"]),
            "exception_count": 0,
            "manifest_audit": manifest_audit,
            "labels": labels,
            "headroom": headroom,
            "probes": probes,
            "identity": identity,
            "finite_source_and_features": finite_source_features,
            "final_decision": decision,
            "stage_0b_allowed": decision == "HASTE_STAGE_0A_PASS_STAGE_0B_ALLOWED",
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
        validation = {
            "proposal_hash_ok": proposal_hash_ok,
            "manifest_json_parsed": True,
            "partial_json_parsed": True,
            "result_decision_recomputed": classify_stage0a(decision_inputs),
            **manifest_audit,
            "exception_count": 0,
            "final_decision": decision,
        }
        _write_json(paths["result_json"], result)
        _write_text(paths["result_md"], _result_markdown(result))
        _write_json(paths["validation"], validation)
        state.update({"status": "completed", "phase": "complete", "completed_model_row_count": len(partial_rows)})
        _write_json(paths["status"], {**state, "completed_at": _utc_now()})
        _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
        return result
    except Exception as exc:
        detail = traceback.format_exc()
        _write_json(
            paths["partial"],
            _partial_payload(
                manifest_hash,
                len(model_manifest),
                partial_rows,
                exception_count=1,
                last_exception=detail,
            ),
        )
        _write_json(
            paths["blocker"],
            {
                "method": "HASTE-VLA",
                "stage": "0A",
                "proposal_hash": PROPOSAL_HASH,
                "exception_type": type(exc).__name__,
                "exception": str(exc),
                "traceback": detail,
                "completed_model_row_count": len(partial_rows),
                "planned_model_row_count": len(model_manifest),
                "scientific_kill": False,
                "failed_at": _utc_now(),
            },
        )
        state.update({"status": "failed", "phase": "failed", "exception_count": 1})
        _write_json(paths["status"], {**state, "failed_at": _utc_now()})
        _write_json(paths["heartbeat"], {**state, "updated_at": _utc_now()})
        raise
    finally:
        stop.set()
        heartbeat.join(timeout=2.0)
        if policy is not None:
            del policy
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", default="/mnt/c/assets/data/libero")
    parser.add_argument("--checkpoint", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--report-root", default=str(REPO_ROOT / "reports" / "haste_vla"))
    parser.add_argument("--run-root", default=str(REPO_ROOT / "runs" / "haste_vla" / "stage0a"))
    args = parser.parse_args()
    try:
        result = run(args)
    except Exception:
        traceback.print_exc()
        return 1
    print(json.dumps({"final_decision": result["final_decision"]}, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
