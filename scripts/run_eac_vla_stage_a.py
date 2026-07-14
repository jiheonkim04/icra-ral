"""Freeze and preflight EAC-VLA Stage A manifest."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
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

from tca_map.smolvla.eac_vla import PROPOSAL_HASH, chunk_sha256, eac_commitment_prefix  # noqa: E402
from tca_map.smolvla.official_canonical_eval import _make_noise, _postprocess_chunk  # noqa: E402
from tca_map.smolvla.official_closed_loop_scaleup import (  # noqa: E402
    _extract_single_env,
    _rss_mb,
    _successes_from_info,
    wilson_interval,
)
from tca_map.smolvla.official_wsl_libero_rollout import (  # noqa: E402
    PolicySpec,
    _cuda_memory,
    _dummy_observation,
    _json_default,
    _load_policy_and_processors,
    _make_env_cfg,
    _round,
    _set_runtime_env,
)


DATE_KST = "2026-07-15"
RESET_SEEDS = [20261211, 20261212]
EAC_RUNTIME_SAMPLES = 2
EAC_RISK_DISPERSION_WEIGHT = 0.67
EAC_RISK_TRANSITION_WEIGHT = 0.33
EAC_FULL_POLICY = "eac_full"
STAGE_A_TASKS = [
    {
        "suite": "libero_spatial",
        "task_id": 0,
        "instruction": "pick up the black bowl between the plate and the ramekin and place it on the plate",
    },
    {
        "suite": "libero_spatial",
        "task_id": 8,
        "instruction": "pick up the black bowl next to the plate and place it on the plate",
    },
    {
        "suite": "libero_object",
        "task_id": 6,
        "instruction": "pick up the butter and place it in the basket",
    },
    {
        "suite": "libero_goal",
        "task_id": 4,
        "instruction": "put the bowl on top of the cabinet",
    },
    {
        "suite": "libero_10",
        "task_id": 2,
        "instruction": "turn on the stove and put the moka pot on it",
    },
]
POLICIES = [
    {
        "policy": "frozen_smolvla_fixed_queue",
        "role": "base",
        "scheduler": "fixed_commitment",
        "commitment": 50,
        "proxy_or_reproduction_label": "unmodified frozen SmolVLA action values with fixed full queue commitment",
    },
    {
        "policy": "aac_entropy_proxy",
        "role": "closest_prior_proxy",
        "scheduler": "dispersion_only_quantile_proxy",
        "commitment": 8,
        "commitment_map": {"short": 2, "medium": 8, "long": 50},
        "quantile_margin": 0.33,
        "proxy_or_reproduction_label": "faithful transparent local proxy, not an official AAC reproduction",
    },
    {
        "policy": "eac_full",
        "role": "ours",
        "scheduler": "selected_validation_config",
        "commitment": 4,
        "config_id": "eac_q33_aggressive_1_4_50",
        "commitment_map": {"short": 1, "medium": 4, "long": 50},
        "quantile_margin": 0.33,
        "proxy_or_reproduction_label": "ours",
    },
    {
        "policy": "eac_no_calibration_no_hysteresis_ablation",
        "role": "key_ablation",
        "scheduler": "raw_risk_fixed_threshold_ablation",
        "commitment": 4,
        "commitment_map": {"short": 1, "medium": 4, "long": 50},
        "raw_thresholds": {"low": 1.0 / 3.0, "high": 2.0 / 3.0},
        "proxy_or_reproduction_label": "key ablation",
    },
    {
        "policy": "fixed_short_replan_baseline",
        "role": "simple_killer",
        "scheduler": "fixed_commitment",
        "commitment": 1,
        "proxy_or_reproduction_label": "strong simple fixed short-replan baseline",
    },
]


def _canonical_json(payload: Any) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), default=_json_default).encode("utf-8")


def _sha256_payload(payload: Any) -> str:
    return hashlib.sha256(_canonical_json(payload)).hexdigest().upper()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# EAC-VLA Stage A Manifest And Preflight",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- training happened: `{report['training_happened']}`",
        f"- validation search happened: `{report['validation_search_happened']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        f"- planned episode count: `{report['planned_episode_count']}`",
        f"- paired cases per policy: `{report['paired_cases_per_policy']}`",
        f"- reset seeds: `{report['reset_seeds']}`",
        f"- policies: `{report['policy_order']}`",
        f"- canonical payload sha256: `{report['canonical_payload_sha256']}`",
        "",
        "Policy identities:",
        "",
        "```json",
        json.dumps(report["policy_identities"], indent=2, sort_keys=True),
        "```",
        "",
        "Preflight records:",
        "",
        "```json",
        json.dumps(report.get("preflight_records", []), indent=2, sort_keys=True),
        "```",
        "",
        "Errors:",
    ]
    errors = list(report.get("errors") or [])
    if errors:
        lines.extend(f"- `{error}`" for error in errors)
    else:
        lines.append("- none")
    lines.extend(["", f"Next step: {report['next_step']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_lines_md(path: Path, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _as_array(value: Any, shape: tuple[int, ...] | None = None) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if shape is not None and tuple(array.shape) != tuple(shape):
        raise ValueError(f"expected shape {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("nonfinite array")
    return array


def _frame_key(record: Mapping[str, Any]) -> tuple[int, int, int]:
    return (int(record["task_index"]), int(record["episode_index"]), int(record["frame_index"]))


def _robust_norm(values: np.ndarray) -> np.ndarray:
    lo = float(np.quantile(values, 0.05))
    hi = float(np.quantile(values, 0.95))
    if hi <= lo:
        return np.zeros_like(values, dtype=np.float64)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _normalize_scalar(value: float, lo: float, hi: float) -> float:
    if hi <= lo:
        return 0.0
    return float(np.clip((float(value) - lo) / (hi - lo), 0.0, 1.0))


def _summarize(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _validation_frame_metrics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if str(row.get("split")) == "val":
            grouped[_frame_key(row)].append(row)

    metrics = []
    dispersion_values = []
    transition_values = []
    for key, rows in sorted(grouped.items()):
        previews = [_as_array(row["base_action_chunk_first_two_preview"], (2, 7)) for row in rows]
        stacked = np.stack(previews, axis=0)
        dispersion = float(np.mean(np.var(stacked, axis=0)))
        first_transition = float(np.mean([np.linalg.norm(item[1] - item[0]) for item in previews]))
        dispersion_values.append(dispersion)
        transition_values.append(first_transition)
        metrics.append(
            {
                "task_index": int(key[0]),
                "episode_index": int(key[1]),
                "frame_index": int(key[2]),
                "first_two_dispersion": dispersion,
                "first_transition_l2": first_transition,
            }
        )

    dispersion_norm = _robust_norm(np.asarray(dispersion_values, dtype=np.float64))
    transition_norm = _robust_norm(np.asarray(transition_values, dtype=np.float64))
    risk = EAC_RISK_DISPERSION_WEIGHT * dispersion_norm + EAC_RISK_TRANSITION_WEIGHT * transition_norm
    for item, u_norm, d_norm, value in zip(metrics, dispersion_norm.tolist(), transition_norm.tolist(), risk.tolist()):
        item["dispersion_norm"] = float(u_norm)
        item["transition_norm"] = float(d_norm)
        item["risk"] = float(value)
    return metrics


def _build_runtime_calibration(args: argparse.Namespace) -> dict[str, Any]:
    artifact = _read_json(Path(args.canonical_artifact))
    selected_config = _read_json(Path(args.selected_config))
    records = list(artifact.get("records") or [])
    metrics = _validation_frame_metrics(records)
    dispersion = np.asarray([float(item["first_two_dispersion"]) for item in metrics], dtype=np.float64)
    transition = np.asarray([float(item["first_transition_l2"]) for item in metrics], dtype=np.float64)
    risk = np.asarray([float(item["risk"]) for item in metrics], dtype=np.float64)
    dispersion_norm = np.asarray([float(item["dispersion_norm"]) for item in metrics], dtype=np.float64)
    q = float(selected_config["quantile_margin"])
    return {
        "source_canonical_artifact": str(args.canonical_artifact),
        "source_canonical_artifact_sha256": _sha256_file(Path(args.canonical_artifact)),
        "selected_config": selected_config,
        "validation_frame_count": len(metrics),
        "risk_weights": {"dispersion": EAC_RISK_DISPERSION_WEIGHT, "transition": EAC_RISK_TRANSITION_WEIGHT},
        "normalizer": {
            "dispersion_p05": float(np.quantile(dispersion, 0.05)),
            "dispersion_p95": float(np.quantile(dispersion, 0.95)),
            "transition_p05": float(np.quantile(transition, 0.05)),
            "transition_p95": float(np.quantile(transition, 0.95)),
        },
        "eac_quantile_thresholds": {
            "low": float(np.quantile(risk, q)),
            "high": float(np.quantile(risk, 1.0 - q)),
            "quantile_margin": q,
        },
        "aac_dispersion_quantile_thresholds": {
            "low": float(np.quantile(dispersion_norm, q)),
            "high": float(np.quantile(dispersion_norm, 1.0 - q)),
            "quantile_margin": q,
        },
        "risk_summary": _summarize(risk.tolist()),
        "dispersion_norm_summary": _summarize(dispersion_norm.tolist()),
        "transition_norm_summary": _summarize([float(item["transition_norm"]) for item in metrics]),
        "frozen_validation_only": True,
        "confirmatory_test_tuning_happened": False,
    }


def _episode_manifest() -> list[dict[str, Any]]:
    episodes = []
    index = 0
    for policy in POLICIES:
        policy_name = str(policy["policy"])
        for task in STAGE_A_TASKS:
            for seed in RESET_SEEDS:
                pair_id = f"{task['suite']}|task_{task['task_id']}|seed_{seed}"
                episodes.append(
                    {
                        "planned_episode_index": index,
                        "episode_id": f"{policy_name}|{pair_id}",
                        "pair_id": pair_id,
                        "policy": policy_name,
                        "suite": task["suite"],
                        "task_id": int(task["task_id"]),
                        "instruction": task["instruction"],
                        "reset_seed": int(seed),
                    }
                )
                index += 1
    return episodes


def _validate_manifest(payload: Mapping[str, Any]) -> list[str]:
    errors = []
    episodes = list(payload.get("episodes") or [])
    policy_order = list(payload.get("policy_order") or [])
    pair_sets = {}
    episode_ids = [row["episode_id"] for row in episodes]
    if len(episode_ids) != len(set(episode_ids)):
        errors.append("duplicate episode ids")
    for policy in policy_order:
        pair_sets[policy] = {row["pair_id"] for row in episodes if row["policy"] == policy}
    if len({tuple(sorted(values)) for values in pair_sets.values()}) != 1:
        errors.append("policy pair identities are not matched")
    if len(episodes) != 50:
        errors.append(f"planned episode count is {len(episodes)}, expected 50")
    per_policy = Counter(row["policy"] for row in episodes)
    if any(count != 10 for count in per_policy.values()):
        errors.append(f"per-policy count mismatch: {dict(per_policy)}")
    if sorted(set(row["reset_seed"] for row in episodes)) != RESET_SEEDS:
        errors.append("reset seed set mismatch")
    return errors


def build_manifest(args: argparse.Namespace) -> dict[str, Any]:
    selected_config = json.loads(Path(args.selected_config).read_text(encoding="utf-8"))
    episodes = _episode_manifest()
    payload = {
        "schema_version": 1,
        "date": f"{DATE_KST} KST",
        "method": "EAC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "mode": "stage_a_manifest",
        "branch": "codex/autonomous-until-paper-governance-v2",
        "selected_config": selected_config,
        "config_id": selected_config["config_id"],
        "policy_order": [item["policy"] for item in POLICIES],
        "policy_identities": POLICIES,
        "reset_seeds": RESET_SEEDS,
        "tasks": STAGE_A_TASKS,
        "episodes": episodes,
        "planned_episode_count": len(episodes),
        "paired_cases_per_policy": len(STAGE_A_TASKS) * len(RESET_SEEDS),
        "official_success_condition": "LIBERO official task success from environment final_info/is_success",
        "policy_order_affects_env_initialization": False,
        "fixed_task_balanced_allocation": True,
        "no_post_hoc_task_or_reset_selection": True,
        "confirmatory_test_identities_used_for_training_or_validation": False,
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "validation_search_happened": False,
        "confirmatory_test_tuning_happened": False,
        "errors": [],
    }
    payload["errors"] = _validate_manifest(payload)
    canonical_payload = {key: value for key, value in payload.items() if key not in {"canonical_payload_sha256", "errors"}}
    payload["canonical_payload_sha256"] = _sha256_payload(canonical_payload)
    payload["final_decision"] = "EAC_STAGE_A_PLAN_FROZEN_PREFLIGHT_PENDING" if not payload["errors"] else "IMPLEMENTATION_FAILURE"
    payload["next_step"] = (
        "Run the EAC Stage A policy preflight before any rollout."
        if not payload["errors"]
        else "Fix only the concrete manifest defect before preflight."
    )
    return payload


def build_preflight(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    _set_runtime_env(args)
    manifest = json.loads(Path(args.stage_a_manifest).read_text(encoding="utf-8"))
    loaded = _load_policy_and_processors(args, PolicySpec("frozen_base"))
    policy = loaded["policy"]
    env_preprocessor = loaded["env_preprocessor"]
    preprocessor = loaded["preprocessor"]
    postprocessor = loaded["postprocessor"]
    dummy = env_preprocessor(_dummy_observation(torch))
    batch = preprocessor(dummy)
    policy.reset()
    noise = _make_noise(policy, int(args.runtime_seed), torch)
    with torch.inference_mode():
        raw_chunk = policy.predict_action_chunk(batch, noise=noise)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    postprocessed_chunk, _ = _postprocess_chunk(raw_chunk, postprocessor, 7)
    errors = list(manifest.get("errors") or [])
    records = []
    for identity in manifest["policy_identities"]:
        commitment = int(identity["commitment"])
        prefix = eac_commitment_prefix(postprocessed_chunk, commitment)
        max_diff = float(np.max(np.abs(prefix - postprocessed_chunk[:commitment]))) if prefix.size else 0.0
        records.append(
            {
                "policy": identity["policy"],
                "role": identity["role"],
                "scheduler": identity["scheduler"],
                "commitment": commitment,
                "prefix_shape": [int(dim) for dim in prefix.shape],
                "prefix_sha256": chunk_sha256(prefix),
                "prefix_max_abs_diff": max_diff,
                "action_values_modified": bool(max_diff > 0.0),
                "proxy_or_reproduction_label": identity["proxy_or_reproduction_label"],
            }
        )
        if max_diff > 0.0:
            errors.append(f"{identity['policy']} changed action values")
    output_shape_ok = [int(dim) for dim in postprocessed_chunk.shape] == [50, 7]
    if not output_shape_ok:
        errors.append(f"postprocessed chunk shape mismatch: {list(postprocessed_chunk.shape)}")
    if not bool(np.isfinite(postprocessed_chunk).all()):
        errors.append("postprocessed chunk contains nonfinite values")
    if (loaded.get("audit") or {}).get("old_custom_libero_7d_route_used"):
        errors.append("old custom LIBERO_7D route used")
    report = {
        **{key: value for key, value in manifest.items() if key != "episodes"},
        "mode": "stage_a_preflight",
        "final_decision": "EAC_STAGE_A_PREFLIGHT_PASS_RUNNER_IMPLEMENTATION_PENDING" if not errors else "IMPLEMENTATION_FAILURE",
        "stage_a_manifest": str(args.stage_a_manifest),
        "stage_a_manifest_sha256": _sha256_payload(manifest),
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "validation_search_happened": False,
        "confirmatory_test_tuning_happened": False,
        "preflight_records": records,
        "policy_count": len(records),
        "checkpoint_policy_count": 0,
        "cuda_ok": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_memory": _cuda_memory(torch),
        "policy_output_shape": [int(dim) for dim in postprocessed_chunk.shape],
        "policy_output_shape_ok": output_shape_ok,
        "policy_output_finite": bool(np.isfinite(postprocessed_chunk).all()),
        "all_policy_prefixes_value_preserving": bool(all(not row["action_values_modified"] for row in records)),
        "no_accidental_checkpoint_reuse": True,
        "old_custom_libero_7d_route_used": bool((loaded.get("audit") or {}).get("old_custom_libero_7d_route_used")),
        "errors": errors,
        "next_step": (
            "Implement the minimal EAC Stage A runner and launch only after runner validation."
            if not errors
            else "Fix only the concrete preflight defect before runner implementation."
        ),
    }
    return report


def _policy_identity_map(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {str(item["policy"]): dict(item) for item in manifest["policy_identities"]}


def _runtime_sample_count(identity: Mapping[str, Any], args: argparse.Namespace) -> int:
    if identity["scheduler"] == "fixed_commitment":
        return 1
    return max(2, int(args.runtime_samples))


def _runtime_noise_seed(args: argparse.Namespace, reset_seed: int, env_step: int, sample_index: int) -> int:
    return int(args.runtime_seed) + int(reset_seed) * 1009 + int(env_step) * 9176 + int(sample_index) * 101


def _postprocessed_chunks(
    *,
    policy: Any,
    batch: Mapping[str, Any],
    postprocessor: Any,
    torch_mod: Any,
    reset_seed: int,
    env_step: int,
    sample_count: int,
    args: argparse.Namespace,
) -> list[np.ndarray]:
    chunks = []
    for sample_index in range(int(sample_count)):
        noise = _make_noise(policy, _runtime_noise_seed(args, reset_seed, env_step, sample_index), torch_mod)
        with torch_mod.inference_mode():
            raw_chunk = policy.predict_action_chunk(batch, noise=noise)
        postprocessed_chunk, _ = _postprocess_chunk(raw_chunk, postprocessor, 7)
        chunks.append(np.asarray(postprocessed_chunk, dtype=np.float32))
    return chunks


def _risk_from_chunks(chunks: Sequence[np.ndarray], calibration: Mapping[str, Any]) -> dict[str, Any]:
    if not chunks:
        raise ValueError("need at least one action chunk")
    previews = np.stack([np.asarray(chunk[:2], dtype=np.float64) for chunk in chunks], axis=0)
    dispersion = float(np.mean(np.var(previews, axis=0))) if len(chunks) >= 2 else 0.0
    transition = float(np.mean([np.linalg.norm(np.asarray(chunk[1], dtype=np.float64) - np.asarray(chunk[0], dtype=np.float64)) for chunk in chunks]))
    norm = calibration["normalizer"]
    dispersion_norm = _normalize_scalar(dispersion, float(norm["dispersion_p05"]), float(norm["dispersion_p95"]))
    transition_norm = _normalize_scalar(transition, float(norm["transition_p05"]), float(norm["transition_p95"]))
    risk = EAC_RISK_DISPERSION_WEIGHT * dispersion_norm + EAC_RISK_TRANSITION_WEIGHT * transition_norm
    return {
        "first_two_dispersion": dispersion,
        "first_transition_l2": transition,
        "dispersion_norm": dispersion_norm,
        "transition_norm": transition_norm,
        "risk": float(risk),
        "chunk_sample_count": int(len(chunks)),
    }


def _three_way_commitment(value: float, thresholds: Mapping[str, Any], commitment_map: Mapping[str, Any]) -> int:
    if float(value) <= float(thresholds["low"]):
        return int(commitment_map["long"])
    if float(value) >= float(thresholds["high"]):
        return int(commitment_map["short"])
    return int(commitment_map["medium"])


def _commitment_for_policy(
    identity: Mapping[str, Any],
    risk_record: Mapping[str, Any],
    calibration: Mapping[str, Any],
) -> tuple[int, dict[str, Any]]:
    scheduler = str(identity["scheduler"])
    if scheduler == "fixed_commitment":
        return int(identity["commitment"]), {"scheduler_value": None, "thresholds": None}
    if scheduler == "dispersion_only_quantile_proxy":
        thresholds = dict(calibration["aac_dispersion_quantile_thresholds"])
        commitment = _three_way_commitment(float(risk_record["dispersion_norm"]), thresholds, identity["commitment_map"])
        return commitment, {"scheduler_value": float(risk_record["dispersion_norm"]), "thresholds": thresholds}
    if scheduler == "selected_validation_config":
        thresholds = dict(calibration["eac_quantile_thresholds"])
        commitment = _three_way_commitment(float(risk_record["risk"]), thresholds, identity["commitment_map"])
        return commitment, {"scheduler_value": float(risk_record["risk"]), "thresholds": thresholds}
    if scheduler == "raw_risk_fixed_threshold_ablation":
        thresholds = dict(identity["raw_thresholds"])
        commitment = _three_way_commitment(float(risk_record["risk"]), thresholds, identity["commitment_map"])
        return commitment, {"scheduler_value": float(risk_record["risk"]), "thresholds": thresholds}
    raise ValueError(f"unknown EAC scheduler: {scheduler}")


def _schedule_prefix(
    identity: Mapping[str, Any],
    chunks: Sequence[np.ndarray],
    calibration: Mapping[str, Any],
) -> dict[str, Any]:
    base_chunk = np.asarray(chunks[0], dtype=np.float32)
    risk_record = _risk_from_chunks(chunks, calibration)
    commitment, scheduler_record = _commitment_for_policy(identity, risk_record, calibration)
    prefix = eac_commitment_prefix(base_chunk, commitment)
    expected = base_chunk[:commitment]
    max_diff = float(np.max(np.abs(prefix - expected))) if prefix.size else 0.0
    return {
        "prefix": prefix,
        "base_chunk": base_chunk,
        "risk": risk_record,
        "commitment": int(commitment),
        "scheduler_value": scheduler_record["scheduler_value"],
        "thresholds": scheduler_record["thresholds"],
        "prefix_shape": [int(dim) for dim in prefix.shape],
        "prefix_sha256": chunk_sha256(prefix),
        "base_chunk_sha256": chunk_sha256(base_chunk),
        "prefix_max_abs_diff": max_diff,
        "action_values_modified": bool(max_diff > 0.0),
    }


def _batch_device(batch: Mapping[str, Any], torch_mod: Any) -> Any:
    for value in batch.values():
        if hasattr(value, "device"):
            return value.device
    return torch_mod.device("cuda" if torch_mod.cuda.is_available() else "cpu")


def _runner_validation_md(path: Path, report: Mapping[str, Any]) -> None:
    lines = [
        "# EAC-VLA Stage A Runner Validation",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- training happened: `{report['training_happened']}`",
        f"- validation search happened: `{report['validation_search_happened']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        f"- planned episode count: `{report['planned_episode_count']}`",
        f"- policy count: `{report['policy_count']}`",
        f"- action values modified: `{report['any_action_values_modified']}`",
        f"- rollout allowed: `{report['stage_a_rollout_allowed']}`",
        "",
        "Calibration:",
        "",
        "```json",
        json.dumps(report["runtime_calibration"], indent=2, sort_keys=True, default=_json_default),
        "```",
        "",
        "Policy validation records:",
        "",
        "```json",
        json.dumps(report["runner_validation_records"], indent=2, sort_keys=True, default=_json_default),
        "```",
        "",
        "Errors:",
    ]
    errors = list(report.get("errors") or [])
    if errors:
        lines.extend(f"- `{error}`" for error in errors)
    else:
        lines.append("- none")
    lines.extend(["", f"Next step: {report['next_step']}"])
    _write_lines_md(path, lines)


def build_runner_validation(args: argparse.Namespace) -> dict[str, Any]:
    import torch

    _set_runtime_env(args)
    manifest = _read_json(Path(args.stage_a_manifest))
    preflight = _read_json(Path(args.stage_a_preflight_output))
    calibration = _build_runtime_calibration(args)
    errors = []
    if manifest.get("final_decision") != "EAC_STAGE_A_PLAN_FROZEN_PREFLIGHT_PENDING":
        errors.append("Stage A manifest decision is not the frozen preflight-pending state")
    if preflight.get("final_decision") != "EAC_STAGE_A_PREFLIGHT_PASS_RUNNER_IMPLEMENTATION_PENDING":
        errors.append("Stage A preflight decision is not runner-implementation-pending pass")
    if calibration["validation_frame_count"] != 400:
        errors.append(f"validation frame count mismatch: {calibration['validation_frame_count']}")

    loaded = _load_policy_and_processors(args, PolicySpec("frozen_base"))
    policy = loaded["policy"]
    env_preprocessor = loaded["env_preprocessor"]
    preprocessor = loaded["preprocessor"]
    postprocessor = loaded["postprocessor"]
    dummy = env_preprocessor(_dummy_observation(torch))
    batch = preprocessor(dummy)

    records = []
    for identity in manifest["policy_identities"]:
        policy.reset()
        chunks = _postprocessed_chunks(
            policy=policy,
            batch=batch,
            postprocessor=postprocessor,
            torch_mod=torch,
            reset_seed=int(args.runtime_seed),
            env_step=0,
            sample_count=_runtime_sample_count(identity, args),
            args=args,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        schedule = _schedule_prefix(identity, chunks, calibration)
        records.append(
            {
                "policy": identity["policy"],
                "role": identity["role"],
                "scheduler": identity["scheduler"],
                "runtime_sample_count": int(len(chunks)),
                "commitment": int(schedule["commitment"]),
                "scheduler_value": schedule["scheduler_value"],
                "thresholds": schedule["thresholds"],
                "risk": schedule["risk"],
                "prefix_shape": schedule["prefix_shape"],
                "prefix_sha256": schedule["prefix_sha256"],
                "base_chunk_sha256": schedule["base_chunk_sha256"],
                "prefix_max_abs_diff": schedule["prefix_max_abs_diff"],
                "action_values_modified": schedule["action_values_modified"],
            }
        )
        if schedule["action_values_modified"]:
            errors.append(f"{identity['policy']} runner modified action values")
        if schedule["prefix_shape"][1:] != [7]:
            errors.append(f"{identity['policy']} prefix action dimension mismatch")

    any_modified = any(bool(row["action_values_modified"]) for row in records)
    report = {
        **{key: value for key, value in manifest.items() if key != "episodes"},
        "mode": "stage_a_runner_validation",
        "final_decision": "EAC_STAGE_A_RUNNER_VALIDATED_READY_FOR_ROLLOUT" if not errors else "IMPLEMENTATION_FAILURE",
        "stage_a_manifest": str(args.stage_a_manifest),
        "stage_a_preflight": str(args.stage_a_preflight_output),
        "stage_a_manifest_sha256": _sha256_file(Path(args.stage_a_manifest)),
        "stage_a_preflight_sha256": _sha256_file(Path(args.stage_a_preflight_output)),
        "runtime_calibration": calibration,
        "runner_validation_records": records,
        "policy_count": len(records),
        "runtime_samples_for_dynamic_schedulers": int(args.runtime_samples),
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "validation_search_happened": False,
        "confirmatory_test_tuning_happened": False,
        "any_action_values_modified": bool(any_modified),
        "all_policy_prefixes_value_preserving": bool(not any_modified),
        "no_accidental_checkpoint_reuse": True,
        "old_custom_libero_7d_route_used": bool((loaded.get("audit") or {}).get("old_custom_libero_7d_route_used")),
        "cuda_ok": bool(torch.cuda.is_available()),
        "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "cuda_memory": _cuda_memory(torch),
        "errors": errors,
        "stage_a_rollout_allowed": bool(not errors),
        "next_step": (
            "Launch the frozen EAC Stage A rollout with the validated runner."
            if not errors
            else "Fix only the concrete runner validation defect before rollout."
        ),
    }
    return report


def _episode_key(row: Mapping[str, Any]) -> tuple[str, str, int, int]:
    return (str(row["policy"]), str(row["suite"]), int(row["task_id"]), int(row["reset_seed"]))


def _result_status_payload(
    *,
    status: str,
    args: argparse.Namespace,
    completed: int,
    planned: int,
    final_decision: str | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return {
        "status": status,
        "updated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "pid": os.getpid(),
        "planned_episode_count": int(planned),
        "completed_episode_count": int(completed),
        "partial_result": str(args.stage_a_partial_output),
        "final_result": str(args.stage_a_output),
        "final_decision": final_decision,
        "error": error,
    }


def _write_stage_status(args: argparse.Namespace, payload: Mapping[str, Any]) -> None:
    if not args.stage_a_status_output:
        return
    _write_json(Path(args.stage_a_status_output), payload)


def trace_one_stage_a_episode(
    *,
    env: Any,
    policy: Any,
    identity: Mapping[str, Any],
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    calibration: Mapping[str, Any],
    seed: int,
    args: argparse.Namespace,
    video_path: Path | None,
) -> dict[str, Any]:
    import torch
    from lerobot.scripts.lerobot_eval import (
        ACTION,
        add_envs_task,
        check_env_attributes_and_types,
        preprocess_observation,
        write_video,
    )

    if env.num_envs != 1:
        raise ValueError("EAC Stage A trace expects batch size 1")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    policy.reset()
    observation, _ = env.reset(seed=[int(seed)])
    max_steps = int(env.call("_max_episode_steps")[0])
    done = np.array([False])
    rewards: list[float] = []
    successes: list[bool] = []
    action_finite = True
    action_shape_ok = True
    action_max_abs = 0.0
    policy_latencies: list[float] = []
    env_latencies: list[float] = []
    pending_actions: list[np.ndarray] = []
    commitment_counts: Counter[int] = Counter()
    risk_values: list[float] = []
    dispersion_values: list[float] = []
    scheduler_values: list[float] = []
    schedule_trace: list[dict[str, Any]] = []
    chunks_generated = 0
    policy_calls = 0
    action_values_modified = False
    prefix_max_abs_diff = 0.0
    terminated_last = False
    truncated_last = False
    frames = []

    capture_video = video_path is not None
    if capture_video:
        frames.append(env.envs[0].render())

    check_env_attributes_and_types(env)
    step = 0
    while not np.all(done) and step < max_steps:
        if not pending_actions:
            lerobot_observation = preprocess_observation(observation)
            lerobot_observation = add_envs_task(env, lerobot_observation)
            lerobot_observation = env_preprocessor(lerobot_observation)
            batch = preprocessor(lerobot_observation)

            sample_count = _runtime_sample_count(identity, args)
            start_policy = time.perf_counter()
            chunks = _postprocessed_chunks(
                policy=policy,
                batch=batch,
                postprocessor=postprocessor,
                torch_mod=torch,
                reset_seed=int(seed),
                env_step=int(step),
                sample_count=sample_count,
                args=args,
            )
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            policy_latencies.append(time.perf_counter() - start_policy)
            policy_calls += int(sample_count)
            chunks_generated += 1

            schedule = _schedule_prefix(identity, chunks, calibration)
            prefix_max_abs_diff = max(prefix_max_abs_diff, float(schedule["prefix_max_abs_diff"]))
            action_values_modified = action_values_modified or bool(schedule["action_values_modified"])
            commitment = int(schedule["commitment"])
            commitment_counts[commitment] += 1
            risk_values.append(float(schedule["risk"]["risk"]))
            dispersion_values.append(float(schedule["risk"]["dispersion_norm"]))
            if schedule["scheduler_value"] is not None:
                scheduler_values.append(float(schedule["scheduler_value"]))
            pending_actions = [np.asarray(row, dtype=np.float32) for row in schedule["prefix"]]
            if len(schedule_trace) < 25:
                schedule_trace.append(
                    {
                        "env_step": int(step),
                        "commitment": commitment,
                        "scheduler_value": schedule["scheduler_value"],
                        "thresholds": schedule["thresholds"],
                        "risk": schedule["risk"],
                        "prefix_sha256": schedule["prefix_sha256"],
                        "prefix_max_abs_diff": schedule["prefix_max_abs_diff"],
                    }
                )

        action_policy_space = pending_actions.pop(0)
        device = _batch_device(batch, torch)
        action_tensor = torch.as_tensor(action_policy_space.reshape(1, 7), dtype=torch.float32, device=device)
        action_transition = {ACTION: action_tensor}
        action_transition = env_postprocessor(action_transition)
        action = action_transition[ACTION]
        action_numpy = action.to("cpu").numpy()
        action_finite = action_finite and bool(np.isfinite(action_numpy).all())
        action_shape_ok = action_shape_ok and action_numpy.shape == (1, 7)
        action_max_abs = max(action_max_abs, float(np.max(np.abs(action_numpy))))

        start_env = time.perf_counter()
        observation, reward, terminated, truncated, info = env.step(action_numpy)
        env_latencies.append(time.perf_counter() - start_env)
        if capture_video:
            frames.append(env.envs[0].render())

        step_successes = _successes_from_info(info, env.num_envs)
        successes.append(bool(step_successes[0]))
        rewards.append(float(np.asarray(reward).reshape(-1)[0]))
        terminated_last = bool(np.asarray(terminated).reshape(-1)[0])
        truncated_last = bool(np.asarray(truncated).reshape(-1)[0])

        done = terminated | truncated | done
        if step + 1 == max_steps:
            done = np.ones_like(done, dtype=bool)
        step += 1

    success = any(successes)
    sum_reward = float(np.sum(rewards)) if rewards else 0.0
    max_reward = float(np.max(rewards)) if rewards else 0.0
    if success:
        termination_reason = "success"
    elif terminated_last:
        termination_reason = "terminated_without_success"
    elif truncated_last or step >= max_steps:
        termination_reason = "max_steps_or_truncated_without_success"
    else:
        termination_reason = "done_without_success"

    saved_video_path = None
    if capture_video and frames:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        write_video(str(video_path), np.stack(frames), env.unwrapped.metadata["render_fps"])
        saved_video_path = str(video_path)

    return {
        "success": bool(success),
        "sum_reward": _round(sum_reward, 6),
        "max_reward": _round(max_reward, 6),
        "episode_length": int(step),
        "termination_reason": termination_reason,
        "failure_status": "success" if success else "unsuccessful",
        "exception": None,
        "action_validity": {
            "finite": bool(action_finite),
            "shape_ok": bool(action_shape_ok),
            "max_abs": _round(action_max_abs, 6),
        },
        "action_values_modified": bool(action_values_modified),
        "prefix_max_abs_diff": _round(prefix_max_abs_diff, 9),
        "action_chunks_generated": int(chunks_generated),
        "policy_calls": int(policy_calls),
        "policy_calls_per_step": _round(policy_calls / max(1, step), 6),
        "commitment_counts": {str(key): int(value) for key, value in sorted(commitment_counts.items())},
        "commitment_mean": _round(float(np.mean(list(commitment_counts.elements()))) if commitment_counts else None, 6),
        "mechanism_activation_fraction": _round(sum(value for key, value in commitment_counts.items() if key != 50) / max(1, chunks_generated), 6),
        "risk_mean": _round(float(np.mean(risk_values)) if risk_values else None, 6),
        "risk_max": _round(float(np.max(risk_values)) if risk_values else None, 6),
        "dispersion_norm_mean": _round(float(np.mean(dispersion_values)) if dispersion_values else None, 6),
        "scheduler_value_mean": _round(float(np.mean(scheduler_values)) if scheduler_values else None, 6),
        "schedule_trace_preview": schedule_trace,
        "env_steps": int(step),
        "policy_latency_mean_s": _round(float(np.mean(policy_latencies)), 6) if policy_latencies else None,
        "policy_latency_max_s": _round(float(np.max(policy_latencies)), 6) if policy_latencies else None,
        "env_step_latency_mean_s": _round(float(np.mean(env_latencies)), 6) if env_latencies else None,
        "env_step_latency_max_s": _round(float(np.max(env_latencies)), 6) if env_latencies else None,
        "peak_vram": _cuda_memory(torch),
        "rss_mb": _rss_mb(),
        "video_path": saved_video_path,
    }


def _mean(values: Sequence[float]) -> float | None:
    clean = [float(value) for value in values if value is not None]
    return _round(float(np.mean(clean)), 6) if clean else None


def summarize_stage_a(scaleup: Mapping[str, Any], manifest: Mapping[str, Any]) -> dict[str, Any]:
    rows = list(scaleup.get("episodes") or [])
    policies = list(manifest["policy_order"])
    by_policy: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_policy_task: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    by_pair: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        by_policy[str(row["policy"])].append(row)
        by_policy_task[(str(row["policy"]), str(row["suite"]), int(row["task_id"]))].append(row)
        by_pair[str(row["pair_id"])][str(row["policy"])] = row

    policy_summary = {}
    for policy in policies:
        policy_rows = by_policy.get(policy, [])
        successes = sum(1 for row in policy_rows if row.get("success"))
        total = len(policy_rows)
        task_rates = []
        for (p, _suite, _task_id), task_rows in by_policy_task.items():
            if p == policy:
                task_rates.append(sum(1 for row in task_rows if row.get("success")) / max(1, len(task_rows)))
        commitment_counter: Counter[str] = Counter()
        for row in policy_rows:
            commitment_counter.update({str(key): int(value) for key, value in (row.get("commitment_counts") or {}).items()})
        policy_summary[policy] = {
            "successes": successes,
            "total": total,
            "success_rate": _round(successes / total, 6) if total else None,
            "success_percent": _round(100 * successes / total, 3) if total else None,
            "wilson_95": wilson_interval(successes, total),
            "task_balanced_success_rate": _mean(task_rates),
            "avg_episode_length": _mean([row.get("episode_length") for row in policy_rows]),
            "avg_policy_latency_s": _mean([row.get("policy_latency_mean_s") for row in policy_rows]),
            "avg_policy_calls_per_step": _mean([row.get("policy_calls_per_step") for row in policy_rows]),
            "avg_action_chunks_generated": _mean([row.get("action_chunks_generated") for row in policy_rows]),
            "avg_mechanism_activation_fraction": _mean([row.get("mechanism_activation_fraction") for row in policy_rows]),
            "avg_risk_mean": _mean([row.get("risk_mean") for row in policy_rows]),
            "commitment_counts": dict(sorted(commitment_counter.items())),
            "action_values_modified": any(bool(row.get("action_values_modified")) for row in policy_rows),
            "action_validity_all_finite": all(bool((row.get("action_validity") or {}).get("finite")) for row in policy_rows) if policy_rows else False,
            "action_validity_all_shape_ok": all(bool((row.get("action_validity") or {}).get("shape_ok")) for row in policy_rows) if policy_rows else False,
            "peak_vram_mb": _round(max([float((row.get("peak_vram") or {}).get("max_allocated_mb") or 0.0) for row in policy_rows] or [0.0]), 3),
        }

    paired = {}
    eac_rows = by_policy.get(EAC_FULL_POLICY, [])
    for policy in policies:
        if policy == EAC_FULL_POLICY:
            continue
        counts = Counter()
        deltas = []
        for pair_id, pair_rows in by_pair.items():
            eac = pair_rows.get(EAC_FULL_POLICY)
            other = pair_rows.get(policy)
            if not eac or not other:
                continue
            eac_success = bool(eac.get("success"))
            other_success = bool(other.get("success"))
            deltas.append(float(eac_success) - float(other_success))
            if eac_success and not other_success:
                counts["win"] += 1
            elif other_success and not eac_success:
                counts["loss"] += 1
            else:
                counts["tie"] += 1
        paired[policy] = {
            "eac_full_wins": int(counts["win"]),
            "eac_full_losses": int(counts["loss"]),
            "ties": int(counts["tie"]),
            "paired_delta_eac_minus_policy": _round(float(np.mean(deltas)) if deltas else None, 6),
        }

    return {
        "policy_summary": policy_summary,
        "paired_vs_eac_full": paired,
        "eac_full_completed_episode_count": len(eac_rows),
        "eac_full_action_values_modified": any(bool(row.get("action_values_modified")) for row in eac_rows),
    }


def choose_stage_a_decision(scaleup: Mapping[str, Any], summary: Mapping[str, Any]) -> str:
    planned = int(scaleup.get("planned_episode_count") or 0)
    completed = int(scaleup.get("completed_episode_count") or 0)
    if int(scaleup.get("infrastructure_failure_count") or 0):
        return "EAC_STAGE_A_INFRASTRUCTURE_FAILURE_REPAIR_REQUIRED"
    if planned and completed < planned:
        return "EAC_STAGE_A_INCOMPLETE_RESUME_REQUIRED"
    policies = summary.get("policy_summary") or {}
    eac = policies.get(EAC_FULL_POLICY) or {}
    if summary.get("eac_full_action_values_modified"):
        return "EAC_STAGE_A_IMPLEMENTATION_FAILURE_ACTION_VALUES_MODIFIED"
    if not eac.get("action_validity_all_finite") or not eac.get("action_validity_all_shape_ok"):
        return "EAC_STAGE_A_IMPLEMENTATION_FAILURE_ACTION_INVALID"
    eac_commitments = eac.get("commitment_counts") or {}
    if eac_commitments and set(eac_commitments) == {"50"}:
        return "EAC_STAGE_A_MECHANISM_INVALID_NONACTING"

    eac_successes = int(eac.get("successes") or 0)
    eac_tb = float(eac.get("task_balanced_success_rate") or 0.0)
    other_stats = {name: stats for name, stats in policies.items() if name != EAC_FULL_POLICY}
    if eac_successes == 0 and any(int(stats.get("successes") or 0) >= 4 for stats in other_stats.values()):
        return "EAC_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE"
    for stats in other_stats.values():
        other_tb = float(stats.get("task_balanced_success_rate") or 0.0)
        other_successes = int(stats.get("successes") or 0)
        if other_tb - eac_tb >= 0.30 and other_successes - eac_successes >= 3:
            return "EAC_STAGE_A_CATASTROPHIC_KILL_CLEARLY_WORSE_THAN_BASELINE_OR_ABLATION"
    if other_stats and all(eac_tb > float(stats.get("task_balanced_success_rate") or 0.0) for stats in other_stats.values()):
        return "EAC_STAGE_A_POSITIVE_TO_STAGE_B_REQUIRED"
    return "EAC_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED"


def _partial_payload(
    *,
    args: argparse.Namespace,
    manifest: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    errors: Sequence[Mapping[str, Any]],
    started: float,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "date": f"{DATE_KST} KST",
        "method": "EAC-VLA",
        "mode": "stage_a_partial",
        "final_decision": "EAC_STAGE_A_IN_PROGRESS",
        "stage_a_manifest": str(args.stage_a_manifest),
        "stage_a_manifest_sha256": _sha256_file(Path(args.stage_a_manifest)),
        "planned_episode_count": int(manifest["planned_episode_count"]),
        "completed_episode_count": len(rows),
        "successful_episode_count": sum(1 for row in rows if row.get("success")),
        "infrastructure_failure_count": sum(1 for row in rows if row.get("failure_status") == "exception"),
        "policy_order": list(manifest["policy_order"]),
        "episodes": list(rows),
        "errors": list(errors),
        "closed_loop_experiment_happened": bool(rows),
        "training_happened": False,
        "validation_search_happened": False,
        "confirmatory_test_tuning_happened": False,
        "elapsed_seconds": _round(time.monotonic() - started, 3),
    }


def write_stage_a_result_md(path: Path, report: Mapping[str, Any]) -> None:
    summary = report["summary"]
    lines = [
        "# EAC-VLA Stage A Result",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- planned episodes: `{report['scaleup']['planned_episode_count']}`",
        f"- completed episodes: `{report['scaleup']['completed_episode_count']}`",
        f"- infrastructure failures: `{report['scaleup']['infrastructure_failure_count']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        "",
        "## Policy Success",
        "",
        "| Policy | Successes | Total | Rate | Task-balanced | Avg policy calls/step | Commitments |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for policy, stats in summary["policy_summary"].items():
        lines.append(
            f"| `{policy}` | `{stats['successes']}` | `{stats['total']}` | `{stats['success_percent']}%` | "
            f"`{stats['task_balanced_success_rate']}` | `{stats['avg_policy_calls_per_step']}` | "
            f"`{stats['commitment_counts']}` |"
        )
    lines.extend(
        [
            "",
            "## Paired Versus EAC Full",
            "",
            "```json",
            json.dumps(summary["paired_vs_eac_full"], indent=2, sort_keys=True, default=_json_default),
            "```",
            "",
            "The EAC scheduler preserves frozen SmolVLA action values and changes only queue commitment length; action-value modification would be an implementation failure.",
        ]
    )
    _write_lines_md(path, lines)


def run_stage_a(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from lerobot.envs.factory import make_env

    _set_runtime_env(args)
    started = time.monotonic()
    manifest = _read_json(Path(args.stage_a_manifest))
    runner_validation = _read_json(Path(args.stage_a_runner_validation_output))
    if runner_validation.get("final_decision") != "EAC_STAGE_A_RUNNER_VALIDATED_READY_FOR_ROLLOUT":
        raise RuntimeError("EAC Stage A runner validation has not passed")
    calibration = runner_validation["runtime_calibration"]
    identities = _policy_identity_map(manifest)
    planned_lookup = {_episode_key(row): dict(row) for row in manifest["episodes"]}
    partial_path = Path(args.stage_a_partial_output)
    if partial_path.exists():
        partial = _read_json(partial_path)
        rows: list[dict[str, Any]] = [dict(row) for row in partial.get("episodes") or []]
        errors: list[dict[str, Any]] = [dict(row) for row in partial.get("errors") or []]
    else:
        rows = []
        errors = []
    completed_keys = {_episode_key(row) for row in rows}

    loaded = _load_policy_and_processors(args, PolicySpec("frozen_base"))
    policy_audit = loaded["audit"]
    launched_this_run = 0
    _write_stage_status(
        args,
        _result_status_payload(
            status="running",
            args=args,
            completed=len(rows),
            planned=int(manifest["planned_episode_count"]),
            final_decision="EAC_STAGE_A_IN_PROGRESS",
        ),
    )

    try:
        for policy_name in manifest["policy_order"]:
            identity = identities[str(policy_name)]
            for task in manifest["tasks"]:
                env = None
                try:
                    env_cfg = _make_env_cfg(str(task["suite"]), [int(task["task_id"])])
                    env = _extract_single_env(make_env(env_cfg, n_envs=1, use_async_envs=False), str(task["suite"]), int(task["task_id"]))
                    for seed in manifest["reset_seeds"]:
                        key = (str(policy_name), str(task["suite"]), int(task["task_id"]), int(seed))
                        if key in completed_keys:
                            continue
                        if args.limit_episodes and launched_this_run >= int(args.limit_episodes):
                            break
                        planned = planned_lookup[key]
                        row = dict(planned)
                        video_path = None
                        if args.capture_failure_videos:
                            video_path = Path(args.video_dir) / policy_name / task["suite"] / f"task_{task['task_id']}_seed_{seed}.mp4"
                        print(f"[eac-stage-a] {policy_name} {task['suite']} task_{task['task_id']} seed {seed}", flush=True)
                        try:
                            trace = trace_one_stage_a_episode(
                                env=env,
                                policy=loaded["policy"],
                                identity=identity,
                                env_preprocessor=loaded["env_preprocessor"],
                                env_postprocessor=loaded["env_postprocessor"],
                                preprocessor=loaded["preprocessor"],
                                postprocessor=loaded["postprocessor"],
                                calibration=calibration,
                                seed=int(seed),
                                args=args,
                                video_path=video_path if args.capture_failure_videos else None,
                            )
                            if args.capture_failure_videos and trace["success"] and trace.get("video_path"):
                                Path(trace["video_path"]).unlink(missing_ok=True)
                                trace["video_path"] = None
                            row.update(trace)
                        except Exception as exc:  # pragma: no cover - simulator boundary
                            exception = {
                                "type": type(exc).__name__,
                                "message": str(exc),
                                "traceback": traceback.format_exc().splitlines()[-24:],
                            }
                            row.update(
                                {
                                    "success": False,
                                    "sum_reward": None,
                                    "max_reward": None,
                                    "episode_length": None,
                                    "termination_reason": "exception",
                                    "failure_status": "exception",
                                    "exception": exception,
                                    "action_validity": {"finite": False, "shape_ok": False, "max_abs": None},
                                    "action_values_modified": None,
                                    "prefix_max_abs_diff": None,
                                    "action_chunks_generated": None,
                                    "policy_calls": None,
                                    "policy_calls_per_step": None,
                                    "commitment_counts": {},
                                    "mechanism_activation_fraction": None,
                                    "peak_vram": _cuda_memory(torch),
                                    "rss_mb": _rss_mb(),
                                    "video_path": None,
                                }
                            )
                            errors.append({"episode_id": row["episode_id"], **exception})
                        rows.append(row)
                        completed_keys.add(key)
                        launched_this_run += 1
                        partial = _partial_payload(args=args, manifest=manifest, rows=rows, errors=errors, started=started)
                        _write_json(partial_path, partial)
                        _write_stage_status(
                            args,
                            _result_status_payload(
                                status="running",
                                args=args,
                                completed=len(rows),
                                planned=int(manifest["planned_episode_count"]),
                                final_decision="EAC_STAGE_A_IN_PROGRESS",
                            ),
                        )
                    if args.limit_episodes and launched_this_run >= int(args.limit_episodes):
                        break
                finally:
                    if env is not None:
                        try:
                            env.close()
                        except Exception:
                            pass
                if args.limit_episodes and launched_this_run >= int(args.limit_episodes):
                    break
            if args.limit_episodes and launched_this_run >= int(args.limit_episodes):
                break
    finally:
        torch.cuda.empty_cache()

    scaleup = {
        "executed": True,
        "planned_episode_count": int(manifest["planned_episode_count"]),
        "completed_episode_count": sum(1 for row in rows if row.get("failure_status") != "exception"),
        "row_count": len(rows),
        "successful_episode_count": sum(1 for row in rows if row.get("success")),
        "infrastructure_failure_count": sum(1 for row in rows if row.get("failure_status") == "exception"),
        "episodes": rows,
        "policy_load_audit": policy_audit,
        "errors": errors,
        "elapsed_seconds": _round(time.monotonic() - started, 3),
    }
    summary = summarize_stage_a(scaleup, manifest)
    final_decision = choose_stage_a_decision(scaleup, summary)
    report = {
        "schema_version": 1,
        "date": f"{DATE_KST} KST",
        "method": "EAC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "mode": "stage_a",
        "branch": "codex/autonomous-until-paper-governance-v2",
        "stage_a_manifest": str(args.stage_a_manifest),
        "stage_a_manifest_sha256": _sha256_file(Path(args.stage_a_manifest)),
        "stage_a_runner_validation": str(args.stage_a_runner_validation_output),
        "stage_a_runner_validation_sha256": _sha256_file(Path(args.stage_a_runner_validation_output)),
        "policy_order": list(manifest["policy_order"]),
        "scaleup": scaleup,
        "summary": summary,
        "closed_loop_experiment_happened": True,
        "training_happened": False,
        "validation_search_happened": False,
        "confirmatory_test_tuning_happened": False,
        "final_decision": final_decision,
        "stage_b_required": final_decision
        in {
            "EAC_STAGE_A_POSITIVE_TO_STAGE_B_REQUIRED",
            "EAC_STAGE_A_NONCATASTROPHIC_TO_STAGE_B_REQUIRED",
        },
        "valid_current_formulation_kill": final_decision
        in {
            "EAC_STAGE_A_CATASTROPHIC_KILL_ZERO_VS_STRONG_BASELINE",
            "EAC_STAGE_A_CATASTROPHIC_KILL_CLEARLY_WORSE_THAN_BASELINE_OR_ABLATION",
        },
    }
    _write_json(Path(args.stage_a_output), report)
    write_stage_a_result_md(Path(args.stage_a_md_output), report)
    _write_stage_status(
        args,
        _result_status_payload(
            status="completed" if final_decision != "EAC_STAGE_A_INCOMPLETE_RESUME_REQUIRED" else "incomplete",
            args=args,
            completed=len(rows),
            planned=int(manifest["planned_episode_count"]),
            final_decision=final_decision,
        ),
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=["freeze-stage-a-manifest", "preflight", "runner-validate", "stage-a"],
        default="freeze-stage-a-manifest",
    )
    parser.add_argument("--base-path", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--runtime-seed", type=int, default=20260715)
    parser.add_argument("--runtime-samples", type=int, default=EAC_RUNTIME_SAMPLES)
    parser.add_argument("--canonical-artifact", default="reports/canonical_frozen_base_prediction_artifact.json")
    parser.add_argument("--selected-config", default="reports/eac_vla/selected_config.json")
    parser.add_argument("--stage-a-manifest", default="reports/eac_vla/stage_a_manifest.json")
    parser.add_argument("--stage-a-manifest-md", default="reports/eac_vla/stage_a_manifest.md")
    parser.add_argument("--stage-a-preflight-output", default="reports/eac_vla/stage_a_preflight.json")
    parser.add_argument("--stage-a-preflight-md", default="reports/eac_vla/stage_a_preflight.md")
    parser.add_argument("--stage-a-runner-validation-output", default="reports/eac_vla/stage_a_runner_validation.json")
    parser.add_argument("--stage-a-runner-validation-md", default="reports/eac_vla/stage_a_runner_validation.md")
    parser.add_argument("--stage-a-partial-output", default="reports/eac_vla/stage_a_partial_result.json")
    parser.add_argument("--stage-a-output", default="reports/eac_vla/stage_a_result.json")
    parser.add_argument("--stage-a-md-output", default="reports/eac_vla/stage_a_result.md")
    parser.add_argument("--stage-a-status-output", default="reports/eac_vla/stage_a_status.json")
    parser.add_argument("--video-dir", default="runs/eac_vla_stage_a_videos")
    parser.add_argument("--capture-failure-videos", action="store_true")
    parser.add_argument("--limit-episodes", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.mode == "freeze-stage-a-manifest":
        report = build_manifest(args)
        _write_json(Path(args.stage_a_manifest), report)
        _write_md(Path(args.stage_a_manifest_md), report)
    elif args.mode == "preflight":
        if not Path(args.stage_a_manifest).exists():
            manifest = build_manifest(args)
            _write_json(Path(args.stage_a_manifest), manifest)
            _write_md(Path(args.stage_a_manifest_md), manifest)
        report = build_preflight(args)
        _write_json(Path(args.stage_a_preflight_output), report)
        _write_md(Path(args.stage_a_preflight_md), report)
    elif args.mode == "runner-validate":
        report = build_runner_validation(args)
        _write_json(Path(args.stage_a_runner_validation_output), report)
        _runner_validation_md(Path(args.stage_a_runner_validation_md), report)
    else:
        report = run_stage_a(args)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "final_decision": report["final_decision"],
                "planned_episode_count": report.get("planned_episode_count")
                or ((report.get("scaleup") or {}).get("planned_episode_count")),
                "policy_count": len(report.get("policy_identities") or report.get("policy_order") or []),
                "error_count": len(report.get("errors") or []),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
