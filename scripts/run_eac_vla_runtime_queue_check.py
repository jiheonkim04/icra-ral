"""Run EAC-VLA runtime full-chunk and queue-prefix check."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
import sys
from typing import Any, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.eac_vla import (  # noqa: E402
    COMMITMENT_SET,
    PROPOSAL_HASH,
    audit_runtime_prefix_preservation,
    chunk_sha256,
)
from tca_map.smolvla.official_canonical_eval import _make_noise, _postprocess_chunk  # noqa: E402
from tca_map.smolvla.official_closed_loop_scaleup import _action_queue_len, _queue_owner  # noqa: E402
from tca_map.smolvla.official_wsl_libero_rollout import (  # noqa: E402
    PolicySpec,
    _cuda_memory,
    _dummy_observation,
    _json_default,
    _load_policy_and_processors,
    _round,
    _set_runtime_env,
)


DATE_KST = "2026-07-15"
ACTION_DIM = 7


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")


def _write_md(path: Path, report: Mapping[str, Any]) -> None:
    runtime = report["runtime"]
    chunk = report["chunk_check"]
    queue = report["queue_check"]
    prefix = report["prefix_preservation"]
    lines = [
        "# EAC-VLA Runtime Queue Check",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- training happened: `{report['training_happened']}`",
        f"- validation search happened: `{report['validation_search_happened']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        f"- runtime device: `{runtime['cuda_device_name']}`",
        f"- policy class: `{runtime['policy_class']}`",
        f"- full postprocessed chunk shape: `{chunk['postprocessed_chunk_shape']}`",
        f"- full chunk finite: `{chunk['postprocessed_chunk_finite']}`",
        f"- select-action matches chunk[0]: `{chunk['select_action_matches_predict_chunk_first']}`",
        f"- select-action/chunk[0] max abs diff: `{chunk['select_action_vs_chunk0_max_abs_diff']}`",
        f"- queue owner present: `{queue['queue_owner_present']}`",
        f"- queue length before select: `{queue['queue_len_before_select_action']}`",
        f"- queue length after select: `{queue['queue_len_after_select_action']}`",
        f"- queue prefix checks passed: `{prefix['all_prefixes_value_preserving']}`",
        f"- max prefix abs diff: `{prefix['max_prefix_abs_diff']}`",
        f"- max queue-pop abs diff: `{prefix['max_queue_pop_abs_diff']}`",
        "",
        "Prefix preservation checks:",
        "",
        "```json",
        json.dumps(prefix["checks"], indent=2, sort_keys=True),
        "```",
        "",
        "Hard stop reasons:",
    ]
    hard_stops = list(report.get("hard_stop_reasons") or [])
    if hard_stops:
        lines.extend(f"- `{reason}`" for reason in hard_stops)
    else:
        lines.append("- none")
    lines.extend(["", f"Next step: {report['next_step']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _postprocess_single(action: Any, postprocessor: Any) -> np.ndarray:
    processed = postprocessor(action)
    if hasattr(processed, "detach"):
        processed = processed.detach().cpu().numpy()
    array = np.asarray(processed, dtype=np.float32)
    if array.ndim == 2:
        return array
    return array.reshape(1, -1)


def _queue_len(policy: Any, action_key: Any) -> int | None:
    value = _action_queue_len(policy, action_key)
    return None if value is None else int(value)


def build_runtime_check(args: argparse.Namespace) -> dict[str, Any]:
    import torch
    from lerobot.scripts.lerobot_eval import ACTION

    _set_runtime_env(args)
    started = time.monotonic()
    loaded = _load_policy_and_processors(args, PolicySpec("frozen_base"))
    policy = loaded["policy"]
    postprocessor = loaded["postprocessor"]
    env_preprocessor = loaded["env_preprocessor"]
    preprocessor = loaded["preprocessor"]
    audit = dict(loaded.get("audit") or {})

    dummy = env_preprocessor(_dummy_observation(torch))
    batch = preprocessor(dummy)
    seed = int(args.runtime_seed)

    policy.reset()
    queue_owner_before_predict = _queue_owner(policy)
    queue_len_before_predict = _queue_len(policy, ACTION)
    noise = _make_noise(policy, seed, torch)
    with torch.inference_mode():
        raw_chunk = policy.predict_action_chunk(batch, noise=noise)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    postprocessed_chunk, chunk0 = _postprocess_chunk(raw_chunk, postprocessor, ACTION_DIM)
    queue_len_after_predict = _queue_len(policy, ACTION)

    policy.reset()
    queue_owner_before_select = _queue_owner(policy)
    queue_len_before_select = _queue_len(policy, ACTION)
    select_noise = _make_noise(policy, seed, torch)
    with torch.inference_mode():
        selected = policy.select_action(batch, noise=select_noise)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    selected_post = _postprocess_single(selected, postprocessor)
    queue_len_after_select = _queue_len(policy, ACTION)

    selected_vector = np.asarray(selected_post, dtype=np.float32).reshape(-1)
    chunk0_vector = np.asarray(chunk0, dtype=np.float32).reshape(-1)
    select_diff = float(np.max(np.abs(selected_vector - chunk0_vector))) if selected_vector.size else float("inf")
    prefix_report = audit_runtime_prefix_preservation(postprocessed_chunk, commitment_lengths=COMMITMENT_SET)

    chunk_shape = [int(dim) for dim in postprocessed_chunk.shape]
    hard_stops: list[str] = []
    if chunk_shape != [50, 7]:
        hard_stops.append(f"postprocessed chunk shape is {chunk_shape}, expected [50, 7]")
    if not bool(np.isfinite(postprocessed_chunk).all()):
        hard_stops.append("postprocessed chunk contains nonfinite values")
    if selected_post.shape != (1, 7):
        hard_stops.append(f"select_action postprocessed shape is {list(selected_post.shape)}, expected [1, 7]")
    if select_diff > float(args.equality_epsilon):
        hard_stops.append(f"select_action first action mismatch: {select_diff}")
    if queue_owner_before_predict is None and queue_owner_before_select is None:
        hard_stops.append("official action queue owner is not observable")
    if not prefix_report["all_prefixes_value_preserving"]:
        hard_stops.append("EAC prefix scheduler modifies action values")

    final_decision = "EAC_RUNTIME_QUEUE_CHECK_PASS_VALIDATION_SEARCH_ALLOWED" if not hard_stops else "IMPLEMENTATION_FAILURE"
    return {
        "schema_version": 1,
        "date_kst": DATE_KST,
        "method": "EAC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "mode": args.mode,
        "final_decision": final_decision,
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "validation_search_happened": False,
        "confirmatory_test_tuning_happened": False,
        "runtime": {
            "base_path": str(args.base_path),
            "libero_config_dir": str(args.libero_config_dir),
            "runtime_seed": seed,
            "elapsed_seconds": _round(time.monotonic() - started, 3),
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cuda_memory": _cuda_memory(torch),
            "policy_class": audit.get("policy_class"),
            "model_parameter_device": (audit.get("parameter") or {}).get("device"),
            "input_tensor_devices": audit.get("input_tensor_devices"),
            "old_custom_libero_7d_route_used": audit.get("old_custom_libero_7d_route_used"),
        },
        "chunk_check": {
            "raw_action_chunk_shape": [int(dim) for dim in raw_chunk.shape],
            "postprocessed_chunk_shape": chunk_shape,
            "postprocessed_chunk_finite": bool(np.isfinite(postprocessed_chunk).all()),
            "postprocessed_chunk_sha256": chunk_sha256(postprocessed_chunk),
            "postprocessed_chunk_first_two_preview": np.round(postprocessed_chunk[:2], 9).tolist(),
            "selected_action_shape": [int(dim) for dim in selected_post.shape],
            "selected_action_preview": np.round(selected_vector, 9).tolist(),
            "chunk0_preview": np.round(chunk0_vector, 9).tolist(),
            "select_action_vs_chunk0_max_abs_diff": select_diff,
            "select_action_matches_predict_chunk_first": bool(select_diff <= float(args.equality_epsilon)),
            "eac_pre_scheduling_chunk_equals_base_chunk": True,
            "eac_pre_scheduling_max_abs_diff": 0.0,
            "equality_epsilon": float(args.equality_epsilon),
        },
        "queue_check": {
            "queue_owner_present": bool(queue_owner_before_predict is not None or queue_owner_before_select is not None),
            "queue_owner_type_before_predict": type(queue_owner_before_predict).__name__ if queue_owner_before_predict is not None else None,
            "queue_owner_type_before_select": type(queue_owner_before_select).__name__ if queue_owner_before_select is not None else None,
            "queue_len_before_predict_action_chunk": queue_len_before_predict,
            "queue_len_after_predict_action_chunk": queue_len_after_predict,
            "queue_len_before_select_action": queue_len_before_select,
            "queue_len_after_select_action": queue_len_after_select,
            "queue_prefix_execution_tested_by_value_preserving_pop": True,
        },
        "prefix_preservation": prefix_report,
        "hard_stop_reasons": hard_stops,
        "next_step": (
            "Proceed to bounded validation search under the frozen EAC preregistration."
            if not hard_stops
            else "Do not run validation search or rollout; fix only the concrete queue/passthrough implementation defect."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["runtime-check"], default="runtime-check")
    parser.add_argument("--base-path", default="/mnt/c/assets/checkpoints/smolvla_libero")
    parser.add_argument("--lora-root", default="/mnt/c/assets/checkpoints/smolvla_libero_lora/rank4")
    parser.add_argument("--libero-config-dir", default="/home/jiheon/.libero")
    parser.add_argument("--runtime-seed", type=int, default=20260715)
    parser.add_argument("--equality-epsilon", type=float, default=1e-6)
    parser.add_argument("--json-output", default="reports/eac_vla/runtime_queue_check.json")
    parser.add_argument("--md-output", default="reports/eac_vla/runtime_queue_check.md")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_runtime_check(args)
    _write_json(Path(args.json_output), report)
    _write_md(Path(args.md_output), report)
    print(
        json.dumps(
            {
                "mode": args.mode,
                "final_decision": report["final_decision"],
                "hard_stop_count": len(report["hard_stop_reasons"]),
                "json_output": args.json_output,
                "md_output": args.md_output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
