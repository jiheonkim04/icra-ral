"""DICD-VLA prototype runner.

The fast path runs a synthetic mechanism smoke that exercises the trainable
adapter, checkpoint roundtrip, action-change checks, and no-privileged-input
guard.  The real closed-loop stage builds on the same helpers for extracting
postprocessed SmolVLA action chunks.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_echo_vla_first_prototype import _postprocess_action, _preprocess_batch  # noqa: E402
from scripts.run_phase_barrier_vla_prototype import _round, _write_json, _write_md  # noqa: E402
from tca_map.smolvla.dicd_vla import (  # noqa: E402
    DICDConfig,
    assert_no_privileged_inference_fields,
    build_dicd_examples,
    direct_chunk_index_action,
    file_sha256,
    load_dicd_checkpoint,
    make_dicd_features,
    predict_dicd_action,
    save_dicd_checkpoint,
    train_dicd_adapter,
)
from tca_map.smolvla.official_closed_loop_scaleup import _json_default  # noqa: E402


DATE_KST = "2026-07-12"
BRANCH = "codex/auto-method-20260712-01-dicd-vla"


def _synthetic_chunk(offset: float, *, chunk_len: int, action_dim: int) -> np.ndarray:
    rows = []
    for index in range(chunk_len):
        base = np.zeros(action_dim, dtype=np.float32)
        base[0] = float(offset + 0.08 * index)
        base[1] = float(0.05 * np.sin(offset + index))
        base[2] = float(-0.04 * np.cos(offset + index))
        base[3] = float(0.03 * index)
        base[4] = float(0.02 * offset)
        base[5] = float(-0.01 * index)
        base[6] = -1.0
        rows.append(base)
    return np.stack(rows, axis=0)


def _synthetic_trace(config: DICDConfig, count: int, delay: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    chunks = [_synthetic_chunk(index * 0.03, chunk_len=config.chunk_len, action_dim=config.action_dim) for index in range(count)]
    executed: list[np.ndarray] = []
    previous = np.zeros(config.action_dim, dtype=np.float32)
    for index, chunk in enumerate(chunks):
        delayed = direct_chunk_index_action(chunk, min(delay, config.chunk_len - 1), config).reshape(-1)
        action = delayed + 0.18 * previous
        action[6] = -1.0
        executed.append(action.astype(np.float32))
        previous = action
    return chunks, executed


def _postprocess_action_chunk(action_chunk: Any, loaded: Mapping[str, Any], *, max_chunk_len: int | None = None) -> np.ndarray:
    import torch

    chunk = action_chunk
    if isinstance(chunk, torch.Tensor):
        tensor = chunk.detach()
    else:
        tensor = torch.as_tensor(chunk)
    if tensor.ndim == 3:
        tensor = tensor[0]
    if tensor.ndim == 1:
        tensor = tensor.reshape(1, -1)
    if tensor.ndim != 2:
        raise ValueError(f"unsupported action chunk rank {tensor.ndim}")
    if max_chunk_len is not None:
        tensor = tensor[: int(max_chunk_len)]
    rows = []
    for row in tensor:
        rows.append(_postprocess_action(row.reshape(1, -1), dict(loaded)).reshape(-1))
    return np.stack(rows, axis=0).astype(np.float32)


def predict_postprocessed_action_chunk(policy: Any, env: Any, observation: Any, loaded: Mapping[str, Any], *, max_chunk_len: int | None = None) -> np.ndarray:
    import torch

    batch = _preprocess_batch(env, observation, dict(loaded))
    with torch.inference_mode():
        raw_chunk = policy.predict_action_chunk(batch)
    return _postprocess_action_chunk(raw_chunk, loaded, max_chunk_len=max_chunk_len)


def _run_synthetic_mechanism_smoke(args: argparse.Namespace) -> dict[str, Any]:
    config = DICDConfig(action_dim=int(args.action_dim), chunk_len=int(args.chunk_len), history_len=int(args.history_len), hidden_dim=int(args.hidden_dim))
    chunks, executed = _synthetic_trace(config, int(args.synthetic_count), int(args.delay))
    full_examples = build_dicd_examples(chunks, executed, delay=int(args.delay), config=config, use_history=True)
    ablation_examples = build_dicd_examples(chunks, executed, delay=int(args.delay), config=config, use_history=False)
    full_model, full_stats = train_dicd_adapter(full_examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=int(args.seed))
    ablation_model, ablation_stats = train_dicd_adapter(ablation_examples, config=config, epochs=int(args.epochs), lr=float(args.lr), seed=int(args.seed))
    checkpoint_path = Path(args.checkpoint_path)
    save_dicd_checkpoint(checkpoint_path, full_model, full_stats)
    loaded_model, loaded_stats = load_dicd_checkpoint(checkpoint_path)
    checkpoint_hash = file_sha256(checkpoint_path)

    probe_index = min(6, len(chunks) - int(args.delay) - 1)
    probe_history = executed[max(0, probe_index - config.history_len) : probe_index]
    probe_features = make_dicd_features(
        chunks[probe_index],
        history=probe_history,
        delay=int(args.delay),
        step_fraction=probe_index / max(1.0, float(len(chunks) - 1)),
        config=config,
        use_history=True,
    )
    ablation_features = make_dicd_features(
        chunks[probe_index],
        history=probe_history,
        delay=int(args.delay),
        step_fraction=probe_index / max(1.0, float(len(chunks) - 1)),
        config=config,
        use_history=False,
    )
    direct = direct_chunk_index_action(chunks[probe_index], int(args.delay), config)
    in_memory = predict_dicd_action(full_model, probe_features)
    reloaded = predict_dicd_action(loaded_model, probe_features)
    ablation = predict_dicd_action(ablation_model, ablation_features)
    assert_no_privileged_inference_fields(["observation", "instruction", "action_chunk", "executed_action_history", "declared_delay", "step_fraction"])

    checks = {
        "chunk_horizon_exceeds_delay": bool(config.chunk_len > int(args.delay)),
        "full_finite_gradients": bool(full_stats["finite_gradients"]),
        "full_loss_decreased": bool(full_stats["loss_decreased"]),
        "ablation_loss_decreased": bool(ablation_stats["loss_decreased"]),
        "checkpoint_reloaded": bool(np.allclose(in_memory, reloaded)),
        "full_changes_direct_chunk_index": bool(np.linalg.norm(in_memory - direct) > float(args.min_action_delta)),
        "full_differs_from_no_history_ablation": bool(np.linalg.norm(in_memory - ablation) > float(args.min_action_delta)),
        "no_privileged_inference_fields": True,
    }
    return {
        "schema_version": "dicd_vla_mechanism_smoke_v1",
        "date_kst": DATE_KST,
        "branch": BRANCH,
        "smoke_type": "synthetic_core_mechanism",
        "real_smolvla_chunk_smoke_happened": False,
        "closed_loop_experiment_happened": False,
        "training_happened": True,
        "config": config.__dict__,
        "delay": int(args.delay),
        "training_example_count_full": int(len(full_examples)),
        "training_example_count_ablation": int(len(ablation_examples)),
        "full_train_stats": full_stats,
        "ablation_train_stats": ablation_stats,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_hash,
        "loaded_stats": loaded_stats,
        "probe": {
            "probe_index": int(probe_index),
            "direct_chunk_index_action": direct.reshape(-1).tolist(),
            "dicd_full_action": in_memory.reshape(-1).tolist(),
            "dicd_reloaded_action": reloaded.reshape(-1).tolist(),
            "dicd_no_history_action": ablation.reshape(-1).tolist(),
            "full_vs_direct_delta_norm": _round(float(np.linalg.norm(in_memory - direct)), 6),
            "full_vs_ablation_delta_norm": _round(float(np.linalg.norm(in_memory - ablation)), 6),
        },
        "checks": checks,
        "mechanism_smoke_passed": bool(all(checks.values())),
        "final_decision": "DICD_SYNTHETIC_MECHANISM_SMOKE_PASSED" if all(checks.values()) else "DICD_SYNTHETIC_MECHANISM_SMOKE_FAILED",
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    report: dict[str, Any]
    try:
        report = _run_synthetic_mechanism_smoke(args)
    except Exception as exc:  # pragma: no cover - runtime boundary
        report = {
            "schema_version": "dicd_vla_mechanism_smoke_v1",
            "date_kst": DATE_KST,
            "branch": BRANCH,
            "errors": [{"type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc().splitlines()[-80:]}],
            "mechanism_smoke_passed": False,
            "final_decision": "DICD_SYNTHETIC_MECHANISM_SMOKE_FAILED",
        }
    report["elapsed_seconds"] = _round(time.monotonic() - started, 3)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    _write_json(output_json, report)
    _write_md(
        output_md,
        [
            "# DICD-VLA Mechanism Smoke Result",
            "",
            f"Date: `{DATE_KST}`",
            "",
            f"Final decision: `{report.get('final_decision')}`",
            "",
            f"- smoke type: `{report.get('smoke_type')}`",
            f"- mechanism smoke passed: `{report.get('mechanism_smoke_passed')}`",
            f"- training happened: `{report.get('training_happened')}`",
            f"- real SmolVLA chunk smoke happened: `{report.get('real_smolvla_chunk_smoke_happened')}`",
            f"- closed-loop experiment happened: `{report.get('closed_loop_experiment_happened')}`",
            f"- checkpoint: `{report.get('checkpoint_path')}`",
            f"- checkpoint sha256: `{report.get('checkpoint_sha256')}`",
            f"- checks: `{report.get('checks')}`",
            f"- probe: `{report.get('probe')}`",
            f"- elapsed seconds: `{report.get('elapsed_seconds')}`",
            "",
            "Next step: run real SmolVLA chunk smoke before Stage A closed-loop rollout.",
        ],
    )
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DICD-VLA prototype smoke.")
    parser.add_argument("--output-json", default="reports/dicd_vla/mechanism_smoke_result.json")
    parser.add_argument("--output-md", default="reports/dicd_vla/mechanism_smoke_result.md")
    parser.add_argument("--checkpoint-path", default="reports/dicd_vla/checkpoints/dicd_synthetic_smoke.pt")
    parser.add_argument("--delay", type=int, default=2)
    parser.add_argument("--chunk-len", type=int, default=8)
    parser.add_argument("--history-len", type=int, default=2)
    parser.add_argument("--action-dim", type=int, default=7)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--synthetic-count", type=int, default=18)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--min-action-delta", type=float, default=1e-4)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    report = run(parse_args(argv))
    print(json.dumps({"final_decision": report.get("final_decision"), "checks": report.get("checks")}, indent=2, sort_keys=True, default=_json_default))
    return 0 if report.get("mechanism_smoke_passed") else 2


if __name__ == "__main__":
    raise SystemExit(main())
