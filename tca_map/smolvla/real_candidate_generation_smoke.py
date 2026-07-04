"""Bounded real candidate-generation smoke for SmolVLA/TCA-Select.

The default path is refusal-only. Real loading and one-sample inference require
three task-local gates:

- ALLOW_REAL_CANDIDATE_GENERATION_SMOKE=1
- ALLOW_HEAVY_IMPORT=1
- ALLOW_SINGLE_SAMPLE_INFERENCE=1

Even with the gates, this is an engineering smoke only. It does not train,
rollout, create simulator environments, use external verifiers, use privileged
state, execute OpenVLA-OFT, download assets, or make paper-grade claims.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from tca_map.inference.tca_select import distributional_tca_select_inference
from tca_map.smolvla.load_only_smoke import (
    _env_flag,
    _external_tokenizer_files,
    _find_files,
    _nvidia_smi,
    _read_tokenizer_dependency,
    _rss_mb,
    _runtime_dependencies,
)


REQUIRED_GATES = [
    "ALLOW_REAL_CANDIDATE_GENERATION_SMOKE",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
]
FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_TINY_TRAINING",
    "ALLOW_GPU_TRAINING",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_RUNTIME_INSTALL",
    "ALLOW_SIMULATOR_IMPORT_SMOKE",
    "ALLOW_SIMULATOR_RENDER_SMOKE",
    "ALLOW_SIMULATOR_RESET_STEP",
    "ALLOW_TINY_ROLLOUT",
]
MAX_RUNTIME_SECONDS = 600
MAX_VRAM_MB = 14336
MAX_CANDIDATES = 4
MAX_HEATMAP_GRID = 8
DEFAULT_ACTION_DIM = 6


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _flatten_action(action: Any) -> list[float]:
    if hasattr(action, "detach"):
        action = action.detach().cpu().reshape(-1).tolist()
    if hasattr(action, "reshape") and hasattr(action, "tolist"):
        action = action.reshape(-1).tolist()
    if isinstance(action, (int, float)):
        return [float(action)]
    return [float(value) for value in list(action)]


def _candidate_action(seed_action: list[float], offset: float, index: int) -> list[float]:
    direction = -1.0 if index % 2 else 1.0
    return [round(float(value) + direction * offset * (dim + 1), 6) for dim, value in enumerate(seed_action)]


def build_candidate_heatmaps(
    seed_action: Any,
    candidate_count: int = MAX_CANDIDATES,
    heatmap_grid: int = MAX_HEATMAP_GRID,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Build a low-res target-conditioned candidate heatmap from one action.

    This is intentionally simple: it turns one decoded action into a small
    local candidate set for interface validation. It is not a learned decoder
    and should not be treated as benchmark evidence.
    """
    if candidate_count < 2 or candidate_count > MAX_CANDIDATES:
        raise ValueError(f"candidate_count must be between 2 and {MAX_CANDIDATES}")
    if heatmap_grid < 2 or heatmap_grid > MAX_HEATMAP_GRID:
        raise ValueError(f"heatmap_grid must be between 2 and {MAX_HEATMAP_GRID}")

    action = _flatten_action(seed_action)
    if not action:
        action = [0.0] * DEFAULT_ACTION_DIM

    candidates: list[dict[str, Any]] = []
    for index in range(candidate_count):
        target_consistent = index == 0 or index % 2 == 0
        if index == 0:
            candidate_action = [round(float(value), 6) for value in action]
        else:
            candidate_action = _candidate_action(action, offset=0.015 * index, index=index)
        candidates.append(
            {
                "index": index,
                "voxel": index,
                "grid_index": [
                    index % heatmap_grid,
                    (index // heatmap_grid) % heatmap_grid,
                    min(index // max(1, heatmap_grid * heatmap_grid), heatmap_grid - 1),
                ],
                "action": candidate_action,
                "logit": round(2.0 - 0.15 * index, 6),
                "target_index": 0 if target_consistent else 1,
                "source": "real_single_sample_candidate_smoke",
            }
        )

    action_heatmap = {
        "grid_size": heatmap_grid,
        "action_dim": len(action),
        "candidates": candidates,
        "low_resolution": True,
        "coarse_to_fine_ready": True,
        "candidate_source": "single_smolvla_select_action_seed",
    }
    masked_heatmap = {
        **action_heatmap,
        "candidates": [
            {
                **candidate,
                "logit": round(
                    float(candidate["logit"]) - (0.45 if candidate["target_index"] == 0 else 0.05),
                    6,
                ),
            }
            for candidate in candidates
        ],
        "candidate_source": "language_masked_candidate_proxy",
    }
    negative_action_heatmap = {
        **action_heatmap,
        "candidates": [
            {
                **candidate,
                "logit": round(
                    float(candidate["logit"]) + (0.25 if candidate["target_index"] != 0 else -0.25),
                    6,
                ),
            }
            for candidate in candidates
        ],
        "candidate_source": "counterfactual_negative_candidate_proxy",
    }
    target_heatmap = {"scores": [1.0, 0.1], "top_index": 0, "low_resolution": True}
    return action_heatmap, masked_heatmap, negative_action_heatmap, target_heatmap


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    result = report.get("result") or {}
    selection = report.get("selection") or {}
    lines = [
        "# Real Candidate-Generation Smoke Runtime Report",
        "",
        "This is a bounded engineering smoke. It is not standard success, rollout success, or paper-grade evidence.",
        "",
        f"- passed: `{result.get('passed')}`",
        f"- decision: `{report.get('decision')}`",
        f"- blocked reason: `{result.get('blocked_reason')}`",
        f"- selected candidate index: `{selection.get('selected_candidate_index')}`",
        f"- selected target index: `{selection.get('selected_target_index')}`",
        f"- model inference performed: `{(report.get('policy') or {}).get('model_inference_performed')}`",
        f"- rollout performed: `{(report.get('policy') or {}).get('rollouts_performed')}`",
        f"- paper claim made: `{(report.get('policy') or {}).get('paper_grade_claims_made')}`",
        "",
        "## Next Step",
        "",
        str(report.get("recommended_next_step")),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def _policy_base(
    required_gate_state: dict[str, bool],
    forbidden_gates_set: list[str],
    *,
    heavy_attempted: bool = False,
    inference_performed: bool = False,
    gpu_job: bool = False,
) -> dict[str, Any]:
    return {
        "real_candidate_generation_smoke": True,
        "bounded_single_sample_only": True,
        "downloads_performed": False,
        "installs_performed": False,
        "heavy_model_imports_performed": heavy_attempted,
        "model_load_performed": inference_performed,
        "single_sample_model_inference_performed": inference_performed,
        "model_inference_performed": inference_performed,
        "candidate_generation_performed": inference_performed,
        "training_performed": False,
        "rollouts_performed": False,
        "simulator_environment_created": False,
        "gpu_jobs_performed": gpu_job,
        "openvla_oft_executed": False,
        "tokens_read_or_written": False,
        "external_verifier_used": False,
        "privileged_inference_used": False,
        "paper_grade_claims_made": False,
        "required_gates": required_gate_state,
        "forbidden_gates_set": forbidden_gates_set,
        "max_runtime_sec": MAX_RUNTIME_SECONDS,
        "max_vram_mb": MAX_VRAM_MB,
    }


def _readiness_inputs(
    plan_report: Path,
    runtime_deps_report: Path,
    load_only_report: Path,
    single_sample_report: Path,
) -> dict[str, Any]:
    plan = _read_json_if_exists(plan_report)
    runtime = _read_json_if_exists(runtime_deps_report)
    load = _read_json_if_exists(load_only_report)
    single = _read_json_if_exists(single_sample_report)
    return {
        "plan_present": bool(plan),
        "plan_passed": bool(plan.get("real_candidate_generation_smoke_plan_passed")),
        "ready_for_implementation": bool(plan.get("ready_for_real_candidate_generation_smoke_implementation")),
        "ready_for_execution_in_plan": bool(plan.get("ready_for_real_candidate_generation_smoke_execution")),
        "runtime_ready": bool(((runtime.get("runtime_dependencies") or {}).get("ready_for_load_only_runtime"))),
        "load_only_passed": bool(((load.get("result") or {}).get("passed"))),
        "single_sample_passed": bool(((single.get("result") or {}).get("passed"))),
    }


def _run_candidate_generation(
    args: argparse.Namespace,
    smolvla_ckpt: Path,
    checkpoint_root: Path,
    hf_home: Path,
    external_dependency: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    from tca_map.smolvla.single_sample_interface_smoke import _build_synthetic_batch, _load_policy

    started = time.monotonic()
    rss_before = _rss_mb()
    gpu_before = _nvidia_smi()
    policy, config = _load_policy(smolvla_ckpt, hf_home, external_dependency, args.device)
    tokenizer_root = Path(external_dependency["root"])
    batch, adapter_metadata = _build_synthetic_batch(config, tokenizer_root, args.task, args.device)
    noise = torch.zeros((1, config.chunk_size, config.max_action_dim), dtype=torch.float32, device=args.device)

    infer_started = time.monotonic()
    with torch.inference_mode():
        seed_action = policy.select_action(batch, noise=noise)
    infer_elapsed = time.monotonic() - infer_started

    action_heatmap, masked_heatmap, negative_heatmap, target_heatmap = build_candidate_heatmaps(
        seed_action,
        candidate_count=args.candidate_count,
        heatmap_grid=args.heatmap_grid,
    )
    selection_started = time.perf_counter()
    selection = distributional_tca_select_inference(
        action_heatmap=action_heatmap,
        target_heatmap=target_heatmap,
        masked_action_heatmap=masked_heatmap,
        negative_action_heatmaps=[negative_heatmap],
        K=args.candidate_count,
        temperature=args.temperature,
        metadata=None,
        external_verifier=None,
    )
    selection_elapsed_ms = (time.perf_counter() - selection_started) * 1000.0

    seed_values = _flatten_action(seed_action)
    selected = selection.get("selected") or {}
    selected_values = _flatten_action(selected.get("action", []))
    action_l1_to_seed = None
    if seed_values and selected_values and len(seed_values) == len(selected_values):
        action_l1_to_seed = sum(abs(a - b) for a, b in zip(seed_values, selected_values)) / len(seed_values)

    cuda_max_allocated_mb = None
    if torch.cuda.is_available():
        cuda_max_allocated_mb = round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)

    del policy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    runtime = {
        "elapsed_sec": round(time.monotonic() - started, 3),
        "single_sample_inference_elapsed_sec": round(infer_elapsed, 3),
        "selection_latency_ms": round(selection_elapsed_ms, 6),
        "rss_before_mb": rss_before,
        "rss_after_mb": _rss_mb(),
        "gpu_before": gpu_before,
        "gpu_after": _nvidia_smi(),
        "cuda_max_allocated_mb": cuda_max_allocated_mb,
        "device": args.device,
        "load_vlm_weights": bool(getattr(config, "load_vlm_weights", False)),
        "task": args.task,
        "batch_keys": sorted(batch.keys()),
        "adapter_metadata": adapter_metadata,
    }
    generation = {
        "seed_action_shape": list(seed_action.detach().cpu().shape),
        "seed_action_preview": [round(float(value), 6) for value in seed_values[: min(6, len(seed_values))]],
        "candidate_count": len(action_heatmap["candidates"]),
        "heatmap_grid": action_heatmap["grid_size"],
        "action_dim": action_heatmap["action_dim"],
        "low_resolution": True,
        "coarse_to_fine_ready": True,
        "action_heatmap": action_heatmap,
        "masked_action_heatmap": masked_heatmap,
        "negative_action_heatmap": negative_heatmap,
        "target_heatmap": target_heatmap,
        "selection": selection,
        "selected_action_l1_to_seed": None if action_l1_to_seed is None else round(action_l1_to_seed, 6),
        "wrong_target_proxy": bool(selected and selected.get("target_index") != target_heatmap["top_index"]),
    }
    return runtime, generation


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    smolvla_ckpt = Path(os.environ.get("SMOLVLA_CKPT") or args.smolvla_ckpt)
    checkpoint_root = Path(os.environ.get("CHECKPOINT_ROOT") or args.checkpoint_root)
    hf_home = Path(os.environ.get("HF_HOME") or args.hf_home)
    plan_report = Path(args.plan_report)
    runtime_deps_report = Path(args.runtime_deps_report)
    load_only_report = Path(args.load_only_report)
    single_sample_report = Path(args.single_sample_report)

    required_gate_state = {name: _env_flag(name) for name in REQUIRED_GATES}
    missing_required = [name for name, value in required_gate_state.items() if not value]
    forbidden_gates_set = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    readiness = _readiness_inputs(plan_report, runtime_deps_report, load_only_report, single_sample_report)
    config_files = _find_files(smolvla_ckpt, ["config.json"])
    tokenizer_files = _find_files(
        smolvla_ckpt,
        [
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "vocab.json",
            "merges.txt",
            "tokenizer.model",
            "sentencepiece.bpe.model",
        ],
    )
    weight_files = _find_files(
        smolvla_ckpt,
        ["model.safetensors", "pytorch_model.bin", "model-00001-of-00001.safetensors"],
        ["*.safetensors", "*.bin"],
    )
    dependency_name = _read_tokenizer_dependency(smolvla_ckpt)
    external_dependency = _external_tokenizer_files(dependency_name, [hf_home, checkpoint_root])
    deps = _runtime_dependencies()
    gpu = _nvidia_smi()
    deps_ready = all(deps.values())
    files_ready = bool(config_files and weight_files and (tokenizer_files or external_dependency["found"]))

    report: dict[str, Any] = {
        "schema_version": "tca-map-real-candidate-generation-smoke-v0",
        "real_candidate_generation_smoke_passed": False,
        "decision": "stop",
        "policy": _policy_base(required_gate_state, forbidden_gates_set),
        "risk_assessment": {
            "task": "bounded real candidate-generation smoke",
            "command": "scripts/133_bounded_real_candidate_generation_smoke.ps1",
            "source_path": str(smolvla_ckpt),
            "expected_runtime_minutes": "<=10",
            "expected_disk_gb": 0,
            "expected_ram_gb": "bounded by one CPU model-load smoke",
            "expected_vram_mb": "<=14336; CPU default",
            "token_login_license_payment_required": False,
            "target_output_paths": [args.report_path, args.markdown_report_path],
            "stop_condition": "missing gates, missing local files, runtime deps unavailable, >10 minutes, >14GB VRAM, any forbidden gate",
            "fallback_plan": "keep candidate-generation at synthetic contract level and continue offline proxy evidence only",
            "decision": "proceed" if not missing_required and not forbidden_gates_set else "stop",
        },
        "paths": {
            "smolvla_ckpt": str(smolvla_ckpt),
            "checkpoint_root": str(checkpoint_root),
            "hf_home": str(hf_home),
            "plan_report": str(plan_report),
            "runtime_deps_report": str(runtime_deps_report),
            "load_only_report": str(load_only_report),
            "single_sample_report": str(single_sample_report),
        },
        "readiness_input_summary": readiness,
        "files": {
            "config_found": config_files,
            "tokenizer_found": tokenizer_files,
            "weights_found": weight_files,
            "external_tokenizer_dependency": external_dependency,
            "files_ready": files_ready,
        },
        "runtime_dependencies": deps,
        "gpu": gpu,
        "runtime": None,
        "generation": None,
        "selection": None,
        "result": {"passed": False, "blocked": True, "blocked_reason": None, "elapsed_sec": None},
        "ready_for_rollout": False,
        "ready_for_benchmark_claim": False,
        "ready_for_paper_claim": False,
        "recommended_next_step": None,
    }

    def block(reason: str, code: int) -> tuple[dict[str, Any], int]:
        report["result"] = {
            "passed": False,
            "blocked": True,
            "blocked_reason": reason,
            "elapsed_sec": round(time.monotonic() - started, 3),
        }
        report["recommended_next_step"] = reason
        return report, code

    if args.candidate_count < 2 or args.candidate_count > MAX_CANDIDATES:
        return block(f"candidate_count must be between 2 and {MAX_CANDIDATES}", 11)
    if args.heatmap_grid < 2 or args.heatmap_grid > MAX_HEATMAP_GRID:
        return block(f"heatmap_grid must be between 2 and {MAX_HEATMAP_GRID}", 12)
    if missing_required:
        return block(
            "Missing required task-local gate(s): " + ", ".join(f"{name}=1" for name in missing_required),
            2,
        )
    if forbidden_gates_set:
        return block("Forbidden gate(s) set: " + ", ".join(forbidden_gates_set), 3)
    if not readiness["plan_passed"] or not readiness["ready_for_implementation"]:
        return block("Real candidate-generation smoke plan is missing or not green for implementation.", 4)
    if readiness["ready_for_execution_in_plan"]:
        return block("Planning report unexpectedly marks execution ready; rerun planner to preserve gate separation.", 5)
    if not readiness["runtime_ready"] or not readiness["load_only_passed"] or not readiness["single_sample_passed"]:
        return block("Prior runtime/load-only/single-sample prerequisites are missing or not passed.", 6)
    if not files_ready:
        return block("SmolVLA checkpoint/tokenizer/weights readiness is incomplete.", 7)
    if not deps_ready:
        missing = [name for name, present in deps.items() if not present]
        return block("Missing runtime dependencies: " + ", ".join(missing), 8)

    try:
        runtime, generation = _run_candidate_generation(
            args=args,
            smolvla_ckpt=smolvla_ckpt,
            checkpoint_root=checkpoint_root,
            hf_home=hf_home,
            external_dependency=external_dependency,
        )
        elapsed = round(time.monotonic() - started, 3)
        cuda_mb = runtime.get("cuda_max_allocated_mb") or 0
        if elapsed > MAX_RUNTIME_SECONDS:
            report["runtime"] = runtime
            report["generation"] = generation
            report["selection"] = generation.get("selection")
            return block("Bounded real candidate-generation smoke exceeded the 10 minute runtime budget.", 9)
        if cuda_mb > MAX_VRAM_MB:
            report["runtime"] = runtime
            report["generation"] = generation
            report["selection"] = generation.get("selection")
            return block("Bounded real candidate-generation smoke exceeded the 14GB VRAM budget.", 10)

        selected = (generation.get("selection") or {}).get("selected") or {}
        report["policy"] = _policy_base(
            required_gate_state,
            forbidden_gates_set,
            heavy_attempted=True,
            inference_performed=True,
            gpu_job=args.device == "cuda",
        )
        report["runtime"] = runtime
        report["generation"] = generation
        report["selection"] = {
            "method": (generation.get("selection") or {}).get("method"),
            "selected_candidate_index": selected.get("index"),
            "selected_target_index": selected.get("target_index"),
            "selected_action_l1_to_seed": generation.get("selected_action_l1_to_seed"),
            "wrong_target_proxy": generation.get("wrong_target_proxy"),
        }
        report["real_candidate_generation_smoke_passed"] = True
        report["decision"] = "bounded_real_candidate_generation_smoke_passed"
        report["result"] = {"passed": True, "blocked": False, "blocked_reason": None, "elapsed_sec": elapsed}
        report["recommended_next_step"] = (
            "Use this as engineering evidence only; next safe task is a report-only synthesis or a bounded offline candidate-generation comparison plan."
        )
        return report, 0
    except Exception as exc:  # noqa: BLE001 - report exact smoke failure.
        report["policy"] = _policy_base(required_gate_state, forbidden_gates_set, heavy_attempted=True)
        report["runtime_error"] = {"type": type(exc).__name__, "message": str(exc)}
        return block(f"Bounded real candidate-generation smoke failed: {type(exc).__name__}: {exc}", 13)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--plan-report", default="reports/real_candidate_generation_smoke_plan_report.json")
    parser.add_argument("--runtime-deps-report", default="reports/smolvla_runtime_deps_report.json")
    parser.add_argument("--load-only-report", default="reports/smolvla_load_only_smoke_report.json")
    parser.add_argument("--single-sample-report", default="reports/smolvla_single_sample_interface_report.json")
    parser.add_argument("--report-path", default="reports/real_candidate_generation_smoke_report.json")
    parser.add_argument("--markdown-report-path", default="reports/real_candidate_generation_smoke_report.md")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--task", default="pick up the object")
    parser.add_argument("--candidate-count", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--heatmap-grid", type=int, default=MAX_HEATMAP_GRID)
    parser.add_argument("--temperature", type=float, default=0.5)
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_path)
    markdown_path = Path(args.markdown_report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, markdown_path)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
