"""Synthetic-tensor candidate-generation contract checker.

This checker validates the data contract that a future learned-policy candidate
generator must satisfy before any model inference is attempted. It uses synthetic
candidate heatmaps only and does not import heavy VLA models, load SmolVLA, run
model inference, train, rollout, use GPU jobs, execute simulators, download
assets, execute OpenVLA-OFT, or make paper claims.
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


FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
    "ALLOW_SINGLE_SAMPLE_INFERENCE",
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
FORBIDDEN_METADATA_KEYS = {
    "simulator_state",
    "privileged_state",
    "object_pose",
    "object_poses",
    "env_state",
    "success_label",
    "rollout_reward",
    "task_success",
}
MAX_CANDIDATES = 8
MAX_GRID = 8
ACTION_DIM = 4


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _synthetic_contract(candidate_count: int, heatmap_grid: int) -> tuple[dict, dict, dict, dict]:
    candidates = []
    for index in range(candidate_count):
        target_consistent = index % 2 == 0
        candidates.append(
            {
                "index": index,
                "voxel": index,
                "grid_index": [index % heatmap_grid, (index // heatmap_grid) % heatmap_grid, 0],
                "action": [
                    round(0.05 * index, 6),
                    round(-0.03 * index, 6),
                    round(0.01 * index, 6),
                    1.0 if target_consistent else -1.0,
                ],
                "logit": round(1.0 - 0.04 * index, 6),
                "target_index": 0 if target_consistent else 1,
                "source": "synthetic_contract",
            }
        )
    action_heatmap = {
        "grid_size": heatmap_grid,
        "action_dim": ACTION_DIM,
        "candidates": candidates,
        "low_resolution": True,
        "coarse_to_fine_ready": True,
    }
    masked_heatmap = {
        **action_heatmap,
        "candidates": [
            {**candidate, "logit": candidate["logit"] - (0.2 if candidate["target_index"] == 0 else 0.02)}
            for candidate in candidates
        ],
    }
    target_heatmap = {"scores": [1.0, 0.05], "top_index": 0}
    metadata = {
        "sample_id": "synthetic_candidate_contract",
        "target_text": "put the object in the target container",
        "uses_privileged_state": False,
        "external_verifier_used": False,
    }
    return action_heatmap, masked_heatmap, target_heatmap, metadata


def _validate_contract(action_heatmap: dict, masked_heatmap: dict, target_heatmap: dict, metadata: dict) -> list[str]:
    errors: list[str] = []
    candidates = action_heatmap.get("candidates") or []
    masked_candidates = masked_heatmap.get("candidates") or []
    if not candidates:
        errors.append("candidate list is empty")
    if len(candidates) > MAX_CANDIDATES:
        errors.append(f"candidate count exceeds {MAX_CANDIDATES}")
    if len(masked_candidates) != len(candidates):
        errors.append("masked heatmap candidate count does not match full heatmap")
    if int(action_heatmap.get("grid_size") or 0) > MAX_GRID:
        errors.append(f"heatmap grid exceeds {MAX_GRID}")
    if int(action_heatmap.get("action_dim") or 0) != ACTION_DIM:
        errors.append(f"action_dim must be {ACTION_DIM}")
    if target_heatmap.get("top_index") is None:
        errors.append("target heatmap has no top_index")
    forbidden_keys = sorted(key for key in metadata if key in FORBIDDEN_METADATA_KEYS)
    if forbidden_keys:
        errors.append("metadata contains forbidden privileged key(s): " + ", ".join(forbidden_keys))
    if metadata.get("uses_privileged_state"):
        errors.append("metadata marks privileged state usage")
    if metadata.get("external_verifier_used"):
        errors.append("metadata marks external verifier usage")
    for index, candidate in enumerate(candidates):
        action = candidate.get("action")
        if not isinstance(action, list) or len(action) != ACTION_DIM:
            errors.append(f"candidate {index} action must be a length-{ACTION_DIM} list")
        if candidate.get("voxel") is None:
            errors.append(f"candidate {index} missing voxel")
        if candidate.get("logit") is None:
            errors.append(f"candidate {index} missing logit")
        if candidate.get("target_index") is None:
            errors.append(f"candidate {index} missing target_index")
    return errors


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Candidate-Generation Contract Check Report",
        "",
        "This is a synthetic-tensor contract check. It is not model inference, rollout success, or paper-grade evidence.",
        "",
        f"- passed: `{report.get('candidate_generation_contract_check_passed')}`",
        f"- candidate count: `{report.get('candidate_count')}`",
        f"- heatmap grid: `{report.get('heatmap_grid')}`",
        f"- selected candidate index: `{(report.get('selection') or {}).get('selected_candidate_index')}`",
        f"- ready for real candidate-generation smoke plan: `{report.get('ready_for_real_candidate_generation_smoke_plan')}`",
        f"- ready for paper claim: `{report.get('ready_for_paper_claim')}`",
        "",
        "## Next Step",
        "",
        str(report.get("recommended_next_step")),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_contract_check(
    readiness_report: Path,
    report_json: Path,
    report_md: Path,
    candidate_count: int = 4,
    heatmap_grid: int = 8,
) -> dict[str, Any]:
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    if candidate_count < 2 or candidate_count > MAX_CANDIDATES:
        raise ValueError(f"candidate_count must be between 2 and {MAX_CANDIDATES}")
    if heatmap_grid < 2 or heatmap_grid > MAX_GRID:
        raise ValueError(f"heatmap_grid must be between 2 and {MAX_GRID}")

    readiness = _read_json_if_exists(readiness_report)
    action_heatmap, masked_heatmap, target_heatmap, metadata = _synthetic_contract(candidate_count, heatmap_grid)
    errors = _validate_contract(action_heatmap, masked_heatmap, target_heatmap, metadata)
    started = time.perf_counter()
    selection = distributional_tca_select_inference(
        action_heatmap=action_heatmap,
        target_heatmap=target_heatmap,
        masked_action_heatmap=masked_heatmap,
        K=min(candidate_count, MAX_CANDIDATES),
        temperature=0.5,
        metadata=None,
        external_verifier=None,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    selected = selection.get("selected") or {}
    passed = bool(
        not forbidden
        and not errors
        and selected
        and not metadata["uses_privileged_state"]
        and not metadata["external_verifier_used"]
    )
    report = {
        "schema_version": "tca-map-candidate-generation-contract-check-v0",
        "candidate_generation_contract_check_passed": passed,
        "decision": "candidate_generation_contract_ready" if passed else "stop",
        "policy": {
            "synthetic_tensor_only": True,
            "report_only": False,
            "downloads_performed": False,
            "installs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "training_performed": False,
            "rollouts_performed": False,
            "simulator_environment_created": False,
            "gpu_jobs_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
            "privileged_inference_used": False,
            "external_verifier_used": False,
            "forbidden_gates_set": forbidden,
        },
        "paths": {"readiness_report": str(readiness_report)},
        "readiness_input_summary": {
            "readiness_report_present": bool(readiness),
            "readiness_plan_passed": bool(readiness.get("candidate_generation_readiness_plan_passed")),
            "ready_for_contract_checker": bool(readiness.get("ready_for_candidate_generation_contract_checker")),
            "ready_for_real_candidate_generation_smoke_execution": bool(
                readiness.get("ready_for_real_candidate_generation_smoke_execution")
            ),
        },
        "candidate_count": candidate_count,
        "heatmap_grid": heatmap_grid,
        "contract_errors": errors,
        "selection": {
            "selected_candidate_index": selected.get("index"),
            "selected_target_index": selected.get("target_index"),
            "latency_ms": round(latency_ms, 6),
        },
        "max_gpu_memory_mb": 0.0,
        "ready_for_real_candidate_generation_smoke_plan": passed,
        "ready_for_real_candidate_generation_smoke_execution": False,
        "ready_for_rollout": False,
        "ready_for_paper_claim": False,
        "recommended_next_step": (
            "Plan a separately gated real candidate-generation smoke. It must require heavy-import/inference risk assessment and must not rollout."
            if passed
            else "Fix candidate-generation contract errors before any model-inference plan."
        ),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, report_md)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness-report", default="reports/candidate_generation_readiness_plan_report.json")
    parser.add_argument("--report-json", default="reports/candidate_generation_contract_check_report.json")
    parser.add_argument("--report-md", default="reports/candidate_generation_contract_check_report.md")
    parser.add_argument("--candidate-count", type=int, default=4)
    parser.add_argument("--heatmap-grid", type=int, default=8)
    args = parser.parse_args(argv)

    report = run_contract_check(
        readiness_report=Path(args.readiness_report),
        report_json=Path(args.report_json),
        report_md=Path(args.report_md),
        candidate_count=args.candidate_count,
        heatmap_grid=args.heatmap_grid,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["candidate_generation_contract_check_passed"] else 2


if __name__ == "__main__":
    sys.exit(main())
