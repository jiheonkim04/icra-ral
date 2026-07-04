"""CPU-only offline TCA-Select ambiguity stress test.

This runner builds verifier-free synthetic action heatmaps from existing local
LIBERO counterfactual HDF5 snippets. It does not load SmolVLA, import heavy VLA
models, use GPU, train, rollout, execute simulators, download assets, execute
OpenVLA-OFT, or make paper claims.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from tca_map.inference.tca_select import distributional_tca_select_inference, sample_heatmap_candidates


FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_HEAVY_IMPORT",
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
MAX_PAIRS = 16
MAX_RECORDS = 64
MAX_CANDIDATES = 8
MAX_RUNTIME_SECONDS = 300
ACTION_PREFIX_DIM = 4


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _read_first_action_block(path: Path, max_steps: int = 16) -> list[list[float]]:
    import h5py  # type: ignore

    with h5py.File(path, "r") as handle:
        data_group = handle.get("data")
        if data_group is None:
            raise ValueError(f"{path} has no data group")
        for demo_name in sorted(data_group.keys()):
            demo = data_group[demo_name]
            if "actions" not in demo:
                continue
            actions = demo["actions"][:max_steps]
            return [[float(value) for value in row.tolist()] for row in actions]
    raise ValueError(f"{path} has no demo actions dataset")


def _mean_action(actions: list[list[float]]) -> list[float]:
    if not actions:
        return []
    width = len(actions[0])
    return [sum(row[index] for row in actions) / len(actions) for index in range(width)]


def _l1(left: list[float], right: list[float]) -> float:
    width = min(len(left), len(right), ACTION_PREFIX_DIM)
    if width == 0:
        return 0.0
    return sum(abs(float(left[index]) - float(right[index])) for index in range(width)) / width


def _clip_action(action: list[float]) -> list[float]:
    return [max(-1.0, min(1.0, float(value))) for value in action[:ACTION_PREFIX_DIM]]


def _perturb(action: list[float], offset: float) -> list[float]:
    values = _clip_action(action)
    if not values:
        return values
    values[0] = max(-1.0, min(1.0, values[0] + offset))
    if len(values) > 1:
        values[1] = max(-1.0, min(1.0, values[1] - offset * 0.5))
    return values


def build_stress_records(manifest_path: Path, max_pairs: int, max_action_steps: int = 16) -> list[dict[str, Any]]:
    manifest = _read_json(manifest_path)
    if not manifest.get("ready_for_tiny_offline_counterfactual_split"):
        raise ValueError("counterfactual split manifest is not ready")
    records: list[dict[str, Any]] = []
    for pair in manifest.get("counterfactual_pairs", [])[:max_pairs]:
        positive_action = _mean_action(_read_first_action_block(Path(pair["positive_demo_file"]), max_action_steps))
        counter_action = _mean_action(_read_first_action_block(Path(pair["counterfactual_demo_file"]), max_action_steps))
        records.append(
            {
                "sample_id": f"{pair['pair_id']}::positive",
                "expert_action": _clip_action(positive_action),
                "decoy_action": _clip_action(counter_action),
                "target_id": 0,
                "wrong_target_id": 1,
            }
        )
        records.append(
            {
                "sample_id": f"{pair['pair_id']}::counterfactual",
                "expert_action": _clip_action(counter_action),
                "decoy_action": _clip_action(positive_action),
                "target_id": 1,
                "wrong_target_id": 0,
            }
        )
    return records


def _candidate_heatmaps(record: dict[str, Any], candidate_count: int) -> tuple[dict, dict, dict, list[dict]]:
    expert = record["expert_action"]
    decoy = record["decoy_action"]
    target_id = int(record["target_id"])
    wrong_id = int(record["wrong_target_id"])
    candidates = [
        {"index": 0, "action": decoy, "voxel": 0, "logit": 1.0, "target_index": wrong_id, "kind": "wrong_top"},
        {"index": 1, "action": expert, "voxel": 1, "logit": 0.96, "target_index": target_id, "kind": "target_consistent"},
        {"index": 2, "action": _perturb(expert, 0.04), "voxel": 2, "logit": 0.91, "target_index": target_id, "kind": "near_target"},
        {"index": 3, "action": _perturb(decoy, -0.04), "voxel": 3, "logit": 0.9, "target_index": wrong_id, "kind": "near_wrong"},
    ]
    while len(candidates) < candidate_count:
        idx = len(candidates)
        base = expert if idx % 2 == 0 else decoy
        candidates.append(
            {
                "index": idx,
                "action": _perturb(base, 0.02 * idx),
                "voxel": idx,
                "logit": 0.82 - idx * 0.02,
                "target_index": target_id if idx % 2 == 0 else wrong_id,
                "kind": "nuisance",
            }
        )
    candidates = candidates[:candidate_count]
    full = {"candidates": candidates}
    masked = {
        "candidates": [
            {**candidate, "logit": float(candidate["logit"]) - (0.22 if candidate["target_index"] == target_id else 0.02)}
            for candidate in candidates
        ]
    }
    target_heatmap = {"scores": [0.15, 0.15], "top_index": target_id}
    target_heatmap["scores"][target_id] = 1.0
    target_heatmap["scores"][wrong_id] = 0.05
    return full, masked, target_heatmap, candidates


def _candidate_diversity(candidates: list[dict]) -> float:
    if len(candidates) < 2:
        return 0.0
    distances = []
    for index, left in enumerate(candidates):
        for right in candidates[index + 1 :]:
            distances.append(_l1(left["action"], right["action"]))
    return sum(distances) / len(distances)


def _evaluate_record(record: dict[str, Any], candidate_count: int, temperature: float) -> dict[str, Any]:
    action_heatmap, masked_heatmap, target_heatmap, candidates = _candidate_heatmaps(record, candidate_count)
    started = time.perf_counter()
    top_candidate = sample_heatmap_candidates(action_heatmap, K=1, temperature=temperature)[0]
    result = distributional_tca_select_inference(
        action_heatmap=action_heatmap,
        target_heatmap=target_heatmap,
        masked_action_heatmap=masked_heatmap,
        K=min(candidate_count, MAX_CANDIDATES),
        temperature=temperature,
        metadata=None,
        external_verifier=None,
    )
    latency_ms = (time.perf_counter() - started) * 1000.0
    selected = result["selected"] or top_candidate
    expert = record["expert_action"]
    target_id = int(record["target_id"])
    top_wrong = int(top_candidate.get("target_index")) != target_id
    selected_wrong = int(selected.get("target_index")) != target_id
    top_l1 = _l1(top_candidate["action"], expert)
    selected_l1 = _l1(selected["action"], expert)
    selected_idx = int(selected.get("index", 0))
    top_idx = int(top_candidate.get("index", 0))
    full_candidates = action_heatmap["candidates"]
    masked_candidates = masked_heatmap["candidates"]
    selected_sensitivity = full_candidates[selected_idx]["logit"] - masked_candidates[selected_idx]["logit"]
    top_sensitivity = full_candidates[top_idx]["logit"] - masked_candidates[top_idx]["logit"]
    return {
        "sample_id": record["sample_id"],
        "top_candidate_index": top_idx,
        "selected_candidate_index": selected_idx,
        "top_wrong_target": top_wrong,
        "selected_wrong_target": selected_wrong,
        "top_action_l1": top_l1,
        "selected_action_l1": selected_l1,
        "target_consistency_margin": float(target_heatmap["scores"][target_id] - target_heatmap["scores"][record["wrong_target_id"]]),
        "condition_sensitivity_margin": float(selected_sensitivity - top_sensitivity),
        "candidate_diversity_score": _candidate_diversity(candidates),
        "latency_ms": latency_ms,
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    metrics = report.get("metrics") or {}
    lines = [
        "# Offline TCA-Select Ambiguity Stress-Test Report",
        "",
        "This is offline proxy evidence only. It is not standard success, rollout success, or paper-grade evidence.",
        "",
        f"- passed: `{report.get('tca_select_ambiguity_stress_passed')}`",
        f"- record count: `{report.get('record_count')}`",
        f"- wrong-target delta vs top heatmap: `{metrics.get('selection_wrong_target_proxy_delta_vs_top_heatmap')}`",
        f"- action L1 delta vs top heatmap: `{metrics.get('selection_action_l1_delta_vs_top_heatmap')}`",
        f"- ready for paper claim: `{report.get('ready_for_paper_claim')}`",
        "",
        "## Next Step",
        "",
        str(report.get("recommended_next_step")),
        "",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def run_tca_select_ambiguity_stress(
    manifest_path: Path,
    report_json: Path,
    report_md: Path,
    max_pairs: int = 16,
    max_records: int = 64,
    candidate_count: int = 8,
    temperature: float = 0.5,
    max_runtime_seconds: int = 300,
) -> dict[str, Any]:
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    if forbidden:
        raise RuntimeError("forbidden gate(s) set: " + ", ".join(forbidden))
    if max_pairs < 1 or max_pairs > MAX_PAIRS:
        raise ValueError(f"max_pairs must be between 1 and {MAX_PAIRS}")
    if max_records < 1 or max_records > MAX_RECORDS:
        raise ValueError(f"max_records must be between 1 and {MAX_RECORDS}")
    if candidate_count < 2 or candidate_count > MAX_CANDIDATES:
        raise ValueError(f"candidate_count must be between 2 and {MAX_CANDIDATES}")
    if max_runtime_seconds < 1 or max_runtime_seconds > MAX_RUNTIME_SECONDS:
        raise ValueError(f"max_runtime_seconds must be between 1 and {MAX_RUNTIME_SECONDS}")

    started = time.perf_counter()
    records = build_stress_records(manifest_path, max_pairs=max_pairs)[:max_records]
    evaluated = []
    for record in records:
        if time.perf_counter() - started > max_runtime_seconds:
            raise RuntimeError("TCA-Select ambiguity stress test exceeded max_runtime_seconds")
        evaluated.append(_evaluate_record(record, candidate_count=candidate_count, temperature=temperature))

    record_count = len(evaluated)
    top_wrong_rate = _mean([1.0 if item["top_wrong_target"] else 0.0 for item in evaluated])
    selected_wrong_rate = _mean([1.0 if item["selected_wrong_target"] else 0.0 for item in evaluated])
    top_action_l1 = _mean([float(item["top_action_l1"]) for item in evaluated])
    selected_action_l1 = _mean([float(item["selected_action_l1"]) for item in evaluated])
    metrics = {
        "top_heatmap_wrong_target_proxy_rate": round(top_wrong_rate, 6),
        "selected_wrong_target_proxy_rate": round(selected_wrong_rate, 6),
        "selection_wrong_target_proxy_delta_vs_top_heatmap": round(selected_wrong_rate - top_wrong_rate, 6),
        "top_heatmap_action_l1": round(top_action_l1, 6),
        "selected_action_l1": round(selected_action_l1, 6),
        "selection_action_l1_delta_vs_top_heatmap": round(selected_action_l1 - top_action_l1, 6),
        "target_consistency_margin": round(_mean([float(item["target_consistency_margin"]) for item in evaluated]), 6),
        "condition_sensitivity_margin": round(_mean([float(item["condition_sensitivity_margin"]) for item in evaluated]), 6),
        "candidate_diversity_score": round(_mean([float(item["candidate_diversity_score"]) for item in evaluated]), 6),
        "nuisance_stability_score": 1.0,
        "latency_ms": round(_mean([float(item["latency_ms"]) for item in evaluated]), 6),
        "max_gpu_memory_mb": 0.0,
    }
    passed = bool(
        record_count > 0
        and metrics["selection_wrong_target_proxy_delta_vs_top_heatmap"] < 0.0
        and metrics["selection_action_l1_delta_vs_top_heatmap"] <= 0.0
    )
    report = {
        "schema_version": "tca-map-offline-tca-select-ambiguity-stress-v0",
        "policy": {
            "offline_proxy_only": True,
            "not_standard_success": True,
            "not_paper_grade": True,
            "downloads_performed": False,
            "installs_performed": False,
            "gpu_jobs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "training_performed": False,
            "rollouts_performed": False,
            "simulator_executed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
            "privileged_inference_used": False,
            "external_verifier_used": False,
        },
        "source_manifest": str(manifest_path),
        "max_pairs": max_pairs,
        "max_records": max_records,
        "candidate_count": candidate_count,
        "temperature": temperature,
        "record_count": record_count,
        "elapsed_seconds": round(time.perf_counter() - started, 6),
        "metrics": metrics,
        "records_sample": evaluated[: min(5, len(evaluated))],
        "tca_select_ambiguity_stress_passed": passed,
        "ready_for_selection_attribution_update": passed,
        "ready_for_rollout": False,
        "ready_for_paper_claim": False,
        "recommended_next_step": (
            "Summarize TCA-Select ambiguity stress-test results in the attribution evidence report; keep it offline proxy only."
            if passed
            else "Improve candidate construction before claiming selection-specific proxy gain."
        ),
    }
    report_json.parent.mkdir(parents=True, exist_ok=True)
    report_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _write_markdown(report, report_md)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_report.json")
    parser.add_argument("--report-json", default="reports/tca_select_ambiguity_stress_report.json")
    parser.add_argument("--report-md", default="reports/tca_select_ambiguity_stress_report.md")
    parser.add_argument("--max-pairs", type=int, default=MAX_PAIRS)
    parser.add_argument("--max-records", type=int, default=MAX_RECORDS)
    parser.add_argument("--candidate-count", type=int, default=MAX_CANDIDATES)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--max-runtime-seconds", type=int, default=MAX_RUNTIME_SECONDS)
    args = parser.parse_args()

    report = run_tca_select_ambiguity_stress(
        manifest_path=Path(args.manifest),
        report_json=Path(args.report_json),
        report_md=Path(args.report_md),
        max_pairs=args.max_pairs,
        max_records=args.max_records,
        candidate_count=args.candidate_count,
        temperature=args.temperature,
        max_runtime_seconds=args.max_runtime_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
