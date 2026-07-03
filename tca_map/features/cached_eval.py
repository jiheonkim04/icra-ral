"""Eval-only smoke path for cached feature records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tca_map.eval import compute_offline_metrics
from tca_map.features.cache import validate_feature_cache, write_dummy_feature_cache
from tca_map.heads import ActionMapHead, TCAMapHead


def _load_records(cache_dir: Path) -> list[dict]:
    features_path = cache_dir / "features.jsonl"
    return [json.loads(line) for line in features_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _candidate_objects(record: dict) -> list[str]:
    objects = record.get("candidate_objects") or []
    if objects:
        return objects
    fallback = []
    for key in ["target", "distractor"]:
        value = record.get(key)
        if isinstance(value, dict) and value.get("name"):
            fallback.append(value["name"])
    return fallback or ["object"]


def evaluate_feature_cache(cache_dir: Path, report_path: Path, prepare_dummy_cache: bool = False) -> dict:
    if prepare_dummy_cache and not (cache_dir / "manifest.json").exists():
        write_dummy_feature_cache(cache_dir, max_samples=4, overwrite=True)

    validation = validate_feature_cache(cache_dir)
    if not validation["valid"]:
        report = {
            "policy": _policy(),
            "cache_dir": str(cache_dir),
            "cache_valid": False,
            "validation_errors": validation["errors"],
            "recommended_next_step": "Create or fix the dummy feature cache before eval smoke.",
        }
        _write_report(report_path, report)
        return report

    records = _load_records(cache_dir)
    action_head = ActionMapHead(grid_size=8)
    tca_head = TCAMapHead(grid_size=8)
    metric_records = []
    for record in records:
        prediction = tca_head.predict(record["hidden_tokens"], _candidate_objects(record))
        expert_action = record["expert_action"]
        metric_records.append(
            {
                "sample_id": record["sample_id"],
                "pred_action": prediction["action"],
                "expert_action": expert_action,
                "pred_voxel": tuple(prediction["action_heatmap"]["top_voxel"]),
                "expert_voxel": action_head.action_to_voxel(expert_action),
                "pred_target": prediction["target"]["top_index"],
                "target_id": record["target"]["object_id"],
                "latency_ms": 0.0,
            }
        )

    metrics = compute_offline_metrics(metric_records)
    metrics.update(
        {
            "mode": "feature_cache_eval_smoke",
            "cache_record_count": len(records),
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "training_performed": False,
            "rollouts_performed": False,
            "openvla_oft_executed": False,
        }
    )
    report = {
        "policy": _policy(),
        "cache_dir": str(cache_dir),
        "cache_valid": True,
        "validation_errors": [],
        "metrics": metrics,
        "recommended_next_step": "Use this eval-only smoke to guard cached-feature/head contracts. Run risk assessment before real SmolVLA feature extraction or training.",
    }
    _write_report(report_path, report)
    return report


def _policy() -> dict:
    return {
        "eval_only": True,
        "downloads_performed": False,
        "gpu_jobs_performed": False,
        "heavy_model_imports_performed": False,
        "model_load_performed": False,
        "model_inference_performed": False,
        "training_performed": False,
        "rollouts_performed": False,
        "openvla_oft_executed": False,
    }


def _write_report(report_path: Path, report: dict) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default="runs/feature_cache/dummy_contract")
    parser.add_argument("--report-path", default="reports/feature_cache_eval_report.json")
    parser.add_argument("--prepare-dummy-cache", action="store_true")
    args = parser.parse_args()

    report = evaluate_feature_cache(
        cache_dir=Path(args.cache_dir),
        report_path=Path(args.report_path),
        prepare_dummy_cache=args.prepare_dummy_cache,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
