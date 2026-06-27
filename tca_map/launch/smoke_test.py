"""Dummy smoke train/eval entrypoint.

This code is intentionally CPU-only and dependency-light.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from tca_map.datasets import make_counterfactual_pairs, make_dummy_samples
from tca_map.eval import compute_offline_metrics
from tca_map.heads import ActionMapHead, TCAMapHead
from tca_map.models import DummyAdapter

REPO_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = REPO_ROOT / "reports"
SMOKE_REPORT_PATH = REPORTS_DIR / "smoke_report.json"


def _relative_report_path(path: Path) -> str:
    return path.relative_to(REPO_ROOT).as_posix()


def _load_json_if_exists(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _update_smoke_report(mode: str, metrics_path: Path, metrics: dict) -> dict:
    train_path = REPORTS_DIR / "dummy_train_metrics.json"
    eval_path = REPORTS_DIR / "dummy_eval_metrics.json"
    existing = _load_json_if_exists(SMOKE_REPORT_PATH) or {}

    train_metrics = metrics if mode == "train" else _load_json_if_exists(train_path)
    eval_metrics = metrics if mode == "eval" else _load_json_if_exists(eval_path)
    train_passed = bool(train_metrics and train_metrics.get("mode") == "train")
    eval_passed = bool(eval_metrics and eval_metrics.get("mode") == "eval")

    report = {
        **existing,
        "train_metrics_path": _relative_report_path(train_path) if train_path.exists() or mode == "train" else None,
        "eval_metrics_path": _relative_report_path(eval_path) if eval_path.exists() or mode == "eval" else None,
        "safe_to_run_pilot_gpu": False,
        "train_smoke_passed": train_passed,
        "eval_smoke_passed": eval_passed,
        "downloads_performed": False,
        "gpu_training_performed": False,
        "real_rollouts_performed": False,
        "recommended_next_step": (
            "Configure local real assets and rerun preflight before any pilot GPU work."
            if train_passed and eval_passed
            else "Run both dummy train and eval smoke before considering real asset checks."
        ),
        "last_updated_by": f"dummy_{mode}_smoke",
        "last_metrics_path": _relative_report_path(metrics_path),
    }
    SMOKE_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def run_smoke(mode: str) -> dict:
    REPORTS_DIR.mkdir(exist_ok=True)
    samples = make_dummy_samples(count=4)
    pairs = make_counterfactual_pairs(samples)
    adapter = DummyAdapter()
    action_head = ActionMapHead(grid_size=8)
    tca_head = TCAMapHead(grid_size=8)

    records = []
    losses = []
    for sample in samples:
        start = time.perf_counter()
        train_result = adapter.train_step(sample, loss_config={})
        hidden = adapter.get_hidden_tokens(sample["observation"], sample["instruction"])
        tca_pred = tca_head.predict(hidden, sample["observation"]["candidate_objects"])
        latency_ms = (time.perf_counter() - start) * 1000.0
        expert_voxel = action_head.action_to_voxel(sample["expert_action"])
        pred_voxel = tuple(tca_pred["action_heatmap"]["top_voxel"])
        records.append(
            {
                "sample_id": sample["sample_id"],
                "pred_action": tca_pred["action"],
                "expert_action": sample["expert_action"],
                "pred_voxel": pred_voxel,
                "expert_voxel": expert_voxel,
                "pred_target": tca_pred["target"]["top_index"],
                "target_id": sample["target"]["object_id"],
                "latency_ms": latency_ms,
            }
        )
        losses.append(train_result["loss"])

    metrics = compute_offline_metrics(records)
    metrics.update(
        {
            "mode": mode,
            "dummy_loss": round(sum(losses) / len(losses), 6),
            "dummy_sample_count": len(samples),
            "counterfactual_pair_count": len(pairs),
            "downloads_performed": False,
            "gpu_training_performed": False,
            "real_rollouts_performed": False,
        }
    )
    output_path = REPORTS_DIR / f"dummy_{mode}_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    smoke_report = _update_smoke_report(mode, output_path, metrics)
    print(json.dumps({"metrics": metrics, "smoke_report": smoke_report}, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    args = parser.parse_args()
    run_smoke(args.mode)


if __name__ == "__main__":
    main()
