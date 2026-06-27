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
    print(json.dumps(metrics, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "eval"], required=True)
    args = parser.parse_args()
    run_smoke(args.mode)


if __name__ == "__main__":
    main()
