"""Offline proxy metrics for dummy and tiny pilots."""

from __future__ import annotations

import math


def _mse(pred: list[float], target: list[float]) -> float:
    return sum((p - t) ** 2 for p, t in zip(pred, target)) / max(1, len(target))


def _l1(pred: list[float], target: list[float]) -> float:
    return sum(abs(p - t) for p, t in zip(pred, target)) / max(1, len(target))


def _voxel_distance(a: tuple[int, int, int], b: tuple[int, int, int]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def compute_offline_metrics(records: list[dict]) -> dict:
    if not records:
        return {
            "offline_standard_proxy": 0.0,
            "standard_proxy_score": 0.0,
            "action_l1": 0.0,
            "action_mse": 0.0,
            "action_voxel_hit_rate": 0.0,
            "distance_to_expert_voxel": 0.0,
            "target_top1_accuracy": 0.0,
            "target_topk_accuracy": 0.0,
            "wrong_target_proxy_rate": 0.0,
            "counterfactual_separation_margin": 0.0,
            "nuisance_stability_score": 1.0,
            "latency_ms": 0.0,
            "max_gpu_memory_mb": 0.0,
        }

    action_l1 = sum(_l1(r["pred_action"], r["expert_action"]) for r in records) / len(records)
    action_mse = sum(_mse(r["pred_action"], r["expert_action"]) for r in records) / len(records)
    voxel_hits = sum(1 for r in records if r["pred_voxel"] == r["expert_voxel"])
    voxel_dist = sum(_voxel_distance(r["pred_voxel"], r["expert_voxel"]) for r in records) / len(records)
    target_hits = sum(1 for r in records if r["pred_target"] == r["target_id"])
    wrong_target = len(records) - target_hits
    offline_proxy = max(0.0, 1.0 - action_l1) * (target_hits / len(records))

    return {
        "offline_standard_proxy": round(offline_proxy, 6),
        "standard_proxy_score": round(offline_proxy, 6),
        "action_l1": round(action_l1, 6),
        "action_mse": round(action_mse, 6),
        "action_voxel_hit_rate": round(voxel_hits / len(records), 6),
        "distance_to_expert_voxel": round(voxel_dist, 6),
        "target_top1_accuracy": round(target_hits / len(records), 6),
        "target_topk_accuracy": round(target_hits / len(records), 6),
        "wrong_target_proxy_rate": round(wrong_target / len(records), 6),
        "counterfactual_separation_margin": 0.0,
        "nuisance_stability_score": 1.0,
        "latency_ms": round(sum(r.get("latency_ms", 0.0) for r in records) / len(records), 6),
        "max_gpu_memory_mb": 0.0,
    }
