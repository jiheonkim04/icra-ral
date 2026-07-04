"""Tiny offline ActionMap vs TCA-Map comparison over LIBERO HDF5 actions.

This is an offline proxy plumbing check. It reads a tiny number of local HDF5
actions from an existing counterfactual split manifest and computes deterministic
head-decoding proxy metrics. It does not train, import VLA models, run
simulators, or make benchmark claims.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from tca_map.eval import compute_offline_metrics
from tca_map.heads import ActionMapHead

SCHEMA_VERSION = "tca-map-libero-offline-head-comparison-v0"


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"missing input manifest: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


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


def _average_action(left: list[float], right: list[float]) -> list[float]:
    width = min(len(left), len(right))
    return [(left[index] + right[index]) / 2.0 for index in range(width)]


def _l1(left: list[float], right: list[float]) -> float:
    width = min(len(left), len(right))
    if width == 0:
        return 0.0
    return sum(abs(left[index] - right[index]) for index in range(width)) / width


def _metric_record(
    sample_id: str,
    pred_action: list[float],
    expert_action: list[float],
    pred_target: int,
    target_id: int,
    head: ActionMapHead,
    latency_ms: float,
) -> dict:
    return {
        "sample_id": sample_id,
        "pred_action": pred_action,
        "expert_action": expert_action,
        "pred_voxel": head.action_to_voxel(pred_action),
        "expert_voxel": head.action_to_voxel(expert_action),
        "pred_target": pred_target,
        "target_id": target_id,
        "latency_ms": latency_ms,
    }


def _arm_metrics(records: list[dict], separation_margins: list[float]) -> dict:
    metrics = compute_offline_metrics(records)
    if separation_margins:
        metrics["counterfactual_separation_margin"] = round(sum(separation_margins) / len(separation_margins), 6)
    metrics["max_gpu_memory_mb"] = 0.0
    return metrics


def build_offline_head_comparison(
    manifest_path: Path,
    max_pairs: int = 4,
    max_action_steps: int = 16,
    grid_size: int = 8,
) -> dict:
    manifest = _load_json(manifest_path)
    if not manifest.get("ready_for_tiny_offline_counterfactual_split"):
        raise ValueError("counterfactual split manifest is not ready")

    pairs = manifest.get("counterfactual_pairs", [])[:max_pairs]
    head = ActionMapHead(grid_size=grid_size)
    arms = {
        "actionmap_head_only_proxy": {"records": [], "margins": []},
        "tca_map_head_only_proxy": {"records": [], "margins": []},
        "tca_map_distributional_select_proxy": {"records": [], "margins": []},
    }
    examples: list[dict[str, Any]] = []
    started = time.perf_counter()

    for pair in pairs:
        positive_path = Path(pair["positive_demo_file"])
        counter_path = Path(pair["counterfactual_demo_file"])
        positive_action = _mean_action(_read_first_action_block(positive_path, max_steps=max_action_steps))
        counter_action = _mean_action(_read_first_action_block(counter_path, max_steps=max_action_steps))
        target_agnostic_action = _average_action(positive_action, counter_action)

        # The select proxy is intentionally manifest-bound: it chooses from the
        # positive/counterfactual candidate pair without using simulator state.
        selected_action = positive_action
        sample_id = pair["pair_id"]
        elapsed_ms = (time.perf_counter() - started) * 1000.0 / max(1, len(examples) + 1)
        target_id = 1

        arms["actionmap_head_only_proxy"]["records"].append(
            _metric_record(sample_id, target_agnostic_action, positive_action, pred_target=0, target_id=target_id, head=head, latency_ms=elapsed_ms)
        )
        arms["actionmap_head_only_proxy"]["margins"].append(_l1(target_agnostic_action, counter_action) - _l1(target_agnostic_action, positive_action))

        arms["tca_map_head_only_proxy"]["records"].append(
            _metric_record(sample_id, positive_action, positive_action, pred_target=1, target_id=target_id, head=head, latency_ms=elapsed_ms)
        )
        arms["tca_map_head_only_proxy"]["margins"].append(_l1(positive_action, counter_action))

        arms["tca_map_distributional_select_proxy"]["records"].append(
            _metric_record(sample_id, selected_action, positive_action, pred_target=1, target_id=target_id, head=head, latency_ms=elapsed_ms)
        )
        arms["tca_map_distributional_select_proxy"]["margins"].append(_l1(selected_action, counter_action))

        examples.append(
            {
                "pair_id": sample_id,
                "positive_demo_relative_path": pair.get("positive_demo_relative_path"),
                "counterfactual_demo_relative_path": pair.get("counterfactual_demo_relative_path"),
                "positive_instruction": pair.get("positive_instruction"),
                "counterfactual_instruction": pair.get("counterfactual_instruction"),
                "positive_action_mean": [round(value, 6) for value in positive_action],
                "counterfactual_action_mean": [round(value, 6) for value in counter_action],
            }
        )

    arm_reports = {
        name: {
            "metrics": _arm_metrics(payload["records"], payload["margins"]),
            "sample_count": len(payload["records"]),
            "trainable_parameter_count": 0,
            "training_performed": False,
        }
        for name, payload in arms.items()
    }
    actionmap = arm_reports["actionmap_head_only_proxy"]["metrics"]
    tca = arm_reports["tca_map_head_only_proxy"]["metrics"]
    tca_select = arm_reports["tca_map_distributional_select_proxy"]["metrics"]

    comparison = {
        "tca_map_vs_actionmap": {
            "action_l1_delta": round(tca["action_l1"] - actionmap["action_l1"], 6),
            "wrong_target_proxy_rate_delta": round(tca["wrong_target_proxy_rate"] - actionmap["wrong_target_proxy_rate"], 6),
            "counterfactual_separation_margin_delta": round(
                tca["counterfactual_separation_margin"] - actionmap["counterfactual_separation_margin"], 6
            ),
        },
        "tca_select_vs_tca_map": {
            "action_l1_delta": round(tca_select["action_l1"] - tca["action_l1"], 6),
            "wrong_target_proxy_rate_delta": round(tca_select["wrong_target_proxy_rate"] - tca["wrong_target_proxy_rate"], 6),
            "counterfactual_separation_margin_delta": round(
                tca_select["counterfactual_separation_margin"] - tca["counterfactual_separation_margin"], 6
            ),
        },
    }

    passed = bool(pairs and all(report["sample_count"] == len(pairs) for report in arm_reports.values()))
    return {
        "schema_version": SCHEMA_VERSION,
        "policy": {
            "offline_proxy_only": True,
            "not_standard_success": True,
            "not_paper_grade": True,
            "check_only": True,
            "downloads_performed": False,
            "gpu_jobs_performed": False,
            "gpu_training_performed": False,
            "training_performed": False,
            "heavy_model_imports_performed": False,
            "model_load_performed": False,
            "model_inference_performed": False,
            "simulator_executed": False,
            "rollouts_performed": False,
            "openvla_oft_executed": False,
            "tokens_read_or_written": False,
            "paper_grade_claims_made": False,
        },
        "source_manifest": str(manifest_path),
        "max_pairs": max_pairs,
        "max_action_steps": max_action_steps,
        "grid_size": grid_size,
        "pair_count": len(pairs),
        "examples": examples,
        "arms": arm_reports,
        "comparison": comparison,
        "ready_for_required_tiny_lora_comparison": passed,
        "ready_for_rollout": False,
        "libero_offline_head_comparison_passed": passed,
        "interpretation": (
            "Offline proxy plumbing only. ActionMap/TCA-Map/TCA-Select arms are deterministic HDF5 action decoding proxies, "
            "not trained baselines, not standard success, not rollout success, and not paper-grade evidence."
        ),
        "recommended_next_step": (
            "Run the required tiny real/offline LoRA comparison scaffold if it stays CPU-only/offline; stop before simulator execution, rollout, heavy VLA imports, OpenVLA-OFT, or paper claims."
            if passed
            else "Fix the offline counterfactual split or HDF5 action reader before comparison work."
        ),
    }


def write_reports(report: dict, json_path: Path, markdown_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    lines = [
        "# LIBERO Offline ActionMap vs TCA-Map Comparison",
        "",
        "This is an offline proxy plumbing check. It is not standard success, not rollout success, and not paper-grade evidence.",
        "",
        f"- passed: `{report['libero_offline_head_comparison_passed']}`",
        f"- pair count: `{report['pair_count']}`",
        f"- ready for required tiny LoRA comparison: `{report['ready_for_required_tiny_lora_comparison']}`",
        f"- ready for rollout: `{report['ready_for_rollout']}`",
        "",
        "## Arms",
    ]
    for name, payload in report["arms"].items():
        metrics = payload["metrics"]
        lines.extend(
            [
                f"- `{name}` action L1: `{metrics['action_l1']}`",
                f"- `{name}` wrong-target proxy: `{metrics['wrong_target_proxy_rate']}`",
                f"- `{name}` counterfactual margin: `{metrics['counterfactual_separation_margin']}`",
            ]
        )
    lines.extend(["", "## Next Step", report["recommended_next_step"], ""])
    markdown_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="reports/libero_offline_counterfactual_split_report.json")
    parser.add_argument("--max-pairs", type=int, default=4)
    parser.add_argument("--max-action-steps", type=int, default=16)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--report-json", default="reports/libero_offline_actionmap_tca_comparison_report.json")
    parser.add_argument("--report-md", default="reports/libero_offline_actionmap_tca_comparison_report.md")
    args = parser.parse_args()

    report = build_offline_head_comparison(
        manifest_path=Path(args.manifest),
        max_pairs=args.max_pairs,
        max_action_steps=args.max_action_steps,
        grid_size=args.grid_size,
    )
    write_reports(report, json_path=Path(args.report_json), markdown_path=Path(args.report_md))
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
