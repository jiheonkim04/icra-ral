#!/usr/bin/env python
"""Analyze the preregistered wrist-dropout condition-generality study."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any


TASKS = [
    ("libero_goal", 0, [20260733, 20260734, 20260735]),
    ("libero_object", 0, [20260733, 20260734, 20260735]),
    ("libero_spatial", 5, [20260731, 20260732, 20260735]),
]
ORIGINAL_TASK_KEY = "libero_spatial/task5"
EXPECTED_PAIR_COUNT = 9


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> dict[str, float | None]:
    if total <= 0:
        return {"low": None, "high": None}
    phat = successes / total
    denom = 1.0 + z * z / total
    center = (phat + z * z / (2.0 * total)) / denom
    half = (z / denom) * math.sqrt((phat * (1.0 - phat) / total) + (z * z / (4.0 * total * total)))
    return {"low": max(0.0, center - half), "high": min(1.0, center + half)}


def binom_cdf(k: int, n: int, p: float = 0.5) -> float:
    return sum(math.comb(n, i) * (p**i) * ((1.0 - p) ** (n - i)) for i in range(k + 1))


def exact_mcnemar_p_value(clean_to_dropout_fail: int, dropout_only_success: int) -> float | None:
    discordant = clean_to_dropout_fail + dropout_only_success
    if discordant == 0:
        return 1.0
    lo = min(clean_to_dropout_fail, dropout_only_success)
    hi = max(clean_to_dropout_fail, dropout_only_success)
    lower = binom_cdf(lo, discordant, 0.5)
    upper = 1.0 - binom_cdf(hi - 1, discordant, 0.5)
    return min(1.0, 2.0 * min(lower, upper))


def task_key(suite: str, task_id: int) -> str:
    return f"{suite}/task{int(task_id)}"


def build_report(run_dir: Path) -> dict[str, Any]:
    summary = load_json(run_dir / "generality_summary.json")
    rows = summary.get("rows") or []
    by_pair: dict[tuple[str, int, int], dict[str, Any]] = defaultdict(dict)
    for row in rows:
        key = (str(row["suite"]), int(row["task_id"]), int(row["reset_identity"]))
        by_pair[key][str(row["condition"])] = row

    pairs: list[dict[str, Any]] = []
    infrastructure_exceptions: list[dict[str, Any]] = []
    for suite, task_id, identities in TASKS:
        for identity in identities:
            payload = by_pair.get((suite, task_id, identity), {})
            clean = payload.get("clean")
            dropout = payload.get("dropout")
            pair = {
                "task_key": task_key(suite, task_id),
                "suite": suite,
                "task_id": task_id,
                "reset_identity": identity,
                "clean_result_path": clean.get("result_path") if clean else None,
                "dropout_result_path": dropout.get("result_path") if dropout else None,
                "clean_completed": bool(clean and clean.get("completed")),
                "dropout_completed": bool(dropout and dropout.get("completed")),
                "clean_success": bool(clean and clean.get("success")),
                "dropout_success": bool(dropout and dropout.get("success")),
                "clean_steps": clean.get("steps") if clean else None,
                "dropout_steps": dropout.get("steps") if dropout else None,
                "clean_action_chunk_count": clean.get("action_chunk_count") if clean else None,
                "dropout_action_chunk_count": dropout.get("action_chunk_count") if dropout else None,
            }
            if clean is None or dropout is None or not pair["clean_completed"] or not pair["dropout_completed"]:
                infrastructure_exceptions.append(pair)
            pairs.append(pair)

    clean_success_count = sum(1 for pair in pairs if pair["clean_success"])
    dropout_success_count = sum(1 for pair in pairs if pair["dropout_success"])
    clean_success_dropout_failure_flips = [
        pair for pair in pairs if pair["clean_success"] and not pair["dropout_success"]
    ]
    dropout_only_successes = [
        pair for pair in pairs if (not pair["clean_success"]) and pair["dropout_success"]
    ]
    tasks_with_flips = sorted({pair["task_key"] for pair in clean_success_dropout_failure_flips})
    task_distribution: dict[str, dict[str, int]] = {}
    for pair in pairs:
        item = task_distribution.setdefault(
            pair["task_key"],
            {"pairs": 0, "clean_successes": 0, "dropout_successes": 0, "clean_to_dropout_failure_flips": 0},
        )
        item["pairs"] += 1
        item["clean_successes"] += int(pair["clean_success"])
        item["dropout_successes"] += int(pair["dropout_success"])
        item["clean_to_dropout_failure_flips"] += int(pair["clean_success"] and not pair["dropout_success"])

    valid = (
        len(pairs) == EXPECTED_PAIR_COUNT
        and len(infrastructure_exceptions) == 0
        and int(summary.get("infrastructure_failure_count") or 0) == 0
        and int(summary.get("row_count") or 0) == EXPECTED_PAIR_COUNT * 2
    )
    clean_meaningfully_nonzero = clean_success_count >= 3
    repeated_degradation = len(clean_success_dropout_failure_flips) >= 3
    broad_enough = len(tasks_with_flips) >= 2 or len(clean_success_dropout_failure_flips) >= 4
    original_task_only = (
        bool(clean_success_dropout_failure_flips)
        and set(tasks_with_flips).issubset({ORIGINAL_TASK_KEY})
    )
    if not valid:
        decision = "WRIST_DROPOUT_EVALUATION_INVALID"
    elif clean_meaningfully_nonzero and repeated_degradation and broad_enough:
        decision = "WRIST_DROPOUT_REPEATED_PROBLEM_CONFIRMED"
    elif original_task_only:
        decision = "WRIST_DROPOUT_TASK5_LOCALIZED"
    else:
        decision = "WRIST_DROPOUT_UNDERPOWERED_ONE_FIXED_EXPANSION_ALLOWED"

    paired_uncertainty = {
        "clean_success_dropout_failure_flips": len(clean_success_dropout_failure_flips),
        "dropout_only_successes": len(dropout_only_successes),
        "discordant_pair_count": len(clean_success_dropout_failure_flips) + len(dropout_only_successes),
        "exact_mcnemar_two_sided_p_value": exact_mcnemar_p_value(
            len(clean_success_dropout_failure_flips), len(dropout_only_successes)
        ),
        "degradation_rate_among_clean_success_pairs": (
            len(clean_success_dropout_failure_flips) / clean_success_count if clean_success_count else None
        ),
        "degradation_rate_wilson_95_ci": wilson_interval(
            len(clean_success_dropout_failure_flips), clean_success_count
        ),
    }
    return {
        "schema_version": "2026-07-18.epoch5_wrist_dropout_generality_result.v1",
        "stage": "epoch_5_wrist_dropout_condition_generality_frozen_prior_only",
        "decision": decision,
        "run_dir": str(run_dir),
        "policy": "frozen official X-VLA-Libero",
        "condition": "wrist_blackout",
        "task_count": 3,
        "reset_identities_per_task": 3,
        "clean_episode_count": sum(1 for row in rows if row.get("condition") == "clean"),
        "dropout_episode_count": sum(1 for row in rows if row.get("condition") == "dropout"),
        "clean_success_count": clean_success_count,
        "dropout_success_count": dropout_success_count,
        "paired_clean_success_dropout_failure_flip_count": len(clean_success_dropout_failure_flips),
        "tasks_with_flips": tasks_with_flips,
        "task_distribution": task_distribution,
        "paired_uncertainty": paired_uncertainty,
        "infrastructure_exceptions": infrastructure_exceptions,
        "model_forward_count": int(summary.get("model_forward_count") or 0),
        "cuda_devices": summary.get("cuda_devices") or [],
        "peak_cuda_max_allocated_mib": summary.get("peak_cuda_max_allocated_mib"),
        "training_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "ours_rollout_happened": False,
        "control_rollout_happened": False,
        "method_selected": False,
        "broad_natural_reset_sweep_happened": False,
        "decision_rule": {
            "confirm": "valid, clean_success_count>=3, paired clean-success/dropout-failure flips>=3, and flips across >=2 tasks or >=4 identities",
            "localized": "valid and flips occur only on original libero_spatial/task5",
            "underpowered": "valid but neither confirmed nor localized",
            "invalid": "missing/incomplete pairs or infrastructure failures",
        },
        "pairs": pairs,
        "next_action": (
            "Select an actual external prior before any learned Ours method."
            if decision == "WRIST_DROPOUT_REPEATED_PROBLEM_CONFIRMED"
            else "Do not design another method under this condition until the steer permits the next bounded action."
        ),
    }


def write_markdown(report: dict[str, Any], output_path: Path) -> None:
    lines = [
        "# Wrist-Dropout Condition Generality Result",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Run: `{report['run_dir']}`",
        "- Policy: frozen official X-VLA only; no Ours, training, optimizer, or checkpoint.",
        "",
        "## Aggregate",
        "",
        f"- Clean successes: `{report['clean_success_count']}/9`",
        f"- Dropout successes: `{report['dropout_success_count']}/9`",
        f"- Paired clean-success → dropout-failure flips: `{report['paired_clean_success_dropout_failure_flip_count']}`",
        f"- Tasks with flips: `{', '.join(report['tasks_with_flips']) if report['tasks_with_flips'] else 'NONE'}`",
        f"- Model forward count: `{report['model_forward_count']}`",
        f"- CUDA devices: `{', '.join(report['cuda_devices']) if report['cuda_devices'] else 'NOT_RECORDED'}`",
        f"- Peak CUDA max allocated MiB: `{report['peak_cuda_max_allocated_mib']}`",
        "",
        "## Paired uncertainty",
        "",
        f"- Exact McNemar two-sided p-value: `{report['paired_uncertainty']['exact_mcnemar_two_sided_p_value']}`",
        f"- Degradation rate among clean-success pairs: `{report['paired_uncertainty']['degradation_rate_among_clean_success_pairs']}`",
        f"- Wilson 95% CI: `{report['paired_uncertainty']['degradation_rate_wilson_95_ci']}`",
        "",
        "## Task distribution",
        "",
        "| task | pairs | clean successes | dropout successes | clean→dropout flips |",
        "|---|---:|---:|---:|---:|",
    ]
    for task_key, item in sorted(report["task_distribution"].items()):
        lines.append(
            f"| {task_key} | {item['pairs']} | {item['clean_successes']} | "
            f"{item['dropout_successes']} | {item['clean_to_dropout_failure_flips']} |"
        )
    lines.extend(
        [
            "",
            "## Pairs",
            "",
            "| task | identity | clean success | dropout success | clean steps | dropout steps | clean chunks | dropout chunks |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for pair in report["pairs"]:
        lines.append(
            f"| {pair['task_key']} | {pair['reset_identity']} | {pair['clean_success']} | "
            f"{pair['dropout_success']} | {pair['clean_steps']} | {pair['dropout_steps']} | "
            f"{pair['clean_action_chunk_count']} | {pair['dropout_action_chunk_count']} |"
        )
    lines.extend(["", f"Next action: {report['next_action']}", ""])
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    report = build_report(args.run_dir)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_markdown(report, args.output_md)
    print(json.dumps({"decision": report["decision"], "output_json": str(args.output_json)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
