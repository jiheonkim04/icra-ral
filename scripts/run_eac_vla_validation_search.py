"""Run EAC-VLA bounded validation-only design search."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.smolvla.eac_vla import PROPOSAL_HASH  # noqa: E402


DATE_KST = "2026-07-15"
VALIDATION_SCORE_WEIGHTS = {
    "risk_exposure_reduction_proxy": 0.45,
    "clean_action_value_passthrough": 0.20,
    "mechanism_activation": 0.15,
    "runtime_action_validity": 0.10,
    "latency_penalty": -0.05,
    "oscillation_penalty": -0.05,
}
CONFIGS = [
    {"config_id": "eac_q25_balanced_2_8_50", "quantile_margin": 0.25, "commitment_map": {"short": 2, "medium": 8, "long": 50}},
    {"config_id": "eac_q33_balanced_2_8_50", "quantile_margin": 0.33, "commitment_map": {"short": 2, "medium": 8, "long": 50}},
    {"config_id": "eac_q40_balanced_2_8_50", "quantile_margin": 0.40, "commitment_map": {"short": 2, "medium": 8, "long": 50}},
    {"config_id": "eac_q25_aggressive_1_4_50", "quantile_margin": 0.25, "commitment_map": {"short": 1, "medium": 4, "long": 50}},
    {"config_id": "eac_q33_aggressive_1_4_50", "quantile_margin": 0.33, "commitment_map": {"short": 1, "medium": 4, "long": 50}},
    {"config_id": "eac_q40_aggressive_1_4_50", "quantile_margin": 0.40, "commitment_map": {"short": 1, "medium": 4, "long": 50}},
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(dict(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_md(path: Path, report: Mapping[str, Any]) -> None:
    selected = report["selected_config"]
    lines = [
        "# EAC-VLA Validation Search",
        "",
        f"Date: `{DATE_KST}`",
        "",
        f"Proposal hash: `{PROPOSAL_HASH}`",
        "",
        f"Final decision: `{report['final_decision']}`",
        "",
        f"- closed-loop experiment happened: `{report['closed_loop_experiment_happened']}`",
        f"- training happened: `{report['training_happened']}`",
        f"- validation search happened: `{report['validation_search_happened']}`",
        f"- confirmatory-test tuning happened: `{report['confirmatory_test_tuning_happened']}`",
        f"- confirmatory records used for tuning: `{report['confirmatory_records_used_for_tuning']}`",
        f"- tried config count: `{report['tried_config_count']}`",
        f"- selected config: `{selected['config_id']}`",
        f"- selected validation score: `{selected['validation_score']}`",
        f"- selected commitment counts: `{selected['commitment_counts']}`",
        f"- selected policy calls per step proxy: `{selected['policy_calls_per_step_proxy']}`",
        f"- selected oscillation fraction: `{selected['oscillation_fraction']}`",
        f"- selected risk exposure reduction proxy: `{selected['score_components']['risk_exposure_reduction_proxy']}`",
        "",
        "Validation score weights:",
        "",
        "```json",
        json.dumps(report["validation_score_weights"], indent=2, sort_keys=True),
        "```",
        "",
        "Tried configurations:",
        "",
        "```json",
        json.dumps(report["tried_configs"], indent=2, sort_keys=True),
        "```",
        "",
        "Reference baselines:",
        "",
        "```json",
        json.dumps(report["reference_baselines"], indent=2, sort_keys=True),
        "```",
        "",
        "Hard stop reasons:",
    ]
    hard_stops = list(report.get("hard_stop_reasons") or [])
    if hard_stops:
        lines.extend(f"- `{reason}`" for reason in hard_stops)
    else:
        lines.append("- none")
    lines.extend(["", f"Next step: {report['next_step']}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _frame_key(record: Mapping[str, Any]) -> tuple[int, int, int]:
    return (int(record["task_index"]), int(record["episode_index"]), int(record["frame_index"]))


def _as_array(value: Any, shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if tuple(array.shape) != shape:
        raise ValueError(f"expected {shape}, got {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("nonfinite array")
    return array


def _robust_norm(values: np.ndarray) -> np.ndarray:
    lo, hi = np.quantile(values, [0.05, 0.95])
    if hi <= lo:
        return np.zeros_like(values)
    return np.clip((values - lo) / (hi - lo), 0.0, 1.0)


def _summarize(values: Sequence[float]) -> dict[str, Any]:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {"count": 0, "mean": None, "min": None, "p50": None, "p95": None, "max": None}
    return {
        "count": int(arr.size),
        "mean": float(np.mean(arr)),
        "min": float(np.min(arr)),
        "p50": float(np.quantile(arr, 0.50)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _validation_frame_metrics(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[int, int, int], list[Mapping[str, Any]]] = defaultdict(list)
    for row in records:
        if str(row.get("split")) == "val":
            grouped[_frame_key(row)].append(row)

    metrics = []
    dispersion_values = []
    transition_values = []
    for key, rows in sorted(grouped.items()):
        previews = []
        for row in rows:
            previews.append(_as_array(row["base_action_chunk_first_two_preview"], (2, 7)))
        stacked = np.stack(previews, axis=0)
        dispersion = float(np.mean(np.var(stacked, axis=0)))
        first_transition = float(np.mean([np.linalg.norm(item[1] - item[0]) for item in previews]))
        dispersion_values.append(dispersion)
        transition_values.append(first_transition)
        metrics.append(
            {
                "task_index": int(key[0]),
                "episode_index": int(key[1]),
                "frame_index": int(key[2]),
                "first_two_dispersion": dispersion,
                "first_transition_l2": first_transition,
            }
        )

    risk = 0.67 * _robust_norm(np.asarray(dispersion_values, dtype=np.float64)) + 0.33 * _robust_norm(np.asarray(transition_values, dtype=np.float64))
    for item, value in zip(metrics, risk.tolist()):
        item["risk"] = float(value)
    return metrics


def _assign_commitments(risk: np.ndarray, config: Mapping[str, Any]) -> np.ndarray:
    q = float(config["quantile_margin"])
    lower = float(np.quantile(risk, q))
    upper = float(np.quantile(risk, 1.0 - q))
    cmap = dict(config["commitment_map"])
    commitments = np.full(risk.shape, int(cmap["medium"]), dtype=np.int64)
    commitments[risk <= lower] = int(cmap["long"])
    commitments[risk >= upper] = int(cmap["short"])
    return commitments


def _oscillation_fraction(metrics: Sequence[Mapping[str, Any]], commitments: np.ndarray) -> float:
    by_sequence: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for item, commitment in zip(metrics, commitments.tolist()):
        by_sequence[(int(item["task_index"]), int(item["episode_index"]))].append((int(item["frame_index"]), int(commitment)))
    changes = 0
    total = 0
    for rows in by_sequence.values():
        ordered = [commitment for _, commitment in sorted(rows)]
        for left, right in zip(ordered, ordered[1:]):
            total += 1
            changes += int(left != right)
    return 0.0 if total == 0 else float(changes / total)


def _score_config(
    config: Mapping[str, Any],
    metrics: Sequence[Mapping[str, Any]],
    *,
    clean_ok: bool,
    runtime_ok: bool,
) -> dict[str, Any]:
    risk = np.asarray([float(item["risk"]) for item in metrics], dtype=np.float64)
    commitments = _assign_commitments(risk, config)
    counts = Counter(int(item) for item in commitments.tolist())
    max_share = max(counts.values()) / float(len(commitments))
    risk_exposure = float(np.mean(risk * (1.0 - (commitments / 50.0))))
    oracle_exposure = float(np.mean(risk * (1.0 - (1.0 / 50.0))))
    risk_exposure_reduction_proxy = 0.0 if oracle_exposure <= 0.0 else float(risk_exposure / oracle_exposure)
    policy_calls_per_step = float(np.mean(1.0 / commitments))
    latency_penalty = float((policy_calls_per_step - (1.0 / 50.0)) / (1.0 - (1.0 / 50.0)))
    oscillation = _oscillation_fraction(metrics, commitments)
    activation = float(1.0 - max_share)
    short_risk = float(np.mean(risk[commitments == min(counts)])) if np.any(commitments == min(counts)) else None
    long_risk = float(np.mean(risk[commitments == max(counts)])) if np.any(commitments == max(counts)) else None
    monotonic_ok = bool(short_risk is not None and long_risk is not None and short_risk > long_risk)
    components = {
        "risk_exposure_reduction_proxy": risk_exposure_reduction_proxy,
        "clean_action_value_passthrough": 1.0 if clean_ok else 0.0,
        "mechanism_activation": activation,
        "runtime_action_validity": 1.0 if runtime_ok and monotonic_ok else 0.0,
        "latency_penalty": latency_penalty,
        "oscillation_penalty": oscillation,
    }
    score = sum(float(VALIDATION_SCORE_WEIGHTS[name]) * float(value) for name, value in components.items())
    return {
        "config_id": str(config["config_id"]),
        "quantile_margin": float(config["quantile_margin"]),
        "commitment_map": dict(config["commitment_map"]),
        "validation_score": float(score),
        "score_components": components,
        "commitment_counts": {str(key): int(value) for key, value in sorted(counts.items())},
        "max_commitment_share": float(max_share),
        "policy_calls_per_step_proxy": policy_calls_per_step,
        "oscillation_fraction": oscillation,
        "risk_summary_by_commitment": {
            str(commitment): _summarize(risk[commitments == commitment].tolist())
            for commitment in sorted(counts)
        },
        "risk_monotonicity_short_gt_long": monotonic_ok,
    }


def _fixed_commitment_baseline(metrics: Sequence[Mapping[str, Any]], commitment: int) -> dict[str, Any]:
    risk = np.asarray([float(item["risk"]) for item in metrics], dtype=np.float64)
    commitments = np.full(risk.shape, int(commitment), dtype=np.int64)
    risk_exposure = float(np.mean(risk * (1.0 - (commitments / 50.0))))
    oracle_exposure = float(np.mean(risk * (1.0 - (1.0 / 50.0))))
    return {
        "commitment": int(commitment),
        "risk_exposure_reduction_proxy": 0.0 if oracle_exposure <= 0.0 else float(risk_exposure / oracle_exposure),
        "policy_calls_per_step_proxy": float(np.mean(1.0 / commitments)),
        "oscillation_fraction": 0.0,
    }


def build_validation_search(args: argparse.Namespace) -> dict[str, Any]:
    canonical = _read_json(Path(args.canonical_base_artifact))
    stage0 = _read_json(Path(args.stage0_audit))
    runtime = _read_json(Path(args.runtime_queue_check))
    metrics = _validation_frame_metrics(canonical["records"])
    clean_ok = bool((stage0.get("action_value_passthrough_summary") or {}).get("max", 1.0) <= 1e-6)
    runtime_ok = runtime.get("final_decision") == "EAC_RUNTIME_QUEUE_CHECK_PASS_VALIDATION_SEARCH_ALLOWED"
    tried = [_score_config(config, metrics, clean_ok=clean_ok, runtime_ok=runtime_ok) for config in CONFIGS]
    tried = sorted(tried, key=lambda item: (-float(item["validation_score"]), str(item["config_id"])))
    selected = dict(tried[0])
    hard_stops = []
    if stage0.get("final_decision") != "AUDIT_PASS_PROCEED_TO_VALIDATION_SEARCH":
        hard_stops.append("Stage 0 audit is not passed")
    if not runtime_ok:
        hard_stops.append("runtime queue check is not passed")
    if len(CONFIGS) != 6:
        hard_stops.append("validation search budget is not exactly six configurations")
    if len(metrics) != 400:
        hard_stops.append(f"validation frame count changed: {len(metrics)}")
    if selected["score_components"]["runtime_action_validity"] < 1.0:
        hard_stops.append("selected config failed runtime action validity or risk monotonicity")

    final_decision = "EAC_VALIDATION_SEARCH_SELECT_CONFIG_STAGE_A_MANIFEST_READY" if not hard_stops else "DESIGN_FAILURE"
    return {
        "schema_version": 1,
        "date_kst": DATE_KST,
        "method": "EAC-VLA",
        "proposal_hash": PROPOSAL_HASH,
        "mode": args.mode,
        "final_decision": final_decision,
        "closed_loop_experiment_happened": False,
        "training_happened": False,
        "validation_search_happened": True,
        "confirmatory_test_tuning_happened": False,
        "confirmatory_records_used_for_tuning": False,
        "source_canonical_base_artifact": str(args.canonical_base_artifact),
        "source_stage0_audit": str(args.stage0_audit),
        "source_runtime_queue_check": str(args.runtime_queue_check),
        "validation_score_weights": dict(VALIDATION_SCORE_WEIGHTS),
        "search_budget": {
            "max_total_configurations": 6,
            "actual_total_configurations": len(CONFIGS),
            "threshold_quantile_values": [0.25, 0.33, 0.40],
            "commitment_maps": [
                {"short": 2, "medium": 8, "long": 50},
                {"short": 1, "medium": 4, "long": 50},
            ],
            "random_seeds_used": 0,
            "architecture_choices": 0,
            "confirmatory_test_identities_used": False,
        },
        "validation_frame_count": len(metrics),
        "tried_config_count": len(tried),
        "tried_configs": tried,
        "selected_config": selected,
        "selected_config_id": selected["config_id"],
        "reference_baselines": {
            "frozen_smolvla_fixed_queue": _fixed_commitment_baseline(metrics, 50),
            "fixed_short_replan_baseline_commitment_2": _fixed_commitment_baseline(metrics, 2),
            "fixed_short_replan_baseline_commitment_1": _fixed_commitment_baseline(metrics, 1),
        },
        "risk_summary": _summarize([float(item["risk"]) for item in metrics]),
        "hard_stop_reasons": hard_stops,
        "next_step": (
            "Freeze the EAC Stage A matched manifest and preflight the five policy identities."
            if not hard_stops
            else "Do not run Stage A; classify validation search failure and pivot or repair only the concrete defect."
        ),
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["validation-search"], default="validation-search")
    parser.add_argument("--canonical-base-artifact", default="reports/canonical_frozen_base_prediction_artifact.json")
    parser.add_argument("--stage0-audit", default="reports/eac_vla/stage_0_audit.json")
    parser.add_argument("--runtime-queue-check", default="reports/eac_vla/runtime_queue_check.json")
    parser.add_argument("--json-output", default="reports/eac_vla/validation_search.json")
    parser.add_argument("--md-output", default="reports/eac_vla/validation_search.md")
    parser.add_argument("--selected-config-output", default="reports/eac_vla/selected_config.json")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_validation_search(args)
    _write_json(Path(args.json_output), report)
    _write_md(Path(args.md_output), report)
    _write_json(Path(args.selected_config_output), report["selected_config"])
    print(
        json.dumps(
            {
                "mode": args.mode,
                "final_decision": report["final_decision"],
                "hard_stop_count": len(report["hard_stop_reasons"]),
                "selected_config_id": report["selected_config_id"],
                "selected_validation_score": report["selected_config"]["validation_score"],
                "json_output": args.json_output,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
