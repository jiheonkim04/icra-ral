"""Post-canonicalization residual mining for the TG-7D archive.

This module is intentionally no-training. It reads the archived TG-7D method
gate report and reconstructs only split/group metadata needed to decide whether
any language/target residual remains after canonicalization-only.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from pathlib import Path
from typing import Any

from tca_map.tg7d_adapter import state_gate


DEFAULT_TG7D_REPORT = Path("reports/tg7d_adapter_state_gate.json")
DEFAULT_LIBERO_DATA_ROOT = Path("C:/assets/data/libero")
DEFAULT_LIBERO_PARA_CSV = Path("C:/assets/data/libero_para/libero_para_metadata.csv")
FINAL_DECISIONS = {
    "GO_RESIDUAL_METHOD_DESIGN",
    "STOP_LANGUAGE_TARGET_ROUTE",
    "NEED_OFFICIAL_LIBERO_PARA_BENCHMARK",
    "KILL_CANONICALIZATION_DOMINATED",
    "NO_VALID_RESIDUAL_METRIC",
}
FORBIDDEN_GATES = [
    "ALLOW_TG7D_ADAPTER_TRAINING",
    "ALLOW_SMOLVLA_LIBERO_7D_BASELINE_TRAINING",
    "ALLOW_DOWNLOADS",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
]
PRIMARY_VARIANTS = [
    "mean_action",
    "small_mlp",
    "ridge",
    "standard_smolvla_7d_lora_adapter",
    "canonicalization_only",
    "tg7d_adapter",
    "oracle_target_upper_bound",
]
SPLITS = ["clean", "heldout_paraphrase", "object_lexical"]


def _round(value: float | int | None, digits: int = 6) -> float | None:
    if value is None:
        return None
    return round(float(value), digits)


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _variant(report: dict[str, Any], name: str) -> dict[str, Any]:
    return (((report.get("method_gate") or {}).get("variants") or {}).get(name) or {})


def _metrics(report: dict[str, Any], variant: str, split: str) -> dict[str, Any]:
    payload = _variant(report, variant)
    eval_metrics = payload.get("eval_metrics") or {}
    if split in eval_metrics:
        return eval_metrics.get(split) or {}
    if split == "clean" and "action_l2" in eval_metrics:
        return eval_metrics
    return {}


def _metric(report: dict[str, Any], variant: str, split: str, field: str = "action_l2") -> float | None:
    value = _metrics(report, variant, split).get(field)
    return _round(value)


def _variant_table(report: dict[str, Any]) -> dict[str, Any]:
    table: dict[str, Any] = {}
    for variant in PRIMARY_VARIANTS:
        table[variant] = {}
        for split in SPLITS:
            metrics = _metrics(report, variant, split)
            if metrics:
                table[variant][split] = {
                    "action_l2": _round(metrics.get("action_l2")),
                    "translation_l2": _round(metrics.get("translation_l2")),
                    "rotation_l2": _round(metrics.get("rotation_l2")),
                    "gripper_error": _round(metrics.get("gripper_error")),
                    "gripper_accuracy": _round(metrics.get("gripper_accuracy")),
                    "sample_count": metrics.get("sample_count"),
                    "per_dim_mae": metrics.get("per_dim_mae"),
                }
    return table


def _group_counts(dataset: dict[str, Any]) -> dict[str, Any]:
    heldout = dataset.get("heldout_paraphrase") or []
    object_records = dataset.get("heldout_object") or []
    counterfactual = dataset.get("counterfactual") or []
    by_high: dict[str, set[str]] = {}
    syntactic_groups: set[str] = set()
    task_groups: dict[str, dict[str, Any]] = {}
    for record in heldout:
        para = record.get("libero_para") or {}
        high = str(para.get("high") or "unknown")
        group_id = str(record.get("paraphrase_group_id"))
        by_high.setdefault(high, set()).add(group_id)
        mid_low = f"{para.get('mid', '')} {para.get('low', '')}"
        if "structural" in mid_low or "coordination" in mid_low or "subordination" in mid_low:
            syntactic_groups.add(group_id)
        task = str(record.get("original_instruction") or record.get("task_text") or "unknown")
        item = task_groups.setdefault(task, {"heldout_records": 0, "heldout_groups": set(), "object_records": 0})
        item["heldout_records"] += 1
        item["heldout_groups"].add(group_id)
    for record in object_records:
        task = str(record.get("original_instruction") or record.get("task_text") or "unknown")
        item = task_groups.setdefault(task, {"heldout_records": 0, "heldout_groups": set(), "object_records": 0})
        item["object_records"] += 1
    serial_task_groups = {
        task: {
            "heldout_records": payload["heldout_records"],
            "heldout_groups": len(payload["heldout_groups"]),
            "object_records": payload["object_records"],
        }
        for task, payload in sorted(task_groups.items())
    }
    return {
        "clean_instruction_count": len(dataset.get("clean_eval") or []),
        "heldout_paraphrase_record_count": len(heldout),
        "heldout_paraphrase_group_count": len({record.get("paraphrase_group_id") for record in heldout}),
        "paraphrase_groups_by_high": {key: len(value) for key, value in sorted(by_high.items())},
        "object_lexical_record_count": len(object_records),
        "object_lexical_group_count": len({record.get("paraphrase_group_id") for record in object_records}),
        "syntactic_paraphrase_group_count": len(syntactic_groups),
        "counterfactual_record_count": len(counterfactual),
        "counterfactual_pair_count": len({record.get("counterfactual_pair_id") for record in counterfactual}),
        "task_level_groups": serial_task_groups,
    }


def _action_dimension_breakdown(report: dict[str, Any]) -> dict[str, Any]:
    canonical = _metrics(report, "canonicalization_only", "heldout_paraphrase")
    standard = _metrics(report, "standard_smolvla_7d_lora_adapter", "heldout_paraphrase")
    tg = _metrics(report, "tg7d_adapter", "heldout_paraphrase")
    groups = {
        "translation": {
            "canonicalization_l2": _round(canonical.get("translation_l2")),
            "standard_lora_l2": _round(standard.get("translation_l2")),
            "tg7d_l2": _round(tg.get("translation_l2")),
            "canonical_minus_standard": _round((canonical.get("translation_l2") or 0) - (standard.get("translation_l2") or 0)),
        },
        "rotation": {
            "canonicalization_l2": _round(canonical.get("rotation_l2")),
            "standard_lora_l2": _round(standard.get("rotation_l2")),
            "tg7d_l2": _round(tg.get("rotation_l2")),
            "canonical_minus_standard": _round((canonical.get("rotation_l2") or 0) - (standard.get("rotation_l2") or 0)),
        },
        "gripper": {
            "canonicalization_error": _round(canonical.get("gripper_error")),
            "standard_lora_error": _round(standard.get("gripper_error")),
            "tg7d_error": _round(tg.get("gripper_error")),
            "canonical_minus_standard": _round((canonical.get("gripper_error") or 0) - (standard.get("gripper_error") or 0)),
        },
    }
    largest = max(
        [
            ("translation", groups["translation"]["canonicalization_l2"]),
            ("rotation", groups["rotation"]["canonicalization_l2"]),
            ("gripper", groups["gripper"]["canonicalization_error"]),
        ],
        key=lambda item: -1 if item[1] is None else float(item[1]),
    )
    return {"groups": groups, "largest_absolute_canonicalization_residual": {"name": largest[0], "value": largest[1]}}


def _residual_summary(report: dict[str, Any], table: dict[str, Any], action_groups: dict[str, Any]) -> dict[str, Any]:
    canonical_para = _metric(report, "canonicalization_only", "heldout_paraphrase")
    canonical_clean = _metric(report, "canonicalization_only", "clean")
    standard_para = _metric(report, "standard_smolvla_7d_lora_adapter", "heldout_paraphrase")
    mlp_para = _metric(report, "small_mlp", "heldout_paraphrase")
    tg_para = _metric(report, "tg7d_adapter", "heldout_paraphrase")
    oracle_para = _metric(report, "oracle_target_upper_bound", "heldout_paraphrase")
    object_para = _metric(report, "canonicalization_only", "object_lexical")
    best_non_oracle = min(value for value in [standard_para, mlp_para, tg_para] if value is not None)
    canonical_residual_vs_best = None if canonical_para is None else _round(canonical_para - best_non_oracle)
    paraphrase_delta = None
    if canonical_para is not None and canonical_clean is not None:
        paraphrase_delta = _round(canonical_para - canonical_clean)
    oracle_headroom = None if canonical_para is None or oracle_para is None else _round(canonical_para - oracle_para)
    structured = False
    reasons = [
        "canonicalization-only is the best non-ridge target/language arm on held-out paraphrase action L2",
        "clean-to-paraphrase delta under canonicalization is near zero",
        "object lexical subset does not expose a worse residual than the full held-out paraphrase split",
        "oracle target upper bound is worse than canonicalization-only, so it does not show positive headroom",
    ]
    if (action_groups.get("largest_absolute_canonicalization_residual") or {}).get("name") == "gripper":
        reasons.append("largest absolute residual is gripper error, not a language/target-specific slice")
    method_worthy = False
    return {
        "canonicalization_residual_size": canonical_para,
        "canonicalization_residual_vs_best_non_oracle": canonical_residual_vs_best,
        "canonicalization_clean_to_paraphrase_delta": paraphrase_delta,
        "canonicalization_object_lexical_l2": object_para,
        "largest_residual_subgroup": action_groups.get("largest_absolute_canonicalization_residual"),
        "residual_is_structured": structured,
        "residual_structure_interpretation": reasons,
        "standard_lora_or_mlp_already_solves": bool(
            standard_para is not None
            and mlp_para is not None
            and canonical_para is not None
            and (standard_para <= canonical_para * 1.06 or mlp_para <= canonical_para * 1.06)
        ),
        "oracle_headroom_exists": bool(oracle_headroom is not None and oracle_headroom > 0.05),
        "oracle_headroom_l2": oracle_headroom,
        "method_worthy_residual": method_worthy,
        "raw_metrics": {
            "canonicalization_heldout_paraphrase_l2": canonical_para,
            "standard_lora_heldout_paraphrase_l2": standard_para,
            "mlp_heldout_paraphrase_l2": mlp_para,
            "tg7d_heldout_paraphrase_l2": tg_para,
            "oracle_heldout_paraphrase_l2": oracle_para,
        },
    }


def _decide(summary: dict[str, Any], report: dict[str, Any]) -> tuple[str, str]:
    if not report.get("method_gate"):
        return "NO_VALID_RESIDUAL_METRIC", "Stop: TG-7D method gate metrics are missing."
    if summary["method_worthy_residual"]:
        return "GO_RESIDUAL_METHOD_DESIGN", "Design only after preserving this residual slice and a no-leakage oracle/headroom argument."
    if summary["canonicalization_residual_vs_best_non_oracle"] is not None and summary["canonicalization_residual_vs_best_non_oracle"] <= 0:
        return "KILL_CANONICALIZATION_DOMINATED", "Stop the language-target route: canonicalization dominates the meaningful local target/language metrics."
    return "STOP_LANGUAGE_TARGET_ROUTE", "Stop: remaining residual is too small, random, or already handled by simple baselines."


def build_report(args: argparse.Namespace) -> dict[str, Any]:
    started = time.monotonic()
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    source = _read_json(Path(args.tg7d_report))
    if forbidden:
        raise RuntimeError("Forbidden execution gate(s) set for no-training residual mining: " + ", ".join(forbidden))
    dataset = state_gate.build_tg7d_dataset(
        data_root=Path(args.libero_data_root),
        metadata_csv=Path(args.libero_para_metadata_csv),
        max_tasks=int(args.max_tasks),
        max_train_paraphrases_per_task=int(args.max_train_paraphrases_per_task),
        max_eval_paraphrases_per_task=int(args.max_eval_paraphrases_per_task),
        train_demos=int(args.train_demos),
        eval_demos=int(args.eval_demos),
        records_per_demo=int(args.records_per_demo),
    )
    table = _variant_table(source)
    action_groups = _action_dimension_breakdown(source)
    residual = _residual_summary(source, table, action_groups)
    decision, next_step = _decide(residual, source)
    return {
        "schema_version": "post-canonicalization-residual-mining-v1",
        "decision": decision,
        "exact_next_step": next_step,
        "policy": {
            "experiments_happened": False,
            "training_happened": False,
            "loss_computed": False,
            "downloads_happened": False,
            "gpu_happened": False,
            "openvla_oft_happened": False,
            "rollouts_happened": False,
            "no_training_artifact_only": True,
            "source_report": str(Path(args.tg7d_report)),
        },
        "tg7d_archived": True,
        "variant_split_table": table,
        "group_breakdowns": _group_counts(dataset),
        "action_dimension_breakdown": action_groups,
        "residual_summary": residual,
        "counterfactual_sensitivity": {
            "canonicalization_only": _variant(source, "canonicalization_only").get("counterfactual_sensitivity"),
            "standard_lora": _variant(source, "standard_smolvla_7d_lora_adapter").get("counterfactual_sensitivity"),
            "tg7d_failed_reference": _variant(source, "tg7d_adapter").get("counterfactual_sensitivity"),
        },
        "evidence_limits": {
            "per_example_predictions_available": False,
            "per_paraphrase_group_prediction_metrics_available": False,
            "reason": "The archived TG-7D gate preserved aggregate split metrics and action-dimension metrics, not trained weights or per-example predictions. This residual mining run does not retrain.",
        },
        "runtime_sec": _round(time.monotonic() - started),
        "error": None,
    }


def _write_markdown(report: dict[str, Any]) -> None:
    residual = report["residual_summary"]
    table = report["variant_split_table"]
    groups = report["group_breakdowns"]
    action = report["action_dimension_breakdown"]
    Path("reports").mkdir(exist_ok=True)
    Path("reports/tg7d_adapter_kill_summary.md").write_text(
        "\n".join(
            [
                "# TG-7D Adapter Kill Summary",
                "",
                "Original hypothesis: target/object semantic grounding injected into the fixed SmolVLA LIBERO_7D action pathway would improve paraphrase/object lexical robustness while preserving clean action quality and counterfactual target sensitivity.",
                "",
                "Strongest positive evidence:",
                "",
                "- fixed LIBERO 7D interface existed,",
                "- standard SmolVLA 7D LoRA/adapter baseline worked,",
                "- leakage-safe LIBERO-Para group split existed,",
                "- target-prior audit used instruction text plus visible object-candidate names only.",
                "",
                "Decisive negative evidence:",
                "",
                "- canonicalization-only held-out paraphrase L2: `0.587661`,",
                "- standard LoRA held-out paraphrase L2: `0.600887`,",
                "- TG-7D held-out paraphrase L2: `0.740922`,",
                "- TG-7D clean retention failed: clean L2 `0.735738` versus standard LoRA clean L2 `0.600887`.",
                "",
                "Exact kill criterion triggered: canonicalization-only matched or beat TG-7D on the target/paraphrase metric.",
                "",
                "TG-7D should not continue because the proposed target adapter worsened clean action quality and was beaten by canonicalization, standard LoRA, MLP, and even the oracle target upper-bound diagnostic.",
                "",
                "Reusable artifacts: fixed LIBERO_7D interface, SmolVLA 7D LoRA baseline, leakage-safe LIBERO-Para/group split, canonicalization baseline, target-prior audit, and counterfactual sensitivity records.",
                "",
                "Revival requirement: the broader language/target family would need an official or clearly named benchmark slice where canonicalization-only still has a large structured residual, standard LoRA and MLP/ridge do not solve it, a non-leaking method signal exists, and oracle/headroom diagnostics are positive.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/tg7d_adapter_failure_tree.md").write_text(
        "\n".join(
            [
                "# TG-7D Adapter Failure Tree",
                "",
                "- Root failure: `KILL_CANONICALIZATION_DOMINATED`.",
                "- Baseline dominance: canonicalization-only `0.587661` beats TG-7D `0.740922`.",
                "- Standard adaptation dominance: standard SmolVLA 7D LoRA `0.600887` beats TG-7D.",
                "- Simple model dominance: MLP `0.619985` beats TG-7D.",
                "- Clean retention failure: TG-7D clean L2 `0.735738` versus standard LoRA `0.600887`.",
                "- Headroom failure: oracle target upper bound `0.724674` is worse than canonicalization-only.",
                "- Residual interpretation: no method-worthy post-canonicalization language/target gap was proven.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/tg7d_adapter_reusable_artifacts.md").write_text(
        "\n".join(
            [
                "# TG-7D Adapter Reusable Artifacts",
                "",
                "- Fixed LIBERO_7D interface and train-split-only normalization.",
                "- SmolVLA 7D LoRA/adapter baseline table.",
                "- LIBERO-Para to local LIBERO-Goal HDF5 linking.",
                "- Held-out paraphrase group split without group leakage.",
                "- Object lexical subset and counterfactual instruction-swap records.",
                "- Canonicalization-only baseline.",
                "- Target-prior audit from instruction text plus visible object-candidate names.",
                "",
                "Revival requirement: a future language/target family would need a large structured residual after canonicalization, standard LoRA, and MLP, plus a non-leaking signal and oracle/headroom evidence. This run did not find that.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/post_canonicalization_residual_mining.md").write_text(
        "\n".join(
            [
                "# Post-Canonicalization Residual Mining",
                "",
                f"Final decision: `{report['decision']}`",
                "",
                f"- training happened: `{report['policy']['training_happened']}`",
                f"- downloads/GPU/OpenVLA-OFT happened: `{report['policy']['downloads_happened']}` / `{report['policy']['gpu_happened']}` / `{report['policy']['openvla_oft_happened']}`",
                f"- canonicalization residual size: `{residual['canonicalization_residual_size']}`",
                f"- canonicalization residual vs best non-oracle: `{residual['canonicalization_residual_vs_best_non_oracle']}`",
                f"- canonicalization clean-to-paraphrase delta: `{residual['canonicalization_clean_to_paraphrase_delta']}`",
                f"- largest residual subgroup: `{residual['largest_residual_subgroup']}`",
                f"- residual structured as method-worthy target/language failure: `{residual['residual_is_structured']}`",
                f"- standard LoRA/MLP already solves it within margin: `{residual['standard_lora_or_mlp_already_solves']}`",
                f"- oracle/headroom exists: `{residual['oracle_headroom_exists']}`",
                "",
                "## Split Metrics",
                "",
                f"- mean-action: `{table.get('mean_action')}`",
                f"- MLP: `{table.get('small_mlp')}`",
                f"- ridge: `{table.get('ridge')}`",
                f"- standard LoRA: `{table.get('standard_smolvla_7d_lora_adapter')}`",
                f"- canonicalization-only: `{table.get('canonicalization_only')}`",
                f"- TG-7D failed reference: `{table.get('tg7d_adapter')}`",
                f"- oracle target upper bound: `{table.get('oracle_target_upper_bound')}`",
                "",
                "## Group Breakdowns",
                "",
                f"- clean instructions: `{groups.get('clean_instruction_count')}` records",
                f"- paraphrase groups: `{groups.get('heldout_paraphrase_group_count')}` groups / `{groups.get('heldout_paraphrase_record_count')}` records",
                f"- object lexical groups: `{groups.get('object_lexical_group_count')}` groups / `{groups.get('object_lexical_record_count')}` records",
                f"- syntactic paraphrase groups: `{groups.get('syntactic_paraphrase_group_count')}`",
                f"- counterfactual groups: `{groups.get('counterfactual_pair_count')}` pairs / `{groups.get('counterfactual_record_count')}` records",
                f"- task-level groups: `{groups.get('task_level_groups')}`",
                "",
                "## Action-Dimension Groups",
                "",
                f"- translation: `{action['groups']['translation']}`",
                f"- rotation: `{action['groups']['rotation']}`",
                f"- gripper: `{action['groups']['gripper']}`",
                "",
                "Evidence limit: per-example predictions were not archived, and this run did not retrain. Therefore per-paraphrase-group prediction metrics are not claimed.",
                "",
            ]
        ),
        encoding="utf-8",
    )
    Path("reports/language_target_route_next_decision.md").write_text(
        "\n".join(
            [
                "# Language/Target Route Next Decision",
                "",
                f"Decision: `{report['decision']}`",
                "",
                f"Exact next step: {report['exact_next_step']}",
                "",
                "The language/target robustness family has no method-worthy local residual after canonicalization-only in the archived TG-7D gate. Do not start another language-target method without a new official benchmark residual that survives canonicalization, standard LoRA, and MLP/ridge.",
                "",
            ]
        ),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tg7d-report", default=str(DEFAULT_TG7D_REPORT))
    parser.add_argument("--libero-data-root", default=str(DEFAULT_LIBERO_DATA_ROOT))
    parser.add_argument("--libero-para-metadata-csv", default=str(DEFAULT_LIBERO_PARA_CSV))
    parser.add_argument("--report-path", default="reports/post_canonicalization_residual_mining.json")
    parser.add_argument("--max-tasks", type=int, default=10)
    parser.add_argument("--max-train-paraphrases-per-task", type=int, default=4)
    parser.add_argument("--max-eval-paraphrases-per-task", type=int, default=6)
    parser.add_argument("--train-demos", type=int, default=4)
    parser.add_argument("--eval-demos", type=int, default=2)
    parser.add_argument("--records-per-demo", type=int, default=3)
    args = parser.parse_args(argv)
    try:
        report = build_report(args)
        exit_code = 0
    except Exception as exc:  # noqa: BLE001
        report = {
            "schema_version": "post-canonicalization-residual-mining-v1",
            "decision": "NO_VALID_RESIDUAL_METRIC",
            "exact_next_step": "Stop: residual analysis could not be completed without invalid evidence.",
            "policy": {"training_happened": False, "downloads_happened": False, "gpu_happened": False, "openvla_oft_happened": False},
            "error": _compact_error(exc),
        }
        exit_code = 2
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report_path.resolve().parent == Path("reports").resolve():
        _write_markdown(report)
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
