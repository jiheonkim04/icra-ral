"""Official SmolVLA-LIBERO routing design gate.

This module computes oracle upper bounds for base/LoRA routing using the
official SmolVLA-LIBERO path. It does not implement or train a routing method.
It retrains the same bounded rank-4 LoRA baseline only because previous runs did
not save adapter weights or complete per-frame predictions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.official_libero_baseline_scaleup import (
    _add_training_batch_dims,
    _cuda_memory,
    _gradient_summary,
    _json_default,
    _loss_from_output,
    _parameter_summary,
    _postprocess_action,
    _raw_current_action,
    _rss_mb,
    _safe_autocast_status,
    _stat_vector,
    _tensor_devices,
    _to_float,
)
from tca_map.smolvla.official_libero_failure_mining import (
    DEFAULT_TRAIN_INDICES,
    MAX_TRAINING_STEPS,
    _mean,
    _metric_row,
    _phase,
    _task_text_map,
    evaluate_mean_prior_rows,
    evaluate_model_rows,
    select_eval_plan,
    summarize_rows,
)


FINAL_DECISIONS = {
    "GO_DESIGN_BASE_RETENTIVE_ROUTING",
    "GO_DESIGN_FRAME_CONDITIONAL_ROUTING",
    "GO_DESIGN_TASK_ROUTING",
    "NEED_MORE_OFFICIAL_FAILURE_MINING",
    "NO_ROUTING_HEADROOM",
    "ROUTING_NOVELTY_KILLED_BY_MOIRA",
    "NO_METHOD_WORTHY_GAP",
}

FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
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
    "ALLOW_CLOUD_HANDOFF",
]

MAX_RUNTIME_SECONDS = 2 * 60 * 60


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _row_key(row: dict[str, Any]) -> tuple[int, int, int]:
    return (int(row["episode_index"]), int(row["frame_index"]), int(row["task_index"]))


def _paired_rows(base_rows: list[dict[str, Any]], lora_rows: list[dict[str, Any]]) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    lora_by_key = {_row_key(row): row for row in lora_rows}
    pairs = []
    for base in base_rows:
        key = _row_key(base)
        if key not in lora_by_key:
            raise KeyError(f"Missing LoRA row for {key}")
        pairs.append((base, lora_by_key[key]))
    return pairs


def _choose_row(base: dict[str, Any], lora: dict[str, Any], chooser: str) -> dict[str, Any]:
    if chooser == "base":
        chosen = dict(base)
        chosen["selected_expert"] = "frozen_base"
        return chosen
    if chooser == "lora":
        chosen = dict(lora)
        chosen["selected_expert"] = "rank4_lora"
        return chosen
    raise ValueError(chooser)


def frame_oracle_rows(base_rows: list[dict[str, Any]], lora_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for base, lora in _paired_rows(base_rows, lora_rows):
        rows.append(_choose_row(base, lora, "lora" if float(lora["action_l2"]) < float(base["action_l2"]) else "base"))
    return rows


def eval_loss_oracle_rows(base_rows: list[dict[str, Any]], lora_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for base, lora in _paired_rows(base_rows, lora_rows):
        rows.append(_choose_row(base, lora, "lora" if float(lora["eval_loss"]) < float(base["eval_loss"]) else "base"))
    return rows


def task_oracle_rows(base_rows: list[dict[str, Any]], lora_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, str]]:
    task_to_values: dict[int, dict[str, list[float]]] = {}
    for base, lora in _paired_rows(base_rows, lora_rows):
        task = int(base["task_index"])
        task_to_values.setdefault(task, {"base": [], "lora": []})
        task_to_values[task]["base"].append(float(base["action_l2"]))
        task_to_values[task]["lora"].append(float(lora["action_l2"]))
    routing = {}
    for task, values in task_to_values.items():
        routing[str(task)] = "rank4_lora" if float(np.mean(values["lora"])) < float(np.mean(values["base"])) else "frozen_base"
    rows = []
    for base, lora in _paired_rows(base_rows, lora_rows):
        rows.append(_choose_row(base, lora, "lora" if routing[str(base["task_index"])] == "rank4_lora" else "base"))
    return rows, routing


def action_dim_oracle_rows(base_rows: list[dict[str, Any]], lora_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for base, lora in _paired_rows(base_rows, lora_rows):
        per_dim = np.minimum(
            np.asarray(base["per_dim_abs"], dtype=np.float32),
            np.asarray(lora["per_dim_abs"], dtype=np.float32),
        )
        row = dict(base)
        row["selected_expert"] = "per_action_dimension_oracle"
        row["per_dim_abs"] = [round(float(x), 9) for x in per_dim.tolist()]
        row["action_l2"] = round(float(np.linalg.norm(per_dim)), 9)
        row["translation_l2"] = round(float(np.linalg.norm(per_dim[:3])), 9)
        row["rotation_l2"] = round(float(np.linalg.norm(per_dim[3:6])), 9)
        row["gripper_abs"] = round(float(per_dim[6]), 9)
        row["gripper_sign_match"] = bool(base["gripper_sign_match"] or lora["gripper_sign_match"])
        row["eval_loss"] = round(float(min(base["eval_loss"], lora["eval_loss"])), 9)
        row["range_violation_count"] = min(int(base["range_violation_count"]), int(lora["range_violation_count"]))
        rows.append(row)
    return rows


def _improvement(base_value: float, candidate_value: float) -> dict[str, float]:
    absolute = float(base_value) - float(candidate_value)
    relative = absolute / max(abs(float(base_value)), 1e-12)
    return {"absolute": round(absolute, 9), "relative": round(relative, 9)}


def _task_headroom(base_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_task: dict[int, dict[str, list[float]]] = {}
    candidate_by_key = {_row_key(row): row for row in candidate_rows}
    for base in base_rows:
        task = int(base["task_index"])
        by_task.setdefault(task, {"base": [], "candidate": []})
        by_task[task]["base"].append(float(base["action_l2"]))
        by_task[task]["candidate"].append(float(candidate_by_key[_row_key(base)]["action_l2"]))
    output = {}
    for task, values in sorted(by_task.items()):
        base_mean = float(np.mean(values["base"]))
        candidate_mean = float(np.mean(values["candidate"]))
        output[str(task)] = {
            "base_action_l2": round(base_mean, 9),
            "candidate_action_l2": round(candidate_mean, 9),
            "improvement": _improvement(base_mean, candidate_mean),
        }
    return output


def _selector_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        expert = str(row.get("selected_expert", "unknown"))
        counts[expert] = counts.get(expert, 0) + 1
    return counts


def choose_routing_decision(
    *,
    sample_count: int,
    frame_improvement_abs: float,
    frame_improvement_rel: float,
    task_improvement_abs: float,
    task_improvement_rel: float,
    moira_kills_task_only: bool,
) -> str:
    if sample_count < 100:
        return "NEED_MORE_OFFICIAL_FAILURE_MINING"
    frame_headroom = frame_improvement_abs >= 0.005 and frame_improvement_rel >= 0.05
    task_headroom = task_improvement_abs >= 0.005 and task_improvement_rel >= 0.05
    if not frame_headroom:
        return "NO_ROUTING_HEADROOM"
    if not task_headroom:
        return "GO_DESIGN_FRAME_CONDITIONAL_ROUTING"
    if moira_kills_task_only:
        return "ROUTING_NOVELTY_KILLED_BY_MOIRA"
    return "GO_DESIGN_TASK_ROUTING"


def _method_spec(final_decision: str) -> dict[str, Any] | None:
    if final_decision not in {"GO_DESIGN_FRAME_CONDITIONAL_ROUTING", "GO_DESIGN_BASE_RETENTIVE_ROUTING", "GO_DESIGN_TASK_ROUTING"}:
        return None
    router_level = "frame-level or hybrid" if final_decision == "GO_DESIGN_FRAME_CONDITIONAL_ROUTING" else "task-level plus base-retention guard"
    return {
        "method_name": "Frame-Conditional Adapter Retention",
        "problem_statement": "Low-data rank-4 LoRA on official SmolVLA-LIBERO creates negative transfer: some frames improve while others are worse than frozen/base.",
        "precise_failure_gap": "Routing oracle can avoid LoRA-hurt frames, but task-only routing has weak or MoIRA-covered headroom.",
        "model_components": ["frozen/base SmolVLA expert", "rank-4 LoRA adapted expert", "small router/gate", "retention regularizer"],
        "router_input_signals": ["instruction/task embedding", "current state", "visual embeddings if cached", "base-vs-LoRA action disagreement", "chunk/eval-loss proxy or uncertainty"],
        "router_level": router_level,
        "frozen_base_explicit_expert": True,
        "training_objective": "supervise gate to select LoRA only when it improves held-out action proxy while retaining base otherwise; no method training should start before a fixed protocol.",
        "retention_loss": "penalize selecting LoRA when frozen/base has lower action L2 or lower normalized chunk loss on diagnostic supervision.",
        "metrics": ["held-out action L2", "translation L2", "rotation L2", "gripper error/sign", "normalized chunk eval loss", "negative-transfer frame rate"],
        "required_baselines": ["frozen/base SmolVLA", "standard rank-4 LoRA", "mean-action prior", "task oracle", "frame oracle", "MoIRA-style instruction routing"],
        "ablations": ["no frozen/base expert", "instruction-only router", "frame-state-only router", "no retention loss", "weighted adapter merge/soup"],
        "first_experiment": "planning-only next; then, if approved, train a tiny gate on the same official diagnostic split with frozen/base fallback and compare to oracle bounds.",
        "kill_criteria": ["frame oracle below 5%/0.005 headroom", "gate fails to beat frozen/base", "MoIRA-style instruction router matches it", "mean-action or trivial prior explains gains"],
        "novelty_vs_moira": "must be frame/state/action-disagreement aware and base-retentive; instruction-to-adapter routing alone is killed by MoIRA.",
        "novelty_vs_standard_lora": "explicitly avoids LoRA negative-transfer frames instead of always applying the adapter.",
        "novelty_vs_aac": "not an action-chunk length scheduler; AAC is an adjacent temporal-stability baseline, not the routing mechanism.",
        "expected_ral_strength": "low-medium until official rollout exists",
        "expected_kill_risk": "high",
    }


def _write_lines(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def _table_row(values: list[Any]) -> str:
    return "| " + " | ".join(str(value) for value in values) + " |"


def _write_reports(report: dict[str, Any], output_dir: Path) -> None:
    oracle = report["oracle_bounds"]
    metrics = report["aggregate_metrics"]
    novelty = report["moira_comparison"]
    spec = report.get("method_spec")
    decision = report["final_decision"]
    headroom = report["headroom"]

    _write_lines(
        output_dir / "official_smolvla_routing_oracle_bound.md",
        [
            "# Official SmolVLA Routing Oracle Bound",
            "",
            f"Date: {report['date']}",
            "",
            "## Boundary",
            "",
            "- official SmolVLA-LIBERO evidence only",
            "- no method implementation",
            f"- standard rank-4 LoRA retrained only because saved adapter/per-frame rows were unavailable: `{report['regenerated_lora_for_oracle']}`",
            f"- held-out frames: `{report['dataset']['heldout_sample_count']}`",
            f"- task groups: `{report['dataset']['selected_task_count']}`",
            "",
            "## Aggregate Metrics",
            "",
            _table_row(["variant", "action L2", "eval loss", "translation L2", "rotation L2", "gripper abs", "gripper sign", "selected base/LoRA"]),
            _table_row(["---", "---:", "---:", "---:", "---:", "---:", "---:", "---"]),
            *[
                _table_row(
                    [
                        name,
                        item.get("action_l2_mean"),
                        item.get("eval_loss_mean"),
                        item.get("translation_l2_mean"),
                        item.get("rotation_l2_mean"),
                        item.get("gripper_abs_mean"),
                        item.get("gripper_sign_accuracy"),
                        oracle.get(name, {}).get("selector_counts"),
                    ]
                )
                for name, item in metrics.items()
            ],
            "",
            "## Headroom",
            "",
            f"- frame oracle improvement over frozen/base: `{headroom['frame_oracle_over_base']}`",
            f"- task oracle improvement over frozen/base: `{headroom['task_oracle_over_base']}`",
            f"- eval-loss oracle improvement over frozen/base: `{headroom['eval_loss_oracle_over_base']}`",
            f"- hard-gate threshold: at least `0.005` absolute and `5%` relative action-L2 improvement over frozen/base",
            "",
            "## Per-Task Oracle Gains",
            "",
            _table_row(["task", "base action L2", "task-oracle action L2", "abs gain", "rel gain"]),
            _table_row(["---:", "---:", "---:", "---:", "---:"]),
            *[
                _table_row(
                    [
                        task,
                        item["base_action_l2"],
                        item["candidate_action_l2"],
                        item["improvement"]["absolute"],
                        item["improvement"]["relative"],
                    ]
                )
                for task, item in oracle["task_oracle"]["per_task_headroom"].items()
            ],
            "",
            f"Routing upper-bound verdict: `{report['routing_headroom_verdict']}`",
        ],
    )

    _write_lines(
        output_dir / "official_smolvla_routing_vs_moira.md",
        [
            "# Official SmolVLA Routing vs MoIRA",
            "",
            f"Date: {report['date']}",
            "",
            "## MoIRA Comparison",
            "",
            f"- MoIRA does modular instruction routing: `{novelty['moira_modular_instruction_routing']}`",
            f"- MoIRA uses external text-based routing: `{novelty['moira_external_text_router']}`",
            f"- MoIRA uses low-rank adapter experts: `{novelty['moira_lora_experts']}`",
            f"- MoIRA evaluates on LIBERO: `{novelty['moira_libero_eval']}`",
            f"- instruction/task-only routing killed by MoIRA: `{novelty['task_or_instruction_only_killed']}`",
            "",
            "Sources:",
            "",
            "- MoIRA arXiv/html: https://arxiv.org/html/2507.01843v2",
            "- SmolVLA paper: https://arxiv.org/abs/2506.01844",
            "- SmolVLA blog: https://huggingface.co/blog/smolvla",
            "- AAC paper: https://arxiv.org/abs/2604.04161",
            "",
            "## Surviving Differentiator",
            "",
            novelty["surviving_differentiator"],
            "",
            "## Rejected Novelty Claims",
            "",
            *[f"- {item}" for item in novelty["rejected_novelty_claims"]],
            "",
            "## Required Comparison Baselines",
            "",
            *[f"- {item}" for item in novelty["required_comparison_baselines"]],
        ],
    )

    method_lines = [
        "# Official SmolVLA Routing Method Spec",
        "",
        f"Date: {report['date']}",
        "",
        f"Final decision: `{decision}`",
        "",
    ]
    if spec is None:
        method_lines.extend(["No method spec is recommended because the design gate did not reach a GO decision."])
    else:
        method_lines.extend(
            [
                f"Method name: `{spec['method_name']}`",
                "",
                f"Problem statement: {spec['problem_statement']}",
                "",
                f"Precise failure gap: {spec['precise_failure_gap']}",
                "",
                "Model components:",
                "",
                *[f"- {item}" for item in spec["model_components"]],
                "",
                "Router input signals:",
                "",
                *[f"- {item}" for item in spec["router_input_signals"]],
                "",
                f"Router level: `{spec['router_level']}`",
                f"Frozen/base explicit expert: `{spec['frozen_base_explicit_expert']}`",
                "",
                f"Training objective: {spec['training_objective']}",
                "",
                f"Retention loss: {spec['retention_loss']}",
                "",
                "Metrics:",
                "",
                *[f"- {item}" for item in spec["metrics"]],
                "",
                "Required baselines:",
                "",
                *[f"- {item}" for item in spec["required_baselines"]],
                "",
                "Ablations:",
                "",
                *[f"- {item}" for item in spec["ablations"]],
                "",
                f"Exact first experiment: {spec['first_experiment']}",
                "",
                "Kill criteria:",
                "",
                *[f"- {item}" for item in spec["kill_criteria"]],
                "",
                f"Novelty vs MoIRA: {spec['novelty_vs_moira']}",
                f"Novelty vs standard LoRA: {spec['novelty_vs_standard_lora']}",
                f"Novelty vs AAC: {spec['novelty_vs_aac']}",
                f"Expected RA-L strength: `{spec['expected_ral_strength']}`",
                f"Expected kill risk: `{spec['expected_kill_risk']}`",
            ]
        )
    _write_lines(output_dir / "official_smolvla_routing_method_spec.md", method_lines)

    _write_lines(
        output_dir / "official_smolvla_routing_kill_risks.md",
        [
            "# Official SmolVLA Routing Kill Risks",
            "",
            f"Date: {report['date']}",
            "",
            f"Estimated kill risk: `{report['estimated_kill_risk']}`",
            "",
            "Risks:",
            "",
            *[f"- {item}" for item in report["kill_risks"]],
        ],
    )

    _write_lines(
        output_dir / "official_smolvla_routing_next_decision.md",
        [
            "# Official SmolVLA Routing Next Decision",
            "",
            f"Date: {report['date']}",
            "",
            f"Final decision: `{decision}`",
            "",
            f"Reason: {report['decision_reason']}",
            "",
            f"Exact next prompt: {report.get('exact_next_prompt')}",
        ],
    )

    _write_lines(
        output_dir / "official_smolvla_routing_design_gate.md",
        [
            "# Official SmolVLA Routing Design Gate",
            "",
            f"Date: {report['date']}",
            "",
            f"Final decision: `{decision}`",
            "",
            "## Boundary",
            "",
            f"- experiments happened: `{report['policy']['experiments_performed']}`",
            f"- training happened: `{report['policy']['training_performed']}`",
            f"- loss computed: `{report['policy']['loss_computed']}`",
            f"- GPU/download/OpenVLA-OFT: `{report['policy']['gpu_used']}` / `{report['policy']['downloads_performed']}` / `{report['policy']['openvla_oft_executed']}`",
            f"- official dataset/model used: `{report['policy']['official_model_dataset_used']}`",
            f"- old custom route used: `{report['policy']['old_custom_route_used']}`",
            f"- method implemented: `{report['policy']['method_implemented']}`",
            "",
            "## Key Metrics",
            "",
            f"- frozen/base action L2: `{metrics['frozen_base']['action_l2_mean']}`",
            f"- rank-4 LoRA action L2: `{metrics['rank4_lora']['action_l2_mean']}`",
            f"- mean-action prior action L2: `{metrics['mean_action_prior']['action_l2_mean']}`",
            f"- frame oracle action L2: `{metrics['frame_oracle']['action_l2_mean']}`",
            f"- task oracle action L2: `{metrics['task_oracle']['action_l2_mean']}`",
            f"- action-dim oracle action L2: `{metrics['action_dim_oracle']['action_l2_mean']}`",
            "",
            "## Conclusion",
            "",
            report["decision_reason"],
            "",
            f"Exact next prompt: {report.get('exact_next_prompt')}",
        ],
    )


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    date = "2026-07-09 KST"
    os.environ["HF_HOME"] = str(Path(args.hf_home))
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    report: dict[str, Any] = {
        "date": date,
        "status": "started",
        "policy": {
            "experiments_performed": True,
            "training_performed": False,
            "loss_computed": False,
            "gpu_used": False,
            "downloads_performed": False,
            "openvla_oft_executed": False,
            "rollouts_performed": False,
            "official_model_dataset_used": True,
            "old_custom_route_used": False,
            "method_implemented": False,
            "paper_claims_made": False,
        },
        "paths": {
            "checkpoint": str(Path(args.checkpoint_path)),
            "dataset": str(Path(args.dataset_root)),
            "hf_home": str(Path(args.hf_home)),
            "vlm_root": str(Path(args.vlm_root)),
        },
        "regenerated_lora_for_oracle": True,
        "errors": [],
    }

    def fail(message: str, code: int) -> tuple[dict[str, Any], int]:
        report["status"] = "failed"
        report["final_decision"] = "NEED_MORE_OFFICIAL_FAILURE_MINING"
        report["errors"].append({"message": message})
        report["runtime"] = {"total_elapsed_sec": round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, code

    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    if forbidden:
        return fail("Forbidden gate(s) set: " + ", ".join(forbidden), 2)
    if not _env_flag("ALLOW_HEAVY_IMPORT") or not _env_flag("ALLOW_GPU_TRAINING"):
        return fail("Requires ALLOW_HEAVY_IMPORT=1 and ALLOW_GPU_TRAINING=1.", 3)
    if int(args.steps) > MAX_TRAINING_STEPS:
        return fail(f"Training steps exceed previous official cap: {args.steps}", 4)

    try:
        import pandas as pd
        import torch
        import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

        if not torch.cuda.is_available():
            return fail("CUDA unavailable; stopping instead of CPU fallback.", 5)
        report["policy"]["gpu_used"] = True
        torch.manual_seed(int(args.seed))
        np.random.seed(int(args.seed))
        torch.cuda.reset_peak_memory_stats()

        checkpoint_path = Path(args.checkpoint_path)
        dataset_root = Path(args.dataset_root)
        hf_home = Path(args.hf_home)
        vlm_root = Path(args.vlm_root)
        info = _read_json(dataset_root / "meta" / "info.json")
        stats = _read_json(dataset_root / "meta" / "stats.json")
        tasks_df = pd.read_parquet(dataset_root / "meta" / "tasks.parquet")
        fps = float(info.get("fps", 10.0))
        chunk_size = int(args.chunk_size)
        delta_timestamps = {"action": [i / fps for i in range(chunk_size)]}
        action_min = np.asarray(_stat_vector(stats, "action", "min"), dtype=np.float32)
        action_max = np.asarray(_stat_vector(stats, "action", "max"), dtype=np.float32)
        mean_action = np.asarray(_stat_vector(stats, "action", "mean"), dtype=np.float32)

        eval_plan = select_eval_plan(
            dataset_root=dataset_root,
            train_episode=0,
            max_eval_samples=int(args.max_eval_samples),
            min_task_groups=int(args.min_task_groups),
            episodes_per_task=int(args.episodes_per_task),
        )
        eval_dataset = LeRobotDataset(
            "lerobot/libero",
            root=dataset_root,
            episodes=eval_plan["selected_episodes"],
            delta_timestamps=delta_timestamps,
            video_backend=args.video_backend,
        )
        train_dataset = LeRobotDataset(
            "lerobot/libero",
            root=dataset_root,
            episodes=[0],
            delta_timestamps=delta_timestamps,
            video_backend=args.video_backend,
        )

        cfg = PreTrainedConfig.from_pretrained(checkpoint_path, local_files_only=True, cache_dir=hf_home)
        cfg.device = "cuda"
        cfg.load_vlm_weights = True
        cfg.compile_model = False
        cfg.push_to_hub = False
        cfg.vlm_model_name = str(vlm_root)
        cfg.chunk_size = chunk_size
        policy = SmolVLAPolicy.from_pretrained(
            checkpoint_path,
            config=cfg,
            local_files_only=True,
            cache_dir=hf_home,
            token=False,
            strict=False,
        )
        policy.to("cuda")
        policy.eval()
        if hasattr(policy, "reset"):
            policy.reset()
        preprocessor, postprocessor = make_pre_post_processors(
            cfg,
            pretrained_path=str(checkpoint_path),
            preprocessor_overrides={
                "tokenizer_processor": {"tokenizer_name": str(vlm_root)},
                "device_processor": {"device": "cuda"},
            },
            postprocessor_overrides={"device_processor": {"device": "cuda"}},
        )
        probe = _add_training_batch_dims(preprocessor(train_dataset[0]))
        input_devices = _tensor_devices(probe)
        param_summary = _parameter_summary(policy)
        if not str(param_summary["first_parameter_device"]).startswith("cuda") or not all(
            value.startswith("cuda") for value in input_devices.values()
        ):
            return fail(f"CPU fallback detected: params={param_summary}, inputs={input_devices}", 6)

        base_rows = evaluate_model_rows(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=eval_dataset,
            samples=eval_plan["samples"],
            action_min=action_min,
            action_max=action_max,
        )
        mean_rows = evaluate_mean_prior_rows(
            dataset=eval_dataset,
            samples=eval_plan["samples"],
            mean_action=mean_action,
            action_min=action_min,
            action_max=action_max,
        )

        policy.wrap_with_peft(peft_cli_overrides={"method_type": "LORA", "r": 4})
        policy.to("cuda")
        policy.train()
        lora_param_summary = _parameter_summary(policy)
        optimizer = torch.optim.AdamW([param for param in policy.parameters() if param.requires_grad], lr=float(args.lr))
        loss_curve = []
        grad_curve = []
        train_indices = [idx for idx in DEFAULT_TRAIN_INDICES if idx < len(train_dataset)]
        training_started = time.monotonic()
        report["policy"]["training_performed"] = True
        for step in range(int(args.steps)):
            if time.monotonic() - started > MAX_RUNTIME_SECONDS:
                return fail("Routing design gate exceeded hard runtime cap.", 7)
            sample = train_dataset[train_indices[step % len(train_indices)]]
            batch = _add_training_batch_dims(preprocessor(sample))
            if not all(value.startswith("cuda") for value in _tensor_devices(batch).values()):
                return fail("CPU fallback detected in training batch.", 8)
            optimizer.zero_grad(set_to_none=True)
            loss = _loss_from_output(policy.forward(batch))
            loss_value = _to_float(loss)
            report["policy"]["loss_computed"] = True
            if not math.isfinite(loss_value):
                return fail(f"Non-finite training loss at step {step}: {loss_value}", 9)
            loss.backward()
            grad_summary = _gradient_summary(policy)
            if grad_summary["nonzero_grad_tensors"] == 0:
                return fail(f"No nonzero gradients at step {step}.", 10)
            optimizer.step()
            loss_curve.append({"step": step, "loss": round(loss_value, 9), **_cuda_memory(torch)})
            grad_curve.append({"step": step, **grad_summary})
        training_elapsed = time.monotonic() - training_started

        lora_rows = evaluate_model_rows(
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset=eval_dataset,
            samples=eval_plan["samples"],
            action_min=action_min,
            action_max=action_max,
        )

        frame_rows = frame_oracle_rows(base_rows, lora_rows)
        task_rows, task_routing = task_oracle_rows(base_rows, lora_rows)
        instruction_rows = list(task_rows)
        eval_loss_rows = eval_loss_oracle_rows(base_rows, lora_rows)
        action_dim_rows = action_dim_oracle_rows(base_rows, lora_rows)

        aggregate = {
            "frozen_base": summarize_rows(base_rows),
            "rank4_lora": summarize_rows(lora_rows),
            "mean_action_prior": summarize_rows(mean_rows),
            "frame_oracle": summarize_rows(frame_rows),
            "task_oracle": summarize_rows(task_rows),
            "instruction_task_id_oracle": summarize_rows(instruction_rows),
            "eval_loss_oracle": summarize_rows(eval_loss_rows),
            "action_dim_oracle": summarize_rows(action_dim_rows),
        }
        oracle = {
            "frame_oracle": {"selector_counts": _selector_counts(frame_rows)},
            "task_oracle": {
                "selector_counts": _selector_counts(task_rows),
                "task_routing": task_routing,
                "per_task_headroom": _task_headroom(base_rows, task_rows),
            },
            "instruction_task_id_oracle": {"selector_counts": _selector_counts(instruction_rows), "equivalent_to": "task_oracle"},
            "eval_loss_oracle": {"selector_counts": _selector_counts(eval_loss_rows)},
            "action_dim_oracle": {"selector_counts": _selector_counts(action_dim_rows), "diagnostic_only": True},
        }
        base_l2 = float(aggregate["frozen_base"]["action_l2_mean"])
        frame_l2 = float(aggregate["frame_oracle"]["action_l2_mean"])
        task_l2 = float(aggregate["task_oracle"]["action_l2_mean"])
        eval_l2 = float(aggregate["eval_loss_oracle"]["action_l2_mean"])
        frame_improvement = _improvement(base_l2, frame_l2)
        task_improvement = _improvement(base_l2, task_l2)
        eval_improvement = _improvement(base_l2, eval_l2)
        headroom = {
            "frame_oracle_over_base": frame_improvement,
            "task_oracle_over_base": task_improvement,
            "eval_loss_oracle_over_base": eval_improvement,
            "frame_oracle_over_lora": _improvement(float(aggregate["rank4_lora"]["action_l2_mean"]), frame_l2),
            "task_oracle_over_lora": _improvement(float(aggregate["rank4_lora"]["action_l2_mean"]), task_l2),
        }
        moira_kills_task_only = True
        final_decision = choose_routing_decision(
            sample_count=int(aggregate["frozen_base"]["sample_count"]),
            frame_improvement_abs=frame_improvement["absolute"],
            frame_improvement_rel=frame_improvement["relative"],
            task_improvement_abs=task_improvement["absolute"],
            task_improvement_rel=task_improvement["relative"],
            moira_kills_task_only=moira_kills_task_only,
        )
        if final_decision == "GO_DESIGN_FRAME_CONDITIONAL_ROUTING":
            reason = (
                "Frame oracle clears the routing headroom gate, while task/instruction oracle headroom is tiny. "
                "A viable design therefore needs frame/state/action-disagreement signals and an explicit frozen/base fallback, not text-only task routing."
            )
            exact_next_prompt = (
                "Design a Frame-Conditional Adapter Retention method plan for official SmolVLA-LIBERO. "
                "Do not implement it; predeclare frozen/base, rank-4 LoRA, mean-action prior, frame oracle, task oracle, and MoIRA-style instruction router baselines."
            )
            verdict = "frame_headroom_only"
            risk = "high"
        elif final_decision == "NO_ROUTING_HEADROOM":
            reason = "Even frame oracle routing does not clear the 5% relative and 0.005 absolute action-L2 improvement gate over frozen/base."
            exact_next_prompt = None
            verdict = "no_routing_headroom"
            risk = "very high"
        elif final_decision == "ROUTING_NOVELTY_KILLED_BY_MOIRA":
            reason = "Task/instruction routing headroom exists but the surviving design is already covered by MoIRA-style instruction-to-adapter routing."
            exact_next_prompt = None
            verdict = "task_routing_killed_by_moira"
            risk = "very high"
        else:
            reason = f"Decision rule returned {final_decision}."
            exact_next_prompt = None
            verdict = "needs_followup"
            risk = "high"

        report.update(
            {
                "status": "completed",
                "final_decision": final_decision,
                "decision_reason": reason,
                "exact_next_prompt": exact_next_prompt,
                "routing_headroom_verdict": verdict,
                "estimated_kill_risk": risk,
                "dataset": {
                    "total_episodes": int(info.get("total_episodes", 0)),
                    "total_frames": int(info.get("total_frames", 0)),
                    "total_tasks": int(info.get("total_tasks", 0)),
                    "selected_task_count": len(eval_plan["selected_tasks"]),
                    "selected_tasks": eval_plan["selected_tasks"],
                    "selected_episode_count": len(eval_plan["selected_episodes"]),
                    "selected_episodes": eval_plan["selected_episodes"],
                    "heldout_sample_count": eval_plan["sample_count"],
                    "train_episode": 0,
                    "task_examples": [item["task"] for item in eval_plan["selected_tasks"][:5]],
                },
                "training": {
                    "variant": "standard_rank4_lora_recreated_for_oracle",
                    "steps": int(args.steps),
                    "batch_size": 1,
                    "trainable_params": lora_param_summary["trainable_params"],
                    "total_params": lora_param_summary["total_params"],
                    "loss_before": loss_curve[0]["loss"],
                    "loss_after": loss_curve[-1]["loss"],
                    "loss_curve": loss_curve,
                    "last_grad_norm": grad_curve[-1]["grad_norm"],
                    "last_nonzero_grad_tensors": grad_curve[-1]["nonzero_grad_tensors"],
                    "training_elapsed_sec": round(training_elapsed, 3),
                    "steps_per_sec": round(len(loss_curve) / training_elapsed, 6),
                    "autocast_status": _safe_autocast_status(torch),
                },
                "aggregate_metrics": aggregate,
                "oracle_bounds": oracle,
                "headroom": headroom,
                "moira_comparison": {
                    "moira_modular_instruction_routing": True,
                    "moira_external_text_router": True,
                    "moira_lora_experts": True,
                    "moira_libero_eval": True,
                    "task_or_instruction_only_killed": True,
                    "surviving_differentiator": "Base-retentive frame/state/action-disagreement-aware gating that can select frozen/base per frame to avoid negative transfer.",
                    "required_comparison_baselines": [
                        "frozen/base fallback",
                        "standard rank-4 LoRA",
                        "task-specific LoRA experts",
                        "task oracle",
                        "simple instruction embedding router",
                        "MoIRA-style text router",
                        "adapter soup / weighted LoRA merge",
                        "AAC as adjacent temporal-stability baseline",
                    ],
                    "rejected_novelty_claims": [
                        "use LoRA",
                        "route by task",
                        "route by instruction",
                        "use multiple adapters",
                        "external LLM router",
                        "generic MoE for robotics",
                    ],
                },
                "method_spec": _method_spec(final_decision),
                "kill_risks": [
                    "Frozen/base is already stronger than standard rank-4 LoRA on aggregate.",
                    "Task oracle may be too weak even when frame oracle is useful.",
                    "Instruction/task routing alone is killed by MoIRA-style routing.",
                    "A learned gate may fail to approach frame-oracle headroom.",
                    "Offline action L2 may not translate to simulator success without WSL/Linux LIBERO rollout.",
                    "Adapter soup or weighted LoRA merge may match a proposed router.",
                ],
                "runtime": {
                    "total_elapsed_sec": round(time.monotonic() - started, 3),
                    "rss_final_mb": _rss_mb(),
                    "cuda": {"available": True, "device_name": torch.cuda.get_device_name(0), **_cuda_memory(torch)},
                },
            }
        )
        _write_reports(report, Path(args.report_dir))
        return report, 0
    except Exception as exc:
        report["status"] = "failed"
        report["final_decision"] = "NEED_MORE_OFFICIAL_FAILURE_MINING"
        report["errors"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-16:],
            }
        )
        report["runtime"] = {"total_elapsed_sec": round(time.monotonic() - started, 3), "rss_final_mb": _rss_mb()}
        return report, 31


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-path", default=r"C:\assets\checkpoints\smolvla_libero")
    parser.add_argument("--dataset-root", default=r"C:\assets\datasets\lerobot_libero")
    parser.add_argument("--hf-home", default=r"C:\assets\hf_home")
    parser.add_argument("--vlm-root", default=r"C:\assets\hf_home\HuggingFaceTB\SmolVLM2-500M-Video-Instruct")
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--chunk-size", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-eval-samples", type=int, default=200)
    parser.add_argument("--min-task-groups", type=int, default=5)
    parser.add_argument("--episodes-per-task", type=int, default=2)
    parser.add_argument("--report-json", default="reports/official_smolvla_routing_design_gate.json")
    parser.add_argument("--report-dir", default="reports")
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    json_path = Path(args.report_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, sort_keys=True, default=_json_default) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
