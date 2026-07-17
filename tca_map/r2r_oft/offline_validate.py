"""Offline validation gate for frozen R2R-OFT training arms."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.r2r_oft.train_qlora import (
    _git_commit,
    _json_default,
    _load_chunk,
    _make_instance,
    _write_json,
    build_phase_chunk_index,
    select_chunk_for_step,
)
from tca_map.r2r_oft.training_spec import SPEC_ARTIFACT, build_epoch5_training_spec, validate_training_spec


@dataclass(frozen=True)
class OfflineValidationConfig:
    spec_path: Path
    output_path: Path
    openvla_repo: Path
    checkpoint_dir: Path
    hdf5_path: Path
    primary_adapter_dir: Path
    ablation_adapter_dir: Path
    num_chunks: int = 24
    device_index: int = 0


def _load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else build_epoch5_training_spec()
    errors = validate_training_spec(spec)
    if errors:
        raise ValueError(f"invalid training spec: {'; '.join(errors)}")
    return spec


def select_fixed_validation_chunks(
    *,
    hdf5_path: Path,
    spec: dict[str, Any],
    num_chunks: int,
) -> list[dict[str, Any]]:
    """Select fixed validation chunks using the frozen phase cycle."""

    shared = spec["shared_training"]
    grouped = build_phase_chunk_index(
        hdf5_path,
        demo_indices=list(spec["data"]["validation_demo_indices"]),
        chunk_size=int(shared["action_chunk_size"]),
        train_demo_count_for_target_xy=len(spec["data"]["train_demo_indices"]),
    )
    rng = np.random.default_rng(int(shared["seed"]) + 1)
    cycle = list(spec["arms"][0]["sampler"]["cycle_phase_counts"])
    chunks = [
        select_chunk_for_step(grouped, cycle=cycle, step_index_zero_based=step, rng=rng)
        for step in range(int(num_chunks))
    ]
    for index, chunk in enumerate(chunks):
        chunk["validation_index"] = index
    return chunks


def _summarize_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    l1_values = np.asarray([row["l1"] for row in rows], dtype=np.float64)
    summary: dict[str, Any] = {
        "count": int(len(rows)),
        "mean_l1": float(np.mean(l1_values)),
        "median_l1": float(np.median(l1_values)),
    }
    for phase in (0, 1, 2):
        phase_values = np.asarray([row["l1"] for row in rows if int(row["phase_count_on"]) == phase], dtype=np.float64)
        summary[f"phase_{phase}_count"] = int(phase_values.size)
        summary[f"phase_{phase}_mean_l1"] = float(np.mean(phase_values)) if phase_values.size else None
    return summary


def _summarize_delta(rows: list[dict[str, Any]], prior_rows: list[dict[str, Any]]) -> dict[str, Any]:
    prior_by_index = {int(row["validation_index"]): np.asarray(row["predicted_actions"], dtype=np.float64) for row in prior_rows}
    mean_abs = []
    max_abs = []
    for row in rows:
        pred = np.asarray(row["predicted_actions"], dtype=np.float64)
        prior = prior_by_index[int(row["validation_index"])]
        delta = np.abs(pred - prior)
        mean_abs.append(float(np.mean(delta)))
        max_abs.append(float(np.max(delta)))
    return {
        "fixed_chunk_mean_abs_action_delta": float(np.mean(mean_abs)),
        "fixed_chunk_max_abs_action_delta": float(np.max(max_abs)),
    }


def _predict_rows(
    *,
    label: str,
    adapter_dir: Path | None,
    chunks: list[dict[str, Any]],
    config: OfflineValidationConfig,
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    started = time.monotonic()
    sys.path.insert(0, str(config.openvla_repo))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")

    import torch
    from peft import PeftModel, prepare_model_for_kbit_training
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
    from prismatic.vla.datasets import RLDSBatchTransform

    from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key
    from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla

    torch.cuda.set_device(int(config.device_index))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=int(config.device_index))
    device = torch.device(f"cuda:{int(config.device_index)}")

    gen_cfg = GenerateConfig(
        pretrained_checkpoint=str(config.checkpoint_dir),
        use_l1_regression=True,
        use_diffusion=False,
        use_film=False,
        num_images_in_input=2,
        use_proprio=True,
        center_crop=True,
        num_open_loop_steps=NUM_ACTIONS_CHUNK,
        load_in_4bit=True,
        load_in_8bit=False,
        task_suite_name="libero_10",
        num_trials_per_task=1,
    )
    vla = get_vla(gen_cfg)
    vla = prepare_model_for_kbit_training(vla, use_gradient_checkpointing=False)
    if adapter_dir is not None:
        vla = PeftModel.from_pretrained(vla, str(adapter_dir))
    check_unnorm_key(gen_cfg, vla)
    processor = get_processor(gen_cfg)
    action_head = get_action_head(gen_cfg, vla.llm_dim).to(device)
    proprio_projector = get_proprio_projector(gen_cfg, vla.llm_dim, proprio_dim=8).to(device)
    vla.eval()
    action_head.eval()
    proprio_projector.eval()

    action_stats = vla.norm_stats[gen_cfg.unnorm_key]["action"]
    proprio_stats = vla.norm_stats[gen_cfg.unnorm_key]["proprio"]
    action_tokenizer = ActionTokenizer(processor.tokenizer)
    batch_transform = RLDSBatchTransform(
        action_tokenizer,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        prompt_builder_fn=PurePromptBuilder,
        use_wrist_image=True,
        use_proprio=True,
    )
    collator = PaddedCollatorForActionPrediction(
        processor.tokenizer.model_max_length,
        processor.tokenizer.pad_token_id,
        padding_side="right",
    )
    num_patches = vla.vision_backbone.get_num_patches() * vla.vision_backbone.get_num_images_in_input()
    num_patches += 1

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for chunk in chunks:
            sample = _load_chunk(config.hdf5_path, chunk, int(spec["shared_training"]["action_chunk_size"]))
            instance, normalized_actions = _make_instance(
                sample=sample,
                action_stats=action_stats,
                proprio_stats=proprio_stats,
                batch_transform=batch_transform,
            )
            batch = collator([instance])
            labels_for_mask = batch["labels"][:, 1:].to(device)
            current_action_mask = get_current_action_mask(labels_for_mask)
            next_actions_mask = get_next_actions_mask(labels_for_mask)
            output = vla(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                pixel_values=batch["pixel_values"].to(device),
                labels=batch["labels"].to(device),
                output_hidden_states=True,
                proprio=batch["proprio"].to(torch.bfloat16).to(device),
                proprio_projector=proprio_projector,
                use_film=False,
            )
            last_hidden_states = output.hidden_states[-1]
            text_hidden_states = last_hidden_states[:, num_patches:-1]
            actions_hidden_states = (
                text_hidden_states[current_action_mask | next_actions_mask]
                .reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
                .to(torch.bfloat16)
            )
            predicted_actions = action_head.predict_action(actions_hidden_states).detach().float().cpu().numpy()
            target_actions = batch["actions"].detach().float().cpu().numpy()
            rows.append(
                {
                    "policy": label,
                    "validation_index": int(chunk["validation_index"]),
                    "demo_name": sample["demo_name"],
                    "timestep": int(sample["timestep"]),
                    "phase_count_on": int(sample["phase_count_on"]),
                    "l1": float(np.mean(np.abs(predicted_actions - target_actions))),
                    "predicted_actions": predicted_actions.tolist(),
                    "target_actions": target_actions.tolist(),
                    "normalized_action_min": float(np.min(normalized_actions)),
                    "normalized_action_max": float(np.max(normalized_actions)),
                }
            )

    runtime = {
        "elapsed_seconds": float(time.monotonic() - started),
        "cuda_peak_mib": float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)),
    }
    del vla, action_head, proprio_projector
    torch.cuda.empty_cache()
    gc.collect()
    return rows, runtime


def run_offline_validation(config: OfflineValidationConfig) -> dict[str, Any]:
    started = time.monotonic()
    spec = _load_spec(config.spec_path)
    chunks = select_fixed_validation_chunks(hdf5_path=config.hdf5_path, spec=spec, num_chunks=config.num_chunks)
    primary_step_label = config.primary_adapter_dir.parent.name
    ablation_step_label = config.ablation_adapter_dir.parent.name
    primary_label = f"r2r_primary_{primary_step_label}"
    ablation_label = f"uniform_ablation_{ablation_step_label}"
    policies = [
        ("prior_base", None),
        (primary_label, config.primary_adapter_dir),
        (ablation_label, config.ablation_adapter_dir),
    ]
    all_rows: dict[str, list[dict[str, Any]]] = {}
    runtimes: dict[str, dict[str, Any]] = {}
    for label, adapter in policies:
        rows, runtime = _predict_rows(label=label, adapter_dir=adapter, chunks=chunks, config=config, spec=spec)
        all_rows[label] = rows
        runtimes[label] = runtime

    summaries = {label: _summarize_predictions(rows) for label, rows in all_rows.items()}
    for label in (primary_label, ablation_label):
        summaries[label]["delta_vs_prior"] = _summarize_delta(all_rows[label], all_rows["prior_base"])

    prior = summaries["prior_base"]
    primary = summaries[primary_label]
    ablation = summaries[ablation_label]
    criteria = spec["validation_selection"]["offline_pass_criteria"]
    primary_clean = np.mean([primary["phase_0_mean_l1"], primary["phase_2_mean_l1"]])
    prior_clean = np.mean([prior["phase_0_mean_l1"], prior["phase_2_mean_l1"]])
    clean_degradation = float((primary_clean - prior_clean) / max(prior_clean, 1e-8))
    primary_delta = primary["delta_vs_prior"]
    primary_pass = bool(
        np.isfinite(primary["mean_l1"])
        and primary["phase_1_mean_l1"] <= prior["phase_1_mean_l1"]
        and clean_degradation <= float(criteria["clean_phase_validation_l1_relative_degradation_max"])
        and primary_delta["fixed_chunk_mean_abs_action_delta"]
        <= float(criteria["fixed_chunk_mean_abs_action_delta_max"])
        and primary_delta["fixed_chunk_max_abs_action_delta"] <= float(criteria["fixed_chunk_max_abs_action_delta_max"])
        and max(runtime["cuda_peak_mib"] for runtime in runtimes.values()) <= float(criteria["cuda_peak_mib_max"])
    )
    primary_beats_ablation = bool(primary["phase_1_mean_l1"] < ablation["phase_1_mean_l1"])
    decision = "PRIMARY_OFFLINE_PASS_BEATS_ABLATION" if primary_pass and primary_beats_ablation else "OFFLINE_SELECTION_NOT_PASSED"

    report = {
        "schema_version": 1,
        "method": "R2R-OFT",
        "status": "COMPLETE",
        "decision": decision,
        "success": bool(primary_pass and primary_beats_ablation),
        "git_commit": _git_commit(),
        "spec_path": str(config.spec_path),
        "num_chunks": int(config.num_chunks),
        "primary_step_label": primary_step_label,
        "ablation_step_label": ablation_step_label,
        "primary_summary_key": primary_label,
        "ablation_summary_key": ablation_label,
        "validation_phase_counts": {
            str(phase): int(sum(1 for chunk in chunks if int(chunk["phase_count_on"]) == phase)) for phase in (0, 1, 2)
        },
        "chunks": chunks,
        "summaries": summaries,
        "runtimes": runtimes,
        "primary_clean_phase_relative_degradation_vs_prior": clean_degradation,
        "primary_passes_offline_gate": primary_pass,
        "primary_beats_uniform_ablation_on_phase1_l1": primary_beats_ablation,
        "closed_loop_evaluation_happened": False,
        "elapsed_seconds": float(time.monotonic() - started),
    }
    _write_json(config.output_path, report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=SPEC_ARTIFACT)
    parser.add_argument("--output", type=Path, default=Path("runs/openvla_oft_int4/epoch5_r2r_oft_offline_validation.json"))
    parser.add_argument("--openvla-repo", type=Path, default=Path("/mnt/c/assets/repos/openvla-oft"))
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "/home/jiheon/assets/checkpoints/openvla-oft/"
            "moojink_openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
        ),
    )
    parser.add_argument(
        "--hdf5-path",
        type=Path,
        default=Path("/mnt/c/assets/data/libero/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5"),
    )
    parser.add_argument(
        "--primary-adapter-dir",
        type=Path,
        default=Path("runs/openvla_oft_int4/epoch5_r2r_oft_training/r2r_oft_rank4_lambda2_lr2e4_steps64/checkpoints/step_0064/adapter"),
    )
    parser.add_argument(
        "--ablation-adapter-dir",
        type=Path,
        default=Path("runs/openvla_oft_int4/epoch5_r2r_oft_training/uniform_oft_rank4_lambda0_lr2e4_steps64/checkpoints/step_0064/adapter"),
    )
    parser.add_argument("--num-chunks", type=int, default=24)
    parser.add_argument("--device-index", type=int, default=0)
    args = parser.parse_args()
    report = run_offline_validation(
        OfflineValidationConfig(
            spec_path=args.spec,
            output_path=args.output,
            openvla_repo=args.openvla_repo,
            checkpoint_dir=args.checkpoint_dir,
            hdf5_path=args.hdf5_path,
            primary_adapter_dir=args.primary_adapter_dir,
            ablation_adapter_dir=args.ablation_adapter_dir,
            num_chunks=int(args.num_chunks),
            device_index=int(args.device_index),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return 0 if report.get("status") == "COMPLETE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
