"""ATCD offline teacher-signal audit for Epoch 5.

This script asks a narrow pre-training question for the second candidate after
CR-LightVLA:

    Do LightVLA and OpenVLA-OFT produce complementary HDF5 action proposals on
    the fixed task-8 validation chunks strongly enough to justify a later
    QLoRA complementarity-distillation training run?

It performs no training, no optimizer step, no checkpoint write, and no
simulator rollout.  Policy prediction is run one policy per process so the
OpenVLA-OFT and LightVLA source trees do not collide in Python's module cache.
"""

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

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.r2r_oft.offline_validate import select_fixed_validation_chunks  # noqa: E402
from tca_map.r2r_oft.train_qlora import _load_chunk, _make_instance, _write_json  # noqa: E402
from tca_map.r2r_oft.training_spec import SPEC_ARTIFACT, build_epoch5_training_spec, validate_training_spec  # noqa: E402


TASK8_HDF5 = Path("/mnt/c/assets/data/libero/libero_10/KITCHEN_SCENE8_put_both_moka_pots_on_the_stove_demo.hdf5")
OPENVLA_REPO = Path("/mnt/c/assets/repos/openvla-oft")
OPENVLA_CHECKPOINT = Path(
    "/home/jiheon/assets/checkpoints/openvla-oft/"
    "moojink_openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
)
LIGHTVLA_REPO = Path("/mnt/c/assets/repos/LightVLA")
LIGHTVLA_CHECKPOINT = Path("/home/jiheon/assets/checkpoints/lightvla/TTJiang_LightVLA-libero-10")


@dataclass(frozen=True)
class PredictConfig:
    policy: str
    repo: Path
    checkpoint: Path
    output: Path
    spec_path: Path = SPEC_ARTIFACT
    hdf5_path: Path = TASK8_HDF5
    num_chunks: int = 24
    device_index: int = 0


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_spec(path: Path) -> dict[str, Any]:
    spec = json.loads(path.read_text(encoding="utf-8")) if path.exists() else build_epoch5_training_spec()
    errors = validate_training_spec(spec)
    if errors:
        raise ValueError(f"invalid training spec: {'; '.join(errors)}")
    return spec


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    l1_values = np.asarray([row["l1"] for row in rows], dtype=np.float64)
    summary: dict[str, Any] = {
        "count": int(len(rows)),
        "mean_l1": float(np.mean(l1_values)),
        "median_l1": float(np.median(l1_values)),
        "min_l1": float(np.min(l1_values)),
        "max_l1": float(np.max(l1_values)),
    }
    for phase in (0, 1, 2):
        phase_values = np.asarray([row["l1"] for row in rows if int(row["phase_count_on"]) == phase], dtype=np.float64)
        summary[f"phase_{phase}_count"] = int(phase_values.size)
        summary[f"phase_{phase}_mean_l1"] = float(np.mean(phase_values)) if phase_values.size else None
    return summary


def _extract_action_hidden_states(
    *,
    last_hidden_states: Any,
    action_mask: Any,
    num_patches: int,
    num_action_tokens: int,
) -> tuple[Any, str, dict[str, Any]]:
    """Extract continuous-action hidden states across OpenVLA-OFT variants.

    OpenVLA-OFT's training/eval scripts index the text portion with the action
    token mask after slicing out the vision/proprio tokens.  LightVLA's exported
    HF model prunes vision tokens at inference time and its own ``predict_action``
    path reads the final action-token span instead.  This helper first tries the
    original OpenVLA-OFT path, then uses the model-runtime-compatible final-span
    path when the pruner has changed the hidden-state layout.
    """

    text_hidden_states = last_hidden_states[:, num_patches:-1]
    diagnostics = {
        "last_hidden_shape": list(last_hidden_states.shape),
        "text_hidden_shape": list(text_hidden_states.shape),
        "action_mask_shape": list(action_mask.shape),
        "action_mask_true_count": int(action_mask.sum().detach().cpu().item()),
        "num_patches_assumed": int(num_patches),
        "num_action_tokens": int(num_action_tokens),
    }
    if text_hidden_states.shape[1] == action_mask.shape[1] and diagnostics["action_mask_true_count"] == num_action_tokens:
        return text_hidden_states[action_mask], "text_mask_after_unpruned_patch_slice", diagnostics

    if last_hidden_states.shape[1] >= num_action_tokens:
        diagnostics["fallback_reason"] = "text-slice/action-mask layout mismatch after policy-specific token pruning"
        return last_hidden_states[:, -num_action_tokens:, :].reshape(num_action_tokens, -1), "final_action_token_span", diagnostics

    raise RuntimeError(f"cannot extract {num_action_tokens} action hidden states from shape {list(last_hidden_states.shape)}")


def predict_policy(config: PredictConfig) -> dict[str, Any]:
    started = time.monotonic()
    if not config.repo.exists():
        raise FileNotFoundError(config.repo)
    if not config.checkpoint.exists():
        raise FileNotFoundError(config.checkpoint)

    spec = _load_spec(config.spec_path)
    chunks = select_fixed_validation_chunks(hdf5_path=config.hdf5_path, spec=spec, num_chunks=config.num_chunks)

    # Force the selected policy source tree to the front.  Run one policy per
    # process; do not import both OpenVLA-OFT and LightVLA in one interpreter.
    repo_str = str(config.repo)
    if repo_str in sys.path:
        sys.path.remove(repo_str)
    sys.path.insert(0, repo_str)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    import torch
    from peft import prepare_model_for_kbit_training
    from prismatic.models.backbones.llm.prompting import PurePromptBuilder
    from prismatic.training.train_utils import get_current_action_mask, get_next_actions_mask
    from prismatic.util.data_utils import PaddedCollatorForActionPrediction
    from prismatic.vla.action_tokenizer import ActionTokenizer
    from prismatic.vla.constants import ACTION_DIM, NUM_ACTIONS_CHUNK
    from prismatic.vla.datasets import RLDSBatchTransform

    from experiments.robot.libero.run_libero_eval import GenerateConfig, check_unnorm_key
    from experiments.robot.openvla_utils import get_action_head, get_processor, get_proprio_projector, get_vla

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    torch.cuda.set_device(int(config.device_index))
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device=int(config.device_index))
    device = torch.device(f"cuda:{int(config.device_index)}")

    gen_cfg = GenerateConfig(
        pretrained_checkpoint=str(config.checkpoint),
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
    num_patches += 1  # proprio token

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
            action_mask = current_action_mask | next_actions_mask
            extracted_hidden_states, extraction_mode, extraction_diagnostics = _extract_action_hidden_states(
                last_hidden_states=last_hidden_states,
                action_mask=action_mask,
                num_patches=num_patches,
                num_action_tokens=NUM_ACTIONS_CHUNK * ACTION_DIM,
            )
            actions_hidden_states = (
                extracted_hidden_states
                .reshape(1, NUM_ACTIONS_CHUNK * ACTION_DIM, -1)
                .to(torch.bfloat16)
            )
            predicted_actions = action_head.predict_action(actions_hidden_states).detach().float().cpu().numpy()
            target_actions = batch["actions"].detach().float().cpu().numpy()
            rows.append(
                {
                    "policy": config.policy,
                    "validation_index": int(chunk["validation_index"]),
                    "demo_name": sample["demo_name"],
                    "demo_index": int(sample["demo_index"]),
                    "timestep": int(sample["timestep"]),
                    "phase_count_on": int(sample["phase_count_on"]),
                    "l1": float(np.mean(np.abs(predicted_actions - target_actions))),
                    "predicted_actions": predicted_actions.tolist(),
                    "target_actions": target_actions.tolist(),
                    "action_hidden_extraction": extraction_mode,
                    "action_hidden_extraction_diagnostics": extraction_diagnostics,
                    "normalized_action_min": float(np.min(normalized_actions)),
                    "normalized_action_max": float(np.max(normalized_actions)),
                }
            )

    runtime = {
        "elapsed_seconds": float(time.monotonic() - started),
        "cuda_peak_mib": float(torch.cuda.max_memory_allocated(device=device) / (1024 * 1024)),
    }
    report = {
        "schema_version": 1,
        "audit": "ATCD_policy_prediction_rows",
        "policy": config.policy,
        "repo": str(config.repo),
        "checkpoint": str(config.checkpoint),
        "run_libero_eval_source": str((config.repo / "experiments/robot/libero/run_libero_eval.py")),
        "spec_path": str(config.spec_path),
        "hdf5_path": str(config.hdf5_path),
        "num_chunks": int(config.num_chunks),
        "chunks": chunks,
        "summary": _summarize_rows(rows),
        "runtime": runtime,
        "rows": rows,
        "training_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "closed_loop_rollout_happened": False,
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(config.output, report)
    del vla, action_head, proprio_projector
    torch.cuda.empty_cache()
    gc.collect()
    return report


def compare_policy_rows(openvla_path: Path, lightvla_path: Path, output: Path) -> dict[str, Any]:
    openvla = json.loads(openvla_path.read_text(encoding="utf-8"))
    lightvla = json.loads(lightvla_path.read_text(encoding="utf-8"))
    open_rows = {int(row["validation_index"]): row for row in openvla["rows"]}
    light_rows = {int(row["validation_index"]): row for row in lightvla["rows"]}
    if set(open_rows) != set(light_rows):
        raise ValueError("policy row validation indices do not match")

    comparisons: list[dict[str, Any]] = []
    for index in sorted(open_rows):
        o = open_rows[index]
        l = light_rows[index]
        o_l1 = float(o["l1"])
        l_l1 = float(l["l1"])
        if abs(o_l1 - l_l1) <= 1e-8:
            winner = "tie"
        elif o_l1 < l_l1:
            winner = "openvla_oft_int4"
        else:
            winner = "lightvla"
        o_pred = np.asarray(o["predicted_actions"], dtype=np.float64)
        l_pred = np.asarray(l["predicted_actions"], dtype=np.float64)
        comparisons.append(
            {
                "validation_index": index,
                "demo_name": o["demo_name"],
                "demo_index": int(o["demo_index"]),
                "timestep": int(o["timestep"]),
                "phase_count_on": int(o["phase_count_on"]),
                "openvla_l1": o_l1,
                "lightvla_l1": l_l1,
                "oracle_l1": min(o_l1, l_l1),
                "winner": winner,
                "policy_mean_abs_delta": float(np.mean(np.abs(o_pred - l_pred))),
                "policy_max_abs_delta": float(np.max(np.abs(o_pred - l_pred))),
            }
        )

    def mean_for(key: str, rows: list[dict[str, Any]], phase: int | None = None) -> float | None:
        vals = [float(row[key]) for row in rows if phase is None or int(row["phase_count_on"]) == phase]
        return float(np.mean(vals)) if vals else None

    winner_counts = {
        "openvla_oft_int4": sum(1 for row in comparisons if row["winner"] == "openvla_oft_int4"),
        "lightvla": sum(1 for row in comparisons if row["winner"] == "lightvla"),
        "tie": sum(1 for row in comparisons if row["winner"] == "tie"),
    }
    open_mean = mean_for("openvla_l1", comparisons)
    light_mean = mean_for("lightvla_l1", comparisons)
    oracle_mean = mean_for("oracle_l1", comparisons)
    best_single = min(open_mean, light_mean)  # type: ignore[arg-type]
    absolute_gain = float(best_single - oracle_mean)  # type: ignore[operator]
    relative_gain = float(absolute_gain / max(best_single, 1e-8))

    phase_summary = {}
    for phase in (0, 1, 2):
        phase_summary[str(phase)] = {
            "count": sum(1 for row in comparisons if int(row["phase_count_on"]) == phase),
            "openvla_mean_l1": mean_for("openvla_l1", comparisons, phase),
            "lightvla_mean_l1": mean_for("lightvla_l1", comparisons, phase),
            "oracle_mean_l1": mean_for("oracle_l1", comparisons, phase),
            "openvla_wins": sum(
                1 for row in comparisons if int(row["phase_count_on"]) == phase and row["winner"] == "openvla_oft_int4"
            ),
            "lightvla_wins": sum(
                1 for row in comparisons if int(row["phase_count_on"]) == phase and row["winner"] == "lightvla"
            ),
        }

    phase1 = phase_summary["1"]
    phase1_best_single = min(
        phase1["openvla_mean_l1"] if phase1["openvla_mean_l1"] is not None else float("inf"),
        phase1["lightvla_mean_l1"] if phase1["lightvla_mean_l1"] is not None else float("inf"),
    )
    phase1_oracle = phase1["oracle_mean_l1"]
    phase1_abs_gain = float(phase1_best_single - phase1_oracle) if phase1_oracle is not None else 0.0

    pass_signal = bool(
        winner_counts["openvla_oft_int4"] >= 3
        and winner_counts["lightvla"] >= 3
        and absolute_gain >= 0.01
        and relative_gain >= 0.03
        and phase1["count"] >= 6
        and phase1_abs_gain >= 0.01
    )
    decision = "ATCD_TEACHER_SIGNAL_PASS" if pass_signal else "ATCD_TEACHER_SIGNAL_NOT_ENOUGH"

    report = {
        "schema_version": 1,
        "audit": "ATCD_teacher_signal",
        "decision": decision,
        "teacher_signal_pass": pass_signal,
        "openvla_rows": str(openvla_path),
        "lightvla_rows": str(lightvla_path),
        "num_chunks": len(comparisons),
        "winner_counts": winner_counts,
        "openvla_mean_l1": open_mean,
        "lightvla_mean_l1": light_mean,
        "oracle_mean_l1": oracle_mean,
        "oracle_absolute_gain_vs_best_single": absolute_gain,
        "oracle_relative_gain_vs_best_single": relative_gain,
        "phase_summary": phase_summary,
        "phase1_oracle_absolute_gain_vs_best_single": phase1_abs_gain,
        "mean_policy_abs_delta": mean_for("policy_mean_abs_delta", comparisons),
        "max_policy_abs_delta": max(float(row["policy_max_abs_delta"]) for row in comparisons),
        "comparisons": comparisons,
        "training_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "closed_loop_rollout_happened": False,
        "interpretation": (
            "A pass means there is offline per-chunk complementarity worth considering for a later bounded QLoRA "
            "distillation run; it is not a rollout result or prototype GO."
        ),
        "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    _write_json(output, report)
    return report


def _policy_defaults(policy: str) -> tuple[Path, Path]:
    if policy == "openvla_oft_int4":
        return OPENVLA_REPO, OPENVLA_CHECKPOINT
    if policy == "lightvla":
        return LIGHTVLA_REPO, LIGHTVLA_CHECKPOINT
    raise ValueError(f"unknown policy {policy!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    predict = subparsers.add_parser("predict")
    predict.add_argument("--policy", choices=["openvla_oft_int4", "lightvla"], required=True)
    predict.add_argument("--repo", type=Path)
    predict.add_argument("--checkpoint", type=Path)
    predict.add_argument("--output", type=Path, required=True)
    predict.add_argument("--spec", type=Path, default=SPEC_ARTIFACT)
    predict.add_argument("--hdf5-path", type=Path, default=TASK8_HDF5)
    predict.add_argument("--num-chunks", type=int, default=24)
    predict.add_argument("--device-index", type=int, default=0)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--openvla-rows", type=Path, required=True)
    compare.add_argument("--lightvla-rows", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)

    args = parser.parse_args()
    if args.command == "predict":
        default_repo, default_checkpoint = _policy_defaults(args.policy)
        report = predict_policy(
            PredictConfig(
                policy=args.policy,
                repo=args.repo or default_repo,
                checkpoint=args.checkpoint or default_checkpoint,
                output=args.output,
                spec_path=args.spec,
                hdf5_path=args.hdf5_path,
                num_chunks=int(args.num_chunks),
                device_index=int(args.device_index),
            )
        )
    else:
        report = compare_policy_rows(args.openvla_rows, args.lightvla_rows, args.output)
    print(json.dumps(report, indent=2, sort_keys=True, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
