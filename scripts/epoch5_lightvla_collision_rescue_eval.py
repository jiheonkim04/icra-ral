"""Run a bounded Collision-Rescue LightVLA diagnostic on LIBERO task 8.

This is an Epoch 5 official-prior-first method diagnostic. It keeps the
LightVLA model/checkpoint fixed and changes only the inference token-selection
rule:

    original LightVLA: keep each unique first-choice token selected by the
    instruction-conditioned dynamic queries.

    collision rescue: keep those first-choice tokens and, only for queries whose
    first-choice token is selected by more than one query, also keep that query's
    second-choice token.

The rule is fixed before rollout. It has no trainable parameters, no learned
policy selector, and no retuning on residual reset identities.
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
import time
import traceback
import types
from pathlib import Path
from typing import Any

LIGHTVLA_ROOT = "/mnt/c/assets/repos/LightVLA"
if LIGHTVLA_ROOT in sys.path:
    sys.path.remove(LIGHTVLA_ROOT)
sys.path.insert(0, LIGHTVLA_ROOT)

CHECKPOINT = "/home/jiheon/assets/checkpoints/lightvla/TTJiang_LightVLA-libero-10"
IDENTITIES = list(range(20260716, 20260724))
INITIAL_STATE_INDICES = {identity: index for identity, index in zip(IDENTITIES, range(5, 13))}

TELEMETRY: dict[int, list[dict[str, Any]]] = {}
CURRENT_IDENTITY: int | None = None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def beat(path: Path, stage: str) -> None:
    path.write_text(f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {stage}\n", encoding="utf-8")


def summarize_telemetry(identity: int) -> dict[str, Any]:
    rows = TELEMETRY.get(identity, [])
    if not rows:
        return {"calls": 0}
    keys = [
        "first_unique_count",
        "collision_query_count",
        "rescued_second_choice_count",
        "final_unique_count",
        "num_patches",
    ]
    summary: dict[str, Any] = {"calls": len(rows)}
    for key in keys:
        vals = [float(row[key]) for row in rows if key in row]
        if vals:
            summary[f"{key}_mean"] = round(sum(vals) / len(vals), 3)
            summary[f"{key}_min"] = int(min(vals))
            summary[f"{key}_max"] = int(max(vals))
    return summary


def collision_rescue_score_to_mask(self, score):  # noqa: ANN001
    import torch

    bsz, query_count, num_patches = score.shape
    if num_patches != self.num_patches:
        raise RuntimeError(f"score num_patches {num_patches} != pruner num_patches {self.num_patches}")

    k = 2 if num_patches >= 2 else 1
    topk = torch.topk(score, k=k, dim=-1).indices
    first = topk[..., 0]
    second = topk[..., 1] if k == 2 else topk[..., 0]

    mask = torch.zeros(bsz, num_patches, dtype=torch.bool, device=score.device)
    batch = torch.arange(bsz, device=score.device).unsqueeze(1).expand_as(first)
    mask[batch, first] = True

    counts = torch.zeros(bsz, num_patches, dtype=torch.long, device=score.device)
    counts.scatter_add_(1, first, torch.ones_like(first, dtype=torch.long))
    collision = counts.gather(1, first) > 1

    rescued_before = mask.clone()
    if collision.any():
        rescue_batch = batch[collision]
        rescue_second = second[collision]
        mask[rescue_batch, rescue_second] = True

    if CURRENT_IDENTITY is not None:
        first_unique = rescued_before.sum(dim=1).detach().cpu().tolist()
        final_unique = mask.sum(dim=1).detach().cpu().tolist()
        collision_queries = collision.sum(dim=1).detach().cpu().tolist()
        rescued = (mask & ~rescued_before).sum(dim=1).detach().cpu().tolist()
        for i in range(bsz):
            TELEMETRY.setdefault(CURRENT_IDENTITY, []).append(
                {
                    "first_unique_count": int(first_unique[i]),
                    "collision_query_count": int(collision_queries[i]),
                    "rescued_second_choice_count": int(rescued[i]),
                    "final_unique_count": int(final_unique[i]),
                    "num_patches": int(num_patches),
                    "query_count": int(query_count),
                }
            )

    return mask


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        default="/mnt/c/Users/jiheo/tca_map/runs/lightvla_prior/cr_lightvla_task8_all_20260717T1600KST",
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    partial = run_dir / "partial.json"
    result_path = run_dir / "result.json"
    heartbeat = run_dir / "heartbeat.txt"

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("LIBERO_CONFIG_PATH", "/home/jiheon/.libero")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    report: dict[str, Any] = {
        "method": "CR-LightVLA",
        "full_name": "Collision-Rescue LightVLA token pruning",
        "policy": "LightVLA-libero-10-4bit-with-collision-rescue-token-selection",
        "checkpoint": CHECKPOINT,
        "source_root_forced": LIGHTVLA_ROOT,
        "suite": "libero_10",
        "task_id": 8,
        "task_description": "put both moka pots on the stove",
        "reset_identities": IDENTITIES,
        "initial_state_indices": INITIAL_STATE_INDICES,
        "rule_fixed_before_rollout": True,
        "rule": (
            "Keep original LightVLA first-choice unique tokens; for query collisions, "
            "also keep each collided query's second-choice token."
        ),
        "trainable_parameters_added": 0,
        "training_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "ours_design_happened": True,
        "closed_loop_rollout_happened": True,
        "scheduler_bypassed_reason": (
            "official LightVLA run_task waits for >20000 MB free VRAM, impossible on local 16GB RTX 5080"
        ),
        "official_functions": ["GenerateConfig", "initialize_model", "get_libero_env", "run_episode"],
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "episodes": [],
        "errors": [],
    }

    try:
        beat(heartbeat, "import")
        import torch
        from libero.libero import benchmark
        from experiments.robot.libero.libero_utils import get_libero_env
        import experiments.robot.libero.run_libero_eval as run_libero_eval
        from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model, run_episode
        from experiments.robot.robot_utils import get_image_resize_size, set_seed_everywhere

        report["torch"] = getattr(torch, "__version__", None)
        report["run_libero_eval_file"] = getattr(run_libero_eval, "__file__", None)
        report["initialize_model_signature"] = str(inspect.signature(initialize_model))
        report["run_episode_signature"] = str(inspect.signature(run_episode))
        report["cuda_available"] = torch.cuda.is_available()
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        report["cuda_name"] = torch.cuda.get_device_name(0)
        report["cuda_mem_before"] = list(torch.cuda.mem_get_info(0))

        cfg = GenerateConfig()
        cfg.pretrained_checkpoint = CHECKPOINT
        cfg.task_suite_name = "libero_10"
        cfg.num_trials_per_task = 1
        cfg.load_in_4bit = True
        cfg.load_in_8bit = False
        cfg.center_crop = True
        cfg.seed = 20260716
        cfg.local_log_dir = str(run_dir / "val_logs")
        cfg.save_rollout_video = False

        set_seed_everywhere(cfg.seed)
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        beat(heartbeat, "load_model")
        load_started = time.time()
        model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg, device)
        report["load_elapsed_s"] = round(time.time() - load_started, 3)
        pruner = model.language_model.model.pruner
        report["original_score_to_mask"] = repr(pruner.score_to_mask)
        pruner.score_to_mask = types.MethodType(collision_rescue_score_to_mask, pruner)
        report["patched_pruner_class"] = type(pruner).__name__
        report["patched_score_to_mask"] = "collision_rescue_score_to_mask"

        resize_size = get_image_resize_size(cfg)
        task_suite = benchmark.get_benchmark_dict()[cfg.task_suite_name]()
        task = task_suite.get_task(8)

        global CURRENT_IDENTITY
        for identity in IDENTITIES:
            index = INITIAL_STATE_INDICES[identity]
            row: dict[str, Any] = {
                "reset_identity": identity,
                "initial_state_index": index,
                "success": False,
                "completed": False,
            }
            env = None
            try:
                beat(heartbeat, f"episode_{identity}")
                CURRENT_IDENTITY = identity
                cfg.seed = identity
                set_seed_everywhere(cfg.seed)
                initial_state = task_suite.get_task_init_states(8)[index]
                row["initial_state_shape"] = list(getattr(initial_state, "shape", []))
                env, task_description = get_libero_env(task, cfg.model_family, resolution=cfg.env_img_res)
                row["task_description"] = task_description
                episode_started = time.time()
                success, replay_images = run_episode(
                    cfg,
                    env,
                    task_description,
                    model,
                    resize_size,
                    processor=processor,
                    action_head=action_head,
                    proprio_projector=proprio_projector,
                    noisy_action_projector=noisy_action_projector,
                    initial_state=initial_state,
                    device=device,
                )
                row["success"] = bool(success)
                row["completed"] = True
                row["episode_elapsed_s"] = round(time.time() - episode_started, 3)
                row["replay_frame_count"] = len(replay_images)
                row["pruner_telemetry"] = summarize_telemetry(identity)
            except Exception as exc:  # pragma: no cover - runtime artifact path
                row["completed"] = False
                row["error_type"] = type(exc).__name__
                row["error"] = str(exc)
                row["traceback_tail"] = traceback.format_exc()[-3000:]
                report["errors"].append({"reset_identity": identity, "error": row["error"]})
            finally:
                CURRENT_IDENTITY = None
                if env is not None:
                    try:
                        env.close()
                    except Exception:
                        pass
            report["episodes"].append(row)
            write_json(partial, report)

        report["completed_episode_count"] = sum(1 for row in report["episodes"] if row.get("completed"))
        report["successes"] = sum(1 for row in report["episodes"] if row.get("success"))
        report["failures"] = [
            row["reset_identity"] for row in report["episodes"] if row.get("completed") and not row.get("success")
        ]
        report["cuda_max_allocated"] = int(torch.cuda.max_memory_allocated(0))
        report["cuda_mem_after"] = list(torch.cuda.mem_get_info(0))
        del model, action_head, proprio_projector, noisy_action_projector, processor
        gc.collect()
        torch.cuda.empty_cache()
        report["cuda_mem_after_cleanup"] = list(torch.cuda.mem_get_info(0))
        report["completed"] = True
    except Exception as exc:  # pragma: no cover - runtime artifact path
        report["completed"] = False
        report["error_type"] = type(exc).__name__
        report["error"] = str(exc)
        report["traceback_tail"] = traceback.format_exc()[-5000:]
    finally:
        report["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        beat(heartbeat, "finished")
        write_json(result_path, report)
        print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("completed") else 1


if __name__ == "__main__":
    sys.exit(main())
