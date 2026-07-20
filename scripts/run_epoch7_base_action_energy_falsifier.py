#!/usr/bin/env python3
"""Run the frozen X-VLA Base action-energy falsifier.

This is real CUDA VLA inference over official X-VLA-format demonstration
chunks.  It performs no training, simulator construction, outcome read, or
Ours execution.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import h5py
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.epoch7_selective_language_grounding import atomic_write_json, load_json  # noqa: E402
from tca_map.xvla_task1.train_lora import _prepare_xvla_imports  # noqa: E402


DEFAULT_PROTOCOL = REPO_ROOT / "reports/epoch7_selective_language_grounding/method_stage0_base_energy_protocol.json"
DEFAULT_MANIFEST = REPO_ROOT / "reports/epoch7_selective_language_grounding/method_partition_manifest.json"
DEFAULT_DATA_ROOT = Path("/mnt/c/assets/datasets/Libero-XVLA-format/libero_goal")
DEFAULT_XVLA_ROOT = Path("/mnt/c/assets/repos/X-VLA")
MODEL_ID = "2toINF/X-VLA-Libero"
MODEL_REVISION = "129e71460678b7236cee6fc9707f09d9fa0c3590"
XVLA_CACHE_DIR = "/home/jiheon/assets/checkpoints/xvla_hf_cache/transformers"
FAMILY_ORDER = ("act", "obj", "comp")


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes())
    return digest.hexdigest()


def memory_snapshot(torch_module: Any | None = None) -> dict[str, Any]:
    meminfo: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines():
        key, value = line.split(":", 1)
        meminfo[key] = int(value.strip().split()[0])
    gpu = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.used,memory.free,utilization.gpu",
            "--format=csv,noheader,nounits",
        ],
        text=True,
        timeout=10,
    ).strip()
    payload: dict[str, Any] = {
        "mem_total_kib": meminfo.get("MemTotal"),
        "mem_available_kib": meminfo.get("MemAvailable"),
        "swap_total_kib": meminfo.get("SwapTotal"),
        "swap_free_kib": meminfo.get("SwapFree"),
        "swap_used_kib": meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0),
        "nvidia_smi": gpu,
    }
    if torch_module is not None and torch_module.cuda.is_available():
        payload["torch_cuda_allocated_bytes"] = int(torch_module.cuda.memory_allocated())
        payload["torch_cuda_reserved_bytes"] = int(torch_module.cuda.memory_reserved())
        payload["torch_cuda_peak_allocated_bytes"] = int(torch_module.cuda.max_memory_allocated())
    return payload


def validate_lock(protocol_path: Path, manifest_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_json(protocol_path)
    manifest = load_json(manifest_path)
    expected = protocol["partition_manifest"]
    actual_hash = sha256_file(manifest_path)
    if actual_hash != expected["sha256"]:
        raise ValueError(f"partition manifest hash drift: {actual_hash}")
    if manifest["assignment_sha256"] != expected["assignment_sha256"]:
        raise ValueError("partition assignment hash drift")
    if manifest["status"] != expected["status_required"]:
        raise ValueError("partition manifest status drift")
    if protocol["status"] != "FROZEN_BEFORE_BASE_ENERGY_OUTCOMES":
        raise ValueError("protocol is not frozen")
    if protocol["ours_execution_authorized"] or protocol["training_authorized"]:
        raise ValueError("Base falsifier protocol improperly authorizes Ours/training")
    return protocol, manifest


def prepare_official_image_transform() -> Any:
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode

    return transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
        ]
    )


def decode_official_image_bytes(value: Any) -> Any:
    """Mirror X-VLA datasets.utils.decode_image_from_bytes without package import.

    Importing ``datasets.utils`` executes X-VLA's dataset package initializer,
    which requires the unrelated optional ``mmengine`` dependency.  The LIBERO
    path itself only needs this OpenCV decode used by the released handler.
    """
    import cv2
    from PIL import Image

    array = np.frombuffer(value, dtype=np.uint8) if isinstance(value, (bytes, bytearray)) else value
    decoded = cv2.imdecode(array, cv2.IMREAD_COLOR)
    if decoded is None:
        raw = np.frombuffer(array, dtype=np.uint8)
        if raw.size == 2764800:
            decoded = raw.reshape(720, 1280, 3)
        elif raw.size == 921600:
            decoded = raw.reshape(480, 640, 3)
        else:
            raise ValueError(f"unsupported encoded image payload with {raw.size} bytes")
    return Image.fromarray(decoded)


def official_libero_left_action(abs_action_6d: np.ndarray) -> np.ndarray:
    """Apply the released LiberoHandler's gripper threshold exactly."""
    value = np.asarray(abs_action_6d, dtype=np.float32)
    if value.ndim != 2 or value.shape[1] != 10:
        raise ValueError(f"expected [T,10] abs_action_6d, got {value.shape}")
    return np.concatenate([value[:, :9], (value[:, 9:] > 0.0).astype(np.float32)], axis=-1)


def load_sample(path: Path, frame_index: int, image_transform: Any, torch_module: Any) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        left = official_libero_left_action(np.asarray(handle["abs_action_6d"], dtype=np.float32))
        if frame_index + 30 >= len(left):
            raise ValueError(f"insufficient horizon in {path}")
        right = np.zeros_like(left)
        absolute = np.concatenate([left, right], axis=-1)
        proprio = absolute[frame_index]
        action = absolute[frame_index + 1 : frame_index + 31]
        image_index = frame_index + 1
        images = [
            decode_official_image_bytes(handle["observation/third_image"][image_index]),
            decode_official_image_bytes(handle["observation/wrist_image"][image_index]),
        ]
        image_input = [image_transform(image) for image in images]
        image_input.append(torch_module.zeros_like(image_input[0]))
        image_tensor = torch_module.stack(image_input, dim=0).unsqueeze(0)
        image_mask = torch_module.tensor([[True, True, False]], dtype=torch_module.bool)
        canonical_value = handle["language_instruction"][()]
        canonical = canonical_value.decode("utf-8") if isinstance(canonical_value, bytes) else str(canonical_value)
    if action.shape != (30, 20) or proprio.shape != (20,):
        raise ValueError(f"official action shape mismatch: {action.shape}, {proprio.shape}")
    if not (np.isfinite(action).all() and np.isfinite(proprio).all()):
        raise ValueError("nonfinite official action/proprio")
    return {
        "canonical": canonical,
        "image_input": image_tensor,
        "image_mask": image_mask,
        "proprio": torch_module.from_numpy(proprio).unsqueeze(0),
        "action": torch_module.from_numpy(action).unsqueeze(0),
        "action_sha256": array_sha256(action),
        "proprio_sha256": array_sha256(proprio),
        "image_tensor_sha256": array_sha256(image_tensor.numpy()),
    }


def to_device(sample: dict[str, Any], device: Any, torch_module: Any) -> dict[str, Any]:
    return {
        "image_input": sample["image_input"].to(device=device, dtype=torch_module.float32),
        "image_mask": sample["image_mask"].to(device=device),
        "proprio": sample["proprio"].to(device=device, dtype=torch_module.float32),
        "action": sample["action"].to(device=device, dtype=torch_module.float32),
        "domain_id": torch_module.tensor([3], device=device, dtype=torch_module.long),
    }


def fixed_noise(
    action: Any,
    base_seed: int,
    sample_index: int,
    torch_module: Any,
) -> tuple[Any, Any, int]:
    derived_seed = int(base_seed + sample_index * 10000)
    generator = torch_module.Generator(device=action.device)
    generator.manual_seed(derived_seed)
    t = torch_module.rand((1,), generator=generator, device=action.device, dtype=action.dtype)
    epsilon = torch_module.randn(action.shape, generator=generator, device=action.device, dtype=action.dtype)
    return t, epsilon, derived_seed


def evaluate_condition(
    model: Any,
    processor: Any,
    device_sample: dict[str, Any],
    instruction: str,
    t: Any,
    epsilon: Any,
    torch_module: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    input_ids = processor.encode_language(instruction)["input_ids"].to(device_sample["action"].device)
    enc = model.forward_vlm(input_ids, device_sample["image_input"], device_sample["image_mask"])
    action = device_sample["action"]
    noisy = epsilon * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
    proprio_m, noisy_m = model.action_space.preprocess(device_sample["proprio"], noisy)
    prediction = model.transformer(
        domain_id=device_sample["domain_id"],
        action_with_noise=noisy_m,
        t=t,
        proprio=proprio_m,
        **enc,
    )
    loss_dict = model.action_space.compute_loss(prediction, action)
    losses = {key: float(value.detach().float().item()) for key, value in loss_dict.items()}
    total = float(sum(loss_dict.values()).detach().float().item())
    prediction_np = prediction.detach().float().cpu().numpy()
    if not np.isfinite(prediction_np).all() or not np.isfinite(list(losses.values()) + [total]).all():
        raise ValueError("nonfinite prediction or energy")
    record = {
        "instruction": instruction,
        "input_ids_sha256": array_sha256(input_ids.detach().cpu().numpy()),
        "energy_components": losses,
        "energy_total": total,
        "prediction_sha256": array_sha256(prediction_np),
        "prediction_min": float(np.min(prediction_np)),
        "prediction_max": float(np.max(prediction_np)),
        "prediction_mean": float(np.mean(prediction_np)),
        "prediction_std": float(np.std(prediction_np)),
        "finite": True,
    }
    return record, prediction_np


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [row for row in rows if row.get("completed") is True]
    paired = []
    for row in completed:
        means = {
            condition: float(np.mean([seed["conditions"][condition]["energy_total"] for seed in row["seeds"]]))
            for condition in ("canonical", "paraphrase", "hard_negative")
        }
        paired.append(
            {
                "sample_id": row["sample_id"],
                "eval_id": row["eval_id"],
                "family": row["family"],
                **{f"{key}_energy": value for key, value in means.items()},
                "canonical_selective": means["canonical"] < means["hard_negative"],
                "equivalence_drift": means["canonical"] < means["paraphrase"],
                "ranking_violation": means["paraphrase"] >= means["hard_negative"],
                "canonical_negative_prediction_mean_abs_delta": float(
                    np.mean([seed["prediction_deltas"]["canonical_vs_hard_negative_mean_abs"] for seed in row["seeds"]])
                ),
            }
        )

    def fraction(key: str) -> float | None:
        return float(np.mean([bool(row[key]) for row in paired])) if paired else None

    def coverage(key: str, field: str) -> int:
        return len({row[field] for row in paired if row[key]})

    return {
        "completed_samples": len(completed),
        "expected_samples": 30,
        "model_forwards": sum(len(row["seeds"]) * 3 for row in completed),
        "all_finite": all(
            condition["finite"]
            for row in completed
            for seed in row["seeds"]
            for condition in seed["conditions"].values()
        ),
        "canonical_selectivity_accuracy": fraction("canonical_selective"),
        "canonical_selectivity_task_coverage": coverage("canonical_selective", "eval_id"),
        "equivalence_drift_rate": fraction("equivalence_drift"),
        "equivalence_drift_task_coverage": coverage("equivalence_drift", "eval_id"),
        "equivalence_drift_family_coverage": coverage("equivalence_drift", "family"),
        "ranking_violation_rate": fraction("ranking_violation"),
        "ranking_violation_task_coverage": coverage("ranking_violation", "eval_id"),
        "ranking_violation_family_coverage": coverage("ranking_violation", "family"),
        "canonical_negative_prediction_mean_abs_delta": (
            float(np.mean([row["canonical_negative_prediction_mean_abs_delta"] for row in paired]))
            if paired
            else None
        ),
        "paired_rows": paired,
    }


def adjudicate(summary: dict[str, Any], protocol: dict[str, Any], swap_used_kib: int) -> str:
    rules = protocol["decision_rules"]["BASE_ENERGY_HEADROOM_GO"]
    pass_go = (
        summary["completed_samples"] == summary["expected_samples"]
        and summary["model_forwards"] == protocol["energy"]["expected_total_model_forwards"]
        and summary["all_finite"]
        and swap_used_kib == 0
        and summary["canonical_selectivity_accuracy"] >= rules["canonical_selectivity_accuracy_min"]
        and summary["canonical_selectivity_task_coverage"] >= rules["canonical_selectivity_task_coverage_min"]
        and summary["equivalence_drift_rate"] >= rules["equivalence_drift_rate_min"]
        and summary["equivalence_drift_task_coverage"] >= rules["equivalence_drift_task_coverage_min"]
        and summary["equivalence_drift_family_coverage"] >= rules["equivalence_drift_family_coverage_min"]
        and summary["ranking_violation_rate"] >= rules["ranking_violation_rate_min"]
        and summary["ranking_violation_task_coverage"] >= rules["ranking_violation_task_coverage_min"]
        and summary["ranking_violation_family_coverage"] >= rules["ranking_violation_family_coverage_min"]
        and summary["canonical_negative_prediction_mean_abs_delta"]
        >= rules["canonical_negative_prediction_mean_abs_delta_min"]
    )
    return "BASE_ENERGY_HEADROOM_GO" if pass_go else "BASE_ENERGY_FALSIFIER_FAIL"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--xvla-root", type=Path, default=DEFAULT_XVLA_ROOT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    protocol, manifest = validate_lock(args.protocol, args.manifest)
    result: dict[str, Any] = {
        "schema_version": "epoch7.base_action_energy_falsifier.v1",
        "created_at": timestamp(),
        "last_updated_at": timestamp(),
        "execution_type": "VLA_INFERENCE",
        "protocol_path": str(args.protocol),
        "protocol_sha256": sha256_file(args.protocol),
        "manifest_path": str(args.manifest),
        "manifest_sha256": sha256_file(args.manifest),
        "assignment_sha256": manifest["assignment_sha256"],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "training_happened": False,
        "optimizer_step_happened": False,
        "checkpoint_written": False,
        "simulator_constructed": False,
        "simulator_episode_count": 0,
        "reward_done_success_read": False,
        "ours_executed": False,
        "confirmatory_content_read": False,
        "resources_before": memory_snapshot(),
        "samples": [],
    }
    atomic_write_json(args.output, result)
    exit_code = 1
    model = processor = None
    try:
        if result["resources_before"]["swap_used_kib"] != 0:
            raise RuntimeError("nonzero WSL swap before execution")
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        torch.cuda.set_device(0)
        torch.manual_seed(1700)
        torch.cuda.manual_seed_all(1700)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        import_report = _prepare_xvla_imports(args.xvla_root)
        from models.modeling_xvla import XVLA
        from models.processing_xvla import XVLAProcessor

        processor = XVLAProcessor.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=XVLA_CACHE_DIR,
        )
        model = XVLA.from_pretrained(
            MODEL_ID,
            revision=MODEL_REVISION,
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
            cache_dir=XVLA_CACHE_DIR,
        )
        model.eval().to(device="cuda:0", dtype=torch.float32)
        result["runtime"] = {
            "import_report": import_report,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
            "model_training_flag": bool(model.training),
            "device_set": sorted({str(parameter.device) for parameter in model.parameters()}),
        }
        result["resources_after_load"] = memory_snapshot(torch)
        atomic_write_json(args.output, result)

        image_transform = prepare_official_image_transform()
        demo_by_eval = {int(row["eval_id"]): row for row in manifest["stage0_base_energy_samples"]["demonstrations"]}
        paraphrases = manifest["stage0_base_energy_samples"]["paraphrases"]
        negatives = manifest["hard_negative_instructions"]
        canonical = manifest["canonical_instructions"]
        seeds = [int(value) for value in protocol["energy"]["time_noise_seeds"]]

        for sample_index, para in enumerate(paraphrases):
            eval_id = int(para["eval_id"])
            family = str(para["family"])
            demo = demo_by_eval[eval_id]
            path = args.data_root / demo["relative_path"]
            sample_id = f"eval{eval_id}_{family}"
            if sha256_file(path) != demo["file_sha256"]:
                raise ValueError(f"frozen demo hash drift for {sample_id}")
            host_sample = load_sample(path, int(demo["frame_index"]), image_transform, torch)
            if host_sample["canonical"] != canonical[str(eval_id)]:
                raise ValueError(f"canonical instruction drift for {sample_id}")
            device_sample = to_device(host_sample, next(model.parameters()).device, torch)
            row: dict[str, Any] = {
                "sample_id": sample_id,
                "eval_id": eval_id,
                "family": family,
                "demo_relative_path": demo["relative_path"],
                "demo_file_sha256": demo["file_sha256"],
                "frame_index": int(demo["frame_index"]),
                "action_sha256": host_sample["action_sha256"],
                "proprio_sha256": host_sample["proprio_sha256"],
                "image_tensor_sha256": host_sample["image_tensor_sha256"],
                "paraphrase_row_id": para["row_id"],
                "negative_eval_id": int(negatives[str(eval_id)]["eval_id"]),
                "seeds": [],
                "completed": False,
            }
            for base_seed in seeds:
                t, epsilon, derived_seed = fixed_noise(device_sample["action"], base_seed, sample_index, torch)
                condition_text = {
                    "canonical": canonical[str(eval_id)],
                    "paraphrase": para["instruction"],
                    "hard_negative": negatives[str(eval_id)]["instruction"],
                }
                condition_records: dict[str, Any] = {}
                predictions: dict[str, np.ndarray] = {}
                with torch.inference_mode():
                    for condition, instruction in condition_text.items():
                        record, prediction = evaluate_condition(
                            model, processor, device_sample, instruction, t, epsilon, torch
                        )
                        condition_records[condition] = record
                        predictions[condition] = prediction
                seed_record = {
                    "base_seed": base_seed,
                    "derived_seed": derived_seed,
                    "t": float(t.detach().cpu().item()),
                    "epsilon_sha256": array_sha256(epsilon.detach().cpu().numpy()),
                    "conditions": condition_records,
                    "prediction_deltas": {
                        "canonical_vs_paraphrase_mean_abs": float(
                            np.mean(np.abs(predictions["canonical"] - predictions["paraphrase"]))
                        ),
                        "canonical_vs_hard_negative_mean_abs": float(
                            np.mean(np.abs(predictions["canonical"] - predictions["hard_negative"]))
                        ),
                        "paraphrase_vs_hard_negative_mean_abs": float(
                            np.mean(np.abs(predictions["paraphrase"] - predictions["hard_negative"]))
                        ),
                    },
                }
                row["seeds"].append(seed_record)
            row["completed"] = True
            result["samples"].append(row)
            result["last_updated_at"] = timestamp()
            result["summary"] = aggregate(result["samples"])
            result["resources_latest"] = memory_snapshot(torch)
            atomic_write_json(args.output, result)
            print(f"{timestamp()} completed {sample_id} ({len(result['samples'])}/30)", flush=True)

        result["summary"] = aggregate(result["samples"])
        result["resources_at_exit"] = memory_snapshot(torch)
        result["decision"] = adjudicate(
            result["summary"], protocol, int(result["resources_at_exit"]["swap_used_kib"])
        )
        result["completed_at"] = timestamp()
        exit_code = 0
    except Exception as exc:
        result["exception"] = {
            "type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        result["decision"] = "BASE_ENERGY_EXECUTION_FAILURE"
        result["last_updated_at"] = timestamp()
    finally:
        model = processor = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            result["resources_after_cleanup"] = memory_snapshot(torch)
        except Exception as cleanup_exc:
            result["cleanup_exception"] = f"{type(cleanup_exc).__name__}: {cleanup_exc}"
        atomic_write_json(args.output, result)
    print(json.dumps({"decision": result.get("decision"), "summary": result.get("summary")}, indent=2))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
