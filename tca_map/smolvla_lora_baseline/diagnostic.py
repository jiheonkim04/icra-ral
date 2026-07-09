"""Bounded standard SmolVLA LoRA baseline reproduction.

This runner is a baseline diagnostic, not a new method. It checks whether a
standard PEFT LoRA adapter can learn from local LIBERO HDF5 action chunks under
the local RTX 5080 budget.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.smolvla.interface_adapters import ACTION_STRATEGY_GRIPPER_CLOSE, adapt_policy_action_to_env_action
from tca_map.smolvla.libero_learned_policy_rollout import (
    CAMERA_ALIAS_STRATEGY_CURRENT,
    STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3,
    _build_batch,
)
from tca_map.smolvla.load_only_smoke import (
    _external_tokenizer_files,
    _find_files,
    _nvidia_smi,
    _read_tokenizer_dependency,
    _runtime_dependencies,
)
from tca_map.smolvla.offline_demo_action_decoding import _load_first_hdf5_sample
from tca_map.smolvla.vlm_enabled_repeated_offline_decoding import _load_policy_with_vlm


HEAVY_IMPORT_GATE = "ALLOW_HEAVY_IMPORT"
BASELINE_GATE = "ALLOW_SMOLVLA_LORA_BASELINE"
TRAINING_GATE = "ALLOW_SMOLVLA_LORA_BASELINE_TRAINING"
DEFAULT_HDF5_PATH = "C:/assets/data/libero/libero_10/KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5"
DEFAULT_MAX_STEPS = 60
MAX_STEPS_CAP = 200
MAX_VRAM_MB = 15360
MAX_RUNTIME_SECONDS = 14400
LORA_TARGET_MODULES = ["state_proj", "action_in_proj", "action_out_proj"]
FINAL_DECISIONS = {
    "READY_FOR_METHOD_ON_TOP_OF_SMOLVLA_LORA",
    "KILL_NO_REAL_LORA_LEARNING",
    "KILL_MEAN_BASELINE_DOMINATED",
    "KILL_FROZEN_BASELINE_DOMINATED",
    "TOO_HEAVY_LOCAL",
    "ENV_BLOCKED",
}
FORBIDDEN_GATES = [
    "ALLOW_DOWNLOADS",
    "ALLOW_ROLLOUTS",
    "ALLOW_ROLLOUT",
    "ALLOW_POLICY_ROLLOUT",
    "ALLOW_BENCHMARK_ROLLOUT",
    "ALLOW_OPENVLA_OFT",
    "ALLOW_PATCHGUARD_TINY_LORA_TRAINING",
    "ALLOW_PATCHGUARD_VLA_STATE1B",
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _module_version(name: str) -> dict[str, Any]:
    try:
        module = __import__(name)
        return {"available": True, "import_ok": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:  # noqa: BLE001
        return {"available": False, "import_ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _safe_task_text(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_demo"):
        stem = stem[: -len("_demo")]
    return stem.replace("_", " ")


def _demo_sort_key(name: str) -> tuple[str, int | str]:
    prefix, _, suffix = name.rpartition("_")
    if suffix.isdigit():
        return prefix, int(suffix)
    return name, name


def _round(value: float | np.floating[Any], digits: int = 6) -> float:
    return round(float(value), digits)


def _action_chunk(path: Path, demo_name: str, timestep: int, chunk_size: int, action_dim: int) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        actions = np.asarray(handle["data"][demo_name]["actions"], dtype=np.float32)
    end = min(actions.shape[0], timestep + chunk_size)
    chunk = actions[timestep:end, :action_dim]
    if chunk.shape[0] == 0:
        raise ValueError(f"empty action chunk for {demo_name} at timestep {timestep}")
    if chunk.shape[0] < chunk_size:
        pad = np.repeat(chunk[-1:, :], chunk_size - chunk.shape[0], axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32)


def select_records(
    hdf5_path: Path,
    *,
    max_train_demos: int,
    max_eval_demos: int,
    records_per_demo: int,
) -> dict[str, Any]:
    import h5py

    with h5py.File(hdf5_path, "r") as handle:
        demos = sorted(handle["data"].keys(), key=_demo_sort_key)
        if len(demos) < max(2, max_train_demos + 1):
            raise ValueError(f"need at least {max(2, max_train_demos + 1)} demos, found {len(demos)}")
        train_demos = demos[:max_train_demos]
        eval_demos = demos[max_train_demos : max_train_demos + max_eval_demos]
        if not eval_demos:
            eval_demos = demos[max_train_demos : max_train_demos + 1]

        def records_for(demo_names: list[str]) -> list[dict[str, Any]]:
            records: list[dict[str, Any]] = []
            for demo_name in demo_names:
                actions = np.asarray(handle["data"][demo_name]["actions"], dtype=np.float32)
                usable = max(1, actions.shape[0] - 1)
                if records_per_demo <= 1:
                    timesteps = [0]
                else:
                    timesteps = np.linspace(0, usable - 1, num=records_per_demo, dtype=np.int64).tolist()
                for timestep in dict.fromkeys(int(x) for x in timesteps):
                    records.append(
                        {
                            "hdf5_path": str(hdf5_path),
                            "task_name": hdf5_path.stem,
                            "task_text": _safe_task_text(hdf5_path),
                            "demo_name": demo_name,
                            "timestep": int(timestep),
                            "action_length": int(actions.shape[0]),
                            "action_dim": int(actions.shape[1]),
                        }
                    )
            return records

        train_records = records_for(train_demos)
        eval_records = records_for(eval_demos)
    return {
        "hdf5_path": str(hdf5_path),
        "task_name": hdf5_path.stem,
        "task_text": _safe_task_text(hdf5_path),
        "all_demo_count": len(demos),
        "train_demos": train_demos,
        "eval_demos": eval_demos,
        "train_records": train_records,
        "eval_records": eval_records,
        "train_count": len(train_records),
        "eval_count": len(eval_records),
        "split": "deterministic_demo_holdout",
        "records_per_demo": int(records_per_demo),
    }


def _expert_action(path: Path, demo_name: str, timestep: int) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        return np.asarray(handle["data"][demo_name]["actions"][timestep], dtype=np.float32).reshape(-1)


def _build_training_batch(
    *,
    config: Any,
    tokenizer_root: Path,
    record: dict[str, Any],
    device: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    hdf5_path = Path(record["hdf5_path"])
    sample = _load_first_hdf5_sample(hdf5_path, record["demo_name"], int(record["timestep"]))
    batch, batch_metadata = _build_batch(
        config,
        tokenizer_root,
        sample["obs"],
        record["task_text"],
        device,
        CAMERA_ALIAS_STRATEGY_CURRENT,
        STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3,
    )
    action_dim = int(config.output_features["action"].shape[0])
    chunk = _action_chunk(hdf5_path, record["demo_name"], int(record["timestep"]), int(config.chunk_size), action_dim)
    batch["action"] = torch.tensor(chunk[None, :, :], dtype=torch.float32, device=device)
    return batch, {"sample": sample["metadata"], "batch": batch_metadata}


def _forward_loss(policy: Any, batch: dict[str, Any], config: Any, device: str):
    import torch

    noise = torch.zeros((1, int(config.chunk_size), int(config.max_action_dim)), dtype=torch.float32, device=device)
    time_tensor = torch.full((1,), 0.5, dtype=torch.float32, device=device)
    return policy.forward(batch, noise=noise, time=time_tensor)


def _load_policy(
    *,
    smolvla_ckpt: Path,
    hf_home: Path,
    checkpoint_root: Path,
    device: str,
    lora_rank: int | None = None,
    target_modules: list[str] | None = None,
) -> tuple[Any, Any, Path, dict[str, Any]]:
    dependency_name = _read_tokenizer_dependency(smolvla_ckpt)
    external_dependency = _external_tokenizer_files(dependency_name, [hf_home, checkpoint_root])
    if not external_dependency.get("found"):
        raise FileNotFoundError("external tokenizer/VLM dependency root is missing")
    policy, config = _load_policy_with_vlm(smolvla_ckpt, hf_home, external_dependency, device)
    if lora_rank is not None:
        from peft import LoraConfig, get_peft_model

        lora_config = LoraConfig(
            r=int(lora_rank),
            lora_alpha=int(lora_rank) * 2,
            target_modules=target_modules or LORA_TARGET_MODULES,
            lora_dropout=0.0,
            bias="none",
        )
        policy = get_peft_model(policy, lora_config)
    return policy, config, Path(external_dependency["root"]), external_dependency


def _trainable_params(policy: Any) -> dict[str, int]:
    return {
        "total_params": int(sum(p.numel() for p in policy.parameters())),
        "trainable_params": int(sum(p.numel() for p in policy.parameters() if p.requires_grad)),
    }


def _predict_policy_action(
    *,
    policy: Any,
    config: Any,
    tokenizer_root: Path,
    record: dict[str, Any],
    device: str,
) -> tuple[np.ndarray, np.ndarray]:
    import torch

    batch, _metadata = _build_training_batch(config=config, tokenizer_root=tokenizer_root, record=record, device=device)
    batch.pop("action")
    expert = _expert_action(Path(record["hdf5_path"]), record["demo_name"], int(record["timestep"]))
    policy.reset()
    noise = torch.zeros((1, int(config.chunk_size), int(config.max_action_dim)), dtype=torch.float32, device=device)
    with torch.no_grad():
        action = policy.select_action(batch, noise=noise)
    adapter = adapt_policy_action_to_env_action(
        action,
        7,
        strategy=ACTION_STRATEGY_GRIPPER_CLOSE,
        action_scale=1.0,
    )
    return np.asarray(adapter.values, dtype=np.float32).reshape(-1), expert.reshape(-1)


def _metrics_from_predictions(predictions: list[np.ndarray], experts: list[np.ndarray]) -> dict[str, Any]:
    pred = np.stack([np.asarray(item, dtype=np.float32).reshape(-1)[:7] for item in predictions], axis=0)
    expert = np.stack([np.asarray(item, dtype=np.float32).reshape(-1)[:7] for item in experts], axis=0)
    diff = pred - expert
    action_l2 = np.linalg.norm(diff, axis=1)
    first6_l2 = np.linalg.norm(diff[:, :6], axis=1)
    translation_l2 = np.linalg.norm(diff[:, :3], axis=1)
    rotation_l2 = np.linalg.norm(diff[:, 3:6], axis=1)
    gripper_error = np.abs(diff[:, 6])
    pred_gripper = pred[:, 6] >= 0
    expert_gripper = expert[:, 6] >= 0
    per_dim_mae = np.mean(np.abs(diff), axis=0)
    worst_dims = sorted(
        [{"dim": int(idx), "mae": _round(value)} for idx, value in enumerate(per_dim_mae)],
        key=lambda item: item["mae"],
        reverse=True,
    )
    return {
        "sample_count": int(pred.shape[0]),
        "action_l2": _round(np.mean(action_l2)),
        "action_l2_first6": _round(np.mean(first6_l2)),
        "translation_l2": _round(np.mean(translation_l2)),
        "rotation_l2": _round(np.mean(rotation_l2)),
        "gripper_error": _round(np.mean(gripper_error)),
        "gripper_accuracy": _round(np.mean(pred_gripper == expert_gripper)),
        "per_dim_mae": [_round(x) for x in per_dim_mae],
        "worst_action_dimensions": worst_dims[:3],
    }


def evaluate_constant_action(records: list[dict[str, Any]], action: np.ndarray) -> dict[str, Any]:
    predictions = []
    experts = []
    for record in records:
        experts.append(_expert_action(Path(record["hdf5_path"]), record["demo_name"], int(record["timestep"])))
        predictions.append(np.asarray(action, dtype=np.float32).reshape(-1)[:7])
    return _metrics_from_predictions(predictions, experts)


def _evaluate_policy(
    *,
    policy: Any,
    config: Any,
    tokenizer_root: Path,
    records: list[dict[str, Any]],
    device: str,
) -> dict[str, Any]:
    policy.eval()
    predictions = []
    experts = []
    for record in records:
        prediction, expert = _predict_policy_action(
            policy=policy,
            config=config,
            tokenizer_root=tokenizer_root,
            record=record,
            device=device,
        )
        predictions.append(prediction)
        experts.append(expert)
    return _metrics_from_predictions(predictions, experts)


def _mean_train_action(records: list[dict[str, Any]]) -> np.ndarray:
    actions = [
        _expert_action(Path(record["hdf5_path"]), record["demo_name"], int(record["timestep"]))[:7]
        for record in records
    ]
    return np.mean(np.stack(actions, axis=0), axis=0).astype(np.float32)


def _train_lora(
    *,
    policy: Any,
    config: Any,
    tokenizer_root: Path,
    train_records: list[dict[str, Any]],
    device: str,
    max_steps: int,
    learning_rate: float,
) -> dict[str, Any]:
    import torch

    policy.train()
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=float(learning_rate))
    losses: list[float] = []
    for step in range(max_steps):
        record = train_records[step % len(train_records)]
        optimizer.zero_grad(set_to_none=True)
        batch, _metadata = _build_training_batch(config=config, tokenizer_root=tokenizer_root, record=record, device=device)
        loss, _loss_dict = _forward_loss(policy, batch, config, device)
        loss.backward()
        optimizer.step()
        losses.append(_round(loss.detach().cpu()))
    first_window = float(np.mean(losses[: min(5, len(losses))])) if losses else None
    last_window = float(np.mean(losses[-min(5, len(losses)) :])) if losses else None
    loss_decreased = bool(
        losses
        and first_window is not None
        and last_window is not None
        and (last_window <= first_window * 0.95 or last_window <= first_window - 0.005)
    )
    return {
        "max_steps": int(max_steps),
        "optimizer_steps": int(len(losses)),
        "learning_rate": float(learning_rate),
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "loss_first_window_mean": _round(first_window) if first_window is not None else None,
        "loss_last_window_mean": _round(last_window) if last_window is not None else None,
        "loss_delta": _round(losses[-1] - losses[0]) if len(losses) > 1 else 0.0,
        "loss_decreased_meaningfully": loss_decreased,
        "loss_trace": losses,
    }


def _dependency_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python_version": sys.version.split()[0],
        "modules": {
            name: _module_version(name)
            for name in ["torch", "transformers", "accelerate", "peft", "bitsandbytes", "lerobot", "h5py"]
        },
        "nvidia_smi": _nvidia_smi(),
        "runtime_dependencies": _runtime_dependencies(),
        "torch_cuda": {},
        "smokes": {},
    }
    try:
        import torch

        report["torch_cuda"] = {
            "torch_version": torch.__version__,
            "cuda_available": bool(torch.cuda.is_available()),
            "torch_cuda_runtime": torch.version.cuda,
            "device_count": torch.cuda.device_count(),
            "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "total_vram_mb": _round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
            if torch.cuda.is_available()
            else None,
        }
        if torch.cuda.is_available():
            x = torch.ones((8, 8), device="cuda")
            y = (x @ x).sum()
            torch.cuda.synchronize()
            report["smokes"]["tiny_cuda_tensor"] = {"ok": True, "value": float(y.detach().cpu())}
    except Exception as exc:  # noqa: BLE001
        report["smokes"]["tiny_cuda_tensor"] = {"ok": False, "error": _compact_error(exc)}
    try:
        import torch
        from peft import LoraConfig, get_peft_model

        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
        wrapped = get_peft_model(
            model,
            LoraConfig(r=4, lora_alpha=8, target_modules=["0", "2"], lora_dropout=0.0, bias="none"),
        )
        _ = wrapped(torch.randn(2, 8))
        report["smokes"]["peft_dummy_lora"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001
        report["smokes"]["peft_dummy_lora"] = {"ok": False, "error": _compact_error(exc)}
    try:
        import bitsandbytes as bnb
        import torch

        layer = bnb.nn.Linear8bitLt(8, 4, has_fp16_weights=False).cuda()
        _ = layer(torch.randn(2, 8, device="cuda", dtype=torch.float16))
        torch.cuda.synchronize()
        report["smokes"]["bitsandbytes_8bit_linear"] = {"ok": True}
    except Exception as exc:  # noqa: BLE001
        report["smokes"]["bitsandbytes_8bit_linear"] = {"ok": False, "error": _compact_error(exc)}
    return report


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    summary = report.get("summary") or {}
    split = report.get("dataset_split") or {}
    metrics = report.get("metrics") or {}
    training = report.get("training") or {}
    lines = [
        "# SmolVLA LoRA Baseline STATE 1 Result",
        "",
        "Bounded standard LoRA baseline reproduction on local LIBERO HDF5 data. This is not a new method, rollout, full benchmark, OpenVLA-OFT run, or paper claim.",
        "",
        f"- final decision: `{summary.get('final_decision')}`",
        f"- model used: `{summary.get('model_used')}`",
        f"- dataset: `{split.get('hdf5_path')}`",
        f"- split: `{split.get('split')}`",
        f"- train demos: `{split.get('train_demos')}`",
        f"- eval demos: `{split.get('eval_demos')}`",
        f"- train/eval counts: `{split.get('train_count')} / {split.get('eval_count')}`",
        f"- training happened: `{summary.get('training_happened')}`",
        f"- loss computed: `{summary.get('loss_computed')}`",
        f"- LoRA rank: `{summary.get('lora_rank')}`",
        f"- trainable params: `{summary.get('trainable_params')}`",
        f"- VRAM peak MB: `{summary.get('vram_peak_mb')}`",
        f"- runtime sec: `{summary.get('runtime_sec')}`",
        f"- loss start/end: `{training.get('loss_start')} / {training.get('loss_end')}`",
        f"- loss decreased meaningfully: `{training.get('loss_decreased_meaningfully')}`",
        "",
        "## Eval Metrics",
        "",
        "| variant | action L2 | first6 L2 | translation L2 | rotation L2 | gripper error | gripper accuracy |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name in ["mean_action", "frozen_base_smolvla", "standard_lora"]:
        payload = metrics.get(name) or {}
        lines.append(
            "| {name} | {action} | {first6} | {trans} | {rot} | {grip_err} | {grip_acc} |".format(
                name=name,
                action=payload.get("action_l2"),
                first6=payload.get("action_l2_first6"),
                trans=payload.get("translation_l2"),
                rot=payload.get("rotation_l2"),
                grip_err=payload.get("gripper_error"),
                grip_acc=payload.get("gripper_accuracy"),
            )
        )
    lines.extend(
        [
            "",
            f"- LoRA beats mean-action baseline: `{summary.get('lora_beats_mean_action')}`",
            f"- LoRA beats frozen/base baseline: `{summary.get('lora_beats_frozen_base')}`",
            f"- LoRA learns: `{summary.get('lora_learns')}`",
            "",
            f"Exact next step: {summary.get('exact_next_step')}",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    max_steps = min(max(1, int(args.max_steps)), MAX_STEPS_CAP)
    hdf5_path = Path(args.hdf5_path)
    smolvla_ckpt = Path(args.smolvla_ckpt)
    checkpoint_root = Path(args.checkpoint_root)
    hf_home = Path(args.hf_home)

    report: dict[str, Any] = {
        "schema_version": "smolvla-lora-baseline-state1-v1",
        "evidence_label": "smolvla_lora_baseline_state1",
        "decision": "ENV_BLOCKED",
        "policy": {
            "bounded_standard_lora_baseline": True,
            "new_method_created": False,
            "patchguard_continued": False,
            "downloads_performed": False,
            "large_model_or_dataset_downloads_performed": False,
            "rollouts_performed": False,
            "benchmark_rollouts_performed": False,
            "openvla_oft_executed": False,
            "full_vla_finetuning_performed": False,
            "paper_grade_claims_made": False,
            "heavy_import_gate_set": _env_flag(HEAVY_IMPORT_GATE),
            "baseline_gate_set": _env_flag(BASELINE_GATE),
            "training_gate_set": _env_flag(TRAINING_GATE),
            "forbidden_gates_set": forbidden,
            "training_performed": False,
            "loss_computed": False,
            "model_load_performed": False,
        },
        "paths": {
            "hdf5_path": str(hdf5_path),
            "smolvla_ckpt": str(smolvla_ckpt),
            "checkpoint_root": str(checkpoint_root),
            "hf_home": str(hf_home),
        },
        "risk_limits": {
            "max_steps": max_steps,
            "batch_size": 1,
            "lora_rank": int(args.lora_rank),
            "max_vram_mb": MAX_VRAM_MB,
            "max_runtime_seconds": MAX_RUNTIME_SECONDS,
            "device": args.device,
        },
        "dependency_check": {},
        "files": {},
        "dataset_split": {},
        "adapter_probe": {},
        "training": {},
        "metrics": {},
        "summary": {},
        "error": None,
    }

    def finish(decision: str, next_step: str, code: int) -> tuple[dict[str, Any], int]:
        if decision not in FINAL_DECISIONS:
            raise ValueError(f"invalid final decision: {decision}")
        report["decision"] = decision
        report["summary"]["final_decision"] = decision
        report["summary"]["exact_next_step"] = next_step
        report["summary"]["runtime_sec"] = _round(time.monotonic() - started, 3)
        try:
            import torch

            report["summary"]["vram_peak_mb"] = (
                _round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3) if torch.cuda.is_available() else 0.0
            )
        except Exception:
            report["summary"]["vram_peak_mb"] = None
        return report, code

    if not report["policy"]["heavy_import_gate_set"]:
        return finish("ENV_BLOCKED", f"Set {HEAVY_IMPORT_GATE}=1 for this bounded check.", 2)
    if not report["policy"]["baseline_gate_set"]:
        return finish("ENV_BLOCKED", f"Set {BASELINE_GATE}=1 for this bounded check.", 3)
    if forbidden:
        report["error"] = {"message": "Forbidden gate(s) set: " + ", ".join(forbidden)}
        return finish("ENV_BLOCKED", "Clear forbidden rollout/download/OpenVLA-OFT/PatchGuard gates and rerun.", 4)

    try:
        report["dependency_check"] = _dependency_report()
        modules = report["dependency_check"].get("modules") or {}
        torch_cuda = report["dependency_check"].get("torch_cuda") or {}
        peft_ok = bool((modules.get("peft") or {}).get("import_ok"))
        cuda_ok = bool(torch_cuda.get("cuda_available"))
        if not peft_ok:
            return finish("ENV_BLOCKED", "PEFT is unavailable; cannot run real SmolVLA LoRA.", 5)
        if not cuda_ok or args.device != "cuda":
            return finish("TOO_HEAVY_LOCAL", "CUDA is unavailable for the bounded LoRA baseline.", 6)
        if not report["policy"]["training_gate_set"]:
            return finish("ENV_BLOCKED", f"Set {TRAINING_GATE}=1 to allow the bounded standard LoRA baseline.", 7)
        if not hdf5_path.exists():
            return finish("ENV_BLOCKED", f"Missing local LIBERO HDF5 path: {hdf5_path}", 8)

        report["files"] = {
            "smolvla_config_files": _find_files(smolvla_ckpt, ["config.json"]),
            "smolvla_weight_files": _find_files(
                smolvla_ckpt,
                ["model.safetensors", "pytorch_model.bin", "model-00001-of-00001.safetensors"],
                ["*.safetensors", "*.bin"],
            ),
        }
        report["dataset_split"] = select_records(
            hdf5_path,
            max_train_demos=int(args.max_train_demos),
            max_eval_demos=int(args.max_eval_demos),
            records_per_demo=int(args.records_per_demo),
        )

        import torch

        torch.cuda.reset_peak_memory_stats()
        mean_action = _mean_train_action(report["dataset_split"]["train_records"])
        report["metrics"]["mean_action"] = evaluate_constant_action(report["dataset_split"]["eval_records"], mean_action)

        base_policy, base_config, tokenizer_root, external_dependency = _load_policy(
            smolvla_ckpt=smolvla_ckpt,
            hf_home=hf_home,
            checkpoint_root=checkpoint_root,
            device=args.device,
            lora_rank=None,
        )
        report["policy"]["model_load_performed"] = True
        report["adapter_probe"]["external_dependency"] = external_dependency
        report["adapter_probe"]["base_model_params"] = _trainable_params(base_policy)
        report["metrics"]["frozen_base_smolvla"] = _evaluate_policy(
            policy=base_policy,
            config=base_config,
            tokenizer_root=tokenizer_root,
            records=report["dataset_split"]["eval_records"],
            device=args.device,
        )
        del base_policy
        gc.collect()
        torch.cuda.empty_cache()

        lora_policy, lora_config, tokenizer_root, _external_dependency = _load_policy(
            smolvla_ckpt=smolvla_ckpt,
            hf_home=hf_home,
            checkpoint_root=checkpoint_root,
            device=args.device,
            lora_rank=int(args.lora_rank),
        )
        report["adapter_probe"].update(
            {
                "lora_injection_happened": True,
                "target_modules": LORA_TARGET_MODULES,
                "lora_rank": int(args.lora_rank),
                **_trainable_params(lora_policy),
            }
        )
        if report["adapter_probe"]["trainable_params"] <= 0:
            return finish("ENV_BLOCKED", "LoRA injection created no trainable parameters.", 9)

        report["training"] = _train_lora(
            policy=lora_policy,
            config=lora_config,
            tokenizer_root=tokenizer_root,
            train_records=report["dataset_split"]["train_records"],
            device=args.device,
            max_steps=max_steps,
            learning_rate=float(args.learning_rate),
        )
        report["policy"]["training_performed"] = True
        report["policy"]["loss_computed"] = True
        report["metrics"]["standard_lora_train"] = _evaluate_policy(
            policy=lora_policy,
            config=lora_config,
            tokenizer_root=tokenizer_root,
            records=report["dataset_split"]["train_records"],
            device=args.device,
        )
        report["metrics"]["standard_lora"] = _evaluate_policy(
            policy=lora_policy,
            config=lora_config,
            tokenizer_root=tokenizer_root,
            records=report["dataset_split"]["eval_records"],
            device=args.device,
        )
        del lora_policy
        gc.collect()
        torch.cuda.empty_cache()

        mean_metric = report["metrics"]["mean_action"]["action_l2"]
        frozen_metric = report["metrics"]["frozen_base_smolvla"]["action_l2"]
        lora_metric = report["metrics"]["standard_lora"]["action_l2"]
        train_metric = report["metrics"]["standard_lora_train"]["action_l2"]
        vram_peak_mb = _round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3)
        report["summary"].update(
            {
                "model_used": str(smolvla_ckpt),
                "dataset_used": str(hdf5_path),
                "lora_rank": int(args.lora_rank),
                "trainable_params": report["adapter_probe"]["trainable_params"],
                "vram_peak_mb": vram_peak_mb,
                "training_happened": True,
                "loss_computed": True,
                "mean_action_metric": mean_metric,
                "frozen_base_metric": frozen_metric,
                "standard_lora_metric": lora_metric,
                "standard_lora_train_metric": train_metric,
                "train_eval_gap": _round(lora_metric - train_metric),
                "lora_beats_mean_action": bool(lora_metric < mean_metric),
                "lora_beats_frozen_base": bool(lora_metric < frozen_metric),
                "lora_learns": bool(report["training"]["loss_decreased_meaningfully"]),
                "previous_simple_mlp_linear_comparable": False,
                "previous_simple_mlp_linear_note": "Previous simple MLP/linear action-head metrics used different local proxy heads and are not directly comparable to real SmolVLA select_action.",
                "failure_cases_by_action_dimension": report["metrics"]["standard_lora"]["worst_action_dimensions"],
            }
        )
        if vram_peak_mb > MAX_VRAM_MB:
            return finish("TOO_HEAVY_LOCAL", "Standard LoRA baseline exceeded the local VRAM budget.", 10)
        if not report["summary"]["lora_learns"]:
            return finish("KILL_NO_REAL_LORA_LEARNING", "Standard LoRA loss did not decrease meaningfully.", 0)
        if not report["summary"]["lora_beats_mean_action"]:
            return finish("KILL_MEAN_BASELINE_DOMINATED", "Standard LoRA did not beat the mean-action baseline.", 0)
        if not report["summary"]["lora_beats_frozen_base"]:
            return finish("KILL_FROZEN_BASELINE_DOMINATED", "Standard LoRA did not beat frozen/base SmolVLA.", 0)
        return finish(
            "READY_FOR_METHOD_ON_TOP_OF_SMOLVLA_LORA",
            "Plan the next method only after preserving this standard LoRA baseline and predeclaring comparisons against standard LoRA, generic augmentation, cutout/random-erasing, and no-adaptation controls.",
            0,
        )
    except Exception as exc:  # noqa: BLE001
        report["error"] = _compact_error(exc)
        message = str(exc).lower()
        if "out of memory" in message:
            return finish("TOO_HEAVY_LOCAL", "OOM at batch size 1/rank 4; stop local SmolVLA LoRA baseline.", 11)
        return finish("ENV_BLOCKED", "Fix the environment, local data, or SmolVLA LoRA adapter path before rerun.", 12)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf5-path", default=DEFAULT_HDF5_PATH)
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--report-path", default="reports/smolvla_lora_baseline_state1_result.json")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-train-demos", type=int, default=3)
    parser.add_argument("--max-eval-demos", type=int, default=2)
    parser.add_argument("--records-per-demo", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    args = parser.parse_args(argv)

    report, exit_code = build_report(args)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if report_path.suffix == ".json":
        _write_markdown_report(report, report_path.with_suffix(".md"))
    print(json.dumps(report, indent=2, sort_keys=True))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
