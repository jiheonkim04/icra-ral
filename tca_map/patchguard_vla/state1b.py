"""PatchGuard-VLA STATE 1B environment and tiny adapter feasibility gate."""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from tca_map.patchguard_vla.diagnostic import (
    VARIANT_CLEAN,
    VARIANT_CUTOUT_DEFENSE,
    VARIANT_FIXED_VISIBLE_PATCH,
    VARIANT_RANDOM_PATCH,
    apply_patch_variant,
)
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
    _rss_mb,
    _runtime_dependencies,
)
from tca_map.smolvla.offline_demo_action_decoding import _load_first_hdf5_sample
from tca_map.smolvla.vlm_enabled_repeated_offline_decoding import _load_policy_with_vlm


HEAVY_IMPORT_GATE = "ALLOW_HEAVY_IMPORT"
STATE1B_GATE = "ALLOW_PATCHGUARD_VLA_STATE1B"
TRAINING_GATE = "ALLOW_PATCHGUARD_TINY_LORA_TRAINING"
MAX_STEPS_CAP = 30
DEFAULT_MAX_STEPS = 10
MAX_VRAM_MB = 15360
MAX_RUNTIME_SECONDS = 1800
LORA_TARGET_MODULES = ["state_proj", "action_in_proj", "action_out_proj"]
EVAL_VARIANTS = [VARIANT_CLEAN, VARIANT_RANDOM_PATCH, VARIANT_FIXED_VISIBLE_PATCH, VARIANT_CUTOUT_DEFENSE]
TRAINING_VARIANTS = ["standard_clean_lora", "generic_adv_aug_lora", "patchguard_kinematic_lora"]
FINAL_DECISIONS = {
    "READY_FOR_PATCHGUARD_LORA_STATE2",
    "QLORA_BLOCKED_BUT_LORA_POSSIBLE",
    "ENV_BLOCKED_INSTALL_FAILED",
    "KILL_NO_ADAPTER_PATH",
    "KILL_BASELINE_DOMINATED",
    "TOO_HEAVY_LOCAL",
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
    "ALLOW_LIBERO_ROBOSUITE_DIAGNOSTIC_ROLLOUT",
]


def _env_flag(name: str) -> bool:
    return os.environ.get(name) == "1"


def _compact_error(exc: BaseException) -> dict[str, Any]:
    return {
        "type": type(exc).__name__,
        "message": str(exc),
        "traceback_tail": traceback.format_exc().splitlines()[-12:],
    }


def _module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def _module_version(name: str) -> dict[str, Any]:
    if not _module_available(name):
        return {"available": False}
    try:
        module = __import__(name)
        return {"available": True, "import_ok": True, "version": getattr(module, "__version__", None)}
    except Exception as exc:  # noqa: BLE001
        return {"available": True, "import_ok": False, "error": f"{type(exc).__name__}: {exc}"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _finite_list(values: Any, limit: int | None = None) -> list[float]:
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    if limit is not None:
        flat = flat[:limit]
    return [round(float(x), 6) for x in flat]


def _plan_inputs(plan: dict[str, Any], args: argparse.Namespace) -> tuple[Path, list[int], str, str | None]:
    hdf5_path = Path((plan.get("inputs") or {}).get("hdf5_path") or args.hdf5_path)
    timesteps = (plan.get("inputs") or {}).get("selected_timesteps") or []
    if not timesteps:
        timesteps = (((plan.get("planned_sample") or {}).get("hdf5") or {}).get("selected_timesteps") or [])
    cleaned: list[int] = []
    for value in timesteps:
        item = int(value)
        if item >= 0 and item not in cleaned:
            cleaned.append(item)
    task_text = (
        ((plan.get("planned_sample") or {}).get("selected_task_text"))
        or ((plan.get("planned_sample") or {}).get("selected_language"))
        or "turn on the stove and put the moka pot on it"
    )
    demo_name = (((plan.get("planned_sample") or {}).get("hdf5") or {}).get("demo_name"))
    return hdf5_path, cleaned[: max(1, int(args.max_train_samples))], task_text, demo_name


def _action_chunk(path: Path, demo_name: str | None, timestep: int, chunk_size: int, action_dim: int) -> np.ndarray:
    import h5py

    with h5py.File(path, "r") as handle:
        selected_demo = demo_name or sorted(handle["data"].keys())[0]
        actions = np.asarray(handle["data"][selected_demo]["actions"], dtype=np.float32)
    end = min(actions.shape[0], timestep + chunk_size)
    chunk = actions[timestep:end, :action_dim]
    if chunk.shape[0] == 0:
        raise ValueError(f"empty action chunk at timestep {timestep}")
    if chunk.shape[0] < chunk_size:
        pad = np.repeat(chunk[-1:, :], chunk_size - chunk.shape[0], axis=0)
        chunk = np.concatenate([chunk, pad], axis=0)
    return chunk.astype(np.float32)


def _build_training_batch(
    *,
    config: Any,
    tokenizer_root: Path,
    hdf5_path: Path,
    demo_name: str | None,
    timestep: int,
    task_text: str,
    variant: str,
    device: str,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    import torch

    sample = _load_first_hdf5_sample(hdf5_path, demo_name, timestep)
    obs, patch_metadata = apply_patch_variant(sample["obs"], variant, seed=seed + timestep)
    batch, batch_metadata = _build_batch(
        config,
        tokenizer_root,
        obs,
        task_text,
        device,
        CAMERA_ALIAS_STRATEGY_CURRENT,
        STATE_ADAPTER_STRATEGY_EEF_POS_QUAT_FIRST3,
    )
    action_dim = int(config.output_features["action"].shape[0])
    chunk = _action_chunk(hdf5_path, demo_name, timestep, int(config.chunk_size), action_dim)
    batch["action"] = torch.tensor(chunk[None, :, :], dtype=torch.float32, device=device)
    metadata = {
        "sample": sample["metadata"],
        "patch_metadata": patch_metadata,
        "batch_metadata": batch_metadata,
        "variant": variant,
        "timestep": int(timestep),
    }
    return batch, metadata


def _forward_loss(policy: Any, batch: dict[str, Any], config: Any, device: str):
    import torch

    noise = torch.zeros((1, int(config.chunk_size), int(config.max_action_dim)), dtype=torch.float32, device=device)
    time_tensor = torch.full((1,), 0.5, dtype=torch.float32, device=device)
    loss, loss_dict = policy.forward(batch, noise=noise, time=time_tensor)
    return loss, loss_dict


def _decode_action(
    *,
    policy: Any,
    config: Any,
    tokenizer_root: Path,
    hdf5_path: Path,
    demo_name: str | None,
    timestep: int,
    task_text: str,
    variant: str,
    device: str,
    seed: int,
) -> dict[str, Any]:
    import torch

    batch, metadata = _build_training_batch(
        config=config,
        tokenizer_root=tokenizer_root,
        hdf5_path=hdf5_path,
        demo_name=demo_name,
        timestep=timestep,
        task_text=task_text,
        variant=variant,
        device=device,
        seed=seed,
    )
    expert_action = batch.pop("action")[0, 0, :6].detach().cpu().numpy()
    policy.reset()
    noise = torch.zeros((1, int(config.chunk_size), int(config.max_action_dim)), dtype=torch.float32, device=device)
    with torch.no_grad():
        policy_action = policy.select_action(batch, noise=noise)
    adapter = adapt_policy_action_to_env_action(
        policy_action,
        7,
        strategy=ACTION_STRATEGY_GRIPPER_CLOSE,
        action_scale=1.0,
    )
    policy_np = policy_action.detach().cpu().numpy().reshape(-1).astype(np.float32)
    adapted = np.asarray(adapter.values, dtype=np.float32)
    return {
        "variant": variant,
        "timestep": int(timestep),
        "policy_action_preview": _finite_list(policy_np, 6),
        "adapted_action_preview": _finite_list(adapted, 7),
        "expert_action_preview": _finite_list(expert_action, 6),
        "action_l1_to_expert_first6": round(float(np.mean(np.abs(policy_np[:6] - expert_action[:6]))), 6),
        "action_adapter_metadata": adapter.metadata,
        "metadata": metadata,
    }


def _evaluate_policy(
    *,
    policy: Any,
    config: Any,
    tokenizer_root: Path,
    hdf5_path: Path,
    demo_name: str | None,
    timesteps: list[int],
    task_text: str,
    device: str,
    seed: int,
) -> dict[str, Any]:
    policy.eval()
    samples = []
    for timestep in timesteps:
        for variant in EVAL_VARIANTS:
            samples.append(
                _decode_action(
                    policy=policy,
                    config=config,
                    tokenizer_root=tokenizer_root,
                    hdf5_path=hdf5_path,
                    demo_name=demo_name,
                    timestep=timestep,
                    task_text=task_text,
                    variant=variant,
                    device=device,
                    seed=seed,
                )
            )
    clean_by_timestep = {item["timestep"]: item for item in samples if item["variant"] == VARIANT_CLEAN}
    variant_metrics: dict[str, dict[str, Any]] = {}
    for variant in EVAL_VARIANTS:
        items = [item for item in samples if item["variant"] == variant]
        divergences = []
        translation_l2 = []
        l1_to_expert = []
        for item in items:
            clean = clean_by_timestep[item["timestep"]]
            policy_action = np.asarray(item["policy_action_preview"], dtype=np.float32)
            clean_action = np.asarray(clean["policy_action_preview"], dtype=np.float32)
            adapted = np.asarray(item["adapted_action_preview"], dtype=np.float32)
            clean_adapted = np.asarray(clean["adapted_action_preview"], dtype=np.float32)
            divergences.append(float(np.mean(np.abs(policy_action - clean_action))))
            translation_l2.append(float(np.linalg.norm(adapted[:3] - clean_adapted[:3])))
            l1_to_expert.append(float(item["action_l1_to_expert_first6"]))
        variant_metrics[variant] = {
            "sample_count": len(items),
            "mean_policy6_l1_vs_clean": round(float(np.mean(divergences)), 6) if divergences else 0.0,
            "max_policy6_l1_vs_clean": round(float(np.max(divergences)), 6) if divergences else 0.0,
            "mean_translation_l2_vs_clean": round(float(np.mean(translation_l2)), 6) if translation_l2 else 0.0,
            "mean_action_l1_to_expert_first6": round(float(np.mean(l1_to_expert)), 6) if l1_to_expert else None,
        }
    attack_metric = float(
        np.mean(
            [
                variant_metrics[VARIANT_RANDOM_PATCH]["mean_policy6_l1_vs_clean"],
                variant_metrics[VARIANT_FIXED_VISIBLE_PATCH]["mean_policy6_l1_vs_clean"],
            ]
        )
    )
    return {
        "samples": samples,
        "variant_metrics": variant_metrics,
        "attack_divergence_metric": round(attack_metric, 6),
        "fixed_patch_metric": variant_metrics[VARIANT_FIXED_VISIBLE_PATCH]["mean_policy6_l1_vs_clean"],
        "random_patch_metric": variant_metrics[VARIANT_RANDOM_PATCH]["mean_policy6_l1_vs_clean"],
        "cutout_metric": variant_metrics[VARIANT_CUTOUT_DEFENSE]["mean_policy6_l1_vs_clean"],
        "clean_metric": variant_metrics[VARIANT_CLEAN]["mean_action_l1_to_expert_first6"],
    }


def _dependency_report() -> dict[str, Any]:
    report = {
        "python_version": sys.version.split()[0],
        "modules": {
            name: _module_version(name)
            for name in ["torch", "transformers", "accelerate", "peft", "bitsandbytes", "lerobot", "h5py"]
        },
        "nvidia_smi": _nvidia_smi(),
        "torch_cuda": {},
        "runtime_dependencies": _runtime_dependencies(),
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
            "total_vram_mb": round(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024), 3)
            if torch.cuda.is_available()
            else None,
        }
        if torch.cuda.is_available():
            x = torch.ones((16, 16), device="cuda")
            y = (x @ x).sum()
            torch.cuda.synchronize()
            report["smokes"]["tiny_cuda_tensor"] = {
                "ok": True,
                "value": float(y.detach().cpu()),
                "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3),
            }
    except Exception as exc:  # noqa: BLE001
        report["smokes"]["tiny_cuda_tensor"] = {"ok": False, "error": _compact_error(exc)}
    try:
        import torch
        import bitsandbytes as bnb

        lin8 = bnb.nn.Linear8bitLt(8, 4, has_fp16_weights=False).cuda()
        inp = torch.randn(2, 8, device="cuda", dtype=torch.float16)
        out = lin8(inp)
        torch.cuda.synchronize()
        report["smokes"]["bitsandbytes_8bit_linear"] = {
            "ok": True,
            "shape": list(out.shape),
            "dtype": str(out.dtype),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3),
        }
        lin4 = bnb.nn.Linear4bit(8, 4, compute_dtype=torch.float16, quant_type="nf4").cuda()
        out4 = lin4(inp)
        torch.cuda.synchronize()
        report["smokes"]["bitsandbytes_4bit_linear"] = {
            "ok": True,
            "shape": list(out4.shape),
            "dtype": str(out4.dtype),
            "max_allocated_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3),
        }
    except Exception as exc:  # noqa: BLE001
        report["smokes"]["bitsandbytes_cuda_kernels"] = {"ok": False, "error": _compact_error(exc)}
    try:
        import torch
        from peft import LoraConfig, get_peft_model

        model = torch.nn.Sequential(torch.nn.Linear(8, 8), torch.nn.ReLU(), torch.nn.Linear(8, 4))
        wrapped = get_peft_model(
            model,
            LoraConfig(r=4, lora_alpha=8, target_modules=["0", "2"], lora_dropout=0.0, bias="none"),
        )
        out = wrapped(torch.randn(2, 8))
        report["smokes"]["peft_dummy_lora"] = {
            "ok": True,
            "output_shape": list(out.shape),
            "total_params": sum(p.numel() for p in wrapped.parameters()),
            "trainable_params": sum(p.numel() for p in wrapped.parameters() if p.requires_grad),
        }
    except Exception as exc:  # noqa: BLE001
        report["smokes"]["peft_dummy_lora"] = {"ok": False, "error": _compact_error(exc)}
    return report


def _load_lora_policy(
    *,
    smolvla_ckpt: Path,
    hf_home: Path,
    checkpoint_root: Path,
    device: str,
    rank: int,
) -> tuple[Any, Any, Path, dict[str, Any]]:
    from peft import LoraConfig, get_peft_model

    dependency_name = _read_tokenizer_dependency(smolvla_ckpt)
    external_dependency = _external_tokenizer_files(dependency_name, [hf_home, checkpoint_root])
    if not external_dependency.get("found"):
        raise FileNotFoundError("external tokenizer/VLM dependency root is missing")
    policy, config = _load_policy_with_vlm(smolvla_ckpt, hf_home, external_dependency, device)
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank * 2,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=0.0,
        bias="none",
    )
    policy = get_peft_model(policy, lora_config)
    return policy, config, Path(external_dependency["root"]), external_dependency


def _candidate_modules(policy: Any) -> dict[str, Any]:
    import torch

    records = []
    for name, module in policy.named_modules():
        if isinstance(module, torch.nn.Linear):
            low = name.lower()
            if any(key in low for key in ["lm_expert", "state_proj", "action_in_proj", "action_out_proj", "action_time"]):
                records.append({"name": name, "shape": [int(module.out_features), int(module.in_features)]})
    return {
        "target_modules": LORA_TARGET_MODULES,
        "candidate_count": len(records),
        "candidate_examples": records[-30:],
    }


def _train_variant(
    *,
    variant_name: str,
    policy: Any,
    config: Any,
    tokenizer_root: Path,
    hdf5_path: Path,
    demo_name: str | None,
    timesteps: list[int],
    task_text: str,
    device: str,
    max_steps: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    import torch

    policy.train()
    optimizer = torch.optim.AdamW((p for p in policy.parameters() if p.requires_grad), lr=learning_rate)
    losses: list[float] = []
    consistency_losses: list[float] = []
    train_variant_map = {
        "standard_clean_lora": [VARIANT_CLEAN],
        "generic_adv_aug_lora": [VARIANT_CLEAN, VARIANT_RANDOM_PATCH, VARIANT_FIXED_VISIBLE_PATCH],
        "patchguard_kinematic_lora": [VARIANT_CLEAN, VARIANT_RANDOM_PATCH, VARIANT_FIXED_VISIBLE_PATCH],
    }
    train_variants = train_variant_map[variant_name]
    for step in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        step_losses = []
        clean_loss_by_timestep: dict[int, Any] = {}
        for timestep in timesteps:
            for patch_variant in train_variants:
                batch, _metadata = _build_training_batch(
                    config=config,
                    tokenizer_root=tokenizer_root,
                    hdf5_path=hdf5_path,
                    demo_name=demo_name,
                    timestep=timestep,
                    task_text=task_text,
                    variant=patch_variant,
                    device=device,
                    seed=seed + step,
                )
                loss, _loss_dict = _forward_loss(policy, batch, config, device)
                step_losses.append(loss)
                if patch_variant == VARIANT_CLEAN:
                    clean_loss_by_timestep[timestep] = loss
        objective = torch.stack(step_losses).mean()
        consistency = torch.tensor(0.0, dtype=torch.float32, device=device)
        if variant_name == "patchguard_kinematic_lora":
            consistency_terms = []
            for timestep in timesteps:
                clean = clean_loss_by_timestep[timestep].detach()
                for patch_variant in [VARIANT_RANDOM_PATCH, VARIANT_FIXED_VISIBLE_PATCH]:
                    batch, _metadata = _build_training_batch(
                        config=config,
                        tokenizer_root=tokenizer_root,
                        hdf5_path=hdf5_path,
                        demo_name=demo_name,
                        timestep=timestep,
                        task_text=task_text,
                        variant=patch_variant,
                        device=device,
                        seed=seed + step + 123,
                    )
                    patched_loss, _loss_dict = _forward_loss(policy, batch, config, device)
                    consistency_terms.append(torch.abs(patched_loss - clean))
            consistency = torch.stack(consistency_terms).mean() if consistency_terms else consistency
            objective = objective + 0.25 * consistency
        objective.backward()
        optimizer.step()
        losses.append(round(float(objective.detach().cpu()), 6))
        consistency_losses.append(round(float(consistency.detach().cpu()), 6))
    return {
        "loss_start": losses[0] if losses else None,
        "loss_end": losses[-1] if losses else None,
        "loss_delta": round(float(losses[-1] - losses[0]), 6) if len(losses) > 1 else 0.0,
        "loss_trace": losses,
        "kinematic_consistency_loss_trace": consistency_losses,
    }


def _write_markdown_report(report: dict[str, Any], path: Path) -> None:
    summary = report.get("summary") or {}
    dep = report.get("dependency_check") or {}
    training = report.get("training_smoke") or {}
    lines = [
        "# PatchGuard-VLA STATE 1B Result",
        "",
        "Bounded environment and tiny LoRA feasibility gate. This is not a full benchmark, rollout, OpenVLA-OFT run, or paper claim.",
        "",
        f"- final decision: `{summary.get('final_decision')}`",
        f"- dependency install happened: `{summary.get('dependency_install_happened')}`",
        f"- CUDA available: `{((dep.get('torch_cuda') or {}).get('cuda_available'))}`",
        f"- GPU: `{((dep.get('torch_cuda') or {}).get('device_name'))}`",
        f"- PEFT: `{((dep.get('modules') or {}).get('peft') or {}).get('version')}`",
        f"- bitsandbytes: `{((dep.get('modules') or {}).get('bitsandbytes') or {}).get('version')}`",
        f"- bitsandbytes 4-bit smoke: `{(((dep.get('smokes') or {}).get('bitsandbytes_4bit_linear') or {}).get('ok'))}`",
        f"- LoRA injection happened: `{summary.get('lora_injection_happened')}`",
        f"- tiny training smoke happened: `{summary.get('tiny_training_smoke_happened')}`",
        f"- loss computed: `{summary.get('loss_computed')}`",
        f"- VRAM peak MB: `{summary.get('vram_peak_mb')}`",
        f"- runtime sec: `{summary.get('runtime_sec')}`",
        f"- clean metric: `{summary.get('clean_metric')}`",
        f"- patched metric: `{summary.get('patched_metric')}`",
        f"- cutout/random erasing metric: `{summary.get('cutout_metric')}`",
        f"- generic adversarial LoRA metric: `{summary.get('generic_adv_lora_metric')}`",
        f"- PatchGuard metric: `{summary.get('patchguard_metric')}`",
        f"- PatchGuard beats baseline: `{summary.get('patchguard_beats_baseline')}`",
        "",
        "## Variant Metrics",
        "",
        "| variant | attack divergence | clean metric | fixed patch | random patch | cutout | loss start | loss end |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, payload in (training.get("variants") or {}).items():
        eval_after = payload.get("eval_after") or {}
        train = payload.get("train") or {}
        lines.append(
            "| {name} | {attack} | {clean} | {fixed} | {random} | {cutout} | {start} | {end} |".format(
                name=name,
                attack=eval_after.get("attack_divergence_metric"),
                clean=eval_after.get("clean_metric"),
                fixed=eval_after.get("fixed_patch_metric"),
                random=eval_after.get("random_patch_metric"),
                cutout=eval_after.get("cutout_metric"),
                start=train.get("loss_start"),
                end=train.get("loss_end"),
            )
        )
    lines.extend(["", f"Exact next step: {summary.get('exact_next_step')}", ""])
    path.write_text("\n".join(lines), encoding="utf-8")


def build_report(args: argparse.Namespace) -> tuple[dict[str, Any], int]:
    started = time.monotonic()
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    smolvla_ckpt = Path(args.smolvla_ckpt)
    checkpoint_root = Path(args.checkpoint_root)
    hf_home = Path(args.hf_home)
    plan_path = Path(args.plan_report)
    forbidden = [name for name in FORBIDDEN_GATES if _env_flag(name)]
    max_steps = min(max(1, int(args.max_steps)), MAX_STEPS_CAP)

    report: dict[str, Any] = {
        "schema_version": "patchguard-vla-state1b-v1",
        "evidence_label": "patchguard_vla_state1b",
        "decision": "TOO_HEAVY_LOCAL",
        "policy": {
            "bounded_state1b": True,
            "task_local_gates_required": [
                f"{HEAVY_IMPORT_GATE}=1",
                f"{STATE1B_GATE}=1",
                f"{TRAINING_GATE}=1",
            ],
            "downloads_performed": False,
            "large_model_or_dataset_downloads_performed": False,
            "installs_performed_by_runner": False,
            "training_performed": False,
            "rollouts_performed": False,
            "benchmark_rollouts_performed": False,
            "openvla_oft_executed": False,
            "paper_grade_claims_made": False,
            "heavy_import_gate_set": _env_flag(HEAVY_IMPORT_GATE),
            "state1b_gate_set": _env_flag(STATE1B_GATE),
            "training_gate_set": _env_flag(TRAINING_GATE),
            "forbidden_gates_set": forbidden,
        },
        "claims": {
            "benchmark_success_claimed": False,
            "paper_grade_claim_made": False,
            "sota_claimed": False,
        },
        "paths": {
            "plan_report": str(plan_path),
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
        "adapter_probe": {},
        "training_smoke": {},
        "summary": {},
        "error": None,
    }

    def finish(decision: str, next_step: str, code: int) -> tuple[dict[str, Any], int]:
        if decision not in FINAL_DECISIONS:
            raise ValueError(f"invalid final decision: {decision}")
        report["decision"] = decision
        report["summary"]["final_decision"] = decision
        report["summary"]["exact_next_step"] = next_step
        report["summary"]["runtime_sec"] = round(time.monotonic() - started, 3)
        try:
            import torch

            report["summary"]["vram_peak_mb"] = (
                round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3) if torch.cuda.is_available() else 0.0
            )
        except Exception:
            report["summary"]["vram_peak_mb"] = None
        return report, code

    if not report["policy"]["heavy_import_gate_set"]:
        return finish("TOO_HEAVY_LOCAL", f"Set {HEAVY_IMPORT_GATE}=1 for this bounded check.", 2)
    if not report["policy"]["state1b_gate_set"]:
        return finish("TOO_HEAVY_LOCAL", f"Set {STATE1B_GATE}=1 for this bounded check.", 3)
    if forbidden:
        report["error"] = {"message": "Forbidden gate(s) set: " + ", ".join(forbidden)}
        return finish("TOO_HEAVY_LOCAL", "Clear forbidden rollout/download/OpenVLA-OFT gates and rerun STATE 1B.", 4)

    try:
        report["dependency_check"] = _dependency_report()
        modules = report["dependency_check"]["modules"]
        peft_ok = bool((modules.get("peft") or {}).get("import_ok"))
        bnb_ok = bool((modules.get("bitsandbytes") or {}).get("import_ok"))
        bnb_kernel_ok = bool(((report["dependency_check"].get("smokes") or {}).get("bitsandbytes_4bit_linear") or {}).get("ok"))
        cuda_ok = bool((report["dependency_check"].get("torch_cuda") or {}).get("cuda_available"))
        report["summary"]["dependency_install_happened"] = bool(args.dependency_install_happened)
        report["summary"]["peft_status"] = modules.get("peft")
        report["summary"]["bitsandbytes_status"] = modules.get("bitsandbytes")
        report["summary"]["cuda_gpu_status"] = report["dependency_check"].get("torch_cuda")
        if not peft_ok:
            return finish("ENV_BLOCKED_INSTALL_FAILED", "Install/import PEFT successfully before PatchGuard LoRA smoke.", 5)
        if not cuda_ok:
            return finish("TOO_HEAVY_LOCAL", "CUDA is unavailable; do not continue to local LoRA training.", 6)
        if not bnb_ok or not bnb_kernel_ok:
            report["summary"]["qlora_blocked_but_lora_possible"] = True
        if not plan_path.exists():
            report["error"] = {"message": f"Missing plan report: {plan_path}"}
            return finish("TOO_HEAVY_LOCAL", "Restore the STATE 1 plan report before training smoke.", 7)
        plan = _read_json(plan_path)
        hdf5_path, timesteps, task_text, demo_name = _plan_inputs(plan, args)
        if not hdf5_path.exists():
            report["error"] = {"message": f"Missing HDF5 file: {hdf5_path}"}
            return finish("TOO_HEAVY_LOCAL", "Restore local LIBERO HDF5 data before training smoke.", 8)
        if args.device != "cuda":
            return finish("TOO_HEAVY_LOCAL", "STATE 1B training smoke is CUDA-only after CUDA passed.", 9)
        if not report["policy"]["training_gate_set"]:
            return finish("TOO_HEAVY_LOCAL", f"Set {TRAINING_GATE}=1 to allow the bounded 10-step LoRA smoke.", 10)

        import torch

        torch.cuda.reset_peak_memory_stats()
        t_load = time.monotonic()
        policy, config, tokenizer_root, external_dependency = _load_lora_policy(
            smolvla_ckpt=smolvla_ckpt,
            hf_home=hf_home,
            checkpoint_root=checkpoint_root,
            device=args.device,
            rank=int(args.lora_rank),
        )
        report["policy"]["heavy_model_imports_performed"] = True
        report["policy"]["model_load_performed"] = True
        trainable_params = sum(p.numel() for p in policy.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in policy.parameters())
        report["adapter_probe"] = {
            "model_path_tested": str(smolvla_ckpt),
            "external_dependency": external_dependency,
            "load_elapsed_sec": round(time.monotonic() - t_load, 3),
            "lora_injection_happened": True,
            "lora_rank": int(args.lora_rank),
            "target_modules": LORA_TARGET_MODULES,
            "total_params": total_params,
            "trainable_params": trainable_params,
            "candidate_modules": _candidate_modules(policy),
            "estimated_vram_mb_after_load": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3),
        }
        report["summary"]["lora_injection_happened"] = True
        report["summary"]["model_path_tested"] = str(smolvla_ckpt)
        if trainable_params <= 0:
            return finish("KILL_NO_ADAPTER_PATH", "No trainable LoRA parameters were created.", 11)

        report["training_smoke"] = {
            "max_steps": max_steps,
            "learning_rate": float(args.learning_rate),
            "timesteps": timesteps,
            "task_text": task_text,
            "demo_name": demo_name,
            "variants": {},
        }
        base_eval = _evaluate_policy(
            policy=policy,
            config=config,
            tokenizer_root=tokenizer_root,
            hdf5_path=hdf5_path,
            demo_name=demo_name,
            timesteps=timesteps,
            task_text=task_text,
            device=args.device,
            seed=int(args.seed),
        )
        report["training_smoke"]["base_eval_before_training"] = {
            key: value for key, value in base_eval.items() if key != "samples"
        }
        del policy
        gc.collect()
        torch.cuda.empty_cache()

        for variant_name in TRAINING_VARIANTS:
            torch.cuda.reset_peak_memory_stats()
            policy, config, tokenizer_root, _external_dependency = _load_lora_policy(
                smolvla_ckpt=smolvla_ckpt,
                hf_home=hf_home,
                checkpoint_root=checkpoint_root,
                device=args.device,
                rank=int(args.lora_rank),
            )
            train_result = _train_variant(
                variant_name=variant_name,
                policy=policy,
                config=config,
                tokenizer_root=tokenizer_root,
                hdf5_path=hdf5_path,
                demo_name=demo_name,
                timesteps=timesteps,
                task_text=task_text,
                device=args.device,
                max_steps=max_steps,
                learning_rate=float(args.learning_rate),
                seed=int(args.seed),
            )
            report["policy"]["training_performed"] = True
            eval_after = _evaluate_policy(
                policy=policy,
                config=config,
                tokenizer_root=tokenizer_root,
                hdf5_path=hdf5_path,
                demo_name=demo_name,
                timesteps=timesteps,
                task_text=task_text,
                device=args.device,
                seed=int(args.seed),
            )
            report["training_smoke"]["variants"][variant_name] = {
                "train": train_result,
                "eval_after": {key: value for key, value in eval_after.items() if key != "samples"},
                "vram_peak_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 3),
            }
            del policy
            gc.collect()
            torch.cuda.empty_cache()

        variants = report["training_smoke"]["variants"]
        standard_metric = variants["standard_clean_lora"]["eval_after"]["attack_divergence_metric"]
        generic_metric = variants["generic_adv_aug_lora"]["eval_after"]["attack_divergence_metric"]
        patchguard_metric = variants["patchguard_kinematic_lora"]["eval_after"]["attack_divergence_metric"]
        cutout_metric = report["training_smoke"]["base_eval_before_training"]["cutout_metric"]
        clean_metric = variants["patchguard_kinematic_lora"]["eval_after"]["clean_metric"]
        variant_vram_peak_mb = max(
            variant_report["vram_peak_mb"] for variant_report in variants.values()
        )
        report["summary"].update(
            {
                "tiny_training_smoke_happened": True,
                "loss_computed": True,
                "vram_peak_mb": variant_vram_peak_mb,
                "clean_metric": clean_metric,
                "patched_metric": report["training_smoke"]["base_eval_before_training"]["attack_divergence_metric"],
                "standard_lora_metric": standard_metric,
                "generic_adv_lora_metric": generic_metric,
                "patchguard_metric": patchguard_metric,
                "cutout_metric": cutout_metric,
                "random_erasing_metric": cutout_metric,
                "patchguard_beats_baseline": bool(patchguard_metric < generic_metric and patchguard_metric < cutout_metric),
            }
        )
        if variant_vram_peak_mb > MAX_VRAM_MB:
            return finish("TOO_HEAVY_LOCAL", "Tiny LoRA smoke exceeded the 15GB VRAM limit.", 12)
        if not report["summary"]["patchguard_beats_baseline"]:
            return finish(
                "KILL_BASELINE_DOMINATED",
                "Do not proceed to STATE 2; PatchGuard did not beat generic adversarial augmentation and cutout baselines in the tiny smoke.",
                0,
            )
        return finish(
            "READY_FOR_PATCHGUARD_LORA_STATE2",
            "Next run should be an explicitly approved STATE 2 PatchGuard LoRA pilot with stronger patch optimization and held-out local samples.",
            0,
        )
    except Exception as exc:  # noqa: BLE001
        report["error"] = _compact_error(exc)
        message = str(exc).lower()
        if "out of memory" in message:
            return finish("TOO_HEAVY_LOCAL", "OOM at batch size 1/rank 4; stop local PatchGuard LoRA.", 13)
        return finish("KILL_NO_ADAPTER_PATH", "Fix or replace the SmolVLA LoRA adapter path before any training.", 14)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-report", default="reports/vlm_enabled_repeated_offline_decoding_plan_report.json")
    parser.add_argument("--smolvla-ckpt", default="C:/assets/checkpoints/smolvla")
    parser.add_argument("--checkpoint-root", default="C:/assets/checkpoints")
    parser.add_argument("--hf-home", default="C:/assets/hf_home")
    parser.add_argument("--hdf5-path", default="")
    parser.add_argument("--report-path", default="reports/patchguard_vla_state1b_result.json")
    parser.add_argument("--device", default="cuda", choices=["cuda"])
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=DEFAULT_MAX_STEPS)
    parser.add_argument("--max-train-samples", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dependency-install-happened", action="store_true")
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
    sys.exit(main())
