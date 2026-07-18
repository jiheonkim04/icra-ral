"""Execute the single frozen CVLR-XVLA Stage 0 protocol.

The run caches clean agent/wrist Florence2 tokens from the frozen data split,
trains only the preregistered cross-view predictor, and probes action outputs
on the frozen development panel.  It never performs a closed-loop rollout.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import inspect
import json
import math
import os
import pathlib
import random
import shutil
import subprocess
import sys
import time
import traceback
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tca_map.rifa_xvla.stage0 import (
    cuda_report,
    freeze_module,
    install_optional_xvla_shims,
    install_xvla_transformers_compat_patches,
    materialize_xvla_clip,
    matrix_to_rotate6d,
    memory_report,
    nvidia_smi,
    package_version,
    plan_to_libero_actions,
    prepare_live_inputs,
    prepare_offline_inputs,
    sha256_file,
    utcish_timestamp,
)

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = REPO_ROOT / "configs" / "cvlr_xvla_stage0_frozen_contract.json"
DEFAULT_REPORT_JSON = REPO_ROOT / "reports" / "cvlr_xvla_stage0_result.json"
DEFAULT_REPORT_MD = REPO_ROOT / "reports" / "cvlr_xvla_stage0_result.md"
DEFAULT_TELEMETRY_JSON = REPO_ROOT / "reports" / "cvlr_xvla_stage0_runtime_telemetry.json"
DEFAULT_CHECKPOINT_DIR = REPO_ROOT / "reports" / "checkpoints" / "cvlr_xvla_stage0"
EXPECTED_CONTRACT_SHA256 = "6767e529bd43a61760cd75ae8e4b05d235946fcbf4c5f8e05dbae5e35aa72746"
IMPLEMENTATION_LABEL = "CVLR_XVLA_FROZEN_STAGE0_LOCAL_IMPLEMENTATION"


def atomic_write_json(path: pathlib.Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def load_frozen_contract(path: pathlib.Path = DEFAULT_CONTRACT) -> dict[str, Any]:
    if sha256_file(path) != EXPECTED_CONTRACT_SHA256:
        raise ValueError("CVLR Stage 0 frozen contract hash drift")
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("method") != "CVLR_XVLA":
        raise ValueError("frozen method drift")
    if contract.get("execution_classification") != "OURS_VLA_TRAINING":
        raise ValueError("frozen execution classification drift")
    expected_panel = [
        ("libero_goal", 0, [20260733, 20260734, 20260735]),
        ("libero_object", 0, [20260733, 20260734, 20260735]),
        ("libero_spatial", 5, [20260731, 20260732, 20260735]),
    ]
    panel = [
        (row["suite"], int(row["task_id"]), list(row["identities"]))
        for row in contract["panel"]
    ]
    if panel != expected_panel:
        raise ValueError(f"frozen panel drift: {panel!r}")
    split = contract["data_split"]
    if split["training_demo_indices"] != "0..39" or split["validation_demo_indices"] != "40..49":
        raise ValueError("frozen train/validation split drift")
    if split["official_reader_sample_positions_per_demo"] != [0, 9, 18, 27]:
        raise ValueError("frozen sample positions drift")
    budget = contract["training_budget"]
    if int(budget["configuration_count"]) != 1 or int(budget["optimizer_steps_exact"]) != 96:
        raise ValueError("frozen configuration or optimizer budget drift")
    if bool(budget["downloads_allowed"]) or bool(budget["validation_selection_or_tuning"]):
        raise ValueError("downloads and outcome tuning are frozen off")
    safety = contract["action_safety_thresholds"]
    if bool(safety["universal_max_absolute_threshold_used"]):
        raise ValueError("binary gripper and continuous motion may not share a universal threshold")
    if int((contract["bounded_repair"])["current_count"]) != 0:
        raise ValueError("initial frozen contract unexpectedly records a repair")
    return contract


class CVLRLatentPredictor(nn.Module):
    """Low-rank predictor for the missing X-VLA wrist visual-token block."""

    def __init__(
        self,
        *,
        token_dim: int = 1024,
        token_count: int = 50,
        proprio_dim: int = 20,
        bottleneck_dim: int = 128,
        output_tanh_scale: float = 3.0,
    ) -> None:
        super().__init__()
        self.token_dim = int(token_dim)
        self.token_count = int(token_count)
        self.proprio_dim = int(proprio_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.output_tanh_scale = float(output_tanh_scale)
        self.agent_norm = nn.LayerNorm(self.token_dim)
        self.agent_down = nn.Linear(self.token_dim, self.bottleneck_dim)
        self.language_down = nn.Linear(self.token_dim, self.bottleneck_dim)
        self.proprio_down = nn.Linear(self.proprio_dim, self.bottleneck_dim)
        self.token_position = nn.Parameter(torch.zeros(self.token_count, self.bottleneck_dim))
        self.core = nn.Linear(self.bottleneck_dim, self.bottleneck_dim)
        self.output = nn.Linear(self.bottleneck_dim, self.token_dim)
        nn.init.normal_(self.token_position, mean=0.0, std=0.02)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(
        self,
        agent_tokens: torch.Tensor,
        language_mean: torch.Tensor,
        proprio: torch.Tensor,
    ) -> torch.Tensor:
        if agent_tokens.ndim != 3 or tuple(agent_tokens.shape[1:]) != (
            self.token_count,
            self.token_dim,
        ):
            raise ValueError(f"agent token shape drift: {tuple(agent_tokens.shape)}")
        if language_mean.ndim != 2 or language_mean.shape[-1] != self.token_dim:
            raise ValueError("language embedding shape drift")
        if proprio.ndim != 2 or proprio.shape[-1] != self.proprio_dim:
            raise ValueError("proprio shape drift")
        hidden = self.agent_down(self.agent_norm(agent_tokens))
        hidden = hidden + self.language_down(language_mean).unsqueeze(1)
        hidden = hidden + self.proprio_down(proprio).unsqueeze(1)
        hidden = hidden + self.token_position.unsqueeze(0)
        hidden = F.gelu(hidden)
        hidden = F.gelu(self.core(hidden))
        return torch.tanh(self.output(hidden)) * self.output_tanh_scale


class AuxVisualTokenHook:
    """Conditionally replace only X-VLA's wrist auxiliary token slice."""

    def __init__(self, wrist_token_count: int = 50) -> None:
        self.wrist_token_count = int(wrist_token_count)
        self.replacement: torch.Tensor | None = None
        self.missing = False
        self.forward_count = 0
        self.replacement_count = 0
        self.clean_bypass_count = 0

    def activate(self, replacement: torch.Tensor, *, missing: bool) -> None:
        self.replacement = replacement
        self.missing = bool(missing)

    def deactivate(self) -> None:
        self.replacement = None
        self.missing = False

    def __call__(
        self,
        _module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
    ) -> tuple[tuple[Any, ...], dict[str, Any]]:
        if self.replacement is None:
            return args, kwargs
        self.forward_count += 1
        if not self.missing:
            self.clean_bypass_count += 1
            return args, kwargs
        auxiliary = kwargs.get("aux_visual_inputs")
        if not isinstance(auxiliary, torch.Tensor):
            raise RuntimeError("X-VLA transformer did not expose aux_visual_inputs as a keyword tensor")
        replacement = self.replacement.to(device=auxiliary.device, dtype=auxiliary.dtype)
        expected = (auxiliary.shape[0], self.wrist_token_count, auxiliary.shape[-1])
        if tuple(replacement.shape) != tuple(expected):
            raise ValueError(f"replacement shape {tuple(replacement.shape)} != {tuple(expected)}")
        updated = auxiliary.clone()
        updated[:, : self.wrist_token_count, :] = replacement
        new_kwargs = dict(kwargs)
        new_kwargs["aux_visual_inputs"] = updated
        self.replacement_count += 1
        return args, new_kwargs


def trainable_parameter_count(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad))


def parameter_vector(module: nn.Module) -> torch.Tensor:
    values = [parameter.detach().float().reshape(-1).cpu() for parameter in module.parameters()]
    return torch.cat(values) if values else torch.empty(0)


def gradient_global_norm(module: nn.Module) -> tuple[float, int, int]:
    total = 0.0
    tensors = 0
    nonzero = 0
    for parameter in module.parameters():
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach().float()
        tensors += 1
        if not bool(torch.isfinite(grad).all().item()):
            return float("nan"), tensors, nonzero
        if bool(torch.count_nonzero(grad).item()):
            nonzero += 1
        total += float(torch.sum(grad * grad).item())
    return float(math.sqrt(total)), int(tensors), int(nonzero)


def task_key(task: dict[str, Any]) -> str:
    return f"{task['suite']}_task{int(task['task_id'])}"


def read_fixed_official_samples(meta_path: pathlib.Path, positions: list[int]) -> list[dict[str, Any]]:
    from datasets.dataset import InfiniteDataReader  # type: ignore

    desired = set(int(value) for value in positions)
    selected: dict[int, dict[str, Any]] = {}
    reader = InfiniteDataReader(str(meta_path), num_actions=30, num_views=3, training=False, action_mode="ee6d")
    for index, sample in enumerate(reader):
        if index in desired:
            selected[index] = sample
        if len(selected) == len(desired):
            break
    if set(selected) != desired:
        raise RuntimeError(f"official reader positions unavailable for {meta_path}: {sorted(selected)}")
    return [selected[index] for index in positions]


def frozen_predictor_inputs(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        agent = model.vlm._encode_image(inputs["image_input"][:, 0])
        language = model.vlm.get_input_embeddings()(inputs["input_ids"]).mean(dim=1)
    return agent.detach(), language.detach(), inputs["proprio"].detach()


def frozen_clean_latents(
    model: nn.Module,
    inputs: dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        features = model.vlm._encode_image(inputs["image_input"][:, :2].flatten(0, 1))
        agent = features[0:1]
        wrist = features[1:2]
        language = model.vlm.get_input_embeddings()(inputs["input_ids"]).mean(dim=1)
    return agent.detach(), wrist.detach(), language.detach(), inputs["proprio"].detach()


def generate_with_replacement(
    model: nn.Module,
    hook: AuxVisualTokenHook,
    inputs: dict[str, torch.Tensor],
    *,
    replacement: torch.Tensor | None,
    missing: bool,
    denoise_steps: int,
    seed: int,
) -> np.ndarray:
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    if replacement is None:
        hook.deactivate()
    else:
        hook.activate(replacement, missing=missing)
    try:
        with torch.no_grad():
            output = model.generate_actions(**inputs, steps=int(denoise_steps))
        return output.detach().float().cpu().numpy().squeeze(0)
    finally:
        hook.deactivate()


def semantic_action_delta(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
    left_plan = np.asarray(left, dtype=np.float32)
    right_plan = np.asarray(right, dtype=np.float32)
    left_actions = plan_to_libero_actions(left_plan)
    right_actions = plan_to_libero_actions(right_plan)
    translation = left_actions[:, :3] - right_actions[:, :3]
    rotation = left_actions[:, 3:6] - right_actions[:, 3:6]
    raw_gripper = left_plan[:, 9] - right_plan[:, 9]
    discrete_gripper = left_actions[:, 6] - right_actions[:, 6]
    return {
        "translation_rms": float(np.sqrt(np.mean(translation.astype(np.float64) ** 2))),
        "translation_max_abs": float(np.max(np.abs(translation))),
        "rotation_rms": float(np.sqrt(np.mean(rotation.astype(np.float64) ** 2))),
        "rotation_max_abs": float(np.max(np.abs(rotation))),
        "raw_gripper_mean_abs_delta": float(np.mean(np.abs(raw_gripper))),
        "raw_gripper_max_abs_delta": float(np.max(np.abs(raw_gripper))),
        "gripper_flip_count": int(np.count_nonzero(discrete_gripper)),
        "gripper_flip_indices": [int(value) for value in np.flatnonzero(discrete_gripper)],
        "finite": bool(
            np.isfinite(left_plan).all()
            and np.isfinite(right_plan).all()
            and np.isfinite(left_actions).all()
            and np.isfinite(right_actions).all()
        ),
    }


def summarize_semantic_deltas(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    values = [row[key] for row in rows]
    if not values:
        raise ValueError(f"no rows for semantic delta {key}")
    return {
        "count": len(values),
        "mean_translation_rms": float(np.mean([value["translation_rms"] for value in values])),
        "max_translation_rms": float(np.max([value["translation_rms"] for value in values])),
        "mean_rotation_rms": float(np.mean([value["rotation_rms"] for value in values])),
        "max_rotation_rms": float(np.max([value["rotation_rms"] for value in values])),
        "mean_raw_gripper_max_abs_delta": float(
            np.mean([value["raw_gripper_max_abs_delta"] for value in values])
        ),
        "max_raw_gripper_max_abs_delta": float(
            np.max([value["raw_gripper_max_abs_delta"] for value in values])
        ),
        "total_gripper_flips": int(sum(value["gripper_flip_count"] for value in values)),
        "all_finite": bool(all(value["finite"] for value in values)),
    }


def apply_stage0_decision(gates: dict[str, bool]) -> str:
    data = ["target_records_valid", "split_integrity"]
    implementation = [
        "real_xvla_forward_path",
        "cuda_execution",
        "trainable_parameter_count_exact",
        "finite_nonzero_gradients",
        "optimizer_steps_exact",
        "weights_changed",
        "checkpoint_write_and_disk_reload",
        "xvla_frozen",
        "wrist_insertion_path_active",
    ]
    design = ["exact_clean_bypass", "semantic_action_safety", "action_outputs_finite"]
    mechanism = [
        "reconstruction_meaningfully_beats_controls",
        "prediction_noncollapsed",
        "meaningful_full_vs_no_reconstruction_action_effect",
    ]
    if any(not bool(gates.get(name, False)) for name in data):
        return "CVLR_XVLA_STAGE0_DATA_OR_SUPERVISION_FAILURE"
    if any(not bool(gates.get(name, False)) for name in implementation):
        return "CVLR_XVLA_STAGE0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
    if any(not bool(gates.get(name, False)) for name in design):
        return "CVLR_XVLA_STAGE0_DESIGN_FAILURE"
    if any(not bool(gates.get(name, False)) for name in mechanism):
        return "CVLR_XVLA_STAGE0_KEY_COMPONENT_NOT_USEFUL"
    return "CVLR_XVLA_STAGE0_PASS"


def write_markdown(path: pathlib.Path, result: dict[str, Any]) -> None:
    training = result.get("training") or {}
    reconstruction = result.get("reconstruction_validation") or {}
    action = result.get("action_validation") or {}
    lines = [
        "# CVLR-XVLA Stage 0 Result",
        "",
        f"- Decision: `{result.get('decision')}`",
        f"- Execution valid: `{result.get('execution_valid')}`",
        f"- CUDA PID: `{result.get('cuda_pid')}`",
        f"- Peak VRAM MiB: `{(result.get('cuda') or {}).get('max_allocated_mib')}`",
        f"- Trainable parameters: `{training.get('trainable_parameter_count')}`",
        f"- Optimizer steps: `{training.get('optimizer_steps')}`",
        f"- Validation MSE full / zero / AWF: `{reconstruction.get('full_mse_mean')} / "
        f"{reconstruction.get('zero_mse_mean')} / {reconstruction.get('awf_mse_mean')}`",
        f"- Clean translation / rotation max RMS: `"
        f"{((action.get('clean_full_vs_base') or {}).get('max_translation_rms'))} / "
        f"{((action.get('clean_full_vs_base') or {}).get('max_rotation_rms'))}`",
        f"- Dropout gripper flips full vs Base: `"
        f"{((action.get('dropout_full_vs_base') or {}).get('total_gripper_flips'))}`",
        "",
        "## Frozen gates",
        "",
        "| gate | pass |",
        "|---|---|",
    ]
    for name, passed in (result.get("gates") or {}).items():
        lines.append(f"| `{name}` | `{passed}` |")
    lines += [
        "",
        "Continuous translation, continuous rotation, raw gripper score, and final discrete gripper flips were evaluated separately.",
        "No closed-loop rollout, official success measurement, threshold tuning, or privileged inference input occurred in Stage 0.",
    ]
    if result.get("exceptions"):
        lines += ["", "## Exceptions", "", "```json", json.dumps(result["exceptions"], indent=2), "```"]
    write_text(path, "\n".join(lines) + "\n")


@dataclass
class RuntimePaths:
    run_dir: pathlib.Path
    status: pathlib.Path
    heartbeat: pathlib.Path
    partial: pathlib.Path
    result: pathlib.Path
    result_md: pathlib.Path
    exit_code: pathlib.Path


def run_stage0(run_dir: pathlib.Path, contract_path: pathlib.Path = DEFAULT_CONTRACT) -> tuple[int, dict[str, Any]]:
    started = time.monotonic()
    run_dir.mkdir(parents=True, exist_ok=True)
    paths = RuntimePaths(
        run_dir=run_dir,
        status=run_dir / "status.json",
        heartbeat=run_dir / "heartbeat.json",
        partial=run_dir / "partial_result.json",
        result=run_dir / "result.json",
        result_md=run_dir / "result.md",
        exit_code=run_dir / "exit_code.txt",
    )
    contract = load_frozen_contract(contract_path)
    result: dict[str, Any] = {
        "schema_version": "2026-07-18.epoch5_cvlr_xvla_stage0_result.v1",
        "execution_classification": "OURS_VLA_TRAINING",
        "implementation_label": IMPLEMENTATION_LABEL,
        "method": "CVLR_XVLA",
        "stage": "frozen_stage0_latent_reconstruction_and_action_smoke",
        "run_dir": str(run_dir),
        "pid": int(os.getpid()),
        "cuda_pid": int(os.getpid()) if torch.cuda.is_available() else None,
        "contract": {"path": str(contract_path), "sha256": sha256_file(contract_path)},
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "started_at": utcish_timestamp(),
        "execution_valid": False,
        "decision": "CVLR_XVLA_STAGE0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE",
        "exceptions": [],
        "repair_count": int(contract["bounded_repair"]["current_count"]),
        "downloads_used": False,
        "model_offload_used": False,
        "no_closed_loop_rollout": True,
        "official_closed_loop_success_measured": False,
        "no_threshold_tuning": True,
        "no_privileged_inference_input": True,
        "rifa_reopened": False,
        "broad_candidate_search_reopened": False,
        "natural_reset_mining_reopened": False,
        "nvidia_smi_before": nvidia_smi(),
    }

    def heartbeat(stage: str) -> None:
        payload = {"timestamp": utcish_timestamp(), "stage": stage, "pid": int(os.getpid())}
        atomic_write_json(paths.heartbeat, payload)
        atomic_write_json(paths.status, {**payload, "state": "running"})
        atomic_write_json(
            paths.partial,
            {
                "schema_version": result["schema_version"],
                "method": result["method"],
                "stage": stage,
                "pid": int(os.getpid()),
                "contract_sha256": result["contract"]["sha256"],
                "elapsed_seconds": round(time.monotonic() - started, 3),
            },
        )

    atomic_write_json(
        run_dir / "launch_manifest.json",
        {
            "schema_version": "2026-07-18.epoch5_cvlr_xvla_stage0_launch.v1",
            "source_head": result["source_head"],
            "contract_path": str(contract_path),
            "contract_sha256": result["contract"]["sha256"],
            "python": sys.executable,
            "argv": sys.argv,
            "pid": int(os.getpid()),
            "panel": contract["panel"],
            "training_budget": contract["training_budget"],
            "thresholds": {
                "reconstruction": contract["reconstruction_thresholds"],
                "action_effect": contract["action_effect_thresholds"],
                "action_safety": contract["action_safety_thresholds"],
                "validity": contract["validity_thresholds"],
            },
            "closed_loop_rollout_authorized": False,
        },
    )
    write_text(run_dir / "worker_pid.txt", f"{os.getpid()}\n")
    write_text(
        run_dir / "exact_resume_command.txt",
        f"{sys.executable} scripts/run_cvlr_xvla_stage0.py --run-dir {run_dir} --contract {contract_path}\n",
    )

    model: nn.Module | None = None
    hook_handle: Any = None
    env: Any = None
    try:
        heartbeat("risk_and_contract_validation")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable for frozen CVLR Stage 0")
        if not all(pathlib.Path(task["hdf5"]).is_file() for task in contract["panel"]):
            raise FileNotFoundError("one or more frozen LIBERO HDF5 files are missing")
        device = torch.device("cuda:0")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        seed = int(contract["training_budget"]["seed"])
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        result["risk_assessment"] = {
            "source": "tracked local code and already-present X-VLA/LIBERO assets",
            "downloads": "disabled",
            "dataset_files_present": True,
            "cuda_before": cuda_report(),
            "system_ram_before": memory_report(),
            "no_cpu_or_disk_model_offload": True,
        }

        heartbeat("materialize_frozen_clean_records")
        xvla_root = pathlib.Path(contract["xvla"]["source_root"])
        if str(xvla_root) in sys.path:
            sys.path.remove(str(xvla_root))
        sys.path.insert(0, str(xvla_root))
        shims = install_optional_xvla_shims()
        materialized_root = run_dir / "materialized_fixed_samples"
        materialized_root.mkdir(parents=True, exist_ok=False)
        raw_records: list[dict[str, Any]] = []
        materialized_rows: list[dict[str, Any]] = []
        split = contract["data_split"]
        positions = list(split["official_reader_sample_positions_per_demo"])
        for task in contract["panel"]:
            for split_name, indices in (
                ("training", split["stage0_training_demo_indices_per_task"]),
                ("validation", split["stage0_validation_demo_indices_per_task"]),
            ):
                for demo_index in indices:
                    sample_dir = materialized_root / f"{task_key(task)}_{split_name}_demo{int(demo_index)}"
                    materialized = materialize_xvla_clip(
                        pathlib.Path(task["hdf5"]),
                        sample_dir,
                        demo_index=int(demo_index),
                        instruction=str(task["instruction"]),
                        clip_steps=int(split["materialized_clip_steps"]),
                    )
                    materialized.pop("agent_frame")
                    materialized.pop("wrist_frame")
                    for position, sample in zip(
                        positions,
                        read_fixed_official_samples(pathlib.Path(materialized["meta_path"]), positions),
                    ):
                        raw_records.append(
                            {
                                "task": task,
                                "split": split_name,
                                "demo_index": int(demo_index),
                                "reader_position": int(position),
                                "sample": sample,
                            }
                        )
                    materialized_rows.append(materialized)
        training_raw = [row for row in raw_records if row["split"] == "training"]
        validation_raw = [row for row in raw_records if row["split"] == "validation"]
        result["data"] = {
            "training_record_count": len(training_raw),
            "validation_record_count": len(validation_raw),
            "training_demo_indices": split["stage0_training_demo_indices_per_task"],
            "validation_demo_indices": split["stage0_validation_demo_indices_per_task"],
            "official_reader_sample_positions": positions,
            "materialized": materialized_rows,
            "split_overlap": bool(
                set(split["stage0_training_demo_indices_per_task"])
                & set(split["stage0_validation_demo_indices_per_task"])
            ),
            "confirmatory_test_data_used": False,
        }

        heartbeat("load_frozen_xvla")
        result["optional_import_shims_used"] = shims
        result["transformers_compat_patches"] = install_xvla_transformers_compat_patches()
        from models.modeling_xvla import XVLA  # type: ignore
        from models.processing_xvla import XVLAProcessor  # type: ignore

        xvla = contract["xvla"]
        model = XVLA.from_pretrained(
            xvla["model_id"],
            revision=xvla["model_revision"],
            trust_remote_code=True,
            torch_dtype=torch.float32,
            local_files_only=True,
            cache_dir=xvla["cache_dir"],
        )
        processor = XVLAProcessor.from_pretrained(
            xvla["model_id"],
            revision=xvla["model_revision"],
            trust_remote_code=True,
            local_files_only=True,
            cache_dir=xvla["cache_dir"],
        )
        freeze_module(model)
        model.to(device=device, dtype=torch.float32)
        model.eval()
        if "aux_visual_inputs" not in inspect.signature(model.transformer.forward).parameters:
            raise RuntimeError("X-VLA transformer insertion keyword unavailable")
        hook = AuxVisualTokenHook(int(xvla["visual_tokens_per_view"]))
        hook_handle = model.transformer.register_forward_pre_hook(hook, with_kwargs=True)
        result["xvla"] = {
            "model_id": xvla["model_id"],
            "model_revision": xvla["model_revision"],
            "source_revision": xvla["source_revision"],
            "model_class": type(model).__name__,
            "processor_class": type(processor).__name__,
            "device": str(next(model.parameters()).device),
            "dtype": str(next(model.parameters()).dtype),
            "base_trainable_parameter_count": trainable_parameter_count(model),
            "insertion_point": xvla["insertion_point"],
        }
        result["runtime_dependencies"] = {
            name: package_version(name) for name in ["torch", "transformers", "h5py", "scipy"]
        }

        heartbeat("cache_frozen_clean_latents")
        cached: list[dict[str, Any]] = []
        for row in raw_records:
            inputs = prepare_offline_inputs(row["sample"], processor, device, condition="clean")
            agent, wrist, language, proprio = frozen_clean_latents(model, inputs)
            cached.append(
                {
                    "task_key": task_key(row["task"]),
                    "split": row["split"],
                    "demo_index": row["demo_index"],
                    "reader_position": row["reader_position"],
                    "agent": agent.cpu(),
                    "wrist": wrist.cpu(),
                    "language": language.cpu(),
                    "proprio": proprio.cpu(),
                }
            )
        training_records = [row for row in cached if row["split"] == "training"]
        validation_records = [row for row in cached if row["split"] == "validation"]
        target_values = torch.cat([row["wrist"].reshape(-1) for row in cached])
        result["latent_targets"] = {
            "shape": list(training_records[0]["wrist"].shape[1:]),
            "all_finite": bool(torch.isfinite(target_values).all()),
            "nonzero_count": int(torch.count_nonzero(target_values)),
            "element_std": float(target_values.float().std()),
            "record_count": len(cached),
        }

        low_compute = contract["low_compute_parameterization"]
        predictor = CVLRLatentPredictor(
            token_dim=int(low_compute["visual_token_dim"]),
            token_count=int(low_compute["visual_token_count"]),
            proprio_dim=int(low_compute["proprio_dim"]),
            bottleneck_dim=int(low_compute["bottleneck_dim"]),
            output_tanh_scale=float(low_compute["output_tanh_scale"]),
        ).to(device)
        initial_output_zero = bool(
            torch.count_nonzero(predictor.output.weight).item() == 0
            and torch.count_nonzero(predictor.output.bias).item() == 0
        )
        initial_parameters = parameter_vector(predictor)
        budget = contract["training_budget"]
        optimizer = torch.optim.AdamW(
            predictor.parameters(),
            lr=float(budget["learning_rate"]),
            weight_decay=float(budget["weight_decay"]),
        )

        heartbeat("train_frozen_cvlr_predictor")
        predictor.train()
        losses: list[float] = []
        gradient_norms: list[float] = []
        gradient_tensor_counts: list[int] = []
        nonzero_gradient_tensor_counts: list[int] = []
        batch_size = int(budget["batch_size"])
        steps = int(budget["optimizer_steps_exact"])
        for step in range(steps):
            indices = [int((step * batch_size + offset) % len(training_records)) for offset in range(batch_size)]
            batch = [training_records[index] for index in indices]
            agent = torch.cat([row["agent"] for row in batch]).to(device)
            wrist = torch.cat([row["wrist"] for row in batch]).to(device)
            language = torch.cat([row["language"] for row in batch]).to(device)
            proprio = torch.cat([row["proprio"] for row in batch]).to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = predictor(agent, language, proprio)
            loss = F.mse_loss(prediction, wrist)
            if not bool(torch.isfinite(loss).item()):
                raise RuntimeError(f"nonfinite CVLR loss at step {step + 1}")
            loss.backward()
            norm, tensor_count, nonzero_count = gradient_global_norm(predictor)
            if not np.isfinite(norm):
                raise RuntimeError(f"nonfinite CVLR gradients at step {step + 1}")
            torch.nn.utils.clip_grad_norm_(predictor.parameters(), float(budget["max_grad_norm"]))
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(norm))
            gradient_tensor_counts.append(int(tensor_count))
            nonzero_gradient_tensor_counts.append(int(nonzero_count))
            if (step + 1) % 8 == 0 or step + 1 == steps:
                heartbeat(f"train_step_{step + 1}_of_{steps}")

        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "cvlr_xvla_stage0.pt"
        torch.save(predictor.state_dict(), checkpoint_path)
        final_parameters = parameter_vector(predictor)
        reloaded = CVLRLatentPredictor(
            token_dim=int(low_compute["visual_token_dim"]),
            token_count=int(low_compute["visual_token_count"]),
            proprio_dim=int(low_compute["proprio_dim"]),
            bottleneck_dim=int(low_compute["bottleneck_dim"]),
            output_tanh_scale=float(low_compute["output_tanh_scale"]),
        ).to(device)
        reloaded.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        reload_delta = float(torch.norm(parameter_vector(reloaded) - final_parameters).item())
        predictor = reloaded.eval()
        result["training"] = {
            "trainable_parameter_count": trainable_parameter_count(predictor),
            "optimizer_steps": len(losses),
            "first_loss": losses[0],
            "final_loss": losses[-1],
            "losses": losses,
            "gradient_norms": gradient_norms,
            "gradient_tensor_counts": gradient_tensor_counts,
            "nonzero_gradient_tensor_counts": nonzero_gradient_tensor_counts,
            "finite_nonzero_gradients": bool(
                all(np.isfinite(value) and value > 0.0 for value in gradient_norms)
                and all(value > 0 for value in nonzero_gradient_tensor_counts)
            ),
            "weight_change_l2": float(torch.norm(final_parameters - initial_parameters).item()),
            "weights_changed": bool(torch.count_nonzero(final_parameters - initial_parameters).item()),
            "initial_output_projection_exact_zero": initial_output_zero,
            "checkpoint": {
                "path": str(checkpoint_path),
                "sha256": sha256_file(checkpoint_path),
                "bytes": int(checkpoint_path.stat().st_size),
                "reload_parameter_delta_l2": reload_delta,
                "disk_reload_ok": reload_delta == 0.0,
            },
        }

        heartbeat("evaluate_frozen_reconstruction_validation")
        reconstruction_rows: list[dict[str, Any]] = []
        for row in validation_records:
            agent = row["agent"].to(device)
            wrist = row["wrist"].to(device)
            language = row["language"].to(device)
            proprio = row["proprio"].to(device)
            with torch.no_grad():
                full = predictor(agent, language, proprio)
                zero = torch.zeros_like(wrist)
                awf = agent
            reconstruction_rows.append(
                {
                    "task_key": row["task_key"],
                    "demo_index": row["demo_index"],
                    "reader_position": row["reader_position"],
                    "full_mse": float(F.mse_loss(full, wrist).cpu()),
                    "zero_mse": float(F.mse_loss(zero, wrist).cpu()),
                    "awf_mse": float(F.mse_loss(awf, wrist).cpu()),
                    "full_cosine": float(F.cosine_similarity(full.flatten(), wrist.flatten(), dim=0).cpu()),
                    "awf_cosine": float(F.cosine_similarity(awf.flatten(), wrist.flatten(), dim=0).cpu()),
                    "predicted_element_std": float(full.float().std().cpu()),
                    "target_element_std": float(wrist.float().std().cpu()),
                    "finite": bool(torch.isfinite(full).all().item()),
                }
            )
        full_mse = float(np.mean([row["full_mse"] for row in reconstruction_rows]))
        zero_mse = float(np.mean([row["zero_mse"] for row in reconstruction_rows]))
        awf_mse = float(np.mean([row["awf_mse"] for row in reconstruction_rows]))
        per_task: dict[str, Any] = {}
        for key in sorted({row["task_key"] for row in reconstruction_rows}):
            task_rows = [row for row in reconstruction_rows if row["task_key"] == key]
            per_task[key] = {
                "full_mse": float(np.mean([row["full_mse"] for row in task_rows])),
                "zero_mse": float(np.mean([row["zero_mse"] for row in task_rows])),
                "awf_mse": float(np.mean([row["awf_mse"] for row in task_rows])),
            }
            per_task[key]["full_beats_both"] = bool(
                per_task[key]["full_mse"] < per_task[key]["zero_mse"]
                and per_task[key]["full_mse"] < per_task[key]["awf_mse"]
            )
        tasks_beating_both = sum(int(row["full_beats_both"]) for row in per_task.values())
        result["reconstruction_validation"] = {
            "rows": reconstruction_rows,
            "record_count": len(reconstruction_rows),
            "full_mse_mean": full_mse,
            "zero_mse_mean": zero_mse,
            "awf_mse_mean": awf_mse,
            "full_relative_to_better_control": full_mse / min(zero_mse, awf_mse),
            "full_cosine_mean": float(np.mean([row["full_cosine"] for row in reconstruction_rows])),
            "awf_cosine_mean": float(np.mean([row["awf_cosine"] for row in reconstruction_rows])),
            "predicted_element_std_mean": float(
                np.mean([row["predicted_element_std"] for row in reconstruction_rows])
            ),
            "tasks_full_beats_both_controls": int(tasks_beating_both),
            "per_task": per_task,
        }

        heartbeat("evaluate_frozen_live_action_panel")
        os.environ.setdefault("MUJOCO_GL", "egl")
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv

        action_rows: list[dict[str, Any]] = []
        generation_calls = 0
        predictor_context_calls = 0
        for task in contract["panel"]:
            suite = benchmark.get_benchmark_dict()[str(task["suite"])]()
            libero_task = suite.get_task(int(task["task_id"]))
            bddl = pathlib.Path(get_libero_path("bddl_files")) / libero_task.problem_folder / libero_task.bddl_file
            initial_states = suite.get_task_init_states(int(task["task_id"]))
            for identity in task["identities"]:
                identity = int(identity)
                index = identity - 20260711
                env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=128, camera_widths=128)
                try:
                    env.seed(identity)
                    env.reset()
                    obs = env.set_init_state(np.asarray(initial_states[index], dtype=np.float64))
                    for _ in range(10):
                        obs, _reward, _done, _info = env.step(
                            np.asarray([0, 0, 0, 0, 0, 0, -1], dtype=np.float32)
                        )
                    obs["robo_ori"] = matrix_to_rotate6d(env.env.robots[0].controller.ee_ori_mat)
                    obs["robo_pos"] = np.asarray(env.env.robots[0].controller.ee_pos, dtype=np.float32)
                    for condition in contract["conditions"]:
                        inputs = prepare_live_inputs(
                            obs,
                            str(task["instruction"]),
                            processor,
                            device,
                            condition=condition,
                        )
                        agent, language, proprio = frozen_predictor_inputs(model, inputs)
                        predictor_context_calls += 1
                        with torch.no_grad():
                            predicted = predictor(agent, language, proprio)
                        zero = torch.zeros_like(predicted)
                        awf = agent
                        eval_seed = seed + 4000 + len(action_rows)
                        base = generate_with_replacement(
                            model,
                            hook,
                            inputs,
                            replacement=None,
                            missing=False,
                            denoise_steps=int(xvla["denoise_steps"]),
                            seed=eval_seed,
                        )
                        full = generate_with_replacement(
                            model,
                            hook,
                            inputs,
                            replacement=predicted,
                            missing=condition == "mask_1_in_hand_dropout",
                            denoise_steps=int(xvla["denoise_steps"]),
                            seed=eval_seed,
                        )
                        generation_calls += 2
                        output: dict[str, Any] = {
                            "source": "official_live_initial_observation",
                            "suite": task["suite"],
                            "task_id": int(task["task_id"]),
                            "reset_identity": identity,
                            "initial_state_index": int(index),
                            "condition": condition,
                            "full_vs_base": semantic_action_delta(full, base),
                            "raw_gripper_scores": {
                                "BASE": [float(value) for value in base[:, 9]],
                                "CVLR_XVLA": [float(value) for value in full[:, 9]],
                            },
                            "final_discrete_gripper": {
                                "BASE": [float(value) for value in plan_to_libero_actions(base)[:, 6]],
                                "CVLR_XVLA": [float(value) for value in plan_to_libero_actions(full)[:, 6]],
                            },
                            "legal_deployment_inputs_only": True,
                        }
                        if condition == "mask_1_in_hand_dropout":
                            zero_plan = generate_with_replacement(
                                model,
                                hook,
                                inputs,
                                replacement=zero,
                                missing=True,
                                denoise_steps=int(xvla["denoise_steps"]),
                                seed=eval_seed,
                            )
                            awf_plan = generate_with_replacement(
                                model,
                                hook,
                                inputs,
                                replacement=awf,
                                missing=True,
                                denoise_steps=int(xvla["denoise_steps"]),
                                seed=eval_seed,
                            )
                            generation_calls += 2
                            output["full_vs_zero"] = semantic_action_delta(full, zero_plan)
                            output["full_vs_awf"] = semantic_action_delta(full, awf_plan)
                            output["zero_vs_base"] = semantic_action_delta(zero_plan, base)
                            output["awf_vs_base"] = semantic_action_delta(awf_plan, base)
                            output["raw_gripper_scores"].update(
                                {
                                    "CVLR_NO_RECONSTRUCTION_ZERO_FILL": [
                                        float(value) for value in zero_plan[:, 9]
                                    ],
                                    "AWF_DETERMINISTIC_AGENT_TOKEN_FILL": [
                                        float(value) for value in awf_plan[:, 9]
                                    ],
                                }
                            )
                            output["final_discrete_gripper"].update(
                                {
                                    "CVLR_NO_RECONSTRUCTION_ZERO_FILL": [
                                        float(value) for value in plan_to_libero_actions(zero_plan)[:, 6]
                                    ],
                                    "AWF_DETERMINISTIC_AGENT_TOKEN_FILL": [
                                        float(value) for value in plan_to_libero_actions(awf_plan)[:, 6]
                                    ],
                                }
                            )
                        action_rows.append(output)
                        heartbeat(f"action_probe_{task_key(task)}_{identity}_{condition}")
                finally:
                    env.close()
                    env = None

        clean_rows = [row for row in action_rows if row["condition"] == "clean"]
        dropout_rows = [row for row in action_rows if row["condition"] == "mask_1_in_hand_dropout"]
        clean_summary = summarize_semantic_deltas(clean_rows, "full_vs_base")
        dropout_base_summary = summarize_semantic_deltas(dropout_rows, "full_vs_base")
        dropout_zero_summary = summarize_semantic_deltas(dropout_rows, "full_vs_zero")
        dropout_awf_summary = summarize_semantic_deltas(dropout_rows, "full_vs_awf")
        result["action_validation"] = {
            "rows": action_rows,
            "live_initial_observation_row_count": len(action_rows),
            "live_simulator_reset_probe_count": 9,
            "simulator_episode_count": 0,
            "closed_loop_action_step_count": 0,
            "clean_full_vs_base": clean_summary,
            "dropout_full_vs_base": dropout_base_summary,
            "dropout_full_vs_zero": dropout_zero_summary,
            "dropout_full_vs_awf": dropout_awf_summary,
            "semantic_dimensions_reported_separately": True,
            "universal_max_absolute_threshold_used": False,
        }
        result["forward_counts"] = {
            "xvla_visual_encoder_context_calls": len(cached) + predictor_context_calls,
            "xvla_generate_actions_calls": generation_calls,
            "xvla_action_transformer_calls": generation_calls * int(xvla["denoise_steps"]),
            "cvlr_predictor_training_calls": steps,
            "cvlr_predictor_validation_calls": len(validation_records),
            "cvlr_predictor_live_context_calls": predictor_context_calls,
            "aux_hook_forward_calls": int(hook.forward_count),
            "aux_hook_replacement_calls": int(hook.replacement_count),
            "aux_hook_clean_bypass_calls": int(hook.clean_bypass_count),
        }

        reconstruction_thresholds = contract["reconstruction_thresholds"]
        effect = contract["action_effect_thresholds"]
        safety = contract["action_safety_thresholds"]
        validity = contract["validity_thresholds"]
        result["gates"] = {
            "target_records_valid": bool(
                result["latent_targets"]["all_finite"]
                and result["latent_targets"]["nonzero_count"] > 0
                and len(reconstruction_rows) == int(reconstruction_thresholds["target_record_count_exact"])
            ),
            "split_integrity": bool(
                not result["data"]["split_overlap"]
                and len(training_records) == int(split["expected_training_records"])
                and len(validation_records) == int(split["expected_validation_records"])
            ),
            "real_xvla_forward_path": bool(
                result["xvla"]["model_class"] == "XVLA"
                and result["forward_counts"]["xvla_generate_actions_calls"]
                >= int(validity["real_xvla_forward_count_min"])
            ),
            "cuda_execution": bool(
                torch.cuda.is_available() and str(result["xvla"]["device"]).startswith("cuda")
            ),
            "trainable_parameter_count_exact": bool(
                result["training"]["trainable_parameter_count"]
                == int(validity["trainable_parameter_count_exact"])
            ),
            "finite_nonzero_gradients": bool(result["training"]["finite_nonzero_gradients"]),
            "optimizer_steps_exact": bool(
                result["training"]["optimizer_steps"] == int(validity["optimizer_steps_exact"])
            ),
            "weights_changed": bool(
                result["training"]["weights_changed"]
                and result["training"]["weight_change_l2"]
                > float(validity["weight_change_l2_min_exclusive"])
            ),
            "checkpoint_write_and_disk_reload": bool(
                result["training"]["checkpoint"]["disk_reload_ok"]
                and result["training"]["checkpoint"]["reload_parameter_delta_l2"]
                <= float(validity["checkpoint_reload_parameter_delta_l2_max"])
            ),
            "xvla_frozen": bool(
                result["xvla"]["base_trainable_parameter_count"]
                <= int(validity["frozen_xvla_trainable_parameter_count_max"])
            ),
            "wrist_insertion_path_active": bool(
                result["forward_counts"]["aux_hook_replacement_calls"] > 0
                and result["forward_counts"]["aux_hook_clean_bypass_calls"] > 0
            ),
            "reconstruction_meaningfully_beats_controls": bool(
                result["reconstruction_validation"]["full_relative_to_better_control"]
                <= float(reconstruction_thresholds["full_mse_relative_to_better_control_max"])
                and result["reconstruction_validation"]["tasks_full_beats_both_controls"]
                >= int(reconstruction_thresholds["tasks_with_full_mse_better_than_both_controls_min"])
            ),
            "prediction_noncollapsed": bool(
                result["reconstruction_validation"]["predicted_element_std_mean"]
                > float(reconstruction_thresholds["predicted_token_element_std_min_exclusive"])
            ),
            "meaningful_full_vs_no_reconstruction_action_effect": bool(
                dropout_zero_summary["mean_translation_rms"]
                >= float(effect["full_vs_zero_mean_translation_rms_min"])
                or dropout_zero_summary["mean_rotation_rms"]
                >= float(effect["full_vs_zero_mean_rotation_rms_min"])
                or dropout_zero_summary["mean_raw_gripper_max_abs_delta"]
                >= float(effect["full_vs_zero_mean_raw_gripper_max_abs_delta_min"])
            ),
            "exact_clean_bypass": bool(
                clean_summary["max_translation_rms"]
                <= float(safety["clean_full_vs_base_translation_rms_max"])
                and clean_summary["max_rotation_rms"]
                <= float(safety["clean_full_vs_base_rotation_rms_max"])
                and clean_summary["max_raw_gripper_max_abs_delta"]
                <= float(safety["clean_full_vs_base_raw_gripper_max_abs_delta_max"])
                and clean_summary["total_gripper_flips"]
                <= int(safety["clean_full_vs_base_gripper_flip_count_max"])
            ),
            "semantic_action_safety": bool(
                dropout_base_summary["max_translation_rms"]
                <= float(safety["dropout_full_vs_base_translation_rms_max"])
                and dropout_base_summary["max_rotation_rms"]
                <= float(safety["dropout_full_vs_base_rotation_rms_max"])
                and dropout_base_summary["max_raw_gripper_max_abs_delta"]
                <= float(safety["dropout_full_vs_base_raw_gripper_max_abs_delta_max"])
                and dropout_base_summary["total_gripper_flips"]
                <= int(safety["dropout_full_vs_base_gripper_flip_count_max"])
            ),
            "action_outputs_finite": bool(
                clean_summary["all_finite"]
                and dropout_base_summary["all_finite"]
                and dropout_zero_summary["all_finite"]
                and dropout_awf_summary["all_finite"]
            ),
        }
        result["decision"] = apply_stage0_decision(result["gates"])
        result["execution_valid"] = True
        result["frozen_decision_rule_applied"] = True
        result["automatic_progression"] = {
            "stage_a_authorized": result["decision"] == "CVLR_XVLA_STAGE0_PASS",
            "stage_a_executed_within_stage0": False,
            "next_action": (
                "Freeze and launch a separate bounded paired Stage A protocol."
                if result["decision"] == "CVLR_XVLA_STAGE0_PASS"
                else "Archive CVLR without Stage A."
            ),
        }

        heartbeat("persist_tracked_checkpoint")
        DEFAULT_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        tracked_checkpoint = DEFAULT_CHECKPOINT_DIR / "cvlr_xvla_stage0.pt"
        shutil.copy2(checkpoint_path, tracked_checkpoint)
        result["tracked_checkpoint"] = {
            "path": str(tracked_checkpoint),
            "sha256": sha256_file(tracked_checkpoint),
            "bytes": int(tracked_checkpoint.stat().st_size),
        }
    except Exception as exc:  # pragma: no cover - empirical boundary
        result["exceptions"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback_tail": traceback.format_exc().splitlines()[-120:],
            }
        )
        result["execution_valid"] = False
        result["decision"] = "CVLR_XVLA_STAGE0_IMPLEMENTATION_OR_OPTIMIZATION_FAILURE"
        result["automatic_progression"] = {
            "stage_a_authorized": False,
            "stage_a_executed_within_stage0": False,
            "next_action": "Classify the failure and use at most the single permitted narrow repair.",
        }
    finally:
        if hook_handle is not None:
            try:
                hook_handle.remove()
            except Exception:
                pass
        if env is not None:
            try:
                env.close()
            except Exception:
                pass
        try:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                result["cuda"] = cuda_report()
                torch.cuda.empty_cache()
        except Exception as cleanup_exc:
            result.setdefault("cleanup_exceptions", []).append(str(cleanup_exc))
        result["system_ram"] = memory_report()
        result["nvidia_smi_after"] = nvidia_smi()
        result["elapsed_seconds"] = round(time.monotonic() - started, 3)
        result["finished_at"] = utcish_timestamp()
        process_exit_code = 0 if result.get("execution_valid") else 2
        result["process_exit_code"] = process_exit_code
        atomic_write_json(paths.result, result)
        write_markdown(paths.result_md, result)
        shutil.copy2(paths.result, DEFAULT_REPORT_JSON)
        shutil.copy2(paths.result_md, DEFAULT_REPORT_MD)
        atomic_write_json(
            paths.status,
            {
                "timestamp": utcish_timestamp(),
                "stage": "finished",
                "state": "complete" if result.get("execution_valid") else "failed",
                "pid": int(os.getpid()),
                "decision": result.get("decision"),
            },
        )
        atomic_write_json(
            paths.heartbeat,
            {"timestamp": utcish_timestamp(), "stage": "finished", "pid": int(os.getpid())},
        )
        write_text(paths.exit_code, f"{process_exit_code}\n")
        telemetry_artifacts: dict[str, Any] = {}
        for artifact in [
            paths.result,
            paths.result_md,
            paths.status,
            paths.heartbeat,
            paths.partial,
            paths.exit_code,
            run_dir / "launch_manifest.json",
            DEFAULT_REPORT_JSON,
            DEFAULT_REPORT_MD,
        ]:
            if artifact.exists():
                telemetry_artifacts[str(artifact)] = {
                    "sha256": sha256_file(artifact),
                    "bytes": int(artifact.stat().st_size),
                }
        tracked = result.get("tracked_checkpoint") or {}
        if tracked.get("path") and pathlib.Path(tracked["path"]).exists():
            artifact = pathlib.Path(tracked["path"])
            telemetry_artifacts[str(artifact)] = {
                "sha256": sha256_file(artifact),
                "bytes": int(artifact.stat().st_size),
            }
        telemetry = {
            "schema_version": "2026-07-18.epoch5_cvlr_xvla_stage0_telemetry.v1",
            "run_dir": str(run_dir),
            "pid": int(os.getpid()),
            "cuda_pid": result.get("cuda_pid"),
            "decision": result.get("decision"),
            "execution_valid": result.get("execution_valid"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "cuda": result.get("cuda"),
            "system_ram": result.get("system_ram"),
            "nvidia_smi_before": result.get("nvidia_smi_before"),
            "nvidia_smi_after": result.get("nvidia_smi_after"),
            "artifacts": telemetry_artifacts,
            "exceptions": result.get("exceptions"),
        }
        atomic_write_json(DEFAULT_TELEMETRY_JSON, telemetry)
    return int(result["process_exit_code"]), result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=pathlib.Path, required=True)
    parser.add_argument("--contract", type=pathlib.Path, default=DEFAULT_CONTRACT)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    code, result = run_stage0(pathlib.Path(args.run_dir), pathlib.Path(args.contract))
    print(json.dumps({"decision": result.get("decision"), "execution_valid": result.get("execution_valid")}, sort_keys=True))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
