#!/usr/bin/env python3
"""Run the frozen Epoch 8 PCAT Stage 0 causal mechanism test."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

import h5py
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tca_map.xvla_task1.train_lora import _prepare_xvla_imports  # noqa: E402


DEFAULT_PROTOCOL = REPO_ROOT / "reports/epoch8_pcat_stage0_protocol.json"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports/epoch8_pcat_stage0"
DEFAULT_DATA_ROOT = Path("/mnt/c/assets/datasets/Libero-XVLA-format/libero_goal")
DEFAULT_XVLA_ROOT = Path("/mnt/c/assets/repos/X-VLA")
MODEL_ID = "2toINF/X-VLA-Libero"
MODEL_REVISION = "129e71460678b7236cee6fc9707f09d9fa0c3590"
XVLA_CACHE_DIR = "/home/jiheon/assets/checkpoints/xvla_hf_cache/transformers"
ROLE_ORDER = (
    "capacity_control",
    "paraphrase_augmentation",
    "equivalence_only",
    "pcat",
)
FAMILY_ORDER = ("act", "obj", "comp")


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype="<i8").tobytes())
    digest.update(array.tobytes())
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, sort_keys=True) + "\n")


def heartbeat(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{timestamp()} {message}\n", encoding="utf-8")


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
    output: dict[str, Any] = {
        "mem_total_kib": meminfo.get("MemTotal"),
        "mem_available_kib": meminfo.get("MemAvailable"),
        "swap_total_kib": meminfo.get("SwapTotal"),
        "swap_free_kib": meminfo.get("SwapFree"),
        "swap_used_kib": meminfo.get("SwapTotal", 0) - meminfo.get("SwapFree", 0),
        "nvidia_smi": gpu,
    }
    if torch_module is not None and torch_module.cuda.is_available():
        output.update(
            {
                "torch_cuda_allocated_bytes": int(torch_module.cuda.memory_allocated()),
                "torch_cuda_reserved_bytes": int(torch_module.cuda.memory_reserved()),
                "torch_cuda_peak_allocated_bytes": int(torch_module.cuda.max_memory_allocated()),
            }
        )
    return output


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate_protocol(protocol_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_json(protocol_path)
    if protocol["status"] != "FROZEN_BEFORE_STAGE0_OUTCOMES":
        raise ValueError("Stage 0 protocol is not frozen")
    if not protocol["authorization"]["stage0_training_and_offline_evaluation"]:
        raise ValueError("Stage 0 execution is not authorized")
    if protocol["authorization"]["confirmation"]:
        raise ValueError("protocol improperly authorizes confirmation")
    expected_script = protocol["implementation"]["script"]
    if expected_script["path"] != "scripts/run_epoch8_pcat_stage0.py":
        raise ValueError("unexpected execution script path")
    if sha256_file(REPO_ROOT / expected_script["path"]) != expected_script["sha256"]:
        raise ValueError("execution script hash drift")
    for item in protocol["frozen_inputs"]:
        path = REPO_ROOT / item["path"]
        if sha256_file(path) != item["sha256"]:
            raise ValueError(f"frozen input hash drift: {path}")
    supervision = load_json(REPO_ROOT / protocol["supervision"]["path"])
    if supervision["status"] != "ACTION_RESPONSE_SUPERVISION_PREFLIGHT_PASS":
        raise ValueError("action-response supervision preflight did not pass")
    return protocol, supervision


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


def load_sample(
    public_record: dict[str, Any],
    data_root: Path,
    image_transform: Any,
    torch_module: Any,
) -> dict[str, Any]:
    path = data_root / public_record["relative_path"]
    if sha256_file(path) != public_record["file_sha256"]:
        raise ValueError(f"demo file hash drift: {path}")
    frame_index = int(public_record["frame_index"])
    with h5py.File(path, "r") as handle:
        raw = np.asarray(handle["abs_action_6d"], dtype=np.float32)
        left = np.concatenate([raw[:, :9], (raw[:, 9:] > 0.0).astype(np.float32)], axis=-1)
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
    if action.shape != (30, 20) or proprio.shape != (20,):
        raise ValueError(f"unexpected sample shape for {path}: {action.shape} {proprio.shape}")
    if array_sha256(left[frame_index]) != public_record["proprio_sha256"]:
        raise ValueError(f"proprio hash drift: {path}")
    if array_sha256(left[frame_index + 1 : frame_index + 31]) != public_record["action_sha256"]:
        raise ValueError(f"action hash drift: {path}")
    return {
        "eval_id": int(public_record["eval_id"]),
        "relative_path": public_record["relative_path"],
        "canonical_instruction": public_record["canonical_instruction"],
        "image_input": image_tensor,
        "image_mask": image_mask,
        "proprio": torch_module.from_numpy(proprio).unsqueeze(0),
        "action": torch_module.from_numpy(action).unsqueeze(0),
        "image_tensor_sha256": array_sha256(image_tensor.numpy()),
    }


def fixed_time_noise(
    action: Any,
    seed: int,
    item_index: int,
    torch_module: Any,
) -> tuple[Any, Any, int]:
    derived = int(seed + item_index * 10000)
    generator = torch_module.Generator(device=action.device)
    generator.manual_seed(derived)
    t = torch_module.rand((1,), generator=generator, device=action.device, dtype=action.dtype)
    epsilon = torch_module.randn(action.shape, generator=generator, device=action.device, dtype=action.dtype)
    return t, epsilon, derived


def tensor_cpu(value: Any) -> Any:
    return value.detach().float().cpu().contiguous()


def build_adapter_class(torch_module: Any) -> type:
    nn = torch_module.nn

    class ResponseAdapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.context_norm = nn.LayerNorm(1024)
            self.context_projection = nn.Linear(1024, 128)
            self.hidden_1 = nn.Linear(204, 256)
            self.hidden_2 = nn.Linear(256, 256)
            self.output = nn.Linear(256, 10)
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

        @staticmethod
        def time_features(t: Any, dimension: int = 16) -> Any:
            half = dimension // 2
            frequencies = torch_module.exp(
                -math.log(10000.0)
                * torch_module.arange(half, device=t.device, dtype=t.dtype)
                / max(half - 1, 1)
            )
            angles = t.unsqueeze(-1) * frequencies.unsqueeze(0)
            return torch_module.cat([torch_module.sin(angles), torch_module.cos(angles)], dim=-1)

        def forward(
            self,
            context: Any,
            noisy_action: Any,
            proprio: Any,
            t: Any,
            base_prediction: Any,
        ) -> tuple[Any, Any]:
            batch, horizon = noisy_action.shape[:2]
            context_token = torch_module.nn.functional.gelu(
                self.context_projection(self.context_norm(context))
            ).unsqueeze(1).expand(batch, horizon, 128)
            proprio_token = proprio.unsqueeze(1).expand(batch, horizon, 20)
            time_token = self.time_features(t).unsqueeze(1).expand(batch, horizon, 16)
            value = torch_module.cat(
                [context_token, noisy_action, proprio_token, time_token, base_prediction], dim=-1
            )
            value = torch_module.nn.functional.gelu(self.hidden_1(value))
            value = torch_module.nn.functional.gelu(self.hidden_2(value))
            left_residual = self.output(value)
            right_residual = torch_module.zeros_like(left_residual)
            residual = torch_module.cat([left_residual, right_residual], dim=-1)
            return base_prediction + residual, residual

    return ResponseAdapter


def base_prediction(
    model: Any,
    context_entry: dict[str, Any],
    noisy: Any,
    proprio: Any,
    t: Any,
    domain_id: Any,
    device: Any,
    torch_module: Any,
) -> Any:
    enc = {
        "vlm_features": context_entry["vlm_features"].to(device=device, dtype=torch_module.float32),
        "aux_visual_inputs": context_entry["aux_visual_inputs"].to(device=device, dtype=torch_module.float32),
    }
    proprio_m, noisy_m = model.action_space.preprocess(proprio, noisy)
    prediction = model.transformer(
        domain_id=domain_id,
        action_with_noise=noisy_m,
        proprio=proprio_m,
        t=t,
        **enc,
    )
    return prediction


def branch_record(
    context: dict[str, Any],
    prediction: Any,
) -> dict[str, Any]:
    return {
        "context": context["pooled"],
        "base": tensor_cpu(prediction),
    }


def endpoint_record(
    sample: dict[str, Any],
    noisy: Any,
    t: Any,
    branches: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "eval_id": sample["eval_id"],
        "relative_path": sample["relative_path"],
        "target": tensor_cpu(sample["action"]),
        "noisy": tensor_cpu(noisy),
        "proprio": tensor_cpu(sample["proprio"]),
        "t": tensor_cpu(t),
        "branches": branches,
    }


def batch_endpoint(
    items: list[dict[str, Any]],
    side: str,
    condition: str,
    device: Any,
    torch_module: Any,
) -> dict[str, Any]:
    endpoints = [item[side] for item in items]
    return {
        "context": torch_module.cat([endpoint["branches"][condition]["context"] for endpoint in endpoints], dim=0).to(device),
        "base": torch_module.cat([endpoint["branches"][condition]["base"] for endpoint in endpoints], dim=0).to(device),
        "noisy": torch_module.cat([endpoint["noisy"] for endpoint in endpoints], dim=0).to(device),
        "proprio": torch_module.cat([endpoint["proprio"] for endpoint in endpoints], dim=0).to(device),
        "t": torch_module.cat([endpoint["t"] for endpoint in endpoints], dim=0).to(device),
        "target": torch_module.cat([endpoint["target"] for endpoint in endpoints], dim=0).to(device),
    }


def adapter_predict(adapter: Any, batch: dict[str, Any]) -> tuple[Any, Any]:
    return adapter(
        batch["context"], batch["noisy"], batch["proprio"], batch["t"], batch["base"]
    )


def factual_loss(prediction: Any, target: Any, scales: Any, torch_module: Any) -> Any:
    continuous = (prediction[:, :, :9] - target[:, :, :9]) / scales.view(1, 1, 9)
    continuous_loss = torch_module.mean(continuous**2)
    gripper_loss = torch_module.nn.functional.binary_cross_entropy_with_logits(
        prediction[:, :, 9], target[:, :, 9]
    )
    return continuous_loss + gripper_loss


def equivalence_loss(left: Any, right: Any, scales: Any, torch_module: Any) -> Any:
    continuous = (left[:, :, :9] - right[:, :, :9]) / scales.view(1, 1, 9)
    grip = left[:, :, 9] - right[:, :, 9]
    return torch_module.mean(continuous**2) + 0.1 * torch_module.mean(grip**2)


def anchor_loss(residual: Any, scales: Any, torch_module: Any) -> Any:
    continuous = residual[:, :, :9] / scales.view(1, 1, 9)
    return torch_module.mean(continuous**2) + 0.1 * torch_module.mean(residual[:, :, 9] ** 2)


def transport_loss(
    left_own: Any,
    left_cross: Any,
    right_own: Any,
    right_cross: Any,
    left_target: Any,
    right_target: Any,
    scales: Any,
    torch_module: Any,
) -> Any:
    predicted = 0.5 * (
        (left_cross[:, :, :9] - left_own[:, :, :9])
        + (right_own[:, :, :9] - right_cross[:, :, :9])
    ) / scales.view(1, 1, 9)
    real = (right_target[:, :, :9] - left_target[:, :, :9]) / scales.view(1, 1, 9)
    return torch_module.nn.functional.smooth_l1_loss(predicted, real)


def train_role(
    role: str,
    records: list[dict[str, Any]],
    protocol: dict[str, Any],
    action_scales: Any,
    output_dir: Path,
    device: Any,
    torch_module: Any,
) -> tuple[Any, dict[str, Any]]:
    ResponseAdapter = build_adapter_class(torch_module)
    seed = int(protocol["training"]["adapter_seed"])
    torch_module.manual_seed(seed)
    torch_module.cuda.manual_seed_all(seed)
    adapter = ResponseAdapter().to(device=device, dtype=torch_module.float32)
    parameter_count = int(sum(parameter.numel() for parameter in adapter.parameters()))
    optimizer = torch_module.optim.AdamW(
        adapter.parameters(),
        lr=float(protocol["training"]["learning_rate"]),
        weight_decay=float(protocol["training"]["weight_decay"]),
    )
    steps = int(protocol["training"]["optimizer_steps_per_role"])
    batch_size = int(protocol["training"]["batch_pairs"])
    generator = torch_module.Generator(device="cpu")
    generator.manual_seed(seed + ROLE_ORDER.index(role) * 1000)
    order = torch_module.randperm(len(records), generator=generator).tolist()
    cursor = 0
    first_grad_norm = None
    losses: list[float] = []
    started = time.time()
    adapter.train()
    for step in range(steps):
        if cursor + batch_size > len(order):
            order = torch_module.randperm(len(records), generator=generator).tolist()
            cursor = 0
        indices = order[cursor : cursor + batch_size]
        cursor += batch_size
        items = [records[index] for index in indices]
        left_can = batch_endpoint(items, "left", "canonical", device, torch_module)
        right_can = batch_endpoint(items, "right", "canonical", device, torch_module)
        pred_l_can, res_l_can = adapter_predict(adapter, left_can)
        pred_r_can, res_r_can = adapter_predict(adapter, right_can)
        loss = factual_loss(pred_l_can, left_can["target"], action_scales, torch_module)
        loss = loss + factual_loss(pred_r_can, right_can["target"], action_scales, torch_module)

        if role != "capacity_control":
            left_para = batch_endpoint(items, "left", "paraphrase", device, torch_module)
            right_para = batch_endpoint(items, "right", "paraphrase", device, torch_module)
            pred_l_para, _ = adapter_predict(adapter, left_para)
            pred_r_para, _ = adapter_predict(adapter, right_para)
            loss = loss + factual_loss(pred_l_para, left_para["target"], action_scales, torch_module)
            loss = loss + factual_loss(pred_r_para, right_para["target"], action_scales, torch_module)
            if role in {"equivalence_only", "pcat"}:
                loss = loss + float(protocol["loss_weights"]["equivalence"]) * (
                    equivalence_loss(pred_l_can, pred_l_para, action_scales, torch_module)
                    + equivalence_loss(pred_r_can, pred_r_para, action_scales, torch_module)
                )
                loss = loss + float(protocol["loss_weights"]["base_anchor"]) * (
                    anchor_loss(res_l_can, action_scales, torch_module)
                    + anchor_loss(res_r_can, action_scales, torch_module)
                )
        if role == "pcat":
            left_cross = batch_endpoint(items, "left", "cross", device, torch_module)
            right_cross = batch_endpoint(items, "right", "cross", device, torch_module)
            pred_l_cross, _ = adapter_predict(adapter, left_cross)
            pred_r_cross, _ = adapter_predict(adapter, right_cross)
            loss = loss + float(protocol["loss_weights"]["transport"]) * transport_loss(
                pred_l_can,
                pred_l_cross,
                pred_r_can,
                pred_r_cross,
                left_can["target"],
                right_can["target"],
                action_scales,
                torch_module,
            )

        if not bool(torch_module.isfinite(loss).item()):
            raise ValueError(f"nonfinite {role} loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch_module.nn.utils.clip_grad_norm_(
            adapter.parameters(), float(protocol["training"]["gradient_clip_norm"])
        )
        if first_grad_norm is None:
            first_grad_norm = float(grad_norm.detach().float().item())
        optimizer.step()
        losses.append(float(loss.detach().float().item()))
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == steps:
            log = {
                "timestamp": timestamp(),
                "role": role,
                "step": step + 1,
                "loss": losses[-1],
                "cuda_allocated_bytes": int(torch_module.cuda.memory_allocated()),
            }
            append_jsonl(output_dir / "train_log.jsonl", log)
            print(f"[pcat-stage0] {role} step={step + 1}/{steps} loss={losses[-1]:.6f}", flush=True)

    adapter.eval()
    final_parameter_norm = float(
        torch_module.sqrt(
            sum(torch_module.sum(parameter.detach().float() ** 2) for parameter in adapter.parameters())
        ).item()
    )
    output_layer_norm = float(adapter.output.weight.detach().float().norm().item())
    checkpoint = output_dir / f"{role}_adapter.pt"
    torch_module.save(
        {
            "schema_version": "epoch8.pcat_adapter.v1",
            "role": role,
            "state_dict": {key: value.detach().cpu() for key, value in adapter.state_dict().items()},
            "optimizer_steps": steps,
            "parameter_count": parameter_count,
        },
        checkpoint,
    )
    summary = {
        "role": role,
        "optimizer_steps": steps,
        "parameter_count": parameter_count,
        "first_gradient_norm": first_grad_norm,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "minimum_loss": min(losses),
        "loss_decreased": losses[-1] < losses[0],
        "final_parameter_norm": final_parameter_norm,
        "output_layer_weight_norm": output_layer_norm,
        "checkpoint": str(checkpoint.relative_to(REPO_ROOT)),
        "checkpoint_sha256": sha256_file(checkpoint),
        "duration_seconds": time.time() - started,
    }
    return adapter, summary


def predict_single(
    adapter: Any | None,
    endpoint: dict[str, Any],
    condition: str,
    device: Any,
    torch_module: Any,
) -> tuple[Any, Any]:
    branch = endpoint["branches"][condition]
    base = branch["base"].to(device)
    if adapter is None:
        return base, torch_module.zeros_like(base)
    batch = {
        "context": branch["context"].to(device),
        "base": base,
        "noisy": endpoint["noisy"].to(device),
        "proprio": endpoint["proprio"].to(device),
        "t": endpoint["t"].to(device),
    }
    return adapter_predict(adapter, batch)


def official_energy(prediction: Any, target: Any, torch_module: Any) -> float:
    bce = torch_module.nn.functional.binary_cross_entropy_with_logits
    position = (
        torch_module.mean((prediction[:, :, 0:3] - target[:, :, 0:3]) ** 2)
        + torch_module.mean((prediction[:, :, 10:13] - target[:, :, 10:13]) ** 2)
    ) * 500.0
    rotation = (
        torch_module.mean((prediction[:, :, 3:9] - target[:, :, 3:9]) ** 2)
        + torch_module.mean((prediction[:, :, 13:19] - target[:, :, 13:19]) ** 2)
    ) * 10.0
    gripper = 0.5 * (bce(prediction[:, :, 9], target[:, :, 9]) + bce(prediction[:, :, 19], target[:, :, 19]))
    return float((position + rotation + gripper).detach().float().item())


def rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(value, dtype=np.float64) ** 2)))


def evaluate_role(
    role: str,
    adapter: Any | None,
    pair_records: list[dict[str, Any]],
    equivalence_records: list[dict[str, Any]],
    action_scales: np.ndarray,
    action_bounds: dict[str, np.ndarray],
    device: Any,
    torch_module: Any,
) -> dict[str, Any]:
    if adapter is not None:
        adapter.to(device).eval()
    canonical_energy: list[float] = []
    canonical_nrmse: list[float] = []
    equivalence_drift: list[float] = []
    adapter_effect: list[float] = []
    legal_flags: list[bool] = []
    with torch_module.inference_mode():
        for record in equivalence_records:
            endpoint = record["sample"]
            pred_can, _ = predict_single(adapter, endpoint, "canonical", device, torch_module)
            pred_para, _ = predict_single(adapter, endpoint, "paraphrase", device, torch_module)
            target = endpoint["target"].to(device)
            canonical_energy.append(official_energy(pred_can, target, torch_module))
            can_np = pred_can[:, :, :9].detach().float().cpu().numpy()
            para_np = pred_para[:, :, :9].detach().float().cpu().numpy()
            target_np = target[:, :, :9].detach().float().cpu().numpy()
            base_np = endpoint["branches"]["canonical"]["base"][:, :, :9].numpy()
            canonical_nrmse.append(rms((can_np - target_np) / action_scales.reshape(1, 1, 9)))
            equivalence_drift.append(rms((can_np - para_np) / action_scales.reshape(1, 1, 9)))
            adapter_effect.append(rms((can_np - base_np) / action_scales.reshape(1, 1, 9)))
            post_grip = torch_module.sigmoid(pred_can[:, :, 9]).detach().float().cpu().numpy()
            lower = action_bounds["lower"].reshape(1, 1, 9)
            upper = action_bounds["upper"].reshape(1, 1, 9)
            legal_flags.append(
                bool(
                    np.isfinite(can_np).all()
                    and np.isfinite(post_grip).all()
                    and np.all((post_grip >= 0.0) & (post_grip <= 1.0))
                    and np.all((can_np >= lower) & (can_np <= upper))
                )
            )

        transport_rows: list[dict[str, Any]] = []
        for record in pair_records:
            left = record["left"]
            right = record["right"]
            l_own, _ = predict_single(adapter, left, "canonical", device, torch_module)
            l_cross, _ = predict_single(adapter, left, "cross", device, torch_module)
            r_own, _ = predict_single(adapter, right, "canonical", device, torch_module)
            r_cross, _ = predict_single(adapter, right, "cross", device, torch_module)
            predicted = 0.5 * (
                (l_cross[:, :, :9] - l_own[:, :, :9])
                + (r_own[:, :, :9] - r_cross[:, :, :9])
            ).detach().float().cpu().numpy() / action_scales.reshape(1, 1, 9)
            real = (
                right["target"][:, :, :9].numpy() - left["target"][:, :, :9].numpy()
            ) / action_scales.reshape(1, 1, 9)
            predicted_flat = predicted.reshape(-1).astype(np.float64)
            real_flat = real.reshape(-1).astype(np.float64)
            predicted_norm = float(np.linalg.norm(predicted_flat))
            real_norm = float(np.linalg.norm(real_flat))
            cosine = float(np.dot(predicted_flat, real_flat) / max(predicted_norm * real_norm, 1e-12))
            mask = np.abs(real_flat) >= 0.1
            sign_accuracy = float(np.mean(np.sign(predicted_flat[mask]) == np.sign(real_flat[mask]))) if np.any(mask) else None
            transport_rows.append(
                {
                    "pair_id": record["pair_id"],
                    "pair_group": record["pair_group"],
                    "seed": record["seed"],
                    "cosine": cosine,
                    "nrmse": rms(predicted_flat - real_flat),
                    "magnitude_ratio": predicted_norm / max(real_norm, 1e-12),
                    "signed_coordinate_accuracy": sign_accuracy,
                }
            )

    groups: dict[str, Any] = {}
    for group in sorted({row["pair_group"] for row in transport_rows}):
        rows = [row for row in transport_rows if row["pair_group"] == group]
        groups[group] = {
            "count": len(rows),
            "cosine_mean": float(np.mean([row["cosine"] for row in rows])),
            "nrmse_mean": float(np.mean([row["nrmse"] for row in rows])),
            "magnitude_ratio_mean": float(np.mean([row["magnitude_ratio"] for row in rows])),
        }
    return {
        "role": role,
        "validation_equivalence_records": len(equivalence_records),
        "validation_transport_records": len(pair_records),
        "canonical_official_energy_mean": float(np.mean(canonical_energy)),
        "canonical_continuous_nrmse_mean": float(np.mean(canonical_nrmse)),
        "equivalence_drift_nrmse_mean": float(np.mean(equivalence_drift)),
        "adapter_effect_nrmse_mean": float(np.mean(adapter_effect)),
        "legal_action_fraction": float(np.mean(legal_flags)),
        "transport_cosine_mean": float(np.mean([row["cosine"] for row in transport_rows])),
        "transport_nrmse_mean": float(np.mean([row["nrmse"] for row in transport_rows])),
        "transport_magnitude_ratio_mean": float(np.mean([row["magnitude_ratio"] for row in transport_rows])),
        "transport_signed_coordinate_accuracy_mean": float(
            np.mean([row["signed_coordinate_accuracy"] for row in transport_rows if row["signed_coordinate_accuracy"] is not None])
        ),
        "transport_groups": groups,
        "transport_rows": transport_rows,
    }


def adjudicate(
    metrics: dict[str, dict[str, Any]],
    training: dict[str, dict[str, Any]],
    protocol: dict[str, Any],
    execution: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    rules = protocol["decision_rules"]
    base = metrics["base"]
    pcat = metrics["pcat"]
    controls = [metrics[name] for name in ("capacity_control", "paraphrase_augmentation", "equivalence_only")]
    strongest_non_pcat_cosine = max([base["transport_cosine_mean"], *[row["transport_cosine_mean"] for row in controls]])
    strongest_non_pcat_rmse = min([base["transport_nrmse_mean"], *[row["transport_nrmse_mean"] for row in controls]])
    checks = {
        "all_roles_trained_exact_steps": all(
            training[role]["optimizer_steps"] == protocol["training"]["optimizer_steps_per_role"]
            for role in ROLE_ORDER
        ),
        "all_role_losses_decreased": all(training[role]["loss_decreased"] for role in ROLE_ORDER),
        "all_first_gradients_positive_finite": all(
            training[role]["first_gradient_norm"] is not None
            and math.isfinite(training[role]["first_gradient_norm"])
            and training[role]["first_gradient_norm"] > 0
            for role in ROLE_ORDER
        ),
        "pcat_output_nonzero": training["pcat"]["output_layer_weight_norm"] > 0,
        "cuda_model_forwards_positive": execution["base_cuda_forwards"] > 0,
        "swap_remained_zero": execution["resources_after"]["swap_used_kib"] == 0,
        "canonical_retention": pcat["canonical_official_energy_mean"]
        <= rules["canonical_official_energy_ratio_max"] * base["canonical_official_energy_mean"],
        "equivalence_improves_base": pcat["equivalence_drift_nrmse_mean"]
        <= rules["equivalence_vs_base_ratio_max"] * base["equivalence_drift_nrmse_mean"],
        "equivalence_not_worse_than_augmentation": pcat["equivalence_drift_nrmse_mean"]
        <= rules["equivalence_vs_augmentation_ratio_max"]
        * metrics["paraphrase_augmentation"]["equivalence_drift_nrmse_mean"],
        "transport_cosine_absolute": pcat["transport_cosine_mean"] >= rules["transport_cosine_min"],
        "transport_cosine_gain": pcat["transport_cosine_mean"]
        >= strongest_non_pcat_cosine + rules["transport_cosine_gain_min"],
        "transport_rmse_gain": pcat["transport_nrmse_mean"]
        <= rules["transport_nrmse_ratio_max"] * strongest_non_pcat_rmse,
        "transport_all_groups_positive": all(
            row["cosine_mean"] > 0 for row in pcat["transport_groups"].values()
        ),
        "response_magnitude_noncollapsed": rules["transport_magnitude_ratio_min"]
        <= pcat["transport_magnitude_ratio_mean"]
        <= rules["transport_magnitude_ratio_max"],
        "measurable_action_effect": pcat["adapter_effect_nrmse_mean"]
        >= rules["adapter_effect_nrmse_min"],
        "action_legality": pcat["legal_action_fraction"] == 1.0,
    }
    decision = "PCAT_STAGE0_GO" if all(checks.values()) else "PCAT_STAGE0_NO_GO"
    comparison = {
        "strongest_non_pcat_transport_cosine": strongest_non_pcat_cosine,
        "strongest_non_pcat_transport_nrmse": strongest_non_pcat_rmse,
        "checks": checks,
    }
    return decision, comparison


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--xvla-root", type=Path, default=DEFAULT_XVLA_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)

    protocol, supervision = validate_protocol(args.protocol)
    if args.dry_run:
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PROTOCOL_VALID",
                    "training_pairs": len(supervision["training_action_pairs"]),
                    "validation_pairs": len(supervision["validation_action_pairs"]),
                    "confirmation_read": False,
                },
                indent=2,
            )
        )
        return 0

    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_path = args.output_dir / "result.json"
    if result_path.exists():
        raise FileExistsError(f"refusing to overwrite prior Stage 0 result: {result_path}")
    heartbeat_path = args.output_dir / "heartbeat.txt"
    heartbeat(heartbeat_path, "start")

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("HF_HOME", "/home/jiheon/assets/checkpoints/xvla_hf_cache")
    os.environ.setdefault("HF_HUB_CACHE", XVLA_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_CACHE", XVLA_CACHE_DIR)
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")

    result: dict[str, Any] = {
        "schema_version": "epoch8.pcat_stage0_result.v1",
        "created_at": timestamp(),
        "last_updated_at": timestamp(),
        "status": "RUNNING",
        "protocol": {"path": str(args.protocol.relative_to(REPO_ROOT)), "sha256": sha256_file(args.protocol)},
        "execution_classification": "FROZEN_STAGE0_REAL_CUDA_TRAINING_AND_OFFLINE_VALIDATION",
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "confirmation_content_read": False,
        "confirmation_outcome_read": False,
        "simulator_constructed": False,
        "simulator_episode_count": 0,
        "closed_loop_outcome_read": False,
        "one_backbone_resident_limit": 1,
        "training": {},
        "metrics": {},
        "exceptions": [],
        "resources_before": memory_snapshot(),
    }
    atomic_write_json(result_path, result)
    model = processor = None
    exit_code = 1
    try:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if result["resources_before"]["swap_used_kib"] != 0:
            raise RuntimeError("nonzero WSL swap before model load")
        torch.cuda.set_device(0)
        seed = int(protocol["training"]["adapter_seed"])
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        device = torch.device("cuda:0")

        heartbeat(heartbeat_path, "load_base")
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
        model.eval().to(device=device, dtype=torch.float32)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        result["runtime"] = {
            "import_report": import_report,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(0),
            "base_parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
            "base_trainable_parameter_count": int(sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)),
            "base_training_flag": bool(model.training),
        }
        result["resources_after_base_load"] = memory_snapshot(torch)
        atomic_write_json(result_path, result)

        image_transform = prepare_official_image_transform()
        sample_cache: dict[str, dict[str, Any]] = {}
        context_cache: dict[tuple[str, str], dict[str, Any]] = {}
        context_forwards = 0
        transformer_forwards = 0
        domain_id = torch.tensor([3], device=device, dtype=torch.long)

        def get_sample(public: dict[str, Any]) -> dict[str, Any]:
            path = str(public["relative_path"])
            if path not in sample_cache:
                sample_cache[path] = load_sample(public, args.data_root, image_transform, torch)
            return sample_cache[path]

        def get_context(sample: dict[str, Any], instruction: str) -> dict[str, Any]:
            nonlocal context_forwards
            key = (sample["relative_path"], instruction)
            if key not in context_cache:
                inputs = processor.encode_language(instruction)["input_ids"].to(device)
                with torch.inference_mode():
                    enc = model.forward_vlm(
                        inputs,
                        sample["image_input"].to(device=device, dtype=torch.float32),
                        sample["image_mask"].to(device),
                    )
                pooled = enc["vlm_features"].mean(dim=1).detach().float().cpu()
                context_cache[key] = {
                    "pooled": pooled,
                    "vlm_features": enc["vlm_features"].detach().to(device="cpu", dtype=torch.float16),
                    "aux_visual_inputs": enc["aux_visual_inputs"].detach().to(device="cpu", dtype=torch.float16),
                    "input_ids_sha256": array_sha256(inputs.detach().cpu().numpy()),
                }
                context_forwards += 1
                if context_forwards == 1 or context_forwards % 25 == 0:
                    print(f"[pcat-stage0] cached_contexts={context_forwards}", flush=True)
                    heartbeat(heartbeat_path, f"cache_context {context_forwards}")
            return context_cache[key]

        train_language = {
            (int(row["eval_id"]), str(row["family"])): str(row["paraphrase_instruction"])
            for row in supervision["training_equivalence_pairs"]
        }
        validation_language = {
            (int(row["eval_id"]), str(row["family"])): str(row["paraphrase_instruction"])
            for row in supervision["validation_equivalence_pairs"]
        }
        time_noise_seeds = [int(value) for value in protocol["evaluation"]["time_noise_seeds"]]

        def make_pair_records(
            pairs: list[dict[str, Any]],
            training: bool,
            index_offset: int,
        ) -> list[dict[str, Any]]:
            nonlocal transformer_forwards
            output: list[dict[str, Any]] = []
            for pair_index, pair in enumerate(pairs):
                left = get_sample(pair["left"])
                right = get_sample(pair["right"])
                family = FAMILY_ORDER[pair_index % len(FAMILY_ORDER)]
                language_map = train_language if training else validation_language
                left_text = {
                    "canonical": left["canonical_instruction"],
                    "paraphrase": language_map[(left["eval_id"], family)],
                    "cross": right["canonical_instruction"],
                }
                right_text = {
                    "canonical": right["canonical_instruction"],
                    "paraphrase": language_map[(right["eval_id"], family)],
                    "cross": left["canonical_instruction"],
                }
                for seed_index, base_seed in enumerate(time_noise_seeds):
                    item_index = index_offset + pair_index * len(time_noise_seeds) + seed_index
                    left_action = left["action"].to(device=device, dtype=torch.float32)
                    right_action = right["action"].to(device=device, dtype=torch.float32)
                    t, epsilon, derived = fixed_time_noise(left_action, base_seed, item_index, torch)
                    left_noisy = epsilon * t.view(-1, 1, 1) + left_action * (1 - t).view(-1, 1, 1)
                    right_noisy = epsilon * t.view(-1, 1, 1) + right_action * (1 - t).view(-1, 1, 1)
                    endpoints = []
                    for sample, noisy, texts in (
                        (left, left_noisy, left_text),
                        (right, right_noisy, right_text),
                    ):
                        branches: dict[str, dict[str, Any]] = {}
                        for condition, text_value in texts.items():
                            context = get_context(sample, text_value)
                            with torch.inference_mode():
                                prediction = base_prediction(
                                    model,
                                    context,
                                    noisy,
                                    sample["proprio"].to(device=device, dtype=torch.float32),
                                    t,
                                    domain_id,
                                    device,
                                    torch,
                                )
                            branches[condition] = branch_record(context, prediction)
                            transformer_forwards += 1
                        endpoints.append(endpoint_record(sample, noisy, t, branches))
                    output.append(
                        {
                            "pair_id": pair["pair_id"],
                            "pair_group": pair["pair_group"],
                            "family": family,
                            "seed": base_seed,
                            "derived_seed": derived,
                            "left": endpoints[0],
                            "right": endpoints[1],
                        }
                    )
                if pair_index == 0 or (pair_index + 1) % 8 == 0:
                    print(
                        f"[pcat-stage0] {'train' if training else 'validation'}_pairs={pair_index + 1}/{len(pairs)}",
                        flush=True,
                    )
                    heartbeat(heartbeat_path, f"cache_pairs {training} {pair_index + 1}")
            return output

        train_records = make_pair_records(supervision["training_action_pairs"], True, 0)
        validation_pair_records = make_pair_records(
            supervision["validation_action_pairs"], False, 100000
        )

        validation_public: dict[int, dict[str, Any]] = {}
        for pair in supervision["validation_action_pairs"]:
            validation_public[int(pair["left"]["eval_id"])] = pair["left"]
            validation_public[int(pair["right"]["eval_id"])] = pair["right"]
        validation_equivalence_records: list[dict[str, Any]] = []
        for row_index, row in enumerate(supervision["validation_equivalence_pairs"]):
            eval_id = int(row["eval_id"])
            family = str(row["family"])
            sample = get_sample(validation_public[eval_id])
            for seed_index, base_seed in enumerate(time_noise_seeds):
                item_index = 200000 + row_index * len(time_noise_seeds) + seed_index
                action = sample["action"].to(device=device, dtype=torch.float32)
                t, epsilon, derived = fixed_time_noise(action, base_seed, item_index, torch)
                noisy = epsilon * t.view(-1, 1, 1) + action * (1 - t).view(-1, 1, 1)
                branches = {}
                for condition, instruction in (
                    ("canonical", sample["canonical_instruction"]),
                    ("paraphrase", validation_language[(eval_id, family)]),
                ):
                    context = get_context(sample, instruction)
                    with torch.inference_mode():
                        prediction = base_prediction(
                            model,
                            context,
                            noisy,
                            sample["proprio"].to(device=device, dtype=torch.float32),
                            t,
                            domain_id,
                            device,
                            torch,
                        )
                    branches[condition] = branch_record(context, prediction)
                    transformer_forwards += 1
                validation_equivalence_records.append(
                    {
                        "eval_id": eval_id,
                        "family": family,
                        "seed": base_seed,
                        "derived_seed": derived,
                        "sample": endpoint_record(sample, noisy, t, branches),
                    }
                )

        result["cache"] = {
            "unique_samples": len(sample_cache),
            "unique_instruction_contexts": len(context_cache),
            "base_cuda_vlm_forwards": context_forwards,
            "base_cuda_transformer_forwards": transformer_forwards,
            "train_records": len(train_records),
            "validation_pair_records": len(validation_pair_records),
            "validation_equivalence_records": len(validation_equivalence_records),
        }
        result["resources_after_cache"] = memory_snapshot(torch)
        result["last_updated_at"] = timestamp()
        atomic_write_json(result_path, result)

        del context_cache
        del sample_cache
        del model
        model = None
        processor = None
        gc.collect()
        torch.cuda.empty_cache()
        heartbeat(heartbeat_path, "base_unloaded_train_adapters")

        action_scales_np = np.asarray(supervision["construction"]["action_delta_scales"], dtype=np.float32)
        action_scales = torch.from_numpy(action_scales_np).to(device)
        all_train_targets = np.concatenate(
            [
                record[side]["target"][:, :, :9].numpy().reshape(-1, 9)
                for record in train_records
                for side in ("left", "right")
            ],
            axis=0,
        )
        train_min = np.min(all_train_targets, axis=0)
        train_max = np.max(all_train_targets, axis=0)
        expansion = 0.10 * np.maximum(train_max - train_min, 0.02)
        action_bounds = {"lower": train_min - expansion, "upper": train_max + expansion}

        adapters: dict[str, Any] = {}
        training_summaries: dict[str, dict[str, Any]] = {}
        for role in ROLE_ORDER:
            heartbeat(heartbeat_path, f"train {role}")
            adapter, summary = train_role(
                role,
                train_records,
                protocol,
                action_scales,
                args.output_dir,
                device,
                torch,
            )
            adapters[role] = adapter.to("cpu")
            training_summaries[role] = summary
            torch.cuda.empty_cache()
            result["training"] = training_summaries
            result["last_updated_at"] = timestamp()
            atomic_write_json(result_path, result)

        heartbeat(heartbeat_path, "evaluate")
        metrics: dict[str, dict[str, Any]] = {}
        role_adapters: dict[str, Any | None] = {"base": None, **adapters}
        for role in ("base", *ROLE_ORDER):
            adapter = role_adapters[role]
            if adapter is not None:
                adapter = adapter.to(device)
            metrics[role] = evaluate_role(
                role,
                adapter,
                validation_pair_records,
                validation_equivalence_records,
                action_scales_np,
                action_bounds,
                device,
                torch,
            )
            if adapter is not None:
                role_adapters[role] = adapter.to("cpu")
                torch.cuda.empty_cache()
            print(
                f"[pcat-stage0] evaluated {role} cosine={metrics[role]['transport_cosine_mean']:.4f} "
                f"equiv={metrics[role]['equivalence_drift_nrmse_mean']:.4f}",
                flush=True,
            )

        resources_after = memory_snapshot(torch)
        execution = {
            "base_cuda_forwards": context_forwards + transformer_forwards,
            "optimizer_steps": sum(row["optimizer_steps"] for row in training_summaries.values()),
            "trainable_roles": list(ROLE_ORDER),
            "resources_after": resources_after,
        }
        decision, comparison = adjudicate(metrics, training_summaries, protocol, execution)
        result.update(
            {
                "status": "COMPLETE",
                "decision": decision,
                "comparison": comparison,
                "metrics": metrics,
                "training": training_summaries,
                "execution": execution,
                "action_bounds": {key: value.tolist() for key, value in action_bounds.items()},
                "resources_after": resources_after,
                "last_updated_at": timestamp(),
                "confirmation_content_read": False,
                "confirmation_outcome_read": False,
                "closed_loop_outcome_read": False,
            }
        )
        atomic_write_json(result_path, result)
        heartbeat(heartbeat_path, f"complete {decision}")
        print(json.dumps({"result": str(result_path), "decision": decision, "checks": comparison["checks"]}, indent=2), flush=True)
        exit_code = 0
    except Exception as exc:
        result["status"] = "FAILED_EXECUTION"
        result["last_updated_at"] = timestamp()
        result["exceptions"].append(
            {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        try:
            import torch

            result["resources_after"] = memory_snapshot(torch)
        except Exception:
            result["resources_after"] = memory_snapshot()
        atomic_write_json(result_path, result)
        heartbeat(heartbeat_path, f"failed {type(exc).__name__}")
        print(traceback.format_exc(), file=sys.stderr, flush=True)
        exit_code = 1
    finally:
        try:
            del model
            del processor
        except Exception:
            pass
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
