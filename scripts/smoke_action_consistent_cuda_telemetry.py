#!/usr/bin/env python3
"""Telemetry-only CUDA smoke for the exceptional device repair."""

from __future__ import annotations

import gc
import json
import os

import torch


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    device_index = int(torch.cuda.current_device())
    device = torch.device("cuda", device_index)
    device_name = torch.cuda.get_device_name(device_index)
    if device_name != "NVIDIA GeForce RTX 5080":
        raise RuntimeError(f"unexpected CUDA device: {device_name}")
    if not torch.cuda.is_initialized():
        raise RuntimeError("CUDA context did not initialize")

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    tensor = torch.ones((1024,), dtype=torch.float32, device=device)
    torch.cuda.synchronize(device_index)
    allocated_while_live = int(torch.cuda.memory_allocated(device_index))
    reserved_while_live = int(torch.cuda.memory_reserved(device_index))
    peak_allocated_while_live = int(torch.cuda.max_memory_allocated(device_index))
    peak_reserved_while_live = int(torch.cuda.max_memory_reserved(device_index))
    no_cpu_fallback = bool(tensor.is_cuda and tensor.device.index == device_index)
    tensor_sum = float(tensor.sum().item())
    del tensor
    gc.collect()
    torch.cuda.synchronize(device_index)
    allocated_after_free = int(torch.cuda.memory_allocated(device_index))
    reserved_after_free = int(torch.cuda.memory_reserved(device_index))
    torch.cuda.empty_cache()
    reserved_after_empty_cache = int(torch.cuda.memory_reserved(device_index))
    peak_allocated_after_free = int(torch.cuda.max_memory_allocated(device_index))
    peak_reserved_after_free = int(torch.cuda.max_memory_reserved(device_index))

    success = bool(
        no_cpu_fallback
        and tensor_sum == 1024.0
        and allocated_while_live > 0
        and peak_allocated_while_live >= allocated_while_live
        and peak_reserved_while_live >= reserved_while_live
        and peak_allocated_after_free >= allocated_while_live
        and peak_reserved_after_free >= reserved_while_live
        and allocated_after_free == 0
        and reserved_after_empty_cache == 0
    )
    result = {
        "decision": "TELEMETRY_DEVICE_REPAIR_SMOKE_VALID" if success else "TELEMETRY_DEVICE_REPAIR_FAILED",
        "repair_classification": "EXCEPTIONAL_TELEMETRY_DEVICE_REPAIR",
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "cuda_device_count": int(torch.cuda.device_count()),
        "cuda_device_index": device_index,
        "cuda_device": str(device),
        "cuda_device_name": device_name,
        "cuda_allocator_initialized": bool(torch.cuda.is_initialized()),
        "no_cpu_fallback": no_cpu_fallback,
        "tensor_elements": 1024,
        "tensor_sum": tensor_sum,
        "memory": {
            "allocated_while_live_bytes": allocated_while_live,
            "reserved_while_live_bytes": reserved_while_live,
            "peak_allocated_while_live_bytes": peak_allocated_while_live,
            "peak_reserved_while_live_bytes": peak_reserved_while_live,
            "allocated_after_free_bytes": allocated_after_free,
            "reserved_after_free_bytes": reserved_after_free,
            "reserved_after_empty_cache_bytes": reserved_after_empty_cache,
            "peak_allocated_after_free_bytes": peak_allocated_after_free,
            "peak_reserved_after_free_bytes": peak_reserved_after_free,
        },
        "xvla_loaded": False,
        "discovery_outputs_accessed": False,
        "validation_or_confirmatory_outcomes_accessed": False,
        "optimizer_steps": 0,
        "success": success,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
