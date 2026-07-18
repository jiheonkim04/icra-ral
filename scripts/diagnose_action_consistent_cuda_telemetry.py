#!/usr/bin/env python3
"""Diagnose the frozen preflight CUDA telemetry argument without loading X-VLA."""

from __future__ import annotations

import json
import os

import torch


def call_result(label: str, function: object) -> dict[str, object]:
    try:
        function()  # type: ignore[operator]
        return {"label": label, "success": True, "exception": None}
    except Exception as exc:  # diagnostic must record every variant
        return {
            "label": label,
            "success": False,
            "exception": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
        }


def main() -> int:
    allocator_initialized_at_import = bool(torch.cuda.is_initialized())
    cuda_available = bool(torch.cuda.is_available())
    device_count = int(torch.cuda.device_count())
    allocator_initialized_after_availability_queries = bool(torch.cuda.is_initialized())
    exact_preflight_value = torch.device("cuda:0")
    torch.cuda.empty_cache()
    allocator_initialized_after_empty_cache = bool(torch.cuda.is_initialized())

    exact_frozen_call = (
        call_result(
            "exact_frozen_torch_device_before_current_device",
            lambda: torch.cuda.reset_peak_memory_stats(exact_preflight_value),
        )
        if cuda_available
        else None
    )
    allocator_initialized_after_exact_frozen_call = bool(torch.cuda.is_initialized())
    current_device = int(torch.cuda.current_device()) if cuda_available else None
    allocator_initialized_after_current_device = bool(torch.cuda.is_initialized())

    variants: list[dict[str, object]] = []
    if cuda_available:
        variants.append(
            call_result(
                "no_explicit_argument",
                lambda: torch.cuda.reset_peak_memory_stats(),
            )
        )
        variants.append(
            call_result(
                "current_device_integer",
                lambda: torch.cuda.reset_peak_memory_stats(torch.cuda.current_device()),
            )
        )
        variants.append(
            call_result(
                "torch_device_with_current_index",
                lambda: torch.cuda.reset_peak_memory_stats(
                    torch.device("cuda", torch.cuda.current_device())
                ),
            )
        )

    result = {
        "diagnostic": "ACTION_CONSISTENT_MISSING_VIEW_CUDA_TELEMETRY_DEVICE",
        "xvla_loaded": False,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_available": cuda_available,
        "cuda_device_count": device_count,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "current_device": current_device,
        "current_device_name": (
            torch.cuda.get_device_name(current_device) if current_device is not None else None
        ),
        "exact_preflight_argument": {
            "repr": repr(exact_preflight_value),
            "str": str(exact_preflight_value),
            "python_type": f"{type(exact_preflight_value).__module__}.{type(exact_preflight_value).__name__}",
            "device_type": exact_preflight_value.type,
            "device_index": exact_preflight_value.index,
        },
        "cuda_allocator_initialized_at_import": allocator_initialized_at_import,
        "cuda_allocator_initialized_after_availability_queries": allocator_initialized_after_availability_queries,
        "cuda_allocator_initialized_after_empty_cache": allocator_initialized_after_empty_cache,
        "exact_frozen_call_before_current_device": exact_frozen_call,
        "cuda_allocator_initialized_after_exact_frozen_call": allocator_initialized_after_exact_frozen_call,
        "cuda_allocator_initialized_after_current_device": allocator_initialized_after_current_device,
        "cuda_allocator_initialized_after_variant_calls": bool(torch.cuda.is_initialized()),
        "variant_results": variants,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
