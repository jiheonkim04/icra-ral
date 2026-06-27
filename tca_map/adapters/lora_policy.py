"""Low-compute LoRA/QLoRA policy guards.

LoRA and QLoRA are optional support tools. They are not the core novelty and
must not enable full-backbone fine-tuning in the local low-compute protocol.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

DEFAULT_LORA_MODULES = [
    "target_fusion_layers",
    "action_head_projection",
    "small_adapter_layers",
]
LOWCOMPUTE_MAX_TRAINABLE_PARAMS_MILLIONS = 50
_FORBIDDEN_TRUE_KEYS = {
    "full_finetune",
    "full_backbone_finetune",
    "train_backbone",
    "openvla_oft_full_finetune",
    "openvla_oft_full_rollout",
    "openvla_oft_multiseed_sweep",
}


def _walk_items(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    if isinstance(value, dict):
        items: list[tuple[str, Any]] = []
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            items.extend(_walk_items(child, path))
        return items
    return [(prefix, value)]


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() == "true"
    return bool(value)


def validate_lora_policy_config(config: dict) -> dict:
    """Validate optional LoRA/QLoRA config for the low-compute protocol."""
    errors: list[str] = []
    warnings: list[str] = []
    flat_items = _walk_items(config)

    for path, value in flat_items:
        key = path.split(".")[-1]
        if key in _FORBIDDEN_TRUE_KEYS and _bool_value(value):
            errors.append(f"Forbidden local low-compute setting: {path}=true")

    openvla = config.get("openvla_oft", {}) if isinstance(config.get("openvla_oft", {}), dict) else {}
    if _bool_value(openvla.get("enabled", False)):
        if not (_bool_value(openvla.get("frozen_smoke_only", False)) or _bool_value(openvla.get("load_smoke_only", False))):
            errors.append("OpenVLA-OFT may only be enabled for frozen/load smoke in this protocol.")
        if _bool_value(openvla.get("train", False)):
            errors.append("OpenVLA-OFT training is forbidden locally.")

    adapter = config.get("adapter", {}) if isinstance(config.get("adapter", {}), dict) else {}
    qlora = config.get("qlora", {}) if isinstance(config.get("qlora", {}), dict) else {}
    lora = config.get("lora", {}) if isinstance(config.get("lora", {}), dict) else {}

    trainable_params = adapter.get("trainable_params_millions_estimate")
    if trainable_params is None:
        trainable_params = lora.get("trainable_params_millions_estimate")
    if trainable_params is not None and float(trainable_params) > LOWCOMPUTE_MAX_TRAINABLE_PARAMS_MILLIONS:
        errors.append(
            "Trainable parameter estimate exceeds "
            f"{LOWCOMPUTE_MAX_TRAINABLE_PARAMS_MILLIONS}M: {trainable_params}M"
        )

    if _bool_value(qlora.get("enabled", False)) and not _bool_value(qlora.get("explicit_config", False)):
        errors.append("QLoRA requires explicit_config=true because it is an optional memory-saving path.")

    trainable_modules = adapter.get("trainable_modules") or lora.get("trainable_modules") or DEFAULT_LORA_MODULES
    forbidden_modules = [module for module in trainable_modules if "backbone" in str(module).lower()]
    if forbidden_modules:
        errors.append(f"Backbone modules are not allowed in local LoRA trainable_modules: {forbidden_modules}")

    if _bool_value(lora.get("enabled", False)):
        warnings.append("LoRA is optional support, not the main novelty; report it separately from TCA-Select gain.")
    if _bool_value(qlora.get("enabled", False)):
        warnings.append("QLoRA is a memory-saving support path; report it separately from TCA-Select gain.")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "default_trainable_modules": list(DEFAULT_LORA_MODULES),
        "full_backbone_finetuning_allowed": False,
    }


def build_lora_policy(config: dict | None = None) -> dict:
    """Return a normalized optional LoRA policy after validation."""
    normalized = deepcopy(config or {})
    normalized.setdefault("adapter", {})
    normalized["adapter"].setdefault("trainable_modules", list(DEFAULT_LORA_MODULES))
    normalized["adapter"].setdefault("train_backbone", False)
    normalized["adapter"].setdefault("main_novelty", False)
    validation = validate_lora_policy_config(normalized)
    if not validation["passed"]:
        raise ValueError("Invalid low-compute LoRA policy: " + "; ".join(validation["errors"]))
    normalized["validation"] = validation
    return normalized
