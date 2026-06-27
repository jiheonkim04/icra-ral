import pytest

from tca_map.adapters import build_lora_policy, validate_lora_policy_config


def test_head_only_lora_policy_passes():
    config = {
        "adapter": {
            "trainable_modules": ["target_fusion_layers", "action_head_projection"],
            "train_backbone": False,
            "trainable_params_millions_estimate": 20,
        },
        "lora": {"enabled": True},
        "openvla_oft": {"enabled": False},
    }
    result = validate_lora_policy_config(config)
    assert result["passed"] is True
    assert result["full_backbone_finetuning_allowed"] is False


def test_forbidden_full_finetuning_config_fails():
    config = {
        "adapter": {"train_backbone": True},
        "training": {"full_finetune": True},
    }
    result = validate_lora_policy_config(config)
    assert result["passed"] is False
    assert result["errors"]
    with pytest.raises(ValueError):
        build_lora_policy(config)


def test_openvla_oft_large_config_is_rejected():
    config = {
        "openvla_oft": {
            "enabled": True,
            "full_finetune": True,
            "full_rollout": True,
            "multiseed_sweep": True,
        }
    }
    result = validate_lora_policy_config(config)
    assert result["passed"] is False
    assert any("OpenVLA" in error or "openvla" in error.lower() for error in result["errors"])


def test_openvla_oft_frozen_smoke_policy_passes_without_training():
    config = {
        "openvla_oft": {
            "enabled": True,
            "frozen_smoke_only": True,
            "train": False,
        },
        "adapter": {"train_backbone": False},
    }
    result = validate_lora_policy_config(config)
    assert result["passed"] is True


def test_qlora_requires_explicit_config():
    config = {"qlora": {"enabled": True}}
    result = validate_lora_policy_config(config)
    assert result["passed"] is False
    assert any("QLoRA requires explicit_config" in error for error in result["errors"])
