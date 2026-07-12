import torch
import pytest

from tca_map.smolvla.scvc_vla import (
    SCVCConfig,
    apply_sensor_shift,
    assert_no_privileged_inference_fields,
    canonicalize_image,
    known_inverse_affine,
    tensor_stats,
)


def test_known_inverse_recovers_unclipped_shift() -> None:
    config = SCVCConfig(gain=0.5, bias=0.1)
    image = torch.full((1, 3, 8, 8), 0.4)
    shifted = apply_sensor_shift(image, config)
    recovered = known_inverse_affine(shifted, config)

    assert torch.max(torch.abs(recovered - image)).item() < 1e-6


def test_canonicalize_matches_target_stats() -> None:
    config = SCVCConfig()
    image = torch.rand((1, 3, 16, 16)) * 0.2 + 0.3
    target = torch.rand((1, 3, 16, 16)) * 0.4 + 0.1
    target_mean, target_std = tensor_stats(target)
    out = canonicalize_image(
        image,
        target_mean=target_mean,
        target_std=target_std,
        memory={},
        memory_key="camera",
        use_temporal=False,
        config=config,
    )
    out_mean, out_std = tensor_stats(out)

    assert torch.max(torch.abs(out_mean - target_mean)).item() < 0.05
    assert torch.max(torch.abs(out_std - target_std)).item() < 0.05


def test_privileged_fields_rejected() -> None:
    with pytest.raises(ValueError):
        assert_no_privileged_inference_fields(["current_image", "reward"])
