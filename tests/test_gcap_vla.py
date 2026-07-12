import torch
import pytest

from tca_map.smolvla.gcap_vla import (
    CAMERA_KEYS,
    GCAPConfig,
    apply_rect_occlusion,
    assert_no_privileged_inference_fields,
    edge_enhance,
    image_mse,
    occlusion_box,
    repair_camera_tensor,
    transform_batch_images,
)


def _image() -> torch.Tensor:
    y = torch.linspace(0, 1, 32).reshape(1, 1, 32, 1)
    x = torch.linspace(0, 1, 32).reshape(1, 1, 1, 32)
    return torch.cat([x.expand(1, 1, 32, 32), y.expand(1, 1, 32, 32), (x + y).expand(1, 1, 32, 32) / 2], dim=1)


def test_occlusion_box_and_mask_are_nonempty():
    config = GCAPConfig()
    box = occlusion_box(height=32, width=32, identity=20260714, step_fraction=0.35, camera_key=CAMERA_KEYS[0], config=config)
    assert box is not None
    image, mask = apply_rect_occlusion(_image(), box=box, fill_value=0.0)
    assert image.shape == (1, 3, 32, 32)
    assert mask.shape == (1, 1, 32, 32)
    assert mask.sum() > 0


def test_temporal_patch_repair_improves_occluded_region():
    config = GCAPConfig(edge_gain=0.0)
    clean = _image()
    previous = clean.clone()
    box = (8, 8, 20, 20)
    occluded, mask = apply_rect_occlusion(clean, box=box, fill_value=0.0)
    repaired = repair_camera_tensor(occluded, previous=previous, mask=mask, variant="gcap_full", config=config)
    assert image_mse(repaired, clean) < image_mse(occluded, clean)


def test_hold_last_replaces_full_frame_when_masked():
    config = GCAPConfig()
    current = _image()
    previous = torch.ones_like(current) * 0.25
    _, mask = apply_rect_occlusion(current, box=(4, 4, 12, 12), fill_value=0.0)
    repaired = repair_camera_tensor(current, previous=previous, mask=mask, variant="full_frame_hold_last", config=config)
    assert torch.allclose(repaired, previous)


def test_edge_enhance_keeps_shape_and_finiteness():
    config = GCAPConfig(edge_gain=0.1)
    image = _image()
    _, mask = apply_rect_occlusion(image, box=(4, 4, 20, 20), fill_value=0.0)
    enhanced = edge_enhance(image, mask=mask, config=config)
    assert enhanced.shape == image.shape
    assert torch.isfinite(enhanced).all()


def test_transform_batch_updates_memory_without_privileged_fields():
    config = GCAPConfig()
    batch = {key: _image() for key in CAMERA_KEYS}
    memory = {}
    out, diagnostics = transform_batch_images(
        batch,
        variant="gcap_full",
        condition="occluded",
        identity=20260714,
        step_fraction=0.35,
        memory=memory,
        config=config,
    )
    assert set(memory) == set(CAMERA_KEYS)
    assert all(key in out for key in CAMERA_KEYS)
    assert any(value > 0.0 for value in diagnostics["mean_mask_fraction"].values())


def test_privileged_inference_guard():
    with pytest.raises(ValueError):
        assert_no_privileged_inference_fields(["current_observation_image", "sim_state"])
