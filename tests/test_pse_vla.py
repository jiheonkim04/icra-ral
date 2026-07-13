import numpy as np
import pytest
import torch

from tca_map.smolvla.pse_vla import (
    PSEConfig,
    action_l2_delta,
    assert_no_privileged_inference_fields,
    average_action_arrays,
    transform_batch_images,
    transform_image,
    transforms_for_variant,
)


def test_fixed_photometric_transforms() -> None:
    config = PSEConfig()
    image = torch.tensor([[[[0.0, 0.4, 1.0]]]], dtype=torch.float32)

    bright = transform_image(image, "bright_low_contrast", config)
    dark = transform_image(image, "dark_high_contrast", config)

    assert torch.allclose(bright, torch.tensor([[[[0.28, 0.448, 0.7]]]], dtype=torch.float32))
    assert torch.allclose(dark, torch.tensor([[[[0.0, 0.4, 1.0]]]], dtype=torch.float32))


def test_variant_transform_bank_is_fixed() -> None:
    assert transforms_for_variant("pse_full") == ("identity", "bright_low_contrast", "dark_high_contrast")
    assert transforms_for_variant("pse_duplicate_clean") == ("identity", "identity", "identity")
    with pytest.raises(ValueError):
        transforms_for_variant("adaptive_best_transform")


def test_action_averaging_and_delta() -> None:
    actions = [
        np.asarray([[1.0, 0.0, -1.0]], dtype=np.float64),
        np.asarray([[0.0, 1.0, -1.0]], dtype=np.float64),
        np.asarray([[2.0, 2.0, 1.0]], dtype=np.float64),
    ]

    averaged = average_action_arrays(actions)

    assert averaged.shape == (1, 3)
    assert np.allclose(averaged, [[1.0, 1.0, -1.0 / 3.0]])
    assert action_l2_delta(averaged, actions[0]) > 0.0


def test_duplicate_clean_average_is_exact_for_identical_actions() -> None:
    action = np.asarray([[0.1, -0.2, 0.3, 0.4, -0.5, 0.6, -0.7]], dtype=np.float64)

    averaged = average_action_arrays([action, action.copy(), action.copy()])

    assert np.allclose(averaged, action)


def test_transform_batch_reports_image_delta() -> None:
    config = PSEConfig()
    image = torch.full((1, 3, 4, 4), 0.5)
    batch = {"observation.images.camera1": image, "observation.state": torch.zeros((1, 8))}

    transformed, diagnostics = transform_batch_images(batch, transform="bright_low_contrast", config=config)

    assert transformed["observation.state"] is batch["observation.state"]
    assert diagnostics["image_mean_abs_delta"]["observation.images.camera1"] > 0.0


def test_privileged_fields_rejected() -> None:
    with pytest.raises(ValueError):
        assert_no_privileged_inference_fields(["current_image", "reward"])
