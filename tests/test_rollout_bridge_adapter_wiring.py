from pathlib import Path

import numpy as np
import pytest

from tca_map.smolvla import libero_learned_policy_rollout as rollout


class _Feature:
    shape = (3, 8, 8)


def test_rollout_state_tensor_uses_explicit_adapter_metadata():
    obs = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
        "robot0_joint_pos": np.ones(7, dtype=np.float32),
    }

    tensor, metadata = rollout._state_tensor(obs, 6, "cpu")

    assert list(tensor.shape) == [1, 6]
    assert tensor.numpy().tolist()[0] == pytest.approx([0.1, 0.2, 0.3, 1.0, 0.0, 0.0])
    assert metadata["adapter"] == "diagnostic_eef_pos_quat_xyz_6d_state_adapter"
    assert metadata["silent_truncation_performed"] is False
    assert metadata["implicit_padding_performed"] is False


def test_rollout_state_tensor_refuses_silent_padding_or_truncation():
    obs = {"robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32)}

    with pytest.raises(KeyError, match="robot0_eef_quat"):
        rollout._state_tensor(obs, 6, "cpu")

    obs["robot0_eef_quat"] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    with pytest.raises(ValueError, match="refusing silent truncation or padding"):
        rollout._state_tensor(obs, 7, "cpu")


def test_rollout_image_tensor_uses_explicit_alias_metadata():
    image = np.full((4, 4, 3), 255, dtype=np.uint8)
    obs = {"robot0_eye_in_hand_image": image}

    tensor, source, metadata = rollout._image_tensor(obs, "observation.images.camera2", _Feature(), "cpu")

    assert source == "robot0_eye_in_hand_image"
    assert list(tensor.shape) == [1, 3, 8, 8]
    assert metadata["adapter"] == "explicit_image_alias_adapter"
    assert metadata["feature_key"] == "observation.images.camera2"
    assert metadata["source_key"] == "robot0_eye_in_hand_image"
    assert metadata["resized"] is True
    assert metadata["zero_image_fallback_performed"] is False


def test_rollout_image_tensor_refuses_missing_alias():
    with pytest.raises(KeyError, match="no image source found"):
        rollout._image_tensor({}, "observation.images.camera1", _Feature(), "cpu")


def test_rollout_bridge_source_no_longer_contains_implicit_adapter_shortcuts():
    source = Path(rollout.__file__).read_text(encoding="utf-8")

    assert "adapt_policy_action_to_env_action" in source
    assert "adapt_observation_state" in source
    assert "select_image_source" in source
    assert "def _policy_action_to_env_action" not in source
    assert "def _select_image_array" not in source
    assert "values.extend([0.0]" not in source
    assert "values = values[:dim]" not in source
