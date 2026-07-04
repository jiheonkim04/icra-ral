import numpy as np
import pytest

from tca_map.smolvla.interface_adapters import (
    ACTION_STRATEGY_GRIPPER_CLOSE,
    ACTION_STRATEGY_GRIPPER_ZERO_HOLD,
    DIAGNOSTIC_EEF_POS_QUAT_XYZ_6D_STATE_FIELDS,
    StateField,
    adapt_observation_state,
    adapt_policy_action_to_env_action,
    select_image_source,
)


def test_action_adapter_explicitly_maps_6d_policy_to_7d_env():
    result = adapt_policy_action_to_env_action(
        [-2.0, -0.5, 0.25, 0.5, 1.5, 0.0],
        7,
        strategy=ACTION_STRATEGY_GRIPPER_ZERO_HOLD,
    )

    assert result.values == [-1.0, -0.5, 0.25, 0.5, 1.0, 0.0, 0.0]
    assert result.metadata["adapter"] == "explicit_action_adapter"
    assert result.metadata["policy_action_dim"] == 6
    assert result.metadata["env_action_dim"] == 7
    assert result.metadata["strategy"] == ACTION_STRATEGY_GRIPPER_ZERO_HOLD
    assert result.metadata["gripper_value"] == 0.0
    assert result.metadata["clipped_values"] == 2
    assert result.metadata["implicit_padding_performed"] is False
    assert result.metadata["truncation_performed"] is False


def test_action_adapter_supports_named_gripper_strategy():
    result = adapt_policy_action_to_env_action(
        np.zeros(6, dtype=np.float32),
        7,
        strategy=ACTION_STRATEGY_GRIPPER_CLOSE,
    )

    assert result.values[-1] == -1.0
    assert result.metadata["adapter_mode"] == ACTION_STRATEGY_GRIPPER_CLOSE


def test_action_adapter_applies_explicit_action_scale_before_clipping():
    result = adapt_policy_action_to_env_action(
        [0.4, -0.8, 2.0, 0.0, 0.1, -0.1],
        7,
        action_scale=0.5,
        strategy=ACTION_STRATEGY_GRIPPER_ZERO_HOLD,
    )

    assert result.values == pytest.approx([0.2, -0.4, 1.0, 0.0, 0.05, -0.05, 0.0])
    assert result.metadata["action_scale"] == 0.5
    assert result.metadata["scaling_performed"] is True
    assert result.metadata["clipped_values"] == 0


def test_action_adapter_refuses_unsupported_dimension_mapping():
    with pytest.raises(ValueError, match="unsupported action dimension mapping"):
        adapt_policy_action_to_env_action([0.0, 0.1, 0.2], 7)


def test_action_adapter_refuses_invalid_action_scale():
    with pytest.raises(ValueError, match="action_scale"):
        adapt_policy_action_to_env_action([0.0] * 6, 7, action_scale=0.0)


def test_state_adapter_uses_explicit_fields_without_truncation():
    obs = {
        "robot0_eef_pos": np.array([0.1, 0.2, 0.3], dtype=np.float32),
        "robot0_eef_quat": np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }

    result = adapt_observation_state(obs, DIAGNOSTIC_EEF_POS_QUAT_XYZ_6D_STATE_FIELDS, 6)

    assert result.values == pytest.approx([0.1, 0.2, 0.3, 1.0, 0.0, 0.0])
    assert result.metadata["output_dim"] == 6
    assert result.metadata["silent_truncation_performed"] is False
    assert result.metadata["implicit_padding_performed"] is False
    assert result.metadata["uses_privileged_state"] is False


def test_state_adapter_refuses_missing_keys_and_wrong_output_dim():
    obs = {"robot0_eef_pos": [0.1, 0.2, 0.3]}

    with pytest.raises(KeyError, match="robot0_eef_quat"):
        adapt_observation_state(obs, DIAGNOSTIC_EEF_POS_QUAT_XYZ_6D_STATE_FIELDS, 6)

    with pytest.raises(ValueError, match="refusing silent truncation or padding"):
        adapt_observation_state(obs, [StateField("robot0_eef_pos")], 6)


def test_image_alias_adapter_reports_selected_source():
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    obs = {"robot0_eye_in_hand_image": image}

    result = select_image_source(obs, "observation.images.camera2")

    assert result.value is image
    assert result.metadata["adapter"] == "explicit_image_alias_adapter"
    assert result.metadata["feature_key"] == "observation.images.camera2"
    assert result.metadata["source_key"] == "robot0_eye_in_hand_image"
    assert result.metadata["missing"] is False


def test_image_alias_adapter_refuses_missing_source():
    with pytest.raises(KeyError, match="no image source found"):
        select_image_source({}, "observation.images.camera1")
