"""Pure SmolVLA-to-LIBERO interface adapter helpers.

These helpers do not import SmolVLA, create simulator environments, run
inference, or perform rollouts. They make the currently implicit action/state
and image-key bridge behavior explicit so it can be unit-tested before wiring
it into bounded diagnostics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


ACTION_STRATEGY_GRIPPER_ZERO_HOLD = "policy_6d_delta_pose_plus_gripper_zero_hold"
ACTION_STRATEGY_GRIPPER_OPEN = "policy_6d_delta_pose_plus_gripper_open"
ACTION_STRATEGY_GRIPPER_CLOSE = "policy_6d_delta_pose_plus_gripper_close"

GRIPPER_STRATEGY_VALUES: dict[str, float] = {
    ACTION_STRATEGY_GRIPPER_ZERO_HOLD: 0.0,
    ACTION_STRATEGY_GRIPPER_OPEN: 1.0,
    ACTION_STRATEGY_GRIPPER_CLOSE: -1.0,
}

DEFAULT_IMAGE_ALIASES: dict[str, tuple[str, ...]] = {
    "observation.images.camera1": ("agentview_image", "agentview_rgb"),
    "observation.images.camera2": ("robot0_eye_in_hand_image", "eye_in_hand_image"),
    "observation.images.camera3": ("agentview_image", "robot0_eye_in_hand_image"),
    "observation.image": ("agentview_image", "agentview_rgb"),
    "observation.image2": ("robot0_eye_in_hand_image", "eye_in_hand_image"),
    "observation.image3": ("agentview_image", "robot0_eye_in_hand_image"),
}


@dataclass(frozen=True)
class ActionAdapterResult:
    values: list[float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class StateField:
    key: str
    start: int = 0
    stop: int | None = None


@dataclass(frozen=True)
class StateAdapterResult:
    values: list[float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class ImageSourceResult:
    value: Any
    metadata: dict[str, Any]


DIAGNOSTIC_EEF_POS_QUAT_XYZ_6D_STATE_FIELDS: tuple[StateField, ...] = (
    StateField("robot0_eef_pos"),
    StateField("robot0_eef_quat", 0, 3),
)


def _as_flat_float_list(value: Any) -> list[float]:
    """Convert array-like values, including torch tensors, to a flat float list."""
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    array = np.asarray(value, dtype=np.float32).reshape(-1)
    return [float(x) for x in array]


def adapt_policy_action_to_env_action(
    policy_action: Any,
    env_action_dim: int,
    *,
    strategy: str = ACTION_STRATEGY_GRIPPER_ZERO_HOLD,
    action_scale: float = 1.0,
    clip_range: tuple[float, float] = (-1.0, 1.0),
) -> ActionAdapterResult:
    """Adapt a policy action to a LIBERO/RoboSuite env action with metadata.

    The function deliberately refuses unsupported shape changes. The known local
    diagnostic case is a 6D SmolVLA action and a 7D environment action, where the
    seventh component is an explicit gripper strategy rather than silent padding.
    """
    if env_action_dim <= 0:
        raise ValueError("env_action_dim must be positive")
    if strategy not in GRIPPER_STRATEGY_VALUES:
        raise ValueError(f"unsupported action adapter strategy: {strategy}")
    if not np.isfinite(action_scale) or action_scale <= 0:
        raise ValueError("action_scale must be a positive finite value")

    raw_values = _as_flat_float_list(policy_action)
    policy_dim = len(raw_values)
    if policy_dim == 0:
        raise ValueError("policy_action must contain at least one value")

    min_clip, max_clip = clip_range
    if min_clip >= max_clip:
        raise ValueError("clip_range must be increasing")
    scaled_values = [float(value * action_scale) for value in raw_values]
    clipped_values = [float(np.clip(value, min_clip, max_clip)) for value in scaled_values]
    clipped_count = sum(1 for before, after in zip(scaled_values, clipped_values) if before != after)
    gripper_value: float | None = None

    if policy_dim == env_action_dim:
        values = clipped_values
        adapter_mode = "passthrough_same_dim"
    elif policy_dim == 6 and env_action_dim == 7:
        gripper_value = GRIPPER_STRATEGY_VALUES[strategy]
        values = clipped_values + [gripper_value]
        adapter_mode = strategy
    else:
        raise ValueError(
            f"unsupported action dimension mapping: policy_dim={policy_dim}, env_action_dim={env_action_dim}"
        )

    return ActionAdapterResult(
        values=values,
        metadata={
            "adapter": "explicit_action_adapter",
            "adapter_mode": adapter_mode,
            "policy_action_dim": policy_dim,
            "env_action_dim": env_action_dim,
            "strategy": strategy,
            "action_scale": float(action_scale),
            "scaling_performed": bool(action_scale != 1.0),
            "gripper_value": gripper_value,
            "clip_range": [min_clip, max_clip],
            "clipped_values": clipped_count,
            "implicit_padding_performed": False,
            "truncation_performed": False,
        },
    )


def adapt_observation_state(
    obs: Mapping[str, Any],
    fields: Sequence[StateField],
    output_dim: int,
    *,
    adapter_name: str = "explicit_state_adapter",
) -> StateAdapterResult:
    """Build a state vector from explicit fields and slices only.

    The function raises when the selected fields do not exactly match
    ``output_dim``. This prevents the previous silent ``values[:dim]`` behavior.
    """
    if output_dim <= 0:
        raise ValueError("output_dim must be positive")
    if not fields:
        raise ValueError("at least one StateField is required")

    values: list[float] = []
    field_metadata: list[dict[str, Any]] = []
    for field in fields:
        if field.key not in obs:
            raise KeyError(f"missing observation state key: {field.key}")
        array = np.asarray(obs[field.key], dtype=np.float32).reshape(-1)
        start = int(field.start)
        stop = None if field.stop is None else int(field.stop)
        selected = array[start:stop]
        if selected.size == 0:
            raise ValueError(f"state field slice is empty for key: {field.key}")
        values.extend(float(x) for x in selected)
        field_metadata.append(
            {
                "key": field.key,
                "source_length": int(array.size),
                "start": start,
                "stop": stop,
                "selected_length": int(selected.size),
            }
        )

    if len(values) != output_dim:
        raise ValueError(
            f"explicit state adapter produced {len(values)} values, expected {output_dim}; "
            "refusing silent truncation or padding"
        )

    return StateAdapterResult(
        values=values,
        metadata={
            "adapter": adapter_name,
            "output_dim": output_dim,
            "fields": field_metadata,
            "silent_truncation_performed": False,
            "implicit_padding_performed": False,
            "uses_privileged_state": False,
        },
    )


def select_image_source(
    obs: Mapping[str, Any],
    feature_key: str,
    *,
    aliases: Mapping[str, Sequence[str]] | None = None,
) -> ImageSourceResult:
    """Select the observation image source for a policy feature key.

    This is metadata-friendly selection only. Tensor conversion/resizing remains
    a separate step in rollout or smoke code.
    """
    alias_map = aliases or DEFAULT_IMAGE_ALIASES
    candidates = tuple(alias_map.get(feature_key, (feature_key,)))
    for candidate in candidates:
        if candidate in obs:
            return ImageSourceResult(
                value=obs[candidate],
                metadata={
                    "adapter": "explicit_image_alias_adapter",
                    "feature_key": feature_key,
                    "source_key": candidate,
                    "candidates": list(candidates),
                    "missing": False,
                },
            )
    raise KeyError(f"no image source found for {feature_key}; checked {list(candidates)}")
