# Official SmolVLA / LIBERO Action Schema

Date: 2026-07-09 KST

## Official LIBERO Dataset Schema

Official `lerobot/libero` metadata declares:

- `observation.state`: float32, shape `[8]`
- `action`: float32, shape `[7]`
- `observation.images.image`: video, shape `[256, 256, 3]`
- `observation.images.image2`: video, shape `[256, 256, 3]`
- fps: `10.0`
- robot type: `panda`

The LeRobot LIBERO docs describe actions as:

```text
Box(-1, 1, shape=(7,))
```

with 6D end-effector delta and 1D gripper.

## Official LIBERO State Schema

The installed `LiberoProcessorStep` builds the policy state as:

```text
[eef_pos(3), quat_to_axis_angle(eef_quat)(3), gripper_qpos(2)]
```

This produces 8D `observation.state`.

For local HDF5 conversion, the matching source is:

```text
[obs/ee_pos(3), obs/ee_ori(3), obs/gripper_states(2)]
```

## `lerobot/smolvla_libero` Checkpoint Schema

Official checkpoint metadata inspected:

- model: `lerobot/smolvla_libero`
- train dataset: `lerobot/libero`
- output action: `[7]`
- action normalizer tensors: 7D LIBERO action stats
- state normalizer tensors: 8D `observation.state` stats
- policy preprocessor renames:
  - `observation.images.image` to `observation.images.camera1`
  - `observation.images.image2` to `observation.images.camera2`

Important unresolved mismatch:

- checkpoint config declares `observation.state` input shape `[6]`;
- checkpoint normalizer tensors contain 8D `observation.state` stats;
- official `lerobot/libero` dataset metadata contains 8D `observation.state`;
- checkpoint config declares three camera inputs, while official LIBERO metadata exposes two image keys.

This is not a blocker for planning, but it is a blocker for training. The next runnable official step must be a tiny official-asset or tiny-converted-sample processor smoke that proves the shape path.

## Local HDF5 Schema

Representative local HDF5 evidence:

- `actions`: `(272, 7)`, dtype `float64`
- `obs/ee_pos`: `(272, 3)`, dtype `float64`
- `obs/ee_ori`: `(272, 3)`, dtype `float64`
- `obs/gripper_states`: `(272, 2)`, dtype `float64`
- `obs/agentview_rgb`: `(272, 128, 128, 3)`, dtype `uint8`
- `obs/eye_in_hand_rgb`: `(272, 128, 128, 3)`, dtype `uint8`

Observed action ranges in sampled demos stay in the official LIBERO action range, with gripper dimension observed as `-1` / `1`.

## Conversion Rule

The official-compatible conversion must preserve:

- 8D state;
- 7D action;
- gripper action as stored;
- two official image keys;
- LeRobot metadata/stats layout.

It must not:

- map SO100 6D stats onto LIBERO labels;
- silently coerce 8D state to 6D;
- revive the old `LIBERO_7D` adapter route;
- hard-code gripper fills;
- train before a shape-smoke passes.

