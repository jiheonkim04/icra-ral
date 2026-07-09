# Local HDF5 To LeRobot Conversion Plan

Date: 2026-07-09 KST

## Scope

Plan only. No conversion was executed.

Goal: determine whether a tiny local LIBERO HDF5 subset can be converted into official-compatible LeRobot dataset format without reviving the archived custom `LIBERO_7D` adapter route.

## Conversion Feasibility

Decision for conversion route: feasible for a 1-demo tiny conversion.

Reason: local HDF5 demos already contain the required ingredients for the official LIBERO dataset schema:

- 7D continuous actions in `actions`
- EEF position in `obs/ee_pos`
- EEF orientation as 3D orientation vector in `obs/ee_ori`
- gripper position in `obs/gripper_states`
- main camera RGB in `obs/agentview_rgb`
- wrist camera RGB in `obs/eye_in_hand_rgb`

## Target LeRobot Features

For `lerobot/libero` compatibility, create:

```python
features = {
    "observation.images.image": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.images.image2": {
        "dtype": "video",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channel"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": ["state"],
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": ["actions"],
    },
}
```

Create with LeRobot's official writer:

```python
dataset = LeRobotDataset.create(
    repo_id="local/libero_10_tiny_lerobot",
    fps=10,
    root=r"C:\assets\datasets\local_libero_10_tiny_lerobot",
    robot_type="panda",
    features=features,
    use_videos=True,
)
```

Then call `dataset.add_frame(...)`, `dataset.save_episode()`, and `dataset.finalize()`.

## HDF5 To LeRobot Mapping

| local HDF5 key | target LeRobot key | transform |
| --- | --- | --- |
| `obs/agentview_rgb[t]` | `observation.images.image` | resize from `128x128` to `256x256`; verify/apply official 180-degree flip policy |
| `obs/eye_in_hand_rgb[t]` | `observation.images.image2` | resize from `128x128` to `256x256`; verify/apply official 180-degree flip policy |
| `concat(obs/ee_pos[t], obs/ee_ori[t], obs/gripper_states[t])` | `observation.state` | cast to float32, shape `[8]` |
| `actions[t]` | `action` | cast to float32, shape `[7]`; no fill, no truncation |
| natural task string | `task` | use official LIBERO task language when available; otherwise use verified filename-derived task phrase |

## Action Convention

Preserve the local `actions[:, :7]` exactly except dtype conversion to float32.

Do not:

- drop the gripper dimension;
- fill or synthesize gripper values;
- normalize with SO100 stats;
- clip actions during conversion;
- relabel action dimensions based on the archived custom adapter.

Observed local gripper action values are `-1` and `1`, matching the official LIBERO gripper range in the action stats.

## State Convention

Use the official 8D state convention:

```text
[eef_pos(3), axis_angle_or_ee_ori(3), gripper_qpos(2)]
```

Local evidence:

- `obs/ee_pos`: 3D
- `obs/ee_ori`: 3D
- `obs/gripper_states`: 2D
- `obs/ee_states`: 6D, matching `ee_pos + ee_ori`, but missing gripper state

Do not use the old 6D-only `ee_states` path for official conversion.

## Image / Video Requirements

Official `lerobot/libero` is video-backed:

- image keys have dtype `video`;
- image shape is `[256, 256, 3]`;
- fps is `10.0`;
- video shards live under `videos/{video_key}/chunk-000/file-000.mp4`.

Local HDF5 images are `128x128` HWC uint8. The tiny conversion must resize to `256x256` before writing.

Open risk: official `LiberoProcessorStep` flips LIBERO env images by 180 degrees to match HuggingFaceVLA orientation. Before locking the conversion, visually compare a local HDF5 frame against an official `lerobot/libero` sample or apply the same flip if the HDF5 is raw env orientation.

## Dataset Stats Requirements

Use LeRobot's writer/stats path, not custom sidecars:

- `meta/info.json`
- `meta/stats.json`
- `meta/tasks.parquet`
- `meta/episodes/chunk-000/file-000.parquet`
- `data/chunk-000/file-000.parquet`
- `videos/...` if `use_videos=True`

Do not reuse SO100 normalizer stats on LIBERO labels. For a tiny conversion, stats will be tiny-subset stats and must be labeled as smoke-only, not as official full-dataset stats.

## Leakage Rules

Do not use:

- BDDL/eval target labels as model inputs;
- task IDs as language input;
- privileged simulator state at inference;
- filename-derived task strings unless verified against official LIBERO task language;
- eval split labels as training labels.

## Risks

| risk | severity | mitigation |
| --- | --- | --- |
| `smolvla_libero` config lists 6D state while official dataset/stats are 8D | high | Run official asset/sample smoke before training. Preserve 8D dataset schema. |
| config lists `camera3` while official LIBERO dataset has two image keys | high | Run official preprocessor smoke with `smolvla_libero`; do not invent a third camera. |
| local images are 128x128 while official data is 256x256 | medium | Resize before writing; verify visual orientation. |
| video encoding on Windows may fail due codec/backend | medium | Use a tiny dry-run first; if needed, run conversion under WSL/Linux. |
| tiny-subset stats are not full official stats | medium | Label as mini-repro smoke only; do not train paper claims from it. |

## Feasibility Conclusion

A 1-demo tiny LeRobot conversion is possible without the old custom adapter route. It should be implemented as a new official-format conversion utility in the next milestone, with a dry-run mode and a post-conversion `LeRobotDataset` load check.

