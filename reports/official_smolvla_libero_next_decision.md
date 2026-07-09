# Official SmolVLA / LIBERO Next Decision

Date: 2026-07-09 KST

Final decision: `READY_FOR_OFFICIAL_ASSET_APPROVAL`

## Reason

The official asset route is now scoped:

- `lerobot/smolvla_libero` is public, not gated, Apache-2.0, and approximately `0.844 GiB`.
- `lerobot/libero` is public, not gated, Apache-2.0, and approximately `1.803 GiB`.
- Combined selected official route is approximately `2.647 GiB`.
- Exact inspection/load/eval/LoRA commands are documented.
- Local disk has enough space, but the objective explicitly says not to download yet when total size exceeds `2GB`.

Therefore user approval is required before acquiring the official model+dataset pair.

## Local Conversion Alternative

The local HDF5 route is feasible for a planned 1-demo LeRobot-format conversion:

- local actions are already 7D;
- local state components can form official 8D state;
- local images can map to official two-camera LIBERO keys after resize and orientation verification;
- LeRobot's official `LeRobotDataset.create`, `add_frame`, `save_episode`, and `finalize` APIs are available.

This alternative should be the next no-download implementation milestone if official asset approval is not granted.

## Exact Next Command

Only after explicit user approval:

```powershell
$env:HF_HOME='C:\assets\hf_home'
huggingface-cli download lerobot/smolvla_libero --local-dir C:\assets\checkpoints\smolvla_libero
huggingface-cli download lerobot/libero --repo-type dataset --local-dir C:\assets\datasets\lerobot_libero
```

Expected persistent download size: approximately `2.647 GiB` plus overhead.

Do not train after download. The first post-download action must be a tiny official load/processor/sample shape smoke.

