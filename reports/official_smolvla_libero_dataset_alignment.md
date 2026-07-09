# Official SmolVLA / LIBERO Dataset Alignment

Date: 2026-07-09 KST

Branch: `codex/official-smolvla-libero-dataset-alignment`

## Objective

Align the next SmolVLA/LIBERO step with official LeRobot dataset, processor, action, and evaluation conventions before any LoRA baseline reproduction.

This is not a method route and not paper evidence. The archived custom `LIBERO_7D` adapter route remains stopped.

## What Was Inspected

- Official LeRobot LIBERO docs: `https://huggingface.co/docs/lerobot/libero`
- Official SmolVLA docs: `https://huggingface.co/docs/lerobot/en/smolvla`
- Official PEFT docs: `https://huggingface.co/docs/lerobot/peft_training`
- Official LeRobotDataset v3 docs: `https://huggingface.co/docs/lerobot/lerobot-dataset-v3`
- Official `lerobot/libero` dataset card and Hub metadata
- Official `HuggingFaceVLA/libero` dataset card and Hub metadata
- Official `lerobot/smolvla_libero` model card, config, train config, processor JSON, and normalizer metadata
- Installed LeRobot `0.4.4` dataset/env/processor source
- Local HDF5 structure under `C:\assets\data\libero`

Only metadata/config/small JSON/safetensor inspection was performed. No full model or dataset asset was downloaded.

## Official Route Findings

- Official LeRobot LIBERO evaluation requires Linux and MuJoCo, with `MUJOCO_GL=egl` recommended for headless use.
- Official eval entrypoint is `lerobot-eval --env.type=libero`.
- Official LIBERO observations are `observation.state`, `observation.images.image`, and `observation.images.image2`.
- Official `lerobot/libero` metadata has `observation.state` shape `[8]` and `action` shape `[7]`.
- Official continuous action convention is `Box(-1, 1, shape=(7,))`: 6D EEF delta plus 1D gripper.
- `lerobot/smolvla_libero` is public/not gated, Apache-2.0, and uses `lerobot/libero` in its `train_config.json`.
- `lerobot/smolvla_libero` action output is `[7]` with LIBERO action normalizer tensors.
- `lerobot/smolvla_libero` config has an unresolved compatibility wrinkle: policy config lists `observation.state` shape `[6]`, while its normalizer tensors and `lerobot/libero` dataset stats contain 8D `observation.state`. This must be resolved by a tiny official sample smoke before training.
- `lerobot/smolvla_libero` policy preprocessor renames `observation.images.image` to `observation.images.camera1` and `observation.images.image2` to `observation.images.camera2`.
- The config still lists `observation.images.camera3`; no local training should assume how this is handled until an official sample smoke confirms it.

## Local HDF5 Findings

Representative local file:

```text
C:\assets\data\libero\libero_10\KITCHEN_SCENE3_turn_on_the_stove_and_put_the_moka_pot_on_it_demo.hdf5
```

Representative `demo_0` contains:

- `actions`: shape `(272, 7)`, dtype `float64`
- `obs/ee_pos`: shape `(272, 3)`
- `obs/ee_ori`: shape `(272, 3)`
- `obs/gripper_states`: shape `(272, 2)`
- `obs/agentview_rgb`: shape `(272, 128, 128, 3)`, dtype `uint8`
- `obs/eye_in_hand_rgb`: shape `(272, 128, 128, 3)`, dtype `uint8`
- gripper action dimension observed as `-1` / `1`

This is enough for a planned 1-demo LeRobot-format conversion using official 8D state `[ee_pos, ee_ori, gripper_states]` and original 7D actions. It does not justify training yet.

## Decision

Selected decision: `READY_FOR_OFFICIAL_ASSET_APPROVAL`

Reason: the official assets are public, not gated, Apache-2.0, and sizes/commands are known, but the official model+dataset route exceeds the explicit 2 GB no-approval threshold.

The local conversion route is also feasible as a no-download alternative, but it should be the next branch/milestone after this approval decision because it requires implementing and validating a new LeRobot conversion script.

## Hard Stops Preserved

- Training happened: no.
- LoRA happened: no.
- GPU model execution happened: no.
- OpenVLA-OFT happened: no.
- Large asset download happened: no.
- Custom `LIBERO_7D` adapter route used: no.
- Paper claims: no.

